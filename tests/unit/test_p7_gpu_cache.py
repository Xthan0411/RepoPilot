"""Gold-free GPU cache orchestration using real tiny corpora and fake CUDA only."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from repopilot.config import RepoPilotSettings
from repopilot.embeddings import EmbeddingInput, embedding_identity_from_settings
from repopilot.retrieval import ingest_repository
from repopilot.retrieval.corpus import RepositoryChunkCorpusBuilder
from repopilot.schemas import TaskSpec
from repopilot.tools.repository import ReadOnlyRepository

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import p7_gpu_cache as gpu  # noqa: E402
from scripts.p7_holdout_agent_inputs import AGENT_INPUTS_PATH  # noqa: E402
from scripts.p7_holdout_control import EXPECTED_TASK_IDS, MATRIX_PATH  # noqa: E402
from scripts.p7_repository_workload_preflight import (  # noqa: E402
    AGENT_INPUTS_SHA256,
    MATRIX_SHA256,
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name in {"OPENAI_API_KEY", "DATABASE_URL"} or name.startswith("REPOPILOT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _tasks() -> tuple[tuple[TaskSpec, ...], RepoPilotSettings]:
    return gpu.load_tasks(PROJECT_ROOT)


def _checkout(root: Path, task: TaskSpec) -> Path:
    path = root / gpu.CHECKOUT_PATH / task.task_id
    path.mkdir(parents=True)
    (path / ".git").mkdir()
    (path / "example.py").write_text(
        "class Example:\n    def increment(self, value):\n        return value + 1\n",
        encoding="utf-8",
    )
    (path / "README.md").write_text("Example increment API.\n", encoding="utf-8")
    return path


def _fake_git(task: TaskSpec, changes: dict[tuple[str, ...], str] | None = None) -> Any:
    changes = changes or {}

    def git(path: Path, *arguments: str) -> str:
        if arguments in changes:
            return changes[arguments]
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(path)
        if arguments == ("rev-parse", "HEAD"):
            return task.base_commit
        if arguments == ("remote", "get-url", "origin"):
            return str(task.repository_url)
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all", "--ignored"):
            return ""
        raise AssertionError(f"unexpected git operation: {arguments}")

    return git


def test_load_tasks_reads_only_agent_projection_matrix_and_lock_without_gold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = {
        (PROJECT_ROOT / AGENT_INPUTS_PATH).resolve(),
        (PROJECT_ROOT / MATRIX_PATH).resolve(),
        (PROJECT_ROOT / "uv.lock").resolve(),
    }
    original = Path.read_bytes
    seen: list[Path] = []

    def read_bytes(path: Path) -> bytes:
        assert path.resolve() in allowed, f"unexpected evaluator/dependency read: {path}"
        seen.append(path.resolve())
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    tasks, settings = _tasks()
    assert tuple(task.task_id for task in tasks) == EXPECTED_TASK_IDS
    assert set(seen) == allowed
    assert settings.embedding_max_length == 2048
    assert settings.embedding_batch_size == 32


def test_changed_agent_input_bytes_are_rejected_before_matrix_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / AGENT_INPUTS_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes((PROJECT_ROOT / AGENT_INPUTS_PATH).read_bytes() + b"\n")
    with pytest.raises(gpu.GpuCacheError, match="agent_inputs_changed"):
        gpu.load_tasks(tmp_path)


def test_task_order_is_not_inferred_or_reordered_from_untrusted_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads((PROJECT_ROOT / AGENT_INPUTS_PATH).read_bytes())
    payload["tasks"] = list(reversed(payload["tasks"]))
    raw = json.dumps(payload).encode()
    target = tmp_path / AGENT_INPUTS_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    monkeypatch.setattr(gpu, "AGENT_INPUTS_SHA256", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="tasks do not match"):
        gpu.load_tasks(tmp_path)


def test_dependency_lock_changes_rejected_without_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for relative in (AGENT_INPUTS_PATH, MATRIX_PATH, Path("uv.lock")):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative).read_bytes())
    (tmp_path / "uv.lock").write_bytes(b"changed dependency lock")
    monkeypatch.setattr(gpu, "load_gpu", lambda *args: pytest.fail("GPU"))
    with pytest.raises(gpu.GpuCacheError, match="dependency_lock_changed"):
        gpu.load_tasks(tmp_path)


def test_real_fixture_chunk_content_matches_production_and_is_path_portable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks, settings = _tasks()
    task = tasks[0]
    first_root, second_root = tmp_path / "cpu", tmp_path / "gpu"
    first = _checkout(first_root, task)
    _checkout(second_root, task)
    monkeypatch.setattr(gpu, "check_checkout", lambda *args: None)
    first_inputs, first_evidence = gpu.build_inputs(first_root, (task,), settings)
    second_inputs, second_evidence = gpu.build_inputs(second_root, (task,), settings)
    repository = ReadOnlyRepository.from_settings(first, settings)
    manifest = ingest_repository(
        repository, repository_id="production-local", commit=task.base_commit
    )
    corpus = RepositoryChunkCorpusBuilder.from_settings(repository, settings).build(manifest).corpus
    assert {item.content_hash for item in first_inputs} == {
        chunk.content_hash for chunk in corpus.chunks
    }
    assert first_inputs == second_inputs
    assert first_evidence == second_evidence
    assert str(first_root) not in json.dumps(first_evidence)
    assert all("content" not in record for record in first_evidence if isinstance(record, dict))


def test_all_eight_checkouts_are_verified_before_first_ingestion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks, settings = _tasks()
    for task in tasks:
        _checkout(tmp_path, task)
    seen: list[str] = []
    monkeypatch.setattr(gpu, "check_checkout", lambda path, task: seen.append(task.task_id))
    original = ingest_repository

    def ingest(repository: ReadOnlyRepository, **kwargs: Any) -> Any:
        assert seen[:8] == list(EXPECTED_TASK_IDS)
        return original(repository, **kwargs)

    monkeypatch.setattr(gpu, "ingest_repository", ingest)
    inputs, evidence = gpu.build_inputs(tmp_path, tasks, settings)
    assert inputs
    assert len(evidence) == 8


def test_last_missing_checkout_stops_before_any_scan_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks, settings = _tasks()
    calls: list[str] = []

    def check(path: Path, task: TaskSpec) -> None:
        calls.append(task.task_id)
        if task == tasks[-1]:
            raise gpu.GpuCacheError("checkout_missing")

    monkeypatch.setattr(gpu, "check_checkout", check)
    monkeypatch.setattr(gpu, "ingest_repository", lambda *args, **kwargs: pytest.fail("scan"))
    monkeypatch.setattr(gpu, "load_gpu", lambda *args: pytest.fail("GPU"))
    with pytest.raises(gpu.GpuCacheError, match="checkout_missing"):
        gpu.build_inputs(tmp_path, tasks, settings)
    assert tuple(calls) == EXPECTED_TASK_IDS


@pytest.mark.parametrize(
    ("arguments", "result", "code"),
    [
        (("rev-parse", "HEAD"), "0" * 40, "checkout_commit_mismatch"),
        (
            ("remote", "get-url", "origin"),
            "https://github.com/other/project.git",
            "checkout_origin_mismatch",
        ),
        (
            ("status", "--porcelain=v1", "--untracked-files=all", "--ignored"),
            "?? local.txt",
            "checkout_not_clean",
        ),
        (("rev-parse", "--show-toplevel"), "/other/repository", "checkout_root_mismatch"),
    ],
)
def test_checkout_rejects_wrong_commit_origin_dirty_state_and_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
    result: str,
    code: str,
) -> None:
    tasks, _ = _tasks()
    path = _checkout(tmp_path, tasks[0])
    monkeypatch.setattr(gpu, "_git", _fake_git(tasks[0], {arguments: result}))
    with pytest.raises(gpu.GpuCacheError, match=code):
        gpu.check_checkout(path, tasks[0])
    assert (path / "example.py").exists()


@pytest.mark.parametrize("link_git", [False, True])
def test_checkout_rejects_symlink_before_git_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, link_git: bool
) -> None:
    tasks, _ = _tasks()
    real = tmp_path / "real"
    real.mkdir()
    target = tmp_path / "checkout"
    if link_git:
        target.mkdir()
        (target / ".git").symlink_to(real, target_is_directory=True)
    else:
        target.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(gpu, "_git", lambda *args: pytest.fail("git must not run"))
    with pytest.raises(gpu.GpuCacheError, match=r"unsafe_git_dir|checkout_missing"):
        gpu.check_checkout(target, tasks[0])


def test_prepare_fetches_exact_commit_without_reset_and_preserves_existing_dirty_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks, _ = _tasks()
    task = tasks[0]
    operations: list[tuple[str, ...]] = []

    def git(path: Path, *arguments: str) -> str:
        operations.append(arguments)
        return ""

    monkeypatch.setattr(gpu, "_git", git)
    monkeypatch.setattr(gpu, "check_checkout", lambda *args: None)
    gpu.prepare_checkouts(tmp_path, (task,))
    assert ("fetch", "--quiet", "--depth=1", "origin", task.base_commit) in operations
    assert ("checkout", "--quiet", "--detach", task.base_commit) in operations
    assert not any(args[0] in {"reset", "clean", "pull", "submodule"} for args in operations)
    target = tmp_path / gpu.CHECKOUT_PATH / task.task_id
    original = target / "local-work.txt"
    original.write_text("preserve me", encoding="utf-8")
    operations.clear()

    def dirty(*args: object) -> None:
        raise gpu.GpuCacheError("checkout_not_clean")

    monkeypatch.setattr(gpu, "check_checkout", dirty)
    with pytest.raises(gpu.GpuCacheError, match="checkout_not_clean"):
        gpu.prepare_checkouts(tmp_path, (task,))
    assert operations == []
    assert original.read_text() == "preserve me"


def test_failed_checkout_staging_does_not_block_later_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks, _ = _tasks()
    fail = True

    def git(path: Path, *arguments: str) -> str:
        if fail and arguments[0] == "fetch":
            raise subprocess.CalledProcessError(1, ("git", *arguments))
        return ""

    monkeypatch.setattr(gpu, "_git", git)
    monkeypatch.setattr(gpu, "check_checkout", lambda *args: None)
    with pytest.raises(subprocess.CalledProcessError):
        gpu.prepare_checkouts(tmp_path, tasks[:1])
    parent = tmp_path / gpu.CHECKOUT_PATH
    assert list(parent.iterdir()) == []
    fail = False
    gpu.prepare_checkouts(tmp_path, tasks[:1])
    assert (parent / tasks[0].task_id).is_dir()


def _runtime(
    *,
    device: str = "cuda",
    dtype: str = "torch.float32",
    revision: str = "a" * 40,
    length: int = 2048,
    tf32: bool = False,
) -> tuple[Any, Any]:
    parameter = SimpleNamespace(
        device=SimpleNamespace(type=device), dtype=dtype, is_floating_point=lambda: True
    )
    model_type = type(
        "RemoteJinaModel", (), {"__module__": f"transformers_modules.jina.{revision}.modeling_bert"}
    )

    class Model:
        max_seq_length = length

        def parameters(self) -> list[Any]:
            return [parameter]

        def __getitem__(self, index: int) -> Any:
            assert index == 0
            return SimpleNamespace(auto_model=model_type())

    torch = SimpleNamespace(
        __version__="fake-torch",
        version=SimpleNamespace(cuda="fake-cuda"),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=tf32)),
            cudnn=SimpleNamespace(allow_tf32=False),
        ),
        cuda=SimpleNamespace(synchronize=lambda: None, get_device_name=lambda index: "Fake GPU"),
        are_deterministic_algorithms_enabled=lambda: False,
        get_float32_matmul_precision=lambda: "highest",
    )
    return Model(), torch


def test_runtime_evidence_records_actual_loaded_device_code_and_precision() -> None:
    model, torch = _runtime()
    evidence = gpu.validate_runtime(model, torch, "a" * 40)
    assert evidence["device"] == "cuda"
    assert evidence["dtype"] == "torch.float32"
    assert evidence["batch_size"] == 8
    assert evidence["matmul_tf32"] is False
    assert evidence["maximum_length"] == 2048
    assert "a" * 40 in str(evidence["runtime_model_module"])


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"device": "cpu"}, "cuda_fp32_required"),
        ({"dtype": "torch.float16"}, "cuda_fp32_required"),
        ({"revision": "b" * 40}, "runtime_code_revision_mismatch"),
        ({"length": 1024}, "sequence_limit_mismatch"),
        ({"tf32": True}, "matmul_tf32_must_be_disabled"),
    ],
)
def test_runtime_rejects_cpu_lower_precision_unpinned_code_and_changed_policy(
    changes: dict[str, Any], code: str
) -> None:
    model, torch = _runtime(**changes)
    with pytest.raises(gpu.GpuCacheError, match=code):
        gpu.validate_runtime(model, torch, "a" * 40)


def _mock_run(monkeypatch: pytest.MonkeyPatch) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks, settings = _tasks()
    content = "answer = 42\n"
    inputs = (
        EmbeddingInput(content_hash=hashlib.sha256(content.encode()).hexdigest(), content=content),
    )
    identity = embedding_identity_from_settings(settings)
    bindings: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    provider = SimpleNamespace(identity=identity)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            memory_stats=lambda: {"num_alloc_retries": 0, "num_ooms": 0},
            max_memory_allocated=lambda: 1024**2,
        )
    )
    monkeypatch.setattr(gpu, "verify_source", lambda root: ("a" * 40, {"src/file.py": "b" * 40}))
    monkeypatch.setattr(gpu, "load_tasks", lambda root: (tasks, settings))
    monkeypatch.setattr(
        gpu, "build_inputs", lambda *args: (inputs, [{"task_id": tasks[0].task_id}])
    )
    monkeypatch.setattr(
        gpu, "load_gpu", lambda settings: (provider, {"profile": gpu.GPU_PROFILE}, torch)
    )
    monkeypatch.setattr(gpu, "job_lock", lambda path: nullcontext())
    monkeypatch.setattr(
        gpu, "bind_job", lambda path, supplied_identity, binding: bindings.append(binding)
    )
    monkeypatch.setattr(gpu, "_emit", lambda **event: events.append(event))

    def fill(
        path: Path,
        supplied_identity: object,
        supplied_inputs: object,
        supplied_provider: object,
        *,
        max_batches: int | None,
        progress: Any,
    ) -> dict[str, str]:
        assert supplied_identity == identity
        assert supplied_inputs == inputs
        assert supplied_provider is provider
        assert max_batches == 1
        progress({"stage": "batch_ready", "completed_inputs": 1})
        return {"status": "complete"}

    monkeypatch.setattr(gpu, "fill_cache", fill)
    return bindings, events


def test_run_binds_source_model_profile_and_inputs_and_emits_memory_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings, events = _mock_run(monkeypatch)
    result = gpu.run(tmp_path, tmp_path / "job", max_batches=1)
    assert result == {"status": "complete"}
    binding = bindings[0]
    assert binding["controller_commit"] == "a" * 40
    assert binding["execution_source_sha256"] == gpu._digest({"src/file.py": "b" * 40})
    assert binding["agent_inputs_raw_sha256"] == AGENT_INPUTS_SHA256
    assert binding["matrix_sha256"] == MATRIX_SHA256
    assert binding["runtime"] == {"profile": gpu.GPU_PROFILE}
    assert binding["input_count"] == 1
    assert len(binding["input_set_sha256"]) == 64
    assert binding["agent_constructed"] is False and binding["formal_ready"] is False
    assert binding["openai_requests"] == 0
    assert events[-1]["cuda_allocation_retries"] == 0
    assert events[-1]["cuda_ooms"] == 0
    assert events[-1]["peak_allocated_mib"] == 1.0


@pytest.mark.parametrize("changed_at", [2, 3])
def test_run_detects_source_mutation_during_scan_or_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_at: int
) -> None:
    _mock_run(monkeypatch)
    calls = 0

    def source(root: Path) -> tuple[str, dict[str, str]]:
        nonlocal calls
        calls += 1
        return "a" * 40, {"src/file.py": ("c" if calls >= changed_at else "b") * 40}

    monkeypatch.setattr(gpu, "verify_source", source)
    if changed_at == 2:
        monkeypatch.setattr(gpu, "fill_cache", lambda *args, **kwargs: pytest.fail("embedding"))
    with pytest.raises(gpu.GpuCacheError, match="source_changed_during"):
        gpu.run(tmp_path, tmp_path / "job", max_batches=1)


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        (["run", "--max-batches", "0"], "invalid_batch_limit"),
        (["run", "--max-batches", "-1"], "invalid_batch_limit"),
        (["verify", "--max-batches", "1"], "unexpected_batch_limit"),
        (["run", "--archive", "bundle.tar.gz"], "unexpected_archive"),
        (["verify", "--archive", "bundle.tar.gz"], "unexpected_archive"),
        (["export"], "archive_required"),
        (["import"], "archive_required"),
        (["run", "--job-directory", "../outside"], "isolated_gpu_cache_directory_required"),
        (
            ["run", "--job-directory", "artifacts/embedding-cache"],
            "isolated_gpu_cache_directory_required",
        ),
        (
            ["run", "--job-directory", "artifacts/p7-gpu-checkouts-v1"],
            "isolated_gpu_cache_directory_required",
        ),
        (
            ["run", "--job-directory", "artifacts/p7-gpu-cache-v1/nested"],
            "isolated_gpu_cache_directory_required",
        ),
        (
            ["run", "--job-directory", "artifacts/p7-gpu-cache-"],
            "isolated_gpu_cache_directory_required",
        ),
    ],
)
def test_cli_invalid_options_fail_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    code: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["gpu-cache", *arguments, "--repository-root", str(tmp_path)])
    monkeypatch.setattr(gpu, "load_gpu", lambda *args: pytest.fail("GPU"))
    monkeypatch.setattr(gpu, "run", lambda *args, **kwargs: pytest.fail("run"))
    assert gpu.main() == 1
    assert json.loads(capsys.readouterr().out)["code"] == code


@pytest.mark.parametrize("secret", ["OPENAI_API_KEY", "DATABASE_URL"])
def test_run_secret_environment_rejected_before_source_or_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, secret: str
) -> None:
    monkeypatch.setenv(secret, "MUST_NOT_APPEAR_IN_ERROR")
    monkeypatch.setattr(gpu, "verify_source", lambda *args: pytest.fail("source"))
    monkeypatch.setattr(gpu, "load_gpu", lambda *args: pytest.fail("GPU"))
    with pytest.raises(SystemExit, match="embedding_probe_secret_exposure") as failure:
        gpu.run(tmp_path, tmp_path / "job", max_batches=1)
    assert "MUST_NOT_APPEAR_IN_ERROR" not in str(failure.value)


@pytest.mark.parametrize("secret", ["OPENAI_API_KEY", "DATABASE_URL"])
def test_prepare_secret_environment_rejected_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], secret: str
) -> None:
    monkeypatch.setenv(secret, "MUST_NOT_APPEAR_IN_ERROR")
    monkeypatch.setattr(
        sys, "argv", ["gpu-cache", "prepare-checkouts", "--repository-root", str(tmp_path)]
    )
    monkeypatch.setattr(gpu, "verify_source", lambda *args: pytest.fail("source"))
    monkeypatch.setattr(gpu, "prepare_checkouts", lambda *args: pytest.fail("network"))
    assert gpu.main() == 1
    output = capsys.readouterr().out
    assert json.loads(output)["code"] == "gpu_host_secrets_forbidden"
    assert "MUST_NOT_APPEAR_IN_ERROR" not in output
