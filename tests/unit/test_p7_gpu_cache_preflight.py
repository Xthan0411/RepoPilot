"""Fail-closed GPU-cache consumption and bounded CPU-worker orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from repopilot.config import DEFAULT_DATABASE_URL
from repopilot.embeddings import (
    EmbeddingInput,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    LocalEmbeddingCache,
    embedding_request_id,
)
from repopilot.retrieval import ingest_repository
from repopilot.retrieval.corpus import RepositoryChunkCorpusBuilder
from repopilot.storage.memory import InMemoryVectorStore
from repopilot.tools.repository import ReadOnlyRepository

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import p7_gpu_cache_preflight as preflight  # noqa: E402
from scripts import p7_repository_workload_preflight as workload  # noqa: E402
from scripts import p7_workload_runtime_probe as runtime  # noqa: E402
from scripts.p7_gpu_cache_acceptance import ReadOnlyGpuCache  # noqa: E402
from scripts.p7_gpu_cache_store import _read, _write_once  # noqa: E402
from scripts.p7_gpu_cpu_probe import PINNED_IDENTITY  # noqa: E402


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name in {"OPENAI_API_KEY", "DATABASE_URL"} or name.startswith("REPOPILOT_"):
            monkeypatch.delenv(name)


def _input(content: str) -> EmbeddingInput:
    return EmbeddingInput(
        content=content, content_hash=hashlib.sha256(content.encode()).hexdigest()
    )


def _request(*inputs: EmbeddingInput, attempt: int = 1) -> EmbeddingRequest:
    return EmbeddingRequest(
        request_id=embedding_request_id(PINNED_IDENTITY, inputs, attempt),
        identity=PINNED_IDENTITY,
        inputs=inputs,
        attempt=attempt,
    )


class SyntheticProvider:
    identity = PINNED_IDENTITY

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.calls += 1
        return EmbeddingResponse(
            request_id=request.request_id,
            identity=self.identity,
            vectors=tuple(
                EmbeddingVector(content_hash=item.content_hash, values=(1.0,) + (0.0,) * 767)
                for item in request.inputs
            ),
        )


def test_queries_match_workflow_truncation_then_dense_strip() -> None:
    inputs = workload.load_inputs(PROJECT_ROOT)
    settings = workload._settings(inputs, "dense-only-v1")
    queries = preflight.queries_for_tasks(inputs)
    assert len(queries) == 8
    for task, query in zip(inputs.agent_inputs.tasks, queries, strict=True):
        assert query.content == (
            f"{task.issue_title}\n{task.issue_body}"[: settings.search_max_query_length].strip()
        )


def test_only_four_calls_per_exact_query_reach_underlying_provider() -> None:
    base = SyntheticProvider()
    query = _input("allowed issue query")
    provider = preflight.QueryOnlyProvider(base, (query,))
    for _ in range(4):
        provider.embed(_request(query))
    with pytest.raises(preflight.PreflightError, match="query_allowlist_or_call_limit_exceeded"):
        provider.embed(_request(query))
    assert base.calls == provider.calls == 4


def test_duplicate_issue_queries_preserve_four_calls_for_each_task() -> None:
    base = SyntheticProvider()
    query = _input("shared query")
    provider = preflight.QueryOnlyProvider(base, (query, query))
    for _ in range(8):
        provider.embed(_request(query))
    with pytest.raises(preflight.PreflightError, match="call_limit"):
        provider.embed(_request(query))
    assert base.calls == 8


@pytest.mark.parametrize("mode", ["unknown", "retry", "batch", "identity"])
def test_query_gate_rejects_corpus_retry_batch_and_other_identity(mode: str) -> None:
    base = SyntheticProvider()
    query = _input("allowed")
    provider = preflight.QueryOnlyProvider(base, (query,))
    if mode == "unknown":
        request = _request(_input("a missing corpus vector must never be calculated"))
    elif mode == "retry":
        request = _request(query, attempt=2)
    elif mode == "batch":
        request = _request(query, _input("second"))
    else:
        request = _request(query).model_copy(
            update={"identity": PINNED_IDENTITY.model_copy(update={"revision": "other"})}
        )
    with pytest.raises(preflight.PreflightError):
        provider.embed(request)
    assert base.calls == provider.calls == 0


def test_failed_provider_attempt_still_consumes_query_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = SyntheticProvider()
    query = _input("allowed")
    provider = preflight.QueryOnlyProvider(base, (query,))

    def fail(request: EmbeddingRequest) -> EmbeddingResponse:
        raise RuntimeError("no retry")

    monkeypatch.setattr(base, "embed", fail)
    for _ in range(4):
        with pytest.raises(RuntimeError, match="no retry"):
            provider.embed(_request(query))
    with pytest.raises(preflight.PreflightError, match="call_limit"):
        provider.embed(_request(query))
    assert provider.calls == 4


@pytest.mark.parametrize("database", [False, True])
def test_worker_environment_removes_keys_overrides_proxy_and_custom_database(
    monkeypatch: pytest.MonkeyPatch, database: bool
) -> None:
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HTTPS_PROXY",
        "REPOPILOT_EMBEDDING_CACHE_DIR",
        "PYTHONPATH",
        "HF_HOME",
    ):
        monkeypatch.setenv(name, "must-not-reach-worker")
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret@example.invalid/other")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    result = preflight.worker_environment(database=database)
    assert "must-not-reach-worker" not in json.dumps(result)
    assert result["CUDA_VISIBLE_DEVICES"] == ""
    assert result["HF_HUB_OFFLINE"] == result["TRANSFORMERS_OFFLINE"] == "1"
    assert result["OMP_NUM_THREADS"] == result["MKL_NUM_THREADS"] == "8"
    assert (result.get("DATABASE_URL") == DEFAULT_DATABASE_URL) is database


class FakeChild:
    pid = 12345

    def __init__(self, responses: list[int | BaseException]) -> None:
        self.responses = responses
        self.timeouts: list[int | None] = []

    def __enter__(self) -> FakeChild:
        return self

    def __exit__(self, *arguments: object) -> None:
        return None

    def wait(self, timeout: int | None = None) -> int:
        self.timeouts.append(timeout)
        if not self.responses:
            return 0
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.mark.parametrize("code,status", [(0, "passed"), (1, "failed"), (-9, "failed")])
def test_worker_status_uses_exit_code_and_creates_private_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int, status: str
) -> None:
    child = FakeChild([code])
    seen: dict[str, Any] = {}

    def popen(command: tuple[str, ...], **kwargs: Any) -> FakeChild:
        seen.update(kwargs)
        assert command == ("python", "worker")
        return child

    monkeypatch.setattr(subprocess, "Popen", popen)
    result = preflight.run_worker(
        tmp_path, ("python", "worker"), database=False, timeout_seconds=12
    )
    assert result["status"] == status and result["exit_code"] == code
    assert seen["cwd"] == tmp_path and seen["start_new_session"] is True
    assert seen["stderr"] == subprocess.DEVNULL
    assert child.timeouts == [12]


@pytest.mark.parametrize("leader_exits_on_term", [True, False])
def test_timeout_kills_entire_group_even_when_only_leader_exits_after_term(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, leader_exits_on_term: bool
) -> None:
    timeout = subprocess.TimeoutExpired("worker", 1)
    responses: list[int | BaseException] = (
        [timeout, 0] if leader_exits_on_term else [timeout, timeout, -9]
    )
    child = FakeChild(responses)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    result = preflight.run_worker(tmp_path, ("worker",), database=False, timeout_seconds=1)
    assert result["status"] == "timeout"
    assert result["temporary_sandbox_cleanup_verified"] is False
    assert killed == [(child.pid, signal.SIGTERM), (child.pid, signal.SIGKILL)]


def test_timeout_handles_process_group_already_exited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeChild([subprocess.TimeoutExpired("worker", 1), 0])
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)

    def missing(pid: int, sig: int) -> None:
        raise ProcessLookupError("already finished")

    monkeypatch.setattr(os, "killpg", missing)
    report = preflight.run_worker(tmp_path, ("worker",), database=False, timeout_seconds=1)
    assert report["status"] == "timeout"


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(130)])
def test_interruption_terminates_process_group_before_propagating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: BaseException
) -> None:
    child = FakeChild([interruption, 0])
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(type(interruption)):
        preflight.run_worker(tmp_path, ("worker",), database=False, timeout_seconds=1)
    assert killed == [(child.pid, signal.SIGTERM), (child.pid, signal.SIGKILL)]


def _qualification_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = workload.load_inputs(PROJECT_ROOT)
    samples = tuple(_input(value) for value in ("short", "median", "long sample"))
    coverage = SimpleNamespace(
        report={"coverage_passed": True},
        samples=samples,
        gpu_vectors=tuple(
            EmbeddingVector(content_hash=item.content_hash, values=(1.0,) + (0.0,) * 767)
            for item in samples
        ),
        cache=SimpleNamespace(identity=PINNED_IDENTITY),
    )
    monkeypatch.setattr(workload, "load_inputs", lambda root: inputs)
    monkeypatch.setattr(workload, "verify_source", lambda root: ("a" * 40, {"s": "b" * 40}))
    monkeypatch.setattr(preflight, "_checkouts", lambda *args: ())
    monkeypatch.setattr(preflight, "verify_coverage", lambda root, **kwargs: coverage)
    monkeypatch.setattr(preflight, "verify_pinned_cache", lambda path: None)


@pytest.mark.parametrize(
    "worker_status,probe_passed",
    [("failed", False), ("timeout", False), ("passed", False), ("failed", True)],
)
def test_numerical_gate_failure_never_starts_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_status: str, probe_passed: bool
) -> None:
    _qualification_fixture(tmp_path, monkeypatch)
    calls: list[tuple[str, ...]] = []

    def worker(root: Path, command: tuple[str, ...], **kwargs: Any) -> dict[str, Any]:
        calls.append(command)
        assert "scripts.p7_gpu_cpu_probe" in command
        assert kwargs["database"] is False
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"cpu_probe_passed": probe_passed}))
        return {"status": worker_status}

    monkeypatch.setattr(preflight, "run_worker", worker)
    report = preflight.qualify(tmp_path, prepare_retrieval=True)
    assert len(calls) == 1
    assert report["preflight_passed"] is False
    assert report["formal_ready"] is False and report["paid_requests_made"] == 0
    assert "retrieval" not in report


@pytest.mark.parametrize("prepare_retrieval", [False, True])
def test_success_remains_unapproved_and_records_only_requested_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepare_retrieval: bool
) -> None:
    _qualification_fixture(tmp_path, monkeypatch)
    requests: list[Path] = []

    def worker(root: Path, command: tuple[str, ...], **kwargs: Any) -> dict[str, Any]:
        if "scripts.p7_gpu_cpu_probe" in command:
            request = Path(command[command.index("--request") + 1])
            requests.append(request)
            assert request.stat().st_mode & 0o777 == 0o600
            assert len(_read(request)["samples"]) == 3  # type: ignore[arg-type]
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps({"cpu_probe_passed": True}))
        else:
            assert prepare_retrieval and kwargs["database"] is True
            output = Path(command[command.index("--retrieval-worker-output") + 1])
            _write_once(output, {"retrieval_preflight_passed": True})
        return {"status": "passed"}

    monkeypatch.setattr(preflight, "run_worker", worker)
    report = preflight.qualify(tmp_path, prepare_retrieval=prepare_retrieval)
    assert report["preflight_passed"] is prepare_retrieval
    assert report["formal_ready"] is False
    assert report["formal_blockers"] == ["new_source_freeze_and_exact_plan_authorization_required"]
    assert report["paid_requests_made"] == report["provider_requests_made"] == 0
    assert all(not request.exists() for request in requests)


def test_missing_worker_output_does_not_pass_numeric_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _qualification_fixture(tmp_path, monkeypatch)
    calls: list[tuple[str, ...]] = []

    def worker(root: Path, command: tuple[str, ...], **kwargs: Any) -> dict[str, Any]:
        calls.append(command)
        return {"status": "passed"}

    monkeypatch.setattr(preflight, "run_worker", worker)
    report = preflight.qualify(tmp_path, prepare_retrieval=True)
    assert report["preflight_passed"] is False and len(calls) == 1


@pytest.mark.parametrize("change", ["source", "checkouts", "cache"])
def test_qualification_rechecks_immutable_inputs_after_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    _qualification_fixture(tmp_path, monkeypatch)

    def worker(root: Path, command: tuple[str, ...], **kwargs: Any) -> dict[str, Any]:
        if change == "source":
            monkeypatch.setattr(workload, "verify_source", lambda root: ("c" * 40, {}))
        elif change == "checkouts":
            monkeypatch.setattr(preflight, "_checkouts", lambda *args: ("changed",))
        else:

            def changed(path: Path) -> None:
                raise preflight.PreflightError("cache_changed")

            monkeypatch.setattr(preflight, "verify_pinned_cache", changed)
        return {"status": "failed"}

    monkeypatch.setattr(preflight, "run_worker", worker)
    with pytest.raises(preflight.PreflightError):
        preflight.qualify(tmp_path, prepare_retrieval=True)


def _retrieval_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SyntheticProvider, ReadOnlyGpuCache, list[str]]:
    inputs = workload.load_inputs(PROJECT_ROOT)
    settings = workload._settings(inputs, "dense-only-v1")
    cache_root = tmp_path / "vectors"
    writer = LocalEmbeddingCache(cache_root)
    for binding in inputs.environment.bindings:
        checkout = tmp_path / binding.checkout_relative_path
        checkout.mkdir(parents=True)
        (checkout / "example.py").write_text("def increment(value):\n    return value + 1\n")
        (checkout / "README.md").write_text("Example increment value API.\n")
        repository = ReadOnlyRepository.from_settings(checkout, settings)
        manifest = ingest_repository(
            repository, repository_id="fixture", commit=binding.base_commit
        )
        corpus = (
            RepositoryChunkCorpusBuilder.from_settings(repository, settings).build(manifest).corpus
        )
        for chunk in corpus.chunks:
            writer.put(
                PINNED_IDENTITY,
                EmbeddingVector(content_hash=chunk.content_hash, values=(1.0,) + (0.0,) * 767),
            )
    files = {
        f"vectors/{path.relative_to(cache_root).as_posix()}": hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in cache_root.rglob("*.json")
    }
    cache = ReadOnlyGpuCache(cache_root, PINNED_IDENTITY, files)
    accepted = SimpleNamespace(identity=PINNED_IDENTITY, cache=cache)
    provider = SyntheticProvider()
    store = InMemoryVectorStore(dimension=768)
    runtime_tasks: list[str] = []

    def probe(**kwargs: Any) -> dict[str, Any]:
        runtime_tasks.append(kwargs["task"].task_id)
        return {"provider_requests_made": 0, "elapsed_ms": 1}

    monkeypatch.setattr(workload, "load_inputs", lambda root: inputs)
    monkeypatch.setattr(workload, "verify_source", lambda root: ("a" * 40, {"s": "b" * 40}))
    monkeypatch.setattr(preflight, "_checkouts", lambda *args: ())
    monkeypatch.setattr(preflight, "verify_pinned_cache", lambda path: accepted)
    monkeypatch.setattr(
        preflight, "create_cpu_provider", lambda identity: (provider, {"device": "cpu"})
    )
    monkeypatch.setattr(preflight, "PostgresVectorStore", lambda settings: store)
    monkeypatch.setattr(runtime, "probe_runtime", probe)
    return provider, cache, runtime_tasks


def test_all_eight_production_retrieval_paths_use_cache_only_and_32_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, cache, runtime_tasks = _retrieval_fixture(tmp_path, monkeypatch)
    report = preflight.run_retrieval_worker(tmp_path)
    assert report["retrieval_preflight_passed"] is True
    assert report["query_embedding_calls"] == provider.calls == 32
    assert report["source_embedding_calls"] == cache.misses == cache.write_attempts == 0
    assert cache.hits == 64
    assert len(runtime_tasks) == 8
    assert len(report["tasks"]) == 8  # type: ignore[arg-type]


def test_missing_corpus_cache_entry_fails_without_cpu_corpus_embedding_or_later_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, cache, runtime_tasks = _retrieval_fixture(tmp_path, monkeypatch)
    cache._files.clear()
    report = preflight.run_retrieval_worker(tmp_path)
    assert report["retrieval_preflight_passed"] is False
    assert provider.calls == cache.write_attempts == 0 and cache.misses > 0
    assert runtime_tasks == []
    assert len(report["tasks"]) == 1  # type: ignore[arg-type]


def test_main_refuses_existing_report_without_doing_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / preflight.NUMERIC_OUTPUT
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing evidence\n")
    monkeypatch.setattr(sys, "argv", ["preflight", "--repository-root", str(tmp_path)])
    monkeypatch.setattr(preflight, "qualify", lambda *args, **kwargs: pytest.fail("qualification"))
    assert preflight.main() == 1
    assert path.read_bytes() == b"existing evidence\n"
    assert "output_conflict" in capsys.readouterr().out


def test_main_seals_successful_report_as_private_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["preflight", "--repository-root", str(tmp_path), "--prepare-retrieval"]
    )
    monkeypatch.setattr(
        preflight,
        "qualify",
        lambda *args, **kwargs: {"preflight_passed": True, "formal_ready": False},
    )
    assert preflight.main() == 0
    path = tmp_path / preflight.OUTPUT
    assert path.stat().st_mode & 0o777 == 0o600
    assert _read(path)["formal_ready"] is False


def test_main_foreign_failure_message_is_not_disclosed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("OPENAI_API_KEY=secret; postgres://private; source text")

    monkeypatch.setattr(sys, "argv", ["preflight", "--repository-root", str(tmp_path)])
    monkeypatch.setattr(preflight, "qualify", fail)
    assert preflight.main() == 1
    output = capsys.readouterr().out
    assert "qualification_failed" in output
    assert "secret" not in output and "private" not in output and "source text" not in output
    report = _read(tmp_path / preflight.NUMERIC_OUTPUT)
    assert report["failure_code"] == "qualification_failed"
    assert report["formal_ready"] is False and report["paid_requests_made"] == 0
    assert "secret" not in json.dumps(report)


def test_main_rejects_runtime_overrides_before_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("REPOPILOT_EMBEDDING_CACHE_DIR", "other")
    monkeypatch.setattr(sys, "argv", ["preflight", "--repository-root", str(tmp_path)])
    monkeypatch.setattr(preflight, "qualify", lambda *args, **kwargs: pytest.fail("qualification"))
    assert preflight.main() == 1
    assert "unapproved_environment_override" in capsys.readouterr().out


@pytest.mark.parametrize(
    "output",
    [
        "/tmp/report.json",
        "results/other.json",
        "results/p7-holdout/sub/report.json",
        "results/p7-holdout/report.txt",
        "results/p7-holdout/../report.json",
    ],
)
def test_main_only_allows_new_report_paths_under_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], output: str
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["preflight", "--repository-root", str(tmp_path), "--output", output]
    )
    monkeypatch.setattr(preflight, "qualify", lambda *args, **kwargs: pytest.fail("qualification"))
    assert preflight.main() == 1
    assert "unsafe_output_path" in capsys.readouterr().out


def test_main_rejects_symlink_report_path_without_following_or_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / preflight.NUMERIC_OUTPUT
    report.parent.mkdir(parents=True)
    protected = tmp_path / "protected.json"
    protected.write_bytes(b"preserve")
    report.symlink_to(protected)
    monkeypatch.setattr(sys, "argv", ["preflight", "--repository-root", str(tmp_path)])
    monkeypatch.setattr(preflight, "qualify", lambda *args, **kwargs: pytest.fail("qualification"))
    assert preflight.main() == 1
    assert protected.read_bytes() == b"preserve" and report.is_symlink()


@pytest.mark.parametrize(
    "key,dsn", [("secret", DEFAULT_DATABASE_URL), (None, "postgres://elsewhere")]
)
def test_internal_retrieval_worker_rejects_live_key_or_nonlocal_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str | None, dsn: str
) -> None:
    if key:
        monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setenv("DATABASE_URL", dsn)
    output = tmp_path / "internal.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["preflight", "--repository-root", str(tmp_path), "--retrieval-worker-output", str(output)],
    )
    monkeypatch.setattr(preflight, "run_retrieval_worker", lambda *args: pytest.fail("retrieval"))
    assert preflight.main() == 1
    report = _read(output)
    assert report["failure_code"] == "invalid_worker_environment"
