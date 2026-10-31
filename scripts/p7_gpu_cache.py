"""Prepare portable, resumable P7 embeddings; never run an Agent or paid request.

Only prepare-checkouts uses the network (fixed public GitHub commits). Model loading
is offline. GPU output is a separate cache, not an authorization or a formal result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from pydantic import JsonValue

from repopilot.config import DEFAULT_DATABASE_URL, RepoPilotSettings
from repopilot.embeddings import (
    EmbeddingError,
    EmbeddingInput,
    embedding_identity_from_settings,
    embedding_provider_from_settings,
)
from repopilot.embeddings.jina import JinaEmbeddingProvider
from repopilot.evals import ExperimentMatrix, select_experiment_arm
from repopilot.retrieval import RepositoryIngestScanPolicy, ingest_repository
from repopilot.retrieval.corpus import RepositoryChunkCorpusBuilder
from repopilot.schemas import TaskSpec
from repopilot.tools.repository import ReadOnlyRepository
from scripts.p7_embedding_readiness_probe import _validate_network_mode
from scripts.p7_gpu_cache_store import (
    CacheStoreError,
    bind_job,
    export_cache,
    fill_cache,
    import_cache,
    job_lock,
    verify_cache,
)
from scripts.p7_holdout_agent_inputs import AGENT_INPUTS_PATH, HoldoutAgentInputs
from scripts.p7_holdout_control import MATRIX_PATH
from scripts.p7_repository_workload_preflight import (
    AGENT_INPUTS_SHA256,
    MATRIX_SHA256,
    verify_source,
)

JOB_PATH = Path("artifacts/p7-gpu-cache-v1")
CHECKOUT_PATH = Path("artifacts/p7-gpu-checkouts-v1")
GPU_PROFILE = "p7-gpu-cuda-fp32-batch8-v1"


class GpuCacheError(RuntimeError):
    """Only stable, secret-free diagnostic codes may cross the CLI boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise GpuCacheError(code)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _emit(**values: JsonValue) -> None:
    print(json.dumps(values, sort_keys=True), flush=True)


def _contained(root: Path, relative: Path) -> Path:
    _require(not relative.is_absolute() and ".." not in relative.parts, "unsafe_path")
    current = root
    for part in relative.parts:
        current = current / part
        _require(not current.is_symlink(), "unsafe_path")
    _require(current.resolve().is_relative_to(root), "unsafe_path")
    return current


def load_tasks(root: Path) -> tuple[tuple[TaskSpec, ...], RepoPilotSettings]:
    raw = _contained(root, AGENT_INPUTS_PATH).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == AGENT_INPUTS_SHA256, "agent_inputs_changed")
    agent = HoldoutAgentInputs.model_validate_json(raw)
    # Do not use load_experiment_matrix: its reference walk also reads evaluator
    # dataset metadata. The exact digest and model validators suffice for this worker.
    matrix_raw = _contained(root, MATRIX_PATH).read_bytes()
    _require(len(matrix_raw) <= 1_000_000, "matrix_too_large")
    matrix = ExperimentMatrix.model_validate(yaml.safe_load(matrix_raw))
    _require(matrix.content_sha256 == MATRIX_SHA256, "matrix_changed")
    _require(agent.matrix_sha256 == MATRIX_SHA256, "matrix_binding_changed")
    lock_raw = _contained(root, Path(matrix.dependency_lock_path)).read_bytes()
    _require(
        hashlib.sha256(lock_raw).hexdigest() == matrix.dependency_lock_sha256,
        "dependency_lock_changed",
    )
    settings = select_experiment_arm(matrix, "dense-only-v1").settings(
        openai_api_key="gpu-cache-no-llm-request", database_url=DEFAULT_DATABASE_URL
    )
    return agent.tasks, settings


def _job_directory(root: Path, relative: Path) -> Path:
    _require(
        len(relative.parts) == 2
        and relative.parts[0] == "artifacts"
        and relative.parts[1].startswith("p7-gpu-cache-")
        and relative.parts[1] != "p7-gpu-cache-",
        "isolated_gpu_cache_directory_required",
    )
    return _contained(root, relative)


def _git(checkout: Path, *args: str) -> str:
    # A checked-out project's hooks, helpers, and credential configuration are not used.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE"}
    }
    environment.update(
        GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0"
    )
    return subprocess.run(
        (
            "git",
            "-c",
            f"core.hooksPath={os.devnull}",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "protocol.file.allow=never",
            "-C",
            str(checkout),
            *args,
        ),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    ).stdout.strip()


def _url(task: TaskSpec) -> str:
    url = str(task.repository_url or "")
    parsed = urlsplit(url)
    _require(
        parsed.scheme == "https"
        and parsed.netloc == "github.com"
        and not parsed.query
        and not parsed.fragment
        and len(parsed.path.strip("/").split("/")) == 2,
        "unexpected_repository_url",
    )
    return url


def check_checkout(checkout: Path, task: TaskSpec) -> None:
    _require(checkout.is_dir() and not checkout.is_symlink(), "checkout_missing")
    _require(
        (checkout / ".git").is_dir() and not (checkout / ".git").is_symlink(), "unsafe_git_dir"
    )
    _require(
        Path(_git(checkout, "rev-parse", "--show-toplevel")).resolve() == checkout.resolve(),
        "checkout_root_mismatch",
    )
    _require(_git(checkout, "rev-parse", "HEAD") == task.base_commit, "checkout_commit_mismatch")
    _require(
        _git(checkout, "remote", "get-url", "origin") == _url(task), "checkout_origin_mismatch"
    )
    _require(
        not _git(checkout, "status", "--porcelain=v1", "--untracked-files=all", "--ignored"),
        "checkout_not_clean",
    )


def prepare_checkouts(root: Path, tasks: tuple[TaskSpec, ...]) -> None:
    parent = _contained(root, CHECKOUT_PATH)
    parent.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        target = _contained(root, CHECKOUT_PATH / task.task_id)
        _emit(stage="checkout_started", task_id=task.task_id)
        if not target.exists():
            with tempfile.TemporaryDirectory(prefix=".checkout-", dir=parent) as temporary:
                staged = Path(temporary) / "repository"
                staged.mkdir()
                _git(staged, "init", "--quiet")
                _git(staged, "remote", "add", "origin", _url(task))
                _git(staged, "fetch", "--quiet", "--depth=1", "origin", task.base_commit)
                _git(staged, "checkout", "--quiet", "--detach", task.base_commit)
                check_checkout(staged, task)
                _require(not target.exists(), "checkout_destination_conflict")
                staged.rename(target)
        check_checkout(target, task)
        _emit(stage="checkout_ready", task_id=task.task_id)


def build_inputs(
    root: Path, tasks: tuple[TaskSpec, ...], settings: RepoPilotSettings
) -> tuple[tuple[EmbeddingInput, ...], list[JsonValue]]:
    # Validate all repositories before the first expensive scan or embedding call.
    for task in tasks:
        check_checkout(_contained(root, CHECKOUT_PATH / task.task_id), task)
    unique: dict[str, EmbeddingInput] = {}
    evidence: list[JsonValue] = []
    for task in tasks:
        started = time.monotonic()
        _emit(stage="corpus_started", task_id=task.task_id)
        checkout = _contained(root, CHECKOUT_PATH / task.task_id)
        repository = ReadOnlyRepository.from_settings(checkout, settings)
        manifest = ingest_repository(
            repository,
            repository_id="gpu-cache-source",
            commit=task.base_commit,
            scan_policy=RepositoryIngestScanPolicy(),
        )
        corpus = (
            RepositoryChunkCorpusBuilder.from_settings(repository, settings).build(manifest).corpus
        )
        for chunk in corpus.chunks:
            item = EmbeddingInput(content_hash=chunk.content_hash, content=chunk.content)
            previous = unique.setdefault(item.content_hash, item)
            _require(previous == item, "content_hash_conflict")
        # These portable hashes deliberately exclude host-path-derived repository_id.
        record: dict[str, JsonValue] = {
            "task_id": task.task_id,
            "base_commit": task.base_commit,
            "file_count": len(manifest.files),
            "chunk_count": len(corpus.chunks),
            "source_files_sha256": _digest(
                [
                    [file.relative_path, file.content_hash, file.size_bytes]
                    for file in manifest.files
                ]
            ),
            "chunk_contents_sha256": _digest(
                [[c.relative_path, c.start_line, c.end_line, c.content_hash] for c in corpus.chunks]
            ),
        }
        evidence.append(record)
        check_checkout(checkout, task)
        _emit(stage="corpus_ready", elapsed_seconds=round(time.monotonic() - started, 3), **record)
    return tuple(
        sorted(unique.values(), key=lambda item: (len(item.content), item.content_hash))
    ), evidence


def validate_runtime(model: Any, torch: Any, code_revision: str) -> dict[str, JsonValue]:
    """Inspect actual loaded state, not just declared config; no precision conversion."""
    devices = sorted({parameter.device.type for parameter in model.parameters()})
    dtypes = sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
    runtime_module = type(model[0].auto_model).__module__
    _require(devices == ["cuda"] and dtypes == ["torch.float32"], "cuda_fp32_required")
    _require(code_revision in runtime_module.split("."), "runtime_code_revision_mismatch")
    _require(model.max_seq_length == 2048, "sequence_limit_mismatch")
    _require(torch.backends.cuda.matmul.allow_tf32 is False, "matmul_tf32_must_be_disabled")
    torch.cuda.synchronize()
    return {
        "profile": GPU_PROFILE,
        "device": "cuda",
        "dtype": "torch.float32",
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda),
        "runtime_model_module": runtime_module,
        "maximum_length": 2048,
        "batch_size": 8,
        "matmul_tf32": False,
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
    }


def load_gpu(
    settings: RepoPilotSettings,
) -> tuple[JinaEmbeddingProvider, dict[str, JsonValue], Any]:
    _require(
        platform.system() == "Linux" and platform.machine() == "x86_64", "linux_amd64_required"
    )
    torch = importlib.import_module("torch")
    _require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1, "single_cuda_device_required"
    )
    provider = embedding_provider_from_settings(settings)
    _require(isinstance(provider, JinaEmbeddingProvider), "jina_provider_required")
    assert isinstance(provider, JinaEmbeddingProvider)
    model = provider._load_model()
    profile = validate_runtime(model, torch, provider.identity.code_revision)
    for name in ("sentence-transformers", "transformers", "numpy"):
        profile[name] = importlib.metadata.version(name)
    profile["python_version"] = platform.python_version()
    profile["driver_version"] = subprocess.run(
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()
    return provider, profile, torch


def run(root: Path, directory: Path, *, max_batches: int | None) -> dict[str, JsonValue]:
    _validate_network_mode("offline")
    controller, source = verify_source(root)
    tasks, settings = load_tasks(root)
    with job_lock(directory):
        inputs, task_evidence = build_inputs(root, tasks, settings)
        _emit(stage="model_loading_offline", unique_content_count=len(inputs))
        provider, profile, torch = load_gpu(settings)
        identity = embedding_identity_from_settings(settings)
        _require(identity == provider.identity, "embedding_identity_mismatch")
        binding: dict[str, JsonValue] = {
            "controller_commit": controller,
            "execution_source_sha256": _digest(source),
            "agent_inputs_raw_sha256": AGENT_INPUTS_SHA256,
            "matrix_sha256": MATRIX_SHA256,
            "tasks": task_evidence,
            "runtime": profile,
            "input_set_sha256": _digest([item.content_hash for item in inputs]),
            "input_count": len(inputs),
            "agent_constructed": False,
            "openai_requests": 0,
            "formal_ready": False,
        }
        _require(verify_source(root) == (controller, source), "source_changed_during_scan")
        bind_job(directory, identity, binding)
        baseline = torch.cuda.memory_stats()
        started = time.monotonic()

        def progress(event: dict[str, JsonValue]) -> None:
            memory = torch.cuda.memory_stats()
            _emit(
                **event,
                elapsed_seconds=round(time.monotonic() - started, 3),
                cuda_allocation_retries=int(memory.get("num_alloc_retries", 0))
                - int(baseline.get("num_alloc_retries", 0)),
                cuda_ooms=int(memory.get("num_ooms", 0)) - int(baseline.get("num_ooms", 0)),
                peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            )

        result = fill_cache(
            directory, identity, inputs, provider, max_batches=max_batches, progress=progress
        )
        _require(verify_source(root) == (controller, source), "source_changed_during_embedding")
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare-checkouts", "run", "verify", "export", "import")
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--job-directory", type=Path, default=JOB_PATH)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--max-batches", type=int)
    arguments = parser.parse_args()
    try:
        root = arguments.repository_root.resolve(strict=True)
        directory = _job_directory(root, arguments.job_directory)
        _require(arguments.max_batches is None or arguments.max_batches > 0, "invalid_batch_limit")
        _require(
            arguments.max_batches is None or arguments.action == "run", "unexpected_batch_limit"
        )
        _require(
            arguments.archive is None or arguments.action in {"export", "import"},
            "unexpected_archive",
        )
        if arguments.action == "prepare-checkouts":
            _require(
                "OPENAI_API_KEY" not in os.environ and "DATABASE_URL" not in os.environ,
                "gpu_host_secrets_forbidden",
            )
            verify_source(root)
            tasks, _ = load_tasks(root)
            prepare_checkouts(root, tasks)
            result: dict[str, JsonValue] = {"status": "checkouts_ready", "task_count": len(tasks)}
        elif arguments.action == "run":
            result = run(root, directory, max_batches=arguments.max_batches)
        elif arguments.action == "verify":
            result = verify_cache(directory)
        else:
            _require(arguments.archive is not None, "archive_required")
            archive = arguments.archive.expanduser().absolute()
            result = (
                export_cache(directory, archive)
                if arguments.action == "export"
                else import_cache(archive, directory)
            )
        _emit(**result, openai_requests=0, formal_ready=False)
        return 2 if result.get("status") == "paused" else 0
    except (CacheStoreError, GpuCacheError) as error:
        _emit(status="failed", code=error.code, openai_requests=0)
    except EmbeddingError as error:
        _emit(status="failed", code=error.code.value, openai_requests=0)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        _emit(
            status="failed",
            code="invalid_control_or_io",
            error_type=type(error).__name__,
            openai_requests=0,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
