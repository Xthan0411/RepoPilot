"""Bounded qualification of the imported GPU cache, never a paid-run authorization.

Only the eight pinned issue queries may reach the CPU embedding provider during
retrieval. Corpus vectors are read-only and fail closed on any cache miss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from repopilot.config import DEFAULT_DATABASE_URL
from repopilot.embeddings import (
    EmbeddingIdentity,
    EmbeddingInput,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
)
from repopilot.storage import PostgresVectorStore
from scripts import p7_repository_workload_preflight as workload
from scripts.p7_gpu_cache_acceptance import (
    CacheAcceptanceError,
    verify_coverage,
    verify_pinned_cache,
)
from scripts.p7_gpu_cache_store import CacheStoreError, _digest, _read, _safe, _write_once
from scripts.p7_gpu_cpu_probe import CpuProbeError, create_cpu_provider
from scripts.p7_holdout_readiness_v2 import _checkouts

OUTPUT = Path("results/p7-holdout/gpu-cache-preflight-v1.json")
NUMERIC_OUTPUT = Path("results/p7-holdout/gpu-cache-numeric-v1.json")
CACHE = Path("artifacts/p7-gpu-cache-v1")
NUMERIC_TIMEOUT_SECONDS = 900
RETRIEVAL_TIMEOUT_SECONDS = 7200
_ENVIRONMENT_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "SYSTEMROOT",
)


class PreflightError(RuntimeError):
    """Diagnostics never include source text, vectors, DSNs or credentials."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _progress(value: dict[str, JsonValue]) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def queries_for_tasks(inputs: workload.WorkloadInputs) -> tuple[EmbeddingInput, ...]:
    settings = workload._settings(inputs, "dense-only-v1")
    queries: list[EmbeddingInput] = []
    for task in inputs.agent_inputs.tasks:
        # Match the workflow truncation and DenseRetriever's strip exactly.
        text = f"{task.issue_title}\n{task.issue_body}"[: settings.search_max_query_length].strip()
        queries.append(
            EmbeddingInput(content=text, content_hash=hashlib.sha256(text.encode()).hexdigest())
        )
    return tuple(queries)


class QueryOnlyProvider:
    """Permit four exact queries per task (Dense/Hybrid, cold/warm), no retries."""

    def __init__(self, provider: EmbeddingProvider, queries: tuple[EmbeddingInput, ...]) -> None:
        self.provider = provider
        self.remaining = Counter(q.content_hash for q in queries)
        self.remaining = Counter({key: count * 4 for key, count in self.remaining.items()})
        self.calls = 0

    @property
    def identity(self) -> EmbeddingIdentity:
        return self.provider.identity

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if request.identity != self.identity or request.attempt != 1 or len(request.inputs) != 1:
            raise PreflightError("query_only_embedding_required")
        content_hash = request.inputs[0].content_hash
        if self.remaining[content_hash] <= 0:
            raise PreflightError("query_allowlist_or_call_limit_exceeded")
        self.remaining[content_hash] -= 1
        self.calls += 1
        return self.provider.embed(request)


def worker_environment(*, database: bool) -> dict[str, str]:
    """Do not forward API keys, proxy settings, custom endpoints or model overrides."""
    environment = {key: os.environ[key] for key in _ENVIRONMENT_ALLOWLIST if key in os.environ}
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONWARNINGS="ignore",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        CUDA_VISIBLE_DEVICES="",
    )
    if database:
        environment["DATABASE_URL"] = DEFAULT_DATABASE_URL
    return environment


def run_worker(
    root: Path, command: tuple[str, ...], *, database: bool, timeout_seconds: int
) -> dict[str, JsonValue]:
    """Bound the whole child process group; optional-library stderr is not published."""
    started = time.monotonic()
    with subprocess.Popen(
        command,
        cwd=root,
        env=worker_environment(database=database),
        start_new_session=True,
        stderr=subprocess.DEVNULL,
    ) as child:

        def signal_group(sig: signal.Signals) -> None:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, sig)

        def terminate_group() -> None:
            signal_group(signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # The leader can exit while a descendant ignores SIGTERM.
                signal_group(signal.SIGKILL)
                child.wait()

        try:
            code = child.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            terminate_group()
            return {
                "status": "timeout",
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "temporary_sandbox_cleanup_verified": False,
            }
        except BaseException:
            terminate_group()
            raise
    return {
        "status": "passed" if code == 0 else "failed",
        "exit_code": code,
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def run_retrieval_worker(root: Path) -> dict[str, JsonValue]:
    """Use existing real workload stages, injecting only the read-only GPU cache."""
    controller, source = workload.verify_source(root)
    inputs = workload.load_inputs(root)
    if _checkouts(root, inputs.environment):
        raise PreflightError("checkout_unavailable")
    accepted = verify_pinned_cache(root / CACHE)
    base_provider, runtime = create_cpu_provider(accepted.identity)
    provider = QueryOnlyProvider(base_provider, queries_for_tasks(inputs))
    settings = workload._settings(inputs, "dense-only-v1")
    resources = workload.RetrievalResources(provider, accepted.cache, PostgresVectorStore(settings))
    tasks: list[JsonValue] = []
    passed = True
    for task, binding in zip(inputs.agent_inputs.tasks, inputs.environment.bindings, strict=True):
        _progress({"stage": "retrieval_task_started", "task_id": task.task_id})
        evidence = workload.qualify_task(
            root,
            inputs,
            task,
            binding,
            checkout_available=True,
            prepare_retrieval=True,
            resources=resources,
        )
        tasks.append(evidence.model_dump(mode="json"))
        task_passed = all(stage.status == "passed" for stage in evidence.stages)
        passed = passed and task_passed
        _progress(
            {"stage": "retrieval_task_finished", "task_id": task.task_id, "passed": task_passed}
        )
        if not task_passed:
            # Retain this task's stage diagnostics, but avoid costly downstream work.
            break
    if workload.verify_source(root) != (controller, source) or workload.load_inputs(root) != inputs:
        raise PreflightError("source_or_inputs_changed")
    if _checkouts(root, inputs.environment):
        raise PreflightError("checkout_changed")
    verify_pinned_cache(root / CACHE)
    return {
        "retrieval_preflight_passed": passed and len(tasks) == 8 and provider.calls == 32,
        "tasks": tasks,
        "cpu_runtime": runtime,
        "query_embedding_calls": provider.calls,
        "source_embedding_calls": 0,
        "local_cache_writes_enabled": False,
        "corpus_cache_hits": accepted.cache.hits,
        "corpus_cache_misses": accepted.cache.misses,
        "corpus_cache_write_attempts": accepted.cache.write_attempts,
        "database_writes_enabled": True,
        "provider_requests_made": 0,
        "paid_requests_made": 0,
        "agent_constructed": False,
    }


def qualify(root: Path, *, prepare_retrieval: bool) -> dict[str, JsonValue]:
    controller, source = workload.verify_source(root)
    inputs = workload.load_inputs(root)
    report: dict[str, JsonValue] = {
        "preflight_version": "gpu-cache-preflight-v1",
        "controller_commit": controller,
        "checked_at": datetime.now(UTC).isoformat(),
        "execution_source_sha256": _digest(source),
        "matrix_sha256": inputs.matrix.content_sha256,
        "prepare_retrieval": prepare_retrieval,
        "formal_ready": False,
        "formal_blockers": ["new_source_freeze_and_exact_plan_authorization_required"],
        "provider_requests_made": 0,
        "paid_requests_made": 0,
        "agent_constructed": False,
        "preflight_passed": False,
    }
    coverage = verify_coverage(root, progress=_progress)
    report["coverage"] = coverage.report
    request: dict[str, JsonValue] = {
        "identity": coverage.cache.identity.model_dump(mode="json"),
        "samples": [
            {"input": sample.model_dump(mode="json"), "gpu_vector": list(vector.values)}
            for sample, vector in zip(coverage.samples, coverage.gpu_vectors, strict=True)
        ],
        "queries": [query.model_dump(mode="json") for query in queries_for_tasks(inputs)],
    }
    with tempfile.TemporaryDirectory(prefix="repopilot-gpu-acceptance-") as temporary:
        directory = Path(temporary).resolve()
        request_path = directory / "request.json"
        # This private, ephemeral file contains source samples; it is never an artifact.
        _write_once(request_path, request)
        numeric_path = directory / "numeric.json"
        _progress(
            {
                "stage": "cpu_numeric_and_query_probe_started",
                "timeout_seconds": NUMERIC_TIMEOUT_SECONDS,
            }
        )
        numeric = run_worker(
            root,
            (
                sys.executable,
                "-m",
                "scripts.p7_gpu_cpu_probe",
                "--request",
                str(request_path),
                "--output",
                str(numeric_path),
            ),
            database=False,
            timeout_seconds=NUMERIC_TIMEOUT_SECONDS,
        )
        report["numeric_worker"] = numeric
        if numeric_path.is_file():
            if numeric_path.stat().st_size > 128 * 1024:
                raise PreflightError("invalid_numeric_output")
            result = json.loads(numeric_path.read_bytes())
            if not isinstance(result, dict):
                raise PreflightError("invalid_numeric_output")
            report["cpu_probe"] = result
        else:
            result = {}
        if numeric["status"] == "passed" and result.get("cpu_probe_passed") is True:
            if prepare_retrieval:
                retrieval_path = directory / "retrieval.json"
                _progress(
                    {
                        "stage": "retrieval_preflight_started",
                        "timeout_seconds": RETRIEVAL_TIMEOUT_SECONDS,
                    }
                )
                worker = run_worker(
                    root,
                    (
                        sys.executable,
                        "-m",
                        "scripts.p7_gpu_cache_preflight",
                        "--repository-root",
                        str(root),
                        "--retrieval-worker-output",
                        str(retrieval_path),
                    ),
                    database=True,
                    timeout_seconds=RETRIEVAL_TIMEOUT_SECONDS,
                )
                report["retrieval_worker"] = worker
                if retrieval_path.is_file():
                    retrieval = _read(retrieval_path)
                    report["retrieval"] = retrieval
                    report["preflight_passed"] = (
                        worker["status"] == "passed"
                        and retrieval.get("retrieval_preflight_passed") is True
                    )
            else:
                report["numerical_qualification_passed"] = True
    # Revalidate immutable bindings even on an ordinary worker failure or timeout.
    verify_pinned_cache(root / CACHE)
    if workload.verify_source(root) != (controller, source) or workload.load_inputs(root) != inputs:
        raise PreflightError("source_or_inputs_changed")
    if _checkouts(root, inputs.environment):
        raise PreflightError("checkout_changed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--prepare-retrieval", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="New results/p7-holdout/*.json only; never overwritten"
    )
    parser.add_argument("--retrieval-worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output: Path | None = None
    try:
        root = args.repository_root.resolve(strict=True)
        if any(name.startswith("REPOPILOT_") for name in os.environ):
            raise PreflightError("unapproved_environment_override")
        if args.retrieval_worker_output is not None:
            output = _safe(args.retrieval_worker_output)
            if (
                os.environ.get("OPENAI_API_KEY")
                or os.environ.get("DATABASE_URL") != DEFAULT_DATABASE_URL
            ):
                raise PreflightError("invalid_worker_environment")
            report = run_retrieval_worker(root)
            _write_once(output, report)
            return 0 if report["retrieval_preflight_passed"] else 1
        relative = args.output or (OUTPUT if args.prepare_retrieval else NUMERIC_OUTPUT)
        if (
            relative.is_absolute()
            or relative.parent != Path("results/p7-holdout")
            or relative.suffix != ".json"
        ):
            raise PreflightError("unsafe_output_path")
        output = _safe(root / relative)
        if output.exists():
            raise PreflightError("output_conflict")
        report = qualify(root, prepare_retrieval=args.prepare_retrieval)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_once(output, report)
        passed = (
            report.get("preflight_passed") is True
            if args.prepare_retrieval
            else report.get("numerical_qualification_passed") is True
        )
        _progress({"gpu_cache_preflight_passed": passed, "paid_requests": 0, "formal_ready": False})
        return 0 if passed else 1
    except Exception as error:
        # Codes from our own errors are fixed literals; never echo foreign exception text.
        code = (
            error.code
            if isinstance(
                error, (PreflightError, CacheAcceptanceError, CacheStoreError, CpuProbeError)
            )
            else "qualification_failed"
        )
        if output is not None and not output.exists() and code != "output_conflict":
            failure: dict[str, JsonValue] = {
                "preflight_version": "gpu-cache-preflight-v1",
                "preflight_passed": False,
                "failure_code": code,
                "error_type": type(error).__name__,
                "formal_ready": False,
                "provider_requests_made": 0,
                "paid_requests_made": 0,
            }
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                _write_once(output, failure)
            except (OSError, CacheStoreError):
                pass
        _progress(
            {
                "gpu_cache_preflight_passed": False,
                "failure_code": code,
                "paid_requests": 0,
                "formal_ready": False,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
