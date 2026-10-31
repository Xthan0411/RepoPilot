"""Bind the qualified GPU corpus cache to the unchanged CPU query runtime.

This is preparation/evidence plumbing, not authorization. Dynamic Agent queries
are allowed; corpus cache misses fail before any source embedding can occur.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from repopilot.embeddings import (
    EmbeddingIdentity,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
)
from scripts import p7_repository_workload_preflight as workload
from scripts.p7_gpu_cache_acceptance import (
    EXPECTED_COMPLETE_SHA256,
    EXPECTED_INPUTS_SHA256,
    EXPECTED_JOB_SHA256,
    ReadOnlyGpuCache,
    verify_pinned_cache,
)
from scripts.p7_gpu_cache_store import _digest
from scripts.p7_gpu_cpu_probe import PINNED_IDENTITY, create_cpu_provider
from scripts.p7_holdout_control import EXPECTED_TASK_IDS
from scripts.p7_holdout_requalification import _git, _read

QUALIFICATION_PATH = Path("results/p7-holdout/gpu-cache-preflight-v2.json")
QUALIFICATION_RAW_SHA256 = "0f46c7c1c618cdfe4423df7ca7d9eee1a77e1bf2d10cb552db39ecb0719b9b2c"
QUALIFIED_COMMIT = "bfdecc3b4e1d1c6e56cd871f114f1c5fbccf2428"
CACHE_PATH = Path("artifacts/p7-gpu-cache-v1")
AUDITED_RUNNER_PATH = "scripts/p7_holdout_runner.py"
# Reviewed optional embedding-resource injection and historical-budget carry only.
AUDITED_RUNNER_SHA256 = "46ed9afd1d7887a33b09981cf879b84a4bcdcc6edf2ba0de68f9043354d6e42d"
NEW_EXECUTION_PATHS = frozenset(
    {
        "scripts/p7_gpu_execution_profile.py",
        "scripts/p7_holdout_gpu.py",
        "scripts/p7_holdout_readiness_v3.py",
        "scripts/p7_holdout_runner_v3.py",
    }
)


def _require(value: bool, code: str) -> None:
    if not value:
        raise ValueError(code)


def _validate_report(report: dict[str, Any]) -> None:
    """Validate recorded facts without interpreting them as current readiness."""
    _require(
        report.get("sha256") == _digest({k: v for k, v in report.items() if k != "sha256"}),
        "qualification_seal_mismatch",
    )
    _require(
        report.get("controller_commit") == QUALIFIED_COMMIT
        and report.get("matrix_sha256") == workload.MATRIX_SHA256
        and report.get("preflight_passed") is True
        and report.get("prepare_retrieval") is True
        and report.get("formal_ready") is False
        and report.get("agent_constructed") is False
        and report.get("paid_requests_made") == 0
        and report.get("provider_requests_made") == 0,
        "qualification_not_accepted",
    )
    coverage, cpu, retrieval = report["coverage"], report["cpu_probe"], report["retrieval"]
    _require(
        coverage["coverage_passed"] is True
        and coverage["task_count"] == 8
        and coverage["total_vectors"] == coverage["completed_vectors"] == 103236
        and coverage["job_sha256"] == EXPECTED_JOB_SHA256
        and coverage["inputs_sha256"] == EXPECTED_INPUTS_SHA256
        and coverage["complete_sha256"] == EXPECTED_COMPLETE_SHA256
        and coverage["gpu_identity"] == PINNED_IDENTITY.model_dump(mode="json")
        and coverage["cache_misses"] == coverage["cache_write_attempts"] == 0,
        "qualified_cache_identity_mismatch",
    )
    _require(
        cpu["cpu_probe_passed"] is True
        and len(cpu["samples"]) == 3
        and all(sample["passed"] is True for sample in cpu["samples"])
        and len(cpu["queries"]) == 8
        and cpu["runtime"] == retrieval["cpu_runtime"]
        and retrieval["retrieval_preflight_passed"] is True
        and retrieval["query_embedding_calls"] == 32
        and retrieval["source_embedding_calls"] == 0
        and retrieval["corpus_cache_misses"] == retrieval["corpus_cache_write_attempts"] == 0
        and retrieval["local_cache_writes_enabled"] is False
        and retrieval["paid_requests_made"] == retrieval["provider_requests_made"] == 0,
        "qualified_runtime_mismatch",
    )
    for worker in (report["numeric_worker"], report["retrieval_worker"]):
        _require(worker["status"] == "passed" and worker["exit_code"] == 0, "worker_failed")
    _require(
        tuple(item["task_id"] for item in coverage["tasks"]) == EXPECTED_TASK_IDS
        and tuple(item["task_id"] for item in retrieval["tasks"]) == EXPECTED_TASK_IDS,
        "qualification_task_set_mismatch",
    )
    for task in retrieval["tasks"]:
        _require(
            tuple(stage["stage"] for stage in task["stages"]) == workload.STAGES
            and all(
                stage["status"] == "passed" and stage["code"] == "ok" and stage["failure"] is None
                for stage in task["stages"]
            ),
            "qualification_stage_failed",
        )
        metrics = task["stages"][-1]["metrics"]
        _require(
            metrics["setup_offline"] is True
            and metrics["temporary_sandboxes_destroyed"] is True
            and metrics["gold_visible_to_agent"] is False
            and metrics["warm_preparation_within_deadline"] is True
            and metrics["paid_requests_made"] == metrics["provider_requests_made"] == 0,
            "qualification_runtime_failed",
        )


def load_qualification_report(root: Path) -> dict[str, Any]:
    raw = _read(root, QUALIFICATION_PATH, 128_000)
    _require(
        hashlib.sha256(raw).hexdigest() == QUALIFICATION_RAW_SHA256, "qualification_raw_mismatch"
    )
    report: dict[str, Any] = json.loads(raw)
    _validate_report(report)
    return report


def _source_objects_at(root: Path, commit: str) -> dict[str, str]:
    objects: dict[str, str] = {}
    for entry in _git(root, "ls-tree", "-rz", commit, "--", *workload.SOURCE_PATHS).split("\0"):
        if entry:
            metadata, relative = entry.split("\t", 1)
            mode, kind, blob = metadata.split()
            _require(mode in {"100644", "100755"} and kind == "blob", "unsupported_source")
            objects[relative] = blob
    return dict(sorted(objects.items()))


def verify_qualification(root: Path) -> dict[str, Any]:
    """Keep qualified semantics fixed; only the audited execution wiring may differ."""
    report = load_qualification_report(root)
    controller, current = workload.verify_source(root)
    _git(root, "merge-base", "--is-ancestor", QUALIFIED_COMMIT, controller)
    qualified = _source_objects_at(root, QUALIFIED_COMMIT)
    _require(_digest(qualified) == report["execution_source_sha256"], "qualified_source_mismatch")
    _require(set(current) - set(qualified) == NEW_EXECUTION_PATHS, "unapproved_execution_paths")
    _require(set(qualified).issubset(current), "qualified_source_removed")
    _require(
        all(
            current[path] == blob for path, blob in qualified.items() if path != AUDITED_RUNNER_PATH
        ),
        "qualified_semantics_changed",
    )
    _require(
        hashlib.sha256(_read(root, Path(AUDITED_RUNNER_PATH))).hexdigest() == AUDITED_RUNNER_SHA256,
        "unaudited_runner_patch",
    )
    inputs = workload.load_inputs(root)
    expected = tuple((t.task_id, t.base_commit) for t in inputs.agent_inputs.tasks)
    for collection in (report["coverage"]["tasks"], report["retrieval"]["tasks"]):
        _require(
            tuple((t["task_id"], t["base_commit"]) for t in collection) == expected,
            "task_binding_mismatch",
        )
    return report


def verify_cpu_runtime(root: Path, runtime: dict[str, Any]) -> None:
    _require(
        runtime == load_qualification_report(root)["cpu_probe"]["runtime"], "cpu_runtime_drift"
    )


class SingleQueryCpuProvider:
    """Keep the qualified batch shape, without a preflight-only query allowlist."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    @property
    def identity(self) -> EmbeddingIdentity:
        return self.provider.identity

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        _require(
            request.identity == self.identity and len(request.inputs) == 1,
            "single_cpu_query_required",
        )
        return self.provider.embed(request)


def build_gpu_resources(root: Path) -> tuple[EmbeddingProvider, ReadOnlyGpuCache]:
    verify_qualification(root)
    accepted = verify_pinned_cache(root / CACHE_PATH, PINNED_IDENTITY)
    provider, runtime = create_cpu_provider(accepted.identity)
    verify_cpu_runtime(root, runtime)
    return SingleQueryCpuProvider(provider), accepted.cache
