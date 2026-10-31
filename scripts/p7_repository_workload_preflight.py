"""Audit all eight real repositories before a newly frozen Holdout execution.

Default mode only scans, chunks, and runs BM25 in memory. --prepare-retrieval also
embeds the full corpus with offline Jina, writes the local cache and project pgvector
namespaces, exercises Dense/Hybrid twice, and probes the offline runtime setup. Neither
mode executes an Agent, calls an LLM, reads solution patches, or authorizes a paid run.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from repopilot.config import DEFAULT_DATABASE_URL, RepoPilotSettings
from repopilot.embeddings import (
    EmbeddingProvider,
    LocalEmbeddingCache,
    embedding_cache_from_settings,
    embedding_provider_from_settings,
)
from repopilot.evals import ExperimentMatrix, load_experiment_matrix, select_experiment_arm
from repopilot.repair_workflow import (
    FailureStage,
    RepairFailureDetail,
    RepairRetrievalContext,
    build_repair_retrieval_context,
    repository_id_for_task,
    safe_repair_failure_detail,
)
from repopilot.retrieval import RepositoryIngestScanPolicy, ingest_repository
from repopilot.retrieval.bm25 import BM25Policy, BM25Retriever
from repopilot.retrieval.corpus import ChunkCorpus, RepositoryChunkCorpusBuilder
from repopilot.retrieval.manifest import RepositoryManifest
from repopilot.schemas import GitSha, Identifier, Sha256, StrictModel, TaskId, TaskSpec
from repopilot.storage import PostgresVectorStore, VectorNamespace, VectorStore
from repopilot.tools.repository import ReadOnlyRepository
from scripts.p7_holdout_agent_inputs import (
    AGENT_INPUTS_PATH,
    HoldoutAgentInputs,
    _canonical_sha256,
    _validate_content_digest,
)
from scripts.p7_holdout_control import (
    ENVIRONMENT_PATH,
    EXPECTED_TASK_IDS,
    MATRIX_PATH,
    HoldoutControlError,
    HoldoutEnvironmentBinding,
    HoldoutEnvironmentManifest,
    write_control_artifact,
)
from scripts.p7_holdout_readiness_v2 import _checkouts, _safe_path
from scripts.p7_holdout_requalification import _read, _sealed
from scripts.p7_holdout_test_inputs import (
    TEST_INPUTS_RAW_SHA256,
    HoldoutTestInputs,
    load_test_inputs,
)

SCAN_RESULT_PATH = Path("results/p7-holdout/repository-workload-scan-v1.json")
PREPARED_RESULT_PATH = Path("results/p7-holdout/repository-workload-preflight-v1.json")
AGENT_INPUTS_SHA256 = "2a3a197ea7318d42e4b681db8b56b554fc241c19923ba8482566a4bdc9c7661f"
ENVIRONMENT_SHA256 = "b7c6f04283433859e462edb7105ad5411700703fa3557b1281fafc3ccb99890d"
MATRIX_SHA256 = "d75d36d711f23190062aded9e97a8f5abfaba859416466027b227421ea7407e8"
SOURCE_PATHS = ("scripts", "src/repopilot", "configs", "migrations", "pyproject.toml", "uv.lock")
INGEST_SCAN_POLICY = RepositoryIngestScanPolicy()
STAGES = (
    "ingestion",
    "chunking",
    "bm25",
    "dense",
    "hybrid",
    "warm_dense",
    "warm_hybrid",
    "runtime",
)
StageName = Literal[
    "ingestion", "chunking", "bm25", "dense", "hybrid", "warm_dense", "warm_hybrid", "runtime"
]


class WorkloadStage(StrictModel):
    stage: StageName
    status: Literal["passed", "failed", "blocked"]
    code: Literal[
        "ok",
        "stage_failure",
        "checkout_unavailable",
        "upstream_failed",
        "not_requested",
        "warm_preparation_deadline_exhausted",
    ]
    elapsed_ms: int = Field(ge=0)
    failure: RepairFailureDetail | None = None
    metrics: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        expected = {
            "ok": "passed",
            "stage_failure": "failed",
            "checkout_unavailable": "failed",
            "upstream_failed": "blocked",
            "not_requested": "blocked",
            "warm_preparation_deadline_exhausted": "failed",
        }[self.code]
        if self.status != expected or (self.failure is not None and self.code != "stage_failure"):
            raise ValueError("inconsistent workload stage")
        return self


class TaskWorkloadEvidence(StrictModel):
    task_id: TaskId
    base_commit: GitSha
    repository_id: Identifier
    stages: tuple[WorkloadStage, ...]

    @model_validator(mode="after")
    def validate_stages(self) -> Self:
        if tuple(item.stage for item in self.stages) != STAGES:
            raise ValueError("workload stages must be complete and ordered")
        return self


class RepositoryWorkloadPreflight(StrictModel):
    preflight_version: Literal["repository-workload-preflight-v1"] = (
        "repository-workload-preflight-v1"
    )
    controller_commit: GitSha
    checked_at: AwareDatetime
    execution_source_objects: dict[str, GitSha]
    execution_source_sha256: Sha256
    agent_inputs_raw_sha256: Sha256
    environment_raw_sha256: Sha256
    matrix_sha256: Sha256
    test_inputs_raw_sha256: Sha256
    prepare_retrieval: bool
    local_cache_database_writes_enabled: bool
    tasks: tuple[TaskWorkloadEvidence, ...]
    scan_passed: bool
    retrieval_preflight_passed: bool
    agent_constructed: Literal[False] = False
    provider_requests_made: Literal[0] = 0
    paid_requests_made: Literal[0] = 0
    source_rows_loaded: Literal[False] = False
    gold_visible_to_agent: Literal[False] = False
    formal_environment_ready: Literal[False] = False
    formal_blockers: tuple[Literal["new_source_freeze_authorization_required"], ...] = (
        "new_source_freeze_authorization_required",
    )
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        scan = all(stage.status == "passed" for task in self.tasks for stage in task.stages[:3])
        full = all(stage.status == "passed" for task in self.tasks for stage in task.stages)
        if (
            tuple(item.task_id for item in self.tasks) != EXPECTED_TASK_IDS
            or self.scan_passed != scan
            or self.retrieval_preflight_passed != (self.prepare_retrieval and full)
            or self.local_cache_database_writes_enabled != self.prepare_retrieval
            or self.execution_source_sha256 != _canonical_sha256(self.execution_source_objects)
            or self.agent_inputs_raw_sha256 != AGENT_INPUTS_SHA256
            or self.environment_raw_sha256 != ENVIRONMENT_SHA256
            or self.matrix_sha256 != MATRIX_SHA256
            or self.test_inputs_raw_sha256 != TEST_INPUTS_RAW_SHA256
            or not self.execution_source_objects
        ):
            raise ValueError("workload report identity mismatch")
        _validate_content_digest(self)
        return self


@dataclass(frozen=True)
class WorkloadInputs:
    matrix: ExperimentMatrix
    environment: HoldoutEnvironmentManifest
    agent_inputs: HoldoutAgentInputs
    tests: HoldoutTestInputs


@dataclass(frozen=True)
class RetrievalResources:
    provider: EmbeddingProvider
    cache: LocalEmbeddingCache
    store: VectorStore


def load_inputs(root: Path) -> WorkloadInputs:
    agent_raw, environment_raw = _read(root, AGENT_INPUTS_PATH), _read(root, ENVIRONMENT_PATH)
    if (
        hashlib.sha256(agent_raw).hexdigest() != AGENT_INPUTS_SHA256
        or hashlib.sha256(environment_raw).hexdigest() != ENVIRONMENT_SHA256
    ):
        raise ValueError("workload_input_identity_mismatch")
    agent = HoldoutAgentInputs.model_validate_json(agent_raw)
    environment = HoldoutEnvironmentManifest.model_validate_json(environment_raw)
    matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    if (
        matrix.content_sha256 != MATRIX_SHA256
        or environment.matrix_sha256 != MATRIX_SHA256
        or agent.matrix_sha256 != MATRIX_SHA256
        or environment.runtime_command_contract_sha256 != agent.runtime_command_contract_sha256
        or tuple((b.task_id, b.base_commit) for b in environment.bindings)
        != tuple((t.task_id, t.base_commit) for t in agent.tasks)
    ):
        raise ValueError("workload_input_binding_mismatch")
    return WorkloadInputs(
        matrix,
        environment,
        agent,
        load_test_inputs(root, agent_inputs=agent, expected_raw_sha256=TEST_INPUTS_RAW_SHA256),
    )


def verify_source(root: Path) -> tuple[str, dict[str, str]]:
    def git(*args: str) -> str:
        return subprocess.run(
            ("git", *args), cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    if (
        Path(git("rev-parse", "--show-toplevel")).resolve() != root
        or git("status", "--porcelain=v1", "--untracked-files=no")
        or git("status", "--porcelain=v1", "--untracked-files=all", "--", *SOURCE_PATHS)
    ):
        raise ValueError("workload_source_not_clean")
    objects: dict[str, str] = {}
    for entry in git("ls-tree", "-rz", "HEAD", "--", *SOURCE_PATHS).split("\0"):
        if not entry:
            continue
        metadata, name = entry.split("\t", 1)
        mode, kind, blob = metadata.split()
        raw = _read(root, Path(name))
        actual = hashlib.sha1(
            f"blob {len(raw)}\0".encode() + raw, usedforsecurity=False
        ).hexdigest()
        if mode not in {"100644", "100755"} or kind != "blob" or blob != actual:
            raise ValueError("workload_source_bytes_changed")
        objects[name] = blob
    if "scripts/p7_repository_workload_preflight.py" not in objects:
        raise ValueError("workload_controller_not_committed")
    return git("rev-parse", "HEAD"), dict(sorted(objects.items()))


def _settings(inputs: WorkloadInputs, arm: str) -> RepoPilotSettings:
    return select_experiment_arm(inputs.matrix, arm).settings(
        openai_api_key="workload-preflight-never-calls-llm",
        database_url=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
    )


def _blocked(stage: StageName, *, requested: bool = True) -> WorkloadStage:
    return WorkloadStage(
        stage=stage,
        status="blocked",
        code="upstream_failed" if requested else "not_requested",
        elapsed_ms=0,
    )


def _stage[T](
    stage: StageName, operation: Callable[[], T], metrics: Callable[[T], dict[str, JsonValue]]
) -> tuple[WorkloadStage, T | None]:
    print(f"repository_workload_stage_started={stage}", flush=True)
    started = time.perf_counter()
    try:
        value = operation()
        details = metrics(value)
    except Exception as error:
        failure_stage: FailureStage = (
            "ingest" if stage == "ingestion" else "chunking" if stage == "chunking" else "retrieval"
        )
        failure = safe_repair_failure_detail(error, stage=failure_stage)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        print(
            f"repository_workload_stage_finished={stage} status=failed "
            f"cause={failure.cause_code} elapsed_ms={elapsed_ms}",
            flush=True,
        )
        return WorkloadStage(
            stage=stage,
            status="failed",
            code="stage_failure",
            elapsed_ms=elapsed_ms,
            failure=failure,
        ), None
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    print(
        f"repository_workload_stage_finished={stage} status=passed elapsed_ms={elapsed_ms}",
        flush=True,
    )
    return WorkloadStage(
        stage=stage,
        status="passed",
        code="ok",
        elapsed_ms=elapsed_ms,
        metrics=details,
    ), value


def _corpus_metrics(corpus: ChunkCorpus) -> dict[str, JsonValue]:
    return {
        "chunk_count": len(corpus.chunks),
        "source_count": len(corpus.sources),
        "unique_content_count": len({c.content_hash for c in corpus.chunks}),
        "maximum_chunk_characters": max((len(c.content) for c in corpus.chunks), default=0),
        "corpus_sha256": _canonical_sha256(corpus),
    }


def _context_metrics(context: RepairRetrievalContext) -> dict[str, JsonValue]:
    return {
        "repository_content_sha256": context.repository_content_sha256,
        "hit_count": len(context.hits),
        "indexing_latency_ms": context.indexing_latency_ms,
        "retrieval_latency_ms": context.retrieval_latency_ms,
        "hit_ids_sha256": _canonical_sha256(tuple(hit.chunk_id for hit in context.hits)),
    }


def _build_retrieval(
    repository: ReadOnlyRepository,
    task: TaskSpec,
    settings: RepoPilotSettings,
    resources: RetrievalResources,
    corpus: ChunkCorpus,
) -> RepairRetrievalContext:
    context = build_repair_retrieval_context(
        repository,
        task,
        settings,
        embedding_provider=resources.provider,
        embedding_cache=resources.cache,
        vector_store=resources.store,
    )
    identity = resources.provider.identity
    namespace = VectorNamespace(
        repository_id=corpus.repository_id,
        commit=corpus.commit,
        embedding_model=identity.model,
        embedding_version=identity.fingerprint,
        dimension=identity.dimension,
    )
    records = resources.store.list_records(namespace)
    if {(r.chunk_id, r.content_hash) for r in records} != {
        (c.chunk_id, c.content_hash) for c in corpus.chunks
    }:
        raise ValueError("vector_readback_identity_mismatch")
    return context


def _ingest_task(
    task: TaskSpec, settings: RepoPilotSettings
) -> tuple[ReadOnlyRepository, RepositoryManifest]:
    repository = ReadOnlyRepository.from_settings(Path(task.repository_path or ""), settings)
    return repository, ingest_repository(
        repository,
        repository_id=repository_id_for_task(task),
        commit=task.base_commit,
        scan_policy=INGEST_SCAN_POLICY,
    )


def _ingest_metrics(pair: tuple[ReadOnlyRepository, RepositoryManifest]) -> dict[str, JsonValue]:
    repository, manifest = pair
    return {
        "manifest_sha256": manifest.content_sha256,
        "file_count": len(manifest.files),
        "source_bytes": sum(f.size_bytes for f in manifest.files),
        "skipped_count": len(manifest.skipped_sources),
        "skipped_reasons": {
            key: count
            for key, count in Counter(s.reason.value for s in manifest.skipped_sources).items()
        },
        "agent_tool_listing_limit": repository.policy.max_list_entries,
        "internal_scan_listing_limit": INGEST_SCAN_POLICY.max_list_entries,
    }


def qualify_task(
    root: Path,
    inputs: WorkloadInputs,
    task: TaskSpec,
    binding: HoldoutEnvironmentBinding,
    *,
    checkout_available: bool,
    prepare_retrieval: bool,
    resources: RetrievalResources | RepairFailureDetail | None,
) -> TaskWorkloadEvidence:
    settings = _settings(inputs, "dense-only-v1")
    task = task.model_copy(
        update={
            "repository_path": str(root / binding.checkout_relative_path),
            "repository_url": None,
        }
    )
    # Failed checkouts have no usable repository namespace. Do not resolve their
    # missing paths before the stage guard, or one missing checkout aborts all eight.
    repository_id = (
        "unavailable_"
        + _canonical_sha256({"task_id": task.task_id, "base_commit": task.base_commit})[:24]
    )
    stages: list[WorkloadStage] = []
    repository: ReadOnlyRepository | None = None
    manifest: RepositoryManifest | None = None
    if not checkout_available:
        stages.append(
            WorkloadStage(
                stage="ingestion", status="failed", code="checkout_unavailable", elapsed_ms=0
            )
        )
    else:
        scan_result = _stage("ingestion", lambda: _ingest_task(task, settings), _ingest_metrics)
        result, pair = scan_result
        stages.append(result)
        if pair is not None:
            repository, manifest = pair
            repository_id = manifest.repository_id
    corpus: ChunkCorpus | None = None
    if manifest is not None and repository is not None:
        chunk_result = _stage(
            "chunking",
            lambda: (
                RepositoryChunkCorpusBuilder.from_settings(repository, settings)
                .build(manifest)
                .corpus
            ),
            _corpus_metrics,
        )
        result, corpus = chunk_result
        stages.append(result)
    else:
        stages.append(_blocked("chunking"))
    if corpus is not None:
        result, _ = _stage(
            "bm25",
            lambda: BM25Retriever.from_corpus(
                corpus, policy=BM25Policy.from_settings(settings)
            ).search(
                repository_id=repository_id,
                commit=task.base_commit,
                query=f"{task.issue_title}\n{task.issue_body}"[: settings.search_max_query_length],
                top_k=settings.retrieval_default_top_k,
            ),
            lambda result: {"hit_count": len(result.hits)},
        )
        stages.append(result)
    else:
        stages.append(_blocked("bm25"))
    contexts: dict[str, RepairRetrievalContext] = {}
    retrieval_stages: tuple[tuple[StageName, str], ...] = (
        ("dense", "dense-only-v1"),
        ("hybrid", "hybrid-rrf-v1"),
        ("warm_dense", "dense-only-v1"),
        ("warm_hybrid", "hybrid-rrf-v1"),
    )
    for stage, arm in retrieval_stages:
        if not prepare_retrieval or corpus is None or repository is None or resources is None:
            stages.append(_blocked(stage, requested=prepare_retrieval))
            continue
        if isinstance(resources, RepairFailureDetail):
            stages.append(
                WorkloadStage(
                    stage=stage,
                    status="failed",
                    code="stage_failure",
                    elapsed_ms=0,
                    failure=resources,
                )
            )
            continue
        if stage.startswith("warm_") and stage.removeprefix("warm_") not in contexts:
            stages.append(_blocked(stage))
            continue
        context_result = _stage(
            stage,
            partial(
                _build_retrieval,
                repository,
                task,
                _settings(inputs, arm),
                resources,
                corpus,
            ),
            lambda context: {
                **_context_metrics(context),
                "vector_record_count": len(corpus.chunks),
                "vector_readback_verified": True,
            },
        )
        result, context = context_result
        stages.append(result)
        if context is not None:
            contexts[stage] = context
    if prepare_retrieval and "warm_dense" in contexts:
        from scripts.p7_workload_runtime_probe import probe_runtime

        patch = next(
            item.regression_patch for item in inputs.tests.tasks if item.task_id == task.task_id
        )
        result, _ = _stage(
            "runtime",
            lambda: probe_runtime(
                root=root,
                task=task,
                settings=settings,
                binding=binding,
                regression_patch=patch,
                retrieval=contexts["warm_dense"],
            ),
            lambda value: value,
        )
        warm_ms = max(s.elapsed_ms for s in stages if s.stage in {"warm_dense", "warm_hybrid"})
        total_ms = warm_ms + result.elapsed_ms
        if result.status == "passed":
            result = result.model_copy(
                update={
                    "metrics": {
                        **result.metrics,
                        "warm_preparation_elapsed_ms": total_ms,
                        "task_budget_ms": settings.timeout_seconds * 1000,
                        "remaining_task_budget_ms": max(
                            0, settings.timeout_seconds * 1000 - total_ms
                        ),
                        "warm_preparation_within_deadline": total_ms
                        < settings.timeout_seconds * 1000,
                        "remaining_budget_guarantees_agent_completion": False,
                        "new_process_model_cold_load_included": False,
                    }
                }
            )
            if total_ms >= settings.timeout_seconds * 1000:
                result = result.model_copy(
                    update={"status": "failed", "code": "warm_preparation_deadline_exhausted"}
                )
        stages.append(result)
    else:
        stages.append(_blocked("runtime", requested=prepare_retrieval))
    return TaskWorkloadEvidence(
        task_id=task.task_id,
        base_commit=task.base_commit,
        repository_id=repository_id,
        stages=tuple(stages),
    )


@contextmanager
def _offline_environment() -> Iterator[None]:
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "OPENAI_API_KEY")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1"
        )
        os.environ.pop("OPENAI_API_KEY", None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_preflight(root: Path, *, prepare_retrieval: bool = False) -> RepositoryWorkloadPreflight:
    inputs = load_inputs(root)
    controller, source = verify_source(root)
    unavailable = set(_checkouts(root, inputs.environment))
    resources: RetrievalResources | RepairFailureDetail | None = None
    if prepare_retrieval:
        settings = _settings(inputs, "dense-only-v1")
        try:
            resources = RetrievalResources(
                embedding_provider_from_settings(settings),
                embedding_cache_from_settings(settings, project_root=root),
                PostgresVectorStore(settings),
            )
        except Exception as error:
            resources = safe_repair_failure_detail(error, stage="embedding")
    tasks: list[TaskWorkloadEvidence] = []
    with _offline_environment():
        for task, binding in zip(
            inputs.agent_inputs.tasks, inputs.environment.bindings, strict=True
        ):
            print(f"repository_workload_started={task.task_id}", flush=True)
            tasks.append(
                qualify_task(
                    root,
                    inputs,
                    task,
                    binding,
                    checkout_available=task.task_id not in unavailable,
                    prepare_retrieval=prepare_retrieval,
                    resources=resources,
                )
            )
            print(f"repository_workload_completed={task.task_id}", flush=True)
    if (
        verify_source(root) != (controller, source)
        or load_inputs(root) != inputs
        or set(_checkouts(root, inputs.environment)) != unavailable
    ):
        raise ValueError("workload_inputs_changed_during_probe")
    payload = {
        "controller_commit": controller,
        "checked_at": datetime.now(UTC),
        "execution_source_objects": source,
        "execution_source_sha256": _canonical_sha256(source),
        "agent_inputs_raw_sha256": AGENT_INPUTS_SHA256,
        "environment_raw_sha256": ENVIRONMENT_SHA256,
        "matrix_sha256": MATRIX_SHA256,
        "test_inputs_raw_sha256": TEST_INPUTS_RAW_SHA256,
        "prepare_retrieval": prepare_retrieval,
        "local_cache_database_writes_enabled": prepare_retrieval,
        "tasks": tuple(tasks),
        "scan_passed": all(s.status == "passed" for t in tasks for s in t.stages[:3]),
        "retrieval_preflight_passed": prepare_retrieval
        and all(s.status == "passed" for t in tasks for s in t.stages),
    }
    return _sealed(RepositoryWorkloadPreflight, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--prepare-retrieval", action="store_true")
    args = parser.parse_args()
    try:
        root = args.repository_root.resolve(strict=True)
        if any(name.startswith("REPOPILOT_") for name in os.environ):
            raise ValueError("unapproved_environment_override")
        output = _safe_path(
            root, PREPARED_RESULT_PATH if args.prepare_retrieval else SCAN_RESULT_PATH
        )
        if output.exists():
            raise ValueError("output_conflict")
        report = run_preflight(root, prepare_retrieval=args.prepare_retrieval)
        write_control_artifact(output, report)
    except (ValueError, OSError, subprocess.SubprocessError, HoldoutControlError):
        print("repository_workload_preflight_failed=control_or_output_invalid paid_requests=0")
        return 1
    passed = report.retrieval_preflight_passed if args.prepare_retrieval else report.scan_passed
    print(
        f"repository_workload_preflight_passed={str(passed).lower()} tasks=8 "
        "provider_requests=0 paid_requests=0 formal_ready=false"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
