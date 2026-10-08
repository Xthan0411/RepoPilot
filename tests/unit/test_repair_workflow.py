"""Repair workflow input, retrieval, and orchestration-boundary tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from repopilot.agent import repair_decision_request_id
from repopilot.config import RepoPilotSettings
from repopilot.embeddings import LocalEmbeddingCache, embedding_provider_from_settings
from repopilot.enums import (
    DatasetSplit,
    MemoryAccessMode,
    RepairWorkflowErrorCode,
    RepositoryMemoryLookupStatus,
    RetrievalMode,
    VerifiedEpisodeLookupStatus,
)
from repopilot.memory import FileMemoryNamespaceManager, FileRepositoryMemoryStore
from repopilot.providers import FinishAction, ProviderResponse, ProviderUsageDelta
from repopilot.repair_workflow import (
    RepairRunRequest,
    RepairWorkflowError,
    ScriptedProviderManifest,
    build_repair_retrieval_context,
    load_repair_task,
    load_scripted_provider,
    repository_id_for_task,
    run_repair_request,
    run_repair_task,
)
from repopilot.schemas import CommandSpec, MemoryNamespace, TaskSpec
from repopilot.storage import InMemoryVectorStore
from repopilot.tools import ReadOnlyRepository

RUN_ID = UUID("73771928-2208-4dd3-9fc0-ab030ca945d4")
STARTED_AT = datetime(2026, 8, 6, 12, tzinfo=UTC)
IMAGE = f"python@sha256:{'a' * 64}"


def _task(repository_path: str) -> TaskSpec:
    return TaskSpec(
        task_id="repair-workflow-unit",
        repository_path=repository_path,
        base_commit="b" * 40,
        issue_title="Blank parser input raises an exception",
        issue_body="The parse_config function should ignore blank parser lines.",
        test_commands=(
            CommandSpec(
                argv=("python", "-m", "unittest", "-q"),
                timeout_seconds=30,
            ),
        ),
        timeout_seconds=60,
    )


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    package = repository / "parser"
    package.mkdir(parents=True)
    package.joinpath("core.py").write_text(
        "def parse_config(raw: str) -> dict[str, str]:\n"
        "    return dict(line.split('=') for line in raw.splitlines())\n",
        encoding="utf-8",
    )
    return repository


def _manifest(path: Path, task: TaskSpec) -> None:
    path.write_text(task.model_dump_json(indent=2), encoding="utf-8")


def _response(index: int = 0) -> ProviderResponse:
    return ProviderResponse(
        request_id=repair_decision_request_id(RUN_ID, index),
        action=FinishAction(summary="Stop after deterministic evidence collection."),
        usage=ProviderUsageDelta(
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            estimated_cost_usd=Decimal("0"),
            latency_ms=1,
        ),
    )


def test_task_loader_resolves_contained_repository_and_rejects_gold(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    task_path = tmp_path / "task.json"
    _manifest(task_path, _task("repository"))

    loaded = load_repair_task(task_path)

    assert loaded.repository_path == str(repository.resolve())

    payload = json.loads(task_path.read_text(encoding="utf-8"))
    payload["expected_patch"] = "EVALUATOR_ONLY_PATCH_CANARY"
    task_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RepairWorkflowError) as captured:
        load_repair_task(task_path)
    assert captured.value.code is RepairWorkflowErrorCode.INVALID_TASK
    assert "EVALUATOR_ONLY_PATCH_CANARY" not in str(captured.value)


def test_task_loader_rejects_escape_symlink_and_oversized_input(
    tmp_path: Path,
) -> None:
    _repository(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    escape_path = tmp_path / "escape.json"
    _manifest(escape_path, _task(f"../{outside.name}"))

    with pytest.raises(RepairWorkflowError) as captured:
        load_repair_task(escape_path)
    assert captured.value.code is RepairWorkflowErrorCode.INVALID_TASK

    valid_path = tmp_path / "valid.json"
    _manifest(valid_path, _task("repository"))
    symlink = tmp_path / "task-link.json"
    symlink.symlink_to(valid_path)
    with pytest.raises(RepairWorkflowError) as captured:
        load_repair_task(symlink)
    assert captured.value.code is RepairWorkflowErrorCode.INPUT_IO_ERROR

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"{" + b"x" * 1_000_000)
    with pytest.raises(RepairWorkflowError) as captured:
        load_repair_task(oversized)
    assert captured.value.code is RepairWorkflowErrorCode.INPUT_IO_ERROR


def test_scripted_provider_requires_ordered_request_identity(tmp_path: Path) -> None:
    manifest = ScriptedProviderManifest(
        thread_id="repair-workflow-unit",
        run_id=RUN_ID,
        started_at=STARTED_AT,
        responses=(_response(),),
    )
    path = tmp_path / "provider.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    assert load_scripted_provider(path) == manifest

    with pytest.raises(ValidationError):
        ScriptedProviderManifest(
            thread_id="repair-workflow-unit",
            run_id=RUN_ID,
            started_at=STARTED_AT,
            responses=(_response(1),),
        )


def test_bm25_context_exposes_ranked_source_without_evaluator_gold(
    tmp_path: Path,
) -> None:
    repository_path = _repository(tmp_path)
    settings = RepoPilotSettings(retrieval_default_top_k=5, bm25_default_top_k=5)
    repository = ReadOnlyRepository.from_settings(repository_path, settings)

    context = build_repair_retrieval_context(
        repository,
        _task(str(repository_path)),
        settings,
        monotonic=iter((1.0, 1.003, 2.0, 2.004)).__next__,
    )

    assert context.indexing_latency_ms == 3
    assert context.retrieval_latency_ms == 4
    assert context.hits
    assert "path=parser/core.py" in context.hits[0].retrieval_reason
    assert "parser/core.py" in context.repository_summary
    assert "EVALUATOR_ONLY_PATCH_CANARY" not in context.model_dump_json()


@pytest.mark.parametrize("mode", (RetrievalMode.DENSE, RetrievalMode.HYBRID))
def test_repair_context_executes_configured_dense_and_hybrid_retrieval(
    tmp_path: Path,
    mode: RetrievalMode,
) -> None:
    repository_path = _repository(tmp_path)
    settings = RepoPilotSettings(
        retrieval_mode=mode,
        retrieval_default_top_k=5,
        dense_default_top_k=5,
        hybrid_candidate_top_k=5,
    )
    repository = ReadOnlyRepository.from_settings(repository_path, settings)

    context = build_repair_retrieval_context(
        repository,
        _task(str(repository_path)),
        settings,
        embedding_provider=embedding_provider_from_settings(settings),
        embedding_cache=LocalEmbeddingCache(tmp_path / "embedding-cache"),
        vector_store=InMemoryVectorStore(dimension=settings.vector_dimension),
        monotonic=iter((1.0, 1.003, 2.0, 2.004)).__next__,
    )

    assert context.hits
    assert all(hit.dense_score is not None for hit in context.hits)
    if mode is RetrievalMode.DENSE:
        assert all(hit.sparse_score is None for hit in context.hits)
        assert "Top Dense source context" in context.repository_summary
    else:
        assert all(hit.sparse_score is not None for hit in context.hits)
        assert all(hit.fused_score is not None for hit in context.hits)
        assert "Top Hybrid RRF source context" in context.repository_summary


def test_repair_retrieval_loads_repository_memory_across_task_ids(
    tmp_path: Path,
) -> None:
    repository_path = _repository(tmp_path)
    first_task = _task(str(repository_path))
    second_task = first_task.model_copy(update={"task_id": "another-repair-task"})
    settings = RepoPilotSettings(retrieval_default_top_k=5, bm25_default_top_k=5)
    repository = ReadOnlyRepository.from_settings(repository_path, settings)
    store = FileRepositoryMemoryStore(tmp_path / "repository-memory")

    first = build_repair_retrieval_context(
        repository,
        first_task,
        settings,
        repository_memory_store=store,
        memory_created_at=STARTED_AT + timedelta(days=1),
        monotonic=iter((1.0, 1.001, 2.0, 2.001)).__next__,
    )
    second = build_repair_retrieval_context(
        repository,
        second_task,
        settings,
        repository_memory_store=store,
        memory_created_at=STARTED_AT,
        monotonic=iter((3.0, 3.001, 4.0, 4.001)).__next__,
    )

    assert repository_id_for_task(first_task) == repository_id_for_task(second_task)
    assert first.repository_id == second.repository_id
    assert first.repository_content_sha256 == second.repository_content_sha256
    assert first.repository_memory == second.repository_memory
    assert {item.kind.value for item in first.repository_memory} == {
        "repository_map",
        "test_command",
    }
    assert str(repository_path) not in first.model_dump_json()


def test_repair_retrieval_keeps_dev_and_holdout_memory_disjoint(
    tmp_path: Path,
) -> None:
    repository_path = _repository(tmp_path)
    task = _task(str(repository_path))
    settings = RepoPilotSettings(retrieval_default_top_k=5, bm25_default_top_k=5)
    repository = ReadOnlyRepository.from_settings(repository_path, settings)
    manager = FileMemoryNamespaceManager(tmp_path / "namespaced-memory")
    dev_namespace = MemoryNamespace(
        experiment_id="workflow-isolation",
        snapshot_id="snapshot-v1",
        split=DatasetSplit.DEV,
        access_mode=MemoryAccessMode.READ_WRITE,
    )
    holdout_namespace = dev_namespace.model_copy(
        update={
            "split": DatasetSplit.HOLDOUT,
            "access_mode": MemoryAccessMode.READ_ONLY,
        }
    )
    dev_stores = manager.open(dev_namespace, created_at=STARTED_AT)
    holdout_stores = manager.open(holdout_namespace, created_at=STARTED_AT)

    dev = build_repair_retrieval_context(
        repository,
        task,
        settings,
        memory_stores=dev_stores,
        memory_created_at=STARTED_AT,
        monotonic=iter((1.0, 1.001, 2.0, 2.001)).__next__,
    )
    holdout = build_repair_retrieval_context(
        repository,
        task,
        settings,
        memory_stores=holdout_stores,
        memory_created_at=STARTED_AT,
        monotonic=iter((3.0, 3.001, 4.0, 4.001)).__next__,
    )

    assert dev.memory_namespace == dev_namespace
    assert dev.memory_load_audit is not None
    assert dev.memory_load_audit.repository_status is RepositoryMemoryLookupStatus.CURRENT
    assert dev.memory_load_audit.episodic_status is VerifiedEpisodeLookupStatus.EMPTY
    assert dev.repository_memory
    assert holdout.memory_namespace == holdout_namespace
    assert holdout.memory_load_audit is not None
    assert holdout.memory_load_audit.repository_status is RepositoryMemoryLookupStatus.EMPTY
    assert holdout.memory_load_audit.episodic_status is VerifiedEpisodeLookupStatus.EMPTY
    assert holdout.repository_memory == ()
    assert holdout.episodic_memory == ()


def test_run_boundaries_reject_nonempty_output_and_provider_mode_mismatch(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    task_path = tmp_path / "task.json"
    _manifest(task_path, _task(str(repository)))
    settings = RepoPilotSettings()

    with pytest.raises(RepairWorkflowError) as captured:
        run_repair_task(
            task_path=task_path,
            output_dir=tmp_path / "unused",
            sandbox_image_reference=IMAGE,
            settings=settings,
        )
    assert captured.value.code is RepairWorkflowErrorCode.PROVIDER_CONFIGURATION

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    occupied.joinpath("existing.txt").write_text("preserve", encoding="utf-8")
    request = RepairRunRequest(
        thread_id="repair-workflow-unit",
        run_id=RUN_ID,
        started_at=STARTED_AT,
        task=_task(str(repository)),
        sandbox_image_reference=IMAGE,
        provider_responses=(_response(),),
    )
    with pytest.raises(RepairWorkflowError) as captured:
        run_repair_request(request, output_dir=occupied, settings=settings)
    assert captured.value.code is RepairWorkflowErrorCode.INVALID_OUTPUT
    assert occupied.joinpath("existing.txt").read_text(encoding="utf-8") == "preserve"
