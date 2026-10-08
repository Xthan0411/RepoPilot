"""P7-S4 frozen experiment matrix, selection, and preflight tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from repopilot.enums import (
    DatasetSplit,
    ExperimentArmKind,
    ExperimentErrorCode,
    ExperimentPreflightCheckStatus,
    ExperimentPreflightStatus,
    RepairContextMode,
    RetrievalMode,
)
from repopilot.evals import (
    FROZEN_EMBEDDING_CODE_REVISION,
    FROZEN_EMBEDDING_REVISION,
    FROZEN_OPENAI_MODEL,
    BaselineEvaluationExecutor,
    ExperimentError,
    ExperimentMatrix,
    ExperimentPreflightFacts,
    RepairWorkflowEvaluationExecutor,
    build_experiment_executor,
    build_experiment_preflight,
    load_experiment_matrix,
    require_ready_preflight,
    select_experiment_arm,
    write_experiment_preflight,
)

PROJECT_ROOT = Path(__file__).parents[2]
MATRIX_PATH = PROJECT_ROOT / "configs" / "experiments" / "p7-v2.yaml"


def _matrix() -> ExperimentMatrix:
    return load_experiment_matrix(MATRIX_PATH, repository_root=PROJECT_ROOT)


def _facts(**updates: object) -> ExperimentPreflightFacts:
    values: dict[str, object] = {
        "source_commit": "a" * 40,
        "source_clean": True,
        "local_dependencies_available": True,
        "sandbox_image_available": True,
        "provider_credential_available": True,
        "vector_backend_available": True,
        "embedding_snapshot_available": True,
        "output_root_ready": True,
    }
    values.update(updates)
    return ExperimentPreflightFacts(**values)  # type: ignore[arg-type]


def test_frozen_matrix_has_four_default_arms_and_one_optional_memory_arm() -> None:
    matrix = _matrix()

    assert matrix.experiment_id == "repopilot-p7-v2"
    assert matrix.shared_settings.values["model"] == FROZEN_OPENAI_MODEL
    assert matrix.shared_settings.values["embedding_revision"] == FROZEN_EMBEDDING_REVISION
    assert (
        matrix.shared_settings.values["embedding_code_revision"] == FROZEN_EMBEDDING_CODE_REVISION
    )
    assert matrix.prompt_versions == ("repopilot-agent-v1",)
    assert matrix.dev_repetitions == matrix.holdout_repetitions == 1
    assert len(tuple(arm for arm in matrix.arms if arm.enabled_by_default)) == 4
    memory = next(arm for arm in matrix.arms if arm.kind is ExperimentArmKind.HYBRID_MEMORY)
    assert not memory.enabled_by_default
    assert memory.dev_memory_snapshot_id == "p7-dev-empty-v2"
    assert memory.holdout_memory_snapshot_id == "p7-holdout-empty-v2"
    assert all(
        snapshot.namespace.access_mode.value == "read_only" for snapshot in matrix.memory_snapshots
    )


@pytest.mark.parametrize(
    ("arm_id", "executor_type", "context_mode", "retrieval_mode"),
    (
        (
            "mini-swe-safe-v1",
            BaselineEvaluationExecutor,
            RepairContextMode.NONE,
            RetrievalMode.BM25,
        ),
        (
            "no-semantic-rag-v1",
            BaselineEvaluationExecutor,
            RepairContextMode.BM25,
            RetrievalMode.BM25,
        ),
        (
            "dense-only-v1",
            RepairWorkflowEvaluationExecutor,
            RepairContextMode.CONFIGURED,
            RetrievalMode.DENSE,
        ),
        (
            "hybrid-rrf-v1",
            RepairWorkflowEvaluationExecutor,
            RepairContextMode.CONFIGURED,
            RetrievalMode.HYBRID,
        ),
    ),
)
def test_default_arm_selection_builds_the_real_executor(
    arm_id: str,
    executor_type: type[object],
    context_mode: RepairContextMode,
    retrieval_mode: RetrievalMode,
) -> None:
    selection = select_experiment_arm(_matrix(), arm_id)
    executor = build_experiment_executor(
        selection,
        repository_root=PROJECT_ROOT,
        split=DatasetSplit.DEV,
        openai_api_key="test-placeholder",
        database_url="postgresql://repopilot:repopilot@localhost:55433/repopilot",
    )

    assert isinstance(executor, executor_type)
    assert isinstance(
        executor,
        (BaselineEvaluationExecutor, RepairWorkflowEvaluationExecutor),
    )
    assert executor.settings.model == FROZEN_OPENAI_MODEL
    assert executor.settings.retrieval_mode is retrieval_mode
    assert selection.arm.context_mode is context_mode
    assert executor.sandbox_image_reference == selection.matrix.sandbox_image_reference


def test_optional_memory_arm_requires_explicit_selection_and_matching_snapshot() -> None:
    matrix = _matrix()

    with pytest.raises(ExperimentError) as captured:
        select_experiment_arm(matrix, "hybrid-memory-v1")
    assert captured.value.code is ExperimentErrorCode.ARM_UNAVAILABLE

    selection = select_experiment_arm(
        matrix,
        "hybrid-memory-v1",
        include_optional=True,
    )
    with pytest.raises(ExperimentError) as captured:
        build_experiment_executor(
            selection,
            repository_root=PROJECT_ROOT,
            split=DatasetSplit.HOLDOUT,
            openai_api_key="test-placeholder",
            database_url="postgresql://repopilot:repopilot@localhost:55433/repopilot",
        )
    assert captured.value.code is ExperimentErrorCode.MEMORY_SNAPSHOT_INVALID


def test_selection_ignores_repopilot_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REPOPILOT_MODEL", "gpt-env-drift")
    monkeypatch.setenv("REPOPILOT_MAX_STEPS", "999")
    selection = select_experiment_arm(_matrix(), "dense-only-v1")

    settings = selection.settings(
        openai_api_key="test-placeholder",
        database_url="postgresql://repopilot:repopilot@localhost:55433/repopilot",
    )

    assert settings.model == FROZEN_OPENAI_MODEL
    assert settings.max_steps == 30
    assert settings.retrieval_mode is RetrievalMode.DENSE


def test_matrix_rejects_fairness_secret_and_integrity_drift() -> None:
    payload = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))

    drifted = {**payload, "holdout_repetitions": 2}
    with pytest.raises(ValidationError):
        ExperimentMatrix.model_validate(drifted)

    secret = {**payload, "openai_api_key": "must-not-persist"}
    with pytest.raises(ValidationError):
        ExperimentMatrix.model_validate(secret)

    mutable = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
    mutable["shared_settings"]["values"]["model"] = "gpt-5.6"
    with pytest.raises(ValidationError):
        ExperimentMatrix.model_validate(mutable)


def test_preflight_is_ready_only_when_every_static_and_machine_check_passes(
    tmp_path: Path,
) -> None:
    matrix = _matrix()
    checked_at = datetime(2026, 8, 7, 21, tzinfo=UTC)
    ready = build_experiment_preflight(
        matrix,
        repository_root=PROJECT_ROOT,
        checked_at=checked_at,
        facts=_facts(),
    )

    assert ready.status is ExperimentPreflightStatus.READY
    assert ready.ready_for_formal_run
    assert all(item.status is ExperimentPreflightCheckStatus.PASSED for item in ready.checks)
    require_ready_preflight(ready)

    report_path = tmp_path / "preflight.json"
    write_experiment_preflight(report_path, ready)
    with pytest.raises(ExperimentError) as captured:
        write_experiment_preflight(report_path, ready)
    assert captured.value.code is ExperimentErrorCode.OUTPUT_CONFLICT

    blocked = build_experiment_preflight(
        matrix,
        repository_root=PROJECT_ROOT,
        checked_at=checked_at,
        facts=_facts(
            source_clean=False,
            provider_credential_available=False,
            embedding_snapshot_available=None,
        ),
    )
    assert blocked.status is ExperimentPreflightStatus.BLOCKED
    assert not blocked.ready_for_formal_run
    assert {
        item.check_id
        for item in blocked.checks
        if item.status is not ExperimentPreflightCheckStatus.PASSED
    } == {"embedding-snapshot", "provider-credential", "source-snapshot"}
    with pytest.raises(ExperimentError) as captured:
        require_ready_preflight(blocked)
    assert captured.value.code in {
        ExperimentErrorCode.EMBEDDING_SNAPSHOT_INVALID,
        ExperimentErrorCode.PROVIDER_UNAVAILABLE,
        ExperimentErrorCode.SOURCE_SNAPSHOT_UNCLEAN,
    }
