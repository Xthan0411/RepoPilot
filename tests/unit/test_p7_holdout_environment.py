"""P7-S6 formal environment and timeout-projection contract tests."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from repopilot.enums import ExperimentPreflightCheckStatus
from repopilot.evals import FormalFreezeMarker, load_experiment_matrix
from repopilot.evals.dataset import (
    DatasetPreflightReport,
    FrozenDatasetManifest,
)

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.p7_bind_holdout_environments as binder  # noqa: E402
from scripts.p7_holdout_control import (  # noqa: E402
    COMMAND_TIMEOUT_CAP_SECONDS,
    DATASET_PREFLIGHT_PATH,
    EXPECTED_TASK_IDS,
    FREEZE_PATH,
    MATRIX_PATH,
    HoldoutCommandTimeoutBinding,
    HoldoutEnvironmentManifest,
    build_holdout_application_plan,
    build_holdout_preflight,
    build_holdout_timeout_bindings,
    project_holdout_task_timeouts,
)
from scripts.p7_publish_registry_images import (  # noqa: E402
    RESULT_PATH as REGISTRY_MANIFEST_PATH,
)
from scripts.p7_publish_registry_images import (  # noqa: E402
    RegistryManifest,
)
from scripts.p7_sandbox_matrix_preflight import (  # noqa: E402
    DATASET_MANIFEST_PATH,
    SandboxMatrixPreflight,
)
from scripts.p7_sandbox_matrix_preflight import (  # noqa: E402
    RESULT_PATH as SANDBOX_PREFLIGHT_PATH,
)

NOW = datetime(2026, 8, 8, 18, tzinfo=UTC)
CONTROLLER_COMMIT = "c" * 40
COMMITTED_ENVIRONMENT_PATH = PROJECT_ROOT / "results/p7-holdout/environments-v1.json"
COMMITTED_ENVIRONMENT_RAW_SHA256 = (
    "b7c6f04283433859e462edb7105ad5411700703fa3557b1281fafc3ccb99890d"
)


def _inputs() -> tuple[
    FrozenDatasetManifest,
    bytes,
    SandboxMatrixPreflight,
    bytes,
    RegistryManifest,
]:
    dataset_manifest = FrozenDatasetManifest.model_validate_json(
        PROJECT_ROOT.joinpath(DATASET_MANIFEST_PATH).read_bytes()
    )
    preflight_bytes = PROJECT_ROOT.joinpath(SANDBOX_PREFLIGHT_PATH).read_bytes()
    registry_bytes = PROJECT_ROOT.joinpath(REGISTRY_MANIFEST_PATH).read_bytes()
    return (
        dataset_manifest,
        preflight_bytes,
        SandboxMatrixPreflight.model_validate_json(preflight_bytes),
        registry_bytes,
        RegistryManifest.model_validate_json(registry_bytes),
    )


def _manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> HoldoutEnvironmentManifest:
    matrix = load_experiment_matrix(PROJECT_ROOT / MATRIX_PATH, repository_root=PROJECT_ROOT)
    dataset_manifest, preflight_bytes, preflight, registry_bytes, registry = _inputs()
    monkeypatch.setattr(
        binder,
        "_verify_checkout",
        lambda root, relative_path, *, base_commit: root / relative_path,
    )
    return binder.build_environment_manifest(
        root=tmp_path,
        controller_commit=CONTROLLER_COMMIT,
        experiment_id=matrix.experiment_id,
        matrix_sha256=matrix.content_sha256,
        matrix_dataset_manifest_sha256=matrix.dataset.manifest_content_sha256,
        matrix_sandbox_command_timeout_seconds=matrix.shared_settings.values.get(
            "sandbox_command_timeout_seconds"
        ),
        matrix_sandbox_network=matrix.shared_settings.values.get("sandbox_network"),
        dataset_manifest_sha256=dataset_manifest.content_sha256,
        dataset_manifest=dataset_manifest,
        preflight=preflight,
        preflight_bytes=preflight_bytes,
        registry=registry,
        registry_bytes=registry_bytes,
        created_at=NOW,
    )


def test_committed_x86_environment_manifest_is_exact_and_formally_ready() -> None:
    raw = COMMITTED_ENVIRONMENT_PATH.read_bytes()
    manifest = HoldoutEnvironmentManifest.model_validate_json(raw)
    capped = sum(
        timeout.capped_by_sandbox_policy
        for binding in manifest.bindings
        for timeout in (*binding.setup_command_timeouts, *binding.test_command_timeouts)
    )

    assert hashlib.sha256(raw).hexdigest() == COMMITTED_ENVIRONMENT_RAW_SHA256
    assert manifest.controller_commit == "f07fb6dc44131d966431d51c631f5eae353c60c6"
    assert manifest.execution_harness_revision == "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
    assert manifest.registry_manifest_raw_sha256 == (
        "edc0b94b7ec8247e00e2f36c80d5f930e1dc7141602da9e6514f55338841fef9"
    )
    assert len(manifest.bindings) == 8
    assert capped == 17
    assert manifest.formal_environment_ready
    assert manifest.formal_blockers == ()
    assert manifest.paid_requests_made == 0
    assert not manifest.gold_visible_to_agent


def test_timeout_projection_changes_only_command_timeout_fields() -> None:
    dataset_manifest, *_ = _inputs()

    for case in dataset_manifest.external_cases:
        source = case.agent_task("Gold-free formal environment command binding")
        projected = project_holdout_task_timeouts(source)
        source_without_commands = source.model_dump(
            mode="python", exclude={"setup_commands", "test_commands"}
        )
        projected_without_commands = projected.model_dump(
            mode="python", exclude={"setup_commands", "test_commands"}
        )

        assert projected_without_commands == source_without_commands
        for declared, effective in zip(
            (*source.setup_commands, *source.test_commands),
            (*projected.setup_commands, *projected.test_commands),
            strict=True,
        ):
            assert effective.argv == declared.argv
            assert effective.cwd == declared.cwd
            assert effective.timeout_seconds == min(
                declared.timeout_seconds or COMMAND_TIMEOUT_CAP_SECONDS,
                COMMAND_TIMEOUT_CAP_SECONDS,
            )


def test_environment_manifest_binds_all_tasks_images_and_timeout_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path, monkeypatch)
    serialized = manifest.model_dump_json()
    capped = sum(
        timeout.capped_by_sandbox_policy
        for binding in manifest.bindings
        for timeout in (*binding.setup_command_timeouts, *binding.test_command_timeouts)
    )

    assert tuple(item.task_id for item in manifest.bindings) == EXPECTED_TASK_IDS
    assert len({item.sandbox_image_reference for item in manifest.bindings}) == 8
    assert all("@sha256:" in item.sandbox_image_reference for item in manifest.bindings)
    assert all(item.setup_network == item.agent_network == "none" for item in manifest.bindings)
    assert capped == 17
    assert manifest.source_command_contract_sha256 != manifest.runtime_command_contract_sha256
    assert manifest.formal_environment_ready
    assert manifest.formal_blockers == ()
    assert manifest.paid_requests_made == 0
    assert not manifest.gold_visible_to_agent
    assert "expected_patch" not in serialized
    assert "gold_files" not in serialized


def test_timeout_binding_rejects_incorrect_cap_or_evidence() -> None:
    with pytest.raises(ValidationError, match="effective command timeout"):
        HoldoutCommandTimeoutBinding(
            phase="setup",
            command_index=0,
            declared_timeout_seconds=600,
            effective_timeout_seconds=299,
            capped_by_sandbox_policy=True,
        )

    with pytest.raises(ValidationError, match="cap evidence"):
        HoldoutCommandTimeoutBinding(
            phase="test",
            command_index=0,
            declared_timeout_seconds=120,
            effective_timeout_seconds=120,
            capped_by_sandbox_policy=True,
        )


def test_builder_rejects_source_artifact_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = load_experiment_matrix(PROJECT_ROOT / MATRIX_PATH, repository_root=PROJECT_ROOT)
    dataset_manifest, preflight_bytes, preflight, registry_bytes, registry = _inputs()
    monkeypatch.setattr(
        binder,
        "_verify_checkout",
        lambda root, relative_path, *, base_commit: root / relative_path,
    )

    with pytest.raises(
        binder.EnvironmentBindingError,
        match=binder.EnvironmentBindingErrorCode.SOURCE_ARTIFACT_INVALID.value,
    ):
        binder.build_environment_manifest(
            root=tmp_path,
            controller_commit=CONTROLLER_COMMIT,
            experiment_id=matrix.experiment_id,
            matrix_sha256=matrix.content_sha256,
            matrix_dataset_manifest_sha256=matrix.dataset.manifest_content_sha256,
            matrix_sandbox_command_timeout_seconds=matrix.shared_settings.values.get(
                "sandbox_command_timeout_seconds"
            ),
            matrix_sandbox_network=matrix.shared_settings.values.get("sandbox_network"),
            dataset_manifest_sha256=dataset_manifest.content_sha256,
            dataset_manifest=dataset_manifest,
            preflight=preflight,
            preflight_bytes=preflight_bytes,
            registry=registry,
            registry_bytes=registry_bytes + b"\n",
            created_at=NOW,
        )


def test_builder_rejects_timeout_evidence_drift() -> None:
    _, _, preflight, _, _ = _inputs()
    compatibility = preflight.task_evidence[0]
    dataset_manifest, *_ = _inputs()
    setup, test = build_holdout_timeout_bindings(
        dataset_manifest.external_cases[0].agent_task(
            "Gold-free formal environment command binding"
        )
    )
    drifted = compatibility.model_copy(
        update={
            "setup_commands": (
                compatibility.setup_commands[0].model_copy(
                    update={"effective_timeout_seconds": 299}
                ),
                *compatibility.setup_commands[1:],
            )
        }
    )

    with pytest.raises(
        binder.EnvironmentBindingError,
        match=binder.EnvironmentBindingErrorCode.TIMEOUT_CONTRACT_MISMATCH.value,
    ):
        binder._verify_timeout_evidence(
            compatibility=drifted,
            setup_timeouts=setup,
            test_timeouts=test,
        )


def test_formal_environment_advances_runtime_gold_count_to_eight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = load_experiment_matrix(PROJECT_ROOT / MATRIX_PATH, repository_root=PROJECT_ROOT)
    marker = FormalFreezeMarker.model_validate_json(PROJECT_ROOT.joinpath(FREEZE_PATH).read_bytes())
    dataset_preflight = DatasetPreflightReport.model_validate_json(
        PROJECT_ROOT.joinpath(DATASET_PREFLIGHT_PATH).read_bytes()
    )
    dataset_manifest, *_ = _inputs()
    plan = build_holdout_application_plan(
        matrix=matrix,
        marker=marker,
        dataset_manifest=dataset_manifest,
        controller_commit=CONTROLLER_COMMIT,
        created_at=NOW,
    )
    environment = _manifest(tmp_path, monkeypatch)
    preflight = build_holdout_preflight(
        root=tmp_path,
        plan=plan,
        marker=marker,
        dataset_preflight=dataset_preflight,
        checked_at=NOW,
        controller_source_clean=True,
        frozen_runtime_matches=True,
        authorization=None,
        environment_manifest=environment,
    )
    checks = {item.check_id: item for item in preflight.checks}

    assert preflight.runtime_verified_task_count == 8
    assert checks["runtime-gold-validation"].status is ExperimentPreflightCheckStatus.PASSED
    assert checks["task-environments"].status is ExperimentPreflightCheckStatus.PASSED
    assert checks["task-images"].status is ExperimentPreflightCheckStatus.PASSED
    assert checks["task-checkouts"].status is ExperimentPreflightCheckStatus.BLOCKED
    assert not preflight.ready_for_paid_holdout


def test_direct_file_help_has_no_side_effect() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            str(PROJECT_ROOT / "scripts/p7_bind_holdout_environments.py"),
            "--help",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Bind P7-S6 registry" in completed.stdout
