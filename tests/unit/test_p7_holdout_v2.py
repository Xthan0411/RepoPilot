"""Exact consent and fresh controller/readiness binding for the repaired plan."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from scripts import p7_holdout_readiness as legacy  # noqa: E402
from scripts import p7_holdout_v2 as control  # noqa: E402
from scripts.p7_holdout_requalification import (  # noqa: E402
    PLAN_V2_PATH,
    POST_REPAIR_FREEZE_PATH,
    HoldoutApplicationPlanV2,
    PostRepairFreeze,
    _git,
    _read,
    _sealed,
    _sha,
)

NOW = datetime(2026, 10, 29, 5, tzinfo=UTC)


def _readiness_payload(value: control.HoldoutRuntimeReadinessV2) -> dict[str, Any]:
    return {
        **value.model_dump(exclude={"content_sha256"}),
        "provider": value.provider,
        "embedding": value.embedding,
        "vector_backend": value.vector_backend,
        "images": value.images,
    }


@pytest.fixture
def controls(monkeypatch: pytest.MonkeyPatch) -> control.V2Controls:
    freeze = PostRepairFreeze.model_validate_json((ROOT / POST_REPAIR_FREEZE_PATH).read_bytes())
    plan = HoldoutApplicationPlanV2.model_validate_json((ROOT / PLAN_V2_PATH).read_bytes())
    monkeypatch.setattr(control, "load_requalification", lambda root: (freeze, plan))
    authorization = control.build_authorization(
        ROOT,
        approved_at=NOW,
        max_total_cost_usd=Decimal("32.00"),
        overshoot_acknowledged=True,
        readiness_required_before_paid=True,
    )
    original_read = _read

    def read(root: Path, relative: Path, limit: int = 4_000_000) -> bytes:
        if relative == control.AUTHORIZATION_V2_PATH:
            return authorization.model_dump_json().encode()
        return original_read(root, relative, limit)

    monkeypatch.setattr(control, "_read", read)
    return control.load_controls(ROOT)


@pytest.fixture
def ready(
    controls: control.V2Controls,
    monkeypatch: pytest.MonkeyPatch,
) -> control.HoldoutRuntimeReadinessV2:
    historical = legacy.HoldoutRuntimeReadiness.model_validate_json(
        (ROOT / legacy.READINESS_PATH).read_bytes()
    )
    objects = {path: "a" * 40 for path in sorted(control.REQUIRED_EXECUTION_PATHS)}
    monkeypatch.setattr(control, "verify_execution_controller", lambda root: objects)
    monkeypatch.setattr(
        control, "_git", lambda root, *args: "a" * 40 if args[0] == "rev-parse" else ""
    )
    return control.build_readiness(
        controls,
        checked_at=NOW + timedelta(seconds=1),
        execution_controller_objects=objects,
        embedding=historical.embedding,
        vector_backend=historical.vector_backend,
        images=historical.images,
    )


def test_authorization_is_exact_and_requires_readiness(controls: control.V2Controls) -> None:
    auth = controls.authorization
    assert auth.plan_sha256 == controls.plan.content_sha256 == control.APPROVED_PLAN_SHA256
    assert auth.plan_raw_sha256 == _sha(controls.plan_raw)
    assert auth.freeze_raw_sha256 == _sha(controls.freeze_raw)
    assert auth.freeze_marker_sha256 == controls.freeze.content_sha256
    assert auth.matrix_sha256 == controls.plan.matrix_sha256
    assert auth.max_total_cost_usd == Decimal("32.00")
    assert auth.readiness_required_before_paid
    assert auth.final_in_flight_response_overshoot_acknowledged
    assert auth.authorization_version == "holdout-authorization-v2"


@pytest.mark.parametrize(
    "changes",
    [
        {"max_total_cost_usd": Decimal("31.99")},
        {"overshoot_acknowledged": False},
        {"readiness_required_before_paid": False},
    ],
)
def test_builder_rejects_incomplete_consent(
    controls: control.V2Controls,
    changes: dict[str, Any],
) -> None:
    arguments: dict[str, Any] = {
        "approved_at": NOW,
        "max_total_cost_usd": Decimal("32.00"),
        "overshoot_acknowledged": True,
        "readiness_required_before_paid": True,
    }
    arguments.update(changes)
    with pytest.raises(ValidationError):
        control.build_authorization(ROOT, **arguments)


def test_old_authorization_is_never_accepted() -> None:
    with pytest.raises(ValidationError):
        control.PaidHoldoutAuthorizationV2.model_validate_json(
            (ROOT / legacy.AUTHORIZATION_PATH).read_bytes()
        )


@pytest.mark.parametrize("field", ["freeze_marker_sha256", "matrix_sha256", "experiment_id"])
def test_self_rehashed_consent_cannot_cross_bind(
    controls: control.V2Controls,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    payload = controls.authorization.model_dump(exclude={"content_sha256"})
    payload[field] = "other-experiment" if field == "experiment_id" else "0" * 64
    wrong = _sealed(control.PaidHoldoutAuthorizationV2, payload)
    original = _read
    monkeypatch.setattr(
        control,
        "_read",
        lambda root, path, limit=4_000_000: (
            wrong.model_dump_json().encode()
            if path == control.AUTHORIZATION_V2_PATH
            else original(root, path, limit)
        ),
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        control.load_controls(ROOT)


def test_readiness_full_binding_passes(
    controls: control.V2Controls,
    ready: control.HoldoutRuntimeReadinessV2,
) -> None:
    control.validate_readiness_bindings(ROOT, controls, ready)
    assert ready.paid_requests_made == ready.provider.api_requests_made == 0
    assert ready.completed_checks == control.READINESS_CHECKS
    assert ready.qualified_runtime_commit != ready.controller_commit


@pytest.mark.parametrize(
    "field",
    [
        "plan_raw_sha256",
        "plan_content_sha256",
        "authorization_raw_sha256",
        "authorization_content_sha256",
        "freeze_raw_sha256",
        "freeze_content_sha256",
        "canary_raw_sha256",
        "test_inputs_raw_sha256",
        "agent_inputs_raw_sha256",
        "environment_raw_sha256",
        "environment_content_sha256",
        "semantic_objects_sha256",
    ],
)
def test_self_rehashed_readiness_binding_is_rejected(
    controls: control.V2Controls,
    ready: control.HoldoutRuntimeReadinessV2,
    field: str,
) -> None:
    payload = _readiness_payload(ready)
    payload[field] = "0" * 64
    changed = _sealed(control.HoldoutRuntimeReadinessV2, payload)
    with pytest.raises(ValueError, match="binding mismatch"):
        control.validate_readiness_bindings(ROOT, controls, changed)


def test_controller_drift_cannot_reuse_readiness(
    controls: control.V2Controls,
    ready: control.HoldoutRuntimeReadinessV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = {**ready.execution_controller_objects, "scripts/new.py": "b" * 40}
    monkeypatch.setattr(control, "verify_execution_controller", lambda root: changed)
    with pytest.raises(ValueError, match="controller drift"):
        control.validate_readiness_bindings(ROOT, controls, ready)


def test_recorded_controller_must_have_matching_blobs(
    controls: control.V2Controls,
    ready: control.HoldoutRuntimeReadinessV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        control, "_git", lambda root, *args: "b" * 40 if args[0] == "rev-parse" else ""
    )
    with pytest.raises(ValueError, match="recorded controller mismatch"):
        control.validate_readiness_bindings(ROOT, controls, ready)


@pytest.mark.parametrize("field", ["completed_checks", "images", "execution_controller_objects"])
def test_readiness_cannot_omit_checks(
    ready: control.HoldoutRuntimeReadinessV2,
    field: str,
) -> None:
    payload = _readiness_payload(ready)
    payload[field] = {} if field == "execution_controller_objects" else ()
    with pytest.raises(ValidationError):
        _sealed(control.HoldoutRuntimeReadinessV2, payload)


def test_readiness_cannot_predate_consent(
    controls: control.V2Controls,
    ready: control.HoldoutRuntimeReadinessV2,
) -> None:
    later_auth = controls.authorization.model_copy(update={"approved_at": NOW + timedelta(days=1)})
    with pytest.raises(ValueError, match="predates authorization"):
        control.validate_readiness_bindings(
            ROOT, replace(controls, authorization=later_auth), ready
        )


@pytest.mark.parametrize(
    "name", ["OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"]
)
def test_unapproved_provider_overrides_fail_without_echo(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setattr(legacy, "_verify_host_and_environment", lambda: None)
    monkeypatch.setenv(name, "must-not-appear-in-errors")
    with pytest.raises(ValueError, match="unapproved_provider_environment_override") as caught:
        control.verify_runtime_environment()
    assert "must-not-appear" not in str(caught.value)


def test_execution_inventory_checks_actual_bytes(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    for relative in control.REQUIRED_EXECUTION_PATHS:
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("qualified controller\n")
    _git(root, "init", "--quiet")
    _git(root, "add", "scripts")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "Fixture",
    )
    assert set(control.verify_execution_controller(root)) == set(control.REQUIRED_EXECUTION_PATHS)
    target = "scripts/p7_holdout_runner_v2.py"
    _git(root, "update-index", "--assume-unchanged", target)
    (root / target).write_text("changed execution controller\n")
    with pytest.raises(ValueError, match="differs from committed bytes"):
        control.verify_execution_controller(root)


def test_controller_ancestry_failure_is_not_ignored(
    controls: control.V2Controls,
    ready: control.HoldoutRuntimeReadinessV2,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rejected(*args: object) -> str:
        raise subprocess.CalledProcessError(1, "git merge-base")

    monkeypatch.setattr(control, "_git", rejected)
    with pytest.raises(subprocess.CalledProcessError):
        control.validate_readiness_bindings(ROOT, controls, ready)
