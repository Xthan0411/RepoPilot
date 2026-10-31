"""New exact-plan consent and immutable GPU profile, without external execution."""

from __future__ import annotations

import json
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

from scripts import p7_holdout_gpu as control  # noqa: E402
from scripts import p7_holdout_readiness as legacy  # noqa: E402
from scripts import p7_repository_workload_preflight as workload  # noqa: E402
from scripts.p7_gpu_execution_profile import QUALIFICATION_PATH  # noqa: E402
from scripts.p7_holdout_agent_inputs import _canonical_sha256  # noqa: E402
from scripts.p7_holdout_requalification import (  # noqa: E402
    PLAN_V2_PATH,
    HoldoutApplicationPlanV2,
    _git,
    _read,
    _sealed,
)

NOW = datetime(2026, 11, 1, tzinfo=UTC)


def _payload(value: Any) -> dict[str, Any]:
    return {key: getattr(value, key) for key in type(value).model_fields if key != "content_sha256"}


def _fixture_controls(monkeypatch: pytest.MonkeyPatch, root: Path = ROOT) -> control.GpuControls:
    """Use real pinned input bytes, fake only current Git source and external qualification."""
    objects = {path: "a" * 40 for path in sorted(control.REQUIRED_EXECUTION_PATHS)}
    inputs = workload.load_inputs(ROOT)
    report = json.loads((ROOT / QUALIFICATION_PATH).read_bytes())
    original_read = _read
    artifacts: dict[Path, bytes] = {}

    def read(unused_root: Path, path: Path, limit: int = 4_000_000) -> bytes:
        return artifacts[path] if path in artifacts else original_read(ROOT, path, limit)

    monkeypatch.setattr(control, "_read", read)
    monkeypatch.setattr(control, "verify_qualification", lambda root: report)
    monkeypatch.setattr(workload, "load_inputs", lambda root: inputs)
    monkeypatch.setattr(workload, "verify_source", lambda root: ("a" * 40, objects))
    monkeypatch.setattr(control, "_objects_at", lambda root, commit: objects)
    monkeypatch.setattr(
        control, "_git", lambda root, *args: "a" * 40 if args[0] == "rev-parse" else ""
    )
    freeze, plan = control.build_plan(root, frozen_at=NOW)
    artifacts[control.FREEZE_PATH] = freeze.model_dump_json().encode()
    artifacts[control.PLAN_PATH] = plan.model_dump_json().encode()
    auth = control.build_authorization(
        root,
        approved_at=NOW + timedelta(seconds=1),
        max_total_cost_usd=Decimal("32.00"),
        overshoot_acknowledged=True,
        readiness_required_before_paid=True,
        confirm_plan_sha256=plan.content_sha256,
    )
    artifacts[control.AUTHORIZATION_PATH] = auth.model_dump_json().encode()
    return control.load_controls(root)


@pytest.fixture
def controls(monkeypatch: pytest.MonkeyPatch) -> control.GpuControls:
    return _fixture_controls(monkeypatch)


@pytest.fixture
def ready(controls: control.GpuControls) -> control.GpuReadiness:
    historical = legacy.HoldoutRuntimeReadiness.model_validate_json(
        (ROOT / legacy.READINESS_PATH).read_bytes()
    )
    return control.build_readiness(
        controls,
        checked_at=NOW + timedelta(seconds=2),
        execution_controller_objects=controls.freeze.execution_source_objects,
        embedding=historical.embedding,
        vector_backend=historical.vector_backend,
        images=historical.images,
        cpu_profile=controls.freeze.cpu_runtime,
        gpu_cache=controls.freeze.cache_evidence,
    )


def test_plan_preserves_matrix_and_declares_new_runtime_total_budget(
    controls: control.GpuControls,
) -> None:
    plan = controls.plan
    template = HoldoutApplicationPlanV2.model_validate_json((ROOT / PLAN_V2_PATH).read_bytes())
    assert plan.matrix_sha256 == template.matrix_sha256
    assert plan.runtime_profile_overrides_historical_matrix
    assert not plan.new_matrix_claimed
    assert not plan.all_slots_completion_within_budget_guaranteed
    assert len(plan.run_slots) == 32
    assert {slot.run_id for slot in plan.run_slots}.isdisjoint(
        slot.run_id for slot in template.run_slots
    )
    assert plan.frozen_source_commit == controls.freeze.controller_commit
    assert (
        plan.application_budget_usd == plan.carried_unknown_cost_usd + plan.new_execution_budget_usd
    )
    assert plan.carried_unknown_cost_usd == Decimal("1.00")
    assert plan.new_execution_budget_usd == Decimal("31.00")
    assert controls.freeze.carried_attempt.provider_request_count == "unknown"
    assert not controls.freeze.carried_attempt.original_raw_evidence_revalidated
    assert control.load_plan(ROOT) == (controls.freeze, controls.plan)


def test_exact_authorization_and_readiness_bind_all_controls(
    controls: control.GpuControls, ready: control.GpuReadiness
) -> None:
    assert controls.authorization.plan_sha256 == controls.plan.content_sha256
    assert controls.authorization.max_total_cost_usd == Decimal("32.00")
    assert controls.authorization.carried_unknown_cost_usd == Decimal("1.00")
    assert ready.cpu_profile == controls.freeze.cpu_runtime
    assert ready.gpu_cache == controls.freeze.cache_evidence
    assert ready.provider.api_requests_made == ready.paid_requests_made == 0
    control.validate_readiness_bindings(ROOT, controls, ready)


@pytest.mark.parametrize("version", (1, 2))
def test_legacy_authorization_is_not_new_consent(version: int) -> None:
    with pytest.raises(ValidationError):
        control.GpuAuthorization.model_validate_json(
            (ROOT / f"results/p7-holdout/authorization-v{version}.json").read_bytes()
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"max_total_cost_usd": Decimal("33.00")},
        {"overshoot_acknowledged": False},
        {"readiness_required_before_paid": False},
        {"confirm_plan_sha256": "0" * 64},
        {"approved_at": NOW - timedelta(seconds=1)},
    ],
)
def test_authorization_builder_requires_exact_fresh_consent(
    controls: control.GpuControls, changes: dict[str, Any]
) -> None:
    arguments: dict[str, Any] = {
        "approved_at": NOW + timedelta(seconds=1),
        "max_total_cost_usd": Decimal("32.00"),
        "overshoot_acknowledged": True,
        "readiness_required_before_paid": True,
        "confirm_plan_sha256": controls.plan.content_sha256,
    }
    with pytest.raises((ValidationError, ValueError)):
        control.build_authorization(ROOT, **{**arguments, **changes})


@pytest.mark.parametrize(
    "field,value",
    [
        ("carried_unknown_cost_usd", Decimal("0")),
        ("application_budget_usd", Decimal("33")),
        ("new_execution_budget_usd", Decimal("32")),
        ("runtime_profile", "ordinary-cpu-cache"),
        ("run_slots", ()),
    ],
)
def test_plan_rejects_budget_reset_and_runtime_omission(
    controls: control.GpuControls, field: str, value: Any
) -> None:
    payload = _payload(controls.plan)
    with pytest.raises(ValidationError):
        _sealed(control.GpuPlan, {**payload, field: value})


def test_self_rehashed_plan_cannot_change_experimental_choice(
    controls: control.GpuControls, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = _sealed(
        control.GpuPlan,
        {
            **_payload(controls.plan),
            "model": "different-model",
        },
    )
    original = control._read
    monkeypatch.setattr(
        control,
        "_read",
        lambda root, path, limit=4_000_000: (
            changed.model_dump_json().encode()
            if path == control.PLAN_PATH
            else original(root, path, limit)
        ),
    )
    with pytest.raises(ValueError, match="freeze/plan binding mismatch"):
        control.load_plan(ROOT)


@pytest.mark.parametrize(
    "field",
    [
        "freeze_marker_sha256",
        "plan_sha256",
        "plan_raw_sha256",
        "runtime_profile_sha256",
        "carry_forward_sha256",
        "matrix_sha256",
    ],
)
def test_self_rehashed_authorization_cannot_cross_bind(
    controls: control.GpuControls, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    changed = _sealed(
        control.GpuAuthorization,
        {
            **_payload(controls.authorization),
            field: "0" * 64,
        },
    )
    original = control._read
    monkeypatch.setattr(
        control,
        "_read",
        lambda root, path, limit=4_000_000: (
            changed.model_dump_json().encode()
            if path == control.AUTHORIZATION_PATH
            else original(root, path, limit)
        ),
    )
    with pytest.raises(ValueError, match="authorization or source binding mismatch"):
        control.load_controls(ROOT)


@pytest.mark.parametrize(
    "field",
    [
        "plan_raw_sha256",
        "plan_content_sha256",
        "authorization_raw_sha256",
        "authorization_content_sha256",
        "freeze_raw_sha256",
        "freeze_content_sha256",
        "environment_raw_sha256",
        "environment_content_sha256",
        "agent_inputs_raw_sha256",
        "canary_raw_sha256",
        "test_inputs_raw_sha256",
        "semantic_objects_sha256",
    ],
)
def test_self_rehashed_readiness_cannot_cross_bind(
    controls: control.GpuControls, ready: control.GpuReadiness, field: str
) -> None:
    changed = _sealed(
        control.GpuReadiness,
        {
            **_payload(ready),
            field: "0" * 64,
        },
    )
    with pytest.raises(ValueError, match="readiness binding mismatch"):
        control.validate_readiness_bindings(ROOT, controls, changed)


def test_readiness_profile_cannot_be_changed_even_if_rehashed(
    controls: control.GpuControls, ready: control.GpuReadiness
) -> None:
    cpu = {**ready.cpu_profile, "cpu_threads": 4}
    changed = _sealed(
        control.GpuReadiness,
        {
            **_payload(ready),
            "cpu_profile": cpu,
            "runtime_profile_sha256": control._profile_digest(cpu, ready.gpu_cache),
        },
    )
    with pytest.raises(ValueError, match="readiness binding mismatch"):
        control.validate_readiness_bindings(ROOT, controls, changed)


def test_readiness_cannot_predate_consent(
    controls: control.GpuControls, ready: control.GpuReadiness
) -> None:
    later = controls.authorization.model_copy(update={"approved_at": NOW + timedelta(days=1)})
    with pytest.raises(ValueError, match="predates authorization"):
        control.validate_readiness_bindings(ROOT, replace(controls, authorization=later), ready)


def test_readiness_must_include_all_checks(ready: control.GpuReadiness) -> None:
    with pytest.raises(ValidationError, match="incomplete"):
        _sealed(
            control.GpuReadiness,
            {
                **_payload(ready),
                "completed_checks": (),
            },
        )


def test_frozen_source_allows_docs_commit_but_detects_hidden_bytes(
    controls: control.GpuControls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    root = tmp_path.resolve()
    for relative in control.REQUIRED_EXECUTION_PATHS:
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("qualified source\n")
    _git(root, "init", "--quiet")

    def commit(message: str) -> str:
        _git(root, "add", ".")
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
            message,
        )
        return _git(root, "rev-parse", "HEAD")

    source_commit = commit("Qualified source")
    _, objects = workload.verify_source(root)
    freeze = _sealed(
        control.GpuFreeze,
        {
            **_payload(controls.freeze),
            "controller_commit": source_commit,
            "execution_source_objects": objects,
            "execution_source_sha256": _canonical_sha256(objects),
        },
    )
    control.verify_frozen_source(root, freeze)
    (root / "README.md").write_text("documentation only\n")
    commit("Docs")
    control.verify_frozen_source(root, freeze)
    target = "scripts/p7_holdout_gpu.py"
    _git(root, "update-index", "--assume-unchanged", target)
    (root / target).write_text("hidden changed source\n")
    with pytest.raises(ValueError, match="source_bytes_changed"):
        control.verify_frozen_source(root, freeze)


def test_cli_creates_only_unpaid_freeze_and_plan(
    controls: control.GpuControls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setattr(sys, "argv", ["gpu", "--repository-root", str(tmp_path)])
    monkeypatch.setattr(
        control, "build_plan", lambda *args, **kwargs: (controls.freeze, controls.plan)
    )
    writes: list[Path] = []
    monkeypatch.setattr(control, "write_control_artifact", lambda path, model: writes.append(path))
    monkeypatch.setattr(
        control,
        "build_authorization",
        lambda *args, **kwargs: pytest.fail("consent was not granted"),
    )
    assert control.main() == 0
    assert writes == [tmp_path / control.FREEZE_PATH, tmp_path / control.PLAN_PATH]
    assert "authorization_created=false paid_requests=0" in capsys.readouterr().out
