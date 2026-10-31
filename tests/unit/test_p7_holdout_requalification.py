"""Unpaid post-repair freeze evidence, identity, and source-boundary regressions."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from repopilot import enums
from repopilot.evals import FormalFreezeMarker
from repopilot.providers.openai import OpenAIProvider
from repopilot.sandbox import DockerSandbox

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import p7_holdout_requalification as requalification  # noqa: E402
from scripts.p7_holdout_agent_inputs import (  # noqa: E402
    _canonical_sha256,
    _validate_content_digest,
)
from scripts.p7_holdout_control import (  # noqa: E402
    FREEZE_PATH,
    PLAN_PATH,
    HoldoutApplicationPlan,
    PaidHoldoutAuthorization,
    build_holdout_application_plan,
)
from scripts.p7_test_protocol_canary import RESULT_PATH, ProtocolCanaryReport  # noqa: E402

NOW = datetime(2026, 10, 29, 4, tzinfo=UTC)
CONTROLLER = "c" * 40


@pytest.fixture(autouse=True)
def forbid_gold_and_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """These control-plane tests must not open source rows or construct executors."""

    original_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name == "source-rows.json":
            raise AssertionError("requalification must not read evaluator source rows")
        return original_open(path, *args, **kwargs)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("requalification must not construct Provider or Docker")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(OpenAIProvider, "__init__", forbidden)
    monkeypatch.setattr(DockerSandbox, "__init__", forbidden)


def _recorded_qualified_objects() -> dict[str, str]:
    """Validate saved fixture bindings, not the historical source that produced them."""

    freeze = requalification.PostRepairFreeze.model_validate_json(
        (PROJECT_ROOT / requalification.POST_REPAIR_FREEZE_PATH).read_bytes()
    )
    _validate_content_digest(freeze)
    assert freeze.qualified_runtime_commit == requalification.QUALIFIED_COMMIT
    assert tuple(freeze.semantic_objects) == requalification.SEMANTIC_PATHS
    assert freeze.semantic_objects_sha256 == _canonical_sha256(freeze.semantic_objects)
    return dict(freeze.semantic_objects)


@pytest.fixture
def qualified_objects(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    bindings = _recorded_qualified_objects()
    monkeypatch.setattr(
        requalification, "verify_qualified_runtime", lambda root: (CONTROLLER, bindings)
    )
    return bindings


def test_recorded_qualified_objects_match_original_git_history_when_available() -> None:
    bindings = _recorded_qualified_objects()
    present = subprocess.run(
        ("git", "cat-file", "-e", f"{requalification.QUALIFIED_COMMIT}^{{commit}}"),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    if present.returncode != 0:
        pytest.skip("original private development history is not included in the public snapshot")
    paths = requalification.SEMANTIC_PATHS
    objects = requalification._git(
        PROJECT_ROOT,
        "rev-parse",
        *(f"{requalification.QUALIFIED_COMMIT}:{path}" for path in paths),
    ).splitlines()
    assert dict(zip(paths, objects, strict=True)) == bindings


def _build() -> tuple[requalification.PostRepairFreeze, requalification.HoldoutApplicationPlanV2]:
    return requalification.build_requalification(PROJECT_ROOT, frozen_at=NOW)


def _canary() -> ProtocolCanaryReport:
    return ProtocolCanaryReport.model_validate_json((PROJECT_ROOT / RESULT_PATH).read_bytes())


def test_real_canary_build_is_hash_bound_unpaid_and_not_current_calibration(
    qualified_objects: dict[str, str],
) -> None:
    raw = (PROJECT_ROOT / RESULT_PATH).read_bytes()
    canary = _canary()
    freeze, plan = _build()
    historical = FormalFreezeMarker.model_validate_json((PROJECT_ROOT / FREEZE_PATH).read_bytes())

    assert hashlib.sha256(raw).hexdigest() == requalification.CANARY_RAW_SHA256
    assert freeze.canary_raw_sha256 == plan.canary_raw_sha256 == requalification.CANARY_RAW_SHA256
    assert freeze.canary_content_sha256 == canary.content_sha256
    assert freeze.qualified_runtime_commit == plan.frozen_source_commit == canary.controller_commit
    assert freeze.controller_commit == plan.controller_commit == CONTROLLER
    assert freeze.semantic_objects == qualified_objects
    assert freeze.semantic_objects_sha256 == _canonical_sha256(qualified_objects)
    assert freeze.historical_freeze_content_sha256 == historical.content_sha256
    assert freeze.historical_calibration_source_commit == historical.source_commit
    assert freeze.historical_calibration_source_commit != freeze.qualified_runtime_commit
    assert (
        freeze.historical_calibration_report_raw_sha256 == historical.calibration_report_ref.sha256
    )
    assert freeze.current_source_calibration_performed is False
    assert freeze.evaluator_only_holdout_qualification_performed is True
    assert freeze.provider_requests_made == freeze.paid_requests_made == 0
    assert freeze.formal_environment_ready is False
    assert plan.ready_for_paid_holdout is False
    assert plan.authorization_required is True
    assert plan.freeze_marker_sha256 == freeze.content_sha256
    assert (
        freeze.environment_raw_sha256
        == plan.environment_raw_sha256
        == canary.environment_raw_sha256
    )
    assert freeze.agent_inputs_raw_sha256 == plan.agent_inputs_raw_sha256
    assert (
        freeze.test_inputs_raw_sha256
        == plan.test_inputs_raw_sha256
        == canary.test_inputs_raw_sha256
    )
    assert len(freeze.task_ids) == 8
    assert sum(item.tests_run for task in canary.tasks for item in task.baseline) == 198
    assert sum(item.failures for task in canary.tasks for item in task.baseline) == 17
    assert sum(item.errors for task in canary.tasks for item in task.baseline) == 1
    assert sum(item.tests_run for task in canary.tasks for item in task.gold) == 198
    for artifact in (freeze, plan):
        _validate_content_digest(artifact)
        serialized = artifact.model_dump_json()
        assert '"expected_patch"' not in serialized
        assert '"regression_patch"' not in serialized
        assert '"gold_patch"' not in serialized


def test_v2_slots_and_freeze_cannot_reuse_old_authorization(
    qualified_objects: dict[str, str],
) -> None:
    freeze, plan = _build()
    old = HoldoutApplicationPlan.model_validate_json((PROJECT_ROOT / PLAN_PATH).read_bytes())
    authorization = PaidHoldoutAuthorization.model_validate_json(
        (PROJECT_ROOT / "results/p7-holdout/authorization-v1.json").read_bytes()
    )
    assert len(plan.run_slots) == 32
    assert len({slot.run_id for slot in plan.run_slots}) == 32
    assert {slot.run_id for slot in plan.run_slots}.isdisjoint(
        slot.run_id for slot in old.run_slots
    )
    assert {slot.output_path for slot in plan.run_slots}.isdisjoint(
        slot.output_path for slot in old.run_slots
    )
    assert authorization.plan_sha256 == old.content_sha256 != plan.content_sha256
    assert authorization.freeze_marker_sha256 == old.freeze_marker_sha256 != freeze.content_sha256
    assert plan.matrix_sha256 == old.matrix_sha256
    assert plan.model == old.model
    assert plan.prompt_versions == old.prompt_versions
    assert plan.theoretical_application_cost_usd == old.theoretical_application_cost_usd
    assert plan.tuning_from_holdout_forbidden
    for new_slot, old_slot in zip(plan.run_slots, old.run_slots, strict=True):
        assert new_slot.arm_id == old_slot.arm_id
        assert new_slot.task_id == old_slot.task_id
        assert new_slot.configuration_sha256 == old_slot.configuration_sha256
        assert new_slot.base_commit == old_slot.base_commit
        assert new_slot.python_version == old_slot.python_version


@pytest.mark.parametrize(
    "field", ["current_source_calibration_performed", "formal_environment_ready"]
)
def test_freeze_cannot_claim_new_calibration_or_paid_readiness(
    qualified_objects: dict[str, str], field: str
) -> None:
    freeze, _ = _build()
    payload = freeze.model_dump(exclude={"content_sha256"})
    payload[field] = True
    with pytest.raises(ValidationError):
        requalification._sealed(requalification.PostRepairFreeze, payload)


@pytest.mark.parametrize(
    "baseline,changes",
    [
        (False, {"tests_run": 0}),
        (False, {"truncated": True}),
        (False, {"verifier_accepts_evidence": False}),
        (False, {"failures": 1}),
        (False, {"errors": 1}),
        (False, {"failing_test_count": 1}),
        (False, {"exit_code": 1}),
        (False, {"category": enums.TestRunCategory.ASSERTION_FAILURE}),
        (False, {"status": enums.TestStatus.ERROR}),
        (True, {"tests_run": 0}),
        (True, {"tests_run": 1, "failures": 1, "errors": 1}),
        (True, {"exit_code": 0}),
        (True, {"exit_code": None}),
        (True, {"failures": 0, "errors": 0}),
        (True, {"failing_test_count": 0}),
        (True, {"failing_test_count": 2}),
        (True, {"category": enums.TestRunCategory.PASSED}),
        (True, {"status": enums.TestStatus.PASSED}),
    ],
)
def test_completed_evidence_rejects_contradictions(baseline: bool, changes: dict[str, Any]) -> None:
    task = _canary().tasks[0]
    item = task.baseline[0] if baseline else task.gold[0]
    with pytest.raises(ValueError, match="completed test evidence"):
        requalification._validate_completed_evidence(item.model_copy(update=changes))


def test_all_real_baseline_and_gold_evidence_is_completed() -> None:
    for task in _canary().tasks:
        for item in (*task.baseline, *task.gold):
            requalification._validate_completed_evidence(item)


@pytest.mark.parametrize("target", ["canary", "calibration"])
def test_raw_tampering_is_rejected_even_when_json_meaning_is_unchanged(
    qualified_objects: dict[str, str], monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    historical = FormalFreezeMarker.model_validate_json((PROJECT_ROOT / FREEZE_PATH).read_bytes())
    relative = (
        RESULT_PATH if target == "canary" else Path(historical.calibration_report_ref.relative_path)
    )
    original_read = requalification._read

    def changed_read(root: Path, path: Path, limit: int = 4_000_000) -> bytes:
        raw = original_read(root, path, limit)
        return raw + b"\n" if path == relative else raw

    monkeypatch.setattr(requalification, "_read", changed_read)
    with pytest.raises(ValueError, match=r"raw digest|binding mismatch"):
        _build()


@pytest.mark.parametrize("field", ["base_commit", "python_version"])
def test_historical_template_cannot_cross_bind_task_identity(
    qualified_objects: dict[str, str], monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    original = build_holdout_application_plan

    def wrong_template(**kwargs: Any) -> HoldoutApplicationPlan:
        plan = original(**kwargs)
        changed = plan.run_slots[0].model_copy(
            update={
                field: "a" * 40
                if field == "base_commit"
                else ("3.12" if plan.run_slots[0].python_version != "3.12" else "3.11")
            }
        )
        return plan.model_copy(update={"run_slots": (changed, *plan.run_slots[1:])})

    monkeypatch.setattr(requalification, "build_holdout_application_plan", wrong_template)
    with pytest.raises(ValueError, match="binding mismatch"):
        _build()


def _serve_saved_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    freeze: requalification.PostRepairFreeze,
    plan: requalification.HoldoutApplicationPlanV2,
) -> None:
    original_read = requalification._read
    original_git = requalification._git
    saved: dict[Path, bytes] = {
        requalification.POST_REPAIR_FREEZE_PATH: freeze.model_dump_json().encode(),
        requalification.PLAN_V2_PATH: plan.model_dump_json().encode(),
    }

    def read_saved(root: Path, relative: Path, limit: int = 4_000_000) -> bytes:
        if relative in saved:
            return saved[relative]
        return original_read(root, relative, limit)

    def recorded_identity(root: Path, *args: str) -> str:
        # This loader fixture mocks the source-verification boundary. Historical
        # object bytes are checked independently above only when their commit exists.
        recorded_objects = {
            f"{freeze.qualified_runtime_commit}:{path}": blob
            for path, blob in freeze.semantic_objects.items()
        }
        if (
            len(args) > 1
            and args[0] == "rev-parse"
            and all(argument in recorded_objects for argument in args[1:])
        ):
            return "\n".join(recorded_objects[argument] for argument in args[1:])
        if args in (
            ("merge-base", "--is-ancestor", CONTROLLER, CONTROLLER),
            ("merge-base", "--is-ancestor", requalification.QUALIFIED_COMMIT, CONTROLLER),
            (
                "diff",
                "--name-only",
                requalification.QUALIFIED_COMMIT,
                CONTROLLER,
                "--",
                *requalification.SEMANTIC_PATHS,
            ),
        ):
            return ""
        if args == ("rev-parse", f"{CONTROLLER}:{requalification.CONTROLLER_PATH}"):
            return "a" * 40
        return original_git(root, *args)

    monkeypatch.setattr(requalification, "_read", read_saved)
    monkeypatch.setattr(requalification, "_git", recorded_identity)


def test_loader_rebuilds_exact_saved_freeze_and_plan(
    qualified_objects: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, plan = _build()
    _serve_saved_artifacts(monkeypatch, freeze, plan)
    assert requalification.load_requalification(PROJECT_ROOT) == (freeze, plan)


@pytest.mark.parametrize("field", ["configuration_sha256", "base_commit", "python_version"])
def test_loader_rejects_self_rehashed_slot_identity(
    qualified_objects: dict[str, str], monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    freeze, plan = _build()
    values = {
        "configuration_sha256": "a" * 64,
        "base_commit": "a" * 40,
        "python_version": "3.12" if plan.run_slots[0].python_version != "3.12" else "3.11",
    }
    changed = plan.run_slots[0].model_copy(update={field: values[field]})
    payload = plan.model_dump(exclude={"content_sha256"})
    payload["run_slots"] = (changed, *plan.run_slots[1:])
    changed_plan = requalification._sealed(requalification.HoldoutApplicationPlanV2, payload)
    _serve_saved_artifacts(monkeypatch, freeze, changed_plan)
    with pytest.raises(ValueError, match="freeze/plan binding mismatch"):
        requalification.load_requalification(PROJECT_ROOT)


@pytest.mark.parametrize(
    "field", ["freeze_marker_sha256", "environment_raw_sha256", "test_inputs_raw_sha256"]
)
def test_loader_rejects_self_rehashed_cross_file_binding(
    qualified_objects: dict[str, str], monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    freeze, plan = _build()
    payload = plan.model_dump(exclude={"content_sha256"})
    payload["run_slots"] = plan.run_slots
    payload[field] = "0" * 64
    changed_plan = requalification._sealed(requalification.HoldoutApplicationPlanV2, payload)
    _serve_saved_artifacts(monkeypatch, freeze, changed_plan)
    with pytest.raises(ValueError, match="freeze/plan binding mismatch"):
        requalification.load_requalification(PROJECT_ROOT)


def _commit(root: Path, message: str) -> str:
    requalification._git(root, "add", ".")
    requalification._git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        message,
    )
    return requalification._git(root, "rev-parse", "HEAD")


@pytest.fixture
def runtime_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path.resolve()
    requalification._git(root, "init", "--quiet")
    (root / "runtime.py").write_text("qualified runtime\n")
    (root / "runtime-package").mkdir()
    (root / "runtime-package/module.py").write_text("qualified module\n")
    qualified = _commit(root, "Qualified runtime")
    (root / "scripts").mkdir()
    (root / "scripts/controller.py").write_text("committed controller\n")
    _commit(root, "Add requalification controller")
    monkeypatch.setattr(requalification, "QUALIFIED_COMMIT", qualified)
    monkeypatch.setattr(requalification, "SEMANTIC_PATHS", ("runtime-package", "runtime.py"))
    monkeypatch.setattr(requalification, "CONTROLLER_PATH", "scripts/controller.py")
    return root


def test_runtime_verification_accepts_descendant_and_unrelated_untracked_note(
    runtime_repository: Path,
) -> None:
    root = runtime_repository
    (root / "user-note.md").write_text("unrelated note\n")
    controller, objects = requalification.verify_qualified_runtime(root)
    assert controller == requalification._git(root, "rev-parse", "HEAD")
    assert tuple(objects) == requalification.SEMANTIC_PATHS
    assert objects["runtime.py"] == requalification._git(
        root, "rev-parse", f"{requalification.QUALIFIED_COMMIT}:runtime.py"
    )


@pytest.mark.parametrize("change", ["dirty", "untracked", "committed_drift"])
def test_runtime_verification_rejects_semantic_changes(
    runtime_repository: Path, change: str
) -> None:
    root = runtime_repository
    if change == "untracked":
        (root / "runtime-package/new_module.py").write_text("unqualified module\n")
    else:
        (root / "runtime.py").write_text("unqualified runtime\n")
        if change == "committed_drift":
            _commit(root, "Runtime drift")
    with pytest.raises(ValueError, match=r"unclean|runtime drift"):
        requalification.verify_qualified_runtime(root)


def test_runtime_verification_rejects_controller_bytes_hidden_from_status(
    runtime_repository: Path,
) -> None:
    root = runtime_repository
    relative = requalification.CONTROLLER_PATH
    requalification._git(root, "update-index", "--assume-unchanged", "--", relative)
    (root / relative).write_text("uncommitted controller\n")
    with pytest.raises(ValueError, match="controller is not committed"):
        requalification.verify_qualified_runtime(root)


@pytest.mark.parametrize("relative", ["runtime.py", "runtime-package/module.py"])
def test_runtime_verification_rejects_semantic_bytes_hidden_from_status(
    runtime_repository: Path, relative: str
) -> None:
    root = runtime_repository
    requalification._git(root, "update-index", "--assume-unchanged", "--", relative)
    (root / relative).write_text("unqualified runtime hidden from git status\n")
    with pytest.raises(ValueError, match=r"runtime|source|semantic"):
        requalification.verify_qualified_runtime(root)


def test_runtime_verification_rejects_controller_absent_from_commit(
    runtime_repository: Path,
) -> None:
    root = runtime_repository
    relative = requalification.CONTROLLER_PATH
    content = (root / relative).read_bytes()
    requalification._git(root, "rm", "--quiet", "--", relative)
    _commit(root, "Remove controller")
    (root / relative).parent.mkdir(exist_ok=True)
    (root / relative).write_bytes(content)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        requalification.verify_qualified_runtime(root)
