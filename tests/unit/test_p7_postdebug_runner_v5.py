"""Offline v5 carry reconstruction, version separation and direct-execution gates."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from repopilot.enums import EvaluationRunnerErrorCode  # noqa: E402
from scripts import p7_postdebug_runner_v5 as runner  # noqa: E402
from scripts.p7_holdout_requalification import _sealed  # noqa: E402
from tests.unit.test_p7_holdout_runner_v3 import _outcomes  # noqa: E402


@pytest.fixture
def previous(tmp_path, monkeypatch):
    old = runner.v4.PostDebugPlan.model_validate_json((ROOT / runner.v4.PLAN_PATH).read_bytes())
    for name in (*old.input_files, "uv.lock", runner.v4.PLAN_PATH.as_posix()):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    auth = runner.v4.authorization(tmp_path, old, execute=True)
    outcomes = list(_outcomes(SimpleNamespace(plan=old)))
    for index in range(9):
        item = outcomes[index]
        outcomes[index] = runner.shared.HoldoutSlotOutcome(
            **item.model_dump(
                exclude={
                    "attempt_status",
                    "run_status",
                    "termination_reason",
                    "task_success",
                    "usage",
                    "technical_failure_code",
                    "reserved_unknown_cost_usd",
                }
            ),
            attempt_status="technical_failure",
            technical_failure_code=EvaluationRunnerErrorCode.EXECUTION_FAILURE,
            reserved_unknown_cost_usd=Decimal(1),
        )
    outcomes[9] = outcomes[9].model_copy(
        update={
            "usage": outcomes[9].usage.model_copy(
                update={"estimated_cost_usd": Decimal("2.315823125")}
            )
        }
    )
    outcomes = tuple(outcomes)
    monkeypatch.setattr(runner, "_prior_outcomes", lambda *_: outcomes)
    progress = runner.shared._Progress(
        outcomes, (), (), Decimal("4.515823125"), Decimal(9), old.carried_cost_usd
    )
    controls = SimpleNamespace(plan=old, authorization=auth, controller_commit="b" * 40)
    runner.v4.write_report(tmp_path, controls, progress)
    for slot in old.run_slots:
        directory = tmp_path / slot.output_path
        directory.mkdir(parents=True)
        (directory / "fixture.txt").write_text("immutable raw fixture")
    return old, outcomes


@pytest.fixture
def plan(tmp_path, monkeypatch, previous):
    old, _ = previous
    source = {"src/repopilot/agent/repair.py": "a" * 40}
    current = {**source, "scripts/p7_postdebug_runner_v5.py": "b" * 40}
    monkeypatch.setattr(runner.v4.workload, "verify_source", lambda _: ("c" * 40, current))
    monkeypatch.setattr(runner.v3, "_objects_at", lambda *_: source)
    built = runner.build_plan(tmp_path, tmp_path)
    assert set(old.prior_raw_files) <= set(built.prior_raw_files)
    (tmp_path / runner.PLAN_PATH).write_text(built.model_dump_json())
    return built


def test_cumulative_budget_rebuilt_from_raw_and_all_prior_spend(tmp_path, previous):
    old, _ = previous
    assert runner.prior_cost(tmp_path, old) == (Decimal("5.256243125"), Decimal(11))


def test_v5_separate_full_matrix_and_common_prompt_override(plan, previous):
    old, _ = previous
    assert plan.carried_cost_usd == Decimal("16.256243125")
    assert plan.new_execution_budget_usd == Decimal("15.743756875")
    assert plan.expensive_preflight_required is False
    assert plan.fresh_blind_holdout is False
    assert plan.prompt_versions == (runner.PROMPT_VERSION,)
    assert plan.matrix_prompt_versions == old.prompt_versions == ("repopilot-agent-v1",)
    assert len(plan.run_slots) == len({s.run_id for s in plan.run_slots}) == 32
    for original, new in zip(old.run_slots, plan.run_slots, strict=True):
        assert original.run_id != new.run_id
        assert original.output_path != new.output_path
        assert original.configuration_sha256 == new.configuration_sha256
        assert original.arm_id == new.arm_id and original.task_id == new.task_id
        assert "/p7-postdebug-v5-" in new.output_path


def test_versions_are_not_interchangeable(plan, previous):
    with pytest.raises(ValidationError):
        runner.v4.PostDebugPlan.model_validate_json(plan.model_dump_json())
    with pytest.raises(ValidationError):
        runner.PostDebugPlanV5.model_validate_json(previous[0].model_dump_json())


def test_future_prompt_can_read_historical_plan_but_cannot_build_or_execute(
    tmp_path, plan, monkeypatch
):
    monkeypatch.setattr(runner, "PROMPT_VERSION", "repopilot-agent-v3")
    assert runner.PostDebugPlanV5.model_validate_json(plan.model_dump_json()) == plan
    with pytest.raises(ValueError, match="runtime_prompt_version_mismatch"):
        runner.validate_source_and_inputs(tmp_path, plan)
    with pytest.raises(ValueError, match="runtime_prompt_version_mismatch"):
        runner.build_plan(tmp_path, tmp_path)


@pytest.mark.parametrize(
    "changes",
    [
        {"fresh_blind_holdout": True},
        {"application_budget_usd": Decimal(33)},
        {"per_task_cost_limit_usd": Decimal(2)},
        {"prior_unknown_cost_usd": Decimal(0)},
        {"carried_cost_usd": Decimal(0)},
        {"prompt_versions": ("repopilot-agent-v1",)},
        {"matrix_prompt_versions": ("repopilot-agent-v2",)},
        {"run_slots": ()},
    ],
)
def test_resealed_invalid_controls_rejected(plan, changes):
    with pytest.raises((ValidationError, ValueError)):
        _sealed(
            runner.PostDebugPlanV5,
            {
                **plan.model_dump(exclude={"content_sha256"}),
                "run_slots": plan.run_slots,
                **changes,
            },
        )


def _mutate_report(root, changes):
    target = root / runner.v4.REPORT_PATH
    payload = json.loads(target.read_bytes())
    payload.update(changes)
    payload.pop("content_sha256")
    payload["content_sha256"] = runner.shared._canonical_sha256(payload)
    target.write_text(json.dumps(payload))


@pytest.mark.parametrize(
    "change",
    [
        {"known_cost_usd": "0"},
        {"reserved_unknown_cost_usd": "0"},
        {"carried_cost_usd": "0"},
        {"total_accounted_cost_usd": "0"},
        {"outcomes": []},
        {"technical_failure_slots": 0},
        {"fresh_blind_holdout": True},
    ],
)
def test_resealed_prior_report_cannot_hide_raw_cost_or_outcomes(tmp_path, previous, change):
    _mutate_report(tmp_path, change)
    with pytest.raises(ValueError, match="prior_report_raw_mismatch"):
        runner.prior_cost(tmp_path, previous[0])


def test_prior_authorization_must_bind_original_plan(tmp_path, previous):
    auth = runner.v4.PostDebugAuthorization.model_validate_json(
        (tmp_path / runner.v4.AUTH_PATH).read_bytes()
    )
    changed = _sealed(
        runner.v4.PostDebugAuthorization,
        {
            **auth.model_dump(exclude={"content_sha256"}),
            "plan_raw_sha256": "0" * 64,
        },
    )
    (tmp_path / runner.v4.AUTH_PATH).write_text(changed.model_dump_json())
    with pytest.raises(ValueError, match="prior_authorization_mismatch"):
        runner.prior_cost(tmp_path, previous[0])


def test_missing_previous_slot_rejected_without_running_provider(tmp_path, previous, monkeypatch):
    monkeypatch.undo()
    old, _ = previous
    # The original function requires all 32 paths before any paid construction.
    with pytest.raises(ValueError, match="prior_matrix_incomplete"):
        runner._prior_outcomes(tmp_path / "absent", old)


@pytest.fixture
def raw_manifest_fixture(tmp_path, monkeypatch):
    """Use actual manifest/schema validation; replace only the raw history reader."""
    old = runner.v4.PostDebugPlan.model_validate_json((ROOT / runner.v4.PLAN_PATH).read_bytes())
    controls = SimpleNamespace(
        plan=old,
        environment=runner.v4.HoldoutEnvironmentManifest.model_validate_json(
            (ROOT / runner.v4.ENVIRONMENT_PATH).read_bytes()
        ),
        agent_inputs=runner.v4.HoldoutAgentInputs.model_validate_json(
            (ROOT / runner.v4.AGENT_INPUTS_PATH).read_bytes()
        ),
    )
    monkeypatch.setattr(runner.shared, "_checkout", lambda _root, _binding: ROOT)
    reports = {}
    for slot in old.run_slots:
        manifest = runner.shared._expected_manifest(ROOT, controls, slot)
        directory = tmp_path / slot.output_path
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(manifest.model_dump_json())
        reports[directory] = SimpleNamespace(
            run_id=slot.run_id,
            attempts=(
                SimpleNamespace(
                    status="failed",
                    failure_code=EvaluationRunnerErrorCode.EXECUTION_FAILURE,
                    evidence=None,
                    execution_count=1,
                ),
            ),
        )
    monkeypatch.setattr(runner, "inspect_evaluation_run", reports.__getitem__)
    return old, reports


def test_all_previous_raw_slots_are_required_and_reconstructed(tmp_path, raw_manifest_fixture):
    old, _ = raw_manifest_fixture
    outcomes = runner._prior_outcomes(tmp_path, old)
    assert len(outcomes) == 32
    assert all(o.attempt_status == "technical_failure" for o in outcomes)
    assert sum(o.reserved_unknown_cost_usd for o in outcomes) == Decimal(32)


def test_previous_active_slot_blocks_new_run(tmp_path, raw_manifest_fixture):
    old, reports = raw_manifest_fixture
    reports[tmp_path / old.run_slots[0].output_path].attempts[0].status = "started"
    with pytest.raises(ValueError, match="prior_execution_still_running"):
        runner._prior_outcomes(tmp_path, old)


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_versions", ("repopilot-agent-v2",)),
        ("source_commit", "0" * 40),
    ],
)
def test_resealed_prior_manifest_must_match_v4_identity(
    tmp_path, raw_manifest_fixture, field, value
):
    old, _ = raw_manifest_fixture
    path = tmp_path / old.run_slots[0].output_path / "manifest.json"
    manifest = runner.shared.EvaluationRunManifest.model_validate_json(path.read_bytes())
    changed = _sealed(
        runner.shared.EvaluationRunManifest,
        {
            **manifest.model_dump(exclude={"content_sha256"}),
            field: value,
        },
    )
    path.write_text(changed.model_dump_json())
    with pytest.raises(ValueError, match="prior_run_mismatch"):
        runner._prior_outcomes(tmp_path, old)


def test_only_explicit_reviewed_source_changes_are_allowed(tmp_path, previous, monkeypatch):
    before = {"src/repopilot/retrieval/dense.py": "a" * 40}
    monkeypatch.setattr(runner.v3, "_objects_at", lambda *_: before)
    monkeypatch.setattr(
        runner.v4.workload,
        "verify_source",
        lambda _: ("c" * 40, {"src/repopilot/retrieval/dense.py": "b" * 40}),
    )
    with pytest.raises(ValueError, match="unreviewed_repair_scope"):
        runner.build_plan(tmp_path, tmp_path)


def test_runtime_preserves_upstream_and_v4_raw_inventory(tmp_path, plan, previous, monkeypatch):
    old, _ = previous
    actual_inventory = runner.v4.prior_inventory
    monkeypatch.setattr(runner, "_git", lambda *_: "")

    def inventory(root, prior):
        return (
            dict(old.prior_raw_files)
            if isinstance(prior, runner.v3.GpuPlan)
            else actual_inventory(root, prior)
        )

    monkeypatch.setattr(runner.v4, "prior_inventory", inventory)
    assert runner.validate_source_and_inputs(tmp_path, plan) == "c" * 40
    (tmp_path / old.run_slots[0].output_path / "fixture.txt").write_text("tampered")
    with pytest.raises(ValueError, match="prior_results_changed"):
        runner.validate_source_and_inputs(tmp_path, plan)


def test_authorization_and_report_never_overwrite_v4(tmp_path, plan, previous):
    old_auth = (tmp_path / runner.v4.AUTH_PATH).read_bytes()
    old_report = (tmp_path / runner.v4.REPORT_PATH).read_bytes()
    draft = runner.authorization(tmp_path, plan, execute=False)
    assert not (tmp_path / runner.AUTH_PATH).exists()
    auth = runner.authorization(tmp_path, plan, execute=True)
    assert auth.plan_sha256 == draft.plan_sha256 == plan.content_sha256
    assert auth.carried_cost_usd == Decimal("16.256243125")
    assert runner.authorization(tmp_path, plan, execute=True) == auth
    with pytest.raises(ValidationError):
        runner.v4.PostDebugAuthorization.model_validate_json(auth.model_dump_json())
    outcomes = _outcomes(SimpleNamespace(plan=plan))
    progress = runner.shared._Progress(
        outcomes, (), (), Decimal("3.20"), Decimal(0), plan.carried_cost_usd
    )
    controls = SimpleNamespace(plan=plan, authorization=auth, controller_commit="b" * 40)
    report = runner.write_report(tmp_path, controls, progress)
    assert report["report_version"] == "post-debug-rerun-report-v5"
    assert Decimal(report["total_accounted_cost_usd"]) == Decimal("19.456243125")
    assert (tmp_path / runner.v4.AUTH_PATH).read_bytes() == old_auth
    assert (tmp_path / runner.v4.REPORT_PATH).read_bytes() == old_report
    assert runner.write_report(tmp_path, controls, progress) == report


@pytest.mark.parametrize("missing", ["--skip-preflight", "--acknowledge-post-debug", "digest"])
def test_execute_requires_exact_v5_consent(tmp_path, plan, monkeypatch, missing):
    args = [
        "runner",
        "--repository-root",
        str(tmp_path),
        "--execute",
        "--skip-preflight",
        "--acknowledge-post-debug",
        "--confirm-plan-sha256",
        plan.content_sha256,
    ]
    if missing == "digest":
        args[-1] = "0" * 64
    else:
        args.remove(missing)
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner, "load_controls", lambda *_a, **_kw: pytest.fail("not authorized"))
    with pytest.raises(ValueError):
        runner.main()
    assert not (tmp_path / runner.AUTH_PATH).exists()


@pytest.mark.parametrize("maximum", [1, 32])
def test_direct_execution_reuses_resources_with_cumulative_cap(
    tmp_path, plan, monkeypatch, maximum
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--repository-root",
            str(tmp_path),
            "--execute",
            "--skip-preflight",
            "--acknowledge-post-debug",
            "--confirm-plan-sha256",
            plan.content_sha256,
            "--max-new-slots",
            str(maximum),
            "--continue-after-technical-failure",
        ],
    )
    monkeypatch.setattr(runner, "load_controls", lambda *_a, **_kw: SimpleNamespace(plan=plan))
    locks = []

    @contextmanager
    def lock(_root, identity):
        locks.append(identity)
        yield

    monkeypatch.setattr(runner, "_execution_lock", lock)

    def execute(**kwargs):
        assert kwargs["carried_cost_usd"] == Decimal("16.256243125")
        assert kwargs["max_new_slots"] == maximum
        assert kwargs["continue_after_technical_failure"] is True
        assert kwargs["executor_factory_builder"] is runner.v4.executor_factory
        assert kwargs["report_writer"] is runner.write_report
        return 0

    monkeypatch.setattr(runner.shared, "_run_loaded_controls", execute)
    monkeypatch.setattr(
        runner.v4.profile, "verify_qualification", lambda *_: pytest.fail("no preflight")
    )
    monkeypatch.setattr(
        runner.v4.cache_api, "verify_pinned_cache", lambda *_: pytest.fail("no scan")
    )
    assert runner.main() == 0
    assert locks[0] == "p7-paid-evaluation-global"
    assert locks[1].startswith("p7-holdout-v3-")


def test_v5_uses_existing_historical_assets_helper(tmp_path, plan, monkeypatch):
    monkeypatch.setattr(runner.v3, "verify_runtime_environment", lambda: None)
    monkeypatch.setattr(runner, "validate_source_and_inputs", lambda *_: "c" * 40)
    monkeypatch.setenv("OPENAI_API_KEY", "test-never-sent")

    def assets(root, **kwargs):
        assert root == tmp_path and kwargs["plan"] == plan
        assert kwargs["commit"] == "c" * 40
        auth = kwargs["authorize"]()
        assert auth.authorization_version == "post-debug-authorization-v5"
        return SimpleNamespace(plan=plan, authorization=auth)

    monkeypatch.setattr(runner.v4, "historical_assets_controls", assets)
    controls = runner.load_controls(tmp_path, execute=True)
    assert controls.plan.prompt_versions == (runner.PROMPT_VERSION,)
