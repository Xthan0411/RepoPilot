"""Offline v6 controls: historical spend, matched-context baseline, and paid gates."""

from __future__ import annotations

import importlib.util
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

from scripts import p7_postdebug_runner_v6 as runner  # noqa: E402
from scripts.p7_holdout_requalification import _sealed  # noqa: E402
from tests.unit.test_p7_holdout_runner_v3 import _outcomes  # noqa: E402


@pytest.fixture
def previous(tmp_path, monkeypatch):
    old = runner.v5.PostDebugPlanV5.model_validate_json((ROOT / runner.v5.PLAN_PATH).read_bytes())
    matrix = runner.load_experiment_matrix(ROOT / runner.MATRIX_PATH, repository_root=ROOT)
    files = {
        *old.input_files,
        "uv.lock",
        runner.v5.PLAN_PATH.as_posix(),
        matrix.dataset.manifest_path,
        *(arm.baseline_config_ref.relative_path for arm in matrix.arms if arm.baseline_config_ref),
        *(snapshot.manifest_ref.relative_path for snapshot in matrix.memory_snapshots),
    }
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    auth = runner.v5.authorization(tmp_path, old, execute=True)
    outcomes = [
        item.model_copy(
            update={"usage": item.usage.model_copy(update={"estimated_cost_usd": Decimal("0.20")})}
        )
        for item in _outcomes(SimpleNamespace(plan=old))
    ]
    outcomes[0] = outcomes[0].model_copy(
        update={
            "usage": outcomes[0].usage.model_copy(
                update={"estimated_cost_usd": Decimal("0.169348750")}
            )
        }
    )
    outcomes = tuple(outcomes)
    monkeypatch.setattr(runner.v5, "_prior_outcomes", lambda *_: outcomes)
    progress = runner.shared._Progress(
        outcomes, (), (), Decimal("6.369348750"), Decimal(0), old.carried_cost_usd
    )
    runner.v5.write_report(
        tmp_path,
        SimpleNamespace(plan=old, authorization=auth, controller_commit="b" * 40),
        progress,
    )
    for slot in old.run_slots:
        directory = tmp_path / slot.output_path
        directory.mkdir(parents=True)
        (directory / "fixture.txt").write_text("immutable v5 raw fixture")
    return old, outcomes


@pytest.fixture
def plan(tmp_path, monkeypatch, previous):
    source = {"src/repopilot/agent/repair.py": "a" * 40}
    current = {**source, "scripts/p7_postdebug_runner_v6.py": "b" * 40}
    monkeypatch.setattr(runner.v5.workload, "verify_source", lambda _: ("c" * 40, current))
    monkeypatch.setattr(runner.v5.v3, "_objects_at", lambda *_: source)
    built = runner.build_plan(tmp_path, tmp_path)
    (tmp_path / runner.PLAN_PATH).write_text(built.model_dump_json())
    return built


def test_prior_cost_is_recomputed_and_all_unknown_reserves_are_retained(tmp_path, previous):
    old, _ = previous
    assert old.prior_unknown_cost_usd == Decimal(11)
    assert runner.prior_cost(tmp_path, old) == (Decimal("11.625591875"), Decimal(11))


def test_v6_has_new_namespace_50_dollar_cumulative_cap_and_40_slots(plan, previous):
    old, _ = previous
    assert plan.carried_cost_usd == Decimal("22.625591875")
    assert plan.application_budget_usd == Decimal(50)
    assert plan.new_execution_budget_usd == Decimal("27.374408125")
    assert plan.per_task_cost_limit_usd == Decimal(1)
    assert plan.expensive_preflight_required is False
    assert plan.fresh_blind_holdout is False
    assert plan.comparison_basis == "same_initial_context_different_interaction_and_compute"
    assert len(plan.run_slots) == len({slot.run_id for slot in plan.run_slots}) == 40
    assert tuple(slot.arm_id for slot in plan.run_slots[32:]) == (runner.SINGLE_SHOT_ARM_ID,) * 8
    assert tuple(slot.task_id for slot in plan.run_slots[32:]) == runner.EXPECTED_TASK_IDS
    for before, after in zip(old.run_slots, plan.run_slots[:32], strict=True):
        assert before.run_id != after.run_id
        assert before.output_path != after.output_path
        assert before.configuration_sha256 == after.configuration_sha256
        assert (before.arm_id, before.task_id) == (after.arm_id, after.task_id)
    assert set(old.prior_raw_files) <= set(plan.prior_raw_files)


def test_single_shot_configuration_is_explicitly_distinct_from_agent_configuration(tmp_path, plan):
    matrix = runner.load_experiment_matrix(tmp_path / runner.MATRIX_PATH, repository_root=tmp_path)
    hybrid = runner.select_experiment_arm(matrix, "hybrid-rrf-v1")
    snapshot = runner.baseline_configuration(tmp_path)
    assert snapshot["hybrid_configuration"] == hybrid.configuration_snapshot
    assert snapshot["initial_context_source"] == "same_round_hybrid_initial_trace"
    assert snapshot["prompt_version"] == runner.SINGLE_SHOT_PROMPT_VERSION
    assert snapshot["provider_calls"] == 1
    assert snapshot["agent_tools"] is False
    assert snapshot["test_feedback_to_model"] is False
    assert snapshot["transport_retries"] == 0
    assert snapshot["scoring_protocol"] == "isolated-regression-tests-v1"
    assert runner.shared._canonical_sha256(snapshot) == plan.baseline_configuration_sha256
    assert plan.baseline_configuration_sha256 != hybrid.configuration_sha256


def test_v6_and_v5_models_are_not_interchangeable(plan, previous):
    with pytest.raises(ValidationError):
        runner.v5.PostDebugPlanV5.model_validate_json(plan.model_dump_json())
    with pytest.raises(ValidationError):
        runner.PostDebugPlanV6.model_validate_json(previous[0].model_dump_json())


@pytest.mark.parametrize(
    "changes",
    [
        {"application_budget_usd": Decimal(32)},
        {"application_budget_usd": Decimal(51)},
        {"per_task_cost_limit_usd": Decimal(2)},
        {"prior_unknown_cost_usd": Decimal(0)},
        {"carried_cost_usd": Decimal(0)},
        {"new_execution_budget_usd": Decimal(50)},
        {"fresh_blind_holdout": True},
        {"expensive_preflight_required": True},
        {"prompt_versions": ("repopilot-agent-v1",)},
        {"baseline_prompt_version": "repopilot-agent-v2"},
        {"baseline_configuration_sha256": "0" * 64},
        {"comparison_basis": "equal_compute"},
        {"execution_source_sha256": "0" * 64},
        {"run_slots": ()},
    ],
)
def test_invalid_resealed_v6_controls_are_rejected(plan, changes):
    with pytest.raises((ValueError, ValidationError)):
        _sealed(
            runner.PostDebugPlanV6,
            {
                **plan.model_dump(exclude={"content_sha256"}),
                "run_slots": plan.run_slots,
                **changes,
            },
        )


def test_resealed_missing_baseline_slot_is_rejected(plan):
    with pytest.raises((ValueError, ValidationError), match="slot_matrix_changed"):
        _sealed(
            runner.PostDebugPlanV6,
            {
                **plan.model_dump(exclude={"content_sha256"}),
                "run_slots": plan.run_slots[:-1],
            },
        )


@pytest.mark.parametrize(
    "change",
    [
        {"known_cost_usd": "0"},
        {"reserved_unknown_cost_usd": "1"},
        {"carried_cost_usd": "0"},
        {"total_accounted_cost_usd": "0"},
        {"budget_usd": "50"},
        {"outcomes": []},
        {"completed_slots": 31},
        {"technical_failure_slots": 1},
        {"fresh_blind_holdout": True},
    ],
)
def test_resealed_prior_report_cannot_replace_raw_cost_or_outcomes(tmp_path, previous, change):
    path = tmp_path / runner.v5.REPORT_PATH
    payload = json.loads(path.read_bytes())
    payload.update(change)
    payload.pop("content_sha256")
    payload["content_sha256"] = runner.shared._canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="prior_report_raw_mismatch"):
        runner.prior_cost(tmp_path, previous[0])


def test_prior_report_seal_is_required(tmp_path, previous):
    path = tmp_path / runner.v5.REPORT_PATH
    payload = json.loads(path.read_bytes())
    payload["known_cost_usd"] = "0"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="prior_report_seal_mismatch"):
        runner.prior_cost(tmp_path, previous[0])


def test_prior_authorization_cannot_be_rebound_to_another_plan(tmp_path, previous):
    path = tmp_path / runner.v5.AUTH_PATH
    auth = runner.v5.PostDebugAuthorizationV5.model_validate_json(path.read_bytes())
    changed = _sealed(
        runner.v5.PostDebugAuthorizationV5,
        {
            **auth.model_dump(exclude={"content_sha256"}),
            "plan_raw_sha256": "0" * 64,
        },
    )
    path.write_text(changed.model_dump_json())
    with pytest.raises(ValueError, match="prior_authorization_mismatch"):
        runner.prior_cost(tmp_path, previous[0])


def test_unreviewed_source_changes_cannot_enter_v6(tmp_path, previous, monkeypatch):
    before = {"src/repopilot/retrieval/dense.py": "a" * 40}
    monkeypatch.setattr(runner.v5.v3, "_objects_at", lambda *_: before)
    monkeypatch.setattr(
        runner.v5.workload,
        "verify_source",
        lambda _: (
            "c" * 40,
            {"src/repopilot/retrieval/dense.py": "b" * 40},
        ),
    )
    with pytest.raises(ValueError, match="unreviewed_repair_scope"):
        runner.build_plan(tmp_path, tmp_path)


def test_runtime_checks_preserve_old_raw_inventory(tmp_path, plan, previous, monkeypatch):
    old, _ = previous
    original_inventory = runner.v4.prior_inventory
    monkeypatch.setattr(runner, "_git", lambda *_: "")

    def inventory(root, prior):
        if isinstance(prior, runner.v5.v3.GpuPlan):
            return dict(old.prior_raw_files)
        if type(prior) is runner.v4.PostDebugPlan:
            return {}
        return original_inventory(root, prior)

    monkeypatch.setattr(runner.v4, "prior_inventory", inventory)
    assert runner.validate_source_and_inputs(tmp_path, plan) == "c" * 40
    (tmp_path / old.run_slots[0].output_path / "fixture.txt").write_text("tampered")
    with pytest.raises(ValueError, match="prior_results_changed"):
        runner.validate_source_and_inputs(tmp_path, plan)


def test_runtime_prompt_change_preserves_historical_read_but_blocks_execution(
    tmp_path, plan, monkeypatch
):
    monkeypatch.setattr(runner, "PROMPT_VERSION", "repopilot-agent-v99")
    assert runner.PostDebugPlanV6.model_validate_json(plan.model_dump_json()) == plan
    with pytest.raises(ValueError, match="runtime_prompt_version_mismatch"):
        runner.validate_source_and_inputs(tmp_path, plan)


def test_execution_source_change_is_rejected_before_asset_loading(tmp_path, plan, monkeypatch):
    monkeypatch.setattr(
        runner.v5.workload,
        "verify_source",
        lambda _: (
            "d" * 40,
            {**plan.execution_source_objects, "src/repopilot/agent/repair.py": "f" * 40},
        ),
    )
    with pytest.raises(ValueError, match="frozen_execution_source_changed"):
        runner.validate_source_and_inputs(tmp_path, plan)


@pytest.mark.parametrize(
    "name,code",
    [
        ("uv.lock", "lock_changed"),
        (runner.MATRIX_PATH.as_posix(), "frozen_input_changed"),
    ],
)
def test_frozen_input_changes_block_execution(tmp_path, plan, monkeypatch, name, code):
    monkeypatch.setattr(runner, "_git", lambda *_: "")
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b"\n# tampered\n")
    with pytest.raises(ValueError, match=code):
        runner.validate_source_and_inputs(tmp_path, plan)


def test_authorization_is_new_bound_50_dollar_record_and_dry_run_never_writes(tmp_path, plan):
    old_auth = (tmp_path / runner.v5.AUTH_PATH).read_bytes()
    draft = runner.authorization(tmp_path, plan, execute=False)
    assert not (tmp_path / runner.AUTH_PATH).exists()
    auth = runner.authorization(tmp_path, plan, execute=True)
    assert auth.plan_sha256 == draft.plan_sha256 == plan.content_sha256
    assert auth.max_total_cost_usd == Decimal(50)
    assert auth.carried_cost_usd == Decimal("22.625591875")
    assert runner.authorization(tmp_path, plan, execute=True) == auth
    assert (tmp_path / runner.AUTH_PATH).stat().st_mode & 0o777 == 0o600
    assert (tmp_path / runner.v5.AUTH_PATH).read_bytes() == old_auth
    with pytest.raises(ValidationError):
        runner.v5.PostDebugAuthorizationV5.model_validate_json(auth.model_dump_json())


def test_resealed_wrong_current_authorization_binding_is_rejected(tmp_path, plan):
    auth = runner.authorization(tmp_path, plan, execute=True)
    changed = _sealed(
        runner.PostDebugAuthorizationV6,
        {
            **auth.model_dump(exclude={"content_sha256"}),
            "plan_raw_sha256": "0" * 64,
        },
    )
    (tmp_path / runner.AUTH_PATH).write_text(changed.model_dump_json())
    with pytest.raises(ValueError, match="authorization_mismatch"):
        runner.authorization(tmp_path, plan, execute=True)


def test_authorization_cannot_silently_raise_the_cumulative_budget(tmp_path, plan):
    auth = runner.authorization(tmp_path, plan, execute=False)
    with pytest.raises((ValidationError, ValueError), match="budget_changed"):
        _sealed(
            runner.PostDebugAuthorizationV6,
            {
                **auth.model_dump(exclude={"content_sha256"}),
                "max_total_cost_usd": Decimal(51),
            },
        )


@pytest.mark.parametrize("missing", ["--skip-preflight", "--acknowledge-post-debug", "digest"])
def test_execute_requires_exact_v6_consent_before_loading_or_constructing_provider(
    tmp_path, plan, monkeypatch, missing
):
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
    monkeypatch.setattr(runner, "load_controls", lambda *_a, **_kw: pytest.fail("before consent"))
    monkeypatch.setattr(runner, "executor_factory", lambda **_: pytest.fail("no provider"))
    with pytest.raises(ValueError):
        runner.main()
    assert not (tmp_path / runner.AUTH_PATH).exists()


def test_paid_load_requires_credential_before_authorization(tmp_path, plan, monkeypatch):
    monkeypatch.setattr(runner.v5.v3, "verify_runtime_environment", lambda: None)
    monkeypatch.setattr(runner, "validate_source_and_inputs", lambda *_: "c" * 40)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        runner.v4, "historical_assets_controls", lambda *_a, **_kw: pytest.fail("no auth")
    )
    with pytest.raises(ValueError, match="provider_credential_required"):
        runner.load_controls(tmp_path, execute=True)
    assert not (tmp_path / runner.AUTH_PATH).exists()


@pytest.mark.parametrize("execute", [False, True])
def test_loader_reuses_bound_historical_assets_without_paid_execution(
    tmp_path, plan, monkeypatch, execute
):
    monkeypatch.setattr(runner.v5.v3, "verify_runtime_environment", lambda: None)
    monkeypatch.setattr(runner, "validate_source_and_inputs", lambda *_: "c" * 40)
    monkeypatch.setenv("OPENAI_API_KEY", "diagnostic-never-sent")
    monkeypatch.setattr(runner, "executor_factory", lambda **_: pytest.fail("no executor"))

    def assets(root, **kwargs):
        assert root == tmp_path
        assert kwargs["plan"] == plan
        assert kwargs["commit"] == "c" * 40
        auth = kwargs["authorize"]()
        assert auth.authorization_version == "post-debug-authorization-v6"
        assert auth.max_total_cost_usd == Decimal(50)
        return SimpleNamespace(plan=plan, authorization=auth)

    monkeypatch.setattr(runner.v4, "historical_assets_controls", assets)
    controls = runner.load_controls(tmp_path, execute=execute)
    assert controls.plan == plan
    assert (tmp_path / runner.AUTH_PATH).exists() is execute


@pytest.mark.parametrize("maximum", [1, 40])
def test_direct_execution_uses_shared_resume_and_cumulative_budget(
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
        assert kwargs["carried_cost_usd"] == Decimal("22.625591875")
        assert kwargs["max_new_slots"] == maximum
        assert kwargs["continue_after_technical_failure"] is True
        assert kwargs["executor_factory_builder"] is runner.executor_factory
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


def test_report_requires_all_40_slots_and_preserves_old_report(tmp_path, plan):
    old_report = (tmp_path / runner.v5.REPORT_PATH).read_bytes()
    auth = runner.authorization(tmp_path, plan, execute=True)
    controls = SimpleNamespace(plan=plan, authorization=auth, controller_commit="c" * 40)
    outcomes = _outcomes(controls)
    incomplete = runner.shared._Progress(
        outcomes[:32], plan.run_slots[32:], (), Decimal("3.20"), Decimal(0), plan.carried_cost_usd
    )
    with pytest.raises(ValueError, match="matrix_incomplete"):
        runner.write_report(tmp_path, controls, incomplete)
    assert not (tmp_path / runner.REPORT_PATH).exists()
    complete = runner.shared._Progress(
        outcomes, (), (), Decimal(4), Decimal(0), plan.carried_cost_usd
    )
    report = runner.write_report(tmp_path, controls, complete)
    assert report["slot_count"] == report["completed_slots"] == 40
    assert report["budget_usd"] == "50.00"
    assert Decimal(report["total_accounted_cost_usd"]) == Decimal("26.625591875")
    assert report["fresh_blind_holdout"] is False
    assert report["comparison_basis"] == "same_initial_context_different_interaction_and_compute"
    assert runner.write_report(tmp_path, controls, complete) == report
    assert (tmp_path / runner.v5.REPORT_PATH).read_bytes() == old_report
    assert (tmp_path / runner.REPORT_PATH).stat().st_mode & 0o777 == 0o600


def test_dry_run_shared_loop_never_constructs_an_executor(tmp_path, plan, monkeypatch):
    controls = SimpleNamespace(plan=plan)
    monkeypatch.setattr(
        runner.shared,
        "_inspect_progress",
        lambda *_: runner.shared._Progress((), plan.run_slots, (), Decimal(0), Decimal(0)),
    )
    assert (
        runner.shared._run_loaded_controls(
            root=tmp_path,
            controls=controls,
            execute=False,
            confirm_plan_sha256=None,
            max_new_slots=40,
            continue_after_technical_failure=False,
            report_writer=lambda *_: pytest.fail("not complete"),
            executor_factory_builder=lambda **_: pytest.fail("no provider before execution"),
            carried_cost_usd=plan.carried_cost_usd,
        )
        == 0
    )


def test_complete_resume_only_reports_without_constructing_executor(tmp_path, plan, monkeypatch):
    controls = SimpleNamespace(plan=plan)
    outcomes = _outcomes(controls)
    monkeypatch.setattr(
        runner.shared,
        "_inspect_progress",
        lambda *_: runner.shared._Progress(outcomes, (), (), Decimal(4), Decimal(0)),
    )
    reports = []
    assert (
        runner.shared._run_loaded_controls(
            root=tmp_path,
            controls=controls,
            execute=True,
            confirm_plan_sha256=plan.content_sha256,
            max_new_slots=40,
            continue_after_technical_failure=False,
            report_writer=lambda *_: reports.append(True),
            executor_factory_builder=lambda **_: pytest.fail("completed slots cannot rerun"),
            carried_cost_usd=plan.carried_cost_usd,
        )
        == 0
    )
    assert reports == [True]


def test_partial_resume_skips_32_completed_agent_slots_and_runs_next_baseline_only(
    tmp_path, plan, monkeypatch
):
    controls = SimpleNamespace(
        plan=plan, authorization=SimpleNamespace(max_total_cost_usd=Decimal(50))
    )
    outcomes = _outcomes(controls)
    recorded = list(outcomes[:32])
    invoked = []
    constructed = []
    monkeypatch.setenv("OPENAI_API_KEY", "diagnostic-never-sent")

    def progress(*_):
        return runner.shared._Progress(
            tuple(recorded),
            plan.run_slots[len(recorded) :],
            (),
            sum((item.usage.estimated_cost_usd for item in recorded), Decimal(0)),
            Decimal(0),
        )

    def execute_slot(**kwargs):
        slot = kwargs["slot"]
        invoked.append((slot.arm_id, slot.task_id))
        assert slot == plan.run_slots[32]
        assert kwargs["accounted_cost_before"] == plan.carried_cost_usd + Decimal("3.2")
        recorded.append(outcomes[32])

    def factory(**_):
        def build(arm, task):
            constructed.append((arm, task))
            return object()

        return build

    monkeypatch.setattr(runner.shared, "_inspect_progress", progress)
    monkeypatch.setattr(runner.shared, "_execute_slot", execute_slot)
    assert (
        runner.shared._run_loaded_controls(
            root=tmp_path,
            controls=controls,
            execute=True,
            confirm_plan_sha256=plan.content_sha256,
            max_new_slots=1,
            continue_after_technical_failure=False,
            report_writer=lambda *_: pytest.fail("remaining baseline slots are incomplete"),
            executor_factory_builder=factory,
            carried_cost_usd=plan.carried_cost_usd,
        )
        == 0
    )
    assert invoked == constructed == [(runner.SINGLE_SHOT_ARM_ID, runner.EXPECTED_TASK_IDS[0])]


def test_cumulative_budget_includes_carry_before_new_slot_is_constructed(
    tmp_path, plan, monkeypatch
):
    controls = SimpleNamespace(
        plan=plan, authorization=SimpleNamespace(max_total_cost_usd=Decimal(50))
    )
    monkeypatch.setenv("OPENAI_API_KEY", "diagnostic-never-sent")
    monkeypatch.setattr(
        runner.shared,
        "_inspect_progress",
        lambda *_: runner.shared._Progress((), plan.run_slots, (), Decimal(27), Decimal(0)),
    )
    monkeypatch.setattr(runner.shared, "_execute_slot", lambda **_: pytest.fail("budget exhausted"))
    with pytest.raises(runner.shared.HoldoutRunnerError) as caught:
        runner.shared._run_loaded_controls(
            root=tmp_path,
            controls=controls,
            execute=True,
            confirm_plan_sha256=plan.content_sha256,
            max_new_slots=1,
            continue_after_technical_failure=False,
            report_writer=lambda *_: None,
            executor_factory_builder=lambda **_: lambda *_: pytest.fail("no slot executor"),
            carried_cost_usd=plan.carried_cost_usd,
        )
    assert caught.value.code is runner.shared.HoldoutRunnerErrorCode.BUDGET_EXHAUSTED


def test_baseline_manifest_has_its_own_prompt_and_configuration(tmp_path, plan, monkeypatch):
    controls = SimpleNamespace(
        plan=plan,
        environment=runner.v4.HoldoutEnvironmentManifest.model_validate_json(
            (ROOT / runner.v4.ENVIRONMENT_PATH).read_bytes()
        ),
        agent_inputs=runner.v4.HoldoutAgentInputs.model_validate_json(
            (ROOT / runner.v4.AGENT_INPUTS_PATH).read_bytes()
        ),
    )
    monkeypatch.setattr(runner.shared, "_checkout", lambda *_: ROOT)
    agent_slot = next(slot for slot in plan.run_slots if slot.arm_id == "hybrid-rrf-v1")
    baseline_slot = plan.run_slots[32]
    agent = runner.shared._expected_manifest(ROOT, controls, agent_slot)
    baseline = runner.shared._expected_manifest(ROOT, controls, baseline_slot)
    assert agent.prompt_versions == (runner.PROMPT_VERSION,)
    assert baseline.prompt_versions == (runner.SINGLE_SHOT_PROMPT_VERSION,)
    assert baseline.configuration_sha256 == baseline_slot.configuration_sha256
    assert baseline.configuration_sha256 != agent.configuration_sha256
    assert baseline.model == agent.model
    assert baseline.model_provider == agent.model_provider
    assert baseline.task_ids == agent.task_ids
    assert baseline.sandbox_image_reference == agent.sandbox_image_reference
    assert baseline.source_commit == agent.source_commit == plan.frozen_source_commit


def test_baseline_manifest_configuration_tampering_is_rejected(tmp_path, plan, monkeypatch):
    controls = SimpleNamespace(
        plan=plan,
        environment=runner.v4.HoldoutEnvironmentManifest.model_validate_json(
            (ROOT / runner.v4.ENVIRONMENT_PATH).read_bytes()
        ),
        agent_inputs=runner.v4.HoldoutAgentInputs.model_validate_json(
            (ROOT / runner.v4.AGENT_INPUTS_PATH).read_bytes()
        ),
    )
    monkeypatch.setattr(runner.shared, "_checkout", lambda *_: ROOT)
    slot = plan.run_slots[32].model_copy(update={"configuration_sha256": "0" * 64})
    with pytest.raises(runner.shared.HoldoutRunnerError) as caught:
        runner.shared._expected_manifest(ROOT, controls, slot)
    assert caught.value.code is runner.shared.HoldoutRunnerErrorCode.SOURCE_IDENTITY_MISMATCH


@pytest.mark.parametrize("invalid_budget", [False, True])
def test_baseline_manifest_validates_cli_module_plan_without_class_identity_assumption(
    plan, monkeypatch, invalid_budget
):
    # `python -m` defines the controller's models in __main__, while the shared
    # manifest builder imports its canonical module. An alias reproduces that
    # separate class identity without executing main or creating paid authority.
    name = "scripts._v6_cli_identity_fixture"
    spec = importlib.util.spec_from_file_location(name, runner.__file__)
    assert spec is not None and spec.loader is not None
    cli_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, cli_module)
    spec.loader.exec_module(cli_module)
    cli_plan = cli_module.PostDebugPlanV6.model_validate_json(plan.model_dump_json())
    assert not isinstance(cli_plan, runner.PostDebugPlanV6)
    if invalid_budget:
        cli_plan = cli_plan.model_copy(update={"application_budget_usd": Decimal(51)})
    controls = SimpleNamespace(
        plan=cli_plan,
        environment=runner.v4.HoldoutEnvironmentManifest.model_validate_json(
            (ROOT / runner.v4.ENVIRONMENT_PATH).read_bytes()
        ),
        agent_inputs=runner.v4.HoldoutAgentInputs.model_validate_json(
            (ROOT / runner.v4.AGENT_INPUTS_PATH).read_bytes()
        ),
    )
    monkeypatch.setattr(runner.shared, "_checkout", lambda *_: ROOT)
    slot = cli_plan.run_slots[32]
    if invalid_budget:
        with pytest.raises(runner.shared.HoldoutRunnerError) as caught:
            runner.shared._expected_manifest(ROOT, controls, slot)
        assert caught.value.code is runner.shared.HoldoutRunnerErrorCode.INVALID_CONFIGURATION
    else:
        manifest = runner.shared._expected_manifest(ROOT, controls, slot)
        assert manifest.prompt_versions == (runner.SINGLE_SHOT_PROMPT_VERSION,)
        assert manifest.configuration_sha256 == plan.baseline_configuration_sha256
        assert manifest.source_commit == plan.frozen_source_commit


def test_executor_factory_reuses_gpu_resources_and_baseline_never_reindexes(
    tmp_path, plan, monkeypatch
):
    from repopilot.evals import single_shot

    controls = SimpleNamespace(
        plan=plan,
        environment=runner.v4.HoldoutEnvironmentManifest.model_validate_json(
            (ROOT / runner.v4.ENVIRONMENT_PATH).read_bytes()
        ),
        agent_inputs=runner.v4.HoldoutAgentInputs.model_validate_json(
            (ROOT / runner.v4.AGENT_INPUTS_PATH).read_bytes()
        ),
    )
    provider, cache, store = object(), object(), object()
    resource_calls, vector_calls, context_calls = [], [], []

    def resources(root):
        resource_calls.append(root)
        return provider, cache

    def vector_store(settings):
        vector_calls.append(settings)
        return store

    context = object()

    def initial_context(path, task, **bindings):
        context_calls.append((path, task, bindings))
        return context

    monkeypatch.setattr(runner.v4, "gpu_resources", resources)
    monkeypatch.setattr(runner.shared, "PostgresVectorStore", vector_store)
    monkeypatch.setattr(
        runner.shared,
        "embedding_provider_from_settings",
        lambda *_: pytest.fail("shared embedding provider must not be rebuilt"),
    )
    monkeypatch.setattr(
        runner.shared,
        "embedding_cache_from_settings",
        lambda *_a, **_kw: pytest.fail("shared cache must not be rebuilt"),
    )
    monkeypatch.setattr(
        single_shot,
        "OpenAISDKTransport",
        lambda **_: pytest.fail("constructing a slot must not call the Provider"),
    )
    monkeypatch.setattr(single_shot, "load_hybrid_initial_context", initial_context)
    factory = runner.executor_factory(
        root=tmp_path,
        controls=controls,
        openai_api_key="diagnostic-never-sent",
        database_url="postgresql://local-only/test",
    )
    task_id = runner.EXPECTED_TASK_IDS[0]
    hybrid = factory("hybrid-rrf-v1", task_id)
    baseline = factory(runner.SINGLE_SHOT_ARM_ID, task_id)
    second_baseline = factory(runner.SINGLE_SHOT_ARM_ID, runner.EXPECTED_TASK_IDS[1])
    assert isinstance(hybrid, runner.RepairWorkflowEvaluationExecutor)
    assert isinstance(baseline, single_shot.SingleShotEvaluationExecutor)
    assert isinstance(second_baseline, single_shot.SingleShotEvaluationExecutor)
    assert resource_calls == [tmp_path]
    assert len(vector_calls) == 1
    assert hybrid.embedding_provider is provider
    assert hybrid.embedding_cache is cache
    assert hybrid.vector_store is store
    assert baseline.settings == hybrid.settings
    assert baseline.sandbox_image_reference == hybrid.sandbox_image_reference
    assert baseline.evaluation_configuration_sha256 == plan.baseline_configuration_sha256
    assert baseline.sandbox_factory is not None
    assert baseline.transport is None
    assert context_calls == []  # Even trace loading is deferred until this slot executes.
    task = controls.agent_inputs.tasks[0]
    assert baseline.initial_context_factory(task) is context
    slot = next(s for s in plan.run_slots if s.arm_id == "hybrid-rrf-v1" and s.task_id == task_id)
    assert context_calls == [
        (
            tmp_path / slot.output_path,
            task,
            {
                "expected_run_id": slot.run_id,
                "expected_source_commit": plan.frozen_source_commit,
                "expected_configuration_sha256": slot.configuration_sha256,
            },
        )
    ]
