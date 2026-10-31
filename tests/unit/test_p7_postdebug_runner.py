"""Offline checks for explicit post-debug labels, preserved history and budget."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from repopilot.sandbox.docker import DockerImageIdentity  # noqa: E402
from scripts import p7_postdebug_runner as runner  # noqa: E402
from scripts.p7_holdout_requalification import _sealed  # noqa: E402


@pytest.fixture
def plan(monkeypatch: pytest.MonkeyPatch) -> runner.PostDebugPlan:
    old = {"src/repopilot/agent/repair.py": "a" * 40}
    current = {**old, "scripts/p7_postdebug_runner.py": "b" * 40}
    monkeypatch.setattr(runner.workload, "verify_source", lambda _: ("c" * 40, current))
    monkeypatch.setattr(runner.v3, "_objects_at", lambda *_: old)
    monkeypatch.setattr(runner, "prior_cost", lambda *_: (Decimal("0.740420000"), Decimal(2)))
    monkeypatch.setattr(runner, "prior_inventory", lambda *_: {"prior.json": "d" * 64})
    return runner.build_plan(ROOT, ROOT)


def test_plan_is_seen_task_rerun_and_includes_all_prior_spend(plan) -> None:
    assert plan.fresh_blind_holdout is False
    assert plan.evaluation_label == "post_debug_seen_task_rerun"
    assert plan.expensive_preflight_required is False
    assert plan.carried_cost_usd == Decimal("2.740420000")
    assert plan.new_execution_budget_usd == Decimal("29.259580000")
    assert len(plan.run_slots) == 32
    assert len({slot.run_id for slot in plan.run_slots}) == 32
    assert all("/p7-postdebug-v4-" in s.output_path for s in plan.run_slots)


@pytest.mark.parametrize(
    "change",
    [
        {"fresh_blind_holdout": True},
        {"carried_cost_usd": Decimal(0)},
        {"application_budget_usd": Decimal(33)},
        {"per_task_cost_limit_usd": Decimal(2)},
        {"prior_unknown_cost_usd": Decimal(0)},
        {"new_execution_budget_usd": Decimal(32)},
        {"prior_known_cost_usd": Decimal("NaN")},
    ],
)
def test_resealed_misleading_labels_or_budget_rejected(plan, change) -> None:
    with pytest.raises((ValidationError, ValueError)):
        _sealed(
            runner.PostDebugPlan,
            {
                **plan.model_dump(exclude={"content_sha256"}),
                "run_slots": plan.run_slots,
                **change,
            },
        )


def test_resealed_slot_redirect_rejected(plan) -> None:
    slots = list(plan.run_slots)
    slots[0] = slots[0].model_copy(update={"output_path": "results/p7-holdout/raw/old"})
    with pytest.raises(ValidationError, match="slot_identity_mismatch"):
        _sealed(
            runner.PostDebugPlan,
            {
                **plan.model_dump(exclude={"content_sha256"}),
                "run_slots": tuple(slots),
            },
        )


def test_no_requalification_for_changed_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(
        runner.workload,
        "verify_source",
        lambda _: (
            "a" * 40,
            {"src/repopilot/retrieval/dense.py": "b" * 40},
        ),
    )
    monkeypatch.setattr(
        runner.v3,
        "_objects_at",
        lambda *_: {
            "src/repopilot/retrieval/dense.py": "c" * 40,
        },
    )
    with pytest.raises(ValueError, match="unreviewed_repair_scope"):
        runner.build_plan(ROOT, ROOT)


def test_source_and_prior_history_cannot_change(monkeypatch, plan) -> None:
    monkeypatch.setattr(runner, "_git", lambda *_: "")
    assert runner.validate_source_and_inputs(ROOT, plan) == "c" * 40
    monkeypatch.setattr(runner, "prior_inventory", lambda *_: {"prior.json": "e" * 64})
    with pytest.raises(ValueError, match="prior_results_changed"):
        runner.validate_source_and_inputs(ROOT, plan)
    monkeypatch.setattr(runner.workload, "verify_source", lambda _: ("c" * 40, {}))
    with pytest.raises(ValueError, match="frozen_execution_source_changed"):
        runner.validate_source_and_inputs(ROOT, plan)


def test_authorization_only_written_for_explicit_execute(tmp_path, plan) -> None:
    path = tmp_path / runner.PLAN_PATH
    path.parent.mkdir(parents=True)
    path.write_text(plan.model_dump_json())
    draft = runner.authorization(tmp_path, plan, execute=False)
    assert not (tmp_path / runner.AUTH_PATH).exists()
    auth = runner.authorization(tmp_path, plan, execute=True)
    assert auth.plan_sha256 == draft.plan_sha256 == plan.content_sha256
    assert auth.carried_cost_usd == Decimal("2.740420000")
    assert auth.expensive_preflight_waived and auth.post_debug_label_acknowledged
    raw = (tmp_path / runner.AUTH_PATH).read_bytes()
    assert runner.authorization(tmp_path, plan, execute=True) == auth
    assert (tmp_path / runner.AUTH_PATH).read_bytes() == raw
    changed = plan.model_copy(update={"content_sha256": "f" * 64})
    with pytest.raises(ValueError, match="authorization_mismatch"):
        runner.authorization(tmp_path, changed, execute=True)


@pytest.mark.parametrize("missing", ["--skip-preflight", "--acknowledge-post-debug", "digest"])
def test_execute_requires_exact_plan_and_explicit_waiver(
    monkeypatch, tmp_path, plan, missing
) -> None:
    path = tmp_path / runner.PLAN_PATH
    path.parent.mkdir(parents=True)
    path.write_text(plan.model_dump_json())
    args = [
        "runner",
        "--repository-root",
        str(tmp_path),
        "--execute",
        "--confirm-plan-sha256",
        plan.content_sha256,
        "--skip-preflight",
        "--acknowledge-post-debug",
    ]
    if missing == "digest":
        args[args.index(plan.content_sha256)] = "0" * 64
    else:
        args.remove(missing)
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner, "load_controls", lambda *_a, **_kw: pytest.fail("no execution"))
    with pytest.raises(ValueError):
        runner.main()
    assert not (tmp_path / runner.AUTH_PATH).exists()


def test_explicit_run_reuses_runner_without_full_preflight(monkeypatch, tmp_path, plan) -> None:
    path = tmp_path / runner.PLAN_PATH
    path.parent.mkdir(parents=True)
    path.write_text(plan.model_dump_json())
    (tmp_path / runner.v3.PLAN_PATH).write_bytes((ROOT / runner.v3.PLAN_PATH).read_bytes())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--repository-root",
            str(tmp_path),
            "--execute",
            "--confirm-plan-sha256",
            plan.content_sha256,
            "--skip-preflight",
            "--acknowledge-post-debug",
        ],
    )
    monkeypatch.setattr(runner, "load_controls", lambda *_a, **_kw: SimpleNamespace(plan=plan))
    calls = []

    @contextmanager
    def lock(_root, identity):
        calls.append(identity)
        yield

    monkeypatch.setattr(runner, "_execution_lock", lock)
    monkeypatch.setattr(
        runner.profile, "verify_qualification", lambda *_: pytest.fail("no preflight")
    )
    monkeypatch.setattr(
        runner.cache_api, "verify_pinned_cache", lambda *_: pytest.fail("no full scan")
    )

    def run(**kwargs):
        assert kwargs["carried_cost_usd"] == Decimal("2.740420000")
        assert kwargs["max_new_slots"] == 32
        assert kwargs["continue_after_technical_failure"] is False
        return 0

    monkeypatch.setattr(runner.shared, "_run_loaded_controls", run)
    assert runner.main() == 0
    assert calls[0] == "p7-paid-evaluation-global"
    assert calls[1].startswith("p7-holdout-v3-")


def test_prior_inventory_rejects_symlinks(tmp_path) -> None:
    old = runner.v3.GpuPlan.model_validate_json((ROOT / runner.v3.PLAN_PATH).read_bytes())
    base = tmp_path / "results/p7-holdout/raw" / old.plan_id
    base.mkdir(parents=True)
    (base / "link").symlink_to(ROOT / "pyproject.toml")
    with pytest.raises(ValueError, match="unsafe_prior_artifact"):
        runner.prior_inventory(tmp_path, old)


def test_completed_report_honestly_labels_and_carries_budget(tmp_path, plan) -> None:
    from tests.unit.test_p7_holdout_runner_v3 import _outcomes

    controls = SimpleNamespace(
        plan=plan,
        authorization=SimpleNamespace(content_sha256="a" * 64),
        controller_commit="b" * 40,
    )
    outcomes = _outcomes(controls)
    progress = runner.shared._Progress(
        outcomes, (), (), Decimal("3.20"), Decimal(0), plan.carried_cost_usd
    )
    report = runner.write_report(tmp_path, controls, progress)
    assert report["fresh_blind_holdout"] is False
    assert report["total_accounted_cost_usd"] == "5.940420000"
    assert report["slot_count"] == 32
    assert runner.write_report(tmp_path, controls, progress) == report
    assert json.loads((tmp_path / runner.REPORT_PATH).read_text()) == report


def _runtime_fixture(monkeypatch, root: Path, plan):
    """Use real sealed inputs and v3 readiness; replace only Git/Docker host I/O."""
    v3 = runner.v3
    for name in (*plan.input_files, "uv.lock"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (root / runner.PLAN_PATH).write_text(plan.model_dump_json())
    freeze_raw = (ROOT / v3.FREEZE_PATH).read_bytes()
    old_raw = (root / v3.PLAN_PATH).read_bytes()
    auth_raw = (root / v3.AUTHORIZATION_PATH).read_bytes()
    environment_raw = (root / runner.ENVIRONMENT_PATH).read_bytes()
    agent_raw = (root / runner.AGENT_INPUTS_PATH).read_bytes()
    freeze = v3.GpuFreeze.model_validate_json(freeze_raw)
    old = v3.GpuPlan.model_validate_json(old_raw)
    auth = v3.GpuAuthorization.model_validate_json(auth_raw)
    controls = v3.GpuControls(
        freeze=freeze,
        plan=old,
        authorization=auth,
        environment=runner.HoldoutEnvironmentManifest.model_validate_json(environment_raw),
        agent_inputs=runner.HoldoutAgentInputs.model_validate_json(agent_raw),
        controller_commit=old.controller_commit,
        freeze_raw=freeze_raw,
        plan_raw=old_raw,
        authorization_raw=auth_raw,
        environment_raw=environment_raw,
        agent_inputs_raw=agent_raw,
    )
    legacy = v3.legacy.HoldoutRuntimeReadiness.model_validate_json(
        (ROOT / v3.legacy.READINESS_PATH).read_bytes()
    )
    ready = v3.build_readiness(
        controls,
        checked_at=auth.approved_at + timedelta(seconds=1),
        execution_controller_objects=freeze.execution_source_objects,
        embedding=legacy.embedding,
        vector_backend=legacy.vector_backend,
        images=legacy.images,
        cpu_profile=freeze.cpu_runtime,
        gpu_cache=freeze.cache_evidence,
    )
    (root / v3.READINESS_PATH).write_text(ready.model_dump_json())
    monkeypatch.setattr(runner, "_git", lambda *_: "")
    monkeypatch.setattr(v3, "verify_runtime_environment", lambda: None)
    # Heavy probes, providers and embeddings must never be used just to load controls.
    monkeypatch.setattr(
        runner.profile, "verify_qualification", lambda *_: pytest.fail("heavy preflight")
    )
    monkeypatch.setattr(runner, "gpu_resources", lambda *_: pytest.fail("embedding loaded"))
    monkeypatch.setattr(
        runner.shared, "_real_executor_factory", lambda **_: pytest.fail("executor constructed")
    )
    checkouts = {}
    for binding in controls.environment.bindings:
        checkout = root / binding.checkout_relative_path
        checkout.mkdir(parents=True)
        checkouts[checkout] = binding.base_commit

    def git_checkout(checkout, *args):
        assert checkout in checkouts
        if args == ("rev-parse", "HEAD"):
            return checkouts[checkout]
        assert args == ("status", "--porcelain=v1")
        return ""

    monkeypatch.setattr(runner.shared, "_git", git_checkout)
    image_ids = {image.registry_digest_reference: image.local_image_id for image in ready.images}
    image_calls = []

    class OfflineDocker:
        def resolve_image(self, reference):
            image_calls.append(reference)
            return DockerImageIdentity(
                image_id=image_ids[reference], repository_digests=(reference,)
            )

    monkeypatch.setattr(runner.shared, "DockerCliClient", OfflineDocker)
    return ready, image_calls, image_ids


@pytest.mark.parametrize("execute", [False, True])
def test_load_controls_uses_real_v3_readiness_inputs_and_checks_all_assets(
    monkeypatch, tmp_path, plan, execute
) -> None:
    ready, image_calls, _ = _runtime_fixture(monkeypatch, tmp_path, plan)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder-never-sent")

    controls = runner.load_controls(tmp_path, execute=execute)

    assert isinstance(controls, runner.shared._Controls)
    assert isinstance(controls.plan, runner.PostDebugPlan)
    assert isinstance(controls.authorization, runner.PostDebugAuthorization)
    assert isinstance(controls.readiness, runner.v3.GpuReadiness)
    assert isinstance(controls.agent_inputs, runner.HoldoutAgentInputs)
    assert isinstance(controls.environment, runner.HoldoutEnvironmentManifest)
    assert controls.plan == plan
    assert controls.readiness == ready
    assert controls.readiness.ready_for_paid_holdout is True
    assert controls.controller_commit == "c" * 40
    assert len(controls.agent_inputs.tasks) == 8
    assert image_calls == [item.registry_digest_reference for item in ready.images]
    assert len(image_calls) == 8
    assert (tmp_path / runner.AUTH_PATH).exists() is execute
    assert not (tmp_path / "results/p7-holdout/raw" / plan.plan_id).exists()


@pytest.mark.parametrize(
    "binding",
    [
        "plan_content_sha256",
        "plan_raw_sha256",
        "authorization_raw_sha256",
        "environment_raw_sha256",
    ],
)
def test_load_controls_rejects_resealed_stale_historical_asset_bindings(
    monkeypatch, tmp_path, plan, binding
) -> None:
    ready, image_calls, _ = _runtime_fixture(monkeypatch, tmp_path, plan)
    changed = _sealed(
        runner.v3.GpuReadiness,
        {
            **{
                name: getattr(ready, name)
                for name in runner.v3.GpuReadiness.model_fields
                if name != "content_sha256"
            },
            binding: "0" * 64,
        },
    )
    (tmp_path / runner.v3.READINESS_PATH).write_text(changed.model_dump_json())
    with pytest.raises(ValueError, match="historical_asset_bindings_changed"):
        runner.load_controls(tmp_path, execute=False)
    assert not image_calls
    assert not (tmp_path / runner.AUTH_PATH).exists()


def test_load_controls_rejects_regression_input_drift_before_asset_checks(
    monkeypatch, tmp_path, plan
) -> None:
    _, image_calls, _ = _runtime_fixture(monkeypatch, tmp_path, plan)
    target = tmp_path / runner.TEST_INPUTS_PATH
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen_input_changed"):
        runner.load_controls(tmp_path, execute=False)
    assert not image_calls


def test_load_controls_real_asset_verifier_rejects_changed_local_image(
    monkeypatch, tmp_path, plan
) -> None:
    ready, image_calls, image_ids = _runtime_fixture(monkeypatch, tmp_path, plan)
    image_ids[ready.images[0].registry_digest_reference] = "sha256:" + "0" * 64
    with pytest.raises(runner.shared.HoldoutRunnerError, match="source_identity_mismatch"):
        runner.load_controls(tmp_path, execute=False)
    assert image_calls == [ready.images[0].registry_digest_reference]


def test_load_controls_execution_needs_credential_before_authorization(
    monkeypatch, tmp_path, plan
) -> None:
    _, image_calls, _ = _runtime_fixture(monkeypatch, tmp_path, plan)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="provider_credential_required"):
        runner.load_controls(tmp_path, execute=True)
    assert not image_calls
    assert not (tmp_path / runner.AUTH_PATH).exists()
