"""Forty seen-task runs: bounded repair plus a matched-context single-shot control.

The USD 50 cumulative authorization includes all prior spend and unknown reserves.
No expensive corpus preflight is rerun; source, inputs, history and cache-on-read
integrity remain mandatory. A single-shot result is not a fresh blind holdout.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Self
from uuid import UUID, uuid5

from pydantic import model_validator

from repopilot.evals import load_experiment_matrix, select_experiment_arm
from repopilot.evals.runner import EvaluationTaskExecutor, RepairWorkflowEvaluationExecutor
from repopilot.providers.prompts import PROMPT_VERSION
from repopilot.schemas import Sha256
from scripts import p7_holdout_gpu as v3
from scripts import p7_holdout_runner as shared
from scripts import p7_postdebug_runner as v4
from scripts import p7_postdebug_runner_v5 as v5
from scripts import p7_repository_workload_preflight as workload
from scripts.p7_holdout_control import EXPECTED_ARM_IDS, EXPECTED_TASK_IDS, MATRIX_PATH
from scripts.p7_holdout_requalification import _git, _read, _sealed
from scripts.p7_holdout_runner_v2 import _execution_lock

require, sha = v4.require, v4.sha
PLAN_PATH = Path("results/p7-holdout/postdebug-plan-v6.json")
AUTH_PATH = Path("results/p7-holdout/postdebug-authorization-v6.json")
REPORT_PATH = Path("results/p7-holdout/postdebug-report-v6.json")
TOTAL_BUDGET = Decimal("50.00")
SINGLE_SHOT_ARM_ID = "single-shot-hybrid-context-v1"
SINGLE_SHOT_PROMPT_VERSION = "repopilot-single-shot-v1"
ARM_IDS = (*EXPECTED_ARM_IDS, SINGLE_SHOT_ARM_ID)
NAMESPACE = UUID("f3a88d67-7d36-42d8-8842-a33ed42baa11")
REPAIR_PATHS = frozenset(
    {
        "src/repopilot/agent/repair.py",
        "src/repopilot/tools/patch.py",
        "src/repopilot/tools/patch_workspace.py",
        "src/repopilot/tools/tests.py",
        "src/repopilot/repair_workflow.py",
        "src/repopilot/evals/isolated_tests.py",
        "src/repopilot/evals/single_shot.py",
        "scripts/p7_holdout_runner.py",
        "scripts/p7_postdebug_runner_v6.py",
    }
)


def baseline_configuration(root: Path) -> dict[str, object]:
    matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    hybrid = select_experiment_arm(matrix, "hybrid-rrf-v1")
    return {
        "baseline_version": SINGLE_SHOT_ARM_ID,
        "hybrid_configuration": hybrid.configuration_snapshot,
        "initial_context_source": "same_round_hybrid_initial_trace",
        "prompt_version": SINGLE_SHOT_PROMPT_VERSION,
        "provider_calls": 1,
        "agent_tools": False,
        "test_feedback_to_model": False,
        "transport_retries": 0,
        "scoring_protocol": "isolated-regression-tests-v1",
    }


class PostDebugPlanV6(v5.PostDebugPlanV5):
    plan_version: Literal["post-debug-rerun-v6"] = "post-debug-rerun-v6"  # type: ignore[assignment]
    experiment_id: Literal["repopilot-p7-postdebug-v6"] = "repopilot-p7-postdebug-v6"  # type: ignore[assignment]
    application_budget_usd: Decimal = TOTAL_BUDGET
    baseline_configuration_sha256: Sha256
    baseline_prompt_version: Literal["repopilot-single-shot-v1"] = SINGLE_SHOT_PROMPT_VERSION  # type: ignore[assignment]
    comparison_basis: Literal["same_initial_context_different_interaction_and_compute"] = (
        "same_initial_context_different_interaction_and_compute"
    )

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        require(self.application_budget_usd == TOTAL_BUDGET, "budget_changed")
        require(self.per_task_cost_limit_usd == Decimal(1), "task_budget_changed")
        require(
            all(
                x.is_finite() and x >= 0
                for x in (
                    self.prior_known_cost_usd,
                    self.prior_unknown_cost_usd,
                    self.carried_cost_usd,
                    self.new_execution_budget_usd,
                )
            )
            and self.prior_unknown_cost_usd >= v4.V2_UNKNOWN_RESERVE
            and self.carried_cost_usd == self.prior_known_cost_usd + self.prior_unknown_cost_usd
            and self.new_execution_budget_usd == TOTAL_BUDGET - self.carried_cost_usd,
            "invalid_carried_budget",
        )
        require(
            self.execution_source_sha256 == shared._canonical_sha256(self.execution_source_objects)
            and self.plan_id == f"p7-postdebug-v6-{self.execution_source_sha256[:16]}",
            "source_seal_mismatch",
        )
        require(
            self.prompt_versions == (v5.EXPECTED_PROMPT_VERSION,)
            and self.matrix_prompt_versions == ("repopilot-agent-v1",),
            "prompt_override_mismatch",
        )
        require(
            tuple((s.arm_id, s.task_id) for s in self.run_slots)
            == tuple((a, t) for a in ARM_IDS for t in EXPECTED_TASK_IDS),
            "slot_matrix_changed",
        )
        for slot in self.run_slots:
            require(
                slot.run_id == uuid5(NAMESPACE, f"{self.plan_id}:{slot.arm_id}:{slot.task_id}")
                and slot.output_path
                == f"results/p7-holdout/raw/{self.plan_id}/{slot.arm_id}/{slot.task_id}",
                "slot_identity_mismatch",
            )
            if slot.arm_id == SINGLE_SHOT_ARM_ID:
                require(
                    slot.configuration_sha256 == self.baseline_configuration_sha256,
                    "baseline_configuration_mismatch",
                )
        shared._validate_content_digest(self)
        return self


class PostDebugAuthorizationV6(v5.PostDebugAuthorizationV5):
    authorization_version: Literal["post-debug-authorization-v6"] = "post-debug-authorization-v6"  # type: ignore[assignment]
    max_total_cost_usd: Decimal = TOTAL_BUDGET

    @model_validator(mode="after")
    def validate_authorization(self) -> Self:
        require(self.max_total_cost_usd == TOTAL_BUDGET, "budget_changed")
        shared._validate_content_digest(self)
        return self


def prior_cost(root: Path, old: v5.PostDebugPlanV5) -> tuple[Decimal, Decimal]:
    """Audit complete v5 evidence; do not trust a user-supplied cost total."""
    auth = v5.PostDebugAuthorizationV5.model_validate_json(_read(root, v5.AUTH_PATH))
    require(
        auth.plan_sha256 == old.content_sha256
        and auth.plan_raw_sha256 == sha(_read(root, v5.PLAN_PATH))
        and auth.frozen_source_commit == old.frozen_source_commit
        and auth.carried_cost_usd == old.carried_cost_usd,
        "prior_authorization_mismatch",
    )
    report = json.loads(_read(root, v5.REPORT_PATH))
    require(isinstance(report, dict), "prior_report_invalid")
    digest = report.pop("content_sha256", None)
    require(digest == shared._canonical_sha256(report), "prior_report_seal_mismatch")
    outcomes = v5._prior_outcomes(root, old)
    known = sum((o.usage.estimated_cost_usd for o in outcomes if o.usage), Decimal(0))
    unknown = sum((o.reserved_unknown_cost_usd for o in outcomes), Decimal(0))
    technical = sum(o.attempt_status == "technical_failure" for o in outcomes)
    require(
        report.get("report_version") == "post-debug-rerun-report-v5"
        and report.get("plan_sha256") == old.content_sha256
        and report.get("authorization_sha256") == auth.content_sha256
        and report.get("source_commit") == old.frozen_source_commit
        and report.get("evaluation_label") == "post_debug_seen_task_rerun"
        and report.get("fresh_blind_holdout") is False
        and report.get("slot_count") == len(outcomes) == 32
        and report.get("completed_slots") == 32 - technical
        and report.get("technical_failure_slots") == technical
        and report.get("task_success_count") == sum(o.task_success is True for o in outcomes)
        and report.get("outcomes") == [o.model_dump(mode="json") for o in outcomes]
        and Decimal(report["known_cost_usd"]) == known
        and Decimal(report["reserved_unknown_cost_usd"]) == unknown
        and Decimal(report["carried_cost_usd"]) == old.carried_cost_usd
        and Decimal(report["total_accounted_cost_usd"]) == old.carried_cost_usd + known + unknown
        and Decimal(report["budget_usd"]) == v4.TOTAL_BUDGET,
        "prior_report_raw_mismatch",
    )
    return old.prior_known_cost_usd + known, old.prior_unknown_cost_usd + unknown


def build_plan(root: Path, evidence_root: Path) -> PostDebugPlanV6:
    require(PROMPT_VERSION == v5.EXPECTED_PROMPT_VERSION, "runtime_prompt_version_mismatch")
    commit, objects = workload.verify_source(root)
    old_raw = _read(root, v5.PLAN_PATH)
    old = v5.PostDebugPlanV5.model_validate_json(old_raw)
    require(_read(evidence_root, v5.PLAN_PATH) == old_raw, "prior_plan_mismatch")
    previous = v3._objects_at(root, old.frozen_source_commit)
    require(not (set(previous) - set(objects)), "previous_source_removed")
    require(
        {p for p in objects if objects[p] != previous.get(p)} <= REPAIR_PATHS,
        "unreviewed_repair_scope",
    )
    known, unknown = prior_cost(evidence_root, old)
    input_files = dict(old.input_files)
    for name, expected in input_files.items():
        require(sha(_read(root, Path(name))) == expected, "frozen_input_changed")
    for path in (v5.PLAN_PATH, v5.AUTH_PATH, v5.REPORT_PATH):
        input_files[path.as_posix()] = sha(_read(evidence_root, path))
    source_sha = shared._canonical_sha256(objects)
    plan_id = f"p7-postdebug-v6-{source_sha[:16]}"
    baseline_sha = shared._canonical_sha256(baseline_configuration(root))
    templates = (
        *old.run_slots,
        *(
            s.model_copy(
                update={
                    "arm_id": SINGLE_SHOT_ARM_ID,
                    "configuration_sha256": baseline_sha,
                }
            )
            for s in old.run_slots
            if s.arm_id == "hybrid-rrf-v1"
        ),
    )
    slots = tuple(
        s.model_copy(
            update={
                "run_id": uuid5(NAMESPACE, f"{plan_id}:{s.arm_id}:{s.task_id}"),
                "output_path": f"results/p7-holdout/raw/{plan_id}/{s.arm_id}/{s.task_id}",
            }
        )
        for s in templates
    )
    return _sealed(
        PostDebugPlanV6,
        {
            "plan_id": plan_id,
            "created_at": datetime.now(UTC),
            "frozen_source_commit": commit,
            "execution_source_objects": objects,
            "execution_source_sha256": source_sha,
            "dependency_lock_sha256": old.dependency_lock_sha256,
            "dataset_manifest_sha256": old.dataset_manifest_sha256,
            "matrix_sha256": old.matrix_sha256,
            "prompt_versions": old.prompt_versions,
            "matrix_prompt_versions": old.matrix_prompt_versions,
            "run_slots": slots,
            "baseline_configuration_sha256": baseline_sha,
            "previous_plan_raw_sha256": sha(old_raw),
            "input_files": input_files,
            "prior_raw_files": {**old.prior_raw_files, **v4.prior_inventory(evidence_root, old)},
            "prior_known_cost_usd": known,
            "prior_unknown_cost_usd": unknown,
            "carried_cost_usd": known + unknown,
            "new_execution_budget_usd": TOTAL_BUDGET - known - unknown,
        },
    )


def validate_source_and_inputs(root: Path, plan: PostDebugPlanV6) -> str:
    require(PROMPT_VERSION == v5.EXPECTED_PROMPT_VERSION, "runtime_prompt_version_mismatch")
    commit, objects = workload.verify_source(root)
    require(objects == plan.execution_source_objects, "frozen_execution_source_changed")
    _git(root, "merge-base", "--is-ancestor", plan.frozen_source_commit, commit)
    require(sha(_read(root, Path("uv.lock"))) == plan.dependency_lock_sha256, "lock_changed")
    for name, expected in plan.input_files.items():
        require(sha(_read(root, Path(name))) == expected, "frozen_input_changed")
    old_raw = _read(root, v5.PLAN_PATH)
    require(sha(old_raw) == plan.previous_plan_raw_sha256, "prior_plan_changed")
    old = v5.PostDebugPlanV5.model_validate_json(old_raw)
    upstream = v3.GpuPlan.model_validate_json(_read(root, v3.PLAN_PATH))
    fourth = v4.PostDebugPlan.model_validate_json(_read(root, v4.PLAN_PATH))
    inventory = v4.prior_inventory(root, upstream)
    inventory.update(v4.prior_inventory(root, fourth))
    require(inventory == old.prior_raw_files, "upstream_results_changed")
    inventory.update(v4.prior_inventory(root, old))
    require(inventory == plan.prior_raw_files, "prior_results_changed")
    require(
        prior_cost(root, old) == (plan.prior_known_cost_usd, plan.prior_unknown_cost_usd),
        "prior_cost_changed",
    )
    require(
        shared._canonical_sha256(baseline_configuration(root))
        == plan.baseline_configuration_sha256,
        "baseline_configuration_mismatch",
    )
    return commit


def authorization(root: Path, plan: PostDebugPlanV6, *, execute: bool) -> PostDebugAuthorizationV6:
    expected = {
        "plan_sha256": plan.content_sha256,
        "plan_raw_sha256": sha(_read(root, PLAN_PATH)),
        "frozen_source_commit": plan.frozen_source_commit,
        "carried_cost_usd": plan.carried_cost_usd,
    }
    target = root / AUTH_PATH
    if target.exists() or target.is_symlink():
        auth = PostDebugAuthorizationV6.model_validate_json(_read(root, AUTH_PATH))
        require(
            all(getattr(auth, k) == value for k, value in expected.items()),
            "authorization_mismatch",
        )
        return auth
    auth = _sealed(PostDebugAuthorizationV6, {"approved_at": datetime.now(UTC), **expected})
    if execute:
        shared._prepare_output_parent(root, AUTH_PATH.as_posix())
        shared._write_private_create_only(target, auth.model_dump_json(indent=2).encode() + b"\n")
    return auth


def load_controls(root: Path, *, execute: bool) -> shared._Controls:
    v3.verify_runtime_environment()
    plan = PostDebugPlanV6.model_validate_json(_read(root, PLAN_PATH))
    commit = validate_source_and_inputs(root, plan)
    if execute:
        require(bool(os.environ.get("OPENAI_API_KEY", "").strip()), "provider_credential_required")
    return v4.historical_assets_controls(
        root, plan=plan, authorize=lambda: authorization(root, plan, execute=execute), commit=commit
    )


def executor_factory(
    *, root: Path, controls: shared._Controls, openai_api_key: str, database_url: str
) -> shared.ExecutorFactory:
    from repopilot.evals.single_shot import (
        SingleShotEvaluationExecutor,
        load_hybrid_initial_context,
    )

    original = v4.executor_factory(
        root=root, controls=controls, openai_api_key=openai_api_key, database_url=database_url
    )

    def build(arm_id: str, task_id: str) -> EvaluationTaskExecutor:
        if arm_id != SINGLE_SHOT_ARM_ID:
            return original(arm_id, task_id)
        hybrid = original("hybrid-rrf-v1", task_id)
        require(isinstance(hybrid, RepairWorkflowEvaluationExecutor), "hybrid_executor_mismatch")
        assert isinstance(hybrid, RepairWorkflowEvaluationExecutor)
        require(hybrid.sandbox_factory is not None, "isolated_sandbox_required")
        assert hybrid.sandbox_factory is not None
        slot = next(
            s
            for s in controls.plan.run_slots
            if s.arm_id == "hybrid-rrf-v1" and s.task_id == task_id
        )
        return SingleShotEvaluationExecutor(
            settings=hybrid.settings,
            sandbox_image_reference=hybrid.sandbox_image_reference,
            evaluation_configuration_sha256=shared._canonical_sha256(baseline_configuration(root)),
            sandbox_factory=hybrid.sandbox_factory,
            initial_context_factory=lambda task: load_hybrid_initial_context(
                root / slot.output_path,
                task,
                expected_run_id=slot.run_id,
                expected_source_commit=controls.plan.frozen_source_commit,
                expected_configuration_sha256=slot.configuration_sha256,
            ),
        )

    return build


def write_report(
    root: Path, controls: shared._Controls, progress: shared._Progress
) -> dict[str, Any]:
    require(
        not progress.pending_slots
        and not progress.interrupted_slots
        and len(progress.outcomes) == 40,
        "matrix_incomplete",
    )
    payload: dict[str, Any] = {
        "report_version": "post-debug-rerun-report-v6",
        "evaluation_label": "post_debug_seen_task_rerun",
        "fresh_blind_holdout": False,
        "expensive_preflight_waived": True,
        "comparison_basis": "same_initial_context_different_interaction_and_compute",
        "plan_sha256": controls.plan.content_sha256,
        "authorization_sha256": controls.authorization.content_sha256,
        "source_commit": controls.plan.frozen_source_commit,
        "controller_commit": controls.controller_commit,
        "slot_count": 40,
        "completed_slots": len(progress.outcomes) - progress.technical_failures,
        "technical_failure_slots": progress.technical_failures,
        "task_success_count": sum(o.task_success is True for o in progress.outcomes),
        "known_cost_usd": str(progress.known_cost_usd),
        "reserved_unknown_cost_usd": str(progress.reserved_unknown_cost_usd),
        "carried_cost_usd": str(progress.carried_cost_usd),
        "total_accounted_cost_usd": str(progress.accounted_cost_usd),
        "budget_usd": str(TOTAL_BUDGET),
        "outcomes": [o.model_dump(mode="json") for o in progress.outcomes],
    }
    payload["content_sha256"] = shared._canonical_sha256(payload)
    target = root / REPORT_PATH
    if target.exists() or target.is_symlink():
        require(json.loads(_read(root, REPORT_PATH)) == payload, "report_conflict")
    else:
        shared._prepare_output_parent(root, REPORT_PATH.as_posix())
        shared._write_private_create_only(target, (json.dumps(payload, indent=2) + "\n").encode())
    print(
        f"postdebug_complete=true slots=40 successes={payload['task_success_count']} "
        f"technical_failures={progress.technical_failures} "
        f"total_accounted_cost_usd={progress.accounted_cost_usd}",
        flush=True,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--build-plan-from-evidence", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-plan-sha256")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--acknowledge-post-debug", action="store_true")
    parser.add_argument("--max-new-slots", type=int, default=40, choices=range(1, 41))
    parser.add_argument("--continue-after-technical-failure", action="store_true")
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    if args.build_plan_from_evidence:
        require(not args.execute, "build_and_execute_separate")
        plan = build_plan(root, args.build_plan_from_evidence.resolve(strict=True))
        shared._prepare_output_parent(root, PLAN_PATH.as_posix())
        shared._write_private_create_only(
            root / PLAN_PATH, plan.model_dump_json(indent=2).encode() + b"\n"
        )
        print(
            f"postdebug_v6_plan_created=true plan_sha256={plan.content_sha256} "
            f"carried_cost_usd={plan.carried_cost_usd} "
            f"remaining_budget_usd={plan.new_execution_budget_usd}"
        )
        return 0
    plan = PostDebugPlanV6.model_validate_json(_read(root, PLAN_PATH))
    if args.execute:
        require(
            args.skip_preflight and args.acknowledge_post_debug,
            "explicit_waiver_and_label_required",
        )
        require(args.confirm_plan_sha256 == plan.content_sha256, "plan_confirmation_mismatch")
    with _execution_lock(root, "p7-paid-evaluation-global"):
        upstream = v3.GpuPlan.model_validate_json(_read(root, v3.PLAN_PATH))
        with _execution_lock(root, upstream.plan_id):
            controls = load_controls(root, execute=args.execute)
            print(
                "postdebug_v6_start=true expensive_preflight=skipped "
                "cache_integrity=verify_on_read fresh_blind_holdout=false slots=40",
                flush=True,
            )
            return shared._run_loaded_controls(
                root=root,
                controls=controls,
                execute=args.execute,
                confirm_plan_sha256=args.confirm_plan_sha256,
                max_new_slots=args.max_new_slots,
                continue_after_technical_failure=args.continue_after_technical_failure,
                report_writer=write_report,
                executor_factory_builder=executor_factory,
                carried_cost_usd=plan.carried_cost_usd,
            )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except shared.HoldoutRunnerError as error:
        raise SystemExit(str(error)) from None
    except ValueError as error:
        code = (
            str(error)
            if type(error) is ValueError and str(error).isidentifier()
            else "postdebug_v6_validation_failed"
        )
        raise SystemExit(code) from None
