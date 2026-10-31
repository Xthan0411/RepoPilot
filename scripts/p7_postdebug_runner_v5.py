"""Second seen-task rerun; preserve v4 and reuse its assets without a new preflight.

The four arms receive the same prompt/interface fixes. Historical costs are rebuilt
from v4's complete raw ledger, not accepted from a report total. This is not a fresh
blind holdout, and the original cumulative USD 32 authorization remains the ceiling.
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

from repopilot.evals import EvaluationRunManifest, inspect_evaluation_run
from repopilot.providers.prompts import PROMPT_VERSION
from repopilot.schemas import Identifier
from scripts import p7_holdout_gpu as v3
from scripts import p7_holdout_runner as shared
from scripts import p7_postdebug_runner as v4
from scripts import p7_repository_workload_preflight as workload
from scripts.p7_holdout_control import EXPECTED_ARM_IDS, EXPECTED_TASK_IDS
from scripts.p7_holdout_requalification import _git, _read, _sealed
from scripts.p7_holdout_runner_v2 import _execution_lock

require, sha = v4.require, v4.sha
PLAN_PATH = Path("results/p7-holdout/postdebug-plan-v5.json")
AUTH_PATH = Path("results/p7-holdout/postdebug-authorization-v5.json")
REPORT_PATH = Path("results/p7-holdout/postdebug-report-v5.json")
NAMESPACE = UUID("5721dcbb-b776-4c76-a946-241b4389ff8e")
EXPECTED_PROMPT_VERSION = "repopilot-agent-v2"
REPAIR_PATHS = frozenset(
    {
        "src/repopilot/tools/output_projection.py",
        "src/repopilot/tools/registry.py",
        "src/repopilot/providers/prompts.py",
        "src/repopilot/enums.py",
        "src/repopilot/tools/patch.py",
        "src/repopilot/evals/isolated_tests.py",
        "src/repopilot/evals/experiments.py",
        "src/repopilot/repair_workflow.py",
        "src/repopilot/agent/repair.py",
        "scripts/p7_holdout_runner.py",
        "scripts/p7_postdebug_runner.py",
        "scripts/p7_postdebug_runner_v5.py",
    }
)


class PostDebugPlanV5(v4.PostDebugPlan):
    plan_version: Literal["post-debug-rerun-v5"] = "post-debug-rerun-v5"  # type: ignore[assignment]
    experiment_id: Literal["repopilot-p7-postdebug-v5"] = "repopilot-p7-postdebug-v5"  # type: ignore[assignment]
    matrix_prompt_versions: tuple[Identifier, ...]
    prompt_version_basis: Literal["common_postdebug_prompt_override"] = (
        "common_postdebug_prompt_override"
    )

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        require(self.application_budget_usd == v4.TOTAL_BUDGET, "budget_changed")
        require(self.per_task_cost_limit_usd == Decimal("1.00"), "task_budget_changed")
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
            and self.new_execution_budget_usd == v4.TOTAL_BUDGET - self.carried_cost_usd,
            "invalid_carried_budget",
        )
        require(
            self.execution_source_sha256 == shared._canonical_sha256(self.execution_source_objects)
            and self.plan_id == f"p7-postdebug-v5-{self.execution_source_sha256[:16]}",
            "source_seal_mismatch",
        )
        require(
            self.prompt_versions == (EXPECTED_PROMPT_VERSION,)
            and self.matrix_prompt_versions == ("repopilot-agent-v1",),
            "prompt_override_mismatch",
        )
        require(
            tuple((s.arm_id, s.task_id) for s in self.run_slots)
            == tuple((a, t) for a in EXPECTED_ARM_IDS for t in EXPECTED_TASK_IDS),
            "slot_matrix_changed",
        )
        for slot in self.run_slots:
            require(
                slot.run_id == uuid5(NAMESPACE, f"{self.plan_id}:{slot.arm_id}:{slot.task_id}")
                and slot.output_path
                == f"results/p7-holdout/raw/{self.plan_id}/{slot.arm_id}/{slot.task_id}",
                "slot_identity_mismatch",
            )
        shared._validate_content_digest(self)
        return self


class PostDebugAuthorizationV5(v4.PostDebugAuthorization):
    authorization_version: Literal["post-debug-authorization-v5"] = "post-debug-authorization-v5"  # type: ignore[assignment]


def _prior_outcomes(root: Path, old: v4.PostDebugPlan) -> tuple[shared.HoldoutSlotOutcome, ...]:
    outcomes = []
    for slot in old.run_slots:
        directory = root / slot.output_path
        require(directory.is_dir(), "prior_matrix_incomplete")
        report = inspect_evaluation_run(directory)
        manifest = EvaluationRunManifest.model_validate_json(
            _read(root, Path(slot.output_path) / "manifest.json")
        )
        require(
            report.run_id == slot.run_id
            and len(report.attempts) == 1
            and manifest.run_id == slot.run_id
            and manifest.arm_id == slot.arm_id
            and manifest.task_ids == (slot.task_id,)
            and manifest.source_commit == old.frozen_source_commit
            and manifest.experiment_id == old.experiment_id
            and manifest.configuration_sha256 == slot.configuration_sha256
            and manifest.prompt_versions == old.prompt_versions
            and manifest.dataset_manifest_sha256 == old.dataset_manifest_sha256
            and manifest.dependency_lock_sha256 == old.dependency_lock_sha256,
            "prior_run_mismatch",
        )
        attempt = report.attempts[0]
        if attempt.status == "completed" and attempt.evidence is not None:
            outcome = shared._completed_outcome(
                root, slot, attempt.execution_count, attempt.evidence
            )
        elif attempt.status == "failed" and attempt.failure_code is not None:
            outcome = shared._technical_outcome(
                root, slot, attempt.execution_count, attempt.failure_code
            )
        else:
            raise ValueError("prior_execution_still_running")
        outcomes.append(outcome)
    return tuple(outcomes)


def prior_cost(root: Path, old: v4.PostDebugPlan) -> tuple[Decimal, Decimal]:
    """Rebuild all 32 raw slots and compare the complete previous report."""
    auth = v4.PostDebugAuthorization.model_validate_json(_read(root, v4.AUTH_PATH))
    require(
        auth.plan_sha256 == old.content_sha256
        and auth.plan_raw_sha256 == sha(_read(root, v4.PLAN_PATH))
        and auth.frozen_source_commit == old.frozen_source_commit
        and auth.carried_cost_usd == old.carried_cost_usd,
        "prior_authorization_mismatch",
    )
    report = json.loads(_read(root, v4.REPORT_PATH))
    require(isinstance(report, dict), "prior_report_invalid")
    digest = report.pop("content_sha256", None)
    require(digest == shared._canonical_sha256(report), "prior_report_seal_mismatch")
    outcomes = _prior_outcomes(root, old)
    known = sum((o.usage.estimated_cost_usd for o in outcomes if o.usage), Decimal(0))
    unknown = sum((o.reserved_unknown_cost_usd for o in outcomes), Decimal(0))
    technical = sum(o.attempt_status == "technical_failure" for o in outcomes)
    require(
        report.get("report_version") == "post-debug-rerun-report-v4"
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


def build_plan(root: Path, evidence_root: Path) -> PostDebugPlanV5:
    require(PROMPT_VERSION == EXPECTED_PROMPT_VERSION, "runtime_prompt_version_mismatch")
    commit, objects = workload.verify_source(root)
    old_raw = _read(root, v4.PLAN_PATH)
    old = v4.PostDebugPlan.model_validate_json(old_raw)
    require(_read(evidence_root, v4.PLAN_PATH) == old_raw, "prior_plan_mismatch")
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
    for path in (v4.PLAN_PATH, v4.AUTH_PATH, v4.REPORT_PATH):
        input_files[path.as_posix()] = sha(_read(evidence_root, path))
    source_sha = shared._canonical_sha256(objects)
    plan_id = f"p7-postdebug-v5-{source_sha[:16]}"
    slots = tuple(
        slot.model_copy(
            update={
                "run_id": uuid5(NAMESPACE, f"{plan_id}:{slot.arm_id}:{slot.task_id}"),
                "output_path": f"results/p7-holdout/raw/{plan_id}/{slot.arm_id}/{slot.task_id}",
            }
        )
        for slot in old.run_slots
    )
    return _sealed(
        PostDebugPlanV5,
        {
            "plan_id": plan_id,
            "created_at": datetime.now(UTC),
            "frozen_source_commit": commit,
            "execution_source_objects": objects,
            "execution_source_sha256": source_sha,
            "dependency_lock_sha256": old.dependency_lock_sha256,
            "dataset_manifest_sha256": old.dataset_manifest_sha256,
            "matrix_sha256": old.matrix_sha256,
            "prompt_versions": (EXPECTED_PROMPT_VERSION,),
            "matrix_prompt_versions": old.prompt_versions,
            "run_slots": slots,
            "previous_plan_raw_sha256": sha(old_raw),
            "input_files": input_files,
            "prior_raw_files": {**old.prior_raw_files, **v4.prior_inventory(evidence_root, old)},
            "prior_known_cost_usd": known,
            "prior_unknown_cost_usd": unknown,
            "carried_cost_usd": known + unknown,
            "new_execution_budget_usd": v4.TOTAL_BUDGET - known - unknown,
        },
    )


def validate_source_and_inputs(root: Path, plan: PostDebugPlanV5) -> str:
    require(PROMPT_VERSION == EXPECTED_PROMPT_VERSION, "runtime_prompt_version_mismatch")
    commit, objects = workload.verify_source(root)
    require(objects == plan.execution_source_objects, "frozen_execution_source_changed")
    _git(root, "merge-base", "--is-ancestor", plan.frozen_source_commit, commit)
    require(sha(_read(root, Path("uv.lock"))) == plan.dependency_lock_sha256, "lock_changed")
    for name, expected in plan.input_files.items():
        require(sha(_read(root, Path(name))) == expected, "frozen_input_changed")
    old_raw = _read(root, v4.PLAN_PATH)
    require(sha(old_raw) == plan.previous_plan_raw_sha256, "prior_plan_changed")
    old = v4.PostDebugPlan.model_validate_json(old_raw)
    upstream = v3.GpuPlan.model_validate_json(_read(root, v3.PLAN_PATH))
    inventory = v4.prior_inventory(root, upstream)
    require(inventory == old.prior_raw_files, "upstream_results_changed")
    inventory.update(v4.prior_inventory(root, old))
    require(inventory == plan.prior_raw_files, "prior_results_changed")
    require(
        prior_cost(root, old) == (plan.prior_known_cost_usd, plan.prior_unknown_cost_usd),
        "prior_cost_changed",
    )
    return commit


def authorization(root: Path, plan: PostDebugPlanV5, *, execute: bool) -> PostDebugAuthorizationV5:
    expected = {
        "plan_sha256": plan.content_sha256,
        "plan_raw_sha256": sha(_read(root, PLAN_PATH)),
        "frozen_source_commit": plan.frozen_source_commit,
        "carried_cost_usd": plan.carried_cost_usd,
    }
    target = root / AUTH_PATH
    if target.exists() or target.is_symlink():
        auth = PostDebugAuthorizationV5.model_validate_json(_read(root, AUTH_PATH))
        require(all(getattr(auth, k) == v for k, v in expected.items()), "authorization_mismatch")
        return auth
    auth = _sealed(PostDebugAuthorizationV5, {"approved_at": datetime.now(UTC), **expected})
    if execute:
        shared._prepare_output_parent(root, AUTH_PATH.as_posix())
        shared._write_private_create_only(target, auth.model_dump_json(indent=2).encode() + b"\n")
    return auth


def load_controls(root: Path, *, execute: bool) -> shared._Controls:
    v3.verify_runtime_environment()
    plan = PostDebugPlanV5.model_validate_json(_read(root, PLAN_PATH))
    commit = validate_source_and_inputs(root, plan)
    if execute:
        require(bool(os.environ.get("OPENAI_API_KEY", "").strip()), "provider_credential_required")
    return v4.historical_assets_controls(
        root, plan=plan, authorize=lambda: authorization(root, plan, execute=execute), commit=commit
    )


def write_report(
    root: Path, controls: shared._Controls, progress: shared._Progress
) -> dict[str, Any]:
    return v4.write_report(
        root,
        controls,
        progress,
        report_path=REPORT_PATH,
        report_version="post-debug-rerun-report-v5",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--build-plan-from-evidence", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-plan-sha256")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--acknowledge-post-debug", action="store_true")
    parser.add_argument("--max-new-slots", type=int, default=32, choices=range(1, 33))
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
            f"postdebug_v5_plan_created=true plan_sha256={plan.content_sha256} "
            f"carried_cost_usd={plan.carried_cost_usd} "
            f"remaining_budget_usd={plan.new_execution_budget_usd}"
        )
        return 0
    plan = PostDebugPlanV5.model_validate_json(_read(root, PLAN_PATH))
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
                "postdebug_v5_start=true expensive_preflight=skipped "
                "cache_integrity=verify_on_read fresh_blind_holdout=false",
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
                executor_factory_builder=v4.executor_factory,
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
            else "postdebug_v5_validation_failed"
        )
        raise SystemExit(code) from None
