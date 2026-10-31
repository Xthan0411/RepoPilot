"""Versioned post-debug rerun on the original eight tasks, NOT a new blind holdout.

The operator explicitly waives the expensive preflight. Source/input identity,
append-only history, per-read cache integrity, authorization and total-budget gates
remain enforced. No old results or readiness documents are rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Self, cast
from uuid import UUID, uuid5

from pydantic import AwareDatetime, model_validator

from repopilot.evals import inspect_evaluation_run
from repopilot.schemas import GitSha, Identifier, Sha256, StrictModel
from scripts import p7_gpu_cache_acceptance as cache_api
from scripts import p7_gpu_cache_store as cache_store
from scripts import p7_gpu_execution_profile as profile
from scripts import p7_holdout_gpu as v3
from scripts import p7_holdout_runner as shared
from scripts import p7_repository_workload_preflight as workload
from scripts.p7_gpu_cpu_probe import PINNED_IDENTITY, create_cpu_provider
from scripts.p7_holdout_agent_inputs import AGENT_INPUTS_PATH, HoldoutAgentInputs
from scripts.p7_holdout_control import (
    ENVIRONMENT_PATH,
    EXPECTED_ARM_IDS,
    EXPECTED_TASK_IDS,
    MATRIX_PATH,
    HoldoutEnvironmentManifest,
    HoldoutRunSlot,
)
from scripts.p7_holdout_requalification import _git, _read, _sealed
from scripts.p7_holdout_runner_v2 import _execution_lock
from scripts.p7_holdout_test_inputs import (
    TEST_INPUTS_PATH,
    TEST_INPUTS_RAW_SHA256,
    load_test_inputs,
)

PLAN_PATH = Path("results/p7-holdout/postdebug-plan-v4.json")
AUTH_PATH = Path("results/p7-holdout/postdebug-authorization-v4.json")
REPORT_PATH = Path("results/p7-holdout/postdebug-report-v4.json")
NAMESPACE = UUID("526fe66a-8620-4a51-a642-4260dbe43734")
TOTAL_BUDGET = Decimal("32.00")
V2_UNKNOWN_RESERVE = Decimal("1.00")
# Only reviewed repair/diagnostic code may change while reusing corpus qualification.
REPAIR_PATHS = frozenset(
    {
        "src/repopilot/tools/patch.py",
        "src/repopilot/sandbox/patching.py",
        "src/repopilot/agent/repair.py",
        "src/repopilot/agent/graph.py",
        "src/repopilot/repair_workflow.py",
        "src/repopilot/trace.py",
        "scripts/p7_holdout_runner.py",
        "scripts/p7_postdebug_runner.py",
    }
)


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class PostDebugPlan(StrictModel):
    plan_version: Literal["post-debug-rerun-v4"] = "post-debug-rerun-v4"
    evaluation_label: Literal["post_debug_seen_task_rerun"] = "post_debug_seen_task_rerun"
    fresh_blind_holdout: Literal[False] = False
    expensive_preflight_required: Literal[False] = False
    plan_id: Identifier
    experiment_id: Literal["repopilot-p7-postdebug-v4"] = "repopilot-p7-postdebug-v4"
    created_at: AwareDatetime
    frozen_source_commit: GitSha
    execution_source_objects: dict[str, GitSha]
    execution_source_sha256: Sha256
    dependency_lock_sha256: Sha256
    dataset_manifest_sha256: Sha256
    matrix_sha256: Sha256
    prompt_versions: tuple[Identifier, ...]
    run_slots: tuple[HoldoutRunSlot, ...]
    previous_plan_raw_sha256: Sha256
    input_files: dict[str, Sha256]
    prior_raw_files: dict[str, Sha256]
    prior_known_cost_usd: Decimal
    prior_unknown_cost_usd: Decimal
    carried_cost_usd: Decimal
    application_budget_usd: Decimal = TOTAL_BUDGET
    per_task_cost_limit_usd: Decimal = Decimal("1.00")
    new_execution_budget_usd: Decimal
    final_in_flight_response_may_overshoot: Literal[True] = True
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        require(self.application_budget_usd == TOTAL_BUDGET, "budget_changed")
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
            and self.prior_unknown_cost_usd >= V2_UNKNOWN_RESERVE
            and self.carried_cost_usd == self.prior_known_cost_usd + self.prior_unknown_cost_usd
            and self.new_execution_budget_usd == TOTAL_BUDGET - self.carried_cost_usd,
            "invalid_carried_budget",
        )
        require(
            self.execution_source_sha256 == shared._canonical_sha256(self.execution_source_objects),
            "source_seal_mismatch",
        )
        require(
            self.plan_id == f"p7-postdebug-v4-{self.execution_source_sha256[:16]}",
            "plan_id_mismatch",
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


class PostDebugAuthorization(StrictModel):
    authorization_version: Literal["post-debug-authorization-v4"] = "post-debug-authorization-v4"
    approved_at: AwareDatetime
    consent_basis: Literal["operator_execute_exact_plan_cli"] = "operator_execute_exact_plan_cli"
    plan_sha256: Sha256
    plan_raw_sha256: Sha256
    frozen_source_commit: GitSha
    max_total_cost_usd: Decimal = TOTAL_BUDGET
    carried_cost_usd: Decimal
    expensive_preflight_waived: Literal[True] = True
    post_debug_label_acknowledged: Literal[True] = True
    final_in_flight_response_overshoot_acknowledged: Literal[True] = True
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_authorization(self) -> Self:
        require(self.max_total_cost_usd == TOTAL_BUDGET, "budget_changed")
        shared._validate_content_digest(self)
        return self


def prior_inventory(root: Path, old: v3.GpuPlan | PostDebugPlan) -> dict[str, str]:
    base = root / "results/p7-holdout/raw" / old.plan_id
    require(base.is_dir() and not base.is_symlink(), "prior_results_missing")
    files = {}
    for path in sorted(base.rglob("*")):
        require(not path.is_symlink(), "unsafe_prior_artifact")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha(path.read_bytes())
        else:
            require(path.is_dir(), "unsafe_prior_artifact")
    require(bool(files), "prior_results_empty")
    return files


def prior_cost(root: Path, old: v3.GpuPlan) -> tuple[Decimal, Decimal]:
    known, unknown = Decimal(0), V2_UNKNOWN_RESERVE
    for slot in old.run_slots:
        directory = root / slot.output_path
        if not directory.exists():
            continue
        report = inspect_evaluation_run(directory)
        require(report.run_id == slot.run_id and len(report.attempts) == 1, "prior_run_mismatch")
        attempt = report.attempts[0]
        require(attempt.status in {"completed", "failed"}, "prior_execution_still_running")
        if attempt.evidence:
            known += attempt.evidence.usage.estimated_cost_usd
            unknown += Decimal(attempt.execution_count - 1)
        else:
            unknown += Decimal(attempt.execution_count)
    return known, unknown


def build_plan(root: Path, evidence_root: Path) -> PostDebugPlan:
    commit, objects = workload.verify_source(root)
    old_raw = _read(root, v3.PLAN_PATH)
    old = v3.GpuPlan.model_validate_json(old_raw)
    previous = v3._objects_at(root, old.frozen_source_commit)
    require(not (set(previous) - set(objects)), "previous_source_removed")
    changed = {p for p in objects if objects[p] != previous.get(p)}
    require(changed <= REPAIR_PATHS, "unreviewed_repair_scope")
    require(_read(evidence_root, v3.PLAN_PATH) == old_raw, "prior_plan_mismatch")
    known, unknown = prior_cost(evidence_root, old)
    input_paths = (
        v3.PLAN_PATH,
        v3.AUTHORIZATION_PATH,
        ENVIRONMENT_PATH,
        AGENT_INPUTS_PATH,
        MATRIX_PATH,
        TEST_INPUTS_PATH,
        profile.QUALIFICATION_PATH,
    )
    input_files = {p.as_posix(): sha(_read(root, p)) for p in input_paths}
    source_sha = shared._canonical_sha256(objects)
    plan_id = f"p7-postdebug-v4-{source_sha[:16]}"
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
        PostDebugPlan,
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
            "run_slots": slots,
            "previous_plan_raw_sha256": sha(old_raw),
            "input_files": input_files,
            "prior_raw_files": prior_inventory(evidence_root, old),
            "prior_known_cost_usd": known,
            "prior_unknown_cost_usd": unknown,
            "carried_cost_usd": known + unknown,
            "new_execution_budget_usd": TOTAL_BUDGET - known - unknown,
        },
    )


def validate_source_and_inputs(root: Path, plan: PostDebugPlan) -> str:
    commit, objects = workload.verify_source(root)
    require(objects == plan.execution_source_objects, "frozen_execution_source_changed")
    _git(root, "merge-base", "--is-ancestor", plan.frozen_source_commit, commit)
    require(sha(_read(root, Path("uv.lock"))) == plan.dependency_lock_sha256, "lock_changed")
    for name, expected in plan.input_files.items():
        require(sha(_read(root, Path(name))) == expected, "frozen_input_changed")
    old = v3.GpuPlan.model_validate_json(_read(root, v3.PLAN_PATH))
    require(sha(_read(root, v3.PLAN_PATH)) == plan.previous_plan_raw_sha256, "prior_plan_changed")
    require(prior_inventory(root, old) == plan.prior_raw_files, "prior_results_changed")
    return commit


def authorization(root: Path, plan: PostDebugPlan, *, execute: bool) -> PostDebugAuthorization:
    target = root / AUTH_PATH
    expected = {
        "plan_sha256": plan.content_sha256,
        "plan_raw_sha256": sha(_read(root, PLAN_PATH)),
        "frozen_source_commit": plan.frozen_source_commit,
        "carried_cost_usd": plan.carried_cost_usd,
    }
    if target.exists() or target.is_symlink():
        auth = PostDebugAuthorization.model_validate_json(_read(root, AUTH_PATH))
        require(all(getattr(auth, k) == v for k, v in expected.items()), "authorization_mismatch")
        return auth
    auth = _sealed(PostDebugAuthorization, {"approved_at": datetime.now(UTC), **expected})
    if execute:
        shared._prepare_output_parent(root, AUTH_PATH.as_posix())
        shared._write_private_create_only(target, auth.model_dump_json(indent=2).encode() + b"\n")
    return auth


def load_controls(root: Path, *, execute: bool) -> shared._Controls:
    v3.verify_runtime_environment()
    plan = PostDebugPlan.model_validate_json(_read(root, PLAN_PATH))
    commit = validate_source_and_inputs(root, plan)
    if execute:
        require(bool(os.environ.get("OPENAI_API_KEY", "").strip()), "provider_credential_required")
    return historical_assets_controls(
        root, plan=plan, authorize=lambda: authorization(root, plan, execute=execute), commit=commit
    )


def historical_assets_controls(
    root: Path,
    *,
    plan: PostDebugPlan,
    authorize: Callable[[], PostDebugAuthorization],
    commit: str,
) -> shared._Controls:
    """Reuse sealed v3 assets without manufacturing new readiness evidence."""
    # Reuse *historical* image bindings, never claim this is a fresh readiness run.
    ready_raw = _read(root, v3.READINESS_PATH)
    ready = v3.GpuReadiness.model_validate_json(ready_raw)
    old = v3.GpuPlan.model_validate_json(_read(root, v3.PLAN_PATH))
    require(
        ready.ready_for_paid_holdout
        and ready.plan_content_sha256 == old.content_sha256
        and ready.plan_raw_sha256 == sha(_read(root, v3.PLAN_PATH))
        and ready.authorization_raw_sha256 == sha(_read(root, v3.AUTHORIZATION_PATH))
        and ready.environment_raw_sha256 == sha(_read(root, ENVIRONMENT_PATH)),
        "historical_asset_bindings_changed",
    )
    agent_raw = _read(root, AGENT_INPUTS_PATH)
    agent = HoldoutAgentInputs.model_validate_json(agent_raw)
    load_test_inputs(root, agent_inputs=agent, expected_raw_sha256=TEST_INPUTS_RAW_SHA256)
    controls = shared._Controls(
        plan=plan,
        authorization=authorize(),
        environment=HoldoutEnvironmentManifest.model_validate_json(_read(root, ENVIRONMENT_PATH)),
        readiness=ready,
        agent_inputs=agent,
        readiness_raw=ready_raw,
        agent_inputs_raw=agent_raw,
        controller_commit=commit,
    )
    shared._verify_runtime_assets(root, controls)
    return controls


def gpu_resources(root: Path) -> tuple[profile.SingleQueryCpuProvider, cache_api.ReadOnlyGpuCache]:
    """Pinned metadata now; each vector is hash/schema-verified when actually read."""
    profile.load_qualification_report(root)
    directory = cache_api._safe(root / profile.CACHE_PATH)
    job, _, complete = cache_api._pinned_documents(directory)
    require(cache_store._identity(job) == PINNED_IDENTITY, "cache_identity_changed")
    files = complete.get("files")
    require(isinstance(files, dict), "cache_file_manifest_invalid")
    cache = cache_api.ReadOnlyGpuCache(
        directory / "vectors", PINNED_IDENTITY, cast(dict[str, str], files)
    )
    provider, runtime = create_cpu_provider(PINNED_IDENTITY)
    profile.verify_cpu_runtime(root, runtime)
    return profile.SingleQueryCpuProvider(provider), cache


def executor_factory(
    *, root: Path, controls: shared._Controls, openai_api_key: str, database_url: str
) -> shared.ExecutorFactory:
    provider, cache = gpu_resources(root)
    return shared._real_executor_factory(
        root=root,
        controls=controls,
        openai_api_key=openai_api_key,
        database_url=database_url,
        embedding_provider=provider,
        embedding_cache=cache,
    )


def write_report(
    root: Path,
    controls: shared._Controls,
    progress: shared._Progress,
    *,
    report_path: Path = REPORT_PATH,
    report_version: str = "post-debug-rerun-report-v4",
) -> dict[str, Any]:
    require(
        not progress.pending_slots
        and not progress.interrupted_slots
        and len(progress.outcomes) == 32,
        "matrix_incomplete",
    )
    payload = {
        "report_version": report_version,
        "evaluation_label": "post_debug_seen_task_rerun",
        "fresh_blind_holdout": False,
        "expensive_preflight_waived": True,
        "plan_sha256": controls.plan.content_sha256,
        "authorization_sha256": controls.authorization.content_sha256,
        "source_commit": controls.plan.frozen_source_commit,
        "controller_commit": controls.controller_commit,
        "slot_count": 32,
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
    target = root / report_path
    if target.exists() or target.is_symlink():
        require(json.loads(_read(root, report_path)) == payload, "report_conflict")
    else:
        shared._prepare_output_parent(root, report_path.as_posix())
        shared._write_private_create_only(target, (json.dumps(payload, indent=2) + "\n").encode())
    print(
        f"postdebug_complete=true slots=32 successes={payload['task_success_count']} "
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
            f"postdebug_plan_created=true plan_sha256={plan.content_sha256} "
            f"carried_cost_usd={plan.carried_cost_usd} "
            f"remaining_budget_usd={plan.new_execution_budget_usd}"
        )
        return 0
    plan = PostDebugPlan.model_validate_json(_read(root, PLAN_PATH))
    if args.execute:
        require(
            args.skip_preflight and args.acknowledge_post_debug,
            "explicit_waiver_and_label_required",
        )
        require(args.confirm_plan_sha256 == plan.content_sha256, "plan_confirmation_mismatch")
    with _execution_lock(root, "p7-paid-evaluation-global"):
        # Also exclude an old v3 invocation while the rerun is using its budget.
        old = v3.GpuPlan.model_validate_json(_read(root, v3.PLAN_PATH))
        with _execution_lock(root, old.plan_id):
            controls = load_controls(root, execute=args.execute)
            print(
                "postdebug_start=true expensive_preflight=skipped "
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
                executor_factory_builder=executor_factory,
                carried_cost_usd=plan.carried_cost_usd,
            )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except shared.HoldoutRunnerError as error:
        raise SystemExit(str(error)) from None
    except ValueError as error:
        # Only our fixed taxonomy is exposed; validation details may contain input data.
        code = (
            str(error)
            if type(error) is ValueError and str(error).isidentifier()
            else "postdebug_validation_failed"
        )
        raise SystemExit(code) from None
