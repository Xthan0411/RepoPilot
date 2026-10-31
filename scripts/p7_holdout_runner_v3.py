"""Execute the separately authorized GPU-cache holdout, defaulting to a read-only check.

The historical failed attempt keeps its USD 1 unknown-cost reserve. Only new v3
slots are executed, under the same USD 32 total application budget. No cache miss
may trigger corpus embedding and no qualification-only query allowlist is reused.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, ValidationError, model_validator

from repopilot.schemas import GitSha, Identifier, Sha256, StrictModel
from scripts import p7_holdout_readiness_v2 as probes
from scripts import p7_holdout_runner as shared
from scripts import p7_holdout_runner_v2 as previous
from scripts.p7_gpu_cache_acceptance import verify_pinned_cache
from scripts.p7_gpu_execution_profile import CACHE_PATH, build_gpu_resources
from scripts.p7_holdout_gpu import (
    READINESS_PATH,
    GpuAuthorization,
    GpuPlan,
    GpuReadiness,
    load_controls,
    validate_readiness_bindings,
    verify_runtime_environment,
)
from scripts.p7_holdout_readiness import ReadinessError
from scripts.p7_holdout_requalification import _read

EXECUTION_REPORT_V3_PATH = Path("results/p7-holdout/execution-report-v3.json")
CARRIED_UNKNOWN_COST_USD = Decimal("1.00")


class HoldoutExecutionReportV3(StrictModel):
    """New slot outcomes plus an explicit, never forgiven historical cost reserve."""

    schema_version: Literal[3] = 3
    execution_version: Literal["holdout-execution-v3"] = "holdout-execution-v3"
    plan_id: Identifier
    experiment_id: Identifier
    plan_content_sha256: Sha256
    freeze_content_sha256: Sha256
    runtime_profile_sha256: Sha256
    qualification_raw_sha256: Sha256
    carry_forward_sha256: Sha256
    authorization_content_sha256: Sha256
    readiness_raw_sha256: Sha256
    readiness_content_sha256: Sha256
    agent_inputs_raw_sha256: Sha256
    agent_inputs_content_sha256: Sha256
    test_inputs_raw_sha256: Sha256
    test_protocol_version: Literal["isolated-regression-tests-v1"]
    controller_commit: GitSha
    completed_at: AwareDatetime
    slot_count: Literal[32] = 32
    completed_slot_count: int = Field(ge=0, le=32, strict=True)
    technical_failure_slot_count: int = Field(ge=0, le=32, strict=True)
    task_success_count: int = Field(ge=0, le=32, strict=True)
    known_estimated_cost_usd: Decimal = Field(ge=0)
    reserved_unknown_cost_usd: Decimal = Field(ge=0)
    carried_unknown_cost_usd: Decimal = Field(ge=0)
    new_execution_accounted_cost_usd: Decimal = Field(ge=0)
    application_accounted_cost_usd: Decimal = Field(ge=0)
    authorized_cost_usd: Decimal = Field(ge=0)
    new_execution_budget_usd: Decimal = Field(ge=0)
    final_in_flight_overshoot_observed: bool
    outcomes: tuple[shared.HoldoutSlotOutcome, ...]
    aggregation_deferred_to: Literal["P7-S7"] = "P7-S7"
    gold_visible_to_agent: Literal[False] = False
    source_rows_loaded_by_formal_runner: Literal[False] = False
    formal_execution_complete: Literal[True] = True
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        known = sum(
            (item.usage.estimated_cost_usd for item in self.outcomes if item.usage is not None),
            Decimal("0"),
        )
        reserved = sum((item.reserved_unknown_cost_usd for item in self.outcomes), Decimal("0"))
        if (
            len(self.outcomes) != 32
            or len({item.run_id for item in self.outcomes}) != 32
            or self.authorized_cost_usd != Decimal("32.00")
            or self.new_execution_budget_usd != Decimal("31.00")
            or self.carried_unknown_cost_usd != CARRIED_UNKNOWN_COST_USD
            or self.completed_slot_count
            != sum(item.attempt_status == "completed" for item in self.outcomes)
            or self.technical_failure_slot_count
            != sum(item.attempt_status == "technical_failure" for item in self.outcomes)
            or self.task_success_count != sum(item.task_success is True for item in self.outcomes)
            or self.known_estimated_cost_usd != known
            or self.reserved_unknown_cost_usd != reserved
            or self.new_execution_accounted_cost_usd != known + reserved
            or self.application_accounted_cost_usd
            != known + reserved + self.carried_unknown_cost_usd
            or self.final_in_flight_overshoot_observed
            != (self.application_accounted_cost_usd > self.authorized_cost_usd)
        ):
            raise ValueError("v3 formal execution report totals are inconsistent")
        shared._validate_content_digest(self)
        return self


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-plan-sha256")
    parser.add_argument(
        "--max-new-slots", type=int, default=1, choices=range(1, 33), metavar="1..32"
    )
    parser.add_argument(
        "--continue-after-technical-failure",
        action="store_true",
        help="explicitly continue other v3 slots; all unknown-cost reserves remain counted",
    )
    arguments = parser.parse_args()
    try:
        root = arguments.repository_root.resolve(strict=True)
    except OSError:
        raise shared.HoldoutRunnerError(
            shared.HoldoutRunnerErrorCode.INVALID_CONFIGURATION
        ) from None
    controls = _load_and_verify_controls(root, require_credential=arguments.execute)

    def invoke() -> int:
        return shared._run_loaded_controls(
            root=root,
            controls=controls,
            execute=arguments.execute,
            confirm_plan_sha256=arguments.confirm_plan_sha256,
            max_new_slots=arguments.max_new_slots,
            continue_after_technical_failure=arguments.continue_after_technical_failure,
            report_writer=_write_final_report_if_complete,
            executor_factory_builder=_gpu_executor_factory,
            carried_cost_usd=CARRIED_UNKNOWN_COST_USD,
        )

    if not arguments.execute:
        return invoke()
    if arguments.confirm_plan_sha256 != controls.plan.content_sha256:
        raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.INVALID_CONFIGURATION)
    with previous._execution_lock(root, controls.plan.plan_id):
        return invoke()


def _load_and_verify_controls(root: Path, *, require_credential: bool) -> shared._Controls:
    try:
        verify_runtime_environment()
    except ReadinessError as error:
        raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode(error.code.value)) from None
    except ValueError:
        raise shared.HoldoutRunnerError(
            shared.HoldoutRunnerErrorCode.UNAPPROVED_ENVIRONMENT_OVERRIDE
        ) from None
    if require_credential and not os.environ.get("OPENAI_API_KEY", "").strip():
        raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.PROVIDER_CREDENTIAL_REQUIRED)
    controls = _load_bound_readiness(root)
    if probes._missing_dependencies():
        raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.LOCAL_DEPENDENCY_UNAVAILABLE)
    previous._verify_runtime_assets(root, controls)
    if not require_credential:
        # Execute verifies the entire cache in its factory before constructing an Agent.
        # Dry-run must remain model-free, but still checks the same immutable cache files.
        try:
            verify_pinned_cache(root / CACHE_PATH)
        except Exception:
            raise shared.HoldoutRunnerError(
                shared.HoldoutRunnerErrorCode.SOURCE_ARTIFACT_INVALID
            ) from None
    return controls


def _load_bound_readiness(root: Path) -> shared._Controls:
    try:
        controls = load_controls(root)
        readiness_raw = _read(root, READINESS_PATH)
        readiness = GpuReadiness.model_validate_json(readiness_raw)
    except (OSError, ValueError, subprocess.SubprocessError):
        raise shared.HoldoutRunnerError(
            shared.HoldoutRunnerErrorCode.SOURCE_ARTIFACT_INVALID
        ) from None
    try:
        validate_readiness_bindings(root, controls, readiness)
        _validate_budget(controls.plan, controls.authorization)
    except (OSError, ValueError, subprocess.SubprocessError):
        raise shared.HoldoutRunnerError(
            shared.HoldoutRunnerErrorCode.SOURCE_IDENTITY_MISMATCH
        ) from None
    return shared._Controls(
        plan=controls.plan,
        authorization=controls.authorization,
        environment=controls.environment,
        readiness=readiness,
        agent_inputs=controls.agent_inputs,
        readiness_raw=readiness_raw,
        agent_inputs_raw=controls.agent_inputs_raw,
        controller_commit=controls.controller_commit,
    )


def _validate_budget(plan: GpuPlan, authorization: GpuAuthorization) -> None:
    if (
        plan.carried_unknown_cost_usd != CARRIED_UNKNOWN_COST_USD
        or plan.application_budget_usd != Decimal("32.00")
        or plan.new_execution_budget_usd != Decimal("31.00")
        or authorization.max_total_cost_usd != Decimal("32.00")
        or authorization.carried_unknown_cost_usd != CARRIED_UNKNOWN_COST_USD
        or authorization.new_execution_budget_usd != Decimal("31.00")
    ):
        raise ValueError("v3 budget binding mismatch")


def _gpu_executor_factory(
    *,
    root: Path,
    controls: shared._Controls,
    openai_api_key: str,
    database_url: str,
) -> shared.ExecutorFactory:
    if (
        not isinstance(controls.plan, GpuPlan)
        or not isinstance(controls.authorization, GpuAuthorization)
        or not isinstance(controls.readiness, GpuReadiness)
    ):
        raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.SOURCE_IDENTITY_MISMATCH)
    _validate_budget(controls.plan, controls.authorization)
    provider, cache = build_gpu_resources(root)
    return shared._real_executor_factory(
        root=root,
        controls=controls,
        openai_api_key=openai_api_key,
        database_url=database_url,
        embedding_provider=provider,
        embedding_cache=cache,
    )


def _write_final_report_if_complete(
    root: Path, controls: shared._Controls, progress: shared._Progress
) -> HoldoutExecutionReportV3:
    if (
        not isinstance(controls.plan, GpuPlan)
        or not isinstance(controls.authorization, GpuAuthorization)
        or not isinstance(controls.readiness, GpuReadiness)
        or progress.carried_cost_usd != CARRIED_UNKNOWN_COST_USD
        or progress.pending_slots
        or progress.interrupted_slots
        or len(progress.outcomes) != 32
    ):
        raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.OUTPUT_CONFLICT)
    ordered = tuple(
        next(item for item in progress.outcomes if item.run_id == slot.run_id)
        for slot in controls.plan.run_slots
    )
    payload: dict[str, object] = {
        "schema_version": 3,
        "execution_version": "holdout-execution-v3",
        "plan_id": controls.plan.plan_id,
        "experiment_id": controls.plan.experiment_id,
        "plan_content_sha256": controls.plan.content_sha256,
        "freeze_content_sha256": controls.plan.freeze_marker_sha256,
        "runtime_profile_sha256": controls.plan.runtime_profile_sha256,
        "qualification_raw_sha256": controls.plan.qualification_raw_sha256,
        "carry_forward_sha256": controls.plan.carry_forward_sha256,
        "authorization_content_sha256": controls.authorization.content_sha256,
        "readiness_raw_sha256": hashlib.sha256(controls.readiness_raw).hexdigest(),
        "readiness_content_sha256": controls.readiness.content_sha256,
        "agent_inputs_raw_sha256": hashlib.sha256(controls.agent_inputs_raw).hexdigest(),
        "agent_inputs_content_sha256": controls.agent_inputs.content_sha256,
        "test_inputs_raw_sha256": controls.plan.test_inputs_raw_sha256,
        "test_protocol_version": controls.plan.test_protocol_version,
        "controller_commit": controls.controller_commit,
        "completed_at": datetime.now(UTC),
        "slot_count": 32,
        "completed_slot_count": sum(item.attempt_status == "completed" for item in ordered),
        "technical_failure_slot_count": sum(
            item.attempt_status == "technical_failure" for item in ordered
        ),
        "task_success_count": sum(item.task_success is True for item in ordered),
        "known_estimated_cost_usd": progress.known_cost_usd,
        "reserved_unknown_cost_usd": progress.reserved_unknown_cost_usd,
        "carried_unknown_cost_usd": progress.carried_cost_usd,
        "new_execution_accounted_cost_usd": (
            progress.known_cost_usd + progress.reserved_unknown_cost_usd
        ),
        "application_accounted_cost_usd": progress.accounted_cost_usd,
        "authorized_cost_usd": controls.authorization.max_total_cost_usd,
        "new_execution_budget_usd": controls.plan.new_execution_budget_usd,
        "final_in_flight_overshoot_observed": (
            progress.accounted_cost_usd > controls.authorization.max_total_cost_usd
        ),
        "outcomes": ordered,
    }
    # Defaults must be included in the canonical seal, not added after hashing.
    payload.update(
        aggregation_deferred_to="P7-S7",
        gold_visible_to_agent=False,
        source_rows_loaded_by_formal_runner=False,
        formal_execution_complete=True,
    )
    report = HoldoutExecutionReportV3.model_validate(
        {**payload, "content_sha256": shared._canonical_sha256(payload)}
    )
    shared._prepare_output_parent(root, EXECUTION_REPORT_V3_PATH.as_posix())
    target = root / EXECUTION_REPORT_V3_PATH
    if target.is_symlink():
        raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.OUTPUT_CONFLICT)
    if target.exists():
        try:
            existing = HoldoutExecutionReportV3.model_validate_json(target.read_bytes())
        except (OSError, ValidationError):
            raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.OUTPUT_CONFLICT) from None
        if existing != report.model_copy(
            update={
                "completed_at": existing.completed_at,
                "content_sha256": existing.content_sha256,
            }
        ):
            raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.OUTPUT_CONFLICT)
        report = existing
    else:
        shared._write_private_create_only(target, report.model_dump_json(indent=2).encode() + b"\n")
    print(
        "holdout_execution_v3_complete=true slots=32 "
        f"known_cost_usd={report.known_estimated_cost_usd} "
        f"reserved_unknown_cost_usd={report.reserved_unknown_cost_usd} "
        f"carried_unknown_cost_usd={report.carried_unknown_cost_usd} "
        f"application_accounted_cost_usd={report.application_accounted_cost_usd} "
        "aggregation_deferred_to=P7-S7"
    )
    return report


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except shared.HoldoutRunnerError as error:
        raise SystemExit(str(error)) from None
