"""Run the separately authorized, post-repair holdout; default to a zero-request dry run.

Paid execution additionally requires the exact v2 plan digest. The first invocation
defaults to one slot; larger batches must be requested explicitly. Raw task execution,
append-only recovery, and conservative unknown-cost accounting use the existing Runner.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, TypeAdapter, ValidationError, model_validator

from repopilot.schemas import GitSha, Identifier, Sha256, StrictModel
from scripts import p7_holdout_readiness_v2 as probes
from scripts import p7_holdout_runner as shared
from scripts.p7_holdout_readiness import ReadinessError
from scripts.p7_holdout_requalification import HoldoutApplicationPlanV2, _read
from scripts.p7_holdout_v2 import (
    READINESS_V2_PATH,
    HoldoutRuntimeReadinessV2,
    PaidHoldoutAuthorizationV2,
    load_controls,
    validate_readiness_bindings,
    verify_runtime_environment,
)

EXECUTION_REPORT_V2_PATH = Path("results/p7-holdout/execution-report-v2.json")


class HoldoutExecutionReportV2(StrictModel):
    """A v2 raw-run index, never a reinterpretation of historical v1 evidence."""

    schema_version: Literal[2] = 2
    execution_version: Literal["holdout-execution-v2"] = "holdout-execution-v2"
    plan_id: Identifier
    experiment_id: Identifier
    plan_content_sha256: Sha256
    freeze_content_sha256: Sha256
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
    application_accounted_cost_usd: Decimal = Field(ge=0)
    authorized_cost_usd: Decimal = Field(ge=0)
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
            or self.completed_slot_count
            != sum(item.attempt_status == "completed" for item in self.outcomes)
            or self.technical_failure_slot_count
            != sum(item.attempt_status == "technical_failure" for item in self.outcomes)
            or self.task_success_count != sum(item.task_success is True for item in self.outcomes)
            or self.known_estimated_cost_usd != known
            or self.reserved_unknown_cost_usd != reserved
            or self.application_accounted_cost_usd != known + reserved
            or self.final_in_flight_overshoot_observed
            != (self.application_accounted_cost_usd > self.authorized_cost_usd)
        ):
            raise ValueError("v2 formal execution report totals are inconsistent")
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
        help="explicitly continue other slots; failed attempts retain the USD 1 cost reserve",
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
        )

    if not arguments.execute:
        return invoke()
    if arguments.confirm_plan_sha256 != controls.plan.content_sha256:
        raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.INVALID_CONFIGURATION)
    with _execution_lock(root, controls.plan.plan_id):
        return invoke()


@contextmanager
def _execution_lock(root: Path, plan_id: str) -> Iterator[None]:
    """Keep one paid controller per plan; never unlink an inode another process may lock."""

    try:
        TypeAdapter(Identifier).validate_python(plan_id)
    except ValidationError:
        raise shared.HoldoutRunnerError(
            shared.HoldoutRunnerErrorCode.INVALID_CONFIGURATION
        ) from None
    relative = f"results/p7-holdout/locks/{plan_id}.lock"
    shared._prepare_output_parent(root, relative)
    try:
        descriptor = os.open(root / relative, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    except OSError:
        raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.OUTPUT_CONFLICT) from None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise shared.HoldoutRunnerError(
                shared.HoldoutRunnerErrorCode.CONCURRENT_EXECUTION
            ) from None
        except OSError:
            raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.OUTPUT_CONFLICT) from None
        yield
    finally:
        os.close(descriptor)


def _load_and_verify_controls(root: Path, *, require_credential: bool) -> shared._Controls:
    """Validate v2 authority and current runtime before constructing any executor."""

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
    _verify_runtime_assets(root, controls)
    return controls


def _verify_runtime_assets(root: Path, controls: shared._Controls) -> None:
    tasks = {task.task_id: task for task in controls.agent_inputs.tasks}
    for binding in controls.environment.bindings:
        task = tasks.get(binding.task_id)
        slots = tuple(slot for slot in controls.plan.run_slots if slot.task_id == binding.task_id)
        if (
            task is None
            or task.repository_path is not None
            or task.repository_url is None
            or task.base_commit != binding.base_commit
            or len(slots) != 4
            or any(slot.base_commit != binding.base_commit for slot in slots)
        ):
            raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.SOURCE_IDENTITY_MISMATCH)
    try:
        images, failed_images = probes._resolve_images(root, controls.environment)
        failed_checkouts = probes._checkouts(root, controls.environment)
    except (OSError, ValueError):
        raise shared.HoldoutRunnerError(
            shared.HoldoutRunnerErrorCode.SOURCE_ARTIFACT_INVALID
        ) from None
    if failed_images:
        raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.SANDBOX_IMAGE_UNAVAILABLE)
    if failed_checkouts:
        raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.CHECKOUT_UNAVAILABLE)
    if images != controls.readiness.images:
        raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.SOURCE_IDENTITY_MISMATCH)


def _load_bound_readiness(root: Path) -> shared._Controls:
    """Bind the new readiness to exact control bytes and unchanged execution code."""

    try:
        controls = load_controls(root)
        readiness_raw = _read(root, READINESS_V2_PATH)
        readiness = HoldoutRuntimeReadinessV2.model_validate_json(readiness_raw)
    except (OSError, ValueError, subprocess.SubprocessError):
        raise shared.HoldoutRunnerError(
            shared.HoldoutRunnerErrorCode.SOURCE_ARTIFACT_INVALID
        ) from None
    try:
        validate_readiness_bindings(root, controls, readiness)
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


def _write_final_report_if_complete(
    root: Path,
    controls: shared._Controls,
    progress: shared._Progress,
) -> HoldoutExecutionReportV2:
    if (
        not isinstance(controls.plan, HoldoutApplicationPlanV2)
        or not isinstance(controls.authorization, PaidHoldoutAuthorizationV2)
        or not isinstance(controls.readiness, HoldoutRuntimeReadinessV2)
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
        "schema_version": 2,
        "execution_version": "holdout-execution-v2",
        "plan_id": controls.plan.plan_id,
        "experiment_id": controls.plan.experiment_id,
        "plan_content_sha256": controls.plan.content_sha256,
        "freeze_content_sha256": controls.plan.freeze_marker_sha256,
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
        "application_accounted_cost_usd": progress.accounted_cost_usd,
        "authorized_cost_usd": controls.authorization.max_total_cost_usd,
        "final_in_flight_overshoot_observed": (
            progress.accounted_cost_usd > controls.authorization.max_total_cost_usd
        ),
        "outcomes": ordered,
        "aggregation_deferred_to": "P7-S7",
        "gold_visible_to_agent": False,
        "source_rows_loaded_by_formal_runner": False,
        "formal_execution_complete": True,
    }
    report = HoldoutExecutionReportV2.model_validate(
        {**payload, "content_sha256": shared._canonical_sha256(payload)}
    )
    shared._prepare_output_parent(root, EXECUTION_REPORT_V2_PATH.as_posix())
    target = root / EXECUTION_REPORT_V2_PATH
    if target.is_symlink():
        raise shared.HoldoutRunnerError(shared.HoldoutRunnerErrorCode.OUTPUT_CONFLICT)
    if target.exists():
        try:
            existing = HoldoutExecutionReportV2.model_validate_json(target.read_bytes())
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
        "holdout_execution_v2_complete=true slots=32 "
        f"known_cost_usd={report.known_estimated_cost_usd} "
        f"reserved_unknown_cost_usd={report.reserved_unknown_cost_usd} "
        "aggregation_deferred_to=P7-S7"
    )
    return report


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except shared.HoldoutRunnerError as error:
        raise SystemExit(str(error)) from None
