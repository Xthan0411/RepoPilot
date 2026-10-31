"""Zero-provider qualification of the actual isolated test/verification path.

Run on the replacement Linux x86 VM after restoring the exact clean checkouts and
digest-pinned images. This evaluator-only command uses Gold as a known-good candidate
in temporary containers, never constructs an Agent/Provider, and never authorizes paid
execution. Unlike the historical exit-code preflight, every result is interpreted by
RunTestsTool and checked against the production verifier's evidence requirements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import Field, model_validator

from repopilot.config import RepoPilotSettings
from repopilot.enums import PolicyDecision, SandboxCommandPhase, TestRunCategory, TestStatus
from repopilot.evals import load_experiment_matrix, select_experiment_arm
from repopilot.evals.dataset import load_frozen_dataset
from repopilot.evals.isolated_tests import build_isolated_test_sandbox_factory
from repopilot.repair_workflow import RepairSandboxFactory
from repopilot.sandbox import ApplyPatchRequest, RunRequest, SandboxHandle
from repopilot.schemas import GitSha, Sha256, StrictModel, TaskId, TaskSpec, TestRunResult, ToolCall
from repopilot.tools.sandbox import RunTestsArguments, RunTestsResult, RunTestsTool
from repopilot.verifier import _valid_completed_test_evidence
from scripts.p7_holdout_agent_inputs import (
    AGENT_INPUTS_PATH,
    HoldoutAgentInputs,
    _canonical_sha256,
    _validate_content_digest,
)
from scripts.p7_holdout_control import (
    ENVIRONMENT_PATH,
    EXPECTED_TASK_IDS,
    MATRIX_PATH,
    HoldoutEnvironmentManifest,
    write_control_artifact,
)
from scripts.p7_holdout_test_inputs import (
    TEST_INPUTS_PATH,
    TEST_INPUTS_RAW_SHA256,
    load_test_inputs,
)

RESULT_PATH = Path("results/p7-holdout/test-protocol-canary-v2.json")
REQUIRED_CONTROLLER_PATHS = (
    "scripts/p7_test_protocol_canary.py",
    "scripts/p7_holdout_test_inputs.py",
    "scripts/p7_holdout_agent_inputs.py",
    "scripts/p7_holdout_control.py",
    "src/repopilot/evals/isolated_tests.py",
    "src/repopilot/tools/tests.py",
    TEST_INPUTS_PATH.as_posix(),
)


class ProtocolCanaryError(ValueError):
    """A fixed failure code and safe structured counts, never raw evaluator output."""

    def __init__(self, code: str, evidence: tuple[ProtocolTestEvidence, ...] = ()) -> None:
        super().__init__(code)
        self.code = code
        self.evidence = evidence


class ProtocolTestEvidence(StrictModel):
    """Counts and categories only; no hidden assertion text or ephemeral file refs."""

    command_index: int = Field(ge=0)
    status: TestStatus
    category: TestRunCategory
    exit_code: int | None
    tests_run: int = Field(ge=0)
    failures: int = Field(ge=0)
    errors: int = Field(ge=0)
    failing_test_count: int = Field(ge=0)
    truncated: bool
    verifier_accepts_evidence: bool


class ProtocolTaskEvidence(StrictModel):
    task_id: TaskId
    base_commit: GitSha
    regression_patch_sha256: Sha256
    baseline: tuple[ProtocolTestEvidence, ...]
    gold: tuple[ProtocolTestEvidence, ...]
    baseline_reproduced: Literal[True] = True
    gold_verified: Literal[True] = True

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if (
            not self.baseline
            or len(self.baseline) != len(self.gold)
            or tuple(item.command_index for item in self.baseline)
            != tuple(range(len(self.baseline)))
            or tuple(item.command_index for item in self.gold) != tuple(range(len(self.gold)))
            or not all(item.verifier_accepts_evidence for item in (*self.baseline, *self.gold))
            or not any(item.status is TestStatus.FAILED for item in self.baseline)
            or not all(item.status is TestStatus.PASSED for item in self.gold)
        ):
            raise ValueError("test protocol evidence is incomplete")
        return self


class ProtocolCanaryReport(StrictModel):
    protocol_version: Literal["isolated-regression-tests-v1"] = "isolated-regression-tests-v1"
    controller_commit: GitSha
    environment_raw_sha256: Sha256
    test_inputs_raw_sha256: Sha256
    checked_at: str
    tasks: tuple[ProtocolTaskEvidence, ...]
    provider_requests_made: Literal[0] = 0
    paid_requests_made: Literal[0] = 0
    agent_constructed: Literal[False] = False
    gold_persisted_outside_temporary_evaluator: Literal[False] = False
    formal_environment_ready: Literal[False] = False
    formal_blockers: tuple[Literal["new_source_freeze_and_authorization_required"], ...] = (
        "new_source_freeze_and_authorization_required",
    )
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if tuple(task.task_id for task in self.tasks) != EXPECTED_TASK_IDS:
            raise ValueError("full protocol canary must cover all eight tasks")
        if self.formal_blockers != ("new_source_freeze_and_authorization_required",):
            raise ValueError("protocol canary cannot authorize paid execution")
        _validate_content_digest(self)
        return self


def _evidence(result: TestRunResult) -> ProtocolTestEvidence:
    return ProtocolTestEvidence(
        command_index=result.command_index,
        status=result.status,
        category=result.category,
        exit_code=result.exit_code,
        tests_run=result.tests_run,
        failures=result.failures,
        errors=result.errors,
        failing_test_count=len(result.failing_tests),
        truncated=result.truncated,
        verifier_accepts_evidence=_valid_completed_test_evidence(result),
    )


def qualify_task(
    *,
    task: TaskSpec,
    regression_patch: str,
    gold_patch: str,
    settings: RepoPilotSettings,
    sandbox_image_reference: str,
    sandbox_factory: RepairSandboxFactory | None = None,
) -> ProtocolTaskEvidence:
    """Exercise production tools in ephemeral workspaces, retaining no solution diff."""

    factory = sandbox_factory or build_isolated_test_sandbox_factory(
        settings=settings,
        sandbox_image_reference=sandbox_image_reference,
        task=task,
        regression_patch=regression_patch,
    )
    with tempfile.TemporaryDirectory(prefix="repopilot-test-protocol-") as temporary:
        sandbox = factory(Path(temporary))
        handle: SandboxHandle | None = None
        try:
            handle = sandbox.prepare(task)
            for command in task.setup_commands:
                observation = sandbox.run(
                    handle,
                    RunRequest(call_id=uuid4(), command=command, phase=SandboxCommandPhase.SETUP),
                )
                if (
                    observation.exit_code != 0
                    or observation.error_type is not None
                    or observation.policy_decision is not PolicyDecision.ALLOWED
                    or observation.truncated
                ):
                    raise ProtocolCanaryError("protocol_setup_failed")
            tool = RunTestsTool(sandbox=sandbox, handle=handle, task=task)

            def run_tests() -> tuple[ProtocolTestEvidence, ...]:
                results = []
                for index in range(len(task.test_commands)):
                    value = tool.execute(
                        RunTestsArguments(test_index=index),
                        call=ToolCall(
                            call_id=uuid4(),
                            tool_name="run_tests",
                            validated_arguments={"test_index": index},
                            requested_at=datetime.now(UTC),
                        ),
                    )
                    if not isinstance(value, RunTestsResult):
                        raise ProtocolCanaryError("protocol_test_result_invalid")
                    results.append(_evidence(value.test_run))
                return tuple(results)

            baseline = run_tests()
            if not all(item.verifier_accepts_evidence for item in baseline) or not any(
                item.status is TestStatus.FAILED for item in baseline
            ):
                raise ProtocolCanaryError("protocol_baseline_not_reproduced", baseline)
            patch_call = uuid4()
            patch_ref = sandbox.stage_patch(handle, patch_call, gold_patch.encode())
            applied = sandbox.apply_patch(
                handle, ApplyPatchRequest(call_id=patch_call, patch_ref=patch_ref)
            )
            if (
                applied.exit_code != 0
                or applied.error_type is not None
                or applied.policy_decision is not PolicyDecision.ALLOWED
            ):
                raise ProtocolCanaryError("protocol_gold_not_applied")
            gold = run_tests()
            if not all(
                item.verifier_accepts_evidence and item.status is TestStatus.PASSED for item in gold
            ):
                raise ProtocolCanaryError("protocol_gold_failed", gold)
            return ProtocolTaskEvidence(
                task_id=task.task_id,
                base_commit=task.base_commit,
                regression_patch_sha256=hashlib.sha256(regression_patch.encode()).hexdigest(),
                baseline=baseline,
                gold=gold,
            )
        finally:
            if handle is not None:
                sandbox.destroy(handle)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _verified_controller_commit(root: Path) -> str:
    """Reject uncommitted executed modules without rejecting unrelated user notes."""

    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("controller_source_dirty")
    revision = _git(root, "rev-parse", "HEAD")
    for relative in REQUIRED_CONTROLLER_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise SystemExit("controller_source_uncommitted")
        try:
            _git(root, "ls-files", "--error-unmatch", "--", relative)
            expected = _git(root, "rev-parse", f"{revision}:{relative}")
        except subprocess.CalledProcessError:
            raise SystemExit("controller_source_uncommitted") from None
        if _git(root, "hash-object", "--no-filters", "--", relative) != expected:
            raise SystemExit("controller_source_dirty")
    return revision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--task-id", action="append", choices=EXPECTED_TASK_IDS)
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    if platform.system() != "Linux" or platform.machine() not in {"amd64", "x86_64"}:
        raise SystemExit("host_platform_mismatch")
    controller_commit = _verified_controller_commit(root)
    selected = tuple(args.task_id or EXPECTED_TASK_IDS)
    if len(set(selected)) != len(selected):
        raise SystemExit("duplicate_task_id")
    if selected == EXPECTED_TASK_IDS and (root / RESULT_PATH).exists():
        raise SystemExit("output_conflict")
    matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    bundle = load_frozen_dataset(root / matrix.dataset.manifest_path, repository_root=root)
    agent_inputs = HoldoutAgentInputs.model_validate_json((root / AGENT_INPUTS_PATH).read_bytes())
    tests = load_test_inputs(
        root, agent_inputs=agent_inputs, expected_raw_sha256=TEST_INPUTS_RAW_SHA256
    )
    environment_raw = (root / ENVIRONMENT_PATH).read_bytes()
    environment = HoldoutEnvironmentManifest.model_validate_json(environment_raw)
    if (
        environment.matrix_sha256 != matrix.content_sha256
        or agent_inputs.matrix_sha256 != matrix.content_sha256
        or bundle.manifest.content_sha256 != agent_inputs.dataset_manifest_content_sha256
        or environment.runtime_command_contract_sha256
        != agent_inputs.runtime_command_contract_sha256
    ):
        raise SystemExit("source_identity_mismatch")
    settings = select_experiment_arm(matrix, "no-semantic-rag-v1").settings(
        openai_api_key="unused-zero-provider-canary",
        database_url="postgresql://unused-zero-provider-canary",
    )
    outcomes: list[ProtocolTaskEvidence] = []
    failed = False
    for task_id in selected:
        print(f"test_protocol_started={task_id}", flush=True)
        try:
            binding = next(item for item in environment.bindings if item.task_id == task_id)
            source = next(item for item in agent_inputs.tasks if item.task_id == task_id)
            checkout = (root / binding.checkout_relative_path).resolve(strict=True)
            if (
                not checkout.is_relative_to(root)
                or _git(checkout, "rev-parse", "HEAD") != source.base_commit
                or binding.base_commit != source.base_commit
                or _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
            ):
                raise ValueError("checkout_identity_mismatch")
            task = source.model_copy(
                update={"repository_path": str(checkout), "repository_url": None}
            )
            regression = next(item for item in tests.tasks if item.task_id == task_id)
            evaluation = next(
                item for item in bundle.external_evaluation_tasks if item.task.task_id == task_id
            )
            outcomes.append(
                qualify_task(
                    task=task,
                    regression_patch=regression.regression_patch,
                    gold_patch=evaluation.expected_patch,
                    settings=settings,
                    sandbox_image_reference=binding.sandbox_image_reference,
                )
            )
            print(f"test_protocol_passed={task_id}", flush=True)
        except ProtocolCanaryError as error:
            failed = True
            print(
                "test_protocol_failure="
                + json.dumps(
                    {
                        "task_id": task_id,
                        "code": error.code,
                        "test_evidence": [item.model_dump(mode="json") for item in error.evidence],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as error:
            # Never print potentially Gold-bearing exception messages or subprocess output.
            failed = True
            print(
                f"test_protocol_failed={task_id} code=qualification_failed "
                f"exception_type={type(error).__name__}",
                flush=True,
            )
    if failed:
        print(f"test_protocol_canary_passed=false passed={len(outcomes)} paid_requests=0")
        return 1
    if _verified_controller_commit(root) != controller_commit:
        raise SystemExit("controller_source_changed")
    if selected == EXPECTED_TASK_IDS:
        payload = {
            "protocol_version": "isolated-regression-tests-v1",
            "controller_commit": controller_commit,
            "environment_raw_sha256": hashlib.sha256(environment_raw).hexdigest(),
            "test_inputs_raw_sha256": TEST_INPUTS_RAW_SHA256,
            "checked_at": datetime.now(UTC).isoformat(),
            "tasks": tuple(outcomes),
            "provider_requests_made": 0,
            "paid_requests_made": 0,
            "agent_constructed": False,
            "gold_persisted_outside_temporary_evaluator": False,
            "formal_environment_ready": False,
            "formal_blockers": ("new_source_freeze_and_authorization_required",),
        }
        report = ProtocolCanaryReport.model_validate(
            {**payload, "content_sha256": _canonical_sha256(payload)}
        )
        write_control_artifact(root / RESULT_PATH, report)
    print(
        f"test_protocol_canary_passed=true tasks={len(outcomes)} provider_requests=0 "
        "paid_requests=0 formal_ready=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
