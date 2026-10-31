"""Unified-diff policy and patch Tool session tests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest
from pydantic import JsonValue

from repopilot.enums import (
    ErrorType,
    PatchErrorCode,
    PatchStatus,
    PolicyDecision,
    ToolResultStatus,
)
from repopilot.sandbox import ApplyPatchRequest, SandboxHandle
from repopilot.schemas import ArtifactRef, CommandSpec, Observation, TaskSpec, ToolCall
from repopilot.tools import (
    ApplyPatchTool,
    InspectDiffTool,
    PatchLimits,
    PatchToolSession,
    PatchValidationError,
    ToolRegistry,
    parse_unified_diff,
)

CALL_ID = UUID("66666666-6666-4666-8666-666666666666")
INSPECT_CALL_ID = UUID("77777777-7777-4777-8777-777777777777")
STARTED_AT = datetime(2026, 8, 6, 9, tzinfo=UTC)
VALID_DIFF = (
    "diff --git a/pkg/module.py b/pkg/module.py\n"
    "--- a/pkg/module.py\n"
    "+++ b/pkg/module.py\n"
    "@@ -1,2 +1,2 @@\n"
    "-VALUE = 1\n"
    "+VALUE = 2\n"
    " KEEP = True\n"
)
TWO_FILE_DIFF = VALID_DIFF + VALID_DIFF.replace("pkg/module.py", "pkg/other.py")
TWO_HUNK_DIFF = (
    "diff --git a/pkg/module.py b/pkg/module.py\n"
    "--- a/pkg/module.py\n"
    "+++ b/pkg/module.py\n"
    "@@ -1 +1 @@\n"
    "-VALUE = 1\n"
    "+VALUE = 2\n"
    "@@ -3 +3 @@\n"
    "-OTHER = 1\n"
    "+OTHER = 2\n"
)


def limits(**updates: int) -> PatchLimits:
    return PatchLimits.model_validate(
        {
            "max_bytes": 262_144,
            "max_files": 20,
            "max_hunks": 100,
            "max_changed_lines": 2_000,
            **updates,
        }
    )


def make_task() -> TaskSpec:
    return TaskSpec(
        task_id="patch-tool-task",
        repository_path="/tmp/patch-tool-task",
        base_commit="a" * 40,
        issue_title="Change one constant",
        issue_body="Update the existing module constant.",
        test_commands=(CommandSpec(argv=("python", "-m", "unittest", "-q")),),
    )


def make_handle(task: TaskSpec) -> SandboxHandle:
    return SandboxHandle(
        sandbox_id="sandbox-patch-tool",
        task_id=task.task_id,
        repository_commit=task.base_commit,
        prepared_at=STARTED_AT,
    )


def observation(
    *,
    exit_code: int = 0,
    error_type: ErrorType | None = None,
    stderr: str | None = None,
) -> Observation:
    stderr_ref = (
        ArtifactRef(
            relative_path="artifacts/sandbox-patch-tool/error.stderr.txt",
            sha256="e" * 64,
            size_bytes=len(stderr.encode()),
            media_type="text/plain; charset=utf-8",
        )
        if stderr is not None
        else None
    )
    return Observation(
        call_id=CALL_ID,
        started_at=STARTED_AT,
        duration_ms=4,
        exit_code=exit_code,
        stderr_ref=stderr_ref,
        stderr_preview=stderr,
        sandbox_id="sandbox-patch-tool",
        policy_decision=PolicyDecision.ALLOWED,
        error_type=error_type,
    )


class StubPatchSandbox:
    def __init__(
        self,
        *,
        observations: list[Observation] | None = None,
        mismatch_artifact: bool = False,
    ) -> None:
        self.observations = observations or [observation()]
        self.mismatch_artifact = mismatch_artifact
        self.staged: list[bytes] = []
        self.requests: list[ApplyPatchRequest] = []

    def stage_patch(
        self,
        handle: SandboxHandle,
        call_id: UUID,
        content: bytes,
    ) -> ArtifactRef:
        self.staged.append(content)
        return ArtifactRef(
            relative_path=f"artifacts/{handle.sandbox_id}/{call_id}.patch.diff",
            sha256=("f" * 64 if self.mismatch_artifact else hashlib.sha256(content).hexdigest()),
            size_bytes=len(content),
            media_type="text/x-diff; charset=utf-8",
        )

    def apply_patch(
        self,
        handle: SandboxHandle,
        request: ApplyPatchRequest,
    ) -> Observation:
        self.requests.append(request)
        return self.observations.pop(0)


def make_call(
    tool_name: str,
    arguments: dict[str, JsonValue],
    *,
    call_id: UUID = CALL_ID,
) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        tool_name=tool_name,
        validated_arguments=arguments,
        requested_at=STARTED_AT,
    )


def test_parser_returns_bounded_safe_summary() -> None:
    parsed = parse_unified_diff(VALID_DIFF, limits())

    assert parsed.content == VALID_DIFF
    assert parsed.summary.files_changed == ("pkg/module.py",)
    assert parsed.summary.hunk_count == 1
    assert parsed.summary.added_lines == 1
    assert parsed.summary.removed_lines == 1
    assert parsed.summary.sha256 == hashlib.sha256(VALID_DIFF.encode()).hexdigest()


def test_normalization_artifact_is_what_the_sandbox_receives() -> None:
    task = make_task()
    sandbox = StubPatchSandbox()
    session = PatchToolSession(
        sandbox=sandbox,
        handle=make_handle(task),
        task=task,
        limits=limits(),
        preview_bytes=1_000,
    )
    proposal = VALID_DIFF.replace("-1,2 +1,2", "-1,20 +1,1")
    result = session.apply(proposal, call=make_call("apply_patch", {"diff": proposal}))
    assert result.status is PatchStatus.APPLIED
    assert sandbox.staged == [VALID_DIFF.encode()]
    assert result.summary is not None
    assert result.summary.original_sha256 == hashlib.sha256(proposal.encode()).hexdigest()
    assert result.patch_state.diff_ref is not None
    assert result.patch_state.diff_ref.sha256 == result.summary.sha256


@pytest.mark.parametrize(
    ("candidate", "code"),
    [
        (VALID_DIFF.rstrip("\n"), PatchErrorCode.INVALID_DIFF),
        (
            VALID_DIFF.replace("pkg/module.py", "../outside.py"),
            PatchErrorCode.PATH_DENIED,
        ),
        (
            VALID_DIFF.replace("pkg/module.py", ".git/config"),
            PatchErrorCode.PATH_DENIED,
        ),
        (
            VALID_DIFF.replace(
                "diff --git a/pkg/module.py b/pkg/module.py",
                "diff --git a/pkg/module.py b/pkg/other.py",
            ),
            PatchErrorCode.UNSUPPORTED_PATCH,
        ),
        (
            VALID_DIFF.replace("--- a/pkg/module.py\n", "new file mode 100644\n"),
            PatchErrorCode.UNSUPPORTED_PATCH,
        ),
        (
            TWO_HUNK_DIFF.replace("@@ -3 +3 @@", "@@ -3 +1 @@"),
            PatchErrorCode.INVALID_DIFF,
        ),
        (
            VALID_DIFF + VALID_DIFF,
            PatchErrorCode.INVALID_DIFF,
        ),
    ],
)
def test_parser_rejects_unsafe_or_ambiguous_diff(
    candidate: str,
    code: PatchErrorCode,
) -> None:
    with pytest.raises(PatchValidationError) as captured:
        parse_unified_diff(candidate, limits())

    assert captured.value.code is code
    assert candidate not in str(captured.value)


@pytest.mark.parametrize(
    ("candidate", "configured", "code"),
    [
        (VALID_DIFF, {"max_bytes": 20}, PatchErrorCode.PATCH_TOO_LARGE),
        (TWO_FILE_DIFF, {"max_files": 1}, PatchErrorCode.TOO_MANY_FILES),
        (TWO_HUNK_DIFF, {"max_hunks": 1}, PatchErrorCode.TOO_MANY_HUNKS),
        (
            VALID_DIFF,
            {"max_changed_lines": 1},
            PatchErrorCode.TOO_MANY_CHANGED_LINES,
        ),
    ],
)
def test_parser_enforces_configured_complexity_limits(
    candidate: str,
    configured: dict[str, int],
    code: PatchErrorCode,
) -> None:
    with pytest.raises(PatchValidationError) as captured:
        parse_unified_diff(candidate, limits(**configured))

    assert captured.value.code is code


def test_apply_and_inspect_tools_share_one_artifact_backed_patch_state() -> None:
    task = make_task()
    sandbox = StubPatchSandbox()
    session = PatchToolSession(
        sandbox=sandbox,
        handle=make_handle(task),
        task=task,
        limits=limits(),
        preview_bytes=32,
    )
    registry = ToolRegistry(max_output_bytes=65_536)
    registry.register(ApplyPatchTool(session))
    registry.register(InspectDiffTool(session))

    applied = registry.invoke(make_call("apply_patch", {"diff": VALID_DIFF}))
    inspected = registry.invoke(make_call("inspect_diff", {}, call_id=INSPECT_CALL_ID))

    assert applied.status is ToolResultStatus.SUCCESS
    assert applied.payload is not None
    applied_state = cast(dict[str, JsonValue], applied.payload["patch_state"])
    assert applied.payload["status"] == PatchStatus.APPLIED
    assert applied_state["revision"] == 1
    assert sandbox.staged == [VALID_DIFF.encode()]
    assert len(sandbox.requests) == 1
    assert inspected.status is ToolResultStatus.SUCCESS
    assert inspected.payload is not None
    inspected_state = cast(dict[str, JsonValue], inspected.payload["patch_state"])
    assert inspected.payload["available"] is True
    assert inspected_state["diff_ref"] == applied_state["diff_ref"]
    assert inspected.payload["truncated"] is True
    assert len(str(inspected.payload["preview"]).encode()) <= 32


def test_rejected_attempt_does_not_stage_or_replace_current_patch_state() -> None:
    task = make_task()
    sandbox = StubPatchSandbox()
    session = PatchToolSession(
        sandbox=sandbox,
        handle=make_handle(task),
        task=task,
        limits=limits(),
        preview_bytes=64,
    )
    first = session.apply(VALID_DIFF, call=make_call("apply_patch", {"diff": VALID_DIFF}))

    rejected = session.apply(
        VALID_DIFF.replace("pkg/module.py", ".git/config"),
        call=make_call(
            "apply_patch",
            {"diff": "hidden"},
            call_id=INSPECT_CALL_ID,
        ),
    )

    assert first.status is PatchStatus.APPLIED
    assert rejected.status is PatchStatus.REJECTED
    assert rejected.error_code is PatchErrorCode.PATH_DENIED
    assert rejected.patch_state == first.patch_state
    assert len(sandbox.staged) == 1


def test_sandbox_conflict_and_artifact_mismatch_are_typed_rejections() -> None:
    task = make_task()
    conflict = observation(
        exit_code=45,
        error_type=ErrorType.PATCH_ERROR,
        stderr="repopilot_patch_error:apply_conflict\n",
    )
    conflict_session = PatchToolSession(
        sandbox=StubPatchSandbox(observations=[conflict]),
        handle=make_handle(task),
        task=task,
        limits=limits(),
        preview_bytes=64,
    )
    mismatch_session = PatchToolSession(
        sandbox=StubPatchSandbox(mismatch_artifact=True),
        handle=make_handle(task),
        task=task,
        limits=limits(),
        preview_bytes=64,
    )

    conflict_result = conflict_session.apply(
        VALID_DIFF,
        call=make_call("apply_patch", {"diff": VALID_DIFF}),
    )
    mismatch_result = mismatch_session.apply(
        VALID_DIFF,
        call=make_call("apply_patch", {"diff": VALID_DIFF}),
    )

    assert conflict_result.error_code is PatchErrorCode.APPLY_CONFLICT
    assert conflict_result.patch_state.status is PatchStatus.NONE
    assert mismatch_result.error_code is PatchErrorCode.ARTIFACT_MISMATCH
    assert mismatch_result.observation is None


def test_protected_test_rejection_preserves_unchanged_candidate_state() -> None:
    task = make_task()
    rejection = observation(
        exit_code=44,
        error_type=ErrorType.PATCH_ERROR,
        stderr="repopilot_patch_error:protected_test_path\n",
    )
    session = PatchToolSession(
        sandbox=StubPatchSandbox(observations=[rejection]),
        handle=make_handle(task),
        task=task,
        limits=limits(),
        preview_bytes=64,
    )
    before = session.state

    result = session.apply(VALID_DIFF, call=make_call("apply_patch", {"diff": VALID_DIFF}))

    assert result.error_code is PatchErrorCode.PROTECTED_TEST_PATH
    assert result.patch_state == before == session.state
    assert result.observation == rejection
    assert "no candidate source changes were applied" in result.message
    assert "source-only patch" in result.message
    assert not session.inspect().available


def test_inspect_before_success_is_explicitly_unavailable() -> None:
    task = make_task()
    session = PatchToolSession(
        sandbox=StubPatchSandbox(),
        handle=make_handle(task),
        task=task,
        limits=limits(),
        preview_bytes=64,
    )

    result = session.inspect()

    assert result.available is False
    assert result.error_code is PatchErrorCode.INSPECT_UNAVAILABLE
    assert result.patch_state.status is PatchStatus.NONE


def test_registry_rejects_extra_patch_control_fields_before_staging() -> None:
    task = make_task()
    sandbox = StubPatchSandbox()
    session = PatchToolSession(
        sandbox=sandbox,
        handle=make_handle(task),
        task=task,
        limits=limits(),
        preview_bytes=64,
    )
    registry = ToolRegistry()
    registry.register(ApplyPatchTool(session))

    result = registry.invoke(
        make_call(
            "apply_patch",
            {
                "diff": VALID_DIFF,
                "path": "/tmp/host-target",
            },
        )
    )

    assert result.status is ToolResultStatus.ERROR
    assert result.error_type is ErrorType.INVALID_TOOL_ARGUMENTS
    assert sandbox.staged == []
