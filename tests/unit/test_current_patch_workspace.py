"""Offline current-source and cumulative-patch checks with real local git apply."""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from repopilot.enums import PatchErrorCode, PatchStatus, PolicyDecision
from repopilot.sandbox.artifacts import SandboxArtifactStore
from repopilot.sandbox.base import ApplyPatchRequest, ReadFileRequest, SandboxHandle
from repopilot.sandbox.patching import APPLY_PATCH_HELPER
from repopilot.schemas import ArtifactRef, CommandSpec, Observation, TaskSpec, ToolCall
from repopilot.tools.patch import PatchLimits, PatchToolSession, parse_unified_diff
from repopilot.tools.patch_workspace import ExactDiffConflict, apply_exact_diff, cumulative_diff
from repopilot.tools.repository import (
    ReadOnlyRepository,
    RepositoryAccessError,
    RepositoryAccessPolicy,
)


class LocalSandbox:
    def __init__(self, path: Path, store: SandboxArtifactStore, handle: SandboxHandle) -> None:
        self.path = path
        self.store = store
        self.handle = handle
        self.applied: list[bytes] = []
        self.corrupt_after_apply = False
        self.corrupt_read_artifact = False
        self.read_preview_truncated = False
        self.omit_read_registration = False

    def stage_patch(self, handle: SandboxHandle, call_id: UUID, content: bytes) -> ArtifactRef:
        return self.store.persist_patch(call_id, content)

    def apply_patch(self, handle: SandboxHandle, request: ApplyPatchRequest) -> Observation:
        content = self.store.read(request.patch_ref)
        subprocess.run(("git", "apply", "--check", "-"), input=content, cwd=self.path, check=True)
        subprocess.run(("git", "apply", "-"), input=content, cwd=self.path, check=True)
        self.applied.append(content)
        if self.corrupt_after_apply:
            (self.path / "a.py").write_text("unexpected = True\n")
        return self.observation(request.call_id)

    def observation(self, call_id: UUID, reference: ArtifactRef | None = None) -> Observation:
        return Observation(
            call_id=call_id,
            sandbox_id=self.handle.sandbox_id,
            started_at=datetime.now(UTC),
            duration_ms=0,
            exit_code=0,
            policy_decision=PolicyDecision.ALLOWED,
            stdout_ref=reference,
        )

    def read_file(self, handle: SandboxHandle, request: ReadFileRequest) -> Observation:
        content = (self.path / request.relative_path).read_bytes()
        reference = self.store.persist(request.call_id, "read", content)
        if self.corrupt_read_artifact:
            (self.store.root.parent / reference.relative_path).write_bytes(b"tampered")
        return self.observation(request.call_id, reference).model_copy(
            update={"truncated": self.read_preview_truncated}
        )

    def collect_artifacts(self, handle: SandboxHandle) -> tuple[ArtifactRef, ...]:
        return tuple(
            ref
            for ref in self.store.collect()
            if not self.omit_read_registration or not ref.relative_path.endswith(".read.txt")
        )


def setup(
    tmp_path: Path, files: dict[str, str] | None = None, **limit_overrides: int
) -> tuple[PatchToolSession, LocalSandbox, Path]:
    base = tmp_path / "base"
    base.mkdir()
    for path, content in (files or {"a.py": "x = 1\ny = 1\n", "b.py": "z = 1\n"}).items():
        (base / path).write_text(content)
    current = tmp_path / "current"
    shutil.copytree(base, current)
    task = TaskSpec(
        task_id="current-patch",
        repository_path=str(base),
        base_commit="a" * 40,
        issue_title="Test current source",
        issue_body="Apply a correction.",
        test_commands=(CommandSpec(argv=("python", "-m", "pytest")),),
    )
    handle = SandboxHandle(
        sandbox_id="test-sandbox",
        task_id=task.task_id,
        repository_commit=task.base_commit,
        prepared_at=datetime.now(UTC),
    )
    root = tmp_path / "output" / "artifacts"
    source_store = SandboxArtifactStore(root)
    source_store.prepare(handle.sandbox_id, repository_source=base)
    output_store = SandboxArtifactStore(root)
    output_store.prepare("tool-output", repository_source=base)
    sandbox = LocalSandbox(current, source_store, handle)
    limits = {"max_bytes": 20_000, "max_files": 5, "max_hunks": 10, "max_changed_lines": 100}
    limits.update(limit_overrides)
    session = PatchToolSession(
        sandbox=sandbox,
        handle=handle,
        task=task,
        limits=PatchLimits(**limits),
        preview_bytes=5000,
        repository=ReadOnlyRepository(base),
        artifact_store=output_store,
    )
    return session, sandbox, root.parent


def apply(session: PatchToolSession, before: dict[str, str], after: dict[str, str]):
    diff = cumulative_diff(before, after)
    return session.apply(
        diff,
        call=ToolCall(
            call_id=uuid4(),
            tool_name="apply_patch",
            validated_arguments={"diff": diff},
            requested_at=datetime.now(UTC),
        ),
    )


def test_second_incremental_patch_exports_complete_base_diff_and_current_source(tmp_path: Path):
    session, sandbox, output = setup(tmp_path)
    first = apply(session, {"a.py": "x = 1\ny = 1\n"}, {"a.py": "x = 2\ny = 1\n"})
    second = apply(session, {"a.py": "x = 2\ny = 1\n"}, {"a.py": "x = 2\ny = 3\n"})
    assert first.status is second.status is PatchStatus.APPLIED
    assert second.patch_state.revision == 2
    assert second.patch_state.diff_ref and first.patch_state.diff_ref
    full = (output / second.patch_state.diff_ref.relative_path).read_bytes()
    assert b"-x = 1" in full and b"+x = 2" in full and b"+y = 3" in full
    assert full != sandbox.applied[-1]
    assert (output / first.patch_state.diff_ref.relative_path).read_bytes() == sandbox.applied[0]
    fresh = tmp_path / "fresh"
    shutil.copytree(tmp_path / "base", fresh)
    subprocess.run(("git", "apply", "-"), cwd=fresh, input=full, check=True)
    assert (fresh / "a.py").read_bytes() == (sandbox.path / "a.py").read_bytes()
    assert session.read_file("a.py", call_id=uuid4()).content == "x = 2\ny = 3\n"
    assert session.read_file("b.py", call_id=uuid4()).content == "z = 1\n"
    assert "-x = 1" in (session.inspect().preview or "")


def test_cumulative_patch_includes_distinct_second_file(tmp_path: Path):
    session, _, output = setup(tmp_path)
    apply(session, {"a.py": "x = 1\ny = 1\n"}, {"a.py": "x = 2\ny = 1\n"})
    second = apply(session, {"b.py": "z = 1\n"}, {"b.py": "z = 2\n"})
    assert second.status is PatchStatus.APPLIED
    assert second.patch_state.files_changed == ("a.py", "b.py")
    assert second.patch_state.diff_ref
    assert b"diff --git a/a.py" in (output / second.patch_state.diff_ref.relative_path).read_bytes()


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"max_files": 1}, PatchErrorCode.TOO_MANY_FILES),
        ({"max_changed_lines": 2}, PatchErrorCode.TOO_MANY_CHANGED_LINES),
        ({"max_bytes": 140}, PatchErrorCode.PATCH_TOO_LARGE),
    ],
)
def test_cumulative_limits_reject_before_second_mutation(tmp_path: Path, override, expected):
    session, sandbox, _ = setup(tmp_path, **override)
    first = apply(session, {"a.py": "x = 1\ny = 1\n"}, {"a.py": "x = 2\ny = 1\n"})
    assert first.status is PatchStatus.APPLIED
    second = apply(session, {"b.py": "z = 1\n"}, {"b.py": "z = 2\n"})
    assert second.status is PatchStatus.REJECTED and second.error_code is expected
    assert second.patch_state == first.patch_state
    assert len(sandbox.applied) == 1
    assert (sandbox.path / "b.py").read_text() == "z = 1\n"


def test_cumulative_hunk_limit_and_exact_context_conflict(tmp_path: Path):
    base = "".join(f"x{i} = 1\n" for i in range(40))
    once = base.replace("x0 = 1", "x0 = 2")
    twice = once.replace("x39 = 1", "x39 = 2")
    session, sandbox, _ = setup(tmp_path, {"a.py": base}, max_hunks=1)
    assert apply(session, {"a.py": base}, {"a.py": once}).status is PatchStatus.APPLIED
    rejected = apply(session, {"a.py": once}, {"a.py": twice})
    assert rejected.error_code is PatchErrorCode.TOO_MANY_HUNKS
    assert len(sandbox.applied) == 1
    conflict = apply(session, {"a.py": base}, {"a.py": base.replace("x0 = 1", "x0 = 3")})
    assert conflict.error_code is PatchErrorCode.APPLY_CONFLICT
    assert session.read_file("a.py", call_id=uuid4()).content == once


@pytest.mark.parametrize("kind", ["post_apply", "artifact"])
def test_unproven_sandbox_content_fails_closed(tmp_path: Path, kind: str):
    session, sandbox, _ = setup(tmp_path)
    if kind == "post_apply":
        sandbox.corrupt_after_apply = True
    else:
        sandbox.corrupt_read_artifact = True
    rejected = apply(session, {"a.py": "x = 1\ny = 1\n"}, {"a.py": "x = 2\ny = 1\n"})
    assert rejected.status is PatchStatus.REJECTED
    assert rejected.error_code is PatchErrorCode.SANDBOX_ERROR
    assert session.state.revision == 0
    with pytest.raises(RepositoryAccessError):
        session.read_file("a.py", call_id=uuid4())


def test_detects_source_drift_between_validation_and_correction(tmp_path: Path):
    session, sandbox, _ = setup(tmp_path)
    first = apply(session, {"a.py": "x = 1\ny = 1\n"}, {"a.py": "x = 2\ny = 1\n"})
    (sandbox.path / "a.py").write_text("mutated = True\n")
    rejected = apply(session, {"a.py": "x = 2\ny = 1\n"}, {"a.py": "x = 3\ny = 1\n"})
    assert rejected.error_code is PatchErrorCode.SANDBOX_ERROR
    assert rejected.patch_state == first.patch_state and len(sandbox.applied) == 1


def test_current_source_read_never_skips_partial_unicode_lines(tmp_path: Path):
    original = "a = 1\n" + "# 中文内容\n" * 20
    current = original.replace("a = 1", "a = 2")
    session, _, _ = setup(tmp_path, {"a.py": original})
    assert apply(session, {"a.py": original}, {"a.py": current}).status is PatchStatus.APPLIED
    assert session._workspace
    session._workspace.repository = ReadOnlyRepository(
        tmp_path / "base", policy=RepositoryAccessPolicy(max_read_bytes=35)
    )
    window = session.read_file("a.py", call_id=uuid4())
    assert window.truncated and window.content.endswith("\n")
    assert window.next_start_line == window.end_line + 1
    continued = session.read_file("a.py", start_line=window.next_start_line, call_id=uuid4())
    assert (window.content + continued.content) == "".join(
        current.splitlines(keepends=True)[: continued.end_line]
    )


def test_full_verified_artifact_allows_truncated_read_preview(tmp_path: Path):
    session, sandbox, _ = setup(tmp_path)
    sandbox.read_preview_truncated = True
    result = apply(session, {"a.py": "x = 1\ny = 1\n"}, {"a.py": "x = 2\ny = 1\n"})
    assert result.status is PatchStatus.APPLIED
    assert session.read_file("a.py", call_id=uuid4()).content == "x = 2\ny = 1\n"


def test_read_reference_must_be_registered_by_the_agent_sandbox(tmp_path: Path):
    session, sandbox, _ = setup(tmp_path)
    sandbox.omit_read_registration = True
    result = apply(session, {"a.py": "x = 1\ny = 1\n"}, {"a.py": "x = 2\ny = 1\n"})
    assert result.error_code is PatchErrorCode.SANDBOX_ERROR
    assert not sandbox.applied


def test_insert_and_delete_cumulative_patch_uses_correct_offsets(tmp_path: Path):
    before = "x = 1\ny = 1\n"
    inserted = "first = True\n" + before
    deleted = "first = True\ny = 1\n"
    session, _, output = setup(tmp_path)
    assert apply(session, {"a.py": before}, {"a.py": inserted}).status is PatchStatus.APPLIED
    result = apply(session, {"a.py": inserted}, {"a.py": deleted})
    assert result.status is PatchStatus.APPLIED and result.patch_state.diff_ref
    assert session.read_file("a.py", call_id=uuid4()).content == deleted
    full = (output / result.patch_state.diff_ref.relative_path).read_bytes()
    fresh = tmp_path / "fresh"
    shutil.copytree(tmp_path / "base", fresh)
    subprocess.run(("git", "apply", "-"), input=full, cwd=fresh, check=True)
    assert (fresh / "a.py").read_text() == deleted


def test_fully_reverting_first_candidate_is_rejected_before_mutation(tmp_path: Path):
    before = "x = 1\ny = 1\n"
    after = "x = 2\ny = 1\n"
    session, sandbox, _ = setup(tmp_path)
    first = apply(session, {"a.py": before}, {"a.py": after})
    rejected = apply(session, {"a.py": after}, {"a.py": before})
    assert rejected.status is PatchStatus.REJECTED
    assert rejected.patch_state == first.patch_state and len(sandbox.applied) == 1


def test_untracked_legacy_session_cannot_export_last_increment_as_complete(tmp_path: Path):
    session, sandbox, _ = setup(tmp_path)
    session._workspace = None
    first = apply(session, {"a.py": "x = 1\ny = 1\n"}, {"a.py": "x = 2\ny = 1\n"})
    assert first.status is PatchStatus.APPLIED
    second = apply(session, {"a.py": "x = 2\ny = 1\n"}, {"a.py": "x = 3\ny = 1\n"})
    assert second.error_code is PatchErrorCode.SANDBOX_ERROR
    assert second.patch_state == first.patch_state and len(sandbox.applied) == 1


@pytest.mark.parametrize(
    ("source", "hunks", "accepted"),
    [
        ("prefix\nanchor\nold\ntail\n", "@@ -1,3 +99,3 @@\n anchor\n-old\n+new\n tail\n", True),
        (
            "anchor\nold\ntail\ngap\nanchor\nold\ntail\n",
            "@@ -99,3 +99,3 @@\n anchor\n-old\n+new\n tail\n",
            False,
        ),
        ("prefix\nold\n", "@@ -99 +99 @@\n-old\n+new\n", False),
        ("old\n", "@@ -99,0 +99,1 @@\n+new\n", False),
        (
            "anchor\nold\ntail\ngap\nanchor\nold\ntail\n",
            "@@ -1,3 +1,3 @@\n anchor\n-old\n+new\n tail\n",
            True,
        ),
        (
            "A\nold1\nB\nold2\nC\n",
            "@@ -7,3 +7,3 @@\n A\n-old1\n+new1\n B\n@@ -10,3 +10,3 @@\n B\n-old2\n+new2\n C\n",
            False,
        ),
    ],
)
def test_pure_prediction_matches_shipped_sandbox_offset_rules(source, hunks, accepted):
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n" + hunks
    limits = PatchLimits(max_bytes=20_000, max_files=5, max_hunks=10, max_changed_lines=100)
    parsed = parse_unified_diff(diff, limits)
    namespace = {}
    definitions = APPLY_PATCH_HELPER.split("\nname = sys.argv[1]\n", maxsplit=1)[0]
    exec(compile(definitions, "trusted-sandbox-patch-definitions", "exec"), namespace)
    sandbox_hunks = namespace["parse_patch"](parsed.content, 5, 10, 100)[0][1]
    if accepted:
        predicted = apply_exact_diff(parsed.content, {"a.py": source})["a.py"].encode()
        assert predicted == namespace["apply_hunks"](source, sandbox_hunks)
    else:
        with pytest.raises(ExactDiffConflict):
            apply_exact_diff(parsed.content, {"a.py": source})
        with pytest.raises(namespace["PatchFailure"]):
            namespace["apply_hunks"](source, sandbox_hunks)
