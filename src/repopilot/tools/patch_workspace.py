"""Exact, bounded current-source bookkeeping for incremental candidate repairs.

Only clean repository source and the Agent sandbox are read. No evaluator checkout,
test patch, shell command, or model-provided executable participates in this layer.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import stat
from typing import Protocol
from uuid import UUID, uuid5

from repopilot.enums import PolicyDecision, RepositoryErrorCode
from repopilot.sandbox.artifacts import SandboxArtifactStore
from repopilot.sandbox.base import ReadFileRequest, SandboxHandle
from repopilot.schemas import ArtifactRef, Observation
from repopilot.tools.repository import (
    FileReadResult,
    ReadOnlyRepository,
    RepositoryAccessError,
)

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: [^\n]*)?\n$")


class CurrentSourceError(RuntimeError):
    """Current candidate source could not be proven; no raw source is exposed."""


class ExactDiffConflict(CurrentSourceError):
    """Canonical hunks do not exactly match the current verified source."""


class SourceSandbox(Protocol):
    def read_file(self, handle: SandboxHandle, request: ReadFileRequest) -> Observation: ...

    def collect_artifacts(self, handle: SandboxHandle) -> tuple[ArtifactRef, ...]: ...


def apply_exact_diff(content: str, sources: dict[str, str]) -> dict[str, str]:
    """Match the sandbox's exact-offset or unique full-context rules, without fuzz."""

    result = dict(sources)
    lines = content.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        path = lines[index].split(" ", 3)[2].removeprefix("a/")
        if path not in sources:
            raise ExactDiffConflict
        old = sources[path].splitlines(keepends=True)
        output: list[str] = []
        cursor = 0
        index += 1
        if lines[index].startswith("index "):
            index += 1
        index += 2  # Structural validation already checked ---/+++.
        while index < len(lines) and not lines[index].startswith("diff --git "):
            header = _HUNK.fullmatch(lines[index])
            if header is None:
                raise ExactDiffConflict
            old_start, old_count, _new_start, new_count = (
                int(header[1]),
                int(header[2] or "1"),
                int(header[3]),
                int(header[4] or "1"),
            )
            position = old_start - 1 if old_count else old_start
            index += 1
            expected: list[str] = []
            replacement: list[str] = []
            has_context = False
            while index < len(lines) and not lines[index].startswith(("@@ ", "diff --git ")):
                line = lines[index]
                if line[0] in {" ", "-"}:
                    expected.append(line[1:])
                if line[0] in {" ", "+"}:
                    replacement.append(line[1:])
                has_context = has_context or line[0] == " "
                index += 1
            if len(expected) != old_count or len(replacement) != new_count:
                raise ExactDiffConflict
            exact_position = (
                cursor <= position <= len(old) and old[position : position + old_count] == expected
            )
            if not exact_position:
                if old_count == 0 or not has_context:
                    raise ExactDiffConflict
                matches: list[int] = []
                for candidate in range(len(old) - old_count + 1):
                    if old[candidate : candidate + old_count] == expected:
                        matches.append(candidate)
                        if len(matches) > 1:
                            break
                if len(matches) != 1:
                    raise ExactDiffConflict
                position = matches[0]
            if position < cursor or position > len(old):
                raise ExactDiffConflict
            output.extend(old[cursor:position])
            output.extend(replacement)
            cursor = position + old_count
        output.extend(old[cursor:])
        result[path] = "".join(output)
    return result


def cumulative_diff(base: dict[str, str], current: dict[str, str]) -> str:
    """Produce one deterministic complete base-to-current diff, never a last-diff alias."""

    sections: list[str] = []
    for path in sorted(current):
        original = base[path]
        updated = current[path]
        if original == updated:
            continue
        # The existing canonical patch protocol excludes no-newline markers and CRLF.
        if any(
            value and (not value.endswith("\n") or "\r" in value) for value in (original, updated)
        ):
            raise CurrentSourceError
        body = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=3,
            )
        )
        sections.append(f"diff --git a/{path} b/{path}\n" + body)
    return "".join(sections)


class CurrentPatchWorkspace:
    """Immutable base snapshots and verified current contents of candidate paths."""

    def __init__(
        self,
        repository: ReadOnlyRepository,
        store: SandboxArtifactStore,
        sandbox: SourceSandbox,
        handle: SandboxHandle,
    ) -> None:
        self.repository = repository
        self.store = store
        self.sandbox = sandbox
        self.handle = handle
        self.base: dict[str, str] = {}
        self.current: dict[str, str] = {}
        self.poisoned = False

    def candidate(self, diff: str, paths: tuple[str, ...], call_id: UUID) -> dict[str, str]:
        if self.poisoned:
            raise CurrentSourceError
        sources = dict(self.current)
        for path in paths:
            if path not in self.base:
                source = self.repository.read_text_file(path).content
                self.base[path] = source
            if path not in sources:
                sources[path] = self.base[path]
        # Test code may have mutated the workspace since the previous validation.
        self.verify(sources, call_id, "before")
        expected = apply_exact_diff(diff, sources)
        if any(
            len(text.encode("utf-8")) > self.repository.policy.max_file_bytes
            for text in expected.values()
        ):
            raise CurrentSourceError
        return expected

    def verify(self, expected: dict[str, str], call_id: UUID, phase: str) -> None:
        for path, text in expected.items():
            request_id = uuid5(call_id, f"current-source:{phase}:{path}")
            observation = self.sandbox.read_file(
                self.handle,
                ReadFileRequest(
                    call_id=request_id,
                    relative_path=path,
                ),
            )
            if (
                observation.call_id != request_id
                or observation.sandbox_id != self.handle.sandbox_id
                or observation.exit_code != 0
                or observation.error_type is not None
                or observation.policy_decision is not PolicyDecision.ALLOWED
                or observation.stdout_ref is None
            ):
                raise CurrentSourceError
            reference = observation.stdout_ref
            if reference not in self.sandbox.collect_artifacts(self.handle):
                raise CurrentSourceError
            # Preview truncation is safe only when the independently verified full
            # artifact exactly equals the bounded expected file.
            if self._read_artifact(reference) != text.encode("utf-8"):
                raise CurrentSourceError

    def _read_artifact(self, reference: ArtifactRef) -> bytes:
        root = self.store.root
        path = root.parent / reference.relative_path
        if (
            not path.is_relative_to(root)
            or path.is_symlink()
            or any(p.is_symlink() for p in path.parents)
            or reference.size_bytes > self.repository.policy.max_file_bytes
        ):
            raise CurrentSourceError
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise CurrentSourceError
                data = stream.read(self.repository.policy.max_file_bytes + 1)
        except OSError:
            raise CurrentSourceError from None
        if (
            len(data) != reference.size_bytes
            or hashlib.sha256(data).hexdigest() != reference.sha256
        ):
            raise CurrentSourceError
        return data

    def read_file(
        self, path: str, *, start_line: int, end_line: int | None, call_id: UUID
    ) -> FileReadResult:
        if self.poisoned:
            raise RepositoryAccessError(RepositoryErrorCode.IO_ERROR)
        if path not in self.current:
            return self.repository.read_file(path, start_line=start_line, end_line=end_line)
        self.repository._validate_line_window(start_line, end_line)
        try:
            self.verify({path: self.current[path]}, call_id, "read")
        except CurrentSourceError:
            self.poisoned = True
            raise RepositoryAccessError(RepositoryErrorCode.IO_ERROR) from None
        text = self.current[path]
        lines = text.splitlines(keepends=True)
        requested_end = min(len(lines), end_line or len(lines))
        selected_end = min(requested_end, start_line + self.repository.policy.max_read_lines - 1)
        selected: list[str] = []
        used = 0
        for line in lines[start_line - 1 : selected_end]:
            size = len(line.encode("utf-8"))
            if used + size > self.repository.policy.max_read_bytes:
                if not selected:
                    raise RepositoryAccessError(RepositoryErrorCode.FILE_TOO_LARGE)
                break
            selected.append(line)
            used += size
        end = start_line + len(selected) - 1 if selected else None
        truncated = end is not None and end < requested_end
        return FileReadResult(
            relative_path=path,
            start_line=start_line,
            end_line=end,
            content="".join(selected),
            content_bytes=used,
            file_size_bytes=len(text.encode("utf-8")),
            truncated=truncated,
            next_start_line=end + 1 if truncated and end is not None else None,
        )
