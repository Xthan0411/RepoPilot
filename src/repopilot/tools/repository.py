"""Contained, deterministic, read-only access to one repository tree."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Annotated, BinaryIO, Self

from pydantic import AfterValidator, Field, field_validator, model_validator

from repopilot.config import RepoPilotSettings
from repopilot.enums import ErrorType, RepositoryEntryType, RepositoryErrorCode
from repopilot.schemas import RelativePath, StrictModel

DEFAULT_DENIED_NAMES = (".env", ".git", ".hg", ".svn")
PATH_ERROR_CODES = frozenset(
    {
        RepositoryErrorCode.INVALID_PATH,
        RepositoryErrorCode.OUTSIDE_ROOT,
        RepositoryErrorCode.SYMLINK_FORBIDDEN,
        RepositoryErrorCode.DENIED_PATH,
    }
)
ERROR_MESSAGES = {
    RepositoryErrorCode.INVALID_PATH: "repository path is not a canonical relative path",
    RepositoryErrorCode.OUTSIDE_ROOT: "repository path resolves outside the repository root",
    RepositoryErrorCode.SYMLINK_FORBIDDEN: "repository path contains a forbidden symbolic link",
    RepositoryErrorCode.DENIED_PATH: "repository path is denied by read policy",
    RepositoryErrorCode.NOT_FOUND: "repository path does not exist",
    RepositoryErrorCode.NOT_FILE: "repository path is not a regular file",
    RepositoryErrorCode.NOT_DIRECTORY: "repository path is not a directory",
    RepositoryErrorCode.BINARY_FILE: "repository file appears to be binary",
    RepositoryErrorCode.FILE_TOO_LARGE: "repository file exceeds the configured size limit",
    RepositoryErrorCode.DECODE_ERROR: "repository file is not valid UTF-8 text",
    RepositoryErrorCode.IO_ERROR: "repository operation failed",
}


def _validate_repository_location(value: str) -> str:
    """Validate a result path, allowing dot only as the repository root."""

    if value == ".":
        return value
    if not value or "\x00" in value or "\\" in value:
        raise ValueError("path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("path must stay beneath the repository root")
    if path.as_posix() != value:
        raise ValueError("path must be canonical")
    return value


RepositoryLocation = Annotated[str, AfterValidator(_validate_repository_location)]


class RepositoryAccessPolicy(StrictModel):
    """Immutable limits and denied names for repository reads."""

    max_file_bytes: int = Field(default=1_000_000, gt=0, le=10_000_000)
    max_read_bytes: int = Field(default=65_536, gt=0, le=1_000_000)
    max_read_lines: int = Field(default=400, gt=0, le=5_000)
    max_list_entries: int = Field(default=2_000, gt=0, le=20_000)
    denied_names: tuple[str, ...] = DEFAULT_DENIED_NAMES

    @field_validator("denied_names")
    @classmethod
    def validate_denied_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("denied_names must be unique")
        for name in value:
            if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
                raise ValueError("denied_names must contain simple path components")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def validate_byte_limits(self) -> Self:
        if self.max_read_bytes > self.max_file_bytes:
            raise ValueError("max_read_bytes must not exceed max_file_bytes")
        return self

    @classmethod
    def from_settings(cls, settings: RepoPilotSettings) -> Self:
        """Copy repository limits from validated runtime settings."""

        return cls(
            max_file_bytes=settings.repository_max_file_bytes,
            max_read_bytes=settings.repository_max_read_bytes,
            max_read_lines=settings.repository_max_read_lines,
            max_list_entries=settings.repository_max_list_entries,
        )


class RepositoryEntry(StrictModel):
    """One safe directory-listing entry."""

    relative_path: RelativePath
    entry_type: RepositoryEntryType
    size_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_size(self) -> Self:
        if self.entry_type is RepositoryEntryType.FILE and self.size_bytes is None:
            raise ValueError("file entries require size_bytes")
        if self.entry_type is not RepositoryEntryType.FILE and self.size_bytes is not None:
            raise ValueError("non-file entries cannot have size_bytes")
        return self


class DirectoryListing(StrictModel):
    """Bounded, deterministic repository directory output."""

    directory: RepositoryLocation
    entries: tuple[RepositoryEntry, ...]
    truncated: bool


class FileReadResult(StrictModel):
    """Bounded UTF-8 text returned from one regular repository file."""

    relative_path: RelativePath
    start_line: int = Field(gt=0)
    end_line: int | None = Field(default=None, gt=0)
    content: str
    file_size_bytes: int = Field(ge=0)
    content_bytes: int = Field(ge=0)
    truncated: bool
    next_start_line: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_read_window(self) -> Self:
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        if self.content_bytes != len(self.content.encode("utf-8")):
            raise ValueError("content_bytes must match UTF-8 encoded content")
        if self.truncated != (self.next_start_line is not None):
            raise ValueError("truncated reads require exactly one next_start_line")
        if self.next_start_line is not None and (
            self.end_line is None or self.next_start_line != self.end_line + 1
        ):
            raise ValueError("next_start_line must follow end_line")
        return self


class RepositoryTextFile(StrictModel):
    """One complete bounded UTF-8 source file for deterministic indexing."""

    relative_path: RelativePath
    content: str
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_size(self) -> Self:
        if self.size_bytes != len(self.content.encode("utf-8")):
            raise ValueError("size_bytes must match UTF-8 encoded content")
        return self


class RepositoryAccessError(RuntimeError):
    """Safe typed failure for repository path and read operations."""

    def __init__(self, code: RepositoryErrorCode) -> None:
        super().__init__(ERROR_MESSAGES[code])
        self.code = code
        self.error_type = (
            ErrorType.PATH_VIOLATION if code in PATH_ERROR_CODES else ErrorType.TOOL_EXECUTION_ERROR
        )


class ReadOnlyRepository:
    """List and read one canonical repository root without write capabilities."""

    def __init__(
        self,
        root: Path,
        *,
        policy: RepositoryAccessPolicy | None = None,
    ) -> None:
        self._policy = policy or RepositoryAccessPolicy()
        self._root = self._validate_root(root)

    @classmethod
    def from_settings(
        cls,
        root: Path,
        settings: RepoPilotSettings,
    ) -> Self:
        return cls(root, policy=RepositoryAccessPolicy.from_settings(settings))

    @property
    def root(self) -> Path:
        """Return the canonical trusted root."""

        return self._root

    @property
    def policy(self) -> RepositoryAccessPolicy:
        return self._policy

    def list_directory(
        self,
        path: str = ".",
        *,
        recursive: bool = True,
    ) -> DirectoryListing:
        """List entries in deterministic traversal order without following links."""

        normalized, absolute, _ = self._resolve(path, require_directory=True)
        entries: list[RepositoryEntry] = []
        for entry in self._walk(absolute, normalized, recursive=recursive):
            if len(entries) == self._policy.max_list_entries:
                return DirectoryListing(
                    directory=normalized,
                    entries=tuple(entries),
                    truncated=True,
                )
            entries.append(entry)
        return DirectoryListing(
            directory=normalized,
            entries=tuple(entries),
            truncated=False,
        )

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> FileReadResult:
        """Read one bounded UTF-8 line window from a regular file."""

        self._validate_line_window(start_line, end_line)
        normalized, _, parts = self._resolve(path, require_file=True)
        data = self._read_regular_file(parts)
        if b"\x00" in data:
            raise RepositoryAccessError(RepositoryErrorCode.BINARY_FILE)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise RepositoryAccessError(RepositoryErrorCode.DECODE_ERROR) from None
        return self._select_lines(
            normalized,
            text,
            file_size_bytes=len(data),
            start_line=start_line,
            end_line=end_line,
        )

    def read_text_file(self, path: str) -> RepositoryTextFile:
        """Read a complete UTF-8 source file within the configured file-size limit.

        Unlike the Agent-facing line reader, this internal indexing operation is not
        truncated by output byte or line limits.
        """

        normalized, _, parts = self._resolve(path, require_file=True)
        data = self._read_regular_file(parts)
        if b"\x00" in data:
            raise RepositoryAccessError(RepositoryErrorCode.BINARY_FILE)
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            raise RepositoryAccessError(RepositoryErrorCode.DECODE_ERROR) from None
        return RepositoryTextFile(
            relative_path=normalized,
            content=content,
            size_bytes=len(data),
        )

    def _walk(
        self,
        absolute: Path,
        normalized: str,
        *,
        recursive: bool,
    ) -> Iterator[RepositoryEntry]:
        stack: list[tuple[Path, str]] = [(absolute, normalized)]
        while stack:
            directory, prefix = stack.pop()
            try:
                with os.scandir(directory) as iterator:
                    children = sorted(iterator, key=lambda entry: entry.name)
            except OSError:
                raise RepositoryAccessError(RepositoryErrorCode.IO_ERROR) from None
            child_directories: list[tuple[Path, str]] = []
            for child in children:
                if child.name in self._policy.denied_names:
                    continue
                relative = child.name if prefix == "." else f"{prefix}/{child.name}"
                try:
                    metadata = child.stat(follow_symlinks=False)
                except OSError:
                    raise RepositoryAccessError(RepositoryErrorCode.IO_ERROR) from None
                if stat.S_ISLNK(metadata.st_mode):
                    yield RepositoryEntry(
                        relative_path=relative,
                        entry_type=RepositoryEntryType.SYMLINK,
                    )
                elif stat.S_ISDIR(metadata.st_mode):
                    yield RepositoryEntry(
                        relative_path=relative,
                        entry_type=RepositoryEntryType.DIRECTORY,
                    )
                    if recursive:
                        child_directories.append((Path(child.path), relative))
                elif stat.S_ISREG(metadata.st_mode):
                    yield RepositoryEntry(
                        relative_path=relative,
                        entry_type=RepositoryEntryType.FILE,
                        size_bytes=metadata.st_size,
                    )
            stack.extend(reversed(child_directories))

    def _resolve(
        self,
        path: str,
        *,
        require_file: bool = False,
        require_directory: bool = False,
    ) -> tuple[str, Path, tuple[str, ...]]:
        normalized, parts = self._parse_path(path)
        candidate = self._root
        try:
            metadata = os.stat(self._root)
        except FileNotFoundError:
            raise RepositoryAccessError(RepositoryErrorCode.NOT_FOUND) from None
        except OSError:
            raise RepositoryAccessError(RepositoryErrorCode.IO_ERROR) from None
        for part in parts:
            candidate = candidate / part
            try:
                metadata = os.lstat(candidate)
            except FileNotFoundError:
                raise RepositoryAccessError(RepositoryErrorCode.NOT_FOUND) from None
            except OSError:
                raise RepositoryAccessError(RepositoryErrorCode.IO_ERROR) from None
            if stat.S_ISLNK(metadata.st_mode):
                raise RepositoryAccessError(RepositoryErrorCode.SYMLINK_FORBIDDEN)
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            raise RepositoryAccessError(RepositoryErrorCode.NOT_FOUND) from None
        except OSError:
            raise RepositoryAccessError(RepositoryErrorCode.IO_ERROR) from None
        if not resolved.is_relative_to(self._root):
            raise RepositoryAccessError(RepositoryErrorCode.OUTSIDE_ROOT)
        if require_file and not stat.S_ISREG(metadata.st_mode):
            raise RepositoryAccessError(RepositoryErrorCode.NOT_FILE)
        if require_directory and not stat.S_ISDIR(metadata.st_mode):
            raise RepositoryAccessError(RepositoryErrorCode.NOT_DIRECTORY)
        return normalized, resolved, parts

    def _parse_path(self, value: str) -> tuple[str, tuple[str, ...]]:
        if not isinstance(value, str):
            raise RepositoryAccessError(RepositoryErrorCode.INVALID_PATH)
        if value == ".":
            return value, ()
        if not value or "\x00" in value or "\\" in value:
            raise RepositoryAccessError(RepositoryErrorCode.INVALID_PATH)
        parsed = PurePosixPath(value)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise RepositoryAccessError(RepositoryErrorCode.INVALID_PATH)
        if parsed.as_posix() != value:
            raise RepositoryAccessError(RepositoryErrorCode.INVALID_PATH)
        if any(part in self._policy.denied_names for part in parsed.parts):
            raise RepositoryAccessError(RepositoryErrorCode.DENIED_PATH)
        return value, parsed.parts

    def _read_regular_file(self, parts: tuple[str, ...]) -> bytes:
        try:
            stream = self._open_no_follow(parts)
            with stream:
                metadata = os.fstat(stream.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    raise RepositoryAccessError(RepositoryErrorCode.NOT_FILE)
                if metadata.st_size > self._policy.max_file_bytes:
                    raise RepositoryAccessError(RepositoryErrorCode.FILE_TOO_LARGE)
                data = stream.read(self._policy.max_file_bytes + 1)
        except RepositoryAccessError:
            raise
        except OSError:
            raise RepositoryAccessError(RepositoryErrorCode.IO_ERROR) from None
        if len(data) > self._policy.max_file_bytes:
            raise RepositoryAccessError(RepositoryErrorCode.FILE_TOO_LARGE)
        return data

    def _open_no_follow(self, parts: tuple[str, ...]) -> BinaryIO:
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | no_follow
        file_flags = os.O_RDONLY | no_follow
        directory_fd = os.open(self._root, directory_flags)
        try:
            for part in parts[:-1]:
                next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            file_fd = os.open(parts[-1], file_flags, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
        return os.fdopen(file_fd, "rb")

    def _select_lines(
        self,
        relative_path: str,
        text: str,
        *,
        file_size_bytes: int,
        start_line: int,
        end_line: int | None,
    ) -> FileReadResult:
        lines = text.splitlines(keepends=True)
        if start_line > len(lines):
            return FileReadResult(
                relative_path=relative_path,
                start_line=start_line,
                end_line=None,
                content="",
                file_size_bytes=file_size_bytes,
                content_bytes=0,
                truncated=False,
                next_start_line=None,
            )

        maximum_end = start_line + self._policy.max_read_lines - 1
        requested_end = len(lines) if end_line is None else min(end_line, len(lines))
        selected_end = min(requested_end, maximum_end)
        candidate_lines = lines[start_line - 1 : selected_end]
        content, returned_line_count, byte_truncated = self._limit_content_bytes(candidate_lines)
        returned_end = start_line + returned_line_count - 1
        line_truncated = requested_end > maximum_end
        truncated = line_truncated or byte_truncated
        next_start_line = returned_end + 1 if truncated else None
        return FileReadResult(
            relative_path=relative_path,
            start_line=start_line,
            end_line=returned_end,
            content=content,
            file_size_bytes=file_size_bytes,
            content_bytes=len(content.encode("utf-8")),
            truncated=truncated,
            next_start_line=next_start_line,
        )

    def _limit_content_bytes(self, lines: list[str]) -> tuple[str, int, bool]:
        selected: list[str] = []
        used_bytes = 0
        for line in lines:
            encoded = line.encode("utf-8")
            remaining = self._policy.max_read_bytes - used_bytes
            if remaining == 0:
                return "".join(selected), len(selected), True
            if len(encoded) <= remaining:
                selected.append(line)
                used_bytes += len(encoded)
                continue
            prefix = encoded[:remaining].decode("utf-8", errors="ignore")
            selected.append(prefix)
            return "".join(selected), len(selected), True
        return "".join(selected), len(selected), False

    @staticmethod
    def _validate_line_window(start_line: int, end_line: int | None) -> None:
        if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line <= 0:
            raise RepositoryAccessError(RepositoryErrorCode.INVALID_PATH)
        if end_line is not None:
            if isinstance(end_line, bool) or not isinstance(end_line, int) or end_line <= 0:
                raise RepositoryAccessError(RepositoryErrorCode.INVALID_PATH)
            if end_line < start_line:
                raise RepositoryAccessError(RepositoryErrorCode.INVALID_PATH)

    @staticmethod
    def _validate_root(root: Path) -> Path:
        try:
            resolved = root.resolve(strict=True)
        except FileNotFoundError:
            raise RepositoryAccessError(RepositoryErrorCode.NOT_FOUND) from None
        except OSError:
            raise RepositoryAccessError(RepositoryErrorCode.IO_ERROR) from None
        if not resolved.is_dir():
            raise RepositoryAccessError(RepositoryErrorCode.NOT_DIRECTORY)
        return resolved
