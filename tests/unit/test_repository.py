"""Security and boundary tests for contained read-only repository access."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.enums import ErrorType, RepositoryEntryType, RepositoryErrorCode
from repopilot.tools.repository import (
    DirectoryListing,
    FileReadResult,
    ReadOnlyRepository,
    RepositoryAccessError,
    RepositoryAccessPolicy,
    RepositoryEntry,
    RepositoryTextFile,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "repositories" / "read_only"


def assert_repository_error(
    captured: pytest.ExceptionInfo[RepositoryAccessError],
    code: RepositoryErrorCode,
    error_type: ErrorType,
) -> None:
    assert captured.value.code is code
    assert captured.value.error_type is error_type


def test_policy_defaults_and_settings_mapping() -> None:
    settings = RepoPilotSettings(
        repository_max_file_bytes=8_000,
        repository_max_read_bytes=4_000,
        repository_max_read_lines=123,
        repository_max_list_entries=456,
    )
    policy = RepositoryAccessPolicy.from_settings(settings)

    assert policy.max_file_bytes == 8_000
    assert policy.max_read_bytes == 4_000
    assert policy.max_read_lines == 123
    assert policy.max_list_entries == 456
    assert policy.denied_names == (".env", ".git", ".hg", ".svn")


@pytest.mark.parametrize(
    "denied_names",
    [
        (".git", ".git"),
        ("",),
        (".",),
        ("..",),
        ("nested/name",),
        ("nested\\name",),
    ],
)
def test_policy_rejects_invalid_denied_names(denied_names: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        RepositoryAccessPolicy(denied_names=denied_names)


def test_policy_rejects_read_bytes_larger_than_file_limit() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        RepositoryAccessPolicy(max_file_bytes=10, max_read_bytes=11)


def test_repository_rejects_missing_or_non_directory_root(tmp_path: Path) -> None:
    with pytest.raises(RepositoryAccessError) as missing:
        ReadOnlyRepository(tmp_path / "missing")
    assert_repository_error(
        missing,
        RepositoryErrorCode.NOT_FOUND,
        ErrorType.TOOL_EXECUTION_ERROR,
    )

    regular_file = tmp_path / "file.txt"
    regular_file.write_text("content", encoding="utf-8")
    with pytest.raises(RepositoryAccessError) as not_directory:
        ReadOnlyRepository(regular_file)
    assert_repository_error(
        not_directory,
        RepositoryErrorCode.NOT_DIRECTORY,
        ErrorType.TOOL_EXECUTION_ERROR,
    )


def test_repository_maps_root_removed_after_initialization(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    repository = ReadOnlyRepository(root)
    root.rmdir()

    with pytest.raises(RepositoryAccessError) as captured:
        repository.list_directory()

    assert captured.value.code is RepositoryErrorCode.NOT_FOUND


def test_listing_is_deterministic_typed_and_round_trippable() -> None:
    repository = ReadOnlyRepository(FIXTURE_ROOT)

    first = repository.list_directory()
    second = repository.list_directory()

    assert first == second
    assert tuple(entry.relative_path for entry in first.entries) == (
        "README.md",
        "docs",
        "pkg",
        "docs/说明.md",
        "pkg/__init__.py",
        "pkg/parser.py",
        "pkg/service.py",
    )
    assert tuple(entry.entry_type for entry in first.entries) == (
        RepositoryEntryType.FILE,
        RepositoryEntryType.DIRECTORY,
        RepositoryEntryType.DIRECTORY,
        RepositoryEntryType.FILE,
        RepositoryEntryType.FILE,
        RepositoryEntryType.FILE,
        RepositoryEntryType.FILE,
    )
    assert first.truncated is False
    assert DirectoryListing.model_validate_json(first.model_dump_json()) == first


def test_non_recursive_listing_stays_at_requested_directory() -> None:
    repository = ReadOnlyRepository(FIXTURE_ROOT)

    root = repository.list_directory(recursive=False)
    package = repository.list_directory("pkg", recursive=False)

    assert tuple(entry.relative_path for entry in root.entries) == (
        "README.md",
        "docs",
        "pkg",
    )
    assert tuple(entry.relative_path for entry in package.entries) == (
        "pkg/__init__.py",
        "pkg/parser.py",
        "pkg/service.py",
    )


def test_reads_exact_line_window_and_unicode_content() -> None:
    repository = ReadOnlyRepository(FIXTURE_ROOT)

    parser = repository.read_file("pkg/parser.py", start_line=4, end_line=5)
    unicode_document = repository.read_file("docs/说明.md")

    assert parser.content == (
        'def parse(raw: str) -> dict[str, str]:\n    """Return stripped key/value pairs."""\n'
    )
    assert parser.start_line == 4
    assert parser.end_line == 5
    assert parser.truncated is False
    assert parser.next_start_line is None
    assert parser.content_bytes == len(parser.content.encode("utf-8"))
    assert FileReadResult.model_validate_json(parser.model_dump_json()) == parser
    assert "UTF-8 路径" in unicode_document.content


def test_indexing_read_returns_complete_text_beyond_agent_output_limits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("first\nsecond\nthird\n", encoding="utf-8")
    repository = ReadOnlyRepository(
        tmp_path,
        policy=RepositoryAccessPolicy(
            max_file_bytes=100,
            max_read_bytes=5,
            max_read_lines=1,
        ),
    )

    agent_read = repository.read_file("source.py")
    indexing_read = repository.read_text_file("source.py")

    assert agent_read.truncated is True
    assert indexing_read == RepositoryTextFile(
        relative_path="source.py",
        content="first\nsecond\nthird\n",
        size_bytes=19,
    )


@pytest.mark.parametrize(
    ("contents", "expected_code"),
    [
        (b"text\x00binary", RepositoryErrorCode.BINARY_FILE),
        (b"\xff\xfe", RepositoryErrorCode.DECODE_ERROR),
    ],
)
def test_indexing_read_preserves_binary_and_decode_boundaries(
    tmp_path: Path,
    contents: bytes,
    expected_code: RepositoryErrorCode,
) -> None:
    tmp_path.joinpath("invalid.dat").write_bytes(contents)
    repository = ReadOnlyRepository(tmp_path)

    with pytest.raises(RepositoryAccessError) as captured:
        repository.read_text_file("invalid.dat")

    assert captured.value.code is expected_code


def test_read_beyond_end_returns_empty_non_truncated_result() -> None:
    repository = ReadOnlyRepository(FIXTURE_ROOT)

    result = repository.read_file("README.md", start_line=100)

    assert result.content == ""
    assert result.end_line is None
    assert result.truncated is False
    assert result.next_start_line is None


def test_line_limit_truncates_with_next_line(tmp_path: Path) -> None:
    source = tmp_path / "many-lines.txt"
    source.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    repository = ReadOnlyRepository(
        tmp_path,
        policy=RepositoryAccessPolicy(
            max_file_bytes=100,
            max_read_bytes=100,
            max_read_lines=2,
        ),
    )

    result = repository.read_file("many-lines.txt", start_line=2)
    explicit_range = repository.read_file("many-lines.txt", start_line=1, end_line=2)

    assert result.content == "two\nthree\n"
    assert result.end_line == 3
    assert result.truncated is True
    assert result.next_start_line == 4
    assert explicit_range.content == "one\ntwo\n"
    assert explicit_range.truncated is False


@pytest.mark.parametrize(
    ("max_read_bytes", "expected_content", "expected_end", "expected_next"),
    [
        (5, "one\nt", 2, 3),
        (4, "one\n", 1, 2),
    ],
)
def test_byte_limit_preserves_utf8_boundary_and_reports_next_line(
    tmp_path: Path,
    max_read_bytes: int,
    expected_content: str,
    expected_end: int,
    expected_next: int,
) -> None:
    source = tmp_path / "bytes.txt"
    source.write_text("one\ntwo\nthree\n", encoding="utf-8")
    repository = ReadOnlyRepository(
        tmp_path,
        policy=RepositoryAccessPolicy(
            max_file_bytes=100,
            max_read_bytes=max_read_bytes,
            max_read_lines=10,
        ),
    )

    result = repository.read_file("bytes.txt")

    assert result.content == expected_content
    assert result.content_bytes <= max_read_bytes
    assert result.end_line == expected_end
    assert result.truncated is True
    assert result.next_start_line == expected_next


def test_multibyte_content_is_not_returned_as_invalid_utf8(tmp_path: Path) -> None:
    source = tmp_path / "unicode.txt"
    source.write_text("你\nnext\n", encoding="utf-8")
    repository = ReadOnlyRepository(
        tmp_path,
        policy=RepositoryAccessPolicy(
            max_file_bytes=100,
            max_read_bytes=1,
            max_read_lines=10,
        ),
    )

    result = repository.read_file("unicode.txt")

    assert result.content == ""
    assert result.content_bytes == 0
    assert result.truncated is True
    assert result.next_start_line == 2


def test_file_size_limit_is_checked_before_content_processing(tmp_path: Path) -> None:
    source = tmp_path / "large.txt"
    source.write_text("x" * 11, encoding="utf-8")
    repository = ReadOnlyRepository(
        tmp_path,
        policy=RepositoryAccessPolicy(
            max_file_bytes=10,
            max_read_bytes=10,
        ),
    )

    with pytest.raises(RepositoryAccessError) as captured:
        repository.read_file("large.txt")

    assert_repository_error(
        captured,
        RepositoryErrorCode.FILE_TOO_LARGE,
        ErrorType.TOOL_EXECUTION_ERROR,
    )


@pytest.mark.parametrize(
    ("contents", "expected_code"),
    [
        (b"text\x00binary", RepositoryErrorCode.BINARY_FILE),
        (b"\xff\xfe", RepositoryErrorCode.DECODE_ERROR),
    ],
)
def test_binary_and_invalid_utf8_are_rejected(
    tmp_path: Path,
    contents: bytes,
    expected_code: RepositoryErrorCode,
) -> None:
    source = tmp_path / "invalid.dat"
    source.write_bytes(contents)
    repository = ReadOnlyRepository(tmp_path)

    with pytest.raises(RepositoryAccessError) as captured:
        repository.read_file("invalid.dat")

    assert_repository_error(
        captured,
        expected_code,
        ErrorType.TOOL_EXECUTION_ERROR,
    )


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/etc/passwd",
        "../outside",
        "pkg/../README.md",
        "pkg\\parser.py",
        "pkg//parser.py",
        "./README.md",
        "README.md/",
        "bad\x00path",
    ],
)
def test_noncanonical_or_escaping_paths_are_rejected_without_echo(path: str) -> None:
    repository = ReadOnlyRepository(FIXTURE_ROOT)

    with pytest.raises(RepositoryAccessError) as captured:
        repository.read_file(path)

    assert_repository_error(
        captured,
        RepositoryErrorCode.INVALID_PATH,
        ErrorType.PATH_VIOLATION,
    )
    if path:
        assert path not in str(captured.value)


@pytest.mark.parametrize("path", [".env", ".git/config", "pkg/.svn/entries"])
def test_sensitive_or_vcs_paths_are_denied(path: str) -> None:
    repository = ReadOnlyRepository(FIXTURE_ROOT)

    with pytest.raises(RepositoryAccessError) as captured:
        repository.read_file(path)

    assert_repository_error(
        captured,
        RepositoryErrorCode.DENIED_PATH,
        ErrorType.PATH_VIOLATION,
    )


def test_missing_and_wrong_entry_types_have_distinct_errors() -> None:
    repository = ReadOnlyRepository(FIXTURE_ROOT)

    with pytest.raises(RepositoryAccessError) as missing:
        repository.read_file("missing.py")
    assert missing.value.code is RepositoryErrorCode.NOT_FOUND

    with pytest.raises(RepositoryAccessError) as directory_as_file:
        repository.read_file("pkg")
    assert directory_as_file.value.code is RepositoryErrorCode.NOT_FILE

    with pytest.raises(RepositoryAccessError) as file_as_directory:
        repository.list_directory("README.md")
    assert file_as_directory.value.code is RepositoryErrorCode.NOT_DIRECTORY


def test_internal_and_external_symlinks_are_listed_but_never_followed(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    repository_root.joinpath("inside.txt").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    repository_root.joinpath("internal-link").symlink_to("inside.txt")
    repository_root.joinpath("external-link").symlink_to(outside)
    repository_root.joinpath("directory-link").symlink_to(
        FIXTURE_ROOT,
        target_is_directory=True,
    )
    repository = ReadOnlyRepository(repository_root)

    listing = repository.list_directory()
    symlinks = tuple(
        entry.relative_path
        for entry in listing.entries
        if entry.entry_type is RepositoryEntryType.SYMLINK
    )
    assert symlinks == ("directory-link", "external-link", "internal-link")

    for path in ("internal-link", "external-link", "directory-link/README.md"):
        with pytest.raises(RepositoryAccessError) as captured:
            repository.read_file(path)
        assert_repository_error(
            captured,
            RepositoryErrorCode.SYMLINK_FORBIDDEN,
            ErrorType.PATH_VIOLATION,
        )

    with pytest.raises(RepositoryAccessError) as linked_directory:
        repository.list_directory("directory-link")
    assert linked_directory.value.code is RepositoryErrorCode.SYMLINK_FORBIDDEN


def test_listing_skips_denied_names_and_truncates_deterministically(
    tmp_path: Path,
) -> None:
    for filename in ("a.txt", "b.txt", "c.txt"):
        tmp_path.joinpath(filename).write_text(filename, encoding="utf-8")
    tmp_path.joinpath(".env").write_text("secret", encoding="utf-8")
    tmp_path.joinpath(".git").mkdir()
    tmp_path.joinpath(".git", "config").write_text("private", encoding="utf-8")
    repository = ReadOnlyRepository(
        tmp_path,
        policy=RepositoryAccessPolicy(max_list_entries=2),
    )

    result = repository.list_directory()

    assert tuple(entry.relative_path for entry in result.entries) == ("a.txt", "b.txt")
    assert result.truncated is True


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is unavailable")
def test_non_regular_file_is_not_read_or_listed_as_a_file(tmp_path: Path) -> None:
    fifo = tmp_path / "named-pipe"
    os.mkfifo(fifo)
    repository = ReadOnlyRepository(tmp_path)

    with pytest.raises(RepositoryAccessError) as captured:
        repository.read_file("named-pipe")

    assert captured.value.code is RepositoryErrorCode.NOT_FILE
    assert repository.list_directory().entries == ()


@pytest.mark.parametrize(
    ("start_line", "end_line"),
    [(0, None), (-1, None), (True, None), (2, 1), (1, 0), (1, False)],
)
def test_invalid_line_windows_are_rejected(
    start_line: int,
    end_line: int | None,
) -> None:
    repository = ReadOnlyRepository(FIXTURE_ROOT)

    with pytest.raises(RepositoryAccessError) as captured:
        repository.read_file(
            "README.md",
            start_line=start_line,
            end_line=end_line,
        )

    assert captured.value.code is RepositoryErrorCode.INVALID_PATH


def test_repository_access_does_not_modify_fixture_files() -> None:
    paths = tuple(path for path in FIXTURE_ROOT.rglob("*") if path.is_file())

    def fingerprint(path: Path) -> tuple[int, int, str]:
        metadata = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return metadata.st_size, metadata.st_mtime_ns, digest

    before = {path: fingerprint(path) for path in paths}
    repository = ReadOnlyRepository(FIXTURE_ROOT)
    repository.list_directory()
    repository.read_file("README.md")
    repository.read_file("pkg/parser.py", start_line=4, end_line=8)
    after = {path: fingerprint(path) for path in paths}

    assert after == before
    assert not hasattr(repository, "write_file")
    assert not hasattr(repository, "delete_file")
    assert not hasattr(repository, "apply_patch")


def test_repository_entry_requires_size_only_for_regular_files() -> None:
    with pytest.raises(ValidationError, match="require size_bytes"):
        RepositoryEntry(
            relative_path="file.py",
            entry_type=RepositoryEntryType.FILE,
        )
    with pytest.raises(ValidationError, match="cannot have size_bytes"):
        RepositoryEntry(
            relative_path="pkg",
            entry_type=RepositoryEntryType.DIRECTORY,
            size_bytes=0,
        )


def test_repository_text_file_size_must_match_utf8_content() -> None:
    with pytest.raises(ValidationError, match="must match"):
        RepositoryTextFile(
            relative_path="unicode.txt",
            content="你",
            size_bytes=1,
        )
