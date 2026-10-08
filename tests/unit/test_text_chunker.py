"""Markdown, text, configuration, and Agent-visible Issue chunk tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.enums import SymbolType, TextChunkErrorCode, TextSourceKind
from repopilot.retrieval import (
    PythonChunkPolicy,
    TextChunker,
    TextChunkError,
    TextChunkingResult,
    TextSourceChunkResult,
    ingest_repository,
    issue_content,
    issue_relative_path,
    issue_source_id_for,
    text_kind_for_path,
)
from repopilot.schemas import CommandSpec, TaskSpec
from repopilot.tools import ReadOnlyRepository

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "repositories" / "text_chunker"
REPOSITORY_ID = "fixture.text"
COMMIT = "c" * 40


def build_fixture_result(
    *,
    policy: PythonChunkPolicy | None = None,
) -> TextChunkingResult:
    repository = ReadOnlyRepository(FIXTURE_ROOT)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )
    return TextChunker(repository, policy=policy).chunk_manifest(manifest)


def task_spec(*, issue_body: str = "The parser fails for empty input.") -> TaskSpec:
    return TaskSpec(
        task_id="task-001",
        repository_path=".",
        base_commit=COMMIT,
        issue_title="Handle empty input",
        issue_body=issue_body,
        test_commands=(CommandSpec(argv=("pytest", "-q")),),
    )


def assert_text_error(
    captured: pytest.ExceptionInfo[TextChunkError],
    code: TextChunkErrorCode,
) -> None:
    assert captured.value.code is code


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("README.md", (TextSourceKind.MARKDOWN, "markdown")),
        ("guide.rst", (TextSourceKind.TEXT, "text")),
        ("notes.txt", (TextSourceKind.TEXT, "text")),
        ("pyproject.toml", (TextSourceKind.CONFIG, "toml")),
        ("settings.yaml", (TextSourceKind.CONFIG, "yaml")),
        ("settings.yml", (TextSourceKind.CONFIG, "yml")),
        ("data.json", (TextSourceKind.CONFIG, "json")),
        ("app.ini", (TextSourceKind.CONFIG, "ini")),
        ("tool.cfg", (TextSourceKind.CONFIG, "cfg")),
        ("README.MD", None),
        ("data.csv", None),
        ("source.py", None),
    ],
)
def test_text_suffix_ownership_is_exact(
    path: str,
    expected: tuple[TextSourceKind, str] | None,
) -> None:
    assert text_kind_for_path(path) == expected


def test_fixture_sources_are_sorted_typed_and_exclude_unowned_csv() -> None:
    result = build_fixture_result()

    assert tuple(
        (source.relative_path, source.source_kind, source.empty) for source in result.sources
    ) == (
        ("README.md", TextSourceKind.MARKDOWN, False),
        ("config.yaml", TextSourceKind.CONFIG, False),
        ("empty.txt", TextSourceKind.TEXT, True),
        ("notes.txt", TextSourceKind.TEXT, False),
    )
    assert "ignored.csv" not in {source.relative_path for source in result.sources}
    assert result.sources[2].chunks == ()


def test_markdown_headings_build_qualified_sections_and_ignore_fenced_hashes() -> None:
    markdown = build_fixture_result().sources[0]

    assert tuple(
        (
            chunk.start_line,
            chunk.end_line,
            chunk.symbol_name,
            chunk.metadata["strategy"],
        )
        for chunk in markdown.chunks
    ) == (
        (1, 1, None, "markdown_preamble"),
        (3, 4, "Guide", "markdown_heading"),
        (6, 11, "Guide > Install", "markdown_heading"),
        (13, 14, "Guide > Usage", "markdown_heading"),
    )
    assert "# Not a heading" in markdown.chunks[2].content
    assert all(chunk.symbol_type is SymbolType.TEXT for chunk in markdown.chunks)
    assert markdown.chunks[0].chunk_id == (
        "chunk_ef919c6871bf798d9eefa773a48d5817f090cbfa249250fbca2acf7411d1d592"
    )


def test_plain_text_and_configuration_use_overlapping_line_windows() -> None:
    result = build_fixture_result(policy=PythonChunkPolicy(max_lines=3, overlap_lines=1))
    config = result.sources[1]
    notes = result.sources[3]

    assert tuple((chunk.start_line, chunk.end_line) for chunk in config.chunks) == (
        (1, 3),
        (3, 4),
    )
    assert tuple((chunk.start_line, chunk.end_line) for chunk in notes.chunks) == (
        (1, 3),
        (3, 5),
    )
    assert all(chunk.language == "yaml" for chunk in config.chunks)
    assert all(chunk.language == "text" for chunk in notes.chunks)
    assert notes.chunks[0].content.splitlines()[-1] == notes.chunks[1].content.splitlines()[0]


def test_issue_source_contains_only_agent_visible_title_and_body() -> None:
    repository = ReadOnlyRepository(FIXTURE_ROOT)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )
    task = task_spec()

    result = TextChunker(repository).chunk_issue(manifest, task)
    rendered = result.chunks[0].content

    assert result.source_kind is TextSourceKind.ISSUE
    assert result.relative_path == "_repopilot_issues/task-001.md"
    assert result.relative_path == issue_relative_path(task.task_id)
    assert result.source_id == issue_source_id_for(REPOSITORY_ID, COMMIT, task.task_id)
    assert rendered == issue_content(task)
    assert task.issue_title in rendered
    assert task.issue_body in rendered
    assert "expected_patch" not in rendered
    assert "gold_files" not in rendered
    assert "pytest" not in rendered
    assert result.source_content_hash == hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def test_issue_content_change_updates_hash_but_preserves_source_identity() -> None:
    repository = ReadOnlyRepository(FIXTURE_ROOT)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )
    chunker = TextChunker(repository)

    first = chunker.chunk_issue(manifest, task_spec(issue_body="First behavior."))
    second = chunker.chunk_issue(manifest, task_spec(issue_body="Other behavior."))

    assert first.source_id == second.source_id
    assert first.source_content_hash != second.source_content_hash
    assert first.chunks[0].chunk_id != second.chunks[0].chunk_id


def test_markdown_without_headings_and_crlf_are_preserved(tmp_path: Path) -> None:
    tmp_path.joinpath("README.md").write_bytes(b"first\r\nsecond\r\n")
    repository = ReadOnlyRepository(tmp_path)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )

    result = TextChunker(repository).chunk_manifest(manifest)
    chunk = result.chunks[0]

    assert chunk.metadata["strategy"] == "line_window"
    assert chunk.content == "first\r\nsecond\r\n"
    assert chunk.content_hash == hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()


def test_long_markdown_heading_name_is_bounded(tmp_path: Path) -> None:
    heading = "H" * 600
    tmp_path.joinpath("README.md").write_text(f"# {heading}\nbody\n", encoding="utf-8")
    repository = ReadOnlyRepository(tmp_path)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )

    chunk = TextChunker(repository).chunk_manifest(manifest).chunks[0]

    assert chunk.symbol_name is not None
    assert len(chunk.symbol_name) == 512
    assert chunk.metadata["symbol_name_truncated"] is True
    assert len(str(chunk.metadata["symbol_name_sha256"])) == 64


def test_invalid_configuration_is_indexed_as_text_without_parsing(tmp_path: Path) -> None:
    tmp_path.joinpath("broken.yaml").write_text("key: [not closed\n", encoding="utf-8")
    repository = ReadOnlyRepository(tmp_path)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )

    result = TextChunker.from_settings(
        repository,
        RepoPilotSettings(chunk_max_lines=50, chunk_overlap_lines=5),
    ).chunk_manifest(manifest)

    assert result.sources[0].source_kind is TextSourceKind.CONFIG
    assert result.chunks[0].content == "key: [not closed\n"
    assert result.chunks[0].metadata["max_lines"] == 50
    assert result.chunks[0].metadata["overlap_lines"] == 5


def test_same_size_text_change_after_manifest_is_fatal(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("first\n", encoding="utf-8")
    repository = ReadOnlyRepository(tmp_path)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )
    path.write_text("other\n", encoding="utf-8")

    with pytest.raises(TextChunkError) as captured:
        TextChunker(repository).chunk_manifest(manifest)

    assert_text_error(captured, TextChunkErrorCode.SOURCE_CHANGED)
    assert "notes.txt" not in str(captured.value)


def test_missing_text_source_is_safe_unavailable_error(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("first\n", encoding="utf-8")
    repository = ReadOnlyRepository(tmp_path)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )
    path.unlink()

    with pytest.raises(TextChunkError) as captured:
        TextChunker(repository).chunk_manifest(manifest)

    assert_text_error(captured, TextChunkErrorCode.SOURCE_UNAVAILABLE)


def test_text_result_models_reject_inconsistent_empty_and_source_metadata() -> None:
    result = build_fixture_result()
    source = result.sources[0]

    with pytest.raises(ValidationError, match="empty must"):
        TextSourceChunkResult(
            source_id=source.source_id,
            relative_path=source.relative_path,
            source_kind=source.source_kind,
            source_content_hash=source.source_content_hash,
            empty=True,
            chunks=source.chunks,
        )
    with pytest.raises(ValidationError, match="unique sorted"):
        TextChunkingResult(
            repository_id=result.repository_id,
            commit=result.commit,
            sources=(source, source),
        )
