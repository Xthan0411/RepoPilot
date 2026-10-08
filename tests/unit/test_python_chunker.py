"""AST boundaries, fallback, determinism, and safety tests for Python chunks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    PythonChunkErrorCode,
    PythonChunkMode,
    SymbolType,
)
from repopilot.retrieval import (
    PythonAstChunker,
    PythonChunkError,
    PythonChunkingResult,
    PythonChunkPolicy,
    PythonSourceChunkResult,
    RepositoryIngestError,
    chunk_python_repository,
    ingest_repository,
)
from repopilot.tools import ReadOnlyRepository

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "repositories" / "python_chunker"
REPOSITORY_ID = "fixture.python"
COMMIT = "b" * 40


def build_fixture_result() -> PythonChunkingResult:
    repository = ReadOnlyRepository(FIXTURE_ROOT)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )
    return chunk_python_repository(repository, manifest)


def build_result(
    root: Path,
    *,
    policy: PythonChunkPolicy | None = None,
) -> PythonChunkingResult:
    repository = ReadOnlyRepository(root)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )
    return chunk_python_repository(repository, manifest, policy=policy)


def assert_chunk_error(
    captured: pytest.ExceptionInfo[PythonChunkError],
    code: PythonChunkErrorCode,
) -> None:
    assert captured.value.code is code


def test_policy_defaults_settings_mapping_and_validation() -> None:
    assert PythonChunkPolicy() == PythonChunkPolicy(max_lines=200, overlap_lines=20)
    assert PythonChunkPolicy.from_settings(
        RepoPilotSettings(chunk_max_lines=120, chunk_overlap_lines=12)
    ) == PythonChunkPolicy(max_lines=120, overlap_lines=12)

    for values in (
        {"max_lines": 0},
        {"max_lines": 2_001},
        {"overlap_lines": -1},
        {"overlap_lines": 501},
        {"max_lines": 20, "overlap_lines": 20},
        {"max_lines": True},
        {"overlap_lines": False},
    ):
        with pytest.raises(ValidationError):
            PythonChunkPolicy(**values)


def test_fixture_sources_have_explicit_modes_and_exclude_non_python() -> None:
    result = build_fixture_result()

    assert tuple(
        (source.relative_path, source.mode, source.syntax_error_line) for source in result.sources
    ) == (
        ("broken.py", PythonChunkMode.SYNTAX_FALLBACK, 1),
        ("empty.py", PythonChunkMode.EMPTY, None),
        ("source.py", PythonChunkMode.AST, None),
    )
    assert "notes.md" not in {source.relative_path for source in result.sources}
    assert result.sources[1].chunks == ()


def test_fixture_ast_boundaries_qualified_names_and_decorators_are_stable() -> None:
    result = build_fixture_result()
    source = result.sources[2]

    assert tuple(
        (
            chunk.start_line,
            chunk.end_line,
            chunk.symbol_type,
            chunk.symbol_name,
            chunk.metadata["ast_kind"],
        )
        for chunk in source.chunks
    ) == (
        (1, 3, SymbolType.MODULE, "source", "module"),
        (5, 18, SymbolType.CLASS, "Service", "class"),
        (7, 8, SymbolType.CLASS, "Service.Nested", "class"),
        (10, 12, SymbolType.FUNCTION, "Service.build", "function"),
        (14, 18, SymbolType.FUNCTION, "Service.run", "async_function"),
        (15, 16, SymbolType.FUNCTION, "Service.run.inner", "async_function"),
        (21, 22, SymbolType.FUNCTION, "helper", "function"),
        (24, 24, SymbolType.MODULE, "source", "module"),
    )
    assert source.chunks[1].content.startswith("@decorator\nclass Service:")
    assert source.chunks[3].content.startswith("    @classmethod\n")
    assert source.chunks[0].chunk_id == (
        "chunk_1a7d1a686ccdc054864a32a51251feef07bfde77fe7c9740408c5e2b9da24ccc"
    )


def test_chunk_hashes_metadata_and_flattened_order_are_self_consistent() -> None:
    result = build_fixture_result()

    assert result.chunks == tuple(chunk for source in result.sources for chunk in source.chunks)
    assert len({chunk.chunk_id for chunk in result.chunks}) == len(result.chunks)
    for source in result.sources:
        for chunk in source.chunks:
            assert chunk.repository_id == result.repository_id
            assert chunk.commit == result.commit
            assert chunk.metadata["source_id"] == source.source_id
            assert chunk.content_hash == hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            assert chunk.metadata["part_index"] == 0
            assert chunk.metadata["part_count"] == 1


def test_syntax_error_falls_back_to_text_without_executing_source() -> None:
    result = build_fixture_result()
    broken = result.sources[0]

    assert broken.mode is PythonChunkMode.SYNTAX_FALLBACK
    assert len(broken.chunks) == 1
    assert broken.chunks[0].symbol_type is SymbolType.TEXT
    assert broken.chunks[0].symbol_name is None
    assert broken.chunks[0].metadata["chunk_mode"] == "syntax_fallback"
    assert broken.chunks[0].metadata["syntax_error_line"] == 1
    assert broken.chunks[0].content == "def broken(:\n    pass\n"


def test_valid_source_is_parsed_but_never_executed(tmp_path: Path) -> None:
    tmp_path.joinpath("danger.py").write_text(
        'raise RuntimeError("chunking must not execute this")\n',
        encoding="utf-8",
    )

    result = build_result(tmp_path)

    assert result.sources[0].mode is PythonChunkMode.AST
    assert result.chunks[0].content.startswith("raise RuntimeError")


def test_large_function_uses_deterministic_overlapping_line_windows(
    tmp_path: Path,
) -> None:
    lines = ["def long_function():\n"]
    lines.extend(f"    value_{index} = {index}\n" for index in range(1, 10))
    tmp_path.joinpath("long.py").write_text("".join(lines), encoding="utf-8")
    policy = PythonChunkPolicy(max_lines=4, overlap_lines=1)

    first = build_result(tmp_path, policy=policy)
    second = build_result(tmp_path, policy=policy)
    chunks = first.sources[0].chunks

    assert first == second
    assert tuple((chunk.start_line, chunk.end_line) for chunk in chunks) == (
        (1, 4),
        (4, 7),
        (7, 10),
    )
    assert tuple(chunk.metadata["part_index"] for chunk in chunks) == (0, 1, 2)
    assert all(chunk.metadata["part_count"] == 3 for chunk in chunks)
    assert chunks[0].content.splitlines()[-1] == chunks[1].content.splitlines()[0]
    assert chunks[1].content.splitlines()[-1] == chunks[2].content.splitlines()[0]


def test_syntax_fallback_uses_the_same_window_contract(tmp_path: Path) -> None:
    tmp_path.joinpath("broken.py").write_text(
        "def broken(:\n    one = 1\n    two = 2\n    three = 3\n"
        "    four = 4\n    five = 5\n    six = 6\n",
        encoding="utf-8",
    )

    result = build_result(
        tmp_path,
        policy=PythonChunkPolicy(max_lines=3, overlap_lines=1),
    )

    assert result.sources[0].mode is PythonChunkMode.SYNTAX_FALLBACK
    assert tuple((chunk.start_line, chunk.end_line) for chunk in result.sources[0].chunks) == (
        (1, 3),
        (3, 5),
        (5, 7),
    )


def test_comments_are_module_content_but_whitespace_is_empty(tmp_path: Path) -> None:
    tmp_path.joinpath("comments.py").write_text("# searchable comment\n", encoding="utf-8")
    tmp_path.joinpath("whitespace.py").write_text(" \n\t\n", encoding="utf-8")

    result = build_result(tmp_path)

    assert result.sources[0].mode is PythonChunkMode.AST
    assert result.sources[0].chunks[0].symbol_type is SymbolType.MODULE
    assert result.sources[0].chunks[0].content == "# searchable comment\n"
    assert result.sources[1].mode is PythonChunkMode.EMPTY
    assert result.sources[1].chunks == ()


def test_crlf_content_is_preserved_in_chunk_hash(tmp_path: Path) -> None:
    tmp_path.joinpath("windows.py").write_bytes(b"value = 1\r\nother = 2\r\n")

    result = build_result(tmp_path)
    chunk = result.chunks[0]

    assert chunk.content == "value = 1\r\nother = 2\r\n"
    assert chunk.content_hash == hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()


def test_long_qualified_name_is_bounded_with_a_stable_hash(tmp_path: Path) -> None:
    outer = "A" + ("a" * 199)
    middle = "B" + ("b" * 199)
    inner = "C" + ("c" * 199)
    tmp_path.joinpath("names.py").write_text(
        f"class {outer}:\n    class {middle}:\n        class {inner}:\n            pass\n",
        encoding="utf-8",
    )

    result = build_result(tmp_path)
    innermost = next(chunk for chunk in result.chunks if chunk.start_line == 3)

    assert innermost.symbol_name is not None
    assert len(innermost.symbol_name) == 512
    assert innermost.metadata["symbol_name_truncated"] is True
    assert len(str(innermost.metadata["symbol_name_sha256"])) == 64


def test_same_size_content_change_after_manifest_is_fatal(tmp_path: Path) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("value = 1\n", encoding="utf-8")
    repository = ReadOnlyRepository(tmp_path)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )
    source_path.write_text("value = 2\n", encoding="utf-8")

    with pytest.raises(PythonChunkError) as captured:
        chunk_python_repository(repository, manifest)

    assert_chunk_error(captured, PythonChunkErrorCode.SOURCE_CHANGED)
    assert "source.py" not in str(captured.value)


def test_missing_manifest_source_is_safe_unavailable_error(tmp_path: Path) -> None:
    source_path = tmp_path / "source.py"
    source_path.write_text("value = 1\n", encoding="utf-8")
    repository = ReadOnlyRepository(tmp_path)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )
    source_path.unlink()

    with pytest.raises(PythonChunkError) as captured:
        chunk_python_repository(repository, manifest)

    assert_chunk_error(captured, PythonChunkErrorCode.SOURCE_UNAVAILABLE)
    assert "source.py" not in str(captured.value)


def test_manifest_integrity_is_verified_before_any_source_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ReadOnlyRepository(FIXTURE_ROOT)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    ).model_copy(update={"content_sha256": "0" * 64})

    def forbidden_read(self: ReadOnlyRepository, path: str) -> object:
        raise AssertionError("source read must not occur for an invalid Manifest")

    monkeypatch.setattr(ReadOnlyRepository, "read_text_file", forbidden_read)

    with pytest.raises(RepositoryIngestError):
        chunk_python_repository(repository, manifest)


def test_result_models_reject_inconsistent_modes_and_duplicate_sources() -> None:
    result = build_fixture_result()
    ast_source = result.sources[2]

    with pytest.raises(ValidationError, match="empty Python"):
        PythonSourceChunkResult(
            source_id=ast_source.source_id,
            relative_path=ast_source.relative_path,
            mode=PythonChunkMode.EMPTY,
            chunks=ast_source.chunks,
        )
    with pytest.raises(ValidationError, match="unique sorted"):
        PythonChunkingResult(
            repository_id=result.repository_id,
            commit=result.commit,
            sources=(ast_source, ast_source),
        )
    with pytest.raises(ValidationError, match="deterministically sorted"):
        PythonSourceChunkResult(
            source_id=ast_source.source_id,
            relative_path=ast_source.relative_path,
            mode=PythonChunkMode.AST,
            chunks=tuple(reversed(ast_source.chunks)),
        )


def test_chunker_from_settings_uses_validated_runtime_limits() -> None:
    settings = RepoPilotSettings(chunk_max_lines=80, chunk_overlap_lines=8)
    chunker = PythonAstChunker.from_settings(ReadOnlyRepository(FIXTURE_ROOT), settings)

    assert chunker.policy == PythonChunkPolicy(max_lines=80, overlap_lines=8)
