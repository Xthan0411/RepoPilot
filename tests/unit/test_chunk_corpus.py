"""Combined corpus, incremental reuse, and hostile JSONL tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.enums import ChunkCorpusErrorCode
from repopilot.retrieval import (
    ChunkCorpus,
    ChunkCorpusError,
    PythonChunkPolicy,
    RepositoryChunkCorpusBuilder,
    ingest_repository,
    read_chunk_corpus,
    write_chunk_corpus,
)
from repopilot.schemas import CommandSpec, TaskSpec
from repopilot.tools import ReadOnlyRepository

REPOSITORY_ID = "fixture.mixed"
COMMIT = "d" * 40


def populate_mixed_repository(root: Path) -> None:
    root.joinpath("source.py").write_text(
        "def parse(value: str) -> str:\n    return value.strip()\n",
        encoding="utf-8",
    )
    root.joinpath("README.md").write_text(
        "# Parser\n\nUse `parse` for input.\n",
        encoding="utf-8",
    )
    root.joinpath("settings.toml").write_text("enabled = true\n", encoding="utf-8")
    root.joinpath("empty.txt").write_text("", encoding="utf-8")
    root.joinpath("ignored.csv").write_text("not,indexed\n", encoding="utf-8")


def task_spec(*, body: str = "Empty input should return an empty value.") -> TaskSpec:
    return TaskSpec(
        task_id="task-001",
        repository_path=".",
        base_commit=COMMIT,
        issue_title="Handle empty input",
        issue_body=body,
        test_commands=(CommandSpec(argv=("pytest", "-q")),),
    )


def build_corpus(
    root: Path,
    *,
    previous: ChunkCorpus | None = None,
    policy: PythonChunkPolicy | None = None,
    task: TaskSpec | None = None,
) -> tuple[ReadOnlyRepository, Any]:
    repository = ReadOnlyRepository(root)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )
    result = RepositoryChunkCorpusBuilder(repository, policy=policy).build(
        manifest,
        task=task,
        previous=previous,
    )
    return repository, result


def assert_corpus_error(
    captured: pytest.ExceptionInfo[ChunkCorpusError],
    code: ChunkCorpusErrorCode,
) -> None:
    assert captured.value.code is code


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(record, ensure_ascii=False)}\n" for record in records),
        encoding="utf-8",
    )


def test_builder_combines_python_text_issue_and_empty_source_snapshots(
    tmp_path: Path,
) -> None:
    populate_mixed_repository(tmp_path)

    _, result = build_corpus(tmp_path, task=task_spec())
    corpus = result.corpus

    assert result.stats.model_dump() == {
        "reused_sources": 0,
        "rebuilt_sources": 5,
        "removed_sources": 0,
    }
    assert tuple(source.relative_path for source in corpus.sources) == (
        "README.md",
        "_repopilot_issues/task-001.md",
        "empty.txt",
        "settings.toml",
        "source.py",
    )
    empty = next(source for source in corpus.sources if source.relative_path == "empty.txt")
    assert empty.chunk_ids == ()
    assert {chunk.language for chunk in corpus.chunks} == {
        "issue",
        "markdown",
        "python",
        "toml",
    }
    assert "ignored.csv" not in {source.relative_path for source in corpus.sources}


def test_unchanged_sources_reuse_without_reading_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    populate_mixed_repository(tmp_path)
    repository, first = build_corpus(tmp_path, task=task_spec())
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )

    def forbidden_read(self: ReadOnlyRepository, path: str) -> object:
        raise AssertionError("unchanged source must not be re-read")

    monkeypatch.setattr(ReadOnlyRepository, "read_text_file", forbidden_read)
    second = RepositoryChunkCorpusBuilder(repository).build(
        manifest,
        task=task_spec(),
        previous=first.corpus,
    )

    assert second.corpus == first.corpus
    assert second.stats.model_dump() == {
        "reused_sources": 5,
        "rebuilt_sources": 0,
        "removed_sources": 0,
    }


def test_policy_change_rebuilds_every_source_including_empty_and_issue(
    tmp_path: Path,
) -> None:
    populate_mixed_repository(tmp_path)
    _, first = build_corpus(tmp_path, task=task_spec())

    _, second = build_corpus(
        tmp_path,
        task=task_spec(),
        previous=first.corpus,
        policy=PythonChunkPolicy(max_lines=3, overlap_lines=1),
    )

    assert second.stats.model_dump() == {
        "reused_sources": 0,
        "rebuilt_sources": 5,
        "removed_sources": 0,
    }
    assert {source.build_fingerprint for source in first.corpus.sources}.isdisjoint(
        source.build_fingerprint for source in second.corpus.sources
    )


def test_builder_from_settings_uses_shared_window_configuration(tmp_path: Path) -> None:
    populate_mixed_repository(tmp_path)
    repository = ReadOnlyRepository(tmp_path)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )
    builder = RepositoryChunkCorpusBuilder.from_settings(
        repository,
        RepoPilotSettings(chunk_max_lines=40, chunk_overlap_lines=4),
    )

    result = builder.build(manifest)

    assert builder.policy == PythonChunkPolicy(max_lines=40, overlap_lines=4)
    assert all(source.build_fingerprint for source in result.corpus.sources)


def test_only_changed_source_is_rebuilt(tmp_path: Path) -> None:
    populate_mixed_repository(tmp_path)
    _, first = build_corpus(tmp_path, task=task_spec())
    tmp_path.joinpath("README.md").write_text(
        "# Parser\n\nUse `parse` for values.\n",
        encoding="utf-8",
    )

    _, second = build_corpus(tmp_path, task=task_spec(), previous=first.corpus)

    assert second.stats.model_dump() == {
        "reused_sources": 4,
        "rebuilt_sources": 1,
        "removed_sources": 0,
    }
    first_by_path = {source.relative_path: source for source in first.corpus.sources}
    second_by_path = {source.relative_path: source for source in second.corpus.sources}
    assert first_by_path["README.md"].source_id == second_by_path["README.md"].source_id
    assert (
        first_by_path["README.md"].source_content_hash
        != second_by_path["README.md"].source_content_hash
    )


def test_added_and_removed_sources_update_incremental_stats(tmp_path: Path) -> None:
    populate_mixed_repository(tmp_path)
    _, first = build_corpus(tmp_path)
    tmp_path.joinpath("settings.toml").unlink()
    tmp_path.joinpath("notes.txt").write_text("new notes\n", encoding="utf-8")

    _, second = build_corpus(tmp_path, previous=first.corpus)

    assert second.stats.model_dump() == {
        "reused_sources": 3,
        "rebuilt_sources": 1,
        "removed_sources": 1,
    }
    assert "settings.toml" not in {source.relative_path for source in second.corpus.sources}
    assert "notes.txt" in {source.relative_path for source in second.corpus.sources}


def test_issue_change_rebuilds_only_synthetic_issue(tmp_path: Path) -> None:
    populate_mixed_repository(tmp_path)
    _, first = build_corpus(tmp_path, task=task_spec(body="First behavior."))

    _, second = build_corpus(
        tmp_path,
        task=task_spec(body="Changed behavior."),
        previous=first.corpus,
    )

    assert second.stats.model_dump() == {
        "reused_sources": 4,
        "rebuilt_sources": 1,
        "removed_sources": 0,
    }


def test_previous_or_task_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    populate_mixed_repository(tmp_path)
    repository, first = build_corpus(tmp_path)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )

    with pytest.raises(ChunkCorpusError) as previous_error:
        RepositoryChunkCorpusBuilder(repository).build(
            manifest,
            previous=first.corpus.model_copy(update={"repository_id": "other.repo"}),
        )
    assert_corpus_error(previous_error, ChunkCorpusErrorCode.MIXED_CORPUS)

    mismatched_task = task_spec().model_copy(update={"base_commit": "e" * 40})
    with pytest.raises(ChunkCorpusError) as task_error:
        RepositoryChunkCorpusBuilder(repository).build(
            manifest,
            task=mismatched_task,
        )
    assert_corpus_error(task_error, ChunkCorpusErrorCode.MIXED_CORPUS)


def test_synthetic_issue_path_collision_is_rejected(tmp_path: Path) -> None:
    populate_mixed_repository(tmp_path)
    issue_directory = tmp_path / "_repopilot_issues"
    issue_directory.mkdir()
    issue_directory.joinpath("task-001.md").write_text("repository file\n", encoding="utf-8")
    repository = ReadOnlyRepository(tmp_path)
    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )

    with pytest.raises(ChunkCorpusError) as captured:
        RepositoryChunkCorpusBuilder(repository).build(manifest, task=task_spec())

    assert_corpus_error(captured, ChunkCorpusErrorCode.MIXED_CORPUS)


def test_jsonl_round_trip_is_byte_deterministic_and_keeps_header_first(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    populate_mixed_repository(repository_root)
    _, result = build_corpus(repository_root, task=task_spec())
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    write_chunk_corpus(first_path, result.corpus)
    write_chunk_corpus(second_path, result.corpus)
    records = load_jsonl(first_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert records[0]["record_type"] == "header"
    assert all(record["schema_version"] == 1 for record in records)
    assert all(record["record_type"] == "chunk" for record in records[1:])
    assert read_chunk_corpus(first_path) == result.corpus


def test_empty_corpus_round_trip_retains_repository_identity(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    _, result = build_corpus(repository_root)
    path = tmp_path / "empty.jsonl"

    write_chunk_corpus(path, result.corpus)

    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert read_chunk_corpus(path) == result.corpus


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing_version", ChunkCorpusErrorCode.INVALID_DOCUMENT),
        ("unsupported_version", ChunkCorpusErrorCode.UNSUPPORTED_VERSION),
        ("duplicate_chunk", ChunkCorpusErrorCode.DUPLICATE_CHUNK),
        ("unsorted_chunks", ChunkCorpusErrorCode.UNSORTED_CORPUS),
        ("mixed_commit", ChunkCorpusErrorCode.MIXED_CORPUS),
        ("invalid_hash", ChunkCorpusErrorCode.INVALID_DOCUMENT),
        ("unknown_chunk", ChunkCorpusErrorCode.INVALID_DOCUMENT),
    ],
)
def test_jsonl_reader_rejects_typed_hostile_mutations(
    tmp_path: Path,
    mutation: str,
    expected_code: ChunkCorpusErrorCode,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    populate_mixed_repository(repository_root)
    _, result = build_corpus(repository_root, task=task_spec())
    path = tmp_path / "corpus.jsonl"
    write_chunk_corpus(path, result.corpus)
    records = load_jsonl(path)

    if mutation == "missing_version":
        records[0].pop("schema_version")
    elif mutation == "unsupported_version":
        records[0]["schema_version"] = 2
    elif mutation == "duplicate_chunk":
        records.append(records[1])
    elif mutation == "unsorted_chunks":
        records[1], records[2] = records[2], records[1]
    elif mutation == "mixed_commit":
        records[1]["chunk"]["commit"] = "e" * 40
    elif mutation == "invalid_hash":
        records[1]["chunk"]["content_hash"] = "f" * 64
    elif mutation == "unknown_chunk":
        records[0]["sources"][0]["chunk_ids"] = ["chunk_" + ("f" * 64)]
    write_jsonl(path, records)

    with pytest.raises(ChunkCorpusError) as captured:
        read_chunk_corpus(path)

    assert_corpus_error(captured, expected_code)


@pytest.mark.parametrize("payload", [b"", b"\n", b"not-json\n", b"[]\n", b"\xff\xfe"])
def test_jsonl_reader_rejects_invalid_documents(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(payload)

    with pytest.raises(ChunkCorpusError) as captured:
        read_chunk_corpus(path)

    assert_corpus_error(captured, ChunkCorpusErrorCode.INVALID_DOCUMENT)
    decoded = payload.decode("utf-8", errors="ignore")
    if decoded:
        assert decoded not in str(captured.value)


def test_corpus_artifact_io_errors_are_safe(tmp_path: Path) -> None:
    with pytest.raises(ChunkCorpusError) as read_error:
        read_chunk_corpus(tmp_path / "missing.jsonl")
    assert_corpus_error(read_error, ChunkCorpusErrorCode.IO_ERROR)

    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    _, result = build_corpus(repository_root)
    target = tmp_path / "corpus.jsonl"
    target.mkdir()
    with pytest.raises(ChunkCorpusError) as write_error:
        write_chunk_corpus(target, result.corpus)
    assert_corpus_error(write_error, ChunkCorpusErrorCode.IO_ERROR)
    assert not tmp_path.joinpath(".corpus.jsonl.tmp").exists()


def test_corpus_models_reject_duplicate_source_paths(tmp_path: Path) -> None:
    populate_mixed_repository(tmp_path)
    _, result = build_corpus(tmp_path)
    source = result.corpus.sources[0]

    with pytest.raises(ValidationError, match="unique sorted"):
        ChunkCorpus(
            repository_id=result.corpus.repository_id,
            commit=result.corpus.commit,
            sources=(source, source),
            chunks=result.corpus.chunks,
        )
