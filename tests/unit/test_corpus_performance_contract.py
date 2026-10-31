"""Operation-count and exact-output guards for linear corpus preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from repopilot.retrieval import chunker, corpus, ingest_repository, text_chunker
from repopilot.retrieval.manifest import (
    RepositoryIngestError,
    RepositoryManifest,
    verify_repository_manifest_integrity,
)
from repopilot.schemas import CommandSpec, TaskSpec
from repopilot.tools.repository import ReadOnlyRepository

COMMIT = "a" * 40


def _inputs(root: Path, count: int = 4) -> tuple[ReadOnlyRepository, RepositoryManifest]:
    for index in range(count):
        extension = "py" if index % 2 == 0 else "md"
        content = "def f():\n    return 1\n" if extension == "py" else "# Header\n\nBody.\n"
        root.joinpath(f"source_{index:03}.{extension}").write_text(content, encoding="utf-8")
    root.joinpath("broken.py").write_text("def broken(:\n    return 1\n", encoding="utf-8")
    root.joinpath("empty.txt").write_text("", encoding="utf-8")
    repository = ReadOnlyRepository(root)
    return repository, ingest_repository(repository, repository_id="perf.fixture", commit=COMMIT)


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="perf-task",
        repository_path=".",
        base_commit=COMMIT,
        issue_title="Fix behavior",
        issue_body="The result should be two.",
        test_commands=(CommandSpec(argv=("pytest", "-q")),),
    )


class _LegacyBuilder(corpus.RepositoryChunkCorpusBuilder):
    """Retain the old public, per-source validation path as a differential oracle."""

    def _repository_candidates(self, manifest: RepositoryManifest) -> list[corpus._SourceCandidate]:
        candidates = super()._repository_candidates(manifest)
        sources = {source.source_id: source for source in manifest.files}
        for candidate in candidates:
            source = sources[candidate.source_id]
            method = (
                self._python.chunk_source
                if source.relative_path.endswith(".py")
                else self._text.chunk_source
            )
            candidate.build = corpus._source_builder(method, manifest, source)
        return candidates


@pytest.mark.parametrize("include_issue", [False, True])
def test_builder_matches_legacy_outputs_and_incremental_stats(
    tmp_path: Path,
    include_issue: bool,
) -> None:
    repository, manifest = _inputs(tmp_path)
    task = _task() if include_issue else None
    builder = corpus.RepositoryChunkCorpusBuilder(repository)
    legacy = _LegacyBuilder(repository)
    first = builder.build(manifest, task=task)
    baseline = legacy.build(manifest, task=task)
    assert first.model_dump(mode="json") == baseline.model_dump(mode="json")
    assert builder.build(manifest, task=task, previous=first.corpus).model_dump() == (
        legacy.build(manifest, task=task, previous=baseline.corpus).model_dump()
    )

    tmp_path.joinpath("source_000.py").write_text("answer = 42\n", encoding="utf-8")
    tmp_path.joinpath("source_001.md").unlink()
    tmp_path.joinpath("new.toml").write_text("enabled = true\n", encoding="utf-8")
    changed = ingest_repository(repository, repository_id="perf.fixture", commit=COMMIT)
    result = builder.build(changed, task=task, previous=first.corpus)
    reference = legacy.build(changed, task=task, previous=baseline.corpus)
    assert result.model_dump(mode="json") == reference.model_dump(mode="json")
    assert result.stats.rebuilt_sources == 2
    assert result.stats.removed_sources == 1


@pytest.mark.parametrize("count", [2, 24])
@pytest.mark.parametrize("include_issue", [False, True])
def test_full_manifest_checks_are_constant_per_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    include_issue: bool,
) -> None:
    repository, manifest = _inputs(tmp_path, count)
    calls = 0

    def counted(value: RepositoryManifest) -> None:
        nonlocal calls
        calls += 1
        verify_repository_manifest_integrity(value)

    for module in (corpus, chunker, text_chunker):
        monkeypatch.setattr(module, "verify_repository_manifest_integrity", counted)
    corpus.RepositoryChunkCorpusBuilder(repository).build(
        manifest,
        task=_task() if include_issue else None,
    )
    assert calls == (2 if include_issue else 1)


@pytest.mark.parametrize("entry", ["builder", "python", "text"])
def test_public_boundaries_still_reject_tampered_manifest(tmp_path: Path, entry: str) -> None:
    repository, manifest = _inputs(tmp_path)
    tampered = manifest.model_copy(update={"repository_id": "tampered"})
    with pytest.raises(RepositoryIngestError):
        if entry == "builder":
            corpus.RepositoryChunkCorpusBuilder(repository).build(tampered)
        elif entry == "python":
            chunker.PythonAstChunker(repository).chunk_source(tampered, manifest.files[0])
        else:
            source = next(
                source for source in manifest.files if source.relative_path.endswith(".md")
            )
            text_chunker.TextChunker(repository).chunk_source(tampered, source)


@pytest.mark.parametrize("name", ["source_000.py", "source_001.md"])
def test_verified_builder_still_detects_changed_source(tmp_path: Path, name: str) -> None:
    repository, manifest = _inputs(tmp_path)
    tmp_path.joinpath(name).write_text("changed content\n", encoding="utf-8")
    with pytest.raises((chunker.PythonChunkError, text_chunker.TextChunkError)) as captured:
        corpus.RepositoryChunkCorpusBuilder(repository).build(manifest)
    assert isinstance(captured.value, (chunker.PythonChunkError, text_chunker.TextChunkError))
    assert captured.value.code.value == "source_changed"


def test_source_membership_validation_visits_chunks_linearly(tmp_path: Path) -> None:
    repository, manifest = _inputs(tmp_path, 48)
    original = corpus.RepositoryChunkCorpusBuilder(repository).build(manifest).corpus
    visits = 0

    class CountedMetadata(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            nonlocal visits
            if key == "source_id":
                visits += 1
            return super().get(key, default)

    counted = tuple(
        chunk.model_copy(update={"metadata": CountedMetadata(chunk.metadata)})
        for chunk in original.chunks
    )
    actual = corpus.ChunkCorpus(
        repository_id=original.repository_id,
        commit=original.commit,
        sources=original.sources,
        chunks=counted,
    )
    assert actual.model_dump() == original.model_dump()
    assert 0 < visits <= 3 * len(counted) + len(original.sources)


@pytest.mark.parametrize("source_id", [None, "unknown-source", [], {}, 123, True])
def test_direct_corpus_rejects_invalid_chunk_source_id(
    tmp_path: Path,
    source_id: Any,
) -> None:
    repository, manifest = _inputs(tmp_path)
    original = corpus.RepositoryChunkCorpusBuilder(repository).build(manifest).corpus
    metadata = dict(original.chunks[0].metadata)
    if source_id is None:
        metadata.pop("source_id")
    else:
        metadata["source_id"] = source_id
    changed = original.chunks[0].model_copy(update={"metadata": metadata})
    with pytest.raises(ValidationError):
        corpus.ChunkCorpus(
            repository_id=original.repository_id,
            commit=original.commit,
            sources=original.sources,
            chunks=(changed, *original.chunks[1:]),
        )
