"""Tokenizer, Okapi BM25 ranking, persistence, and hostile artifact tests."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.enums import BM25ErrorCode, SymbolType
from repopilot.retrieval import (
    BM25Document,
    BM25Error,
    BM25Policy,
    BM25Retriever,
    BM25TermFrequency,
    ChunkCorpus,
    ChunkSourceSnapshot,
    build_bm25_index,
    chunk_corpus_content_sha256,
    read_bm25_index,
    tokenize_chunk,
    tokenize_code_text,
    verify_bm25_index_corpus,
    verify_bm25_index_integrity,
    write_bm25_index,
)
from repopilot.schemas import CodeChunk

REPOSITORY_ID = "fixture.bm25"
COMMIT = "e" * 40
SOURCE_HASH = "a" * 64
BUILD_FINGERPRINT = "b" * 64


def code_chunk(
    *,
    chunk_id: str,
    path: str,
    content: str,
    symbol_name: str | None = None,
    source_id: str | None = None,
) -> CodeChunk:
    effective_source_id = source_id or f"src_{chunk_id}"
    return CodeChunk(
        chunk_id=chunk_id,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        relative_path=path,
        symbol_name=symbol_name,
        symbol_type=SymbolType.FUNCTION if symbol_name is not None else SymbolType.TEXT,
        start_line=1,
        end_line=max(1, content.count("\n")),
        language="python",
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        metadata={
            "source_id": effective_source_id,
            "source_content_hash": SOURCE_HASH,
            "processor": "test",
        },
    )


def chunk_corpus(*chunks: CodeChunk) -> ChunkCorpus:
    sorted_chunks = tuple(sorted(chunks, key=lambda item: item.relative_path))
    sources = tuple(
        ChunkSourceSnapshot(
            source_id=cast(str, chunk.metadata["source_id"]),
            relative_path=chunk.relative_path,
            source_content_hash=SOURCE_HASH,
            processor="test",
            build_fingerprint=BUILD_FINGERPRINT,
            chunk_ids=(chunk.chunk_id,),
        )
        for chunk in sorted_chunks
    )
    return ChunkCorpus(
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        sources=sources,
        chunks=sorted_chunks,
    )


def assert_bm25_error(
    captured: pytest.ExceptionInfo[BM25Error],
    code: BM25ErrorCode,
) -> None:
    assert captured.value.code is code


def test_tokenizer_preserves_and_splits_identifiers_paths_and_repetitions() -> None:
    assert tokenize_code_text("HTTPParser parse_file camelCase pkg/http-client.py") == (
        "httpparser",
        "http",
        "parser",
        "parse_file",
        "parse",
        "file",
        "camelcase",
        "camel",
        "case",
        "pkg/http-client.py",
        "pkg",
        "http",
        "client",
        "py",
    )
    assert tokenize_code_text("HTTPParser HTTPParser") == (
        "httpparser",
        "http",
        "parser",
        "httpparser",
        "http",
        "parser",
    )
    assert tokenize_code_text("配置 parser") == ("配置", "parser")
    assert tokenize_code_text("x" * 513) == ()


def test_chunk_tokenizer_applies_fixed_path_and_symbol_boosts() -> None:
    chunk = code_chunk(
        chunk_id="chunk_tokenized",
        path="pkg/http_parser.py",
        symbol_name="HTTPParser.parse_file",
        content="return value\n",
    )

    counts = Counter(tokenize_chunk(chunk))

    assert counts["return"] == 1
    assert counts["pkg/http_parser.py"] == 2
    assert counts["http_parser"] == 2
    assert counts["httpparser"] == 3
    assert counts["parse_file"] == 3
    assert counts["parser"] == 5


def test_policy_defaults_settings_mapping_and_invalid_values() -> None:
    assert BM25Policy() == BM25Policy(k1=1.5, b=0.75, default_top_k=10)
    assert BM25Policy.from_settings(
        RepoPilotSettings(bm25_k1=1.2, bm25_b=0.6, bm25_default_top_k=25)
    ) == BM25Policy(k1=1.2, b=0.6, default_top_k=25)

    for values in (
        {"k1": 0},
        {"k1": 5.1},
        {"k1": math.nan},
        {"k1": math.inf},
        {"k1": True},
        {"b": -0.1},
        {"b": 1.1},
        {"b": math.nan},
        {"b": math.inf},
        {"b": False},
        {"default_top_k": 0},
        {"default_top_k": 101},
        {"default_top_k": True},
    ):
        with pytest.raises(ValidationError):
            BM25Policy(**values)


def test_index_is_deterministic_complete_and_repository_pinned() -> None:
    corpus = chunk_corpus(
        code_chunk(
            chunk_id="chunk_zeta",
            path="zeta.py",
            symbol_name="parse_value",
            content="def parse_value(value):\n    return value\n",
        ),
        code_chunk(
            chunk_id="chunk_alpha",
            path="alpha.py",
            symbol_name="format_value",
            content="def format_value(value):\n    return value\n",
        ),
    )

    first = build_bm25_index(corpus)
    second = build_bm25_index(corpus)

    assert first == second
    assert first.repository_id == REPOSITORY_ID
    assert first.commit == COMMIT
    assert first.corpus_sha256 == chunk_corpus_content_sha256(corpus)
    assert tuple(document.chunk_id for document in first.documents) == (
        "chunk_alpha",
        "chunk_zeta",
    )
    assert first.document_count == 2
    assert first.total_document_length == sum(document.length for document in first.documents)
    assert first.average_document_length == (first.total_document_length / first.document_count)
    verify_bm25_index_integrity(first)


def test_standard_okapi_formula_and_explainable_hit() -> None:
    corpus = chunk_corpus(
        code_chunk(
            chunk_id="chunk_parser",
            path="pkg/parser.py",
            symbol_name="Parser.parse",
            content="parse parse other\n",
        ),
    )
    index = build_bm25_index(corpus, policy=BM25Policy(k1=1.5, b=0.75))
    retriever = BM25Retriever(index)

    result = retriever.search(
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        query="parse",
        top_k=1,
    )

    document = index.documents[0]
    term_frequency = dict((item.term, item.count) for item in document.term_frequencies)["parse"]
    inverse_document_frequency = math.log(1 + (1 - 1 + 0.5) / (1 + 0.5))
    expected = (
        inverse_document_frequency * term_frequency * (index.k1 + 1) / (term_frequency + index.k1)
    )
    assert result.hits[0].sparse_score == pytest.approx(expected)
    assert result.hits[0].sparse_rank == 1
    assert result.hits[0].dense_rank is None
    assert result.hits[0].query == "parse"
    assert result.hits[0].retrieval_reason == (
        "BM25 matched [parse]; path=pkg/parser.py; symbol=Parser.parse"
    )


def test_symbol_and_path_tokens_improve_exact_code_localization() -> None:
    corpus = chunk_corpus(
        code_chunk(
            chunk_id="chunk_target",
            path="pkg/http_parser.py",
            symbol_name="HTTPParser.parse_response",
            content="return value\n",
        ),
        code_chunk(
            chunk_id="chunk_other",
            path="pkg/formatter.py",
            symbol_name="Formatter.render",
            content="return value\n",
        ),
    )
    retriever = BM25Retriever.from_corpus(corpus)

    result = retriever.search(
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        query="HTTPParser parse_response",
    )

    assert tuple(hit.chunk_id for hit in result.hits) == ("chunk_target",)
    assert "symbol=HTTPParser.parse_response" in result.hits[0].retrieval_reason


def test_top_k_no_match_and_equal_score_tie_break_are_stable() -> None:
    corpus = chunk_corpus(
        code_chunk(
            chunk_id="chunk_z",
            path="a.py",
            content="shared\n",
        ),
        code_chunk(
            chunk_id="chunk_a",
            path="b.py",
            content="shared\n",
        ),
    )
    retriever = BM25Retriever.from_corpus(
        corpus,
        policy=BM25Policy(default_top_k=1),
    )

    default_result = retriever.search(
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        query="shared",
    )
    both = retriever.search(
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        query="shared",
        top_k=2,
    )
    missing = retriever.search(
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        query="not_present_anywhere",
    )

    assert tuple(hit.chunk_id for hit in default_result.hits) == ("chunk_a",)
    assert tuple(hit.chunk_id for hit in both.hits) == ("chunk_a", "chunk_z")
    assert both.hits[0].sparse_score == both.hits[1].sparse_score
    assert missing.hits == ()


def test_allowlist_filters_before_bm25_ranking_and_reassigns_ranks() -> None:
    retriever = BM25Retriever.from_corpus(
        chunk_corpus(
            code_chunk(
                chunk_id="chunk_a",
                path="a.py",
                content="shared shared\n",
            ),
            code_chunk(
                chunk_id="chunk_b",
                path="b.py",
                content="shared\n",
            ),
        )
    )

    result = retriever.search(
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        query="shared",
        top_k=2,
        allowed_chunk_ids=("chunk_b",),
    )

    assert tuple(hit.chunk_id for hit in result.hits) == ("chunk_b",)
    assert result.hits[0].sparse_rank == 1


@pytest.mark.parametrize("query", ["", " \n ", "!!!"])
def test_blank_or_tokenless_query_has_explicit_error(query: str) -> None:
    retriever = BM25Retriever.from_corpus(chunk_corpus())

    with pytest.raises(BM25Error) as captured:
        retriever.search(
            repository_id=REPOSITORY_ID,
            commit=COMMIT,
            query=query,
        )

    assert_bm25_error(captured, BM25ErrorCode.EMPTY_QUERY)


@pytest.mark.parametrize("top_k", [0, 101, True, 1.5])
def test_invalid_top_k_has_explicit_error(top_k: Any) -> None:
    retriever = BM25Retriever.from_corpus(chunk_corpus())

    with pytest.raises(BM25Error) as captured:
        retriever.search(
            repository_id=REPOSITORY_ID,
            commit=COMMIT,
            query="parser",
            top_k=top_k,
        )

    assert_bm25_error(captured, BM25ErrorCode.INVALID_QUERY)


def test_query_rejects_nul_oversize_and_identity_mismatch() -> None:
    retriever = BM25Retriever.from_corpus(chunk_corpus())

    for query in ("parser\x00secret", "x" * 2_001):
        with pytest.raises(BM25Error) as captured:
            retriever.search(
                repository_id=REPOSITORY_ID,
                commit=COMMIT,
                query=query,
            )
        assert_bm25_error(captured, BM25ErrorCode.INVALID_QUERY)

    for repository_id, commit in (("other.repo", COMMIT), (REPOSITORY_ID, "f" * 40)):
        with pytest.raises(BM25Error) as captured:
            retriever.search(
                repository_id=repository_id,
                commit=commit,
                query="parser",
            )
        assert_bm25_error(captured, BM25ErrorCode.IDENTITY_MISMATCH)


def test_empty_corpus_is_valid_and_round_trips(tmp_path: Path) -> None:
    index = build_bm25_index(chunk_corpus())
    path = tmp_path / "nested" / "bm25.json"

    write_bm25_index(path, index)
    loaded = read_bm25_index(path)

    assert loaded == index
    assert loaded.document_count == 0
    assert loaded.total_document_length == 0
    assert loaded.average_document_length == 0.0
    assert loaded.documents == ()
    assert loaded.document_frequencies == ()
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_index_round_trip_is_byte_deterministic(tmp_path: Path) -> None:
    index = build_bm25_index(
        chunk_corpus(
            code_chunk(
                chunk_id="chunk_parser",
                path="parser.py",
                symbol_name="parse_value",
                content="def parse_value():\n    pass\n",
            )
        )
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_bm25_index(first, index)
    write_bm25_index(second, read_bm25_index(first))

    assert first.read_bytes() == second.read_bytes()


def test_reader_rejects_tampering_invalid_statistics_and_unknown_version(
    tmp_path: Path,
) -> None:
    index = build_bm25_index(
        chunk_corpus(
            code_chunk(
                chunk_id="chunk_parser",
                path="parser.py",
                content="parse\n",
            )
        )
    )
    path = tmp_path / "bm25.json"
    write_bm25_index(path, index)
    raw = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))

    raw["k1"] = 1.7
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BM25Error) as captured:
        read_bm25_index(path)
    assert_bm25_error(captured, BM25ErrorCode.INTEGRITY_MISMATCH)

    raw = cast(dict[str, Any], json.loads(index.model_dump_json()))
    raw["document_count"] = 2
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BM25Error) as captured:
        read_bm25_index(path)
    assert_bm25_error(captured, BM25ErrorCode.INVALID_DOCUMENT)

    raw = cast(dict[str, Any], json.loads(index.model_dump_json()))
    raw["schema_version"] = 2
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BM25Error) as captured:
        read_bm25_index(path)
    assert_bm25_error(captured, BM25ErrorCode.UNSUPPORTED_VERSION)


def test_reader_rejects_missing_malformed_and_non_utf8_documents(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(BM25Error) as captured:
        read_bm25_index(missing)
    assert_bm25_error(captured, BM25ErrorCode.IO_ERROR)

    path = tmp_path / "bm25.json"
    for payload in (b"{", b"[]", b'{"documents":[]}', b"\xff"):
        path.write_bytes(payload)
        with pytest.raises(BM25Error) as captured:
            read_bm25_index(path)
        assert_bm25_error(captured, BM25ErrorCode.INVALID_DOCUMENT)


def test_model_rejects_duplicate_terms_bad_lengths_and_bad_digest() -> None:
    with pytest.raises(ValidationError):
        BM25Document(
            chunk_id="chunk_invalid",
            relative_path="invalid.py",
            length=2,
            term_frequencies=(
                BM25TermFrequency(term="same", count=1),
                BM25TermFrequency(term="same", count=1),
            ),
        )

    index = build_bm25_index(chunk_corpus())
    damaged = index.model_copy(update={"content_sha256": "f" * 64})
    with pytest.raises(BM25Error) as captured:
        BM25Retriever(damaged)
    assert_bm25_error(captured, BM25ErrorCode.INTEGRITY_MISMATCH)


def test_corpus_fingerprint_changes_with_content_or_identity() -> None:
    first = chunk_corpus(code_chunk(chunk_id="chunk_first", path="first.py", content="one\n"))
    second = chunk_corpus(code_chunk(chunk_id="chunk_second", path="first.py", content="two\n"))

    assert chunk_corpus_content_sha256(first) != chunk_corpus_content_sha256(second)
    index = build_bm25_index(first)
    verify_bm25_index_corpus(index, first)
    with pytest.raises(BM25Error) as captured:
        verify_bm25_index_corpus(index, second)
    assert_bm25_error(captured, BM25ErrorCode.IDENTITY_MISMATCH)
