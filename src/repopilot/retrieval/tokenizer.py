"""Deterministic tokenizer for source code, identifiers, and repository paths."""

from __future__ import annotations

import re
from typing import Final, Literal

from repopilot.schemas import CodeChunk

BM25_PATH_BOOST: Final[Literal[2]] = 2
BM25_SYMBOL_BOOST: Final[Literal[3]] = 3
BM25_MAX_TOKEN_LENGTH = 512

_COMPOUND_PATTERN = re.compile(
    r"[A-Za-z0-9_]+(?:[./-][A-Za-z0-9_]+)+",
)
_LEXEME_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\W\d_]+", re.UNICODE)
_CAMEL_PART_PATTERN = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|\d|\Z)|[A-Z]?[a-z]+|[A-Z]+|\d+",
)


def tokenize_code_text(value: str) -> tuple[str, ...]:
    """Return lowercase full identifiers/paths plus their deterministic components.

    Repeated occurrences remain repeated for term-frequency calculation. Within one
    lexical occurrence, duplicate expansions are removed while preserving order.
    """

    tokens: list[str] = []
    compound_starts: dict[int, str] = {
        match.start(): match.group(0).lower() for match in _COMPOUND_PATTERN.finditer(value)
    }
    for match in _LEXEME_PATTERN.finditer(value):
        compound = compound_starts.get(match.start())
        if compound is not None and len(compound) <= BM25_MAX_TOKEN_LENGTH:
            tokens.append(compound)
        tokens.extend(_expand_lexeme(match.group(0)))
    return tuple(tokens)


def tokenize_chunk(
    chunk: CodeChunk,
    *,
    path_boost: int = BM25_PATH_BOOST,
    symbol_boost: int = BM25_SYMBOL_BOOST,
) -> tuple[str, ...]:
    """Tokenize chunk content and apply fixed integer path/symbol field boosts."""

    if path_boost < 1 or symbol_boost < 1:
        raise ValueError("tokenizer field boosts must be positive integers")
    content_tokens = tokenize_code_text(chunk.content)
    path_tokens = tokenize_code_text(chunk.relative_path)
    symbol_tokens = tokenize_code_text(chunk.symbol_name) if chunk.symbol_name is not None else ()
    return content_tokens + path_tokens * path_boost + symbol_tokens * symbol_boost


def _expand_lexeme(value: str) -> tuple[str, ...]:
    normalized = value.lower()
    expanded = [normalized] if len(normalized) <= BM25_MAX_TOKEN_LENGTH else []
    for snake_part in value.split("_"):
        if not snake_part:
            continue
        camel_parts = _CAMEL_PART_PATTERN.findall(snake_part)
        if not camel_parts:
            camel_parts = [snake_part]
        expanded.extend(
            normalized_part
            for part in camel_parts
            if len(normalized_part := part.lower()) <= BM25_MAX_TOKEN_LENGTH
        )
    return tuple(dict.fromkeys(expanded))
