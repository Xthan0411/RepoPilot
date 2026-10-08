"""Integrity-protected content-addressed local embedding cache."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Literal, Self

from pydantic import ValidationError, model_validator

from repopilot.embeddings.base import (
    EMBEDDING_SCHEMA_VERSION,
    EmbeddingError,
    EmbeddingIdentity,
    EmbeddingVector,
)
from repopilot.enums import EmbeddingErrorCode
from repopilot.schemas import Sha256, StrictModel

EMBEDDING_CACHE_SCHEMA_VERSION: Literal[1] = 1
MAX_CACHE_ENTRY_BYTES = 1_000_000


class EmbeddingCacheRecord(StrictModel):
    """One independently verifiable cache document."""

    schema_version: Literal[1] = EMBEDDING_CACHE_SCHEMA_VERSION
    cache_key: Sha256
    identity: EmbeddingIdentity
    vector: EmbeddingVector
    integrity_sha256: Sha256

    @model_validator(mode="after")
    def validate_cache_record(self) -> Self:
        if self.cache_key != embedding_cache_key(
            self.identity,
            self.vector.content_hash,
        ):
            raise ValueError("embedding cache key does not match its payload")
        if len(self.vector.values) != self.identity.dimension:
            raise ValueError("embedding cache vector dimension is inconsistent")
        if self.integrity_sha256 != _record_integrity(
            self.cache_key,
            self.identity,
            self.vector,
        ):
            raise ValueError("embedding cache integrity digest does not match")
        return self


def embedding_cache_key(
    identity: EmbeddingIdentity,
    content_hash: str,
) -> Sha256:
    """Include schema, full model identity, encoding policy, and exact content."""

    if re.fullmatch(r"[0-9a-f]{64}", content_hash) is None:
        raise EmbeddingError(EmbeddingErrorCode.INVALID_INPUT)
    payload = {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "identity_fingerprint": identity.fingerprint,
        "content_hash": content_hash,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class LocalEmbeddingCache:
    """Store one atomic JSON file per content-addressed vector."""

    def __init__(self, root: Path) -> None:
        if root.is_symlink():
            raise EmbeddingError(EmbeddingErrorCode.CACHE_IO_ERROR)
        try:
            self._root = root.resolve()
        except OSError:
            raise EmbeddingError(EmbeddingErrorCode.CACHE_IO_ERROR) from None

    @property
    def root(self) -> Path:
        return self._root

    def get(
        self,
        identity: EmbeddingIdentity,
        content_hash: str,
    ) -> EmbeddingVector | None:
        """Return a verified vector or fail closed on a corrupt existing entry."""

        key = embedding_cache_key(identity, content_hash)
        target = self._entry_path(key)
        self._validate_existing_root()
        if target.parent.is_symlink():
            raise EmbeddingError(EmbeddingErrorCode.CACHE_CORRUPT)
        if not target.exists():
            return None
        if target.is_symlink():
            raise EmbeddingError(EmbeddingErrorCode.CACHE_CORRUPT)
        try:
            if target.stat().st_size > MAX_CACHE_ENTRY_BYTES:
                raise EmbeddingError(EmbeddingErrorCode.CACHE_CORRUPT)
            document = json.loads(target.read_text(encoding="utf-8"))
            record = EmbeddingCacheRecord.model_validate(document)
        except EmbeddingError:
            raise
        except (OSError, UnicodeError):
            raise EmbeddingError(EmbeddingErrorCode.CACHE_IO_ERROR) from None
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
            raise EmbeddingError(EmbeddingErrorCode.CACHE_CORRUPT) from None
        if record.identity != identity or record.vector.content_hash != content_hash:
            raise EmbeddingError(EmbeddingErrorCode.CACHE_CORRUPT)
        return record.vector

    def put(
        self,
        identity: EmbeddingIdentity,
        vector: EmbeddingVector,
    ) -> None:
        """Atomically persist a new vector; never replace conflicting valid data."""

        if len(vector.values) != identity.dimension:
            raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE)
        key = embedding_cache_key(identity, vector.content_hash)
        existing = self.get(identity, vector.content_hash)
        if existing is not None:
            if existing != vector:
                raise EmbeddingError(EmbeddingErrorCode.CACHE_CORRUPT)
            return

        record = EmbeddingCacheRecord(
            cache_key=key,
            identity=identity,
            vector=vector,
            integrity_sha256=_record_integrity(key, identity, vector),
        )
        target = self._entry_path(key)
        parent = target.parent
        temporary_path: Path | None = None
        try:
            self._ensure_root()
            parent.mkdir(parents=False, exist_ok=True)
            if parent.is_symlink() or not parent.is_dir():
                raise EmbeddingError(EmbeddingErrorCode.CACHE_IO_ERROR)
            serialized = f"{record.model_dump_json(indent=2)}\n"
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=parent,
                prefix=f".{key}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(serialized)
                temporary.flush()
                os.fsync(temporary.fileno())
            if target.is_symlink():
                raise EmbeddingError(EmbeddingErrorCode.CACHE_IO_ERROR)
            temporary_path.replace(target)
        except EmbeddingError:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink()
            raise
        except OSError:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink()
            raise EmbeddingError(EmbeddingErrorCode.CACHE_IO_ERROR) from None

    def _validate_existing_root(self) -> None:
        if self._root.exists() and (self._root.is_symlink() or not self._root.is_dir()):
            raise EmbeddingError(EmbeddingErrorCode.CACHE_IO_ERROR)

    def _ensure_root(self) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise EmbeddingError(EmbeddingErrorCode.CACHE_IO_ERROR) from None
        if self._root.is_symlink() or not self._root.is_dir():
            raise EmbeddingError(EmbeddingErrorCode.CACHE_IO_ERROR)

    def _entry_path(self, key: str) -> Path:
        return self._root / key[:2] / f"{key}.json"


def _record_integrity(
    cache_key: str,
    identity: EmbeddingIdentity,
    vector: EmbeddingVector,
) -> Sha256:
    payload = {
        "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "identity": identity.model_dump(mode="json"),
        "vector": vector.model_dump(mode="json"),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
