"""Lazy local Jina code-embedding adapter with explicit remote-code consent."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterable
from typing import Protocol, cast

from repopilot.embeddings.base import (
    EmbeddingError,
    EmbeddingIdentity,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
)
from repopilot.enums import EmbeddingErrorCode, EmbeddingProviderKind


class SentenceEncoder(Protocol):
    """Narrow SentenceTransformer surface used by the Jina adapter."""

    max_seq_length: int

    def get_embedding_dimension(self) -> int | None:
        """Return the model output width."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> object:
        """Encode one bounded batch."""


ModelFactory = Callable[[str, str, str, bool], SentenceEncoder]


class JinaEmbeddingProvider:
    """Load a revision-pinned Jina SentenceTransformer only on first use."""

    def __init__(
        self,
        identity: EmbeddingIdentity,
        *,
        trust_remote_code: bool,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if identity.provider is not EmbeddingProviderKind.JINA_LOCAL:
            raise EmbeddingError(EmbeddingErrorCode.IDENTITY_MISMATCH)
        self._identity = identity
        self._trust_remote_code = trust_remote_code
        self._model_factory = model_factory
        self._model: SentenceEncoder | None = None

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if request.identity != self.identity:
            raise EmbeddingError(EmbeddingErrorCode.IDENTITY_MISMATCH)
        model = self._load_model()
        try:
            raw_vectors = model.encode(
                [item.content for item in request.inputs],
                batch_size=len(request.inputs),
                normalize_embeddings=self.identity.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except EmbeddingError:
            raise
        except Exception:
            raise EmbeddingError(
                EmbeddingErrorCode.PROVIDER_FAILURE,
                retryable=True,
            ) from None

        vectors = _coerce_vectors(raw_vectors)
        if len(vectors) != len(request.inputs):
            raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE)
        try:
            return EmbeddingResponse(
                request_id=request.request_id,
                identity=self.identity,
                vectors=tuple(
                    EmbeddingVector(
                        content_hash=item.content_hash,
                        values=values,
                    )
                    for item, values in zip(request.inputs, vectors, strict=True)
                ),
            )
        except (TypeError, ValueError):
            raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE) from None

    def _load_model(self) -> SentenceEncoder:
        if self._model is not None:
            return self._model
        if self._model_factory is None and not self._trust_remote_code:
            raise EmbeddingError(EmbeddingErrorCode.REMOTE_CODE_DISABLED)
        factory = self._model_factory or _default_model_factory
        try:
            model = factory(
                self.identity.model,
                self.identity.revision,
                self.identity.code_revision,
                self._trust_remote_code,
            )
            model.max_seq_length = self.identity.max_length
            dimension = model.get_embedding_dimension()
        except EmbeddingError:
            raise
        except (ImportError, ModuleNotFoundError):
            raise EmbeddingError(EmbeddingErrorCode.PROVIDER_UNAVAILABLE) from None
        except Exception:
            raise EmbeddingError(EmbeddingErrorCode.PROVIDER_UNAVAILABLE) from None
        if dimension != self.identity.dimension:
            raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE)
        self._model = model
        return model


def _default_model_factory(
    model_name: str,
    revision: str,
    code_revision: str,
    trust_remote_code: bool,
) -> SentenceEncoder:
    try:
        module = importlib.import_module("sentence_transformers")
        sentence_transformer = module.SentenceTransformer
    except (ImportError, AttributeError):
        raise EmbeddingError(EmbeddingErrorCode.PROVIDER_UNAVAILABLE) from None
    model = sentence_transformer(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        model_kwargs={"code_revision": code_revision},
        config_kwargs={"code_revision": code_revision},
    )
    return cast(SentenceEncoder, model)


def _coerce_vectors(raw_vectors: object) -> tuple[tuple[float, ...], ...]:
    if isinstance(raw_vectors, (str, bytes)) or not isinstance(raw_vectors, Iterable):
        raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE)
    vectors: list[tuple[float, ...]] = []
    for raw_row in raw_vectors:
        converter = getattr(raw_row, "tolist", None)
        row = converter() if callable(converter) else raw_row
        if isinstance(row, (str, bytes)) or not isinstance(row, Iterable):
            raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE)
        values: list[float] = []
        for coordinate in row:
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(float(coordinate))
            ):
                raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE)
            values.append(float(coordinate))
        vectors.append(tuple(values))
    return tuple(vectors)
