"""Versioned embedding contracts, safe errors, and bounded batch retries."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from repopilot.enums import EmbeddingErrorCode, EmbeddingProviderKind
from repopilot.schemas import (
    ModelIdentifier,
    Sha256,
    StrictModel,
    VersionIdentifier,
)

EMBEDDING_SCHEMA_VERSION: Literal[1] = 1
MAX_EMBEDDING_DIMENSION = 2_000
MAX_EMBEDDING_COORDINATE = 3.4e38
EMBEDDING_ERROR_MESSAGES = {
    EmbeddingErrorCode.INVALID_INPUT: "embedding input is invalid",
    EmbeddingErrorCode.IDENTITY_MISMATCH: "embedding identity does not match the provider",
    EmbeddingErrorCode.PROVIDER_UNAVAILABLE: "embedding provider is unavailable",
    EmbeddingErrorCode.REMOTE_CODE_DISABLED: "remote embedding model code is not enabled",
    EmbeddingErrorCode.PROVIDER_FAILURE: "embedding provider call failed",
    EmbeddingErrorCode.RETRY_EXHAUSTED: "embedding provider retries were exhausted",
    EmbeddingErrorCode.INVALID_RESPONSE: "embedding provider returned an invalid response",
    EmbeddingErrorCode.CACHE_IO_ERROR: "embedding cache could not be read or written",
    EmbeddingErrorCode.CACHE_CORRUPT: "embedding cache entry is invalid",
    EmbeddingErrorCode.CORPUS_MISMATCH: "embedding corpus identity is inconsistent",
}


class EmbeddingError(RuntimeError):
    """Safe typed failure that omits source text, vectors, and local paths."""

    def __init__(
        self,
        code: EmbeddingErrorCode,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(EMBEDDING_ERROR_MESSAGES[code])
        self.code = code
        self.retryable = retryable


class EmbeddingIdentity(StrictModel):
    """Everything that can change the numerical meaning of an embedding."""

    provider: EmbeddingProviderKind
    model: ModelIdentifier
    revision: VersionIdentifier
    code_revision: VersionIdentifier = "v1"
    dimension: int = Field(gt=0, le=MAX_EMBEDDING_DIMENSION)
    normalize: bool
    max_length: int = Field(gt=0, le=8_192)

    @field_validator("dimension", "max_length", mode="before")
    @classmethod
    def reject_boolean_limits(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("embedding numeric limits must not be booleans")
        return value

    @property
    def fingerprint(self) -> Sha256:
        return identity_fingerprint(self)


class EmbeddingInput(StrictModel):
    """One content-addressed input; chunk identity is deliberately excluded."""

    content_hash: Sha256
    content: str

    @model_validator(mode="after")
    def validate_content_hash(self) -> Self:
        digest = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if digest != self.content_hash:
            raise ValueError("content_hash must match UTF-8 embedding content")
        return self


class EmbeddingRequest(StrictModel):
    """One deterministic, correlated provider batch attempt."""

    schema_version: Literal[1] = EMBEDDING_SCHEMA_VERSION
    request_id: Sha256
    identity: EmbeddingIdentity
    inputs: tuple[EmbeddingInput, ...] = Field(min_length=1, max_length=256)
    attempt: int = Field(gt=0, le=6)

    @field_validator("attempt", mode="before")
    @classmethod
    def reject_boolean_attempt(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("embedding attempt must not be a boolean")
        return value

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        hashes = tuple(item.content_hash for item in self.inputs)
        if len(set(hashes)) != len(hashes):
            raise ValueError("embedding request content hashes must be unique")
        if self.request_id != embedding_request_id(self.identity, self.inputs, self.attempt):
            raise ValueError("embedding request_id does not match its payload")
        return self


class EmbeddingVector(StrictModel):
    """One finite vector correlated by content hash."""

    content_hash: Sha256
    values: tuple[float, ...] = Field(min_length=1, max_length=MAX_EMBEDDING_DIMENSION)

    @field_validator("values", mode="before")
    @classmethod
    def validate_values(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            raise ValueError("embedding values must be a list or tuple")
        for coordinate in value:
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(float(coordinate))
                or abs(float(coordinate)) > MAX_EMBEDDING_COORDINATE
            ):
                raise ValueError("embedding coordinates must be finite float32 values")
        return value


class EmbeddingResponse(StrictModel):
    """Provider output whose exact request correlation is checked by the executor."""

    schema_version: Literal[1] = EMBEDDING_SCHEMA_VERSION
    request_id: Sha256
    identity: EmbeddingIdentity
    vectors: tuple[EmbeddingVector, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_vectors(self) -> Self:
        hashes = tuple(vector.content_hash for vector in self.vectors)
        if len(set(hashes)) != len(hashes):
            raise ValueError("embedding response content hashes must be unique")
        if any(len(vector.values) != self.identity.dimension for vector in self.vectors):
            raise ValueError("embedding response vector dimension is inconsistent")
        return self


class EmbeddingExecutionResult(StrictModel):
    """Vectors plus per-call batching and retry evidence."""

    identity: EmbeddingIdentity
    vectors: tuple[EmbeddingVector, ...]
    batch_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts_and_vectors(self) -> Self:
        hashes = tuple(vector.content_hash for vector in self.vectors)
        if len(set(hashes)) != len(hashes):
            raise ValueError("embedding execution vectors must be unique")
        if self.provider_call_count != self.batch_count + self.retry_count:
            raise ValueError("provider calls must equal batches plus retries")
        if self.batch_count == 0 and self.vectors:
            raise ValueError("vectors require at least one provider batch")
        return self


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Numerical embedding capability independent of Agent decision providers."""

    @property
    def identity(self) -> EmbeddingIdentity:
        """Return the exact model and encoding identity."""

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed one bounded batch or raise :class:`EmbeddingError`."""


def identity_fingerprint(identity: EmbeddingIdentity) -> Sha256:
    """Return the canonical identity used by cache keys and vector namespaces."""

    document = {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "identity": identity.model_dump(mode="json"),
    }
    payload = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def embedding_request_id(
    identity: EmbeddingIdentity,
    inputs: Sequence[EmbeddingInput],
    attempt: int,
) -> Sha256:
    """Correlate one exact ordered batch attempt without retaining source text."""

    payload = {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "identity": identity.model_dump(mode="json"),
        "content_hashes": [item.content_hash for item in inputs],
        "attempt": attempt,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class EmbeddingBatchExecutor:
    """Run deterministic batches and bounded exponential retries."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        batch_size: int,
        max_retries: int,
        retry_backoff_ms: int,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or isinstance(max_retries, bool)
            or isinstance(retry_backoff_ms, bool)
            or not 1 <= batch_size <= 256
            or not 0 <= max_retries <= 5
            or not 0 <= retry_backoff_ms <= 10_000
        ):
            raise EmbeddingError(EmbeddingErrorCode.INVALID_INPUT)
        self._provider = provider
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_backoff_ms = retry_backoff_ms
        self._sleep = sleep

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._provider.identity

    def embed(self, inputs: Sequence[EmbeddingInput]) -> EmbeddingExecutionResult:
        """Embed unique inputs in order; max_retries excludes the first attempt."""

        normalized = tuple(inputs)
        content_hashes = tuple(item.content_hash for item in normalized)
        if len(set(content_hashes)) != len(content_hashes):
            raise EmbeddingError(EmbeddingErrorCode.INVALID_INPUT)
        if not normalized:
            return EmbeddingExecutionResult(
                identity=self.identity,
                vectors=(),
                batch_count=0,
                provider_call_count=0,
                retry_count=0,
            )

        vectors: list[EmbeddingVector] = []
        batch_count = 0
        provider_call_count = 0
        retry_count = 0
        for offset in range(0, len(normalized), self._batch_size):
            batch = normalized[offset : offset + self._batch_size]
            batch_count += 1
            for attempt in range(1, self._max_retries + 2):
                request = EmbeddingRequest(
                    request_id=embedding_request_id(self.identity, batch, attempt),
                    identity=self.identity,
                    inputs=batch,
                    attempt=attempt,
                )
                provider_call_count += 1
                try:
                    response = self._provider.embed(request)
                except EmbeddingError as error:
                    if not error.retryable:
                        raise
                    if attempt > self._max_retries:
                        raise EmbeddingError(EmbeddingErrorCode.RETRY_EXHAUSTED) from None
                    retry_count += 1
                    delay_seconds = self._retry_backoff_ms * (2 ** (attempt - 1)) / 1_000
                    self._sleep(delay_seconds)
                    continue
                except Exception:
                    raise EmbeddingError(EmbeddingErrorCode.PROVIDER_FAILURE) from None
                vectors.extend(_validated_response_vectors(request, response))
                break

        return EmbeddingExecutionResult(
            identity=self.identity,
            vectors=tuple(vectors),
            batch_count=batch_count,
            provider_call_count=provider_call_count,
            retry_count=retry_count,
        )


def _validated_response_vectors(
    request: EmbeddingRequest,
    response: object,
) -> tuple[EmbeddingVector, ...]:
    if not isinstance(response, EmbeddingResponse):
        raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE)
    if response.request_id != request.request_id:
        raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE)
    if response.identity != request.identity:
        raise EmbeddingError(EmbeddingErrorCode.IDENTITY_MISMATCH)
    expected_hashes = tuple(item.content_hash for item in request.inputs)
    actual_hashes = tuple(vector.content_hash for vector in response.vectors)
    if actual_hashes != expected_hashes:
        raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE)
    return response.vectors
