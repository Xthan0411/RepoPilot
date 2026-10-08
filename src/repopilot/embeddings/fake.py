"""Deterministic hash-based Embedding Provider for tests and local smoke flows."""

from __future__ import annotations

import hashlib
import math

from repopilot.embeddings.base import (
    EmbeddingError,
    EmbeddingIdentity,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
)
from repopilot.enums import EmbeddingErrorCode, EmbeddingProviderKind


class FakeEmbeddingProvider:
    """Generate stable vectors without model dependencies, files, or network."""

    def __init__(
        self,
        identity: EmbeddingIdentity,
        *,
        fail_first_calls: int = 0,
    ) -> None:
        if identity.provider is not EmbeddingProviderKind.FAKE:
            raise EmbeddingError(EmbeddingErrorCode.IDENTITY_MISMATCH)
        if (
            isinstance(fail_first_calls, bool)
            or not isinstance(fail_first_calls, int)
            or not 0 <= fail_first_calls <= 100
        ):
            raise EmbeddingError(EmbeddingErrorCode.INVALID_INPUT)
        self._identity = identity
        self._fail_first_calls = fail_first_calls
        self._requests: list[EmbeddingRequest] = []

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    @property
    def calls_made(self) -> int:
        return len(self._requests)

    @property
    def requests(self) -> tuple[EmbeddingRequest, ...]:
        return tuple(self._requests)

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self._requests.append(request)
        if request.identity != self.identity:
            raise EmbeddingError(EmbeddingErrorCode.IDENTITY_MISMATCH)
        if self.calls_made <= self._fail_first_calls:
            raise EmbeddingError(
                EmbeddingErrorCode.PROVIDER_FAILURE,
                retryable=True,
            )
        return EmbeddingResponse(
            request_id=request.request_id,
            identity=self.identity,
            vectors=tuple(
                EmbeddingVector(
                    content_hash=item.content_hash,
                    values=_hash_vector(
                        item.content_hash,
                        identity=self.identity,
                    ),
                )
                for item in request.inputs
            ),
        )


def _hash_vector(
    content_hash: str,
    *,
    identity: EmbeddingIdentity,
) -> tuple[float, ...]:
    seed = (
        f"{identity.fingerprint}:{content_hash}:{identity.dimension}:{int(identity.normalize)}"
    ).encode("ascii")
    coordinates: list[float] = []
    block_index = 0
    while len(coordinates) < identity.dimension:
        block = hashlib.sha256(seed + block_index.to_bytes(8, "big")).digest()
        for offset in range(0, len(block), 4):
            integer = int.from_bytes(block[offset : offset + 4], "big")
            coordinates.append((integer / 4_294_967_295) * 2.0 - 1.0)
            if len(coordinates) == identity.dimension:
                break
        block_index += 1
    if identity.normalize:
        norm = math.sqrt(math.fsum(value * value for value in coordinates))
        if norm == 0.0:
            raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE)
        coordinates = [value / norm for value in coordinates]
    return tuple(coordinates)
