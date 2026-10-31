"""Bounded offline CPU qualification of a few sealed GPU-cache vectors.

This internal worker never builds an Agent, opens a database, or contacts a model API.
The caller owns its subprocess wall-clock timeout and the temporary request file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from pydantic import Field, JsonValue, field_validator, model_validator

from repopilot.embeddings import (
    EmbeddingBatchExecutor,
    EmbeddingIdentity,
    EmbeddingInput,
    EmbeddingProvider,
    EmbeddingVector,
)
from repopilot.embeddings.jina import JinaEmbeddingProvider, SentenceEncoder
from repopilot.schemas import StrictModel

CPU_THREADS = 8
COSINE_MINIMUM = 0.9999
MAX_ABSOLUTE_ERROR = 0.001
NORM_TOLERANCE = 0.0001
MAX_REQUEST_BYTES = 8 * 1024 * 1024
PINNED_IDENTITY = EmbeddingIdentity.model_validate(
    {
        "provider": "jina_local",
        "model": "jinaai/jina-embeddings-v2-base-code",
        "revision": "516f4baf13dec4ddddda8631e019b5737c8bc250",
        "code_revision": "3baf9e3ac750e76e8edd3019170176884695fb94",
        "dimension": 768,
        "normalize": True,
        "max_length": 2048,
    }
)


class CpuProbeError(RuntimeError):
    """A safe error code without request contents or local credentials."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class NumericSample(StrictModel):
    input: EmbeddingInput
    gpu_vector: tuple[float, ...]

    @field_validator("gpu_vector", mode="before")
    @classmethod
    def reject_non_coordinates(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)) or any(
            isinstance(coordinate, bool) or not isinstance(coordinate, (int, float))
            for coordinate in value
        ):
            raise ValueError("GPU vector coordinates must be numeric, not coerced")
        return value

    @model_validator(mode="after")
    def validate_vector(self) -> NumericSample:
        # Reuse strict finite-coordinate and content-address contracts.
        EmbeddingVector(content_hash=self.input.content_hash, values=self.gpu_vector)
        if len(self.input.content.encode("utf-8")) > 2 * 1024 * 1024:
            raise ValueError("sample exceeds worker content limit")
        _vector_norm(self.gpu_vector, PINNED_IDENTITY.dimension)
        return self


class ProbeRequest(StrictModel):
    identity: EmbeddingIdentity
    samples: tuple[NumericSample, ...] = Field(min_length=3, max_length=3)
    queries: tuple[EmbeddingInput, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_request(self) -> ProbeRequest:
        if self.identity != PINNED_IDENTITY:
            raise ValueError("unexpected probe identity")
        if len({sample.input.content_hash for sample in self.samples}) != 3:
            raise ValueError("numeric samples must be distinct")
        if any(not query.content or len(query.content) > 256 for query in self.queries):
            raise ValueError("queries must contain 1 to 256 characters")
        return self


CpuLoader = Callable[[EmbeddingIdentity], tuple[EmbeddingProvider, dict[str, JsonValue]]]


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise CpuProbeError(code)


def _offline() -> None:
    _require(
        os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "offline_environment_required",
    )


@contextmanager
def no_network() -> Iterator[None]:
    """Block ordinary Python network connection paths during the pure probe."""

    def blocked(*args: object, **kwargs: object) -> Any:
        raise CpuProbeError("network_access_forbidden")

    targets = (
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
        (socket, "create_connection"),
        (socket, "getaddrinfo"),
    )
    originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
    try:
        for owner, name, _ in originals:
            setattr(owner, name, blocked)
        yield
    finally:
        for owner, name, value in originals:
            setattr(owner, name, value)


def create_cpu_provider(
    identity: EmbeddingIdentity,
) -> tuple[EmbeddingProvider, dict[str, JsonValue]]:
    """Explicit CPU/FP32 loader shared with the bounded retrieval qualifier.

    This function permits a database environment for the retrieval caller; the pure
    probe rejects it separately. No model downloads are allowed in either caller.
    """
    _offline()
    _require(identity == PINNED_IDENTITY, "embedding_identity_mismatch")
    _require(
        platform.system() == "Linux" and platform.machine() == "x86_64",
        "linux_amd64_required",
    )
    torch = importlib.import_module("torch")
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
    sentence_transformers = importlib.import_module("sentence_transformers")

    def factory(name: str, revision: str, code_revision: str, trust: bool) -> SentenceEncoder:
        return cast(
            SentenceEncoder,
            sentence_transformers.SentenceTransformer(
                name,
                revision=revision,
                device="cpu",
                trust_remote_code=trust,
                local_files_only=True,
                model_kwargs={"code_revision": code_revision},
                config_kwargs={"code_revision": code_revision},
            ),
        )

    provider = JinaEmbeddingProvider(identity, trust_remote_code=True, model_factory=factory)
    model = cast(Any, provider._load_model())
    model.eval()
    devices = sorted({parameter.device.type for parameter in model.parameters()})
    dtypes = sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
    runtime_module = type(model[0].auto_model).__module__
    _require(devices == ["cpu"] and dtypes == ["torch.float32"], "cpu_fp32_required")
    _require(identity.code_revision in runtime_module.split("."), "runtime_code_revision_mismatch")
    _require(model.max_seq_length == identity.max_length, "sequence_limit_mismatch")
    _require(
        torch.get_num_threads() == CPU_THREADS and torch.get_num_interop_threads() == 1,
        "cpu_thread_contract_mismatch",
    )
    profile: dict[str, JsonValue] = {
        "profile": "p7-cpu-fp32-batch1-threads8-v1",
        "device": "cpu",
        "dtype": "torch.float32",
        "parameter_devices": devices,
        "parameter_dtypes": cast(JsonValue, dtypes),
        "cpu_threads": CPU_THREADS,
        "interop_threads": 1,
        "batch_size": 1,
        "maximum_length": identity.max_length,
        "runtime_model_module": runtime_module,
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "local_files_only": True,
    }
    for package in ("sentence-transformers", "transformers", "numpy"):
        profile[package] = importlib.metadata.version(package)
    return provider, profile


def _vector_norm(values: tuple[float, ...], dimension: int) -> float:
    _require(len(values) == dimension, "vector_dimension_mismatch")
    _require(
        all(
            not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
            for v in values
        ),
        "non_finite_vector",
    )
    norm = math.sqrt(math.fsum(value * value for value in values))
    _require(abs(norm - 1.0) <= NORM_TOLERANCE, "vector_not_normalized")
    return norm


def _emit(**event: JsonValue) -> None:
    print(json.dumps(event, sort_keys=True, allow_nan=False), flush=True)


def run_probe(request: object, *, loader: CpuLoader | None = None) -> dict[str, JsonValue]:
    """Run exactly three numeric comparisons then up to eight independent queries."""
    report: dict[str, JsonValue] = {
        "probe_version": "gpu-cache-cpu-probe-v1",
        "status": "failed",
        "cpu_probe_passed": False,
        "formal_ready": False,
        "provider_requests_made": 0,
        "paid_requests_made": 0,
        "agent_constructed": False,
        "sample_scope": "three_deterministic_samples_not_full_cache_equivalence",
        "thresholds": {
            "cosine_minimum": COSINE_MINIMUM,
            "maximum_absolute_error": MAX_ABSOLUTE_ERROR,
            "normalization_tolerance": NORM_TOLERANCE,
        },
        "samples": [],
        "queries": [],
        "local_embedding_calls": 0,
    }
    stage = "request_validation"
    samples: list[JsonValue] = []
    queries: list[JsonValue] = []
    query_seconds: list[float] = []
    report["samples"] = samples
    report["queries"] = queries
    try:
        _offline()
        _require(
            not any(
                name in {"OPENAI_API_KEY", "DATABASE_URL"} or name.startswith("REPOPILOT_")
                for name in os.environ
            ),
            "runtime_credentials_or_overrides_forbidden",
        )
        parsed = ProbeRequest.model_validate(request)
        with no_network():
            stage = "model_load"
            _emit(stage="cpu_model_loading")
            started = time.monotonic()
            provider, runtime = (loader or create_cpu_provider)(parsed.identity)
            _require(provider.identity == parsed.identity, "embedding_identity_mismatch")
            report["model_load_seconds"] = round(time.monotonic() - started, 6)
            report["runtime"] = runtime
            executor = EmbeddingBatchExecutor(
                provider, batch_size=1, max_retries=0, retry_backoff_ms=0
            )
            _emit(stage="cpu_model_loaded", elapsed_seconds=report["model_load_seconds"])
            for index, sample in enumerate(parsed.samples):
                stage = "numeric_sample"
                _emit(stage="cpu_numeric_started", index=index)
                started = time.monotonic()
                report["local_embedding_calls"] = index + 1
                vector = executor.embed((sample.input,)).vectors[0].values
                elapsed = time.monotonic() - started
                cpu_norm = _vector_norm(vector, parsed.identity.dimension)
                gpu_norm = _vector_norm(sample.gpu_vector, parsed.identity.dimension)
                cosine = math.fsum(
                    left * right for left, right in zip(vector, sample.gpu_vector, strict=True)
                ) / (cpu_norm * gpu_norm)
                maximum = max(
                    abs(left - right) for left, right in zip(vector, sample.gpu_vector, strict=True)
                )
                passed = cosine >= COSINE_MINIMUM and maximum <= MAX_ABSOLUTE_ERROR
                samples.append(
                    {
                        "index": index,
                        "content_hash": sample.input.content_hash,
                        "input_characters": len(sample.input.content),
                        "elapsed_seconds": round(elapsed, 6),
                        "cosine_similarity": cosine,
                        "maximum_absolute_error": maximum,
                        "cpu_norm": cpu_norm,
                        "gpu_norm": gpu_norm,
                        "passed": passed,
                    }
                )
                _emit(stage="cpu_numeric_completed", index=index, passed=passed)
            _require(
                all(isinstance(sample, dict) and sample["passed"] is True for sample in samples),
                "numeric_tolerance_failed",
            )
            for index, query in enumerate(parsed.queries):
                stage = "query_timing"
                _emit(stage="cpu_query_started", index=index)
                started = time.monotonic()
                report["local_embedding_calls"] = 3 + index + 1
                vector = executor.embed((query,)).vectors[0].values
                elapsed = time.monotonic() - started
                norm = _vector_norm(vector, parsed.identity.dimension)
                query_seconds.append(elapsed)
                report["query_total_seconds"] = round(sum(query_seconds), 6)
                report["query_max_seconds"] = round(max(query_seconds), 6)
                queries.append(
                    {
                        "index": index,
                        "content_hash": query.content_hash,
                        "input_characters": len(query.content),
                        "elapsed_seconds": round(elapsed, 6),
                        "norm": norm,
                    }
                )
                _emit(stage="cpu_query_completed", index=index, elapsed_seconds=round(elapsed, 6))
        report["status"] = "passed"
        report["cpu_probe_passed"] = True
    except Exception as error:
        report["failure_code"] = (
            error.code if isinstance(error, CpuProbeError) else "cpu_probe_failed"
        )
        report["failure_stage"] = stage
        _emit(stage="cpu_probe_failed", failure_code=report["failure_code"])
    return report


def _safe_path(path: Path) -> Path:
    absolute = path.absolute()
    _require(".." not in absolute.parts, "unsafe_worker_path")
    _require(
        not any(item.is_symlink() for item in (absolute, *absolute.parents)), "unsafe_worker_path"
    )
    return absolute


def _unique_pairs(items: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    output: dict[str, JsonValue] = {}
    for key, value in items:
        _require(key not in output, "duplicate_request_key")
        output[key] = value
    return output


def _request_body(value: object) -> object:
    """Support the controller's sealed temporary file without loosening its payload."""
    if not isinstance(value, dict) or not ({"schema_version", "sha256"} & value.keys()):
        return value
    _require(
        type(value.get("schema_version")) is int and value["schema_version"] == 1,
        "request_schema_mismatch",
    )
    payload = {key: item for key, item in value.items() if key != "sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    _require(value.get("sha256") == hashlib.sha256(encoded).hexdigest(), "request_seal_mismatch")
    return {key: item for key, item in payload.items() if key != "schema_version"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        request_path = _safe_path(arguments.request)
        output_path = _safe_path(arguments.output)
        metadata = request_path.stat()
        _require(
            stat.S_ISREG(metadata.st_mode) and metadata.st_size <= MAX_REQUEST_BYTES,
            "invalid_request_file",
        )
        _require(not output_path.exists(), "output_already_exists")
        request = json.loads(request_path.read_bytes(), object_pairs_hook=_unique_pairs)
        report = run_probe(_request_body(request))
        payload = (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode()
        descriptor = os.open(
            output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
        return 0 if report["cpu_probe_passed"] is True else 1
    except Exception as error:
        _emit(
            stage="cpu_probe_worker_failed",
            failure_code=error.code if isinstance(error, CpuProbeError) else "worker_io_failed",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
