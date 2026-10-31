"""Read-only acceptance of the transferred GPU cache, without any model or API call.

The three expected seals identify the operator-reported artifact. Rebuilding all
eight source corpora proves coverage; it does not prove numerical CPU/GPU parity
or authorize an Agent run. Source text is returned only in three in-memory samples.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import JsonValue, ValidationError

from repopilot.embeddings import (
    EmbeddingIdentity,
    EmbeddingInput,
    EmbeddingVector,
    LocalEmbeddingCache,
    embedding_cache_key,
    embedding_identity_from_settings,
)
from repopilot.embeddings.cache import MAX_CACHE_ENTRY_BYTES, EmbeddingCacheRecord
from repopilot.retrieval import RepositoryIngestScanPolicy, ingest_repository
from repopilot.retrieval.corpus import RepositoryChunkCorpusBuilder
from repopilot.schemas import TaskSpec
from repopilot.tools.repository import ReadOnlyRepository
from scripts import p7_gpu_cache as gpu
from scripts import p7_gpu_cache_store as store
from scripts.p7_holdout_control import ENVIRONMENT_PATH, HoldoutEnvironmentManifest
from scripts.p7_repository_workload_preflight import (
    AGENT_INPUTS_SHA256,
    ENVIRONMENT_SHA256,
    MATRIX_SHA256,
)

EXPECTED_JOB_SHA256 = "acae68987d6d0dd9dcf05e01b2ade0777420d06e9f865e3927b02d5f4ff6baf3"
EXPECTED_INPUTS_SHA256 = "bb99f1ba68fd14be43a63e8cdb3a96d8002fd774836b6b83724bf3ea6b383ce3"
EXPECTED_COMPLETE_SHA256 = "ec4f3a90025b01ef22df8c20bf80b0d4b8ad822742c55699a53095b02d9c4214"
SAMPLE_POLICY = "unique-content-length-ascending-short-median-long-v1"


class CacheAcceptanceError(RuntimeError):
    """Only a stable, non-sensitive diagnostic crosses the worker boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise CacheAcceptanceError(code)


def _safe(path: Path) -> Path:
    try:
        return store._safe(path)
    except store.CacheStoreError:
        raise CacheAcceptanceError("unsafe_cache_path") from None


class ReadOnlyGpuCache(LocalEmbeddingCache):
    """Every corpus read is pinned; a miss cannot fall through to CPU embedding."""

    def __init__(
        self,
        root: Path,
        identity: EmbeddingIdentity,
        files: Mapping[str, str],
    ) -> None:
        super().__init__(_safe(root))
        self.identity = identity
        self._files = dict(files)
        self.hits = 0
        self.misses = 0
        self.write_attempts = 0

    def get(self, identity: EmbeddingIdentity, content_hash: str) -> EmbeddingVector:
        _require(identity == self.identity, "cache_identity_mismatch")
        key = embedding_cache_key(identity, content_hash)
        relative = f"vectors/{key[:2]}/{key}.json"
        expected = self._files.get(relative)
        target = _safe(self.root / key[:2] / f"{key}.json")
        if expected is None or not target.exists():
            self.misses += 1
            raise CacheAcceptanceError("corpus_cache_miss")
        try:
            # O_NOFOLLOW covers the final component; the parent walk above rejects
            # pre-existing links. Never create directories or recover writer files.
            descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as source:
                info = os.fstat(source.fileno())
                _require(
                    stat.S_ISREG(info.st_mode)
                    and info.st_nlink == 1
                    and info.st_size <= MAX_CACHE_ENTRY_BYTES,
                    "unsafe_cache_entry",
                )
                raw = source.read(MAX_CACHE_ENTRY_BYTES + 1)
            _require(hashlib.sha256(raw).hexdigest() == expected, "cache_entry_changed")
            record = EmbeddingCacheRecord.model_validate_json(raw)
            _require(
                record.identity == identity and record.vector.content_hash == content_hash,
                "cache_entry_identity_mismatch",
            )
            store._validate_vector(identity, record.vector)
        except (OSError, ValidationError, store.CacheStoreError):
            raise CacheAcceptanceError("cache_entry_invalid") from None
        self.hits += 1
        return record.vector

    def put(self, identity: EmbeddingIdentity, vector: EmbeddingVector) -> None:
        self.write_attempts += 1
        raise CacheAcceptanceError("cache_write_forbidden")


@dataclass(frozen=True)
class AcceptedGpuCache:
    cache: ReadOnlyGpuCache
    identity: EmbeddingIdentity
    job: dict[str, JsonValue]
    report: dict[str, JsonValue]


@dataclass(frozen=True)
class CoverageResult:
    report: dict[str, JsonValue]
    samples: tuple[EmbeddingInput, ...]
    gpu_vectors: tuple[EmbeddingVector, ...]
    cache: ReadOnlyGpuCache


def _pinned_documents(directory: Path) -> tuple[dict[str, JsonValue], ...]:
    try:
        documents = tuple(
            store._read(directory / name) for name in ("job.json", "inputs.json", "complete.json")
        )
    except store.CacheStoreError:
        raise CacheAcceptanceError("cache_metadata_invalid") from None
    for document, expected in zip(
        documents,
        (EXPECTED_JOB_SHA256, EXPECTED_INPUTS_SHA256, EXPECTED_COMPLETE_SHA256),
        strict=True,
    ):
        _require(document["sha256"] == expected, "cache_pin_mismatch")
    return documents


def verify_pinned_cache(
    directory: Path,
    expected_identity: EmbeddingIdentity | None = None,
) -> AcceptedGpuCache:
    """Verify exact sealed metadata before and after full read-only integrity checks."""
    directory = _safe(directory)
    before = _pinned_documents(directory)
    try:
        identity = store._identity(before[0])
        if expected_identity is not None:
            _require(identity == expected_identity, "cache_identity_mismatch")
        report = store.verify_cache(directory)
    except store.CacheStoreError:
        raise CacheAcceptanceError("cache_integrity_failed") from None
    after = _pinned_documents(directory)
    _require(before == after, "cache_metadata_changed")
    binding = after[0].get("binding")
    _require(isinstance(binding, dict), "cache_binding_invalid")
    binding = cast(dict[str, JsonValue], binding)
    _require(
        binding.get("agent_inputs_raw_sha256") == AGENT_INPUTS_SHA256
        and binding.get("matrix_sha256") == MATRIX_SHA256
        and binding.get("agent_constructed") is False
        and binding.get("formal_ready") is False
        and binding.get("openai_requests") == 0,
        "cache_binding_mismatch",
    )
    files = after[2].get("files")
    _require(isinstance(files, dict), "cache_file_manifest_invalid")
    cache = ReadOnlyGpuCache(directory / "vectors", identity, cast(dict[str, str], files))
    return AcceptedGpuCache(cache, identity, after[0], report)


def _check_checkout(checkout: Path, task: TaskSpec) -> None:
    """CPU checkouts have no required origin; their exact commit and bytes matter."""
    _require(checkout.is_dir() and not checkout.is_symlink(), "checkout_missing")
    _require(
        (checkout / ".git").is_dir() and not (checkout / ".git").is_symlink(),
        "unsafe_checkout_git_dir",
    )
    try:
        _require(
            Path(gpu._git(checkout, "rev-parse", "--show-toplevel")).resolve() == checkout.resolve()
            and gpu._git(checkout, "rev-parse", "HEAD") == task.base_commit
            and not gpu._git(
                checkout, "status", "--porcelain=v1", "--untracked-files=all", "--ignored"
            ),
            "checkout_identity_mismatch",
        )
    except (OSError, subprocess.SubprocessError):
        raise CacheAcceptanceError("checkout_inspection_failed") from None


def verify_coverage(
    root: Path,
    *,
    directory: Path | None = None,
    progress: Callable[[dict[str, JsonValue]], None] | None = None,
) -> CoverageResult:
    """Rebuild all portable corpus hashes, returning only three in-memory samples."""
    root = root.resolve()
    tasks, settings = gpu.load_tasks(root)
    environment_raw = gpu._contained(root, ENVIRONMENT_PATH).read_bytes()
    _require(
        hashlib.sha256(environment_raw).hexdigest() == ENVIRONMENT_SHA256,
        "environment_identity_mismatch",
    )
    environment = HoldoutEnvironmentManifest.model_validate_json(environment_raw)
    _require(
        environment.matrix_sha256 == MATRIX_SHA256
        and tuple((b.task_id, b.base_commit) for b in environment.bindings)
        == tuple((t.task_id, t.base_commit) for t in tasks),
        "environment_binding_mismatch",
    )
    directory = _safe(directory if directory is not None else root / gpu.JOB_PATH)
    if progress:
        progress({"stage": "cache_integrity_started"})
    accepted = verify_pinned_cache(directory, embedding_identity_from_settings(settings))
    if progress:
        progress({"stage": "cache_integrity_passed", **accepted.report})
    binding = cast(dict[str, JsonValue], accepted.job["binding"])
    expected_tasks = binding.get("tasks")
    _require(
        isinstance(expected_tasks, list) and len(expected_tasks) == len(tasks),
        "corpus_task_binding_mismatch",
    )
    expected_tasks = cast(list[JsonValue], expected_tasks)
    checkouts = tuple(
        gpu._contained(root, Path(item.checkout_relative_path)) for item in environment.bindings
    )
    for task, checkout in zip(tasks, checkouts, strict=True):
        _check_checkout(checkout, task)
    unique: dict[str, EmbeddingInput] = {}
    evidence: list[JsonValue] = []
    for task, checkout in zip(tasks, checkouts, strict=True):
        if progress:
            progress({"stage": "corpus_started", "task_id": task.task_id})
        repository = ReadOnlyRepository.from_settings(checkout, settings)
        manifest = ingest_repository(
            repository,
            repository_id="gpu-cache-source",
            commit=task.base_commit,
            scan_policy=RepositoryIngestScanPolicy(),
        )
        corpus = (
            RepositoryChunkCorpusBuilder.from_settings(repository, settings).build(manifest).corpus
        )
        for chunk in corpus.chunks:
            item = EmbeddingInput(content_hash=chunk.content_hash, content=chunk.content)
            previous = unique.setdefault(item.content_hash, item)
            _require(previous == item, "content_hash_conflict")
        record: dict[str, JsonValue] = {
            "task_id": task.task_id,
            "base_commit": task.base_commit,
            "file_count": len(manifest.files),
            "chunk_count": len(corpus.chunks),
            "source_files_sha256": gpu._digest(
                [[f.relative_path, f.content_hash, f.size_bytes] for f in manifest.files]
            ),
            "chunk_contents_sha256": gpu._digest(
                [[c.relative_path, c.start_line, c.end_line, c.content_hash] for c in corpus.chunks]
            ),
        }
        _require(record == expected_tasks[len(evidence)], "corpus_fingerprint_mismatch")
        evidence.append(record)
        _check_checkout(checkout, task)
        if progress:
            progress({"stage": "corpus_passed", **record})
    _require(evidence == binding.get("tasks"), "corpus_fingerprint_mismatch")
    ordered = tuple(
        sorted(unique.values(), key=lambda item: (len(item.content), item.content_hash))
    )
    hashes = [item.content_hash for item in ordered]
    _require(
        len(hashes) == binding.get("input_count")
        and store._digest(hashes) == binding.get("input_set_sha256"),
        "corpus_input_set_mismatch",
    )
    documents = _pinned_documents(directory)
    _require(documents[1].get("input_hashes") == hashes, "corpus_input_order_mismatch")
    _require(len(ordered) >= 3, "insufficient_sample_inputs")
    sample_indices = (0, len(ordered) // 2, len(ordered) - 1)
    samples = tuple(ordered[index] for index in sample_indices)
    vectors = tuple(accepted.cache.get(accepted.identity, item.content_hash) for item in samples)
    # Recheck source identity after the full scan, not just when each task finished.
    for task, checkout in zip(tasks, checkouts, strict=True):
        _check_checkout(checkout, task)
    verified_after = verify_pinned_cache(directory, accepted.identity)
    _require(verified_after.report == accepted.report, "cache_changed_during_coverage")
    report: dict[str, JsonValue] = {
        **accepted.report,
        "coverage_passed": True,
        "task_count": len(tasks),
        "tasks": evidence,
        "input_set_sha256": binding["input_set_sha256"],
        "gpu_runtime": binding.get("runtime"),
        "gpu_identity": accepted.job["identity"],
        "gpu_controller_commit": binding.get("controller_commit"),
        "gpu_execution_source_sha256": binding.get("execution_source_sha256"),
        "sample_policy": SAMPLE_POLICY,
        "samples": [
            {"content_hash": item.content_hash, "character_count": len(item.content)}
            for item in samples
        ],
        "sample_cache_hits": accepted.cache.hits,
        "cache_misses": accepted.cache.misses,
        "cache_write_attempts": accepted.cache.write_attempts,
        "model_loads": 0,
        "provider_requests": 0,
        "openai_requests": 0,
        "formal_ready": False,
    }
    return CoverageResult(report, samples, vectors, accepted.cache)
