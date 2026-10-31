"""Checkpointed, transferable embedding cache; no models, DBs or agents are loaded.

Callers must hold ``job_lock`` around bind/fill operations; export and import
acquire their own locks. Source text is only used in memory: exported artifacts
contain identities, hashes and vectors.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tarfile
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from pydantic import JsonValue, ValidationError

from repopilot.embeddings import (
    EmbeddingBatchExecutor,
    EmbeddingError,
    EmbeddingIdentity,
    EmbeddingInput,
    EmbeddingProvider,
    EmbeddingVector,
    LocalEmbeddingCache,
    embedding_cache_key,
)
from repopilot.embeddings.cache import MAX_CACHE_ENTRY_BYTES

BATCH_SIZE = 8
MAX_INPUTS = 200_000
MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024**3
_HASH = re.compile(r"[a-f0-9]{64}\Z")
_VECTOR = re.compile(r"vectors/([a-f0-9]{2})/([a-f0-9]{64})\.json\Z")
_TEMP = re.compile(r"\.([a-f0-9]{64})\.[^.]+\.tmp\Z")
_METADATA_PREFIX = ".repopilot-gpu-cache-metadata-v1."
_METADATA_TEMP = re.compile(
    re.escape(_METADATA_PREFIX)
    + r"(job\.json|inputs\.json|complete\.json)\.([a-f0-9]{64})\.[a-z0-9_]{8}\.tmp\Z"
)


class CacheStoreError(RuntimeError):
    """A secret-free operational failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _RegularTarInfo(tarfile.TarInfo):
    """Reject extended headers before tarfile can allocate their declared payload."""

    def _proc_member(self, tar: tarfile.TarFile) -> tarfile.TarInfo:
        if self.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
            raise CacheStoreError("unsafe_archive_member")
        return super()._proc_member(tar)  # type: ignore[no-any-return, misc]


def _bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _safe(path: Path) -> Path:
    path = path.absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise CacheStoreError("unsafe_path")
    return path


def _pairs(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise CacheStoreError("duplicate_json_key")
        result[key] = value
    return result


def _read(path: Path) -> dict[str, JsonValue]:
    _safe(path)
    try:
        if not path.is_file() or path.stat().st_size > MAX_METADATA_BYTES:
            raise CacheStoreError("invalid_document")
        value = json.loads(path.read_bytes(), object_pairs_hook=_pairs)
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise CacheStoreError("invalid_document")
        payload = {key: item for key, item in value.items() if key != "sha256"}
        if value.get("sha256") != _digest(payload):
            raise CacheStoreError("document_integrity_mismatch")
        return cast(dict[str, JsonValue], value)
    except (OSError, ValueError, TypeError):
        raise CacheStoreError("invalid_document") from None


def _write_once(path: Path, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    value = {"schema_version": 1, **payload}
    value["sha256"] = _digest(value)
    if path.exists() or path.is_symlink():
        if _read(path) != value:
            raise CacheStoreError("binding_mismatch")
        return value
    raw = _bytes(value) + b"\n"
    if len(raw) > MAX_METADATA_BYTES:
        raise CacheStoreError("metadata_too_large")
    temporary: Path | None = None
    try:
        prefix = f"{_METADATA_PREFIX}{path.name}.{value['sha256']}."
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=prefix, suffix=".tmp", delete=False
        ) as f:
            temporary = Path(f.name)
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)  # Atomic publication without overwriting an existing binding.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return value


def _recover_metadata_temporaries(directory: Path) -> None:
    """Recover only this writer's reserved temp names, while holding the job lock."""
    for path in directory.iterdir():
        match = _METADATA_TEMP.fullmatch(path.name)
        if match is None:
            continue
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink not in (1, 2)
        ):
            raise CacheStoreError("unsafe_metadata_temporary")
        if info.st_nlink == 2:
            # A crash after publication leaves exactly these two links. Never
            # unlink a temp that instead points to an unrelated file/inode.
            target = directory / match[1]
            try:
                published = target.lstat()
            except OSError:
                raise CacheStoreError("unsafe_metadata_temporary") from None
            if (
                not stat.S_ISREG(published.st_mode)
                or (published.st_dev, published.st_ino) != (info.st_dev, info.st_ino)
                or _read(target)["sha256"] != match[2]
            ):
                raise CacheStoreError("unsafe_metadata_temporary")
        # A single-link temp was never published and may contain a partial JSON
        # write. Its reserved name, owner and permissions establish our scope.
        path.unlink()


@contextmanager
def job_lock(directory: Path) -> Iterator[None]:
    """Fail immediately if another process holds this job's writer lock."""
    directory = _safe(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = _safe(directory / ".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CacheStoreError("unsafe_lock")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CacheStoreError("job_locked") from None
        _recover_metadata_temporaries(directory)
        yield
    finally:
        os.close(descriptor)


def bind_job(
    directory: Path, identity: EmbeddingIdentity, binding: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """Bind a dedicated directory to immutable, caller-supplied execution provenance."""
    directory = _safe(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not (directory / "job.json").exists() and any(
        p.name != ".lock" for p in directory.iterdir()
    ):
        raise CacheStoreError("job_directory_not_empty")
    return _write_once(
        directory / "job.json", {"identity": identity.model_dump(mode="json"), "binding": binding}
    )


def _identity(job: dict[str, JsonValue]) -> EmbeddingIdentity:
    try:
        return EmbeddingIdentity.model_validate(job["identity"])
    except (KeyError, ValidationError):
        raise CacheStoreError("invalid_identity") from None


def _vector_path(identity: EmbeddingIdentity, content_hash: str) -> str:
    key = embedding_cache_key(identity, content_hash)
    return f"vectors/{key[:2]}/{key}.json"


def _validate_vector(identity: EmbeddingIdentity, vector: EmbeddingVector) -> None:
    if len(vector.values) != identity.dimension or not all(map(math.isfinite, vector.values)):
        raise CacheStoreError("invalid_vector")
    if identity.normalize and abs(math.sqrt(math.fsum(x * x for x in vector.values)) - 1) > 1e-4:
        raise CacheStoreError("vector_not_normalized")


def _inventory(directory: Path) -> dict[str, str]:
    """Inventory regular files, rejecting links and noncanonical directory names."""
    files: dict[str, str] = {}
    total = 0
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            if relative != "vectors" and not re.fullmatch(r"vectors/[a-f0-9]{2}", relative):
                raise CacheStoreError("unexpected_directory")
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CacheStoreError("unsafe_file")
        if relative == ".lock":
            continue
        limit = MAX_CACHE_ENTRY_BYTES if relative.startswith("vectors/") else MAX_METADATA_BYTES
        if info.st_size > limit:
            raise CacheStoreError("file_too_large")
        total += info.st_size
        if len(files) >= MAX_INPUTS + 3 or total > MAX_TOTAL_BYTES:
            raise CacheStoreError("cache_too_large")
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def _hashes(document: dict[str, JsonValue]) -> list[str]:
    hashes = document.get("input_hashes")
    if (
        not isinstance(hashes, list)
        or not 0 < len(hashes) <= MAX_INPUTS
        or any(not isinstance(h, str) or not _HASH.fullmatch(h) for h in hashes)
    ):
        raise CacheStoreError("invalid_inputs")
    result = cast(list[str], hashes)
    if len(set(result)) != len(result):
        raise CacheStoreError("duplicate_inputs")
    return result


def _validate_input_binding(job: dict[str, JsonValue], hashes: list[str]) -> None:
    binding = job.get("binding")
    if (
        not isinstance(binding, dict)
        or type(binding.get("input_count")) is not int
        or binding.get("input_count") != len(hashes)
        or binding.get("input_set_sha256") != _digest(hashes)
    ):
        raise CacheStoreError("input_binding_mismatch")


def fill_cache(
    directory: Path,
    identity: EmbeddingIdentity,
    inputs: Sequence[EmbeddingInput],
    provider: EmbeddingProvider,
    *,
    max_batches: int | None = None,
    progress: Callable[[dict[str, JsonValue]], None] | None = None,
) -> dict[str, JsonValue]:
    """Persist each batch before starting another; resume only missing content."""
    directory = _safe(directory)
    job = _read(directory / "job.json")
    if _identity(job) != identity or provider.identity != identity:
        raise CacheStoreError("identity_mismatch")
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise CacheStoreError("invalid_batch_limit")
    unique: dict[str, EmbeddingInput] = {}
    for item in inputs:
        if hashlib.sha256(item.content.encode()).hexdigest() != item.content_hash:
            raise CacheStoreError("input_hash_mismatch")
        if item.content_hash in unique and unique[item.content_hash] != item:
            raise CacheStoreError("input_hash_conflict")
        unique.setdefault(item.content_hash, item)
    hashes = _hashes({"input_hashes": list(unique)})
    _validate_input_binding(job, hashes)
    input_document = _write_once(
        directory / "inputs.json", {"input_hashes": cast(JsonValue, hashes)}
    )
    if (directory / "complete.json").exists():
        return verify_cache(directory)
    expected = {_vector_path(identity, h) for h in hashes}
    # Only remove interrupted temporary writes belonging to this exact input set.
    vector_root = directory / "vectors"
    _safe(vector_root)
    for path in vector_root.glob("*/*"):
        match = _TEMP.fullmatch(path.name)
        if match and f"vectors/{path.parent.name}/{match[1]}.json" in expected:
            _safe(path)
            if not path.is_file() or path.stat().st_nlink != 1:
                raise CacheStoreError("unsafe_file")
            path.unlink()
    if set(_inventory(directory)) - expected - {"job.json", "inputs.json"}:
        raise CacheStoreError("unexpected_files")
    cache = LocalEmbeddingCache(vector_root)
    misses: list[EmbeddingInput] = []
    for item in unique.values():
        cached = cache.get(identity, item.content_hash)
        if cached is None:
            misses.append(item)
        else:
            _validate_vector(identity, cached)
    initial_hits, batches = len(hashes) - len(misses), 0
    executor = EmbeddingBatchExecutor(
        provider, batch_size=BATCH_SIZE, max_retries=0, retry_backoff_ms=0
    )
    completed = initial_hits
    for offset in range(0, len(misses), BATCH_SIZE):
        if max_batches is not None and batches >= max_batches:
            break
        result = executor.embed(misses[offset : offset + BATCH_SIZE])
        for vector in result.vectors:
            _validate_vector(identity, vector)
        for vector in result.vectors:
            cache.put(identity, vector)
        completed += len(result.vectors)
        batches += 1
        if progress:
            progress(
                {
                    "status": "running",
                    "completed_vectors": completed,
                    "total_vectors": len(hashes),
                    "batches_this_invocation": batches,
                }
            )
    status: dict[str, JsonValue] = {
        "status": "complete" if completed == len(hashes) else "paused",
        "completed_vectors": completed,
        "total_vectors": len(hashes),
        "cache_hits": initial_hits,
        "batches_this_invocation": batches,
    }
    if completed == len(hashes):
        inventory = _inventory(directory)
        if set(inventory) != expected | {"job.json", "inputs.json"}:
            raise CacheStoreError("unexpected_files")
        _write_once(
            directory / "complete.json",
            {
                "job_sha256": job["sha256"],
                "inputs_sha256": input_document["sha256"],
                "files": {name: inventory[name] for name in sorted(expected)},
            },
        )
        verify_cache(directory)
    return status


def verify_cache(directory: Path) -> dict[str, JsonValue]:
    """Validate all bindings, exact inventory, vector integrity and normalization."""
    directory = _safe(directory)
    job, inputs, complete = (
        _read(directory / name) for name in ("job.json", "inputs.json", "complete.json")
    )
    identity, hashes = _identity(job), _hashes(inputs)
    _validate_input_binding(job, hashes)
    if (
        complete.get("job_sha256") != job["sha256"]
        or complete.get("inputs_sha256") != inputs["sha256"]
    ):
        raise CacheStoreError("binding_mismatch")
    expected = {_vector_path(identity, h) for h in hashes}
    inventory = _inventory(directory)
    if set(inventory) != expected | {"job.json", "inputs.json", "complete.json"}:
        raise CacheStoreError("unexpected_files")
    expected_directories = {"vectors", *(str(Path(name).parent) for name in expected)}
    if {
        str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_dir()
    } != expected_directories:
        raise CacheStoreError("unexpected_directory")
    if complete.get("files") != {name: inventory[name] for name in sorted(expected)}:
        raise CacheStoreError("file_integrity_mismatch")
    cache = LocalEmbeddingCache(directory / "vectors")
    for content_hash in hashes:
        try:
            vector = cache.get(identity, content_hash)
        except EmbeddingError:
            raise CacheStoreError("cache_corrupt") from None
        if vector is None:
            raise CacheStoreError("cache_incomplete")
        _validate_vector(identity, vector)
    return {
        "status": "complete",
        "completed_vectors": len(hashes),
        "total_vectors": len(hashes),
        "job_sha256": job["sha256"],
        "inputs_sha256": inputs["sha256"],
        "complete_sha256": complete["sha256"],
    }


def export_cache(directory: Path, archive: Path) -> dict[str, JsonValue]:
    """Serialize export against writers; callers must not already hold the lock."""
    with job_lock(directory):
        return _export_locked(directory, archive)


def _export_locked(directory: Path, archive: Path) -> dict[str, JsonValue]:
    """Export only a fully verified cache, refusing any existing destination."""
    result = verify_cache(directory)
    archive = _safe(archive)
    if archive.exists():
        raise CacheStoreError("archive_exists")
    if archive.is_relative_to(directory.absolute()):
        raise CacheStoreError("archive_inside_job")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=archive.parent, prefix=".gpu-cache-", delete=False
        ) as f:
            temporary = Path(f.name)
            with tarfile.open(
                fileobj=f, mode="w:gz", format=tarfile.USTAR_FORMAT, compresslevel=1
            ) as tar:
                for name in sorted(_inventory(directory)):
                    info = tar.gettarinfo(str(directory / name), arcname=name)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o600
                    with (directory / name).open("rb") as source:
                        tar.addfile(info, source)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, archive)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {**result, "archive_sha256": _file_digest(archive)}


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def import_cache(archive: Path, destination: Path) -> dict[str, JsonValue]:
    """Serialize installers separately from the destination's future job lock."""
    destination = _safe(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _safe(destination.parent / f".{destination.name}.import.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CacheStoreError("unsafe_lock")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CacheStoreError("job_locked") from None
        return _import_locked(archive, destination)
    finally:
        os.close(descriptor)


def _import_locked(archive: Path, destination: Path) -> dict[str, JsonValue]:
    """Extract defensively into an owned temporary directory; never merge caches."""
    archive, destination = _safe(archive), _safe(destination)
    if not archive.is_file() or archive.stat().st_size > MAX_TOTAL_BYTES:
        raise CacheStoreError("invalid_archive")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise CacheStoreError("destination_not_empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".gpu-import-"))
    try:
        total, names = 0, set()
        with tarfile.open(archive, mode="r|gz", tarinfo=_RegularTarInfo) as tar:
            for member in tar:
                match = _VECTOR.fullmatch(member.name)
                metadata = member.name in {"job.json", "inputs.json", "complete.json"}
                limit = MAX_METADATA_BYTES if metadata else MAX_CACHE_ENTRY_BYTES
                if (
                    not member.isfile()
                    or member.sparse
                    or member.name in names
                    or not (metadata or (match and match[1] == match[2][:2]))
                    or not 0 < member.size <= limit
                ):
                    raise CacheStoreError("unsafe_archive_member")
                names.add(member.name)
                total += member.size
                if len(names) > MAX_INPUTS + 3 or total > MAX_TOTAL_BYTES:
                    raise CacheStoreError("archive_too_large")
                target = temporary / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise CacheStoreError("invalid_archive")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if target.stat().st_size != member.size:
                    raise CacheStoreError("invalid_archive")
        result = verify_cache(temporary)
        _safe(destination)
        if destination.exists() and any(destination.iterdir()):
            raise CacheStoreError("destination_not_empty")
        temporary.replace(destination)
        return {**result, "archive_sha256": _file_digest(archive)}
    except (tarfile.TarError, OSError, EOFError):
        raise CacheStoreError("invalid_archive") from None
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
