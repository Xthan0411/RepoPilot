"""Offline checkpoint, integrity, and hostile archive tests for the GPU worker."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

from repopilot.embeddings import (
    EmbeddingError,
    EmbeddingIdentity,
    EmbeddingInput,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    FakeEmbeddingProvider,
    LocalEmbeddingCache,
    embedding_cache_key,
)
from repopilot.enums import EmbeddingErrorCode, EmbeddingProviderKind

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import p7_gpu_cache_store as store  # noqa: E402


def _identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(
        provider=EmbeddingProviderKind.FAKE,
        model="fake-code-embedding",
        revision="v1",
        dimension=4,
        normalize=True,
        max_length=2048,
    )


def _inputs(count: int = 10) -> tuple[EmbeddingInput, ...]:
    return tuple(
        EmbeddingInput(
            content=f"source content {i}",
            content_hash=hashlib.sha256(f"source content {i}".encode()).hexdigest(),
        )
        for i in range(count)
    )


def _bind(directory: Path, count: int = 10) -> None:
    store.bind_job(
        directory,
        _identity(),
        {
            "device": "fake-test",
            "input_count": count,
            "input_set_sha256": store._digest([i.content_hash for i in _inputs(count)]),
        },
    )


def _complete(directory: Path, count: int = 10) -> None:
    with store.job_lock(directory):
        _bind(directory, count)
        store.fill_cache(directory, _identity(), _inputs(count), FakeEmbeddingProvider(_identity()))


class FailsSecondBatch(FakeEmbeddingProvider):
    attempts = 0

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.attempts += 1
        if self.attempts == 2:
            raise EmbeddingError(EmbeddingErrorCode.PROVIDER_FAILURE, retryable=True)
        return super().embed(request)


def test_failed_second_batch_keeps_first_and_resumes_only_misses(tmp_path: Path) -> None:
    _bind(tmp_path, 19)
    provider = FailsSecondBatch(_identity())
    with pytest.raises(EmbeddingError):
        store.fill_cache(tmp_path, _identity(), _inputs(19), provider)
    assert provider.attempts == 2  # No automatic retries, even for a retryable error.
    assert len(list((tmp_path / "vectors").rglob("*.json"))) == 8
    assert not (tmp_path / "complete.json").exists()
    resumed = FakeEmbeddingProvider(_identity())
    result = store.fill_cache(tmp_path, _identity(), _inputs(19), resumed)
    assert result["status"] == "complete"
    assert result["cache_hits"] == 8
    assert [len(r.inputs) for r in resumed.requests] == [8, 3]
    assert store.verify_cache(tmp_path)["total_vectors"] == 19


def test_pause_does_not_claim_completion_and_resume_is_idempotent(tmp_path: Path) -> None:
    _bind(tmp_path)
    progress: list[dict[str, object]] = []
    result = store.fill_cache(
        tmp_path,
        _identity(),
        _inputs(),
        FakeEmbeddingProvider(_identity()),
        max_batches=1,
        progress=lambda item: progress.append(dict(item)),
    )
    assert result["status"] == "paused"
    assert progress[0]["completed_vectors"] == 8
    assert not (tmp_path / "complete.json").exists()
    provider = FakeEmbeddingProvider(_identity())
    store.fill_cache(tmp_path, _identity(), _inputs(), provider)
    assert provider.calls_made == 1
    store.fill_cache(tmp_path, _identity(), _inputs(), provider)
    assert provider.calls_made == 1
    assert "source content" not in (tmp_path / "job.json").read_text()
    assert "source content" not in (tmp_path / "inputs.json").read_text()


def test_partial_batch_write_and_orphan_temporary_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bind(tmp_path)
    original = LocalEmbeddingCache.put
    calls = 0

    def interrupted(
        self: LocalEmbeddingCache, identity: EmbeddingIdentity, vector: EmbeddingVector
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("simulated interrupted persistence")
        original(self, identity, vector)

    with monkeypatch.context() as patch:
        patch.setattr(LocalEmbeddingCache, "put", interrupted)
        with pytest.raises(OSError):
            store.fill_cache(tmp_path, _identity(), _inputs(), FakeEmbeddingProvider(_identity()))
    assert len(list((tmp_path / "vectors").rglob("*.json"))) == 3
    key = embedding_cache_key(_identity(), _inputs()[3].content_hash)
    temporary = tmp_path / "vectors" / key[:2] / f".{key}.interrupted.tmp"
    temporary.parent.mkdir(exist_ok=True)
    temporary.write_bytes(b"incomplete json")
    provider = FakeEmbeddingProvider(_identity())
    result = store.fill_cache(tmp_path, _identity(), _inputs(), provider)
    assert result["cache_hits"] == 3
    assert provider.calls_made == 1
    assert len(provider.requests[0].inputs) == 7
    assert not temporary.exists()
    assert store.verify_cache(tmp_path)["status"] == "complete"


@pytest.mark.parametrize("control", ["job.json", "inputs.json", "complete.json"])
@pytest.mark.parametrize("published", [False, True])
def test_metadata_publication_interruption_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str, published: bool
) -> None:
    original_link, original_unlink = os.link, Path.unlink
    prefix = f"{store._METADATA_PREFIX}{control}."

    def interrupted_link(source: Path, target: Path) -> None:
        if target.name == control:
            if published:
                original_link(source, target)
            raise OSError("simulated abrupt process exit")
        original_link(source, target)

    def skip_crash_cleanup(path: Path, missing_ok: bool = False) -> None:
        if not path.name.startswith(prefix):
            original_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", interrupted_link)
        # A killed process does not execute _write_once's finally block.
        patch.setattr(Path, "unlink", skip_crash_cleanup)
        with pytest.raises(OSError, match="abrupt"), store.job_lock(tmp_path):
            _bind(tmp_path)
            store.fill_cache(tmp_path, _identity(), _inputs(), FakeEmbeddingProvider(_identity()))
    orphan = next(tmp_path.glob(f"{prefix}*"))
    assert orphan.stat().st_nlink == (2 if published else 1)
    assert (tmp_path / control).exists() is published
    provider = FakeEmbeddingProvider(_identity())
    with store.job_lock(tmp_path):
        assert not orphan.exists()
        _bind(tmp_path)
        result = store.fill_cache(tmp_path, _identity(), _inputs(), provider)
    assert result["status"] == "complete"
    assert provider.calls_made == (0 if control == "complete.json" else 2)
    assert store.verify_cache(tmp_path)["total_vectors"] == 10


def _metadata_temporary(directory: Path, control: str = "job.json", digest: str = "0" * 64) -> Path:
    return directory / f"{store._METADATA_PREFIX}{control}.{digest}.abc12345.tmp"


def test_partial_metadata_write_is_recovered_before_first_binding(tmp_path: Path) -> None:
    orphan = _metadata_temporary(tmp_path)
    orphan.write_bytes(b'{"schema_version":')
    orphan.chmod(0o600)
    with store.job_lock(tmp_path):
        assert not orphan.exists()
        _bind(tmp_path)


@pytest.mark.parametrize("name", [".metadata-user-notes", ".metadata-abcdefgh"])
def test_unowned_metadata_files_are_never_cleaned(tmp_path: Path, name: str) -> None:
    unrelated = tmp_path / name
    unrelated.write_text("user data")
    with store.job_lock(tmp_path), pytest.raises(store.CacheStoreError, match="not_empty"):
        _bind(tmp_path)
    assert unrelated.read_text() == "user data"


@pytest.mark.parametrize("kind", ["symlink", "wrong_mode", "unrelated_hardlink", "third_link"])
def test_unsafe_metadata_temporaries_are_preserved(tmp_path: Path, kind: str) -> None:
    directory = tmp_path / "job"
    directory.mkdir()
    unrelated = tmp_path / "user-file"
    unrelated.write_text("user data")
    unrelated.chmod(0o600)
    orphan = _metadata_temporary(directory)
    if kind == "symlink":
        orphan.symlink_to(unrelated)
    elif kind == "wrong_mode":
        orphan.write_text("partial")
        orphan.chmod(0o644)
    else:
        os.link(unrelated, orphan)
        if kind == "third_link":
            os.link(unrelated, directory / "job.json")
    with (
        pytest.raises(store.CacheStoreError, match="unsafe_metadata_temporary"),
        store.job_lock(directory),
    ):
        pytest.fail("unsafe temporary accepted")
    assert orphan.exists()
    assert unrelated.read_text() == "user data"


def test_metadata_published_link_must_match_named_payload_hash(tmp_path: Path) -> None:
    _bind(tmp_path)
    orphan = _metadata_temporary(tmp_path)
    os.link(tmp_path / "job.json", orphan)
    with (
        pytest.raises(store.CacheStoreError, match="unsafe_metadata_temporary"),
        store.job_lock(tmp_path),
    ):
        pytest.fail("wrong metadata digest accepted")
    assert orphan.exists()
    assert (tmp_path / "job.json").exists()


def test_export_recovers_completed_metadata_publication(tmp_path: Path) -> None:
    directory = tmp_path / "job"
    _complete(directory)
    completed = json.loads((directory / "complete.json").read_bytes())
    orphan = _metadata_temporary(directory, "complete.json", completed["sha256"])
    os.link(directory / "complete.json", orphan)
    assert store.export_cache(directory, tmp_path / "cache.tar.gz")["status"] == "complete"
    assert not orphan.exists()
    assert (directory / "complete.json").stat().st_nlink == 1


def test_job_binding_and_input_order_are_immutable(tmp_path: Path) -> None:
    _bind(tmp_path)
    _bind(tmp_path)
    with pytest.raises(store.CacheStoreError, match="binding_mismatch"):
        store.bind_job(tmp_path, _identity(), {"device": "different"})
    with pytest.raises(store.CacheStoreError, match="input_binding_mismatch"):
        store.fill_cache(
            tmp_path, _identity(), tuple(reversed(_inputs())), FakeEmbeddingProvider(_identity())
        )
    with pytest.raises(store.CacheStoreError, match="input_binding_mismatch"):
        store.fill_cache(tmp_path, _identity(), _inputs(3), FakeEmbeddingProvider(_identity()))
    assert not (tmp_path / "inputs.json").exists()


def test_invalid_input_hash_never_reaches_provider(tmp_path: Path) -> None:
    _bind(tmp_path, 1)
    altered = _inputs(1)[0].model_copy(update={"content": "tampered"})
    provider = FakeEmbeddingProvider(_identity())
    with pytest.raises(store.CacheStoreError, match="input_hash_mismatch"):
        store.fill_cache(tmp_path, _identity(), (altered,), provider)
    assert provider.calls_made == 0


def test_non_normalized_provider_vectors_are_not_saved(tmp_path: Path) -> None:
    class Unnormalized(FakeEmbeddingProvider):
        def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
            response = super().embed(request)
            return response.model_copy(
                update={
                    "vectors": tuple(
                        v.model_copy(update={"values": tuple(x * 2 for x in v.values)})
                        for v in response.vectors
                    )
                }
            )

    _bind(tmp_path)
    with pytest.raises(store.CacheStoreError, match="vector_not_normalized"):
        store.fill_cache(tmp_path, _identity(), _inputs(), Unnormalized(_identity()))
    assert not list((tmp_path / "vectors").rglob("*.json"))


def test_lock_is_exclusive_reusable_and_rejects_symlinks(tmp_path: Path) -> None:
    directory = tmp_path / "job"
    with (
        store.job_lock(directory),
        pytest.raises(store.CacheStoreError, match="job_locked"),
        store.job_lock(directory),
    ):
        pytest.fail("second lock acquired")
    with store.job_lock(directory):
        pass
    link = tmp_path / "alias"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(store.CacheStoreError, match="unsafe_path"), store.job_lock(link):
        pytest.fail("linked directory accepted")


def test_incomplete_or_corrupt_cache_cannot_export(tmp_path: Path) -> None:
    job, archive = tmp_path / "job", tmp_path / "cache.tar.gz"
    _bind(job)
    with pytest.raises(store.CacheStoreError):
        store.export_cache(job, archive)
    assert not archive.exists()
    store.fill_cache(job, _identity(), _inputs(), FakeEmbeddingProvider(_identity()))
    vector = next((job / "vectors").rglob("*.json"))
    vector.write_bytes(vector.read_bytes() + b" ")
    with pytest.raises(store.CacheStoreError, match="file_integrity_mismatch"):
        store.export_cache(job, archive)
    assert not archive.exists()


def test_complete_inventory_rejects_extra_file_empty_directory_and_link(tmp_path: Path) -> None:
    _complete(tmp_path, 1)
    extra = tmp_path / "extra.json"
    extra.write_text("{}")
    with pytest.raises(store.CacheStoreError, match="unexpected_files"):
        store.verify_cache(tmp_path)
    extra.unlink()
    expected_prefix = embedding_cache_key(_identity(), _inputs(1)[0].content_hash)[:2]
    extra_directory = tmp_path / "vectors" / ("ff" if expected_prefix != "ff" else "aa")
    extra_directory.mkdir()
    with pytest.raises(store.CacheStoreError, match="unexpected_directory"):
        store.verify_cache(tmp_path)
    extra_directory.rmdir()
    extra.symlink_to(tmp_path / "job.json")
    with pytest.raises(store.CacheStoreError, match="unsafe_file"):
        store.verify_cache(tmp_path)


def test_resealed_subset_manifest_is_rejected_by_job_binding(tmp_path: Path) -> None:
    _complete(tmp_path)
    inputs = json.loads((tmp_path / "inputs.json").read_text())
    inputs["input_hashes"] = inputs["input_hashes"][:1]
    inputs.pop("sha256")
    inputs["sha256"] = store._digest(inputs)
    (tmp_path / "inputs.json").write_text(json.dumps(inputs))
    with pytest.raises(store.CacheStoreError, match="input_binding_mismatch"):
        store.verify_cache(tmp_path)


def test_archive_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    source, destination, archive = (tmp_path / p for p in ("source", "destination", "cache.tar.gz"))
    _complete(source)
    exported = store.export_cache(source, archive)
    destination.mkdir()
    imported = store.import_cache(archive, destination)
    assert imported == exported
    assert store.verify_cache(destination) == store.verify_cache(source)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        assert ".lock" not in names
        assert len(names) == 13
    with pytest.raises(store.CacheStoreError, match="archive_exists"):
        store.export_cache(source, archive)
    with pytest.raises(store.CacheStoreError, match="destination_not_empty"):
        store.import_cache(archive, destination)


def _archive(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/tmp/escape",
        "vectors/aa/../escape",
        "job.json/",
        "./job.json",
        "vectors\\aa\\x.json",
    ],
)
def test_import_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    archive, destination = tmp_path / "bad.tar.gz", tmp_path / "destination"
    member = tarfile.TarInfo(name)
    member.size = 2
    _archive(archive, [(member, b"{}")])
    with pytest.raises(store.CacheStoreError, match="unsafe_archive_member"):
        store.import_cache(archive, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".gpu-import-*"))


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE]
)
def test_import_rejects_nonregular_members(tmp_path: Path, kind: bytes) -> None:
    archive = tmp_path / "bad.tar.gz"
    member = tarfile.TarInfo("job.json")
    member.type, member.linkname = kind, "../../outside"
    _archive(archive, [(member, b"")])
    with pytest.raises(store.CacheStoreError, match="unsafe_archive_member"):
        store.import_cache(archive, tmp_path / "destination")


def test_import_rejects_duplicate_and_oversize_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, destination = tmp_path / "bad.tar.gz", tmp_path / "destination"
    member = tarfile.TarInfo("job.json")
    member.size = 2
    _archive(archive, [(member, b"{}"), (member, b"{}")])
    with pytest.raises(store.CacheStoreError, match="unsafe_archive_member"):
        store.import_cache(archive, destination)
    monkeypatch.setattr(store, "MAX_METADATA_BYTES", 1)
    _archive(archive, [(member, b"{}")])
    with pytest.raises(store.CacheStoreError, match="unsafe_archive_member"):
        store.import_cache(archive, destination)


def test_import_failure_preserves_existing_contents(tmp_path: Path) -> None:
    destination = tmp_path / "existing-cpu-cache"
    destination.mkdir()
    marker = destination / "keep"
    marker.write_text("user data")
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(b"not gzip")
    with pytest.raises(store.CacheStoreError, match="destination_not_empty"):
        store.import_cache(archive, destination)
    assert marker.read_text() == "user data"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(store.CacheStoreError, match="invalid_archive"):
        store.import_cache(archive, empty)
    assert empty.is_dir() and not list(empty.iterdir())
    assert not list(tmp_path.glob(".gpu-import-*"))


@pytest.mark.parametrize(
    "kind",
    [
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
        tarfile.GNUTYPE_LONGNAME,
        tarfile.GNUTYPE_LONGLINK,
        tarfile.GNUTYPE_SPARSE,
    ],
)
def test_import_rejects_extended_headers_before_payload_allocation(
    tmp_path: Path, kind: bytes
) -> None:
    import gzip

    member = tarfile.TarInfo("extended-header")
    member.type = kind
    member.size = store.MAX_TOTAL_BYTES + 1
    archive = tmp_path / "header-bomb.tar.gz"
    # No huge body is needed: the header itself must be rejected before it is read.
    archive.write_bytes(gzip.compress(member.tobuf(format=tarfile.GNU_FORMAT)))
    with pytest.raises(store.CacheStoreError, match="unsafe_archive_member"):
        store.import_cache(archive, tmp_path / "destination")
    assert not list(tmp_path.glob(".gpu-import-*"))


def test_import_enforces_total_extracted_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "total-limit.tar.gz"
    first, second = tarfile.TarInfo("job.json"), tarfile.TarInfo("inputs.json")
    first.size = second.size = 512
    _archive(archive, [(first, b"a" * 512), (second, b"b" * 512)])
    assert archive.stat().st_size < 700
    monkeypatch.setattr(store, "MAX_TOTAL_BYTES", 700)
    with pytest.raises(store.CacheStoreError, match="archive_too_large"):
        store.import_cache(archive, tmp_path / "destination")
    assert not list(tmp_path.glob(".gpu-import-*"))


def test_normalization_is_rechecked_on_resume(tmp_path: Path) -> None:
    _bind(tmp_path)
    vector = EmbeddingVector(content_hash=_inputs()[0].content_hash, values=(1.0, 1.0, 1.0, 1.0))
    LocalEmbeddingCache(tmp_path / "vectors").put(_identity(), vector)
    provider = FakeEmbeddingProvider(_identity())
    with pytest.raises(store.CacheStoreError, match="vector_not_normalized"):
        store.fill_cache(tmp_path, _identity(), _inputs(), provider)
    assert provider.calls_made == 0


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_batch_limit_is_rejected(tmp_path: Path, limit: int) -> None:
    _bind(tmp_path)
    with pytest.raises(store.CacheStoreError, match="invalid_batch_limit"):
        store.fill_cache(
            tmp_path, _identity(), _inputs(), FakeEmbeddingProvider(_identity()), max_batches=limit
        )
