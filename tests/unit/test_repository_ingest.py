"""Determinism, safety, and artifact tests for repository ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from repopilot.enums import (
    ErrorType,
    RepositoryEntryType,
    RepositoryErrorCode,
    RepositoryIngestErrorCode,
)
from repopilot.retrieval import (
    RepositoryIgnorePolicy,
    RepositoryIngestError,
    RepositoryIngestor,
    RepositoryIngestScanPolicy,
    RepositoryManifest,
    RepositorySource,
    SkippedRepositorySource,
    ingest_repository,
    read_repository_manifest,
    source_id_for,
    verify_repository_manifest_integrity,
    write_repository_manifest,
)
from repopilot.retrieval.manifest import repository_manifest_content_sha256
from repopilot.tools import (
    DirectoryListing,
    ReadOnlyRepository,
    RepositoryAccessError,
    RepositoryAccessPolicy,
    RepositoryTextFile,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "repositories" / "read_only"
REPOSITORY_ID = "fixture.repo"
COMMIT = "a" * 40


def build_fixture_manifest() -> RepositoryManifest:
    return ingest_repository(
        ReadOnlyRepository(FIXTURE_ROOT),
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )


def assert_ingest_error(
    captured: pytest.ExceptionInfo[RepositoryIngestError],
    code: RepositoryIngestErrorCode,
) -> None:
    assert captured.value.code is code


def test_default_ignore_policy_is_versioned_sorted_and_security_preserving() -> None:
    policy = RepositoryIgnorePolicy()

    assert policy.policy_version == 1
    assert policy.ignored_names == tuple(sorted(policy.ignored_names))
    assert {".env", ".git", ".hg", ".svn", ".venv", "__pycache__"}.issubset(policy.ignored_names)


@pytest.mark.parametrize(
    "ignored_names",
    [
        (".env", ".git", ".hg"),
        (".env", ".git", ".hg", ".svn", ".svn"),
        (".env", ".git", ".hg", ".svn", ""),
        (".env", ".git", ".hg", ".svn", "nested/name"),
        (".env", ".git", ".hg", ".svn", ".."),
    ],
)
def test_ignore_policy_rejects_missing_security_names_and_invalid_components(
    ignored_names: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError):
        RepositoryIgnorePolicy(ignored_names=ignored_names)


def test_ingestor_merges_custom_repository_denials_into_manifest(tmp_path: Path) -> None:
    tmp_path.joinpath("visible.py").write_text("visible = True\n", encoding="utf-8")
    tmp_path.joinpath("private").mkdir()
    tmp_path.joinpath("private", "secret.py").write_text("secret = True\n", encoding="utf-8")
    repository = ReadOnlyRepository(
        tmp_path,
        policy=RepositoryAccessPolicy(denied_names=(".env", ".git", ".hg", ".svn", "private")),
    )

    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )

    assert "private" in manifest.ignore_policy.ignored_names
    assert tuple(source.relative_path for source in manifest.files) == ("visible.py",)


def test_source_id_is_stable_and_separates_repository_commit_and_path() -> None:
    expected = source_id_for(REPOSITORY_ID, COMMIT, "pkg/说明.py")

    assert expected == "src_bfdb1f7537af726651957d639ede53acb6a2153cdf0ac3d183678c81474e3ea4"
    assert expected == source_id_for(REPOSITORY_ID, COMMIT, "pkg/说明.py")
    assert expected.startswith("src_")
    assert len(expected) == 68
    assert expected != source_id_for("other.repo", COMMIT, "pkg/说明.py")
    assert expected != source_id_for(REPOSITORY_ID, "b" * 40, "pkg/说明.py")
    assert expected != source_id_for(REPOSITORY_ID, COMMIT, "pkg/other.py")


def test_fixture_manifest_is_complete_sorted_and_repeatable() -> None:
    first = build_fixture_manifest()
    second = build_fixture_manifest()

    assert first == second
    assert tuple(source.relative_path for source in first.files) == (
        "README.md",
        "docs/说明.md",
        "pkg/__init__.py",
        "pkg/parser.py",
        "pkg/service.py",
    )
    assert first.skipped_sources == ()
    assert all(source.size_bytes >= 0 for source in first.files)
    assert all(len(source.content_hash) == 64 for source in first.files)
    assert len({source.source_id for source in first.files}) == len(first.files)
    verify_repository_manifest_integrity(first)
    # The internal scan ceiling is not part of source identity. Existing complete
    # inventories retain their pre-separation digest and manifest schema.
    assert first.content_sha256 == (
        "1abeb9390f06d7099094be1542f0dfe02bb1bc6b5cf1eedfb9cc67ef7a9749e6"
    )
    assert first.schema_version == 1
    assert "scan_policy" not in first.model_dump()


def test_manifest_order_does_not_depend_on_repository_listing_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = build_fixture_manifest()
    original = ReadOnlyRepository.list_directory

    def reversed_listing(
        self: ReadOnlyRepository,
        path: str = ".",
        *,
        recursive: bool = True,
    ) -> DirectoryListing:
        listing = original(self, path, recursive=recursive)
        return listing.model_copy(update={"entries": tuple(reversed(listing.entries))})

    monkeypatch.setattr(ReadOnlyRepository, "list_directory", reversed_listing)

    assert build_fixture_manifest() == expected


def test_content_change_updates_hash_without_redefining_path_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    repository = ReadOnlyRepository(tmp_path)
    first = ingest_repository(repository, repository_id=REPOSITORY_ID, commit=COMMIT)
    source.write_text("value = 2\n", encoding="utf-8")
    second = ingest_repository(repository, repository_id=REPOSITORY_ID, commit=COMMIT)

    assert first.files[0].source_id == second.files[0].source_id
    assert first.files[0].content_hash != second.files[0].content_hash
    assert first.content_sha256 != second.content_sha256


def test_empty_and_unicode_files_are_valid_sources(tmp_path: Path) -> None:
    tmp_path.joinpath("empty.py").write_text("", encoding="utf-8")
    tmp_path.joinpath("说明.md").write_text("内容\n", encoding="utf-8")

    manifest = ingest_repository(
        ReadOnlyRepository(tmp_path),
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )

    assert tuple(source.relative_path for source in manifest.files) == ("empty.py", "说明.md")
    assert manifest.files[0].size_bytes == 0
    assert manifest.files[0].content_hash == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_ignored_names_are_not_traversed_or_exposed(tmp_path: Path) -> None:
    tmp_path.joinpath("visible.py").write_text("visible = True\n", encoding="utf-8")
    for ignored in (".git", ".venv", "__pycache__"):
        directory = tmp_path / ignored
        directory.mkdir()
        directory.joinpath("secret.py").write_text("do_not_index = True\n", encoding="utf-8")

    manifest = ingest_repository(
        ReadOnlyRepository(tmp_path),
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )
    serialized = manifest.model_dump_json()

    assert tuple(source.relative_path for source in manifest.files) == ("visible.py",)
    assert "secret.py" not in serialized
    assert "do_not_index" not in serialized


def test_binary_invalid_utf8_oversized_and_symlink_sources_are_typed_skips(
    tmp_path: Path,
) -> None:
    tmp_path.joinpath("binary.dat").write_bytes(b"x\x00y")
    tmp_path.joinpath("invalid.dat").write_bytes(b"\xff\xfe")
    tmp_path.joinpath("large.txt").write_text("x" * 11, encoding="utf-8")
    tmp_path.joinpath("valid.py").write_text("ok = 1\n", encoding="utf-8")
    tmp_path.joinpath("linked.py").symlink_to("valid.py")
    repository = ReadOnlyRepository(
        tmp_path,
        policy=RepositoryAccessPolicy(
            max_file_bytes=10,
            max_read_bytes=10,
        ),
    )

    manifest = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )

    assert tuple(source.relative_path for source in manifest.files) == ("valid.py",)
    assert tuple(
        (source.relative_path, source.entry_type, source.reason, source.size_bytes)
        for source in manifest.skipped_sources
    ) == (
        (
            "binary.dat",
            RepositoryEntryType.FILE,
            RepositoryErrorCode.BINARY_FILE,
            3,
        ),
        (
            "invalid.dat",
            RepositoryEntryType.FILE,
            RepositoryErrorCode.DECODE_ERROR,
            2,
        ),
        (
            "large.txt",
            RepositoryEntryType.FILE,
            RepositoryErrorCode.FILE_TOO_LARGE,
            11,
        ),
        (
            "linked.py",
            RepositoryEntryType.SYMLINK,
            RepositoryErrorCode.SYMLINK_FORBIDDEN,
            None,
        ),
    )


def test_internal_listing_limit_is_fatal_instead_of_producing_partial_manifest(
    tmp_path: Path,
) -> None:
    for name in ("a.py", "b.py", "c.py"):
        tmp_path.joinpath(name).write_text(f"{name}\n", encoding="utf-8")
    repository = ReadOnlyRepository(tmp_path)

    with pytest.raises(RepositoryIngestError) as captured:
        ingest_repository(
            repository,
            repository_id=REPOSITORY_ID,
            commit=COMMIT,
            scan_policy=RepositoryIngestScanPolicy(max_list_entries=2),
        )

    assert_ingest_error(captured, RepositoryIngestErrorCode.LISTING_TRUNCATED)
    assert "a.py" not in str(captured.value)


def test_size_change_between_listing_and_read_is_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.joinpath("source.py").write_text("one", encoding="utf-8")

    def changed_source(
        self: ReadOnlyRepository,
        path: str,
    ) -> RepositoryTextFile:
        return RepositoryTextFile(
            relative_path=path,
            content="longer",
            size_bytes=6,
        )

    monkeypatch.setattr(ReadOnlyRepository, "read_text_file", changed_source)

    with pytest.raises(RepositoryIngestError) as captured:
        ingest_repository(
            ReadOnlyRepository(tmp_path),
            repository_id=REPOSITORY_ID,
            commit=COMMIT,
        )

    assert_ingest_error(captured, RepositoryIngestErrorCode.SOURCE_CHANGED)
    assert "source.py" not in str(captured.value)


def test_post_listing_type_change_is_mapped_to_safe_source_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.joinpath("source.py").write_text("one", encoding="utf-8")

    def missing_source(
        self: ReadOnlyRepository,
        path: str,
    ) -> RepositoryTextFile:
        raise RepositoryAccessError(RepositoryErrorCode.NOT_FOUND)

    monkeypatch.setattr(ReadOnlyRepository, "read_text_file", missing_source)

    with pytest.raises(RepositoryIngestError) as captured:
        ingest_repository(
            ReadOnlyRepository(tmp_path),
            repository_id=REPOSITORY_ID,
            commit=COMMIT,
        )

    assert_ingest_error(captured, RepositoryIngestErrorCode.SOURCE_CHANGED)


def test_repository_io_error_remains_a_repository_access_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.joinpath("source.py").write_text("one", encoding="utf-8")

    def failed_read(
        self: ReadOnlyRepository,
        path: str,
    ) -> RepositoryTextFile:
        raise RepositoryAccessError(RepositoryErrorCode.IO_ERROR)

    monkeypatch.setattr(ReadOnlyRepository, "read_text_file", failed_read)

    with pytest.raises(RepositoryAccessError) as captured:
        ingest_repository(
            ReadOnlyRepository(tmp_path),
            repository_id=REPOSITORY_ID,
            commit=COMMIT,
        )

    assert captured.value.code is RepositoryErrorCode.IO_ERROR
    assert captured.value.error_type is ErrorType.TOOL_EXECUTION_ERROR


def test_manifest_models_reject_invalid_inventory_combinations() -> None:
    source = RepositorySource(
        source_id=source_id_for(REPOSITORY_ID, COMMIT, "b.py"),
        relative_path="b.py",
        size_bytes=0,
        content_hash="0" * 64,
    )

    with pytest.raises(ValidationError, match="sorted paths"):
        RepositoryManifest.build(
            repository_id=REPOSITORY_ID,
            commit=COMMIT,
            ignore_policy=RepositoryIgnorePolicy(),
            files=(
                source,
                source.model_copy(
                    update={
                        "source_id": source_id_for(REPOSITORY_ID, COMMIT, "a.py"),
                        "relative_path": "a.py",
                    }
                ),
            ),
        )
    with pytest.raises(ValidationError, match="does not match"):
        RepositoryManifest.build(
            repository_id=REPOSITORY_ID,
            commit=COMMIT,
            ignore_policy=RepositoryIgnorePolicy(),
            files=(source.model_copy(update={"source_id": f"src_{'f' * 64}"}),),
        )
    with pytest.raises(ValidationError, match="symlink_forbidden"):
        SkippedRepositorySource(
            relative_path="link",
            entry_type=RepositoryEntryType.SYMLINK,
            reason=RepositoryErrorCode.BINARY_FILE,
        )


def test_manifest_round_trip_is_byte_deterministic_and_contains_no_source_text(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    repository_root.joinpath("source.py").write_text(
        "super_secret_source_value = 7\n",
        encoding="utf-8",
    )
    manifest = ingest_repository(
        ReadOnlyRepository(repository_root),
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    write_repository_manifest(first_path, manifest)
    write_repository_manifest(second_path, manifest)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert b"super_secret_source_value" not in first_path.read_bytes()
    assert read_repository_manifest(first_path) == manifest


def test_manifest_read_rejects_integrity_tamper(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    write_repository_manifest(path, build_fixture_manifest())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["files"][0]["content_hash"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RepositoryIngestError) as captured:
        read_repository_manifest(path)

    assert_ingest_error(captured, RepositoryIngestErrorCode.INTEGRITY_MISMATCH)


def test_manifest_read_rejects_rehashed_semantic_tamper(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    write_repository_manifest(path, build_fixture_manifest())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["files"] = list(reversed(payload["files"]))
    payload["content_sha256"] = repository_manifest_content_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RepositoryIngestError) as captured:
        read_repository_manifest(path)

    assert_ingest_error(captured, RepositoryIngestErrorCode.INVALID_DOCUMENT)


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        ("not json", RepositoryIngestErrorCode.INVALID_DOCUMENT),
        ("[]", RepositoryIngestErrorCode.INVALID_DOCUMENT),
        ('{"schema_version": 2}', RepositoryIngestErrorCode.UNSUPPORTED_VERSION),
    ],
)
def test_manifest_read_has_stable_document_errors(
    tmp_path: Path,
    payload: str,
    expected_code: RepositoryIngestErrorCode,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(RepositoryIngestError) as captured:
        read_repository_manifest(path)

    assert_ingest_error(captured, expected_code)
    assert payload not in str(captured.value)


def test_manifest_read_rejects_invalid_utf8_as_invalid_document(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(RepositoryIngestError) as captured:
        read_repository_manifest(path)

    assert_ingest_error(captured, RepositoryIngestErrorCode.INVALID_DOCUMENT)


def test_manifest_artifact_io_errors_are_safe(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(RepositoryIngestError) as read_error:
        read_repository_manifest(missing)
    assert_ingest_error(read_error, RepositoryIngestErrorCode.IO_ERROR)

    target_directory = tmp_path / "manifest.json"
    target_directory.mkdir()
    with pytest.raises(RepositoryIngestError) as write_error:
        write_repository_manifest(target_directory, build_fixture_manifest())
    assert_ingest_error(write_error, RepositoryIngestErrorCode.IO_ERROR)
    assert not tmp_path.joinpath(".manifest.json.tmp").exists()


def test_ingestor_object_exposes_effective_policy_and_empty_repository(
    tmp_path: Path,
) -> None:
    ingestor = RepositoryIngestor(ReadOnlyRepository(tmp_path))

    manifest = ingestor.build(repository_id=REPOSITORY_ID, commit=COMMIT)

    assert ingestor.ignore_policy == manifest.ignore_policy
    assert ingestor.scan_policy == RepositoryIngestScanPolicy()
    assert manifest.files == ()
    assert manifest.skipped_sources == ()


@pytest.mark.parametrize("invalid_limit", [0, -1, 20_001, True, 2.0, "2000", None])
def test_internal_scan_limit_is_strict_positive_and_bounded(invalid_limit: object) -> None:
    with pytest.raises(ValidationError):
        RepositoryIngestScanPolicy.model_validate({"max_list_entries": invalid_limit})


def test_internal_scan_policy_is_versioned_and_does_not_change_agent_policy(
    tmp_path: Path,
) -> None:
    repository = ReadOnlyRepository(tmp_path, policy=RepositoryAccessPolicy(max_list_entries=7))
    scan_policy = RepositoryIngestScanPolicy(max_list_entries=19)
    ingestor = RepositoryIngestor(repository, scan_policy=scan_policy)

    assert ingestor.scan_policy == scan_policy
    assert ingestor.scan_policy.policy_version == 1
    assert repository.policy.max_list_entries == 7
    assert RepositoryIngestScanPolicy().max_list_entries == 20_000
    with pytest.raises(ValidationError):
        RepositoryIngestScanPolicy.model_validate({"policy_version": 2})


def test_over_2000_real_entries_index_completely_while_agent_listing_stays_bounded(
    tmp_path: Path,
) -> None:
    for index in range(2_005):
        (tmp_path / f"source_{index:04d}.py").write_text("value = 1\n", encoding="utf-8")
    repository = ReadOnlyRepository(tmp_path)
    first = ingest_repository(repository, repository_id=REPOSITORY_ID, commit=COMMIT)
    second = ingest_repository(repository, repository_id=REPOSITORY_ID, commit=COMMIT)
    listing = repository.list_directory()

    assert first == second
    assert len(first.files) == 2_005
    assert first.files[-1].relative_path == "source_2004.py"
    assert first.skipped_sources == ()
    assert listing.truncated
    assert len(listing.entries) == repository.policy.max_list_entries == 2_000


def test_internal_scan_ceiling_counts_directories_and_accepts_exact_boundary(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "pkg"
    nested.mkdir()
    (nested / "source.py").write_text("value = 1\n", encoding="utf-8")
    repository = ReadOnlyRepository(tmp_path)
    exact = ingest_repository(
        repository,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        scan_policy=RepositoryIngestScanPolicy(max_list_entries=2),
    )
    larger = ingest_repository(repository, repository_id=REPOSITORY_ID, commit=COMMIT)

    assert tuple(source.relative_path for source in exact.files) == ("pkg/source.py",)
    assert exact == larger
    with pytest.raises(RepositoryIngestError) as captured:
        ingest_repository(
            repository,
            repository_id=REPOSITORY_ID,
            commit=COMMIT,
            scan_policy=RepositoryIngestScanPolicy(max_list_entries=1),
        )
    assert_ingest_error(captured, RepositoryIngestErrorCode.LISTING_TRUNCATED)
