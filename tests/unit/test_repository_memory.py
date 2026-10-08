"""Repository Memory admission, invalidation, and file-safety contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repopilot.enums import (
    RepositoryMemoryErrorCode,
    RepositoryMemoryLookupStatus,
)
from repopilot.memory import (
    FileRepositoryMemoryStore,
    RepositoryMemoryError,
    build_confirmed_fact_record,
    build_repository_map_record,
    build_test_command_record,
    command_spec_sha256,
)
from repopilot.retrieval import (
    RepositoryIgnorePolicy,
    RepositoryManifest,
    RepositorySource,
    source_id_for,
)
from repopilot.schemas import CommandSpec, TaskSpec

NOW = datetime(2026, 8, 7, 8, 0, tzinfo=UTC)
REPOSITORY_ID = "repo_memory_fixture"
COMMIT = "a" * 40


def make_manifest(
    *,
    repository_id: str = REPOSITORY_ID,
    commit: str = COMMIT,
    parser_hash: str = "1" * 64,
) -> RepositoryManifest:
    paths = (
        ("README.md", "2" * 64),
        ("pkg/parser.py", parser_hash),
        ("tests/test_parser.py", "3" * 64),
    )
    return RepositoryManifest.build(
        repository_id=repository_id,
        commit=commit,
        ignore_policy=RepositoryIgnorePolicy(),
        files=tuple(
            RepositorySource(
                source_id=source_id_for(repository_id, commit, path),
                relative_path=path,
                size_bytes=100 + index,
                content_hash=content_hash,
            )
            for index, (path, content_hash) in enumerate(paths)
        ),
    )


def make_task(
    *,
    commit: str = COMMIT,
    secret_argument: str = "--token=SECRET_COMMAND_CANARY",
) -> TaskSpec:
    return TaskSpec(
        task_id="repository-memory-task",
        repository_path="/tmp/repository-memory",
        base_commit=commit,
        issue_title="Parser regression",
        issue_body="The parser should ignore blank values.",
        test_commands=(
            CommandSpec(
                argv=("python", "-m", "pytest", "-q", secret_argument),
                cwd="tests",
                timeout_seconds=30,
            ),
        ),
        timeout_seconds=60,
    )


def test_builders_admit_only_manifest_task_or_source_confirmed_records() -> None:
    manifest = make_manifest()
    task = make_task()

    repository_map = build_repository_map_record(manifest, created_at=NOW)
    test_command = build_test_command_record(
        manifest,
        task,
        test_index=0,
        created_at=NOW,
    )
    fact = build_confirmed_fact_record(
        manifest,
        key="parser-entrypoint",
        fact="The parser implementation is in pkg/parser.py.",
        evidence_paths=("pkg/parser.py",),
        created_at=NOW,
    )

    assert repository_map.repository_content_sha256 == manifest.content_sha256
    assert '"top_level_entries":["README.md","pkg","tests"]' in repository_map.content
    assert test_command.evidence[0].content_sha256 == command_spec_sha256(task.test_commands[0])
    assert "SECRET_COMMAND_CANARY" not in test_command.content
    assert fact.evidence[0].relative_path == "pkg/parser.py"

    with pytest.raises(RepositoryMemoryError) as missing_evidence:
        build_confirmed_fact_record(
            manifest,
            key="unsupported-fact",
            fact="This fact has no current source.",
            evidence_paths=("missing.py",),
            created_at=NOW,
        )

    assert missing_evidence.value.code is RepositoryMemoryErrorCode.UNCONFIRMED_RECORD


def test_file_store_round_trips_sorted_records_and_is_idempotent(tmp_path: Path) -> None:
    manifest = make_manifest()
    task = make_task()
    records = (
        build_repository_map_record(manifest, created_at=NOW),
        build_test_command_record(
            manifest,
            task,
            test_index=0,
            created_at=NOW,
        ),
        build_confirmed_fact_record(
            manifest,
            key="parser-entrypoint",
            fact="The parser implementation is in pkg/parser.py.",
            evidence_paths=("pkg/parser.py",),
            created_at=NOW,
        ),
    )
    root = tmp_path / "repository-memory"
    store = FileRepositoryMemoryStore(root)

    assert store.put(records[1], manifest=manifest, task=task)
    assert store.put(records[0], manifest=manifest)
    assert store.put(records[2], manifest=manifest)
    assert not store.put(records[0], manifest=manifest)
    lookup = store.lookup(manifest)

    assert lookup.status is RepositoryMemoryLookupStatus.CURRENT
    assert lookup.records == tuple(sorted(records, key=lambda item: item.record_id))
    assert lookup.stale_record_count == 0
    record_path = (
        root
        / manifest.repository_id
        / manifest.commit
        / manifest.content_sha256
        / f"{records[0].record_id}.json"
    )
    assert record_path.stat().st_mode & 0o777 == 0o600
    assert record_path.parent.stat().st_mode & 0o777 == 0o700


def test_lookup_distinguishes_empty_commit_stale_and_content_stale(
    tmp_path: Path,
) -> None:
    store = FileRepositoryMemoryStore(tmp_path / "repository-memory")
    original = make_manifest()
    assert store.lookup(original).status is RepositoryMemoryLookupStatus.EMPTY
    store.put(build_repository_map_record(original, created_at=NOW), manifest=original)

    changed_commit = make_manifest(commit="b" * 40)
    commit_lookup = store.lookup(changed_commit)
    changed_content = make_manifest(parser_hash="f" * 64)
    content_lookup = store.lookup(changed_content)

    assert commit_lookup.status is RepositoryMemoryLookupStatus.COMMIT_STALE
    assert commit_lookup.records == ()
    assert commit_lookup.stale_record_count == 1
    assert content_lookup.status is RepositoryMemoryLookupStatus.CONTENT_STALE
    assert content_lookup.records == ()
    assert content_lookup.stale_record_count == 1


def test_store_rejects_namespace_stale_evidence_and_duplicate_conflicts(
    tmp_path: Path,
) -> None:
    store = FileRepositoryMemoryStore(tmp_path / "repository-memory")
    manifest = make_manifest()
    first = build_confirmed_fact_record(
        manifest,
        key="parser-entrypoint",
        fact="The parser implementation is in pkg/parser.py.",
        evidence_paths=("pkg/parser.py",),
        created_at=NOW,
    )
    conflicting = build_confirmed_fact_record(
        manifest,
        key="parser-entrypoint",
        fact="A conflicting statement uses the same stable fact key.",
        evidence_paths=("pkg/parser.py",),
        created_at=NOW,
    )
    store.put(first, manifest=manifest)

    with pytest.raises(RepositoryMemoryError) as duplicate:
        store.put(conflicting, manifest=manifest)
    with pytest.raises(RepositoryMemoryError) as namespace:
        store.put(first, manifest=make_manifest(repository_id="repo_other"))
    with pytest.raises(RepositoryMemoryError) as commit:
        store.put(first, manifest=make_manifest(commit="b" * 40))
    with pytest.raises(RepositoryMemoryError) as content:
        store.put(first, manifest=make_manifest(parser_hash="f" * 64))

    assert duplicate.value.code is RepositoryMemoryErrorCode.DUPLICATE_CONFLICT
    assert namespace.value.code is RepositoryMemoryErrorCode.NAMESPACE_MISMATCH
    assert commit.value.code is RepositoryMemoryErrorCode.COMMIT_STALE
    assert content.value.code is RepositoryMemoryErrorCode.CONTENT_STALE


def test_store_detects_tampering_capacity_and_symlink_paths(tmp_path: Path) -> None:
    manifest = make_manifest()
    root = tmp_path / "repository-memory"
    store = FileRepositoryMemoryStore(root, max_records_per_version=1)
    record = build_repository_map_record(manifest, created_at=NOW)
    store.put(record, manifest=manifest)
    path = (
        root
        / manifest.repository_id
        / manifest.commit
        / manifest.content_sha256
        / f"{record.record_id}.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["content"] = "tampered"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RepositoryMemoryError) as integrity:
        store.lookup(manifest)

    assert integrity.value.code is RepositoryMemoryErrorCode.INTEGRITY_MISMATCH

    limited_root = tmp_path / "limited"
    limited = FileRepositoryMemoryStore(limited_root, max_records_per_version=1)
    limited.put(record, manifest=manifest)
    second = build_confirmed_fact_record(
        manifest,
        key="second-fact",
        fact="A second source-backed fact.",
        evidence_paths=("README.md",),
        created_at=NOW,
    )
    with pytest.raises(RepositoryMemoryError) as capacity:
        limited.put(second, manifest=manifest)

    target = tmp_path / "actual-memory"
    target.mkdir()
    link = tmp_path / "linked-memory"
    link.symlink_to(target, target_is_directory=True)
    unsafe = FileRepositoryMemoryStore(link)
    with pytest.raises(RepositoryMemoryError) as path_error:
        unsafe.put(record, manifest=manifest)

    assert capacity.value.code is RepositoryMemoryErrorCode.CAPACITY_EXCEEDED
    assert path_error.value.code is RepositoryMemoryErrorCode.PATH_UNSAFE


def test_store_caps_repository_versions_without_deleting_history(tmp_path: Path) -> None:
    store = FileRepositoryMemoryStore(
        tmp_path / "repository-memory",
        max_versions_per_repository=1,
    )
    original = make_manifest()
    store.put(build_repository_map_record(original, created_at=NOW), manifest=original)
    changed = make_manifest(parser_hash="f" * 64)

    with pytest.raises(RepositoryMemoryError) as capacity:
        store.put(build_repository_map_record(changed, created_at=NOW), manifest=changed)

    assert capacity.value.code is RepositoryMemoryErrorCode.CAPACITY_EXCEEDED
    assert store.lookup(original).status is RepositoryMemoryLookupStatus.CURRENT


def test_store_rejects_relative_root_and_invalid_constructor_limits(
    tmp_path: Path,
) -> None:
    with pytest.raises(RepositoryMemoryError) as path_error:
        FileRepositoryMemoryStore(Path("relative-memory"))

    invalid_settings = (
        {"max_records_per_version": 0},
        {"max_records_per_version": True},
        {"max_versions_per_repository": 0},
        {"max_versions_per_repository": 257},
        {"max_record_bytes": 1023},
        {"max_record_bytes": True},
    )
    for settings in invalid_settings:
        with pytest.raises(ValueError):
            FileRepositoryMemoryStore(tmp_path / "repository-memory", **settings)

    assert path_error.value.code is RepositoryMemoryErrorCode.PATH_UNSAFE
