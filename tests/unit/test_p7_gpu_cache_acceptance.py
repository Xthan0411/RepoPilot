"""Offline tiny-cache coverage and fail-closed corpus-consumer tests."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from repopilot.config import RepoPilotSettings
from repopilot.embeddings import (
    EmbeddingIdentity,
    EmbeddingInput,
    FakeEmbeddingProvider,
    embedding_cache_key,
    embedding_identity_from_settings,
)

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import p7_gpu_cache as gpu  # noqa: E402
from scripts import p7_gpu_cache_acceptance as acceptance  # noqa: E402
from scripts import p7_gpu_cache_store as store  # noqa: E402
from scripts.p7_holdout_control import ENVIRONMENT_PATH  # noqa: E402
from scripts.p7_repository_workload_preflight import (  # noqa: E402
    AGENT_INPUTS_SHA256,
    MATRIX_SHA256,
)


def test_retained_operator_job_has_the_exact_pinned_canonical_seal() -> None:
    document = store._read(PROJECT_ROOT / "results/p7-holdout/gpu-cache-job-v1.json")
    assert document["sha256"] == acceptance.EXPECTED_JOB_SHA256
    binding = document["binding"]
    assert isinstance(binding, dict)
    assert binding["input_count"] == 103236
    assert binding["formal_ready"] is False


def _tiny_cache(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[EmbeddingIdentity, tuple[EmbeddingInput, ...]]:
    identity = embedding_identity_from_settings(RepoPilotSettings())
    inputs = tuple(
        EmbeddingInput(content=value, content_hash=hashlib.sha256(value.encode()).hexdigest())
        for value in ("short", "median source content", "long " * 100)
    )
    _fill(directory, identity, inputs, [], monkeypatch)
    return identity, inputs


def _fill(
    directory: Path,
    identity: EmbeddingIdentity,
    inputs: tuple[EmbeddingInput, ...],
    tasks: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with store.job_lock(directory):
        store.bind_job(
            directory,
            identity,
            {
                "input_count": len(inputs),
                "input_set_sha256": store._digest([item.content_hash for item in inputs]),
                "agent_inputs_raw_sha256": AGENT_INPUTS_SHA256,
                "matrix_sha256": MATRIX_SHA256,
                "agent_constructed": False,
                "formal_ready": False,
                "openai_requests": 0,
                "runtime": {"profile": "fake-only-test"},
                "tasks": tasks,
            },
        )
        store.fill_cache(directory, identity, inputs, FakeEmbeddingProvider(identity))
    report = store.verify_cache(directory)
    for name in ("JOB", "INPUTS", "COMPLETE"):
        monkeypatch.setattr(acceptance, f"EXPECTED_{name}_SHA256", report[f"{name.lower()}_sha256"])


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_full_pinned_verification_and_reads_make_no_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, inputs = _tiny_cache(tmp_path, monkeypatch)
    before = _snapshot(tmp_path)
    result = acceptance.verify_pinned_cache(tmp_path, identity)
    assert result.report["completed_vectors"] == 3
    assert result.cache.get(identity, inputs[0].content_hash).content_hash == inputs[0].content_hash
    assert result.cache.hits == 1
    assert result.cache.misses == 0
    assert result.cache.write_attempts == 0
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("name", ["JOB", "INPUTS", "COMPLETE"])
def test_valid_but_other_cache_seals_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    _tiny_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(acceptance, f"EXPECTED_{name}_SHA256", "0" * 64)
    monkeypatch.setattr(store, "verify_cache", lambda *args: pytest.fail("unexpected full scan"))
    with pytest.raises(acceptance.CacheAcceptanceError, match="cache_pin_mismatch"):
        acceptance.verify_pinned_cache(tmp_path)


def test_expected_model_identity_cannot_be_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, _ = _tiny_cache(tmp_path, monkeypatch)
    different = identity.model_copy(update={"revision": "other-revision"})
    with pytest.raises(acceptance.CacheAcceptanceError, match="cache_identity_mismatch"):
        acceptance.verify_pinned_cache(tmp_path, different)


def test_cache_miss_and_put_fail_without_recomputation_or_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, inputs = _tiny_cache(tmp_path, monkeypatch)
    cache = acceptance.verify_pinned_cache(tmp_path).cache
    before = _snapshot(tmp_path)
    with pytest.raises(acceptance.CacheAcceptanceError, match="corpus_cache_miss"):
        cache.get(identity, "0" * 64)
    vector = cache.get(identity, inputs[0].content_hash)
    with pytest.raises(acceptance.CacheAcceptanceError, match="cache_write_forbidden"):
        cache.put(identity, vector)
    assert cache.misses == 1
    assert cache.write_attempts == 1
    assert _snapshot(tmp_path) == before


def test_consumer_rechecks_file_hash_even_if_changed_record_is_internally_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, inputs = _tiny_cache(tmp_path, monkeypatch)
    cache = acceptance.verify_pinned_cache(tmp_path).cache
    key = embedding_cache_key(identity, inputs[0].content_hash)
    path = tmp_path / "vectors" / key[:2] / f"{key}.json"
    # Whitespace leaves the per-record semantic integrity valid but changes the
    # exact bytes bound by the accepted completion manifest.
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(acceptance.CacheAcceptanceError, match="cache_entry_changed"):
        cache.get(identity, inputs[0].content_hash)


def test_consumer_reports_deleted_entry_as_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, inputs = _tiny_cache(tmp_path, monkeypatch)
    cache = acceptance.verify_pinned_cache(tmp_path).cache
    key = embedding_cache_key(identity, inputs[0].content_hash)
    path = tmp_path / "vectors" / key[:2] / f"{key}.json"
    path.unlink()
    with pytest.raises(acceptance.CacheAcceptanceError, match="corpus_cache_miss"):
        cache.get(identity, inputs[0].content_hash)
    assert cache.misses == 1


@pytest.mark.parametrize("component", ["root", "shard", "entry"])
def test_consumer_rejects_symlink_at_each_path_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    directory = tmp_path / "job"
    identity, inputs = _tiny_cache(directory, monkeypatch)
    cache = acceptance.verify_pinned_cache(directory).cache
    key = embedding_cache_key(identity, inputs[0].content_hash)
    targets = {
        "root": directory / "vectors",
        "shard": directory / "vectors" / key[:2],
        "entry": directory / "vectors" / key[:2] / f"{key}.json",
    }
    path = targets[component]
    moved = tmp_path / "moved"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=component != "entry")
    with pytest.raises(acceptance.CacheAcceptanceError, match="unsafe_cache_path"):
        cache.get(identity, inputs[0].content_hash)


def test_consumer_rejects_hard_link_added_after_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "job"
    identity, inputs = _tiny_cache(directory, monkeypatch)
    cache = acceptance.verify_pinned_cache(directory).cache
    key = embedding_cache_key(identity, inputs[0].content_hash)
    os.link(directory / "vectors" / key[:2] / f"{key}.json", tmp_path / "linked")
    with pytest.raises(acceptance.CacheAcceptanceError, match="unsafe_cache_entry"):
        cache.get(identity, inputs[0].content_hash)


def test_consumer_rejects_fifo_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, inputs = _tiny_cache(tmp_path, monkeypatch)
    cache = acceptance.verify_pinned_cache(tmp_path).cache
    key = embedding_cache_key(identity, inputs[0].content_hash)
    path = tmp_path / "vectors" / key[:2] / f"{key}.json"
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(acceptance.CacheAcceptanceError, match="unsafe_cache_entry"):
        cache.get(identity, inputs[0].content_hash)


def test_full_verification_detects_corrupt_or_missing_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, inputs = _tiny_cache(tmp_path, monkeypatch)
    key = embedding_cache_key(identity, inputs[0].content_hash)
    (tmp_path / "vectors" / key[:2] / f"{key}.json").write_bytes(b"corrupt")
    with pytest.raises(acceptance.CacheAcceptanceError, match="cache_integrity_failed"):
        acceptance.verify_pinned_cache(tmp_path)


def _coverage_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[EmbeddingInput, ...]:
    tasks, _ = gpu.load_tasks(PROJECT_ROOT)
    settings = RepoPilotSettings()
    environment_raw = (PROJECT_ROOT / ENVIRONMENT_PATH).read_bytes()
    environment_path = tmp_path / ENVIRONMENT_PATH
    environment_path.parent.mkdir(parents=True)
    environment_path.write_bytes(environment_raw)
    environment = json.loads(environment_raw)
    for index, binding in enumerate(environment["bindings"]):
        checkout = tmp_path / binding["checkout_relative_path"]
        checkout.mkdir(parents=True)
        (checkout / ".git").mkdir()
        (checkout / "sample.py").write_text(
            f"def sample_{index}(value):\n    return value + {index}\n", encoding="utf-8"
        )
        (checkout / "README.md").write_text(
            "Shared documentation content.\n" + "extra " * index, encoding="utf-8"
        )
    monkeypatch.setattr(gpu, "CHECKOUT_PATH", Path("artifacts/p7-holdout/checkouts"))
    monkeypatch.setattr(gpu, "check_checkout", lambda *args: None)
    monkeypatch.setattr(acceptance, "_check_checkout", lambda *args: None)
    inputs, records = gpu.build_inputs(tmp_path, tasks, settings)
    monkeypatch.setattr(gpu, "load_tasks", lambda *args: (tasks, settings))
    _fill(
        tmp_path / gpu.JOB_PATH,
        embedding_identity_from_settings(settings),
        inputs,
        records,
        monkeypatch,
    )
    return inputs


def test_coverage_rebuilds_all_task_fingerprints_and_returns_fixed_three_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _coverage_fixture(tmp_path, monkeypatch)
    before = _snapshot(tmp_path / gpu.JOB_PATH)
    events: list[Any] = []
    result = acceptance.verify_coverage(tmp_path, progress=events.append)
    assert result.samples == (inputs[0], inputs[len(inputs) // 2], inputs[-1])
    assert [vector.content_hash for vector in result.gpu_vectors] == [
        item.content_hash for item in result.samples
    ]
    assert result.report["coverage_passed"] is True
    assert result.report["task_count"] == 8
    assert result.report["formal_ready"] is False
    assert result.report["cache_misses"] == 0
    assert result.report["provider_requests"] == 0
    assert result.report["model_loads"] == 0
    assert sum(event["stage"] == "corpus_passed" for event in events) == 8
    assert _snapshot(tmp_path / gpu.JOB_PATH) == before
    serialized = json.dumps(result.report)
    assert all(item.content not in serialized for item in result.samples)


def test_modified_cpu_source_fingerprint_is_not_accepted_as_complete_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _coverage_fixture(tmp_path, monkeypatch)
    path = tmp_path / gpu.CHECKOUT_PATH / "django__django-15902" / "sample.py"
    path.write_text("def changed():\n    return 'changed'\n", encoding="utf-8")
    with pytest.raises(acceptance.CacheAcceptanceError, match="corpus_fingerprint_mismatch"):
        acceptance.verify_coverage(tmp_path)


def test_environment_bytes_are_checked_before_cache_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _coverage_fixture(tmp_path, monkeypatch)
    path = tmp_path / ENVIRONMENT_PATH
    path.write_bytes(path.read_bytes() + b"\n")
    monkeypatch.setattr(acceptance, "verify_pinned_cache", lambda *args: pytest.fail("cache scan"))
    with pytest.raises(acceptance.CacheAcceptanceError, match="environment_identity_mismatch"):
        acceptance.verify_coverage(tmp_path)


def test_checkout_guard_checks_root_commit_all_untracked_and_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = gpu.load_tasks(PROJECT_ROOT)[0][0]
    (tmp_path / ".git").mkdir()
    observed: list[tuple[str, ...]] = []

    def git(checkout: Path, *arguments: str) -> str:
        observed.append(arguments)
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(checkout)
        if arguments == ("rev-parse", "HEAD"):
            return task.base_commit
        return ""

    monkeypatch.setattr(gpu, "_git", git)
    acceptance._check_checkout(tmp_path, task)
    assert observed[-1] == ("status", "--porcelain=v1", "--untracked-files=all", "--ignored")


def test_checkout_guard_rejects_git_directory_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = gpu.load_tasks(PROJECT_ROOT)[0][0]
    (tmp_path / "git-target").mkdir()
    (tmp_path / ".git").symlink_to(tmp_path / "git-target", target_is_directory=True)
    monkeypatch.setattr(gpu, "_git", lambda *args: pytest.fail("unsafe git invocation"))
    with pytest.raises(acceptance.CacheAcceptanceError, match="unsafe_checkout_git_dir"):
        acceptance._check_checkout(tmp_path, task)
