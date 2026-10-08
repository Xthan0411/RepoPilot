from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from repopilot.enums import (
    DatasetManifestErrorCode,
    DatasetSplit,
    ModelProvider,
)
from repopilot.evals import (
    DatasetManifestError,
    build_evaluation_run_manifest,
    load_frozen_dataset,
)

pytestmark = pytest.mark.security

REPOSITORY_ROOT = Path(__file__).parents[2].resolve()
MANIFEST = REPOSITORY_ROOT / "data/manifests/p7-v1/manifest.json"


def test_agent_and_evaluation_run_manifest_do_not_disclose_dataset_gold() -> None:
    bundle = load_frozen_dataset(MANIFEST, repository_root=REPOSITORY_ROOT)
    holdout = bundle.external_evaluation_tasks
    run_manifest = build_evaluation_run_manifest(
        experiment_id="gold-isolation",
        run_id=UUID("11111111-2222-3333-4444-555555555555"),
        created_at=datetime(2026, 8, 7, tzinfo=UTC),
        source_commit="a" * 40,
        dependency_lock_sha256="b" * 64,
        dataset_manifest_sha256=bundle.manifest.content_sha256,
        configuration={"profile": "security", "temperature": 0.0},
        split=DatasetSplit.HOLDOUT,
        arm_id="hybrid",
        tasks=holdout,
        memory_namespace=None,
        model_provider=ModelProvider.FAKE,
        model="fake-deterministic",
        prompt_versions=("repair-v1",),
        sandbox_image_reference=f"python@sha256:{'c' * 64}",
    )

    agent_payload = json.dumps(
        [item.model_dump(mode="json") for item in bundle.agent_tasks()],
        sort_keys=True,
    )
    run_payload = run_manifest.model_dump_json()
    assert "expected_patch" not in agent_payload
    assert "gold_files" not in agent_payload
    assert "+    __slots__ = ()" not in agent_payload
    assert "+    __slots__ = ()" not in run_payload
    assert "source-rows.json" not in run_payload


def test_symlink_root_and_symlink_source_snapshot_fail_closed(tmp_path: Path) -> None:
    real_root = _copy_dataset_inputs(tmp_path / "real")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(DatasetManifestError) as root_error:
        load_frozen_dataset(
            linked_root / "data/manifests/p7-v1/manifest.json",
            repository_root=linked_root,
        )
    assert root_error.value.code is DatasetManifestErrorCode.PATH_UNSAFE

    source_rows = real_root / "data/manifests/p7-v1/source-rows.json"
    source_copy = real_root / "source-copy.json"
    shutil.copyfile(source_rows, source_copy)
    source_rows.unlink()
    source_rows.symlink_to(source_copy)
    with pytest.raises(DatasetManifestError) as source_error:
        load_frozen_dataset(
            real_root / "data/manifests/p7-v1/manifest.json",
            repository_root=real_root,
        )
    assert source_error.value.code is DatasetManifestErrorCode.PATH_UNSAFE


def test_manifest_path_escape_is_rejected_before_file_read(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    root.mkdir(exist_ok=True)
    outside = tmp_path.parent / "outside-dataset-manifest.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(DatasetManifestError) as error:
        load_frozen_dataset(outside, repository_root=root)
    assert error.value.code is DatasetManifestErrorCode.PATH_UNSAFE


def _copy_dataset_inputs(destination: Path) -> Path:
    root = destination.resolve()
    (root / "data/manifests").mkdir(parents=True)
    (root / "tests/fixtures").mkdir(parents=True)
    shutil.copytree(
        REPOSITORY_ROOT / "data/manifests/p7-v1",
        root / "data/manifests/p7-v1",
    )
    shutil.copytree(
        REPOSITORY_ROOT / "tests/fixtures/p5_repair_tasks",
        root / "tests/fixtures/p5_repair_tasks",
    )
    return root
