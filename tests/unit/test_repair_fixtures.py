"""P5 repair fixture identity, gold isolation, and hostile manifest tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

import pytest

from repopilot.evals import (
    REPAIR_FIXTURE_TEST_ARGV,
    RepairFixtureError,
    RepairFixtureErrorCode,
    load_repair_fixture_manifest,
    snapshot_repair_repository,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "p5_repair_tasks"
MANIFEST = FIXTURE_ROOT / "manifest.json"


def _copy_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "p5_repair_tasks"
    shutil.copytree(FIXTURE_ROOT, destination)
    return destination / "manifest.json"


def _manifest_payload(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_manifest_loads_two_sorted_hermetic_snapshot_pinned_tasks() -> None:
    manifest = load_repair_fixture_manifest(MANIFEST)

    assert manifest.suite_id == "p5.repair-fixtures.v1"
    assert tuple(case.evaluation_task.task.task_id for case in manifest.cases) == (
        "p5-config-parser-blank",
        "p5-parser-service-async",
    )
    for case in manifest.cases:
        task = case.evaluation_task.task
        assert task.repository_path is not None
        assert Path(task.repository_path).is_absolute()
        assert task.setup_commands == ()
        assert task.test_commands[0].argv == REPAIR_FIXTURE_TEST_ARGV
        assert task.test_commands[0].timeout_seconds == 30
        assert task.timeout_seconds == 60
        snapshot = snapshot_repair_repository(Path(task.repository_path))
        assert snapshot.base_commit == task.base_commit
        assert snapshot.sha256 == case.repository_sha256
        assert case.expected_base_exit_code == 1


def test_agent_projection_contains_no_evaluator_gold_or_patch_canary() -> None:
    manifest = load_repair_fixture_manifest(MANIFEST)
    agent_payload = json.dumps(
        [task.model_dump(mode="json") for task in manifest.agent_tasks()],
        sort_keys=True,
    )

    assert "expected_patch" not in agent_payload
    assert "gold_files" not in agent_payload
    assert '"split"' not in agent_payload
    assert "diff --git" not in agent_payload
    for case in manifest.cases:
        assert case.evaluation_task.expected_patch not in agent_payload
        for gold_file in case.evaluation_task.gold_files:
            assert gold_file not in agent_payload
    assert "diff --git" not in repr(manifest)


def test_repository_content_tampering_is_rejected_before_task_use(
    tmp_path: Path,
) -> None:
    manifest_path = _copy_fixture(tmp_path)
    parser = manifest_path.parent / "repositories" / "config_parser" / "config_parser" / "parser.py"
    parser.write_text(parser.read_text(encoding="utf-8") + "\nTAMPERED = True\n", encoding="utf-8")

    with pytest.raises(RepairFixtureError) as captured:
        load_repair_fixture_manifest(manifest_path)

    assert captured.value.code is RepairFixtureErrorCode.SNAPSHOT_MISMATCH
    assert str(parser) not in str(captured.value)


def test_base_commit_mismatch_is_distinct_from_snapshot_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path = _copy_fixture(tmp_path)
    payload = _manifest_payload(manifest_path)
    payload["cases"][0]["evaluation_task"]["task"]["base_commit"] = "f" * 40
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RepairFixtureError) as captured:
        load_repair_fixture_manifest(manifest_path)

    assert captured.value.code is RepairFixtureErrorCode.BASE_COMMIT_MISMATCH


def test_patch_scope_mismatch_and_duplicate_task_ids_are_invalid(
    tmp_path: Path,
) -> None:
    for mutation in ("gold", "task_id"):
        case_root = tmp_path / mutation
        case_root.mkdir()
        manifest_path = _copy_fixture(case_root)
        payload = _manifest_payload(manifest_path)
        if mutation == "gold":
            payload["cases"][0]["evaluation_task"]["gold_files"] = ["other.py"]
        else:
            payload["cases"][1]["evaluation_task"]["task"]["task_id"] = payload["cases"][0][
                "evaluation_task"
            ]["task"]["task_id"]
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(RepairFixtureError) as captured:
            load_repair_fixture_manifest(manifest_path)

        assert captured.value.code is RepairFixtureErrorCode.MANIFEST_INVALID


def test_repository_symlink_escape_and_in_tree_symlink_are_rejected(
    tmp_path: Path,
) -> None:
    manifest_path = _copy_fixture(tmp_path)
    repository = manifest_path.parent / "repositories" / "config_parser"
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.rmtree(repository)
    repository.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RepairFixtureError) as escaped:
        load_repair_fixture_manifest(manifest_path)

    assert escaped.value.code is RepairFixtureErrorCode.PATH_VIOLATION

    manifest_path = _copy_fixture(tmp_path / "in-tree")
    repository = manifest_path.parent / "repositories" / "config_parser"
    (repository / "source-link.py").symlink_to(repository / "config_parser" / "parser.py")

    with pytest.raises(RepairFixtureError) as linked:
        load_repair_fixture_manifest(manifest_path)

    assert linked.value.code is RepairFixtureErrorCode.REPOSITORY_UNAVAILABLE
