"""Regression-test projection identity and formal executor isolation wiring."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import p7_holdout_test_inputs as projection  # noqa: E402
from scripts import p7_test_protocol_canary as canary  # noqa: E402
from scripts.p7_holdout_agent_inputs import (  # noqa: E402
    AGENT_INPUTS_PATH,
    HoldoutAgentInputs,
    _canonical_sha256,
)
from scripts.p7_holdout_test_inputs import (  # noqa: E402
    TEST_INPUTS_PATH,
    TEST_INPUTS_RAW_SHA256,
    HoldoutTestInputs,
    build_test_inputs,
    load_test_inputs,
)


def _agent_inputs() -> HoldoutAgentInputs:
    return HoldoutAgentInputs.model_validate_json((PROJECT_ROOT / AGENT_INPUTS_PATH).read_bytes())


def test_regression_projection_is_exactly_rebuildable_and_contains_no_solutions() -> None:
    raw = (PROJECT_ROOT / TEST_INPUTS_PATH).read_bytes()
    artifact = load_test_inputs(
        PROJECT_ROOT, agent_inputs=_agent_inputs(), expected_raw_sha256=TEST_INPUTS_RAW_SHA256
    )
    assert hashlib.sha256(raw).hexdigest() == TEST_INPUTS_RAW_SHA256
    assert build_test_inputs(PROJECT_ROOT) == artifact
    assert len(artifact.tasks) == 8
    assert not artifact.solution_patch_serialized
    assert not artifact.agent_context_allowed
    for item in json.loads(raw)["tasks"]:
        assert set(item) == {
            "task_id",
            "base_commit",
            "regression_patch",
            "regression_patch_sha256",
        }
    assert b'"expected_patch"' not in raw
    assert b'"gold_files"' not in raw


def test_runtime_projection_loader_does_not_open_source_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _agent_inputs()
    original = Path.open

    def guarded(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name != TEST_INPUTS_PATH.name:
            raise AssertionError("runtime loader read material beyond test projection")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert (
        len(
            load_test_inputs(
                PROJECT_ROOT, agent_inputs=inputs, expected_raw_sha256=TEST_INPUTS_RAW_SHA256
            ).tasks
        )
        == 8
    )


def test_raw_digest_rejects_self_consistently_rehashed_test_patch(tmp_path: Path) -> None:
    payload = json.loads((PROJECT_ROOT / TEST_INPUTS_PATH).read_bytes())
    payload["tasks"][0]["regression_patch"] += "\n"
    payload["tasks"][0]["regression_patch_sha256"] = hashlib.sha256(
        payload["tasks"][0]["regression_patch"].encode()
    ).hexdigest()
    payload["content_sha256"] = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    path = tmp_path / TEST_INPUTS_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="raw digest"):
        load_test_inputs(
            tmp_path, agent_inputs=_agent_inputs(), expected_raw_sha256=TEST_INPUTS_RAW_SHA256
        )


@pytest.mark.parametrize("change", ["solution", "reorder", "digest"])
def test_regression_projection_rejects_invalid_payload(change: str) -> None:
    payload = json.loads((PROJECT_ROOT / TEST_INPUTS_PATH).read_bytes())
    if change == "solution":
        payload["tasks"][0]["expected_patch"] = "not allowed"
    elif change == "reorder":
        payload["tasks"] = list(reversed(payload["tasks"]))
    else:
        payload["tasks"][0]["regression_patch"] += "\n"
    payload["content_sha256"] = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    with pytest.raises(ValidationError):
        HoldoutTestInputs.model_validate(payload)


def test_projection_cannot_be_rebound_to_different_agent_inputs() -> None:
    inputs = _agent_inputs().model_copy(update={"content_sha256": "0" * 64})
    with pytest.raises(ValueError, match="binding mismatch"):
        load_test_inputs(
            PROJECT_ROOT, agent_inputs=inputs, expected_raw_sha256=TEST_INPUTS_RAW_SHA256
        )


def test_projection_rejects_symlink(tmp_path: Path) -> None:
    path = tmp_path / TEST_INPUTS_PATH
    path.parent.mkdir(parents=True)
    path.symlink_to(PROJECT_ROOT / TEST_INPUTS_PATH)
    with pytest.raises(ValueError, match="unsafe"):
        load_test_inputs(
            tmp_path, agent_inputs=_agent_inputs(), expected_raw_sha256=TEST_INPUTS_RAW_SHA256
        )


def test_projection_cli_rejects_reformatted_existing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = build_test_inputs(PROJECT_ROOT)
    path = tmp_path / TEST_INPUTS_PATH
    path.parent.mkdir(parents=True)
    path.write_text(artifact.model_dump_json())
    monkeypatch.setattr(projection, "build_test_inputs", lambda root: artifact)
    monkeypatch.setattr(sys, "argv", ["projection", "--repository-root", str(tmp_path)])
    with pytest.raises(ValueError, match="raw digest"):
        projection.main()


def _controller_repository(tmp_path: Path) -> Path:
    for relative in canary.REQUIRED_CONTROLLER_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("committed fixture\n")
    for arguments in (
        ("init", "--quiet"),
        ("add", "."),
        (
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "Controller fixture",
        ),
    ):
        canary._git(tmp_path, *arguments)
    return tmp_path


def test_canary_controller_allows_unrelated_untracked_notes(tmp_path: Path) -> None:
    root = _controller_repository(tmp_path)
    (root / "user-note.md").write_text("not controller source\n")
    assert canary._verified_controller_commit(root) == canary._git(root, "rev-parse", "HEAD")


def test_canary_rejects_required_untracked_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _controller_repository(tmp_path)
    missing = "scripts/uncommitted_runtime.py"
    (root / missing).write_text("uncommitted code\n")
    monkeypatch.setattr(
        canary, "REQUIRED_CONTROLLER_PATHS", (*canary.REQUIRED_CONTROLLER_PATHS, missing)
    )
    with pytest.raises(SystemExit, match="controller_source_uncommitted"):
        canary._verified_controller_commit(root)


@pytest.mark.parametrize("hidden_by_index", [False, True])
def test_canary_rejects_modified_required_module(tmp_path: Path, hidden_by_index: bool) -> None:
    root = _controller_repository(tmp_path)
    relative = canary.REQUIRED_CONTROLLER_PATHS[0]
    if hidden_by_index:
        canary._git(root, "update-index", "--assume-unchanged", "--", relative)
    (root / relative).write_text("modified runtime\n")
    with pytest.raises(SystemExit, match="controller_source_dirty"):
        canary._verified_controller_commit(root)


def test_canary_rejects_required_path_missing_from_head(tmp_path: Path) -> None:
    root = _controller_repository(tmp_path)
    relative = canary.REQUIRED_CONTROLLER_PATHS[0]
    canary._git(root, "rm", "--cached", "--", relative)
    canary._git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "Remove runtime from HEAD",
    )
    assert (root / relative).is_file()
    with pytest.raises(SystemExit, match="controller_source_uncommitted"):
        canary._verified_controller_commit(root)
