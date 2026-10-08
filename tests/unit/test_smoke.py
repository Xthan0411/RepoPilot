"""Unit tests for deterministic P0 smoke orchestration and artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import SecretStr, ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    ErrorType,
    ModelProvider,
    RunStatus,
    SandboxNetworkMode,
    TerminationReason,
)
from repopilot.providers.base import ProviderError
from repopilot.schemas import FinalResult
from repopilot.smoke import (
    SmokeInputError,
    SmokeIOError,
    SmokeManifest,
    SmokeTrace,
    load_smoke_manifest,
    run_smoke,
    run_smoke_manifest,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
MANIFEST_PATH = FIXTURES / "p0_smoke_manifest.json"
OTHER_REQUEST_ID = UUID("77777777-7777-4777-8777-777777777777")


def load_payload() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
    )


def test_smoke_manifest_round_trip_and_gold_isolation() -> None:
    manifest = load_smoke_manifest(MANIFEST_PATH)
    serialized = manifest.model_dump_json()

    assert SmokeManifest.model_validate_json(serialized) == manifest
    assert "expected_patch" not in serialized
    assert "gold_files" not in serialized


def test_smoke_writes_deterministic_stopped_result_and_trace(tmp_path: Path) -> None:
    settings = RepoPilotSettings(
        openai_api_key=SecretStr("sensitive-openai-test-value"),
        langsmith_api_key=SecretStr("sensitive-langsmith-test-value"),
        langsmith_tracing=True,
    )
    output_dir = tmp_path / "output"

    first = run_smoke(MANIFEST_PATH, output_dir, settings)
    first_result_bytes = first.result_path.read_bytes()
    first_trace_bytes = first.trace_path.read_bytes()
    second = run_smoke(MANIFEST_PATH, output_dir, settings)

    assert second.result_path.read_bytes() == first_result_bytes
    assert second.trace_path.read_bytes() == first_trace_bytes
    assert FinalResult.model_validate_json(first_result_bytes) == first.result
    assert SmokeTrace.model_validate_json(first_trace_bytes) == first.trace
    assert first.result.status is RunStatus.STOPPED
    assert first.result.termination_reason is TerminationReason.AGENT_STOPPED
    assert first.result.patch_path is None
    assert first.result.task_metrics.task_success is False
    assert first.result.task_metrics.tool_steps == 0
    assert first.result.trace_path == "trace.json"
    assert first.trace.result_status is RunStatus.STOPPED
    assert first.trace.request.context_window is not None
    assert first.trace.request.context_window.fits_character_budget
    assert first.trace.settings["openai_api_key"] == "<redacted>"
    assert first.trace.settings["langsmith_api_key"] == "<redacted>"
    assert "sensitive-openai-test-value" not in first_trace_bytes.decode()
    assert "sensitive-langsmith-test-value" not in first_trace_bytes.decode()


@pytest.mark.parametrize(
    "settings",
    [
        RepoPilotSettings(
            model_provider=ModelProvider.OPENAI,
            model="gpt-test",
            openai_api_key=SecretStr("provider-test-value"),
        ),
        RepoPilotSettings(model="another-fake"),
        RepoPilotSettings(sandbox_network=SandboxNetworkMode.SETUP_ONLY),
    ],
)
def test_smoke_rejects_non_fake_or_networked_settings(
    tmp_path: Path,
    settings: RepoPilotSettings,
) -> None:
    with pytest.raises(SmokeInputError):
        run_smoke(MANIFEST_PATH, tmp_path, settings)


def test_smoke_rejects_manifest_request_response_mismatch() -> None:
    payload = load_payload()
    payload["request_id"] = str(OTHER_REQUEST_ID)

    with pytest.raises(ValidationError, match="request_id"):
        SmokeManifest.model_validate(payload)


def test_smoke_rejects_duplicate_output_names() -> None:
    payload = load_payload()
    payload["trace_filename"] = payload["result_filename"]

    with pytest.raises(ValidationError, match="must be different"):
        SmokeManifest.model_validate(payload)


def test_smoke_rejects_unauthorized_fake_tool_without_writing(tmp_path: Path) -> None:
    payload = load_payload()
    payload["allowed_tools"] = ["search_code"]
    manifest = SmokeManifest.model_validate(payload)

    with pytest.raises(ProviderError) as captured:
        run_smoke_manifest(manifest, tmp_path, RepoPilotSettings())

    assert captured.value.error_type is ErrorType.UNKNOWN_TOOL
    assert not tmp_path.joinpath("result.json").exists()


def test_smoke_maps_input_and_output_file_failures_to_safe_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(SmokeIOError, match="unable to read"):
        load_smoke_manifest(missing)

    invalid = tmp_path / "invalid.json"
    secret_value = "sensitive-invalid-manifest-value"
    invalid.write_text(f'{{"unknown": "{secret_value}"', encoding="utf-8")
    with pytest.raises(SmokeInputError) as captured:
        load_smoke_manifest(invalid)
    assert secret_value not in str(captured.value)

    output_file = tmp_path / "not-a-directory"
    output_file.write_text("occupied", encoding="utf-8")
    with pytest.raises(SmokeIOError, match="output directory"):
        run_smoke(MANIFEST_PATH, output_file, RepoPilotSettings())


def test_smoke_rejects_output_path_that_resolves_outside_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output_dir.joinpath("linked").symlink_to(outside, target_is_directory=True)
    payload = load_payload()
    payload["result_filename"] = "linked/result.json"
    manifest = SmokeManifest.model_validate(payload)

    with pytest.raises(SmokeIOError, match="escapes"):
        run_smoke_manifest(manifest, output_dir, RepoPilotSettings())
