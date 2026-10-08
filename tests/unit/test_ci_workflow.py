"""Structural tests for the ordinary no-secret, no-paid-service CI workflow."""

from pathlib import Path
from typing import Any, cast

import yaml

PROJECT_ROOT = Path(__file__).parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
EXPECTED_COMMANDS = (
    "uv sync --locked",
    "uv lock --check",
    "uv run pytest -q",
    "uv run ruff check .",
    "uv run ruff format --check .",
    "uv run mypy src/repopilot",
)


def load_workflow() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8")),
    )


def test_ci_workflow_has_read_only_permissions_and_quality_commands() -> None:
    workflow = load_workflow()
    quality = workflow["jobs"]["quality"]
    steps = quality["steps"]
    commands = tuple(step["run"] for step in steps if "run" in step)

    assert workflow["permissions"] == {"contents": "read"}
    assert quality["runs-on"] == "ubuntu-latest"
    assert quality["timeout-minutes"] == 15
    assert commands == EXPECTED_COMMANDS


def test_ci_workflow_pins_python_and_uv_without_external_services() -> None:
    workflow = load_workflow()
    serialized = WORKFLOW_PATH.read_text(encoding="utf-8").lower()
    setup_step = workflow["jobs"]["quality"]["steps"][1]

    assert setup_step["with"]["python-version"] == "3.12"
    assert setup_step["with"]["version"] == "0.11.32"
    assert setup_step["uses"].startswith("astral-sh/setup-uv@")
    assert "secrets." not in serialized
    assert "openai_api_key" not in serialized
    assert "docker" not in serialized
