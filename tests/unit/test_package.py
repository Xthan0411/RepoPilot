"""Smoke tests for the installable RepoPilot package."""

import repopilot


def test_package_version() -> None:
    assert repopilot.__version__ == "0.1.0"
