"""P7-S6 all-task Sandbox preflight contract tests."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from repopilot.evals.dataset import load_frozen_dataset

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p7_django_sandbox_canary import (  # noqa: E402
    RESULT_PATH as CANARY_RESULT_PATH,
)
from scripts.p7_django_sandbox_canary import (  # noqa: E402
    CanaryError,
    DjangoSandboxCanary,
    _verify_ref,
    build_dockerfile,
)
from scripts.p7_sandbox_matrix_preflight import (  # noqa: E402
    _DISTRIBUTIONS,
    _FLASK_BUILD_TOOLS,
    _FLASK_TASK_ID,
    _SCM_BUILD_TOOLS,
    _SYMPY_TEST_TOOLS,
    _TASK_BUILD_TOOLS,
    _TASK_SCM_PRETEND_VERSIONS,
    DATASET_MANIFEST_PATH,
    RESULT_PATH,
    SandboxMatrixPreflight,
    _dependency_packages,
    _safe_name,
    _selected_task_ids,
    build_conda_adapter_dockerfile,
    build_python_adapter_dockerfile,
)
from scripts.p7_swebench_environment import (  # noqa: E402
    EXPECTED_TASK_IDS,
    QUALIFICATION_PATH,
    SweBenchUpstreamQualification,
)


def test_committed_x86_matrix_evidence_is_complete_and_cross_referenced() -> None:
    qualification = SweBenchUpstreamQualification.model_validate_json(
        PROJECT_ROOT.joinpath(QUALIFICATION_PATH).read_bytes()
    )
    canary = DjangoSandboxCanary.model_validate_json(
        PROJECT_ROOT.joinpath(CANARY_RESULT_PATH).read_bytes()
    )
    preflight_bytes = PROJECT_ROOT.joinpath(RESULT_PATH).read_bytes()
    preflight = SandboxMatrixPreflight.model_validate_json(preflight_bytes)

    _verify_ref(PROJECT_ROOT, qualification.baseline_report_ref)
    _verify_ref(PROJECT_ROOT, qualification.gold_report_ref)
    assert preflight.controller_commit == "5f2076fdfc6bfb9963e4524a9727434d7f5d2707"
    assert tuple(item.task_id for item in preflight.task_evidence) == EXPECTED_TASK_IDS
    assert preflight.upstream_qualification_sha256 == qualification.content_sha256
    assert canary.upstream_qualification_sha256 == qualification.content_sha256
    assert preflight.adapter_verified_task_count == preflight.required_task_count == 8
    assert not preflight.python_mismatch_task_ids
    assert preflight.paid_requests_made == 0
    assert not preflight.gold_visible_to_agent
    assert preflight.compatibility_preflight_passed
    assert not preflight.formal_environment_ready
    assert preflight.formal_blockers == (
        "registry_digest_required",
        "task_command_timeout_contract_mismatch",
    )
    assert hashlib.sha256(preflight_bytes).hexdigest() == (
        "9504aed7904436f807d890077f927dc3a4c0a6cfa890d383811535c2c8889c1c"
    )


def test_all_frozen_tasks_have_safe_project_adapter_distributions() -> None:
    assert tuple(_DISTRIBUTIONS) == EXPECTED_TASK_IDS

    image = f"swebench/task@sha256:{'a' * 64}"
    for distribution in _DISTRIBUTIONS.values():
        dockerfile = build_dockerfile(image, distribution_name=distribution)
        assert f"python -m pip uninstall -y {distribution}" in dockerfile
        assert "no-index = true" in dockerfile
        assert "no-build-isolation = false" in dockerfile


@pytest.mark.parametrize("task_id", EXPECTED_TASK_IDS)
def test_task_identity_maps_to_safe_local_image_name(task_id: str) -> None:
    safe_name = _safe_name(task_id)

    assert safe_name
    assert "_" not in safe_name
    assert set(safe_name) <= set("abcdefghijklmnopqrstuvwxyz0123456789.-")


def test_unsafe_task_identity_is_rejected() -> None:
    with pytest.raises(CanaryError, match="task_id_invalid"):
        _safe_name("task/id")


def test_flask_uses_frozen_python_adapter_with_pinned_bootstrap() -> None:
    bundle = load_frozen_dataset(
        PROJECT_ROOT / DATASET_MANIFEST_PATH,
        repository_root=PROJECT_ROOT,
    )
    task = next(
        item.task
        for item in bundle.external_evaluation_tasks
        if item.task.task_id == _FLASK_TASK_ID
    )
    packages = _dependency_packages(task)

    dockerfile = build_python_adapter_dockerfile(
        f"python@sha256:{'a' * 64}",
        bootstrap_packages=packages,
        build_tools=_FLASK_BUILD_TOOLS,
    )

    assert packages == task.setup_commands[0].argv[4:]
    assert "setuptools==70.0.0" in dockerfile
    assert "pytest==8.4.2" in dockerfile
    assert "wheel==0.45.1" in dockerfile
    assert 'SHELL ["/bin/sh", "-c"]' in dockerfile
    assert "/sbin/nologin" in dockerfile
    assert "/bin/bash" not in dockerfile
    assert "no-index = true" in dockerfile
    assert "expected_patch" not in dockerfile
    assert "OPENAI" not in dockerfile


def test_source_build_metadata_maps_to_pinned_conda_adapter_tools() -> None:
    assert tuple(_TASK_BUILD_TOOLS) == EXPECTED_TASK_IDS[3:]
    assert _TASK_BUILD_TOOLS["pydata__xarray-4094"] == _SCM_BUILD_TOOLS
    assert "setuptools-scm==7.1.0" in _SCM_BUILD_TOOLS
    assert "setuptools-scm-git-archive==1.4.1" in _TASK_BUILD_TOOLS["pydata__xarray-5131"]
    assert _TASK_BUILD_TOOLS["pytest-dev__pytest-11143"] == _SCM_BUILD_TOOLS
    assert _TASK_BUILD_TOOLS["sympy__sympy-20590"] == _SYMPY_TEST_TOOLS
    assert "pytest==7.4.0" in _SYMPY_TEST_TOOLS

    dockerfile = build_conda_adapter_dockerfile(
        f"swebench/task@sha256:{'a' * 64}",
        distribution_name="xarray",
        build_tools=_TASK_BUILD_TOOLS["pydata__xarray-5131"],
    )

    assert "python -m pip install" in dockerfile
    assert "--no-deps setuptools-scm==7.1.0" in dockerfile
    assert "setuptools-scm-git-archive==1.4.1" in dockerfile
    assert "python -m pip uninstall -y xarray" in dockerfile
    assert dockerfile.index("python -m pip install") < dockerfile.index("python -m pip uninstall")
    assert "no-index = true" in dockerfile

    no_git_dockerfile = build_conda_adapter_dockerfile(
        f"swebench/task@sha256:{'a' * 64}",
        distribution_name="pytest",
        build_tools=_SCM_BUILD_TOOLS,
        scm_pretend_version=_TASK_SCM_PRETEND_VERSIONS["pytest-dev__pytest-11143"],
    )
    assert 'SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTEST="8.0.0.dev53+g6995257cf"' in no_git_dockerfile


def test_conda_adapter_rejects_unpinned_build_tools() -> None:
    with pytest.raises(CanaryError, match="build_tools_invalid"):
        build_conda_adapter_dockerfile(
            f"swebench/task@sha256:{'a' * 64}",
            distribution_name="pytest",
            build_tools=("setuptools-scm>=6.2.3",),
        )
    with pytest.raises(CanaryError, match="scm_pretend_version_invalid"):
        build_conda_adapter_dockerfile(
            f"swebench/task@sha256:{'a' * 64}",
            distribution_name="pytest",
            build_tools=(),
            scm_pretend_version="8.0.0; touch /tmp/unsafe",
        )


def test_diagnostic_task_selection_is_ordered_and_deduplicated() -> None:
    assert _selected_task_ids(None) == EXPECTED_TASK_IDS
    assert _selected_task_ids(
        [
            "sympy__sympy-20590",
            "pydata__xarray-4094",
            "sympy__sympy-20590",
        ]
    ) == ("pydata__xarray-4094", "sympy__sympy-20590")

    with pytest.raises(CanaryError, match="task_selection_invalid"):
        _selected_task_ids([])


def test_python_adapter_rejects_unpinned_or_shell_packages() -> None:
    image = f"python@sha256:{'a' * 64}"

    with pytest.raises(CanaryError, match="bootstrap_packages_invalid"):
        build_python_adapter_dockerfile(
            image,
            bootstrap_packages=("pytest",),
            build_tools=_FLASK_BUILD_TOOLS,
        )
    with pytest.raises(CanaryError, match="bootstrap_packages_invalid"):
        build_python_adapter_dockerfile(
            image,
            bootstrap_packages=("pytest==8.4.2;curl",),
            build_tools=_FLASK_BUILD_TOOLS,
        )


def test_direct_file_help_invocation_resolves_preflight_imports() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            str(PROJECT_ROOT / "scripts/p7_sandbox_matrix_preflight.py"),
            "--help",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Qualify all frozen P7 holdout tasks" in completed.stdout
