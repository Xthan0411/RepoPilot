"""P7-S3 baseline configuration, fairness, and information-boundary tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    BaselineErrorCode,
    BaselineKind,
    RepairContextMode,
    RetrievalMode,
)
from repopilot.evals import (
    BASELINE_TOOL_NAMES,
    BaselineDefinition,
    BaselineError,
    BaselineEvaluationExecutor,
    baseline_configuration_snapshot,
    build_baseline_runtime_contract,
    load_baseline_definition,
    require_comparable_baselines,
)
from repopilot.repair_workflow import build_repair_no_precomputed_context
from repopilot.schemas import CommandSpec, TaskSpec
from repopilot.tools import ReadOnlyRepository

PROJECT_ROOT = Path(__file__).parents[2]
BASELINE_CONFIGS = PROJECT_ROOT / "configs" / "baselines"
MINI_SWE_CONFIG = BASELINE_CONFIGS / "mini-swe-safe-v1.yaml"
NO_SEMANTIC_CONFIG = BASELINE_CONFIGS / "no-semantic-rag-v1.yaml"
IMAGE = f"python@sha256:{'a' * 64}"


def _settings(**updates: object) -> RepoPilotSettings:
    values: dict[str, object] = {
        "max_steps": 20,
        "max_tokens": 20_000,
        "timeout_seconds": 120,
        "retrieval_mode": RetrievalMode.BM25,
        "query_rewrite_enabled": False,
    }
    values.update(updates)
    return RepoPilotSettings.model_validate(values)


def _task(repository: Path) -> TaskSpec:
    return TaskSpec(
        task_id="baseline-no-precomputed-context",
        repository_path=str(repository),
        base_commit="b" * 40,
        issue_title="Return normalized parser output",
        issue_body="The parser should normalize blank input.",
        test_commands=(
            CommandSpec(
                argv=("python", "-m", "unittest", "-q"),
                timeout_seconds=30,
            ),
        ),
        timeout_seconds=60,
    )


def test_frozen_baselines_share_runtime_fairness_but_not_context() -> None:
    mini_swe = load_baseline_definition(MINI_SWE_CONFIG)
    no_semantic = load_baseline_definition(NO_SEMANTIC_CONFIG)
    settings = _settings()

    mini_contract = build_baseline_runtime_contract(
        mini_swe,
        settings=settings,
        sandbox_image_reference=IMAGE,
    )
    no_semantic_contract = build_baseline_runtime_contract(
        no_semantic,
        settings=settings,
        sandbox_image_reference=IMAGE,
    )

    assert mini_swe.kind is BaselineKind.MINI_SWE_SAFE
    assert mini_contract.context_mode is RepairContextMode.NONE
    assert mini_contract.retrieval_mode is None
    assert no_semantic.kind is BaselineKind.NO_SEMANTIC_RAG
    assert no_semantic_contract.context_mode is RepairContextMode.BM25
    assert no_semantic_contract.retrieval_mode is RetrievalMode.BM25
    assert mini_contract.fairness == no_semantic_contract.fairness
    assert require_comparable_baselines((mini_contract, no_semantic_contract)) == (
        mini_contract.fairness.content_sha256
    )
    assert mini_contract.content_sha256 != no_semantic_contract.content_sha256
    assert mini_contract.fairness.tool_names == BASELINE_TOOL_NAMES
    snapshot = json.dumps(
        baseline_configuration_snapshot(no_semantic_contract),
        sort_keys=True,
    )
    assert "expected_patch" not in snapshot
    assert "gold_files" not in snapshot
    assert "test_patch" not in snapshot
    assert "database_url" not in snapshot
    assert "api_key" not in snapshot


@pytest.mark.parametrize(
    ("mutator", "error_code"),
    (
        (
            lambda text: text.replace("schema_version: 1", "schema_version: 2"),
            BaselineErrorCode.UNSUPPORTED_VERSION,
        ),
        (
            lambda text: f"{text}\nexpected_patch: evaluator-canary\n",
            BaselineErrorCode.INFORMATION_BOUNDARY,
        ),
        (
            lambda text: text.replace(
                "  - search_code",
                "  - search_code\n  - unrestricted_shell",
            ),
            BaselineErrorCode.CAPABILITY_MISMATCH,
        ),
        (
            lambda text: text.replace(
                "arbitrary_shell_enabled: false",
                "arbitrary_shell_enabled: true",
            ),
            BaselineErrorCode.CAPABILITY_MISMATCH,
        ),
    ),
)
def test_loader_rejects_unsupported_leaky_or_more_powerful_configs(
    tmp_path: Path,
    mutator: object,
    error_code: BaselineErrorCode,
) -> None:
    assert callable(mutator)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        mutator(MINI_SWE_CONFIG.read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    with pytest.raises(BaselineError) as captured:
        load_baseline_definition(invalid)

    assert captured.value.code is error_code
    assert "evaluator-canary" not in str(captured.value)


def test_runtime_rejects_semantic_retrieval_rewrite_and_fairness_drift() -> None:
    mini_swe = load_baseline_definition(MINI_SWE_CONFIG)
    no_semantic = load_baseline_definition(NO_SEMANTIC_CONFIG)

    with pytest.raises(BaselineError) as captured:
        build_baseline_runtime_contract(
            no_semantic,
            settings=_settings(retrieval_mode=RetrievalMode.DENSE),
            sandbox_image_reference=IMAGE,
        )
    assert captured.value.code is BaselineErrorCode.CAPABILITY_MISMATCH

    with pytest.raises(BaselineError) as captured:
        build_baseline_runtime_contract(
            mini_swe,
            settings=_settings(query_rewrite_enabled=True),
            sandbox_image_reference=IMAGE,
        )
    assert captured.value.code is BaselineErrorCode.CAPABILITY_MISMATCH

    first = build_baseline_runtime_contract(
        mini_swe,
        settings=_settings(),
        sandbox_image_reference=IMAGE,
    )
    drifted = build_baseline_runtime_contract(
        no_semantic,
        settings=_settings(max_steps=21),
        sandbox_image_reference=IMAGE,
    )
    with pytest.raises(BaselineError) as captured:
        require_comparable_baselines((first, drifted))
    assert captured.value.code is BaselineErrorCode.FAIRNESS_VIOLATION

    with pytest.raises(BaselineError) as captured:
        build_baseline_runtime_contract(
            mini_swe,
            settings=_settings(),
            sandbox_image_reference="python:3.12",
        )
    assert captured.value.code is BaselineErrorCode.FAIRNESS_VIOLATION


def test_definition_rejects_wrong_identity_and_executor_cannot_accept_memory() -> None:
    payload = load_baseline_definition(MINI_SWE_CONFIG).model_dump(mode="python")
    payload["context_mode"] = RepairContextMode.BM25
    with pytest.raises(ValidationError):
        BaselineDefinition.model_validate(payload)

    assert "memory_stores" not in BaselineEvaluationExecutor.__dataclass_fields__


def test_mini_swe_context_has_repository_identity_but_no_preloaded_evidence(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    repository.joinpath("parser.py").write_text(
        "def normalize(value: str) -> str:\n    return value.strip()\n",
        encoding="utf-8",
    )
    settings = _settings()
    context = build_repair_no_precomputed_context(
        ReadOnlyRepository.from_settings(repository, settings),
        _task(repository),
    )

    assert context.repository_id.startswith("repo_")
    assert len(context.repository_content_sha256) == 64
    assert context.hits == ()
    assert context.repository_memory == ()
    assert context.episodic_memory == ()
    assert context.memory_namespace is None
    assert context.indexing_latency_ms == 0
    assert context.retrieval_latency_ms == 0
    assert "parser.py" not in context.repository_summary
