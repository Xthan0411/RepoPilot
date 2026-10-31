"""Typer CLI tests for exit codes, determinism, and safe error output."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

import repopilot.cli as cli_module
from repopilot.cli import app
from repopilot.enums import CliExitCode, RepairWorkflowErrorCode
from repopilot.evals import BM25BaselineReport
from repopilot.repair_workflow import RepairWorkflowError
from repopilot.replay import TraceReplayResult
from repopilot.retrieval import RepositoryIngestScanPolicy, ingest_repository
from repopilot.schemas import FinalResult
from repopilot.trace import AgentTrace

FIXTURES = Path(__file__).parents[1] / "fixtures"
MANIFEST_PATH = FIXTURES / "p0_smoke_manifest.json"
P1_MANIFEST_PATH = FIXTURES / "p1_agent_manifest.json"
P2_MANIFEST_PATH = FIXTURES / "p2_bm25_baseline_manifest.json"
CONFIG_PATH = Path(__file__).parents[2] / "configs" / "default.yaml"
RUNNER = CliRunner()
CONFIG_ENV_NAMES = (
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "REPOPILOT_PROFILE",
    "REPOPILOT_MODEL_PROVIDER",
    "REPOPILOT_MODEL",
    "REPOPILOT_TEMPERATURE",
    "DATABASE_URL",
    "REPOPILOT_MAX_STEPS",
    "REPOPILOT_MAX_REPLANS",
    "REPOPILOT_MAX_TOKENS",
    "REPOPILOT_TIMEOUT_SECONDS",
    "REPOPILOT_MAX_COST_USD",
    "REPOPILOT_REPOSITORY_MAX_FILE_BYTES",
    "REPOPILOT_REPOSITORY_MAX_READ_BYTES",
    "REPOPILOT_REPOSITORY_MAX_READ_LINES",
    "REPOPILOT_REPOSITORY_MAX_LIST_ENTRIES",
    "REPOPILOT_TOOL_MAX_OUTPUT_BYTES",
    "REPOPILOT_SEARCH_MAX_MATCHES",
    "REPOPILOT_SEARCH_MAX_QUERY_LENGTH",
    "REPOPILOT_CHUNK_MAX_LINES",
    "REPOPILOT_CHUNK_OVERLAP_LINES",
    "REPOPILOT_BM25_K1",
    "REPOPILOT_BM25_B",
    "REPOPILOT_BM25_DEFAULT_TOP_K",
    "REPOPILOT_VECTOR_SCHEMA",
    "REPOPILOT_VECTOR_TABLE",
    "REPOPILOT_VECTOR_DIMENSION",
    "REPOPILOT_DATABASE_CONNECT_TIMEOUT_SECONDS",
    "REPOPILOT_DATABASE_STATEMENT_TIMEOUT_MS",
    "REPOPILOT_SANDBOX_NETWORK",
    "REPOPILOT_LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def clean_configuration_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in CONFIG_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def load_payload() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
    )


def invoke_smoke(manifest: Path, output_dir: Path) -> Any:
    return RUNNER.invoke(
        app,
        [
            "smoke",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--config",
            str(CONFIG_PATH),
        ],
    )


def invoke_run_fixture(manifest: Path, output_dir: Path) -> Any:
    return RUNNER.invoke(
        app,
        [
            "run-fixture",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--config",
            str(CONFIG_PATH),
        ],
    )


def invoke_replay(trace_path: Path) -> Any:
    return RUNNER.invoke(
        app,
        [
            "replay",
            "--trace",
            str(trace_path),
        ],
    )


def invoke_eval_bm25(manifest: Path, output_dir: Path) -> Any:
    return RUNNER.invoke(
        app,
        [
            "eval-bm25",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--config",
            str(CONFIG_PATH),
        ],
    )


def invoke_repair(task: Path, output_dir: Path) -> Any:
    return RUNNER.invoke(
        app,
        [
            "repair",
            "--task",
            str(task),
            "--sandbox-image",
            f"python@sha256:{'a' * 64}",
            "--output-dir",
            str(output_dir),
            "--provider-script",
            str(task.parent / "provider.json"),
            "--config",
            str(CONFIG_PATH),
        ],
    )


def test_cli_smoke_returns_json_and_writes_deterministic_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    first = invoke_smoke(MANIFEST_PATH, output_dir)

    assert first.exit_code == CliExitCode.SUCCESS
    stdout_result = FinalResult.model_validate_json(first.stdout)
    assert stdout_result.status.value == "stopped"
    result_bytes = output_dir.joinpath("result.json").read_bytes()
    trace_bytes = output_dir.joinpath("trace.json").read_bytes()

    second = invoke_smoke(MANIFEST_PATH, output_dir)
    assert second.exit_code == CliExitCode.SUCCESS
    assert second.stdout == first.stdout
    assert output_dir.joinpath("result.json").read_bytes() == result_bytes
    assert output_dir.joinpath("trace.json").read_bytes() == trace_bytes


def test_cli_usage_errors_use_typer_exit_code_two() -> None:
    result = RUNNER.invoke(app, ["smoke"])

    assert result.exit_code == CliExitCode.USAGE_ERROR


def test_cli_missing_manifest_is_io_error(tmp_path: Path) -> None:
    result = invoke_smoke(tmp_path / "missing.json", tmp_path / "output")

    assert result.exit_code == CliExitCode.IO_ERROR
    assert "unable to read smoke manifest" in result.stderr


def test_cli_invalid_manifest_is_safe_input_error(tmp_path: Path) -> None:
    secret_value = "sensitive-cli-invalid-value"
    manifest = tmp_path / "invalid.json"
    manifest.write_text(f'{{"private": "{secret_value}"', encoding="utf-8")

    result = invoke_smoke(manifest, tmp_path / "output")

    assert result.exit_code == CliExitCode.INVALID_INPUT
    assert "invalid smoke manifest" in result.stderr
    assert secret_value not in result.output


def test_cli_output_file_collision_is_io_error(tmp_path: Path) -> None:
    output_path = tmp_path / "occupied"
    output_path.write_text("not a directory", encoding="utf-8")

    result = invoke_smoke(MANIFEST_PATH, output_path)

    assert result.exit_code == CliExitCode.IO_ERROR
    assert "output directory" in result.stderr


def test_cli_rejects_real_provider_without_echoing_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_value = "sensitive-cli-provider-value"
    monkeypatch.setenv("REPOPILOT_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("REPOPILOT_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", secret_value)

    result = invoke_smoke(MANIFEST_PATH, tmp_path / "output")

    assert result.exit_code == CliExitCode.INVALID_INPUT
    assert "model_provider=fake" in result.stderr
    assert secret_value not in result.output


def test_cli_rejects_networked_smoke_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REPOPILOT_SANDBOX_NETWORK", "setup_only")

    result = invoke_smoke(MANIFEST_PATH, tmp_path / "output")

    assert result.exit_code == CliExitCode.INVALID_INPUT
    assert "sandbox_network=none" in result.stderr


def test_cli_maps_provider_capability_failure_to_adapter_exit_code(
    tmp_path: Path,
) -> None:
    payload = load_payload()
    payload["allowed_tools"] = ["search_code"]
    manifest = tmp_path / "unauthorized.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = invoke_smoke(manifest, tmp_path / "output")

    assert result.exit_code == CliExitCode.ADAPTER_ERROR
    assert "outside allowed_tools" in result.stderr
    assert not tmp_path.joinpath("output", "result.json").exists()


def test_cli_hides_unexpected_internal_error_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_value = "sensitive-internal-error-value"

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError(secret_value)

    monkeypatch.setattr(cli_module, "run_smoke", fail)
    result = invoke_smoke(MANIFEST_PATH, tmp_path / "output")

    assert result.exit_code == CliExitCode.INTERNAL_ERROR
    assert "unexpected internal error" in result.stderr
    assert secret_value not in result.output


def test_cli_run_fixture_and_offline_replay_are_deterministic(tmp_path: Path) -> None:
    output_dir = tmp_path / "p1"

    first = invoke_run_fixture(P1_MANIFEST_PATH, output_dir)

    assert first.exit_code == CliExitCode.SUCCESS
    result = FinalResult.model_validate_json(first.stdout)
    trace = AgentTrace.model_validate_json(output_dir.joinpath("trace.json").read_bytes())
    result_bytes = output_dir.joinpath("result.json").read_bytes()
    trace_bytes = output_dir.joinpath("trace.json").read_bytes()
    replayed = invoke_replay(output_dir / "trace.json")

    assert result.termination_reason.value == "agent_stopped"
    assert len(trace.events) == 12
    assert replayed.exit_code == CliExitCode.SUCCESS
    assert TraceReplayResult.model_validate_json(replayed.stdout).final_state == trace.final_state

    second = invoke_run_fixture(P1_MANIFEST_PATH, output_dir)
    assert second.exit_code == CliExitCode.SUCCESS
    assert second.stdout == first.stdout
    assert output_dir.joinpath("result.json").read_bytes() == result_bytes
    assert output_dir.joinpath("trace.json").read_bytes() == trace_bytes


def test_cli_run_fixture_missing_manifest_is_io_error(tmp_path: Path) -> None:
    result = invoke_run_fixture(tmp_path / "missing.json", tmp_path / "output")

    assert result.exit_code == CliExitCode.IO_ERROR
    assert "unable to read Agent run manifest" in result.stderr


def test_cli_replay_missing_trace_is_io_error(tmp_path: Path) -> None:
    result = invoke_replay(tmp_path / "missing.json")

    assert result.exit_code == CliExitCode.IO_ERROR
    assert "could not be read or written" in result.stderr


def test_cli_replay_rejects_tampered_or_unsupported_trace(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    assert invoke_run_fixture(P1_MANIFEST_PATH, output_dir).exit_code == CliExitCode.SUCCESS
    trace_path = output_dir / "trace.json"
    payload = cast(dict[str, Any], json.loads(trace_path.read_text(encoding="utf-8")))
    payload["final_state"]["repository_summary"] = "tampered"
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    tampered = invoke_replay(trace_path)

    assert tampered.exit_code == CliExitCode.INVALID_INPUT
    assert "digest does not match" in tampered.stderr

    payload["schema_version"] = 999
    trace_path.write_text(json.dumps(payload), encoding="utf-8")
    unsupported = invoke_replay(trace_path)
    assert unsupported.exit_code == CliExitCode.INVALID_INPUT
    assert "schema version is unsupported" in unsupported.stderr


def test_cli_eval_bm25_generates_baseline_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "p2"

    result = invoke_eval_bm25(P2_MANIFEST_PATH, output_dir)

    assert result.exit_code == CliExitCode.SUCCESS
    report = BM25BaselineReport.model_validate_json(result.stdout)
    assert report.benchmark_id == "p2.synthetic-smoke"
    assert report.summary.task_count == 3
    assert report.summary.mean_file_recall_at_1 == 1.0
    assert set(path.name for path in output_dir.iterdir()) == {
        "baseline.json",
        "bm25-index.json",
        "chunks.jsonl",
        "summary.md",
        "tasks.csv",
        "tasks.jsonl",
    }
    assert "EVALUATOR_ONLY_PATCH_CANARY" not in result.stdout


def test_cli_eval_bm25_maps_input_output_and_retrieval_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = invoke_eval_bm25(tmp_path / "missing.json", tmp_path / "missing-output")
    assert missing.exit_code == CliExitCode.IO_ERROR
    assert "unable to read BM25 baseline input" in missing.stderr

    occupied = tmp_path / "occupied"
    occupied.write_text("file", encoding="utf-8")
    output_error = invoke_eval_bm25(P2_MANIFEST_PATH, occupied)
    assert output_error.exit_code == CliExitCode.IO_ERROR
    assert "unable to write BM25 baseline artifacts" in output_error.stderr

    monkeypatch.setattr(
        "repopilot.evals.baseline.ingest_repository",
        partial(ingest_repository, scan_policy=RepositoryIngestScanPolicy(max_list_entries=1)),
    )
    retrieval_error = invoke_eval_bm25(P2_MANIFEST_PATH, tmp_path / "retrieval")
    assert retrieval_error.exit_code == CliExitCode.ADAPTER_ERROR
    assert "indexing or retrieval failed" in retrieval_error.stderr


def test_cli_bm25_inventory_is_independent_of_agent_tool_listing_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPOPILOT_REPOSITORY_MAX_LIST_ENTRIES", "1")
    result = invoke_eval_bm25(P2_MANIFEST_PATH, tmp_path / "retrieval")
    assert result.exit_code == CliExitCode.SUCCESS
    assert BM25BaselineReport.model_validate_json(result.stdout).summary.task_count == 3


@pytest.mark.parametrize(
    ("error_code", "exit_code"),
    (
        (RepairWorkflowErrorCode.INVALID_TASK, CliExitCode.INVALID_INPUT),
        (RepairWorkflowErrorCode.OUTPUT_IO_ERROR, CliExitCode.IO_ERROR),
        (RepairWorkflowErrorCode.SANDBOX_PREPARATION, CliExitCode.ADAPTER_ERROR),
        (RepairWorkflowErrorCode.TRACE_FAILURE, CliExitCode.INTERNAL_ERROR),
    ),
)
def test_cli_repair_maps_typed_workflow_failures_without_leaking_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_code: RepairWorkflowErrorCode,
    exit_code: CliExitCode,
) -> None:
    secret = "sensitive-repair-cli-value"

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RepairWorkflowError(error_code)

    monkeypatch.setattr(cli_module, "run_repair_task", fail)
    task = tmp_path / "task.json"
    task.write_text("{}", encoding="utf-8")

    result = invoke_repair(task, tmp_path / "output")

    assert result.exit_code == exit_code
    assert "error:" in result.stderr
    assert secret not in result.output
