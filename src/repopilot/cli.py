"""Typer command-line entry point for RepoPilot."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from repopilot.config import (
    DEFAULT_CONFIG_PATH,
    EVAL_CONFIG_PATH,
    ConfigurationError,
    load_settings,
)
from repopilot.enums import (
    CliExitCode,
    LocalizationEvalErrorCode,
    RepairWorkflowErrorCode,
    TraceErrorCode,
)
from repopilot.evals import LocalizationEvalError, run_bm25_baseline
from repopilot.providers.base import ProviderError
from repopilot.repair_workflow import RepairWorkflowError, run_repair_task
from repopilot.replay import replay_trace
from repopilot.smoke import SmokeInputError, SmokeIOError, run_smoke
from repopilot.trace import TraceArtifactError, read_trace
from repopilot.workflow import (
    AgentRunInputError,
    AgentRunIOError,
    run_agent_fixture,
)

app = typer.Typer(
    name="repopilot",
    help="Evaluation-driven repository Agent tooling.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Run explicit RepoPilot workflows."""


@app.command("smoke")
def smoke_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Self-contained P0 smoke manifest JSON.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for deterministic result and trace JSON.",
        ),
    ] = Path("results/p0-smoke"),
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="RepoPilot YAML configuration profile.",
        ),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Run one deterministic Fake-provider action without claiming task success."""

    try:
        settings = load_settings(config)
        artifacts = run_smoke(manifest, output_dir, settings)
    except (ConfigurationError, SmokeInputError) as error:
        _abort(str(error), CliExitCode.INVALID_INPUT)
    except SmokeIOError as error:
        _abort(str(error), CliExitCode.IO_ERROR)
    except ProviderError as error:
        _abort(str(error), CliExitCode.ADAPTER_ERROR)
    except Exception:
        _abort("unexpected internal error", CliExitCode.INTERNAL_ERROR)

    typer.echo(artifacts.result.model_dump_json(indent=2))


@app.command("run-fixture")
def run_fixture_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Self-contained P1 Fake Provider/Sandbox manifest JSON.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for deterministic Agent result and Trace JSON.",
        ),
    ] = Path("results/p1-fixture"),
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="RepoPilot YAML configuration profile.",
        ),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Run the complete deterministic P1 Agent loop and persist its Trace."""

    try:
        settings = load_settings(config)
        artifacts = run_agent_fixture(manifest, output_dir, settings)
    except (ConfigurationError, AgentRunInputError) as error:
        _abort(str(error), CliExitCode.INVALID_INPUT)
    except AgentRunIOError as error:
        _abort(str(error), CliExitCode.IO_ERROR)
    except TraceArtifactError as error:
        exit_code = (
            CliExitCode.IO_ERROR
            if error.code is TraceErrorCode.IO_ERROR
            else CliExitCode.INTERNAL_ERROR
        )
        _abort(str(error), exit_code)
    except Exception:
        _abort("unexpected internal error", CliExitCode.INTERNAL_ERROR)

    typer.echo(artifacts.result.model_dump_json(indent=2))


@app.command("replay")
def replay_command(
    trace_path: Annotated[
        Path,
        typer.Option(
            "--trace",
            help="Versioned RepoPilot Agent Trace JSON.",
        ),
    ],
) -> None:
    """Verify and replay one Trace without Provider, tool, or Sandbox calls."""

    try:
        replayed = replay_trace(read_trace(trace_path))
    except TraceArtifactError as error:
        exit_code = (
            CliExitCode.IO_ERROR
            if error.code is TraceErrorCode.IO_ERROR
            else CliExitCode.INVALID_INPUT
        )
        _abort(str(error), exit_code)
    except Exception:
        _abort("unexpected internal error", CliExitCode.INTERNAL_ERROR)

    typer.echo(replayed.model_dump_json(indent=2))


@app.command("eval-bm25")
def eval_bm25_command(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="Versioned evaluator-only BM25 task manifest JSON.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for corpus, index, task metrics, and summary artifacts.",
        ),
    ] = Path("results/p2-baseline"),
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="RepoPilot YAML configuration profile.",
        ),
    ] = EVAL_CONFIG_PATH,
) -> None:
    """Build and evaluate one fixed, local, gold-isolated BM25 baseline."""

    try:
        settings = load_settings(config)
        artifacts = run_bm25_baseline(manifest, output_dir, settings)
    except ConfigurationError as error:
        _abort(str(error), CliExitCode.INVALID_INPUT)
    except LocalizationEvalError as error:
        if error.code in {
            LocalizationEvalErrorCode.INPUT_IO_ERROR,
            LocalizationEvalErrorCode.OUTPUT_IO_ERROR,
        }:
            exit_code = CliExitCode.IO_ERROR
        elif error.code is LocalizationEvalErrorCode.RETRIEVAL_FAILURE:
            exit_code = CliExitCode.ADAPTER_ERROR
        else:
            exit_code = CliExitCode.INVALID_INPUT
        _abort(str(error), exit_code)
    except Exception:
        _abort("unexpected internal error", CliExitCode.INTERNAL_ERROR)

    typer.echo(artifacts.report.model_dump_json(indent=2))


@app.command("repair")
def repair_command(
    task: Annotated[
        Path,
        typer.Option(
            "--task",
            help="Agent-only TaskSpec JSON with one local repository path.",
        ),
    ],
    sandbox_image: Annotated[
        str,
        typer.Option(
            "--sandbox-image",
            help="Immutable Docker image reference pinned by sha256 digest.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Empty directory for result, Trace, and Sandbox artifacts.",
        ),
    ] = Path("results/repair"),
    provider_script: Annotated[
        Path | None,
        typer.Option(
            "--provider-script",
            help="Ordered Fake Provider decisions; forbidden for OpenAI runs.",
        ),
    ] = None,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="RepoPilot YAML configuration profile.",
        ),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Run the production retrieval-to-verified-repair workflow."""

    try:
        settings = load_settings(config)
        artifacts = run_repair_task(
            task_path=task,
            output_dir=output_dir,
            sandbox_image_reference=sandbox_image,
            settings=settings,
            provider_script_path=provider_script,
        )
    except ConfigurationError as error:
        _abort(str(error), CliExitCode.INVALID_INPUT)
    except RepairWorkflowError as error:
        _abort(str(error), _repair_exit_code(error.code))
    except Exception:
        _abort("unexpected internal error", CliExitCode.INTERNAL_ERROR)

    typer.echo(artifacts.result.model_dump_json(indent=2))


def _repair_exit_code(code: RepairWorkflowErrorCode) -> CliExitCode:
    if code in {
        RepairWorkflowErrorCode.INPUT_IO_ERROR,
        RepairWorkflowErrorCode.INVALID_OUTPUT,
        RepairWorkflowErrorCode.OUTPUT_IO_ERROR,
    }:
        return CliExitCode.IO_ERROR
    if code in {
        RepairWorkflowErrorCode.INVALID_TASK,
        RepairWorkflowErrorCode.INVALID_PROVIDER_SCRIPT,
        RepairWorkflowErrorCode.PROVIDER_CONFIGURATION,
    }:
        return CliExitCode.INVALID_INPUT
    if code is RepairWorkflowErrorCode.TRACE_FAILURE:
        return CliExitCode.INTERNAL_ERROR
    return CliExitCode.ADAPTER_ERROR


def _abort(message: str, exit_code: CliExitCode) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=int(exit_code))
