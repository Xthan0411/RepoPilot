"""Deterministic P1 fixture workflow with full Agent Trace artifacts."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, ValidationError, model_validator

from repopilot.agent import HappyPathAgentGraph, decision_request_id
from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    ErrorCategory,
    ModelProvider,
    ReproductionStatus,
    RunStatus,
    SandboxNetworkMode,
    TerminationReason,
    TestStatus,
    ValidationStatus,
)
from repopilot.providers import FakeProvider, ProviderResponse
from repopilot.replay import replay_trace
from repopilot.sandbox import FakeSandbox, FakeSandboxStep, SandboxHandle
from repopilot.schemas import (
    AgentState,
    BudgetState,
    FinalResult,
    LocalizationMetrics,
    RelativePath,
    ReproductionResult,
    StrictModel,
    TaskMetrics,
    TaskSpec,
    UsageMetrics,
    ValidationResult,
)
from repopilot.tools import (
    ReadOnlyRepository,
    RunTestsTool,
    build_read_only_tool_registry,
)
from repopilot.trace import AgentTrace, write_trace


class AgentRunInputError(ValueError):
    """Safely reportable fixture or deterministic-run configuration failure."""


class AgentRunIOError(RuntimeError):
    """Safely reportable fixture or artifact I/O failure."""


class AgentRunManifest(StrictModel):
    """Self-contained Fake Provider/Sandbox script for one P1 Agent run."""

    schema_version: Literal[1] = 1
    thread_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    run_id: UUID
    started_at: AwareDatetime
    task: TaskSpec
    repository_root: RelativePath
    repository_summary: str = ""
    provider_responses: tuple[ProviderResponse, ...] = Field(min_length=1)
    sandbox_handle: SandboxHandle
    sandbox_steps: tuple[FakeSandboxStep, ...] = Field(min_length=1)
    result_filename: RelativePath = "result.json"
    trace_filename: RelativePath = "trace.json"

    @model_validator(mode="after")
    def validate_correlations(self) -> Self:
        if (
            self.sandbox_handle.task_id != self.task.task_id
            or self.sandbox_handle.repository_commit != self.task.base_commit
        ):
            raise ValueError("sandbox handle must match task identity")
        for index, response in enumerate(self.provider_responses):
            if response.request_id != decision_request_id(self.run_id, index):
                raise ValueError("provider response request IDs must follow decision order")
        if self.result_filename == self.trace_filename:
            raise ValueError("result_filename and trace_filename must be different")
        return self


@dataclass(frozen=True)
class AgentRunArtifacts:
    """In-memory P1 outcome and its deterministic artifact paths."""

    result: FinalResult
    trace: AgentTrace
    result_path: Path
    trace_path: Path


def load_agent_run_manifest(path: Path) -> AgentRunManifest:
    """Read and validate one fixture without echoing rejected values."""

    try:
        serialized = path.read_text(encoding="utf-8")
    except OSError:
        raise AgentRunIOError("unable to read Agent run manifest") from None
    try:
        return AgentRunManifest.model_validate_json(serialized)
    except ValidationError as error:
        raise AgentRunInputError(_safe_validation_message("Agent run manifest", error)) from None


def run_agent_fixture(
    manifest_path: Path,
    output_dir: Path,
    settings: RepoPilotSettings,
) -> AgentRunArtifacts:
    """Load and execute one deterministic full P1 Agent fixture."""

    manifest = load_agent_run_manifest(manifest_path)
    repository_root = _resolve_repository_root(
        manifest_path,
        manifest.repository_root,
    )
    return run_agent_manifest(
        manifest,
        repository_root=repository_root,
        output_dir=output_dir,
        settings=settings,
    )


def run_agent_manifest(
    manifest: AgentRunManifest,
    *,
    repository_root: Path,
    output_dir: Path,
    settings: RepoPilotSettings,
) -> AgentRunArtifacts:
    """Execute the Agent, verify replay, then write Trace and result atomically."""

    _validate_run_settings(settings)
    repository = ReadOnlyRepository.from_settings(repository_root, settings)
    registry = build_read_only_tool_registry(repository, settings)
    sandbox = FakeSandbox(
        handle=manifest.sandbox_handle,
        steps=manifest.sandbox_steps,
    )
    registry.register(
        RunTestsTool.prepare(
            sandbox=sandbox,
            task=manifest.task,
        )
    )
    provider = FakeProvider(
        manifest.provider_responses,
        provider_name=settings.model,
    )
    initial = AgentState(
        task=manifest.task,
        repository_summary=manifest.repository_summary,
        budget=BudgetState(
            max_steps=settings.max_steps,
            max_tokens=settings.max_tokens,
            max_cost_usd=settings.max_cost_usd,
            started_at=manifest.started_at,
            deadline_at=(manifest.started_at + timedelta(seconds=settings.timeout_seconds)),
        ),
    )
    graph = HappyPathAgentGraph.from_settings(
        provider=provider,
        registry=registry,
        settings=settings,
        clock=lambda: manifest.started_at,
    )
    final_state = graph.run(
        thread_id=manifest.thread_id,
        run_id=manifest.run_id,
        state=initial,
    )
    trace = graph.build_trace(manifest.thread_id, settings=settings)
    replay_trace(trace)
    result = _build_result(manifest, trace, final_state)
    result_path, trace_path = _prepare_output_paths(manifest, output_dir)
    write_trace(trace_path, trace)
    _write_result(result_path, result)
    return AgentRunArtifacts(
        result=result,
        trace=trace,
        result_path=result_path,
        trace_path=trace_path,
    )


def _validate_run_settings(settings: RepoPilotSettings) -> None:
    if settings.model_provider is not ModelProvider.FAKE:
        raise AgentRunInputError("P1 fixture requires model_provider=fake")
    if settings.model != "fake-deterministic":
        raise AgentRunInputError("P1 fixture requires model=fake-deterministic")
    if settings.sandbox_network is not SandboxNetworkMode.NONE:
        raise AgentRunInputError("P1 fixture requires sandbox_network=none")


def _build_result(
    manifest: AgentRunManifest,
    trace: AgentTrace,
    final_state: AgentState,
) -> FinalResult:
    responses = tuple(
        event.provider_response for event in trace.events if event.provider_response is not None
    )
    termination = final_state.termination_reason
    if termination is None:
        raise AgentRunInputError("Agent run did not reach a terminal state")
    return FinalResult(
        task_id=manifest.task.task_id,
        run_id=manifest.run_id,
        status=(
            RunStatus.STOPPED
            if termination is TerminationReason.AGENT_STOPPED
            else RunStatus.FAILED
        ),
        termination_reason=termination,
        patch_path=None,
        reproduction_result=ReproductionResult(
            status=ReproductionStatus.NOT_ATTEMPTED,
        ),
        validation_result=ValidationResult(
            status=ValidationStatus.NOT_RUN,
        ),
        localization_metrics=LocalizationMetrics(
            file_recall_at_1=0,
            file_recall_at_3=0,
            file_recall_at_5=0,
            mean_reciprocal_rank=0,
            first_relevant_file_rank=None,
            indexing_latency_ms=0,
            retrieval_latency_ms=0,
        ),
        task_metrics=TaskMetrics(
            issue_reproduced=False,
            patch_applied=False,
            tests_passed=final_state.test_state.status is TestStatus.PASSED,
            task_success=False,
            tool_steps=len(final_state.tool_results),
        ),
        usage=UsageMetrics(
            input_tokens=sum(response.usage.input_tokens for response in responses),
            output_tokens=sum(response.usage.output_tokens for response in responses),
            total_tokens=sum(response.usage.total_tokens for response in responses),
            estimated_cost_usd=sum(
                (response.usage.estimated_cost_usd for response in responses),
                start=Decimal("0"),
            ),
            latency_ms=sum(response.usage.latency_ms for response in responses),
        ),
        trace_path=manifest.trace_filename,
        error_category=_error_category(termination),
    )


def _error_category(reason: TerminationReason) -> ErrorCategory | None:
    if reason is TerminationReason.AGENT_STOPPED:
        return None
    if reason is TerminationReason.RETRIEVAL_EXHAUSTED:
        return ErrorCategory.RETRIEVAL_MISS
    if reason in {
        TerminationReason.MAX_STEPS,
        TerminationReason.TOKEN_LIMIT,
        TerminationReason.TIMEOUT,
        TerminationReason.COST_LIMIT,
    }:
        return ErrorCategory.BUDGET_EXHAUSTED
    if reason is TerminationReason.SANDBOX_ERROR:
        return ErrorCategory.SANDBOX_FAILURE
    if reason is TerminationReason.PROVIDER_ERROR:
        return ErrorCategory.MODEL_FORMAT_ERROR
    if reason is TerminationReason.TESTS_FAILED:
        return ErrorCategory.TEST_MISREAD
    return ErrorCategory.TOOL_MISUSE


def _resolve_repository_root(manifest_path: Path, relative_root: str) -> Path:
    try:
        fixture_root = manifest_path.resolve(strict=True).parent
        repository_root = (fixture_root / relative_root).resolve(strict=True)
    except OSError:
        raise AgentRunIOError("unable to resolve fixture repository") from None
    if not repository_root.is_relative_to(fixture_root):
        raise AgentRunIOError("fixture repository escapes the manifest directory")
    return repository_root


def _prepare_output_paths(
    manifest: AgentRunManifest,
    output_dir: Path,
) -> tuple[Path, Path]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        root = output_dir.resolve(strict=True)
        result_path = _contained_output_path(root, manifest.result_filename)
        trace_path = _contained_output_path(root, manifest.trace_filename)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise AgentRunIOError("unable to prepare Agent output directory") from None
    return result_path, trace_path


def _contained_output_path(root: Path, relative_path: str) -> Path:
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root):
        raise AgentRunIOError("Agent output path escapes the output directory")
    return target


def _write_result(path: Path, result: FinalResult) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(f"{result.model_dump_json(indent=2)}\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise AgentRunIOError("unable to write Agent result") from None


def _safe_validation_message(source: str, error: ValidationError) -> str:
    messages: list[str] = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in item["loc"]) or "root"
        messages.append(f"{location}: {item['msg']}")
    return f"invalid {source}: {'; '.join(messages)}"
