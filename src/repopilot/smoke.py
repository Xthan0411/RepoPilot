"""Deterministic P0 smoke workflow with no Agent loop or external execution."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal, Self, cast
from uuid import UUID

from pydantic import (
    AwareDatetime,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    ModelProvider,
    ReproductionStatus,
    RunStatus,
    SandboxNetworkMode,
    TerminationReason,
    ValidationStatus,
)
from repopilot.providers.base import ProviderRequest, ProviderResponse
from repopilot.providers.fake import FakeProvider
from repopilot.providers.prompts import build_context_aware_provider_request
from repopilot.schemas import (
    AgentState,
    BudgetState,
    FinalResult,
    Identifier,
    LocalizationMetrics,
    NonEmptyText,
    RelativePath,
    ReproductionResult,
    StrictModel,
    TaskMetrics,
    TaskSpec,
    UsageMetrics,
    ValidationResult,
)


class SmokeInputError(ValueError):
    """A safely reportable manifest or smoke-configuration error."""


class SmokeIOError(RuntimeError):
    """A safely reportable input/output failure."""


class SmokeManifest(StrictModel):
    """Self-contained deterministic input for the P0 CLI vertical slice."""

    schema_version: Literal[1] = 1
    run_id: UUID
    request_id: UUID
    requested_at: AwareDatetime
    task: TaskSpec
    allowed_tools: tuple[Identifier, ...] = Field(min_length=1)
    provider_response: ProviderResponse
    repository_summary: str = ""
    result_filename: RelativePath = "result.json"
    trace_filename: RelativePath = "trace.json"

    @field_validator("allowed_tools")
    @classmethod
    def validate_unique_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("allowed_tools must be unique")
        return value

    @model_validator(mode="after")
    def validate_correlations(self) -> Self:
        if self.provider_response.request_id != self.request_id:
            raise ValueError("provider response request_id must match the manifest request_id")
        if self.result_filename == self.trace_filename:
            raise ValueError("result_filename and trace_filename must be different")
        return self


class SmokeTrace(StrictModel):
    """Auditable trace for the intentionally shallow P0 smoke run."""

    schema_version: Literal[1] = 1
    run_mode: Literal["p0_smoke"] = "p0_smoke"
    generated_at: AwareDatetime
    provider_name: NonEmptyText
    settings: dict[str, JsonValue]
    request: ProviderRequest
    response: ProviderResponse
    result_status: RunStatus


@dataclass(frozen=True)
class SmokeArtifacts:
    """In-memory result plus the two files written by a smoke run."""

    result: FinalResult
    trace: SmokeTrace
    result_path: Path
    trace_path: Path


def load_smoke_manifest(path: Path) -> SmokeManifest:
    """Read and validate a manifest without echoing rejected input values."""

    try:
        serialized = path.read_text(encoding="utf-8")
    except OSError:
        raise SmokeIOError("unable to read smoke manifest") from None
    try:
        return SmokeManifest.model_validate_json(serialized)
    except ValidationError as error:
        raise SmokeInputError(_safe_validation_message("smoke manifest", error)) from None


def run_smoke(
    manifest_path: Path,
    output_dir: Path,
    settings: RepoPilotSettings,
) -> SmokeArtifacts:
    """Load one manifest and run the deterministic P0 vertical slice."""

    manifest = load_smoke_manifest(manifest_path)
    return run_smoke_manifest(manifest, output_dir, settings)


def run_smoke_manifest(
    manifest: SmokeManifest,
    output_dir: Path,
    settings: RepoPilotSettings,
) -> SmokeArtifacts:
    """Produce a trace and honest stopped result from one Fake provider action."""

    _validate_smoke_settings(settings)
    state = _build_initial_state(manifest, settings)
    request = build_context_aware_provider_request(
        request_id=manifest.request_id,
        run_id=manifest.run_id,
        state=state,
        allowed_tools=manifest.allowed_tools,
        requested_at=manifest.requested_at,
        max_chars=settings.openai_prompt_max_chars,
    )
    provider = FakeProvider(
        (manifest.provider_response,),
        provider_name=settings.model,
    )
    response = provider.generate(request)
    trace = SmokeTrace(
        generated_at=manifest.requested_at,
        provider_name=provider.provider_name,
        settings=cast(dict[str, JsonValue], settings.redacted_snapshot()),
        request=request,
        response=response,
        result_status=RunStatus.STOPPED,
    )
    result = _build_stopped_result(manifest, response)
    result_path, trace_path = _prepare_output_paths(manifest, output_dir)
    _write_json(trace_path, trace.model_dump_json(indent=2))
    _write_json(result_path, result.model_dump_json(indent=2))
    return SmokeArtifacts(
        result=result,
        trace=trace,
        result_path=result_path,
        trace_path=trace_path,
    )


def _validate_smoke_settings(settings: RepoPilotSettings) -> None:
    if settings.model_provider is not ModelProvider.FAKE:
        raise SmokeInputError("P0 smoke requires model_provider=fake")
    if settings.model != "fake-deterministic":
        raise SmokeInputError("P0 smoke requires model=fake-deterministic")
    if settings.sandbox_network is not SandboxNetworkMode.NONE:
        raise SmokeInputError("P0 smoke requires sandbox_network=none")


def _build_initial_state(
    manifest: SmokeManifest,
    settings: RepoPilotSettings,
) -> AgentState:
    return AgentState(
        task=manifest.task,
        repository_summary=manifest.repository_summary,
        budget=BudgetState(
            max_steps=settings.max_steps,
            max_tokens=settings.max_tokens,
            max_cost_usd=settings.max_cost_usd,
            started_at=manifest.requested_at,
            deadline_at=manifest.requested_at + timedelta(seconds=settings.timeout_seconds),
        ),
    )


def _build_stopped_result(
    manifest: SmokeManifest,
    response: ProviderResponse,
) -> FinalResult:
    usage = response.usage
    return FinalResult(
        task_id=manifest.task.task_id,
        run_id=manifest.run_id,
        status=RunStatus.STOPPED,
        termination_reason=TerminationReason.AGENT_STOPPED,
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
            tests_passed=False,
            task_success=False,
            tool_steps=0,
        ),
        usage=UsageMetrics(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=usage.estimated_cost_usd,
            latency_ms=usage.latency_ms,
        ),
        trace_path=manifest.trace_filename,
        error_category=None,
    )


def _prepare_output_paths(
    manifest: SmokeManifest,
    output_dir: Path,
) -> tuple[Path, Path]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        root = output_dir.resolve(strict=True)
        result_path = _contained_path(root, manifest.result_filename)
        trace_path = _contained_path(root, manifest.trace_filename)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise SmokeIOError("unable to prepare smoke output directory") from None
    return result_path, trace_path


def _contained_path(root: Path, relative_path: str) -> Path:
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root):
        raise SmokeIOError("smoke output path escapes the output directory")
    return target


def _write_json(path: Path, serialized: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(f"{serialized}\n", encoding="utf-8")
        temporary_path.replace(path)
    except OSError:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise SmokeIOError("unable to write smoke output") from None


def _safe_validation_message(source: str, error: ValidationError) -> str:
    messages: list[str] = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in item["loc"]) or "root"
        messages.append(f"{location}: {item['msg']}")
    return f"invalid {source}: {'; '.join(messages)}"
