"""Narrow contracts for model-provider adapters."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal, Protocol, Self, runtime_checkable
from uuid import UUID

from pydantic import AwareDatetime, Field, field_validator, model_validator

from repopilot.context import ContextWindowSnapshot
from repopilot.enums import ErrorType, ProviderActionKind, ProviderErrorCode
from repopilot.schemas import (
    AgentState,
    Identifier,
    ModelIdentifier,
    NonEmptyText,
    StrictModel,
    ToolCall,
    VersionIdentifier,
)


class ProviderError(RuntimeError):
    """Safe provider failure carrying the shared runtime error taxonomy."""

    def __init__(
        self,
        message: str,
        *,
        error_type: ErrorType,
        code: ProviderErrorCode,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.code = code


class ProviderRequest(StrictModel):
    """One model decision request with an explicit tool capability boundary."""

    request_id: UUID
    run_id: UUID
    state: AgentState
    allowed_tools: tuple[Identifier, ...] = ()
    requested_at: AwareDatetime
    context_window: ContextWindowSnapshot | None = None

    @field_validator("allowed_tools")
    @classmethod
    def validate_unique_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("allowed_tools must be unique")
        return value

    @model_validator(mode="after")
    def validate_context_namespace(self) -> Self:
        window = self.context_window
        if window is not None and (
            window.request_id != self.request_id
            or window.run_id != self.run_id
            or window.task_id != self.state.task.task_id
            or window.repository_commit != self.state.task.base_commit
        ):
            raise ValueError("context window must match Provider request namespace")
        return self


class ProviderUsageDelta(StrictModel):
    """Usage attributable to exactly one provider decision."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    cache_write_input_tokens: int = Field(default=0, ge=0)
    reasoning_output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: Decimal = Field(ge=0)
    latency_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total_tokens(self) -> Self:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens plus output_tokens")
        if self.cached_input_tokens + self.cache_write_input_tokens > self.input_tokens:
            raise ValueError("input token detail counts cannot exceed input_tokens")
        if self.reasoning_output_tokens > self.output_tokens:
            raise ValueError("reasoning_output_tokens cannot exceed output_tokens")
        return self


class ToolAction(StrictModel):
    """A provider request to invoke exactly one validated tool."""

    kind: Literal[ProviderActionKind.TOOL_CALL] = ProviderActionKind.TOOL_CALL
    tool_call: ToolCall
    considered_tools: tuple[Identifier, ...] = Field(min_length=1)
    selection_reason: NonEmptyText

    @field_validator("considered_tools")
    @classmethod
    def validate_unique_considered_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("considered_tools must be unique")
        return value

    @model_validator(mode="after")
    def validate_selected_tool(self) -> Self:
        if self.tool_call.tool_name not in self.considered_tools:
            raise ValueError("selected tool must appear in considered_tools")
        return self


class FinishAction(StrictModel):
    """A request to enter verification, not an authoritative success claim."""

    kind: Literal[ProviderActionKind.FINISH] = ProviderActionKind.FINISH
    summary: NonEmptyText


ProviderAction = Annotated[ToolAction | FinishAction, Field(discriminator="kind")]


class ProviderResponse(StrictModel):
    """One structured provider action correlated to its request."""

    request_id: UUID
    action: ProviderAction
    usage: ProviderUsageDelta
    provider_response_id: str | None = Field(default=None, min_length=1, max_length=128)
    model: ModelIdentifier | None = None
    prompt_version: VersionIdentifier | None = None
    attempts: int = Field(default=1, ge=1, le=6)


@runtime_checkable
class ModelProviderAdapter(Protocol):
    """Interface consumed by the future Agent decision node."""

    @property
    def provider_name(self) -> str:
        """Return a stable provider implementation identifier."""

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Return one structured action or raise a typed provider error."""
