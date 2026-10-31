"""Typed, deterministic dispatch for the capabilities exposed to an Agent."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Protocol, cast

from pydantic import BaseModel, Field, JsonValue, ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    ErrorType,
    RepositoryErrorCode,
    ToolErrorCode,
    ToolResultStatus,
)
from repopilot.sandbox.base import SandboxError
from repopilot.schemas import (
    Identifier,
    StrictModel,
    ToolCall,
    ToolExecutionResult,
)
from repopilot.tools.repository import RepositoryAccessError

ToolOutputProjector = Callable[[ToolCall, dict[str, JsonValue], bytes], dict[str, JsonValue]]


class ToolOutputProjectionError(ValueError):
    """A safe source preview cannot represent even one complete source line."""


TOOL_ERROR_MESSAGES: Mapping[ToolErrorCode, str] = {
    ToolErrorCode.UNKNOWN_TOOL: "requested tool is not registered",
    ToolErrorCode.INVALID_ARGUMENTS: "tool arguments failed validation",
    ToolErrorCode.DUPLICATE_CALL: "tool call ID has already been used",
    ToolErrorCode.OUTPUT_TOO_LARGE: "tool output exceeds the configured byte limit",
    ToolErrorCode.REPOSITORY_ERROR: "repository operation failed",
    ToolErrorCode.TOOL_EXECUTION_ERROR: "tool execution failed",
}


class RegisteredTool(Protocol):
    """Minimal capability contract accepted by :class:`ToolRegistry`."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def arguments_model(self) -> type[BaseModel]: ...

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        """Execute one already-validated argument object."""


class ToolRegistrationError(ValueError):
    """Safe startup failure caused by an invalid or duplicate registration."""


class ToolArgumentError(ValueError):
    """Safe tool-specific argument failure not expressible in a static schema."""


class ToolDescriptor(StrictModel):
    """Agent-visible name, purpose, and JSON argument schema for one tool."""

    name: Identifier
    description: str = Field(min_length=1, max_length=1_000)
    argument_schema: dict[str, JsonValue]


class ToolRegistry:
    """Register unique capabilities and invoke each call ID at most once."""

    def __init__(
        self,
        *,
        max_output_bytes: int = 65_536,
        output_projector: ToolOutputProjector | None = None,
    ) -> None:
        if (
            isinstance(max_output_bytes, bool)
            or not isinstance(max_output_bytes, int)
            or not 0 < max_output_bytes <= 1_000_000
        ):
            raise ToolRegistrationError("max_output_bytes must be between 1 and 1000000")
        self._max_output_bytes = max_output_bytes
        self._output_projector = output_projector
        self._tools: dict[str, RegisteredTool] = {}
        self._seen_call_ids: set[str] = set()

    @classmethod
    def from_settings(
        cls,
        settings: RepoPilotSettings,
        *,
        output_projector: ToolOutputProjector | None = None,
    ) -> ToolRegistry:
        """Construct a Registry using the validated global output limit."""

        return cls(
            max_output_bytes=settings.tool_max_output_bytes, output_projector=output_projector
        )

    def register(self, tool: RegisteredTool) -> None:
        """Register one valid, uniquely named tool."""

        try:
            ToolDescriptor(
                name=tool.name,
                description=tool.description,
                argument_schema=cast(
                    dict[str, JsonValue],
                    tool.arguments_model.model_json_schema(),
                ),
            )
        except (AttributeError, TypeError, ValidationError):
            raise ToolRegistrationError("tool registration metadata is invalid") from None
        if tool.name in self._tools:
            raise ToolRegistrationError("tool name is already registered")
        self._tools[tool.name] = tool

    def list_tools(self) -> tuple[ToolDescriptor, ...]:
        """Return deterministic Agent-visible descriptors sorted by tool name."""

        return tuple(
            ToolDescriptor(
                name=tool.name,
                description=tool.description,
                argument_schema=cast(
                    dict[str, JsonValue],
                    tool.arguments_model.model_json_schema(),
                ),
            )
            for tool in sorted(self._tools.values(), key=lambda item: item.name)
        )

    def invoke(self, call: ToolCall) -> ToolExecutionResult:
        """Claim and execute a call, converting every failure to a safe envelope."""

        call_id = str(call.call_id)
        if call_id in self._seen_call_ids:
            return self._error(
                call,
                error_type=ErrorType.DUPLICATE_TOOL_CALL,
                error_code=ToolErrorCode.DUPLICATE_CALL,
            )
        self._seen_call_ids.add(call_id)

        tool = self._tools.get(call.tool_name)
        if tool is None:
            return self._error(
                call,
                error_type=ErrorType.UNKNOWN_TOOL,
                error_code=ToolErrorCode.UNKNOWN_TOOL,
            )
        try:
            arguments = tool.arguments_model.model_validate(call.validated_arguments)
        except ValidationError:
            return self._error(
                call,
                error_type=ErrorType.INVALID_TOOL_ARGUMENTS,
                error_code=ToolErrorCode.INVALID_ARGUMENTS,
            )

        try:
            output = tool.execute(arguments, call=call)
        except ToolArgumentError:
            return self._error(
                call,
                error_type=ErrorType.INVALID_TOOL_ARGUMENTS,
                error_code=ToolErrorCode.INVALID_ARGUMENTS,
            )
        except RepositoryAccessError as error:
            return self._error(
                call,
                error_type=error.error_type,
                error_code=ToolErrorCode.REPOSITORY_ERROR,
                repository_error_code=error.code,
            )
        except SandboxError as error:
            return self._error(
                call,
                error_type=error.error_type,
                error_code=ToolErrorCode.TOOL_EXECUTION_ERROR,
            )
        except Exception:
            return self._error(
                call,
                error_type=ErrorType.TOOL_EXECUTION_ERROR,
                error_code=ToolErrorCode.TOOL_EXECUTION_ERROR,
            )

        try:
            raw_payload = output.model_dump(mode="json")
            if not isinstance(raw_payload, dict):
                raise TypeError("tool result must serialize to an object")
            payload = cast(dict[str, JsonValue], raw_payload)
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except Exception:
            return self._error(
                call,
                error_type=ErrorType.TOOL_EXECUTION_ERROR,
                error_code=ToolErrorCode.TOOL_EXECUTION_ERROR,
            )
        if len(encoded) > self._max_output_bytes:
            return self._error(
                call,
                error_type=ErrorType.TOOL_EXECUTION_ERROR,
                error_code=ToolErrorCode.OUTPUT_TOO_LARGE,
                truncated=True,
            )
        if self._output_projector is not None:
            try:
                payload = self._output_projector(call, payload, encoded)
                encoded = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                if len(encoded) > self._max_output_bytes:
                    raise ToolOutputProjectionError
            except ToolOutputProjectionError:
                return self._error(
                    call,
                    error_type=ErrorType.TOOL_EXECUTION_ERROR,
                    error_code=ToolErrorCode.OUTPUT_TOO_LARGE,
                    truncated=True,
                )
            except Exception:
                return self._error(
                    call,
                    error_type=ErrorType.TOOL_EXECUTION_ERROR,
                    error_code=ToolErrorCode.TOOL_EXECUTION_ERROR,
                )
        return ToolExecutionResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolResultStatus.SUCCESS,
            payload=payload,
            output_bytes=len(encoded),
            truncated=payload.get("truncated") is True,
        )

    @staticmethod
    def _error(
        call: ToolCall,
        *,
        error_type: ErrorType,
        error_code: ToolErrorCode,
        repository_error_code: RepositoryErrorCode | None = None,
        truncated: bool = False,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolResultStatus.ERROR,
            error_type=error_type,
            error_code=error_code,
            repository_error_code=repository_error_code,
            message=TOOL_ERROR_MESSAGES[error_code],
            truncated=truncated,
        )
