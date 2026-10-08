"""OpenAI Responses API adapter with strict one-action parsing and safe retries."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from decimal import Decimal
from typing import Any, Protocol, cast
from uuid import uuid5

import openai
from openai import OpenAI
from pydantic import Field, JsonValue, ValidationError, field_validator

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    ErrorType,
    ModelProvider,
    ProviderErrorCode,
    ReasoningEffort,
)
from repopilot.providers.base import (
    FinishAction,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    ProviderUsageDelta,
    ToolAction,
)
from repopilot.providers.prompts import (
    PROMPT_VERSION,
    SYSTEM_INSTRUCTIONS,
    PromptBuildError,
    build_agent_prompt,
)
from repopilot.schemas import StrictModel, ToolCall
from repopilot.tools.registry import RegisteredTool, ToolDescriptor

FINISH_TOOL_NAME = "finish"
_SAFE_PROVIDER_ERROR_MESSAGE = "OpenAI provider request failed"
_SAFE_FORMAT_ERROR_MESSAGE = "OpenAI response failed structured validation"
_MAX_RETRY_DELAY_SECONDS = 30.0
MonotonicClock = Callable[[], float]
Sleeper = Callable[[float], None]


class OpenAITransportError(RuntimeError):
    """Normalized SDK/HTTP failure without response bodies or credentials."""

    def __init__(
        self,
        code: ProviderErrorCode,
        *,
        retryable: bool,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(_SAFE_PROVIDER_ERROR_MESSAGE)
        self.code = code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class ResponsesTransport(Protocol):
    """Minimal injectable boundary around ``client.responses.create``."""

    def create(self, payload: Mapping[str, JsonValue]) -> Mapping[str, object]:
        """Return one JSON-compatible Responses API object."""


class OpenAIPricing(StrictModel):
    """Versioned per-million-token rates supplied by validated configuration."""

    input_per_million_usd: Decimal = Field(gt=0, le=1_000)
    cached_input_per_million_usd: Decimal = Field(ge=0, le=1_000)
    cache_write_per_million_usd: Decimal = Field(ge=0, le=1_000)
    output_per_million_usd: Decimal = Field(gt=0, le=1_000)


class OpenAIProviderConfig(StrictModel):
    """Non-secret OpenAI request, retry, prompt, and pricing configuration."""

    model: str = Field(pattern=r"^gpt-[A-Za-z0-9][A-Za-z0-9._-]{0,123}$")
    reasoning_effort: ReasoningEffort
    max_output_tokens: int = Field(gt=0, le=128_000)
    max_retries: int = Field(ge=0, le=5)
    retry_backoff_ms: int = Field(ge=0, le=10_000)
    prompt_max_chars: int = Field(ge=4_096, le=200_000)
    pricing: OpenAIPricing

    @field_validator("model")
    @classmethod
    def require_pinned_model(cls, value: str) -> str:
        """Reject mutable aliases so traces retain a reproducible model ID."""

        if value in {"gpt-5.6", "gpt-5-latest"} or value.endswith("-latest"):
            raise ValueError("model must be a pinned model ID")
        return value


class _FinishArguments(StrictModel):
    summary: str = Field(min_length=1, max_length=2_000)


class _Capability:
    def __init__(self, tool: RegisteredTool) -> None:
        try:
            self.name = tool.name
            self.arguments_model = tool.arguments_model
            self.descriptor = ToolDescriptor(
                name=tool.name,
                description=tool.description,
                argument_schema=cast(
                    dict[str, JsonValue],
                    tool.arguments_model.model_json_schema(),
                ),
            )
        except (AttributeError, TypeError, ValidationError):
            raise ProviderError(
                "OpenAI tool metadata is invalid",
                error_type=ErrorType.VALIDATION_ERROR,
                code=ProviderErrorCode.INVALID_CONFIGURATION,
            ) from None


class OpenAISDKTransport:
    """Official Python SDK transport with SDK retries disabled for auditability."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: int,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key,
            timeout=float(timeout_seconds),
            max_retries=0,
        )

    def create(self, payload: Mapping[str, JsonValue]) -> Mapping[str, object]:
        try:
            create = cast(Callable[..., Any], self._client.responses.create)
            response = create(**dict(payload))
            dumped = response.model_dump(mode="json")
            if not isinstance(dumped, dict):
                raise TypeError
            return cast(Mapping[str, object], dumped)
        except openai.APITimeoutError:
            raise OpenAITransportError(
                ProviderErrorCode.TIMEOUT,
                retryable=True,
            ) from None
        except openai.APIConnectionError:
            raise OpenAITransportError(
                ProviderErrorCode.CONNECTION_ERROR,
                retryable=True,
            ) from None
        except openai.AuthenticationError:
            raise OpenAITransportError(
                ProviderErrorCode.AUTHENTICATION_ERROR,
                retryable=False,
            ) from None
        except openai.PermissionDeniedError:
            raise OpenAITransportError(
                ProviderErrorCode.PERMISSION_DENIED,
                retryable=False,
            ) from None
        except openai.RateLimitError as error:
            raise OpenAITransportError(
                ProviderErrorCode.RATE_LIMITED,
                retryable=True,
                retry_after_seconds=_retry_after_seconds(error),
            ) from None
        except openai.BadRequestError:
            raise OpenAITransportError(
                ProviderErrorCode.BAD_REQUEST,
                retryable=False,
            ) from None
        except openai.NotFoundError:
            raise OpenAITransportError(
                ProviderErrorCode.NOT_FOUND,
                retryable=False,
            ) from None
        except openai.ConflictError as error:
            raise OpenAITransportError(
                ProviderErrorCode.CONFLICT,
                retryable=True,
                retry_after_seconds=_retry_after_seconds(error),
            ) from None
        except openai.InternalServerError as error:
            raise OpenAITransportError(
                ProviderErrorCode.SERVER_ERROR,
                retryable=True,
                retry_after_seconds=_retry_after_seconds(error),
            ) from None
        except openai.APIError:
            raise OpenAITransportError(
                ProviderErrorCode.SERVER_ERROR,
                retryable=False,
            ) from None
        except (AttributeError, TypeError, ValueError):
            raise OpenAITransportError(
                ProviderErrorCode.RESPONSE_MALFORMED,
                retryable=False,
            ) from None


class OpenAIProvider:
    """Convert one bounded Agent decision into exactly one structured action."""

    def __init__(
        self,
        *,
        config: OpenAIProviderConfig,
        tools: Iterable[RegisteredTool],
        transport: ResponsesTransport,
        clock: MonotonicClock = time.monotonic,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        capabilities: dict[str, _Capability] = {}
        for tool in tools:
            capability = _Capability(tool)
            if capability.name == FINISH_TOOL_NAME or capability.name in capabilities:
                raise ProviderError(
                    "OpenAI tools must be unique and cannot use the reserved finish name",
                    error_type=ErrorType.VALIDATION_ERROR,
                    code=ProviderErrorCode.INVALID_CONFIGURATION,
                )
            capabilities[capability.name] = capability
        if not capabilities:
            raise ProviderError(
                "OpenAI provider requires at least one Agent tool",
                error_type=ErrorType.VALIDATION_ERROR,
                code=ProviderErrorCode.INVALID_CONFIGURATION,
            )
        self._config = config
        self._capabilities = capabilities
        self._transport = transport
        self._clock = clock
        self._sleeper = sleeper

    @classmethod
    def from_settings(
        cls,
        *,
        settings: RepoPilotSettings,
        tools: Iterable[RegisteredTool],
        transport: ResponsesTransport | None = None,
        clock: MonotonicClock = time.monotonic,
        sleeper: Sleeper = time.sleep,
    ) -> OpenAIProvider:
        """Construct the adapter from redaction-aware validated settings."""

        if settings.model_provider is not ModelProvider.OPENAI or settings.openai_api_key is None:
            raise ProviderError(
                "OpenAI provider settings are incomplete",
                error_type=ErrorType.CONFIG_ERROR,
                code=ProviderErrorCode.INVALID_CONFIGURATION,
            )
        config = OpenAIProviderConfig(
            model=settings.model,
            reasoning_effort=settings.openai_reasoning_effort,
            max_output_tokens=settings.openai_max_output_tokens,
            max_retries=settings.openai_max_retries,
            retry_backoff_ms=settings.openai_retry_backoff_ms,
            prompt_max_chars=settings.openai_prompt_max_chars,
            pricing=OpenAIPricing(
                input_per_million_usd=settings.openai_input_cost_per_million_usd,
                cached_input_per_million_usd=(settings.openai_cached_input_cost_per_million_usd),
                cache_write_per_million_usd=(settings.openai_cache_write_cost_per_million_usd),
                output_per_million_usd=settings.openai_output_cost_per_million_usd,
            ),
        )
        resolved_transport = transport or OpenAISDKTransport(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout_seconds=settings.openai_request_timeout_seconds,
        )
        return cls(
            config=config,
            tools=tools,
            transport=resolved_transport,
            clock=clock,
            sleeper=sleeper,
        )

    @property
    def provider_name(self) -> str:
        return f"openai-responses:{self._config.model}:{PROMPT_VERSION}"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate one strict tool/finish action or raise a typed safe error."""

        unknown = set(request.allowed_tools).difference(self._capabilities)
        if unknown:
            raise ProviderError(
                "Provider request contains an unavailable tool capability",
                error_type=ErrorType.VALIDATION_ERROR,
                code=ProviderErrorCode.INVALID_CONFIGURATION,
            )
        try:
            prompt = build_agent_prompt(
                request,
                max_chars=self._config.prompt_max_chars,
            )
        except PromptBuildError:
            raise ProviderError(
                "Provider prompt exceeds its configured safe limit",
                error_type=ErrorType.MODEL_FORMAT_ERROR,
                code=ProviderErrorCode.PROMPT_TOO_LARGE,
            ) from None

        payload = self._request_payload(request, prompt)
        started = self._clock()
        raw, attempts = self._create_with_retries(payload)
        latency_ms = max(0, round((self._clock() - started) * 1_000))
        return self._parse_response(
            request,
            raw,
            attempts=attempts,
            latency_ms=latency_ms,
        )

    def _request_payload(
        self,
        request: ProviderRequest,
        prompt: str,
    ) -> dict[str, JsonValue]:
        tools = [
            _function_tool(self._capabilities[name].descriptor) for name in request.allowed_tools
        ]
        tools.append(_finish_tool())
        return {
            "model": self._config.model,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": prompt,
            "tools": cast(list[JsonValue], tools),
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "reasoning": {"effort": self._config.reasoning_effort.value},
            "max_output_tokens": self._config.max_output_tokens,
            "store": False,
            "truncation": "disabled",
            "metadata": {
                "repopilot_prompt_version": PROMPT_VERSION,
                "repopilot_task_id": request.state.task.task_id,
            },
        }

    def _create_with_retries(
        self,
        payload: Mapping[str, JsonValue],
    ) -> tuple[Mapping[str, object], int]:
        attempts = 0
        while True:
            attempts += 1
            try:
                return self._transport.create(payload), attempts
            except OpenAITransportError as error:
                if not error.retryable or attempts > self._config.max_retries:
                    raise ProviderError(
                        _SAFE_PROVIDER_ERROR_MESSAGE,
                        error_type=ErrorType.PROVIDER_ERROR,
                        code=error.code,
                    ) from None
                self._sleeper(self._retry_delay(error, attempts))
            except Exception:
                raise ProviderError(
                    _SAFE_PROVIDER_ERROR_MESSAGE,
                    error_type=ErrorType.PROVIDER_ERROR,
                    code=ProviderErrorCode.CONNECTION_ERROR,
                ) from None

    def _retry_delay(
        self,
        error: OpenAITransportError,
        attempts: int,
    ) -> float:
        if error.retry_after_seconds is not None:
            return min(max(error.retry_after_seconds, 0.0), _MAX_RETRY_DELAY_SECONDS)
        base = float(self._config.retry_backoff_ms) / 1_000
        return min(base * (2.0 ** (attempts - 1)), _MAX_RETRY_DELAY_SECONDS)

    def _parse_response(
        self,
        request: ProviderRequest,
        raw: Mapping[str, object],
        *,
        attempts: int,
        latency_ms: int,
    ) -> ProviderResponse:
        if raw.get("status") != "completed":
            raise _format_error(ProviderErrorCode.RESPONSE_INCOMPLETE)
        response_id = _required_text(raw.get("id"))
        response_model = _required_text(raw.get("model"))
        if response_model != self._config.model:
            raise _format_error(ProviderErrorCode.MODEL_MISMATCH)
        output = raw.get("output")
        if not isinstance(output, list):
            raise _format_error(ProviderErrorCode.RESPONSE_MALFORMED)
        calls: list[Mapping[str, object]] = []
        for item in output:
            if not isinstance(item, Mapping):
                raise _format_error(ProviderErrorCode.RESPONSE_MALFORMED)
            item_type = item.get("type")
            if item_type == "function_call":
                calls.append(cast(Mapping[str, object], item))
            elif item_type != "reasoning":
                raise _format_error(ProviderErrorCode.RESPONSE_MALFORMED)
        if len(calls) != 1:
            raise _format_error(ProviderErrorCode.TOOL_COUNT_INVALID)

        action = self._parse_action(request, response_id, calls[0])
        usage = _parse_usage(
            raw.get("usage"),
            pricing=self._config.pricing,
            latency_ms=latency_ms,
        )
        return ProviderResponse(
            request_id=request.request_id,
            action=action,
            usage=usage,
            provider_response_id=response_id,
            model=response_model,
            prompt_version=PROMPT_VERSION,
            attempts=attempts,
        )

    def _parse_action(
        self,
        request: ProviderRequest,
        response_id: str,
        call: Mapping[str, object],
    ) -> ToolAction | FinishAction:
        if call.get("status") not in {None, "completed"}:
            raise _format_error(ProviderErrorCode.RESPONSE_INCOMPLETE)
        provider_call_id = _required_text(call.get("call_id"))
        name = _required_text(call.get("name"))
        arguments = _parse_arguments(call.get("arguments"))
        if name == FINISH_TOOL_NAME:
            try:
                finish = _FinishArguments.model_validate(arguments)
            except ValidationError:
                raise _format_error(ProviderErrorCode.TOOL_ARGUMENTS_INVALID) from None
            return FinishAction(summary=finish.summary)
        if name not in request.allowed_tools or name not in self._capabilities:
            raise ProviderError(
                _SAFE_FORMAT_ERROR_MESSAGE,
                error_type=ErrorType.UNKNOWN_TOOL,
                code=ProviderErrorCode.TOOL_NOT_ALLOWED,
            )
        capability = self._capabilities[name]
        declared_properties = capability.descriptor.argument_schema.get("properties")
        if not isinstance(declared_properties, dict) or set(arguments) != set(declared_properties):
            raise _format_error(ProviderErrorCode.TOOL_ARGUMENTS_INVALID)
        try:
            encoded_arguments = json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            validated = capability.arguments_model.model_validate_json(
                encoded_arguments,
                strict=True,
            )
            serialized = validated.model_dump(mode="json")
        except (TypeError, ValidationError, ValueError):
            raise _format_error(ProviderErrorCode.TOOL_ARGUMENTS_INVALID) from None
        if not isinstance(serialized, dict):
            raise _format_error(ProviderErrorCode.TOOL_ARGUMENTS_INVALID)
        return ToolAction(
            tool_call=ToolCall(
                call_id=uuid5(
                    request.run_id,
                    f"openai:{response_id}:{provider_call_id}",
                ),
                tool_name=name,
                validated_arguments=cast(dict[str, JsonValue], serialized),
                requested_at=request.requested_at,
            ),
            considered_tools=request.allowed_tools,
            selection_reason=f"OpenAI selected {name} for the active Agent step.",
        )


def _function_tool(descriptor: ToolDescriptor) -> dict[str, JsonValue]:
    return {
        "type": "function",
        "name": descriptor.name,
        "description": descriptor.description,
        "strict": True,
        "parameters": _strict_schema(descriptor.argument_schema),
    }


def _finish_tool() -> dict[str, JsonValue]:
    return {
        "type": "function",
        "name": FINISH_TOOL_NAME,
        "description": (
            "Request deterministic verification after the active finish step. "
            "This does not claim repair success."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief evidence summary for deterministic verification.",
                }
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    }


def _strict_schema(schema: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    cleaned = _clean_schema_value(dict(schema))
    if not isinstance(cleaned, dict) or cleaned.get("type") != "object":
        raise ProviderError(
            "Agent tool schema must have an object root",
            error_type=ErrorType.VALIDATION_ERROR,
            code=ProviderErrorCode.INVALID_CONFIGURATION,
        )
    return cleaned


def _clean_schema_value(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return [_clean_schema_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    cleaned: dict[str, JsonValue] = {}
    for key, item in value.items():
        if key in {"default", "examples", "title"}:
            continue
        cleaned[key] = _clean_schema_value(item)
    properties = cleaned.get("properties")
    if cleaned.get("type") == "object" and isinstance(properties, dict):
        cleaned["additionalProperties"] = False
        cleaned["required"] = list(properties)
    return cleaned


def _parse_arguments(value: object) -> dict[str, object]:
    if not isinstance(value, str) or len(value) > 100_000:
        raise _format_error(ProviderErrorCode.TOOL_ARGUMENTS_INVALID)
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_unique_object,
            parse_constant=lambda _: _reject_json_constant(),
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise _format_error(ProviderErrorCode.TOOL_ARGUMENTS_INVALID) from None
    if not isinstance(parsed, dict):
        raise _format_error(ProviderErrorCode.TOOL_ARGUMENTS_INVALID)
    return cast(dict[str, object], parsed)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant() -> None:
    raise ValueError("non-finite JSON constant")


def _parse_usage(
    value: object,
    *,
    pricing: OpenAIPricing,
    latency_ms: int,
) -> ProviderUsageDelta:
    if not isinstance(value, Mapping):
        raise _format_error(ProviderErrorCode.USAGE_INVALID)
    input_tokens = _nonnegative_int(value.get("input_tokens"))
    output_tokens = _nonnegative_int(value.get("output_tokens"))
    total_tokens = _nonnegative_int(value.get("total_tokens"))
    input_details = value.get("input_tokens_details")
    output_details = value.get("output_tokens_details")
    cached = 0
    cache_write = 0
    reasoning = 0
    if input_details is not None:
        if not isinstance(input_details, Mapping):
            raise _format_error(ProviderErrorCode.USAGE_INVALID)
        cached = _optional_nonnegative_int(input_details.get("cached_tokens"))
        cache_write = _optional_nonnegative_int(input_details.get("cache_write_tokens"))
    if output_details is not None:
        if not isinstance(output_details, Mapping):
            raise _format_error(ProviderErrorCode.USAGE_INVALID)
        reasoning = _optional_nonnegative_int(output_details.get("reasoning_tokens"))
    if (
        total_tokens != input_tokens + output_tokens
        or cached + cache_write > input_tokens
        or reasoning > output_tokens
    ):
        raise _format_error(ProviderErrorCode.USAGE_INVALID)
    regular_input = input_tokens - cached - cache_write
    million = Decimal(1_000_000)
    estimated_cost = (
        Decimal(regular_input) * pricing.input_per_million_usd
        + Decimal(cached) * pricing.cached_input_per_million_usd
        + Decimal(cache_write) * pricing.cache_write_per_million_usd
        + Decimal(output_tokens) * pricing.output_per_million_usd
    ) / million
    return ProviderUsageDelta(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached,
        cache_write_input_tokens=cache_write,
        reasoning_output_tokens=reasoning,
        estimated_cost_usd=estimated_cost,
        latency_ms=latency_ms,
    )


def _required_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise _format_error(ProviderErrorCode.RESPONSE_MALFORMED)
    return value


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _format_error(ProviderErrorCode.USAGE_INVALID)
    return value


def _optional_nonnegative_int(value: object) -> int:
    if value is None:
        return 0
    return _nonnegative_int(value)


def _format_error(code: ProviderErrorCode) -> ProviderError:
    return ProviderError(
        _SAFE_FORMAT_ERROR_MESSAGE,
        error_type=ErrorType.MODEL_FORMAT_ERROR,
        code=code,
    )


def _retry_after_seconds(error: openai.APIStatusError) -> float | None:
    value = error.response.headers.get("retry-after")
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


__all__ = [
    "FINISH_TOOL_NAME",
    "OpenAIPricing",
    "OpenAIProvider",
    "OpenAIProviderConfig",
    "OpenAISDKTransport",
    "OpenAITransportError",
    "ResponsesTransport",
]
