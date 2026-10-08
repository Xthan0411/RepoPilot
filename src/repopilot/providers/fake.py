"""Deterministic scripted model provider for tests and local smoke flows."""

from collections.abc import Iterable

from repopilot.enums import ErrorType, ProviderErrorCode
from repopilot.providers.base import (
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    ToolAction,
)


class FakeProviderConfigurationError(ProviderError):
    """The injected Fake script is unusable."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            error_type=ErrorType.MODEL_FORMAT_ERROR,
            code=ProviderErrorCode.INVALID_CONFIGURATION,
        )


class ProviderScriptExhaustedError(ProviderError):
    """No scripted response remains for the next request."""

    def __init__(self) -> None:
        super().__init__(
            "fake provider response script is exhausted",
            error_type=ErrorType.PROVIDER_ERROR,
            code=ProviderErrorCode.SCRIPT_EXHAUSTED,
        )


class FakeProvider:
    """Return prevalidated responses in a deterministic order."""

    def __init__(
        self,
        responses: Iterable[ProviderResponse],
        *,
        provider_name: str = "fake-deterministic",
    ) -> None:
        self._responses = tuple(responses)
        if not self._responses:
            raise FakeProviderConfigurationError(
                "fake provider requires at least one scripted response"
            )
        normalized_name = provider_name.strip()
        if not normalized_name:
            raise FakeProviderConfigurationError("fake provider name must not be blank")
        self._provider_name = normalized_name
        self._next_response = 0
        self._calls_made = 0
        self._requests: list[ProviderRequest] = []

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def calls_made(self) -> int:
        return self._calls_made

    @property
    def remaining_responses(self) -> int:
        return len(self._responses) - self._next_response

    @property
    def requests(self) -> tuple[ProviderRequest, ...]:
        """Return the exact validated requests received in call order."""

        return tuple(self._requests)

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self._calls_made += 1
        self._requests.append(request)
        if self._next_response >= len(self._responses):
            raise ProviderScriptExhaustedError()

        response = self._responses[self._next_response]
        self._validate_response(request, response)
        self._next_response += 1
        return response

    @staticmethod
    def _validate_response(
        request: ProviderRequest,
        response: ProviderResponse,
    ) -> None:
        if response.request_id != request.request_id:
            raise FakeProviderConfigurationError(
                "scripted response request_id does not match the provider request"
            )
        if isinstance(response.action, ToolAction):
            if response.action.tool_call.tool_name not in request.allowed_tools:
                raise ProviderError(
                    "scripted response selected a tool outside allowed_tools",
                    error_type=ErrorType.UNKNOWN_TOOL,
                    code=ProviderErrorCode.TOOL_NOT_ALLOWED,
                )
            if not set(response.action.considered_tools).issubset(request.allowed_tools):
                raise FakeProviderConfigurationError(
                    "scripted considered_tools exceed the request capability boundary"
                )
