"""Contract tests for the Provider Protocol and deterministic FakeProvider."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from repopilot.enums import ErrorType, ProviderActionKind
from repopilot.providers.base import (
    FinishAction,
    ModelProviderAdapter,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    ToolAction,
)
from repopilot.providers.fake import (
    FakeProvider,
    FakeProviderConfigurationError,
    ProviderScriptExhaustedError,
)
from repopilot.schemas import AgentState

FIXTURES = Path(__file__).parents[1] / "fixtures"
REQUEST_ID = UUID("44444444-4444-4444-8444-444444444444")
OTHER_REQUEST_ID = UUID("77777777-7777-4777-8777-777777777777")
REQUESTED_AT = datetime(2026, 7, 27, 8, 1, tzinfo=UTC)


def load_fixture(name: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((FIXTURES / name).read_text(encoding="utf-8")),
    )


def make_request(
    *,
    request_id: UUID = REQUEST_ID,
    allowed_tools: tuple[str, ...] = ("search_code", "read_file"),
) -> ProviderRequest:
    return ProviderRequest(
        request_id=request_id,
        run_id=UUID("22222222-2222-4222-8222-222222222222"),
        state=AgentState.model_validate(load_fixture("valid_agent_state.json")),
        allowed_tools=allowed_tools,
        requested_at=REQUESTED_AT,
    )


def make_response() -> ProviderResponse:
    return ProviderResponse.model_validate(load_fixture("valid_provider_response.json"))


def test_provider_request_and_response_round_trip() -> None:
    request = make_request()
    response = make_response()

    assert ProviderRequest.model_validate_json(request.model_dump_json()) == request
    assert ProviderResponse.model_validate_json(response.model_dump_json()) == response
    assert isinstance(response.action, ToolAction)
    assert response.action.kind is ProviderActionKind.TOOL_CALL


def test_fake_provider_implements_protocol_and_returns_script_in_order() -> None:
    response = make_response()
    provider = FakeProvider((response,))
    adapter: ModelProviderAdapter = provider

    assert isinstance(adapter, ModelProviderAdapter)
    assert adapter.provider_name == "fake-deterministic"
    assert adapter.generate(make_request()) == response
    assert provider.calls_made == 1
    assert provider.remaining_responses == 0

    with pytest.raises(ProviderScriptExhaustedError) as captured:
        provider.generate(make_request(request_id=OTHER_REQUEST_ID))

    assert captured.value.error_type is ErrorType.PROVIDER_ERROR
    assert provider.calls_made == 2


def test_fake_provider_requires_a_nonempty_script_and_name() -> None:
    with pytest.raises(FakeProviderConfigurationError) as empty_script:
        FakeProvider(())
    assert empty_script.value.error_type is ErrorType.MODEL_FORMAT_ERROR

    with pytest.raises(FakeProviderConfigurationError, match="name"):
        FakeProvider((make_response(),), provider_name=" ")


def test_provider_request_rejects_duplicate_allowed_tools() -> None:
    with pytest.raises(ValidationError, match="allowed_tools must be unique"):
        make_request(allowed_tools=("read_file", "read_file"))


def test_provider_request_allows_finish_only_capability_boundary() -> None:
    request = make_request(allowed_tools=())

    assert request.allowed_tools == ()


def test_tool_action_requires_selected_tool_in_considered_set() -> None:
    payload = load_fixture("valid_provider_response.json")["action"]
    payload["considered_tools"] = ["search_code"]

    with pytest.raises(ValidationError, match="selected tool"):
        ToolAction.model_validate(payload)


def test_finish_action_cannot_carry_an_authoritative_success_claim() -> None:
    action = FinishAction(summary="Evidence collection is complete.")
    assert action.kind is ProviderActionKind.FINISH

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FinishAction.model_validate(
            {
                "kind": "finish",
                "summary": "Done.",
                "success": True,
            }
        )


def test_provider_response_requires_a_known_discriminated_action() -> None:
    payload = load_fixture("valid_provider_response.json")
    payload["action"]["kind"] = "unknown"

    with pytest.raises(ValidationError):
        ProviderResponse.model_validate(payload)


def test_fake_provider_rejects_mismatched_request_correlation() -> None:
    provider = FakeProvider((make_response(),))
    with pytest.raises(FakeProviderConfigurationError, match="request_id") as captured:
        provider.generate(make_request(request_id=OTHER_REQUEST_ID))

    assert captured.value.error_type is ErrorType.MODEL_FORMAT_ERROR
    assert provider.remaining_responses == 1


def test_fake_provider_enforces_allowed_tool_boundary() -> None:
    with pytest.raises(ProviderError, match="outside allowed_tools") as captured:
        FakeProvider((make_response(),)).generate(make_request(allowed_tools=("search_code",)))

    assert captured.value.error_type is ErrorType.UNKNOWN_TOOL


def test_fake_provider_rejects_considered_tools_outside_capability_boundary() -> None:
    with pytest.raises(FakeProviderConfigurationError, match="capability") as captured:
        FakeProvider((make_response(),)).generate(make_request(allowed_tools=("read_file",)))

    assert captured.value.error_type is ErrorType.MODEL_FORMAT_ERROR
