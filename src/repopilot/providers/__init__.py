"""Replaceable model-provider adapters."""

from repopilot.providers.base import (
    FinishAction,
    ModelProviderAdapter,
    ProviderAction,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    ProviderUsageDelta,
    ToolAction,
)
from repopilot.providers.fake import FakeProvider
from repopilot.providers.openai import (
    OpenAIPricing,
    OpenAIProvider,
    OpenAIProviderConfig,
    OpenAISDKTransport,
    OpenAITransportError,
    ResponsesTransport,
)
from repopilot.providers.prompts import (
    PROMPT_VERSION,
    SYSTEM_INSTRUCTIONS,
    AgentPromptComposition,
    PromptBuildError,
    build_agent_prompt,
    build_context_aware_provider_request,
    compose_agent_prompt,
)

__all__ = [
    "PROMPT_VERSION",
    "SYSTEM_INSTRUCTIONS",
    "AgentPromptComposition",
    "FakeProvider",
    "FinishAction",
    "ModelProviderAdapter",
    "OpenAIPricing",
    "OpenAIProvider",
    "OpenAIProviderConfig",
    "OpenAISDKTransport",
    "OpenAITransportError",
    "PromptBuildError",
    "ProviderAction",
    "ProviderError",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderUsageDelta",
    "ResponsesTransport",
    "ToolAction",
    "build_agent_prompt",
    "build_context_aware_provider_request",
    "compose_agent_prompt",
]
