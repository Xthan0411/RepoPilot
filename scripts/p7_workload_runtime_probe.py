"""Exercise pre-decision production boundaries without constructing/running an Agent."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from pydantic import JsonValue, SecretStr

from repopilot.agent.repair import (
    STEP_TOOL_CAPABILITIES,
    initialize_repair_plan,
    repair_decision_request_id,
)
from repopilot.agent.state import from_graph_state, to_graph_state
from repopilot.config import RepoPilotSettings
from repopilot.enums import RepairWorkflowErrorCode, SandboxNetworkMode
from repopilot.evals.isolated_tests import build_isolated_test_sandbox_factory
from repopilot.providers.openai import OpenAIProvider, OpenAISDKTransport
from repopilot.providers.prompts import build_agent_prompt, build_context_aware_provider_request
from repopilot.repair_workflow import (
    FailureStage,
    RepairRetrievalContext,
    RepairRunRequest,
    RepairWorkflowError,
    _build_repair_registry,
    _run_setup_commands,
    safe_repair_failure_detail,
)
from repopilot.schemas import AgentState, BudgetState, TaskSpec
from repopilot.tools.repository import ReadOnlyRepository
from scripts.p7_holdout_control import HoldoutEnvironmentBinding

_DUMMY_KEY = "workload-preflight-no-provider-request"


class _BlockedSDKTransport(OpenAISDKTransport):
    """Initialize the installed SDK normally, but never delegate a request to it."""

    def create(self, payload: Mapping[str, JsonValue]) -> Mapping[str, object]:
        raise RuntimeError("provider_requests_forbidden")

    def close(self) -> None:
        self._client.close()


def probe_runtime(
    *,
    root: Path,
    task: TaskSpec,
    settings: RepoPilotSettings,
    binding: HoldoutEnvironmentBinding,
    regression_patch: str,
    retrieval: RepairRetrievalContext,
) -> dict[str, JsonValue]:
    """Run offline setup and serialize the first request, never send it.

    The caller validates the frozen task/image/regression projection. Only the
    isolated evaluator sees regression tests. Gold patches are not an input.
    Temporary sandboxes and files are destroyed even if construction fails.
    """

    started = time.monotonic()
    stage: FailureStage = "input"
    try:
        if settings.sandbox_network is not SandboxNetworkMode.NONE:
            raise ValueError("offline_setup_required")
        if task.task_id != binding.task_id or task.base_commit != binding.base_commit:
            raise ValueError("task_binding_mismatch")
        if task.repository_path is None:
            raise ValueError("local_repository_required")
        Path(task.repository_path).resolve(strict=True).relative_to(root.resolve(strict=True))
        # Discard even an accidentally supplied real credential before SDK construction.
        safe_settings = settings.model_copy(update={"openai_api_key": SecretStr(_DUMMY_KEY)})
        request = RepairRunRequest(
            thread_id="repository-workload-preflight",
            run_id=uuid4(),
            started_at=datetime.now(UTC),
            task=task,
            sandbox_image_reference=binding.sandbox_image_reference,
        )
        repository = ReadOnlyRepository.from_settings(Path(task.repository_path), safe_settings)
        stage = "sandbox_preparation"
        factory = build_isolated_test_sandbox_factory(
            settings=safe_settings,
            sandbox_image_reference=binding.sandbox_image_reference,
            task=task,
            regression_patch=regression_patch,
        )
        with tempfile.TemporaryDirectory(prefix="repopilot-workload-probe-") as directory:
            sandbox = factory(Path(directory))
            handle = sandbox.prepare(task)
            try:
                stage = "setup"
                observations = _run_setup_commands(sandbox, handle, request, safe_settings)
                stage = "provider_configuration"
                registry, tools = _build_repair_registry(
                    repository=repository,
                    sandbox=sandbox,
                    handle=handle,
                    task=task,
                    settings=safe_settings,
                )
                initial = AgentState(
                    run_id=request.run_id,
                    task=task,
                    repository_summary=retrieval.repository_summary,
                    memory_namespace=retrieval.memory_namespace,
                    memory_load_audit=retrieval.memory_load_audit,
                    repository_memory=retrieval.repository_memory,
                    episodic_memory=retrieval.episodic_memory,
                    retrieved_context=retrieval.hits,
                    observations=observations,
                    budget=BudgetState(
                        max_steps=safe_settings.max_steps,
                        max_tokens=safe_settings.max_tokens,
                        max_cost_usd=safe_settings.max_cost_usd,
                        started_at=request.started_at,
                        deadline_at=request.started_at
                        + timedelta(seconds=safe_settings.timeout_seconds),
                    ),
                )
                graph_state = to_graph_state(initial)
                planned = from_graph_state({**graph_state, **initialize_repair_plan(graph_state)})
                assert planned.active_step_id is not None
                allowed = STEP_TOOL_CAPABILITIES[planned.active_step_id]
                provider_request = build_context_aware_provider_request(
                    request_id=repair_decision_request_id(request.run_id, 0),
                    run_id=request.run_id,
                    state=planned,
                    allowed_tools=allowed,
                    requested_at=datetime.now(UTC),
                    max_chars=safe_settings.openai_prompt_max_chars,
                )
                prompt = build_agent_prompt(
                    provider_request, max_chars=safe_settings.openai_prompt_max_chars
                )
                transport = _BlockedSDKTransport(
                    api_key=_DUMMY_KEY,
                    timeout_seconds=safe_settings.openai_request_timeout_seconds,
                )
                try:
                    provider = OpenAIProvider.from_settings(
                        settings=safe_settings, tools=tools, transport=transport
                    )
                    payload = provider._request_payload(provider_request, prompt)
                    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
                finally:
                    transport.close()
                metrics: dict[str, JsonValue] = {
                    "setup_command_count": len(observations),
                    "setup_offline": True,
                    "registered_tool_count": len(registry.list_tools()),
                    "first_decision_allowed_tools": list(allowed),
                    "provider_payload_bytes": len(encoded),
                    "provider_payload_sha256": hashlib.sha256(encoded).hexdigest(),
                    "provider_sdk_constructed": True,
                    "provider_requests_made": 0,
                    "paid_requests_made": 0,
                    "agent_constructed": False,
                    "gold_visible_to_agent": False,
                }
            finally:
                previous_stage = stage
                stage = "cleanup"
                sandbox.destroy(handle)
                stage = previous_stage
        metrics["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        metrics["temporary_sandboxes_destroyed"] = True
        return metrics
    except Exception as error:
        raise RepairWorkflowError(
            RepairWorkflowErrorCode.SANDBOX_PREPARATION,
            detail=safe_repair_failure_detail(error, stage=stage),
        ) from None
