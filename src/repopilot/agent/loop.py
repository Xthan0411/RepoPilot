"""Compiled deterministic happy-path Agent loop."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Literal, Self, cast
from uuid import UUID

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import JsonValue, ValidationError

from repopilot.agent.graph import (
    AgentStateGraphError,
    build_in_memory_checkpointer,
)
from repopilot.agent.nodes import (
    AgentLoopError,
    Clock,
    HappyPathNodes,
    initialize_plan,
    system_utc_now,
)
from repopilot.agent.state import (
    AgentGraphState,
    ThreadIdentity,
    from_graph_state,
    to_graph_state,
)
from repopilot.config import RepoPilotSettings
from repopilot.context import CONTEXT_POLICY_VERSION
from repopilot.enums import AgentStateErrorCode, TraceErrorCode, WorkingMemoryErrorCode
from repopilot.memory import (
    WorkingMemoryCheckpoint,
    WorkingMemoryError,
    WorkingMemoryStore,
    build_working_memory_checkpoint,
    context_policy_version_from_events,
    verify_working_memory_integrity,
)
from repopilot.providers.base import ModelProviderAdapter
from repopilot.schemas import AgentState
from repopilot.tools.registry import ToolRegistry
from repopilot.trace import AgentTrace, TraceArtifactError

INITIALIZE_NODE = "initialize_plan"
DECIDE_NODE = "decide"
EXECUTE_NODE = "execute_tool"
OBSERVE_NODE = "observe"
FINISH_NODE = "finish"
TERMINATE_NODE = "terminate"


class HappyPathAgentGraph:
    """Run the fixed search/read/test/finish loop with injected Fakes."""

    def __init__(
        self,
        *,
        provider: ModelProviderAdapter,
        registry: ToolRegistry,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        max_replans: int = 1,
        prompt_max_chars: int = 50_000,
        clock: Clock = system_utc_now,
    ) -> None:
        self._checkpointer = checkpointer or build_in_memory_checkpointer()
        self._provider_name = provider.provider_name
        self._clock = clock
        self._nodes = HappyPathNodes(
            provider=provider,
            registry=registry,
            max_replans=max_replans,
            prompt_max_chars=prompt_max_chars,
            clock=clock,
        )
        builder = StateGraph(AgentGraphState)
        builder.add_node(INITIALIZE_NODE, RunnableLambda(self._nodes.initialize))
        builder.add_node(DECIDE_NODE, RunnableLambda(self._nodes.decide))
        builder.add_node(EXECUTE_NODE, RunnableLambda(self._nodes.execute_tool))
        builder.add_node(OBSERVE_NODE, RunnableLambda(self._nodes.observe))
        builder.add_node(FINISH_NODE, RunnableLambda(self._nodes.finish))
        builder.add_node(TERMINATE_NODE, RunnableLambda(self._nodes.terminate))
        builder.add_edge(START, INITIALIZE_NODE)
        builder.add_edge(INITIALIZE_NODE, DECIDE_NODE)
        builder.add_conditional_edges(
            DECIDE_NODE,
            self._nodes.route_after_decision,
            {
                "execute_tool": EXECUTE_NODE,
                "finish": FINISH_NODE,
                "terminate": TERMINATE_NODE,
            },
        )
        builder.add_edge(EXECUTE_NODE, OBSERVE_NODE)
        builder.add_conditional_edges(
            OBSERVE_NODE,
            self._nodes.route_after_observation,
            {
                "decide": DECIDE_NODE,
                "terminate": TERMINATE_NODE,
            },
        )
        builder.add_edge(FINISH_NODE, END)
        builder.add_edge(TERMINATE_NODE, END)
        self._graph: CompiledStateGraph[
            AgentGraphState,
            None,
            AgentGraphState,
            AgentGraphState,
        ] = builder.compile(
            checkpointer=self._checkpointer,
            name="repopilot-happy-path",
        )
        recovery_builder = StateGraph(AgentGraphState)
        recovery_builder.add_node(DECIDE_NODE, RunnableLambda(self._nodes.decide))
        recovery_builder.add_node(
            EXECUTE_NODE,
            RunnableLambda(self._nodes.execute_tool),
        )
        recovery_builder.add_node(OBSERVE_NODE, RunnableLambda(self._nodes.observe))
        recovery_builder.add_node(FINISH_NODE, RunnableLambda(self._nodes.finish))
        recovery_builder.add_node(
            TERMINATE_NODE,
            RunnableLambda(self._nodes.terminate),
        )
        recovery_builder.add_conditional_edges(
            START,
            _route_recovery,
            {
                "decide": DECIDE_NODE,
                "execute_tool": EXECUTE_NODE,
                "observe": OBSERVE_NODE,
                "finish": FINISH_NODE,
            },
        )
        recovery_builder.add_conditional_edges(
            DECIDE_NODE,
            self._nodes.route_after_decision,
            {
                "execute_tool": EXECUTE_NODE,
                "finish": FINISH_NODE,
                "terminate": TERMINATE_NODE,
            },
        )
        recovery_builder.add_edge(EXECUTE_NODE, OBSERVE_NODE)
        recovery_builder.add_conditional_edges(
            OBSERVE_NODE,
            self._nodes.route_after_observation,
            {
                "decide": DECIDE_NODE,
                "terminate": TERMINATE_NODE,
            },
        )
        recovery_builder.add_edge(FINISH_NODE, END)
        recovery_builder.add_edge(TERMINATE_NODE, END)
        self._recovery_graph: CompiledStateGraph[
            AgentGraphState,
            None,
            AgentGraphState,
            AgentGraphState,
        ] = recovery_builder.compile(
            checkpointer=self._checkpointer,
            name="repopilot-happy-path-recovery",
        )

    @property
    def checkpointer(self) -> BaseCheckpointSaver[str]:
        return self._checkpointer

    @classmethod
    def from_settings(
        cls,
        *,
        provider: ModelProviderAdapter,
        registry: ToolRegistry,
        settings: RepoPilotSettings,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        clock: Clock = system_utc_now,
    ) -> Self:
        """Apply the validated recovery limit from runtime settings."""

        return cls(
            provider=provider,
            registry=registry,
            checkpointer=checkpointer,
            max_replans=settings.max_replans,
            prompt_max_chars=settings.openai_prompt_max_chars,
            clock=clock,
        )

    def run(
        self,
        *,
        thread_id: str,
        run_id: UUID,
        state: AgentState,
    ) -> AgentState:
        """Start a fresh thread and execute the graph through honest finish."""

        config = self._thread_config(thread_id)
        existing = self._read_optional(config)
        if existing is not None:
            code = (
                AgentStateErrorCode.TASK_MISMATCH
                if existing.task.task_id != state.task.task_id
                else AgentStateErrorCode.THREAD_ALREADY_EXISTS
            )
            raise AgentStateGraphError(code)
        if state.run_id is not None and state.run_id != run_id:
            raise AgentStateGraphError(AgentStateErrorCode.STATE_CONFLICT)
        try:
            initial = AgentState.model_validate(
                {
                    **state.model_dump(mode="python"),
                    "run_id": run_id,
                }
            )
            initialize_plan(to_graph_state(initial))
            output = self._graph.invoke(to_graph_state(initial), config)
            if not isinstance(output, Mapping):
                raise TypeError
            return from_graph_state(cast(Mapping[str, object], output))
        except (AgentLoopError, AgentStateGraphError):
            raise
        except (ValidationError, ValueError):
            raise AgentStateGraphError(AgentStateErrorCode.STATE_CONFLICT) from None
        except Exception:
            raise AgentStateGraphError(AgentStateErrorCode.CHECKPOINT_ERROR) from None

    def load(self, thread_id: str) -> AgentState:
        """Return the latest valid checkpoint, including a failed mid-loop state."""

        state = self._read_optional(self._thread_config(thread_id))
        if state is None:
            raise AgentStateGraphError(AgentStateErrorCode.THREAD_NOT_FOUND)
        return state

    def save_working_memory(
        self,
        thread_id: str,
        *,
        store: WorkingMemoryStore,
        sequence: int,
        created_at: datetime | None = None,
    ) -> WorkingMemoryCheckpoint:
        """Explicitly persist one nonterminal checkpoint and its Trace prefix."""

        state = self.load(thread_id)
        if state.run_id is None:
            raise WorkingMemoryError(WorkingMemoryErrorCode.INVALID_SNAPSHOT)
        events = self._nodes.trace_recorder.events(state.run_id)
        checkpoint = build_working_memory_checkpoint(
            task_id=state.task.task_id,
            run_id=state.run_id,
            source_thread_id=thread_id,
            repository_commit=state.task.base_commit,
            context_policy_version=context_policy_version_from_events(events),
            sequence=sequence,
            created_at=created_at or self._clock(),
            state=state,
            trace_events=events,
        )
        store.save(checkpoint)
        return checkpoint

    def resume_from_working_memory(
        self,
        *,
        thread_id: str,
        checkpoint: WorkingMemoryCheckpoint,
    ) -> AgentState:
        """Continue one verified snapshot in a distinct, empty LangGraph thread."""

        verify_working_memory_integrity(checkpoint)
        config = self._thread_config(thread_id)
        if (
            thread_id == checkpoint.source_thread_id
            or checkpoint.context_policy_version != CONTEXT_POLICY_VERSION
            or self._read_optional(config) is not None
        ):
            raise WorkingMemoryError(WorkingMemoryErrorCode.RECOVERY_CONFLICT)
        graph_state = to_graph_state(checkpoint.state)
        _route_recovery(graph_state)
        try:
            self._nodes.trace_recorder.restore(
                checkpoint.run_id,
                checkpoint.trace_events,
            )
            output = self._recovery_graph.invoke(graph_state, config)
            if not isinstance(output, Mapping):
                raise TypeError
            return from_graph_state(cast(Mapping[str, object], output))
        except (AgentLoopError, AgentStateGraphError, WorkingMemoryError):
            raise
        except (ValidationError, ValueError):
            raise WorkingMemoryError(WorkingMemoryErrorCode.INVALID_SNAPSHOT) from None
        except Exception:
            raise AgentStateGraphError(AgentStateErrorCode.CHECKPOINT_ERROR) from None

    def build_trace(
        self,
        thread_id: str,
        *,
        settings: RepoPilotSettings,
    ) -> AgentTrace:
        """Build an integrity-protected Trace from recorded node transitions."""

        final_state = self.load(thread_id)
        if final_state.run_id is None:
            raise TraceArtifactError(TraceErrorCode.INVALID_DOCUMENT)
        events = self._nodes.trace_recorder.events(final_state.run_id)
        if not events:
            raise TraceArtifactError(TraceErrorCode.INVALID_DOCUMENT)
        return AgentTrace.build(
            run_id=final_state.run_id,
            task_id=final_state.task.task_id,
            thread_id=thread_id,
            provider_name=self._provider_name,
            settings=cast(dict[str, JsonValue], settings.redacted_snapshot()),
            events=events,
            final_state=final_state,
        )

    def _read_optional(self, config: RunnableConfig) -> AgentState | None:
        try:
            snapshot = self._graph.get_state(config)
            if not snapshot.values:
                return None
            return from_graph_state(cast(Mapping[str, object], snapshot.values))
        except ValidationError:
            raise AgentStateGraphError(AgentStateErrorCode.CHECKPOINT_ERROR) from None
        except AgentStateGraphError:
            raise
        except Exception:
            raise AgentStateGraphError(AgentStateErrorCode.CHECKPOINT_ERROR) from None

    @staticmethod
    def _thread_config(thread_id: str) -> RunnableConfig:
        try:
            identity = ThreadIdentity(thread_id=thread_id)
        except ValidationError:
            raise AgentStateGraphError(AgentStateErrorCode.INVALID_THREAD_ID) from None
        return {"configurable": {"thread_id": identity.thread_id}}


def _route_recovery(
    state: AgentGraphState,
) -> Literal["decide", "execute_tool", "observe", "finish"]:
    """Select the first unfinished node without re-running initialization."""

    snapshot = from_graph_state(state)
    if snapshot.termination_reason is not None or snapshot.active_step_id is None:
        raise WorkingMemoryError(WorkingMemoryErrorCode.TERMINAL_SNAPSHOT)
    if snapshot.pending_tool_result is not None:
        return "observe"
    if snapshot.pending_tool_call is not None:
        return "execute_tool"
    if snapshot.finish_summary is not None:
        return "finish"
    return "decide"
