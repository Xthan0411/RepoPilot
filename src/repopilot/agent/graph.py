"""Minimal thread-scoped LangGraph with safe checkpoint access."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from repopilot.agent.state import (
    AgentGraphState,
    AgentGraphUpdate,
    StateReductionConflict,
    ThreadIdentity,
    apply_state_update,
    from_graph_state,
    to_graph_state,
)
from repopilot.enums import AgentStateErrorCode, ErrorType
from repopilot.schemas import AgentState, TaskId

CHECKPOINT_NODE = "checkpoint_state"
CHECKPOINT_ALLOWED_TYPES = (
    ("repopilot.enums", "AgentFailureKind"),
    ("repopilot.enums", "DatasetSplit"),
    ("repopilot.enums", "ErrorType"),
    ("repopilot.enums", "MemoryAccessMode"),
    ("repopilot.enums", "PatchErrorCode"),
    ("repopilot.enums", "PatchStatus"),
    ("repopilot.enums", "PlanStepStatus"),
    ("repopilot.enums", "PolicyDecision"),
    ("repopilot.enums", "ProviderErrorCode"),
    ("repopilot.enums", "RepositoryErrorCode"),
    ("repopilot.enums", "RepositoryMemoryConfirmation"),
    ("repopilot.enums", "RepositoryMemoryKind"),
    ("repopilot.enums", "RepositoryMemoryLookupStatus"),
    ("repopilot.enums", "TerminationReason"),
    ("repopilot.enums", "TestRunCategory"),
    ("repopilot.enums", "TestStatus"),
    ("repopilot.enums", "ToolErrorCode"),
    ("repopilot.enums", "ToolResultStatus"),
    ("repopilot.enums", "VerifiedEpisodeLookupStatus"),
    ("repopilot.schemas", "ArtifactRef"),
    ("repopilot.schemas", "AgentFailure"),
    ("repopilot.schemas", "BudgetState"),
    ("repopilot.schemas", "CommandSpec"),
    ("repopilot.schemas", "MemoryLoadAudit"),
    ("repopilot.schemas", "MemoryNamespace"),
    ("repopilot.schemas", "Observation"),
    ("repopilot.schemas", "PatchState"),
    ("repopilot.schemas", "PlanStep"),
    ("repopilot.schemas", "RepositoryMemoryItem"),
    ("repopilot.schemas", "RetrievalHit"),
    ("repopilot.schemas", "TaskSpec"),
    ("repopilot.schemas", "TestState"),
    ("repopilot.schemas", "TestRunResult"),
    ("repopilot.schemas", "ToolCall"),
    ("repopilot.schemas", "ToolExecutionResult"),
    ("repopilot.schemas", "VerifiedEpisodeMemoryItem"),
)
STATE_ERROR_MESSAGES = {
    AgentStateErrorCode.INVALID_THREAD_ID: "thread ID is invalid",
    AgentStateErrorCode.THREAD_NOT_FOUND: "thread checkpoint does not exist",
    AgentStateErrorCode.THREAD_ALREADY_EXISTS: "thread checkpoint already exists",
    AgentStateErrorCode.TASK_MISMATCH: "thread checkpoint belongs to another task",
    AgentStateErrorCode.STATE_CONFLICT: "state update conflicts with checkpoint invariants",
    AgentStateErrorCode.CHECKPOINT_ERROR: "checkpoint operation failed",
}


GraphFailureCause = Literal[
    "graph_recursion_limit",
    "validation_error",
    "invalid_value",
    "local_io_error",
    "unexpected_exception",
]
_GRAPH_FAILURE_CAUSES = frozenset(
    {
        "graph_recursion_limit",
        "validation_error",
        "invalid_value",
        "local_io_error",
        "unexpected_exception",
    }
)


class AgentStateGraphError(RuntimeError):
    """Safe typed failure for state validation or checkpoint operations."""

    def __init__(
        self, code: AgentStateErrorCode, *, cause_type: GraphFailureCause | None = None
    ) -> None:
        if cause_type is not None and cause_type not in _GRAPH_FAILURE_CAUSES:
            raise ValueError("unapproved graph failure cause")
        super().__init__(STATE_ERROR_MESSAGES[code])
        self.code = code
        self.cause_type = cause_type
        self.error_type = (
            ErrorType.INTERNAL_ERROR
            if code is AgentStateErrorCode.CHECKPOINT_ERROR
            else ErrorType.VALIDATION_ERROR
        )


def build_in_memory_checkpointer() -> InMemorySaver:
    """Create the strict, process-local saver shared by RepoPilot graphs."""

    return InMemorySaver(
        serde=JsonPlusSerializer(
            allowed_msgpack_modules=CHECKPOINT_ALLOWED_TYPES,
        )
    )


def checkpoint_state(_: AgentGraphState) -> dict[str, object]:
    """Create a named, side-effect-free graph step for the initial checkpoint."""

    return {}


class AgentStateGraph:
    """Start, query, and safely update process-local Agent checkpoints."""

    def __init__(
        self,
        *,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> None:
        self._checkpointer = checkpointer or build_in_memory_checkpointer()
        builder = StateGraph(AgentGraphState)
        builder.add_node(CHECKPOINT_NODE, RunnableLambda(checkpoint_state))
        builder.add_edge(START, CHECKPOINT_NODE)
        builder.add_edge(CHECKPOINT_NODE, END)
        self._graph: CompiledStateGraph[
            AgentGraphState,
            None,
            AgentGraphState,
            AgentGraphState,
        ] = builder.compile(checkpointer=self._checkpointer, name="repopilot-state")

    @property
    def checkpointer(self) -> BaseCheckpointSaver[str]:
        """Expose the injected saver for controlled composition and tests."""

        return self._checkpointer

    def start(self, thread_id: str, state: AgentState) -> AgentState:
        """Create a new thread and persist its validated initial state."""

        config = self._thread_config(thread_id)
        existing = self._read_optional(config)
        if existing is not None:
            code = (
                AgentStateErrorCode.TASK_MISMATCH
                if existing.task.task_id != state.task.task_id
                else AgentStateErrorCode.THREAD_ALREADY_EXISTS
            )
            raise AgentStateGraphError(code)
        try:
            output = self._graph.invoke(to_graph_state(state), config)
            if not isinstance(output, Mapping):
                raise TypeError
            restored = from_graph_state(cast(Mapping[str, object], output))
        except AgentStateGraphError:
            raise
        except Exception:
            raise AgentStateGraphError(AgentStateErrorCode.CHECKPOINT_ERROR) from None
        if restored != state:
            raise AgentStateGraphError(AgentStateErrorCode.CHECKPOINT_ERROR)
        return restored

    def load(
        self,
        thread_id: str,
        *,
        expected_task_id: TaskId | None = None,
    ) -> AgentState:
        """Restore the latest complete checkpoint for one thread."""

        state = self._read_optional(self._thread_config(thread_id))
        if state is None:
            raise AgentStateGraphError(AgentStateErrorCode.THREAD_NOT_FOUND)
        if expected_task_id is not None and state.task.task_id != expected_task_id:
            raise AgentStateGraphError(AgentStateErrorCode.TASK_MISMATCH)
        return state

    def update(
        self,
        thread_id: str,
        update: AgentGraphUpdate,
    ) -> AgentState:
        """Validate a partial update, apply reducers, and create a checkpoint."""

        current = self.load(thread_id)
        if not update:
            raise AgentStateGraphError(AgentStateErrorCode.STATE_CONFLICT)
        try:
            expected = apply_state_update(current, update)
        except StateReductionConflict:
            raise AgentStateGraphError(AgentStateErrorCode.STATE_CONFLICT) from None
        config = self._thread_config(thread_id)
        try:
            self._graph.update_state(
                config,
                cast(dict[str, object], dict(update)),
                as_node=CHECKPOINT_NODE,
            )
            restored = self.load(thread_id)
        except AgentStateGraphError:
            raise
        except Exception:
            raise AgentStateGraphError(AgentStateErrorCode.CHECKPOINT_ERROR) from None
        if restored != expected:
            raise AgentStateGraphError(AgentStateErrorCode.CHECKPOINT_ERROR)
        return restored

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
