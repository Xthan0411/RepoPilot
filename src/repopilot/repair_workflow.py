"""Production repair workflow wiring retrieval, Agent, Docker, Trace, and Verifier."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal, Protocol, Self, cast
from uuid import UUID, uuid4, uuid5

from pydantic import Field, ValidationError, field_validator, model_validator

from repopilot.agent import RepairAgentGraph, repair_decision_request_id
from repopilot.config import RepoPilotSettings
from repopilot.embeddings import (
    EmbeddingError,
    EmbeddingIndexer,
    EmbeddingProvider,
    LocalEmbeddingCache,
    embedding_cache_from_settings,
    embedding_executor_from_settings,
    embedding_provider_from_settings,
)
from repopilot.enums import (
    ErrorCategory,
    MemoryAccessMode,
    ModelProvider,
    PolicyDecision,
    RepairContextMode,
    RepairWorkflowErrorCode,
    RepositoryMemoryErrorCode,
    RepositoryMemoryLookupStatus,
    ReproductionStatus,
    RetrievalMode,
    RunStatus,
    SandboxCommandPhase,
    SandboxLifecycle,
    SandboxNetworkMode,
    TerminationReason,
    VerifiedEpisodeLookupStatus,
)
from repopilot.memory import (
    MemoryNamespaceError,
    MemoryStoreBundle,
    RepositoryMemoryError,
    RepositoryMemoryStore,
    VerifiedEpisodeError,
    VerifiedEpisodeExecutionEvidence,
    VerifiedEpisodicRecord,
    build_repository_map_record,
    build_test_command_record,
    context_policy_version_from_events,
    project_repository_memory_record,
    project_verified_episode_record,
)
from repopilot.providers import (
    FakeProvider,
    ModelProviderAdapter,
    OpenAIProvider,
    ProviderResponse,
)
from repopilot.replay import replay_trace
from repopilot.retrieval import (
    BM25Error,
    BM25Policy,
    BM25Retriever,
    ChunkCorpusError,
    DenseRetrievalError,
    DenseRetriever,
    HybridPolicy,
    HybridRetrievalError,
    HybridRetriever,
    RepositoryChunkCorpusBuilder,
    RepositoryIngestError,
    ingest_repository,
)
from repopilot.sandbox import (
    DockerSandbox,
    RunRequest,
    SandboxAdapter,
    SandboxError,
    SandboxHandle,
    SandboxPolicy,
)
from repopilot.schemas import (
    AgentState,
    ArtifactRef,
    BudgetState,
    FinalResult,
    LocalizationMetrics,
    MemoryLoadAudit,
    MemoryNamespace,
    Observation,
    RelativePath,
    RepositoryMemoryItem,
    ReproductionResult,
    RetrievalHit,
    StrictModel,
    TaskId,
    TaskSpec,
    UsageMetrics,
    VerifiedEpisodeMemoryItem,
)
from repopilot.storage import PostgresVectorStore, VectorStore, VectorStoreError
from repopilot.tools import (
    ApplyPatchTool,
    InspectDiffTool,
    InspectSymbolTool,
    ListFilesTool,
    PatchToolSession,
    ReadFileTool,
    ReadOnlyRepository,
    RegisteredTool,
    RepositoryAccessError,
    RunTestsTool,
    SearchCodeTool,
    ToolRegistry,
    build_reproduction_result,
)
from repopilot.tools.patch import PatchCapableSandbox
from repopilot.trace import (
    AgentTrace,
    TraceArtifactError,
    write_trace,
)
from repopilot.verifier import (
    FinalResultContext,
    RuleBasedVerifier,
    VerificationEvidence,
    build_final_result,
)

REPAIR_RUN_SCHEMA_VERSION: Literal[1] = 1
SCRIPTED_PROVIDER_SCHEMA_VERSION: Literal[1] = 1
ARTIFACT_MANIFEST_SCHEMA_VERSION: Literal[1] = 1
RESULT_FILENAME: RelativePath = "result.json"
TRACE_FILENAME: RelativePath = "trace.json"
ARTIFACT_MANIFEST_FILENAME: RelativePath = "artifacts.json"
ARTIFACT_ROOT: RelativePath = "artifacts"
_MAX_INPUT_BYTES = 1_000_000
_IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:~-]{0,439}@sha256:[0-9a-f]{64}$")
MonotonicClock = Callable[[], float]
RepairSandboxFactory = Callable[[Path], "RepairSandbox"]


class RepairWorkflowError(RuntimeError):
    """Safe orchestration failure carrying no source, command, path, or secret."""

    def __init__(self, code: RepairWorkflowErrorCode) -> None:
        super().__init__(
            {
                RepairWorkflowErrorCode.INPUT_IO_ERROR: "repair input is unavailable",
                RepairWorkflowErrorCode.INVALID_TASK: "repair task is invalid",
                RepairWorkflowErrorCode.INVALID_PROVIDER_SCRIPT: (
                    "scripted Provider input is invalid"
                ),
                RepairWorkflowErrorCode.INVALID_OUTPUT: (
                    "repair output directory must be empty and non-symlinked"
                ),
                RepairWorkflowErrorCode.PROVIDER_CONFIGURATION: (
                    "repair Provider configuration is invalid"
                ),
                RepairWorkflowErrorCode.RETRIEVAL_FAILURE: ("repair retrieval preparation failed"),
                RepairWorkflowErrorCode.MEMORY_FAILURE: ("repair Memory operation failed"),
                RepairWorkflowErrorCode.SANDBOX_PREPARATION: ("repair Sandbox preparation failed"),
                RepairWorkflowErrorCode.SETUP_FAILURE: "task setup command failed",
                RepairWorkflowErrorCode.AGENT_FAILURE: "repair Agent execution failed",
                RepairWorkflowErrorCode.TRACE_FAILURE: "repair Trace validation failed",
                RepairWorkflowErrorCode.CLEANUP_FAILURE: "repair Sandbox cleanup failed",
                RepairWorkflowErrorCode.OUTPUT_IO_ERROR: "repair output could not be written",
            }[code]
        )
        self.code = code


class RepairSandbox(SandboxAdapter, PatchCapableSandbox, Protocol):
    """Combined runtime capabilities needed by one repair workflow."""


class ScriptedProviderManifest(StrictModel):
    """Offline Provider decisions plus exact run identity for Fake-mode CLI runs."""

    schema_version: Literal[1] = SCRIPTED_PROVIDER_SCHEMA_VERSION
    thread_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    run_id: UUID
    started_at: datetime
    responses: tuple[ProviderResponse, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_response_order(self) -> Self:
        if self.started_at.tzinfo is None or self.started_at.utcoffset() is None:
            raise ValueError("started_at must be timezone-aware")
        for index, response in enumerate(self.responses):
            if response.request_id != repair_decision_request_id(self.run_id, index):
                raise ValueError("scripted Provider request IDs must follow decision order")
        return self


class RepairRunRequest(StrictModel):
    """Fully resolved Agent-only task and trusted runtime identities."""

    schema_version: Literal[1] = REPAIR_RUN_SCHEMA_VERSION
    thread_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    run_id: UUID
    started_at: datetime
    task: TaskSpec
    sandbox_image_reference: str = Field(min_length=1, max_length=512, repr=False)
    context_mode: RepairContextMode = RepairContextMode.BM25
    evaluation_configuration_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    provider_responses: tuple[ProviderResponse, ...] = ()

    @field_validator("sandbox_image_reference")
    @classmethod
    def require_digest_pinned_image(cls, value: str) -> str:
        if _IMAGE_PATTERN.fullmatch(value) is None:
            raise ValueError("sandbox image must be an immutable digest reference")
        return value

    @model_validator(mode="after")
    def validate_runtime_identity(self) -> Self:
        if self.started_at.tzinfo is None or self.started_at.utcoffset() is None:
            raise ValueError("started_at must be timezone-aware")
        if self.task.repository_path is None or self.task.repository_url is not None:
            raise ValueError("P5 repair workflow requires one local repository path")
        for index, response in enumerate(self.provider_responses):
            if response.request_id != repair_decision_request_id(self.run_id, index):
                raise ValueError("Provider request IDs must follow decision order")
        return self


class RepairRetrievalContext(StrictModel):
    """Agent-visible BM25 context and reporting-only latency."""

    repository_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    repository_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repository_summary: str = Field(max_length=12_000)
    memory_namespace: MemoryNamespace | None = None
    memory_load_audit: MemoryLoadAudit | None = None
    repository_memory: tuple[RepositoryMemoryItem, ...] = ()
    episodic_memory: tuple[VerifiedEpisodeMemoryItem, ...] = ()
    hits: tuple[RetrievalHit, ...]
    indexing_latency_ms: int = Field(ge=0)
    retrieval_latency_ms: int = Field(ge=0)


class RepairArtifactManifest(StrictModel):
    """Stable index of the retained files rooted beneath one run output."""

    schema_version: Literal[1] = ARTIFACT_MANIFEST_SCHEMA_VERSION
    task_id: TaskId
    run_id: UUID
    thread_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    provider_name: str = Field(min_length=1, max_length=256)
    sandbox_image_reference: str = Field(min_length=1, max_length=512)
    context_mode: RepairContextMode = RepairContextMode.BM25
    settings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_configuration_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    prompt_versions: tuple[str, ...]
    artifact_root: RelativePath = ARTIFACT_ROOT
    artifacts: tuple[ArtifactRef, ...]
    result_path: RelativePath = RESULT_FILENAME
    trace_path: RelativePath = TRACE_FILENAME

    @model_validator(mode="after")
    def validate_unique_artifacts(self) -> Self:
        paths = tuple(item.relative_path for item in self.artifacts)
        if len(set(paths)) != len(paths):
            raise ValueError("artifact paths must be unique")
        if tuple(sorted(paths)) != paths:
            raise ValueError("artifact paths must be sorted")
        artifact_prefix = f"{self.artifact_root}/"
        if any(not path.startswith(artifact_prefix) for path in paths):
            raise ValueError("artifact paths must remain beneath artifact_root")
        if _IMAGE_PATTERN.fullmatch(self.sandbox_image_reference) is None:
            raise ValueError("sandbox image must be an immutable digest reference")
        return self


@dataclass(frozen=True)
class RepairRunArtifacts:
    """Completed in-memory repair plus its fixed output files."""

    result: FinalResult
    trace: AgentTrace
    final_state: AgentState
    artifact_manifest: RepairArtifactManifest
    result_path: Path
    trace_path: Path
    artifact_manifest_path: Path
    verified_episode: VerifiedEpisodicRecord | None


def load_repair_task(path: Path) -> TaskSpec:
    """Load one Agent-only TaskSpec and resolve its local repository path."""

    payload = _read_bounded_input(path, RepairWorkflowErrorCode.INPUT_IO_ERROR)
    try:
        task = TaskSpec.model_validate_json(payload)
    except (ValidationError, UnicodeDecodeError):
        raise RepairWorkflowError(RepairWorkflowErrorCode.INVALID_TASK) from None
    if task.repository_path is None or task.repository_url is not None:
        raise RepairWorkflowError(RepairWorkflowErrorCode.INVALID_TASK)
    raw_repository = Path(task.repository_path)
    try:
        manifest_root = path.parent.resolve(strict=True)
        if raw_repository.is_absolute():
            repository = raw_repository.resolve(strict=True)
        else:
            repository = (manifest_root / raw_repository).resolve(strict=True)
            if not repository.is_relative_to(manifest_root):
                raise OSError
    except OSError:
        raise RepairWorkflowError(RepairWorkflowErrorCode.INVALID_TASK) from None
    if not repository.is_dir():
        raise RepairWorkflowError(RepairWorkflowErrorCode.INVALID_TASK)
    return task.model_copy(update={"repository_path": str(repository)})


def load_scripted_provider(path: Path) -> ScriptedProviderManifest:
    """Load a bounded Fake Provider script without echoing rejected content."""

    payload = _read_bounded_input(
        path,
        RepairWorkflowErrorCode.INVALID_PROVIDER_SCRIPT,
    )
    try:
        return ScriptedProviderManifest.model_validate_json(payload)
    except (ValidationError, UnicodeDecodeError):
        raise RepairWorkflowError(RepairWorkflowErrorCode.INVALID_PROVIDER_SCRIPT) from None


def run_repair_task(
    *,
    task_path: Path,
    output_dir: Path,
    sandbox_image_reference: str,
    settings: RepoPilotSettings,
    provider_script_path: Path | None = None,
    memory_stores: MemoryStoreBundle | None = None,
) -> RepairRunArtifacts:
    """CLI-facing task loader with explicit Fake/OpenAI identity behavior."""

    task = load_repair_task(task_path)
    if settings.model_provider is ModelProvider.FAKE:
        if provider_script_path is None:
            raise RepairWorkflowError(RepairWorkflowErrorCode.PROVIDER_CONFIGURATION)
        script = load_scripted_provider(provider_script_path)
        request = RepairRunRequest(
            thread_id=script.thread_id,
            run_id=script.run_id,
            started_at=script.started_at,
            task=task,
            sandbox_image_reference=sandbox_image_reference,
            provider_responses=script.responses,
        )
    else:
        if provider_script_path is not None:
            raise RepairWorkflowError(RepairWorkflowErrorCode.PROVIDER_CONFIGURATION)
        run_id = uuid4()
        request = RepairRunRequest(
            thread_id=f"repair-{run_id.hex[:20]}",
            run_id=run_id,
            started_at=datetime.now(UTC),
            task=task,
            sandbox_image_reference=sandbox_image_reference,
        )
    return run_repair_request(
        request,
        output_dir=output_dir,
        settings=settings,
        memory_stores=memory_stores,
    )


def run_repair_request(
    request: RepairRunRequest,
    *,
    output_dir: Path,
    settings: RepoPilotSettings,
    sandbox: RepairSandbox | None = None,
    sandbox_factory: RepairSandboxFactory | None = None,
    memory_stores: MemoryStoreBundle | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_cache: LocalEmbeddingCache | None = None,
    vector_store: VectorStore | None = None,
    monotonic: MonotonicClock = time.perf_counter,
) -> RepairRunArtifacts:
    """Execute one complete repair and persist verified, replayable artifacts."""

    if sandbox is not None and sandbox_factory is not None:
        raise ValueError("provide either sandbox or sandbox_factory")
    if request.context_mode is RepairContextMode.NONE and memory_stores is not None:
        raise ValueError("Memory requires a precomputed repository context")
    if (
        request.context_mode is RepairContextMode.BM25
        and settings.retrieval_mode is not RetrievalMode.BM25
    ):
        raise ValueError("BM25 context requires BM25 retrieval settings")
    if (
        request.context_mode is RepairContextMode.CONFIGURED
        and settings.retrieval_mode is RetrievalMode.BM25
    ):
        raise ValueError("configured experiment context requires Dense or Hybrid retrieval")
    root = _prepare_output_directory(output_dir)
    repository = _repository(request.task, settings)
    retrieval = (
        build_repair_no_precomputed_context(repository, request.task)
        if request.context_mode is RepairContextMode.NONE
        else build_repair_retrieval_context(
            repository,
            request.task,
            settings,
            memory_stores=memory_stores,
            memory_created_at=request.started_at,
            embedding_provider=embedding_provider,
            embedding_cache=embedding_cache,
            vector_store=vector_store,
            monotonic=monotonic,
        )
    )
    if sandbox is not None:
        active_sandbox = sandbox
    elif sandbox_factory is not None:
        active_sandbox = sandbox_factory(root)
    else:
        active_sandbox = cast(
            RepairSandbox,
            DockerSandbox(
                policy=SandboxPolicy.from_settings(settings),
                image_reference=request.sandbox_image_reference,
                artifact_root=root / ARTIFACT_ROOT,
            ),
        )
    handle: SandboxHandle | None = None
    primary_error: RepairWorkflowError | None = None
    completed: (
        tuple[
            FinalResult,
            AgentTrace,
            AgentState,
            tuple[ArtifactRef, ...],
        ]
        | None
    ) = None
    try:
        try:
            handle = active_sandbox.prepare(request.task)
        except SandboxError:
            raise RepairWorkflowError(RepairWorkflowErrorCode.SANDBOX_PREPARATION) from None
        setup_observations = _run_setup_commands(
            active_sandbox,
            handle,
            request,
            settings,
        )
        registry, tools = _build_repair_registry(
            repository=repository,
            sandbox=active_sandbox,
            handle=handle,
            task=request.task,
            settings=settings,
        )
        provider = _build_provider(
            request=request,
            settings=settings,
            tools=tools,
        )
        initial = AgentState(
            task=request.task,
            repository_summary=retrieval.repository_summary,
            memory_namespace=retrieval.memory_namespace,
            memory_load_audit=retrieval.memory_load_audit,
            repository_memory=retrieval.repository_memory,
            episodic_memory=retrieval.episodic_memory,
            retrieved_context=retrieval.hits,
            observations=setup_observations,
            budget=BudgetState(
                max_steps=settings.max_steps,
                max_tokens=settings.max_tokens,
                max_cost_usd=settings.max_cost_usd,
                started_at=request.started_at,
                deadline_at=(request.started_at + timedelta(seconds=settings.timeout_seconds)),
            ),
        )
        graph = RepairAgentGraph.from_settings(
            provider=provider,
            registry=registry,
            settings=settings,
            clock=(
                (lambda: request.started_at)
                if settings.model_provider is ModelProvider.FAKE
                else (lambda: datetime.now(UTC))
            ),
        )
        try:
            final_state = graph.run(
                thread_id=request.thread_id,
                run_id=request.run_id,
                state=initial,
            )
            trace = graph.build_trace(request.thread_id, settings=settings)
            replay_trace(trace)
        except TraceArtifactError:
            raise RepairWorkflowError(RepairWorkflowErrorCode.TRACE_FAILURE) from None
        except Exception:
            raise RepairWorkflowError(RepairWorkflowErrorCode.AGENT_FAILURE) from None
        result = _build_result(
            request=request,
            final_state=final_state,
            trace=trace,
            retrieval=retrieval,
        )
        try:
            artifacts = tuple(
                sorted(
                    active_sandbox.collect_artifacts(handle),
                    key=lambda item: item.relative_path,
                )
            )
        except SandboxError:
            raise RepairWorkflowError(RepairWorkflowErrorCode.AGENT_FAILURE) from None
        completed = (result, trace, final_state, artifacts)
    except RepairWorkflowError as error:
        primary_error = error
    finally:
        if handle is not None and active_sandbox.lifecycle is SandboxLifecycle.PREPARED:
            try:
                active_sandbox.destroy(handle)
            except SandboxError:
                raise RepairWorkflowError(RepairWorkflowErrorCode.CLEANUP_FAILURE) from None
    if primary_error is not None:
        raise primary_error
    if completed is None:
        raise RepairWorkflowError(RepairWorkflowErrorCode.AGENT_FAILURE)

    result, trace, final_state, artifacts = completed
    prompt_versions = tuple(
        dict.fromkeys(
            response.prompt_version
            for event in trace.events
            if (response := event.provider_response) is not None
            and response.prompt_version is not None
        )
    )
    artifact_manifest = RepairArtifactManifest(
        task_id=request.task.task_id,
        run_id=request.run_id,
        thread_id=request.thread_id,
        provider_name=trace.provider_name,
        sandbox_image_reference=request.sandbox_image_reference,
        context_mode=request.context_mode,
        settings_sha256=_settings_sha256(settings),
        evaluation_configuration_sha256=request.evaluation_configuration_sha256,
        prompt_versions=prompt_versions,
        artifacts=artifacts,
    )
    result_path = root / RESULT_FILENAME
    trace_path = root / TRACE_FILENAME
    artifact_manifest_path = root / ARTIFACT_MANIFEST_FILENAME
    try:
        write_trace(trace_path, trace)
        _write_model(result_path, result)
        _write_model(artifact_manifest_path, artifact_manifest)
    except (OSError, TraceArtifactError):
        raise RepairWorkflowError(RepairWorkflowErrorCode.OUTPUT_IO_ERROR) from None
    verified_episode: VerifiedEpisodicRecord | None = None
    if (
        memory_stores is not None
        and memory_stores.namespace.access_mode is MemoryAccessMode.READ_WRITE
        and result.status is RunStatus.SUCCESS
    ):
        try:
            verified_episode = memory_stores.episodic.admit(
                repository_id=retrieval.repository_id,
                repository_content_sha256=retrieval.repository_content_sha256,
                result=result,
                trace=trace,
                execution=VerifiedEpisodeExecutionEvidence(
                    sandbox_image_reference=request.sandbox_image_reference,
                    settings_sha256=artifact_manifest.settings_sha256,
                    prompt_versions=artifact_manifest.prompt_versions,
                    context_policy_versions=(context_policy_version_from_events(trace.events),),
                    artifacts=artifact_manifest.artifacts,
                ),
                artifact_root=root,
                created_at=request.started_at,
            )
        except (
            MemoryNamespaceError,
            VerifiedEpisodeError,
            ValidationError,
            ValueError,
        ):
            raise RepairWorkflowError(RepairWorkflowErrorCode.MEMORY_FAILURE) from None
    return RepairRunArtifacts(
        result=result,
        trace=trace,
        final_state=final_state,
        artifact_manifest=artifact_manifest,
        result_path=result_path,
        trace_path=trace_path,
        artifact_manifest_path=artifact_manifest_path,
        verified_episode=verified_episode,
    )


def build_repair_retrieval_context(
    repository: ReadOnlyRepository,
    task: TaskSpec,
    settings: RepoPilotSettings,
    *,
    repository_memory_store: RepositoryMemoryStore | None = None,
    memory_stores: MemoryStoreBundle | None = None,
    memory_created_at: datetime | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_cache: LocalEmbeddingCache | None = None,
    vector_store: VectorStore | None = None,
    monotonic: MonotonicClock = time.perf_counter,
) -> RepairRetrievalContext:
    """Build the configured BM25, Dense, or Hybrid context before Docker mutation."""

    if repository_memory_store is not None and memory_stores is not None:
        raise ValueError("provide either legacy Repository Memory or a namespace bundle")
    repository_id = repository_id_for_task(task)
    try:
        indexing_started = monotonic()
        manifest = ingest_repository(
            repository,
            repository_id=repository_id,
            commit=task.base_commit,
        )
        corpus = (
            RepositoryChunkCorpusBuilder.from_settings(
                repository,
                settings,
            )
            .build(manifest)
            .corpus
        )
        sparse_retriever = (
            BM25Retriever.from_corpus(
                corpus,
                policy=BM25Policy.from_settings(settings),
            )
            if settings.retrieval_mode in (RetrievalMode.BM25, RetrievalMode.HYBRID)
            else None
        )
        dense_retriever: DenseRetriever | None = None
        if settings.retrieval_mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID):
            effective_provider = (
                embedding_provider
                if embedding_provider is not None
                else embedding_provider_from_settings(settings)
            )
            executor = embedding_executor_from_settings(settings, effective_provider)
            effective_store = (
                vector_store if vector_store is not None else PostgresVectorStore(settings)
            )
            EmbeddingIndexer(
                executor=executor,
                cache=(
                    embedding_cache
                    if embedding_cache is not None
                    else embedding_cache_from_settings(settings)
                ),
                vector_store=effective_store,
            ).index(corpus)
            dense_retriever = DenseRetriever(
                repository_id=repository_id,
                commit=task.base_commit,
                executor=executor,
                vector_store=effective_store,
            )
        retriever = HybridRetriever(
            corpus=corpus,
            sparse_retriever=sparse_retriever,
            dense_retriever=dense_retriever,
            policy=HybridPolicy.from_settings(settings),
        )
        indexing_latency_ms = max(
            0,
            round((monotonic() - indexing_started) * 1_000),
        )
        query = f"{task.issue_title}\n{task.issue_body}"[: settings.search_max_query_length]
        retrieval_started = monotonic()
        search = retriever.search(
            query,
            top_k=settings.retrieval_default_top_k,
        )
        retrieval_latency_ms = max(
            0,
            round((monotonic() - retrieval_started) * 1_000),
        )
    except (
        BM25Error,
        ChunkCorpusError,
        DenseRetrievalError,
        EmbeddingError,
        HybridRetrievalError,
        RepositoryAccessError,
        RepositoryIngestError,
        VectorStoreError,
        ValidationError,
        ValueError,
    ):
        raise RepairWorkflowError(RepairWorkflowErrorCode.RETRIEVAL_FAILURE) from None

    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    summaries: list[str] = []
    for hit in search.hits[:5]:
        chunk = chunks.get(hit.chunk_id)
        if chunk is None:
            raise RepairWorkflowError(RepairWorkflowErrorCode.RETRIEVAL_FAILURE)
        content = chunk.content[:800]
        summaries.append(
            f"{chunk.relative_path}:{chunk.start_line}-{chunk.end_line}"
            f" symbol={chunk.symbol_name or '<module>'}\n{content}"
        )
    mode_label = {
        RetrievalMode.BM25: "BM25",
        RetrievalMode.DENSE: "Dense",
        RetrievalMode.HYBRID: "Hybrid RRF",
    }[settings.retrieval_mode]
    summary = f"{mode_label} localization produced no source hit."
    if summaries:
        summary = f"Top {mode_label} source context (untrusted):\n" + "\n---\n".join(summaries)
    repository_memory: tuple[RepositoryMemoryItem, ...] = ()
    episodic_memory: tuple[VerifiedEpisodeMemoryItem, ...] = ()
    repository_status: RepositoryMemoryLookupStatus | None = None
    episodic_status: VerifiedEpisodeLookupStatus | None = None
    expired_episode_count = 0
    effective_repository_store = (
        memory_stores.repository if memory_stores is not None else repository_memory_store
    )
    if effective_repository_store is not None:
        if memory_created_at is None:
            raise ValueError("memory_created_at is required when Memory is enabled")
        try:
            write_repository_memory = (
                memory_stores is None
                or memory_stores.namespace.access_mode is MemoryAccessMode.READ_WRITE
            )
            if write_repository_memory:
                effective_repository_store.put(
                    build_repository_map_record(
                        manifest,
                        created_at=memory_created_at,
                    ),
                    manifest=manifest,
                )
                for test_index in range(len(task.test_commands)):
                    effective_repository_store.put(
                        build_test_command_record(
                            manifest,
                            task,
                            test_index=test_index,
                            created_at=memory_created_at,
                        ),
                        manifest=manifest,
                        task=task,
                    )
            lookup = effective_repository_store.lookup(manifest)
            repository_status = lookup.status
            if memory_stores is None and lookup.status is not RepositoryMemoryLookupStatus.CURRENT:
                raise RepositoryMemoryError(RepositoryMemoryErrorCode.CONTENT_STALE)
            if lookup.status is RepositoryMemoryLookupStatus.CURRENT:
                repository_memory = tuple(
                    project_repository_memory_record(record) for record in lookup.records
                )
        except (MemoryNamespaceError, RepositoryMemoryError):
            raise RepairWorkflowError(RepairWorkflowErrorCode.MEMORY_FAILURE) from None
    if memory_stores is not None:
        assert memory_created_at is not None
        try:
            episode_lookup = memory_stores.episodic.lookup(
                repository_id=repository_id,
                repository_commit=manifest.commit,
                repository_content_sha256=manifest.content_sha256,
                as_of=memory_created_at,
            )
            episodic_status = episode_lookup.status
            expired_episode_count = episode_lookup.expired_record_count
            if episode_lookup.status is VerifiedEpisodeLookupStatus.CURRENT:
                episodic_memory = tuple(
                    project_verified_episode_record(
                        record,
                        as_of=memory_created_at,
                    )
                    for record in episode_lookup.records
                )
        except (MemoryNamespaceError, VerifiedEpisodeError, ValidationError):
            raise RepairWorkflowError(RepairWorkflowErrorCode.MEMORY_FAILURE) from None
    memory_load_audit = (
        None
        if memory_stores is None
        else MemoryLoadAudit(
            namespace=memory_stores.namespace,
            repository_status=repository_status,
            episodic_status=episodic_status,
            repository_records_loaded=len(repository_memory),
            episodic_records_loaded=len(episodic_memory),
            expired_episode_count=expired_episode_count,
        )
    )
    return RepairRetrievalContext(
        repository_id=repository_id,
        repository_content_sha256=manifest.content_sha256,
        repository_summary=summary[:12_000],
        memory_namespace=(None if memory_stores is None else memory_stores.namespace),
        memory_load_audit=memory_load_audit,
        repository_memory=repository_memory,
        episodic_memory=episodic_memory,
        hits=search.hits,
        indexing_latency_ms=indexing_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
    )


def build_repair_no_precomputed_context(
    repository: ReadOnlyRepository,
    task: TaskSpec,
) -> RepairRetrievalContext:
    """Build identity-only context for the safety-adapted mini-swe baseline."""

    repository_id = repository_id_for_task(task)
    try:
        manifest = ingest_repository(
            repository,
            repository_id=repository_id,
            commit=task.base_commit,
        )
    except (
        RepositoryAccessError,
        RepositoryIngestError,
        ValidationError,
        ValueError,
    ):
        raise RepairWorkflowError(RepairWorkflowErrorCode.RETRIEVAL_FAILURE) from None
    return RepairRetrievalContext(
        repository_id=repository_id,
        repository_content_sha256=manifest.content_sha256,
        repository_summary=(
            "No precomputed repository localization is available; use the bounded "
            "repository tools to gather evidence."
        ),
        hits=(),
        indexing_latency_ms=0,
        retrieval_latency_ms=0,
    )


def repository_id_for_task(task: TaskSpec) -> str:
    """Hash trusted repository location without persisting or prompting the host path."""

    if task.repository_path is not None:
        try:
            location = str(Path(task.repository_path).resolve(strict=True))
        except OSError:
            raise RepairWorkflowError(RepairWorkflowErrorCode.INVALID_TASK) from None
        location_kind = "local_path"
    elif task.repository_url is not None:
        location = str(task.repository_url)
        location_kind = "repository_url"
    else:
        raise RepairWorkflowError(RepairWorkflowErrorCode.INVALID_TASK)
    encoded = json.dumps(
        {
            "kind": location_kind,
            "location": location,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"repo_{hashlib.sha256(encoded).hexdigest()[:24]}"


def _repository(task: TaskSpec, settings: RepoPilotSettings) -> ReadOnlyRepository:
    if task.repository_path is None:
        raise RepairWorkflowError(RepairWorkflowErrorCode.INVALID_TASK)
    try:
        return ReadOnlyRepository.from_settings(Path(task.repository_path), settings)
    except (RepositoryAccessError, ValueError):
        raise RepairWorkflowError(RepairWorkflowErrorCode.INVALID_TASK) from None


def _build_repair_registry(
    *,
    repository: ReadOnlyRepository,
    sandbox: RepairSandbox,
    handle: SandboxHandle,
    task: TaskSpec,
    settings: RepoPilotSettings,
) -> tuple[ToolRegistry, tuple[RegisteredTool, ...]]:
    patch_session = PatchToolSession.from_settings(
        sandbox=sandbox,
        handle=handle,
        task=task,
        settings=settings,
    )
    tools: tuple[RegisteredTool, ...] = (
        ListFilesTool(repository),
        ReadFileTool(repository),
        SearchCodeTool.from_settings(repository, settings),
        InspectSymbolTool.from_settings(repository, settings),
        RunTestsTool(sandbox=sandbox, handle=handle, task=task),
        ApplyPatchTool(patch_session),
        InspectDiffTool(patch_session),
    )
    registry = ToolRegistry.from_settings(settings)
    for tool in tools:
        registry.register(tool)
    return registry, tools


def _build_provider(
    *,
    request: RepairRunRequest,
    settings: RepoPilotSettings,
    tools: tuple[RegisteredTool, ...],
) -> ModelProviderAdapter:
    if settings.model_provider is ModelProvider.FAKE:
        if settings.model != "fake-deterministic" or not request.provider_responses:
            raise RepairWorkflowError(RepairWorkflowErrorCode.PROVIDER_CONFIGURATION)
        return FakeProvider(
            request.provider_responses,
            provider_name=settings.model,
        )
    if request.provider_responses:
        raise RepairWorkflowError(RepairWorkflowErrorCode.PROVIDER_CONFIGURATION)
    try:
        return OpenAIProvider.from_settings(
            settings=settings,
            tools=tools,
        )
    except Exception:
        raise RepairWorkflowError(RepairWorkflowErrorCode.PROVIDER_CONFIGURATION) from None


def _run_setup_commands(
    sandbox: RepairSandbox,
    handle: SandboxHandle,
    request: RepairRunRequest,
    settings: RepoPilotSettings,
) -> tuple[Observation, ...]:
    observations: list[Observation] = []
    for index, command in enumerate(request.task.setup_commands):
        call_id = uuid5(request.run_id, f"setup:{index}")
        try:
            observation = sandbox.run(
                handle,
                RunRequest(
                    call_id=call_id,
                    command=command,
                    phase=SandboxCommandPhase.SETUP,
                    network_requested=(settings.sandbox_network is SandboxNetworkMode.SETUP_ONLY),
                ),
            )
        except SandboxError:
            raise RepairWorkflowError(RepairWorkflowErrorCode.SETUP_FAILURE) from None
        observations.append(observation)
        if (
            observation.exit_code != 0
            or observation.error_type is not None
            or observation.policy_decision is not PolicyDecision.ALLOWED
        ):
            raise RepairWorkflowError(RepairWorkflowErrorCode.SETUP_FAILURE)
    return tuple(observations)


def _build_result(
    *,
    request: RepairRunRequest,
    final_state: AgentState,
    trace: AgentTrace,
    retrieval: RepairRetrievalContext,
) -> FinalResult:
    responses = tuple(
        event.provider_response for event in trace.events if event.provider_response is not None
    )
    usage = UsageMetrics(
        input_tokens=sum(response.usage.input_tokens for response in responses),
        output_tokens=sum(response.usage.output_tokens for response in responses),
        total_tokens=sum(response.usage.total_tokens for response in responses),
        estimated_cost_usd=sum(
            (response.usage.estimated_cost_usd for response in responses),
            start=Decimal("0"),
        ),
        latency_ms=sum(response.usage.latency_ms for response in responses),
    )
    localization = LocalizationMetrics(
        file_recall_at_1=0,
        file_recall_at_3=0,
        file_recall_at_5=0,
        mean_reciprocal_rank=0,
        first_relevant_file_rank=None,
        indexing_latency_ms=retrieval.indexing_latency_ms,
        retrieval_latency_ms=retrieval.retrieval_latency_ms,
    )
    reproduction = (
        ReproductionResult(status=ReproductionStatus.NOT_ATTEMPTED)
        if final_state.baseline_test_result is None
        else build_reproduction_result(final_state.baseline_test_result)
    )
    validation_test = (
        final_state.test_state.last_result if final_state.test_state.commands_run >= 2 else None
    )
    decision = RuleBasedVerifier().verify(
        VerificationEvidence(
            reproduction_result=reproduction,
            patch_state=final_state.patch_state,
            validation_test_run=validation_test,
            tool_steps=len(final_state.tool_results),
        )
    )
    context = FinalResultContext(
        task_id=request.task.task_id,
        run_id=request.run_id,
        localization_metrics=localization,
        usage=usage,
        trace_path=TRACE_FILENAME,
    )
    if final_state.termination_reason in {
        TerminationReason.AGENT_STOPPED,
        TerminationReason.TESTS_FAILED,
        TerminationReason.INVALID_PATCH,
    }:
        return build_final_result(decision, context)
    if final_state.termination_reason is None:
        raise RepairWorkflowError(RepairWorkflowErrorCode.AGENT_FAILURE)
    metrics = decision.task_metrics.model_copy(update={"task_success": False})
    return FinalResult(
        task_id=request.task.task_id,
        run_id=request.run_id,
        status=RunStatus.STOPPED,
        termination_reason=final_state.termination_reason,
        patch_path=decision.patch_path,
        reproduction_result=decision.reproduction_result,
        validation_result=decision.validation_result,
        localization_metrics=localization,
        task_metrics=metrics,
        usage=usage,
        trace_path=TRACE_FILENAME,
        error_category=_runtime_error_category(final_state.termination_reason),
    )


def _runtime_error_category(reason: TerminationReason) -> ErrorCategory:
    if reason is TerminationReason.RETRIEVAL_EXHAUSTED:
        return ErrorCategory.RETRIEVAL_MISS
    if reason in {
        TerminationReason.MAX_STEPS,
        TerminationReason.TOKEN_LIMIT,
        TerminationReason.TIMEOUT,
        TerminationReason.COST_LIMIT,
    }:
        return ErrorCategory.BUDGET_EXHAUSTED
    if reason is TerminationReason.SANDBOX_ERROR:
        return ErrorCategory.SANDBOX_FAILURE
    if reason is TerminationReason.PROVIDER_ERROR:
        return ErrorCategory.MODEL_FORMAT_ERROR
    if reason is TerminationReason.INVALID_PATCH:
        return ErrorCategory.INVALID_PATCH
    if reason is TerminationReason.TESTS_FAILED:
        return ErrorCategory.TEST_MISREAD
    return ErrorCategory.TOOL_MISUSE


def _prepare_output_directory(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise OSError
        path.mkdir(parents=True, exist_ok=True)
        root = path.resolve(strict=True)
        if not root.is_dir() or any(root.iterdir()):
            raise OSError
        return root
    except OSError:
        raise RepairWorkflowError(RepairWorkflowErrorCode.INVALID_OUTPUT) from None


def _read_bounded_input(
    path: Path,
    code: RepairWorkflowErrorCode,
) -> bytes:
    try:
        if path.is_symlink():
            raise OSError
        payload = path.read_bytes()
    except OSError:
        raise RepairWorkflowError(code) from None
    if len(payload) > _MAX_INPUT_BYTES:
        raise RepairWorkflowError(code)
    return payload


def _write_model(path: Path, model: StrictModel) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(f"{model.model_dump_json(indent=2)}\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def _settings_sha256(settings: RepoPilotSettings) -> str:
    encoded = json.dumps(
        settings.redacted_snapshot(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ARTIFACT_MANIFEST_FILENAME",
    "ARTIFACT_ROOT",
    "RESULT_FILENAME",
    "TRACE_FILENAME",
    "RepairArtifactManifest",
    "RepairRetrievalContext",
    "RepairRunArtifacts",
    "RepairRunRequest",
    "RepairSandbox",
    "RepairSandboxFactory",
    "RepairWorkflowError",
    "ScriptedProviderManifest",
    "build_repair_retrieval_context",
    "load_repair_task",
    "load_scripted_provider",
    "repository_id_for_task",
    "run_repair_request",
    "run_repair_task",
]
