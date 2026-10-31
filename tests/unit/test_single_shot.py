"""Offline one-response baseline, exact initial-context reuse and independent evaluator."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid5

import pytest
from test_isolated_repair_workflow import (
    PRIVATE_MARKER,
    REGRESSION_PATCH,
    RUN_ID,
    SOURCE,
    STARTED_AT,
    _candidate_patch,
    _fixture_task,
    _TrustedFixtureSandbox,
)

from repopilot.agent import repair_decision_request_id
from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    DatasetSplit,
    ModelProvider,
    RepairContextMode,
    RunStatus,
    TerminationReason,
)
from repopilot.evals.isolated_tests import IsolatedTestSandbox
from repopilot.evals.runner import (
    EvaluationExecutionRequest,
    EvaluationRunnerError,
    EvaluationTaskExecutionError,
    RepairWorkflowEvaluationExecutor,
    build_evaluation_run_manifest,
    inspect_evaluation_run,
    run_evaluation,
)
from repopilot.evals.single_shot import (
    SINGLE_SHOT_ARM_ID,
    SINGLE_SHOT_PROMPT_VERSION,
    SingleShotContext,
    SingleShotEvaluationExecutor,
    SingleShotTrace,
    _digest,
    _issue_digest,
    load_hybrid_initial_context,
    verify_single_shot_trace,
)
from repopilot.providers import FinishAction, ProviderResponse, ProviderUsageDelta
from repopilot.providers.prompts import _parse_rendered_payload, build_agent_prompt
from repopilot.repair_workflow import RepairRetrievalContext
from repopilot.schemas import EvaluationTaskSpec, FinalResult, RetrievalHit
from repopilot.trace import AgentTrace

IMAGE = f"python@sha256:{'a' * 64}"
MODEL = "gpt-5-mini-2025-08-07"


class Transport:
    def __init__(self, response, events):
        self.response = response
        self.events = events
        self.requests = []

    def create(self, payload):
        self.events.append("provider")
        self.requests.append(payload)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _response(diff=None):
    return {
        "id": "resp_fixture",
        "model": MODEL,
        "status": "completed",
        "usage": {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140},
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(
                            {
                                "diff": _candidate_patch(correct_patch=True)
                                if diff is None
                                else diff,
                            }
                        ),
                    }
                ],
            }
        ],
    }


def _context(task):
    inputs = {
        "task_id": task.task_id,
        "base_commit": task.base_commit,
        "issue_title": task.issue_title,
        "issue_body": task.issue_body,
        "repository_summary": "calculator.py\n" + SOURCE,
        "retrieved_context": [],
    }
    return SingleShotContext(
        **inputs,
        hybrid_run_id=RUN_ID,
        hybrid_execution_run_id=RUN_ID,
        hybrid_manifest_sha256="a" * 64,
        hybrid_trace_sha256="b" * 64,
        hybrid_prompt_sha256="c" * 64,
        raw_issue_sha256=_issue_digest(task),
        initial_context_sha256=_digest(inputs),
    )


@pytest.fixture
def fixture(tmp_path):
    task = _fixture_task(tmp_path)
    events = []
    transport = Transport(_response(), events)

    def sandbox_factory(root):
        events.append("sandbox")
        assert events[0] == "provider"
        return IsolatedTestSandbox(
            task=task,
            regression_patch=REGRESSION_PATCH,
            artifact_root=root / "artifacts",
            agent_factory=lambda path: _TrustedFixtureSandbox(path, tmp_path / "agent"),
            evaluator_factory=lambda path: _TrustedFixtureSandbox(path, tmp_path / "evaluator"),
        )

    settings = RepoPilotSettings(
        model_provider=ModelProvider.OPENAI,
        model=MODEL,
        openai_api_key="fake-no-provider-call",
        openai_max_retries=5,
    )
    executor = SingleShotEvaluationExecutor(
        settings=settings,
        sandbox_image_reference=IMAGE,
        evaluation_configuration_sha256="c" * 64,
        sandbox_factory=sandbox_factory,
        initial_context_factory=_context,
        transport=transport,
    )
    request = EvaluationExecutionRequest(
        task=task,
        run_id=RUN_ID,
        attempt_id="fixture",
        execution_index=0,
        thread_id="one-shot",
        started_at=datetime.now(UTC),
        output_dir=tmp_path / "out",
    )
    return executor, request, transport, events


def test_exactly_one_response_then_real_isolated_verification(fixture):
    executor, request, transport, events = fixture
    evidence = executor.execute(request)
    assert evidence.task_success and evidence.status is RunStatus.SUCCESS
    assert events == ["provider", "sandbox"]
    assert len(transport.requests) == 1
    payload = transport.requests[0]
    assert "tools" not in payload and "previous_response_id" not in payload
    assert payload["text"]["format"]["strict"] is True
    assert payload["max_output_tokens"] == executor.settings.openai_max_output_tokens
    assert payload["reasoning"]["effort"] == executor.settings.openai_reasoning_effort.value
    assert PRIVATE_MARKER not in json.dumps(payload)
    assert "test_commands" not in payload["input"]
    trace = SingleShotTrace.model_validate_json((request.output_dir / "trace.json").read_bytes())
    result = FinalResult.model_validate_json((request.output_dir / "result.json").read_bytes())
    verify_single_shot_trace(trace, result)
    assert trace.execution_order == (
        "provider_response",
        "sandbox_setup",
        "baseline",
        "patch",
        "validation",
    )
    assert trace.baseline.test_run.failures == 1
    assert trace.validation.test_run.failures == 0
    assert evidence.usage.estimated_cost_usd > 0
    assert Path(request.task.repository_path, "calculator.py").read_text() == SOURCE
    assert PRIVATE_MARKER not in (request.output_dir / "trace.json").read_text()


@pytest.mark.parametrize(
    "diff,reason",
    [
        (_candidate_patch(correct_patch=False), "tests_failed"),
        ("not a unified diff", "invalid_patch"),
        ("", "invalid_patch"),
        (REGRESSION_PATCH, "invalid_patch"),
    ],
)
def test_no_retry_or_feedback_after_failed_candidate(fixture, diff, reason):
    executor, request, transport, _ = fixture
    transport.response = _response(diff)
    evidence = executor.execute(request)
    assert not evidence.task_success
    assert evidence.termination_reason.value == reason
    assert len(transport.requests) == 1


@pytest.mark.parametrize("bad", ["refusal", "incomplete", "schema", "model", "tools", "duplicate"])
def test_known_usage_response_failure_is_retained_without_evaluator_or_retry(fixture, bad):
    executor, request, transport, events = fixture
    response = transport.response
    if bad == "refusal":
        response["output"][0]["content"] = [{"type": "refusal", "refusal": "private refusal"}]
    elif bad == "incomplete":
        response["status"] = "incomplete"
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    elif bad == "schema":
        response["output"][0]["content"][0]["text"] = '{"diff":"", "extra":true}'
    elif bad == "model":
        response["model"] = "other-model"
    elif bad == "tools":
        response["output"] = [{"type": "function_call", "name": "run_tests"}]
    else:
        response["output"].append(response["output"][0])
    evidence = executor.execute(request)
    assert not evidence.task_success
    assert evidence.usage.input_tokens == 100
    assert evidence.usage.estimated_cost_usd > 0
    assert len(transport.requests) == 1 and events == ["provider"]
    trace = SingleShotTrace.model_validate_json((request.output_dir / "trace.json").read_bytes())
    assert trace.provider_failure is not None
    assert trace.candidate_diff is None
    assert trace.execution_order == ("provider_response",)
    assert "private refusal" not in (request.output_dir / "trace.json").read_text()


@pytest.mark.parametrize("bad", ["transport", "usage"])
def test_unknown_usage_fails_conservatively_without_retries(fixture, bad):
    executor, request, transport, events = fixture
    if bad == "transport":
        transport.response = RuntimeError("SECRET SHOULD NOT LEAK")
    else:
        del transport.response["usage"]
    with pytest.raises(EvaluationTaskExecutionError) as error:
        executor.execute(request)
    assert "SECRET" not in str(error.value)
    assert events == ["provider"] and len(transport.requests) == 1


def test_wrong_context_is_rejected_before_provider(fixture):
    executor, request, transport, events = fixture
    context = _context(request.task).model_copy(update={"base_commit": "f" * 40})
    executor = replace(executor, initial_context_factory=lambda _: context)
    with pytest.raises(EvaluationTaskExecutionError):
        executor.execute(request)
    assert transport.requests == [] and events == []


@pytest.mark.parametrize("kind", ["token_limit", "cost_limit", "timeout"])
def test_budget_exhaustion_after_response_never_enters_evaluator(fixture, kind):
    executor, request, transport, events = fixture
    if kind == "token_limit":
        executor = replace(
            executor, settings=executor.settings.model_copy(update={"max_tokens": 100})
        )
    elif kind == "cost_limit":
        executor = replace(
            executor,
            settings=executor.settings.model_copy(update={"max_cost_usd": Decimal("0.000001")}),
        )
    else:
        executor = replace(
            executor, settings=executor.settings.model_copy(update={"timeout_seconds": 2})
        )
        original = transport.create

        def create(payload):
            result = original(payload)
            # Simulate time spent by this response without sleeping in an offline test.
            object.__setattr__(request, "started_at", datetime.now(UTC) - timedelta(seconds=5))
            return result

        transport.create = create
    evidence = executor.execute(request)
    assert evidence.termination_reason.value == kind
    assert events == ["provider"] and len(transport.requests) == 1
    result = FinalResult.model_validate_json((request.output_dir / "result.json").read_bytes())
    trace = SingleShotTrace.model_validate_json((request.output_dir / "trace.json").read_bytes())
    verify_single_shot_trace(trace, result)


def test_expired_deadline_makes_no_provider_call(fixture):
    executor, request, transport, events = fixture
    request = replace(request, started_at=STARTED_AT)
    with pytest.raises(EvaluationTaskExecutionError):
        executor.execute(request)
    assert transport.requests == [] and events == []


@pytest.mark.parametrize("stage", ["setup", "baseline", "patch", "validation", "cleanup"])
def test_evaluator_deadline_never_reports_success_and_preserves_usage(fixture, stage):
    executor, request, transport, _ = fixture
    original_factory = executor.sandbox_factory
    calls = []

    def factory(root):
        sandbox = original_factory(root)
        original_prepare = sandbox.prepare
        original_run = sandbox.run
        original_patch = sandbox.apply_patch
        original_destroy = sandbox.destroy
        run_count = 0

        def finish(name, result):
            calls.append(name)
            if name == stage:
                object.__setattr__(
                    request, "started_at", datetime.now(UTC) - timedelta(seconds=9999)
                )
            return result

        def prepare(task):
            return finish("setup", original_prepare(task))

        def run(handle, test_request):
            nonlocal run_count
            run_count += 1
            return finish(
                "baseline" if run_count == 1 else "validation", original_run(handle, test_request)
            )

        def patch(handle, patch_request):
            return finish("patch", original_patch(handle, patch_request))

        def destroy(handle):
            return finish("cleanup", original_destroy(handle))

        sandbox.prepare, sandbox.run = prepare, run
        sandbox.apply_patch, sandbox.destroy = patch, destroy
        return sandbox

    executor = replace(executor, sandbox_factory=factory)
    evidence = executor.execute(request)
    assert evidence.termination_reason is TerminationReason.TIMEOUT
    assert not evidence.task_success and evidence.usage.estimated_cost_usd > 0
    assert len(transport.requests) == 1
    assert calls[-1] == "cleanup"
    trace = SingleShotTrace.model_validate_json((request.output_dir / "trace.json").read_bytes())
    result = FinalResult.model_validate_json((request.output_dir / "result.json").read_bytes())
    assert trace.provider_failure is None and trace.evaluation_stop == "timeout"
    verify_single_shot_trace(trace, result)
    expected = ["setup", "baseline", "patch", "validation", "cleanup"]
    assert calls == expected[: expected.index(stage) + 1] + (
        [] if stage == "cleanup" else ["cleanup"]
    )


def test_trace_mutation_is_not_accepted(fixture):
    executor, request, _, _ = fixture
    executor.execute(request)
    payload = json.loads((request.output_dir / "trace.json").read_text())
    payload["candidate_diff"] = "tampered"
    with pytest.raises(ValueError, match=r"candidate/patch mismatch|digest"):
        SingleShotTrace.model_validate(payload)


def test_provider_failure_result_cannot_be_resealed_as_successful_evaluation(fixture):
    executor, request, transport, _ = fixture
    transport.response["status"] = "incomplete"
    executor.execute(request)
    result = FinalResult.model_validate_json((request.output_dir / "result.json").read_bytes())
    trace = SingleShotTrace.model_validate_json((request.output_dir / "trace.json").read_bytes())
    forged = result.model_copy(update={"termination_reason": TerminationReason.INVALID_PATCH})
    payload = trace.model_dump(mode="json", exclude={"content_sha256"})
    payload["final_result_sha256"] = _digest(forged.model_dump(mode="json"))
    resealed = SingleShotTrace.model_validate({**payload, "content_sha256": _digest(payload)})
    with pytest.raises(ValueError, match="provider failure outcome"):
        verify_single_shot_trace(resealed, forged)


def test_raw_reasoning_is_never_persisted(fixture):
    executor, request, transport, _ = fixture
    transport.response["output"].insert(
        0,
        {
            "type": "reasoning",
            "summary": [{"text": "PRIVATE_REASONING_CANARY"}],
        },
    )
    evidence = executor.execute(request)
    assert evidence.task_success
    assert "PRIVATE_REASONING_CANARY" not in (request.output_dir / "trace.json").read_text()


def _manifest(task, *, arm=SINGLE_SHOT_ARM_ID, configuration=None, provider=ModelProvider.OPENAI):
    evaluation = EvaluationTaskSpec(
        task=task,
        expected_patch="PRIVATE_GOLD",
        gold_files=("calculator.py",),
        split=DatasetSplit.DEV,
    )
    manifest = build_evaluation_run_manifest(
        experiment_id="single-shot-test",
        run_id=uuid5(RUN_ID, arm),
        created_at=STARTED_AT,
        source_commit="b" * 40,
        dependency_lock_sha256="c" * 64,
        dataset_manifest_sha256="d" * 64,
        configuration=configuration or {"baseline_version": "single-shot-v1"},
        split=DatasetSplit.DEV,
        arm_id=arm,
        tasks=(evaluation,),
        memory_namespace=None,
        model_provider=provider,
        model=MODEL if provider is ModelProvider.OPENAI else "fake-deterministic",
        prompt_versions=(SINGLE_SHOT_PROMPT_VERSION,)
        if provider is ModelProvider.OPENAI
        else ("fake-v1",),
        sandbox_image_reference=IMAGE,
    )
    return manifest, evaluation


def test_standard_evaluation_runner_reopens_independent_trace(fixture, tmp_path):
    executor, request, _, _ = fixture
    manifest, task = _manifest(request.task)
    executor = replace(executor, evaluation_configuration_sha256=manifest.configuration_sha256)
    report = run_evaluation(
        manifest=manifest,
        tasks=(task,),
        output_dir=tmp_path / "run",
        executor=executor,
    )
    assert report.completed_attempts == 1 and report.failed_attempts == 0
    assert inspect_evaluation_run(tmp_path / "run") == report


def test_hybrid_loader_uses_initial_state_only_and_verifies_completed_evidence(
    fixture, tmp_path, monkeypatch
):
    _, request, _, _ = fixture
    task = request.task.model_copy(
        update={
            "issue_body": "sk-fixtureSECRET01234567890 "
            + "long issue text " * 2000
            + "CLIPPED_ISSUE_TAIL",
        }
    )
    raw_summary = "sk-fixtureSECRET01234567890 " + "repository text " * 700
    raw_hit = RetrievalHit(
        chunk_id="fixture_chunk",
        sparse_rank=1,
        sparse_score=1,
        query="Find increment",
        retrieval_reason="source content " * 2000,
    )
    monkeypatch.setattr(
        "repopilot.repair_workflow.build_repair_retrieval_context",
        lambda *a, **k: RepairRetrievalContext(
            repository_id="fixture",
            repository_content_sha256="e" * 64,
            repository_summary=raw_summary,
            hits=(raw_hit,),
            indexing_latency_ms=0,
            retrieval_latency_ms=0,
        ),
    )
    manifest, evaluation = _manifest(task, arm="hybrid-rrf-v1", provider=ModelProvider.FAKE)

    def responses(task, run_id):
        return (
            ProviderResponse(
                request_id=repair_decision_request_id(run_id, 0),
                action=FinishAction(summary="FINAL_STATE_CANARY_NOT_FOR_BASELINE"),
                usage=ProviderUsageDelta(
                    input_tokens=1,
                    output_tokens=1,
                    total_tokens=2,
                    estimated_cost_usd=Decimal(0),
                    latency_ms=1,
                ),
                prompt_version="fake-v1",
            ),
        )

    # This source run is itself offline; no model usage or evaluator gold is required.
    def source_sandbox(root):
        return IsolatedTestSandbox(
            task=task,
            regression_patch=REGRESSION_PATCH,
            artifact_root=root / "artifacts",
            agent_factory=lambda path: _TrustedFixtureSandbox(path, tmp_path / "source-agent"),
            evaluator_factory=lambda path: _TrustedFixtureSandbox(path, tmp_path / "source-eval"),
        )

    source_executor = RepairWorkflowEvaluationExecutor(
        settings=RepoPilotSettings(),
        sandbox_image_reference=IMAGE,
        context_mode=RepairContextMode.NONE,
        evaluation_configuration_sha256=manifest.configuration_sha256,
        sandbox_factory=source_sandbox,
        provider_responses_factory=responses,
    )
    run_dir = tmp_path / "hybrid"
    report = run_evaluation(
        manifest=manifest, tasks=(evaluation,), output_dir=run_dir, executor=source_executor
    )
    assert report.completed_attempts == 1
    context = load_hybrid_initial_context(
        run_dir,
        task,
        expected_run_id=manifest.run_id,
        expected_source_commit=manifest.source_commit,
        expected_configuration_sha256=manifest.configuration_sha256,
    )
    assert "FINAL_STATE_CANARY" not in context.model_dump_json()
    assert "PRIVATE_GOLD" not in context.model_dump_json()
    assert "test_commands" not in context.model_dump_json()
    trace_path = next(run_dir.glob("executions/*/*/trace.json"))
    source_trace = AgentTrace.model_validate_json(trace_path.read_bytes())
    first = next(e.provider_request for e in source_trace.events if e.provider_request is not None)
    rendered = build_agent_prompt(first, max_chars=first.context_window.max_prompt_chars)
    projected = _parse_rendered_payload(rendered)
    assert context.issue_body == projected["task"]["issue_body"]
    assert context.repository_summary == projected["state"]["repository_summary"]
    assert list(context.retrieved_context) == projected["state"]["retrieved_context"]
    assert context.issue_body != task.issue_body
    assert context.repository_summary != raw_summary
    assert "CLIPPED_ISSUE_TAIL" not in context.model_dump_json()
    assert "sk-fixtureSECRET01234567890" not in context.model_dump_json()
    assert list(context.retrieved_context) != [raw_hit.model_dump(mode="json")]
    assert context.raw_issue_sha256 == _issue_digest(task)
    assert context.initial_context_sha256 == _digest(context.model_input())
    with pytest.raises(ValueError, match="provenance"):
        load_hybrid_initial_context(run_dir, task, expected_run_id=RUN_ID)
    trace = next(run_dir.glob("executions/*/*/trace.json"))
    trace.write_bytes(trace.read_bytes() + b" ")
    with pytest.raises(EvaluationRunnerError):
        load_hybrid_initial_context(run_dir, task)
