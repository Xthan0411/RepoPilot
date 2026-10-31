"""Real ingestion/chunk/BM25 and fake-only dense/runtime workload qualification."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue, ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.embeddings import embedding_identity_from_settings
from repopilot.embeddings.base import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
)
from repopilot.enums import RepositoryIngestErrorCode
from repopilot.repair_workflow import repository_id_for_task
from repopilot.retrieval.manifest import RepositoryIngestError
from repopilot.schemas import TaskSpec
from repopilot.storage.memory import InMemoryVectorStore

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import p7_repository_workload_preflight as preflight  # noqa: E402
from scripts import p7_workload_runtime_probe as runtime  # noqa: E402
from scripts.p7_holdout_control import EXPECTED_TASK_IDS  # noqa: E402


class SyntheticEmbeddingProvider:
    """No model loading or requests; retain production identity and 768-D protocol."""

    def __init__(self, settings: RepoPilotSettings) -> None:
        self.identity = embedding_identity_from_settings(settings)
        self.calls = 0

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.calls += 1
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert "OPENAI_API_KEY" not in os.environ
        return EmbeddingResponse(
            request_id=request.request_id,
            identity=request.identity,
            vectors=tuple(
                EmbeddingVector(
                    content_hash=item.content_hash,
                    values=(1 / math.sqrt(self.identity.dimension),) * self.identity.dimension,
                )
                for item in request.inputs
            ),
        )


def _fixtures(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[preflight.WorkloadInputs, SyntheticEmbeddingProvider, list[str]]:
    inputs = preflight.load_inputs(PROJECT_ROOT)
    for binding in inputs.environment.bindings:
        checkout = root / binding.checkout_relative_path
        checkout.mkdir(parents=True)
        (checkout / "example.py").write_text("def function(value):\n    return value + 1\n")
        (checkout / "README.md").write_text("A test repository with function and value.\n")
    settings = preflight._settings(inputs, "dense-only-v1")
    provider = SyntheticEmbeddingProvider(settings)
    store = InMemoryVectorStore(dimension=768)
    runtime_tasks: list[str] = []

    def probe(**kwargs: object) -> dict[str, JsonValue]:
        task = cast(TaskSpec, kwargs["task"])
        assert task.repository_url is None
        assert Path(task.repository_path or "").is_relative_to(root)
        assert "OPENAI_API_KEY" not in os.environ
        runtime_tasks.append(task.task_id)
        return {"provider_requests_made": 0, "registered_tool_count": 7, "elapsed_ms": 1}

    monkeypatch.setattr(preflight, "load_inputs", lambda supplied_root: inputs)
    monkeypatch.setattr(
        preflight, "verify_source", lambda supplied_root: ("a" * 40, {"code": "b" * 40})
    )
    monkeypatch.setattr(preflight, "_checkouts", lambda *args: ())
    monkeypatch.setattr(preflight, "embedding_provider_from_settings", lambda settings: provider)
    monkeypatch.setattr(preflight, "PostgresVectorStore", lambda settings: store)
    monkeypatch.setattr(runtime, "probe_runtime", probe)
    return inputs, provider, runtime_tasks


def test_default_scans_all_eight_real_corpora_without_embedding_or_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, provider, runtime_tasks = _fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(
        preflight, "embedding_provider_from_settings", lambda *args: pytest.fail("embedding")
    )
    monkeypatch.setattr(preflight, "PostgresVectorStore", lambda *args: pytest.fail("database"))
    report = preflight.run_preflight(tmp_path)
    assert tuple(task.task_id for task in report.tasks) == EXPECTED_TASK_IDS
    assert report.scan_passed and not report.retrieval_preflight_passed
    assert not report.local_cache_database_writes_enabled
    assert provider.calls == 0 and runtime_tasks == []
    assert not report.agent_constructed and report.provider_requests_made == 0
    assert report.paid_requests_made == 0 and not report.formal_environment_ready
    for task, binding in zip(report.tasks, inputs.environment.bindings, strict=True):
        assert task.base_commit == binding.base_commit
        assert [stage.status for stage in task.stages[:3]] == ["passed"] * 3
        assert all(stage.code == "not_requested" for stage in task.stages[3:])
        assert task.stages[0].metrics["agent_tool_listing_limit"] == 2000
        assert task.stages[0].metrics["internal_scan_listing_limit"] == 20_000
        assert task.stages[1].metrics["chunk_count"] == 2


def test_preparation_exercises_production_dense_hybrid_index_and_readback_using_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, provider, runtime_tasks = _fixtures(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "secret-not-for-model-code")
    report = preflight.run_preflight(tmp_path, prepare_retrieval=True)
    assert report.scan_passed and report.retrieval_preflight_passed
    assert report.local_cache_database_writes_enabled
    assert tuple(runtime_tasks) == EXPECTED_TASK_IDS
    assert provider.calls > 0
    for task in report.tasks:
        assert all(stage.status == "passed" for stage in task.stages)
        for stage in task.stages[3:7]:
            assert stage.metrics["vector_readback_verified"] is True
            assert stage.metrics["vector_record_count"] == 2
        metrics = task.stages[-1].metrics
        assert metrics["warm_preparation_within_deadline"] is True
        assert metrics["task_budget_ms"] == 900_000
        assert metrics["remaining_budget_guarantees_agent_completion"] is False
        assert metrics["new_process_model_cold_load_included"] is False
    assert os.environ["OPENAI_API_KEY"] == "secret-not-for-model-code"
    assert "secret-not-for-model-code" not in report.model_dump_json()


def test_one_ingestion_failure_does_not_hide_other_seven_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixtures(tmp_path, monkeypatch)
    ingest = preflight._ingest_task

    def fail_one(task: TaskSpec, settings: RepoPilotSettings) -> object:
        if task.task_id == EXPECTED_TASK_IDS[0]:
            raise RepositoryIngestError(RepositoryIngestErrorCode.LISTING_TRUNCATED)
        return ingest(task, settings)

    monkeypatch.setattr(preflight, "_ingest_task", fail_one)
    report = preflight.run_preflight(tmp_path)
    assert not report.scan_passed and len(report.tasks) == 8
    failed = report.tasks[0]
    assert failed.stages[0].failure is not None
    assert failed.stages[0].failure.stage == "ingest"
    assert failed.stages[0].failure.cause_code == "listing_truncated"
    assert failed.stages[1].code == "upstream_failed"
    assert all(t.stages[2].status == "passed" for t in report.tasks[1:])


def test_missing_checkout_is_recorded_with_complete_task_stage_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(
        preflight, "_checkouts", lambda *args: (EXPECTED_TASK_IDS[0], EXPECTED_TASK_IDS[5])
    )
    report = preflight.run_preflight(tmp_path)
    assert len(report.tasks) == 8 and not report.scan_passed
    assert report.tasks[0].stages[0].code == "checkout_unavailable"
    assert report.tasks[5].stages[0].code == "checkout_unavailable"
    assert report.tasks[6].stages[2].status == "passed"


def test_truly_absent_checkout_does_not_abort_remaining_repository_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, _, _ = _fixtures(tmp_path, monkeypatch)
    missing = tmp_path / inputs.environment.bindings[0].checkout_relative_path
    for path in missing.iterdir():
        path.unlink()
    missing.rmdir()
    assert not missing.exists()
    monkeypatch.setattr(
        preflight,
        "_checkouts",
        lambda root, environment: tuple(
            binding.task_id
            for binding in environment.bindings
            if not (root / binding.checkout_relative_path).is_dir()
        ),
    )

    report = preflight.run_preflight(tmp_path)

    assert len(report.tasks) == 8 and not report.scan_passed
    failed = report.tasks[0]
    assert failed.repository_id.startswith("unavailable_")
    assert failed.stages[0].code == "checkout_unavailable"
    assert tuple(stage.stage for stage in failed.stages) == preflight.STAGES
    assert all(stage.status == "blocked" for stage in failed.stages[1:])
    for evidence, task, binding in zip(
        report.tasks[1:],
        inputs.agent_inputs.tasks[1:],
        inputs.environment.bindings[1:],
        strict=True,
    ):
        local_task = task.model_copy(
            update={
                "repository_path": str(tmp_path / binding.checkout_relative_path),
                "repository_url": None,
            }
        )
        assert evidence.repository_id == repository_id_for_task(local_task)
        assert all(stage.status == "passed" for stage in evidence.stages[:3])


def test_retrieval_failure_is_safe_and_blocks_warm_retry_but_not_other_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixtures(tmp_path, monkeypatch)
    build = preflight._build_retrieval

    def failure(*args: object) -> object:
        task = cast(TaskSpec, args[1])
        if task.task_id == EXPECTED_TASK_IDS[0]:
            raise OSError("secret-postgresql-password: source issue text")
        return build(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(preflight, "_build_retrieval", failure)
    report = preflight.run_preflight(tmp_path, prepare_retrieval=True)
    assert report.scan_passed and not report.retrieval_preflight_passed
    assert report.tasks[0].stages[3].status == "failed"
    assert report.tasks[0].stages[5].code == "upstream_failed"
    assert all(stage.status == "passed" for task in report.tasks[1:] for stage in task.stages)
    assert "secret-postgresql-password" not in report.model_dump_json()
    assert "source issue text" not in report.model_dump_json()


def test_resource_construction_failure_is_reported_for_all_eight_without_losing_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixtures(tmp_path, monkeypatch)

    def unavailable(*args: object) -> None:
        raise OSError("secret-cache-path")

    monkeypatch.setattr(preflight, "embedding_provider_from_settings", unavailable)
    report = preflight.run_preflight(tmp_path, prepare_retrieval=True)
    assert report.scan_passed and not report.retrieval_preflight_passed
    assert len(report.tasks) == 8
    assert all(task.stages[3].status == "failed" for task in report.tasks)
    assert "secret-cache-path" not in report.model_dump_json()


def test_partial_coverage_cannot_be_sealed_as_valid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixtures(tmp_path, monkeypatch)
    report = preflight.run_preflight(tmp_path)
    payload = report.model_dump(mode="python")
    payload["tasks"] = tuple(task for task in report.tasks if "pytest" not in task.task_id)
    with pytest.raises(ValidationError, match="identity mismatch"):
        preflight.RepositoryWorkloadPreflight.model_validate(payload)


def test_cli_creates_private_scan_evidence_and_never_overwrites_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["probe", "--repository-root", str(tmp_path)])
    assert preflight.main() == 0
    output = tmp_path / preflight.SCAN_RESULT_PATH
    original = output.read_bytes()
    assert output.stat().st_mode & 0o777 == 0o600
    assert preflight.main() == 1
    assert output.read_bytes() == original
    assert not (tmp_path / preflight.PREPARED_RESULT_PATH).exists()


def test_scan_and_full_results_use_distinct_create_only_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["probe", "--repository-root", str(tmp_path)])
    assert preflight.main() == 0
    scan = (tmp_path / preflight.SCAN_RESULT_PATH).read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["probe", "--repository-root", str(tmp_path), "--prepare-retrieval"]
    )
    assert preflight.main() == 0
    assert (tmp_path / preflight.SCAN_RESULT_PATH).read_bytes() == scan
    assert (tmp_path / preflight.PREPARED_RESULT_PATH).stat().st_mode & 0o777 == 0o600


def test_source_byte_audit_rejects_hidden_assume_unchanged_edit(tmp_path: Path) -> None:
    script = tmp_path / "scripts/p7_repository_workload_preflight.py"
    script.parent.mkdir()
    script.write_text("# source snapshot\n")
    for args in (("init", "--quiet"), ("add", "."), ("commit", "--quiet", "-m", "fixture")):
        subprocess.run(
            ("git", "-c", "user.name=Unit", "-c", "user.email=unit@example.test", *args),
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
    controller, objects = preflight.verify_source(tmp_path)
    assert len(controller) == 40 and len(objects) == 1
    subprocess.run(
        ("git", "update-index", "--assume-unchanged", str(script.relative_to(tmp_path))),
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    script.write_text("# hidden changed bytes\n")
    with pytest.raises(ValueError, match="source_bytes_changed"):
        preflight.verify_source(tmp_path)


def test_inputs_are_bound_to_original_gold_free_projections() -> None:
    inputs = preflight.load_inputs(PROJECT_ROOT)
    assert inputs.agent_inputs.task_ids == EXPECTED_TASK_IDS
    assert not inputs.agent_inputs.evaluator_gold_serialized
    assert not inputs.tests.solution_patch_serialized
    assert tuple(item.task_id for item in inputs.environment.bindings) == EXPECTED_TASK_IDS
