"""Build the frozen Agent-only P7 holdout input projection without Provider calls.

This is an evaluator-side, one-time projection command. The paid holdout runner imports
the strict artifact model from this module but never invokes ``main`` and never opens the
source-row snapshot that also contains evaluator Gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, ValidationError, model_validator
from pydantic_core import to_jsonable_python

from repopilot.enums import DatasetSplit
from repopilot.evals import evaluation_task_set_sha256, load_experiment_matrix
from repopilot.evals.dataset import FrozenDatasetManifest, load_frozen_dataset
from repopilot.schemas import Identifier, Sha256, StrictModel, TaskId, TaskSpec
from scripts.p7_holdout_control import (
    ENVIRONMENT_PATH,
    EXPECTED_TASK_IDS,
    MATRIX_PATH,
    HoldoutEnvironmentManifest,
    holdout_command_contract_sha256,
    project_holdout_task_timeouts,
    write_control_artifact,
)

AGENT_INPUTS_PATH = Path("results/p7-holdout/agent-inputs-v1.json")
AGENT_INPUTS_VERSION: Literal["holdout-agent-inputs-v1"] = "holdout-agent-inputs-v1"


class HoldoutAgentInputs(StrictModel):
    """Gold-free task material consumed by the formal paid runner."""

    schema_version: Literal[1] = 1
    input_version: Literal["holdout-agent-inputs-v1"] = AGENT_INPUTS_VERSION
    experiment_id: Identifier
    matrix_sha256: Sha256
    dataset_manifest_raw_sha256: Sha256
    dataset_manifest_content_sha256: Sha256
    source_rows_raw_sha256: Sha256
    source_evaluator_task_set_sha256: Sha256
    runtime_command_contract_sha256: Sha256
    projected_at: AwareDatetime
    task_ids: tuple[TaskId, ...]
    tasks: tuple[TaskSpec, ...]
    projection_scope: Literal["agent_visible_fields_only"] = "agent_visible_fields_only"
    source_rows_loaded_by_formal_runner: Literal[False] = False
    evaluator_gold_serialized: Literal[False] = False
    provider_requests_made: Literal[0] = 0
    paid_requests_made: Literal[0] = 0
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        if self.task_ids != EXPECTED_TASK_IDS:
            raise ValueError("holdout Agent input task IDs changed")
        if tuple(task.task_id for task in self.tasks) != self.task_ids:
            raise ValueError("holdout Agent tasks do not match declared IDs")
        if any(
            task.repository_url is None or task.repository_path is not None for task in self.tasks
        ):
            raise ValueError("committed Agent inputs must use frozen repository URLs")
        if holdout_command_contract_sha256(self.tasks) != self.runtime_command_contract_sha256:
            raise ValueError("holdout Agent input command contract changed")
        _validate_content_digest(self)
        return self


def build_holdout_agent_inputs(*, root: Path, projected_at: datetime) -> HoldoutAgentInputs:
    """Project the already-frozen evaluator bundle into a Gold-free runtime artifact."""

    matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    dataset_path = root / matrix.dataset.manifest_path
    dataset_raw = dataset_path.read_bytes()
    manifest = FrozenDatasetManifest.model_validate_json(dataset_raw)
    bundle = load_frozen_dataset(dataset_path, repository_root=root)
    environment = HoldoutEnvironmentManifest.model_validate_json(
        (root / ENVIRONMENT_PATH).read_bytes()
    )
    evaluations = tuple(bundle.external_evaluation_tasks)
    declared = next(item for item in matrix.dataset.task_sets if item.split is DatasetSplit.HOLDOUT)
    if (
        tuple(item.task.task_id for item in evaluations) != EXPECTED_TASK_IDS
        or evaluation_task_set_sha256(evaluations) != declared.task_set_sha256
        or manifest.content_sha256 != matrix.dataset.manifest_content_sha256
        or environment.matrix_sha256 != matrix.content_sha256
    ):
        raise ValueError("frozen evaluator task identity changed")
    tasks = tuple(project_holdout_task_timeouts(item.task) for item in evaluations)
    if holdout_command_contract_sha256(tasks) != environment.runtime_command_contract_sha256:
        raise ValueError("runtime task projection does not match accepted environment")
    payload: dict[str, object] = {
        "schema_version": 1,
        "input_version": AGENT_INPUTS_VERSION,
        "experiment_id": matrix.experiment_id,
        "matrix_sha256": matrix.content_sha256,
        "dataset_manifest_raw_sha256": hashlib.sha256(dataset_raw).hexdigest(),
        "dataset_manifest_content_sha256": manifest.content_sha256,
        "source_rows_raw_sha256": manifest.source_rows_ref.sha256,
        "source_evaluator_task_set_sha256": declared.task_set_sha256,
        "runtime_command_contract_sha256": environment.runtime_command_contract_sha256,
        "projected_at": projected_at,
        "task_ids": EXPECTED_TASK_IDS,
        "tasks": tasks,
        "projection_scope": "agent_visible_fields_only",
        "source_rows_loaded_by_formal_runner": False,
        "evaluator_gold_serialized": False,
        "provider_requests_made": 0,
        "paid_requests_made": 0,
    }
    return HoldoutAgentInputs.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve(strict=True)
    if (root / AGENT_INPUTS_PATH).exists():
        try:
            existing = HoldoutAgentInputs.model_validate_json(
                (root / AGENT_INPUTS_PATH).read_bytes()
            )
        except (OSError, ValidationError):
            raise SystemExit("existing_holdout_agent_inputs_invalid") from None
        print(
            "holdout_agent_inputs_valid=true "
            f"tasks={len(existing.tasks)} gold_serialized=false paid_requests=0"
        )
        return 0
    artifact = build_holdout_agent_inputs(root=root, projected_at=datetime.now(UTC))
    serialized = artifact.model_dump_json()
    forbidden = ('"expected_patch"', '"gold_files"', '"gold_patch"', '"test_patch"')
    if any(token in serialized for token in forbidden):
        raise SystemExit("holdout_agent_inputs_gold_boundary_violation")
    write_control_artifact(root / AGENT_INPUTS_PATH, artifact)
    print(
        "holdout_agent_inputs_created=true tasks=8 gold_serialized=false "
        "provider_requests=0 paid_requests=0"
    )
    return 0


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_content_digest(value: StrictModel) -> None:
    payload = value.model_dump(mode="python", exclude={"content_sha256"})
    if _canonical_sha256(payload) != getattr(value, "content_sha256", None):
        raise ValueError("holdout Agent input content digest does not match")


if __name__ == "__main__":
    raise SystemExit(main())
