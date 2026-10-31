"""Project regression tests for an isolated evaluator, never solution patches.

Only the explicit projection command reads the evaluator-owned dataset. The paid
controller uses ``load_test_inputs`` and never opens the source-row snapshot. This
artifact is not Agent context and is not permission to execute a paid evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from repopilot.schemas import GitSha, Sha256, StrictModel, TaskId
from scripts.p7_holdout_agent_inputs import (
    AGENT_INPUTS_PATH,
    HoldoutAgentInputs,
    _canonical_sha256,
    _validate_content_digest,
)
from scripts.p7_holdout_control import EXPECTED_TASK_IDS, MATRIX_PATH, write_control_artifact

TEST_INPUTS_PATH = Path("results/p7-holdout/regression-tests-v1.json")
TEST_PROTOCOL_VERSION: Literal["isolated-regression-tests-v1"] = "isolated-regression-tests-v1"
MAX_TEST_INPUT_BYTES = 4_000_000
# Frozen test-only projection, not a paid-execution authorization.
TEST_INPUTS_RAW_SHA256 = "eeff30648f37afe50e512e48a7c828b6eb8149652161c266d55f81e46818fb4e"


class RegressionTestInput(StrictModel):
    """One test-only patch bound to a clean base revision."""

    task_id: TaskId
    base_commit: GitSha
    regression_patch: str = Field(min_length=1, max_length=1_000_000, repr=False)
    regression_patch_sha256: Sha256

    @model_validator(mode="after")
    def validate_patch(self) -> Self:
        if (
            hashlib.sha256(self.regression_patch.encode()).hexdigest()
            != self.regression_patch_sha256
        ):
            raise ValueError("regression patch digest mismatch")
        return self


class HoldoutTestInputs(StrictModel):
    """Evaluator-private input projection with no solution/expected patch fields."""

    schema_version: Literal[1] = 1
    test_protocol_version: Literal["isolated-regression-tests-v1"] = TEST_PROTOCOL_VERSION
    agent_inputs_content_sha256: Sha256
    source_rows_raw_sha256: Sha256
    dataset_manifest_content_sha256: Sha256
    tasks: tuple[RegressionTestInput, ...]
    solution_patch_serialized: Literal[False] = False
    agent_context_allowed: Literal[False] = False
    paid_requests_made: Literal[0] = 0
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if tuple(item.task_id for item in self.tasks) != EXPECTED_TASK_IDS:
            raise ValueError("regression task set mismatch")
        _validate_content_digest(self)
        return self


def build_test_inputs(root: Path) -> HoldoutTestInputs:
    """Evaluator-only generation, separate from the formal runtime loader."""

    from repopilot.evals import load_experiment_matrix
    from repopilot.evals.dataset import SweBenchSourceRow, load_frozen_dataset

    matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    bundle = load_frozen_dataset(root / matrix.dataset.manifest_path, repository_root=root)
    agent_inputs = HoldoutAgentInputs.model_validate_json((root / AGENT_INPUTS_PATH).read_bytes())
    source_raw = (root / bundle.manifest.source_rows_ref.relative_path).read_bytes()
    if (
        hashlib.sha256(source_raw).hexdigest() != agent_inputs.source_rows_raw_sha256
        or bundle.manifest.content_sha256 != agent_inputs.dataset_manifest_content_sha256
        or matrix.content_sha256 != agent_inputs.matrix_sha256
    ):
        raise ValueError("regression projection source mismatch")
    rows = TypeAdapter(tuple[SweBenchSourceRow, ...]).validate_json(source_raw)
    tasks = tuple(
        RegressionTestInput(
            task_id=row.instance_id,
            base_commit=row.base_commit,
            regression_patch=row.test_patch,
            regression_patch_sha256=hashlib.sha256(row.test_patch.encode()).hexdigest(),
        )
        for row in rows
    )
    if tuple((item.task_id, item.base_commit) for item in tasks) != tuple(
        (item.task_id, item.base_commit) for item in agent_inputs.tasks
    ):
        raise ValueError("regression projection task mismatch")
    payload = {
        "schema_version": 1,
        "test_protocol_version": TEST_PROTOCOL_VERSION,
        "agent_inputs_content_sha256": agent_inputs.content_sha256,
        "source_rows_raw_sha256": agent_inputs.source_rows_raw_sha256,
        "dataset_manifest_content_sha256": agent_inputs.dataset_manifest_content_sha256,
        "tasks": tasks,
        "solution_patch_serialized": False,
        "agent_context_allowed": False,
        "paid_requests_made": 0,
    }
    return HoldoutTestInputs.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def load_test_inputs(
    root: Path, *, agent_inputs: HoldoutAgentInputs, expected_raw_sha256: str
) -> HoldoutTestInputs:
    """Load only test projection bytes and require an independently frozen raw digest."""

    path = root / TEST_INPUTS_PATH
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError("unsafe regression projection path")
    with path.open("rb") as stream:
        raw = stream.read(MAX_TEST_INPUT_BYTES + 1)
    if len(raw) > MAX_TEST_INPUT_BYTES or hashlib.sha256(raw).hexdigest() != expected_raw_sha256:
        raise ValueError("regression projection raw digest mismatch")
    artifact = HoldoutTestInputs.model_validate_json(raw)
    if (
        artifact.agent_inputs_content_sha256 != agent_inputs.content_sha256
        or artifact.source_rows_raw_sha256 != agent_inputs.source_rows_raw_sha256
        or artifact.dataset_manifest_content_sha256 != agent_inputs.dataset_manifest_content_sha256
        or tuple((item.task_id, item.base_commit) for item in artifact.tasks)
        != tuple((item.task_id, item.base_commit) for item in agent_inputs.tasks)
    ):
        raise ValueError("regression projection binding mismatch")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    artifact = build_test_inputs(root)
    path = root / TEST_INPUTS_PATH
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink()
            or HoldoutTestInputs.model_validate_json(path.read_bytes()) != artifact
        ):
            raise ValueError("regression projection output conflict")
    else:
        write_control_artifact(path, artifact)
    if hashlib.sha256(path.read_bytes()).hexdigest() != TEST_INPUTS_RAW_SHA256:
        raise ValueError("regression projection raw digest mismatch")
    print(
        "holdout_test_inputs_valid=true tasks=8 solution_patch_serialized=false "
        "agent_context_allowed=false paid_requests=0 "
        f"raw_sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
