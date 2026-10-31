"""Exact v2 consent and shared zero-paid identity gates for the repaired Holdout.

The caller must have direct user approval before building authorization. Neither an
authorization nor a readiness model executes a task. All legacy v1 controls remain
separate; the qualified runtime and current execution controller are different facts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, model_validator

from repopilot.embeddings import embedding_identity_from_settings
from repopilot.evals import load_experiment_matrix, select_experiment_arm
from repopilot.schemas import GitSha, Identifier, Sha256, StrictModel
from scripts import p7_holdout_readiness as legacy
from scripts.p7_holdout_agent_inputs import (
    AGENT_INPUTS_PATH,
    HoldoutAgentInputs,
    _validate_content_digest,
)
from scripts.p7_holdout_control import (
    ENVIRONMENT_PATH,
    EXPECTED_TASK_IDS,
    MATRIX_PATH,
    HoldoutEnvironmentManifest,
)
from scripts.p7_holdout_requalification import (
    CONTROLLER_PATH,
    PLAN_V2_PATH,
    POST_REPAIR_FREEZE_PATH,
    HoldoutApplicationPlanV2,
    PostRepairFreeze,
    _git,
    _read,
    _sealed,
    _sha,
    load_requalification,
)

AUTHORIZATION_V2_PATH = Path("results/p7-holdout/authorization-v2.json")
READINESS_V2_PATH = Path("results/p7-holdout/runtime-readiness-v2.json")
APPROVED_PLAN_ID = "p7-holdout-v2-d26d33699d59b6fc"
APPROVED_PLAN_SHA256 = "3660cc5962ab8810b9b74fab1c4820188340111d7798e0ef4c620f8d1aafb3c0"
APPROVED_PLAN_RAW_SHA256 = "b0654760cea101d8d1b295ca1c2c56e05e924a88f8d70c730e326706414253a2"
APPROVED_FREEZE_RAW_SHA256 = "19bfbaae0c7ac122b96aa0be5385510787a63a9a70de05cee581f77441fc9ef8"
READINESS_CHECKS = (
    "authorization",
    "checkouts",
    "controller",
    "embedding",
    "images",
    "local-dependencies",
    "output-paths",
    "provider-presence",
    "qualified-runtime",
    "vector-backend",
)
REQUIRED_EXECUTION_PATHS = (
    "scripts/__init__.py",
    "scripts/p7_holdout_v2.py",
    CONTROLLER_PATH,
    "scripts/p7_holdout_readiness_v2.py",
    "scripts/p7_holdout_runner_v2.py",
    "scripts/p7_holdout_readiness.py",
    "scripts/p7_holdout_runner.py",
    "scripts/p7_embedding_readiness_probe.py",
)


class PaidHoldoutAuthorizationV2(StrictModel):
    """One human approval for an immutable plan, not a reusable budget allowance."""

    authorization_version: Literal["holdout-authorization-v2"] = "holdout-authorization-v2"
    authorization_id: Literal["p7-s6-v2-usd32-ethan"] = "p7-s6-v2-usd32-ethan"
    approved_by: Literal["ethan"] = "ethan"
    approved_at: AwareDatetime
    experiment_id: Identifier
    plan_id: Identifier
    plan_sha256: Sha256
    plan_raw_sha256: Sha256
    freeze_marker_sha256: Sha256
    freeze_raw_sha256: Sha256
    matrix_sha256: Sha256
    max_total_cost_usd: Decimal
    paid_execution_approved: Literal[True]
    final_in_flight_response_overshoot_acknowledged: Literal[True]
    readiness_required_before_paid: Literal[True]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_authorization(self) -> Self:
        if (
            self.plan_id != APPROVED_PLAN_ID
            or self.plan_sha256 != APPROVED_PLAN_SHA256
            or self.plan_raw_sha256 != APPROVED_PLAN_RAW_SHA256
            or self.freeze_raw_sha256 != APPROVED_FREEZE_RAW_SHA256
            or self.max_total_cost_usd != Decimal("32.00")
        ):
            raise ValueError("v2 authorization does not match human approval")
        _validate_content_digest(self)
        return self


@dataclass(frozen=True)
class V2Controls:
    freeze: PostRepairFreeze
    plan: HoldoutApplicationPlanV2
    authorization: PaidHoldoutAuthorizationV2
    environment: HoldoutEnvironmentManifest
    agent_inputs: HoldoutAgentInputs
    controller_commit: str
    freeze_raw: bytes
    plan_raw: bytes
    authorization_raw: bytes
    environment_raw: bytes
    agent_inputs_raw: bytes


class HoldoutRuntimeReadinessV2(StrictModel):
    """Fresh replacement-VM checks; never a reused historical v1 readiness claim."""

    schema_version: Literal[2] = 2
    readiness_version: Literal["holdout-runtime-readiness-v2"] = "holdout-runtime-readiness-v2"
    experiment_id: Identifier
    controller_commit: GitSha
    checked_at: AwareDatetime
    host_platform: Literal["linux/amd64"] = "linux/amd64"
    plan_raw_sha256: Sha256
    plan_content_sha256: Sha256
    authorization_raw_sha256: Sha256
    authorization_content_sha256: Sha256
    environment_raw_sha256: Sha256
    environment_content_sha256: Sha256
    freeze_raw_sha256: Sha256
    freeze_content_sha256: Sha256
    agent_inputs_raw_sha256: Sha256
    test_inputs_raw_sha256: Sha256
    canary_raw_sha256: Sha256
    qualified_runtime_commit: GitSha
    semantic_objects_sha256: Sha256
    execution_controller_objects: dict[str, GitSha]
    allowed_secret_environment: tuple[str, ...] = legacy.ALLOWED_SECRET_ENVIRONMENT
    provider: legacy.ProviderCredentialEvidence
    embedding: legacy.EmbeddingReadinessEvidence
    vector_backend: legacy.VectorReadinessEvidence
    images: tuple[legacy.ReadinessImageEvidence, ...]
    completed_checks: tuple[str, ...]
    paid_requests_made: Literal[0] = 0
    gold_visible_to_agent: Literal[False] = False
    ready_for_paid_holdout: Literal[True] = True
    formal_blockers: tuple[()] = ()
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_readiness(self) -> Self:
        if (
            self.completed_checks != READINESS_CHECKS
            or self.allowed_secret_environment != legacy.ALLOWED_SECRET_ENVIRONMENT
            or tuple(image.task_id for image in self.images) != EXPECTED_TASK_IDS
            or tuple(self.execution_controller_objects)
            != tuple(sorted(self.execution_controller_objects))
            or not set(REQUIRED_EXECUTION_PATHS).issubset(self.execution_controller_objects)
        ):
            raise ValueError("v2 readiness checks or source inventory incomplete")
        _validate_content_digest(self)
        return self


def build_authorization(
    root: Path,
    *,
    approved_at: datetime,
    max_total_cost_usd: Decimal,
    overshoot_acknowledged: bool,
    readiness_required_before_paid: bool,
) -> PaidHoldoutAuthorizationV2:
    """Create only from the explicit approval received in the current conversation."""

    freeze, plan = load_requalification(root)
    payload: dict[str, object] = {
        "approved_at": approved_at,
        "experiment_id": plan.experiment_id,
        "plan_id": plan.plan_id,
        "plan_sha256": plan.content_sha256,
        "plan_raw_sha256": _sha(_read(root, PLAN_V2_PATH)),
        "freeze_marker_sha256": freeze.content_sha256,
        "freeze_raw_sha256": _sha(_read(root, POST_REPAIR_FREEZE_PATH)),
        "matrix_sha256": plan.matrix_sha256,
        "max_total_cost_usd": max_total_cost_usd,
        "paid_execution_approved": True,
        "final_in_flight_response_overshoot_acknowledged": overshoot_acknowledged,
        "readiness_required_before_paid": readiness_required_before_paid,
    }
    return _sealed(PaidHoldoutAuthorizationV2, payload)


def load_controls(root: Path) -> V2Controls:
    freeze, plan = load_requalification(root)
    plan_raw = _read(root, PLAN_V2_PATH)
    freeze_raw = _read(root, POST_REPAIR_FREEZE_PATH)
    authorization_raw = _read(root, AUTHORIZATION_V2_PATH)
    environment_raw = _read(root, ENVIRONMENT_PATH)
    agent_raw = _read(root, AGENT_INPUTS_PATH)
    authorization = PaidHoldoutAuthorizationV2.model_validate_json(authorization_raw)
    environment = HoldoutEnvironmentManifest.model_validate_json(environment_raw)
    agent = HoldoutAgentInputs.model_validate_json(agent_raw)
    if (
        plan.plan_id != APPROVED_PLAN_ID
        or plan.content_sha256 != APPROVED_PLAN_SHA256
        or authorization.experiment_id != plan.experiment_id
        or authorization.plan_sha256 != plan.content_sha256
        or authorization.plan_raw_sha256 != _sha(plan_raw)
        or authorization.freeze_marker_sha256 != freeze.content_sha256
        or authorization.freeze_raw_sha256 != _sha(freeze_raw)
        or authorization.matrix_sha256 != plan.matrix_sha256
        or authorization.approved_at < plan.created_at
        or _sha(environment_raw) != plan.environment_raw_sha256
        or _sha(agent_raw) != plan.agent_inputs_raw_sha256
    ):
        raise ValueError("v2 authorization or source binding mismatch")
    return V2Controls(
        freeze,
        plan,
        authorization,
        environment,
        agent,
        _git(root, "rev-parse", "HEAD"),
        freeze_raw,
        plan_raw,
        authorization_raw,
        environment_raw,
        agent_raw,
    )


def verify_runtime_environment() -> None:
    import os

    legacy._verify_host_and_environment()
    if any(
        name in os.environ
        for name in (
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "OPENAI_ORG_ID",
            "OPENAI_PROJECT_ID",
        )
    ):
        raise ValueError("unapproved_provider_environment_override")


def verify_execution_controller(root: Path) -> dict[str, str]:
    """Bind the whole scripts directory, including indirect probe/helper imports."""

    if _git(root, "status", "--porcelain=v1", "--untracked-files=all", "--", "scripts"):
        raise ValueError("execution controller source unclean")
    objects: dict[str, str] = {}
    for entry in _git(root, "ls-tree", "-rz", "HEAD", "--", "scripts").split("\0"):
        if not entry:
            continue
        metadata, relative = entry.split("\t", 1)
        mode, kind, blob = metadata.split()
        if mode not in ("100644", "100755") or kind != "blob":
            raise ValueError("unsupported execution source entry")
        raw = _read(root, Path(relative))
        if (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw, usedforsecurity=False).hexdigest()
            != blob
        ):
            raise ValueError("executed controller differs from committed bytes")
        objects[relative] = blob
    if not set(REQUIRED_EXECUTION_PATHS).issubset(objects):
        raise ValueError("execution controller incomplete")
    return dict(sorted(objects.items()))


def _readiness_bindings(controls: V2Controls) -> dict[str, object]:
    return {
        "experiment_id": controls.plan.experiment_id,
        "plan_raw_sha256": _sha(controls.plan_raw),
        "plan_content_sha256": controls.plan.content_sha256,
        "authorization_raw_sha256": _sha(controls.authorization_raw),
        "authorization_content_sha256": controls.authorization.content_sha256,
        "environment_raw_sha256": _sha(controls.environment_raw),
        "environment_content_sha256": controls.environment.content_sha256,
        "freeze_raw_sha256": _sha(controls.freeze_raw),
        "freeze_content_sha256": controls.freeze.content_sha256,
        "agent_inputs_raw_sha256": _sha(controls.agent_inputs_raw),
        "test_inputs_raw_sha256": controls.freeze.test_inputs_raw_sha256,
        "canary_raw_sha256": controls.freeze.canary_raw_sha256,
        "qualified_runtime_commit": controls.freeze.qualified_runtime_commit,
        "semantic_objects_sha256": controls.freeze.semantic_objects_sha256,
    }


def build_readiness(
    controls: V2Controls,
    *,
    checked_at: datetime,
    execution_controller_objects: dict[str, str],
    embedding: legacy.EmbeddingReadinessEvidence,
    vector_backend: legacy.VectorReadinessEvidence,
    images: tuple[legacy.ReadinessImageEvidence, ...],
) -> HoldoutRuntimeReadinessV2:
    payload = {
        **_readiness_bindings(controls),
        "controller_commit": controls.controller_commit,
        "checked_at": checked_at,
        "execution_controller_objects": execution_controller_objects,
        "provider": legacy.ProviderCredentialEvidence(),
        "embedding": embedding,
        "vector_backend": vector_backend,
        "images": images,
        "completed_checks": READINESS_CHECKS,
    }
    return _sealed(HoldoutRuntimeReadinessV2, payload)


def validate_readiness_bindings(
    root: Path,
    controls: V2Controls,
    readiness: HoldoutRuntimeReadinessV2,
) -> None:
    if any(
        getattr(readiness, key) != value for key, value in _readiness_bindings(controls).items()
    ):
        raise ValueError("v2 readiness binding mismatch")
    _git(
        root,
        "merge-base",
        "--is-ancestor",
        controls.plan.controller_commit,
        readiness.controller_commit,
    )
    _git(
        root, "merge-base", "--is-ancestor", readiness.controller_commit, controls.controller_commit
    )
    if readiness.execution_controller_objects != verify_execution_controller(root):
        raise ValueError("readiness execution controller drift")
    recorded = {
        path: _git(root, "rev-parse", f"{readiness.controller_commit}:{path}")
        for path in readiness.execution_controller_objects
    }
    if recorded != readiness.execution_controller_objects:
        raise ValueError("readiness recorded controller mismatch")
    if readiness.checked_at < controls.authorization.approved_at:
        raise ValueError("readiness predates authorization")
    if tuple(
        (image.task_id, image.registry_digest_reference) for image in readiness.images
    ) != tuple(
        (binding.task_id, binding.sandbox_image_reference)
        for binding in controls.environment.bindings
    ):
        raise ValueError("readiness image binding mismatch")
    matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    settings = select_experiment_arm(matrix, "dense-only-v1").settings(
        openai_api_key="readiness-identity-placeholder",
        database_url="postgresql://repopilot:repopilot@localhost:55433/repopilot",
    )
    identity = embedding_identity_from_settings(settings)
    if (
        readiness.embedding.identity_sha256 != identity.fingerprint
        or readiness.embedding.model != identity.model
        or readiness.embedding.revision != identity.revision
        or readiness.embedding.code_revision != identity.code_revision
        or readiness.vector_backend.schema_name != settings.vector_schema
        or readiness.vector_backend.table_name != settings.vector_table
    ):
        raise ValueError("readiness embedding/vector identity mismatch")
