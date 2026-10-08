"""Run the authorized P7-S5 dev matrix without loading holdout evaluation data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.embeddings import (
    EmbeddingInput,
    EmbeddingRequest,
    embedding_identity_from_settings,
    embedding_provider_from_settings,
    embedding_request_id,
)
from repopilot.enums import EvaluationAttemptStatus, EvaluationRunnerErrorCode
from repopilot.evals import (
    CalibrationError,
    DevCalibrationPlan,
    DevCalibrationPreflight,
    DevCalibrationPreflightFacts,
    DevCalibrationReport,
    FileEvaluationRunStore,
    PaidCalibrationAuthorization,
    build_dev_calibration_plan,
    build_dev_calibration_preflight,
    build_evaluation_run_report,
    build_paid_calibration_authorization,
    load_dev_calibration_tasks,
    load_experiment_matrix,
    run_dev_calibration,
    select_experiment_arm,
    write_calibration_artifact,
)
from repopilot.sandbox import DockerCliClient, DockerClientError
from repopilot.schemas import StrictModel
from repopilot.storage import PostgresVectorStore, VectorStoreError

MATRIX_PATH = Path("configs/experiments/p7-v2.yaml")
FIRST_ATTEMPT_PLAN_PATH = Path("results/p7-dev/plan-v2.json")
PRIOR_PLAN_PATH = Path("results/p7-dev/plan-v4.json")
PRIOR_REPORT_PATH = Path("results/p7-dev/calibration-report-v4.json")
AUTHORIZATION_PATH = Path("results/p7-dev/authorization-v5.json")
PLAN_PATH = Path("results/p7-dev/plan-v5.json")
READY_PREFLIGHT_PATH = Path("results/p7-dev/preflight-ready-v5.json")
REPORT_PATH = Path("results/p7-dev/calibration-report-v5.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approve-application-cost-usd", required=True)
    arguments = parser.parse_args()

    root = arguments.repository_root.resolve(strict=True)
    approved_cost = _approved_cost(arguments.approve_application_cost_usd)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key.strip():
        raise SystemExit("OPENAI_API_KEY is unavailable")
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://repopilot:repopilot@localhost:55433/repopilot",
    )
    source_commit = _git(root, "rev-parse", "HEAD")
    if not _source_is_clean(root):
        raise SystemExit("source tree is not clean before calibration")

    matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    tasks = load_dev_calibration_tasks(matrix, repository_root=root)
    carried_cost = _verified_carried_cost(root, matrix_sha256=matrix.content_sha256)
    now = datetime.now(UTC)
    authorization = _load_or_create(
        root / AUTHORIZATION_PATH,
        PaidCalibrationAuthorization,
        lambda: build_paid_calibration_authorization(
            matrix=matrix,
            authorization_id="p7-dev-user-approved-v5",
            approved_by=arguments.approved_by,
            approved_at=now,
            max_total_cost_usd=approved_cost,
        ),
    )
    plan = _load_or_create(
        root / PLAN_PATH,
        DevCalibrationPlan,
        lambda: build_dev_calibration_plan(
            matrix=matrix,
            authorization=authorization,
            tasks=tasks,
            created_at=authorization.approved_at,
            source_commit=source_commit,
            source_dirty=False,
            carried_reported_cost_usd=carried_cost,
        ),
    )
    if plan.source_commit != source_commit:
        raise SystemExit("existing calibration plan belongs to another source commit")

    settings = select_experiment_arm(matrix, "dense-only-v1").settings(
        openai_api_key=api_key,
        database_url=database_url,
    )
    facts = DevCalibrationPreflightFacts(
        source_commit=source_commit,
        source_clean=True,
        local_dependencies_available=all(
            importlib.util.find_spec(module) is not None
            for module in ("openai", "psycopg", "sentence_transformers")
        ),
        sandbox_image_available=_sandbox_image_available(matrix.sandbox_image_reference),
        provider_credential_available=True,
        vector_backend_available=_vector_backend_available(settings),
        embedding_snapshot_available=_embedding_snapshot_available(settings),
        output_root_ready=True,
    )
    preflight = _load_or_create(
        root / READY_PREFLIGHT_PATH,
        DevCalibrationPreflight,
        lambda: build_dev_calibration_preflight(
            matrix=matrix,
            tasks=tasks,
            repository_root=root,
            checked_at=authorization.approved_at,
            facts=facts,
        ),
    )
    if not preflight.ready_for_paid_dev or preflight.source_dirty:
        raise SystemExit("dev calibration preflight is blocked")

    if (root / REPORT_PATH).exists():
        report = DevCalibrationReport.model_validate_json((root / REPORT_PATH).read_bytes())
    else:
        report = run_dev_calibration(
            plan=plan,
            matrix=matrix,
            authorization=authorization,
            preflight=preflight,
            tasks=tasks,
            repository_root=root,
            openai_api_key=api_key,
            database_url=database_url,
        )
        write_calibration_artifact(root / REPORT_PATH, report)
    print(
        f"calibration_id={report.calibration_id} arms={len(report.runs)} "
        f"attempts={sum(item.total_attempts for item in report.runs)} "
        f"reported_cost_usd={report.total_estimated_cost_usd} "
        f"application_cost_usd={carried_cost + report.total_estimated_cost_usd}"
    )
    return 0


def _approved_cost(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        raise SystemExit("approved application cost is invalid") from None
    if parsed != Decimal("8.00"):
        raise SystemExit("P7-S5 requires the exact approved USD 8.00 application plan")
    return parsed


def _verified_carried_cost(root: Path, *, matrix_sha256: str) -> Decimal:
    """Carry every verified prior dev cost without trusting a mutable summary alone."""

    if (root / PRIOR_PLAN_PATH).is_file() and (root / PRIOR_REPORT_PATH).is_file():
        try:
            prior_plan = DevCalibrationPlan.model_validate_json(
                (root / PRIOR_PLAN_PATH).read_bytes()
            )
            prior_report = DevCalibrationReport.model_validate_json(
                (root / PRIOR_REPORT_PATH).read_bytes()
            )
        except (OSError, ValidationError):
            raise SystemExit("prior calibration controls are unavailable or invalid") from None
        if (
            prior_plan.matrix_sha256 != matrix_sha256
            or prior_report.matrix_sha256 != matrix_sha256
            or prior_report.plan_sha256 != prior_plan.content_sha256
            or prior_report.holdout_accessed
        ):
            raise SystemExit("prior calibration controls do not match the dev retry")
        observed_cost = Decimal("0")
        for slot in prior_plan.run_slots:
            store = FileEvaluationRunStore(root / slot.output_path)
            manifest = store.load_manifest()
            events = store.read_events(manifest)
            store.verify_completed_history(manifest, events)
            report = build_evaluation_run_report(manifest, events)
            if report.pending_attempts or report.failed_attempts:
                raise SystemExit("prior calibration history is incomplete")
            observed_cost += sum(
                (
                    attempt.evidence.usage.estimated_cost_usd
                    for attempt in report.attempts
                    if attempt.evidence is not None
                ),
                Decimal("0"),
            )
        if observed_cost != prior_report.total_estimated_cost_usd:
            raise SystemExit("prior calibration cost does not match retained results")
        return prior_plan.carried_reported_cost_usd + observed_cost

    try:
        prior_plan = _load_legacy_plan(root / FIRST_ATTEMPT_PLAN_PATH)
    except (OSError, ValidationError, ValueError):
        raise SystemExit("prior calibration plan is unavailable or invalid") from None
    if prior_plan.matrix_sha256 != matrix_sha256 or prior_plan.holdout_accessed:
        raise SystemExit("prior calibration plan does not match the dev retry")

    carried = Decimal("0")
    recovered = 0
    for slot in prior_plan.run_slots:
        store = FileEvaluationRunStore(root / slot.output_path)
        manifest = store.load_manifest()
        if (
            manifest.run_id != slot.run_id
            or manifest.source_commit != prior_plan.source_commit
            or manifest.configuration_sha256 != slot.configuration_sha256
            or manifest.split.value != "dev"
        ):
            raise SystemExit("prior calibration run identity is invalid")
        events = store.read_events(manifest)
        report = build_evaluation_run_report(manifest, events)
        if report.pending_attempts or report.completed_attempts or not report.failed_attempts:
            raise SystemExit("prior calibration history is not the expected failed attempt")
        for attempt in report.attempts:
            if (
                attempt.status is not EvaluationAttemptStatus.FAILED
                or attempt.failure_code is not EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY
            ):
                raise SystemExit("prior calibration failure is not recoverable")
            terminal = next(
                event
                for event in reversed(events)
                if event.attempt_id == attempt.attempt_id
                and event.status is EvaluationAttemptStatus.FAILED
            )
            evidence = store.recover_execution_evidence(
                manifest=manifest,
                event=terminal,
            )
            carried += evidence.usage.estimated_cost_usd
            recovered += 1
    expected = len(prior_plan.task_ids) * len(prior_plan.run_slots) * prior_plan.repetitions
    if recovered != expected:
        raise SystemExit("prior calibration cost ledger is incomplete")
    return carried


def _load_legacy_plan(path: Path) -> DevCalibrationPlan:
    raw = json.loads(path.read_bytes())
    if not isinstance(raw, dict):
        raise ValueError
    recorded_digest = raw.pop("content_sha256", None)
    encoded = json.dumps(
        raw,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    if recorded_digest != hashlib.sha256(encoded).hexdigest():
        raise ValueError
    raw["carried_reported_cost_usd"] = "0"
    migrated = json.dumps(
        raw,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    raw["content_sha256"] = hashlib.sha256(migrated).hexdigest()
    return DevCalibrationPlan.model_validate(raw)


def _load_or_create[ModelT: StrictModel](
    path: Path,
    model_type: type[ModelT],
    factory: Callable[[], ModelT],
) -> ModelT:
    if path.exists():
        try:
            return model_type.model_validate_json(path.read_bytes())
        except (OSError, ValidationError):
            raise SystemExit("existing calibration control artifact is invalid") from None
    value = factory()
    write_calibration_artifact(path, value)
    return value


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_is_clean(root: Path) -> bool:
    status = _git(root, "status", "--porcelain=v1")
    if not status:
        return True
    return all(
        line[3:].startswith("results/p7-dev/") for line in status.splitlines() if len(line) > 3
    )


def _sandbox_image_available(reference: str) -> bool:
    try:
        DockerCliClient().resolve_image(reference)
    except DockerClientError:
        return False
    return True


def _vector_backend_available(settings: RepoPilotSettings) -> bool:
    try:
        PostgresVectorStore(settings).initialize()
    except VectorStoreError:
        return False
    return True


def _embedding_snapshot_available(settings: RepoPilotSettings) -> bool:
    try:
        identity = embedding_identity_from_settings(settings)
        content = "def calibration_probe() -> int:\n    return 1\n"
        item = EmbeddingInput(
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            content=content,
        )
        request = EmbeddingRequest(
            request_id=embedding_request_id(identity, (item,), 1),
            identity=identity,
            inputs=(item,),
            attempt=1,
        )
        response = embedding_provider_from_settings(settings).embed(request)
    except Exception:
        return False
    return len(response.vectors) == 1 and len(response.vectors[0].values) == 768


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CalibrationError as error:
        raise SystemExit(str(error)) from None
