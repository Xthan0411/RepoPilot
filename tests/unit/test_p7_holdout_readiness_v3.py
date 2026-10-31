"""GPU-wrapper readiness contracts without real models, Docker, DB, or Provider calls."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import JsonValue

from repopilot.config import DEFAULT_DATABASE_URL, RepoPilotSettings
from repopilot.embeddings import EmbeddingRequest, EmbeddingResponse, EmbeddingVector
from repopilot.evals import load_experiment_matrix, select_experiment_arm

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts import p7_holdout_readiness as legacy
from scripts import p7_holdout_readiness_v2 as v2
from scripts import p7_holdout_readiness_v3 as readiness
from scripts.p7_gpu_cache_acceptance import CacheAcceptanceError
from scripts.p7_gpu_cpu_probe import PINNED_IDENTITY, CpuProbeError
from scripts.p7_holdout_control import MATRIX_PATH
from scripts.p7_holdout_gpu import READINESS_V3_PATH, V3Controls

PROJECT_ROOT = Path(__file__).parents[2]


def _old() -> legacy.HoldoutRuntimeReadiness:
    return legacy.HoldoutRuntimeReadiness.model_validate_json(
        (PROJECT_ROOT / "results/p7-holdout/runtime-readiness-v1.json").read_bytes()
    )


def _observation() -> readiness.CpuReadinessObservation:
    return readiness.CpuReadinessObservation(
        embedding=_old().embedding.model_copy(update={"cache_preparation_network_used": False}),
        cpu_profile={
            "profile": "p7-cpu-fp32-batch1-threads8-v1",
            "device": "cpu",
            "dtype": "torch.float32",
            "batch_size": 1,
            "cpu_threads": 8,
        },
    )


def _settings() -> RepoPilotSettings:
    matrix = load_experiment_matrix(PROJECT_ROOT / MATRIX_PATH, repository_root=PROJECT_ROOT)
    return select_experiment_arm(matrix, "dense-only-v1").settings(
        openai_api_key="provider-not-used", database_url=DEFAULT_DATABASE_URL
    )


def _configured(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[V3Controls, list[str], list[dict[str, object]]]:
    controls = cast(
        V3Controls,
        SimpleNamespace(
            plan=SimpleNamespace(
                run_slots=tuple(SimpleNamespace(output_path=f"runs/slot-{i}") for i in range(32))
            ),
            environment=SimpleNamespace(),
        ),
    )
    old = _old()
    matrix = load_experiment_matrix(PROJECT_ROOT / MATRIX_PATH, repository_root=PROJECT_ROOT)
    calls: list[str] = []
    builds: list[dict[str, object]] = []
    report = {
        "status": "complete",
        "completed_vectors": 103236,
        "total_vectors": 103236,
        "job_sha256": "a" * 64,
        "inputs_sha256": "b" * 64,
        "complete_sha256": "c" * 64,
    }

    def cache(*args: object) -> object:
        calls.append("cache")
        return SimpleNamespace(report=report, identity=PINNED_IDENTITY)

    def cpu(*args: object) -> readiness.CpuReadinessObservation:
        calls.append("cpu")
        return _observation()

    def vector(*args: object) -> legacy.VectorReadinessEvidence:
        calls.append("vector")
        return old.vector_backend

    def metadata(*args: object) -> bool:
        calls.append("metadata")
        return True

    def build(**kwargs: object) -> legacy.ProviderCredentialEvidence:
        calls.append("build")
        builds.append(kwargs)
        return legacy.ProviderCredentialEvidence()

    monkeypatch.setenv("OPENAI_API_KEY", "secret-key-never-persist")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret-database@localhost/repopilot")
    monkeypatch.setattr(readiness, "verify_runtime_environment", lambda: None)
    monkeypatch.setattr(readiness, "load_controls", lambda root: controls)
    monkeypatch.setattr(readiness, "verify_execution_controller", lambda root: {"script": "a" * 40})
    monkeypatch.setattr(
        readiness,
        "_local_gates",
        lambda *args: (
            old.images,
            {
                "missing_image_task_ids": (),
                "invalid_checkout_task_ids": (),
                "conflicting_output_indices": (),
            },
        ),
    )
    monkeypatch.setattr(v2, "_missing_dependencies", lambda: ())
    monkeypatch.setattr(readiness, "load_experiment_matrix", lambda *args, **kwargs: matrix)
    monkeypatch.setattr(readiness, "verify_pinned_cache", cache)
    monkeypatch.setattr(readiness, "_probe_cpu", cpu)
    monkeypatch.setattr(legacy, "_probe_vector_backend", vector)
    monkeypatch.setattr(readiness, "_cache_metadata_unchanged", metadata)
    monkeypatch.setattr(readiness, "build_readiness", build)
    return controls, calls, builds


def test_success_verifies_cache_once_single_cpu_probe_and_private_create_only_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    controls, calls, builds = _configured(monkeypatch)
    assert readiness.qualify(tmp_path) == 0
    assert calls == ["cache", "cpu", "vector", "metadata", "build"]
    output = tmp_path / READINESS_V3_PATH
    assert output.stat().st_mode & 0o777 == 0o600
    assert len(builds) == 1 and builds[0]["controls"] is controls
    assert builds[0]["embedding"] == _observation().embedding
    assert builds[0]["cpu_profile"] == _observation().cpu_profile
    assert cast(dict[str, object], builds[0]["gpu_cache"])["total_vectors"] == 103236
    serialized = output.read_text() + capsys.readouterr().out
    assert "secret-key" not in serialized and "secret-database" not in serialized
    assert "cpu_probe_count=1" in serialized and "provider_requests=0" in serialized


def test_local_gate_checks_all_32_outputs_plus_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_gates = readiness._local_gates
    controls, _, _ = _configured(monkeypatch)
    monkeypatch.setattr(v2, "_resolve_images", lambda *args: (_old().images, ()))
    monkeypatch.setattr(v2, "_checkouts", lambda *args: ())
    seen: list[tuple[str, ...]] = []

    def conflicts(root: Path, paths: tuple[str, ...]) -> tuple[int, ...]:
        seen.append(paths)
        return (32,)

    monkeypatch.setattr(v2, "_output_conflicts", conflicts)
    _, diagnostics = local_gates(tmp_path, controls)
    assert len(seen[0]) == 33
    assert seen[0][0] == str(READINESS_V3_PATH)
    assert seen[0][-1] == "runs/slot-31"
    assert diagnostics["conflicting_output_indices"] == (32,)


def test_missing_requirements_merge_before_any_full_cache_or_model_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, calls, builds = _configured(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    monkeypatch.setattr(v2, "_missing_dependencies", lambda: ("torch", "psycopg"))
    monkeypatch.setattr(
        readiness,
        "_local_gates",
        lambda *args: (
            (),
            {
                "missing_image_task_ids": ("django__django-15902",),
                "invalid_checkout_task_ids": ("sympy__sympy-20590",),
                "conflicting_output_indices": (1,),
            },
        ),
    )
    assert readiness.qualify(tmp_path) == 1
    assert calls == [] and builds == []
    output = capsys.readouterr().out
    for code in (
        "sandbox_image_unavailable",
        "checkout_unavailable",
        "output_conflict",
        "local_dependency_unavailable",
        "provider_credential_required",
    ):
        assert code in output
    assert not (tmp_path / READINESS_V3_PATH).exists()


def test_bad_cache_blocks_model_and_database_without_raw_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, calls, _ = _configured(monkeypatch)

    def broken(*args: object) -> None:
        raise CacheAcceptanceError("secret-from-broken-cache")

    monkeypatch.setattr(readiness, "verify_pinned_cache", broken)
    assert readiness.qualify(tmp_path) == 1
    assert calls == []
    output = capsys.readouterr().out
    assert "gpu_cache_unavailable" in output and "secret-from-broken-cache" not in output


def test_missing_exact_authorization_blocks_even_local_asset_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, calls, builds = _configured(monkeypatch)

    def missing(root: Path) -> V3Controls:
        raise ValueError("authorization_required")

    monkeypatch.setattr(readiness, "load_controls", missing)
    monkeypatch.setattr(readiness, "_local_gates", lambda *args: pytest.fail("local gates called"))
    with pytest.raises(ValueError, match="authorization_required"):
        readiness.qualify(tmp_path)
    assert calls == [] and builds == []
    assert not (tmp_path / READINESS_V3_PATH).exists()


def test_cpu_and_vector_failures_are_both_reported_without_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, calls, _ = _configured(monkeypatch)

    def broken(*args: object) -> None:
        raise legacy.ReadinessError(legacy.ReadinessErrorCode.EMBEDDING_SNAPSHOT_UNAVAILABLE)

    monkeypatch.setattr(readiness, "_probe_cpu", broken)
    monkeypatch.setattr(legacy, "_probe_vector_backend", broken)
    assert readiness.qualify(tmp_path) == 1
    assert calls == ["cache"]
    output = capsys.readouterr().out
    assert "cpu_embedding_profile_unavailable" in output
    assert "vector_backend_unavailable" in output
    assert not (tmp_path / READINESS_V3_PATH).exists()


@pytest.mark.parametrize("drift", ["controls", "objects", "assets", "cache"])
def test_final_drift_rejected_without_repeating_full_cache_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    controls, calls, builds = _configured(monkeypatch)
    if drift == "controls":
        values = iter((controls, object()))
        monkeypatch.setattr(readiness, "load_controls", lambda root: next(values))
    elif drift == "objects":
        objects = iter(({"script": "a" * 40}, {"script": "b" * 40}))
        monkeypatch.setattr(readiness, "verify_execution_controller", lambda root: next(objects))
    elif drift == "assets":
        attempts = iter((False, True))
        original = readiness._local_gates

        def gate(*args: Any) -> tuple[tuple[legacy.ReadinessImageEvidence, ...], dict[str, object]]:
            images, detail = original(*args)
            if next(attempts):
                detail["invalid_checkout_task_ids"] = ("django__django-15902",)
            return images, detail

        monkeypatch.setattr(readiness, "_local_gates", gate)
    else:
        monkeypatch.setattr(readiness, "_cache_metadata_unchanged", lambda *args: False)
    assert readiness.qualify(tmp_path) == 1
    assert calls.count("cache") == 1 and calls.count("cpu") == 1
    assert builds == [] and not (tmp_path / READINESS_V3_PATH).exists()


def test_existing_readiness_not_overwritten_even_on_create_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configured(monkeypatch)
    output = tmp_path / READINESS_V3_PATH
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing")
    assert readiness.qualify(tmp_path) == 1
    assert output.read_bytes() == b"existing"


def test_cache_after_probe_checks_all_three_seals_without_vector_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        readiness, "_pinned_documents", lambda path: tuple({"sha256": key} for key in "abc")
    )
    report: dict[str, JsonValue] = {"job_sha256": "a", "inputs_sha256": "b", "complete_sha256": "c"}
    assert readiness._cache_metadata_unchanged(tmp_path, report)
    report["complete_sha256"] = "changed"
    assert not readiness._cache_metadata_unchanged(tmp_path, report)


def _worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key in {"OPENAI_API_KEY", "DATABASE_URL"} or key.startswith("REPOPILOT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


class FakeProvider:
    identity = PINNED_IDENTITY

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension
        self.requests: list[EmbeddingRequest] = []

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        # Deliberately bypass the response schema so malformed-provider cases
        # exercise the worker's own dimension guard, not Pydantic construction.
        return EmbeddingResponse.model_construct(
            request_id=request.request_id,
            identity=request.identity,
            vectors=(
                EmbeddingVector(
                    content_hash=request.inputs[0].content_hash,
                    values=(1.0, *((0.0,) * (self.dimension - 1))),
                ),
            ),
        )


def test_cpu_worker_embeds_exactly_one_fixed_short_input_batch_one_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _worker_env(monkeypatch)
    provider = FakeProvider()
    monkeypatch.setattr(
        readiness, "create_cpu_provider", lambda identity: (provider, _observation().cpu_profile)
    )
    result = readiness._cpu_probe()
    assert len(provider.requests) == 1 and len(provider.requests[0].inputs) == 1
    assert provider.requests[0].inputs[0].content == legacy.PROBE_CONTENT
    assert result.embedding == _observation().embedding
    assert result.cpu_profile == _observation().cpu_profile
    assert "vectors" not in result.model_dump_json()


@pytest.mark.parametrize(
    "key,value",
    [
        ("OPENAI_API_KEY", "secret"),
        ("DATABASE_URL", "secret"),
        ("REPOPILOT_EMBEDDING_MODEL", "other"),
        ("HF_HUB_OFFLINE", "0"),
        ("TRANSFORMERS_OFFLINE", "0"),
        ("CUDA_VISIBLE_DEVICES", "0"),
    ],
)
def test_cpu_worker_rejects_secrets_or_online_mode_before_model_load(
    monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    _worker_env(monkeypatch)
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(readiness, "create_cpu_provider", lambda _: pytest.fail("model loaded"))
    with pytest.raises(CpuProbeError, match="cpu_probe_environment_invalid"):
        readiness._cpu_probe()


def test_cpu_worker_blocks_network_and_restores_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    _worker_env(monkeypatch)
    original = socket.create_connection

    def forbidden(identity: object) -> Any:
        socket.create_connection(("example.invalid", 443))
        pytest.fail("network allowed")

    monkeypatch.setattr(readiness, "create_cpu_provider", forbidden)
    with pytest.raises(CpuProbeError, match="network_access_forbidden"):
        readiness._cpu_probe()
    assert socket.create_connection is original


def test_cpu_worker_rejects_wrong_vector_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    _worker_env(monkeypatch)
    monkeypatch.setattr(readiness, "create_cpu_provider", lambda _: (FakeProvider(767), {}))
    with pytest.raises(CpuProbeError, match="vector_dimension_mismatch"):
        readiness._cpu_probe()


class FakeProcess:
    pid = 81726
    returncode = 0

    def __init__(self, output: bytes, *, timeout: bool = False) -> None:
        self.output = output
        self.timeout = timeout
        self.timeouts: list[int | None] = []
        self.exited = False

    def __enter__(self) -> FakeProcess:
        return self

    def __exit__(self, *args: object) -> None:
        self.exited = True

    def communicate(self, timeout: int | None = None) -> tuple[bytes, None]:
        self.timeouts.append(timeout)
        if self.timeout and len(self.timeouts) == 1:
            assert timeout is not None
            raise subprocess.TimeoutExpired("secret-command", timeout)
        return self.output, None


def test_worker_process_strips_credentials_exits_before_profile_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeProcess(_observation().model_dump_json().encode())
    observed: list[dict[str, Any]] = []

    def popen(command: object, **kwargs: Any) -> FakeProcess:
        observed.append({"command": command, **kwargs})
        return child

    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")
    monkeypatch.setenv("DATABASE_URL", "secret-dsn")
    monkeypatch.setenv("HTTPS_PROXY", "secret-proxy")
    monkeypatch.setenv("REPOPILOT_MODEL", "override")
    monkeypatch.setenv("HF_HOME", "/unapproved-cache")
    monkeypatch.setattr(subprocess, "Popen", popen)

    def runtime(root: Path, profile: object) -> None:
        assert child.exited
        assert profile == _observation().cpu_profile

    monkeypatch.setattr(readiness, "verify_cpu_runtime", runtime)
    assert readiness._probe_cpu(tmp_path, _settings()) == _observation()
    assert child.timeouts == [900]
    assert observed[0]["start_new_session"] is True
    environment = observed[0]["env"]
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["OMP_NUM_THREADS"] == "8"
    assert not set(environment).intersection(
        {"OPENAI_API_KEY", "DATABASE_URL", "HTTPS_PROXY", "REPOPILOT_MODEL", "HF_HOME"}
    )


@pytest.mark.parametrize("failure", ["exit", "json", "oversize", "identity", "profile"])
def test_invalid_worker_observation_or_changed_runtime_cannot_certify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    payload = _observation().model_dump(mode="json")
    if failure == "identity":
        payload["embedding"]["identity_sha256"] = "a" * 64
    output = json.dumps(payload).encode()
    if failure == "json":
        output = b"secret-invalid-json"
    elif failure == "oversize":
        output = b"x" * (readiness.MAX_PROBE_OUTPUT_BYTES + 1)
    child = FakeProcess(output)
    if failure == "exit":
        child.returncode = 1
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)

    def runtime(*args: object) -> None:
        if failure == "profile":
            raise ValueError("secret-runtime-details")

    monkeypatch.setattr(readiness, "verify_cpu_runtime", runtime)
    with pytest.raises(legacy.ReadinessError) as error:
        readiness._probe_cpu(tmp_path, _settings())
    assert "secret" not in str(error.value)


def test_timed_out_worker_group_killed_and_waited_without_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = FakeProcess(b"", timeout=True)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(legacy.ReadinessError):
        readiness._probe_cpu(tmp_path, _settings())
    assert signals == [(child.pid, signal.SIGKILL)]
    assert child.timeouts == [900, None] and child.exited


@pytest.mark.parametrize("worker", [False, True])
def test_cli_never_publishes_raw_exception_or_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    worker: bool,
) -> None:
    def broken(*args: object) -> Any:
        raise RuntimeError("private-path secret-token")

    monkeypatch.setattr(readiness, "_cpu_probe" if worker else "qualify", broken)
    monkeypatch.setattr(
        sys,
        "argv",
        ["readiness", "--repository-root", str(tmp_path), *(["--cpu-probe"] if worker else [])],
    )
    assert readiness.main() == 1
    output = capsys.readouterr()
    assert "secret-token" not in output.out + output.err
    assert "private-path" not in output.out + output.err
