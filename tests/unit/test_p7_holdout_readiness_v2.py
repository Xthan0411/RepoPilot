"""Zero-paid v2 readiness gates; all external runtime probes are replaced by fakes."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from repopilot.evals import load_experiment_matrix

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import p7_holdout_readiness as legacy  # noqa: E402
from scripts import p7_holdout_readiness_v2 as readiness  # noqa: E402
from scripts.p7_holdout_control import (  # noqa: E402
    ENVIRONMENT_PATH,
    MATRIX_PATH,
    HoldoutEnvironmentManifest,
)
from scripts.p7_holdout_v2 import READINESS_V2_PATH, V2Controls  # noqa: E402
from scripts.p7_publish_registry_images import RegistryManifest  # noqa: E402


def _environment() -> HoldoutEnvironmentManifest:
    return HoldoutEnvironmentManifest.model_validate_json(
        (PROJECT_ROOT / ENVIRONMENT_PATH).read_bytes()
    )


def _mock_registry(monkeypatch: pytest.MonkeyPatch) -> RegistryManifest:
    raw = (PROJECT_ROOT / readiness.REGISTRY_PATH).read_bytes()
    monkeypatch.setattr(readiness, "_read", lambda *args: raw)
    return RegistryManifest.model_validate_json(raw)


@pytest.mark.parametrize("mutation", [None, "platform", "id", "digest", "shape", "missing"])
def test_local_images_require_platform_digest_and_config_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str | None
) -> None:
    registry = _mock_registry(monkeypatch)
    records = {item.registry_digest_reference: item for item in registry.images}
    calls: list[tuple[str, ...]] = []

    def inspect(arguments: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        expected = records[arguments[-1]]
        item = {
            "Id": expected.local_image_id,
            "RepoDigests": [expected.registry_digest_reference],
            "Architecture": "amd64",
            "Os": "linux",
        }
        if mutation == "missing":
            raise subprocess.CalledProcessError(1, arguments, stderr="secret-must-not-leak")
        if mutation == "platform":
            item["Architecture"] = "arm64"
        if mutation == "id":
            item["Id"] = "sha256:" + "0" * 64
        if mutation == "digest":
            item["RepoDigests"] = []
        return subprocess.CompletedProcess(
            arguments, 0, stdout=json.dumps([] if mutation == "shape" else [item])
        )

    monkeypatch.setattr(subprocess, "run", inspect)
    images, failures = readiness._resolve_images(tmp_path, _environment())

    assert len(calls) == 8
    assert all(call[:3] == ("docker", "image", "inspect") for call in calls)
    assert len(images) == (8 if mutation is None else 0)
    assert len(failures) == (0 if mutation is None else 8)


def test_registry_tampering_stops_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_registry(monkeypatch)
    environment = _environment().model_copy(update={"registry_manifest_raw_sha256": "0" * 64})
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: pytest.fail("Docker was invoked")
    )
    with pytest.raises(ValueError, match="registry_identity_mismatch"):
        readiness._resolve_images(tmp_path, environment)


@pytest.mark.parametrize("mutation", [None, "base", "root", "dirty", "extra", "ignored"])
def test_checkout_gate_rejects_wrong_revision_and_all_dirty_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str | None
) -> None:
    environment = _environment()
    bindings = {tmp_path / b.checkout_relative_path: b for b in environment.bindings}
    for path in bindings:
        path.mkdir(parents=True)
    status_calls: list[tuple[str, ...]] = []

    def git(path: Path, *args: str) -> str:
        if args == ("rev-parse", "--show-toplevel"):
            return str(tmp_path if mutation == "root" else path)
        if args == ("rev-parse", "HEAD"):
            return "0" * 40 if mutation == "base" else bindings[path].base_commit
        status_calls.append(args)
        return {
            "dirty": " M src/example.py",
            "extra": "?? extra.txt",
            "ignored": "!! .ignored-test-result",
        }.get(mutation or "", "")

    monkeypatch.setattr(readiness, "_git", git)
    failures = readiness._checkouts(tmp_path, environment)
    assert len(failures) == (0 if mutation is None else 8)
    assert all("--untracked-files=all" in call and "--ignored" in call for call in status_calls)


def test_checkout_gate_rejects_symlink_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment()
    target = tmp_path / "unexpected"
    target.mkdir()
    for binding in environment.bindings:
        path = tmp_path / binding.checkout_relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(readiness, "_git", lambda *args: pytest.fail("Git called on symlink"))
    assert len(readiness._checkouts(tmp_path, environment)) == 8


def test_output_gate_rejects_existing_path_symlink_and_file_parent(tmp_path: Path) -> None:
    (tmp_path / "existing").mkdir()
    (tmp_path / "file-parent").write_text("occupied")
    (tmp_path / "symlink").symlink_to(tmp_path / "nonexistent", target_is_directory=True)
    paths = ("fresh/output", "existing", "file-parent/output", "symlink/output", "../escape")
    assert readiness._output_conflicts(tmp_path, paths) == (1, 2, 3, 4)
    assert not (tmp_path / "fresh").exists()


def test_missing_dependencies_are_reported_without_importing_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def find(name: str) -> object | None:
        observed.append(name)
        return None if name in {"torch", "sentence_transformers"} else object()

    monkeypatch.setattr(importlib.util, "find_spec", find)
    assert readiness._missing_dependencies() == ("sentence_transformers", "torch")
    assert tuple(observed) == readiness.REQUIRED_DEPENDENCIES


def _configured(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[V2Controls, list[bool], list[dict[str, object]]]:
    """Replace every external probe; the orchestration still performs its final recheck."""

    controls = cast(V2Controls, SimpleNamespace(plan=SimpleNamespace(run_slots=())))
    old = legacy.HoldoutRuntimeReadiness.model_validate_json(
        (PROJECT_ROOT / "results/p7-holdout/runtime-readiness-v1.json").read_bytes()
    )
    matrix = load_experiment_matrix(PROJECT_ROOT / MATRIX_PATH, repository_root=PROJECT_ROOT)
    calls: list[bool] = []
    builds: list[dict[str, object]] = []

    def embedding(*args: object, offline: bool) -> legacy.EmbeddingReadinessEvidence:
        calls.append(offline)
        return old.embedding

    def build(**kwargs: object) -> legacy.ProviderCredentialEvidence:
        builds.append(kwargs)
        return legacy.ProviderCredentialEvidence()

    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-key")
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
    monkeypatch.setattr(readiness, "_missing_dependencies", lambda: ())
    monkeypatch.setattr(readiness, "load_experiment_matrix", lambda *args, **kwargs: matrix)
    monkeypatch.setattr(legacy, "_probe_embedding", embedding)
    monkeypatch.setattr(legacy, "_probe_vector_backend", lambda settings: old.vector_backend)
    monkeypatch.setattr(readiness, "build_readiness", build)
    return controls, calls, builds


@pytest.mark.parametrize("prepare_cache", [False, True])
def test_success_requires_final_offline_probe_then_creates_private_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepare_cache: bool
) -> None:
    controls, probes, builds = _configured(monkeypatch)
    assert readiness.qualify(tmp_path, prepare_embedding_cache=prepare_cache) == 0
    assert probes == ([False, True] if prepare_cache else [True])
    output = tmp_path / READINESS_V2_PATH
    assert output.stat().st_mode & 0o777 == 0o600
    assert len(builds) == 1 and builds[0]["controls"] is controls
    embedding = cast(legacy.EmbeddingReadinessEvidence, builds[0]["embedding"])
    assert embedding.cache_preparation_network_used == prepare_cache
    assert embedding.final_probe_network == "offline"
    content = output.read_text()
    assert "secret-openai-key" not in content and "secret-database" not in content


def test_missing_local_requirements_are_merged_before_expensive_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, probes, builds = _configured(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setattr(readiness, "_missing_dependencies", lambda: ("torch", "psycopg"))
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
    monkeypatch.setattr(
        legacy,
        "_probe_vector_backend",
        lambda *args: pytest.fail("Database touched before prerequisites"),
    )
    assert readiness.qualify(tmp_path, prepare_embedding_cache=True) == 1
    assert not probes and not builds and not (tmp_path / READINESS_V2_PATH).exists()
    output = capsys.readouterr().out
    for code in (
        "provider_credential_required",
        "local_dependency_unavailable",
        "sandbox_image_unavailable",
        "checkout_unavailable",
        "output_conflict",
    ):
        assert code in output
    assert "secret-database" not in output


@pytest.mark.parametrize("prepare_cache", [False, True])
def test_probe_failures_are_merged_without_persisting_errors_or_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    prepare_cache: bool,
) -> None:
    _, _, builds = _configured(monkeypatch)
    probes: list[bool] = []

    def broken_embedding(*args: object, offline: bool) -> None:
        probes.append(offline)
        raise legacy.ReadinessError(
            legacy.ReadinessErrorCode.EMBEDDING_SNAPSHOT_UNAVAILABLE,
            diagnostic={"sensitive": "secret-openai-key"},
        )

    def broken_vector(*args: object) -> None:
        raise legacy.ReadinessError(
            legacy.ReadinessErrorCode.VECTOR_BACKEND_UNAVAILABLE,
            diagnostic={"sensitive": "secret-database"},
        )

    monkeypatch.setattr(legacy, "_probe_embedding", broken_embedding)
    monkeypatch.setattr(legacy, "_probe_vector_backend", broken_vector)
    assert readiness.qualify(tmp_path, prepare_embedding_cache=prepare_cache) == 1
    assert probes == [not prepare_cache]
    assert not builds and not (tmp_path / READINESS_V2_PATH).exists()
    output = capsys.readouterr().out
    assert "embedding_snapshot_unavailable" in output and "vector_backend_unavailable" in output
    assert "secret-openai-key" not in output and "secret-database" not in output


@pytest.mark.parametrize("change", ["controller", "controls", "assets"])
def test_drift_during_long_probe_blocks_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    controls, _, builds = _configured(monkeypatch)
    if change == "controller":
        values = iter(({"script": "a" * 40}, {"script": "b" * 40}))
        monkeypatch.setattr(readiness, "verify_execution_controller", lambda root: next(values))
    elif change == "controls":
        changed = iter((controls, object()))
        monkeypatch.setattr(readiness, "load_controls", lambda root: next(changed))
    else:
        calls = 0
        original = readiness._local_gates

        def local(*args: object) -> tuple[tuple[object, ...], dict[str, object]]:
            nonlocal calls
            calls += 1
            images, details = original(*args)  # type: ignore[arg-type]
            if calls == 2:
                details["invalid_checkout_task_ids"] = ("django__django-15902",)
            return images, details

        monkeypatch.setattr(readiness, "_local_gates", local)
    assert readiness.qualify(tmp_path) == 1
    assert not builds and not (tmp_path / READINESS_V2_PATH).exists()


def test_create_only_write_never_overwrites_existing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configured(monkeypatch)
    output = tmp_path / READINESS_V2_PATH
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing-readiness")
    assert readiness.qualify(tmp_path) == 1
    assert output.read_bytes() == b"existing-readiness"


def test_online_success_without_final_offline_success_cannot_certify_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, builds = _configured(monkeypatch)
    probe = legacy._probe_embedding
    calls: list[bool] = []

    def offline_failure(*args: object, offline: bool) -> legacy.EmbeddingReadinessEvidence:
        calls.append(offline)
        if offline:
            raise legacy.ReadinessError(legacy.ReadinessErrorCode.EMBEDDING_SNAPSHOT_UNAVAILABLE)
        return probe(*args, offline=offline)  # type: ignore[arg-type]

    monkeypatch.setattr(legacy, "_probe_embedding", offline_failure)
    assert readiness.qualify(tmp_path, prepare_embedding_cache=True) == 1
    assert calls == [False, True]
    assert not builds and not (tmp_path / READINESS_V2_PATH).exists()


@pytest.mark.parametrize("exception", [ValueError, OSError, subprocess.SubprocessError])
def test_cli_does_not_expose_control_or_subprocess_error_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exception: type[Exception],
) -> None:
    def rejected(*args: object, **kwargs: object) -> int:
        raise exception("secret-control-or-subprocess-value")

    monkeypatch.setattr(readiness, "qualify", rejected)
    monkeypatch.setattr(sys, "argv", ["readiness", "--repository-root", str(tmp_path)])
    assert readiness.main() == 1
    output = capsys.readouterr().out
    assert "control_identity_or_authorization_invalid" in output
    assert "secret-control-or-subprocess-value" not in output


@pytest.mark.parametrize("state", ["clean", "dirty", "untracked", "ignored"])
def test_real_git_checkout_observes_dirty_and_extra_files(tmp_path: Path, state: str) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "tracked.txt").write_text("base\n")
    (checkout / ".gitignore").write_text("ignored.txt\n")
    for args in (("init", "--quiet"), ("add", "."), ("commit", "--quiet", "-m", "fixture")):
        subprocess.run(
            ("git", "-c", "user.name=Unit Test", "-c", "user.email=unit@example.test", *args),
            cwd=checkout,
            check=True,
            capture_output=True,
        )
    commit = readiness._git(checkout, "rev-parse", "HEAD")
    environment = _environment()
    binding = environment.bindings[0].model_copy(
        update={
            "checkout_relative_path": "checkout",
            "base_commit": commit,
            "checkout_commit": commit,
        }
    )
    environment = environment.model_copy(update={"bindings": (binding,)})
    if state != "clean":
        filename = {"dirty": "tracked.txt", "untracked": "extra.txt", "ignored": "ignored.txt"}[
            state
        ]
        (checkout / filename).write_text("new material\n")
    assert readiness._checkouts(tmp_path, environment) == (
        () if state == "clean" else (binding.task_id,)
    )
