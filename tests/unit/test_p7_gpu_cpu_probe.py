"""Pure CPU worker contracts; no optional model libraries or network required."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from repopilot.embeddings import (
    EmbeddingIdentity,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
)

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts import p7_gpu_cpu_probe as probe


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name in {"OPENAI_API_KEY", "DATABASE_URL"} or name.startswith("REPOPILOT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _input(content: str) -> dict[str, str]:
    return {"content": content, "content_hash": hashlib.sha256(content.encode()).hexdigest()}


def _vector() -> list[float]:
    return [1.0] + [0.0] * 767


def _request() -> dict[str, Any]:
    return {
        "identity": probe.PINNED_IDENTITY.model_dump(mode="json"),
        "samples": [
            {"input": _input(f"private repository sample {index}"), "gpu_vector": _vector()}
            for index in range(3)
        ],
        "queries": [_input(f"private issue query {index}") for index in range(8)],
    }


class FakeProvider:
    identity = probe.PINNED_IDENTITY

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = _vector() if vector is None else vector
        self.requests: list[EmbeddingRequest] = []

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        assert len(request.inputs) == 1
        return EmbeddingResponse(
            request_id=request.request_id,
            identity=request.identity,
            vectors=(
                EmbeddingVector(
                    content_hash=request.inputs[0].content_hash,
                    values=tuple(self.vector),
                ),
            ),
        )


def test_exactly_three_samples_then_eight_queries_batch_one_no_source_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = FakeProvider()
    result = probe.run_probe(_request(), loader=lambda identity: (provider, {"device": "cpu"}))
    assert result["status"] == "passed"
    assert result["cpu_probe_passed"] is True
    assert result["formal_ready"] is False
    assert result["local_embedding_calls"] == 11
    assert result["provider_requests_made"] == 0
    assert len(provider.requests) == 11
    assert len(result["samples"]) == 3  # type: ignore[arg-type]
    assert len(result["queries"]) == 8  # type: ignore[arg-type]
    serialized = json.dumps(result) + capsys.readouterr().out
    assert "private repository" not in serialized
    assert "private issue" not in serialized
    assert "gpu_vector" not in serialized
    assert "query_total_seconds" in result and "query_max_seconds" in result
    assert result["thresholds"] == {
        "cosine_minimum": 0.9999,
        "maximum_absolute_error": 0.001,
        "normalization_tolerance": 0.0001,
    }


@pytest.mark.parametrize("angle", [0.002, 0.02])
def test_numeric_failure_records_all_three_and_does_not_run_queries(angle: float) -> None:
    import math

    vector = [math.cos(angle), math.sin(angle)] + [0.0] * 766
    provider = FakeProvider(vector)
    result = probe.run_probe(_request(), loader=lambda identity: (provider, {}))
    assert result["failure_code"] == "numeric_tolerance_failed"
    assert result["cpu_probe_passed"] is False
    assert len(provider.requests) == 3
    assert result["queries"] == []


@pytest.mark.parametrize("coordinate", [True, "1", float("nan"), float("inf")])
def test_invalid_gpu_coordinate_rejected_before_loading(coordinate: object) -> None:
    request = _request()
    request["samples"][0]["gpu_vector"][0] = coordinate
    result = probe.run_probe(request, loader=lambda _: pytest.fail("model loaded"))
    assert result["cpu_probe_passed"] is False
    assert result["failure_stage"] == "request_validation"


@pytest.mark.parametrize(
    "change", ["identity", "query_length", "sample_count", "duplicate", "hash"]
)
def test_request_bounds_are_enforced_before_model(change: str) -> None:
    request = _request()
    if change == "identity":
        request["identity"]["revision"] = "main"
    elif change == "query_length":
        request["queries"][0] = _input("x" * 257)
    elif change == "sample_count":
        request["samples"].append(copy.deepcopy(request["samples"][0]))
    elif change == "duplicate":
        request["samples"][1] = copy.deepcopy(request["samples"][0])
    else:
        request["queries"][0]["content"] = "different"
    result = probe.run_probe(request, loader=lambda _: pytest.fail("model loaded"))
    assert result["failure_stage"] == "request_validation"
    assert result["local_embedding_calls"] == 0


@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "DATABASE_URL", "REPOPILOT_EMBEDDING_MODEL"])
def test_credentials_or_overrides_rejected_without_echo(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "secret-never-print")
    result = probe.run_probe(_request(), loader=lambda _: pytest.fail("model loaded"))
    assert result["failure_code"] == "runtime_credentials_or_overrides_forbidden"
    assert "secret-never-print" not in json.dumps(result)


def test_offline_flags_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE")
    result = probe.run_probe(_request(), loader=lambda _: pytest.fail("model loaded"))
    assert result["failure_code"] == "offline_environment_required"


def test_network_connection_blocked_and_restored() -> None:
    original = socket.create_connection

    def loader(identity: EmbeddingIdentity) -> Any:
        socket.create_connection(("example.invalid", 443))
        pytest.fail("network not blocked")

    result = probe.run_probe(_request(), loader=loader)
    assert result["failure_code"] == "network_access_forbidden"
    assert socket.create_connection is original


def test_local_embedding_failure_is_safe_without_retry() -> None:
    class Broken(FakeProvider):
        def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
            self.requests.append(request)
            raise RuntimeError("private secret text")

    provider = Broken()
    result = probe.run_probe(_request(), loader=lambda _: (provider, {}))
    assert result["failure_code"] == "cpu_probe_failed"
    assert len(provider.requests) == 1
    assert result["local_embedding_calls"] == 1
    assert "private secret" not in json.dumps(result)


def test_non_normalized_cpu_vector_fails_closed() -> None:
    provider = FakeProvider([0.5] + [0.0] * 767)
    result = probe.run_probe(_request(), loader=lambda _: (provider, {}))
    assert result["failure_code"] == "vector_not_normalized"
    assert len(provider.requests) == 1


def _fake_loader_modules(monkeypatch: pytest.MonkeyPatch, *, device: str = "cpu") -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    torch = SimpleNamespace(
        __version__="fake",
        set_num_threads=lambda n: arguments.update(threads=n),
        set_num_interop_threads=lambda n: arguments.update(interop=n),
        get_num_threads=lambda: arguments["threads"],
        get_num_interop_threads=lambda: arguments["interop"],
        are_deterministic_algorithms_enabled=lambda: False,
        get_float32_matmul_precision=lambda: "highest",
        backends=SimpleNamespace(mkldnn=SimpleNamespace(enabled=True)),
    )
    auto_type = type("PinnedModel", (), {})
    auto_type.__module__ = (
        "transformers_modules.jina." + probe.PINNED_IDENTITY.code_revision + ".model"
    )

    class FakeModel:
        max_seq_length = 0

        def __init__(self, name: str, **kwargs: Any) -> None:
            arguments.update(name=name, **kwargs)

        def parameters(self) -> Any:
            return [
                SimpleNamespace(
                    device=SimpleNamespace(type=device),
                    dtype="torch.float32",
                    is_floating_point=lambda: True,
                )
            ]

        def get_embedding_dimension(self) -> int:
            return 768

        def eval(self) -> None:
            arguments["eval"] = True

        def __getitem__(self, index: int) -> Any:
            return SimpleNamespace(auto_model=auto_type())

    modules = {
        "torch": torch,
        "sentence_transformers": SimpleNamespace(SentenceTransformer=FakeModel),
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "fake")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    return arguments


def test_cpu_loader_explicit_offline_pinned_fp32_and_fixed_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _fake_loader_modules(monkeypatch)
    provider, runtime = probe.create_cpu_provider(probe.PINNED_IDENTITY)
    assert provider.identity == probe.PINNED_IDENTITY
    assert arguments["device"] == "cpu"
    assert arguments["local_files_only"] is True
    assert arguments["revision"] == probe.PINNED_IDENTITY.revision
    assert arguments["model_kwargs"] == {"code_revision": probe.PINNED_IDENTITY.code_revision}
    assert arguments["config_kwargs"] == {"code_revision": probe.PINNED_IDENTITY.code_revision}
    assert arguments["threads"] == 8 and arguments["interop"] == 1
    assert runtime["parameter_devices"] == ["cpu"]
    assert runtime["parameter_dtypes"] == ["torch.float32"]


def test_loader_rejects_actual_cuda_despite_declared_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_loader_modules(monkeypatch, device="cuda")
    with pytest.raises(probe.CpuProbeError, match="cpu_fp32_required"):
        probe.create_cpu_provider(probe.PINNED_IDENTITY)


def test_worker_writes_safe_create_only_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path, output_path = tmp_path / "request.json", tmp_path / "output.json"
    request_path.write_text(json.dumps(_request()))
    monkeypatch.setattr(probe, "create_cpu_provider", lambda _: (FakeProvider(), {}))
    assert probe.main(["--request", str(request_path), "--output", str(output_path)]) == 0
    assert output_path.stat().st_mode & 0o777 == 0o600
    before = output_path.read_bytes()
    assert probe.main(["--request", str(request_path), "--output", str(output_path)]) == 1
    assert output_path.read_bytes() == before


@pytest.mark.parametrize("tamper", [False, True])
def test_controller_sealed_request_validated_before_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: bool
) -> None:
    from scripts.p7_gpu_cache_store import _write_once

    request, output = tmp_path / "request.json", tmp_path / "output.json"
    _write_once(request, _request())
    if tamper:
        document = json.loads(request.read_text())
        document["queries"][0] = _input("tampered query")
        request.write_text(json.dumps(document))
        monkeypatch.setattr(probe, "create_cpu_provider", lambda _: pytest.fail("model loaded"))
    else:
        monkeypatch.setattr(probe, "create_cpu_provider", lambda _: (FakeProvider(), {}))
    assert probe.main(["--request", str(request), "--output", str(output)]) == int(tamper)
    assert output.exists() is not tamper


def test_duplicate_json_keys_rejected_without_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, output = tmp_path / "request.json", tmp_path / "output.json"
    request.write_text('{"identity": {}, "identity": {}}')
    monkeypatch.setattr(probe, "create_cpu_provider", lambda _: pytest.fail("model loaded"))
    assert probe.main(["--request", str(request), "--output", str(output)]) == 1
    assert not output.exists()


def test_symlink_request_rejected(tmp_path: Path) -> None:
    target, link, output = tmp_path / "real.json", tmp_path / "link.json", tmp_path / "out.json"
    target.write_text(json.dumps(_request()))
    link.symlink_to(target)
    assert probe.main(["--request", str(link), "--output", str(output)]) == 1
    assert not output.exists()
