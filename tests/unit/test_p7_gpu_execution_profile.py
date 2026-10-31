"""Qualified evidence stays pinned; GPU corpus and CPU queries remain separate."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from repopilot.embeddings import (
    EmbeddingInput,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    embedding_request_id,
)

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from scripts import p7_gpu_execution_profile as profile  # noqa: E402
from scripts.p7_gpu_cache_acceptance import CacheAcceptanceError  # noqa: E402
from scripts.p7_gpu_cache_store import _digest  # noqa: E402
from scripts.p7_gpu_cpu_probe import PINNED_IDENTITY  # noqa: E402


def _report() -> dict[str, Any]:
    return profile.load_qualification_report(ROOT)


class SyntheticProvider:
    identity = PINNED_IDENTITY

    def __init__(self) -> None:
        self.calls_made = 0

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.calls_made += 1
        return EmbeddingResponse(
            request_id=request.request_id,
            identity=self.identity,
            vectors=tuple(
                EmbeddingVector(content_hash=item.content_hash, values=(1.0,) + (0.0,) * 767)
                for item in request.inputs
            ),
        )


def test_original_report_has_qualified_shape_and_counts() -> None:
    report = _report()
    assert len(report["retrieval"]["tasks"]) == 8
    assert sum(len(t["stages"]) for t in report["retrieval"]["tasks"]) == 64
    assert report["retrieval"]["corpus_cache_hits"] == 453956


def test_original_report_binds_historical_source_when_available() -> None:
    report = _report()
    present = subprocess.run(
        ("git", "cat-file", "-e", f"{profile.QUALIFIED_COMMIT}^{{commit}}"),
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if present.returncode != 0:
        pytest.skip("original private development history is not included in the public snapshot")
    assert (
        _digest(profile._source_objects_at(ROOT, profile.QUALIFIED_COMMIT))
        == (report["execution_source_sha256"])
    )


def test_report_raw_bytes_cannot_be_reformatted(tmp_path: Path) -> None:
    target = tmp_path / profile.QUALIFICATION_PATH
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(_report(), indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="qualification_raw_mismatch"):
        profile.load_qualification_report(tmp_path)


@pytest.mark.parametrize("mutation", ["task", "sample", "miss", "source", "paid", "stage"])
def test_resealed_but_invalid_report_is_not_accepted(mutation: str) -> None:
    report = copy.deepcopy(_report())
    if mutation == "task":
        report["retrieval"]["tasks"].pop()
    elif mutation == "sample":
        report["cpu_probe"]["samples"][0]["passed"] = False
    elif mutation == "miss":
        report["retrieval"]["corpus_cache_misses"] = 1
    elif mutation == "source":
        report["retrieval"]["source_embedding_calls"] = 1
    elif mutation == "paid":
        report["paid_requests_made"] = 1
    else:
        report["retrieval"]["tasks"][0]["stages"][0]["status"] = "failed"
    report["sha256"] = _digest({k: v for k, v in report.items() if k != "sha256"})
    with pytest.raises(ValueError):
        profile._validate_report(report)


@pytest.mark.parametrize(
    "field,value", [("cpu_threads", 4), ("batch_size", 8), ("dtype", "torch.float16")]
)
def test_runtime_drift_is_rejected(field: str, value: Any) -> None:
    runtime = _report()["cpu_probe"]["runtime"]
    profile.verify_cpu_runtime(ROOT, runtime)
    runtime[field] = value
    with pytest.raises(ValueError, match="cpu_runtime_drift"):
        profile.verify_cpu_runtime(ROOT, runtime)


def _request(*contents: str) -> EmbeddingRequest:
    inputs = tuple(
        EmbeddingInput(content=c, content_hash=hashlib.sha256(c.encode()).hexdigest())
        for c in contents
    )
    return EmbeddingRequest(
        identity=PINNED_IDENTITY,
        inputs=inputs,
        request_id=embedding_request_id(PINNED_IDENTITY, inputs, 1),
        attempt=1,
    )


def test_dynamic_queries_allowed_without_preflight_allowlist() -> None:
    base = SyntheticProvider()
    provider = profile.SingleQueryCpuProvider(base)
    for index in range(8):
        request = _request(f"new Agent query {index}")
        assert provider.embed(request).request_id == request.request_id
    assert base.calls_made == 8
    with pytest.raises(ValueError, match="single_cpu_query_required"):
        provider.embed(_request("first", "second"))
    assert base.calls_made == 8


def test_resource_factory_verifies_cache_before_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(profile, "verify_qualification", lambda root: calls.append("qualification"))

    def bad_cache(*args: Any) -> Any:
        calls.append("cache")
        raise CacheAcceptanceError("corpus_cache_miss")

    monkeypatch.setattr(profile, "verify_pinned_cache", bad_cache)
    monkeypatch.setattr(
        profile, "create_cpu_provider", lambda identity: pytest.fail("model loaded")
    )
    with pytest.raises(CacheAcceptanceError):
        profile.build_gpu_resources(ROOT)
    assert calls == ["qualification", "cache"]


def test_resource_factory_keeps_exact_cache_and_query_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = SyntheticProvider()
    cache = object()
    monkeypatch.setattr(profile, "verify_qualification", lambda root: _report())
    monkeypatch.setattr(
        profile,
        "verify_pinned_cache",
        lambda *args: SimpleNamespace(identity=PINNED_IDENTITY, cache=cache),
    )
    monkeypatch.setattr(
        profile, "create_cpu_provider", lambda identity: (base, _report()["cpu_probe"]["runtime"])
    )
    provider, actual_cache = profile.build_gpu_resources(ROOT)
    assert actual_cache is cache
    assert isinstance(provider, profile.SingleQueryCpuProvider)
    assert provider.provider is base
