"""Load the frozen Jina snapshot in a fresh process and emit secret-free evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from repopilot.config import DEFAULT_DATABASE_URL
from repopilot.embeddings import (
    EmbeddingInput,
    EmbeddingRequest,
    embedding_identity_from_settings,
    embedding_provider_from_settings,
    embedding_request_id,
)
from repopilot.evals import load_experiment_matrix, select_experiment_arm

MATRIX_PATH = Path("configs/experiments/p7-v2.yaml")
PROBE_CONTENT = "def formal_holdout_embedding_probe() -> int:\n    return 1\n"
PROBE_CONTENT_SHA256 = hashlib.sha256(PROBE_CONTENT.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--network-mode", choices=("offline", "online"), required=True)
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve(strict=True)
    _validate_network_mode(arguments.network_mode)
    matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    settings = select_experiment_arm(matrix, "dense-only-v1").settings(
        openai_api_key="embedding-probe-provider-not-used",
        database_url=DEFAULT_DATABASE_URL,
    )
    identity = embedding_identity_from_settings(settings)
    item = EmbeddingInput(content_hash=PROBE_CONTENT_SHA256, content=PROBE_CONTENT)
    request = EmbeddingRequest(
        request_id=embedding_request_id(identity, (item,), 1),
        identity=identity,
        inputs=(item,),
        attempt=1,
    )
    response = embedding_provider_from_settings(settings).embed(request)
    if len(response.vectors) != 1 or len(response.vectors[0].values) != 768:
        raise SystemExit("embedding_probe_invalid_response")
    evidence = {
        "model": identity.model,
        "revision": identity.revision,
        "code_revision": identity.code_revision,
        "identity_sha256": identity.fingerprint,
        "normalized": identity.normalize,
        "maximum_length": identity.max_length,
        "declared_dimension": identity.dimension,
        "probe_content_sha256": PROBE_CONTENT_SHA256,
        "observed_vector_count": len(response.vectors),
        "observed_dimension": len(response.vectors[0].values),
    }
    print(json.dumps(evidence, separators=(",", ":"), sort_keys=True))
    return 0


def _validate_network_mode(mode: Literal["offline", "online"]) -> None:
    expected = "1" if mode == "offline" else "0"
    if os.environ.get("HF_HUB_OFFLINE") != expected:
        raise SystemExit("embedding_probe_network_contract_mismatch")
    if os.environ.get("TRANSFORMERS_OFFLINE") != expected:
        raise SystemExit("embedding_probe_network_contract_mismatch")
    if "OPENAI_API_KEY" in os.environ or "DATABASE_URL" in os.environ:
        raise SystemExit("embedding_probe_secret_exposure")
    if any(name.startswith("REPOPILOT_") for name in os.environ):
        raise SystemExit("embedding_probe_override_exposure")


if __name__ == "__main__":
    raise SystemExit(main())
