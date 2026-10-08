# P7-S6 authorization and zero-paid runtime readiness

## Status and purpose

This is still P7-S6. P7-S7 begins only after all formal Holdout raw runs have reached a
retained terminal state; it owns aggregation, denominators, and consistency reporting.

The user explicitly approved the exact USD 32.00 P7-S6 application plan and acknowledged
that the final in-flight Provider response can make the observed bill slightly exceed the
application-side estimate. The create-only
[`authorization-v1.json`](../results/p7-holdout/authorization-v1.json) binds:

- plan content SHA-256 `e998481dab733bcb3c6590db35caf75fb2d27e48dc4a007ff9916ef5f0e3f7f7`;
- matrix content SHA-256 `d75d36d711f23190062aded9e97a8f5abfaba859416466027b227421ea7407e8`;
- formal-freeze content SHA-256 `9dbac71aca6063cc864567b1c69e7b0bfe5f866d3035fb91f86a5d621aabe7b5`;
- `approved_by=ethan`, exact `max_total_cost_usd=32.00`, paid execution approval, and
  explicit final-response overshoot acknowledgement.

The authorization file was created with mode `0600`. Its raw SHA-256 is
`37df6c2210f71c3046cfc9c79101b4911b00e4143cc9a47999276fd7009aec58` and
its validated content SHA-256 is
`561f47755a255c22b1a3c6a604071b2981f68df49b05dc12e65e8b57f0aa3402`.
P7-S5's USD 8 authorization is not reused.

`scripts/p7_holdout_readiness.py` consumes that authorization, the frozen plan, the
accepted 8/8 environment manifest, and the existing dataset preflight. It never loads the
source-row snapshot or a Gold patch and never calls OpenAI. Only after every gate passes
does it write private, create-only
`results/p7-holdout/runtime-readiness-v1.json`.

### Audited post-freeze compatibility patch

The first x86 readiness attempt correctly stopped before any Provider or embedding probe
with `source_identity_mismatch`. The formal freeze points to source commit
`a1fdecc1d33857cb5655d2b129e129f240dbbcfb`, while the later eight-task environment
qualification required one change under the otherwise frozen runtime:
`src/repopilot/sandbox/docker.py` gives archived workspace entries the deterministic
1980-01-02 UTC mtime instead of tar's epoch-zero default. This prevents ZIP-based Python
build backends from rejecting the staged workspace; it changes archive metadata only,
not the model, Prompt, retrieval, task, Gold, budget, or scoring rule.

Readiness does not broadly exempt the Sandbox file or weaken the source check. It accepts
only this exact compatibility transition, already contained by the environment manifest's
controller ancestry:

- frozen blob `ac0f4c4050350555644fb81e354a658d0d1de73f`;
- qualified blob `67c326a0ec8408c5dcd3661d099ec730e8986721`;
- full binary-diff SHA-256
  `eab26db5938e04385aa770b203bf736ae142171d6979b3395ca19c363fa0faf1`;
- exactly one changed frozen path: `src/repopilot/sandbox/docker.py`.

Both commit ancestry edges, both Git blobs, the patch digest, the current controller
blob, and the complete changed-path set must match. Any additional frozen path or byte
still fails closed. The accepted identity is serialized into the zero-paid readiness
artifact as `frozen_runtime` evidence.

## Frozen runtime configuration

Readiness derives settings directly from `configs/experiments/p7-v2.yaml`; it does not
merge normal `REPOPILOT_*` configuration. The paid runner remains fixed to:

| Field | Frozen value |
| --- | --- |
| Arms | `dense-only-v1`, `hybrid-rrf-v1`, `mini-swe-safe-v1`, `no-semantic-rag-v1` |
| Tasks / repetitions | 8 Holdout tasks × 1 repetition = 32 slots |
| Model | `gpt-5.6-terra`, reasoning effort `medium`, maximum output 4,096 |
| Prompt | `repopilot-agent-v1` |
| Per-task limits | 30 steps, 1 replan, 100,000 tokens, USD 1.00, 900 seconds |
| Retrieval | Top-K 10, candidate Top-K 100, RRF k=60, query rewrite disabled |
| Embedding | Jina code model, exact primary/code revisions, normalized 768 dimensions, max length 2,048 |
| Sandbox | task-specific immutable digest, task Agent/test network `none`, 300-second command cap |

Only `OPENAI_API_KEY` and `DATABASE_URL` may enter at runtime. Their values are never
serialized. A blank Key blocks readiness. The Provider check deliberately proves only
non-blank credential presence and records `provider_api_requests_made=0`; it does not
claim the Key is valid before the first real Provider call. `DATABASE_URL` defaults to
`postgresql://repopilot:repopilot@localhost:55433/repopilot` when absent. Any
`REPOPILOT_*` variable is rejected instead of being allowed to alter a formal arm. The
local embedding model, revisions, budget, Prompt, model, retrieval settings, and success
rule have no environment override.

## Readiness probes

The command first validates Linux amd64, a clean tracked controller, exact content hashes,
authorization identity, frozen Agent source plus the single audited archive-metadata
compatibility patch, all 32 unused output paths, eight exact clean checkouts, and eight
already-local registry digest references. Image inspection never pulls or mutates an
image.

It then runs three runtime probes:

1. **Provider credential:** non-blank presence only; no OpenAI endpoint is called.
2. **Jina embedding:** `scripts/p7_embedding_readiness_probe.py` runs in a fresh child
   interpreter. Optional `--prepare-embedding-cache` permits one host-side online download
   using only the frozen primary and remote-code revisions. A second fresh interpreter
   must then load and encode with `HF_HUB_OFFLINE=1` and
   `TRANSFORMERS_OFFLINE=1`. The child receives a minimal cache/runtime environment
   allowlist and no OpenAI Key, database URL, `REPOPILOT_*` value, or unrelated secret.
   This host cache preparation does not change the task Sandbox network policy.
3. **pgvector:** the production adapter applies its idempotent migration and verifies the
   `vector` extension, schema/table metadata, and dimension 768. The DSN is not retained.

The final artifact binds raw and content hashes for the plan, authorization, and
environment; current controller commit; eight Docker image observations; secret-free
embedding and pgvector identities; the complete ready preflight; zero Provider/paid
requests; no Agent-visible Gold; and empty blockers.

## Aggregate diagnostic and termination

Structural failures such as a wrong host, dirty controller, malformed artifact, or
authorization mismatch terminate immediately because downstream probes would be unsafe.
Once those prerequisites pass, the command checks all independent runtime gates and
reports them together. Missing images are attempted across all eight tasks rather than
stopping at the first one. A blocked run prints one bounded JSON diagnostic containing
only booleans, stable codes, and missing task IDs, then exits `1` without writing the
formal artifact or making a paid request.

Success exits `0` and prints:

```text
holdout_runtime_readiness_passed=true tasks=8 images=8 embedding_dimension=768 vector_dimension=768 provider_requests=0 paid_requests=0 ready_for_paid_holdout=true blockers=0
```

Stable readiness errors are:

| Code | Meaning |
| --- | --- |
| `invalid_configuration` | Runtime secret input or typed settings are malformed. |
| `host_platform_mismatch` | Host is not Linux amd64. |
| `controller_source_dirty` | A tracked controller file differs from the readiness commit. |
| `source_artifact_invalid` | Plan, authorization, environment, freeze, or dataset evidence is missing/malformed. |
| `source_identity_mismatch` | Cross-artifact hashes, tasks, images, controller, or the exact frozen-runtime/compatibility-patch identity differ. Frozen-runtime failures include a bounded `stage` diagnostic. |
| `authorization_mismatch` | Consent is not bound to this exact plan/matrix/freeze or lacks overshoot acknowledgement. |
| `unapproved_environment_override` | A `REPOPILOT_*` override is present. |
| `local_dependency_unavailable` | OpenAI, psycopg, or sentence-transformers runtime is absent. |
| `provider_credential_required` | `OPENAI_API_KEY` is blank or absent. |
| `sandbox_image_unavailable` | One or more immutable digest images are not already local. |
| `embedding_snapshot_unavailable` | Pinned Jina cache preparation or final offline 768-dimension probe failed. |
| `vector_backend_unavailable` | PostgreSQL/pgvector connection, migration, extension, schema, or dimension failed. |
| `readiness_blocked` | Typed preflight still contains a non-passing check. |
| `output_conflict` | The create-only readiness path already exists or cannot be created privately. |

Invalid configurations include approving anything except USD 32.00, omitting the
overshoot acknowledgement, reusing a dev authorization, changing or reordering an arm or
task, using a mutable image, changing a checkout/Python/timeout projection, exposing
Gold, supplying a `REPOPILOT_*` override, changing either Jina revision, using a
non-768 vector schema, leaving any planned output path present, or trying to overwrite a
readiness artifact.

## Boundary after this checkpoint

The accepted x86 artifact has raw SHA-256
`6608e286837149b44f27a0d4d378afa4f1985326db6394608ab3f768329cb90c`, content
SHA-256 `326de84fa40ca9dece4107beb15d7b5dc5dc068ea8d7635e12bbaf300c99e761`,
`ready_for_paid_holdout=true`, eight images, zero paid/Provider requests, no Gold, and no
blockers. It authorizes no tuning and itself performs no Holdout task.

The resumable 32-slot controller now consumes only this exact readiness identity and the
separate Gold-free Agent input projection. It retains every terminal result and
interrupted/technical attempt, stops scheduling when the application ledger reaches the
authorized limit, and defers all aggregation to P7-S7. Its operator, budget, termination,
error, and illegal-state contract is documented in
[`p7-s6-formal-execution.md`](p7-s6-formal-execution.md).
