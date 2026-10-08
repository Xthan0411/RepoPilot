# P7-S6 formal holdout execution

## Scope and current boundary

P7-S6 owns the raw paid execution only. It runs the frozen four-arm, eight-task,
one-repetition application plan and retains append-only evidence. It does not calculate
success-rate, localization, ranking, or comparative claims; evaluator-side aggregation
starts in P7-S7 only after all 32 slots have a retained terminal record.

The x86 runtime-readiness artifact passed on 2026-08-08 with raw SHA-256
`6608e286837149b44f27a0d4d378afa4f1985326db6394608ab3f768329cb90c` and content
SHA-256 `326de84fa40ca9dece4107beb15d7b5dc5dc068ea8d7635e12bbaf300c99e761`.
It binds 8/8 local immutable images, the audited post-freeze Sandbox compatibility
patch, pinned offline Jina output dimension 768, pgvector 0.8.5 dimension 768, a
non-blank Provider credential, zero Provider/paid requests, no Agent-visible Gold, and
no blockers.

## Frozen run configuration

| Field | Formal value |
| --- | --- |
| Arms | `dense-only-v1`, `hybrid-rrf-v1`, `mini-swe-safe-v1`, `no-semantic-rag-v1` |
| Tasks | Eight frozen Holdout tasks, in plan order |
| Repetitions / slots | 1 / 32 |
| Agent source | frozen commit `a1fdecc1d33857cb5655d2b129e129f240dbbcfb` plus the one audited archive-mtime compatibility patch |
| Model / Prompt | `gpt-5.6-terra`, medium reasoning, 4,096 max output, `repopilot-agent-v1` |
| Per-slot limits | 30 steps, 1 replan, 100,000 tokens, USD 1.00, 900 seconds |
| Retrieval | Top-K 10, candidate Top-K 100, RRF k=60, query rewrite off |
| Embedding / vector | pinned local Jina, normalized 768 dimensions / pgvector |
| Sandbox | task-specific immutable registry digest, network `none`, effective command cap 300 seconds |
| Authorization | exact plan content SHA-256 `e998481dab733bcb3c6590db35caf75fb2d27e48dc4a007ff9916ef5f0e3f7f7`, USD 32.00, final in-flight overshoot acknowledged |

Only `OPENAI_API_KEY` and `DATABASE_URL` may enter from the runtime environment. The
runner rejects every `REPOPILOT_*` variable so the model, Prompt, retrieval, budgets,
task, or success rule cannot be changed through configuration. It forces Hugging Face
and Transformers offline mode before constructing the real executor. Secret values are
never serialized.

## Gold boundary

The frozen SWE-bench source-row file contains both Agent-visible Issue data and
evaluator-only patches. The formal runner must therefore never open it. A one-time,
evaluator-side command, `scripts/p7_holdout_agent_inputs.py`, projects only the eight
validated `TaskSpec` objects into private create-only
`results/p7-holdout/agent-inputs-v1.json`:

- raw SHA-256 `2a3a197ea7318d42e4b681db8b56b554fc241c19923ba8482566a4bdc9c7661f`;
- content SHA-256 `7250647265fc2faaf162d4ad7ea347e704457cbe022ceaf4e4b0884b1ba11821`;
- exact runtime command-contract SHA-256
  `feea5cb1e8c5fbaff8c1dcdb65c58d2a67d127b4e7d8dee9a9f5608a5d008b7b`;
- no `expected_patch`, `gold_files`, `gold_patch`, `test_patch`, FAIL_TO_PASS, or
  PASS_TO_PASS payload.

The generic P7-S1 Runner wrapper requires evaluator fields to calculate a task-set
identity. P7-S6 fills them with an explicit unusable withheld sentinel. That wrapper is
never passed to the executor; the execution request contains only `TaskSpec`. The
manifest retains only the derived hash and contains neither the sentinel, Issue body,
host checkout path, nor Gold. Tests also guard the source-row path so a later refactor
cannot silently load it in the formal controller.

## Operator contract

The default command is a read-only preflight. It validates the exact plan,
authorization, environment, readiness, Agent-only projection, frozen runtime, clean
tracked controller, 8 clean checkouts, 8 already-local image IDs, and existing
append-only progress. It does not construct a Provider executor:

```bash
uv run python -m scripts.p7_holdout_runner \
  --repository-root ~/work/repopilot
```

Paid execution requires both an explicit mutation switch and the exact approved plan
content hash. `--max-new-slots` bounds work started by one invocation without changing
the frozen plan:

```bash
uv run python -m scripts.p7_holdout_runner \
  --repository-root ~/work/repopilot \
  --execute \
  --confirm-plan-sha256 e998481dab733bcb3c6590db35caf75fb2d27e48dc4a007ff9916ef5f0e3f7f7 \
  --max-new-slots 1
```

The first one-slot invocation is a real formal slot, not an extra smoke task. If its
terminal evidence is structurally valid, the same command can be rerun with
`--max-new-slots 31`. A normal Agent failure such as `invalid_patch`, `tests_failed`,
`cost_limit`, or `provider_error` is valid measured output and does not require an
override. A controller/executor technical failure stops scheduling immediately.
Continuing other slots after reviewing such evidence requires the explicit
`--continue-after-technical-failure` flag; it does not erase or retry the failed slot.

The command is restart-safe. Closing SSH, terminating the Python process, or restarting
the VM leaves a `started` event and immutable partial execution directory. The next
invocation reconstructs history, records `interrupted`, reserves unknown cost for the
prior attempt, and starts a distinct execution index only if budget remains. Completed
and failed slots are deduplicated. No mutable checkpoint decides progress.

## Budget ledger and stopping rules

The application ledger is:

`known estimated cost from completed evidence + USD 1.00 reserve for each started attempt without trusted usage`.

Before every new attempt the ledger plus another USD 1.00 reserve must be at most the
authorized USD 32.00. This check executes before the Provider executor receives the
task. If a returned final in-flight response itself makes known cost exceed USD 32.00,
the raw evidence is retained and the controller exits with `budget_exhausted`; it never
schedules another slot. This is an application-side guard, not a provider billing hard
cap, and matches the user's explicit overshoot acknowledgement.

`--max-new-slots` reaching its limit is an intentional partial success and prints
`holdout_execution_invocation_complete=true` with remaining slots. Budget exhaustion,
identity drift, output corruption, or technical failure exits non-zero. Provider request
count is not guessed from slot count: actual token/cost/latency evidence comes from the
terminal Agent result.

## Retained outputs

Each plan slot writes under its fixed
`results/p7-holdout/raw/<plan>/<arm>/<task>/` path:

- one immutable `manifest.json`;
- globally ordered append-only lifecycle events;
- a new no-overwrite execution directory for every started/retried attempt;
- verified `result.json`, `trace.json`, and `artifacts.json` for a completed Agent result.

The controller reopens and hashes these bytes before trusting completion. Once all 32
slots are terminal it creates, never overwrites,
`results/p7-holdout/execution-report-v1.json`. That report is only a provenance/cost/raw
outcome index. It records known cost, conservative unknown-cost reserves, normal Agent
outcomes, technical failures, exact plan/readiness/input identities, and
`aggregation_deferred_to=P7-S7`; it contains no evaluator Gold or aggregate benchmark
claim.

## Stable errors and illegal states

| Code | Meaning |
| --- | --- |
| `invalid_configuration` | Execute confirmation, CLI values, or formal typed configuration is invalid. |
| `host_platform_mismatch` | Host is not Linux amd64/x86_64. |
| `controller_source_dirty` | A tracked controller file differs from the formal runner commit. |
| `source_artifact_invalid` | A required control/readiness/Agent-input artifact is missing or malformed. |
| `source_identity_mismatch` | Any cross-artifact hash, frozen byte, task, image, checkout, manifest, or retained artifact identity differs. |
| `unapproved_environment_override` | Any `REPOPILOT_*` variable is present. |
| `provider_credential_required` | Paid mode lacks a non-blank `OPENAI_API_KEY`. |
| `checkout_unavailable` | A task checkout is missing, dirty, unsafe, or at the wrong commit. |
| `sandbox_image_unavailable` | An exact registry digest is not already available with the readiness image ID. |
| `output_conflict` | An output path is unsafe, symlinked, malformed, concurrent, or conflicts with immutable history. |
| `budget_exhausted` | Another attempt cannot be reserved, or the final in-flight response crossed the approved ledger. |
| `technical_failure` | A controller/executor failure or retry-limit state requires operator review. |

Illegal configurations include a wrong plan confirmation, missing/extra/reordered slot,
different arm/config/model/Prompt/budget/task/image/checkout, mutable image, new frozen
Agent byte beyond the audited patch, Gold-bearing Agent input, dirty tracked source,
secret-bearing artifact, `REPOPILOT_*` override, non-local Jina behavior, vector dimension
drift, output overwrite, symlink traversal, silent retry, unreserved interrupted attempt,
or starting P7-S7 while any slot is still pending/interrupted.
