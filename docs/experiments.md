# P7 experiment matrix and preflight

P7-S4 originally froze the experiment design in `p7-v1.yaml`. P7-S5 local model
verification found two pre-task reproducibility defects, so v1 remains historical and
is not edited in place. The current source of truth is
[p7-v2.yaml](../configs/experiments/p7-v2.yaml), whose canonical content digest is
<code>d75d36d711f23190062aded9e97a8f5abfaba859416466027b227421ea7407e8</code>.
Changing any arm, budget, model, task set, prompt, image, or artifact digest invalidates
the matrix.

## Arms and the variable under test

| Arm | Precomputed context | Retrieval | Memory | Default |
|---|---|---|---|---|
| mini-swe-safe-v1 | none | none | off | yes |
| no-semantic-rag-v1 | bounded repository context | BM25 | off | yes |
| dense-only-v1 | bounded repository context | Jina cosine search | off | yes |
| hybrid-rrf-v1 | bounded repository context | BM25 + Dense, RRF k=60 | off | yes |
| hybrid-memory-v1 | bounded repository context | BM25 + Dense, RRF k=60 | split-specific read-only snapshot | no |

The optional Memory arm is disabled by default because both checked-in snapshots are
empty and no extra formal-run budget has been approved. Selecting it requires explicit
opt-in and a Memory store whose experiment, snapshot, split, and read-only access mode
exactly match the matrix.

Dense and Hybrid are production execution paths, not labels. The repair workflow now
chunks the repository, builds or refreshes the revision-scoped vector namespace, embeds
the issue query once, and routes through Dense or RRF according to the selected arm.
Unit tests use a Fake embedding provider and an in-memory vector store; formal runs use
the frozen local Jina model and pgvector.

## Frozen control variables

- Dataset: repopilot-p7-v1, exactly 2 dev fixtures and 8 holdout Issues, with separate
  Runner task-set digests.
- Agent model: OpenAI gpt-5.6-terra, reasoning effort medium, maximum output 4,096
  tokens, Prompt repopilot-agent-v1.
- Temperature: the shared setting is 0.0, but the Responses reasoning-provider adapter
  intentionally does not transmit a temperature parameter. All arms therefore use the
  same provider-default sampling behavior rather than claiming deterministic decoding.
- Embedding: jinaai/jina-embeddings-v2-base-code at immutable revision
  516f4baf13dec4ddddda8631e019b5737c8bc250, normalized 768-dimensional vectors,
  maximum length 2,048. Its delegated `jinaai/jina-bert-v2-qk-post-norm` executable
  code is separately pinned at `3baf9e3ac750e76e8edd3019170176884695fb94` and is part
  of the embedding fingerprint. Transformers is constrained to `>=4.57,<5`; the tested
  lock resolves 4.57.6 because the immutable Jina code is incompatible with 5.x.
- Retrieval: Top-K 10, Hybrid candidate Top-K 100, RRF k=60, query rewrite disabled.
- Agent budget per task: 30 steps, 1 replan, 100,000 total tokens, USD 1.00, 900 seconds.
- Provider limits: 120-second request timeout, 2 retries, 50,000 prompt characters.
- Repetitions: one dev and one holdout run per enabled arm. This is explicitly
  budget-constrained; reports must not claim a variance estimate.
- Sandbox: python:3.12-alpine digest
  sha256:6d43704baacd1bfbe7c295d7f13079d5d8104ed33568873133f8fc69980419df,
  no network during test execution, 1 CPU, 1,024 MiB memory, and 256 PIDs.
- Tools and success criterion: the same bounded repository/patch/test tools and
  rule-based-verifier-v1 for every arm.

The model choice follows the current OpenAI model guide: the GPT-5.6 family is the
recommended current family and Terra is the balanced intelligence/cost tier. The matrix
uses the exact Terra model ID, never gpt-5.6, gpt-5-latest, or another mutable alias.
Configured sub-272K pricing is USD 2.50/M uncached input tokens, USD 0.25/M cached input
tokens, and USD 15/M output tokens.

Upstream references:

- <https://developers.openai.com/api/docs/guides/latest-model>
- <https://developers.openai.com/api/docs/models/gpt-5.6-terra>
- <https://huggingface.co/jinaai/jina-embeddings-v2-base-code/commits/main>

## Runtime environment boundary

Formal settings are built directly from the matrix and do not call the normal
environment-merging configuration loader. REPOPILOT_MODEL, REPOPILOT_TEMPERATURE,
budget variables, retrieval variables, and all other REPOPILOT_* values cannot override
a formal arm.

Only OPENAI_API_KEY and DATABASE_URL may be supplied at execution time. Neither value is
persisted, and both are redacted in runtime snapshots. The local Embedding model cannot
be replaced through an environment variable.

## Termination and error behavior

Agent runs retain these terminal reasons:

success, max_steps, token_limit, timeout, cost_limit, sandbox_error, invalid_patch,
tests_failed, tool_error, provider_error, retrieval_exhausted, repeated_action, and
agent_stopped.

Experiment loading and preflight add stable error codes:

input_io_error, invalid_configuration, unsupported_version, integrity_mismatch,
dataset_mismatch, fairness_violation, arm_unavailable, model_snapshot_invalid,
embedding_snapshot_invalid, memory_snapshot_invalid, dependency_lock_mismatch,
source_snapshot_unclean, sandbox_image_unavailable, provider_unavailable,
vector_backend_unavailable, gold_boundary_violation, output_conflict, and secret_exposure.

Formal execution is rejected rather than degraded when any check is blocked or not run.
Technical failures remain in raw results; the Runner does not delete failed attempts or
silently rerun them under a different configuration.

## Invalid configurations

Validation fails closed for, among other cases:

- a missing, duplicated, reordered, renamed, or extra arm;
- a mutable model alias, non-digest Sandbox image, non-40-character Jina revision, or
  changed Prompt version;
- budget, repetition, pricing, tool, verifier, Sandbox resource, Top-K, or RRF drift;
- query rewrite enabled in one arm;
- Dense/Hybrid names that do not map to their real retrieval execution mode;
- a baseline receiving Dense, Hybrid, Memory, evaluator gold, or arbitrary shell access;
- a frozen Memory snapshot that is writable, missing, cross-split, or changed;
- task IDs, task-set digest, dataset content digest, uv.lock digest, or referenced file
  digest mismatch;
- secret fields or evaluator-only gold, gold_files, gold_patch, oracle, or test_patch
  fields in the matrix;
- any unapproved environment override, conflicting output root, unavailable
  image/provider/vector backend, or unclean source snapshot.

## Selection and preflight

The four default arms can be selected by stable ID:

    from pathlib import Path
    from repopilot.evals import load_experiment_matrix, select_experiment_arm

    root = Path.cwd()
    matrix = load_experiment_matrix(
        root / "configs/experiments/p7-v2.yaml",
        repository_root=root,
    )
    selection = select_experiment_arm(matrix, "hybrid-rrf-v1")

build_experiment_executor then constructs the baseline adapter or production repair
executor for that exact selection. It accepts the two secret connections and optional
test doubles, but no model, budget, or retrieval override.

The historical [P7-S4 preflight](../results/p7-preflight/preflight.json) remains bound to
v1. P7-S5 used versioned dev-only preflights that do not load the holdout source-row
artifact. The final v5 preflight passed the dependency lock, exact two-task dev set,
local dependencies, pinned Jina snapshots, digest-pinned Sandbox image, clean source
snapshot, provider credential presence, output root, and live pgvector checks.

Paid calibration additionally requires an integrity-protected authorization bound to
the v2 matrix. Four default arms × two dev tasks × USD 1.00 per-task application budget
gives a planned ceiling of USD 8.00. This is not represented as a provider-billing hard
cap: usage is known only after a response, so the final in-flight call can overshoot a
task budget; provider-side billing controls remain necessary for an absolute account
limit. The orchestrator stops before another arm if reported cumulative cost exceeds the
authorization and retains the raw overrun evidence. Missing approval, a different ceiling, a blank
credential, any non-dev task, a blocked runtime preflight, output overwrite, or attempt
to freeze a dirty/mismatched commit fails closed. The ignored local credential is not
persisted in any control or result artifact. The dev-discovered v1→v2 changes are
recorded in
[calibration-change-log-v1.json](../results/p7-dev/calibration-change-log-v1.json); no
holdout result or gold informed them.

The final v5 calibration is retained in
[calibration-report-v5.json](../results/p7-dev/calibration-report-v5.json). Its four arms
and eight Runner attempts are complete: five executions reached deterministic Verifier
success and three were retained as `invalid_patch/patch_not_applied`. The v5 run cost USD
`0.546178750`; all four development attempts together cost an estimated USD
`1.275710625`. These two fixtures and one repetition validate the workflow and expose
badcases, but are not a reportable arm-quality benchmark. In particular, the neutral
zero-valued localization fields in repair `FinalResult` keep evaluator gold outside the
Agent executor and must be recomputed evaluator-side in P7-S7, not presented as measured
retrieval quality.

The [formal freeze marker](../results/p7-dev/formal-freeze-v1.json) binds the v2 matrix,
lock, dataset, Prompt, model, both Jina revisions, Sandbox image, four arm hashes, v5
report bytes, and clean code commit `a1fdecc1d33857cb5655d2b129e129f240dbbcfb`.
P7-S6 must consume this identity without holdout-informed tuning.

## P7-S6 formal holdout control and paid-run boundary

The first formal application plan is retained at
[plan-v1.json](../results/p7-holdout/plan-v1.json). It contains exactly four default arms,
eight frozen holdout tasks, one repetition, and 32 task-specific run/output identities.
Every slot binds the frozen arm hash, repository base commit, and required Python version.
The theoretical application limit is exactly USD 32.00. The optional Memory arm is not
included; P7-S5's USD 8 approval is not reusable; a new authorization must bind this plan
and explicitly acknowledge that the last in-flight Provider response can overshoot the
application-side estimate.

The corresponding [preflight-v1.json](../results/p7-holdout/preflight-v1.json) was built
from clean controller commit `a4a425e8e3f2def67588d90fa59ca31df2e862b6`. It verifies
the formal freeze identity, unchanged frozen Agent runtime, Gold-free plan projection,
and empty output roots. It records `paid_requests_made=0`. Later x86 qualification has
advanced the environment evidence without changing that immutable historical artifact:

- the pinned upstream SWE-bench harness proves baseline-fail/Gold-pass for 8/8 tasks;
- RepoPilot's stricter Sandbox adapter has complete x86 evidence for Django, Seaborn, and
  Flask and xarray 5131; the corrected four-task x86 diagnostic then passed xarray 4094,
  both pytest tasks, and SymPy. Staged diagnostics therefore cover all eight tasks;
- a subsequent single run on clean controller commit `5f2076fdfc6bfb9963e4524a9727434d7f5d2707`
  produced the complete [eight-task RepoPilot Sandbox artifact](../results/p7-holdout/sandbox-matrix-preflight-v1.json):
  all eight matched Python, set up offline, failed baseline, and passed evaluator Gold;
- all eight adapted images are published to the fixed private `us-west1` repository under
  immutable digest references. The [registry manifest](../results/p7-holdout/registry-manifest-v1.json)
  proves 8/8 pullbacks, and an independent Artifact Registry control-plane query matched
  all eight recorded digests;
- the declared 600/900-second command timeouts exceed the frozen 300-second Sandbox
  policy. `holdout-command-timeout-contract-v1` now resolves this deterministically as
  `min(declared, 300)` without changing the source dataset or matrix. The committed
  [x86 environment artifact](../results/p7-holdout/environments-v1.json) binds all eight
  immutable task images and clean checkouts, records 17 capped commands, and closes the
  environment gate with zero paid requests and no Agent-visible Gold;
- the exact-plan USD 32 authorization is committed with explicit final in-flight response
  overshoot acknowledgement; it has made zero paid requests;
- the create-only x86 runtime-readiness artifact passes all gates: exact controller and
  audited compatibility patch, 8/8 local image identities, non-blank Provider credential
  without an API call, pinned Jina cache plus fresh-process offline 768-dimensional probe,
  pgvector 0.8.5 dimension 768, zero paid requests, no Agent-visible Gold, and no blockers;
- a content-addressed Agent-only task projection separates Issue/repository/test commands
  from evaluator patch fields. The formal runner never opens the evaluator source rows,
  requires the exact approved plan hash, uses each task's immutable image, reconstructs
  progress from append-only events, and applies a known-cost plus conservative unknown-cost
  ledger before every paid attempt.

The all-task qualification driver no longer fails at the first incompatible project. It
attempts every selected task, emits a bounded aggregate failure list, and accepts repeated
`--task-id` arguments for diagnostic subsets. Subset runs never write the formal 8/8
artifact. Exact xarray/pytest source metadata is mapped to pinned evaluator-only build
tools. Because `.git` is deliberately absent, exact upstream package versions are passed
through fixed project-specific setuptools-scm environment variables; SymPy receives a
pinned pytest 7.4 toolchain that its frozen test command requires. Agent setup and tests
still run with network `none`, and no Provider request is made during qualification.
The debugging chronology and remaining formal boundary are documented in
[`p7-s6-environment-debugging-postmortem.md`](p7-s6-environment-debugging-postmortem.md).
The registry target, authentication lifetime, mutation confirmation, pullback identity
proof, resume rules, and stable errors are documented in
[`p7-s6-registry-publication.md`](p7-s6-registry-publication.md).
The command-timeout projection, 8-task mapping, evidence binding, and error taxonomy are
documented in [`p7-s6-timeout-contract.md`](p7-s6-timeout-contract.md).
Runtime identity, Gold isolation, execution/resume behavior, budget accounting, outputs,
and stable errors are documented in
[`p7-s6-formal-execution.md`](p7-s6-formal-execution.md).

This was a correctness boundary, not a transient model failure. The tasks require Python
3.9, 3.10, and 3.12.13 and execute dependency setup commands; using the generic Python
3.12 image could have turned environment incompatibility into a false Agent failure. The
accepted x86 manifest now proves, for every task, an exact clean base checkout, matching
Python, immutable task image, offline setup success, baseline failure, Gold success,
Agent network `none`, and evaluator-only Gold. The remaining P7-S6 work is the authorized
32-slot raw execution, beginning with one real retained formal slot. No holdout result may
change the frozen behavior, and P7-S7 cannot aggregate until every slot is terminal.
