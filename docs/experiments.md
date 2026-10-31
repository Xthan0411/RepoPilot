# P7 experiments and execution evidence

## P7-S9 non-paid reproduction accepted (2026-10-31)

The [clean-run receipt](../results/p7-clean-run/receipt-v1.json) records a fresh
full Git checkout with a separate virtual environment and cached locked install.
The scripted investigation/replay, successful and unsuccessful trusted-fixture
repair contracts, and synthetic BM25 report passed. The ordinary offline suite
passed 2,112 tests (10 skipped, 21 deselected); lint/format/types also passed.

This is installation/workflow evidence, not another model experiment or a new
Docker/environment qualification. The test-only adapter executes a hard-coded
trusted fixture locally, and all model actions are scripted. No live Provider,
embedding-model or Docker calls were made. v6 outcomes and accounting remain
unchanged. Public entry points are the [README](../README.md),
[architecture](architecture.md), [evaluation](evaluation.md) and
[threat model](threat-model.md). Next is P7-S10 release/demo work; P8 is deferred.

## P7-S8 failure analysis accepted (2026-10-31)

[Error analysis](error-analysis.md) covers all 20 unsuccessful slots and the nine
specified taxonomy labels without inventing root causes. Five trace-linked cases
separate observations, interpretation limits and an unimplemented backlog.
The [machine-readable inventory](../results/p7-holdout/error-analysis-v6.json)
binds the S7 aggregate and each original trace/result hash. Its offline builder
first reuses the S7 evidence verification, then extracts terminal, budget, context,
Provider, patch, test, replan and failure-history facts for exactly those 20 slots.

Terminal counts are 9 `token_limit`, 3 `tests_failed`, 1 `repeated_action`, and 7
single-shot `invalid_patch` (3 `invalid_diff`, 4 `apply_conflict`). Runtime error
categories instead record 9 `budget_exhausted`, 10 `invalid_patch`, 1 `tool_misuse`;
the three Flask validation failures show why these are not interchangeable root
cause labels. All machine cases use `root_cause=not_inferred`.

The cases document early relevant reads followed by cumulative token exhaustion,
an applied patch without a subsequent test, Flask's public-issue/evaluator API
difference, an extractive summary dropping a symbol location before a repeated
query, and the control's output-contract limitation. These do not establish that
more tokens, a better summary or a corrected prompt would solve the tasks.

There are no new Provider requests, embeddings, database calls, sandbox tests,
candidate repairs or rescores. v6 and the cost ledger remain unchanged. Next is
P7-S9 documentation and clean-directory non-paid reproduction, then S10 release
work; optional P8 remains deferred.

## P7-S7 offline aggregation accepted (2026-10-31)

The user explicitly declined another baseline or paid rerun. v6 is the closing
experiment, with unchanged outcomes and its documented output-format confound.
This stage executes no Provider, embedding, retrieval, database or sandbox work.

Outputs: `results/p7-holdout/aggregate-v6.json`, `aggregate-v6-slots.csv`, and
`aggregate-v6.md`. `scripts/p7_aggregate_report.py` rebuilds them deterministically
through `repopilot.evals.reports`. It checks sealed controls, frozen inputs/dataset,
every planned slot, Runner history, nested artifact bytes, Agent replay and
single-shot context provenance before comparing raw outcomes and cost with v6.
Missing evidence, duplicate slots, changed bytes and summary drift fail closed.
Historical carry remains the already-audited sealed-plan ledger, not a new invoice.

The execution table retains 8 as each arm's planned denominator and separates
technical failures from completed unsuccessful repairs. It reports reproduction,
patch application, validation, tool/Provider counts, Decimal cost, tokens, Provider
latency and executor wall p50/p95 (R7 linear interpolation). Single-shot model tool
steps are zero; its baseline/apply/validate operations are evaluator steps, not
model interaction. Its 8/8 reproduction rate is evaluator-side, after the response.

Localization is rejoined to frozen gold only offline, ignoring neutral runtime zeros.
Same-task retained chunk identifiers are mapped through Dense/BM25 metadata and
complete initial-summary headers, never inferred from gold or successful patches.
Dense Recall@1/3/5 = .125/.500/.500, MRR=.291667 (all 8); BM25 = .375/.625/.750,
MRR=.525 (all 8). Hybrid Recall@1/3=.375/.500 (all 8); Recall@5 has 6/8 observable
cases and MRR has 4/8. Full-arm values remain N/A, not optimistic subset scores.
Mini's initial retrieval is N/A. Single-shot context scores have a separate basis.
First relevant successful read/search tool step is available per trace; elapsed
time is not retained. Real initial indexing/retrieval timers are retained in result
files for the three retrieval arms. They include runtime preparation with cached
vectors, not earlier GPU cache construction. Mini/single-shot zeros here mean no
independent initial retrieval, not zero-cost indexing.

Ordinary-task traces contain 275 model tool calls and zero observed policy refusals.
This does not measure attack interception, host execution, or secret exposure.
Those fields are N/A rather than fabricated zero-risk/100%-safe claims. All metrics
carry explicit denominators/availability, with raw trace/result hashes per CSV row.

The subsequent P7-S8 analysis is recorded above. It adds no paid experiment and
does not alter v6.

## Accepted v6 execution evidence (2026-10-31)

All **40 slots completed, with zero technical failures**. Raw Runner histories,
result files and report outcomes match exactly. The original Agent arms score
**19/32** (v5: 18/32); the additional single-shot control scores **1/8**, giving
20/40 total. The five arms reuse eight issues, not forty independent issues.

| Arm | v5 successes | v6 successes | v6 model cost estimate | v6 total tokens |
|---|---:|---:|---:|---:|
| dense-only-v1 | 4/8 | 4/8 | USD1.926355625 | 585,623 |
| hybrid-rrf-v1 | 5/8 | 6/8 | USD1.810553750 | 543,507 |
| mini-swe-safe-v1 | 4/8 | 5/8 | USD1.618186875 | 475,918 |
| no-semantic-rag-v1 | 5/8 | 4/8 | USD2.056925000 | 617,778 |
| single-shot-hybrid-context-v1 | — | 1/8 | USD0.303288750 | 42,550 |

Hybrid and the control both solve pytest-11143. Five tasks are solved only by
Hybrid, none only by the control, and two by neither. The descriptive rate difference
is 62.5 percentage points (75% vs 12.5%). **This is not a clean estimate of a pure
architecture gain**, for the implementation limitation below and the deliberate
difference in interaction/compute. Do not market it as a general sixfold capability
improvement, statistical significance, or an official full SWE-bench score.

### Single-shot format-contract limitation

All eight baseline requests have exactly the same issue, repository summary and
retrieval content actually rendered in the matching Hybrid first request, including
clipping, redaction and hit selection. Each uses one unique response, zero retries
and no model tool calls or feedback. Model/reasoning/output limits match. Independent
trace, deadline, usage and verifier checks pass. No refusal or truncation occurred.

However, the Agent's `ApplyPatchArguments.diff` schema explicitly requires canonical
`diff --git a/path b/path` headers, while the single-shot instructions say only
"ordinary unified diff headers". Django, Seaborn and Flask omit the git header and
are rejected as `invalid_diff`. This is an implementation/prompt-contract limitation
in the control, not a VM failure or missing experiment.

In a **post-hoc in-memory diagnostic only**, adding the header lets the existing
parser accept all three (its existing normalization also corrects one Seaborn hunk
count). Applying those candidates with the frozen exact-context helper to verified
revision-zero source artifacts still conflicts in all three: Django's docstring and
comment differ, Seaborn's indentation differs, and two Flask hunks do not match.
No full-checkout `git apply --check` or tests were run for this diagnostic, no
Provider was called, and no recorded result was changed. It would be incorrect to
claim that the header alone explains three lost successes or to adjust the score
to 4/8. The other four controls directly report `apply_conflict`; pytest-11143 is
the only success. A corrected-format control must be separately versioned if run.

### What the bounded correction actually did

Four normal validation failures enter the one-correction branch, all on Flask.
Every case rereads current source, applies a second incremental patch and exports
the true base-to-current diff. Only Hybrid then passes. Its trace is:
`hybrid-rrf-v1/pallets__flask-4992/executions/attempt_41d5291b77de52014e02123b279457b892d67e042ec746f97fbe4218431c2158/0000/trace.json`
(relative to the v6 raw directory). Sequence 14 applies the first patch, 17 fails
with TypeError, 18 replans, 20 reads current source, 23 applies the second patch and
26 passes one test. The final interface is `text: bool = True` with text/binary
opening selected from that flag.

The other three Flask corrections still fail with TypeError. For example Dense
keeps adjusting a `mode` parameter instead of implementing the expected interface.
Of the seven v5 `tests_failed` slots, three now succeed, three still fail tests and
one becomes `repeated_action`. Only Hybrid/Flask is a directly evidenced second-patch
rescue; Dense/BM25 xarray-4094 both pass on their first candidate in this round.

Across all 32 Agent slots there are five improvements and four regressions, net +1.
The 13 unsuccessful Agent outcomes are nine `token_limit`, three `tests_failed` and
one `repeated_action`. These are not infrastructure failures. Do not pool the best
outcome across rounds or compare 20/40 directly with v5's 18/32.

### Acceptance and accounting

Verified 128 frozen source blobs, 13 input hashes and 1,081 earlier raw-file hashes
(v3 112, v4 448, v5 521), 40 manifests, 80 history events and 519 execution-local
unique artifact references. All 32 Agent traces pass replay; eight independent
single-shot traces and their first-context bindings pass verification. All 19 Agent
successes and one control success have failed baselines, applied source patches and
distinct nonempty passing validations. All 58 Agent test observations reparse from
stored output. All 27 applied Agent revisions replay against full before/after
artifacts; the four second revisions export verified cumulative patches.

This is a retained-evidence audit, not a Docker or official upstream suite rerun.
All raw outcomes remain unchanged. Total model estimate is **USD7.715310000**;
cumulative accounting is **USD30.340901875 / 50**, comprising known usage estimates
USD19.340901875 plus historical unknown-cost reserve USD11. Actual invoice and
cloud costs are not reconciled. Remaining accounting headroom is USD19.659098125,
not permission to start another run. Execution-event span is 4302.109139 seconds
(about 71m42s), 19:10:14Z–20:21:56Z on 2026-10-31. Caches and sequential index
preparation affect wall-clock comparisons.

Evidence: [original report](../results/p7-holdout/postdebug-report-v6.json),
[authorization](../results/p7-holdout/postdebug-authorization-v6.json),
[audit receipt](../results/p7-holdout/postdebug-receipt-v6.json),
[paired comparison](../results/p7-holdout/postdebug-comparison-v5-v6.json).
Archive SHA256:
`bf1a850a556b0137fb2bd6f2e58327ab002f9706b0318eda5ced03ddea5bdb4d`.
Full evidence is retained in ignored `artifacts/p7-postdebug-v6-evidence-v1/`.
S6 execution evidence is complete; this receipt does not finish S7 localization,
safety and latency aggregation, S8 full taxonomy, or S9/S10 reproduction/release.

## Historical v6 design frozen before execution

The user approved a USD50 cumulative application ceiling, including the accepted
v5 carry USD22.625591875. USD11 of unknown historical spend remains reserved;
USD27.374408125 is available for the new run, excluding cloud charges and allowing
the already disclosed final in-flight response overshoot. The new matrix preserves
all four original arms (32 slots) and adds eight `single-shot-hybrid-context-v1`
slots. It is a seen-task rerun, not a new blind holdout.

The control reads only the matching v6 Hybrid trace's first rendered issue,
repository summary and retrieved context after clipping, redaction and hit selection,
with exact content/provenance hashes. It submits one
structured-output request to the same frozen model, reasoning and output limits.
There are no model tools, plan loop, second response, transport retry or evaluator
feedback. Deterministic setup/baseline/application/validation follows the response.
Report the comparison as the gain from interactive execution **given the same initial
retrieval**, not the gain over an issue-only raw model, and report actual tokens/costs
because the two arms intentionally have different interaction/compute budgets.

The repair change gives at most one normal failed-validation correction within the
existing shared replan limit and unchanged task budgets. It requires fresh current
source reads, exports the complete base-to-current patch, enforces cumulative patch
limits, and disallows stale validation evidence. Only static allowlisted exception
types are added to sanitized pytest failure rows; source/messages/stacks/parameter
values and custom exception names remain private. These are general repairs, not
task-specific hints. The observed results and limitations are recorded above.

## Accepted v5 execution evidence (2026-10-31)

The complete v5 matrix has **32 completed slots, zero technical failures and 18
successful repairs**. Each arm runs the same eight tasks once. These are post-debug
seen-task results, not a fresh blind holdout or an official full SWE-bench score.

| Arm | v4 successes / 8 | v5 successes / 8 | v5 technical failures | v5 estimated model cost |
|---|---:|---:|---:|---:|
| dense-only-v1 | 4 | 4 (50%) | 0 | USD 1.883914375 |
| hybrid-rrf-v1 | 3 | 5 (62.5%) | 0 | USD 1.747911875 |
| mini-swe-safe-v1 | 2 | 4 (50%) | 0 | USD 1.225762500 |
| no-semantic-rag-v1 (BM25) | 1 | 5 (62.5%) | 0 | USD 1.511760000 |

The raw histories reproduce all report outcomes exactly. Relative to v4, 13 failed
slots became successful and five successful slots regressed, a net gain of eight.
Do not combine the best result from each round. Hybrid ties the BM25 arm; this small
single-repetition sample does not demonstrate Hybrid superiority or identify which
individual prompt/tool-output change caused the improvement.

The 14 unsuccessful outcomes are seven `tests_failed`, five `token_limit` and two
`repeated_action`. No `invalid_patch` termination or technical exception occurred in
v5. In particular, no `protected_test_path` correction was exercised: the dedicated
correction branch has offline test coverage, not an observed recovery success here.

The v5 model cost estimate is USD **6.369348750**, with no new unknown-cost reserve.
The historical carry remains USD **16.256243125**, including USD 11 of unknown-cost
reserve. Total application accounting is **USD 22.625591875 / 32**; actual provider
invoice reconciliation and VM/GPU charges are excluded. The execution-event span is
**63m24s**, from 2026-10-31 10:01:27.858794Z to 11:04:52.114203Z; it excludes earlier
environment/cache preparation and must not be represented as model inference time.

Evidence: [original v5 report](../results/p7-holdout/postdebug-report-v5.json),
[VM authorization](../results/p7-holdout/postdebug-authorization-v5.json),
[local audit receipt](../results/p7-holdout/postdebug-receipt-v5.json),
[paired comparison](../results/p7-holdout/postdebug-comparison-v4-v5.json).
The archive SHA256 is
`007a0e8d3b167f9aba56e433d07588f0a308f43a673634fe871c5f223bf40763`;
full raw evidence and the archive are retained locally in ignored
`artifacts/p7-postdebug-v5-evidence-v1/`. Historical v4 evidence remains unchanged.

Acceptance checked 32 manifests / 64 history events, 125 frozen source blobs, ten
input hashes and 560 prior raw-file hashes. All 32 complete traces (824 events)
passed integrity/replay; all 329 execution-local unique artifact references match
their bytes and manifests. All 18 successful slots have a failed baseline, an
applied production-only patch and a distinct nonempty passing validation with the
same command index/hash. Re-parsing the 36 retained baseline/validation outputs
reproduces their summaries exactly. Fifteen success baselines are assertion
failures; three Django baselines are classified `test_error` (157 tests, one error),
not assertion failures. Their separate validations have zero failures/errors.
All 26 applied patches target production source only. Six large-output projections
(two read, four search) across four slots have verified full artifacts, usable
previews and continuation metadata. No Provider call or sandbox test rerun occurred
during this local audit; 61 offline controller regression tests also passed.

This acceptance summary is not the full S7 localization/safety/latency aggregation.
S7/S8 reporting and S9/S10 reproduction/demo/release work remain before the core MVP
gate. No further paid rerun is needed to begin those deliverables.

### Initial trace-linked cases for S8

Paths below are relative to the archive's
`results/p7-holdout/raw/p7-postdebug-v5-00779c01190b4e38/`. These are initial audited
examples, not a claim that the full S8 taxonomy and improvement backlog are done.

- **Large-output path works in a real repair:** Hybrid/Django,
  `hybrid-rrf-v1/django__django-15902/executions/attempt_8a04bfb44f7996a8120767290ad5c54f8a82e13f9da0be4ec1f43c87742ecb6f/0000/trace.json`.
  Event 5 projects a 14,209-byte search response with 50 returned matches into a
  readable preview plus full artifact; event 23 applies only `django/forms/formsets.py`;
  event 26 passes 157 tests. The same v4 slot failed technically, but its missing
  original ContextErrorCode prevents a per-case proof of the old exact exception.
- **An applied patch can still fail the contract:** Hybrid/Flask,
  `hybrid-rrf-v1/pallets__flask-4992/executions/attempt_31cdc88ae880e4f7e7d1962ea77d1c83b9ce2dfef1ecd789192f03af7f9f09c6/0000/trace.json`.
  Event 17 applies a `mode: str` interface; event 20 fails
  `test_config_from_file_toml`. The v4 successful candidate used a `text: bool`
  interface. The original issue itself suggests `mode="b"`, and retained sanitized
  output lacks the exception stack: the interface difference is observable, but
  the precise sole cause is not established by this evidence.
- **Applied is not validated:** Dense/pytest-11148,
  `dense-only-v1/pytest-dev__pytest-11148/executions/attempt_83ee00c72b1b45635e18a922a1d72a41a508a707fac123ce0ed23e95998b1bd5/0000/trace.json`.
  Overlapping reads precede the patch at event 29. The model asks for `run_tests`
  at event 31, but that response brings usage to 100,730/100,000 tokens; event 32
  terminates before the test tool executes. This is budget exhaustion before
  validation, not a failed patch test. BM25 solves the same task with two passing
  tests at 60,844 tokens. One paired observation does not prove causal superiority.

## v5 repair and execution contract

The user requested a bounded follow-up repair after the v4 audit. Large successful
read/search outputs now have immutable full artifacts plus usable bounded previews;
context failures preserve typed diagnostics. The common prompt is explicitly v2:
tests/evaluator inputs are read-only, remaining budgets are visible, and only a
pre-apply protected-test rejection may share the existing single correction allowance.
Other safety denials remain terminal. The historical v1 matrix remains unchanged;
the v5 plan records the common prompt override separately from arm settings.

The independent v5 runner reconstructs v4 outcomes and carries known USD5.256243125
plus unknown reserve USD11, for USD16.256243125 total and USD15.743756875 remaining
within the original USD32 application ceiling. It reuses the existing qualified assets
and checks identities/cache reads; no long environment/retrieval preflight is repeated.
All four arms run all eight tasks once in a new output root. No v4 success is merged
with a v5 retry to create a score. Local Fake tests establish execution correctness,
not a model success rate. The user executed v5 on the VM; the local acceptance audit
makes no paid requests.

Frozen [v5 plan](../results/p7-holdout/postdebug-plan-v5.json):
`p7-postdebug-v5-00779c01190b4e38`, source
`3522dd1d2c091fa64d22b1da603379139d5073bd`, content SHA256
`7bda422b5740f206e08d9ce5cb4cfb5c9d9bc48aeab9db4f9074bdcf561430a6`.
It binds 560 prior raw files across v3/v4. Its create-only authorization is written
only when the operator explicitly executes this exact plan; no local paid run or
new runtime-readiness artifact is asserted.

> **Historical v4 acceptance (superseded by the v5 results above):**
> v4 has 32 terminal slots: 23 completed, 9 technical failures, 10 successful repairs.
> These are **post-debug seen-task reruns**, not a new blind holdout. The full eight
> tasks remain in each arm's denominator, including technical failures. No claim of
> an official full SWE-bench harness score or statistically significant arm ranking
> follows from this small, single-repetition sample.

## Accepted v4 results

| Arm | Success / all tasks | Success rate | Completed without technical error | Technical failures |
|---|---:|---:|---:|---:|
| dense-only-v1 | 4/8 | 50% | 6 | 2 |
| hybrid-rrf-v1 | 3/8 | 37.5% | 5 | 3 |
| mini-swe-safe-v1 | 2/8 | 25% | 6 | 2 |
| no-semantic-rag-v1 | 1/8 | 12.5% | 6 | 2 |

The 32 slots are eight issues evaluated across four arms, not 32 distinct issues.
Success means the frozen isolated regression protocol observed a failing baseline,
an applied production-only patch and a distinct nonempty passing validation. All ten
successes were checked against the recorded commands, logs, patch hashes and verifier
results. This audit did not independently rerun the official upstream full test suite.

Sources: [original report](../results/p7-holdout/postdebug-report-v4.json),
[local verification receipt](../results/p7-holdout/postdebug-receipt-v4.json),
[bound plan](../results/p7-holdout/postdebug-plan-v4.json),
[VM-created authorization](../results/p7-holdout/postdebug-authorization-v4.json).
The original archive SHA256 is
`84e7fa00eea12197a238970d1c0c901fbff1d8cb91b68eaead7d186a12701950`.
Archive and full raw evidence are retained locally under ignored
`artifacts/p7-postdebug-v4-evidence-v1/`, not implicitly published to Git. Verification
covered all 32 execution histories, 23 complete trace replays and 211 artifact refs.

Plan `p7-postdebug-v4-2cb3f8ced909187d` binds source
`d710c49104754264fdb3feb54c49967d8f177aee`; execution controller was
`442096d82434d875e35353ef9992c95ec8280885`. The user explicitly waived expensive
preflight after the minimal repair; historical runtime assets and read-only GPU
corpus vectors were reused with source/input/budget checks and per-vector read
integrity. No new qualification pass is invented. Prior v2/v3 results remain intact.

### Costs, timing and outstanding limitations

- Completed v4 cost estimate: USD **4.515823125**. Nine failed slots retain USD **9**
  unknown-cost reserve. Historical carry remains USD **2.740420000**. Total application
  accounting is USD **16.256243125 / 32**, **not an actual invoice** and excluding VM/GPU
  charges. Failed sidecars record at least USD 0.634817500, already covered by the
  unknown reserve, not added twice or used to clear it.
- First execution start to last terminal event: **58m14s** (2026-10-31
  08:14:21.586860Z–09:12:35.222082Z), excluding earlier GPU/cache preparation.
  Sanitized test stdout contains a fixed `in 0.00s`; it is not a performance metric.
- The thirteen completed failures are six protected evaluator-test path denials,
  four cumulative token limits, one failed validation, one repeated action and one
  exhausted retrieval. No invalid-diff/apply-conflict rejection occurred this run.
- Nine technical failures share agent/state_conflict/invalid_value after observe.
  A real-tool offline reproduction confirms a generic defect: a large read/search
  result lacks an artifact reference required by the next context assembly. The
  remote original payload/ContextErrorCode was not retained, so this is a strong
  shared-cause candidate, not proof of every remote failure's exact cause.
- Next: S7 reproducible localization, safety, cost/latency aggregates and consistency
  tests, then S8 trace-linked failure analysis. No automatic paid rerun is authorized
  merely by receiving this evidence; the remaining defect and original results stay
  visible in the final report.

## Historical matrix and qualification

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
