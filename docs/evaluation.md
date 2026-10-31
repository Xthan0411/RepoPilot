# Evaluation: accepted v6 results

RepoPilot completed **40 execution slots: 8 previously seen tasks × 5 arms × 1
repetition**. Twenty slots passed the frozen isolated regression tests; twenty did
not. This is a post-debug, seen-task evaluation, not a fresh blind holdout, 40
independent problems, or a full official SWE-bench score. No further paid
experiments are part of this closeout.

Start with the [aggregate report](../results/p7-holdout/aggregate-v6.md), its
[machine-readable metrics](../results/p7-holdout/aggregate-v6.json), and the
[failure analysis](error-analysis.md). The latter distinguishes observed stopping
mechanisms from untested explanations.

## Results and denominators

| Arm | Successful slots | Unsuccessful slots | Estimated model cost, USD |
|---|---:|---:|---:|
| `dense-only-v1` | 4/8 (50.0%) | 4/8 (50.0%) | 1.926355625 |
| `hybrid-rrf-v1` | 6/8 (75.0%) | 2/8 (25.0%) | 1.810553750 |
| `mini-swe-safe-v1` | 5/8 (62.5%) | 3/8 (37.5%) | 1.618186875 |
| `no-semantic-rag-v1` (BM25) | 4/8 (50.0%) | 4/8 (50.0%) | 2.056925000 |
| `single-shot-hybrid-context-v1` | 1/8 (12.5%) | 7/8 (87.5%) | 0.303288750 |
| All slots | 20/40 (50.0%) | 20/40 (50.0%) | 7.715310000 |

All 40 slots completed; technical failures and missing slots were both **0/40**.
The four interactive Agent arms passed 19/32 slots, distinct from the combined
20/40 figure. Failures stay in their original arm's denominator; results are not
assembled by selecting the best attempts across historical versions.

The four Agent arms stopped unsuccessfully on cumulative token limits in **9/32**
slots, failed validation in **3/32**, and a repeated action in **1/32**. The
single-shot arm had **3/8** format rejections and **4/8** application conflicts.
These are mutually exclusive final outcomes, not counts of every intermediate
error. See the [trace-linked failure inventory](../results/p7-holdout/error-analysis-v6.json).

Success requires a reproduced baseline failure, an applied source patch, and a
non-empty passing validation of that candidate using the configured test command.
It does not establish correctness against every upstream test. Across completed
slots, baseline reproduction was 40/40, patch application was 24/40, and validation
passed in 20/40. In the single-shot arm, baseline reproduction is evaluator work
after the model response, not feedback supplied to the model.

Costs are estimates from retained Provider usage, not reconciled invoices. The
v6 USD 7.715310000 excludes cloud resources and historical runs. Application
accounting including historical carry is **USD 30.340901875 / 50.00**; carry
includes **USD 11.00 of unknown-cost reserves**, not confirmed expenditure.

## What this comparison can and cannot show

- Dense, BM25 and Hybrid provide different initial retrieval context; Mini has
  no precomputed retrieval context. All four use the bounded interactive repair
  workflow. There was **no Memory ablation** in these results.
- The single-shot control receives the same selected initial issue/repository
  evidence as Hybrid's first rendered request, but makes one model request with
  no interactive tool feedback. It is not the same full prompt, interaction
  budget, or computation budget as the Agent arm.
- A confirmed format-contract confound remains: the single-shot prompt requests
  ordinary unified diff, while the parser requires `diff --git` headers. Three
  control candidates lacked those headers. Thus **75.0% versus 12.5% is only a
  descriptive result of this implementation**, not evidence of a causal sixfold
  capability gain. No hypothetical repaired control result replaces a score.
- These tasks were inspected during earlier debugging, and each arm has only
  one repetition per task. The run cannot establish unseen-task generalization,
  variance, statistical significance, or the independent effect of each repair.

The [experiment record](experiments.md) documents the format mismatch, historical
runs, bounded correction behavior, and additional limitations.

## Metric availability and safety limits

The [aggregate JSON](../results/p7-holdout/aggregate-v6.json) records exact
numerators, denominators, per-slot usage, and availability. Provider latency is
the sum of reported request latency; task wall time spans runner start to terminal
state, including preparation and cleanup. Eight samples per arm do not establish
production tail latency. Index preparation timing includes cached-vector/index
work, not the earlier GPU embedding job or a cold embedding benchmark.

File localization is reconstructed offline from retained ranked evidence and
frozen evaluator gold filenames; runtime placeholder zeros are ignored. Mini's
initial-retrieval metrics are N/A. Hybrid's Recall@5 is recoverable for only 6/8
tasks and MRR for 4/8, so full-arm values are N/A rather than zero or a favorable
subset mean. Single-shot localization describes its rendered initial context,
not an independent retrieval ranking. Time to first relevant repository tool
result is also N/A because per-tool completion timestamps were not retained.

The traces retain 275 tool executions and zero policy denials. Ordinary repair
tasks do not measure attack-blocking effectiveness. Host execution, secret
exposure and adversarial interception rates are **not established by this run**;
absence of a recorded incident is not a 100% safety result. Security tests and
the [threat model](threat-model.md) are separate evidence.

## Frozen execution identity

The [v6 plan](../results/p7-holdout/postdebug-plan-v6.json) binds source commit
`ff8f7c0144b56749c6d1515ff9c7f83da53680e6`, dependency/source hashes, dataset,
execution slots and accounting. The recorded provider/model is OpenAI
`gpt-5.6-terra`, with `medium` reasoning and a 4,096-token output limit. Agent
requests use `repopilot-agent-v2`; the control uses `repopilot-single-shot-v1`.
The v6 plan explicitly overrides the older prompt version in the
[base matrix](../configs/experiments/p7-v2.yaml). These describe the executed run,
not a claim about current model availability or deterministic responses.

## Offline verification, not a paid rerun

With the project's Python environment already installed and the original archive
restored under `artifacts/p7-postdebug-v6-evidence-v1`, run these from the repository
root:

```bash
.venv/bin/python -m scripts.p7_aggregate_report \
  --evidence-root artifacts/p7-postdebug-v6-evidence-v1 \
  --check

.venv/bin/python -m scripts.p7_error_analysis \
  --evidence-root artifacts/p7-postdebug-v6-evidence-v1 \
  --check
```

The evidence root is the extracted archive root containing `results/`, not the
nested raw-plan directory. These checks validate retained identities, hashes,
traces, artifacts and saved reports without rewriting them or making Provider,
Docker, embedding or database calls. Expected summaries include
`aggregate_verified=true, slots=40` and `analysis_verified=true, failed_slots=20`,
with `check_only=true` and zero Provider/test requests. Missing or mismatched raw
evidence is an error, not permission to regenerate experimental outcomes.

The complete raw archive is **not published in the Git checkout**; the tracked
[aggregate JSON](../results/p7-holdout/aggregate-v6.json),
[per-slot export](../results/p7-holdout/aggregate-v6-slots.csv),
[execution report](../results/p7-holdout/postdebug-report-v6.json) and failure
inventory remain readable without it. Full independent offline verification
requires access to that archive. A fresh paid execution additionally requires the
frozen runtime inputs, model credentials, cache/database setup, and access to the
digest-pinned images in the project's private registry. Their references do not
guarantee continuing access or retention. Existing reports therefore do **not**
make the full evaluation a one-command public rerun.

For local setup and explicitly labeled non-paid examples, return to the
[README](../README.md). This page does not authorize new experiments or provide
a VM deployment procedure.
