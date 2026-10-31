# Architecture

RepoPilot is a single-Agent, tool-using repair workflow built with LangGraph. It is
not a master/subagent system: one decision component selects the next permitted
action; deterministic code controls transitions, budgets, execution, and success.
The current MVP targets bounded Python-repository repair tasks, not arbitrary
autonomous development.

This document describes the implemented runtime. The accepted v6 experiment also
has a frozen controller and evaluator-specific wiring; those are not implicit
features of every CLI invocation.

## Repair path

```text
TaskSpec + validated settings + local repository + pinned image
    |
    +-- ingest / chunk / initial retrieval / optional Memory
    |
    +-- prepare sandbox / run declared setup
    |
    v
single Agent plan:
    reproduce baseline -> localize -> read source -> patch -> validate -> finish
    ordinary validation failure -> [if eligible] fresh read / correction -> validate again
    each decision: context -> Provider action -> typed tool -> observation
    each transition: budget checks + state/trace recording
    |
    v
rule-based Verifier -> result.json + trace.json + artifacts.json
    |
    +-- optional verified-episode admission in a writable Memory namespace
```

The baseline must show an actual test failure, not merely a nonzero process exit.
`inspect_symbol` supplies structural information; it does not replace a successful,
nonempty `read_file` for each patch target. This is a file-level evidence guard,
not proof that the model understood every changed line.

The graph permits at most one validation-driven repair after an applied candidate
has an ordinary, nontruncated `FAILED` test result. It shares `max_replans` with
empty-search recovery; if that allowance is already spent, there is no extra repair.
Timeouts, policy refusals, and infrastructure errors do not qualify. A second patch
requires fresh reads of the current patched workspace. Exported patch evidence is
cumulative from the original source to the current source, while each submitted
correction is incremental.

A separate one-correction allowance covers malformed/conflicting patches and a
typed protected-test rejection that proves the candidate was not applied. It does
not relax general path policy or retry arbitrary safety failures. Call IDs remain
globally unique within a run; relevant action fingerprints are revision-scoped so
a necessary read or validation after a source change is not mistaken for a repeat.

The Verifier requires baseline reproduction, an artifact-backed applied patch,
and an independent passing run of the same configured test command. The workflow
accepts only validation after the latest applied patch. Model `finish` text,
earlier passes, and patch application alone cannot establish success. Passing the
configured tests is not a proof of general correctness or full-suite coverage.

## Module map

Paths below are relative to `src/repopilot/` unless marked otherwise.

| Layer | Main paths | Responsibility |
|---|---|---|
| Entry/configuration | `cli.py`, `config.py`, `schemas.py`, `enums.py` | Explicit commands, validated settings, typed state and error contracts |
| Orchestration | `repair_workflow.py` | Build repository context, sandbox, tools and Provider; cleanup; verify and persist results |
| Agent | `agent/repair.py`, `agent/state.py`, `agent/budget.py` | LangGraph transitions, stage capabilities, bounded recovery, cumulative usage limits |
| Provider | `providers/base.py`, `providers/openai.py`, `providers/fake.py`, `providers/prompts.py` | Structured action contract, versioned prompt projection, usage and typed failures; deterministic offline fixtures |
| Tools | `tools/registry.py`, `tools/repository.py`, `tools/search.py`, `tools/symbols.py` | Schema validation and bounded read/search/inspection capabilities |
| Mutation/tests | `tools/patch.py`, `tools/patch_workspace.py`, `tools/tests.py`, `tools/output_projection.py` | Validated diffs, current-source reads, cumulative patch evidence, structured test outcomes and bounded previews |
| Retrieval | `retrieval/ingest.py`, `retrieval/corpus.py`, `retrieval/chunker.py`, `retrieval/text_chunker.py`, `retrieval/bm25.py`, `retrieval/dense.py`, `retrieval/hybrid.py` | Commit/content-bound source ingestion, chunks, sparse/dense search and deterministic RRF |
| Embeddings/storage | `embeddings/`, `storage/postgres.py` | Pinned local Jina model, content-addressed cache, pgvector namespaces; fake/in-memory implementations for tests |
| Context/Memory | `context/`, `memory/` | Budgeted evidence selection, provenance, snapshots, repository records and verified historical episodes |
| Sandbox | `sandbox/policy.py`, `sandbox/docker.py`, `sandbox/artifacts.py` | Pure policy decisions, Docker enforcement, task-local files and immutable artifact references |
| Verification/audit | `verifier.py`, `trace.py`, `replay.py` | Evidence-based outcome, hashed state/observation history and offline replay |
| Evaluation | `evals/runner.py`, `evals/isolated_tests.py`, `evals/single_shot.py`, `evals/reports.py` | Append-only attempts, hidden regression-test execution, one-shot control and offline metrics |

## Retrieval, context, and Memory are different layers

Repository ingestion records source identity and rejects incomplete listings rather
than silently indexing a truncated corpus. Python chunks use AST structure with
bounded line windows and a syntax-error fallback; supported documentation/config
text uses deterministic text chunks. Dense search uses the pinned Jina embedding
identity and pgvector; Hybrid combines BM25 and dense rankings with RRF. Initial
retrieval happens before Agent source mutation. The runtime `search_code` tool is
bounded literal search, not a new semantic retrieval or model call.

`ContextManager` selects evidence under per-source and total estimated-token
budgets, records deduplication/exclusion decisions, and uses deterministic
extractive summaries and artifact projections. Eligible `read_file`/`search_code`
outputs within the Registry's hard cap retain an artifact and a bounded preview.
Source-read previews use complete lines and a continuation position; a first line
too long to fit is rejected rather than silently truncated. The trace
records the selected context, transformations, prompt hash, and actual Provider
usage. A per-request context limit is distinct from the run's cumulative token
limit: repeated individually valid prompts can still exhaust the latter.

Memory is optional wiring, not a second Agent or an unconditional default:

- Working Memory stores versioned nonterminal state and its trace prefix for
  controlled recovery. It does not by itself snapshot a live Docker workspace.
- Repository Memory stores commit/content-bound repository facts and commands.
- Verified Episodic Memory admits artifact-backed successful runs; retrieved text
  remains historical evidence, not current-run proof or trusted instructions.
- Namespace bundles separate experiment/snapshot/split/access mode; holdout
  namespaces are read-only. Stale or expired records are not model context.

## Tool, execution, and evaluator boundaries

The Provider returns one typed tool action or a finish request. It cannot supply
host shell commands, Docker options, or success labels. The graph restricts tools
by stage; the registry validates arguments and call identity; each tool then
enforces its own path/output/mutation contract. `run_tests(test_index)` resolves a
trusted task-owned command rather than accepting model-generated command text.
The host application owns Provider credentials; they are not forwarded into task
containers.

Ordinary repair runs use one task sandbox. The evaluation adapter instead maintains
two isolated workspaces: the Agent receives the clean base plus its own edits; a
private evaluator receives the regression-test patch and mirrored candidate edits.
Only sanitized test summaries return to the Agent. Protected regression-test paths
are rejected before either workspace applies the candidate; their private list is
not exposed as model context. The expected solution patch is evaluator-only data,
not a repair input. See [the threat model](threat-model.md) for the exact scope and
limitations of these controls.

`trace.json` contains correlated before/after state, decisions, tool observations,
context accounting, and usage. `replay_trace` checks integrity, ordering,
correlations and state continuity without calling a Provider or executing tools.
Replay is an audit of recorded evidence, not a rerun of the model or tests. The
evaluation runner additionally verifies result/trace/artifact identities and keeps
append-only attempts; interrupted and technical failures are not erased.

## CLI versus the accepted experiment

| Entry | What it actually wires |
|---|---|
| `repopilot repair` | One local `TaskSpec`, explicit pinned image, configured initial retrieval (BM25 by default); Fake mode requires a scripted response file. It does not select a matrix, authorization, or isolated evaluator automatically. |
| `repopilot run-fixture` / `smoke` | Deterministic fixture/smoke paths; not evidence of real-model repair ability. |
| `repopilot replay` | Offline Agent trace audit only. |
| `scripts/p7_postdebug_runner_v6.py` | Frozen 40-slot, seen-task diagnostic execution: four Agent arms plus a same-initial-context single-shot control, sealed plan/authorization/source/input checks, prior-cost accounting, qualified images, isolated tests, and GPU-precomputed cache reuse. |

The four Agent arms are Dense, Hybrid-RRF, `no-semantic-rag` (BM25, **not no
retrieval**), and `mini-swe-safe` (no precomputed retrieval context, still using the
safe tool workflow). The single-shot control receives the same round's Hybrid
initial issue/summary/hits, makes one model call, and gets no tool loop or test
feedback. It is a comparison with different interaction and compute, not a
compute-matched causal estimate.

V6 explicitly waived another expensive corpus preflight; it did not waive source,
input, authorization, history, or cache-integrity checks. Its USD50 authorization
is cumulative and includes prior spend and unknown-cost reserves. That experiment
is complete, uses already-seen tasks, and must not be described as a new blind
holdout or the full SWE-bench benchmark. Current analysis is linked from
[aggregate-v6.md](../results/p7-holdout/aggregate-v6.md) and
[error-analysis.md](error-analysis.md).

## Deliberate limits

MCP transport and metadata-first/on-demand Skill loading remain deferred P8 work;
the current typed Tool Registry is not an MCP implementation. There is no
master/subagent delegation, unrestricted terminal tool, production multi-tenant
service, or learned verifier. Docker isolation, local hash verification, prompt
trust labels, and test success each have narrower guarantees than those features
would require. Deployment and privacy limits are documented in the threat model.
