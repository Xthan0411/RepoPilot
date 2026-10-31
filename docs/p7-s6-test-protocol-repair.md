# P7-S6 test-protocol repair and requalification

## Current status — 2026-10-28

The local code repair and replacement-VM eight-task protocol acceptance passed. The
original report is retained byte-for-byte and independently checked. Formal paid
Holdout remains **0/32**. Historical August v1 environment,
readiness, freeze, plan, and authorization artifacts are retained unchanged. They do
not authorize the repaired source or prove that the replacement VM is ready.

The replacement VM restored all eight digest-pinned images and clean base checkouts.
Its ordinary regression suite passed **1256 tests, 21 skipped**; those skips are not
substitutes for the separately executed eight-task canary. Jina, pgvector, and Provider
credential readiness on the replacement VM have not yet been accepted.

## Why the old qualification was insufficient

Two code-path defects were found before paid execution:

1. The test-result interpreter recognized unittest output, but seven Holdout tasks use
   pytest. A process exit code alone was not sufficient evidence for the production
   Verifier; valid pytest failures/passes were not interpreted correctly.
2. The historical environment preflight applied each task's regression `test_patch` in
   a temporary evaluator. The formal runner instead supplied a clean base checkout to
   its test tools. Selectors for newly introduced regression tests could therefore
   fail to collect, rather than reproduce the actual bug.

This was a gap between an environment check and the actual Agent → test tool → Verifier
path, not an LLM quality or Prompt problem. Local E2E fixtures exercise the repair graph;
the real eight-task canary exercises the production test tool and the Verifier's evidence
helper, without constructing an Agent or Provider.

## Implemented boundaries

### Result interpretation

`src/repopilot/tools/tests.py` selects pytest parsing from the declared command, checks
the final terminal summary against the exit code and failure/error rows, and handles
ANSI output and parameterized node IDs. Collection/import/usage errors, truncated
output, no tests, and all-skipped/all-xfailed output cannot count as a successful repair.
Unittest parsing remains supported.

`tests_run` counts completed unconditional outcomes: passed + failed + errors. It is
not a unique-node or collected-test count; a passed test with a teardown error has two
outcomes. Mixed passed/skipped output follows ordinary pytest success semantics. This
repair does not implement the complete upstream SWE-bench FAIL_TO_PASS/PASS_TO_PASS
grader or make arbitrary test-runner stdout a tamper-proof attestation.

### Separate Agent and evaluator sandboxes

`src/repopilot/evals/isolated_tests.py` supplies the same wrapper to all four arms:

- Agent file/retrieval tools see the clean base plus the candidate patch, never the
  added regression patch or a reference solution.
- A private temporary checkout and separate Docker sandbox receive the regression
  patch. Only exact declared setup/test commands are allowed; runtime network remains
  disabled. Repository code is not executed on the host.
- Setup is performed in both sandboxes. Candidate patches are mirrored to both;
  changes overlapping hidden regression paths are rejected. A failed mirror poisons
  the session so it cannot later claim a valid test result.
- Raw evaluator logs remain private. The Agent and public Trace/artifacts receive a
  reconstructed summary, counts, and sanitized failure IDs, not hidden assertion
  bodies or traceback text. Parameter values are hashed. Infrastructure failures stay
  errors, although their fine-grained category can conservatively collapse to
  `execution_error` after sanitization.
- Cleanup removes temporary evaluator checkouts and raw logs. Public, hash-verified
  sanitized artifacts remain available for replay and Verifier checks.

### Test-only input projection

`scripts/p7_holdout_test_inputs.py` generated
`results/p7-holdout/regression-tests-v1.json`, bound to the exact eight task/base commits,
Agent projection, and dataset/source hashes. Its raw SHA-256 is
`eeff30648f37afe50e512e48a7c828b6eb8149652161c266d55f81e46818fb4e`.
It contains only regression test patches, not reference solution patches, and is not
Agent context. The formal runtime reads this bounded, hash-checked projection; only
the explicit evaluator-side generator reads the original dataset rows.

## Replacement-VM acceptance — accepted

The report at `results/p7-holdout/test-protocol-canary-v2.json` is 8,657 bytes:

- Raw SHA-256: `732ae6863bc3d28f9a5c584e775672e97c87f9844026ac83df0f299e907fae46`.
- Content SHA-256: `327ae01c9ff1a0e45af53b4f9f2c5b76d0dd4a57e517a537edaba6e3666c1759`.
- Qualified runtime: `19cb83dd0370c4f450319040fe604f39cd002b3c`.
- Checked at `2026-10-29T02:40:11.410410+00:00` (October 28 in New York).

| Task | Baseline outcomes | Failures / test errors | Gold passing outcomes |
|---|---:|---:|---:|
| django-15902 | 157 | 0 / 1 | 157 |
| seaborn-3190 | 1 | 1 / 0 | 1 |
| flask-4992 | 1 | 1 / 0 | 1 |
| xarray-4094 | 1 | 1 / 0 | 1 |
| xarray-5131 | 34 | 10 / 0 | 34 |
| pytest-11143 | 1 | 1 / 0 | 1 |
| pytest-11148 | 2 | 2 / 0 | 2 |
| sympy-20590 | 1 | 1 / 0 | 1 |
| Total | 198 | 17 / 1 | 198 |

All baseline exits are 1 and all Gold exits are 0, with no reported truncation.
Django's completed `test_error` is an accepted failing baseline, not an infrastructure
`status=error`. These are completed outcome counts, not necessarily unique test nodes.
The sanitized report is not signed remote attestation: it does not retain the raw
output/observation references needed to independently replay every Verifier decision.
The raw digest, schema, task/base/projection bindings, and count/status consistency are
validated; its zero Provider/paid facts are recorded evidence, not an Agent benchmark.

The following command documents how this evidence was produced. Do not rerun a completed
canary just to regenerate its already retained create-only artifact.

After restoring the exact clean checkouts, dependencies, and eight digest-pinned
images on a Linux amd64 VM, run the zero-provider qualification command:

```bash
cd ~/work/repopilot
uv run python -m scripts.p7_test_protocol_canary \
  --repository-root ~/work/repopilot
```

No OpenAI key is required for this command. It constructs no Agent or Provider. Gold
is used only as a known-good candidate inside temporary evaluator-owned resources.
Baseline and Gold results go through the production `RunTestsTool` and Verifier
evidence checks. Each task must have a genuine baseline test failure and valid passing
Gold results; missing tests or setup failures cannot qualify.

The command checks all requested tasks and emits safe structured diagnostics rather
than hidden test/Gold output. Optional `--task-id` selections are diagnostics only and
write no full acceptance artifact. A successful full eight-task run creates, without
overwriting, `results/p7-holdout/test-protocol-canary-v2.json`.

Even that report records `formal_environment_ready=false` and
`new_source_freeze_and_authorization_required`. It does not qualify Jina, pgvector,
Provider credentials, or a new paid execution plan.

## Completion boundary and next work

Local tests exercise a real trusted pytest fixture through the repair graph, including
correct/wrong patches, missing regression tests, collection errors, and a hidden
assertion marker that must not enter Provider requests, Trace, or public artifacts.
They do not substitute for the eight actual task images on Linux amd64.

The initial **1254 passed, 21 skipped** validation was pre-commit. A clean replacement-VM
clone at `1f70118` exposed a historical readiness test that used moving `HEAD` and wrongly
expected the legacy freeze to authorize the newly committed repair. This failure also
reproduces locally, so it is not a VM compatibility issue. The positive regression now
uses the exact controller in retained v1 readiness; a new real-Git negative regression
proves that v1 rejects the pinned repair commit. CI fetches complete Git history for
these checks. The runtime guard and historical artifact bytes are unchanged.

With this test/CI correction, working-tree validation is **1256 passed, 21 skipped**;
Ruff check/188-file format check and strict mypy (84 source/test files) pass. The delivery
gate additionally requires rerunning the suite in a fresh clone of the final committed
bundle, not just the pre-commit working tree. The 21 skips require explicit Docker
image/database configuration and do not qualify the eight real task environments.
The canary also requires its executed modules and test projection to be committed and
byte-identical to HEAD; unrelated untracked notes/bundles are tolerated.

The parser and evaluator changes affect test semantics and are **outside** the old
archive-mtime-only compatibility exception. The old runtime source guard remains in
place and rejects these source changes; no hash, freeze, or authorization was weakened.
Before any paid run: qualify the replacement VM and eight-task protocol, version and
freeze the repaired source/test protocol, create a newly bound plan/readiness, and
obtain explicit authorization for that plan. Do not overwrite v1 evidence or reuse its
USD 32 approval against changed identities. After all 32 formal slots reach retained
terminal outcomes, P7-S7 can aggregate the actual results.

## Post-repair freeze and plan

`scripts/p7_holdout_requalification.py` accepts only the original canary raw digest and
the qualified `19cb83d` source. It rechecks completed-test evidence, exact task/base/test
patch bindings, historical freeze/calibration, unchanged matrix/lock/dataset, and the
runtime Git tree/blob inventory. The semantic scope includes `scripts/__init__.py` and
the shared protocol/input/control helpers. Dirty tracked files, untracked semantic
files, source drift, symlink inputs, oversized input, and existing output are rejected.

The separate create-only outputs `post-repair-freeze-v2.json` and `plan-v2.json` in
`results/p7-holdout/` were generated and reloaded successfully after builder commit
`5fce45371e6b408b5201cabaabeb0f44e8887078`. Plan ID is
`p7-holdout-v2-d26d33699d59b6fc`, content SHA-256
`3660cc5962ab8810b9b74fab1c4820188340111d7798e0ef4c620f8d1aafb3c0`.
They do not overwrite v1 artifacts. The new freeze explicitly
records evaluator-only Holdout qualification and `current_source_calibration_performed=false`;
the historical dev calibration still belongs to its original source commit. It does
not copy the historical `holdout_accessed=false` claim into the new evidence.

The v2 plan retains four arms × eight tasks × one repetition and the USD 1/task,
USD 32/application planned ceilings, but uses new UUIDs/output directories. It is a
declaration, not authorization. `load_requalification` rebuilds all bindings and slots
from the pinned inputs, rejecting self-rehashed mismatches. The qualified runtime
commit is distinct from the freeze/plan builder commit. Later readiness/runner adapters
must bind their own executed source identity and cannot claim that they ran in this canary.

The user has now explicitly approved that exact v2 plan at USD 32, including final
in-flight response overshoot and mandatory readiness before paid execution. Separate
v2 authorization/readiness/runner controls bind that approval, all executed scripts,
the qualified runtime, and each retained input. They reuse the offline embedding/vector
probes and append-only scheduler, add a plan-level cross-process lock, and keep unknown
interrupted/technical cost reserved. Old consent, changed bindings, missing dependencies,
and budget exhaustion fail closed. The default runner is a dry-run; paid mode defaults
to one new slot. A final-slot overshoot retains the complete report before stopping.

Next: follow [the v2 execution guide](p7-s6-v2-execution.md) to transfer the verified
package and run new-VM readiness plus dry-run with zero Provider requests. Only after
all runtime gates pass may the authorized paid slots begin. Local fake-backed tests
do not qualify the VM. The requalification command itself still creates no authorization,
calls no Provider, and makes no cloud mutation.
