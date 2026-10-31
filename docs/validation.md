# Public release validation

P7-S10 closes the agreed core MVP with reports and publication checks. The user
waived a demo video and requested no additional model experiment. This validation
does not change any accepted v6 score, frozen control, model prompt or runtime code.

## Environment and scope

The candidate was built in a separate clone of the existing public repository,
rooted at `309b67a41a576394826a3aec32fbde2adc145f4f`. It has an independent virtualenv
and no private development Git history, credentials, raw execution directories,
model weights or GPU cache. Installation used `uv sync --locked --offline` with
already cached Python 3.12.12/packages on Darwin arm64, uv 0.9.26. This is not a new
OS, an empty-cache installation or a GitHub-hosted CI result.

The selected 278 source/test/script/config/migration/CI/packaging inputs have the
canonical inventory SHA-256
`ff94707d160e1fc8f0892d405bfdac108974bd99f43c71de091986c5173f02c5`.
The inventory is sorted `{path, sha256}` objects, serialized as compact JSON with
sorted keys. Scope: `src/`, `tests/`, `scripts/`, `configs/`, `migrations/`,
`.github/`, `pyproject.toml`, `uv.lock`, `compose.yaml`, `.python-version`.
This is an input fingerprint, not a signature or a hash of every release file.

## Checks

The exact final counts and commands are recorded in the
[machine receipt](../results/p7-release/release-check-v1.json).

- Ordinary offline tests: **2,112 passed, 13 skipped, 21 deselected in 54.95s**;
  excludes `integration`, `real_model` and `eval` markers.
- Ruff lint and formatting: passed; 246 Python files formatted.
- Strict mypy: passed for 89 source files.
- `uv lock --check --offline`: passed; 104 locked packages resolved. The check
  required access to the existing host uv cache but made no dependency download.
- Both offline v6 report builders reproduced the accepted 40-slot aggregate and
  20-slot failure inventory from the separately retained original raw archive.
  Raw evidence was supplied outside the public checkout and is not published.
- Relative documentation links, exact upstream license blobs, candidate paths,
  file types/sizes and suspicious credential patterns were checked.

No live Provider request, paid experiment, Docker test, embedding-model load,
cloud deployment, billing change or resource cleanup was performed in this slice.
Some unit tests execute only trusted, synthetic fixtures on the host; they do not
establish Docker isolation or new autonomous-model capability.

### Explicit skips and test-only portability changes

Ten tests need excluded original evidence archives. Three further tests compare
against private historical Git objects and explicitly skip in the public clone.
They are not counted as passed. Infrastructure/model marker exclusions are also
separate from these skips.

The first public test run exposed fixtures that assumed private history existed.
Only the test fixtures were adapted: portable exact-patch and source-drift tests
use temporary Git repositories; requalification fixtures validate the recorded
schema, content seal and object bindings at the already mocked Git boundary.
Historical object comparisons remain separate tests and run when their commits
are available. The production readiness/source-freeze gates were not weakened.

The earlier [S9 receipt](../results/p7-clean-run/receipt-v1.json) remains immutable:
its 2,112 passed / 10 skipped / 21 deselected result belongs to the full-history
checkout described there, not this curated public snapshot.

## Evidence index

| Claim | Retained evidence |
|---|---|
| Architecture and trust boundaries | [Architecture](architecture.md), [threat model](threat-model.md) |
| All five arms and 40 planned slots | [Aggregate JSON](../results/p7-holdout/aggregate-v6.json), [readable report](../results/p7-holdout/aggregate-v6.md) |
| Execution inputs/source identity | [v6 plan](../results/p7-holdout/postdebug-plan-v6.json), [acceptance receipt](../results/p7-holdout/postdebug-receipt-v6.json) |
| All 20 unsuccessful slots | [Failure inventory](../results/p7-holdout/error-analysis-v6.json), [five cases](error-analysis.md) |
| Scripted/local-fixture reproduction | [S9 receipt](../results/p7-clean-run/receipt-v1.json) |
| Final interpretation and limits | [Final report](final-report.md), [evaluation guide](evaluation.md) |
| Publication and attribution | [Snapshot boundaries](publication-notes.md), [third-party notices](../THIRD_PARTY_NOTICES.md) |

Private development commit IDs in historical records identify retained evidence;
they are not promises that those objects are in the public Git history. Summary
inspection, raw-evidence reconstruction and a new paid experiment have different
prerequisites. The last requires separate credentials, infrastructure and authority.

## Repeat the ordinary checks

Use a shell without model/database/tracing overrides; no API key is needed.

```bash
uv sync --locked
uv lock --check
uv run pytest -q -rs -m "not integration and not real_model and not eval"
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy src/repopilot
```

The first two commands can download packages when caches are empty. The configured
GitHub workflow runs the ordinary checks with no model credentials. Its actual
status must be read from the published commit's Actions run; local validation is
not a claim that remote CI has passed.

Publication scanning found no unresolved credential matches within its reviewed
scope. Known test sentinels and the documented local database default are not live
secrets. A regex scan does not certify the absence of all secrets, security flaws
or licensing obligations. Cloud resource IDs retained in immutable evidence are
non-secret identifiers, not credentials.
