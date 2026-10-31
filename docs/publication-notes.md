# Public snapshot boundaries

This update continues the existing public history rooted at
`309b67a41a576394826a3aec32fbde2adc145f4f`; it does not merge or publish the internal
development repository's history. The existing MIT license and upstream license
texts are retained. No force push or replacement root is needed.

## Included

- Current Python source, scripts, tests/fixtures, configs, migrations, Compose file,
  locked dependencies and no-secret/no-paid-service CI.
- Frozen dataset/Memory manifests and selected historical controls and reports.
- Final v6 aggregate, per-slot export, failed-case inventory, receipts and final report.
- English/Chinese README, architecture/evaluation/safety documentation and notices.

## Excluded

- Internal Git history, project handoff/phase files and chronological development log.
- Private study/interview notes, the local `github_submission/` directory and bundles.
- Real `.env*` files, credential/private-key files, caches, virtualenvs and model weights.
- Local `artifacts/`, GPU caches, raw holdout runs and raw `results/p7-dev/p7-dev-*`
  execution directories. Development spending authorizations remain omitted.

Selected historical holdout controls retain exact bytes because their hashes and
relationships are regression inputs. They contain historical approvals and non-secret
image/project identities, not credentials or permission to spend now. The sample
database password `repopilot` is only a local development default. Do not expose that
database publicly or reuse its defaults for a deployed service.

## Verification and history

The public checkout does not contain private development commit objects. A portable
regression reconstructs the exact pre/post archive-metadata patch in a temporary Git
repository and checks both blob identities and the full patch digest. A second portable
test verifies rejection of additional source-path drift. The original-commit negative
regression is preserved but explicitly skips when those commits are unavailable.
Production readiness checks have not been weakened or bypassed.

The requalification unit-test fixture uses sealed recorded object bindings for its
existing mocked Git boundary. The historical object comparison and GPU qualification
source comparison are separate optional checks, not fabricated successes. Temporary
real-Git source-drift tests still execute without private history.

Other raw-evidence-dependent tests explicitly skip when their archives are absent.
Frozen controller source-ancestry checks require the retained development checkout.
Offline aggregate/failed-case reconstruction instead uses the current reporting code,
frozen inputs and separate original raw archive; it does not need to execute the
historical controller or contact a Provider. Read the
[evaluation guide](evaluation.md) before interpreting a summary as a rerun guarantee.
Public test counts and the exact scan scope are recorded in [validation](validation.md).

## Evidence and licensing boundaries

Issue excerpts and evaluator-only patches remain upstream material; do not feed
`source-rows.json` or `regression-tests-v1.json` to Agent context or retrieval.
[Third-party notices](../THIRD_PARTY_NOTICES.md) identify their origins and preserved
license texts. Summaries and receipts can point to excluded raw evidence without
promising that those files are included in this checkout.

No test transcript, regex scan or hash proves the absence of every secret or every
security flaw. Publication checks are scoped safeguards, not certification. No new
video, model request, cloud deployment, cleanup or billing change is part of this update.
