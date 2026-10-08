# RepoPilot P7 Dataset v1

This frozen set contains two checked-in development fixtures and eight public holdout
Issues selected from SWE-bench Lite. It is a small project evaluation set, not a claim of
statistical representativeness or a replacement for the complete benchmark.

## Source identity

- Dataset: [SWE-bench Lite](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite)
- Frozen dataset revision: `69611d31007e1c6731db8bd5b5c3f2d33f5bab6e`
- Dataset card SHA-256:
  `01d69da06269ac911e8d716a868838d53c9bf0a16b9b2fee9097cb5c84bff3d3`
- Citation: Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World
  GitHub Issues?*, ICLR 2024.
- SWE-bench implementation license: MIT. The dataset card does not declare one blanket
  license for all upstream repository content, so every selected case records the
  repository license observed at its exact base commit.

`source-rows.json` is a canonical snapshot containing only the eight reviewed upstream
rows. It contains evaluator-only gold patches and must never be placed in Agent Context,
Provider requests, repository indexes, or prompts. `load_frozen_dataset()` projects the
inner `TaskSpec` separately from each `EvaluationTaskSpec`.

## Frozen cases

| Split | Case | Repository license | Evidence status |
|---|---|---|---|
| dev | `p5-config-parser-blank` | RepoPilot MIT | checked-in verified fixture |
| dev | `p5-parser-service-async` | RepoPilot MIT | checked-in verified fixture |
| holdout | `django__django-15902` | BSD-3-Clause | metadata verified |
| holdout | `mwaskom__seaborn-3190` | BSD-3-Clause | metadata verified |
| holdout | `pallets__flask-4992` | BSD-3-Clause | baseline failed, gold passed |
| holdout | `pydata__xarray-4094` | Apache-2.0 | metadata verified |
| holdout | `pydata__xarray-5131` | Apache-2.0 | metadata verified |
| holdout | `pytest-dev__pytest-11143` | MIT | metadata verified |
| holdout | `pytest-dev__pytest-11148` | MIT | metadata verified |
| holdout | `sympy__sympy-20590` | BSD-3-Clause | metadata verified |

For every public case, GitHub metadata confirms the solution PR is merged and its base
SHA equals the frozen SWE-bench `base_commit`. Repository license file paths, Git blob
SHAs, pinned URLs, setup commands, test commands, Python versions, gold files, and
FAIL_TO_PASS identities are stored in `manifest.json`.

## Preflight scope

Flask 4992 was cloned at its exact base commit. Its upstream test patch was applied, then
the focused test was run in a read-only-rootfs, resource-limited, digest-pinned
`python:3.12-alpine` container:

1. Base + test patch: exit 1 with the expected unexpected-`text`-argument failure.
2. Base + test patch + gold patch: exit 0, one test passed.

The normalized setup transcript and raw focused pytest outputs are integrity-referenced
from `results/p7-dataset-v1/preflight-report.json`. The other seven public cases are
metadata-frozen but have not yet been executed locally. They must not be reported as
reproducible successes until a later preflight upgrades their evidence status.

Trusted setup may use network access only to acquire the exact public commit and install
declared dependencies. Agent execution remains governed by the existing Sandbox network
policy. P7-S2 performs no model call and does not expose holdout gold to the Agent.

## Exclusions

`candidate-decisions.json` retains every included candidate plus explicit exclusions.
The first set excludes an Astropy task for compiled-stack cost, a Matplotlib task for
large native/system setup, and an old Requests task whose tests require an HTTP service
and a legacy environment. Excluded candidates are not silently dropped and do not enter
task-count denominators.
