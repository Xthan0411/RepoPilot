# P7-S6 formal command-timeout contract

## Purpose

The frozen holdout tasks declare individual setup/test command timeouts between 120 and
900 seconds. The frozen P7-v2 experiment matrix independently limits every Sandbox
command to 300 seconds. RepoPilot's production Sandbox correctly denies a command whose
declared timeout exceeds that policy ceiling, so using the source tasks directly would
turn evaluator configuration drift into 17 deterministic policy denials before the
Agent could be evaluated.

The formal contract resolves that mismatch without changing the frozen dataset, Agent,
Prompt, command argv, command cwd, task-wide timeout, model budget, or Sandbox policy:

```text
effective_command_timeout = min(declared_command_timeout, 300 seconds)
```

The projection is evaluator control-plane data. All four enabled arms receive the exact
same projected task, and the projection is content-addressed before any paid request.

## Why this boundary

Two simpler-looking alternatives are rejected:

- raising `sandbox_command_timeout_seconds` to 600/900 would change the frozen P7-v2
  matrix after dev calibration;
- editing `data/manifests/p7-v1/manifest.json` would change the frozen dataset identity
  and task-set hash.

The chosen projection preserves the source artifacts byte-for-byte and applies the
already-qualified behavior used by the 8/8 x86 Sandbox preflight. The Agent prompt only
contains the unchanged task-wide timeout and command counts; individual command timeout
metadata is not rendered into the Prompt.

## Exact projection

| Task | Task-wide timeout | Setup declared → effective | Test declared → effective |
| --- | ---: | --- | --- |
| `django__django-15902` | 1800 | 600→300, 600→300 | 600→300 |
| `mwaskom__seaborn-3190` | 1800 | 900→300, 900→300 | 300→300 |
| `pallets__flask-4992` | 900 | 600→300, 600→300 | 120→120 |
| `pydata__xarray-4094` | 1800 | 900→300, 900→300 | 300→300 |
| `pydata__xarray-5131` | 1800 | 900→300, 900→300 | 300→300 |
| `pytest-dev__pytest-11143` | 1200 | 600→300, 600→300 | 300→300 |
| `pytest-dev__pytest-11148` | 1200 | 600→300, 600→300 | 300→300 |
| `sympy__sympy-20590` | 1200 | 600→300, 600→300 | 300→300 |

There are 24 commands in total. Seventeen are capped and seven already fit the policy.
The task-wide timeout remains unchanged; the formal Agent wall-clock budget remains the
separate frozen matrix value of 900 seconds.

## Inputs and output

`scripts/p7_bind_holdout_environments.py` consumes only committed or already-qualified
control artifacts:

| Input | Required identity |
| --- | --- |
| Matrix | `configs/experiments/p7-v2.yaml`, content SHA-256 `d75d36d711f23190062aded9e97a8f5abfaba859416466027b227421ea7407e8` |
| Dataset | `data/manifests/p7-v1/manifest.json`, content SHA-256 `89496bc2200149601333dadb9a6c7ac6b7d660b4bc7acf29c23597d33b660f60` |
| Compatibility | `sandbox-matrix-preflight-v1.json`, raw SHA-256 `9504aed7904436f807d890077f927dc3a4c0a6cfa890d383811535c2c8889c1c` |
| Registry | `registry-manifest-v1.json`, raw SHA-256 `edc0b94b7ec8247e00e2f36c80d5f930e1dc7141602da9e6514f55338841fef9` |
| Checkouts | eight clean exact-base repositories under `artifacts/p7-holdout/checkouts/` |

The create-only output is `results/p7-holdout/environments-v1.json`. It binds:

- exact matrix, dataset, compatibility, registry, harness, controller, checkout, Python,
  and image-digest identities;
- declared/effective timeout and cap decision for every command;
- a per-task timeout projection SHA-256;
- source and projected-runtime Gold-free command-contract SHA-256 values;
- offline setup, baseline-fail, evaluator-Gold-pass, no-network, and Gold-isolation
  evidence;
- `formal_environment_ready=true`, empty `formal_blockers`, and zero paid/Provider
  requests.

## Runtime configuration and environment variables

The binder must run on the qualified Linux amd64 VM from a clean tracked controller
checkout. The fixed runtime configuration is:

- command cap: 300 seconds;
- setup network: `none`;
- Agent/test network: `none`;
- image references: task-specific immutable Artifact Registry digests;
- output mode: private `0600`, create-only JSON.

No environment variable is required. The binder does not read `.env`, source rows, or
evaluator Gold patches; request a cloud token; call Artifact Registry; pull/build an
image; invoke OpenAI/Jina; access pgvector; run a task; or apply evaluator Gold. It
therefore has no Provider or registry mutation cost.

## Validation and invalid configurations

Before creating the artifact, the binder verifies:

1. Linux amd64 host, clean tracked controller, and absent output path.
2. Frozen matrix cap is exactly 300 and network is exactly `none`.
3. Exact raw/content identities for the compatibility and registry artifacts.
4. Exact eight-task order across dataset, compatibility, and registry evidence.
5. Per-task base commit, Python, local adapter identity, and remote digest identity.
6. Clean checkout commit and path containment.
7. Every recorded preflight declared/effective timeout equals the new formal projection,
   for setup, baseline, and Gold observations.
8. No evaluator Gold field is serialized into the environment artifact.

Invalid configurations include raising or lowering the cap, enabling setup network,
editing either source artifact, reordering tasks, substituting an image or checkout,
changing a command timeout without new qualification, supplying a dirty checkout, or
attempting to overwrite an existing environment manifest.

## Stable termination reasons

Success exits `0` and prints:

```text
holdout_environment_binding_passed=true tasks=8 registry_digests=8 timeout_cap_seconds=300 paid_requests=0 formal_environment_ready=true blockers=0
```

Failure exits `1`, writes no artifact, makes no paid request, and prints one stable code:

| Code | Meaning |
| --- | --- |
| `invalid_configuration` | CLI/root or typed configuration is invalid. |
| `host_platform_mismatch` | Host is not Linux amd64. |
| `controller_source_dirty` | Tracked controller files differ from the binding commit. |
| `source_artifact_invalid` | A required artifact is missing, malformed, or byte/content mismatched. |
| `source_identity_mismatch` | Matrix, dataset, task, Python, image, or contract identity differs. |
| `checkout_unavailable` | A required checkout cannot be opened as a Git repository. |
| `checkout_identity_mismatch` | Checkout escapes the root, is dirty, or is at the wrong commit. |
| `timeout_contract_mismatch` | Qualified declared/effective command evidence differs from the formal projection. |
| `output_conflict` | The create-only environment manifest path already exists. |

## Current checkpoint

The qualified x86 VM created the formal artifact from clean controller commit
`f07fb6dc44131d966431d51c631f5eae353c60c6`. The committed
[`environments-v1.json`](../results/p7-holdout/environments-v1.json) is byte-identical to
the VM output, with raw SHA-256
`b7c6f04283433859e462edb7105ad5411700703fa3557b1281fafc3ccb99890d`
and validated content SHA-256
`5b6135dba064954ac163db3dd3a2d3e543a356856193390692666a638995154d`.
Its typed summary records 8 bindings, 17 capped commands, `formal_environment_ready=true`,
empty blockers, zero paid requests, and no Agent-visible Gold. The VM create-only file
had mode `0600`; version control retains the evidence bytes but cannot represent private
read/write bits independently of ordinary non-executable file mode.

The environment gate is complete. The separate USD 32 authorization and
Provider/Jina/pgvector runtime readiness subsequently passed, and the fail-closed paid
32-slot runner is implemented. This environment artifact alone grants none of those
permissions; the formal controller validates their exact identities independently.
