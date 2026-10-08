# P7-S6 immutable registry publication

## Purpose and current boundary

The Sandbox matrix proves that eight exact local Docker config image IDs can execute the
frozen holdout tasks on `linux/amd64`. A local image ID is not a portable execution
reference, so formal holdout execution remains blocked until every qualified image is
published and verified through an immutable registry manifest digest.

`scripts/p7_publish_registry_images.py` implements that boundary without invoking an
Agent, model Provider, task setup, tests, pgvector, Jina, or evaluator Gold. Its only
external mutation is publishing the eight already-qualified images to one private
Artifact Registry repository.

The fixed target is deliberately narrow:

| Field | Required value |
| --- | --- |
| Project | `repopilot-504912` |
| Location | `us-west1` |
| Repository | `repopilot-p7-s6` |
| Registry root | `us-west1-docker.pkg.dev/repopilot-504912/repopilot-p7-s6` |
| Repository format/mode | Docker / Standard |
| Tags | immutable |
| Vulnerability scanning | disabled |
| Active principal | `repopilot-eval-publisher@repopilot-504912.iam.gserviceaccount.com` |
| Credential source | attached Compute Engine service account metadata |
| Per-image tag | `p7-s6-v1` |
| Output | `results/p7-holdout/registry-manifest-v1.json` |

The service account has repository-level Artifact Registry Writer rather than project
Owner, Editor, or repository administrator. No service-account key exists.

## Inputs and identity checks

The publisher consumes the committed
`results/p7-holdout/sandbox-matrix-preflight-v1.json`. Its raw SHA-256 is fixed as
`9504aed7904436f807d890077f927dc3a4c0a6cfa890d383811535c2c8889c1c`. The script
revalidates the typed artifact, exact eight-task order, zero Python mismatches, zero paid
requests, and Gold isolation before inspecting Docker.

It then requires:

- a clean tracked controller checkout;
- a `linux/amd64` host;
- no existing output artifact;
- the exact metadata and active `gcloud` principal;
- the exact repository resource name, format, mode, immutable-tag setting, and disabled
  scanning state;
- all eight exact local config image IDs with `linux/amd64` platform and positive size.

No environment variable is required. Before invoking `gcloud` or Docker, the subprocess
environment removes API-key, token, secret, password, credential, and `DATABASE_URL`
values so unrelated Provider or database secrets cannot cross this boundary.

## Read-only and mutating modes

The default command is read-only:

```bash
uv run python -m scripts.p7_publish_registry_images \
  --repository-root ~/work/repopilot
```

It performs all local, identity, and repository checks but does not request a metadata
token, authenticate Docker, tag an image, push an image, or create an artifact. Success
prints:

```text
registry_publish_preflight_passed=true images=8 execute=false external_mutations=0 paid_requests=0
```

Mutation requires both `--execute` and an exact typed confirmation:

```bash
uv run python -m scripts.p7_publish_registry_images \
  --repository-root ~/work/repopilot \
  --execute \
  --confirm-registry \
  us-west1-docker.pkg.dev/repopilot-504912/repopilot-p7-s6
```

An incorrect, absent, or unnecessary confirmation fails before registry mutation.

## Authentication lifecycle

Execution retrieves a short-lived OAuth token from the Compute Engine metadata server.
The token is passed only through Docker login stdin and is never printed, persisted in an
artifact, placed in a command argument, or copied into a container. Docker receives an
isolated temporary `DOCKER_CONFIG`, so the publisher neither consumes nor overwrites the
operator's normal Docker credentials. It always attempts logout before the temporary
directory is removed, including expected publication failures.

If a token expires during a long multi-image upload, the run fails without an output
artifact. A rerun receives a new token and resumes from verified remote tags.

## Publication, digest verification, and resume

For each frozen task in dataset order, the publisher derives the registry image name
from the same validated task-to-safe-name mapping used by the Sandbox preflight. It never
accepts an operator-supplied image ID, image name, tag, or digest.

Before pushing, it attempts to pull the immutable target tag:

- if the tag is absent, it tags the exact local config image ID, pushes it, pulls it back,
  and records `publication_outcome=published`;
- if the tag exists, it pulls it and accepts it only when the remote image resolves to
  the exact qualified local config image ID, recording
  `publication_outcome=already_present`;
- if the existing tag resolves to any other config image ID, it stops with
  `remote_tag_conflict`. It never tries to move, delete, or overwrite an immutable tag.

After a successful pull, the script extracts exactly one repository manifest digest for
the task image and creates the portable `...@sha256:...` reference. The final manifest
must contain eight unique local config IDs, eight unique registry manifest digests, exact
task order, successful remote pulls, zero Provider/paid requests, and no Agent-visible
Gold.

Remote publication may be partially complete if a later task fails. That state is
intentional and recoverable; no remote image is automatically deleted. The create-only
manifest is written only after all eight images pass.

## Stable failures and invalid configurations

Registry publication is evaluator infrastructure, not an Agent run. It exits `1` with a
stable `registry_publish_failed=true code=...` marker and never adds a failure to the
Agent evaluation denominator.

| Code | Meaning |
| --- | --- |
| `invalid_confirmation` | Execute confirmation is absent/wrong, or supplied in read-only mode. |
| `host_platform_mismatch` | Host is not Linux x86_64/amd64. |
| `controller_source_dirty` | Tracked repository files differ from the publisher commit. |
| `source_preflight_invalid` | The committed 8/8 source artifact is missing, changed, or invalid. |
| `output_conflict` | The create-only registry manifest path already exists. |
| `metadata_unavailable` | Attached service-account identity or short-lived token is unavailable/invalid. |
| `identity_mismatch` | Metadata or active `gcloud` principal is not the fixed publisher account. |
| `repository_invalid` | Project/location/name/format/mode/tag/scanning policy differs from the contract. |
| `local_image_unavailable` | An exact qualified local image cannot be inspected. |
| `local_image_identity_mismatch` | Docker resolved a different local config image ID. |
| `local_image_platform_mismatch` | A local image is not `linux/amd64`. |
| `authentication_failed` | Short-lived Docker login failed. |
| `remote_probe_failed` | Remote tag pull failed for a reason other than explicit absence. |
| `remote_tag_conflict` | An immutable tag already points to a different config image. |
| `publish_failed` | Local tagging or remote push failed. |
| `remote_digest_invalid` | Pullback did not yield one exact registry manifest digest. |
| `logout_failed` | Isolated Docker credential logout failed. |
| `command_unavailable` | A required local command could not be invoked or timed out. |

The script never enables scanning, creates a repository, changes IAM, changes the frozen
Agent/task/Prompt/Gold/success rule, accepts a mutable tag as a formal reference, or
deletes a remote artifact.

## Result and remaining gate

Successful execution creates a SHA-256-protected
`p7-registry-manifest-v1` artifact and prints:

```text
registry_publish_passed=true images=8 immutable=true credential=compute_engine_metadata paid_requests=0 formal_ready=false blockers=1
```

The formal execution completed from publisher commit
`a5c43fdc9e3f2c33db0ab0e6f7af3a5cd46dd170`: all eight images were newly published,
pulled back, and accepted with zero Provider requests. The durable artifact is
[`registry-manifest-v1.json`](../results/p7-holdout/registry-manifest-v1.json), with raw
SHA-256 `edc0b94b7ec8247e00e2f36c80d5f930e1dc7141602da9e6514f55338841fef9`
and validated content SHA-256
`5f52d0aac381330342daf1dc52bcf09961d18282f9ee391fe005a73627e27b8d`.

All eight reported digests were then queried independently through the Artifact Registry
control plane and matched the artifact with status zero. The apparent equality between
local `.Id` and registry digest was also audited: the VM ran Docker Engine 29.7.2 with
`io.containerd.snapshotter.v1`, so the local containerd image descriptor and remote
manifest identity were represented by the same digest in this environment.

`registry_digest_required` is satisfied. The subsequent x86 environment binding also
resolved `task_command_timeout_contract_mismatch`; the formal environment now has eight
immutable digest bindings and no blocker. Its fail-closed implementation and accepted
artifact are documented in
[`p7-s6-timeout-contract.md`](p7-s6-timeout-contract.md).
