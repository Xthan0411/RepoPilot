# RepoPilot Docker Sandbox Threat Model

> Version: P4 Gate / policy schema 1
> Status: policy, Docker lifecycle/execution, adversarial security, resource, and cleanup Gate complete

## 1. Scope and security claim

RepoPilot eventually executes untrusted repository code and tests. The Sandbox must assume
that the Issue text, model output, repository files, dependency metadata, test commands,
test process, and subprocesses are hostile.

P4-S1 makes this policy claim:

> A request that violates the policy is rejected with a typed, auditable decision before a
> Docker client or host process is called.

P4-S2 adds this narrower runtime claim:

> A successful `prepare` has created a distinct digest-pinned Docker container whose
> inspected state matches the approved offline, non-root, non-privileged, read-only,
> capability-free, resource-limited posture; every normal failure path attempts exact
> container rollback, and repeated destroy is safe.

P4-S3 adds bounded, policy-approved task-command execution, no-follow file reads, output
previews, persistent artifacts, timeout cleanup, and setup-only network transitions.
P4-S4 and P4-S5 add adversarial command/path/network/secret/resource and cleanup proof.
Patch application and diff enforcement remain P5 work.

## 2. Assets

- Host filesystem, user home, SSH/AWS/Docker/Codex configuration, and API keys.
- Docker daemon and socket, which are equivalent to broad host control.
- Host network and internal services reachable from it.
- CPU, memory, process table, disk/tmpfs, execution time, and log capacity.
- Repository checkout identity and the separation between repository and artifact data.
- Evaluator gold data, traces, diffs, test output, and audit integrity.
- Other RepoPilot tasks, containers, vector namespaces, and working directories.

## 3. Trust boundaries

| Component | Trust level | Boundary |
|---|---|---|
| Agent/model | Untrusted planner | May select a tool or task-owned command index; never supplies shell text, Docker options, mounts, secrets, UID, capabilities, or resource limits |
| Task manifest | Evaluator-controlled but validated | Provides exact setup/test `CommandSpec` values; still cannot override hard policy invariants |
| Repository and tests | Hostile data/code | Enter only an isolated task workspace; cannot become policy instructions |
| Sandbox Policy | Trusted, pure decision layer | Runs before Docker; emits allow/deny plus a safe request fingerprint |
| Docker adapter | Trusted enforcement layer, P4-S2+ | Must translate the approved request exactly and fail closed on unsupported controls |
| Docker daemon/runtime | Privileged dependency | Never exposed to the container; daemon errors become typed Sandbox failures |
| Artifact store | Trusted host-side output boundary | Receives bounded copied output; is not a general host bind mount |

## 4. Hard invariants

These rules cannot be widened through environment variables:

1. No host subprocess or shell is used to execute repository commands.
2. A command must exactly equal one command in the task's setup or test tuple.
3. The executable must be one of `pip`, `pip3`, `pytest`, `python`, `python3`, or `uv`.
   Shells, Docker/Podman clients, `sudo`, `mount`, `nsenter`, `ssh`, `curl`, arbitrary
   repository executables, and absolute executable paths are not allowlisted.
4. `argv` remains a tuple passed directly to the runtime; shell metacharacters are data.
5. Containers are non-root, non-privileged, capability-free, read-only at the root
   filesystem, and use `no-new-privileges`.
6. Docker socket mounting and every arbitrary host bind mount are denied.
7. Workspace input is copied into a per-container tmpfs by P4-S2; it is not implemented by
   weakening rule 6. Sensitive roots, symlinks, and non-regular files are excluded.
8. Image references must contain an immutable `@sha256:<64 lowercase hex>` digest.
   Image selection is deployment/code controlled rather than an environment override.
9. Environment forwarding is name-allowlisted to `CI`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD`,
   `PYTHONDONTWRITEBYTECODE`, and `PYTHONUNBUFFERED`. Values are supplied by the trusted
   adapter, not copied from the host. Provider keys, cloud credentials, `HOME`,
   `DOCKER_HOST`, proxy variables, and arbitrary host environment are denied.
10. Repository and artifact path capabilities are separate. Absolute paths, backslashes,
    `..`, `.git`, `.env`, and sensitive dot-directory roots are denied.
11. Every execution has finite CPU, memory, PID, tmpfs, timeout, output, and artifact
    limits.
12. A denied request cannot be converted to an execution attempt. The future adapter must
    accept an allowed policy record or repeat the same validation itself.

## 5. Configurable limits

Configuration precedence remains defaults < YAML < environment < explicit override.
Values outside these hard ranges fail configuration loading:

| Setting | Default | Valid range | Meaning |
|---|---:|---:|---|
| `REPOPILOT_SANDBOX_NETWORK` | `none` | `none`, `setup_only` | Maximum network capability |
| `REPOPILOT_SANDBOX_CPU_LIMIT` | `1.0` | `(0, 8]` | CPU quota expressed as CPUs |
| `REPOPILOT_SANDBOX_MEMORY_LIMIT_MB` | `1024` | `128–8192` | Container memory ceiling |
| `REPOPILOT_SANDBOX_PIDS_LIMIT` | `256` | `16–1024` | Maximum processes/threads |
| `REPOPILOT_SANDBOX_TMPFS_LIMIT_MB` | `256` | `16–2048` | Writable tmpfs ceiling; cannot exceed memory |
| `REPOPILOT_SANDBOX_COMMAND_TIMEOUT_SECONDS` | `300` | `1–3600` | Per-command policy ceiling |
| `REPOPILOT_SANDBOX_MAX_OUTPUT_BYTES` | `1000000` | `1–10000000` | Captured output ceiling |
| `REPOPILOT_SANDBOX_MAX_ARTIFACT_BYTES` | `20000000` | `1–100000000` | Complete artifact ceiling; cannot be below output |

An untrusted preparation request may ask for less than these limits, never more.
Booleans are rejected as numeric values. P4-S2 must map these fields to explicit Docker
options and verify the created container configuration.

## 6. Network semantics

- `none`: setup and tests run without network.
- `setup_only`: only a task-owned setup command with an explicit network request may have
  network. Tests and all other operations remain offline.
- If P4-S2 cannot prove that network is disconnected after setup, it must fail instead of
  running tests. A permanently networked container does not satisfy `setup_only`.
- Network permission does not forward host proxy variables, credentials, Docker socket,
  SSH agent, or host networking.

## 7. Command and path semantics

The Agent calls `run_tests(test_index)`; it does not send an `argv`. The tool resolves the
index against `TaskSpec`, and the policy compares the complete `CommandSpec`, including
arguments, cwd, and timeout, with the correct setup/test tuple.

Application-specific arguments cannot be made safe by attempting to parse every pytest,
pip, uv, or Python option. Containment therefore relies on the absence of host mounts and
secrets, a task-local workspace, non-root execution, read-only rootfs, network policy, and
resource limits. This is why exact task ownership is necessary but not sufficient.

`RelativePath` rejects absolute paths, `..`, backslashes, non-canonical spellings, and NUL.
The policy additionally separates:

- repository read/write paths, which cannot enter `artifacts/`;
- artifact writes, which must enter `artifacts/<name>`;
- sensitive roots such as `.git`, `.ssh`, `.aws`, `.docker`, `.config`, and `.codex`.

P4-S3 must use no-follow filesystem operations and reject symlinks or path changes between
validation and access. String validation alone does not prevent symlink/TOCTOU attacks.

## 8. Threat/control matrix

| Threat | P4-S1 control | Typed denial | Runtime proof |
|---|---|---|---|
| Agent injects shell command | Exact task tuple plus executable allowlist; no shell string | `command_not_configured`, `executable_denied` | P4-S4 passes `$()`, `;`, and `&&` as one argv value; neither host nor container canary is created |
| Malicious manifest requests `sh`, Docker, or host executable | Hard executable allowlist | `executable_denied` | P4-S4 hostile policy matrix denies shells, Docker, sudo, curl, and absolute executables |
| Mutable or attacker-chosen image | Immutable digest syntax; no runtime image env override | `image_denied` | P4-S2 verifies local RepoDigest and inspect image ID |
| Root/privileged container | UID/GID nonzero, privileged false | `root_user_denied`, `privileged_denied` | P4-S4 real inspect proves UID/GID 65532, read-only rootfs, non-privileged execution |
| Capability or namespace escape | Empty capabilities, no-new-privileges, no Docker socket | `capability_denied`, `no_new_privileges_required`, `docker_socket_denied` | P4-S4 real inspect/probe proves all capabilities dropped, no-new-privileges, and socket absence |
| Host file/credential access | No host bind mounts; no secret environment forwarding | `host_mount_denied`, `environment_denied` | P4-S4 preserves an external host-file canary and excludes fake API/cloud/SSH secrets from container and artifacts |
| Path traversal/artifact overwrite | Canonical paths, no-follow open, immutable artifact namespace | `path_denied` | P4-S4 rejects schema/policy traversal plus runtime file and cwd directory symlinks |
| Network exfiltration | Default none; setup-only phase rule | `network_denied` | P4-S4 proves test network requests are denied and real offline DNS/direct TCP probes fail |
| Fork bomb/resource exhaustion | CPU/memory/PID/tmpfs/time/output limits | `resource_limit_denied`, `timeout_denied` | P4-S5 proves PID `EAGAIN`, CPU timeout, bounded memory termination, recovery, and cleanup |
| Log/artifact disk exhaustion | Output and artifact byte ceilings | `resource_limit_denied` | P4-S5 real flood stops at 8192 retained bytes, destroys the container, and preserves the prefix artifact |
| Cross-task workspace reuse | Task ID correlation and per-task lifecycle | `task_mismatch` | P4-S2 creates two simultaneous distinct containers/tmpfs workspaces |
| Sensitive input leaks through audit | SHA-256 request fingerprint; no argv, host path, image, or env values | N/A | Unit tests and Trace review |

## 9. Audit contract

Each policy decision records:

- policy version;
- prepare/command/path operation;
- task ID;
- allowed or denied;
- `policy_denied` and a stable error code when denied;
- a deterministic SHA-256 of the complete request;
- safe executable name or canonical contained path when applicable;
- effective timeout and whether network was enabled.

It deliberately omits full argv, image reference, raw invalid path, host mount paths, and
environment values. Detailed command output belongs in bounded artifacts, not in policy
logs.

## 10. Denial versus runtime termination

P4-S1 performs no process execution, so it introduces no process exit status or new Agent
termination reason:

- a policy refusal becomes `PolicyDecision.DENIED` with
  `ErrorType.POLICY_DENIED`;
- failure to create, inspect, run, or clean a real container later becomes
  `TerminationReason.SANDBOX_ERROR`;
- an executed command exceeding its deadline becomes `TerminationReason.TIMEOUT` and a
  tool timeout observation;
- a nonzero test result is test evidence, not automatically a Sandbox failure.

## 11. P4-S2 lifecycle enforcement

- `DockerCliClient` uses fixed argv with `shell=False`; the host subprocess is only the
  trusted Docker control plane and never executes a repository command.
- Image resolution is local-only. The requested `name@sha256:<digest>` must appear in
  Docker `RepoDigests`, and the created container's image ID must equal the resolved ID.
  No implicit pull or mutable-tag fallback occurs.
- `prepare` compiles the policy result to `network=none`, UID/GID 65532, read-only rootfs,
  `cap-drop=ALL`, `no-new-privileges`, explicit CPU/memory/PID limits, fixed safe
  environment values, and one size/owner/mode-constrained `/workspace` tmpfs.
- The trusted bootstrap command only creates `/workspace/repository` and
  `/workspace/artifacts`, then waits. The configured working directory remains
  `/workspace`; otherwise Docker would pre-create the repository directory as root before
  the non-root process starts.
- Repository content crosses the boundary through a generated tar stream consumed by
  container `tar` as UID/GID 65532. There is no host bind mount. Copying is bounded by the
  tmpfs limit and omits `.git`, `.env`, sensitive dot roots, symlinks, devices, sockets,
  FIFOs, and other non-regular inputs.
- Docker inspect is normalized and checked before a `SandboxHandle` is returned. Identity,
  image ID, running state, user, network, rootfs, privilege, capabilities, security
  options, quotas, tmpfs, mounts, sensitive environment names, RepoPilot labels, and
  working directory must match or preparation fails closed.
- Stable lifecycle errors distinguish policy, source, daemon, image, create, start, copy,
  inspect, posture, unsupported-operation, and cleanup failures. They contain no daemon
  stderr, image reference, host source path, or environment values.
- A failure after container creation force-removes that exact returned ID. `destroy` can
  be retried after failure and treats absence as success only when Docker explicitly says
  `No such container`; daemon unavailability is not mistaken for successful cleanup.
- The real integration test prepares two simultaneous containers, inspects both, destroys
  both, verifies they are absent, and leaves no `com.repopilot.managed=true` containers.

## 12. P4-S3 run/read/artifact enforcement

- `RunRequest` carries a setup/test phase and strict `network_requested` boolean. Policy
  compares the complete `CommandSpec` to the corresponding task tuple before Docker is
  called. A denied request returns a denied `Observation` with no exit code or output.
- Docker exec still starts only the trusted fixed Python wrapper. The wrapper opens the
  repository root and each requested cwd component using directory FDs plus `O_NOFOLLOW`,
  calls `fchdir`, then replaces itself with the exact allowlisted task argv via `execvp`.
  No task-provided value becomes shell syntax.
- `read_file` uses a separate fixed isolated-Python helper. It walks path components with
  `openat`, `O_DIRECTORY`, and `O_NOFOLLOW`, requires a regular final file, and emits only
  the requested positive line range. Runtime symlinks fail as `path_violation`.
- A setup command receives network only after `none` is disconnected and bridge is
  connected. In `finally`, bridge is disconnected, `none` is restored, and Docker inspect
  must report exactly `("none",)`. Any transition or proof failure destroys the task
  container; tests never request network.
- Docker CLI stdout/stderr are drained incrementally. Each Agent-visible preview is
  bounded by `sandbox_max_output_bytes`; complete combined streams are bounded by
  `sandbox_max_artifact_bytes`. Preview truncation does not discard the complete artifact.
  Crossing the artifact ceiling or wall-clock deadline kills the CLI and force-removes the
  exact task container so a detached child cannot continue running.
- The host artifact root is trusted input and must be disjoint from the repository source.
  Each sandbox receives a new `0700` namespace. Fixed call-ID filenames are written
  `0600`, fsynced, linked without overwrite, and represented by relative path, SHA-256,
  size, and media type. Artifacts remain collectable after Docker cleanup.
- Nonzero task exit codes are ordinary evidence. Policy denial, path violation,
  `tool_timeout`, output/artifact failure, Docker/runtime failure, duplicate call ID,
  network-transition failure, and cleanup failure retain separate typed outcomes.
- The real Python-image integration test proves setup network restoration, bounded
  command output with full artifacts, nonzero exit code, line-range read, runtime symlink
  rejection, timeout under five seconds, post-timeout artifact collection, and container
  absence.

## 13. P4-S4 adversarial core-security proof

- The host-free `security` matrix exercises shells, Docker, sudo, curl, absolute
  executables, allowlisted-but-unconfigured commands, shell metacharacters, test-phase
  network requests, absolute/traversal/backslash paths, sensitive roots, artifact
  cross-root reads, and strict-boolean confusion. Denials retain typed policy errors.
- The opt-in real-Docker test uses only temporary fake host files and fake environment
  secrets. It never opens the user's real SSH, cloud, Docker, Codex, or API-key files.
- An external host canary remains byte-identical. `$()`, `;`, and `&&` sent as one
  configured argument create neither the requested host target nor a container target,
  proving that task argv does not become shell syntax.
- Docker inspect plus an in-container probe prove `network=none`, UID/GID 65532,
  non-privileged/read-only/capability-free/no-new-privileges execution, no bind mount,
  no Docker socket, no forwarded HOME, and no sensitive environment names.
- Offline DNS and direct TCP attempts fail. Explicit network intent on a test command is
  rejected before execution and creates no artifact.
- `.env`, `.git`, and artifact-cross-root reads are denied before Docker. Symlinks created
  after preparation cannot be read, and a symlink used as command `cwd` prevents the
  command from starting.
- Every retained stdout/stderr artifact is scanned for the fake host canary, host path,
  injection payload, and fake secret values. No canary appears, and the container is
  absent after cleanup.

## 14. P4-S5 resource-abuse and cleanup proof

- Every real resource attack uses a separate `network=none` container with 0.25 CPU,
  128 MB memory, 16 PIDs, 16 MB tmpfs, a maximum five-second command deadline, 1 KiB
  per-stream previews, and an 8 KiB combined artifact ceiling. Docker inspect proves the
  effective quotas before attack code starts.
- A CPU busy loop emits one evidence line, reaches its one-second wall-clock deadline,
  returns `tool_timeout` without a trusted exit code, and causes automatic whole-container
  destruction. Its evidence artifact remains readable afterward.
- A bounded fork-pressure probe tries up to 64 sleeping children. The PID cgroup returns
  `EAGAIN` before that count, the parent kills and reaps every child, and a second command
  in the same container proves PID capacity recovered before explicit destruction.
- A process attempts up to 512 MB of allocation inside the 128 MB container. Only Python
  `MemoryError` or Linux OOM exit 137 is accepted; completing the allocation is a test
  failure. The exact container is then force-removed.
- A process floods stdout and stderr. Streaming stops at exactly 8192 retained combined
  bytes, returns `tool_execution_error`, and force-removes the container so the child
  cannot continue after the Docker CLI is killed.
- All retained streams are reread after container destruction and checked against their
  declared byte size and SHA-256. P1 Trace storage is host-side rather than
  container-owned; patch/diff artifacts begin with P5 `apply_patch` and are not
  prematurely claimed by this Gate.
- The tracked cleanup audit records immutable image identities, attack limits, expected
  outcomes, 8 integration tests, 30 security tests, 819 total tests with zero skip, and
  zero managed-container residue.

## 15. P5-S2/P5-S3 test and patch evidence

- `run_tests` accepts only a strict task-owned command index. Structured parsing keeps a
  real assertion/test-body failure separate from import failure, zero tests, unknown
  nonzero exit, truncation, timeout, policy refusal, and Sandbox failure; ambiguous
  evidence cannot become a reproduced defect or passing validation.
- Agent patch input is a bounded diff string, never a host path, shell command, or direct
  overwrite request. The Tool parser denies sensitive/traversal paths, unsupported
  binary/create/delete/rename/mode semantics, malformed hunks, duplicate targets, and
  configured complexity overflow before Docker.
- Validated patch bytes are stored in the current `0700` Sandbox artifact namespace as a
  `0600` immutable file. Before use, its registered relative path, media identity, size,
  regular-file status, no-follow open, and SHA-256 are checked again.
- Only the single verified artifact crosses into Docker, through fixed stdin into the
  task tmpfs. There is still no bind mount, Docker socket, forwarded environment value,
  task-selected destination, or host repository write.
- The fixed helper independently verifies the diff and artifact, then traverses existing
  target parents with directory FDs and `O_NOFOLLOW`. Final targets must be regular UTF-8
  files, so a symlink introduced after preparation cannot redirect a patch.
- All hunks and all replacement bytes are validated before the first replacement.
  Temporary files are staged first. Replacement failure restores original bytes in
  reverse order; inability to prove rollback destroys the container.
- `inspect_diff` returns a bounded preview and metadata only. Full bytes remain in the
  hash-valid artifact. Rejected attempts do not replace the last applied PatchState, and
  neither patch application nor model self-report can declare run success.
- Real Docker evidence covers two successful fixture patches, a multi-file late conflict
  that leaves the earlier target unchanged and permits recovery, and runtime symlink
  replacement that fails without changing the symlink target.

## 16. P6-S6 Memory namespace and poisoning boundary

- Repository and Verified Episodic Stores are opened only as one
  `experiment/snapshot/split/access_mode` bundle. Holdout is schema-enforced read-only;
  dev, holdout, and different snapshots resolve to disjoint directories.
- A 0600 integrity-protected manifest freezes the namespace declaration. Mutable snapshot
  aliases, copied-model validation bypasses, access-mode drift, unexpected contents,
  relative/symlink paths, and digest tampering fail closed with typed namespace errors.
- Lookups require exact repository commit and Manifest content identity. Stale Repository
  text and expired Episodic text are count-only audit outcomes, never model context.
- A verified historical episode remains `historical_evidence`, not trusted control or
  current-run observed evidence. System instructions explicitly prohibit following
  instructions found in Memory; Context/Trace retain separate source, trust, inclusion,
  exclusion, and token-budget records.
- Host-free security tests inject instruction-shaped episode text and prove the trust
  boundary remains intact. Real Docker E2E proves a second same-repository run can load
  the first verified episode without crossing namespace or bypassing Verifier admission.
- Python process-level access controls are not an OS security boundary: trusted
  application code holding a raw filesystem path could bypass the wrapper. Production
  deployment must expose only the namespace bundle to the runner and protect the Memory
  root with operating-system identity and permissions.

## 17. P7-S1 evaluation provenance and overwrite boundary

- Every evaluation run begins with one immutable, integrity-protected manifest binding
  source commit, dependency lock, dataset/task set, complete redacted configuration,
  split/arm/repetitions, Memory namespace, Provider/model/prompt identities, and a
  digest-pinned Sandbox image. Reopening with different content is configuration drift,
  not a request to mutate history.
- A stable task/repetition slot has an append-only transition history. Crash recovery
  preserves the old partial execution, records it as interrupted, and allocates fresh
  run/thread/output identities. Completed and explicitly failed slots are deduplicated;
  retry count is bounded.
- Manifest and event publication uses private directories/files, fsync, and atomic
  no-overwrite semantics. Relative/symlink roots, sequence gaps, transition mismatch,
  concurrent writers, existing nonempty output directories, and oversized input fail
  closed.
- Completion evidence is not trusted from an in-memory return value. The runner reopens
  fixed artifact names without following symlinks, enforces size bounds, hashes actual
  bytes, parses the FinalResult/Trace/artifact manifest, and cross-checks task/run/status,
  termination, success, usage, model, prompt, configuration, and Sandbox identities.
  Completed historical artifacts are reverified before deduplication.
- The production execution adapter receives only Agent-visible `TaskSpec`; the dataset's
  expected patch and other evaluator gold are used only in the task-set digest. The
  manifest also replaces the local absolute repository path with a stable placeholder.
- A local process with write permission to the evaluation root can still replace files
  after publication or race reads. Integrity verification detects ordinary tampering but
  is not a cryptographic signature or remote append-only ledger. Public formal results
  should additionally be bound to a source commit and published artifact checksum.

## 18. P7-S2 dataset provenance, gold isolation, and selection boundary

- The frozen dataset binds one exact SWE-bench Lite revision, a selected source-row
  snapshot, every complete source row, the checked-in fixture manifest, case metadata,
  candidate decisions, preflight claims, and runtime transcripts with content digests.
  Symlink/relative/escaping paths, digest mismatch, unsupported versions, duplicate
  identities, split mismatch, and audit drift fail closed.
- Public source rows contain expected and test patches. They are evaluator-controlled
  input, never repository content or Agent Context. The loader exposes
  `EvaluationTaskSpec` to evaluator code and an explicit inner `TaskSpec` projection to
  execution code. Tests inject known gold canaries and prove they do not enter Agent task
  JSON or the P7-S1 evaluation-run manifest.
- Every selected repository and solution PR uses an immutable base commit. GitHub metadata
  was checked to confirm the PR is merged and its base SHA matches the dataset row.
  Mutable branch/tree URLs are not accepted as repository acquisition identity.
- Setup commands are trusted evaluator configuration, not model output. They are limited
  to shell-free `python -m pip install`, contained requirement paths, and no VCS/URL or
  custom package index. Test commands are limited to pytest or the fixed Django runner.
  Setup network access remains a supply-chain risk and is permitted only before Agent
  execution for declared dependencies.
- Repository licenses are not inferred from current default branches. Each included case
  records the SPDX identifier, license path, Git blob SHA, and URL at its exact base
  commit. The SWE-bench project citation/license is recorded separately because the
  dataset card does not establish one blanket license for all upstream repository text.
- Runtime evidence is intentionally two-level. `metadata_verified` proves identities and
  metadata only. `baseline_failed_gold_passed` additionally requires a digest-pinned
  container, nonzero base+test exit, zero gold exit, expected failure marker, and
  hash-valid setup/baseline/gold logs. Only Flask 4992 currently has that stronger claim.
- A malicious upstream Issue can still contain prompt-injection text. Dataset provenance
  does not make Issue text trusted; the existing Context boundary must continue treating
  it as untrusted task data. A malicious dependency or upstream repository can also attack
  the setup container; no host bind mount, Docker socket, secret, or Agent-selected setup
  source is permitted.

## 19. P7-S6 holdout environment and paid-execution boundary

- Paid holdout consent is separate from dev calibration. It must bind the exact 32-slot
  plan, USD 32 application limit, and final in-flight response overshoot acknowledgement;
  missing or mismatched consent keeps Provider checks and execution not-run.
- Frozen Agent behavior is compared against the formal source commit independently from
  the newer control-plane commit. Adding orchestration, reports, or documentation cannot
  silently change Agent, Prompt, Tool, model, retrieval, budget, or success semantics.
- Every frozen task needs its own exact-base clean checkout and evaluator-qualified,
  digest-pinned image matching the declared Python version. A generic image, mutable tag,
  dirty/wrong commit, unsafe path, incomplete task set, unpinned harness, online Agent
  setup, or missing baseline-fail/gold-pass evidence blocks the entire formal run.
- Dependency download is an image/setup preparation privilege only. Formal Agent and test
  execution retain network `none`; task text and repository content remain untrusted.
- Gold may be used by the evaluator to qualify environments and later compute metrics,
  but the application plan, Agent TaskSpec, Provider request, Trace, and runtime artifacts
  must remain Gold-free. Environment failures are preflight failures and cannot be counted
  as Agent failures in the formal denominator.
- Control artifacts use canonical SHA-256, fixed private paths, mode 0600, and create-only
  writes. Formal raw runs continue to use the P7-S1 append-only Runner and preserve every
  technical retry/failure. Results cannot be used to tune the frozen behavior mid-run.
- Runtime readiness gives the Jina child only a minimal cache/runtime environment
  allowlist, never sends the OpenAI credential, DSN, or unrelated secret, and never
  calls an OpenAI endpoint. The optional host-side Jina cache preparation uses exact
  primary/remote-code revisions; acceptance requires a second fresh interpreter with
  Hugging Face and Transformers offline modes enabled. Neither this setup privilege nor
  pgvector initialization changes the task Sandbox's network-`none` boundary.
- The formal controller never opens the evaluator source-row file. It consumes a
  content-addressed `TaskSpec`-only projection; the generic Runner's required evaluator
  fields use an unusable sentinel only for a derived identity and are neither serialized
  in the run manifest nor passed to the executor. Regression tests fail if Gold fields,
  Issue text, or an absolute checkout path enter the manifest.
- Paid mode requires both `--execute` and the exact authorized plan content SHA. Every
  attempt is append-only, task-specific images are re-resolved against readiness, output
  parents reject symlinks, and tracked controller drift fails closed. Interrupted
  attempts retain partial evidence and consume a conservative USD 1 unknown-cost reserve.
  A technical failure stops other slots unless the operator explicitly reviews and opts
  to continue; no result can be deleted or silently retried under changed configuration.

## 20. Residual risk and later phase obligations

- Docker/container isolation is not a VM boundary; kernel and daemon vulnerabilities
  remain outside the Python policy layer.
- Digest pinning identifies bytes but does not establish provenance, signatures,
  vulnerability status, or SBOM quality.
- Dependency installation during `setup_only` can still fetch malicious packages into
  the isolated workspace.
- Docker tmpfs does not provide a transactional filesystem across multiple files.
  RepoPilot uses preflight, staged replacements, and compensating rollback; a kernel,
  daemon, power, or process failure during the small replacement window can still make
  the container untrustworthy. Such a container must be destroyed rather than reused.
- The MVP patch grammar intentionally excludes file creation, deletion, rename, binary
  changes, mode changes, whitespace-bearing paths, and files without a representable
  final newline. Supporting them later requires separate policy and rollback proofs.
- P5-S4 must keep success rule-based: baseline reproduction, artifact-backed applied
  patch, and explicit post-patch pass are all required. P5-S5/P5-S6 must not let Provider
  output weaken these boundaries or bypass the Tool/Sandbox path.
