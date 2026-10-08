# P4 Docker Sandbox Security and Cleanup Audit

- Evidence: `real_local_docker`
- Recorded: `2026-08-06`
- Platform: Docker Desktop arm64
- Network during attacks: `none`
- Claim boundary: local containment/mechanism evidence, not production-security or
  VM-isolation certification

## Recorded Gate

| Gate | Result |
| --- | ---: |
| Integration | 8 passed, 811 deselected |
| Security | 30 passed, 789 deselected |
| Full suite with all real fixtures enabled | 819 passed, 0 skipped |
| Ruff | passed |
| Format | 116 files |
| Strict mypy | 102 files |
| Lock consistency | 103 packages resolved |
| Managed container residue | 0 |

The Gate used already-local immutable Alpine and Python repository digests. It made no
model call, forwarded no real credential, and pulled no mutable image.

## Resource Attack Matrix

| Attack | Enforced limit | Required outcome | Recorded result |
| --- | --- | --- | --- |
| CPU busy loop | 0.25 CPU, 1 s deadline | `tool_timeout`, no trusted exit code | passed; container automatically destroyed |
| Fork/process pressure | 16 PIDs | `EAGAIN` before 64 children | passed; children reaped and follow-up command ran |
| Memory allocation | 128 MB | `MemoryError` or OOM exit 137; never allocate 512 MB | passed; container removed |
| stdout/stderr flood | 1 KiB preview, 8 KiB combined artifact | bounded prefix plus `tool_execution_error` | passed; exactly 8192 bytes retained |
| Core command/path/network/secret attacks | offline, no bind/socket/secrets, no-follow paths | deny or contain every attack | passed |

All resource-pressure programs existed only in a copied tmpfs workspace inside a
non-root, non-privileged, capability-free, read-only-rootfs container. The host process
only invoked fixed Docker control-plane argv.

## Cleanup and Artifact Evidence

- CPU timeout and log overflow force-removed the whole task container, preventing a child
  process from continuing after the Docker CLI process was killed.
- PID pressure killed and reaped every child, then a second command proved PID capacity
  was available before explicit container destruction.
- Memory pressure ended within the bounded container and the exact container was removed.
- Every stdout/stderr artifact remained readable after destruction; stored size and
  SHA-256 were checked against its `ArtifactRef`.
- P1 Trace storage is host-side and not owned by the ephemeral container. Patch/diff
  artifacts are intentionally not claimed here because `apply_patch` and `inspect_diff`
  begin in P5.

## Residual Boundary

Docker is not a virtual-machine security boundary. Kernel/runtime vulnerabilities,
image provenance/signing, hostile dependency supply chains during explicitly networked
setup, and production multi-tenant hardening remain outside this local MVP proof.
