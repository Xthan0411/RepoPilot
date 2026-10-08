# P7 Baseline execution contract

P7-S3 provides two executable comparison arms. They are intentionally thin adapters over
the same repair workflow so task input, model budget, Sandbox, patch policy, test command,
Verifier, Trace, and result schema remain directly comparable.

## Arms

| Arm | Precomputed context | Runtime repository access | Memory | Semantic retrieval |
|---|---|---|---|---|
| `mini-swe-safe-v1` | None | Bounded RepoPilot tools | Disabled | None |
| `no-semantic-rag-v1` | BM25 Top-K | The same bounded tools | Disabled | None |

`mini-swe-safe-v1` is a safety-adapted mini-swe-style ablation, not an unmodified
mini-swe-agent deployment. It keeps the minimal “issue plus tool loop” comparison but does
not expose arbitrary Bash. RepoPilot's frozen security contract permits only configured
setup/test commands, bounded repository reads/search, bounded unified-diff application,
and diff inspection. Calling this arm “Bash-only” without the `safe` qualification would
overstate what was implemented.

Both arms currently use RepoPilot's evidence-gated repair state machine. Therefore P7 can
attribute their difference to precomputed BM25 context, but cannot claim an independent
external agent architecture comparison.

The checked-in definitions are:

- `configs/baselines/mini-swe-safe-v1.yaml`
- `configs/baselines/no-semantic-rag-v1.yaml`

## Runtime and fairness contract

The baseline YAML contains mechanism choices only. It cannot set a model, temperature,
budget, image, test command, resource limit, or success rule. Those values are bound from
one shared `RepoPilotSettings` object and one digest-pinned Sandbox image.

The resulting fairness hash covers:

- model provider, pinned model, temperature, reasoning effort, and per-call output limit;
- maximum steps, replans, tokens, cost, and wall-clock timeout;
- Sandbox image, network mode, CPU, memory, PID, tmpfs, command timeout, and artifact
  limits;
- repository read/list/search limits;
- patch bytes, files, hunks, and changed-line limits;
- the exact tool set and `rule-based-verifier-v1` success criterion.

`require_comparable_baselines(...)` fails before execution when any covered field differs.
The two arm-specific fields deliberately excluded from the fairness hash are the
precomputed context mode and retrieval mode.

The Eval Runner persists the full baseline contract in its immutable run manifest.
Repair artifacts retain two separate hashes:

- `settings_sha256`: the shared RepoPilot runtime settings;
- `evaluation_configuration_sha256`: the complete experiment-arm configuration checked
  by the Eval Runner.

This separation prevents either a mechanism change or a supposedly harmless runtime
change from being hidden behind one ambiguous hash.

## Information boundary

The evaluator owns `EvaluationTaskSpec`, including `expected_patch`, `gold_files`, and
split. A baseline executor receives only the inner `TaskSpec`. Its constructor has no
Memory-store parameter, and its YAML schema rejects evaluator-only keys.

The model-visible task contains the issue, base commit, local repository snapshot, and
the already-frozen setup/test commands. It does not contain the expected patch, gold file
labels, or hidden test-patch bytes. The deterministic Docker smoke uses a gold canary and
an independently authored passing patch, then scans every retained run file to prove the
canary and evaluator gold path did not cross the boundary.

## Tool and success contract

Both arms register the same safety-bounded tools:

`apply_patch`, `inspect_diff`, `inspect_symbol`, `list_files`, `read_file`, `run_tests`,
and `search_code`.

The repair graph exposes only the capability needed by its active step. The Agent cannot
invent a command for `run_tests`; it selects a test index from the frozen task manifest.
It can request finish, but only the deterministic rule-based Verifier can emit task
success. Success requires a reproduced failure, an artifact-backed applied patch, and an
independent passing execution of the same configured test command.

## Termination and errors

The adapters do not invent favorable termination states. They retain the common
`TerminationReason` vocabulary:

`success`, `max_steps`, `token_limit`, `timeout`, `cost_limit`, `sandbox_error`,
`invalid_patch`, `tests_failed`, `tool_error`, `provider_error`,
`retrieval_exhausted`, `repeated_action`, and `agent_stopped`.

Preflight failures use `BaselineErrorCode`:

- `input_io_error`: missing, symlinked, empty, oversized, or unreadable YAML;
- `invalid_configuration`: the supplied definition cannot be strictly validated;
- `unsupported_version`: a non-v1 schema;
- `capability_mismatch`: wrong arm identity/context/retrieval, extra tools, query rewrite,
  semantic retrieval, Memory, or arbitrary Shell;
- `fairness_violation`: mutable image or model/budget/Sandbox/tool-policy drift;
- `information_boundary`: evaluator-only gold/oracle/test-patch keys in baseline config.

Once execution starts, orchestration failures remain typed `RepairWorkflowErrorCode`
values and are recorded by the Eval Runner as `execution_failure`; corrupt or mismatched
result bytes are recorded as `artifact_integrity`. Normal Agent failures remain terminal
results rather than disappearing from the denominator.

Illegal examples include `python:3.12` instead of an image digest, Dense retrieval in
`no-semantic-rag-v1`, enabled query rewrite or Memory, an added `shell` tool, a changed
budget between arms, duplicate/reordered tools, a mismatched arm ID/context pair, or any
gold/test-patch field.

## Deterministic checks

Configuration and boundary checks:

```bash
uv run pytest tests/unit/test_evaluation_baselines.py -q
```

Real Docker smoke for both arms:

```bash
REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE=python@sha256:<local-image-digest> \
  uv run pytest \
  tests/integration/test_repair_workflow_integration.py::test_baseline_adapters_share_fairness_and_keep_evaluator_gold_out_of_docker \
  -q
```

This smoke validates adapter wiring, not model quality. P7-S4 will freeze the full
experiment matrix and perform cross-arm preflight; P7-S5 and P7-S6 own dev calibration
and formal holdout execution respectively.
