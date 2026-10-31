# Technical guide and configuration entry points

The previous public snapshot's phase-by-phase guide remains in Git at
`309b67a41a576394826a3aec32fbde2adc145f4f`. Its pending-experiment status and old
execution commands are historical, not instructions to restart a paid run.

Current entry points:

- [README](../README.md): installation and explicitly non-paid examples.
- [Architecture](architecture.md): runtime ownership, retrieval/context/Memory,
  typed tools, patch/validation versioning and evaluator separation.
- [Evaluation](evaluation.md): accepted results, read-only report verification,
  private-artifact requirements and limits of the comparison.
- [Threat model](threat-model.md): authority boundaries, controls and residual risks.
- [Default configuration](../configs/default.yaml) and
  [settings implementation](../src/repopilot/config.py): generic runtime settings.
- [CLI implementation](../src/repopilot/cli.py): exact command contracts.

Ordinary settings apply YAML, process environment and explicit overrides. A local
`.env` file is not loaded automatically. Do not put secrets in tracked YAML or task
files. The [example environment](../.env.example) is a template, not authorization.

The frozen experiment controller additionally verifies exact inputs and historical
source identity. Its old sealed records cannot authorize changed source or a new
cloud deployment. No new paid experiment is planned for the final closeout.
