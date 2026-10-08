# Publication validation / 公开快照验证

Validated on 2026-10-08 using Python 3.12.12 and the existing locked development
environment. Package imports were explicitly pointed at the copied source, and the
full test suite ran from a separate temporary repository with a fresh single-commit
history. Original development commits were absent from that verification repository.

| Check | Result |
| --- | --- |
| Full pytest suite | 1,156 passed; 21 skipped |
| Infrastructure/model tests | Skipped when their explicit Docker, database, or real-model prerequisites were absent |
| Ruff lint | Passed |
| Ruff formatting | Passed; 200 files checked |
| Strict mypy, `src/repopilot` | Passed; 81 source files checked |
| `uv lock --check --offline` | Passed; 104 locked packages resolved |
| Deterministic `run-fixture` CLI | Completed with the documented `stopped/agent_stopped` result |
| Offline trace replay CLI | Completed; 12 trace events verified |
| Local Markdown links | No missing targets |
| Original working checkout | Tracked-file hashes, HEAD, index, and prior status preserved; only the new submission directory added |

No live model, paid holdout, registry publication, or real infrastructure integration
run was performed during publication preparation. The test run reused the installed
development dependencies; it was not a new dependency download or a GitHub-hosted CI run.
The uv check needed access to macOS system configuration outside the filesystem sandbox
after uv panicked inside that sandbox; the successful check remained offline and did not
rewrite the lockfile.

The publication-specific readiness test builds temporary frozen/qualified Git commits
and verifies the original expected blob and patch hashes. It does not bypass production
readiness checks or import private Git history into the public snapshot.

## Repeat the ordinary checks

```bash
uv sync --locked
uv lock --check
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy src/repopilot
```

## 中文说明

独立的新 Git 历史副本中，测试为 1156 通过、21 跳过；跳过项有明确的真实 Docker、数据库或模型前置条件。Lint、格式、核心严格类型检查、锁文件检查、确定性 CLI 与 Trace 回放均通过。此次验证复用了现有依赖环境，没有进行模型调用、付费评测或 GitHub CI 运行。
