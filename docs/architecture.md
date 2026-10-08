# Architecture / 架构

RepoPilot treats repository repair as an evidence-producing workflow. The provider
proposes one structured action at a time; the graph and tools enforce what can happen,
and a separate verifier decides whether the retained evidence supports success.

## Execution path

1. Load and validate an agent-only `TaskSpec`. Evaluator patch fields are rejected.
2. Ingest the permitted repository content and construct retrieval/context inputs.
3. Prepare a digest-pinned sandbox and run task-owned setup commands.
4. Execute `reproduce → search → inspect → patch → validate → finish` with capabilities
   restricted to the current step.
5. Replay the completed trace, independently verify reproduction/patch/test evidence,
   collect artifact hashes, destroy the sandbox, and write the final result.

The independent verifier requires a failing reproduction, an applied patch, and passing
validation evidence. Tool failures, environment failures, invalid patches, and budget
termination have separate recorded outcomes.

## Code map

| Boundary | Implementation | Reason |
| --- | --- | --- |
| Orchestration | [`agent/repair.py`](../src/repopilot/agent/repair.py), [`repair_workflow.py`](../src/repopilot/repair_workflow.py) | Keep permitted actions and transitions explicit |
| Provider | [`providers/`](../src/repopilot/providers/) | Swap deterministic fixtures and real model calls behind one contract |
| Repository and tools | [`tools/`](../src/repopilot/tools/), [`retrieval/`](../src/repopilot/retrieval/) | Bound reads, indexing, queries, and executable capabilities |
| Sandbox | [`sandbox/`](../src/repopilot/sandbox/) | Contain repository code and preserve bounded test artifacts |
| Verifier | [`verifier.py`](../src/repopilot/verifier.py) | Derive success from evidence independently of model output |
| Trace | [`trace.py`](../src/repopilot/trace.py), [`replay.py`](../src/repopilot/replay.py) | Check order, identity, and integrity without executing the run again |
| Context | [`context/manager.py`](../src/repopilot/context/manager.py) | Select inputs under source-specific and global budgets |
| Memory | [`memory/`](../src/repopilot/memory/) | Give checkpoints, repository facts, and verified experience separate lifecycles |
| Evaluation | [`evals/`](../src/repopilot/evals/), [`scripts/`](../scripts/) | Freeze comparable controls and retain append-only execution evidence |

## Trust and resource boundaries

Issue text, repository content, tool outputs, and retrieved memory remain untrusted
inputs. They cannot expand tools or modify success criteria. Gold patches are
evaluator-only. Large outputs enter context through bounded artifact references.

Docker preparation copies regular allowlisted files through a filtered archive rather
than bind-mounting the host. `.git`, credential roots, symlinks, and environment files
are excluded. The sandbox uses a non-root user, dropped capabilities, a read-only root,
bounded tmpfs/CPU/memory/PIDs, and finite command deadlines.

Working Memory restores one task/run with its contiguous trace prefix. Repository
Memory invalidates records when the commit or manifest content changes. Episodic Memory
admits experience only after replay and verifier checks. Development and holdout
namespaces cannot be mixed.

## 中文说明

项目将“模型建议”和“可信执行结果”分开：模型每次提出一个结构化动作，状态机决定该阶段允许的能力，工具和 Sandbox 执行动作，独立 Verifier 根据复现、补丁和测试产物判定结果。

Trace 记录调用顺序、状态变化和预算，可在不重跑工具的情况下离线校验。三类 Memory 使用不同生命周期：Working Memory 用于同任务恢复，Repository Memory 随 commit/content 变化失效，Episodic Memory 只接受经过验证的修复经验。评测 Gold 始终留在评测端。
