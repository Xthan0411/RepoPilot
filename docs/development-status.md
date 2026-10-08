# Development status / 开发进度

Publication snapshot prepared on **2026-10-08**, from source commit
`c92d59f84f3bd063a730bf6ad9984202a1a39f83`. The latest development log is dated
**2026-08-08**. Publication preparation does not advance the implementation phase.

| Phase | Status in the development records | Evidence boundary |
| --- | --- | --- |
| P0 | Complete | Package, CLI, validated configuration, typed contracts, fake smoke |
| P1 | Complete | Bounded agent loop, tools, checkpoints, trace, and replay |
| P2 | Complete | Ingestion, chunking, BM25, synthetic localization reports |
| P3 | Complete | Embeddings, vector storage, dense/hybrid retrieval, query rewrite controls |
| P4 | Complete | Hardened Docker sandbox and hostile-input/resource tests |
| P5 | Complete | Repair fixtures, reproduction, patching, independent verification, provider boundary, E2E |
| P6 | Complete | Context budgets, working/repository/episodic memory, namespace isolation |
| P7-S1–S4 | Complete | Resumable evaluation protocol, frozen dataset, baselines, experiment matrices |
| P7-S5 | Complete | Final development calibration and formal freeze |
| P7-S6 | In progress | Eight task environments qualified; formal runner ready; paid slots 0/32 |
| P7-S7 | Pending | Formal aggregation waits for all planned slots to become terminal |
| P8 | Pending | Final evidence-based documentation and reproducibility closure |

## Recorded evaluation checkpoint

The final v5 development calibration covers two development fixtures, four default
arms, and one repetition, for eight attempts:

| Arm | Verified successes | Invalid-patch outcomes | Recorded estimated USD |
| --- | ---: | ---: | ---: |
| `dense-only-v1` | 2 | 0 | 0.186140000 |
| `hybrid-rrf-v1` | 1 | 1 | 0.138182500 |
| `mini-swe-safe-v1` | 0 | 2 | 0.064572500 |
| `no-semantic-rag-v1` | 2 | 0 | 0.157283750 |
| Total | 5 | 3 | 0.546178750 |

All eight runner attempts completed; three agent outcomes remained invalid patches.
Runner completion is distinct from verified repair success. The
[v5 report](../results/p7-dev/calibration-report-v5.json) records
`holdout_accessed=false`. The raw dev execution trees are omitted from this publication
snapshot, so the report is retained as a historical summary, not a self-contained replay.

The [8-task sandbox qualification](../results/p7-holdout/sandbox-matrix-preflight-v1.json)
records baseline failure and evaluator-gold success on Linux amd64, with no provider
requests and no gold exposed to the agent. This proves environment compatibility;
it does not measure agent repair performance. The later
[runtime-readiness artifact](../results/p7-holdout/runtime-readiness-v1.json)
records eight image identities, offline 768-dimensional Jina/pgvector readiness, and zero
paid requests in the original execution environment.

The formal holdout plan has four default arms × eight tasks × one repetition = 32 slots.
No formal holdout slot has started in the recorded checkpoint. Formal conclusions must
wait for execution and P7-S7 aggregation.

## Publication and reproduction

The executable core, configuration, lockfile, scripts, dataset manifests, and retained
JSON evidence are copied without modifying their bytes. One readiness regression test
now creates isolated Git fixture commits instead of requiring the private development
history; it still verifies the exact frozen/result blob identities and patch digest.
The new repository
will have a fresh history; historical source commits and private registry assets do
not become available merely because their identities are recorded here.

The historical holdout authorization is retained as a non-secret regression input:
tests check its exact bytes and binding to the original plan. It is not authorization
for a new user, environment, or experiment. This publication work performs no model
call, registry mutation, or paid evaluation.

## 中文说明

本快照整理于 2026-10-08，最新开发记录停在 2026-08-08。整理公开材料没有推进开发阶段：P0–P6 与 P7-S1–S5 已完成，P7-S6 仍在进行。

最终开发集是 2 个 fixture、4 个默认实验臂、1 次重复，共 8 次尝试；其中 5 次修复经过验证，3 次为 invalid-patch。8/8 Runner 完成不能写成 8/8 修复成功；8/8 环境验证也不能写成 Agent 成功率。正式 Holdout 为 0/32，P7-S7 汇总和 P8 收尾尚待完成。

公开仓库保留历史报告和测试所需输入，不带原 Git 历史、本地原始开发 Trace 或私人开发费用授权。历史 Holdout 授权仅作为精确字节回归输入保留，不代表对新实验的付费许可。
