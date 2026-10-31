# RepoPilot

[English](README.md) · [简体中文](README.zh-CN.md)

RepoPilot 是一个面向 Python 仓库 Issue 的修复 Agent 项目。它把问题复现、代码检索、
源码读取、补丁应用、测试验证和证据保存串成闭环；使用 LangGraph 单 Agent 工作流，
不是 Master/Sub-agent 系统。模型声明“完成”不等于修复成功，最终结论由独立 Verifier 判定。

## 最终结果

v6 是 **8 个已见任务 × 5 个实验臂 × 每槽 1 次**的调试后复测；40 槽全部完成，
技术失败为 0。四个交互 Agent 臂合计 19/32 成功，单次生成对照为 1/8。

| 实验臂 | 成功 / 计划槽位 | 本轮模型估算费用（USD） |
|---|---:|---:|
| Dense | 4/8 | 1.92636 |
| Hybrid：BM25 + Dense + RRF | 6/8 | 1.81055 |
| Mini-SWE-safe：无初始检索 | 5/8 | 1.61819 |
| BM25：无语义 RAG | 4/8 | 2.05693 |
| 单次生成：相同初始 Hybrid 证据 | 1/8 | 0.30329 |

来源：[正式聚合报告](results/p7-holdout/aggregate-v6.md)、
[机器指标](results/p7-holdout/aggregate-v6.json)、[失败分析](docs/error-analysis.md)。
本轮模型估算 USD 7.71531；累计应用记账 USD 30.340901875 / 50，包含 USD 11 未知费用预留，
不是已核对账单，也不包含云资源费用。

**不能把 75% 对 12.5% 宣传为架构使能力提升六倍。** 单次对照的 diff 格式提示弱于 Agent
工具契约，三例因此被拒绝；交互量与计算预算也不同。此外，任务已经被调试过程见过，
每臂只有 8 题、每题一次，没有检验泛化、显著性或方差。这不是完整官方 SWE-bench 成绩。

## 实现与边界

- 单 Agent 的 reproduce → locate/read → patch → validate → verify 工作流。
- AST/文本切块，BM25、Dense、RRF 检索；上下文预算、摘要和产物引用。
- Working、Repository、Verified Episodic Memory：有实现，未在 v6 做效果消融。
- 类型化工具、路径/命令/补丁限制、Docker 执行边界和有限纠错。
- 隔离评测器、追加式执行历史、Trace 回放、哈希核验与显式失败结果。

MCP/Skill、Multi-agent、SFT、自研推理服务和生产多租户部署不计为已实现能力。
Docker 不是 VM 边界；普通任务中的零策略拒绝不代表攻击拦截率。

## 不用 API Key 的本地体验

使用完整 Git checkout、Python 3.12 和 uv，在仓库根目录执行：

```bash
uv sync --locked
.venv/bin/repopilot --help
```

[英文 README 的三段快速示例](README.md#quick-start-no-api-key-or-docker)依次演示：
脚本化调查与回放、真实本地可信小夹具的成功/失败修复契约、BM25 合成报告。
不需要模型密钥、Docker、数据库或 Jina；默认是 Fake Provider。
安装通常需要网络，只有依赖已缓存时才使用 `--offline`。

示例不是自主 LLM 修复，也不是 Docker 验收。生产 `repair` CLI 和历史评测 controller
有不同输入与付费门禁，不会自动继承历史 USD 50 上限。原始评测包、私有镜像和 GPU 缓存不随
公开仓库提供；能读结果，不等于可以一条命令重跑完整实验。

## 项目收尾与文档

本次收尾不补录视频、不追加模型实验；用最终报告、既有成功/失败证据和发布检查交付。

- [最终技术报告](docs/final-report.md)
- [架构与代码入口](docs/architecture.md)
- [评测设计、费用与局限](docs/evaluation.md)
- [验证记录](docs/validation.md)与[公开范围](docs/publication-notes.md)
- [安全边界](docs/threat-model.md)与[第三方归因](THIRD_PARTY_NOTICES.md)

原创代码沿用 [MIT](LICENSE)。上游 Issue、测试补丁和依赖保持各自条款，不被统一重新授权。
