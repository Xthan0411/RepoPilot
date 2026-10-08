# Publication notes / 公开快照说明

This directory is a curated snapshot of the unfinished RepoPilot project, prepared for
a public project link. The original development checkout remains unchanged.

## Included

- Python source, evaluation scripts, tests and fixtures.
- Locked dependencies, Python version, default/baseline/experiment configurations,
  pgvector Compose configuration and migration.
- GitHub Actions quality workflow, blank-key `.env.example`, and publication ignore rules.
- Frozen dataset and memory namespace manifests, preserving their integrity hashes.
- Selected historical reports and qualification evidence used by regression tests.
- English and Chinese README files, architecture, progress, technical documentation,
  license, and upstream attribution.

## Omitted

- Original `.git` history and all temporary `*.bundle` files.
- `.env.p7-local`, other real credential/config files, local virtual environments,
  caches, generated wheels, local `artifacts/`, and scratch `tmp/` directories.
- Private study/interview-preparation notes and internal handoff/phase planning documents.
- Raw `results/p7-dev/p7-dev-*/` execution trees, including provider traces and local paths.
- Development spending authorization files `results/p7-dev/authorization-*.json`.
- The internal chronological development log; its current checkpoint is summarized in
  [development status](development-status.md).

Historical holdout control artifacts are retained byte-for-byte because the tests verify
their exact hashes and cross-artifact relationships. They contain non-secret image/project
identities and a historical approval record, but no API keys, cloud access tokens, service
account private keys, or database credentials. The default database password in sample
configuration is the local development placeholder `repopilot`.

## How to read the evidence

The JSON reports retain historical provenance and may refer to omitted raw files. Those
references are not a promise of complete public replay. P2/P3 synthetic artifacts and
P7 dataset preflight inputs remain present; the latest P7 dev report is a summary only.

Frozen holdout controllers check original commit ancestry, image digests, clean checkouts,
offline model caches, and explicit paid-run controls. The fresh publication history lacks
those original commits, and its private registry references grant no image access.
Readers can run the deterministic quick start and ordinary test suite without rebuilding
the historical cloud experiment.

One publication-specific test adjustment makes the exact archive-metadata compatibility
regression independent of the original Git history. It reconstructs the pre-patch sandbox
bytes in a temporary repository and checks the same frozen/result Git blobs and complete
patch SHA-256. Production source and frozen evidence are unchanged.

Upstream issue excerpts and evaluator-only patches in `data/manifests/p7-v1/source-rows.json`
remain evaluator material. Do not feed that file to agent context or retrieval. Its
upstream sources and license identities are recorded in the dataset manifest and
[third-party notices](../THIRD_PARTY_NOTICES.md).

## 中文说明

提交目录保留实现、测试、配置、依赖锁和选定证据；排除真实密钥、本机环境、临时 bundle、私人复习材料、原始开发 Trace 和开发费用授权。原工作区文件与 Git 状态不做提交、重置或改写。

历史 Holdout 控制产物用于精确 hash 回归，按原字节保留。其中的云项目、镜像和历史审批标识不是凭据。新公开仓库不包含原 commit 历史，也不提供私人镜像访问；快速示例与普通测试可以运行，原冻结付费实验不能直接照搬到新环境。
