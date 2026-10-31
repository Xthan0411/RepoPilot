# P7-S6：真实仓库工作负载修复与零付费预检

## 当前结论

2026-10-29，用户回传 v2 readiness 通过后，首个 `dense-only-v1 / django__django-15902`
slot 成为 technical failure。当前不是 Agent 成功率 0/32，而是 **1 个技术终态、0 个有效
Agent 完成结果、31 pending**。P7-S6 尚未完成，P7-S7 尚未开始。

用户粘贴的事件时间相差 0.134960 秒，execution 目录没有结果；后续零模型诊断确认
Django 和 SymPy 的 `RepositoryIngestError / listing_truncated`。这些是用户回传终端
证据，不是已下载并校验的完整 raw run。缺少原始 Trace/usage 不能单凭耗时断言账单为零。
账本中的 USD 1 是未知费用保守预留，不是已确认收费；保留原失败事件和预留，不清零。

根因是内部完整索引扫描和 Agent 单次工具返回共用了 `repository_max_list_entries=2000`。
小 fixture 与单向量 readiness 没覆盖真实仓库规模。mini-swe 的无预加载检索路径也会 ingest，
因此不是换一个 arm 就能规避的问题。此前粘贴的诊断只有六个 task，不能据此声称八任务通过。

## 本次改动与不变项

- 内部 `RepositoryIngestScanPolicy(policy_version=1, max_list_entries=20000)` 独立于
  Agent 上限 2000；可注入更小正整数用于受限场景，禁止 bool、零、负数或大于 20000。
  仍限制文件大小、拒绝 symlink/敏感目录；扫描截断即失败，不静默部分索引。
  成功扫描相同内容时，旧 manifest/hash 不变；不新增环境变量。
- 真实仓库复查又发现 AST 排序边界：长父类/函数切成多个窗口后，嵌套定义会破坏最终
  chunk 顺序。现在按既有结果契约排序最终 chunks，不改变内容、ID、行边界或切块参数。
- 技术失败新增 create-only `failure-diagnostic-v1.json`，记录白名单阶段/原因，绑定原
  manifest/FAILED event/attempt/run 身份。旧 event schema/hash 不变，无 sidecar 的历史事件
  仍可读取；新文件若篡改、冲突或路径不安全则拒绝。没有异常文本、traceback、Prompt、Key、
  数据库 URL；请求次数为 unknown，不据此豁免费用预留。它不能追溯补出旧失败的异常。
- 新预检固定八个 task，不读取 solution Gold，不构造或运行 Agent，不发送 Provider 请求。
  默认扫描/切块/BM25；`--prepare-retrieval` 进一步真实运行本地离线 Jina 全库嵌入、cache、
  pgvector 写入/读回、Dense/Hybrid 冷暖路径，以及隔离 Sandbox 的 offline setup、七个工具
  注册、SDK 初始化和首轮 Prompt/payload 组装。SDK 只收到 dummy key，transport 禁止发送。
  各阶段独立记录失败并继续其他任务，避免逐任务往返排错。
- full 模式会消耗 VM CPU/内存/磁盘并写本项目 cache/pgvector namespace，创建并清理临时
  Sandbox；**零模型费用不等于零云主机费用**。只使用已缓存的固定模型版本，不自动下载。
- 不修改 task 选择、模型、Prompt、检索排序参数、900 秒任务预算、300 秒命令上限、评分或
  历史冻结 artifact。预热与冷暖耗时会明确记录；预热不代表提升模型能力，也不能隐藏端到端
  成本/时延。暖路径复用已加载的 Jina provider，不包含正式 runner 新进程的模型冷加载，
  不能作为首次启动的完整时限验收，也不保证剩余时间足够 Agent 完成。

本地真实 SymPy 的只读复现实测：旧扫描 2000 条并截断，新扫描 2042 条完整，manifest 有
1816 个源文件和 12 个跳过项；Agent cap 仍 2000、checkout clean。此证据不替代 VM 八任务验收。
修复后的本地扫描/切块还覆盖 xarray-4094（5299 chunks）、pytest-11143（6790）、
pytest-11148（6785）、SymPy（45372）。这只是无模型、无数据库的四仓库检查，
不是八任务 Dense/Hybrid 或 Agent 成功率。

## 1. 上传并转移修复包

文件名：`repopilot-p7-s6-workload-repair-full.bundle`。使用交付消息中的 SHA-256 核验。
本地 Mac 文件在 `/Users/ethan/repopilot/`。浏览器的“上传文件”先上传到 **Cloud Shell**，
不是 VM。若当前提示符是 `jerryxu0411@repopilot-eval-x86`，在另一个 Cloud Shell 标签操作，
或先 `tmux detach-client`（若在 tmux 内），再从 VM `exit` 返回 Cloud Shell。

以下只在 **Cloud Shell** 执行：

```bash
hostname
ls -lh ~/repopilot-p7-s6-workload-repair-full.bundle
sha256sum ~/repopilot-p7-s6-workload-repair-full.bundle

gcloud compute scp \
  ~/repopilot-p7-s6-workload-repair-full.bundle \
  repopilot-eval-x86:~/repopilot-p7-s6-workload-repair-full.bundle \
  --project=repopilot-504912 \
  --zone=us-west1-b \
  --tunnel-through-iap

gcloud compute ssh repopilot-eval-x86 \
  --project=repopilot-504912 \
  --zone=us-west1-b \
  --tunnel-through-iap
```

若 Cloud Shell 显示没有 active account，先通过其登录授权提示恢复个人账号。
不要在 VM 用 service account 执行连接它自己的 SSH 命令，也不要粘贴 Key/token 到聊天。

## 2. 在 VM 更新代码

以下提示符应为 `jerryxu0411@repopilot-eval-x86`：

```bash
cd ~/work/repopilot
sha256sum ~/repopilot-p7-s6-workload-repair-full.bundle
git status --short
git status --porcelain=v1 --untracked-files=no
```

最后一条应为空。`results/p7-holdout/raw/`、readiness、execution report 的未跟踪文件需要保留，
不要删掉。下面以 `&&` 保证 fetch 失败后不会合并旧的 FETCH_HEAD：

```bash
git fetch ~/repopilot-p7-s6-workload-repair-full.bundle main && \
git merge --ff-only FETCH_HEAD && \
git rev-parse HEAD && \
uv sync --frozen --extra jina && \
.venv/bin/python -m pytest -q -m 'not real_model and not eval'
```

任一步失败先停止，把错误发回；不要用 force/reset/删除结果绕过。`--extra jina` 防止清除
可选 embedding 依赖；后续直接用 `.venv/bin/python`，不让默认 uv sync 意外移除它们。

## 3. 先运行八仓库扫描，再安排完整检索预检

进入 tmux，若已经在 tmux 中则跳过创建命令：

```bash
tmux new-session -A -s p7-workload
```

### 3.1 当前先执行：扫描、切块、BM25（不加载模型）

在 VM 的该 tmux 窗口执行：

```bash
cd ~/work/repopilot
.venv/bin/python -m scripts.p7_repository_workload_preflight \
  --repository-root ~/work/repopilot

p7_workload_scan_exit_code=$?
printf 'workload_scan_exit_code=%s\n' "$p7_workload_scan_exit_code"

jq '{
  controller_commit, scan_passed, prepare_retrieval,
  provider_requests_made, paid_requests_made,
  tasks: [.tasks[] | {task_id, stages: .stages[:3]}]
}' results/p7-holdout/repository-workload-scan-v1.json

sha256sum results/p7-holdout/repository-workload-scan-v1.json
```

先把此摘要和原始 `repository-workload-scan-v1.json` 回传。默认模式后三类检索/运行时
阶段标为 `not_requested`，属于预期；`scan_passed=true` 只验收前三阶段。
必须先看八仓库 chunk/unique-content 规模，再安排 3.2，避免在 CPU VM 上盲目启动
可能较长的全库 embedding。本地 SymPy 已有 45372 chunks，不能给出未经实测的耗时承诺。

### 3.2 扫描验收、确认资源安排之后：完整离线检索预检

以下不是当前必须立即执行的步骤。它会加载本地 Jina 并生成完整向量缓存。

```bash
cd ~/work/repopilot
set +e
set +u
set +o pipefail

set -a
source .env.p7-holdout
set +a
unset OPENAI_API_KEY

docker compose up -d --wait pgvector && \
.venv/bin/python -m scripts.p7_repository_workload_preflight \
  --repository-root ~/work/repopilot \
  --prepare-retrieval

p7_workload_exit_code=$?
printf 'workload_preflight_exit_code=%s\n' "$p7_workload_exit_code"
```

**不用重新运行旧 v2 付费命令。** 本步不用 OpenAI Key；`source` 仅用于已有数据库连接，
随即 unset Key。Jina 模型快照应已由 v2 readiness 缓存。脚本逐阶段输出开始/结束与耗时；
全库 CPU embedding 可能明显慢于之前的单向量探针。等待某个任务期间不应关闭 VM，
也不要因终端暂时无输出重复启动；零 Provider 请求仍会产生 VM 运行费用。

预检成功的标准：退出码 0，`repository_workload_preflight_passed=true tasks=8`，
`provider_requests=0 paid_requests=0`。`formal_ready=false` 是刻意保留的门禁，不是预检失败。
full 报告为 `results/p7-holdout/repository-workload-preflight-v1.json`，0600/create-only。
默认 scan-only 用另一路径 `repository-workload-scan-v1.json`，其通过不等于 full 通过。
若报告已有，不要覆盖或删除；先交回现有报告，后续使用明确的新版本路径/控制器处理。

## 4. 回传安全摘要和报告

```bash
cd ~/work/repopilot
jq '{
  preflight_version,
  controller_commit,
  prepare_retrieval,
  task_count: (.tasks | length),
  scan_passed,
  retrieval_preflight_passed,
  provider_requests_made,
  paid_requests_made,
  formal_environment_ready,
  formal_blockers,
  failures: [
    .tasks[] | .task_id as $task |
    .stages[] | select(.status != "passed") |
    {task_id: $task, stage, status, code, failure}
  ]
}' results/p7-holdout/repository-workload-preflight-v1.json

sha256sum results/p7-holdout/repository-workload-preflight-v1.json
git status --porcelain=v1 --untracked-files=no
```

把摘要贴回，并下载原始 `repository-workload-preflight-v1.json` 上传到本 chat。
不要上传 `.env.p7-holdout`。如果未生成报告，提供终端稳定错误码；不要执行删除/重跑付费。

## 5. 预检之后才做什么

先验收完整八任务报告和原 v2 失败/账本证据，再生成独立的新 runtime freeze/plan。
新计划必须显式保留并结转旧失败的费用记录和 unknown reserve，不能把技术重试包装成
免费重置 32 美元。具体总额度/待执行槽位和 source identity 提供给用户确认后，才能建新授权。
最后跑新 readiness、dry-run，再运行正式矩阵。所有32槽位的保留终态及证据齐全后才进入
P7-S7 的汇总、错误分类与报告；不可提前宣称项目的真实任务成功率。
