# P7-S6 v2：恢复后的授权、readiness 与正式运行

**2026-10-29 暂停执行：本页已成为历史 v2 操作记录，不要继续其中的付费命令。**
首个 Dense/Django slot 因内部 ingest 的 2,000 条上限技术失败，当前为 1 个 technical
terminal、0 个有效 Agent 完成结果、31 pending。源码修复必须重新资格验收/冻结/授权，
旧 readiness 与授权不能套用。当前操作入口是
[真实工作负载修复与零付费预检](p7-s6-repository-workload-repair.md)。
以下章节保留原始 v2 流程以供审计，不代表新的运行指令。

## 这次批准的范围

- Plan ID：`p7-holdout-v2-d26d33699d59b6fc`。
- Plan content SHA-256：`3660cc5962ab8810b9b74fab1c4820188340111d7798e0ef4c620f8d1aafb3c0`。
- 原始 plan SHA-256：`b0654760cea101d8d1b295ca1c2c56e05e924a88f8d70c730e326706414253a2`。
- 4 组 × 8 任务 × 1 次，每个任务上限 USD 1.00，应用计划上限 USD 32.00；
  用户明确接受最后一个 in-flight Provider response 可能小幅越界。
- USD 32 不包含 GCP VM、磁盘和镜像存储等费用。
- 新授权保存到 `results/p7-holdout/authorization-v2.json`，绑定 plan、freeze、matrix；
  不复用历史授权。所有 readiness 检查通过前不得发起付费请求。
- 授权原始 SHA-256：`920470eb1af7e258b4616300670d7f7a214a8c2dd5fa2f69ed66165955dbf261`。
- 已验收的 Agent/测试语义固定在 `19cb83d`。新控制器单独记录 Git commit 和整个
  `scripts/` 的逐文件字节身份；生成 readiness 后不要修改执行代码或换包。

默认执行入口只有 dry-run。真正运行必须显式带 `--execute` 和上面的完整 plan hash。
本轮本地开发/打包不调用 Provider；以下 VM 操作由用户执行。

## 1. Cloud Shell：上传并传给 VM

先看提示符。`jerryxu0411@repopilot-eval-x86` 是 VM，**不是 Cloud Shell**。
如果在 VM 的 tmux 内，先单独输入 `tmux detach-client`；回到 VM 的外层 SSH 后，
再单独输入 `exit`。直到 `hostname` 输出 `cs-...`、提示符包含 `@cloudshell`。
如果已在 Cloud Shell，不要再输入 `exit`。

使用 Cloud Shell 的 Upload 功能上传本次提供的
`repopilot-p7-s6-v2-controls-full.bundle`。上传落在 Cloud Shell home，不会自动出现在 VM。
在 **Cloud Shell** 中执行：

```bash
hostname
ls -lh ~/repopilot-p7-s6-v2-controls-full.bundle
sha256sum ~/repopilot-p7-s6-v2-controls-full.bundle
```

必须先确认文件存在且 SHA-256 与本次 chat 的交付值完全一致，再继续：

```bash
gcloud compute scp \
  ~/repopilot-p7-s6-v2-controls-full.bundle \
  repopilot-eval-x86:~/repopilot-p7-s6-v2-controls-full.bundle \
  --project=repopilot-504912 \
  --zone=us-west1-b \
  --tunnel-through-iap
```

传输成功后，从 **Cloud Shell** 连接 VM：

```bash
gcloud compute ssh repopilot-eval-x86 \
  --project=repopilot-504912 \
  --zone=us-west1-b \
  --tunnel-through-iap
```

不要在 VM 内再次 SSH 到自己，也不要给 publisher service account 添加 Compute 管理权限。

## 2. VM：更新代码（不改已有 task checkout / image）

以下所有命令均在 **VM** 内执行：

```bash
hostname
cd ~/work/repopilot
sha256sum ~/repopilot-p7-s6-v2-controls-full.bundle
git status --porcelain=v1 --untracked-files=no
```

应为主机 `repopilot-eval-x86`、相同 bundle hash，且最后一条没有输出。
有 tracked 改动时停止并发回结果，不 reset、不覆盖。确认后：

```bash
git fetch ~/repopilot-p7-s6-v2-controls-full.bundle main && \
git merge --ff-only FETCH_HEAD && \
git rev-parse HEAD
```

应与本次 chat 给出的交付 commit 一致。已有的未跟踪 Canary JSON 如果和 bundle 中
文件完全相同，Git 可以保留；若 Git 报会覆盖未跟踪文件，停止并发回原始错误，勿删除证据。
不用重新拉取 8 个镜像、重建 checkout 或重跑已通过的 Canary。

## 3. VM：tmux、依赖和 pgvector

为避免网页断线打断长任务，进入持久会话：

```bash
tmux new-session -A -s p7-v2
```

进入后再执行下面的命令（不是在 Cloud Shell 主机执行）：

```bash
cd ~/work/repopilot
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen --extra jina && \
docker compose up -d --wait pgvector
```

只有两条都成功才继续。第一次安装 Jina/Torch 或下载缓存可能较久；出现失败时保留输出。
后续直接使用 `.venv/bin/python`，避免一次普通 `uv sync`/`uv run` 又移除 Jina extra。
不要升级系统、改锁文件或在 task checkout 中运行宿主机 pip/pytest。

## 4. VM：安全加载 Key（不要发给 chat）

先检查是否已保留本 VM 的配置：

```bash
cd ~/work/repopilot
test -s .env.p7-holdout
printf 'existing_env_file_exit_code=%s\n' "$?"
```

输出 `0` 表示已有非空文件，跳过下面的创建步骤。输出 `1` 才执行创建；输入 Key 时
屏幕不显示字符是正常现象。不要把 Key 粘贴到普通命令行或 chat：

```bash
umask 077
read -r -s -p "请输入 OPENAI_API_KEY（输入不可见）: " P7_OPENAI_KEY
printf '\n'
if test -n "$P7_OPENAI_KEY" && test ! -e .env.p7-holdout; then
  {
    printf 'OPENAI_API_KEY=%q\n' "$P7_OPENAI_KEY"
    printf 'DATABASE_URL=%q\n' \
      'postgresql://repopilot:repopilot@localhost:55433/repopilot'
  } > .env.p7-holdout
else
  printf 'not_written_empty_key_or_existing_file\n'
fi
unset P7_OPENAI_KEY
```

如果提示已有文件但之前是空文件，请停止让我们检查，不要盲目覆盖。
加载并只打印存在性：

```bash
chmod 600 .env.p7-holdout
set -a
source .env.p7-holdout
set +a
if test -n "${OPENAI_API_KEY:-}"; then
  printf 'openai_key_present=yes\n'
else
  printf 'openai_key_present=no\n'
fi
stat -c 'mode=%a path=%n' .env.p7-holdout
```

应为 `yes` 和 `mode=600`。禁止 `REPOPILOT_*`、`OPENAI_BASE_URL`、`OPENAI_API_BASE`、
`OPENAI_ORG_ID`、`OPENAI_PROJECT_ID` 覆盖；不要粘贴 `env` 或 `.env` 原文。
Key 存在性检查不证明 Key 有效或模型访问权限；readiness 不调用 OpenAI。

## 5. VM：只运行零付费 readiness 和 dry-run

```bash
cd ~/work/repopilot
.venv/bin/python -m scripts.p7_holdout_readiness_v2 \
  --repository-root ~/work/repopilot \
  --prepare-embedding-cache
p7_readiness_exit_code=$?
printf 'readiness_v2_exit_code=%s\n' "$p7_readiness_exit_code"
```

先检查授权、源码、镜像、checkout、输出路径、依赖和 Key；明显缺项会一起报告。
可选缓存准备只下载冻结的 Jina 主/代码 revision，不向子进程传 Key/数据库 URL。
随后必须在新进程中离线生成 768 维向量，并验证 pgvector 768 维。
成功时输出 `holdout_runtime_readiness_v2_passed=true`、`provider_requests=0`、
`paid_requests=0` 和 `readiness_v2_exit_code=0`。

只有成功后才运行下面这个 **dry-run**（没有 `--execute`）：

```bash
.venv/bin/python -m scripts.p7_holdout_runner_v2 \
  --repository-root ~/work/repopilot
p7_dry_run_exit_code=$?
printf 'dry_run_v2_exit_code=%s\n' "$p7_dry_run_exit_code"
sha256sum results/p7-holdout/runtime-readiness-v2.json
```

首次应显示 32 pending slots、0 已运行、费用账本 0，以及
`holdout_execution_preflight_passed=true execute=false ... provider_requests=0`。
**先把本节两段输出发回 chat；不要把下面付费命令和前面整段一起复制。**
readiness 是 create-only；成功后无需重复生成，不得删掉或覆盖来掩盖失败。

## 6. VM：首次付费运行（与第 5 节分开）

在第 5 节全部通过并确认其输出后，沿用同一包、同一 tmux 和已加载的配置：

```bash
.venv/bin/python -m scripts.p7_holdout_runner_v2 \
  --repository-root ~/work/repopilot \
  --execute \
  --confirm-plan-sha256 3660cc5962ab8810b9b74fab1c4820188340111d7798e0ef4c620f8d1aafb3c0 \
  --max-new-slots 1
p7_first_slot_exit_code=$?
printf 'first_slot_v2_exit_code=%s\n' "$p7_first_slot_exit_code"
```

这是 32 次正式运行中的第 1 次，不是额外样本。正常 Agent 失败也必须保留，不为提高
成功率重跑。技术错误、账本预算不足、源码/证据漂移会停止。不要开启第二个执行窗口。
首次 terminal 记录通过结构检查后，继续同一命令并把 `--max-new-slots 1` 改成 `31`；
完成的 slot 自动跳过。若发生技术错误，不要自行加 `--continue-after-technical-failure`。

从 tmux 脱离用 `tmux detach-client`；重连 VM 后用 `tmux attach -t p7-v2`。
SSH 网页断开不会要求重建 VM；不要用关闭 billing 的方式暂停任务。

## 7. 结果、停止原因与 S7 边界

- 原始结果在 `results/p7-holdout/raw/p7-holdout-v2-d26d33699d59b6fc/`，保留 manifest、
  append-only 事件、每次执行的 result/trace/artifacts；中断不覆盖。
- 账本 = 已知估算费用 + 每个缺少可信 usage 的已启动 attempt 预留 USD 1。
  下一 attempt 需再预留 USD 1；不足即停止。最后响应越界后不调度任何新 slot。
- 计划级非阻塞锁保护进度检查、预算及执行。`concurrent_execution` 表示已有进程持锁，
  不是让你删除 lock 文件；进程退出会释放锁，文件保留属于正常现象。
- `control_identity_or_authorization_invalid` / `source_identity_mismatch`：控制文件、代码
  或绑定不匹配。`local_dependency_unavailable`：依赖不全；`embedding_snapshot_unavailable`：
  缓存/最终离线探针未通过；`vector_backend_unavailable`：数据库门禁未通过。
- `checkout_unavailable` / `sandbox_image_unavailable`：任务工作树或镜像不符合固定身份。
  `output_conflict`：已有输出、危险路径或不可覆盖的历史；不要删除证据来继续。
- `technical_failure` 需要诊断，不能改称 Agent 未解决，也不能隐藏该 attempt。
  `budget_exhausted` 需要查看已保留的实际账本；不得扩大预算或更改计划后静默继续。
- 32 个 slot 都有 terminal outcome 才生成 `execution-report-v2.json`。
  该文件是原始证据/费用索引，不是聚合 benchmark 结论。P7-S7 随后计算分母、成功率、
  定位、时延/成本、失败分类，并标明所有缺失和技术失败。

本轮 Fake 测试与本地新克隆验证不能替代第 5 节的真实 VM readiness；不要预先创建
或提交一份伪造的 `runtime-readiness-v2.json`。
