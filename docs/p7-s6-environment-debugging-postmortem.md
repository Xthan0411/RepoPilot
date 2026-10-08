# P7-S6 task-environment debugging postmortem

## Status and scope

P7-S6 must evaluate one frozen Agent across eight SWE-bench-derived tasks without
misclassifying environment failures as Agent failures. The tasks span Django, Seaborn,
Flask, xarray, pytest, and SymPy, require Python 3.9, 3.10, and 3.12.13, and use several
generations of Python packaging metadata.

The pinned upstream SWE-bench harness proved baseline failure and evaluator Gold success
for all eight official task images. RepoPilot then imposed a stricter execution boundary:

- exact clean base checkout plus evaluator-only test patch;
- non-root UID/GID 65532;
- read-only container root filesystem;
- dropped capabilities and `no-new-privileges`;
- no host repository bind mount and no `.git` in the workspace archive;
- network disabled during task setup and tests;
- Gold applied only in a temporary evaluator checkout;
- bounded command time, output, memory, CPU, PID, and tmpfs limits.

The upstream result therefore qualified the task images, but it did not prove that the
same tasks could run through RepoPilot's stricter Sandbox adapter.

Staged diagnostic runs first proved all eight tasks individually. The final full matrix
then ran from clean controller commit `5f2076fdfc6bfb9963e4524a9727434d7f5d2707` on
`linux/amd64` and produced one same-commit 8/8 compatibility artifact. Its raw SHA-256 is
`9504aed7904436f807d890077f927dc3a4c0a6cfa890d383811535c2c8889c1c`; all eight tasks
matched Python, completed setup offline, reproduced baseline failure, and passed with
evaluator-only Gold. Compatibility qualification is complete. All eight images were
subsequently published and control-plane verified under immutable registry digests;
the formal timeout projection was then bound on the same qualified x86 environment.
The resulting 8/8 environment artifact has no blockers. Formal paid execution remains
blocked on separate authorization and runtime-readiness probes, not task compatibility.

## Failure chronology

### 1. Invocation and non-root package state

The first Django canary exposed two controller assumptions:

- direct-file invocation could not resolve the `scripts` package;
- pip's user install state was not writable for the Sandbox UID.

The canary gained a direct-file import bootstrap, a writable `/workspace` HOME/user site,
and explicit offline pip configuration. These changes remained evaluator infrastructure;
they did not change the Agent or task.

### 2. Archive timestamp incompatibility

Seaborn failed while reading an installed package archive. RepoPilot's deterministic
workspace tar used Unix epoch zero, but ZIP metadata cannot represent dates before 1980.
The archive implementation moved to a fixed 1980-01-02 UTC timestamp, preserving
determinism while remaining valid for ZIP-producing build paths.

### 3. Flask runtime, shell, and wheel assumptions

The official Flask image exposed three separate mismatches:

- observed Python 3.11 did not match the frozen task's declared Python 3.12.13;
- the frozen Python runtime was Alpine and provided `/bin/sh`, not `/bin/bash`;
- editable metadata generation required `wheel`, which was not installed.

The adapter reused the already frozen digest-pinned Python 3.12 image, installed the
task's fully pinned dependencies during trusted image preparation, used `/bin/sh`, and
added a pinned wheel. Sandbox setup and tests remained offline.

### 4. First all-task strategy was fail-fast

The initial matrix driver stopped on the first failing task. Each VM run therefore
revealed only the next project's first compatibility problem. This serialized discovery
across heterogeneous build systems and caused avoidable operator round trips.

The driver was changed to:

- continue after bounded task-local failures;
- print a stable aggregate list at the end;
- support repeated `--task-id` arguments for diagnostic subsets;
- never write the formal artifact for a subset;
- preserve zero Provider calls throughout environment qualification.

The first batched five-task run immediately proved the value of this change: xarray 5131
passed, while one execution reported all four remaining failures.

### 5. SCM build metadata without `.git`

Xarray 4094 and both pytest tasks installed `setuptools-scm` successfully but then failed
to infer package versions. RepoPilot intentionally omits `.git` from the Sandbox archive,
so installing the plugin alone was insufficient. Xarray 5131 passed because its build
metadata defined a fallback version.

The adapter now supplies project-specific, non-secret
`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_*` values. Each value equals the version previously
installed in the corresponding digest-pinned upstream image:

| Task | Distribution | Frozen build version |
| --- | --- | --- |
| `pydata__xarray-4094` | xarray | `0.15.2.dev110+ga64cf2d54` |
| `pytest-dev__pytest-11143` | pytest | `8.0.0.dev53+g6995257cf` |
| `pytest-dev__pytest-11148` | pytest | `8.0.0.dev46+g2f7415cfb` |

The values are syntax-validated, bound to task identity in the evidence model, and do not
weaken the `.git` exclusion.

### 6. SymPy's frozen test command lacked pytest

SymPy completed editable installation and produced the expected failing baseline, but its
Gold command failed with `No module named pytest`. The frozen task invokes
`python -m pytest`, while neither its setup command nor the upstream image supplied
pytest.

Changing the frozen task would have changed the evaluation contract. Instead, the
evaluator adapter installs a fully pinned pytest 7.4 toolchain during trusted image
preparation. Task setup and tests still run with network disabled.

## Validation mistake and correction

An initial local amd64 probe directly copied `/testbed` from each official image. It used
the intended non-root, no-network, read-only-rootfs controls, but accidentally retained
`.git`. Xarray and pytest therefore inferred versions and produced a false-positive setup
result.

The correction was to validate the actual semantic boundary, not just the container
flags:

1. explicitly prove `.git` is absent;
2. call RepoPilot's real `_qualify_task` and DockerSandbox path;
3. apply the evaluator-only test patch;
4. prove baseline exits nonzero;
5. apply Gold only in the temporary evaluator workspace;
6. prove Gold exits zero;
7. verify no RepoPilot-managed container remains.

All four previously failing tasks then produced `baseline=(1,)` and `gold=(0,)` locally.
The subsequent GCP x86 diagnostic independently passed all four tasks with
`python_match=true` and `paid_requests=0`.

## Outcome

- Upstream harness qualification: 8 baseline failures and 8 Gold passes.
- RepoPilot staged x86 diagnostic coverage: 8/8 tasks passed.
- Final same-commit x86 compatibility artifact: 8/8 tasks passed with zero Provider
  requests and no Agent-visible Gold.
- Repository regression gate: 1,111 passed, 21 expected skips.
- Ruff, formatting, strict mypy, dependency lock, and container cleanup checks passed.
- Frozen Agent, Prompt, Tool schema, task set, Gold, and success rule were unchanged.
- Immutable registry publication: 8/8 images published, pullback verified, and
  independently matched against the Artifact Registry control plane.
- Formal environment binding: 8/8 immutable task images and clean checkouts bound; 17 of
  24 command timeouts projected through the frozen 300-second ceiling; no blockers.

## Interview framing: “项目中遇到的最大困难是什么？”

### 90-second answer

RepoPilot 最难的部分不是把 LLM 接进循环，而是保证评测结果真的反映 Agent 能力。我的
holdout 有 8 个 SWE-bench 任务，跨 6 个仓库、3 个 Python 版本和多代 packaging 系统。
官方 harness 的 baseline/Gold 都能通过，但接入我自己的非 root、只读 rootfs、无网络、
不携带 `.git` 的 Sandbox 后，出现了 archive 时间戳、Alpine shell、editable build、
setuptools-scm 版本推导和缺失 pytest 等问题。如果直接计入正式结果，会把环境失败误判成
Agent 失败。

我先把 upstream 环境验证和 RepoPilot Sandbox 验证分层，然后把 fail-fast 预检改成一次
遍历全部任务并聚合错误。接着按 exact base commit 审计每个项目的构建元数据，用固定
镜像准备依赖解决兼容性，同时保持 Agent setup/test 无网络、Gold 不进入 Agent。过程中我
还发现本机探针因为保留 `.git` 产生了假阳性，于是改用真实 DockerSandbox 重做
baseline-fail/Gold-pass 验证。最终 8 个任务都获得了 x86 兼容性证据，全程没有模型调用，
也没有为了“跑通”而修改冻结任务或成功标准。这个问题让我认识到 Agent 评测最重要的
工程能力之一，是区分模型 badcase、工具链 bug 和环境基础设施失败，并让三者都有可审计
证据。

### Likely follow-up questions

**为什么不把 `.git` 直接复制进 Sandbox？**

`.git` 会扩大可见数据和攻击面，可能包含 remote、历史对象、hooks 或本地配置；Agent
修复任务只需要工作树。保留 exclusion，再显式提供构建所需的非秘密版本，比开放整个
Git 元数据边界更小。

**为什么不直接修改 task setup，让 SymPy 安装 pytest？**

任务已经被冻结并参与四个 arm 的公平比较。修改 task 会改变数据集和 freeze identity。
pytest 属于执行环境缺失，所以放在 evaluator image preparation，并固定完整版本集合。

**为什么镜像构建允许网络，而任务 setup/test 不允许？**

受信准备阶段只安装经过审计的固定包，并由最终镜像 ID/digest 绑定；运行阶段面对不可信
仓库和模型行为，网络必须关闭。这把 supply-chain preparation 和 Agent execution 权限
分开。

**这次最大的工程失误是什么？**

最初同时存在 fail-fast 和验证语义不完整两个问题：一次只看到一个错误，而且本地探针
没有复现 `.git` exclusion。修复不是继续堆单点补丁，而是改成全量错误采集，并用真实
production boundary 复验。

**如何快速判断 badcase 在哪一层？**

先看阶段：镜像 build、setup、baseline、Agent、patch apply、Gold/test。再比较 upstream
harness、RepoPilot Fake/adapter 和真实 Sandbox 三条链路。模型尚未调用时发生的 build、
setup、baseline/Gold 资格错误不能归因给 Agent；只有环境资格通过后的正式运行才进入
Agent 结果分母。
