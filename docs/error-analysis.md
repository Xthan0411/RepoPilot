# P7-S8：v6 失败分析与未实施改进

本轮收尾保留 v6 的全部原始结果，不再追加模型实验。40 个执行槽位均正常结束，
20 个满足冻结隔离回归测试，20 个未成功；它们是同一组 **8 个已见任务 × 5 个实验臂**，
不是 40 个独立问题，也不是新盲测或完整官方 SWE-bench 成绩。

结论是：执行链路已有可核验的成功案例，但预算分配、重复探索、接口约定和补丁输出契约
仍有明确局限。记录到的停止原因可以准确计数；“为什么模型这样选择”不能仅凭终态标签断言。
本分析不修改 Agent、Prompt、评分、旧工件或费用账本，也不把假设中的修复计成成功。

## 1. 证据范围和阅读方式

- [S7 聚合报告](../results/p7-holdout/aggregate-v6.md)：结果、分母、成本、定位指标及缺失边界。
- [逐失败槽位清单](../results/p7-holdout/error-analysis-v6.json)：20 条失败的 Trace/Result
  路径、SHA-256、终态、预算、工具和上下文事件，绑定已核验的 S7 aggregate。
- [实验局限](experiments.md)：单次对照的格式提示不对齐、已见任务调试、单次运行与历史费用。
- 原始包 SHA-256：`bf1a850a556b0137fb2bd6f2e58327ab002f9706b0318eda5ced03ddea5bdb4d`。
  完整 raw 保存在本地忽略目录，不随 Git 默认公开。下文 Trace 链接只有在恢复原始证据后可用；
  JSON 清单保留可定位的相对路径和内容摘要。

下文 `seq` 指 Trace 的 `events[].sequence`，从 0 开始，不是工具调用次数。
`root_cause=not_inferred` 表示自动清单只提取事实，不自动生成因果判断。
人工案例分为“事实”“解释边界”“待验证改进”；机器校验不能替代因果实验。

## 2. 完整失败分布

| 记录到的停止机制 | Agent 四臂（32 槽） | 单次对照（8 槽） | 解释 |
|---|---:|---:|---|
| `token_limit` | 9 | 0 | 累计 Provider token 超过 100,000 后停止，不是单请求上下文溢出 |
| `tests_failed` | 3 | 0 | 两个候选补丁都应用成功，但第二次验证仍失败 |
| `repeated_action` | 1 | 0 | 重复动作指纹命中，新的重复工具调用没有执行 |
| `invalid_patch` / `invalid_diff` | 0 | 3 | 候选缺少解析器要求的 git diff 文件头 |
| `invalid_patch` / `apply_conflict` | 0 | 4 | 候选旧上下文与当前源码不匹配，未应用 |
| 合计未成功 | **13** | **7** | 20 个失败保留在各自实验臂分母中 |

没有 v6 technical failure；这不表示所有模型决策正确，也不表示历史环境故障不存在。
以上是互斥的最终停止机制。中间发生的错误不能再作为新失败叠加计数，例如 Hybrid / pytest-11148
中途有一次 `apply_conflict`，最后仍成功，不能计入这 20 条失败。
来源：[完整逐槽指标](../results/p7-holdout/aggregate-v6.json)及[失败清单](../results/p7-holdout/error-analysis-v6.json)。

下表覆盖全部 40 槽，便于核对遗漏。✓=成功，B=累计 token 终止，T=测试仍失败，
R=重复动作，F=格式拒绝，C=应用冲突。缩写只用于展示，不替换原始错误码。

| 任务 | Dense | Hybrid | Mini | BM25 | 单次对照 |
|---|---|---|---|---|---|
| django-15902 | ✓ | ✓ | B | B | F |
| seaborn-3190 | B | B | ✓ | ✓ | F |
| flask-4992 | T | ✓ | T | T | F |
| xarray-4094 | ✓ | ✓ | ✓ | ✓ | C |
| xarray-5131 | ✓ | ✓ | ✓ | ✓ | C |
| pytest-11143 | ✓ | ✓ | ✓ | ✓ | ✓ |
| pytest-11148 | B | ✓ | ✓ | B | C |
| sympy-20590 | B | B | R | B | C |

## 3. 规格中的九类错误怎样使用

运行时 `ErrorCategory` 是粗粒度枚举，不是人工根因标签。本轮三个 Flask 验证失败
的原始 `error_category` 都是 `invalid_patch`，但不表示 diff 格式错误；其终止原因是
`tests_failed`。全部失败的原枚举计数为 `budget_exhausted=9`、`invalid_patch=10`、
`tool_misuse=1`，不能拿它直接替代前述机制分类。自动清单保留原枚举，不改写它；
下面额外说明证据能支持到哪一层。类别之间可能重叠，不相加为成功率分母。

| 规格类别 | 本轮可支持的标注 | 不能由此推出 |
|---|---|---|
| `retrieval_miss` | 三个 token-limit 的 SymPy 槽位未观察到对冻结 gold 相关文件的成功 read/search；Mini SymPy 也未观察到。只作定位缺口线索 | 不能证明初始检索完全没有线索；九个预算失败中六个已实际读到相关文件，不能全部归为检索失败 |
| `wrong_plan` | Mini SymPy 重复查找；Hybrid SymPy 修改其他文件且未验证（案例 B、D） | 未验证补丁不等于已证实的错误方案；不能从一次选择断言规划器整体失效 |
| `tool_misuse` | 重复参数触发保护，属于可观测的无进展工具提议（案例 D） | 不等于非法宿主命令或安全越权；v6 没有据此证明模型滥用权限 |
| `context_loss` | Mini SymPy 的抽取式摘要未保留此前符号结果中的定位信息（案例 D） | 摘要发生、信息缺失与重复共现，不等于已证明“摘要导致失败” |
| `invalid_patch` | 七个单次对照补丁被拒绝；三个格式、四个应用冲突（案例 E） | 未进入验证不能叫测试失败；不把修格式后的潜在结果当实测 |
| `test_misread` | 尚不能确认模型误读；三个 Flask 的 API 选择与冻结测试要求有差异，原枚举为 `invalid_patch`（案例 C） | 不能直接说模型看见完整隐藏测试却忽略它；公开反馈只有白名单异常类型 |
| `sandbox_failure` | v6 无技术失败；上述失败不是已确认的容器/环境故障 | 不等于安全边界已全面验证，也不能抹去先前轮次环境适配成本 |
| `budget_exhausted` | 九个累计 token 停止机制已确认（案例 A、B）；步数与 USD 1 上限未触发 | 提高上限是否能修好未知；不是 OpenAI 账户余额不足或 VM 内存耗尽 |
| `model_format_error` | 三个对照候选缺 git 文件头，但对照提示本身未清楚要求（案例 E） | 不能把系统输出契约不对齐全部归咎模型；无证据表明响应被截断 |

定位线索来自 S7 的 `slots[].localization.first_relevant_repository_tool`，是 evaluator-side
离线比对，不向 Agent 注入 gold 路径。未观察到不等于模型从未见过任何相关文本。

## 4. 具体案例

### A. 已找到代码，却在准备打补丁时耗尽预算

对象：Dense / pytest-11148。[Trace][dense-pytest] `seq 8/20/32/34/35`。

事实：

- `seq 8` 已读取相关 `src/_pytest/pathlib.py` 的 450–620 行；`seq 20` 再读 470–630，
  `seq 32` 再读 450–625。三次窗口明显重叠。
- `seq 34` 发起请求前累计 99,971 tokens，仅剩 29；上下文渲染估算为 11,895 tokens。
  本次 Provider 实际累计增量为 11,065，最终变成 111,036 / 100,000。
- 响应提议 `apply_patch`，但响应记账后触发预算检查，补丁没有执行；`seq 35` 终止。
  最终 12 次 Provider 决策、11 个工具结果、0 次补丁应用，成本约 USD 0.358。
- 最后请求已有摘要替换，七个原上下文项被摘要代替；不是“完全没做上下文管理”。
  原始 token 统计以 Provider 回传为准，渲染估算不能冒充账单 token。

解释边界：早期定位成功与重复读取是事实。低信息增益探索和重复输入开销可能挤压实现预算；
是否因为摘要缺失细节而重读、最后那个候选是否会通过，均未验证。
冻结实现的 `BudgetController.before_decision()` 只检查已消费量，响应到达后再检查超限；
这解释了“还有 29 tokens 却仍发起一次大请求”的机制，不是日志误计。

待验证改进：在新请求前按预计输入和输出余量做准入；避免同版本重复读取；为补丁与验证
保留预算。仅加一句“注意剩余预算”不足以解决此机制，因为本轮提示已经有剩余预算。
这些改进本片不实施，不能预先宣称挽救此任务。

### B. 补丁已应用，但没有得到验证机会

对象：Hybrid / SymPy-20590。[Trace][hybrid-sympy] `seq 8/11/26/32/34/35`。

事实：多次读取 `sympy/core/symbol.py`；`seq 32` 已应用向 `Symbol.__slots__` 加入
`_assumptions` 的修改。之后 `seq 34` 提议 `run_tests`，该次响应把累计 token 从
90,344 提高到 100,845，触发停止，验证工具没有执行。最终补丁状态是 applied，
但没有新补丁后的通过证据。冻结 gold 相关 `_print_helpers.py` 的成功 read/search 未被观察到。

解释边界：只能确定“落盘但未验证”和“定位证据有缺口”，不能称为 `tests_failed`，
也不能凭与 gold 修改文件不同就认定候选必错。不同实现可能解决同一问题。
此例展示为什么项目的成功条件必须同时要求最新补丁和绑定其版本的测试通过记录。

待验证改进：把验证余量纳入任务预算分配，减少探索阶段耗尽全部 token。
不放宽成功判定，不使用旧 baseline 或其他补丁版本的测试结果兜底。

### C. 两次补丁都能应用，为什么 Flask 仍失败

对象：Dense / Flask-4992；对照为同轮 Hybrid / Flask-4992。
[Dense Trace][dense-flask] `seq 14/15/17/23/26/29`；
[Hybrid Trace][hybrid-flask] `seq 17/18/20/23/26`。

事实：

- Dense 第一次验证 `seq 14` 失败，`seq 15` 消耗一次纠错；`seq 17/23` 读取修改后的当前源码，
  `seq 26` 第二个补丁应用成功，但 `seq 29` 仍为 TypeError 类测试失败。
- Mini、BM25 同题也各有两个应用成功的补丁和三次测试（baseline + 两次验证）。
  三个失败臂都没有补丁应用冲突或当前源码读取失败，最终仍采用 `mode` 参数方案。
- [冻结 source rows](../data/manifests/p7-v1/source-rows.json) 中，该 issue 举例的是
  `mode="b"`；冻结 evaluator 测试调用的是 `text=False`。这是接口约定差异，
  不是本轮发现后修改了评分。这里仅在离线分析中读取隐藏测试。
- Hybrid 首次也失败；`seq 18` 纠错、`seq 20` 读取当前版本、`seq 23` 改成
  `text: bool = True` 和相应的文本/二进制打开分支，`seq 26` 同一测试命令通过（1 passed）。
  四次正常验证失败纠错只有这一次挽救成功。

解释边界：可确认候选接口与冻结测试契约不匹配；不能把三例都写成“测试工具不工作”，
也不能武断写成“模型无视错误”。Agent 看到的是受限异常类别，不是完整隐藏调用。
issue 与测试对公共 API 的约定差异使该题存在评测解释上的限制；仍按原协议保留失败，
不据此加分，也不把 Hybrid 的一次成功当作普遍因果证明。

待验证改进：在未来独立任务的评测设计中，检查 issue 与目标 API 契约的一致性；
纠错时生成可从公开代码和反馈验证的假设。不要把 `text=False` 作为本题答案硬编码进 Agent，
也不要扩大隐藏测试泄露来换成功率。

### D. 摘要保留了元数据，却丢掉了符号位置

对象：Mini / SymPy-20590。[Trace][mini-sympy] `seq 11/43/44`。

事实：`seq 11` 已完成 `inspect_symbol(symbol="Boolean", path="sympy", kind="class")`；
随后 `seq 43` 在相同 patch revision 0 下再次提议相同参数，call id 不同但动作指纹相同。
重复调用被拒于执行之前，`seq 44` 以 `repeated_action` 停止；此时累计 88,749 tokens，
未触 100,000 上限，且没有补丁。

离线重建并核对该请求的 sealed prompt 可见：旧符号结果项被 `summary_replacement`
替换；抽取式摘要保留了 JSON 开头的扫描计数/跳过文件等元数据，没有保留结果中的
`sympy/logic/boolalg.py` 符号定位。完整结果仍在原 Trace/工件，不等于数据被删掉。

解释边界：信息丢失和重复查找均有证据，但不能证明两者的因果关系，也不能断言保留
一个路径就能解决 SymPy。重复保护按设计工作，关闭保护可能只会增加成本。

待验证改进：按工具 schema 摘要，优先保留符号名称、路径、行号；遇到同版本重复查询，
给出已有结果的有界提醒或复用，而不是无上限重新探索。先用离线契约测试证明摘要信息保留，
未来若获授权再在新任务上衡量效果。本片只记录，不改变冻结行为。

### E. 七个单次对照补丁为何都没进入成功验证

对象：七个单次对照失败，代表：[Django Trace][single-django]、[Xarray-5131 Trace][single-xarray]。
关键字段为 `/candidate_diff`、`/patch/error_code`、`/patch/state`、`/validation`，
完整七条见[失败清单](../results/p7-holdout/error-analysis-v6.json)。

事实：七个响应都正常完成、各一次请求、零重试；补丁均未应用，`validation=null`。
Django/Seaborn/Flask 缺少 `diff --git` 头部，属于三个 `invalid_diff`。
另四个已有补丁头，但旧 hunk 正文与所需当前源码不匹配，返回 `apply_conflict`；
Xarray-5131 的候选还没有上下文行。它们不是测试运行失败，更不是 VM 环境闪退。

对照提示只要求 ordinary unified diff，而 Agent 工具 schema 明确要求 git-style 头部，
因此已确认存在输出契约不对齐。先前只在内存里给三个既有候选补头，解析器虽能接受，
仍发现精确原文冲突；没有重新执行测试或调整分数。详情保留在[实验局限](experiments.md)。

解释边界：当前 6/8 对 1/8 是本版实现的实测，不是干净的“系统带来 6 倍能力”结论。
没有交互读取可能影响候选准确性，但不能从冲突直接断言检索漏召回。
既有候选补头后仍冲突，也不证明未来修正提示后的新候选一定失败。

待验证改进：未来对照实验统一输出契约和解析提示，保留单次、无工具反馈边界；
先以 Fake Provider 验证格式契约。用户已决定本项目不再追加该付费实验，v6 分数不变。

## 5. 改进 backlog（未实施、未证明提升）

这些是未来方向，不是 P7 收尾的新阻塞项。优先级按已观察机制与改动范围排序，
不是预测成功率；本片不新增依赖、预算、授权或实验。

| 优先级 | 最小改动方向 | 对应证据 | 首先如何离线验证 | 尚需验证的收益 |
|---|---|---|---|---|
| P1 | 请求前预计 token 准入、保留补丁/验证余量 | A/B；九个 token 终态 | 剩余额度小于预计输入时不发送请求；严格预算与最终验证版本绑定不退化 | 是否减少无补丁/未验证终止，而非更早失败 |
| P1 | 工具感知摘要，保留符号定位；重复查询有界复用 | D；A 的重叠窗口 | 摘要保留 path/line/name；同参数同 revision 复用，revision 改变不误用旧结果 | 是否减少重复探索而不损失修复信息 |
| P1 | 未来单次对照统一格式契约 | E | 相同可应用补丁在两种入口获得一致格式处理，仍零工具反馈 | 公平性改善可确定；任务成功增益未知 |
| P2 | 在冻结前审查公开需求与 evaluator API 契约 | C | 评测设计记录接口约定及歧义；gold 仍不进入 Agent 输入 | 是否减少因需求歧义造成的误解；不能回填本轮分数 |
| P2 | 用新任务测量定位不足与上下文压缩的各自作用 | SymPy、S7 缺失定位指标 | 记录完整排序元数据和每步时间，缺失显式 N/A | 检索改进、模型更换或 SFT 是否必要，目前不足以判定 |

不推荐据这 8 个已见任务直接进行 SFT、增加多 Agent 或放宽 Sandbox：本轮没有区分这些
选择的证据，也没有对应预算和独立验证集。未来若做改动应建立新版本，保留历史，
不挑选跨轮最佳结果拼出成功率。

## 6. 本片验收和下一片

`scripts/p7_error_analysis.py` 先通过 S7 的全部原始证据检查并与已保存 aggregate 比对，
再生成20条失败清单；`--check` 只比对，不写入。结果绑定原始 Trace/Result，
自动事实和人工解释分离，未知根因不强行打标签。离线检查不调用 Provider、Docker、
embedding 或数据库；不需要用户重新启动 VM 实验。

下一片 **P7-S9：README 与干净环境复现**。整理入口、架构、安装、示例、评测局限和安全文档，
在干净目录用非付费示例验证可运行性，记录环境与输出；Fake Provider 示例必须明确标注，
不能冒充新增真实模型成绩。S10 再完成 demo、发布前检查与交付。P8 MCP/Skill 仍后置。

[dense-pytest]: ../artifacts/p7-postdebug-v6-evidence-v1/results/p7-holdout/raw/p7-postdebug-v6-2a29ac299ffc5629/dense-only-v1/pytest-dev__pytest-11148/executions/attempt_004809f8f8dbf92b44d952550ee1da763b5ed2e30787e10e5650a514fa6fee0b/0000/trace.json
[hybrid-sympy]: ../artifacts/p7-postdebug-v6-evidence-v1/results/p7-holdout/raw/p7-postdebug-v6-2a29ac299ffc5629/hybrid-rrf-v1/sympy__sympy-20590/executions/attempt_378818df721bed6c38ceed5e14a72502d99033d56d8f4acc19fa14ef9761ede7/0000/trace.json
[dense-flask]: ../artifacts/p7-postdebug-v6-evidence-v1/results/p7-holdout/raw/p7-postdebug-v6-2a29ac299ffc5629/dense-only-v1/pallets__flask-4992/executions/attempt_96d04e71c984251286cc1a272f5f154c5e1e306aab0a00c5c6be17c238d6899b/0000/trace.json
[hybrid-flask]: ../artifacts/p7-postdebug-v6-evidence-v1/results/p7-holdout/raw/p7-postdebug-v6-2a29ac299ffc5629/hybrid-rrf-v1/pallets__flask-4992/executions/attempt_41d5291b77de52014e02123b279457b892d67e042ec746f97fbe4218431c2158/0000/trace.json
[mini-sympy]: ../artifacts/p7-postdebug-v6-evidence-v1/results/p7-holdout/raw/p7-postdebug-v6-2a29ac299ffc5629/mini-swe-safe-v1/sympy__sympy-20590/executions/attempt_c117c7e87251d3d46299cb7c33752c521ce6263ddd4d3f12c47cfe876393b727/0000/trace.json
[single-django]: ../artifacts/p7-postdebug-v6-evidence-v1/results/p7-holdout/raw/p7-postdebug-v6-2a29ac299ffc5629/single-shot-hybrid-context-v1/django__django-15902/executions/attempt_c15b5917c609f3bbb11ff2d4b5c3df9260fdee7aec2190c1bfe0c626e797d2c2/0000/trace.json
[single-xarray]: ../artifacts/p7-postdebug-v6-evidence-v1/results/p7-holdout/raw/p7-postdebug-v6-2a29ac299ffc5629/single-shot-hybrid-context-v1/pydata__xarray-5131/executions/attempt_5c6447b09194c90c62ff1b679f570a9c515f109dfd6c50e3dae089bc2db53ab3/0000/trace.json
