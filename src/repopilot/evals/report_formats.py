"""Deterministic text/CSV views of the offline evaluation report."""

# Chinese report punctuation is intentional.
# ruff: noqa: RUF001

from __future__ import annotations

import csv
import io
from decimal import Decimal
from typing import Any


def _number(value: object, places: int = 3, divisor: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{Decimal(str(value)) / divisor:.{places}f}"


def _rate(value: dict[str, Any]) -> str:
    if value["rate"] is None:
        return f"N/A ({value['numerator']}/{value['denominator']})"
    return (
        f"{_number(Decimal(value['rate']) * 100, 1)}% ({value['numerator']}/{value['denominator']})"
    )


def render_slots_csv(report: dict[str, Any]) -> str:
    """Machine-readable rows; null is an empty cell, never a fabricated zero."""
    stream = io.StringIO(newline="")
    columns = (
        "arm_id",
        "task_id",
        "attempt_status",
        "task_success",
        "termination_reason",
        "issue_reproduced",
        "patch_applied",
        "tests_passed",
        "tool_steps",
        "total_tokens",
        "known_cost_usd",
        "provider_latency_ms",
        "event_wall_ms",
        "policy_rejected_tool_calls",
        "localization_basis",
        "localization_status",
        "file_recall_at_1",
        "file_recall_at_3",
        "file_recall_at_5",
        "mean_reciprocal_rank",
        "first_relevant_file_rank",
        "trace_path",
        "trace_sha256",
        "result_sha256",
    )
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for slot in report["slots"]:
        row = {key: slot["execution"].get(key) for key in columns}
        row.update(slot["localization"]["metrics"])
        row.update(
            localization_basis=slot["localization"]["basis"],
            localization_status=slot["localization"]["status"],
            trace_path=slot["trace_ref"]["relative_path"],
            trace_sha256=slot["trace_ref"]["sha256"],
            result_sha256=slot["result_ref"]["sha256"],
        )
        # Avoid spreadsheet formula interpretation when a report is opened in a UI.
        writer.writerow(
            {
                key: (
                    "'" + value
                    if isinstance(value, str) and value.startswith(("=", "+", "-", "@"))
                    else value
                )
                for key, value in row.items()
            }
        )
    return stream.getvalue()


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RepoPilot v6 离线聚合报告",
        "",
        "## 口径与边界",
        "",
        f"{report['distinct_task_count']} 个已见 Issue × 5 个配置，"
        f"共 {report['planned_slots']} 个槽位。"
        "每槽位仅一次执行；不是 40 个独立任务，也不是全量 SWE-bench 得分。",
        "本报告只读保留证据，不重跑模型、测试、检索或索引。原始 v6 分数不变。",
        "",
        "**单次生成对照的 diff 格式提示弱于 Agent 工具契约。** Hybrid 6/8 与单次生成 1/8 "
        "只能作本轮描述性对比，不能将 62.5 个百分点全归因于 harness，也不能宣称通用能力提升 6 倍。"
        "用户已决定不追加对照实验，按此局限收尾。",
        "",
        "## 执行、成本与时延",
        "",
        "| 配置 | 成功/计划槽位 | 技术失败 | 复现率 | 应用补丁槽位率 | 测试通过率 | "
        "本轮费用 USD | 平均模型工具步数 | 平均 token | "
        "平均 Provider 秒 | 任务 wall P50 / P95 秒 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in report["arms"]:
        e = arm["execution"]
        lines.append(
            f"| {arm['arm_id']} | {_rate(e['success_rate_all_slots'])} "
            f"| {e['technical_failure_slots']} | {_rate(e['reproduction_rate_completed'])} "
            f"| {_rate(e['patch_apply_rate_completed'])} "
            f"| {_rate(e['test_pass_rate_completed'])} | {_number(e['known_cost_usd'], 9)} "
            f"| {_number(e['tool_steps']['mean'], 2)} | {_number(e['total_tokens']['mean'], 1)} "
            f"| {_number(e['provider_latency_ms']['mean'], divisor=1000)} "
            f"| {_number(e['event_wall_ms']['p50'], divisor=1000)} / "
            f"{_number(e['event_wall_ms']['p95'], divisor=1000)} |"
        )
    lines += [
        "",
        "复现、补丁应用、测试通过率按 completed 槽位统计；分子分母同时展示。"
        "工具调用/补丁拒绝等按调用计数的指标见 JSON，不与槽位率混用。",
        "Provider 时间是 usage.latency_ms 累计；任务 wall 时间来自 runner started→terminal，"
        "包含准备、执行、验证和清理。P50/P95 使用 R7 线性插值；"
        "每臂只有 8 个样本，不代表生产尾延迟。",
        "单次生成的模型工具步数为 0；评测器依然会隔离应用补丁并运行验证，不等于没有评测器执行。",
        "单次生成臂的复现率指响应完成后评测器确认 baseline 失败，不代表模型自主复现或获得反馈。",
        "",
        "## 文件定位",
        "",
        "仅在评测端连接冻结 gold 文件名。忽略 FinalResult 中的占位零值。"
        "按保留 chunk 顺序去重成文件排名；无法还原的排名不补零。",
        "Dense/BM25/Hybrid 初始检索排名与 single-shot 实际渲染上下文不是同一个测量口径。"
        "Mini 不做初始检索，此项 N/A；工具内定位单独保存在每槽 JSON。",
        "",
        "| 配置 | Recall@1 | Recall@3 | Recall@5 | MRR |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in report["arms"]:
        summary = arm["localization"]
        metrics = summary.get("metrics", summary)
        cells = []
        for name in (
            "file_recall_at_1",
            "file_recall_at_3",
            "file_recall_at_5",
            "mean_reciprocal_rank",
        ):
            metric = metrics[name]
            count = metric["available_task_count"]
            if 0 < count < summary["task_count"]:
                cells.append(f"N/A（仅 {count}/{summary['task_count']} 可测）")
            else:
                cells.append(f"{_number(metric['mean'])} (n={count})")
        lines.append(f"| {arm['arm_id']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Recall@K = 前 K 个不同文件命中 gold 的比例；按可测任务宏平均，n 是该指标可测任务数。"
        "MRR 是首个相关文件排名倒数的宏平均。完整排名没有命中时为 0；证据不足则 N/A。",
        "Hybrid 与单次生成的 Recall@5 仅 6/8、MRR 仅 4/8 可恢复，"
        "全臂比较表标 N/A；JSON 中可测子集均值仅作诊断，不能与其他臂完整均值排名。",
        "首次相关文件工具耗时没有逐工具完成时间记录，标 N/A。下面的首次相关工具步数仅统计"
        "实际成功且非空的 read/search 返回，不含初始上下文，也不包括仅在请求中提到路径。",
        "",
        "| 配置 | 平均索引准备秒 | 平均初始查询秒 | "
        "首个相关 read/search 平均工具步（仅已观察子集） |",
        "| --- | --- | --- | --- |",
    ]
    for arm in report["arms"]:
        loc = arm["localization"]
        timing, tool = loc["timing"], loc["first_relevant_repository_tool"]
        lines.append(
            f"| {arm['arm_id']} | "
            f"{_number(timing['indexing_latency_ms']['mean'], divisor=1000)} "
            f"| {_number(timing['retrieval_latency_ms']['mean'], divisor=1000)} "
            f"| {_number(tool['mean_observed_tool_steps'], 2)} "
            f"(n={tool['observed_task_count']}/{loc['task_count']}) |"
        )
    lines += [
        "",
        "索引准备计时包含 ingest、chunking、稀疏索引和/或已缓存向量读取与数据库索引；"
        "不含此前 GPU 缓存生成，不是冷启动 embedding 性能基准。初始查询来自独立 runtime timer。"
        "Mini/单次生成没有独立初始检索计时，保留零值在聚合中按 N/A 处理。",
        "",
        "## 安全观测与预算",
        "",
    ]
    overall = report["overall"]
    lines += [
        f"保留 trace 中工具调用 {overall['tool_calls']} 次，"
        f"策略拒绝 {overall['policy_rejected_tool_calls']} 次。"
        "这是普通修复任务中的调用观测，不是恶意测试拦截率。",
        "路径穿越/命令注入等攻击拦截率、宿主机执行与 secret 暴露独立审计结果：N/A。"
        "不能因为没有失败日志就填 0 次暴露或 100% 安全。已有安全单测与本次真实实验是不同证据。",
        "",
        f"本轮已知模型估算费用：USD {report['accounting']['known_cost_usd']}。"
        f"历史带入：USD {report['accounting']['carried_cost_usd']}"
        f"（含未知费用预留 USD {report['historical_unknown_reserve_usd']}）。"
        f"累计应用记账：USD {report['accounting']['total_accounted_cost_usd']} / "
        f"{report['accounting']['budget_usd']}。不含云资源费用，不代表已核对的 Provider 账单。",
        "",
        "## 重建与一致性",
        "",
        f"已验证 {report['planned_slots']} 个槽位、"
        f"{report['verified_artifact_references']} 个嵌套 artifact 引用；"
        "缺失槽位为 0。任何缺失、重复、hash 漂移或 raw/report 不一致均拒绝生成正式聚合。",
        "历史 carry 读取已经验收过的 sealed plan，不在本步骤重新扫描历史各轮全部证据。",
        f"执行 source commit：`{report['source_commit']}`。",
        f"原始报告 SHA-256：`{report['sources']['report']['sha256']}`。",
        f"本聚合内容 SHA-256：`{report['content_sha256']}`。",
        "",
        "同目录 `aggregate-v6.json` 保留各槽明细、分母、可用性和 trace/result 字节引用；"
        "`aggregate-v6-slots.csv` 是便于筛选的扁平导出，空单元格表示不可测，不是零。",
        "",
    ]
    return "\n".join(lines)
