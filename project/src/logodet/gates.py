"""验证门框架（gates）。

每个阶段（S0…S7）都有一组机械判据。它们共享同一套结构，所以抽出来：

  - GateResult   一条门的结果：PASS / FAIL / WARN / SKIP + 明细行
  - run_gate     执行一条门，门自身崩溃也转成 FAIL 而不是糊一屏堆栈
  - GateRunner   汇总、打印、写 markdown 报告、决定退出码

设计约定：
  * **阻塞门**（blocking=True）FAIL → 退出码 1，禁止进入下一阶段
  * **非阻塞门** FAIL → 只标红并触发降级，不拦路
  * 所有门只输出数字和判据，不做主观判断
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable


@dataclass
class GateResult:
    gate: str
    name: str
    status: str  # PASS / FAIL / WARN / SKIP
    detail: str
    blocking: bool
    rows: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


def run_gate(
    fn: Callable[[], GateResult],
    gate: str,
    name: str,
    blocking: bool,
    *,
    verbose: bool = True,
) -> GateResult:
    """执行一条门。门内部抛异常时转成 FAIL，保留异常类型与消息。"""
    try:
        r = fn()
    except Exception as e:
        r = GateResult(gate, name, "FAIL", f"{type(e).__name__}: {e}", blocking=blocking)

    if verbose:
        print(f"\n[{r.gate}] {r.name} ... {r.status}")
        for line in r.detail.splitlines():
            print(f"      {line}")
        for row in r.rows:
            cells = list(row) + [""] * (3 - len(row))
            print(f"        {str(cells[0]):<30} {str(cells[1]):<24} {cells[2]}")
    return r


def md_table(rows: list[tuple[str, ...]], headers: tuple[str, ...] = ("项", "值", "说明")) -> str:
    """把明细行渲染成 markdown 表格。列数按最宽的一行对齐。"""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    width = max(width, 2)
    head = list(headers[:width]) + [""] * (width - len(headers))
    padded = [tuple(list(r) + [""] * (width - len(r))) for r in rows]
    lines = [
        "| " + " | ".join(head) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in padded]
    return "\n".join(lines)


class GateRunner:
    """收集一组门的结果，负责汇总打印与报告落盘。"""

    def __init__(self, stage: str, title: str) -> None:
        self.stage = stage
        self.title = title
        self.results: list[GateResult] = []

    def run(
        self, fn: Callable[[], GateResult], gate: str, name: str, blocking: bool
    ) -> GateResult:
        r = run_gate(fn, gate, name, blocking)
        self.results.append(r)
        return r

    def add(self, r: GateResult) -> GateResult:
        self.results.append(r)
        return r

    @property
    def blocking_failures(self) -> list[GateResult]:
        return [r for r in self.results if r.blocking and r.failed]

    def summary_rows(self) -> list[tuple[str, ...]]:
        return [
            (r.gate, r.name, "是" if r.blocking else "否", r.status,
             r.detail.splitlines()[0] if r.detail else "")
            for r in self.results
        ]

    def print_summary(self) -> int:
        print("\n" + "=" * 78)
        for r in self.results:
            flag = "阻塞" if r.blocking else "非阻塞"
            print(f"  {r.gate}  {r.name:<24} {flag:<6} {r.status}")
        print("=" * 78)

        fails = self.blocking_failures
        if fails:
            print(f"\n  {self.stage} 未通过：{len(fails)} 条阻塞门失败 → 禁止进入下一阶段")
            return 1
        warns = [r for r in self.results if r.status == "WARN"]
        tail = f"（含 {len(warns)} 条 WARN）" if warns else ""
        print(f"\n  {self.stage} 通过{tail}")
        return 0

    def write_report(self, path: Path, extra_sections: list[tuple[str, str]] | None = None) -> None:
        lines = [
            f"# {self.title}",
            "",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "由验证脚本自动生成，**不要手改** —— 重跑脚本即可刷新。",
            "",
            "## 验证门总览",
            "",
            md_table(self.summary_rows(), headers=("门", "名称", "阻塞", "结果", "说明")),
        ]
        for title, body in extra_sections or []:
            lines += ["", f"## {title}", "", body]
        for r in self.results:
            if r.rows:
                lines += ["", f"## {r.gate} {r.name} 明细", "", md_table(r.rows)]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
