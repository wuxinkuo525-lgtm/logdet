#!/usr/bin/env python
"""诊断：切片之间的交叉污染有多严重？

S6-L4 里出现一个我原本标注错的现象：只给 T1 成员加了 ε=0.30 的抖动，
但 SIZE_large（0.5808）与 DENSITY_1（0.6190）也远低于 CLEAN（0.9583），
而我在门里把它们标成了「未被加抖动，应接近 CLEAN」。

假设：这些切片**内部本身就含有 T1 成员**，所以被连带拉低。
S3 侦察已提示 T1 与 SIZE_small 几乎不重叠（全量 20,270 框里只有 28 个交集）
—— 因为被画面边缘裁掉的多是大目标，所以 T1 应该大量集中在 SIZE_large。

这不是 bug，但它是一条**必须写进报告的解读警告**：
切片之间不独立，某一轴上的差异会通过重叠泄漏到其它轴。
若模型在截断框上差，SIZE_large 也会跟着显得差，
此时**不能**得出「模型对大目标不行」的结论。

用法：
    python scripts/s6_diagnose_crosstalk.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.eval.slice_eval import default_slices, slice_members  # noqa: E402
from logodet.paths import P  # noqa: E402


def main() -> int:
    subset = sys.argv[1] if len(sys.argv) > 1 else "val_hard_pool"
    gt = json.loads(
        (P.artifact("eval") / "gt" / f"{subset}.json").read_text(encoding="utf-8")
    )
    ann = pd.read_parquet(P.artifact("tables") / "annotations.parquet")
    img = pd.read_parquet(P.artifact("splits") / "split_images.parquet")

    keep_ann = {int(a["id"]) for a in gt["annotations"]}
    keep_img = {int(i["id"]) for i in gt["images"]}
    specs = default_slices()
    members = slice_members(
        ann[ann["ann_id"].isin(keep_ann)], img[img["image_id"].isin(keep_img)], specs
    )

    names = [s.name for s in specs]
    t1 = members["T1_truncated"]

    print("=" * 78)
    print(f" 切片交叉污染诊断（子集 {subset}）")
    print("=" * 78)

    print(f"\nT1_truncated 共 {len(t1):,} 框。它在各切片里的占比：\n")
    print(f"  {'切片':<16} {'切片框数':>8} {'其中 T1':>8} {'T1 占该切片':>12} {'占全部T1':>10}")
    rows = []
    for n in names:
        s = members[n]
        inter = len(s & t1)
        share_in = inter / len(s) if s else 0.0
        share_of = inter / len(t1) if t1 else 0.0
        rows.append((n, len(s), inter, share_in, share_of))
        print(f"  {n:<16} {len(s):>8,} {inter:>8,} {share_in:>11.1%} {share_of:>9.1%}")

    print("\n结论：")
    big = [r for r in rows if r[0] != "T1_truncated" and r[3] > 0.15]
    for n, tot, inter, si, so in sorted(big, key=lambda r: -r[3]):
        print(f"  {n} 里有 {si:.1%} 是 T1 成员 —— 给 T1 加抖动必然把它一起拉低")
    print("\n  这解释了 L4 里 SIZE_large / DENSITY_1 为何明显低于 CLEAN。")
    print("  不是 bug，而是切片天然重叠的结果。")

    print("\n" + "=" * 78)
    print(" 完整的两两重叠矩阵（行占列的比例）")
    print("=" * 78)
    print("\n" + " " * 17 + "".join(f"{n[:11]:>13}" for n in names))
    for a in names:
        sa = members[a]
        line = f"  {a:<15}"
        for b in names:
            sb = members[b]
            v = len(sa & sb) / len(sa) if sa else 0.0
            line += f"{v:>12.1%} "
        print(line)

    out = P.artifact("metrics") / f"s6_crosstalk_{subset}.parquet"
    pd.DataFrame(
        [
            {
                "slice_a": a,
                "slice_b": b,
                "n_a": len(members[a]),
                "n_b": len(members[b]),
                "n_intersect": len(members[a] & members[b]),
                "frac_of_a": len(members[a] & members[b]) / len(members[a])
                if members[a] else 0.0,
            }
            for a in names
            for b in names
        ]
    ).to_parquet(out, index=False)
    print(f"\n矩阵已落盘 {out.relative_to(P.artifacts)}")

    print("\n" + "=" * 78)
    print(" 报告解读警告（必须写进 S7 的结论部分）")
    print("=" * 78)
    print("""
  切片之间不独立。若模型在某一轴上表现差，与该轴重叠高的其它切片
  也会跟着显得差 —— 这是重叠造成的，不是那一维本身的问题。

  具体到本数据集：截断的 logo 多为大目标（被画面边缘裁掉的通常尺寸大），
  所以 T1 与 SIZE_large 高度重叠。看到 SIZE_large 指标低时，
  必须先排除「是不是被 T1 拉低的」，才能谈「模型对大目标不行」。

  可行的排除方法：报 SIZE_large 时同时报 SIZE_large ∖ T1（去掉截断框）。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
