#!/usr/bin/env python
"""S2 诊断：类名与品牌目录名不一致（R9）的分型分析。

建表跑出两个异常，需要定性才能决定标签权威来源：
    classes 行数 2,993，比期望的 3,000 少 7
    R9 命中 13,347 框（6.87%），远高于侦察阶段的"预期 ≈0"

侦察阶段用 regex 时把 `&amp;` 转义差异误判成不一致，所以以为是假警报。
用 lxml 正式解析后仍有 6.87%，说明存在**真实的标签噪声**。
本脚本回答：这些不一致是什么形态？该以目录名还是 XML 名为准？
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.paths import P  # noqa: E402

SUFFIX = re.compile(r"-\d+$")


def classify(dir_name: str, cls_name: str) -> str:
    if dir_name.lower() == cls_name.lower():
        return "仅大小写不同"
    sd, sc = SUFFIX.sub("", dir_name), SUFFIX.sub("", cls_name)
    if sd == sc:
        return "仅 -N 后缀不同"
    if sd.lower() == sc.lower():
        return "大小写 + 后缀"
    return "完全不同"


def main() -> int:
    t = P.artifact("tables")
    ann = pd.read_parquet(t / "annotations.parquet")
    img = pd.read_parquet(t / "images.parquet")
    inv = pd.read_parquet(t / "file_inventory.parquet")

    a = ann.merge(img[["image_id", "brand_dir"]], on="image_id", how="left")
    mis = a[a["class_name_norm"] != a["brand_dir"]].copy()

    print("=" * 74)
    print(" R9 分型：类名 vs 品牌目录名")
    print("=" * 74)
    print(f"\n不一致框数 : {len(mis):,} / {len(a):,}  ({len(mis) / len(a):.2%})")
    print(f"涉及图像   : {mis['image_id'].nunique():,}")
    print(f"涉及目录   : {mis['brand_dir'].nunique():,} / {a['brand_dir'].nunique():,}")

    mis["kind"] = [
        classify(d, c) for d, c in zip(mis["brand_dir"], mis["class_name_norm"])
    ]
    print("\n分型分布：")
    for k, v in mis["kind"].value_counts().items():
        print(f"  {k:<16} {v:>8,}  ({v / len(mis):>6.1%})")

    print("\n各分型样例（目录 → XML 类名，按框数降序）：")
    for kind in mis["kind"].unique():
        sub = mis[mis["kind"] == kind]
        top = sub.groupby(["brand_dir", "class_name_norm"]).size().sort_values(ascending=False)
        print(f"\n  [{kind}]  共 {len(sub):,} 框，{len(top):,} 种组合")
        for (d, c), n in top.head(6).items():
            print(f"    {n:>6,}  {d!r:<32} → {c!r}")

    print("\n" + "=" * 74)
    print(" 7 个「目录名从未作为类名出现」的目录")
    print("=" * 74)
    dirs = set(inv["brand_dir"].unique())
    cls = set(ann["class_name_norm"].unique())
    missing = sorted(dirs - cls)
    print(f"\n目录数 {len(dirs)}   类名数 {len(cls)}   缺口 {len(missing)}")
    for d in missing:
        sub = a[a["brand_dir"] == d]
        vc = sub["class_name_norm"].value_counts()
        items = ", ".join(f"{k!r}×{v}" for k, v in vc.head(3).items())
        print(f"  {d!r:<24} {len(sub):>4} 框 → {items}")

    # 反向：是否有类名对应多个目录（同一类被拆到多个目录）
    print("\n" + "=" * 74)
    print(" 一个类名横跨多个品牌目录的情况")
    print("=" * 74)
    span = a.groupby("class_name_norm")["brand_dir"].nunique()
    multi = span[span > 1].sort_values(ascending=False)
    print(f"\n横跨多目录的类名数 : {len(multi):,}")
    for c, n in multi.head(10).items():
        ds = sorted(a[a["class_name_norm"] == c]["brand_dir"].unique())
        print(f"  {c!r:<30} 跨 {n} 个目录: {ds[:4]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
