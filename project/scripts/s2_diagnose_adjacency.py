#!/usr/bin/env python
"""验证假设：R9 的类名错配是否为「下拉列表选错相邻项」。

观察到的错配对高度集中在字母序相邻的品牌上：
    littmann → liu gong        Esso → Esselte
    Glock → Gurkha             maybach → meyba
    vaude → vidyo              INOHERB → Isabel Maran
    snow peak → timbuk2        yonho → yutong

若成立，说明标注者是从一个**排序后的下拉列表**里选类别，偶尔点到了
相邻条目。这条结论直接决定标签权威来源：目录名来自爬取时用的品牌查询词，
而 XML 名来自易出错的人工下拉选择 —— 前者更可信。

判据：在全部 3,000 个类名的字典序列表里，错配对的**排名距离**分布。
若集中在很小的值（1~5），假设成立；若均匀散布在 0~3000，则不成立。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.paths import P  # noqa: E402

SUFFIX = re.compile(r"-\d+$")


def main() -> int:
    t = P.artifact("tables")
    ann = pd.read_parquet(t / "annotations.parquet")
    img = pd.read_parquet(t / "images.parquet")
    inv = pd.read_parquet(t / "file_inventory.parquet")

    # 参照列表：全部品牌目录名按不区分大小写的字典序排序，
    # 这是标注工具下拉列表最可能的呈现顺序
    all_dirs = sorted(inv["brand_dir"].unique(), key=lambda s: s.lower())
    rank = {d: i for i, d in enumerate(all_dirs)}

    a = ann.merge(img[["image_id", "brand_dir"]], on="image_id", how="left")
    mis = a[a["class_name_norm"] != a["brand_dir"]].copy()

    # 只看两边都在参照列表里的对（XML 名若不是任何目录名则无法定位排名）
    pairs = (
        mis.groupby(["brand_dir", "class_name_norm"]).size().reset_index(name="n_boxes")
    )
    pairs["rank_dir"] = pairs["brand_dir"].map(rank)
    pairs["rank_xml"] = pairs["class_name_norm"].map(rank)
    located = pairs.dropna(subset=["rank_dir", "rank_xml"]).copy()
    located["dist"] = (located["rank_dir"] - located["rank_xml"]).abs().astype(int)

    n_all = len(pairs)
    n_loc = len(located)
    print("=" * 74)
    print(" 假设检验：错配是否为下拉列表相邻项误选")
    print("=" * 74)
    print(f"\n参照列表长度（品牌目录数）: {len(all_dirs):,}")
    print(f"错配组合数                : {n_all:,}")
    print(f"其中两端都能在列表定位的  : {n_loc:,}")

    if n_loc == 0:
        print("无法定位，检验中止")
        return 1

    d = located["dist"].to_numpy()
    w = located["n_boxes"].to_numpy()

    print("\n排名距离分布（按组合数）：")
    for lo, hi, label in [(0, 1, "距离 1（紧邻）"), (2, 3, "距离 2-3"),
                          (4, 10, "距离 4-10"), (11, 100, "距离 11-100"),
                          (101, 10**9, "距离 >100")]:
        m = (d >= lo) & (d <= hi)
        print(f"  {label:<16} {m.sum():>5,} 组  ({m.sum() / n_loc:>6.1%})   "
              f"{w[m].sum():>7,} 框")

    print(f"\n距离中位数 : {int(np.median(d))}")
    print(f"距离均值   : {d.mean():.1f}")
    print(f"距离 <= 3 的占比 : {(d <= 3).mean():.1%}（组合）  "
          f"{w[d <= 3].sum() / w.sum():.1%}（框）")

    # 随机基线：若错配是随机选类，距离应近似均匀分布在 [1, N]，均值约 N/3
    n = len(all_dirs)
    print(f"\n随机误选的期望距离均值 ≈ {n / 3:.0f}（均匀分布假设下）")
    verdict = "成立" if d.mean() < n / 30 else "不成立"
    print(f"→ 假设{verdict}：实测均值 {d.mean():.1f} vs 随机期望 {n / 3:.0f}")

    print("\n距离最小的 15 组：")
    for r in located.nsmallest(15, "dist").itertuples():
        print(f"  距离 {r.dist:>2}  {r.n_boxes:>5,} 框  "
              f"{r.brand_dir!r:<30} → {r.class_name_norm!r}")

    print("\n距离最大的 8 组（若有大距离，说明存在另一类错误）：")
    for r in located.nlargest(8, "dist").itertuples():
        print(f"  距离 {r.dist:>5}  {r.n_boxes:>5,} 框  "
              f"{r.brand_dir!r:<30} → {r.class_name_norm!r}")

    # 拆开看：仅后缀差异 vs 完全不同
    located["same_brand"] = [
        SUFFIX.sub("", a_) == SUFFIX.sub("", b_)
        for a_, b_ in zip(located["brand_dir"], located["class_name_norm"])
    ]
    print("\n按是否同品牌拆分：")
    for flag, g in located.groupby("same_brand"):
        label = "同品牌（仅 -N 后缀差）" if flag else "跨品牌"
        print(f"  {label:<22} {len(g):>4} 组  {g['n_boxes'].sum():>7,} 框  "
              f"距离中位数 {int(g['dist'].median())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
