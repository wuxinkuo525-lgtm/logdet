#!/usr/bin/env python
"""用刚标好的 60 张验证代理修正方案是否真的有效。

背景：VLM 预标显示两个代理都远未达标
    P2 precision 0.143（17/25 是嵌套标注）
    P3 precision 0.176（14 自然形状 + 13 类内双模态）

**P2 的失败模式是可修的**：
    嵌套 → 一个框几乎被另一个完全包含，containment ≈ 1
    真遮挡 → 两个同尺度实例部分互压，containment 明显 < 1
  加判据：containment = area(a∩b)/min(area_a, area_b) < 阈值

**P3 的失败模式不可用阈值修**：
    根因是类内「徽标 vs 字标」双模态（lexus-1 目录里既有 Ⓛ 徽标图 ar≈1.2
    又有 LEXUS 字标图 ar≈4.7），MAD z-score 必然把少数模态判为离群。
    这是标签结构问题，调 z 阈值只会同比例地放大或缩小两类，precision 不变。
  可测的补救：只在**长宽比单模态**的类上启用 P3。

验证方式：把修正规则套回 25/35 张已标样本，看存活样本的 precision。
这是无需重新标注就能量化修法收益的办法。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.paths import P  # noqa: E402


def pairwise_features(ann: pd.DataFrame) -> pd.DataFrame:
    """为每个框算出 (最佳伙伴的 IoF, containment, 面积比)。

    containment 用 min(area) 作分母，所以「小框完全落在大框内」时 → 1.0，
    正是嵌套标注的特征；两个同尺度实例部分互压时明显 < 1。
    """
    iof = np.zeros(len(ann), dtype="float32")
    contain = np.zeros(len(ann), dtype="float32")
    ratio = np.ones(len(ann), dtype="float32")
    pos = {a: i for i, a in enumerate(ann["ann_id"].to_numpy())}

    multi = ann.groupby("image_id")["ann_id"].transform("size") >= 2
    for _, g in ann[multi].groupby("image_id", sort=False):
        x1 = g["x1"].to_numpy("float64"); y1 = g["y1"].to_numpy("float64")
        x2 = g["x2"].to_numpy("float64"); y2 = g["y2"].to_numpy("float64")
        area = (x2 - x1) * (y2 - y1)
        ids = g["ann_id"].to_numpy()

        ix1 = np.maximum(x1[:, None], x1[None, :])
        iy1 = np.maximum(y1[:, None], y1[None, :])
        ix2 = np.minimum(x2[:, None], x2[None, :])
        iy2 = np.minimum(y2[:, None], y2[None, :])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        np.fill_diagonal(inter, 0.0)

        r_iof = inter / np.maximum(area[:, None], 1e-9)
        min_area = np.minimum(area[:, None], area[None, :])
        r_con = inter / np.maximum(min_area, 1e-9)

        best = r_iof.argmax(axis=1)
        rows = np.arange(len(ids))
        for k, aid in enumerate(ids):
            j = pos[aid]
            b = best[k]
            iof[j] = r_iof[k, b]
            contain[j] = r_con[k, b]
            ratio[j] = area[k] / max(area[b], 1e-9)

    out = ann[["ann_id", "image_id", "class_id", "area"]].copy()
    out["iof"] = iof
    out["containment"] = contain
    out["area_ratio"] = ratio
    return out


def main() -> int:
    t = P.artifact("tables")
    ann = pd.read_parquet(t / "annotations.parquet")
    sheet = pd.read_csv(P.artifact("slices") / "review_sheet.csv")

    print("=" * 78)
    print(" 代理修正方案验证")
    print("=" * 78)

    # ---------------- P2 ----------------
    print("\n【P2】加入 containment 判据")
    feat = pairwise_features(ann)
    lab = sheet[sheet["slice"] == "P2"][["ann_id", "vlm_label", "vlm_category"]]
    m = lab.merge(feat, on="ann_id", how="left")

    print("\n已标样本的 containment 分布（按归因）：")
    for cat, g in m.groupby("vlm_category"):
        print(f"  {cat:<12} n={len(g):>2}  containment "
              f"中位数={g['containment'].median():.3f} "
              f"min={g['containment'].min():.3f} max={g['containment'].max():.3f}")

    print("\n扫描 containment 上限，看存活样本的 precision：")
    print(f"  {'上限':<8} {'存活':<6} {'其中1':<6} {'其中0':<6} {'precision':<10} {'全量池子':<10}")
    best = None
    for thr in (1.01, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50):
        sel = m[(m["iof"] >= 0.15) & (m["containment"] < thr)]
        n1 = int((sel["vlm_label"] == "1").sum())
        n0 = int((sel["vlm_label"] == "0").sum())
        prec = n1 / (n1 + n0) if (n1 + n0) else float("nan")
        pool = int(((feat["iof"] >= 0.15) & (feat["containment"] < thr)).sum())
        print(f"  {thr:<8.2f} {len(sel):<6} {n1:<6} {n0:<6} "
              f"{prec if not np.isnan(prec) else float('nan'):<10.3f} {pool:<10,}")
        if (n1 + n0) >= 3 and (best is None or prec > best[1]):
            best = (thr, prec, pool)

    if best:
        print(f"\n  最佳 containment 上限 = {best[0]:.2f}  "
              f"precision = {best[1]:.3f}  全量池子 = {best[2]:,} 框")
        if best[1] >= 0.6:
            print("  → 达到 0.60 门槛，P2 可用修正版")
        else:
            print("  → 仍未达 0.60 门槛")

    # 三个真遮挡样本的具体数值，看它们是否可分
    print("\n  三个真遮挡样本 vs 嵌套样本的可分性：")
    for cat in ("real_occl", "nesting"):
        g = m[m["vlm_category"] == cat]
        print(f"    {cat:<10} iof∈[{g['iof'].min():.3f},{g['iof'].max():.3f}]  "
              f"containment∈[{g['containment'].min():.3f},{g['containment'].max():.3f}]  "
              f"area_ratio∈[{g['area_ratio'].min():.2f},{g['area_ratio'].max():.2f}]")

    # 真正的判别特征是面积比：真遮挡是两个同尺度实例部分互压，
    # 嵌套是小子框套在大 lockup 框里，面积比会很悬殊。
    print("\n  加入面积比判据（要求两框尺度接近）：")
    print(f"  {'面积比下限':<10} {'上限':<8} {'存活':<6} {'1':<4} {'0':<4} "
          f"{'precision':<10} {'全量池子':<10}")
    m["ar_sym"] = np.minimum(m["area_ratio"], 1 / m["area_ratio"].clip(lower=1e-9))
    feat["ar_sym"] = np.minimum(
        feat["area_ratio"], 1 / feat["area_ratio"].clip(lower=1e-9)
    )
    best2 = None
    for lo in (0.3, 0.4, 0.5, 0.6, 0.7):
        sel = m[(m["iof"] >= 0.15) & (m["containment"] < 0.8) & (m["ar_sym"] >= lo)]
        n1 = int((sel["vlm_label"] == "1").sum())
        n0 = int((sel["vlm_label"] == "0").sum())
        prec = n1 / (n1 + n0) if (n1 + n0) else float("nan")
        pool = int(
            ((feat["iof"] >= 0.15) & (feat["containment"] < 0.8)
             & (feat["ar_sym"] >= lo)).sum()
        )
        print(f"  {lo:<10.2f} {'0.80':<8} {len(sel):<6} {n1:<4} {n0:<4} "
              f"{prec:<10.3f} {pool:<10,}")
        if (n1 + n0) >= 3 and (best2 is None or prec > best2[1]):
            best2 = (lo, prec, pool)

    if best2:
        print(f"\n  最佳：面积比下限 {best2[0]:.2f} + containment<0.80  "
              f"→ precision {best2[1]:.3f}，全量池子 {best2[2]:,} 框")
        print(f"  {'达到' if best2[1] >= 0.6 else '仍未达到'} 0.60 门槛")

    # ---------------- P3 ----------------
    print("\n" + "=" * 78)
    print("\n【P3】只在长宽比单模态的类上启用")
    a = ann.copy()
    # 双模态判据：类内 log(ar) 的极差 / MAD 很大，或用双峰间隔
    g = a.groupby("class_id")["log_ar"]
    stat = g.agg(n="size", std="std", ptp=lambda s: s.max() - s.min()).reset_index()
    # 简单可解释的判据：类内 log_ar 极差 > 1.0（即最宽/最窄相差 e^1≈2.7 倍）视为多模态
    stat["multimodal"] = stat["ptp"] > 1.0
    print(f"  类数 {len(stat):,}   判为多模态（类内 log_ar 极差>1.0）"
          f"{int(stat['multimodal'].sum()):,} 类 "
          f"({stat['multimodal'].mean():.1%})")

    uni = set(stat.loc[~stat["multimodal"], "class_id"])
    lab3 = sheet[sheet["slice"] == "P3"][["ann_id", "vlm_label", "vlm_category"]]
    m3 = lab3.merge(a[["ann_id", "class_id"]], on="ann_id", how="left")
    m3["class_unimodal"] = m3["class_id"].isin(uni)

    print("\n  已标 P3 样本按「所属类是否单模态」拆分：")
    for flag, g3 in m3.groupby("class_unimodal"):
        n1 = int((g3["vlm_label"] == "1").sum())
        n0 = int((g3["vlm_label"] == "0").sum())
        prec = n1 / (n1 + n0) if (n1 + n0) else float("nan")
        label = "单模态类" if flag else "多模态类"
        print(f"    {label}  n={len(g3):>2}  1={n1} 0={n0}  precision={prec:.3f}")

    print("\n  已标样本里 bimodal 归因的样本，其类是否真被判为多模态：")
    bi = m3[m3["vlm_category"] == "bimodal"]
    print(f"    bimodal 样本 {len(bi)} 个，其中落在多模态类的 "
          f"{int((~bi['class_unimodal']).sum())} 个 "
          f"({(~bi['class_unimodal']).mean():.0%})")

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
