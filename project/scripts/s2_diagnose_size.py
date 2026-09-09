#!/usr/bin/env python
"""S2 诊断：尺寸分布与论文 Fig.5D 差 3pp 的根因。

实测（绝对像素面积，COCO 阈值 32²/96²）：
    small   1.80%   论文 4.81%   差 -3.01pp
    medium 29.81%   论文 29.79%  差 +0.02pp   ← 几乎完全吻合
    large  68.40%   论文 65.40%  差 +3.00pp

关键线索：**medium 几乎完全吻合，只有 small/large 两端偏**。
若是坐标基准判错或 W/H 取错，三档会一起偏移，不会只偏两端。
所以坐标解析大概率是对的，差异来自**阈值口径不同**。

最可能的解释：论文在**缩放后的图像**上统计（检测器输入尺寸，如 416 或 608），
box 面积随之缩小，于是更多框落进 small。

检验方法：反解出能精确复现论文分布的两个面积阈值 A1、A2，
再看 A1/32² 与 A2/96² 是否为同一个比例 —— 若是，就说明只差一个统一缩放，
且该比例的平方根就是论文用的缩放系数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.paths import P  # noqa: E402

PAPER = {"small": 4.81, "medium": 29.79, "large": 65.40}
S_THR, L_THR = 32**2, 96**2


def dist(area: np.ndarray, s_thr: float, l_thr: float) -> dict[str, float]:
    n = len(area)
    s = float((area < s_thr).sum()) / n * 100
    l = float((area > l_thr).sum()) / n * 100
    return {"small": s, "medium": 100 - s - l, "large": l}


def show(label: str, d: dict[str, float]) -> float:
    worst = max(abs(d[k] - PAPER[k]) for k in PAPER)
    print(f"  {label:<40} "
          f"s={d['small']:5.2f}% m={d['medium']:5.2f}% l={d['large']:5.2f}%  "
          f"最大偏差 {worst:4.2f}pp")
    return worst


def main() -> int:
    t = P.artifact("tables")
    ann = pd.read_parquet(t / "annotations.parquet")
    img = pd.read_parquet(t / "images.parquet")
    a = ann.merge(img[["image_id", "img_w", "img_h"]], on="image_id",
                  how="left", suffixes=("", "_img"))

    area = a["area"].to_numpy(dtype="float64")
    W = a["img_w"].to_numpy(dtype="float64")
    H = a["img_h"].to_numpy(dtype="float64")

    print("=" * 78)
    print(" 尺寸分布口径诊断")
    print("=" * 78)
    print(f"\n论文 Fig.5D 目标： s=4.81%  m=29.79%  l=65.40%")
    print(f"框数 {len(area):,}\n")

    print("--- 候选口径 ---")
    show("① 绝对像素面积（当前实现）", dist(area, S_THR, L_THR))

    # ② 反解阈值：让 small 恰好 4.81%、small+medium 恰好 34.60%
    q1 = float(np.quantile(area, PAPER["small"] / 100))
    q2 = float(np.quantile(area, (PAPER["small"] + PAPER["medium"]) / 100))
    print(f"\n--- 反解阈值 ---")
    print(f"  使 small=4.81% 的面积阈值 A1 = {q1:,.0f}   （COCO 32²={S_THR}）")
    print(f"  使 s+m=34.60% 的面积阈值 A2 = {q2:,.0f}   （COCO 96²={L_THR}）")
    r1, r2 = q1 / S_THR, q2 / L_THR
    print(f"  比例 A1/32² = {r1:.3f}     A2/96² = {r2:.3f}")
    if abs(r1 - r2) / max(r1, r2) < 0.15:
        k = np.sqrt((r1 + r2) / 2)
        print(f"  → 两个比例接近，可用单一缩放解释。等效线性缩放系数 k ≈ {k:.3f}")
        print(f"     即论文的框尺寸约为我们的 {1 / k:.2f} 分之一 → 图像被缩小到约 1/{1/k:.2f}")
    else:
        print("  → 两个比例差异较大，不能用单一缩放解释，需考虑别的口径")

    # ③ 把图像按长边缩放到若干常见检测输入尺寸，重算分布
    print("\n--- 假设：论文在缩放后的图像上统计 ---")
    long_side = np.maximum(W, H)
    best = (None, 1e9)
    for target in (320, 416, 448, 512, 600, 608, 640, 800, 1000, 1333):
        scale = np.minimum(1.0, target / long_side)  # 只缩不放
        d = dist(area * scale**2, S_THR, L_THR)
        w = show(f"③ 长边缩放到 {target}px", d)
        if w < best[1]:
            best = (target, w)
    # 也试不限制只缩小
    for target in (416, 608):
        scale = target / long_side
        d = dist(area * scale**2, S_THR, L_THR)
        w = show(f"③b 长边强制={target}px（可放大）", d)
        if w < best[1]:
            best = (f"{target}-force", w)

    # ④ 相对面积口径
    print("\n--- 假设：论文用相对面积（框面积 / 图面积）---")
    rel = area / (W * H)
    for s_thr, l_thr, label in ((0.01, 0.09, "0.01 / 0.09"),
                                (0.005, 0.05, "0.005 / 0.05"),
                                (0.002, 0.02, "0.002 / 0.02")):
        show(f"④ 相对面积阈值 {label}", dist(rel, s_thr, l_thr))

    print("\n" + "=" * 78)
    print(f" 最佳匹配：{best[0]}  最大偏差 {best[1]:.2f}pp")
    print("=" * 78)

    print("\n--- 面积分布参考 ---")
    for q in (1, 5, 10, 25, 50, 75, 90, 99):
        v = np.quantile(area, q / 100)
        print(f"  p{q:<3} area={v:>12,.0f}  sqrt={np.sqrt(v):>7.1f}px")
    print(f"\n  图像长边 中位数 {np.median(long_side):.0f}px  "
          f"p10 {np.quantile(long_side, .1):.0f}  p90 {np.quantile(long_side, .9):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
