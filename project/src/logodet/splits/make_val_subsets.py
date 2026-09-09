"""两个 val 子集：val2k_repr 与 val_hard_pool。

**它们是故意反着造的**，各自承担一件互相冲突的事：

    val2k_repr    保持分布（无偏）  → 报总体指标，可外推
    val_hard_pool 刻意富集（有偏）  → 报切片对比，不可外推

根本矛盾：代表性与样本量在长尾切片上直接冲突。SIZE small 只占 1.80%，
2,000 张代表性抽样里只剩约 44 个框，算不出可信的 AP；而把 small 富集到
足够量，总体分布就被扭曲了，总体指标不能再用。

类比：val2k_repr 是人口抽样调查（能推断全国平均身高），
val_hard_pool 是病例对照研究（能回答"某因素与患病有没有关系"，
但绝不能用这组人推断人群患病率）。

难例三轴（S3 侦察实测，S2 已把 P2/P3 降级）：
    T1 截断      全量 20,270 框 → val 期望 2,027   充足
    SIZE small   全量  3,490 框 → val 期望   349   略低于 400 配额，取全部
    DENSITY 5+   全量  7,773 框 → val 期望   777   充足
三轴几乎不重叠（T1∩small 仅 28 框），说明它们测的确实是不同的东西。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..seeds import seed_for


@dataclass
class SubsetReport:
    name: str
    n_images: int = 0
    n_boxes: int = 0
    strata_used: int = 0
    per_axis: dict[str, int] = field(default_factory=dict)
    shortfalls: list[str] = field(default_factory=list)


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """最大余数法分配整数配额，保证各层之和精确等于 total。

    直接 round 会因舍入误差导致总数偏离目标（2,000 抽成 1,997 之类）。
    """
    raw = weights / weights.sum() * total
    base = np.floor(raw).astype(int)
    remainder = total - base.sum()
    if remainder > 0:
        order = np.argsort(-(raw - base))
        base[order[:remainder]] += 1
    return base


def make_val2k_repr(
    val_images: pd.DataFrame,
    *,
    target: int = 2000,
    seed_purpose: str = "val2k_repr",
) -> tuple[pd.DataFrame, SubsetReport]:
    """按 supercat × img_area_bin × n_boxes_bin 严格比例抽样。

    val_images 需含 supercat / img_area_bin / n_boxes_bin 三列。
    """
    rng = seed_for(seed_purpose)
    rep = SubsetReport(name="val2k_repr")

    strata_cols = ["supercat", "img_area_bin", "n_boxes_bin"]
    grp = val_images.groupby(strata_cols, observed=True, dropna=False)
    keys = list(grp.groups.keys())
    counts = np.array([len(grp.groups[k]) for k in keys], dtype="float64")
    rep.strata_used = len(keys)

    quota = _largest_remainder(counts, min(target, len(val_images)))

    picked: list[np.ndarray] = []
    for k, q in zip(keys, quota):
        if q <= 0:
            continue
        idx = np.asarray(grp.groups[k])
        if q >= len(idx):
            picked.append(idx)
            if q > len(idx):
                rep.shortfalls.append(f"{k}: 需 {q} 只有 {len(idx)}")
        else:
            picked.append(rng.choice(idx, size=int(q), replace=False))

    sel = val_images.loc[np.concatenate(picked)].copy()
    rep.n_images = len(sel)
    rep.n_boxes = int(sel["n_boxes"].sum())
    return sel, rep


def make_val_hard_pool(
    val_images: pd.DataFrame,
    val_ann: pd.DataFrame,
    *,
    quota_per_axis: int = 400,
    seed_purpose: str = "val_hard_pool",
) -> tuple[pd.DataFrame, SubsetReport]:
    """按三轴定额富集，并配等量 clean 对照。

    以**框**定配额（每轴 >= quota_per_axis 个 GT 框），但选择单位是**图**
    —— 因为推理是按图做的，且一张图里的其他框也会进入评测。
    """
    rng = seed_for(seed_purpose)
    rep = SubsetReport(name="val_hard_pool")

    a = val_ann.merge(val_images[["image_id", "n_boxes"]], on="image_id", how="left")
    axes = {
        "T1": a["is_hard_t1"],
        "SIZE_small": a["area_bin"] == "small",
        "DENSITY_5plus": a["n_boxes"] >= 5,
    }

    chosen: set[int] = set()
    for name, mask in axes.items():
        sub = a[mask]
        avail_boxes = len(sub)
        # 按图聚合：每张图在该轴上贡献多少框
        by_img = sub.groupby("image_id").size().sort_values(ascending=False)
        rep.per_axis[name] = avail_boxes

        if avail_boxes <= quota_per_axis:
            # 不足配额 → 取全部
            chosen |= set(by_img.index)
            rep.shortfalls.append(
                f"{name}: 只有 {avail_boxes} 框 < 配额 {quota_per_axis}，已取全部"
            )
            continue

        # 随机顺序累加图，直到该轴框数达标。
        # 不按"贡献框数降序"取 —— 那样会系统性偏向多框图，
        # 让该轴的样本集中在密集场景，引入新的偏差。
        order = rng.permutation(by_img.index.to_numpy())
        cum = 0
        for img_id in order:
            if cum >= quota_per_axis:
                break
            if img_id not in chosen:
                chosen.add(int(img_id))
            cum += int(by_img.loc[img_id])

    # 等量 clean 对照：三轴都不命中的图
    hard_mask = axes["T1"] | axes["SIZE_small"] | axes["DENSITY_5plus"]
    hard_imgs = set(a.loc[hard_mask, "image_id"].unique())
    clean_pool = np.array(sorted(set(val_images["image_id"]) - hard_imgs))
    n_clean = min(len(chosen), len(clean_pool))
    if n_clean < len(chosen):
        rep.shortfalls.append(f"clean 对照只有 {len(clean_pool)} 张，少于难例 {len(chosen)}")
    clean_sel = rng.choice(clean_pool, size=n_clean, replace=False)

    all_ids = sorted(chosen | set(int(i) for i in clean_sel))
    sel = val_images[val_images["image_id"].isin(all_ids)].copy()
    sel["hard_pool_role"] = np.where(
        sel["image_id"].isin(chosen), "hard", "clean_control"
    )
    rep.n_images = len(sel)
    rep.n_boxes = int(sel["n_boxes"].sum())
    return sel, rep
