"""数据划分：trainval / val。

设计经 S3 侦察（scripts/s3_recon.py）大幅简化，三个原计划的复杂分支
实测都不需要：

  1. **分层键就是 brand_dir**
     实测每张图恰好只有 1 个类别（多类图 0 张）。这是 S2 决定
     「以目录名为权威标签」的直接结果 —— 一张图只属于一个品牌目录。
     原计划的「一图多类时取框数最多的类」逻辑因此不必要。

  2. **长尾兜底不必要**
     实测每类最少 4 张图（不是预想的 1 张），n_c==1 的类为 0。
     10% 配额下每个类都能分到 >=1 张 val 图，不存在「类在 val 缺席」。

  3. **簇约束近乎空操作，但保留**
     弱去重（同 brand_dir 内 file_size + (W,H) 全同）实测 158,593 簇 /
     158,654 图，99.96% 是单图簇，最大簇仅 2 张，0 个类的最大簇会超出
     val 配额。保留该约束是因为成本为零且逻辑正确。

     ⚠️ 诚实声明：这个弱判据只能抓「字节大小完全相同」的重复，
     抓不到缩放过、加过水印的近重复。真 pHash 推迟到训练阶段 ——
     setup 阶段从不训练，训练/验证泄漏对任何上报数字的影响为 0。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..seeds import seed_for


def check_split_ratio(
    n_val: int, n_total: int, n_classes: int, val_ratio: float
) -> tuple[bool, float, float]:
    """校验实际 val 比例是否合理。返回 (是否通过, 实际比例, 容许上界)。

    抽成纯函数是为了能直接测 —— 划分算法本身是自洽的，正常输入很难
    触发这个断言，所以它守的是**代码回归**而不是异常输入。

    上界不能只看 val_ratio：「每类至少 1 张 val」这条约束给出一个
    不可突破的结构下限 n_classes/n_total。类很小时（如 10 类 × 4 图）
    该下限是 25%，远高于 10% 的目标，此时超出容许带并非 bug。
    真实数据里 3,000/158,654 = 1.89%，远低于 10%，该项不起作用。
    """
    actual = n_val / n_total
    floor_ratio = n_classes / n_total
    upper = max(val_ratio * 1.3, floor_ratio * 1.15)
    return (val_ratio * 0.8 <= actual <= upper), actual, upper


@dataclass
class SplitReport:
    val_ratio_target: float = 0.0
    n_trainval: int = 0
    n_val: int = 0
    n_clusters: int = 0
    n_multi_clusters: int = 0
    classes_absent_in_val: list[str] = field(default_factory=list)
    classes_absent_in_trainval: list[str] = field(default_factory=list)
    structural_floor_ratio: float = 0.0
    per_class_val_ratio: pd.Series | None = None

    def as_rows(self) -> list[tuple[str, ...]]:
        total = self.n_trainval + self.n_val
        return [
            ("目标 val 比例", f"{self.val_ratio_target:.1%}", ""),
            ("trainval", f"{self.n_trainval:,}", f"{self.n_trainval / total:.2%}"),
            ("val", f"{self.n_val:,}", f"{self.n_val / total:.2%}"),
            ("结构下限", f"{self.structural_floor_ratio:.2%}",
             "= 类数/图数，「每类至少 1 张 val」不可突破的下限"),
            ("簇总数", f"{self.n_clusters:,}", f"其中多图簇 {self.n_multi_clusters:,}"),
            ("val 缺席的类", f"{len(self.classes_absent_in_val):,}", "应为 0"),
            ("trainval 缺席的类", f"{len(self.classes_absent_in_trainval):,}", "应为 0"),
        ]


def assign_clusters(images: pd.DataFrame, inventory: pd.DataFrame) -> pd.Series:
    """弱去重：同一 brand_dir 内 file_size 与 (W,H) 全同 → 同一簇。

    返回与 images 等长的 cluster_id。
    """
    m = images.merge(
        inventory[["image_rel", "image_bytes"]].rename(columns={"image_rel": "rel_path"}),
        on="rel_path",
        how="left",
    )
    if m["image_bytes"].isna().any():
        raise ValueError(
            f"{int(m['image_bytes'].isna().sum())} 张图在 inventory 里查不到 image_bytes"
        )
    return m.groupby(["brand_dir", "image_bytes", "img_w", "img_h"], dropna=False).ngroup()


def make_split(
    images: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    val_ratio: float = 0.10,
    seed_purpose: str = "split_agnostic",
) -> tuple[pd.DataFrame, SplitReport]:
    """按类分层、以簇为原子单位，划出 trainval / val。

    返回 (带 split 与 cluster_id 列的 images 副本, 报告)。
    """
    out = images.copy()
    out["cluster_id"] = assign_clusters(images, inventory).to_numpy()

    rng = seed_for(seed_purpose)
    rep = SplitReport(val_ratio_target=val_ratio)
    rep.n_clusters = int(out["cluster_id"].nunique())
    sizes = out.groupby("cluster_id").size()
    rep.n_multi_clusters = int((sizes > 1).sum())

    split = pd.Series("trainval", index=out.index, dtype="object")

    for _, grp in out.groupby("brand_dir", sort=True):
        # 簇为原子单位：同一簇的图必须同去向。
        #
        # 用 .groups（**索引标签**）而不是 .indices（子组内的位置）。
        # 早期写成 .indices + split.iloc[...] 时，那些 0..52 的局部位置
        # 被当成全量 Series 的位置，导致每个类都往整表最前面几十行写，
        # 结果 val 只有 216 张（应为 ~15,865）、2,995 个类在 val 缺席。
        cluster_labels = list(grp.groupby("cluster_id").groups.values())
        order = rng.permutation(len(cluster_labels))

        target = val_ratio * len(grp)
        # 至少给 val 留 1 张：实测最小类有 4 张图，0.1*4=0.4，
        # 不兜底的话该类在 val 会缺席
        target = max(target, 1.0)

        cum = 0
        for k in order:
            labels = cluster_labels[k]
            if cum >= target:
                break
            # 只要还没到配额就整簇放 val；允许最后一簇略微超出，
            # 因为簇不可拆。实测最大簇仅 2 张，超出量可忽略。
            split.loc[labels] = "val"
            cum += len(labels)

    out["split"] = split.to_numpy()
    rep.n_trainval = int((out["split"] == "trainval").sum())
    rep.n_val = int((out["split"] == "val").sum())

    # 内部自检：划分是整个 S3 的地基，宁可在这里炸也不要把
    # 一个悄悄错掉的划分传给下游。判据见 check_split_ratio 的说明。
    n_classes = out["brand_dir"].nunique()
    ok, actual, upper = check_split_ratio(rep.n_val, len(out), n_classes, val_ratio)
    rep.structural_floor_ratio = n_classes / len(out)
    if not ok:
        raise AssertionError(
            f"实际 val 比例 {actual:.4f} 偏离目标 {val_ratio:.2f} 过多"
            f"（容许上界 {upper:.4f}，其中「每类至少 1 张」的结构下限为 "
            f"{rep.structural_floor_ratio:.4f}）。检查簇约束或配额逻辑。"
        )

    per_cls = out.groupby("brand_dir")["split"].agg(
        n="size", n_val=lambda s: int((s == "val").sum())
    )
    per_cls["val_ratio"] = per_cls["n_val"] / per_cls["n"]
    rep.per_class_val_ratio = per_cls["val_ratio"]
    rep.classes_absent_in_val = sorted(per_cls.index[per_cls["n_val"] == 0])
    rep.classes_absent_in_trainval = sorted(
        per_cls.index[per_cls["n_val"] == per_cls["n"]]
    )

    return out, rep


def image_area_bin(ann: pd.DataFrame) -> pd.Series:
    """图级尺寸标签 = 该图**最大框**的 area_bin。

    用于 val2k_repr 的分层。84.6% 的图只有 1 个框，对它们这就是精确值。
    """
    idx = ann.groupby("image_id")["area"].idxmax()
    return ann.loc[idx].set_index("image_id")["area_bin"]
