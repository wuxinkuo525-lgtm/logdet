#!/usr/bin/env python
"""S3 前置侦察：划分算法会踩到的四个边界情况。

沿用 S2 的做法 —— 先用真实数据量化边界，再动手写划分逻辑。
S2 的教训是：按假设直写会白做（P1 代理）或做错（坐标平移）。

四个问题：
  Q1 弱去重簇的规模分布 —— 簇是划分的原子单位。若存在超大簇，
     10% 的 val 配额可能被单个簇撑爆，导致某类全进 val 或全进 trainval。
  Q2 类别长尾 —— n_c==1 的类无法同时出现在两侧，必须登记缺席。
  Q3 val_hard_pool 三轴（T1 / small / density 5+）在 10% val 下的可用量，
     决定富集配额定多少才现实。
  Q4 分层键的选择 —— 一图多类时取"框数最多的类"，需确认多类图占比。

用法：
    python scripts/s3_recon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.paths import P  # noqa: E402

VAL_RATIO = 0.10


def main() -> int:
    t = P.artifact("tables")
    images = pd.read_parquet(t / "images.parquet")
    ann = pd.read_parquet(t / "annotations.parquet")
    inv = pd.read_parquet(t / "file_inventory.parquet")

    print("=" * 78)
    print(" S3 前置侦察")
    print("=" * 78)
    print(f"\nimages {len(images):,}  annotations {len(ann):,}")

    # ---- Q1 弱去重簇 -------------------------------------------------------
    # 判据：同一 brand_dir 内，file_size 完全相同且 (W,H) 相同 → 同一簇。
    # 真 pHash 推迟到训练阶段（setup 不训练，泄漏对上报数字影响为 0）。
    print("\n" + "-" * 78)
    print("Q1 弱去重簇规模分布")
    print("-" * 78)

    img = images.merge(
        inv[["image_rel", "image_bytes"]].rename(columns={"image_rel": "rel_path"}),
        on="rel_path", how="left",
    )
    missing_size = int(img["image_bytes"].isna().sum())
    print(f"  缺 image_bytes 的图: {missing_size}")

    key = ["brand_dir", "image_bytes", "img_w", "img_h"]
    grp = img.groupby(key, dropna=False)
    img["cluster_id"] = grp.ngroup()
    sizes = img.groupby("cluster_id").size()

    print(f"  簇总数        : {len(sizes):,}")
    print(f"  单图簇        : {int((sizes == 1).sum()):,} ({(sizes == 1).mean():.2%})")
    print(f"  多图簇        : {int((sizes > 1).sum()):,}")
    print(f"  多图簇覆盖图数 : {int(sizes[sizes > 1].sum()):,} "
          f"({sizes[sizes > 1].sum() / len(img):.2%} 的图像)")
    print(f"  最大簇        : {int(sizes.max())} 张")
    print(f"  簇大小分布    : "
          + ", ".join(f"{k}张={v:,}" for k, v in sizes.value_counts().nlargest(6).items()))

    # 超大簇会不会撑爆某类的 val 配额？
    per_class = img.groupby("brand_dir").agg(
        n_img=("image_id", "size"), max_cluster=("cluster_id", lambda s: s.value_counts().max())
    )
    per_class["val_quota"] = (per_class["n_img"] * VAL_RATIO).apply(np.ceil)
    over = per_class[per_class["max_cluster"] > per_class["val_quota"]]
    print(f"\n  最大簇 > val 配额 的类: {len(over):,} / {len(per_class):,} "
          f"({len(over) / len(per_class):.1%})")
    print("    这些类里，把整簇放进 val 会超配额；算法需允许略微超出或改放 trainval")
    if len(over):
        print("    样例（类 / 图数 / 最大簇 / val配额）：")
        for r in over.nlargest(5, "max_cluster").itertuples():
            print(f"      {r.Index[:28]:<30} {r.n_img:>4} {int(r.max_cluster):>4} "
                  f"{int(r.val_quota):>4}")

    # ---- Q2 类别长尾 -------------------------------------------------------
    print("\n" + "-" * 78)
    print("Q2 类别长尾")
    print("-" * 78)
    nimg = images.groupby("brand_dir").size()
    for thr in (1, 2, 5, 10, 20, 50):
        n = int((nimg <= thr).sum())
        print(f"  图数 <= {thr:<3} 的类: {n:>5,} ({n / len(nimg):>6.2%})")
    print(f"  图数 中位数 / 均值 : {nimg.median():.0f} / {nimg.mean():.1f}")
    print(f"  图数 最小 / 最大   : {nimg.min()} / {nimg.max()}")
    n1 = int((nimg == 1).sum())
    print(f"\n  → n_c == 1 的类 {n1} 个：无法同时出现在两侧，必须登记 absent_in_val")

    # 供 S3 之后选 Top-N 用
    print(f"\n  n_images >= 120 的类: {int((nimg >= 120).sum()):,}  "
          f"（每类 val 需 >=15 框的经验门槛）")

    # ---- Q3 val_hard_pool 三轴可用量 ---------------------------------------
    print("\n" + "-" * 78)
    print("Q3 难例三轴在 10% val 下的可用量")
    print("-" * 78)
    a = ann.merge(images[["image_id", "n_boxes"]], on="image_id", how="left")
    axes = {
        "T1 截断": a["is_hard_t1"],
        "SIZE small": a["area_bin"] == "small",
        "DENSITY 5+": a["n_boxes"] >= 5,
    }
    print(f"  {'轴':<14} {'全量框':>10} {'占比':>8} {'val 期望':>10} {'够 400?':>8}")
    for name, mask in axes.items():
        n_full = int(mask.sum())
        n_val = int(n_full * VAL_RATIO)
        ok = "是" if n_val >= 400 else "否"
        print(f"  {name:<14} {n_full:>10,} {mask.mean():>7.2%} {n_val:>10,} {ok:>8}")

    print("\n  三轴并集（去重后的框数）：")
    union = axes["T1 截断"] | axes["SIZE small"] | axes["DENSITY 5+"]
    print(f"    全量 {int(union.sum()):,} 框 / val 期望 {int(union.sum() * VAL_RATIO):,} 框")
    print(f"    涉及图像 全量 {a.loc[union, 'image_id'].nunique():,} 张")

    print("\n  两两重叠：")
    names = list(axes)
    for i, na in enumerate(names):
        row = " / ".join(
            f"{names[j]}={int((axes[na] & axes[names[j]]).sum()):,}" for j in range(len(names))
        )
        print(f"    {na:<14} {row}")

    # ---- Q4 分层键 ---------------------------------------------------------
    print("\n" + "-" * 78)
    print("Q4 分层键：一图多类的占比")
    print("-" * 78)
    multi_cls = images["n_classes_in_img"] > 1
    print(f"  含多个类别的图: {int(multi_cls.sum()):,} ({multi_cls.mean():.2%})")
    print("  → 占比很低，用「框数最多的类」作分层键的副作用可控")
    print(f"  每图类别数分布: "
          + ", ".join(f"{k}类={v:,}" for k, v in
                      images['n_classes_in_img'].value_counts().nlargest(5).items()))

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
