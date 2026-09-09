#!/usr/bin/env python
"""S3 步骤一：数据划分 + 两个 val 子集。

产出到 runs/splits/：
    split_images.parquet   全量 158,654 图，带 split / cluster_id / 子集标记
    val2k_repr.parquet     代表性子集，报总体指标
    val_hard_pool.parquet  难例富集池，报切片对比
    split_report.json      划分统计 + 配置指纹

用法：
    python scripts/s3_make_splits.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.config import fingerprint, load_yaml  # noqa: E402
from logodet.paths import P  # noqa: E402
from logodet.splits.make_split import image_area_bin, make_split  # noqa: E402
from logodet.splits.make_val_subsets import (  # noqa: E402
    make_val2k_repr,
    make_val_hard_pool,
)


def main() -> int:
    cfg = load_yaml("splits.yaml") if (PROJECT_DIR / "configs" / "splits.yaml").is_file() else {}
    val_ratio = float(cfg.get("val_ratio", 0.10))
    target2k = int(cfg.get("val2k_target", 2000))
    quota = int(cfg.get("hard_pool_quota_per_axis", 400))

    t = P.artifact("tables")
    splits_dir = P.artifact("splits")

    images = pd.read_parquet(t / "images.parquet")
    ann = pd.read_parquet(t / "annotations.parquet")
    inv = pd.read_parquet(t / "file_inventory.parquet")

    print("=" * 78)
    print(" S3 步骤一：数据划分")
    print("=" * 78)
    t0 = time.time()

    # ---- 划分 --------------------------------------------------------------
    print(f"\n[1/3] 分层划分（val_ratio={val_ratio:.0%}，簇为原子单位）...")
    img, rep = make_split(images, inv, val_ratio=val_ratio)
    for k, v, note in rep.as_rows():
        print(f"      {k:<18} {v:<14} {note}")

    if rep.classes_absent_in_val:
        print(f"      [warn] {len(rep.classes_absent_in_val)} 个类在 val 缺席，"
              f"样例 {rep.classes_absent_in_val[:5]}")

    r = rep.per_class_val_ratio
    print(f"      逐类 val 比例  中位数={r.median():.3f} "
          f"p05={r.quantile(.05):.3f} p95={r.quantile(.95):.3f} "
          f"min={r.min():.3f} max={r.max():.3f}")

    # ---- 图级派生列 --------------------------------------------------------
    img["img_area_bin"] = img["image_id"].map(image_area_bin(ann)).astype("string")

    # ---- 两个 val 子集 -----------------------------------------------------
    val_img = img[img["split"] == "val"].copy()
    val_ann = ann[ann["image_id"].isin(set(val_img["image_id"]))].copy()
    print(f"\n      val 集：{len(val_img):,} 图 / {len(val_ann):,} 框")

    print(f"\n[2/3] 构造 val2k_repr（{target2k} 图，108 层严格比例）...")
    v2k, rep2k = make_val2k_repr(val_img, target=target2k)
    print(f"      实际抽出 {rep2k.n_images:,} 图 / {rep2k.n_boxes:,} 框，"
          f"用到 {rep2k.strata_used} 个层")
    for s in rep2k.shortfalls[:5]:
        print(f"      [warn] {s}")

    print(f"\n[3/3] 构造 val_hard_pool（每轴 >= {quota} 框 + 等量 clean 对照）...")
    vhp, rephp = make_val_hard_pool(val_img, val_ann, quota_per_axis=quota)
    print(f"      {rephp.n_images:,} 图 / {rephp.n_boxes:,} 框")
    print(f"      难例图 {int((vhp['hard_pool_role'] == 'hard').sum()):,} / "
          f"clean 对照 {int((vhp['hard_pool_role'] == 'clean_control').sum()):,}")
    for name, n in rephp.per_axis.items():
        print(f"      轴 {name:<16} val 内可用 {n:,} 框")
    for s in rephp.shortfalls:
        print(f"      [warn] {s}")

    # ---- 落盘 --------------------------------------------------------------
    img["in_val2k_repr"] = img["image_id"].isin(set(v2k["image_id"]))
    img["in_val_hard_pool"] = img["image_id"].isin(set(vhp["image_id"]))

    img.to_parquet(splits_dir / "split_images.parquet", index=False)
    v2k.to_parquet(splits_dir / "val2k_repr.parquet", index=False)
    vhp.to_parquet(splits_dir / "val_hard_pool.parquet", index=False)

    # 供后续选 Top-N 参考
    n_img_per_class = images.groupby("brand_dir").size()
    topn_curve = {
        str(thr): int((n_img_per_class >= thr).sum())
        for thr in (50, 80, 100, 120, 150, 200)
    }

    report = {
        "config_fingerprint": fingerprint(cfg),
        "val_ratio_target": val_ratio,
        "n_images": int(len(img)),
        "n_trainval": rep.n_trainval,
        "n_val": rep.n_val,
        "val_ratio_actual": rep.n_val / len(img),
        "n_clusters": rep.n_clusters,
        "n_multi_clusters": rep.n_multi_clusters,
        "classes_absent_in_val": rep.classes_absent_in_val,
        "classes_absent_in_trainval": rep.classes_absent_in_trainval,
        "per_class_val_ratio": {
            "median": float(r.median()), "p05": float(r.quantile(.05)),
            "p95": float(r.quantile(.95)), "min": float(r.min()), "max": float(r.max()),
        },
        "val2k_repr": {
            "n_images": rep2k.n_images, "n_boxes": rep2k.n_boxes,
            "strata_used": rep2k.strata_used, "shortfalls": rep2k.shortfalls,
        },
        "val_hard_pool": {
            "n_images": rephp.n_images, "n_boxes": rephp.n_boxes,
            "per_axis_boxes": rephp.per_axis, "shortfalls": rephp.shortfalls,
            "n_hard": int((vhp["hard_pool_role"] == "hard").sum()),
            "n_clean_control": int((vhp["hard_pool_role"] == "clean_control").sum()),
        },
        "topn_candidate_curve": topn_curve,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (splits_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n产物写入 {splits_dir}")
    print(f"总耗时 {time.time() - t0:.1f}s。下一步：python scripts/s3_verify_splits.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
