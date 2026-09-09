#!/usr/bin/env python
"""S5 步骤一：物化 COCO GT（派生视图）。

parquet 是唯一真源，COCO JSON 可随时删除重建 —— 所以永远不会出现
"手改了 JSON 导致真源漂移"。文件头写入源表 sha256 与配置指纹，
任何指标都能反查到它算在哪个确切的 GT 版本上。

产出到 runs/eval/gt/：
    val2k_repr.json       报总体指标用
    val_hard_pool.json    报切片对比用
    infer_union.json      推理并集（3,079 图），推理时的图像清单

用法：
    python scripts/s5_build_gt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.config import file_sha256, fingerprint, load_yaml  # noqa: E402
from logodet.data.adapters.to_coco_json import build_coco_gt, write_coco_gt  # noqa: E402
from logodet.paths import P  # noqa: E402


def main() -> int:
    t = P.artifact("tables")
    sp = P.artifact("splits")
    gt_dir = P.artifact("eval") / "gt"

    images = pd.read_parquet(sp / "split_images.parquet")
    ann = pd.read_parquet(t / "annotations.parquet")
    v2k = pd.read_parquet(sp / "val2k_repr.parquet")
    vhp = pd.read_parquet(sp / "val_hard_pool.parquet")

    src_fp = {
        "annotations.parquet": file_sha256(t / "annotations.parquet")[:16],
        "split_images.parquet": file_sha256(sp / "split_images.parquet")[:16],
    }
    cfg_fp = fingerprint(
        {"dataset": load_yaml("dataset.yaml"), "splits": load_yaml("splits.yaml")}
    )

    print("=" * 78)
    print(" S5 步骤一：物化 COCO GT")
    print("=" * 78)
    print(f"\n源表指纹 {src_fp}")
    print(f"配置指纹 {cfg_fp[:16]}…\n")

    subsets = {
        "val2k_repr": sorted(v2k["image_id"]),
        "val_hard_pool": sorted(vhp["image_id"]),
        "infer_union": sorted(set(v2k["image_id"]) | set(vhp["image_id"])),
    }

    rows = []
    for name, ids in subsets.items():
        coco = build_coco_gt(
            images, ann,
            image_ids=ids,
            class_agnostic=True,
            info={
                "subset": name,
                "source_tables": src_fp,
                "config_fingerprint": cfg_fp,
                "coord_convention": "0-based, xywh derived from half-open [x1,x2)",
                "class_agnostic": True,
            },
        )
        path = write_coco_gt(gt_dir / f"{name}.json", coco)
        sz = path.stat().st_size / 1024
        rows.append((name, len(coco["images"]), len(coco["annotations"]), sz))
        print(f"  {name:<16} {len(coco['images']):>6,} 图  "
              f"{len(coco['annotations']):>6,} 框  {sz:>8.1f} KB")

    # 一致性自检：并集的图数应等于两子集去重后的数量
    n_union = len(subsets["infer_union"])
    n_expect = len(set(subsets["val2k_repr"]) | set(subsets["val_hard_pool"]))
    assert n_union == n_expect, f"并集 {n_union} != 期望 {n_expect}"
    overlap = len(set(subsets["val2k_repr"]) & set(subsets["val_hard_pool"]))
    print(f"\n  两子集重叠 {overlap} 图，并集 {n_union} 图（= 实际推理量）")

    (gt_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_tables": src_fp,
                "config_fingerprint": cfg_fp,
                "subsets": {
                    n: {"n_images": i, "n_annotations": a} for n, i, a, _ in rows
                },
                "overlap_images": overlap,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n产物写入 {gt_dir}")
    print("下一步：python scripts/s5_l0_selfcheck.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
