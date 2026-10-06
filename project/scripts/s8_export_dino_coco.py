#!/usr/bin/env python
"""S8 步骤一：为 DINO/MMDetection 导出 3000 细类 COCO 格式标注。

**这是纯格式转换，不是重新划分。** 图片属于 trainval 还是 val 的关系
完全来自已经冻结的 `runs/splits/split_images.parquet`，本脚本只读不改。

category_id 直接用 classes.parquet 的 class_id（1..3000），不塌缩成单类
——单类版本的 GT（`runs/eval/gt/*.json`）继续保留给 class-agnostic 评测用，
两者互不覆盖、互不干扰。

产出（`runs/coco/`，全部可删可重建，parquet 才是真源）：
    instances_train_dino.json   trainval 侧，3000 类
    instances_val_dino.json     val 侧（完整 17,216 图，不是 val2k/hard_pool 子集）
    fine_to_super_mapping.json  class_id -> superclass_id 的确定性映射，
                                 供 9-superclass 辅助 hierarchy 头使用

用法：
    python scripts/s8_export_dino_coco.py
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


def verify_prerequisites(classes: pd.DataFrame, ann: pd.DataFrame, split: pd.DataFrame) -> None:
    """转换前置检查。任何一条不满足就直接抛异常停下，不静默降级、不自动改数据。"""
    problems: list[str] = []

    cid = classes["class_id"].sort_values().to_numpy()
    if not (cid.min() == 1 and cid.max() == 3000 and len(cid) == 3000 and classes["class_id"].is_unique):
        problems.append(f"classes.parquet 的 class_id 不是完整无重复的 1..3000（实际 {len(cid)} 个，"
                         f"范围 [{cid.min() if len(cid) else '-'}, {cid.max() if len(cid) else '-'}]）")

    ann_cid = ann["class_id"]
    if ann_cid.isna().any():
        problems.append("annotations.class_id 含 NaN")
    if ann_cid.nunique() != 3000:
        problems.append(f"annotations 中出现的 class_id 种类数 = {ann_cid.nunique()}，不是 3000")

    m1 = ann[["class_id", "class_name"]].drop_duplicates()
    if not m1["class_id"].is_unique:
        problems.append("同一个 class_id 在 annotations 里对应了不止一个 class_name")
    joined = m1.merge(classes[["class_id", "class_name"]], on="class_id", suffixes=("_ann", "_cls"))
    if (joined["class_name_ann"] != joined["class_name_cls"]).any():
        problems.append("annotations 与 classes 两表对同一 class_id 给出的 class_name 不一致")

    ann_img = ann.merge(split[["image_id", "split"]], on="image_id", how="left")
    if ann_img["split"].isna().any():
        problems.append("存在annotations指向的image_id在split表里找不到")
    cid_by_split = ann_img.groupby("split")["class_id"].apply(set)
    if set(cid_by_split.get("trainval", set())) != set(cid_by_split.get("val", set())):
        only_val = set(cid_by_split.get("val", set())) - set(cid_by_split.get("trainval", set()))
        only_tv = set(cid_by_split.get("trainval", set())) - set(cid_by_split.get("val", set()))
        problems.append(f"trainval 与 val 使用的 class_id 集合不一致："
                         f"仅val独有 {sorted(only_val)[:10]}，仅trainval独有 {sorted(only_tv)[:10]}")

    sc_per_class = classes.groupby("class_id")["supercat"].nunique()
    if (sc_per_class > 1).any():
        problems.append(f"classes.parquet 里有 {(sc_per_class > 1).sum()} 个 class_id 对应不止一个 supercat")
    ann_sc_per_class = ann.groupby("class_id")["supercat"].nunique()
    if (ann_sc_per_class > 1).any():
        problems.append(f"annotations.parquet 里有 {(ann_sc_per_class > 1).sum()} 个 class_id 对应不止一个 supercat")
    cross = classes[["class_id", "supercat"]].merge(
        ann[["class_id", "supercat"]].drop_duplicates(), on="class_id", suffixes=("_cls", "_ann")
    )
    if (cross["supercat_cls"] != cross["supercat_ann"]).any():
        problems.append("classes 表与 annotations 表对同一 class_id 给出的 supercat 不一致")

    if problems:
        raise RuntimeError(
            "前置校验未通过，已停止（不会自动重新划分或修复）：\n" + "\n".join(f"  - {p}" for p in problems)
        )


def build_fine_to_super(classes: pd.DataFrame) -> dict:
    """class_id(1..3000) -> superclass_id(1..9) 的确定性映射。

    superclass_id 按 supercat 名称的字典序分配（而不是「先出现先编号」），
    这样换一次运行环境、换一次 pandas 版本，编号都不会变。
    """
    supercats = sorted(classes["supercat"].unique())
    assert len(supercats) == 9, f"超类数应为 9，实际 {len(supercats)}"
    super_id_of = {name: i + 1 for i, name in enumerate(supercats)}

    fine_to_super = {
        int(r.class_id): super_id_of[r.supercat] for r in classes.itertuples()
    }
    return {
        "superclasses": [{"id": sid, "name": name} for name, sid in super_id_of.items()],
        "fine_to_super": fine_to_super,
    }


def main() -> int:
    t = P.artifact("tables")
    sp = P.artifact("splits")
    coco_dir = P.artifact("coco")

    classes = pd.read_parquet(t / "classes.parquet")
    ann = pd.read_parquet(t / "annotations.parquet")
    split = pd.read_parquet(sp / "split_images.parquet")

    print("=" * 78)
    print(" S8 步骤一：导出 3000 细类 COCO 标注（供 DINO/MMDetection 用）")
    print("=" * 78)

    print("\n[1/3] 前置校验 ...")
    verify_prerequisites(classes, ann, split)
    print("      通过：class_id 完整覆盖 1..3000、trainval/val 用同一套 mapping、"
          "每个细类唯一映射到一个 superclass")

    src_fp = {
        "annotations.parquet": file_sha256(t / "annotations.parquet")[:16],
        "classes.parquet": file_sha256(t / "classes.parquet")[:16],
        "split_images.parquet": file_sha256(sp / "split_images.parquet")[:16],
    }
    cfg_fp = fingerprint({"dataset": load_yaml("dataset.yaml"), "splits": load_yaml("splits.yaml")})
    print(f"\n源表指纹 {src_fp}")

    # ---- fine -> super 映射 --------------------------------------------------
    print("\n[2/3] 生成 fine_class_id -> superclass_id 映射 ...")
    mapping = build_fine_to_super(classes)
    mapping_path = coco_dir / "fine_to_super_mapping.json"
    mapping_path.write_text(
        json.dumps({**mapping, "source_tables": src_fp}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"      {len(mapping['superclasses'])} 个 superclass，"
          f"{len(mapping['fine_to_super'])} 条 fine->super 映射 → {mapping_path.name}")

    # ---- 逐 split 导出 COCO（split 归属完全来自 split_images.parquet，只读不改）---
    print("\n[3/3] 按已冻结的 split 导出 COCO 标注 ...")
    split_to_file = {"trainval": "instances_train_dino.json", "val": "instances_val_dino.json"}
    manifest_subsets = {}
    cat_signatures = {}

    for split_name, out_name in split_to_file.items():
        ids = sorted(split.loc[split["split"] == split_name, "image_id"])
        coco = build_coco_gt(
            split, ann,
            image_ids=ids,
            class_agnostic=False,
            info={
                "subset": split_name,
                "source_tables": src_fp,
                "config_fingerprint": cfg_fp,
                "coord_convention": "0-based, xywh derived from half-open [x1,x2)",
                "class_agnostic": False,
                "category_granularity": "fine (3000 classes, class_id from classes.parquet)",
                "split_membership_note": (
                    "images/annotations 的 trainval/val 归属完全取自 "
                    "runs/splits/split_images.parquet 里已冻结的 split 列，"
                    "本脚本只做格式转换，未做任何重新划分。"
                ),
            },
        )
        path = write_coco_gt(coco_dir / out_name, coco)
        sz = path.stat().st_size / 1024 / 1024
        n_img, n_ann, n_cat = len(coco["images"]), len(coco["annotations"]), len(coco["categories"])
        cat_signatures[split_name] = tuple(sorted(c["id"] for c in coco["categories"]))
        manifest_subsets[split_name] = {"n_images": n_img, "n_annotations": n_ann, "n_categories": n_cat}
        print(f"      {split_name:<10} {n_img:>7,} 图  {n_ann:>7,} 框  {n_cat:>4} 类  {sz:>7.1f} MB  → {out_name}")

    # ---- 自检：两侧 category 完全一致；split 归属未被扰动 ----------------------
    assert cat_signatures["trainval"] == cat_signatures["val"] == tuple(range(1, 3001)), (
        "trainval / val 两份 COCO 文件的 category_id 集合不一致，或不是完整 1..3000"
    )
    print("\n      自检：trainval / val 的 category_id 集合完全一致，均为完整 1..3000")

    (coco_dir / "manifest_dino.json").write_text(
        json.dumps(
            {
                "source_tables": src_fp,
                "config_fingerprint": cfg_fp,
                "subsets": manifest_subsets,
                "note": "test split 不存在——本项目只有 trainval/val 两个划分（官方 zip 不含 split 清单，无法复现三分）。",
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nTrain images: {manifest_subsets['trainval']['n_images']}")
    print(f"Validation images: {manifest_subsets['val']['n_images']}")
    print("Test images: 0（不存在 test split，本项目只有 trainval/val）")
    print(f"\n产物写入 {coco_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
