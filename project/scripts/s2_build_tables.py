#!/usr/bin/env python
"""S2 步骤一：XML → 统一中间表（parquet 为唯一真源）。

产出三张表到 runs/tables/：
    images.parquet       每图一行（158,654 行）
    annotations.parquet  每框一行（清洗后约 194,263 行）
    classes.parquet      每类一行（3,000 行）

数据源走 zip 而非磁盘 —— 实测快 82.5 倍，见 ingest/xml_source.py。

坐标处理：侦察已判定数据是 **0-based**（存在 xmin==0，且 xmax 从不超过 W），
所以**不做任何平移**，内部统一 0-based 半开区间 [x1, x2)。

用法：
    python scripts/s2_build_tables.py
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
from logodet.curate.clean_rules import CleanStats, clean_box  # noqa: E402
from logodet.curate.derive_geom import (  # noqa: E402
    add_edge_distance,
    add_geometry,
    add_iof_occlusion,
    add_shape_outlier,
    add_truncation_label,
    overlap_matrix,
    summarize_slices,
)
from logodet.ingest.voc_parse import normalize_class_name, parse_voc_xml  # noqa: E402
from logodet.ingest.xml_source import iter_xml_from_zip  # noqa: E402
from logodet.paths import P  # noqa: E402


def main() -> int:
    cfg = load_yaml("dataset.yaml")
    slices_cfg = load_yaml("slices.yaml") if (PROJECT_DIR / "configs" / "slices.yaml").is_file() else {}
    zip_path = P.raw / cfg["source"]["zip_name"]
    top = cfg["parse"]["archive_top_dir"]
    tables = P.artifact("tables")

    print("=" * 78)
    print(" S2 步骤一：XML → 中间表")
    print("=" * 78)

    inv = pd.read_parquet(tables / "file_inventory.parquet")
    print(f"\n清单表 {len(inv):,} 行")
    meta = inv.set_index("xml_rel")[["supercat", "brand_dir", "stem", "image_rel"]]

    # ---- 解析 --------------------------------------------------------------
    print(f"\n[1/4] 从 zip 解析 XML ...")
    t0 = time.time()
    stats = CleanStats()

    img_records: list[dict] = []
    box_records: list[dict] = []
    n_raw_boxes = 0

    for e in iter_xml_from_zip(zip_path, top_dir=top):
        ann = parse_voc_xml(e.rel_path, e.data)
        if ann.errors:
            stats.hit("R7", f"{e.rel_path}: {ann.errors[0]}")

        try:
            m = meta.loc[e.rel_path]
        except KeyError:
            # zip 里有但清单里没有 —— S1 的四象限门应该已经拦住，这里兜底
            stats.hit("R7", f"{e.rel_path}: 不在 inventory 中")
            continue

        W, H = ann.width, ann.height
        if not W or not H:
            stats.hit("R4", e.rel_path)
            from PIL import Image

            with Image.open(P.dataset / m["image_rel"]) as im:
                W, H = im.size
            size_source = "pil_corrected"
        else:
            size_source = "xml"

        disk_name = f"{m['stem']}{Path(m['image_rel']).suffix}"
        if ann.filename and ann.filename != disk_name:
            stats.hit("R8", f"{e.rel_path}: XML={ann.filename!r} disk={disk_name!r}")

        img_records.append(
            {
                "rel_path": m["image_rel"],
                "xml_rel_path": e.rel_path,
                "supercat": m["supercat"],
                "brand_dir": m["brand_dir"],
                "stem": m["stem"],
                "img_w": int(W),
                "img_h": int(H),
                "size_source": size_source,
                "verified": ann.verified or "",
                "segmented": ann.segmented if ann.segmented is not None else -1,
                "n_boxes_raw": len(ann.boxes),
            }
        )

        seen: set[tuple] = set()
        for b in ann.boxes:
            n_raw_boxes += 1
            name_xml = normalize_class_name(b.name_raw)
            if name_xml != b.name_raw:
                stats.hit("R6", f"{e.rel_path}: {b.name_raw!r} → {name_xml!r}")

            # 类标签以**目录名**为准，不用 XML 里的 <name>。
            #
            # 依据（scripts/s2_diagnose_adjacency.py 实测）：XML 类名与目录名
            # 不一致的 340 个组合中，两者在字典序列表里的**排名距离中位数为 1**，
            # 距离<=3 覆盖 99.6% 的框，而随机误选的期望距离是 1000。
            # 即 XML 名带有系统性的「下拉列表选中相邻项」人为错误。
            # 目录名则来自爬取时使用的品牌查询词，是系统性产物。
            #
            # 另一个硬理由：用目录名恰好重建 3,000 类 = 论文的基准类空间；
            # 用 XML 名只有 2,993 类，直接失去与论文对账的能力。
            name_dir = m["brand_dir"]
            if name_xml != name_dir:
                stats.hit("R9", f"{e.rel_path}: dir={name_dir!r} xml={name_xml!r}")

            cleaned = clean_box(
                b.xmin, b.ymin, b.xmax, b.ymax, int(W), int(H), stats, where=e.rel_path
            )
            if cleaned is None:
                continue
            x1, y1, x2, y2 = cleaned

            key = (name_dir, round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3))
            if key in seen:
                stats.hit("R5", f"{e.rel_path}: {key}")
                continue
            seen.add(key)

            box_records.append(
                {
                    "xml_rel_path": e.rel_path,
                    "class_name": name_dir,          # 权威标签
                    "class_name_xml": name_xml,      # XML 原标注，留作审计
                    "class_name_raw": b.name_raw,
                    "label_disagree": name_xml != name_dir,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "truncated": b.truncated,
                    "difficult": b.difficult,
                    "order_in_file": b.order_in_file,
                }
            )

    print(f"      {len(img_records):,} 图 / 原始 {n_raw_boxes:,} 框 → "
          f"清洗后 {len(box_records):,} 框，{time.time() - t0:.1f}s")

    # ---- 建 images 表（稳定 id）--------------------------------------------
    print("\n[2/4] 组装 images / annotations / classes 表 ...")
    images = pd.DataFrame(img_records)
    # image_id 按 rel_path 字典序递增：稳定、可复现、连续（pycocotools 对连续 id 更友好）
    images = images.sort_values("rel_path", kind="mergesort").reset_index(drop=True)
    images.insert(0, "image_id", np.arange(1, len(images) + 1, dtype="int64"))

    ann = pd.DataFrame(box_records)
    xml2id = dict(zip(images["xml_rel_path"], images["image_id"]))
    ann["image_id"] = ann["xml_rel_path"].map(xml2id).astype("int64")

    # ann_id 按 (image_id, XML 内原始顺序) 递增
    ann = ann.sort_values(["image_id", "order_in_file"], kind="mergesort").reset_index(drop=True)
    ann.insert(0, "ann_id", np.arange(1, len(ann) + 1, dtype="int64"))

    # 类别表 + class_id（按权威类名字典序，稳定映射）
    #
    # 类空间从**全部品牌目录**构造而非从 annotations 里出现过的类名构造：
    # 否则若某个目录的框全被清洗丢弃，该类就会凭空消失，类数不再等于 3,000。
    names = sorted(images["brand_dir"].unique())
    class_id = {n: i + 1 for i, n in enumerate(names)}
    ann["class_id"] = ann["class_name"].map(class_id).astype("int32")
    ann["brand_key"] = ann["class_name"].str.replace(r"-\d+$", "", regex=True)

    # 把图像尺寸与超类带到框级，后续几何计算与切片都要用
    ann = ann.merge(
        images[["image_id", "img_w", "img_h", "supercat"]], on="image_id", how="left"
    )

    # ---- 几何 + 三个难例轴 --------------------------------------------------
    print("\n[3/4] 派生几何与难例切片 ...")
    t1 = time.time()
    sl = slices_cfg or {}
    ann = add_geometry(ann)
    ann = add_edge_distance(
        ann,
        min_px=float(sl.get("edge", {}).get("min_px", 2.0)),
        frac=float(sl.get("edge", {}).get("frac", 0.005)),
    )
    ann = add_truncation_label(ann)
    ann = add_iof_occlusion(ann, threshold=float(sl.get("p2", {}).get("iof_threshold", 0.3)))
    ann = add_shape_outlier(
        ann,
        z_threshold=float(sl.get("p3", {}).get("z_threshold", 3.0)),
        min_class_size=int(sl.get("p3", {}).get("min_class_size", 20)),
    )
    # hard_any 只由 T1 定义。
    #
    # 原定义是 T1|P2|P3 的并集（28,359 框）。但 60 张 VLM 预标核验显示
    # P2 precision=0.143、P3 precision=0.176，双双远低于 0.60 门槛：
    #   P2 的命中 17/25 是同品牌标识的嵌套标注框，不是遮挡
    #   P3 的命中 27/35 是类内徽标/字标双模态或自然形状差异
    # 两者已降级为探索性列（详见 configs/slices.yaml 的 proxy_verdict）。
    # 列仍然保留 —— 计算已经做完，且是错误分析的有用线索。
    hard_any_cols = [sl.get("proxy_verdict", {}).get("hard_any_definition", "is_hard_t1")]
    ann["hard_any"] = ann[hard_any_cols].any(axis=1)
    ann["is_clean"] = ~ann["hard_any"]
    ann["iscrowd"] = np.int8(0)
    print(f"      完成，{time.time() - t1:.1f}s（hard_any 定义 = {' | '.join(hard_any_cols)}）")

    # 图级派生
    nb = ann.groupby("image_id").size().rename("n_boxes")
    nc = ann.groupby("image_id")["class_id"].nunique().rename("n_classes_in_img")
    images = images.merge(nb, on="image_id", how="left").merge(nc, on="image_id", how="left")
    images["n_boxes"] = images["n_boxes"].fillna(0).astype("int32")
    images["n_classes_in_img"] = images["n_classes_in_img"].fillna(0).astype("int32")
    images["n_boxes_bin"] = pd.cut(
        images["n_boxes"], bins=[-1, 1, 2, 4, 10**9], labels=["1", "2", "3-4", "5+"]
    ).astype("string")

    # classes 表以完整类空间（3,000 个品牌目录）为骨架左连接统计值，
    # 保证「某类的框全被清洗丢弃」时它依然在表里（统计值为 0），类数恒为 3,000
    skeleton = pd.DataFrame(
        {"class_id": [class_id[n] for n in names], "class_name": names}
    )
    skeleton["brand_key"] = skeleton["class_name"].str.replace(r"-\d+$", "", regex=True)
    sc_of_dir = images.drop_duplicates("brand_dir").set_index("brand_dir")["supercat"]
    skeleton["supercat"] = skeleton["class_name"].map(sc_of_dir)

    agg = (
        ann.groupby("class_id")
        .agg(
            n_boxes=("ann_id", "size"),
            n_images=("image_id", "nunique"),
            ar_median=("p3_ar_median", "first"),
            ar_mad=("p3_ar_mad", "first"),
            p3_valid=("p3_valid", "first"),
            n_truncated=("is_hard_t1", "sum"),
            n_label_disagree=("label_disagree", "sum"),
        )
        .reset_index()
    )
    classes = skeleton.merge(agg, on="class_id", how="left")
    for c in ("n_boxes", "n_images", "n_truncated", "n_label_disagree"):
        classes[c] = classes[c].fillna(0).astype("int32")
    classes["p3_valid"] = classes["p3_valid"].fillna(False).astype(bool)
    classes["n_subclass_in_brand"] = classes.groupby("brand_key")["class_id"].transform("size")

    # ---- 落盘 --------------------------------------------------------------
    print("\n[4/4] 写 parquet ...")
    images.to_parquet(tables / "images.parquet", index=False)
    ann.to_parquet(tables / "annotations.parquet", index=False)
    classes.to_parquet(tables / "classes.parquet", index=False)

    sl_summary = summarize_slices(ann, hard_any_cols=hard_any_cols)
    ov = overlap_matrix(ann)

    report = {
        "config_fingerprint": fingerprint({"dataset": cfg, "slices": slices_cfg}),
        "n_images": int(len(images)),
        "n_boxes_raw": int(n_raw_boxes),
        "n_boxes_clean": int(len(ann)),
        "n_classes": int(len(classes)),
        "coord_basis": "0-based (侦察判定，无平移)",
        "clean_rule_counts": stats.counts,
        "clean_rule_samples": {k: v[:5] for k, v in stats.samples.items() if v},
        "slice_summary": sl_summary.to_dict(orient="records"),
        "overlap_matrix": ov.to_dict(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (tables / "parse_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n      images.parquet      {len(images):,} 行")
    print(f"      annotations.parquet {len(ann):,} 行")
    print(f"      classes.parquet     {len(classes):,} 行")
    print("\n清洗规则命中：")
    for rule, n in stats.counts.items():
        if n:
            print(f"      {rule}  {n:,}")
    print("\n难例切片体量：")
    for r in sl_summary.to_dict(orient="records"):
        print(f"      {r['slice']:<14} {r['role']:<5} {r['n_ann']:>8,} 框 / "
              f"{r['n_img']:>8,} 图  ({r['rate']:.2%})")

    print(f"\n总耗时 {time.time() - t0:.1f}s。下一步：python scripts/s2_verify_tables.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
