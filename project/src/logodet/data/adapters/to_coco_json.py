"""parquet → COCO GT JSON（按需生成的物化视图）。

parquet 是唯一真源，COCO JSON 是**派生视图**：可随时删除重建，
所以永远不会出现"手改了 JSON 导致真源漂移"的情况。

文件头必须写入 info 段（源表 sha256 + 配置指纹 + 切片定义），
这样任何一个指标都能反查到它是在哪个确切的 GT 版本上算出来的。

坐标转换：内部是 0-based 半开区间 [x1, x2)，COCO 的 bbox 是
[x, y, width, height] —— 直接取 x1, y1, x2-x1, y2-y1，不做任何平移。
S2 的 G4b 门已确认数据本身是 0-based。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def build_coco_gt(
    images: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    image_ids: Sequence[int] | None = None,
    class_agnostic: bool = True,
    ignore_ann_ids: set[int] | None = None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 COCO 格式的 GT 字典。

    Args:
        ignore_ann_ids: 这些 ann 的 iscrowd 置 1。切片评测靠它实现
            「保留全图 GT，但只让切片成员参与 recall 分母」——
            pycocotools 对 iscrowd=1 的 GT 用 IoF 匹配，且匹配上的 det
            既不计 TP 也不计 FP（被吸收）。
    """
    img = images
    if image_ids is not None:
        keep = set(int(i) for i in image_ids)
        img = img[img["image_id"].isin(keep)]
    img = img.sort_values("image_id", kind="mergesort")

    ids = set(img["image_id"])
    ann = annotations[annotations["image_id"].isin(ids)].sort_values(
        ["image_id", "ann_id"], kind="mergesort"
    )

    if class_agnostic:
        categories = [{"id": 1, "supercategory": "logo", "name": "logo"}]
        cat_of = np.ones(len(ann), dtype="int64")
    else:
        cats = sorted(ann["class_id"].unique())
        name_of = (
            ann.drop_duplicates("class_id").set_index("class_id")["class_name"].to_dict()
            if "class_name" in ann.columns
            else {}
        )
        categories = [
            {"id": int(c), "supercategory": "logo", "name": str(name_of.get(c, c))}
            for c in cats
        ]
        cat_of = ann["class_id"].to_numpy(dtype="int64")

    ignore = ignore_ann_ids or set()
    iscrowd = np.where(ann["ann_id"].isin(ignore).to_numpy(), 1, 0).astype("int64")

    x1 = ann["x1"].to_numpy(dtype="float64")
    y1 = ann["y1"].to_numpy(dtype="float64")
    x2 = ann["x2"].to_numpy(dtype="float64")
    y2 = ann["y2"].to_numpy(dtype="float64")
    w = x2 - x1
    h = y2 - y1

    coco = {
        "info": {
            "description": "LogoDet-3K class-agnostic logo detection GT",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(info or {}),
        },
        "licenses": [],
        "images": [
            {
                "id": int(r.image_id),
                "file_name": str(r.rel_path),
                "width": int(r.img_w),
                "height": int(r.img_h),
            }
            for r in img.itertuples()
        ],
        "categories": categories,
        "annotations": [
            {
                "id": int(aid),
                "image_id": int(iid),
                "category_id": int(cid),
                "bbox": [float(a), float(b), float(c), float(d)],
                "area": float(c * d),
                "iscrowd": int(ic),
            }
            for aid, iid, cid, a, b, c, d, ic in zip(
                ann["ann_id"].to_numpy(),
                ann["image_id"].to_numpy(),
                cat_of,
                x1, y1, w, h,
                iscrowd,
            )
        ],
    }
    return coco


def write_coco_gt(path: Path, coco: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    return path


def build_coco_dt(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """predictions.parquet → pycocotools 的 detection 列表。

    用 list-of-dict 而不是落临时文件：cocoGt.loadRes() 直接吃 list，
    省掉一次序列化往返。
    """
    x1 = predictions["x1"].to_numpy(dtype="float64")
    y1 = predictions["y1"].to_numpy(dtype="float64")
    w = predictions["x2"].to_numpy(dtype="float64") - x1
    h = predictions["y2"].to_numpy(dtype="float64") - y1
    return [
        {
            "image_id": int(i),
            "category_id": int(c),
            "bbox": [float(a), float(b), float(ww), float(hh)],
            "score": float(s),
        }
        for i, c, a, b, ww, hh, s in zip(
            predictions["image_id"].to_numpy(),
            predictions["category_id"].to_numpy(),
            x1, y1, w, h,
            predictions["score"].to_numpy(),
        )
    ]
