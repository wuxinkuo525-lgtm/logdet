#!/usr/bin/env python
"""S7 步骤一：跑两个 baseline 的推理。

一次前向同时产出：

    L1 rpn_proposals   backbone→rpn 的 class-agnostic proposals
    L3 coco_collapsed  再过 roi_heads、把 91 个 COCO 类塌缩成单类 "logo"

外加一个**负控制**（不需要额外推理，只改 category_id）：

    L3-strict          同样的框，但 category_id=2（GT 里只有 1）
                       → AP 必须精确为 0。若不为 0，说明评测器在跨类别匹配，
                         那么类塌缩的结果就是假的。

设备模式由 scripts/s7_select_device.py 选定并验证（hybrid 已验一致性
max_box_diff=0.004px，快 1.81x）。

用法：
    python scripts/s7_run_baselines.py
    python scripts/s7_run_baselines.py --limit 100   # 冒烟
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from logodet.baselines.torchvision_detector import (  # noqa: E402
    forward_one,
    load_model,
)
from logodet.config import file_sha256, fingerprint, load_yaml  # noqa: E402
from logodet.data.adapters.to_torchvision import to_torchvision  # noqa: E402
from logodet.data.core_dataset import CoreDataset, ZipImageReader  # noqa: E402
from logodet.paths import P  # noqa: E402


def build_core(limit: int | None) -> tuple[CoreDataset, list[int]]:
    sp, t = P.artifact("splits"), P.artifact("tables")
    img = pd.read_parquet(sp / "split_images.parquet")
    ann = pd.read_parquet(t / "annotations.parquet")
    v2k = pd.read_parquet(sp / "val2k_repr.parquet")
    vhp = pd.read_parquet(sp / "val_hard_pool.parquet")
    ids = sorted(set(v2k["image_id"]) | set(vhp["image_id"]))
    if limit:
        ids = ids[:limit]

    cfg = load_yaml("dataset.yaml")
    zp = P.raw / cfg["source"]["zip_name"]
    reader = None
    if zp.is_file():
        reader = ZipImageReader(zp, cfg["parse"]["archive_top_dir"])
        reader.build_index(set(img[img["image_id"].isin(set(ids))]["rel_path"]))
    core = CoreDataset(img, ann, dataset_root=P.dataset, image_ids=ids,
                       load_image=True, zip_reader=reader)
    return core, ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（冒烟用）")
    ap.add_argument("--device-mode", default="", help="覆盖自动选定的模式")
    args = ap.parse_args()

    choice_path = P.artifact("metrics") / "s7_device_choice.json"
    mode = args.device_mode
    if not mode:
        if not choice_path.is_file():
            print("FAIL: 缺设备选型结果，先跑 python scripts/s7_select_device.py")
            return 1
        mode = json.loads(choice_path.read_text(encoding="utf-8"))["chosen"]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("=" * 78)
    print(f" S7 baseline 推理（device_mode={mode}, run_id={run_id}）")
    print("=" * 78)

    core, ids = build_core(args.limit or None)
    print(f"\n推理集合 {len(core):,} 图")

    model, weights = load_model()
    print(f"模型 Faster R-CNN ResNet50-FPN v2 / {weights}")

    prop_rows: list[dict] = []
    det_rows: list[dict] = []
    timers: dict[str, float] = {}
    n_prop, n_det = [], []

    t0 = time.time()
    for i in range(len(core)):
        s = core[i]
        img_t, _ = to_torchvision(s)
        out = forward_one(model, img_t, s.image_id, mode=mode, timers=timers)

        n_prop.append(len(out.prop_boxes))
        n_det.append(len(out.det_boxes))

        for (x1, y1, x2, y2), sc in zip(out.prop_boxes, out.prop_scores):
            prop_rows.append({
                "image_id": s.image_id, "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2), "score": float(sc),
                "category_id": 1,
            })
        for (x1, y1, x2, y2), sc, lb in zip(
            out.det_boxes, out.det_scores, out.det_labels
        ):
            det_rows.append({
                "image_id": s.image_id, "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2), "score": float(sc),
                "category_id": 1,            # 类塌缩：91 类 → 单类 logo
                "coco_category_id": int(lb),  # 保留原类别，便于错误分析
            })

        if (i + 1) % 200 == 0:
            el = time.time() - t0
            rate = (i + 1) / el
            eta = (len(core) - i - 1) / rate / 60
            print(f"  {i + 1:>5,}/{len(core):,}  {rate:.2f} img/s  "
                  f"剩余约 {eta:.1f} 分钟", flush=True)

    dt = time.time() - t0
    rate = len(core) / dt

    pred_dir = P.artifact("predictions")
    df_prop = pd.DataFrame(prop_rows)
    df_det = pd.DataFrame(det_rows)
    df_prop["baseline"] = "L1_rpn_proposals"
    df_prop["run_id"] = run_id
    df_det["baseline"] = "L3_coco_collapsed"
    df_det["run_id"] = run_id
    df_prop.to_parquet(pred_dir / "L1_rpn_proposals.parquet", index=False)
    df_det.to_parquet(pred_dir / "L3_coco_collapsed.parquet", index=False)

    manifest = {
        "run_id": run_id,
        "device_mode": mode,
        "model": "fasterrcnn_resnet50_fpn_v2",
        "weights": str(weights),
        "weights_sha256_16": file_sha256(
            Path.home() / ".cache/torch/hub/checkpoints"
            / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
        )[:16],
        "config_fingerprint": fingerprint({
            "dataset": load_yaml("dataset.yaml"),
            "splits": load_yaml("splits.yaml"),
        }),
        "n_images": len(core),
        "seconds": round(dt, 1),
        "img_per_sec": round(rate, 3),
        "stage_seconds_per_img_ms": {
            k: round(v / len(core) * 1000, 1) for k, v in timers.items()
        },
        "proposals": {
            "rows": len(df_prop),
            "per_image_median": int(np.median(n_prop)),
            "per_image_max": int(max(n_prop)),
            "score_note": "objectness 为降序合成值，不可跨图比较 → 只报 AR，禁止报 AP",
        },
        "detections": {
            "rows": len(df_det),
            "per_image_median": int(np.median(n_det)),
            "per_image_max": int(max(n_det)),
            "class_collapse": "91 个 COCO 类 → category_id=1",
        },
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (pred_dir / f"manifest_{run_id}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n完成：{len(core):,} 图 / {dt / 60:.1f} 分钟 / {rate:.2f} img/s")
    print(f"  L1 proposals  {len(df_prop):>9,} 行  "
          f"每图中位数 {int(np.median(n_prop))}，最大 {max(n_prop)}")
    print(f"  L3 detections {len(df_det):>9,} 行  "
          f"每图中位数 {int(np.median(n_det))}，最大 {max(n_det)}")
    print(f"\n产物写入 {pred_dir}")
    print("下一步：python scripts/s7_evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
