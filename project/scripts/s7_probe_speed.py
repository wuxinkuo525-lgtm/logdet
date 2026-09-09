#!/usr/bin/env python
"""S7 前置探针：实测 baseline 的推理速度，把工期估算变成实测数字。

不猜。前几个阶段的经验反复证明：本机的性能特征与常规直觉偏差很大
（磁盘逐文件 20 文件/秒、多 worker 反而慢 41 倍），S7 的工期必须实测。

本脚本只测已缓存的 torchvision 模型（Faster R-CNN v2，167MB 已在
~/.cache/torch/hub/checkpoints/）。OWLv2 需先下载，另行测。

用法：
    python scripts/s7_probe_speed.py [样本数]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from logodet.config import load_yaml  # noqa: E402
from logodet.data.adapters.to_torchvision import to_torchvision  # noqa: E402
from logodet.data.core_dataset import CoreDataset, ZipImageReader  # noqa: E402
from logodet.data.loader import build_loader  # noqa: E402
from logodet.paths import P  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N_INFER = 3079  # 推理全集：val2k ∪ hard_pool


def build_core(n: int) -> CoreDataset:
    sp, t = P.artifact("splits"), P.artifact("tables")
    img = pd.read_parquet(sp / "split_images.parquet")
    ann = pd.read_parquet(t / "annotations.parquet")
    v2k = pd.read_parquet(sp / "val2k_repr.parquet")
    vhp = pd.read_parquet(sp / "val_hard_pool.parquet")
    ids = sorted(set(v2k["image_id"]) | set(vhp["image_id"]))[:n]

    cfg = load_yaml("dataset.yaml")
    zp = P.raw / cfg["source"]["zip_name"]
    reader = None
    if zp.is_file():
        reader = ZipImageReader(zp, cfg["parse"]["archive_top_dir"])
        reader.build_index(set(img[img["image_id"].isin(set(ids))]["rel_path"]))
    return CoreDataset(img, ann, dataset_root=P.dataset, image_ids=ids,
                       load_image=True, zip_reader=reader)


def fmt_eta(rate: float) -> str:
    if rate <= 0:
        return "n/a"
    sec = N_INFER / rate
    return f"{sec / 60:.1f} 分钟" if sec >= 60 else f"{sec:.0f} 秒"


def main() -> int:
    device = torch.device(load_yaml("runtime.yaml").get("device", "cpu"))
    print("=" * 78)
    print(f" S7 推理速度探针（device={device}，样本 {N} 图，推理全集 {N_INFER:,} 图）")
    print("=" * 78)

    core = build_core(N)
    loader = build_loader(core, to_torchvision, batch_size=1,
                          num_workers=0, shuffle=False)

    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    print("\n加载 Faster R-CNN ResNet50-FPN v2（权重已缓存）...")
    t0 = time.time()
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.01)
    model.eval().to(device)
    print(f"  加载 + 上设备耗时 {time.time() - t0:.1f}s")

    # ---- 预热：MPS 首次前向要编译 kernel，不预热会把编译时间算进吞吐 ----
    print("\n预热 3 张 ...")
    warm = 0
    with torch.inference_mode():
        for images, _ in loader:
            model([images[0].to(device)])
            warm += 1
            if warm >= 3:
                break
    if device.type == "mps":
        torch.mps.synchronize()

    # ---- 正式计时 ----------------------------------------------------------
    print(f"\n计时 {N} 张 ...")
    t0 = time.time()
    n = 0
    n_boxes = []
    with torch.inference_mode():
        for images, targets in loader:
            out = model([images[0].to(device)])
            n_boxes.append(int(out[0]["boxes"].shape[0]))
            n += 1
    if device.type == "mps":
        torch.mps.synchronize()
    dt = time.time() - t0
    rate = n / dt

    print("\n" + "=" * 78)
    print(" 实测结果")
    print("=" * 78)
    print(f"\n  Faster R-CNN 全模型（含 RPN + ROI head）")
    print(f"    {n} 图 / {dt:.1f}s  →  {rate:.2f} img/s  ({1000 / rate:.0f} ms/img)")
    print(f"    推理全集 {N_INFER:,} 图预计：{fmt_eta(rate)}")
    print(f"    每图输出框数：中位数 {int(np.median(n_boxes))}，"
          f"最大 {max(n_boxes)}（box_score_thresh=0.01）")

    # ---- 只跑 RPN（proposals baseline）------------------------------------
    print("\n只跑 backbone + RPN（proposals baseline）...")
    from torchvision.models.detection.image_list import ImageList

    t0 = time.time()
    n2 = 0
    n_props = []
    with torch.inference_mode():
        for images, _ in loader:
            x = images[0].to(device)
            transformed, _ = model.transform([x])
            feats = model.backbone(transformed.tensors)
            props, _ = model.rpn(transformed, feats)
            n_props.append(int(props[0].shape[0]))
            n2 += 1
    if device.type == "mps":
        torch.mps.synchronize()
    dt2 = time.time() - t0
    rate2 = n2 / dt2
    print(f"    {n2} 图 / {dt2:.1f}s  →  {rate2:.2f} img/s  ({1000 / rate2:.0f} ms/img)")
    print(f"    推理全集预计：{fmt_eta(rate2)}")
    print(f"    每图 proposals：中位数 {int(np.median(n_props))}，最大 {max(n_props)}")

    print("\n" + "=" * 78)
    print(" 工期结论")
    print("=" * 78)
    print(f"""
  已实测（权重已缓存，零下载）：
    baseline-1 RPN proposals     {fmt_eta(rate2):>12}
    baseline-2 Faster R-CNN COCO {fmt_eta(rate):>12}

  说明：两者可**共用一次前向** —— RPN 的输出本就是 Faster R-CNN 的中间结果。
        若合并实现，总耗时约等于单独跑 Faster R-CNN 的 {fmt_eta(rate)}。

  尚未实测：
    baseline-3 OWLv2 zero-shot   需先下载约 1.6GB 权重（按 S1 实测 8.2 MB/s ≈ 3.3 分钟）
                                 ViT-B/16 + 960px 输入，MPS 速度未知，是最大不确定项
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
