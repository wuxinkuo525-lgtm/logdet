#!/usr/bin/env python
"""诊断：Faster R-CNN 在 MPS 上为何慢到不可用。

首轮探针 40 张图跑了 21 分钟仍未完成（<0.03 img/s）。若属实，
推理全集 3,079 图要 27 小时 —— S7 不可能这样做。

**主假设：MPS 算子回退**。torchvision 的检测模型里有若干算子
（`nms`、`roi_align`、部分索引操作）在 MPS 后端缺失，
`PYTORCH_ENABLE_MPS_FALLBACK=1` 会让它们悄悄回退到 CPU。
每次回退都要 GPU→CPU→GPU 同步一次，而 RPN 每张图要对上千个 proposal
做 NMS —— 同步开销会彻底吃掉 GPU 的收益，甚至比纯 CPU 还慢。

这个假设是可测的：**逐段计时 + 对比纯 CPU**。
若纯 CPU 明显更快，就说明瓶颈是回退同步而非算力。

本脚本刻意只用 3-5 张图，避免又跑成 20 分钟。

用法：
    python scripts/s7_diagnose_device.py [每档图数]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from logodet.config import load_yaml  # noqa: E402
from logodet.data.adapters.to_torchvision import to_torchvision  # noqa: E402
from logodet.data.core_dataset import CoreDataset, ZipImageReader  # noqa: E402
from logodet.paths import P  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N_INFER = 3079


def build_core(n: int) -> CoreDataset:
    sp, t = P.artifact("splits"), P.artifact("tables")
    img = pd.read_parquet(sp / "split_images.parquet")
    ann = pd.read_parquet(t / "annotations.parquet")
    v2k = pd.read_parquet(sp / "val2k_repr.parquet")
    ids = sorted(v2k["image_id"])[:n]
    cfg = load_yaml("dataset.yaml")
    zp = P.raw / cfg["source"]["zip_name"]
    reader = None
    if zp.is_file():
        reader = ZipImageReader(zp, cfg["parse"]["archive_top_dir"])
        reader.build_index(set(img[img["image_id"].isin(set(ids))]["rel_path"]))
    return CoreDataset(img, ann, dataset_root=P.dataset, image_ids=ids,
                       load_image=True, zip_reader=reader)


def sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()


def timed_stage(name: str, fn, dev: torch.device, reps: int = 1) -> float:
    sync(dev)
    t0 = time.time()
    for _ in range(reps):
        out = fn()
    sync(dev)
    dt = (time.time() - t0) / reps
    print(f"      {name:<28} {dt * 1000:>9.1f} ms")
    return dt


def probe(device_str: str, tensors: list[torch.Tensor]) -> dict[str, float]:
    dev = torch.device(device_str)
    print(f"\n{'=' * 74}")
    print(f" device = {device_str}")
    print("=" * 74)

    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    t0 = time.time()
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, box_score_thresh=0.05
    )
    model.eval().to(dev)
    print(f"  模型加载 + 上设备 {time.time() - t0:.1f}s")

    x = tensors[0].to(dev)
    out: dict[str, float] = {}

    print("\n  分段计时（单张，已预热）：")
    with torch.inference_mode():
        # 预热：MPS 首次前向要编译 kernel
        model([x])
        sync(dev)

        tr, _ = model.transform([x])
        out["transform"] = timed_stage("transform（resize+norm）",
                                       lambda: model.transform([x]), dev)
        feats = model.backbone(tr.tensors)
        out["backbone"] = timed_stage("backbone（ResNet50-FPN）",
                                      lambda: model.backbone(tr.tensors), dev)
        out["rpn"] = timed_stage("rpn（含 NMS）",
                                 lambda: model.rpn(tr, feats), dev)
        props, _ = model.rpn(tr, feats)
        out["roi"] = timed_stage("roi_heads（含 roi_align+NMS）",
                                 lambda: model.roi_heads(feats, props,
                                                         tr.image_sizes), dev)
        out["full"] = timed_stage("整模型 forward", lambda: model([x]), dev)

    # 多张连跑，测稳定吞吐
    print(f"\n  连跑 {len(tensors)} 张：")
    with torch.inference_mode():
        sync(dev)
        t0 = time.time()
        for t in tensors:
            model([t.to(dev)])
        sync(dev)
        dt = time.time() - t0
    rate = len(tensors) / dt
    out["rate"] = rate
    eta = N_INFER / rate
    print(f"      {len(tensors)} 图 / {dt:.1f}s  →  {rate:.3f} img/s")
    print(f"      推理全集 {N_INFER:,} 图预计：{eta / 60:.1f} 分钟"
          f"（{eta / 3600:.1f} 小时）")
    return out


def main() -> int:
    print("=" * 74)
    print(f" Faster R-CNN 设备诊断（每档 {N} 张）")
    print("=" * 74)

    core = build_core(N)
    tensors = [to_torchvision(core[i])[0] for i in range(len(core))]
    print(f"\n已解码 {len(tensors)} 张，尺寸样例 {tuple(tensors[0].shape)}")

    results: dict[str, dict[str, float]] = {}
    results["cpu"] = probe("cpu", tensors)
    if torch.backends.mps.is_available():
        results["mps"] = probe("mps", tensors)

    print("\n" + "=" * 74)
    print(" 对比结论")
    print("=" * 74)
    if "mps" in results:
        c, m = results["cpu"], results["mps"]
        print(f"\n  {'阶段':<28}{'CPU (ms)':>12}{'MPS (ms)':>12}{'MPS/CPU':>10}")
        for k in ("transform", "backbone", "rpn", "roi", "full"):
            ratio = m[k] / c[k] if c[k] > 0 else float("nan")
            flag = "  ← MPS 更慢" if ratio > 1.2 else ""
            print(f"  {k:<28}{c[k] * 1000:>12.1f}{m[k] * 1000:>12.1f}"
                  f"{ratio:>9.2f}x{flag}")
        print(f"\n  吞吐  CPU {c['rate']:.3f} img/s   MPS {m['rate']:.3f} img/s")
        best = "cpu" if c["rate"] >= m["rate"] else "mps"
        print(f"  → 应采用 device={best}")
        print(f"\n  推理全集 {N_INFER:,} 图：")
        for name, r in (("cpu", c["rate"]), ("mps", m["rate"])):
            print(f"    {name:<5} {N_INFER / r / 60:>8.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
