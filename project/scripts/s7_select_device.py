#!/usr/bin/env python
"""S7 步骤零：选定推理设备模式。

hybrid（backbone+rpn 走 MPS，roi_heads 走 CPU）理论上比全 CPU 快 1.6 倍，
但跨设备可能引入数值差异。照搬 S0-G4 的原则：**先验证一致性再采用**，
不能因为"看起来能跑"就直接上。

判据：
  1. hybrid 与纯 CPU 的检测框数量、坐标、分数必须一致
  2. hybrid 确实更快，否则没有引入复杂度的理由

用法：
    python scripts/s7_select_device.py [验证图数]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from logodet.baselines.torchvision_detector import (  # noqa: E402
    forward_one,
    load_model,
    verify_device_consistency,
)
from logodet.config import load_yaml  # noqa: E402
from logodet.data.adapters.to_torchvision import to_torchvision  # noqa: E402
from logodet.data.core_dataset import CoreDataset, ZipImageReader  # noqa: E402
from logodet.paths import P  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
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


def main() -> int:
    print("=" * 76)
    print(f" S7 设备模式选型（验证 {N} 张）")
    print("=" * 76)

    core = build_core(N)
    tensors = [to_torchvision(core[i])[0] for i in range(len(core))]
    model, weights = load_model()
    print(f"\n模型：Faster R-CNN ResNet50-FPN v2（{weights}）")

    modes = ["cpu"]
    if torch.cuda.is_available():
        modes.append("cuda")
    if torch.backends.mps.is_available():
        modes.append("hybrid")

    # ---- 一致性 ------------------------------------------------------------
    # CUDA 原生支持 roi_align/NMS（不像 MPS 缺算子），不需要 hybrid 式的
    # 跨设备切分，但"不需要切分"不等于"不需要验证"——照样跑一遍 cpu 对照。
    consistency: dict[str, tuple[bool, dict[str, float]]] = {}
    for other in ("cuda", "hybrid"):
        if other not in modes:
            continue
        print(f"\n[1/2] {other} vs cpu 数值一致性 ...")
        ok, stats = verify_device_consistency(model, tensors[:4], other_mode=other)
        consistency[other] = (ok, stats)
        for k, v in stats.items():
            print(f"      {k:<22} {v:.6g}")
        print(f"      判定：{'一致' if ok else f'不一致 → 禁用 {other}'}")

    # ---- 速度 --------------------------------------------------------------
    print("\n[2/2] 速度实测（已预热）...")
    rates: dict[str, float] = {}
    for mode in modes:
        timers: dict[str, float] = {}
        forward_one(model, tensors[0], 0, mode=mode)  # 预热
        t0 = time.time()
        for i, t in enumerate(tensors):
            forward_one(model, t, i, mode=mode, timers=timers)
        dt = time.time() - t0
        rates[mode] = len(tensors) / dt
        seg = "  ".join(f"{k}={v / len(tensors) * 1000:.0f}ms"
                        for k, v in timers.items())
        print(f"      {mode:<8} {rates[mode]:>6.3f} img/s   "
              f"全集 {N_INFER / rates[mode] / 60:>5.1f} 分钟   [{seg}]")

    # ---- 决策 --------------------------------------------------------------
    # 候选：cpu 永远可信任（无跨设备问题）；cuda/hybrid 只有一致性验证通过、
    # 且比 cpu 快出有意义的margin（>15%，覆盖跨设备搬运的固定开销）才采用。
    # 多个候选都合格时选最快的那个。
    chosen = "cpu"
    reason = "只有 CPU 可用" if len(modes) == 1 else "其余候选未通过一致性或不够快"
    best_rate = rates["cpu"]
    for other in ("cuda", "hybrid"):
        if other not in modes:
            continue
        ok, _ = consistency[other]
        if not ok:
            print(f"\n  [跳过] {other}：与 CPU 数值不一致，按 S0-G4 的原则拒绝采用")
            continue
        if rates[other] <= rates["cpu"] * 1.15:
            print(f"\n  [跳过] {other}：仅快 {rates[other] / rates['cpu']:.2f}x，"
                  f"不足以抵消跨设备复杂度")
            continue
        if rates[other] > best_rate:
            chosen = other
            best_rate = rates[other]
            reason = (f"一致性通过且快 {rates[other] / rates['cpu']:.2f}x "
                      f"（{N_INFER / rates['cpu'] / 60:.0f} → "
                      f"{N_INFER / rates[other] / 60:.0f} 分钟）")

    print("\n" + "=" * 76)
    print(f" 选定：device_mode = {chosen}")
    print(f" 理由：{reason}")
    print("=" * 76)

    out = P.artifact("metrics") / "s7_device_choice.json"
    out.write_text(json.dumps({
        "chosen": chosen,
        "reason": reason,
        "consistency": {k: v[1] for k, v in consistency.items()},
        "rates_img_per_sec": rates,
        "eta_minutes": {k: N_INFER / v / 60 for k, v in rates.items()},
        "n_verify_images": N,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {out.relative_to(P.artifacts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
