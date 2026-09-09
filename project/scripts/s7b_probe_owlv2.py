#!/usr/bin/env python
"""S7b 步骤一：OWLv2 下载 + 几何对齐验证 + 速度探针。

三件事按重要性排序：

  1. **几何对齐**（决定性）—— OWLv2 先 pad 成正方形再 resize 960，
     post_process 的 target_sizes 传错会让框系统性错位，
     而错位的表现（AP≈0）与"模型确实找不到"**无法区分**。
     所以用合成图做独立验证：已知位置的色块，看框能不能落上去。

  2. **速度**（决定工期）—— OWLv2 是纯 ViT，没有 roi_align 与逐类 NMS，
     预期能吃到 MPS 的加速（对比 S7 里 Faster R-CNN 的 roi_heads 慢 85 倍）。
     但按 S7 的教训：不实测不算数。

  3. **目视抽检** —— 真实图上画框落盘，人眼确认。

用法：
    python scripts/s7b_probe_owlv2.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from logodet.baselines.owlv2_detector import (  # noqa: E402
    MODEL_ID,
    detect,
    load_owlv2,
    synthetic_alignment_check,
)
from logodet.config import load_yaml  # noqa: E402
from logodet.data.core_dataset import CoreDataset, ZipImageReader  # noqa: E402
from logodet.paths import P  # noqa: E402

N_INFER = 3079
N_PROBE = 12


def _json_safe(o):
    """numpy 标量转 Python 原生类型。json 不认 np.bool_ / np.float32。"""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"无法序列化 {type(o).__name__}")


def build_core(n: int, split: str = "val") -> CoreDataset:
    sp, t = P.artifact("splits"), P.artifact("tables")
    img = pd.read_parquet(sp / "split_images.parquet")
    ann = pd.read_parquet(t / "annotations.parquet")
    pool = img[img["split"] == split]
    ids = sorted(pool["image_id"])[:n]
    cfg = load_yaml("dataset.yaml")
    zp = P.raw / cfg["source"]["zip_name"]
    reader = None
    if zp.is_file():
        reader = ZipImageReader(zp, cfg["parse"]["archive_top_dir"])
        reader.build_index(set(img[img["image_id"].isin(set(ids))]["rel_path"]))
    return CoreDataset(img, ann, dataset_root=P.dataset, image_ids=ids,
                       load_image=True, zip_reader=reader)


def main() -> int:
    print("=" * 78)
    print(f" S7b OWLv2 探针（{MODEL_ID}）")
    print("=" * 78)

    print("\n[1/4] 加载模型（首次会下载约 1.6GB）...")
    t0 = time.time()
    processor, model = load_owlv2("cpu")
    n_param = sum(p.numel() for p in model.parameters())
    print(f"      完成 {time.time() - t0:.1f}s，参数量 {n_param / 1e6:.0f}M")

    # ---- 2. 几何对齐（决定性验证）------------------------------------------
    print("\n[2/4] 几何对齐验证（合成图，已知色块位置）...")
    ok_align, cases = synthetic_alignment_check(processor, model, device="cpu")
    for c in cases:
        if not c["detected"]:
            print(f"      {c['prompt']:<16} {c['wh']}  未检出 → 无法判定对齐")
            continue
        print(f"      {c['prompt']:<16} {c['wh']}  "
              f"中心偏移 {c['center_offset_px']}  "
              f"落在色块内={c['center_inside']}  score={c['score']}")
    print(f"      判定：{'对齐正确' if ok_align else '对齐错误 → 禁止继续'}")
    if not ok_align:
        print("\n      target_sizes 的传法有问题。若框沿长边被拉伸，")
        print("      说明传了原图 (H, W) 而不是 padded 方形边长。")
        return 1

    # ---- 3. 速度 -----------------------------------------------------------
    print(f"\n[3/4] 速度实测（{N_PROBE} 张，已预热）...")
    core = build_core(N_PROBE)
    imgs = [core[i].image for i in range(len(core))]

    rates: dict[str, float] = {}
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devices:
        model.to(dev)
        detect(processor, model, imgs[0], ["a logo"], device=dev)  # 预热
        if dev == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        for k, im in enumerate(imgs):
            detect(processor, model, im, ["a logo"], device=dev, image_id=k)
        if dev == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        rates[dev] = len(imgs) / dt
        print(f"      {dev:<5} {rates[dev]:>6.3f} img/s   "
              f"全集 {N_INFER / rates[dev] / 60:>6.1f} 分钟")

    best_dev = max(rates, key=rates.get)
    print(f"      → 选定 device={best_dev}"
          f"（{N_INFER / rates[best_dev] / 60:.1f} 分钟）")

    # ---- 4. 真实图目视抽检 -------------------------------------------------
    print("\n[4/4] 真实图画框落盘（目视抽检）...")
    from PIL import Image, ImageDraw

    model.to(best_dev)
    out_dir = P.artifact("cache") / "s7b_owlv2_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_box = []
    for i in range(min(8, len(core))):
        s = core[i]
        o = detect(processor, model, s.image, ["a logo"], device=best_dev,
                   threshold=0.05, image_id=s.image_id)
        n_box.append(len(o.boxes_xyxy))
        im = Image.fromarray(s.image).convert("RGB")
        d = ImageDraw.Draw(im)
        # GT 画绿框，预测画红框 —— 一眼能看出是否对齐
        for (x1, y1, x2, y2) in s.boxes_xyxy:
            d.rectangle([x1, y1, x2, y2], outline=(40, 200, 60), width=3)
        for (x1, y1, x2, y2), sc in zip(o.boxes_xyxy, o.scores):
            d.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=2)
            d.text((x1 + 2, max(0, y1 - 12)), f"{sc:.2f}", fill=(255, 40, 40))
        im.save(out_dir / f"{s.image_id}_gt{s.n_boxes}_pred{len(o.boxes_xyxy)}.jpg",
                quality=88)
    print(f"      8 张已落盘到 {out_dir.relative_to(P.artifacts)}")
    print(f"      绿框=GT  红框=OWLv2 预测（threshold=0.05）")
    print(f"      每图预测数：{n_box}")

    result = {
        "model_id": MODEL_ID,
        "n_params_m": round(n_param / 1e6),
        "alignment_ok": bool(ok_align),
        "alignment_cases": cases,
        "rates_img_per_sec": {k: float(v) for k, v in rates.items()},
        "chosen_device": best_dev,
        "eta_minutes": {k: float(N_INFER / v / 60) for k, v in rates.items()},
        "target_sizes_convention": "padded square side = max(H, W)，不是 (H, W)",
    }
    out = P.artifact("metrics") / "s7b_owlv2_probe.json"
    # numpy 的 bool_ / float32 不能直接进 json，统一转成 Python 原生类型
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_safe),
        encoding="utf-8",
    )
    print(f"\n已写入 {out.relative_to(P.artifacts)}")
    print("\n下一步：python scripts/s7b_select_prompt.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
