#!/usr/bin/env python
"""S7b 步骤三：OWLv2 全量推理。

prompt 由 s7b_select_prompt.py 在 **trainval** 上选定（不碰 val，无泄漏）。
设备由 s7b_probe_owlv2.py 实测选定（MPS 比 CPU 快 4.3 倍 —— OWLv2 是纯 ViT，
没有 roi_align 与逐类 NMS，所以不会像 S7 的 Faster R-CNN 那样在 MPS 上崩）。

启动时会**重跑一次几何对齐验证**：代码路径合并后必须确认没退化。
这条不通过就直接退出，不浪费 23 分钟去跑一堆错位的框。

用法：
    python scripts/s7b_run_owlv2.py
    python scripts/s7b_run_owlv2.py --limit 50
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from logodet.baselines.owlv2_detector import (  # noqa: E402
    MODEL_ID,
    detect,
    load_owlv2,
    synthetic_alignment_check,
)
from logodet.config import fingerprint, load_yaml  # noqa: E402
from logodet.data.core_dataset import CoreDataset, ZipImageReader  # noqa: E402
from logodet.paths import P  # noqa: E402

THRESHOLD = 0.02
MAX_DETS = 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mdir = P.artifact("metrics")
    sel_path = mdir / "s7b_prompt_selection.json"
    probe_path = mdir / "s7b_owlv2_probe.json"
    for p in (sel_path, probe_path):
        if not p.is_file():
            print(f"FAIL: 缺 {p.name}，先跑 s7b_probe_owlv2.py 与 s7b_select_prompt.py")
            return 1
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    queries = sel["chosen_queries"]
    device = probe["chosen_device"]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("=" * 78)
    print(f" S7b OWLv2 推理（device={device}, run_id={run_id}）")
    print("=" * 78)
    print(f"\nprompt = {queries}")
    print(f"  选型依据：trainval {sel['n_images']} 图，AP50 最优（无 val 泄漏）")

    processor, model = load_owlv2(device)

    # ---- 启动前重验几何对齐 ------------------------------------------------
    print("\n几何对齐复验（代码路径已合并，必须确认无退化）...")
    ok, cases = synthetic_alignment_check(processor, model, device=device)
    for c in cases:
        if c.get("detected"):
            print(f"  {c['prompt']:<16} 中心偏移 {c['center_offset_px']}  "
                  f"落在色块内={c['center_inside']}")
    if not ok:
        print("  对齐验证失败 → 中止，避免白跑 23 分钟出一堆错位的框")
        return 1
    print("  对齐正确")

    # ---- 推理集合 ---------------------------------------------------------
    sp, t = P.artifact("splits"), P.artifact("tables")
    images = pd.read_parquet(sp / "split_images.parquet")
    ann = pd.read_parquet(t / "annotations.parquet")
    v2k = pd.read_parquet(sp / "val2k_repr.parquet")
    vhp = pd.read_parquet(sp / "val_hard_pool.parquet")
    ids = sorted(set(v2k["image_id"]) | set(vhp["image_id"]))
    if args.limit:
        ids = ids[: args.limit]

    cfg = load_yaml("dataset.yaml")
    zp = P.raw / cfg["source"]["zip_name"]
    reader = ZipImageReader(zp, cfg["parse"]["archive_top_dir"]) if zp.is_file() else None
    if reader:
        reader.build_index(set(images[images["image_id"].isin(set(ids))]["rel_path"]))
    core = CoreDataset(images, ann, dataset_root=P.dataset, image_ids=ids,
                       load_image=True, zip_reader=reader)
    print(f"\n推理集合 {len(core):,} 图")

    rows: list[dict] = []
    n_det: list[int] = []
    q_hits = np.zeros(len(queries), dtype="int64")
    t0 = time.time()
    for i in range(len(core)):
        s = core[i]
        o = detect(processor, model, s.image, queries, device=device,
                   threshold=THRESHOLD, max_dets=MAX_DETS, image_id=s.image_id)
        n_det.append(len(o.boxes_xyxy))
        for qi in o.query_idx:
            q_hits[int(qi)] += 1
        for (x1, y1, x2, y2), sc, qi in zip(o.boxes_xyxy, o.scores, o.query_idx):
            rows.append({
                "image_id": s.image_id, "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2), "score": float(sc),
                "category_id": 1, "query_idx": int(qi),
            })
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            rate = (i + 1) / el
            print(f"  {i + 1:>5,}/{len(core):,}  {rate:.2f} img/s  "
                  f"剩余约 {(len(core) - i - 1) / rate / 60:.1f} 分钟", flush=True)

    dt = time.time() - t0
    df = pd.DataFrame(rows)
    df["baseline"] = "L2_owlv2_zeroshot"
    df["run_id"] = run_id
    out_path = P.artifact("predictions") / "L2_owlv2_zeroshot.parquet"
    df.to_parquet(out_path, index=False)

    manifest = {
        "run_id": run_id,
        "baseline": "L2_owlv2_zeroshot",
        "model_id": MODEL_ID,
        "n_params_m": probe.get("n_params_m"),
        "device": device,
        "queries": queries,
        "prompt_selection": {
            "split": "trainval",
            "n_images": sel["n_images"],
            "metric": sel["selection_metric"],
            "no_val_leakage": True,
        },
        "threshold": THRESHOLD,
        "max_dets": MAX_DETS,
        "alignment_verified": True,
        "alignment_cases": cases,
        "config_fingerprint": fingerprint({
            "dataset": load_yaml("dataset.yaml"),
            "splits": load_yaml("splits.yaml"),
        }),
        "n_images": len(core),
        "seconds": round(dt, 1),
        "img_per_sec": round(len(core) / dt, 3),
        "detections": {
            "rows": len(df),
            "per_image_median": int(np.median(n_det)),
            "per_image_max": int(max(n_det)),
            "per_image_zero": int(sum(1 for x in n_det if x == 0)),
        },
        "query_hit_counts": {q: int(c) for q, c in zip(queries, q_hits)},
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (P.artifact("predictions") / f"manifest_owlv2_{run_id}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n完成：{len(core):,} 图 / {dt / 60:.1f} 分钟 / {len(core) / dt:.2f} img/s")
    print(f"  detections {len(df):,} 行  每图中位数 {int(np.median(n_det))}，"
          f"最大 {max(n_det)}，零检出 {manifest['detections']['per_image_zero']} 张")
    print("  各 query 命中数：")
    for q, c in zip(queries, q_hits):
        print(f"    {q:<20} {int(c):>8,}")
    print(f"\n产物写入 {out_path.relative_to(P.artifacts)}")
    print("下一步：python scripts/s7_evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
