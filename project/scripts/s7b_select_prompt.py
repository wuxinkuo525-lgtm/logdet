#!/usr/bin/env python
"""S7b 步骤二：在 trainval 抽样上选文本 prompt。

### 为什么必须用 trainval

OWLv2 对文本 prompt 高度敏感，prompt 是**超参**。若在 val2k_repr 上比较
prompt 再报同一集合的指标，就是在报告用的集合上调超参 —— 数据泄漏，
报出来的数字会系统性偏高。所以选型只用 trainval 抽样
（本项目从不在 trainval 上报任何数字）。

### 一次前向评所有候选

OWLv2 的每个文本 query 各自与视觉特征做点积，**query 之间互不影响**。
所以把全部候选 query 一次性喂进去，拿到 per-query 分数后再按子集取 max，
与分别前向完全等价 —— 6 个候选的开销从 6 次前向降到 1 次。

用法：
    python scripts/s7b_select_prompt.py [图数]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from logodet.baselines.owlv2_detector import (  # noqa: E402
    MODEL_ID,
    PROMPT_CANDIDATES,
    load_owlv2,
    raw_forward,
)
from logodet.config import load_yaml  # noqa: E402
from logodet.data.adapters.to_coco_json import build_coco_dt, build_coco_gt  # noqa: E402
from logodet.eval.coco_eval import DEFAULT_MAX_DETS, evaluate  # noqa: E402
from logodet.paths import P  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
THRESHOLD = 0.02
MAX_DETS = 100


@torch.inference_mode()
def detect_all_queries(
    processor, model, image_rgb: np.ndarray, queries: list[str], device: str
) -> tuple[np.ndarray, np.ndarray]:
    """一次前向拿到 (boxes, per_query_scores)。

    直接复用 owlv2_detector.raw_forward —— 全项目只有那一处做 OWLv2 的
    坐标换算。早期这里有一份重复实现，与 detect() 构成两条几何路径，
    是静默不一致的温床，已合并。
    """
    return raw_forward(processor, model, image_rgb, queries, device)


def main() -> int:
    sp, t = P.artifact("splits"), P.artifact("tables")
    images = pd.read_parquet(sp / "split_images.parquet")
    ann = pd.read_parquet(t / "annotations.parquet")

    # ---- 只用 trainval，绝不碰 val ----------------------------------------
    pool = images[images["split"] == "trainval"]
    rng = np.random.default_rng(6129)
    ids = sorted(rng.choice(pool["image_id"].to_numpy(), size=min(N, len(pool)),
                            replace=False).tolist())
    assert not set(ids) & set(images.loc[images["split"] == "val", "image_id"]), \
        "选型样本混入了 val —— 这会造成数据泄漏"

    print("=" * 78)
    print(f" S7b prompt 选型（trainval 抽样 {len(ids)} 图）")
    print("=" * 78)
    print(f"\n候选 prompt：{list(PROMPT_CANDIDATES)}")

    # 全部候选的 query 去重，一次前向拿 per-query 分数
    all_q: list[str] = []
    for qs in PROMPT_CANDIDATES.values():
        for q in qs:
            if q not in all_q:
                all_q.append(q)
    qidx = {q: i for i, q in enumerate(all_q)}
    print(f"去重后 query 共 {len(all_q)} 条：{all_q}")

    from logodet.data.core_dataset import CoreDataset, ZipImageReader

    cfg = load_yaml("dataset.yaml")
    zp = P.raw / cfg["source"]["zip_name"]
    reader = ZipImageReader(zp, cfg["parse"]["archive_top_dir"]) if zp.is_file() else None
    if reader:
        reader.build_index(set(images[images["image_id"].isin(set(ids))]["rel_path"]))
    core = CoreDataset(images, ann, dataset_root=P.dataset, image_ids=ids,
                       load_image=True, zip_reader=reader)

    probe = json.loads(
        (P.artifact("metrics") / "s7b_owlv2_probe.json").read_text(encoding="utf-8")
    ) if (P.artifact("metrics") / "s7b_owlv2_probe.json").is_file() else {}
    device = probe.get("chosen_device", "mps")
    print(f"device = {device}\n")

    processor, model = load_owlv2(device)

    # ---- 一次前向，缓存 per-query 分数 -------------------------------------
    cache: list[tuple[int, np.ndarray, np.ndarray]] = []
    t0 = time.time()
    for i in range(len(core)):
        s = core[i]
        b, sc = detect_all_queries(processor, model, s.image, all_q, device)
        cache.append((s.image_id, b, sc))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(core)}  {(i + 1) / (time.time() - t0):.2f} img/s",
                  flush=True)
    print(f"  前向完成 {time.time() - t0:.1f}s\n")

    # ---- GT --------------------------------------------------------------
    gt_dict = build_coco_gt(images, ann, image_ids=ids, class_agnostic=True,
                            info={"subset": "trainval_prompt_selection"})
    import contextlib
    import io

    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()

    # ---- 逐候选评测 --------------------------------------------------------
    print(f"{'候选':<10}{'queries':<44}{'AP':>8}{'AP50':>9}{'AR@100':>9}")
    rows = []
    for name, qs in PROMPT_CANDIDATES.items():
        cols = [qidx[q] for q in qs]
        recs = []
        for image_id, b, sc in cache:
            if not len(b):
                continue
            best = sc[:, cols].max(axis=1)  # 子集内取 max —— 与单独前向等价
            keep = best >= THRESHOLD
            bb, ss = b[keep], best[keep]
            if len(ss) > MAX_DETS:
                order = np.argsort(-ss)[:MAX_DETS]
                bb, ss = bb[order], ss[order]
            for (x1, y1, x2, y2), s_ in zip(bb, ss):
                recs.append({"image_id": image_id, "x1": float(x1), "y1": float(y1),
                             "x2": float(x2), "y2": float(y2),
                             "score": float(s_), "category_id": 1})
        if not recs:
            rows.append({"name": name, "queries": qs, "AP": 0.0, "AP50": 0.0,
                         "AR@100": 0.0, "n_dt": 0})
            print(f"{name:<10}{str(qs)[:42]:<44}{0:>8.4f}{0:>9.4f}{0:>9.4f}")
            continue
        r = evaluate(gt, build_coco_dt(pd.DataFrame(recs)), image_ids=ids,
                     max_dets=DEFAULT_MAX_DETS)
        rows.append({"name": name, "queries": qs, "n_dt": len(recs),
                     **{k: float(v) for k, v in r.metrics.items()}})
        print(f"{name:<10}{str(qs)[:42]:<44}"
              f"{r.metrics['AP']:>8.4f}{r.metrics['AP50']:>9.4f}"
              f"{r.metrics.get('AR@100', float('nan')):>9.4f}")

    df = pd.DataFrame(rows)
    # 用 AP50 选：class-agnostic 单类检测里 AP50 最能反映"找到没找到"，
    # 而 AP@[.5:.95] 会被定位精度主导，那是另一个问题
    best = df.loc[df["AP50"].idxmax()]

    print("\n" + "=" * 78)
    print(f" 选定 prompt = {best['name']}  →  {best['queries']}")
    print(f" 依据：trainval {len(ids)} 图上 AP50={best['AP50']:.4f}（AP={best['AP']:.4f}）")
    print("=" * 78)

    out = P.artifact("metrics") / "s7b_prompt_selection.json"
    out.write_text(json.dumps({
        "model_id": MODEL_ID,
        "selection_split": "trainval",
        "n_images": len(ids),
        "no_val_leakage": True,
        "selection_metric": "AP50",
        "selection_rationale": (
            "class-agnostic 单类检测里 AP50 最能反映「找到没找到」；"
            "AP@[.5:.95] 会被定位精度主导，那是另一个问题"
        ),
        "threshold": THRESHOLD,
        "chosen": best["name"],
        "chosen_queries": best["queries"],
        "all_results": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {out.relative_to(P.artifacts)}")
    print("下一步：python scripts/s7b_run_owlv2.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
