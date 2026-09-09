#!/usr/bin/env python
"""S2 步骤三：生成代理核验清单。

**核验预算相对初版计划大幅缩减，因为 T1 有了真标注。**

初版计划：三个代理各抽 25 张，共 75 张，全靠人工判断。
现在：
    T1 截断  →  **0 张**。有 truncated 真标注，且几何代理已用它校准
                （precision=0.857 / recall=0.961 / MCC=0.897，见 S2-G11）
    P2 互遮挡 →  25 张
    P3 形状离群 → 35 张
    合计 60 张 → 你只需复核我标「不确定」和「判为非难例」的约 15-20 张

抽样策略：不是均匀随机，而是**按代理值分层**（阈值附近 / 中段 / 极端），
因为阈值附近的判断最有信息量 —— 那里才决定阈值定得对不对。

用法：
    python scripts/s2_make_review_sheet.py
产出：
    runs/slices/review_sheet.csv          待填表
    runs/slices/crops/<slice>/<ann_id>.jpg 裁剪图（原图 + 红框 + 邻域）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.paths import P  # noqa: E402
from logodet.seeds import seed_for  # noqa: E402

# 每个切片抽多少张，以及分层比例（阈值附近 / 中段 / 极端）
QUOTA = {"P2": 25, "P3": 35}
STRATA = (("阈值附近", 0.4), ("中段", 0.35), ("极端", 0.25))

# 裁剪图：目标框外扩多少倍，用于展示邻域上下文
CONTEXT_SCALE = 2.2
CROP_MAX_SIDE = 640


def _pick_stratified(sub: pd.DataFrame, value_col: str, n: int, rng) -> pd.DataFrame:
    """按代理值分层抽样。阈值附近权重最高 —— 那里的判断最有信息量。"""
    if len(sub) <= n:
        return sub
    s = sub.sort_values(value_col).reset_index(drop=True)
    thirds = np.array_split(s.index.to_numpy(), 3)
    picks: list[pd.DataFrame] = []
    for (label, frac), idx in zip(STRATA, thirds):
        k = max(1, int(round(n * frac)))
        k = min(k, len(idx))
        chosen = rng.choice(idx, size=k, replace=False)
        part = s.loc[chosen].copy()
        part["stratum"] = label
        picks.append(part)
    out = pd.concat(picks, ignore_index=True)
    return out.head(n)


def _render_crop(img_path: Path, box: tuple[float, float, float, float],
                 out_path: Path) -> bool:
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            half = max(bw, bh) * CONTEXT_SCALE / 2
            cx0 = max(0, int(cx - half))
            cy0 = max(0, int(cy - half))
            cx1 = min(W, int(cx + half))
            cy1 = min(H, int(cy + half))
            crop = im.crop((cx0, cy0, cx1, cy1))

            d = ImageDraw.Draw(crop)
            d.rectangle(
                [x1 - cx0, y1 - cy0, x2 - cx0, y2 - cy0],
                outline=(255, 32, 32), width=max(2, int(min(crop.size) * 0.006)),
            )
            if max(crop.size) > CROP_MAX_SIDE:
                r = CROP_MAX_SIDE / max(crop.size)
                crop = crop.resize((int(crop.width * r), int(crop.height * r)))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out_path, quality=88)
        return True
    except Exception as e:  # 单张失败不该中断整批
        print(f"    [warn] 渲染失败 {img_path.name}: {e}")
        return False


def main() -> int:
    t = P.artifact("tables")
    ann = pd.read_parquet(t / "annotations.parquet")
    img = pd.read_parquet(t / "images.parquet")
    a = ann.merge(img[["image_id", "rel_path"]], on="image_id", how="left")

    slices_dir = P.artifact("slices")
    crops_root = slices_dir / "crops"

    spec = {
        "P2": ("is_hard_p2", "p2_occ_iof", "同图另一个框压住了它 → 是否真的构成遮挡"),
        "P3": ("is_hard_p3", "p3_ar_z", "同类里长宽比离群 → 是否由遮挡/截断/形变导致"),
    }

    print("=" * 78)
    print(" S2 步骤三：生成代理核验清单")
    print("=" * 78)
    print("\nT1 截断不需要人工核验 —— 有 truncated 真标注，")
    print("且几何代理已用它校准：precision=0.857 recall=0.961 MCC=0.897（S2-G11）\n")

    rows: list[pd.DataFrame] = []
    for name, (flag, value_col, question) in spec.items():
        sub = a[a[flag]].copy()
        rng = seed_for(f"review_{name}")
        picked = _pick_stratified(sub, value_col, QUOTA[name], rng)
        picked["slice"] = name
        picked["proxy_value"] = picked[value_col].round(4)
        picked["question"] = question
        rows.append(picked)
        print(f"  {name}: 候选 {len(sub):,} 框 → 抽 {len(picked)} 张")

    sheet = pd.concat(rows, ignore_index=True)

    print("\n渲染裁剪图 ...")
    ok = 0
    crop_paths: list[str] = []
    for r in sheet.itertuples():
        out = crops_root / r.slice / f"{r.ann_id}.jpg"
        if _render_crop(P.dataset / r.rel_path, (r.x1, r.y1, r.x2, r.y2), out):
            ok += 1
        crop_paths.append(str(out.relative_to(P.artifacts)))
    sheet["crop_path"] = crop_paths
    print(f"  成功 {ok}/{len(sheet)}")

    out_cols = [
        "ann_id", "slice", "stratum", "proxy_value", "question",
        "class_name", "supercat", "rel_path", "crop_path",
        "bw", "bh", "ar", "area_bin", "truncated",
    ]
    sheet["vlm_label"] = ""
    sheet["vlm_reason"] = ""
    sheet["human_label"] = ""
    sheet["final_label"] = ""
    sheet["note"] = ""
    out_cols += ["vlm_label", "vlm_reason", "human_label", "final_label", "note"]

    csv_path = slices_dir / "review_sheet.csv"
    sheet[out_cols].to_csv(csv_path, index=False)

    print(f"\n清单已写入 {csv_path}")
    print(f"裁剪图在  {crops_root}")
    print("\n填表说明：")
    print("  vlm_label / human_label / final_label 取值 1=确实难例  0=不是  ?=不确定")
    print("  流程：我先逐张填 vlm_label + vlm_reason，")
    print("        你只复核 vlm_label ∈ {0, ?} 的那些（约 15-20 张，5 分钟）")
    print("  报告口径必须写成 'VLM-assisted, N/60 human-reviewed' 并注明模型，")
    print("  禁止简写为「人工核验」—— 这是学术诚信要求。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
