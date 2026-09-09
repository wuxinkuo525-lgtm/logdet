#!/usr/bin/env python
"""诊断：密度切片为什么不是单调的？

S7 的切片表里密度维度出现一个反直觉排序（L2 OWLv2，AP50）：

    DENSITY_1   0.2708
    DENSITY_2   0.3540   ← 比单框图更高
    DENSITY_3_4 0.2883
    DENSITY_5+  0.2094

直觉预期是单调递减（图里目标越多越难）。第一反应是「被尺寸混淆了」——
S6 已经证明切片之间不独立，密度轴与尺寸轴天然相关：

    DENSITY_1     中位 area 31,893   large 占 77.0%   T1 占 33.9%
    DENSITY_2     中位 area  7,448   large 占 46.7%   T1 占 20.3%
    DENSITY_3_4   中位 area  2,342   large 占 19.6%   T1 占  9.8%
    DENSITY_5+    中位 area  1,980   large 占 14.1%   T1 占  3.8%

但这个方向**解释不了**观察：D1 是四档里最偏大目标的，而尺寸是最强的
正向因子（small 0.0666 → large 0.3486），所以尺寸混淆应该让 D1 显得
更**高**，不是更低。也就是说 D2 > D1 这件事在扣掉尺寸后可能更明显。

本脚本做受控对比来验证：**只在 large + 非截断的框内比密度**，
让密度成为唯一变量。同时对两个 baseline 各跑一遍 —— 若两个架构完全
不同的模型给出同向结果，则更可能是数据性质而非某个模型的怪癖。

⚠️ 预先说明局限：受控之后 D2/D3_4/D5+ 三格的框数会掉到 200 的
可靠线以下。所以本脚本的产出**不是结论，是一个有量化支撑的开放问题**，
留给训练阶段用更大的 val 子集回答（val_full 有 17,216 图）。

用法：
    python scripts/s7_diagnose_density.py [subset]      # 默认 val_hard_pool
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.data.adapters.to_coco_json import build_coco_dt  # noqa: E402
from logodet.eval.slice_eval import (  # noqa: E402
    SliceSpec,
    evaluate_slices,
    slice_members,
)
from logodet.paths import P  # noqa: E402

BASELINES = [
    ("L2 OWLv2 zero-shot", "L2_owlv2_zeroshot"),
    ("L3 COCO 类塌缩", "L3_coco_collapsed"),
]

# 受控格的可靠线。默认 200（与 configs/splits.yaml 的 min_ann_for_reliable
# 一致），但受控切片必然更小，所以这里只用它来**标注**而不是过滤 ——
# 把不可靠的格子藏起来比诚实标出来更糟。
RELIABLE_MIN = 200


def _density_bins() -> list[tuple[str, int, int | None]]:
    return [("D1", 1, 1), ("D2", 2, 2), ("D3_4", 3, 4), ("D5plus", 5, None)]


def _make_specs(controlled: bool) -> list[SliceSpec]:
    """controlled=True 时额外要求 large + 非截断，使密度成为唯一变量。"""

    def build(lo: int, hi: int | None):
        def pred(a: pd.DataFrame) -> pd.Series:
            n = a["n_boxes"]
            m = (n >= lo) if hi is None else ((n >= lo) & (n <= hi))
            if controlled:
                m = m & (a["area_bin"] == "large") & ~a["is_hard_t1"]
            return m

        return pred

    tag = "_large_noT1" if controlled else ""
    axis = "controlled_density" if controlled else "density"
    return [
        SliceSpec(f"{name}{tag}", axis, build(lo, hi),
                  "受控：尺寸与截断已固定" if controlled else "原始密度切片")
        for name, lo, hi in _density_bins()
    ]


def main() -> int:
    subset = sys.argv[1] if len(sys.argv) > 1 else "val_hard_pool"
    gt_path = P.artifact("eval") / "gt" / f"{subset}.json"
    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    ann = pd.read_parquet(P.artifact("tables") / "annotations.parquet")
    img = pd.read_parquet(P.artifact("splits") / "split_images.parquet")
    keep_ann = {int(a["id"]) for a in gt["annotations"]}
    keep_img = {int(i["id"]) for i in gt["images"]}
    ann = ann[ann["ann_id"].isin(keep_ann)]
    img_sub = img[img["image_id"].isin(keep_img)]

    print("=" * 78)
    print(f" 密度维度受控对比诊断（子集 {subset}）")
    print("=" * 78)

    # ---- 先量清各档的成分，证明「尺寸混淆」这个解释不成立 -----------------
    a = ann.merge(img_sub[["image_id", "n_boxes"]], on="image_id", how="left",
                  suffixes=("", "_img"))
    print("\n各密度档的成分（说明为什么尺寸混淆解释不了 D2 > D1）：\n")
    print(f"  {'档':<10} {'框数':>6} {'中位 area':>10} {'large 占比':>10} "
          f"{'small 占比':>10} {'T1 占比':>8}")
    comp_rows = []
    for name, lo, hi in _density_bins():
        n = a["n_boxes"]
        m = (n >= lo) if hi is None else ((n >= lo) & (n <= hi))
        s = a[m]
        row = {
            "bin": name,
            "n_ann": len(s),
            "median_area": float(s["area"].median()) if len(s) else float("nan"),
            "frac_large": float((s["area_bin"] == "large").mean()) if len(s) else 0.0,
            "frac_small": float((s["area_bin"] == "small").mean()) if len(s) else 0.0,
            "frac_t1": float(s["is_hard_t1"].mean()) if len(s) else 0.0,
        }
        comp_rows.append(row)
        print(f"  {name:<10} {row['n_ann']:>6,} {row['median_area']:>10,.0f} "
              f"{row['frac_large']:>9.1%} {row['frac_small']:>9.1%} "
              f"{row['frac_t1']:>7.1%}")

    print("\n  → D1 是四档里最偏大目标的（large 77%，中位面积是 D2 的 4.3 倍），")
    print("    而尺寸是最强的正向因子。所以尺寸混淆只会让 D1 显得**更高**，")
    print("    解释不了 D2 > D1 —— 必须扣掉尺寸再看。")

    # ---- 受控对比：只在 large + 非截断框内比密度 --------------------------
    specs_raw = _make_specs(controlled=False)
    specs_ctrl = _make_specs(controlled=True)
    mem_raw = slice_members(ann, img_sub, specs_raw)
    mem_ctrl = slice_members(ann, img_sub, specs_ctrl)

    out_rows: list[dict] = []
    for tag, fname in BASELINES:
        pred_path = P.artifact("predictions") / f"{fname}.parquet"
        if not pred_path.exists():
            print(f"\n跳过 {tag}：{pred_path.name} 不存在")
            continue
        pred = pd.read_parquet(pred_path)
        dt = build_coco_dt(pred[pred["image_id"].isin(keep_img)])

        for label, specs, mem in (("原始", specs_raw, mem_raw),
                                  ("受控", specs_ctrl, mem_ctrl)):
            res = evaluate_slices(gt, dt, mem, specs, min_ann_for_reliable=RELIABLE_MIN,
                                  n_boot=0)
            by = {r.name: r for r in res}
            print(f"\n{tag} —— {label}"
                  f"{'（只在 large + 非截断框内）' if label == '受控' else ''}\n")
            print(f"  {'切片':<22} {'框数':>6} {'可靠':>5} {'AP50':>9} {'AP':>9} "
                  f"{'AR@300':>9}")
            for s in specs:
                r = by.get(s.name)
                if r is None:
                    continue
                ap50 = r.metrics.get("AP50", float("nan"))
                ap = r.metrics.get("AP", float("nan"))
                ar = r.metrics.get("AR@300", float("nan"))
                flag = "是" if r.n_ann >= RELIABLE_MIN else "**否**"
                print(f"  {s.name:<22} {r.n_ann:>6,} {flag:>5} {ap50:>9.4f} "
                      f"{ap:>9.4f} {ar:>9.4f}")
                out_rows.append({
                    "baseline": fname, "mode": label, "slice": s.name,
                    "n_ann": r.n_ann, "reliable": r.n_ann >= RELIABLE_MIN,
                    "AP50": ap50, "AP": ap, "AR@300": ar,
                })

    # ---- 落盘 --------------------------------------------------------------
    df = pd.DataFrame(out_rows)
    comp = pd.DataFrame(comp_rows)
    out = P.artifact("metrics") / f"s7_density_controlled_{subset}.parquet"
    df.to_parquet(out, index=False)
    comp.to_parquet(
        P.artifact("metrics") / f"s7_density_composition_{subset}.parquet", index=False
    )
    print(f"\n结果已落盘 {out.relative_to(P.artifacts)}")

    # ---- 判读 --------------------------------------------------------------
    print("\n" + "=" * 78)
    print(" 判读")
    print("=" * 78)

    ctrl = df[df["mode"] == "受控"]
    verdicts = []
    for fname in ctrl["baseline"].unique():
        c = ctrl[ctrl["baseline"] == fname].set_index("slice")
        try:
            d1 = c.loc["D1_large_noT1", "AP50"]
            d2 = c.loc["D2_large_noT1", "AP50"]
        except KeyError:
            continue
        verdicts.append((fname, d1, d2, d2 - d1))
        print(f"\n  {fname}")
        print(f"    D1（受控）AP50 = {d1:.4f}   D2（受控）AP50 = {d2:.4f}   "
              f"差 {d2 - d1:+.4f}")

    if verdicts and all(v[3] > 0 for v in verdicts):
        print("\n  → D2 > D1 在扣掉尺寸与截断之后**依然成立**，且在两个架构完全")
        print("    不同的模型上同向复现。所以它不是尺寸混淆，也不像单个模型的怪癖。")

    n_small = int((~ctrl["reliable"]).sum())
    print(f"\n  ⚠️ 但受控格里有 {n_small} 个的框数低于 {RELIABLE_MIN} 的可靠线。")
    print("    因此这**不是结论，是一个有量化支撑的开放问题**：")
    print("      · 待答：单框图里的 logo 是否系统性地更「贴满画面」，")
    print("        从而落在开放词表模型的舒适区之外？")
    print("      · 怎么答：训练阶段改用 val_full（17,216 图）重跑本脚本，")
    print("        受控格样本量会放大约 14 倍，足以给出可靠判据。")
    print("\n  报告中引用这些数字时必须带上「受控格样本量不足」这句限定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
