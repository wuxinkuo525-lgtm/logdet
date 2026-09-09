#!/usr/bin/env python
"""合并人工复核结果，重算代理 precision，并更新 slices.yaml 的核验来源。

运行前：在 review_sheet.csv 的 human_label 列填 1 / 0 / ?（只填需要复核的行即可）
运行后：
    * final_label = human_label 非空则取它，否则取 vlm_label
    * 重算两个代理的 precision（? 不计入分母）
    * 把 human_reviewed 数量与最终 precision 写回 configs/slices.yaml
    * 若 precision 跨过 0.60 门槛，明确提示裁决需要改变

用法：
    python scripts/s2_merge_human_labels.py
    python scripts/s2_merge_human_labels.py --dry-run   # 只看结果不写文件
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.paths import P  # noqa: E402

VALID = {"1", "0", "?"}
THRESHOLD = 0.60
SLICES_YAML = PROJECT_DIR / "configs" / "slices.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写回文件")
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="ANN_ID=LABEL",
        help="直接写入人工标签，如 --set 73736=0 106646=? "
        "（避免手改 CSV 出格式错；不带此参数则读 CSV 里已填的值）",
    )
    args = ap.parse_args()

    sheet_path = P.artifact("slices") / "review_sheet.csv"
    df = pd.read_csv(sheet_path, dtype={"human_label": "string", "vlm_label": "string"})

    # ---- --set 写入 --------------------------------------------------------
    if args.set:
        idx = {int(a): i for i, a in enumerate(df["ann_id"])}
        for item in args.set:
            if "=" not in item:
                print(f"FAIL: --set 参数格式应为 ANN_ID=LABEL，收到 {item!r}")
                return 1
            k, v = item.split("=", 1)
            k, v = k.strip(), v.strip()
            if not k.isdigit() or int(k) not in idx:
                print(f"FAIL: ann_id {k!r} 不在清单中")
                return 1
            if v not in VALID:
                print(f"FAIL: 标签只能是 1 / 0 / ?，收到 {v!r}（ann_id={k}）")
                return 1
            df.loc[idx[int(k)], "human_label"] = v
        print(f"已写入 {len(args.set)} 个人工标签")

    # ---- 校验填写格式 ------------------------------------------------------
    filled = df["human_label"].notna() & (df["human_label"].str.strip() != "")
    bad = df[filled & ~df["human_label"].str.strip().isin(VALID)]
    if len(bad):
        print("FAIL: human_label 只能填 1 / 0 / ?，以下行取值非法：")
        for r in bad.itertuples():
            print(f"  ann_id={r.ann_id}  填的是 {r.human_label!r}")
        return 1

    n_human = int(filled.sum())
    print("=" * 74)
    print(" 合并人工复核结果")
    print("=" * 74)
    print(f"\n已填人工标签 : {n_human} / {len(df)} 行")

    if n_human == 0:
        print("\n human_label 全为空，无需合并。先在 CSV 里填写再运行。")
        return 0

    # ---- 合并 -------------------------------------------------------------
    df["human_label"] = df["human_label"].fillna("").str.strip()
    df["final_label"] = df.apply(
        lambda r: r["human_label"] if r["human_label"] else r["vlm_label"], axis=1
    )

    changed = df[filled & (df["human_label"] != df["vlm_label"])]
    print(f"人工与 VLM 判断不同 : {len(changed)} 行")
    for r in changed.itertuples():
        print(f"  ann_id={r.ann_id:<8} {r.slice}  VLM={r.vlm_label} → 人工={r.human_label}"
              f"   ({r.vlm_category})")

    # ---- 重算 precision ---------------------------------------------------
    print("\n" + "-" * 74)
    results: dict[str, dict] = {}
    for sl_name, key in (("P2", "p2_occlusion"), ("P3", "p3_shape_outlier")):
        sub = df[df["slice"] == sl_name]
        for stage, col in (("VLM 预标", "vlm_label"), ("合并人工后", "final_label")):
            n1 = int((sub[col] == "1").sum())
            n0 = int((sub[col] == "0").sum())
            nq = int((sub[col] == "?").sum())
            prec = n1 / (n1 + n0) if (n1 + n0) else float("nan")
            mark = "达标" if prec >= THRESHOLD else "未达标"
            print(f"{sl_name} {stage:<12} 1={n1:>2} 0={n0:>2} ?={nq:>2}  "
                  f"precision={prec:.3f}  {mark}")
            if col == "final_label":
                results[key] = {
                    "precision": round(float(prec), 4),
                    "label_1": n1, "label_0": n0, "label_unknown": nq,
                    "verdict": "KEPT" if prec >= THRESHOLD else "DEMOTED",
                }
        print()

    # ---- 裁决是否需要改变 --------------------------------------------------
    flipped = [k for k, v in results.items() if v["verdict"] == "KEPT"]
    if flipped:
        print("!" * 74)
        print(f" 注意：{flipped} 的 precision 已跨过 {THRESHOLD} 门槛。")
        print(" 需要重新评估是否恢复为主力切片轴，并同步修改：")
        print("   configs/slices.yaml 的 proxy_verdict / hard_any_definition")
        print("   scripts/s2_build_tables.py 的 hard_any 组成")
        print("   README.md 的发现 ④ 与切片体量表")
        print("!" * 74)
    else:
        print(f"两个代理合并人工后仍未达 {THRESHOLD} 门槛 → 维持降级裁决，无需改动切片轴。")

    if args.dry_run:
        print("\n--dry-run，未写回任何文件。")
        return 0

    # ---- 写回 -------------------------------------------------------------
    df.to_csv(sheet_path, index=False)
    print(f"\n已更新 {sheet_path}")

    cfg = yaml.safe_load(SLICES_YAML.read_text(encoding="utf-8"))
    pv = cfg.setdefault("proxy_verdict", {})
    for key, vals in results.items():
        pv.setdefault(key, {}).update(
            {
                "precision_after_human_review": vals["precision"],
                "label_1_final": vals["label_1"],
                "label_0_final": vals["label_0"],
                "label_unknown_final": vals["label_unknown"],
                "verdict_after_human_review": vals["verdict"],
            }
        )
    rp = cfg.setdefault("review_provenance", {})
    rp["human_reviewed"] = n_human
    rp["human_vlm_disagreements"] = int(len(changed))
    SLICES_YAML.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    print(f"已更新 {SLICES_YAML.relative_to(PROJECT_DIR)}")
    print(f"\n报告口径应写成：VLM-assisted, {n_human}/{len(df)} human-reviewed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
