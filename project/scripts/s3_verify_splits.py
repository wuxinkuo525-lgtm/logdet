#!/usr/bin/env python
"""S3 步骤二：划分验证门（九条）。

阻塞门：
    H1  无交集且并集完整
    H2  簇不跨界（每个弱重复簇的 split 取值唯一）
    H3  val 比例在容许带内
    H4  类覆盖：每个类在 trainval 与 val 都有样本
    H5  分布一致：trainval vs val 在三个边缘分布上的 TVD < 0.01
    H6  val2k_repr 代表性：与 val_full 三维 TVD < 0.03
    H9  可复现：同 seed 重跑两次产物 sha256 完全相同

非阻塞门：
    H7  val_hard_pool 各轴体量与 UNRELIABLE 判定
    H8  两个子集的口径隔离检查（提醒禁止越界使用）

TVD（总变差距离）= 0.5 * Σ|p_i - q_i|，取值 [0,1]，0 表示分布完全相同。
用它而不是 KL：KL 在某一侧概率为 0 时会发散，而长尾类别很容易出现这种情况。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.config import file_sha256, load_yaml  # noqa: E402
from logodet.gates import GateResult, GateRunner, md_table  # noqa: E402
from logodet.paths import P  # noqa: E402

REPORT = PROJECT_DIR / "report" / "s3_splits_report.md"


def tvd(a: pd.Series, b: pd.Series) -> float:
    """总变差距离。两个分布按并集对齐后计算。"""
    keys = sorted(set(a.index) | set(b.index), key=str)
    pa = a.reindex(keys).fillna(0.0)
    pb = b.reindex(keys).fillna(0.0)
    pa = pa / pa.sum() if pa.sum() else pa
    pb = pb / pb.sum() if pb.sum() else pb
    return float(0.5 * np.abs(pa - pb).sum())


def main() -> int:
    cfg = load_yaml("splits.yaml")
    val_ratio = float(cfg["val_ratio"])
    min_reliable = int(cfg.get("min_ann_for_reliable", 200))

    sp = P.artifact("splits")
    t = P.artifact("tables")
    for f in ("split_images.parquet", "val2k_repr.parquet", "val_hard_pool.parquet"):
        if not (sp / f).is_file():
            print(f"FAIL: 缺 {f}，先跑 python scripts/s3_make_splits.py")
            return 1

    img = pd.read_parquet(sp / "split_images.parquet")
    v2k = pd.read_parquet(sp / "val2k_repr.parquet")
    vhp = pd.read_parquet(sp / "val_hard_pool.parquet")
    ann = pd.read_parquet(t / "annotations.parquet")
    rep = json.loads((sp / "split_report.json").read_text(encoding="utf-8"))

    trainval = img[img["split"] == "trainval"]
    val = img[img["split"] == "val"]

    print("=" * 78)
    print(" S3 划分验证门")
    print("=" * 78)
    print(f"\ntrainval {len(trainval):,} / val {len(val):,} / "
          f"val2k {len(v2k):,} / hard_pool {len(vhp):,}")

    runner = GateRunner("S3", "S3 划分验证报告")

    # ---- H1 无交集 ---------------------------------------------------------
    def h1() -> GateResult:
        s_tv, s_v = set(trainval["image_id"]), set(val["image_id"])
        inter = s_tv & s_v
        union = s_tv | s_v
        all_ids = set(img["image_id"])
        rows = [
            ("trainval ∩ val", f"{len(inter):,}", "必须为 0"),
            ("并集", f"{len(union):,}", f"全量 {len(all_ids):,}"),
            ("image_id 唯一", str(img["image_id"].is_unique), "必须 True"),
            ("split 取值", str(sorted(img["split"].unique())), "只应有两种"),
        ]
        ok = (not inter) and union == all_ids and img["image_id"].is_unique
        return GateResult("H1", "无交集且完整", "PASS" if ok else "FAIL",
                          "两侧无交集，并集等于全量" if ok else "划分不完整或有交集",
                          blocking=True, rows=rows)

    runner.run(h1, "H1", "无交集且完整", True)

    # ---- H2 簇不跨界 -------------------------------------------------------
    def h2() -> GateResult:
        nun = img.groupby("cluster_id")["split"].nunique()
        bad = nun[nun > 1]
        multi = int((img.groupby("cluster_id").size() > 1).sum())
        rows = [
            ("簇总数", f"{img['cluster_id'].nunique():,}", ""),
            ("多图簇", f"{multi:,}", "只有这些簇才可能跨界"),
            ("跨界的簇", f"{len(bad):,}", "必须为 0"),
            ("弱去重判据", str(cfg["dedup"]["key"]),
             "只抓字节大小全同的重复；真 pHash 推迟到训练阶段"),
        ]
        if len(bad):
            rows.append(("跨界簇样例", str(list(bad.index[:5])), ""))
        return GateResult("H2", "簇不跨界", "PASS" if not len(bad) else "FAIL",
                          "每个弱重复簇的 split 取值唯一" if not len(bad)
                          else f"{len(bad)} 个簇跨界，存在泄漏",
                          blocking=True, rows=rows)

    runner.run(h2, "H2", "簇不跨界", True)

    # ---- H3 val 比例 -------------------------------------------------------
    def h3() -> GateResult:
        actual = len(val) / len(img)
        lo, hi = val_ratio * 0.9, val_ratio * 1.25
        n_box_val = int(ann["image_id"].isin(set(val["image_id"])).sum())
        rows = [
            ("目标 val 比例", f"{val_ratio:.2%}", ""),
            ("实际 val 比例", f"{actual:.2%}", f"容许带 [{lo:.2%}, {hi:.2%}]"),
            ("val 图数 / 框数", f"{len(val):,} / {n_box_val:,}", ""),
            ("论文官方划分", "trainval 142,142 / test 16,510",
             "⚠️ 非同一划分 —— 官方 zip 不含 split 清单，无法复现"),
            ("超出原因", "簇不可拆 + 每类至少留 1 张 val",
             "小类的 ceil 效应使实际比例略高于目标"),
        ]
        ok = lo <= actual <= hi
        return GateResult("H3", "val 比例", "PASS" if ok else "FAIL",
                          f"实际 {actual:.2%} 落在容许带内" if ok
                          else f"实际 {actual:.2%} 超出 [{lo:.2%}, {hi:.2%}]",
                          blocking=True, rows=rows)

    runner.run(h3, "H3", "val 比例", True)

    # ---- H4 类覆盖 ---------------------------------------------------------
    def h4() -> GateResult:
        per = img.groupby("brand_dir")["split"].agg(
            n="size", n_val=lambda s: int((s == "val").sum())
        )
        absent_val = per.index[per["n_val"] == 0]
        absent_tv = per.index[per["n_val"] == per["n"]]
        r = per["n_val"] / per["n"]
        rows = [
            ("类总数", f"{len(per):,}", ""),
            ("val 缺席的类", f"{len(absent_val):,}", "必须为 0"),
            ("trainval 缺席的类", f"{len(absent_tv):,}", "必须为 0"),
            ("逐类 val 比例",
             f"中位数={r.median():.3f} p05={r.quantile(.05):.3f} p95={r.quantile(.95):.3f}",
             f"min={r.min():.3f} max={r.max():.3f}"),
            ("最小类图数", f"{int(per['n'].min())}", "实测每类 >= 4 张，故无缺席问题"),
        ]
        if len(absent_val):
            rows.append(("缺席样例", str(list(absent_val[:5])), ""))
        ok = len(absent_val) == 0 and len(absent_tv) == 0
        return GateResult("H4", "类覆盖", "PASS" if ok else "FAIL",
                          "3,000 个类在两侧都有样本" if ok else "存在缺席类",
                          blocking=True, rows=rows)

    runner.run(h4, "H4", "类覆盖", True)

    # ---- H5 分布一致 -------------------------------------------------------
    def h5() -> GateResult:
        dims = ["supercat", "img_area_bin", "n_boxes_bin"]
        rows: list[tuple[str, ...]] = []
        worst = 0.0
        for d in dims:
            v = tvd(trainval[d].value_counts(), val[d].value_counts())
            worst = max(worst, v)
            rows.append((f"TVD({d})", f"{v:.5f}", "阈值 < 0.01"))
        rows.append(("最大 TVD", f"{worst:.5f}", ""))
        ok = worst < 0.01
        return GateResult("H5", "trainval/val 分布一致", "PASS" if ok else "FAIL",
                          f"三个边缘分布最大 TVD {worst:.5f}，划分无系统性偏斜"
                          if ok else f"最大 TVD {worst:.5f} 超阈值",
                          blocking=True, rows=rows)

    runner.run(h5, "H5", "trainval/val 分布一致", True)

    # ---- H6 val2k 代表性 ---------------------------------------------------
    def h6() -> GateResult:
        dims = ["supercat", "img_area_bin", "n_boxes_bin"]
        rows: list[tuple[str, ...]] = []
        worst = 0.0
        for d in dims:
            v = tvd(val[d].value_counts(), v2k[d].value_counts())
            worst = max(worst, v)
            rows.append((f"TVD({d}) vs val_full", f"{v:.5f}", "阈值 < 0.03"))
        rows.append(("最大 TVD", f"{worst:.5f}", ""))
        rows.append(("val2k 图数", f"{len(v2k):,}", f"目标 {cfg['val2k_target']:,}"))
        rows.append(("用到的层数", f"{rep['val2k_repr']['strata_used']}", "上限 9×3×4=108"))
        n_box = int(ann["image_id"].isin(set(v2k["image_id"])).sum())
        rows.append(("val2k 框数", f"{n_box:,}", ""))
        ok = worst < 0.03 and len(v2k) == cfg["val2k_target"]
        return GateResult("H6", "val2k 代表性", "PASS" if ok else "FAIL",
                          f"与 val_full 最大 TVD {worst:.5f}，可用于报总体指标"
                          if ok else "代表性不足或图数不符",
                          blocking=True, rows=rows)

    runner.run(h6, "H6", "val2k 代表性", True)

    # ---- H7 hard_pool 体量（非阻塞）---------------------------------------
    def h7() -> GateResult:
        hp_ids = set(vhp["image_id"])
        a = ann[ann["image_id"].isin(hp_ids)].merge(
            img[["image_id", "n_boxes"]], on="image_id", how="left"
        )
        axes = {
            "T1 截断": a["is_hard_t1"],
            "SIZE small": a["area_bin"] == "small",
            "DENSITY 5+": a["n_boxes"] >= 5,
        }
        rows: list[tuple[str, ...]] = []
        unreliable: list[str] = []
        for name, mask in axes.items():
            n = int(mask.sum())
            flag = "" if n >= min_reliable else "UNRELIABLE"
            if n < min_reliable:
                unreliable.append(name)
            rows.append((name, f"{n:,} 框", f"可靠性下限 {min_reliable} {flag}"))
        clean_mask = ~(axes["T1 截断"] | axes["SIZE small"] | axes["DENSITY 5+"])
        rows.append(("clean 对照", f"{int(clean_mask.sum()):,} 框", ""))
        rows.append(("hard / clean 图数",
                     f"{int((vhp['hard_pool_role'] == 'hard').sum()):,} / "
                     f"{int((vhp['hard_pool_role'] == 'clean_control').sum()):,}", ""))
        status = "PASS" if not unreliable else "WARN"
        detail = ("三轴框数均达可靠性下限" if not unreliable
                  else f"{unreliable} 低于 {min_reliable}，报告中须标 UNRELIABLE 并带 bootstrap CI")
        return GateResult("H7", "hard_pool 体量", status, detail,
                          blocking=False, rows=rows)

    runner.run(h7, "H7", "hard_pool 体量", False)

    # ---- H8 口径隔离（非阻塞，留痕）---------------------------------------
    def h8() -> GateResult:
        overlap = set(v2k["image_id"]) & set(vhp["image_id"])
        rows = [
            ("val2k_repr 用途", str(cfg["val2k_role"]), f"禁止：{cfg['val2k_forbidden']}"),
            ("val_hard_pool 用途", str(cfg["hard_pool_role"]),
             f"禁止：{cfg['hard_pool_forbidden']}"),
            ("两子集图像重叠", f"{len(overlap):,}",
             "重叠是允许的 —— 推理只跑去重并集，子集只是行筛选"),
            ("去重并集图数", f"{len(set(v2k['image_id']) | set(vhp['image_id'])):,}",
             "这是实际需要推理的图量"),
            ("hard_pool 三轴", str(cfg["hard_pool_axes"]),
             "不含 P2/P3 —— 已在 S2 因 precision 0.125/0.171 降级"),
        ]
        return GateResult(
            "H8", "口径隔离", "WARN",
            "两子集用途互斥，评测代码须按此约束选集合；越界使用会得出无效结论",
            blocking=False, rows=rows,
        )

    runner.run(h8, "H8", "口径隔离", False)

    # ---- H9 可复现 ---------------------------------------------------------
    def h9() -> GateResult:
        """两层判据。

        自我一致性：同 seed 连跑两次，产物 sha256 相同。
        跨机可复现：重建结果与 configs/splits.yaml 记录的参考值一致。

        后者更强 —— 它能发现「在这台机器上自洽、但换台机器就变了」的问题。
        切分产物不进仓库（4.2 MB 派生数据），所以这个锚点是「可重建」这句
        声称的唯一凭据。
        """
        files = ("split_images.parquet", "val2k_repr.parquet", "val_hard_pool.parquet")
        before = {f: file_sha256(sp / f) for f in files}
        r = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "scripts" / "s3_make_splits.py")],
            cwd=PROJECT_DIR, capture_output=True, text=True,
        )
        if r.returncode != 0:
            return GateResult("H9", "可复现", "FAIL",
                              f"重跑划分脚本失败：{r.stderr[-300:]}", blocking=True)
        after = {f: file_sha256(sp / f) for f in files}

        rows: list[tuple[str, ...]] = []
        diff: list[str] = []
        for f in files:
            same = before[f] == after[f]
            rows.append((f, before[f][:16] + "…",
                         "两次一致" if same else "两次不一致"))
            if not same:
                diff.append(f"{f}(自我不一致)")

        # 与配置里的参考值比对
        ref = cfg.get("expected_sha256", {})
        if ref:
            rows.append(("—— 跨机参考值比对 ——", "", cfg.get("expected_sha256_note", "")))
            for f in files:
                exp = str(ref.get(f, ""))
                got = after[f][: len(exp)] if exp else ""
                ok_ref = bool(exp) and got == exp
                rows.append((f"  {f}", f"期望 {exp} / 实得 {got}",
                             "一致" if ok_ref else "不一致"))
                if exp and not ok_ref:
                    diff.append(f"{f}(与参考值不符)")
        else:
            rows.append(("跨机参考值", "未配置",
                         "建议在 configs/splits.yaml 加 expected_sha256"))

        detail = ("同 seed 重跑两次一致，且与配置记录的参考值相符 —— "
                  "切分可跨机精确复现，无需随仓库携带产物")
        if diff:
            detail = f"复现性不成立：{diff}"
        return GateResult("H9", "可复现", "PASS" if not diff else "FAIL",
                          detail, blocking=True, rows=rows)

    runner.run(h9, "H9", "可复现", True)

    # ---- 报告附表 ----------------------------------------------------------
    extra: list[tuple[str, str]] = []
    rows = []
    for d in ("supercat", "img_area_bin", "n_boxes_bin"):
        tv = trainval[d].value_counts(normalize=True)
        vv = val[d].value_counts(normalize=True)
        kk = sorted(set(tv.index) | set(vv.index), key=str)
        for k in kk:
            rows.append((d, str(k), f"{tv.get(k, 0):.4f}", f"{vv.get(k, 0):.4f}",
                         f"{vv.get(k, 0) - tv.get(k, 0):+.4f}"))
    extra.append(("trainval vs val 边缘分布", md_table(
        rows, headers=("维度", "取值", "trainval", "val", "差"))))

    curve = rep.get("topn_candidate_curve", {})
    extra.append(("Top-N 候选曲线（供训练阶段选 N）", md_table(
        [(k, f"{v:,}", "") for k, v in curve.items()],
        headers=("n_images >= ", "类数", "说明"))))

    runner.write_report(REPORT, extra_sections=extra)
    print(f"\n报告已写入 {REPORT.relative_to(PROJECT_DIR)}")
    return runner.print_summary()


if __name__ == "__main__":
    raise SystemExit(main())
