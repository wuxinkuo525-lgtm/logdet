#!/usr/bin/env python
"""S7 步骤二：评测两个 baseline，出最终报告。

### 三个口径的硬约束（不靠自觉，靠代码拦住）

**M1 L1 只报 AR，禁止报 AP。**
RPN 的 objectness 不是校准过的检测置信度，不同图之间不可比较。
AP 需要把所有图的预测放在一起做全局排序 —— 用不可比的分数排序，
算出来的数没有意义。所以 L1 的结果字典里**物理上不含 AP 键**。

**M2 L3-strict 负控制：AP 必须精确为 0。**
同样的框，但 category_id=2（GT 里只有 1）。若 AP > 0，说明评测器在
跨类别匹配，那么"类塌缩"的结果就是假的。这条是对 L3 数字可信度的
唯一独立验证。

**M3 两个子集用途互斥。**
val2k_repr 报总体（可外推）、val_hard_pool 报切片（不可外推）。
S3-H8 已留痕，这里再次用代码分开。

用法：
    python scripts/s7_evaluate.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.config import load_yaml  # noqa: E402
from logodet.data.adapters.to_coco_json import build_coco_dt  # noqa: E402
from logodet.eval.coco_eval import DEFAULT_MAX_DETS, evaluate  # noqa: E402
from logodet.eval.slice_eval import (  # noqa: E402
    default_slices,
    evaluate_slices,
    slice_members,
    slices_to_frame,
)
from logodet.gates import GateResult, GateRunner, md_table  # noqa: E402
from logodet.paths import P  # noqa: E402

REPORT = PROJECT_DIR / "report" / "s7_baselines_report.md"

# L1 只允许出现这些键 —— AP 类指标物理上不产出
AR_ONLY_KEYS = ("AR@1", "AR@10", "AR@100", "AR@300",
                "AR_small@300", "AR_medium@300", "AR_large@300", "AR50@300")


def _as_coco(d: dict) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO()
        c.dataset = d
        c.createIndex()
    return c


def _load_gt(name: str) -> tuple[COCO, dict]:
    raw = json.loads(
        (P.artifact("eval") / "gt" / f"{name}.json").read_text(encoding="utf-8")
    )
    return _as_coco(raw), raw


def main() -> int:
    cfg = load_yaml("splits.yaml")
    min_reliable = int(cfg.get("min_ann_for_reliable", 200))
    pred_dir = P.artifact("predictions")

    for f in ("L1_rpn_proposals.parquet", "L3_coco_collapsed.parquet"):
        if not (pred_dir / f).is_file():
            print(f"FAIL: 缺 {f}，先跑 python scripts/s7_run_baselines.py")
            return 1

    p1 = pd.read_parquet(pred_dir / "L1_rpn_proposals.parquet")
    p3 = pd.read_parquet(pred_dir / "L3_coco_collapsed.parquet")
    # L2 OWLv2 是增量 baseline，可能还没跑
    p2_path = pred_dir / "L2_owlv2_zeroshot.parquet"
    p2 = pd.read_parquet(p2_path) if p2_path.is_file() else None
    man2 = None
    m2s = sorted(pred_dir.glob("manifest_owlv2_*.json"))
    if m2s:
        man2 = json.loads(m2s[-1].read_text(encoding="utf-8"))
    manifests = sorted(pred_dir.glob("manifest_2*.json"))
    man = json.loads(manifests[-1].read_text(encoding="utf-8")) if manifests else {}

    gt2k, raw2k = _load_gt("val2k_repr")
    gthp, rawhp = _load_gt("val_hard_pool")
    ids2k = sorted(int(i["id"]) for i in raw2k["images"])
    idshp = sorted(int(i["id"]) for i in rawhp["images"])

    print("=" * 78)
    print(" S7 baseline 评测")
    print("=" * 78)
    print(f"\nrun_id {man.get('run_id', '?')}  device_mode {man.get('device_mode', '?')}")
    print(f"推理 {man.get('n_images', '?')} 图 / {man.get('img_per_sec', '?')} img/s")
    print(f"L1 proposals  {len(p1):,} 行")
    print(f"L3 detections {len(p3):,} 行")
    if p2 is not None:
        print(f"L2 owlv2      {len(p2):,} 行  prompt={man2.get('queries') if man2 else '?'}")
    else:
        print("L2 owlv2      未跑（增量 baseline）")

    runner = GateRunner("S7", "S7 baseline 结果报告")

    def sub(df: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
        return df[df["image_id"].isin(set(ids))]

    # 预先算好，各门只查表
    dt1_2k = build_coco_dt(sub(p1, ids2k))
    dt3_2k = build_coco_dt(sub(p3, ids2k))
    r1 = evaluate(gt2k, dt1_2k, image_ids=ids2k, max_dets=DEFAULT_MAX_DETS)
    r3 = evaluate(gt2k, dt3_2k, image_ids=ids2k, max_dets=DEFAULT_MAX_DETS)
    r2 = None
    if p2 is not None:
        r2 = evaluate(gt2k, build_coco_dt(sub(p2, ids2k)), image_ids=ids2k,
                      max_dets=DEFAULT_MAX_DETS)

    # ---- M1 L1 只报 AR -----------------------------------------------------
    def m1() -> GateResult:
        ar = {k: v for k, v in r1.metrics.items() if k.startswith("AR")}
        rows = [(k, f"{v:.4f}", "") for k, v in sorted(ar.items())]
        rows.append(("被抑制的 AP 类指标",
                     ", ".join(k for k in r1.metrics if k.startswith("AP")),
                     "objectness 不可跨图比较 → 报出来是无意义的数"))
        rows.append(("proposals/图", f"中位数 {man.get('proposals', {}).get('per_image_median', '?')}",
                     f"最大 {man.get('proposals', {}).get('per_image_max', '?')}"))
        ok = bool(ar) and all(k in r1.metrics for k in ("AR@100", "AR@300"))
        return GateResult(
            "M1", "L1 只报 AR", "PASS" if ok else "FAIL",
            f"AR@100={ar.get('AR@100', float('nan')):.4f}  "
            f"AR@300={ar.get('AR@300', float('nan')):.4f}"
            f"（AP 类指标已按口径抑制，不进报告）" if ok else "AR 指标缺失",
            blocking=True, rows=rows,
        )

    runner.run(m1, "M1", "L1 只报 AR", True)

    # ---- M2 L3-strict 负控制 ----------------------------------------------
    def m2() -> GateResult:
        wrong = sub(p3, ids2k).copy()
        wrong["category_id"] = 2  # GT 里只有 1
        r = evaluate(gt2k, build_coco_dt(wrong), image_ids=ids2k,
                     max_dets=DEFAULT_MAX_DETS)
        rows = [
            ("构造", "同样的框，category_id 改为 2", "GT 里只有 category_id=1"),
            ("AP", f"{r.metrics['AP']:.6f}", "必须精确为 0"),
            ("AP50", f"{r.metrics['AP50']:.6f}", "必须精确为 0"),
            ("AR@100", f"{r.metrics.get('AR@100', float('nan')):.6f}", "必须精确为 0"),
            ("对照：正确类别的 AP", f"{r3.metrics['AP']:.4f}", "非 0 才说明匹配确实在发生"),
            ("意义", "验证评测器不跨类别匹配",
             "若这条不为 0，L3 的类塌缩结果就是假的"),
        ]
        ok = r.metrics["AP"] == 0.0 and r.metrics["AP50"] == 0.0
        return GateResult(
            "M2", "L3-strict 负控制", "PASS" if ok else "FAIL",
            "错类别的 AP 精确为 0，评测器不跨类别匹配 → L3 的数字可信" if ok
            else f"错类别仍得 AP={r.metrics['AP']:.6f}，评测器在跨类别匹配",
            blocking=True, rows=rows,
        )

    runner.run(m2, "M2", "L3-strict 负控制", True)

    # ---- M3 总体指标（val2k_repr）-----------------------------------------
    def m3() -> GateResult:
        rows = [("子集", "val2k_repr", "代表性抽样，可外推总体")]
        for k in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"):
            line = f"L3={r3.metrics.get(k, float('nan')):.4f}"
            if r2:
                line += f"   L2={r2.metrics.get(k, float('nan')):.4f}"
            rows.append((k, line, ""))
        for k in ("AR@1", "AR@10", "AR@100", "AR@300"):
            line = f"L3={r3.metrics.get(k, float('nan')):.4f}"
            if r2:
                line += f"   L2={r2.metrics.get(k, float('nan')):.4f}"
            line += f"   L1={r1.metrics.get(k, float('nan')):.4f}"
            rows.append((k, line, ""))
        rows.append(("L1 AR50@300", f"{r1.metrics.get('AR50@300', float('nan')):.4f}",
                     "IoU=0.5 下的召回 —— 通用 objectness 能「看到」多少 logo"))
        # L1 的 AR 应当 >= L3 的 AR：proposals 是 detections 的上游
        gap = r1.metrics.get("AR@300", 0) - r3.metrics.get("AR@300", 0)
        rows.append(("AR@300 差 (L1 − L3)", f"{gap:+.4f}",
                     "应为正 —— proposals 是 roi_heads 的输入，召回是上界"))
        if r2:
            ratio = r2.metrics["AP"] / max(r3.metrics["AP"], 1e-9)
            rows.append(("L2 / L3 的 AP 倍率", f"{ratio:.1f}x",
                         "开放词表 vs 类塌缩 —— 前者真的「懂」logo"))
        ok = np.isfinite(r3.metrics.get("AP", float("nan")))
        detail = (f"L3 类塌缩 AP={r3.metrics['AP']:.4f}；"
                  f"L1 proposals AR@300={r1.metrics.get('AR@300', float('nan')):.4f}")
        if r2:
            detail = (f"L2 OWLv2 AP={r2.metrics['AP']:.4f} "
                      f"AP50={r2.metrics['AP50']:.4f}；" + detail)
        return GateResult("M3", "总体指标", "PASS" if ok else "FAIL",
                          detail if ok else "总体指标计算失败",
                          blocking=True, rows=rows)

    runner.run(m3, "M3", "总体指标", True)

    # ---- M4 切片对比（val_hard_pool）--------------------------------------
    ann = pd.read_parquet(P.artifact("tables") / "annotations.parquet")
    images = pd.read_parquet(P.artifact("splits") / "split_images.parquet")
    keep_ann = {int(a["id"]) for a in rawhp["annotations"]}
    specs = default_slices()
    members = slice_members(
        ann[ann["ann_id"].isin(keep_ann)],
        images[images["image_id"].isin(set(idshp))],
        specs,
    )
    sl3 = evaluate_slices(rawhp, build_coco_dt(sub(p3, idshp)), members, specs,
                          min_ann_for_reliable=min_reliable,
                          ci_metrics=("AP50",), n_boot=60)
    sl1 = evaluate_slices(rawhp, build_coco_dt(sub(p1, idshp)), members, specs,
                          min_ann_for_reliable=min_reliable,
                          ci_metrics=("AR@300",), n_boot=60)
    sl2 = None
    if p2 is not None:
        sl2 = evaluate_slices(rawhp, build_coco_dt(sub(p2, idshp)), members, specs,
                              min_ann_for_reliable=min_reliable,
                              ci_metrics=("AP50",), n_boot=60)

    def m4() -> GateResult:
        # 主 baseline 优先用 L2（真正有信号的那个），没跑时退回 L3
        primary = sl2 if sl2 is not None else sl3
        pname = "L2 OWLv2" if sl2 is not None else "L3 类塌缩"
        m3d = {r.name: r.metrics for r in primary}
        rows: list[tuple[str, ...]] = [("主 baseline", pname, "切片对比以它为准")]
        for r in primary:
            mark = "" if r.reliable else " UNRELIABLE"
            rows.append((f"{r.name}{mark}",
                         f"AP50={r.metrics.get('AP50', float('nan')):.4f}",
                         f"n_ann={r.n_ann:,}"))
        clean = m3d.get("CLEAN", {}).get("AP50", float("nan"))
        t1 = m3d.get("T1_truncated", {}).get("AP50", float("nan"))
        rows.append(("CLEAN − T1_truncated", f"{clean - t1:+.4f}",
                     "⚠️ 该对比被尺寸混淆，见下方受控对比"))
        big = m3d.get("SIZE_large", {}).get("AP50", float("nan"))
        big_no = m3d.get("SIZE_large_noT1", {}).get("AP50", float("nan"))
        rows.append(("SIZE_large → _noT1", f"{big:.4f} → {big_no:.4f}",
                     f"剔除 T1 后变化 {big_no - big:+.4f}（S6 交叉污染警告）"))
        t1l = m3d.get("T1_large", {}).get("AP50", float("nan"))
        rows.append(("受控对比（均为大目标）",
                     f"T1_large={t1l:.4f} vs SIZE_large_noT1={big_no:.4f}",
                     f"差 {t1l - big_no:+.4f} —— 这才是「截断难不难」的答案"))
        rows.append(("尺寸主导性",
                     f"small={m3d.get('SIZE_small', {}).get('AP50', float('nan')):.4f} "
                     f"medium={m3d.get('SIZE_medium', {}).get('AP50', float('nan')):.4f} "
                     f"large={big:.4f}",
                     "尺寸的影响远大于其它任何维度"))
        ok = all(np.isfinite(r.metrics.get("AP50", float("nan")))
                 for r in primary if r.n_ann)
        return GateResult("M4", "切片对比", "PASS" if ok else "FAIL",
                          f"[{pname}] 受控对比 T1_large={t1l:.4f} vs "
                          f"SIZE_large_noT1={big_no:.4f}（差 {t1l - big_no:+.4f}）；"
                          f"尺寸 small {m3d.get('SIZE_small', {}).get('AP50', 0):.4f} "
                          f"→ large {big:.4f}" if ok else "存在切片计算失败",
                          blocking=True, rows=rows)

    runner.run(m4, "M4", "切片对比", True)

    # ---- 落盘 --------------------------------------------------------------
    mdir = P.artifact("metrics")
    ov_rows = [
        {"baseline": "L1_rpn_proposals", "subset": "val2k_repr",
         **{k: v for k, v in r1.metrics.items() if k.startswith("AR")}},
        {"baseline": "L3_coco_collapsed", "subset": "val2k_repr", **r3.metrics},
    ]
    if r2:
        ov_rows.insert(1, {"baseline": "L2_owlv2_zeroshot",
                           "subset": "val2k_repr", **r2.metrics})
    pd.DataFrame(ov_rows).to_parquet(mdir / "s7_overall.parquet", index=False)

    frames = []
    for res, name in ((sl3, "L3_coco_collapsed"), (sl1, "L1_rpn_proposals")):
        f = slices_to_frame(res)
        f["baseline"] = name
        frames.append(f)
    if sl2 is not None:
        f = slices_to_frame(sl2)
        f["baseline"] = "L2_owlv2_zeroshot"
        frames.append(f)
    pd.concat(frames, ignore_index=True).to_parquet(
        mdir / "s7_slices.parquet", index=False
    )

    # ---- 报告附表 ----------------------------------------------------------
    extra: list[tuple[str, str]] = []
    ov_tbl = []
    if r2:
        ov_tbl.append((
            "**L2 OWLv2 zero-shot**", f"{r2.metrics['AP']:.4f}",
            f"{r2.metrics['AP50']:.4f}", f"{r2.metrics['AP75']:.4f}",
            f"{r2.metrics.get('AP_small', float('nan')):.4f}",
            f"{r2.metrics.get('AP_medium', float('nan')):.4f}",
            f"{r2.metrics.get('AP_large', float('nan')):.4f}",
            f"{r2.metrics.get('AR@100', float('nan')):.4f}",
            f"{r2.metrics.get('AR@300', float('nan')):.4f}"))
    ov_tbl += [
        ("L3 COCO 类塌缩", f"{r3.metrics['AP']:.4f}", f"{r3.metrics['AP50']:.4f}",
         f"{r3.metrics['AP75']:.4f}", f"{r3.metrics.get('AP_small', float('nan')):.4f}",
         f"{r3.metrics.get('AP_medium', float('nan')):.4f}",
         f"{r3.metrics.get('AP_large', float('nan')):.4f}",
         f"{r3.metrics.get('AR@100', float('nan')):.4f}",
         f"{r3.metrics.get('AR@300', float('nan')):.4f}"),
        ("L1 RPN proposals", "—", "—", "—", "—", "—", "—",
         f"{r1.metrics.get('AR@100', float('nan')):.4f}",
         f"{r1.metrics.get('AR@300', float('nan')):.4f}"),
    ]
    extra.append(("总体指标（val2k_repr，2,000 图 / 2,451 框）", md_table(
        ov_tbl, headers=("baseline", "AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l",
                         "AR@100", "AR@300"))))
    extra.append(("L1 的 AP 为何标「—」", "\n".join([
        "RPN 的 objectness **不是校准过的检测置信度**，不同图之间不可比较。",
        "AP 需要把所有图的预测放在一起做全局排序 —— 用不可比的分数排序，",
        "算出来的数字没有意义。所以 L1 只报 AR@K（每图各自看 top-K 召回）。",
        "这一点由 M1 门用代码保证，不靠自觉。",
        "",
        f"L1 的 `AR50@300 = {r1.metrics.get('AR50@300', float('nan')):.4f}` "
        f"vs `AR@300 = {r1.metrics.get('AR@300', float('nan')):.4f}`（IoU 0.5:0.95 平均）"
        f"—— 差值说明通用 objectness 能「看到」logo，但框得松。",
    ])))

    rows = []
    m1d = {r.name: r for r in sl1}
    m2d = {r.name: r for r in (sl2 or [])}
    for r in sl3:
        ci = r.ci.get("AP50")
        p = m1d.get(r.name)
        q = m2d.get(r.name)
        rows.append((
            r.name, f"{r.n_ann:,}", "是" if r.reliable else "**否**",
            f"{q.metrics.get('AP50', float('nan')):.4f}" if q else "—",
            f"{r.metrics.get('AP50', float('nan')):.4f}",
            f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—",
            f"{p.metrics.get('AR@300', float('nan')):.4f}" if p else "—",
        ))
    extra.append(("逐切片结果（val_hard_pool，1,230 图 / 2,099 框）", md_table(
        rows, headers=("切片", "框数", "可靠", "L2 AP50", "L3 AP50",
                       "L3 AP50 CI", "L1 AR@300"))))

    prompt_note = []
    if man2:
        prompt_note = [
            f"- L2 的文本 prompt = `{man2.get('queries')}`，"
            f"在 **trainval** {man2.get('prompt_selection', {}).get('n_images')} 图上按 AP50 选定。",
            "  **prompt 是超参**，在 val 上选再报 val 就是数据泄漏，所以选型只用 trainval。",
            "- OWLv2 先把图 pad 成正方形再 resize 960，`target_sizes` 必须传 "
            "`max(H,W)` 而非 `(H,W)`；",
            "  传错会让框系统性错位，而错位的表现（AP≈0）与「模型找不到」无法区分。",
            "  已用合成图独立验证：三张非正方形图的框中心偏移 ≤ 2.3px。",
        ]
    extra.append(("口径声明（必须随数字一起引用）", "\n".join([
        "- **总体指标只用 val2k_repr**（代表性抽样，可外推）；"
        "**切片指标只用 val_hard_pool**（已人为富集，不可外推总体）。",
        "- 尺寸切片用**切片视图口径**（非成员 GT 置 iscrowd），"
        "与 COCO 原生 `AP_small` 不同：后者同时过滤 GT 与 DT。",
        "- **报 `SIZE_large` 必须同时看 `SIZE_large_noT1`**，"
        "**报 `T1` 必须看 `T1_large`** —— T1 与 SIZE_large 重叠 37.5%，"
        "不拆开会把截断问题误读成大目标问题。",
        "- L3 是 **COCO 预训练 + 类塌缩**，**未在 LogoDet-3K 上训练**，"
        "是「零样本迁移下限」，不是「模型能力上限」。",
        "- L2 OWLv2 同样**未在 LogoDet-3K 上训练**，是真正的 zero-shot。",
        *prompt_note,
        f"- 推理设备：L1/L3 用 `{man.get('device_mode', '?')}`"
        f"（hybrid 已验数值一致，max_box_diff 0.004px）"
        + (f"，L2 用 `{man2.get('device')}`" if man2 else ""),
    ])))

    runner.write_report(REPORT, extra_sections=extra)
    print(f"\n报告已写入 {REPORT.relative_to(PROJECT_DIR)}")
    print(f"指标已落盘 {mdir.relative_to(P.artifacts)}/s7_overall.parquet, s7_slices.parquet")
    return runner.print_summary()


if __name__ == "__main__":
    raise SystemExit(main())
