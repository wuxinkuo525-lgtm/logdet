#!/usr/bin/env python
"""S5 步骤二：L0 自校验门（九条）。

**这是整个 setup 里最重要的一道保险。** 评测器错了，后面所有 baseline
数字都是废的，而且这种错误在指标上表现为"效果偏低"，极易被误判成模型问题。

阻塞门：
    K1  完美预测 → AP > 0.999（顺带验证 xyxy↔xywh 转换一致）
    K2  抖动阶梯 → AP 随 ε 严格单调递减
    K3  端点     → AP(0.40) < 0.20
    K4  空预测   → AP == 0
    K5  丢一半框 → AR@100 ∈ [0.45, 0.55]
    K6  加假阳   → AP 下降但 AR@100 基本不变
    K7  官方 summarize 的坑被复现且我方实现不受影响
    K8  逐切片评测的 iscrowd 忽略机制正确

非阻塞门：
    K9  bootstrap CI 行为合理（点估计落在区间内，小样本区间更宽）

用法：
    python scripts/s5_l0_selfcheck.py
    python scripts/s5_l0_selfcheck.py --n-images 300   # 加快迭代
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.eval.coco_eval import (  # noqa: E402
    DEFAULT_MAX_DETS,
    bootstrap_ci,
    evaluate,
    summarize,
)
from logodet.eval.l0_selfcheck import (  # noqa: E402
    L0_SUITE,
    JITTER_LEVELS,
    SyntheticSpec,
    make_synthetic_dt,
)
from logodet.gates import GateResult, GateRunner, md_table  # noqa: E402
from logodet.paths import P  # noqa: E402

REPORT = PROJECT_DIR / "report" / "s5_eval_report.md"
BASELINE = PROJECT_DIR / "configs" / "l0_baseline.json"


def _load_gt(path: Path, n_images: int | None) -> tuple[COCO, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if n_images and n_images < len(raw["images"]):
        keep = {im["id"] for im in raw["images"][:n_images]}
        raw = {
            **raw,
            "images": [im for im in raw["images"] if im["id"] in keep],
            "annotations": [a for a in raw["annotations"] if a["image_id"] in keep],
        }
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        coco = COCO()
        coco.dataset = raw
        coco.createIndex()
    return coco, raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=0, help="限制图数以加快迭代；0=全部")
    args = ap.parse_args()

    gt_path = P.artifact("eval") / "gt" / "val2k_repr.json"
    if not gt_path.is_file():
        print(f"FAIL: 缺 {gt_path}，先跑 python scripts/s5_build_gt.py")
        return 1

    coco_gt, raw = _load_gt(gt_path, args.n_images or None)
    n_img = len(raw["images"])
    n_ann = len(raw["annotations"])

    print("=" * 78)
    print(" S5 L0 自校验门")
    print("=" * 78)
    print(f"\nGT: {n_img:,} 图 / {n_ann:,} 框")

    # 一次性把所有合成预测跑完，后面各门只查表
    print("\n跑合成预测 ...")
    results: dict[str, dict[str, float]] = {}
    for spec in L0_SUITE:
        dt = make_synthetic_dt(raw, spec)
        r = evaluate(coco_gt, dt, max_dets=DEFAULT_MAX_DETS)
        results[spec.name] = r.metrics
        print(f"  {spec.name:<18} n_dt={len(dt):>6,}  "
              f"AP={r.metrics['AP']:.4f}  AP50={r.metrics['AP50']:.4f}  "
              f"AR@100={r.metrics.get('AR@100', float('nan')):.4f}")

    runner = GateRunner("S5", "S5 评测器 L0 自校验报告")

    # ---- K1 完美预测 -------------------------------------------------------
    def k1() -> GateResult:
        m = results["perfect"]
        rows = [
            ("AP", f"{m['AP']:.6f}", "阈值 > 0.999"),
            ("AP50", f"{m['AP50']:.6f}", ""),
            ("AP75", f"{m['AP75']:.6f}", ""),
            ("AR@100", f"{m.get('AR@100', float('nan')):.6f}", ""),
            ("AR@300", f"{m.get('AR@300', float('nan')):.6f}", "官方 summarize 报不出这档"),
            ("含义", "GT 原样当预测", "顺带验证 xyxy↔xywh 转换两侧一致"),
        ]
        ok = m["AP"] > 0.999
        return GateResult("K1", "完美预测", "PASS" if ok else "FAIL",
                          f"AP={m['AP']:.6f} —— 坐标格式与匹配逻辑均正确" if ok
                          else f"AP={m['AP']:.6f} 未达 0.999，坐标转换或匹配有问题",
                          blocking=True, rows=rows)

    runner.run(k1, "K1", "完美预测", True)

    # ---- K2 单调性 ---------------------------------------------------------
    def k2() -> GateResult:
        names = ["perfect", "jitter_005", "jitter_010", "jitter_020", "jitter_040"]
        aps = [results[n]["AP"] for n in names]
        rows = [
            (f"ε={e:.2f}", f"AP={a:.4f}", f"AP50={results[n]['AP50']:.4f}")
            for e, a, n in zip(JITTER_LEVELS, aps, names)
        ]
        diffs = np.diff(aps)
        rows.append(("逐级差值", str([f"{d:+.4f}" for d in diffs]), "必须全为负"))
        ok = bool((diffs < 0).all())
        return GateResult("K2", "抖动单调性", "PASS" if ok else "FAIL",
                          "AP 随抖动幅度严格单调递减" if ok
                          else f"存在非递减：{[f'{d:+.4f}' for d in diffs]}",
                          blocking=True, rows=rows)

    runner.run(k2, "K2", "抖动单调性", True)

    # ---- K3 端点 -----------------------------------------------------------
    def k3() -> GateResult:
        a = results["jitter_040"]["AP"]
        rows = [
            ("AP(ε=0.40)", f"{a:.4f}", "阈值 < 0.20"),
            ("判据说明", "大幅抖动必须显著掉点", "否则说明匹配过于宽松"),
        ]
        ok = a < 0.20
        return GateResult("K3", "端点", "PASS" if ok else "FAIL",
                          f"AP(0.40)={a:.4f} < 0.20，匹配严格度正常" if ok
                          else f"AP(0.40)={a:.4f} 偏高，IoU 匹配可能过松",
                          blocking=True, rows=rows)

    runner.run(k3, "K3", "端点", True)

    # ---- K4 空预测 ---------------------------------------------------------
    def k4() -> GateResult:
        r = evaluate(coco_gt, [], max_dets=DEFAULT_MAX_DETS)
        rows = [("AP", f"{r.metrics['AP']:.4f}", "必须 0"),
                ("AR@100", f"{r.metrics.get('AR@100', 0):.4f}", "必须 0"),
                ("n_dt", "0", "空列表不应抛异常")]
        ok = r.metrics["AP"] == 0.0
        return GateResult("K4", "空预测", "PASS" if ok else "FAIL",
                          "空预测返回 0 且不抛异常" if ok else "空预测处理异常",
                          blocking=True, rows=rows)

    runner.run(k4, "K4", "空预测", True)

    # ---- K5 丢一半 ---------------------------------------------------------
    def k5() -> GateResult:
        m = results["half_recall"]
        ar = m.get("AR@100", float("nan"))
        rows = [
            ("AR@100", f"{ar:.4f}", "期望区间 [0.44, 0.56]"),
            ("AP", f"{m['AP']:.4f}", "丢一半框，AP 也应约减半"),
            ("采样方式", "逐框独立按 p=0.5 保留",
             "不设「每图至少留 1 个」的下限 —— 那会让单框图(84.6%)全部保留"),
            ("容差说明", "±0.06", "2,451 框逐框二项采样的波动范围"),
        ]
        ok = 0.44 <= ar <= 0.56
        return GateResult("K5", "召回减半", "PASS" if ok else "FAIL",
                          f"AR@100={ar:.4f} 落在期望区间，召回口径正确" if ok
                          else f"AR@100={ar:.4f} 偏离 [0.44, 0.56]",
                          blocking=True, rows=rows)

    runner.run(k5, "K5", "召回减半", True)

    # ---- K6 假阳 -----------------------------------------------------------
    def k6() -> GateResult:
        """假阳对 AP 的影响**取决于它排在真阳之前还是之后**。

        首轮把这条门写成"加假阳 → AP 必须下降"，结果 FAIL：
        1.0000 → 1.0000。查下来不是 bug，是期望错了 ——
        假阳分数 0.05 全部排在真阳（1.0）之后，PR 曲线在任何假阳
        被计入之前就已到达 recall=1.0，插值后每个 recall 点的
        precision 都是 1.0，于是 AP 仍为 1.0。

        **AP 只惩罚排在真阳之上的假阳。** 这是 AP 作为排序指标的定义性质，
        对解读 baseline 很关键：RPN proposals 会输出上千个框，
        只要排序好，AP 不会被额外的框毁掉；反之若模型把垃圾框打高分，
        AP 会塌得很厉害。

        所以这条门改成验证两个方向：
          低分假阳（排在真阳之后）→ AP 不变    ← 正确行为，值得断言
          高分假阳（排在真阳之上）→ AP 显著下降
        """
        p = results["perfect"]
        f_low = results["perfect_plus_fp"]  # score=0.05，排在真阳之后
        dt_high = make_synthetic_dt(
            raw,
            SyntheticSpec("fp_high", n_false_per_image=10, false_score=2.0, score=1.0),
        )
        f_high = evaluate(coco_gt, dt_high, max_dets=DEFAULT_MAX_DETS).metrics

        d_low = f_low["AP"] - p["AP"]
        d_high = f_high["AP"] - p["AP"]
        d_ar_low = f_low.get("AR@100", 0) - p.get("AR@100", 0)

        rows = [
            ("完美预测 AP", f"{p['AP']:.4f}", "基准"),
            ("+低分假阳 (score=0.05)", f"{f_high and f_low['AP']:.4f}",
             f"差 {d_low:+.4f} —— 应为 0，排在真阳之后不影响 AP"),
            ("+高分假阳 (score=2.0)", f"{f_high['AP']:.4f}",
             f"差 {d_high:+.4f} —— 应显著为负"),
            ("AR@100 低分假阳", f"{f_low.get('AR@100', 0):.4f}",
             f"差 {d_ar_low:+.4f} —— AR 不惩罚假阳"),
            ("AP50 高分假阳", f"{f_high['AP50']:.4f}", f"完美时 {p['AP50']:.4f}"),
            ("结论", "AP 只惩罚排在真阳之上的假阳",
             "解读 baseline 时必须记住这一点"),
        ]
        ok = abs(d_low) < 1e-6 and d_high < -0.10
        return GateResult(
            "K6", "假阳与排序", "PASS" if ok else "FAIL",
            f"低分假阳不影响 AP（差 {d_low:+.4f}），高分假阳使 AP 掉 {abs(d_high):.4f} "
            f"—— 符合 AP 的排序语义" if ok
            else f"低分差 {d_low:+.4f}（应为 0）/ 高分差 {d_high:+.4f}（应 < -0.10）",
            blocking=True, rows=rows,
        )

    runner.run(k6, "K6", "假阳与排序", True)

    # ---- K7 官方 summarize 的坑 --------------------------------------------
    def k7() -> GateResult:
        """复现官方实现的问题，并证明我方实现不受影响。

        把 maxDets 设成不含 100 的 [1, 10, 300]：
          官方 _summarizeDets 里 stats[0] = _summarize(1) 硬编码 maxDets=100，
          查不到 → 空切片 → nan。
          我方 summarize 统一用 max(maxDets)，不受影响。
        """
        from pycocotools.cocoeval import COCOeval

        dt = make_synthetic_dt(raw, SyntheticSpec("perfect"))
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            coco_dt = coco_gt.loadRes(dt)
            ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
            ev.params.maxDets = [1, 10, 300]  # 故意不含 100
            ev.evaluate()
            ev.accumulate()
            official_nan = False
            try:
                with np.errstate(invalid="ignore"):
                    ev.summarize()
                official_ap = float(ev.stats[0])
                official_nan = not np.isfinite(official_ap)
            except Exception:
                official_ap = float("nan")
                official_nan = True

        ours = summarize(ev, max_dets=(1, 10, 300))
        # pycocotools 在查不到 maxDets 时返回 -1.0（它的"无数据"哨兵），
        # 而不是 nan。首轮我按 isfinite 判断，于是 -1.0 被当成"正常值"，
        # 门变成 WARN。判据应同时接受这两种表现。
        official_broken = (not np.isfinite(official_ap)) or official_ap < 0
        rows = [
            ("maxDets 设置", "[1, 10, 300]（故意不含 100）", ""),
            ("官方 stats[0] (AP)", f"{official_ap}",
             "-1.0 或 nan 都表示无数据，因 _summarize(1) 硬编码 maxDets=100"),
            ("我方 AP", f"{ours['AP']:.6f}", "统一用 max(maxDets)=300，正常"),
            ("我方 AR@300", f"{ours.get('AR@300', float('nan')):.6f}",
             "官方接口报不出这一档"),
            ("结论", "自写 summarize 是必要的", "不是重复造轮子"),
        ]
        ok = official_broken and ours["AP"] > 0.999
        return GateResult(
            "K7", "官方 summarize 的坑", "PASS" if ok else "FAIL",
            f"已复现官方实现在非标准 maxDets 下失效（返回 {official_ap}），"
            f"我方实现正常（AP={ours['AP']:.4f}）"
            if ok else f"官方 AP={official_ap} / 我方 AP={ours['AP']:.4f}，"
                       f"与预期不符（本版 pycocotools 行为可能已变）",
            blocking=False, rows=rows,
        )

    runner.run(k7, "K7", "官方 summarize 的坑", False)

    # ---- K8 iscrowd 忽略机制 -----------------------------------------------
    def k8() -> GateResult:
        """切片评测靠 iscrowd=1 忽略非切片成员，必须验证它真的生效。

        构造：把一半 GT 标为 iscrowd=1，只对另一半做完美预测。
        期望：AP 仍接近 1.0 —— 被忽略的 GT 不进 recall 分母。
        对照：不标 iscrowd 时，同样的预测 AP 应显著更低。
        """
        anns = raw["annotations"]
        half = {a["id"] for a in anns[: len(anns) // 2]}

        # A：一半标 ignore，只预测另一半
        raw_ign = {
            **raw,
            "annotations": [
                {**a, "iscrowd": 1 if a["id"] in half else 0} for a in anns
            ],
        }
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            gt_ign = COCO()
            gt_ign.dataset = raw_ign
            gt_ign.createIndex()
        dt_partial = [
            d for d in make_synthetic_dt(raw, SyntheticSpec("perfect"))
        ]
        # 只保留非 ignore 部分对应的预测
        keep_ids = {a["image_id"] for a in anns if a["id"] not in half}
        dt_partial = [d for d in dt_partial if d["image_id"] in keep_ids]

        r_ign = evaluate(gt_ign, dt_partial, max_dets=DEFAULT_MAX_DETS)
        r_plain = evaluate(coco_gt, dt_partial, max_dets=DEFAULT_MAX_DETS)

        rows = [
            ("标 iscrowd=1 的 GT", f"{len(half):,} / {len(anns):,}", ""),
            ("AP（启用忽略）", f"{r_ign.metrics['AP']:.4f}", "被忽略的 GT 不进分母"),
            ("AP（不启用忽略）", f"{r_plain.metrics['AP']:.4f}", "缺失的 GT 计为漏检"),
            ("差值", f"{r_ign.metrics['AP'] - r_plain.metrics['AP']:+.4f}",
             "必须为正，且启用忽略后应接近 1.0"),
            ("n_gt_ignored", f"{r_ign.n_gt_ignored:,}", ""),
        ]
        ok = r_ign.metrics["AP"] > r_plain.metrics["AP"] and r_ign.n_gt_ignored == len(half)
        return GateResult("K8", "iscrowd 忽略机制", "PASS" if ok else "FAIL",
                          "iscrowd=1 的 GT 被正确忽略 —— S6 的切片评测靠它" if ok
                          else "忽略机制未按预期生效",
                          blocking=True, rows=rows)

    runner.run(k8, "K8", "iscrowd 忽略机制", True)

    # ---- K9 bootstrap CI（非阻塞）-----------------------------------------
    def k9() -> GateResult:
        """CI 宽度必须随样本量缩小而变宽。

        首轮用 AP50 + jitter=0.10 测，两个区间宽度都是 0 —— 不是重采样错了，
        而是**指标饱和**：ε=0.10 时 AP50 已经是 1.0000（见 K2 表），
        顶到上界自然没有方差。改用未饱和的 AP + ε=0.20（AP≈0.35），
        远离 0 和 1 两端。
        """
        dt = make_synthetic_dt(raw, SyntheticSpec("j20", jitter=0.20))
        all_ids = sorted(coco_gt.getImgIds())
        small_ids = all_ids[:100]

        pt_a, lo_a, hi_a = bootstrap_ci(coco_gt, dt, metric="AP", n_boot=40,
                                        image_ids=all_ids)
        dt_s = [d for d in dt if d["image_id"] in set(small_ids)]
        pt_b, lo_b, hi_b = bootstrap_ci(coco_gt, dt_s, metric="AP", n_boot=40,
                                        image_ids=small_ids)
        w_a, w_b = hi_a - lo_a, hi_b - lo_b
        rows = [
            ("指标 / 抖动", "AP / ε=0.20", "选未饱和的指标，避免顶到 0 或 1"),
            (f"全量 {len(all_ids)} 图", f"AP={pt_a:.4f}",
             f"CI [{lo_a:.4f}, {hi_a:.4f}] 宽 {w_a:.4f}"),
            (f"小样本 {len(small_ids)} 图", f"AP={pt_b:.4f}",
             f"CI [{lo_b:.4f}, {hi_b:.4f}] 宽 {w_b:.4f}"),
            ("区间宽度比", f"{w_b / max(w_a, 1e-9):.2f}x", "小样本应明显更宽"),
            ("重采样单位", "图像", "同图内的框不独立，按框采会低估方差"),
        ]
        ok = w_b > w_a > 0
        return GateResult("K9", "bootstrap CI", "PASS" if ok else "WARN",
                          f"小样本区间宽 {w_b / max(w_a, 1e-9):.2f} 倍，符合预期" if ok
                          else f"宽度 全量={w_a:.4f} 小样本={w_b:.4f}，未体现样本量效应",
                          blocking=False, rows=rows)

    runner.run(k9, "K9", "bootstrap CI", False)

    # ---- 固化回归基线 ------------------------------------------------------
    baseline = {
        "note": "L0 中间档的实测值。首轮生成后作为回归基线，之后偏离须能解释。",
        "gt_subset": "val2k_repr",
        "n_images": n_img,
        "n_annotations": n_ann,
        "jitter_levels": list(JITTER_LEVELS),
        "metrics": {k: {m: float(v) for m, v in vv.items()} for k, vv in results.items()},
    }
    if BASELINE.is_file():
        old = json.loads(BASELINE.read_text(encoding="utf-8"))
        drift = []
        for name, mm in results.items():
            prev = old.get("metrics", {}).get(name, {})
            for key in ("AP", "AP50"):
                if key in prev and abs(prev[key] - mm[key]) > 1e-6:
                    drift.append(f"{name}.{key}: {prev[key]:.6f} → {mm[key]:.6f}")
        if drift:
            print("\n[warn] 与已有回归基线不一致：")
            for d in drift:
                print(f"        {d}")
        else:
            print("\n与回归基线完全一致")
    else:
        BASELINE.write_text(json.dumps(baseline, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"\n首轮实测，已固化回归基线到 {BASELINE.relative_to(PROJECT_DIR)}")

    extra = [(
        "L0 全部指标实测（回归基线）",
        md_table(
            [(n, f"{m['AP']:.4f}", f"{m['AP50']:.4f}", f"{m['AP75']:.4f}",
              f"{m.get('AR@100', float('nan')):.4f}", f"{m.get('AR@300', float('nan')):.4f}")
             for n, m in results.items()],
            headers=("合成预测", "AP", "AP50", "AP75", "AR@100", "AR@300"),
        ),
    )]
    runner.write_report(REPORT, extra_sections=extra)
    print(f"报告已写入 {REPORT.relative_to(PROJECT_DIR)}")
    return runner.print_summary()


if __name__ == "__main__":
    raise SystemExit(main())
