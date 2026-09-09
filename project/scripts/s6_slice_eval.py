#!/usr/bin/env python
"""S6：难例切片评测器验证门（八条）。

还没有真实预测（S7 才跑 baseline），所以沿用 L0 的思路：
**用合成预测验证切片评测器本身是对的、且有分辨力。**

阻塞门：
    L1  切片 GT 守恒（每个视图的非 ignore 数 == 成员数）
    L2  同轴切片互斥且完备（尺寸三档 / 密度四档构成划分）
    L3  完美预测在每个切片上都得 AP=1.0（切片视图不引入偏差）
    L4  **差异化抖动可被检出** —— 只给 T1 框加大抖动，T1 切片 AP 必须
        显著低于 CLEAN 切片。这条证明切片评测真有分辨力而不是摆设
    L5  UNRELIABLE 标记与 bootstrap CI 行为正确

非阻塞门：
    L6  与 COCO 原生 areaRng 的口径差异对账（两者都正确但不同，须说清）
    L7  报告完整性（每个切片都有 n_ann / 可靠性标记 / 指标）

用法：
    python scripts/s6_slice_eval.py
"""

from __future__ import annotations

import argparse
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
from logodet.eval.coco_eval import DEFAULT_MAX_DETS, evaluate  # noqa: E402
from logodet.eval.l0_selfcheck import SyntheticSpec, jitter_boxes  # noqa: E402
from logodet.eval.slice_eval import (  # noqa: E402
    check_partition,
    default_slices,
    evaluate_slices,
    make_slice_gt,
    slice_members,
    slices_to_frame,
)
from logodet.gates import GateResult, GateRunner, md_table  # noqa: E402
from logodet.paths import P  # noqa: E402

REPORT = PROJECT_DIR / "report" / "s6_slices_report.md"


def _as_coco(d: dict) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO()
        c.dataset = d
        c.createIndex()
    return c


def make_differential_dt(
    gt_dict: dict,
    member_ann_ids: set[int],
    *,
    eps_member: float,
    eps_other: float,
    seed: int = 6129,
) -> list[dict]:
    """造"只对某切片成员加大抖动"的预测。

    这是 L4 的核心工具：如果切片评测有分辨力，那么被加大抖动的那个切片
    指标应显著更低，而其它切片基本不变。
    """
    rng = np.random.default_rng(seed)
    sizes = {im["id"]: (float(im["width"]), float(im["height"]))
             for im in gt_dict["images"]}
    out: list[dict] = []
    for a in gt_dict["annotations"]:
        if a.get("iscrowd", 0) == 1:
            continue
        iw, ih = sizes[int(a["image_id"])]
        x, y, w, h = a["bbox"]
        xyxy = np.array([[x, y, x + w, y + h]], dtype="float64")
        eps = eps_member if int(a["id"]) in member_ann_ids else eps_other
        j = jitter_boxes(xyxy, eps, rng, img_w=iw, img_h=ih)[0]
        out.append(
            {
                "image_id": int(a["image_id"]),
                "category_id": int(a["category_id"]),
                "bbox": [float(j[0]), float(j[1]),
                         float(j[2] - j[0]), float(j[3] - j[1])],
                "score": 1.0,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="val_hard_pool",
                    help="用哪个子集评切片（默认 val_hard_pool）")
    args = ap.parse_args()

    cfg = load_yaml("splits.yaml")
    min_reliable = int(cfg.get("min_ann_for_reliable", 200))

    gt_path = P.artifact("eval") / "gt" / f"{args.subset}.json"
    if not gt_path.is_file():
        print(f"FAIL: 缺 {gt_path}，先跑 python scripts/s5_build_gt.py")
        return 1

    gt_dict = json.loads(gt_path.read_text(encoding="utf-8"))
    coco_gt = _as_coco(gt_dict)

    t = P.artifact("tables")
    sp = P.artifact("splits")
    ann = pd.read_parquet(t / "annotations.parquet")
    images = pd.read_parquet(sp / "split_images.parquet")

    keep_ann = {int(a["id"]) for a in gt_dict["annotations"]}
    keep_img = {int(i["id"]) for i in gt_dict["images"]}
    ann_sub = ann[ann["ann_id"].isin(keep_ann)]
    img_sub = images[images["image_id"].isin(keep_img)]

    specs = default_slices()
    members = slice_members(ann_sub, img_sub, specs)

    print("=" * 78)
    print(f" S6 切片评测器验证门（子集 {args.subset}）")
    print("=" * 78)
    print(f"\nGT: {len(keep_img):,} 图 / {len(keep_ann):,} 框")
    print(f"可靠性下限: {min_reliable} 框\n")
    for s in specs:
        n = len(members[s.name])
        flag = "" if n >= min_reliable else "  UNRELIABLE"
        print(f"  {s.name:<16} {n:>6,} 框{flag}")

    runner = GateRunner("S6", f"S6 切片评测报告（{args.subset}）")

    # ---- L1 切片 GT 守恒 ---------------------------------------------------
    def l1() -> GateResult:
        rows: list[tuple[str, ...]] = []
        bad = 0
        for s in specs:
            ids = members[s.name]
            sl = make_slice_gt(gt_dict, ids)
            n_keep = sum(1 for a in sl["annotations"] if a.get("iscrowd", 0) == 0)
            n_ign = sum(1 for a in sl["annotations"] if a.get("iscrowd", 0) == 1)
            ok = n_keep == len(ids) and n_keep + n_ign == len(sl["annotations"])
            if not ok:
                bad += 1
            rows.append((s.name, f"非ignore {n_keep:,} / 成员 {len(ids):,}",
                         f"ignore {n_ign:,}，合计 {n_keep + n_ign:,}"))
        rows.append(("全体框数", f"{len(gt_dict['annotations']):,}",
                     "每个视图的总框数都必须等于它"))
        return GateResult("L1", "切片 GT 守恒", "PASS" if bad == 0 else "FAIL",
                          "每个切片视图的非 ignore 数都等于成员数，总数守恒" if bad == 0
                          else f"{bad} 个切片视图守恒不成立",
                          blocking=True, rows=rows)

    runner.run(l1, "L1", "切片 GT 守恒", True)

    # ---- L2 同轴互斥完备 ---------------------------------------------------
    def l2() -> GateResult:
        rows: list[tuple[str, ...]] = []
        bad: list[str] = []
        for axis, names in (
            ("size", ["SIZE_small", "SIZE_medium", "SIZE_large"]),
            ("density", ["DENSITY_1", "DENSITY_2", "DENSITY_3_4", "DENSITY_5plus"]),
        ):
            ok, st = check_partition(members, keep_ann, names)
            if not ok:
                bad.append(axis)
            rows.append((
                f"{axis} 轴（{len(names)} 档）",
                f"并集 {st['union']:,} / 全体 {st['total']:,}",
                f"缺 {st['missing']} 多 {st['extra']} 两两重叠 {st['pairwise_overlap']}",
            ))
        # T1 与 CLEAN 不是划分（T1 与 SIZE_small 可以同时成立），只报重叠
        t1, cl = members["T1_truncated"], members["CLEAN"]
        rows.append(("T1 ∩ CLEAN", f"{len(t1 & cl):,}", "必须 0 —— CLEAN 定义为三轴都不命中"))
        rows.append(("T1 ∩ SIZE_small", f"{len(t1 & members['SIZE_small']):,}",
                     "允许非 0 —— 不同轴之间不要求互斥"))
        if t1 & cl:
            bad.append("T1/CLEAN")
        # 残差切片按定义必须与 T1 无交集，且是对应基切片的真子集
        for base, resid in (("SIZE_large", "SIZE_large_noT1"),
                            ("DENSITY_1", "DENSITY_1_noT1")):
            sb, sr = members[base], members[resid]
            no_t1 = len(sr & t1) == 0
            subset_ok = sr <= sb
            rows.append((
                f"{resid}",
                f"{len(sr):,} 框，∩T1={len(sr & t1)}",
                f"是 {base} 子集：{subset_ok}；与 T1 无交集：{no_t1}",
            ))
            if not (no_t1 and subset_ok):
                bad.append(resid)
        return GateResult("L2", "同轴互斥完备", "PASS" if not bad else "FAIL",
                          "尺寸三档与密度四档各自构成划分；残差切片定义自洽" if not bad
                          else f"不成立：{bad}",
                          blocking=True, rows=rows)

    runner.run(l2, "L2", "同轴互斥完备", True)

    # ---- L3 完美预测在每个切片都满分 ---------------------------------------
    def l3() -> GateResult:
        dt = make_differential_dt(gt_dict, set(), eps_member=0.0, eps_other=0.0)
        res = evaluate_slices(gt_dict, dt, members, specs,
                              min_ann_for_reliable=min_reliable,
                              ci_only_when_unreliable=True, n_boot=0)
        rows: list[tuple[str, ...]] = []
        bad: list[str] = []
        for r in res:
            ap = r.metrics.get("AP", float("nan"))
            rows.append((r.name, f"AP={ap:.6f}", f"n_ann={r.n_ann:,}"))
            if r.n_ann > 0 and not (ap > 0.999):
                bad.append(r.name)
        return GateResult("L3", "完美预测逐切片", "PASS" if not bad else "FAIL",
                          "所有非空切片 AP 均 > 0.999 —— 切片视图本身不引入偏差" if not bad
                          else f"以下切片未达满分：{bad}",
                          blocking=True, rows=rows)

    runner.run(l3, "L3", "完美预测逐切片", True)

    # ---- L4 差异化抖动可被检出（关键）-------------------------------------
    def l4() -> GateResult:
        """只给 T1 成员加大抖动，看 T1 切片是否被检出。

        这是整个 S6 最重要的一条：如果切片评测没有分辨力，
        那么"难例上表现更差"这个结论就无从得出，S7 的对比也就没有意义。
        """
        t1_ids = members["T1_truncated"]
        dt = make_differential_dt(gt_dict, t1_ids, eps_member=0.30, eps_other=0.02)
        res = evaluate_slices(gt_dict, dt, members, specs,
                              min_ann_for_reliable=min_reliable,
                              ci_only_when_unreliable=True, n_boot=0)
        m = {r.name: r.metrics.get("AP", float("nan")) for r in res}
        m50 = {r.name: r.metrics.get("AP50", float("nan")) for r in res}

        overall = evaluate(coco_gt, dt, max_dets=DEFAULT_MAX_DETS).metrics
        gap = m["CLEAN"] - m["T1_truncated"]

        rows = [
            ("抖动设置", "T1 成员 ε=0.30，其余 ε=0.02", ""),
            ("T1_truncated", f"AP={m['T1_truncated']:.4f}", f"AP50={m50['T1_truncated']:.4f}"),
            ("CLEAN 对照", f"AP={m['CLEAN']:.4f}", f"AP50={m50['CLEAN']:.4f}"),
            ("差距 CLEAN − T1", f"{gap:+.4f}", "阈值 > 0.30"),
            ("总体（未切片）", f"AP={overall['AP']:.4f}",
             "应落在两者之间 —— 总体是混合结果"),
        ]
        # 其余切片各自含多少 T1 成员 —— 不标出来的话，看到它们偏低会误判。
        # 首轮我给它们写的注释是"未被加抖动，应接近 CLEAN"，那是错的：
        # SIZE_large 里有 37.5% 是 T1，被连带拉低是必然的。
        for name in ("SIZE_small", "SIZE_medium", "SIZE_large", "SIZE_large_noT1",
                     "DENSITY_1", "DENSITY_1_noT1", "DENSITY_5plus"):
            s = members.get(name, set())
            frac = len(s & t1_ids) / len(s) if s else 0.0
            rows.append((f"  {name}", f"AP={m.get(name, float('nan')):.4f}",
                         f"其中 {frac:.1%} 是 T1 成员 → 被连带拉低的幅度与此成正比"))

        ok = gap > 0.30 and m["T1_truncated"] < m["CLEAN"]
        # 总体应在两者之间（允许一点余量，因为总体的图集合不同）
        between = min(m["T1_truncated"], m["CLEAN"]) - 0.05 <= overall["AP"] <= max(
            m["T1_truncated"], m["CLEAN"]
        ) + 0.05
        rows.append(("总体是否落在区间内", str(between), "非阻塞判据，仅作合理性参考"))

        return GateResult(
            "L4", "差异化抖动可检出", "PASS" if ok else "FAIL",
            f"T1 切片 AP={m['T1_truncated']:.4f} 显著低于 CLEAN={m['CLEAN']:.4f}"
            f"（差 {gap:+.4f}）—— 切片评测具备分辨力" if ok
            else f"差距仅 {gap:+.4f}，切片评测分辨力不足",
            blocking=True, rows=rows,
        )

    runner.run(l4, "L4", "差异化抖动可检出", True)

    # ---- L5 UNRELIABLE 与 CI ----------------------------------------------
    def l5() -> GateResult:
        dt = make_differential_dt(gt_dict, members["T1_truncated"],
                                  eps_member=0.20, eps_other=0.10)
        res = evaluate_slices(gt_dict, dt, members, specs,
                              min_ann_for_reliable=min_reliable,
                              ci_metrics=("AP50",), n_boot=60,
                              ci_only_when_unreliable=True)
        rows: list[tuple[str, ...]] = []
        bad: list[str] = []
        for r in res:
            expect_ci = (r.n_ann > 0) and (r.n_ann < min_reliable)
            has_ci = "AP50" in r.ci
            mark = "UNRELIABLE" if not r.reliable else ""
            ci_txt = (f"CI [{r.ci['AP50'][0]:.4f}, {r.ci['AP50'][1]:.4f}]"
                      if has_ci else "无 CI（样本充足）")
            rows.append((r.name, f"n_ann={r.n_ann:,} {mark}".strip(), ci_txt))
            if expect_ci != has_ci:
                bad.append(f"{r.name}(n={r.n_ann}, has_ci={has_ci})")
        rows.append(("规则", f"n_ann < {min_reliable} → 标 UNRELIABLE 且必须带 CI",
                     "样本充足的切片不算 CI（每次要重跑 n_boot 遍评测，很贵）"))
        return GateResult("L5", "UNRELIABLE 与 CI", "PASS" if not bad else "FAIL",
                          "不可靠切片全部带 CI，可靠切片不带 —— 符合规则" if not bad
                          else f"不符规则：{bad}",
                          blocking=True, rows=rows)

    runner.run(l5, "L5", "UNRELIABLE 与 CI", True)

    # ---- L6 与 COCO 原生 areaRng 对账（非阻塞）----------------------------
    def l6() -> GateResult:
        """尺寸切片有两种算法，都正确但不同 —— 必须说清用的是哪个。

        原生 areaRng 同时过滤 GT 与 DT；切片视图只过滤 GT（非成员置 ignore）、
        全部 DT 参与。所以切片视图对假阳更严格，数值一般更低。
        """
        dt = make_differential_dt(gt_dict, set(), eps_member=0.15, eps_other=0.15)
        native = evaluate(coco_gt, dt, max_dets=DEFAULT_MAX_DETS).metrics
        res = evaluate_slices(gt_dict, dt, members, specs,
                              min_ann_for_reliable=min_reliable,
                              ci_only_when_unreliable=True, n_boot=0)
        sl = {r.name: r.metrics.get("AP", float("nan")) for r in res}

        rows = [("口径差异", "原生同时过滤 GT+DT；切片视图只过滤 GT",
                 "切片视图对假阳更严格，数值一般更低或相等")]
        for lbl in ("small", "medium", "large"):
            n = native.get(f"AP_{lbl}", float("nan"))
            s = sl.get(f"SIZE_{lbl}", float("nan"))
            rows.append((f"{lbl}", f"原生 AP_{lbl}={n:.4f}",
                         f"切片视图={s:.4f}  差 {s - n:+.4f}"))
        rows.append(("主口径", "切片视图",
                     "更贴近「在完整检测输出下，该类目标被召回得如何」"))
        return GateResult(
            "L6", "与原生 areaRng 对账", "WARN",
            "两种口径的数值差异已逐档记录；报告中必须注明用的是切片视图口径",
            blocking=False, rows=rows,
        )

    runner.run(l6, "L6", "与原生 areaRng 对账", False)

    # ---- L7 报告完整性 -----------------------------------------------------
    def l7() -> GateResult:
        dt = make_differential_dt(gt_dict, members["T1_truncated"],
                                  eps_member=0.25, eps_other=0.05)
        res = evaluate_slices(gt_dict, dt, members, specs,
                              min_ann_for_reliable=min_reliable,
                              ci_metrics=("AP50",), n_boot=60,
                              ci_only_when_unreliable=True)
        df = slices_to_frame(res)
        out = P.artifact("metrics") / f"s6_slice_demo_{args.subset}.parquet"
        df.to_parquet(out, index=False)

        required = ["slice", "axis", "n_ann", "n_img", "reliable", "AP", "AP50"]
        missing = [c for c in required if c not in df.columns]
        n_nan = int(df["AP"].isna().sum())
        rows = [
            ("切片数", str(len(df)), ""),
            ("必需列齐全", str(not missing), str(missing) if missing else ""),
            ("AP 为 nan 的切片", str(n_nan), "只允许空切片为 nan"),
            ("落盘", str(out.relative_to(P.artifacts)), "S7 会用同样的表结构"),
        ]
        empty = int((df["n_ann"] == 0).sum())
        ok = not missing and n_nan == empty
        return GateResult("L7", "报告完整性", "PASS" if ok else "FAIL",
                          f"{len(df)} 个切片的表结构完整，可供 S7 直接复用" if ok
                          else "表结构不完整",
                          blocking=False, rows=rows)

    runner.run(l7, "L7", "报告完整性", False)

    # ---- L8 残差切片能拆解交叉污染（关键）---------------------------------
    def l8() -> GateResult:
        """验证 `*_noT1` 残差切片真的能把 T1 污染拆出来。

        构造：只给 T1 成员加抖动。期望：
            SIZE_large        被拉低（含 37.5% 的 T1）
            SIZE_large_noT1   回到 CLEAN 水平（已剔除全部 T1）

        若后者确实回到 CLEAN 附近，就证明前者的下降**全部**来自 T1 污染，
        而不是"模型对大目标不行"。没有这条，S7 的结论很容易写错。
        """
        t1_ids = members["T1_truncated"]
        dt = make_differential_dt(gt_dict, t1_ids, eps_member=0.30, eps_other=0.02)
        res = evaluate_slices(gt_dict, dt, members, specs,
                              min_ann_for_reliable=min_reliable,
                              ci_only_when_unreliable=True, n_boot=0)
        m = {r.name: r.metrics.get("AP", float("nan")) for r in res}
        clean = m["CLEAN"]

        rows: list[tuple[str, ...]] = [
            ("CLEAN 基准", f"AP={clean:.4f}", "未含任何 T1 成员"),
        ]
        checks: list[bool] = []
        for base, resid in (("SIZE_large", "SIZE_large_noT1"),
                            ("DENSITY_1", "DENSITY_1_noT1")):
            sb, sr = members[base], members[resid]
            frac = len(sb & t1_ids) / len(sb) if sb else 0.0
            recovered = m[resid] - m[base]
            close = abs(m[resid] - clean) < 0.10
            checks.append(close)
            rows.append((
                base,
                f"AP={m[base]:.4f}（含 {frac:.1%} T1）",
                f"剔除后 {resid}={m[resid]:.4f}，回升 {recovered:+.4f}",
            ))
            rows.append((
                f"  {resid} vs CLEAN",
                f"差 {m[resid] - clean:+.4f}",
                "回到 CLEAN 水平（|差| < 0.10）" if close else "仍有差距，需另找原因",
            ))
        rows.append((
            "解读规则",
            "报 SIZE_large 必须同时报 SIZE_large_noT1",
            "否则会把 T1 污染误读成「模型对大目标不行」",
        ))
        ok = all(checks)
        return GateResult(
            "L8", "残差切片拆解污染", "PASS" if ok else "FAIL",
            "剔除 T1 后两个残差切片都回到 CLEAN 水平 —— 证明下降全部来自交叉污染"
            if ok else "残差切片未回到 CLEAN 水平，下降另有原因",
            blocking=True, rows=rows,
        )

    runner.run(l8, "L8", "残差切片拆解污染", True)

    # ---- 报告附表：一份完整的切片演示 --------------------------------------
    dt_demo = make_differential_dt(gt_dict, members["T1_truncated"],
                                   eps_member=0.25, eps_other=0.05)
    res_demo = evaluate_slices(gt_dict, dt_demo, members, specs,
                               min_ann_for_reliable=min_reliable,
                               ci_metrics=("AP50",), n_boot=60,
                               ci_only_when_unreliable=True)
    rows = []
    for r in res_demo:
        ci = r.ci.get("AP50")
        rows.append((
            r.name, r.axis, f"{r.n_ann:,}", f"{r.n_img:,}",
            "是" if r.reliable else "**UNRELIABLE**",
            f"{r.metrics.get('AP', float('nan')):.4f}",
            f"{r.metrics.get('AP50', float('nan')):.4f}",
            f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—",
        ))
    extra = [
        ("切片评测演示（合成预测：T1 成员 ε=0.25，其余 ε=0.05）",
         md_table(rows, headers=("切片", "轴", "框数", "图数", "可靠",
                                 "AP", "AP50", "AP50 CI"))),
        ("口径声明", "\n".join([
            "- **切片视图口径**：非成员 GT 置 `iscrowd=1`（保留但忽略），全部 det 参与。",
            "- 与 COCO 原生 `AP_small` 等**不同**：后者同时过滤 GT 与 DT。",
            "- 本子集为 `val_hard_pool`，**分布已被人为富集，禁止用它报总体指标**。",
            f"- 框数 < {min_reliable} 的切片标 UNRELIABLE，必须带 bootstrap CI，禁止下强结论。",
        ])),
    ]
    runner.write_report(REPORT, extra_sections=extra)
    print(f"\n报告已写入 {REPORT.relative_to(PROJECT_DIR)}")
    return runner.print_summary()


if __name__ == "__main__":
    raise SystemExit(main())
