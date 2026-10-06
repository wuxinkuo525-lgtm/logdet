#!/usr/bin/env python
"""S9 步骤三（下）：把一次 DINO 训练按 BASELINE_EVALUATION 的全部表格出一遍，并与 baseline 并排。

输入是 s9_dino_predict.py 的产物（runs/predictions/dino/<run-name>/）。
baseline 的数字直接读 s7_evaluate.py 落盘的 runs/metrics/s7_*.parquet，不重算、不手抄。

出的表（口径与 BASELINE_EVALUATION §3 / §4 完全相同，同一套 evaluate / evaluate_slices）：
    1. 总体指标      val2k_repr，单类口径
    2. 召回细分      val2k_repr，AR50@300 与分尺寸 AR
    3. 逐切片结果    val_hard_pool，12 个切片的 AP50 / AR@300
    4. 受控对比      截断、尺寸、密度
    5. 3000 类指标   「品牌也要认对」，baseline 没有这个口径，单独列
    6. 超参与条件    从训练当时落盘的配置里读

产物：
    report/s9_dino_<run-name>_report.md
    runs/metrics/s9_dino_<run-name>_overall.parquet / _slices.parquet

用法（baseline 环境 .venv）：
    python scripts/s9_dino_report.py --run-name shared
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
from logodet.dino_run import resolve_eval_ann, validate_prediction_bundle  # noqa: E402
from logodet.data.adapters.to_coco_json import build_coco_dt  # noqa: E402
from logodet.eval.coco_eval import DEFAULT_MAX_DETS, evaluate  # noqa: E402
from logodet.eval.slice_eval import (  # noqa: E402
    default_slices,
    evaluate_slices,
    slice_members,
    slices_to_frame,
)
from logodet.gates import md_table  # noqa: E402
from logodet.paths import P  # noqa: E402

NAN = float("nan")
BASELINES = (("L2_owlv2_zeroshot", "L2 OWLv2 zero-shot"), ("L3_coco_collapsed", "L3 COCO 类塌缩"),
             ("L1_rpn_proposals", "L1 RPN proposals"))


def fmt(x: float) -> str:
    return "—" if x is None or x != x else f"{x:.4f}"


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def as_coco(d: dict) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO()
        c.dataset = d
        c.createIndex()
    return c


def single_class(d: dict) -> dict:
    """多类 COCO 标注塌缩成 categories=[logo]，与 runs/eval/gt/ 下 baseline 用的 GT 同一口径。"""
    return {**d, "categories": [{"id": 1, "name": "logo", "supercategory": "logo"}],
            "annotations": [{**a, "category_id": 1} for a in d["annotations"]]}


def subset_ann(d: dict, image_ids: set[int]) -> dict:
    return {**d, "images": [im for im in d["images"] if im["id"] in image_ids],
            "annotations": [a for a in d["annotations"] if a["image_id"] in image_ids]}


def load_predictions(path: Path, name: str) -> pd.DataFrame:
    z = np.load(path)
    box = z[f"{name}_box"]
    return pd.DataFrame({
        "image_id": z[f"{name}_image_id"], "category_id": z[f"{name}_category_id"].astype("int64"),
        "x1": box[:, 0], "y1": box[:, 1], "x2": box[:, 2], "y2": box[:, 3], "score": z[f"{name}_score"],
    })


def eval_on(gt: dict, preds: pd.DataFrame) -> dict[str, float]:
    ids = sorted(int(im["id"]) for im in gt["images"])
    dt = build_coco_dt(preds[preds["image_id"].isin(set(ids))])
    return evaluate(as_coco(gt), dt, image_ids=ids, max_dets=DEFAULT_MAX_DETS).metrics


def conditions_table(man: dict) -> str:
    c = man["conditions"]
    sched = "；".join(
        f"{s['type']}(" + ", ".join(f"{k}={s[k]}" for k in ("start_factor", "end", "milestones", "gamma") if k in s) + ")"
        for s in c["param_scheduler"]) or "恒定学习率"
    last = c.get("last_train_log", {})
    rows = [
        ("checkpoint", f"`{Path(c['checkpoint']).name}`", f"第 {c['checkpoint_iter']:,} iter / 计划 {c['max_iters']:,} iter"),
        ("预训练权重", f"`{c['pretrained']}`", "MMDetection 原生 DINO-R50 COCO"),
        ("模型", f"{c['neck']} + {c['head']}", f"{c['num_queries']} query，每图最多 {c['max_per_img']} 框"),
        ("训练数据", f"`{c['train_ann']}`",
         f"{c['train_images']:,} 图 / {c['train_boxes']:,} 框 / {c['num_classes']:,} 类中出现 {c['train_classes_present']:,} 类"),
        ("训练量", f"{c['max_iters']:,} iter × batch {c['batch_size']}",
         f"≈ {c['max_iters'] * c['batch_size'] / c['train_images']:.1f} epoch"),
        ("优化器", f"{c['optimizer']}，lr {c['lr']:g}，weight_decay {c['weight_decay']:g}",
         f"backbone lr ×{c['backbone_lr_mult']:g}，梯度裁剪 {c['grad_clip']:g}，AMP {'开' if c['amp'] else '关'}"),
        ("学习率计划", sched, ""),
        ("训练随机种子", str(c.get("seed", "旧记录未保存")), "初始化、数据采样和增强"),
        ("训练输入", f"短边 {c['train_short_sides'][0]}–{c['train_short_sides'][-1]} 多尺度",
         f"长边上限 {c['train_long_side_cap']}，随机水平翻转"),
        ("测试输入", f"{c['test_scale']}", "keep_ratio"),
        ("辅助分支", f"prototype ×{c['loss_prototype_weight']:g}，hierarchy ×{c['loss_hierarchy_weight']:g}",
         f"prototype 温度 {c['proto_temperature']:g}"),
        ("训练末尾 loss", "，".join(f"{k}={last[k]:.4f}" for k in
                                ("loss", "loss_cls", "loss_bbox", "loss_iou", "loss_prototype", "loss_hierarchy")
                                if k in last) or "无日志", f"iter {last.get('iter', '?')}"),
        ("融合权重 α/β/γ", " / ".join(f"{last[k]:.4f}" for k in ("fusion_alpha", "fusion_beta", "fusion_gamma")
                                  if k in last) or "无日志", "初值各 1/3"),
        ("推理", f"{man['gpu']}，{man['img_per_sec']} img/s",
         f"torch {man['versions']['torch']} / mmdet {man['versions']['mmdet']} / mmcv {man['versions']['mmcv']}"),
        ("预测生成时间", man["created_utc"], f"{man['n_images']:,} 图"),
    ]
    return md_table(rows, headers=("项", "值", "说明"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()

    pred_dir = P.artifact("predictions") / "dino" / args.run_name
    man = validate_prediction_bundle(pred_dir)
    agn = load_predictions(pred_dir / "predictions.npz", "agnostic")
    cls = load_predictions(pred_dir / "predictions.npz", "classwise")
    label = f"**DINO {args.run_name}**"
    mdir = P.artifact("metrics")
    base_ov = pd.read_parquet(mdir / "s7_overall.parquet").set_index("baseline")
    base_sl = pd.read_parquet(mdir / "s7_slices.parquet")
    min_reliable = int(load_yaml("splits.yaml").get("min_ann_for_reliable", 200))

    gt2k = read_json(P.artifact("eval") / "gt" / "val2k_repr.json")
    gthp = read_json(P.artifact("eval") / "gt" / "val_hard_pool.json")
    ids2k = {int(im["id"]) for im in gt2k["images"]}
    idshp = sorted(int(im["id"]) for im in gthp["images"])
    own_ids = {im['id'] for im in read_json(resolve_eval_ann(man, pred_dir))['images']}
    validate_prediction_bundle(pred_dir, ids2k | set(idshp) | own_ids)

    # ---- 1/2 总体（val2k_repr，单类）----------------------------------------
    ov = eval_on(gt2k, agn)
    def base(name: str, key: str) -> float:
        return float(base_ov.loc[name].get(key, NAN)) if name in base_ov.index else NAN

    keys1 = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR@1", "AR@10", "AR@100", "AR@300")
    t1 = [(label, *(fmt(ov[k]) for k in keys1))]
    t1 += [(title, *(fmt(base(name, k)) for k in keys1)) for name, title in BASELINES]
    keys2 = ("AR50@300", "AR_small@300", "AR_medium@300", "AR_large@300")
    t2 = [(label, *(fmt(ov[k]) for k in keys2))]
    t2 += [(title, *(fmt(base(name, k)) for k in keys2)) for name, title in BASELINES]

    # ---- 3/4 切片（val_hard_pool，单类）-------------------------------------
    ann = pd.read_parquet(P.artifact("tables") / "annotations.parquet")
    images = pd.read_parquet(P.artifact("splits") / "split_images.parquet")
    specs = default_slices()
    members = slice_members(ann[ann["ann_id"].isin({int(a["id"]) for a in gthp["annotations"]})],
                            images[images["image_id"].isin(set(idshp))], specs)
    sl = evaluate_slices(gthp, build_coco_dt(agn[agn["image_id"].isin(set(idshp))]), members, specs,
                         min_ann_for_reliable=min_reliable, ci_metrics=("AP50",), n_boot=60)
    mine = {r.name: r for r in sl}
    def bs(name: str, slice_name: str, key: str) -> float:
        hit = base_sl[(base_sl["baseline"] == name) & (base_sl["slice"] == slice_name)]
        return float(hit.iloc[0][key]) if len(hit) else NAN

    t3 = [(r.name, f"{r.n_ann:,}", "是" if r.reliable else "**否**",
           f"**{fmt(r.metrics.get('AP50', NAN))}**", fmt(bs("L2_owlv2_zeroshot", r.name, "AP50")),
           fmt(bs("L3_coco_collapsed", r.name, "AP50")),
           f"**{fmt(r.metrics.get('AR@300', NAN))}**", fmt(bs("L2_owlv2_zeroshot", r.name, "AR@300")),
           fmt(bs("L3_coco_collapsed", r.name, "AR@300")), fmt(bs("L1_rpn_proposals", r.name, "AR@300")))
          for r in sl]

    def ap50(model: str, s: str) -> float:
        return mine[s].metrics.get("AP50", NAN) if model == "DINO" else bs(model, s, "AP50")
    def ar300(model: str, s: str) -> float:
        return mine[s].metrics.get("AR@300", NAN) if model == "DINO" else bs(model, s, "AR@300")
    models = (("DINO", label), ("L2_owlv2_zeroshot", "L2 OWLv2"), ("L3_coco_collapsed", "L3 类塌缩"))
    t4a = [(title, fmt(ap50(m, "SIZE_small")), fmt(ap50(m, "SIZE_large")),
            f"{ap50(m, 'SIZE_large') / max(ap50(m, 'SIZE_small'), 1e-9):.1f} 倍") for m, title in models]
    t4b = [(title, fmt(ap50(m, "T1_large")), fmt(ap50(m, "SIZE_large_noT1")),
            f"{ap50(m, 'T1_large') - ap50(m, 'SIZE_large_noT1'):+.4f}") for m, title in models]
    dens = ("DENSITY_1", "DENSITY_2", "DENSITY_3_4", "DENSITY_5plus")
    t4c = [(f"{title} AP50", *(fmt(ap50(m, d)) for d in dens)) for m, title in models]
    t4c += [(f"{title} AR@300", *(fmt(ar300(m, d)) for d in dens)) for m, title in models]

    # ---- 5 3000 类（品牌也要认对）-------------------------------------------
    val_dino = read_json(P.artifact("coco") / "instances_val_dino.json")
    own = read_json(resolve_eval_ann(man, pred_dir))
    c2k = eval_on(subset_ann(val_dino, ids2k), cls)
    own_cls = eval_on(own, cls)
    own_agn = eval_on(single_class(own), agn)
    own_label = man.get("own_eval_name", Path(man["own_eval_ann"]).stem)
    own_name = f"{own_label}（{len(own['images']):,} 图）"
    keys5 = ("AP", "AP50", "AP75", "AR@1", "AR@10", "AR@100", "AR@300")
    t5 = [
        (f"val2k_repr（{len(ids2k):,} 图）", "3000 类", *(fmt(c2k[k]) for k in keys5)),
        (f"val2k_repr（{len(ids2k):,} 图）", "单类", *(fmt(ov[k]) for k in keys5)),
        (own_name, "3000 类", *(fmt(own_cls[k]) for k in keys5)),
        (own_name, "单类", *(fmt(own_agn[k]) for k in keys5)),
    ]

    # ---- 落盘 ---------------------------------------------------------------
    pd.DataFrame([
        {"run": args.run_name, "subset": "val2k_repr", "protocol": "agnostic", **ov},
        {"run": args.run_name, "subset": "val2k_repr", "protocol": "3000class", **c2k},
        {"run": args.run_name, "subset": own_label, "protocol": "agnostic", **own_agn},
        {"run": args.run_name, "subset": own_label, "protocol": "3000class", **own_cls},
    ]).to_parquet(mdir / f"s9_dino_{args.run_name}_overall.parquet", index=False)
    frame = slices_to_frame(sl)
    frame["baseline"] = f"dino_{args.run_name}"
    frame.to_parquet(mdir / f"s9_dino_{args.run_name}_slices.parquet", index=False)

    sections = [
        ("1. 总体指标（val2k_repr，2,000 图 / 2,451 框，单类口径）",
         md_table(t1, headers=("模型", "AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l", "AR@1", "AR@10", "AR@100", "AR@300"))),
        ("2. 召回细分（val2k_repr，单类口径）",
         md_table(t2, headers=("模型", "AR50@300", "AR_small@300", "AR_medium@300", "AR_large@300"))),
        ("3. 逐切片结果（val_hard_pool，1,230 图 / 2,099 框，单类口径）",
         md_table(t3, headers=("切片", "框数", "可靠", "DINO AP50", "L2 AP50", "L3 AP50",
                               "DINO AR@300", "L2 AR@300", "L3 AR@300", "L1 AR@300"))),
        ("4a. 尺寸跨度（AP50，small → large）", md_table(t4a, headers=("模型", "SIZE_small", "SIZE_large", "跨度"))),
        ("4b. 截断的受控对比（AP50，均为大目标）",
         md_table(t4b, headers=("模型", "T1_large", "SIZE_large_noT1", "差"))),
        ("4c. 密度", md_table(t4c, headers=("模型 / 指标", "D1", "D2", "D3_4", "D5+"))),
        ("5. 3000 类口径（框对且品牌认对）与单类口径对照",
         md_table(t5, headers=("评测集", "口径", "AP", "AP50", "AP75", "AR@1", "AR@10", "AR@100", "AR@300"))),
        ("6. 这次训练的超参与条件", conditions_table(man)),
        ("口径声明", "\n".join([
            "- 表 1–4 与 `BASELINE_EVALUATION.md` §3 / §4 同口径、同一套评测代码；baseline 的数字读自 "
            "`runs/metrics/s7_overall.parquet` / `s7_slices.parquet`，未重算。",
            "- **单类口径**：3000 个品牌塌缩成一个类「logo」，只问框没框到。DINO 每个 query 只保留最高分的类，"
            "每图最多 300 框。",
            "- **3000 类口径**：品牌也要认对。baseline 没有这个口径（它们都不认品牌），所以表 5 没有 baseline 行。"
            "AP 取 maxDets=300，与 mmdet 训练日志里的 `coco/bbox_mAP`（maxDets=100）不是同一个数。",
            "- 总体指标只用 val2k_repr，切片指标只用 val_hard_pool；L1 禁报 AP。",
            "- L2 每图最多 100 框，所以它的 AR@300 = AR@100；预算对比请看 AR@100。",
            "- baseline 三者未在 LogoDet-3K 上训练；DINO 在表 6 所列数据上训练过，val 未参与训练。",
            "- BASELINE_EVALUATION §4 解读四的密度受控格（`D*_large_noT1`）、§7 的 prompt 选型不在本报告内。",
        ])),
    ]
    body = [f"# S9 DINO 评测报告：{args.run_name}", "",
            "> 由 `scripts/s9_dino_report.py` 自动生成，重跑即刷新，不要手改。", ""]
    for title, content in sections:
        body += [f"## {title}", "", content, ""]
    report = PROJECT_DIR / "report" / f"s9_dino_{args.run_name}_report.md"
    report.write_text("\n".join(body), encoding="utf-8")

    for title, content in sections[:-1]:
        print(f"\n## {title}\n\n{content}")
    print(f"\n报告已写入 {report.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
