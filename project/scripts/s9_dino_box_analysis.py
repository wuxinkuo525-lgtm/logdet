#!/usr/bin/env python
"""S9：对一组 DINO 预测做框级分析——不只问「分数多少」，而是逐个框看「错在哪」。

汇总的 AP 只说明好坏；这里对 val2k_repr（2,000 图）里的每一个标注框、每一个预测框逐一归类：

    标注框（找到了吗）       检出 / 分数太低 / 定位不准 / 完全漏检
    预测框（为什么是错的）   正确 / 重复框 / 定位错 / 背景误检
    品牌（框对了，认对了吗） 品牌对 / 错成同一大类的品牌 / 错成别的大类 / 没有框到

并按尺寸、截断、密度、大类、长宽比分组；最后把每类错误挑若干例子画成拼图，供人工逐张看。
**统计是客观的；错误的原因（模型问题、标注错误、遮挡到人也认不出……）只能看图判断，需要人来做。**

工作点：单类口径下，在 val2k_repr 上取使 F1（IoU ≥ 0.5）最大的分数阈值，阈值以上的框算「模型给出的框」。
阈值本身是在 val 上选的，只用于本分析的归类，不用于报告任何正式指标。

用法（baseline 环境 .venv）：
    python scripts/s9_dino_box_analysis.py --run-name shared_b4_decay_e12

产物：
    report/s9_dino_<run>_box_analysis.md          统计表 + 拼图路径 + 口径说明
    runs/metrics/s9_box_<run>_gt.parquet          每个标注框的归类
    runs/metrics/s9_box_<run>_pred.parquet        每个（阈值以上的）预测框的归类
    runs/analysis/box_<run>/sheet_*.jpg           各类错误的例图拼图
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.gates import md_table  # noqa: E402
from logodet.paths import P  # noqa: E402

GT_ORDER = ["检出", "分数太低", "定位不准", "完全漏检"]
PRED_ORDER = ["正确", "重复框", "框了一部分", "定位错", "背景误检"]
BRAND_ORDER = ["品牌对", "错成同一大类", "错成别的大类", "没有框到"]
SEED = 20261007


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a [N,4] × b [M,4]（x1,y1,x2,y2）→ [N,M]。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def iof_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """预测框有多大比例落在标注框里：交集 / 预测框面积。[N_gt, M_pred]。"""
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((len(gt), len(pred)))
    x1 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    y1 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    y2 = np.minimum(gt[:, None, 3], pred[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / np.maximum((pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]), 1e-9)[None, :]


def greedy_match(scores: np.ndarray, ious: np.ndarray, thr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """COCO 式贪心匹配：按分数从高到低，每个预测框配给 IoU 最大且还没被配走的标注框。

    返回 (pred_gt[M]：配上的标注框下标或 -1, gt_iou[N]：配上时的 IoU 或 0)。
    """
    n_gt, n_pred = ious.shape
    pred_gt = np.full(n_pred, -1)
    gt_taken = np.zeros(n_gt, bool)
    gt_iou = np.zeros(n_gt)
    for j in np.argsort(-scores, kind="stable"):
        if n_gt == 0:
            break
        cand = np.where(~gt_taken & (ious[:, j] >= thr))[0]
        if len(cand):
            g = cand[np.argmax(ious[cand, j])]
            pred_gt[j], gt_taken[g], gt_iou[g] = g, True, ious[g, j]
    return pred_gt, gt_iou


def load_preds(run: str, kind: str, image_ids: set[int]) -> pd.DataFrame:
    z = np.load(P.artifact("predictions") / "dino" / run / "predictions.npz")
    df = pd.DataFrame({"image_id": z[f"{kind}_image_id"], "score": z[f"{kind}_score"],
                       "category_id": z[f"{kind}_category_id"].astype(int)})
    box = z[f"{kind}_box"]
    df[["x1", "y1", "x2", "y2"]] = box
    return df[df["image_id"].isin(image_ids)].reset_index(drop=True)


def f1_threshold(gt: pd.DataFrame, pred: pd.DataFrame) -> tuple[float, float]:
    """单类口径：在全部预测上做一次贪心匹配，按分数扫一遍找 F1 最大的阈值。"""
    tp_scores, all_scores = [], []
    for img, p in pred.groupby("image_id"):
        g = gt[gt["image_id"] == img]
        pg, _ = greedy_match(p["score"].to_numpy(), iou_matrix(g[["x1", "y1", "x2", "y2"]].to_numpy(),
                                                                 p[["x1", "y1", "x2", "y2"]].to_numpy()))
        all_scores.append(p["score"].to_numpy())
        tp_scores.append(p["score"].to_numpy()[pg >= 0])
    s, t = np.concatenate(all_scores), np.concatenate(tp_scores)
    order = np.sort(s)[::-1]
    t_sorted = np.sort(t)[::-1]
    best = (0.0, 0.5)
    for thr in np.unique(np.quantile(order[: min(len(order), 50 * len(gt))], np.linspace(0, 1, 400))):
        kept = (s >= thr).sum()
        tp = np.searchsorted(-t_sorted, -thr, side="right")
        if kept:
            prec, rec = tp / kept, tp / len(gt)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best[0]:
                best = (f1, float(thr))
    return best[1], best[0]


def classify(gt: pd.DataFrame, pred: pd.DataFrame, thr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_rows, pred_rows = [], []
    for img, g in gt.groupby("image_id"):
        p = pred[pred["image_id"] == img]
        gb, pb = g[["x1", "y1", "x2", "y2"]].to_numpy(), p[["x1", "y1", "x2", "y2"]].to_numpy()
        ious = iou_matrix(gb, pb)
        iofs = iof_matrix(gb, pb)
        keep = p["score"].to_numpy() >= thr
        pg, giou = greedy_match(p["score"].to_numpy()[keep], ious[:, keep])
        kept_idx = np.where(keep)[0]
        for gi, ann in enumerate(g["ann_id"].to_numpy()):
            if giou[gi] > 0:
                outcome = "检出"
            elif (ious[gi, ~keep] >= 0.5).any():
                outcome = "分数太低"
            elif keep.any() and ious[gi, keep].max() >= 0.1:
                outcome = "定位不准"
            else:
                outcome = "完全漏检"
            best_any = int(np.argmax(ious[gi])) if ious.shape[1] else -1
            gt_rows.append(dict(ann_id=ann, image_id=img, outcome=outcome, match_iou=giou[gi],
                                best_iou_kept=ious[gi, keep].max() if keep.any() else 0.0,
                                best_iou_any=ious[gi].max() if ious.shape[1] else 0.0,
                                best_pred_box=pb[best_any].tolist() if best_any >= 0 else None,
                                best_pred_score=float(p["score"].to_numpy()[best_any]) if best_any >= 0 else 0.0))
        matched = set(pg[pg >= 0].tolist())
        for k, j in enumerate(kept_idx):
            col = ious[:, j]
            if pg[k] >= 0:
                kind = "正确"
            elif (col >= 0.5).any():
                kind = "重复框"
            elif col.size and iofs[:, j].max() >= 0.8:
                kind = "框了一部分"
            elif col.size and col.max() >= 0.1:
                kind = "定位错"
            else:
                kind = "背景误检"
            pred_rows.append(dict(image_id=img, score=float(p["score"].iloc[j]), kind=kind,
                                  max_iou=float(col.max()) if col.size else 0.0,
                                  x1=pb[j, 0], y1=pb[j, 1], x2=pb[j, 2], y2=pb[j, 3]))
        del matched
    return pd.DataFrame(gt_rows), pd.DataFrame(pred_rows)


def brand_outcomes(gt: pd.DataFrame, cls_pred: pd.DataFrame, super_of: dict[int, str]) -> pd.DataFrame:
    """每个标注框：位置对上（IoU ≥ 0.5）的品牌预测里分数最高的那个，猜的是什么品牌。"""
    rows = []
    for img, g in gt.groupby("image_id"):
        p = cls_pred[cls_pred["image_id"] == img]
        ious = iou_matrix(g[["x1", "y1", "x2", "y2"]].to_numpy(), p[["x1", "y1", "x2", "y2"]].to_numpy())
        for gi, (ann, true_c) in enumerate(zip(g["ann_id"], g["class_id"])):
            hit = np.where(ious[gi] >= 0.5)[0] if ious.shape[1] else np.array([], int)
            if not len(hit):
                rows.append(dict(ann_id=ann, brand="没有框到", pred_class=-1, pred_score=0.0))
                continue
            j = hit[np.argmax(p["score"].to_numpy()[hit])]
            pc = int(p["category_id"].iloc[j])
            if pc == true_c:
                brand = "品牌对"
            elif super_of.get(pc) == super_of.get(int(true_c)):
                brand = "错成同一大类"
            else:
                brand = "错成别的大类"
            rows.append(dict(ann_id=ann, brand=brand, pred_class=pc, pred_score=float(p["score"].iloc[j])))
    return pd.DataFrame(rows)


def share_table(df: pd.DataFrame, by: str, col: str, order: list[str], by_order: list | None = None) -> str:
    t = pd.crosstab(df[by], df[col]).reindex(columns=order, fill_value=0)
    if by_order is not None:
        t = t.reindex([b for b in by_order if b in t.index])
    rows = []
    for key, r in t.iterrows():
        n = int(r.sum())
        rows.append((str(key), f"{n:,}", *[f"{100 * r[c] / n:.1f}%" for c in order]))
    tot = t.sum()
    n = int(tot.sum())
    rows.append(("**合计**", f"{n:,}", *[f"{100 * tot[c] / n:.1f}%" for c in order]))
    return md_table(rows, headers=(by, "框数", *order))


def _font(size: int):
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def contact_sheet(items: list[dict], out: Path, title: str, cols: int = 4, cell: int = 380) -> None:
    """items: dict(path, gt_box, pred_box, label)。绿框 = 标注，红框 = 预测。"""
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    label_h, head_h = 54, 40
    sheet = Image.new("RGB", (cols * cell, head_h + rows * (cell + label_h)), (250, 250, 248))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), title, fill=(0, 0, 0), font=_font(22))
    small = _font(15)
    for k, it in enumerate(items):
        img = Image.open(it["path"]).convert("RGB")
        d = ImageDraw.Draw(img)
        w = max(2, img.width // 200)
        gtb = it.get("gt_box")
        for box in (gtb if gtb and isinstance(gtb[0], (list, tuple)) else [gtb] if gtb is not None else []):
            d.rectangle(box, outline=(0, 200, 0), width=w)
        if it.get("pred_box") is not None:
            d.rectangle(it["pred_box"], outline=(230, 0, 0), width=w)
        img.thumbnail((cell - 8, cell - 8))
        x, y = (k % cols) * cell, head_h + (k // cols) * (cell + label_h)
        sheet.paste(img, (x + 4, y + 4))
        draw.multiline_text((x + 6, y + cell), it["label"], fill=(30, 30, 30), font=small, spacing=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--examples", type=int, default=12, help="每类错误画几个例子")
    args = ap.parse_args()
    run = args.run_name
    rng = np.random.default_rng(SEED)

    gt_json = json.loads((P.artifact("eval") / "gt" / "val2k_repr.json").read_text(encoding="utf-8"))
    ids = {int(im["id"]) for im in gt_json["images"]}
    ann = pd.read_parquet(P.artifact("tables") / "annotations.parquet")
    ann = ann[ann["ann_id"].isin({int(a["id"]) for a in gt_json["annotations"]})].copy()
    images = pd.read_parquet(P.artifact("splits") / "split_images.parquet").set_index("image_id")
    classes = pd.read_parquet(P.artifact("tables") / "classes.parquet").set_index("class_id")
    super_of = classes["supercat"].to_dict()
    name_of = classes["class_name"].to_dict()
    ann["n_boxes_bin"] = ann["image_id"].map(images["n_boxes_bin"])
    gt_boxes_of = {img: g[["x1", "y1", "x2", "y2"]].to_numpy().tolist() for img, g in ann.groupby("image_id")}
    ann["截断"] = np.where(ann["truncated"] == 1, "截断", "完整")
    ann["长宽比"] = pd.cut(ann["ar"], [0, 0.67, 1.5, 3, np.inf], labels=["竖长 (<0.67)", "近方 (0.67–1.5)", "横长 (1.5–3)", "很扁 (>3)"]).astype(str)

    man = json.loads((P.artifact("predictions") / "dino" / run / "manifest.json").read_text(encoding="utf-8"))
    train_ann_name = man["conditions"]["train_ann"]
    train_path = next(p for p in (P.artifact("coco") / "debug" / train_ann_name, P.artifact("coco") / train_ann_name) if p.is_file())
    train_counts = pd.Series([a["category_id"] for a in json.loads(train_path.read_text(encoding="utf-8"))["annotations"]]).value_counts()

    agn = load_preds(run, "agnostic", ids)
    cls = load_preds(run, "classwise", ids)
    thr, f1 = f1_threshold(ann, agn)
    gt_out, pred_out = classify(ann, agn, thr)
    gt_out = gt_out.merge(ann[["ann_id", "area_bin", "截断", "n_boxes_bin", "supercat", "长宽比", "class_id",
                               "x1", "y1", "x2", "y2", "class_name"]], on="ann_id")
    br = brand_outcomes(ann, cls, super_of)
    gt_out = gt_out.merge(br, on="ann_id")
    gt_out["训练集里该品牌的框数"] = gt_out["class_id"].map(train_counts).fillna(0).astype(int)
    gt_out["品牌在训练集里"] = np.where(gt_out["训练集里该品牌的框数"] > 0, "见过", "没见过")

    mdir = P.artifact("metrics")
    gt_out.drop(columns=["best_pred_box"]).to_parquet(mdir / f"s9_box_{run}_gt.parquet", index=False)
    pred_out.to_parquet(mdir / f"s9_box_{run}_pred.parquet", index=False)

    # ---- 统计 ----
    n_gt, n_pred = len(gt_out), len(pred_out)
    tp = gt_out[gt_out["outcome"] == "检出"]
    ious = tp["match_iou"].to_numpy()
    sections = []
    sections.append(("0. 工作点", md_table([
        ("评测集", "val2k_repr", f"{len(ids):,} 图 / {n_gt:,} 个标注框"),
        ("分数阈值", f"{thr:.4f}", f"单类口径下使 F1（IoU ≥ 0.5）最大，F1 = {f1:.3f}"),
        ("阈值以上的预测框", f"{n_pred:,}", f"平均每图 {n_pred / len(ids):.2f} 个；每图原始输出 300 个"),
    ])))
    sections.append(("1. 标注框：找到了吗（单类口径）", share_table(gt_out.assign(全部="全部"), "全部", "outcome", GT_ORDER) + "\n\n"
                     "- **检出**：阈值以上有框与它 IoU ≥ 0.5 且配对成功\n"
                     "- **分数太低**：模型其实框到了（IoU ≥ 0.5），但分数在阈值以下\n"
                     "- **定位不准**：阈值以上有框压在它附近（0.1 ≤ IoU < 0.5），但没框准\n"
                     "- **完全漏检**：300 个框里都没有框准的，阈值以上也没有框靠近它"))
    sections.append(("2. 预测框：为什么是错的（阈值以上）", share_table(pred_out.assign(全部="全部"), "全部", "kind", PRED_ORDER) + "\n\n"
                     "- **重复框**：框准了一个已经被别的框配走的标注（同一个 logo 框了两次）\n"
                     "- **框了一部分**：预测框 80% 以上落在某个标注框里，但只框住了它的一部分（比如只框了文字或只框了图形）\n"
                     "- **定位错**：压在某个标注附近但没框准（0.1 ≤ IoU < 0.5）\n"
                     "- **背景误检**：附近没有任何标注（IoU < 0.1），把背景或没标注的东西当成了 logo"))
    sections.append(("3. 检出的框准不准", md_table([
        ("检出框与标注的 IoU 中位数", f"{np.median(ious):.3f}", ""),
        ("IoU ≥ 0.75 的比例", f"{(ious >= 0.75).mean() * 100:.1f}%", "严格标准下也算对"),
        ("IoU ≥ 0.9 的比例", f"{(ious >= 0.9).mean() * 100:.1f}%", "几乎完全重合"),
    ])))
    for title, by, order in (("4a. 按尺寸", "area_bin", ["small", "medium", "large"]),
                             ("4b. 按是否截断", "截断", ["完整", "截断"]),
                             ("4c. 按图内框数（密度）", "n_boxes_bin", None),
                             ("4d. 按大类", "supercat", None),
                             ("4e. 按长宽比", "长宽比", ["竖长 (<0.67)", "近方 (0.67–1.5)", "横长 (1.5–3)", "很扁 (>3)"])):
        sections.append((title, share_table(gt_out, by, "outcome", GT_ORDER, order)))

    # 品牌
    p_super = classes["supercat"].value_counts(normalize=True)
    chance_same = float((p_super ** 2).sum())
    wrong = gt_out[gt_out["brand"].isin(["错成同一大类", "错成别的大类"])]
    same_share = (wrong["brand"] == "错成同一大类").mean() if len(wrong) else float("nan")
    brand_key = classes["brand_key"].to_dict()
    variant_share = (wrong["class_id"].map(brand_key).to_numpy() == wrong["pred_class"].map(brand_key).to_numpy()).mean() if len(wrong) else float("nan")
    sections.append(("5. 品牌：框对了，认对了吗（3000 类口径）", share_table(gt_out.assign(全部="全部"), "全部", "brand", BRAND_ORDER) + "\n\n" + md_table([
        ("认错的框里，错成同一大类的比例", f"{same_share * 100:.1f}%", f"随机乱猜时约 {chance_same * 100:.1f}%（按 9 个大类的品牌数算）"),
        ("认错的框里，错成同一品牌另一款的比例", f"{variant_share * 100:.1f}%", "比如 icbc-2 认成 icbc-1；3,000 类里有 251 类属于这种多款式品牌"),
    ]) + "\n\n- 「品牌」取位置对上（IoU ≥ 0.5）的品牌预测里分数最高的那个，不设分数阈值。"))
    sections.append(("5a. 品牌：训练时见没见过", share_table(gt_out, "品牌在训练集里", "brand", BRAND_ORDER, ["见过", "没见过"]) +
                     f"\n\n- 训练集：`{train_ann_name}`。没见过的品牌不可能认对，它们的比例就是这个训练集下认品牌成绩的硬上限。"))
    pairs = (wrong.assign(true=wrong["class_id"].map(name_of), guess=wrong["pred_class"].map(name_of),
                          ts=wrong["class_id"].map(super_of), gs=wrong["pred_class"].map(super_of))
             .groupby(["true", "guess", "ts", "gs"]).size().sort_values(ascending=False).head(15))
    sections.append(("5b. 最常见的认错（标注 → 模型猜的）", md_table(
        [(t, g, f"{ts} → {gs}", str(n)) for (t, g, ts, gs), n in pairs.items()],
        headers=("标注的品牌", "模型猜的品牌", "大类", "次数"))))
    bins = pd.cut(pred_out["score"], [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    cal = pred_out.assign(bin=bins.astype(str), ok=pred_out["kind"] == "正确").groupby("bin")["ok"].agg(["size", "mean"])
    sections.append(("6. 分数可信吗（阈值以上的框，按分数分段的正确率）", md_table(
        [(b, f"{int(r['size']):,}", f"{r['mean'] * 100:.1f}%") for b, r in cal.iterrows()], headers=("分数段", "框数", "其中正确"))))

    # ---- 例图拼图 ----
    sheet_dir = P.artifacts / "analysis" / f"box_{run}"
    data_root = P.dataset
    path_of = images["rel_path"].to_dict()
    sheets = []

    def pick(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.iloc[rng.permutation(len(df))[:n]] if len(df) > n else df

    for outcome, note in (("完全漏检", "绿 = 漏掉的标注；红 = 离它最近的预测框（可能很远或分数很低）"),
                          ("定位不准", "绿 = 标注；红 = 压在附近但没框准的预测"),
                          ("分数太低", "绿 = 标注；红 = 框准了但分数低于阈值的预测")):
        sub = pick(gt_out[gt_out["outcome"] == outcome], args.examples)
        items = [dict(path=data_root / path_of[r.image_id], gt_box=[r.x1, r.y1, r.x2, r.y2], pred_box=r.best_pred_box,
                      label=f"ann {r.ann_id} | {r.class_name[:22]}\n{r.area_bin}, {r['截断']}, IoU {r.best_iou_any:.2f}, score {r.best_pred_score:.3f}")
                 for _, r in sub.iterrows()]
        out = sheet_dir / f"sheet_gt_{outcome}.jpg"
        contact_sheet(items, out, f"{run} | 标注框：{outcome}（{note}）")
        sheets.append((f"标注框：{outcome}", out))
    fp = pred_out[pred_out["kind"] == "背景误检"].sort_values("score", ascending=False).head(args.examples)
    items = [dict(path=data_root / path_of[r.image_id], gt_box=gt_boxes_of.get(r.image_id), pred_box=[r.x1, r.y1, r.x2, r.y2],
                  label=f"image {r.image_id} | score {r.score:.3f}\n附近没有任何标注") for _, r in fp.iterrows()]
    out = sheet_dir / "sheet_pred_背景误检_高分.jpg"
    contact_sheet(items, out, f"{run} | 分数最高的背景误检（红 = 预测，绿 = 这张图的全部标注；看是否其实是漏标的 logo）")
    sheets.append(("分数最高的背景误检", out))
    part = pick(pred_out[pred_out["kind"] == "框了一部分"], args.examples)
    items = [dict(path=data_root / path_of[r.image_id], gt_box=gt_boxes_of.get(r.image_id), pred_box=[r.x1, r.y1, r.x2, r.y2],
                  label=f"image {r.image_id} | score {r.score:.3f}\n只框住了标注的一部分") for _, r in part.iterrows()]
    out = sheet_dir / "sheet_pred_框了一部分.jpg"
    contact_sheet(items, out, f"{run} | 只框了 logo 的一部分（红 = 预测，绿 = 标注）")
    sheets.append(("只框了 logo 的一部分", out))
    for kind in ("错成同一大类", "错成别的大类"):
        sub = pick(gt_out[gt_out["brand"] == kind], args.examples)
        items = [dict(path=data_root / path_of[r.image_id], gt_box=[r.x1, r.y1, r.x2, r.y2], pred_box=None,
                      label=f"标注 {r.class_name[:20]}\n猜成 {name_of.get(r.pred_class, '?')[:20]} ({super_of.get(r.pred_class, '?')})")
                 for _, r in sub.iterrows()]
        out = sheet_dir / f"sheet_brand_{kind}.jpg"
        contact_sheet(items, out, f"{run} | 品牌{kind}（绿 = 标注）")
        sheets.append((f"品牌{kind}", out))
    sections.append(("7. 给人工看的例图", "\n".join([
        "每类随机抽 " + str(args.examples) + f" 个（随机种子 {SEED}），背景误检取分数最高的。绿框 = 标注，红框 = 模型的预测。",
        "", *[f"- {name}：`{path.relative_to(P.artifacts.parent).as_posix()}`" for name, path in sheets], "",
        "**这一步需要人来做**：逐张判断每个错误的原因并记录，例如——",
        "- 漏检 / 定位不准：logo 太小、太模糊、被遮挡、形状特殊，还是**标注本身有问题**（框画错、漏标）",
        "- 高分背景误检：是不是**图里确实有 logo 但没被标注**（那就不是模型的错）",
        "- 品牌认错：两个品牌是否本来就长得很像，还是标注的品牌名有误（项目早期发现过 13,347 个框的品牌名与目录不一致）",
        "统计只能说明错误分布在哪里，原因要看图才能下结论。",
    ])))
    sections.append(("口径说明", "\n".join([
        "- 评测集只用 val2k_repr（代表性抽样），与 `BASELINE_EVALUATION.md` 的总体指标同一批图。",
        "- 分数阈值是在 val2k_repr 上选的，只用于本分析的归类；正式指标仍以 AP / AR 为准。",
        "- 标注框与预测框的配对用 COCO 式贪心匹配（按分数从高到低），IoU 门槛 0.5。",
        "- 第 5 节的品牌判定不设分数阈值：位置对上的品牌预测里取分数最高的一个。",
    ])))

    body = [f"# S9 框级分析：{run}", "", "> 由 `scripts/s9_dino_box_analysis.py` 自动生成，重跑即刷新。", ""]
    for title, content in sections:
        body += [f"## {title}", "", content, ""]
    report = PROJECT_DIR / "report" / f"s9_dino_{run}_box_analysis.md"
    report.write_text("\n".join(body), encoding="utf-8")
    for title, content in sections:
        print(f"\n## {title}\n\n{content}")
    print(f"\n报告已写入 {report.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
