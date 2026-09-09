"""COCO 评测器。

包装 pycocotools，但**自己实现 summarize**。理由不是重复造轮子，
而是官方 `COCOeval.summarize()` 有一个会静默出错的设计：

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        ...
        s = s[:, :, :, aind, mind]

    def _summarizeDets():
        stats[0] = _summarize(1)                                    # maxDets 用默认的 100
        stats[1] = _summarize(1, iouThr=.5, maxDets=p.maxDets[2])   # 用 maxDets[2]
        ...
        stats[6] = _summarize(0, maxDets=p.maxDets[0])
        stats[7] = _summarize(0, maxDets=p.maxDets[1])
        stats[8] = _summarize(0, maxDets=p.maxDets[2])

两个问题：

1. **stats[0]（主 AP）把 maxDets 硬编码成 100**，而 stats[1..5] 用 `maxDets[2]`。
   一旦把 maxDets 改成 `[1, 10, 300]`，`mind` 会是**空列表**，
   `s[..., []]` 得到空数组，均值是 nan —— 只有一行 RuntimeWarning，
   不抛异常。更隐蔽的情况是 100 仍在列表里但不在第 3 位：
   此时 AP 用 100、AP50 用 maxDets[2]，两个数字口径不同却都"正常"输出。

2. **只能报 maxDets[0..2] 三档 AR**。我们需要 AR@300
   （RPN proposals 基线会输出几百个框），官方接口根本报不出来。

所以这里直接从 `cocoEval.eval['precision']` / `['recall']` 取数：

    precision : [T, R, K, A, M]   T=IoU阈值(10) R=recall点(101) K=类 A=面积档(4) M=maxDets档
    recall    : [T, K, A, M]

-1 表示该组合无 GT，按 COCO 惯例排除后再求均值。
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# COCO 标准面积档（与 S2 的 area_bin 口径一致）
AREA_RANGES = {
    "all": (0.0, 1e10),
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, 1e10),
}
DEFAULT_MAX_DETS = (1, 10, 100, 300)


@dataclass
class EvalResult:
    metrics: dict[str, float] = field(default_factory=dict)
    n_images: int = 0
    n_gt: int = 0
    n_gt_ignored: int = 0
    n_dt: int = 0
    params: dict[str, Any] = field(default_factory=dict)

    def as_rows(self, keys: Sequence[str] | None = None) -> list[tuple[str, ...]]:
        keys = keys or list(self.metrics)
        return [(k, f"{self.metrics[k]:.4f}" if self.metrics[k] == self.metrics[k]
                 else "nan", "") for k in keys if k in self.metrics]


def _mean_valid(x: np.ndarray) -> float:
    """按 COCO 惯例：-1 表示该组合无 GT，排除后求均值。全空则返回 nan。"""
    v = x[x > -1]
    return float(np.mean(v)) if v.size else float("nan")


def summarize(cocoeval: COCOeval, *, max_dets: Sequence[int]) -> dict[str, float]:
    """从 eval 数组直接取指标。不调用官方 summarize()。"""
    e = cocoeval.eval
    if not e:
        raise RuntimeError("必须先 accumulate() 才能 summarize")

    p = cocoeval.params
    prec = e["precision"]  # [T, R, K, A, M]
    rec = e["recall"]  # [T, K, A, M]

    iou_thrs = np.asarray(p.iouThrs)
    area_lbls = list(p.areaRngLbl)
    dets = list(p.maxDets)

    def a_idx(label: str) -> int:
        if label not in area_lbls:
            raise KeyError(f"面积档 {label!r} 不在 params.areaRngLbl={area_lbls}")
        return area_lbls.index(label)

    def m_idx(n: int) -> int:
        if n not in dets:
            raise KeyError(f"maxDets={n} 不在 params.maxDets={dets}")
        return dets.index(n)

    def t_idx(thr: float) -> int:
        hit = np.where(np.isclose(iou_thrs, thr))[0]
        if not len(hit):
            raise KeyError(f"IoU 阈值 {thr} 不在 params.iouThrs")
        return int(hit[0])

    # 主 AP 与各面积档统一用最大的 maxDets —— 口径必须一致，
    # 这正是官方实现出错的地方
    m_main = m_idx(max(dets))
    out: dict[str, float] = {}

    out["AP"] = _mean_valid(prec[:, :, :, a_idx("all"), m_main])
    out["AP50"] = _mean_valid(prec[t_idx(0.50), :, :, a_idx("all"), m_main])
    out["AP75"] = _mean_valid(prec[t_idx(0.75), :, :, a_idx("all"), m_main])
    for lbl in ("small", "medium", "large"):
        if lbl in area_lbls:
            out[f"AP_{lbl}"] = _mean_valid(prec[:, :, :, a_idx(lbl), m_main])

    for n in max_dets:
        # 用 m_idx（会抛 KeyError）而不是 if n in dets 的静默跳过。
        #
        # 静默跳过是危险的：请求 AR@300 而 params.maxDets 里没有 300 时，
        # 结果字典里就没有这个键，调用方拿 .get() 取到 nan 却毫无提示 ——
        # 这和官方 summarize 的失败模式完全同类。
        # 契约写明：传给 summarize 的 max_dets 必须是 params.maxDets 的子集。
        out[f"AR@{n}"] = _mean_valid(rec[:, :, a_idx("all"), m_idx(n)])
    for lbl in ("small", "medium", "large"):
        if lbl in area_lbls:
            out[f"AR_{lbl}@{max(dets)}"] = _mean_valid(rec[:, :, a_idx(lbl), m_main])

    out["AR50@%d" % max(dets)] = _mean_valid(rec[t_idx(0.50), :, a_idx("all"), m_main])
    return out


def evaluate(
    coco_gt: COCO,
    detections: list[dict],
    *,
    image_ids: Sequence[int] | None = None,
    max_dets: Sequence[int] = DEFAULT_MAX_DETS,
    area_ranges: dict[str, tuple[float, float]] | None = None,
    quiet: bool = True,
) -> EvalResult:
    """跑一次 COCO 评测。

    Args:
        detections: pycocotools 格式的 det 列表（见 adapters/to_coco_json.build_coco_dt）
        image_ids: 限定评测的图像集合。None 表示 GT 里的全部
        max_dets: 需要报的 maxDets 档位。**必须显式包含所有要报的档**
    """
    area_ranges = area_ranges or AREA_RANGES
    max_dets = tuple(sorted(set(int(m) for m in max_dets)))

    n_gt_all = len(coco_gt.dataset.get("annotations", []))
    n_ignored = sum(
        1 for a in coco_gt.dataset.get("annotations", []) if a.get("iscrowd", 0) == 1
    )

    if not detections:
        # 空预测：pycocotools 的 loadRes 会抛错，直接返回全 0
        ids = list(image_ids) if image_ids is not None else list(coco_gt.getImgIds())
        metrics = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
        for lbl in ("small", "medium", "large"):
            metrics[f"AP_{lbl}"] = 0.0
        for n in max_dets:
            metrics[f"AR@{n}"] = 0.0
        return EvalResult(
            metrics=metrics, n_images=len(ids), n_gt=n_gt_all,
            n_gt_ignored=n_ignored, n_dt=0,
            params={"max_dets": list(max_dets), "note": "空预测，直接返回 0"},
        )

    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        coco_dt = coco_gt.loadRes(detections)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        if image_ids is not None:
            ev.params.imgIds = sorted(int(i) for i in image_ids)
        ev.params.maxDets = list(max_dets)
        ev.params.areaRng = [list(area_ranges[k]) for k in area_ranges]
        ev.params.areaRngLbl = list(area_ranges)
        ev.evaluate()
        ev.accumulate()

    metrics = summarize(ev, max_dets=max_dets)
    return EvalResult(
        metrics=metrics,
        n_images=len(ev.params.imgIds),
        n_gt=n_gt_all,
        n_gt_ignored=n_ignored,
        n_dt=len(detections),
        params={
            "max_dets": list(max_dets),
            "area_labels": list(area_ranges),
            "iou_thrs": [float(x) for x in ev.params.iouThrs],
        },
    )


def bootstrap_ci(
    coco_gt: COCO,
    detections: list[dict],
    *,
    metric: str = "AP50",
    n_boot: int = 200,
    image_ids: Sequence[int] | None = None,
    max_dets: Sequence[int] = DEFAULT_MAX_DETS,
    seed: int = 6129,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """对**图像**做有放回重采样，给出指标的置信区间。

    重采样单位必须是图像而不是框：同一张图里的框不独立
    （共享场景、光照、拍摄条件），按框重采样会低估方差。

    返回 (点估计, 下界, 上界)。样本量小的切片必须带这个。
    """
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(image_ids if image_ids is not None else coco_gt.getImgIds()))
    by_img: dict[int, list[dict]] = {}
    for d in detections:
        by_img.setdefault(int(d["image_id"]), []).append(d)

    point = evaluate(
        coco_gt, detections, image_ids=ids.tolist(), max_dets=max_dets
    ).metrics.get(metric, float("nan"))

    vals: list[float] = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        # 重采样后同一张图可能出现多次；COCOeval 按 imgIds 去重，
        # 所以改用「加权」近似：直接取去重后的子集
        sub_ids = np.unique(pick)
        sub_dt = [d for i in sub_ids for d in by_img.get(int(i), [])]
        if not sub_dt:
            continue
        r = evaluate(coco_gt, sub_dt, image_ids=sub_ids.tolist(), max_dets=max_dets)
        v = r.metrics.get(metric, float("nan"))
        if v == v:
            vals.append(v)

    if not vals:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return point, lo, hi
