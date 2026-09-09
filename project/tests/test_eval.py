"""评测器的回归测试。

评测器是全项目最不能出错的一环 —— 它错了，所有 baseline 数字都是废的，
而且错误在指标上表现为"效果偏低"，极易被误判成模型问题。

这里用**手工构造的极小 GT**（3 图 5 框）做单元测试，与 s5_l0_selfcheck.py
在真实 2,000 图上的端到端自校验互补：前者定位精确，后者覆盖真实分布。
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logodet.eval.coco_eval import (  # noqa: E402
    AREA_RANGES,
    DEFAULT_MAX_DETS,
    _mean_valid,
    bootstrap_ci,
    evaluate,
)
from logodet.eval.l0_selfcheck import (  # noqa: E402
    SyntheticSpec,
    jitter_boxes,
    make_synthetic_dt,
)


def _toy_gt(n_img: int = 3) -> dict:
    """3 图 5 框的玩具 GT。框尺寸刻意跨越 small/medium/large 三档。"""
    images = [
        {"id": i, "file_name": f"{i}.jpg", "width": 400, "height": 300}
        for i in range(1, n_img + 1)
    ]
    # (image_id, x, y, w, h)
    boxes = [
        (1, 10, 10, 20, 20),    # area 400   → small
        (1, 100, 100, 60, 60),  # area 3600  → medium
        (2, 50, 50, 120, 120),  # area 14400 → large
        (2, 200, 30, 30, 30),   # area 900   → small
        (3, 20, 20, 150, 100),  # area 15000 → large
    ]
    anns = [
        {
            "id": k + 1,
            "image_id": iid,
            "category_id": 1,
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(w * h),
            "iscrowd": 0,
        }
        for k, (iid, x, y, w, h) in enumerate(boxes)
        if iid <= n_img
    ]
    return {
        "info": {},
        "licenses": [],
        "images": images,
        "categories": [{"id": 1, "name": "logo", "supercategory": "logo"}],
        "annotations": anns,
    }


def _coco(d: dict) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO()
        c.dataset = d
        c.createIndex()
    return c


def _perfect_dt(d: dict, score: float = 1.0) -> list[dict]:
    return [
        {
            "image_id": a["image_id"],
            "category_id": a["category_id"],
            "bbox": list(a["bbox"]),
            "score": score,
        }
        for a in d["annotations"]
        if a.get("iscrowd", 0) == 0
    ]


# ---------------------------------------------------------------------------
# 基本正确性
# ---------------------------------------------------------------------------


def test_perfect_prediction_gives_ap_one():
    """最重要的一条：GT 原样当预测必须得满分。

    它同时是 xyxy↔xywh 转换的探针 —— 任何格式不一致都会让这条失败。
    """
    d = _toy_gt()
    r = evaluate(_coco(d), _perfect_dt(d))
    assert r.metrics["AP"] == pytest.approx(1.0, abs=1e-6)
    assert r.metrics["AP50"] == pytest.approx(1.0, abs=1e-6)
    assert r.metrics["AP75"] == pytest.approx(1.0, abs=1e-6)
    assert r.metrics["AR@100"] == pytest.approx(1.0, abs=1e-6)


def test_empty_prediction_gives_zero():
    d = _toy_gt()
    r = evaluate(_coco(d), [])
    assert r.metrics["AP"] == 0.0
    assert r.metrics["AR@100"] == 0.0
    assert r.n_dt == 0


def test_reports_ar_at_300():
    """官方 summarize 报不出 AR@300，我方必须能报。"""
    d = _toy_gt()
    r = evaluate(_coco(d), _perfect_dt(d), max_dets=(1, 10, 100, 300))
    assert "AR@300" in r.metrics
    assert r.metrics["AR@300"] == pytest.approx(1.0, abs=1e-6)


def test_nonstandard_max_dets_still_works():
    """maxDets 不含 100 时也必须正常 —— 官方实现在这里会返回 -1。"""
    d = _toy_gt()
    r = evaluate(_coco(d), _perfect_dt(d), max_dets=(1, 10, 300))
    assert r.metrics["AP"] == pytest.approx(1.0, abs=1e-6)
    assert "AR@300" in r.metrics
    assert "AR@100" not in r.metrics  # 没要求这一档就不该出现


def test_missing_max_dets_key_raises():
    """要求一个不存在的档位应当明确报错，而不是静默给 nan。"""
    from logodet.eval.coco_eval import summarize
    from pycocotools.cocoeval import COCOeval

    d = _toy_gt()
    gt = _coco(d)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(_perfect_dt(d))
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.params.maxDets = [1, 10]
        ev.evaluate()
        ev.accumulate()
    with pytest.raises(KeyError, match="maxDets"):
        summarize(ev, max_dets=(1, 10, 999))


# ---------------------------------------------------------------------------
# 排序语义
# ---------------------------------------------------------------------------


def test_low_score_false_positives_do_not_hurt_ap():
    """排在真阳之后的假阳不影响 AP —— 这是 AP 的排序语义。

    L0 首轮把这条写反了（期望 AP 下降），FAIL 之后才查清。
    """
    d = _toy_gt()
    dt = _perfect_dt(d, score=1.0)
    rng = np.random.default_rng(0)
    for im in d["images"]:
        for _ in range(20):
            dt.append(
                {
                    "image_id": im["id"],
                    "category_id": 1,
                    "bbox": [
                        float(rng.uniform(0, 300)), float(rng.uniform(0, 200)),
                        30.0, 30.0,
                    ],
                    "score": 0.01,
                }
            )
    r = evaluate(_coco(d), dt)
    assert r.metrics["AP"] == pytest.approx(1.0, abs=1e-6)


def test_high_score_false_positives_hurt_ap():
    """排在真阳之上的假阳必须显著拉低 AP。"""
    d = _toy_gt()
    dt = _perfect_dt(d, score=0.5)
    rng = np.random.default_rng(0)
    for im in d["images"]:
        for _ in range(20):
            dt.append(
                {
                    "image_id": im["id"],
                    "category_id": 1,
                    "bbox": [
                        float(rng.uniform(0, 300)), float(rng.uniform(0, 200)),
                        30.0, 30.0,
                    ],
                    "score": 0.9,  # 高于真阳
                }
            )
    r = evaluate(_coco(d), dt)
    assert r.metrics["AP"] < 0.6


def test_ar_ignores_false_positives():
    """AR 只看能否召回，不惩罚假阳。"""
    d = _toy_gt()
    dt = _perfect_dt(d, score=0.5)
    base = evaluate(_coco(d), dt).metrics["AR@100"]
    for im in d["images"]:
        dt.append({"image_id": im["id"], "category_id": 1,
                   "bbox": [0.0, 0.0, 5.0, 5.0], "score": 0.9})
    after = evaluate(_coco(d), dt).metrics["AR@100"]
    assert after == pytest.approx(base, abs=1e-6)


# ---------------------------------------------------------------------------
# iscrowd 忽略机制（S6 的切片评测靠它）
# ---------------------------------------------------------------------------


def test_iscrowd_gt_is_excluded_from_recall():
    d = _toy_gt()
    # 把 ann_id 1,2（image 1 的两个框）标为 ignore
    for a in d["annotations"]:
        a["iscrowd"] = 1 if a["id"] in (1, 2) else 0
    # 只预测 image 2、3 的框
    dt = [
        {"image_id": a["image_id"], "category_id": 1,
         "bbox": list(a["bbox"]), "score": 1.0}
        for a in d["annotations"]
        if a["iscrowd"] == 0
    ]
    r = evaluate(_coco(d), dt)
    assert r.n_gt_ignored == 2
    # 被忽略的 GT 不进 recall 分母 → 仍是满分
    assert r.metrics["AP"] == pytest.approx(1.0, abs=1e-6)


def test_without_iscrowd_missing_gt_counts_as_miss():
    d = _toy_gt()
    dt = [
        {"image_id": a["image_id"], "category_id": 1,
         "bbox": list(a["bbox"]), "score": 1.0}
        for a in d["annotations"]
        if a["id"] not in (1, 2)
    ]
    r = evaluate(_coco(d), dt)
    assert r.n_gt_ignored == 0
    assert r.metrics["AP"] < 1.0


# ---------------------------------------------------------------------------
# 面积分档
# ---------------------------------------------------------------------------


def test_area_ranges_match_coco_convention():
    assert AREA_RANGES["small"] == (0.0, 32.0**2)
    assert AREA_RANGES["medium"] == (32.0**2, 96.0**2)
    assert AREA_RANGES["large"][0] == 96.0**2


def test_per_area_metrics_present():
    d = _toy_gt()
    r = evaluate(_coco(d), _perfect_dt(d))
    for lbl in ("small", "medium", "large"):
        assert f"AP_{lbl}" in r.metrics


# ---------------------------------------------------------------------------
# 抖动与合成预测
# ---------------------------------------------------------------------------


def test_jitter_zero_is_identity():
    boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
    out = jitter_boxes(boxes, 0.0, np.random.default_rng(0), img_w=400, img_h=300)
    assert np.allclose(out, boxes)


def test_jitter_scales_with_box_size():
    """抖动幅度必须与框自身尺寸成比例，否则 AP 曲线会被尺寸分布主导。"""
    rng = np.random.default_rng(0)
    small = np.array([[0.0, 0.0, 10.0, 10.0]])
    large = np.array([[0.0, 0.0, 200.0, 200.0]])
    ds = np.abs(jitter_boxes(small, 0.2, np.random.default_rng(1),
                             img_w=400, img_h=300) - small).max()
    dl = np.abs(jitter_boxes(large, 0.2, np.random.default_rng(1),
                             img_w=400, img_h=300) - large).max()
    assert dl > ds * 5


def test_jitter_keeps_boxes_valid():
    rng = np.random.default_rng(0)
    boxes = np.array([[0.0, 0.0, 399.0, 299.0], [395.0, 295.0, 400.0, 300.0]])
    out = jitter_boxes(boxes, 0.5, rng, img_w=400, img_h=300)
    assert (out[:, 0] >= 0).all() and (out[:, 1] >= 0).all()
    assert (out[:, 2] <= 400).all() and (out[:, 3] <= 300).all()
    assert (out[:, 2] > out[:, 0]).all() and (out[:, 3] > out[:, 1]).all()


def test_synthetic_perfect_matches_gt_count():
    d = _toy_gt()
    dt = make_synthetic_dt(d, SyntheticSpec("perfect"))
    assert len(dt) == len(d["annotations"])


def test_synthetic_skips_iscrowd_gt():
    """iscrowd=1 的 GT 不该被用来造预测，否则完美预测拿不到满分。"""
    d = _toy_gt()
    d["annotations"][0]["iscrowd"] = 1
    dt = make_synthetic_dt(d, SyntheticSpec("perfect"))
    assert len(dt) == len(d["annotations"]) - 1


def test_synthetic_half_recall_has_no_per_image_floor():
    """逐框独立采样，不保证每图至少留 1 个 —— 否则单框图会全部保留。"""
    d = _toy_gt(n_img=3)
    dt = make_synthetic_dt(d, SyntheticSpec("half", keep_ratio=0.5), seed=7)
    assert 0 < len(dt) < len(d["annotations"])


def test_synthetic_is_reproducible():
    d = _toy_gt()
    a = make_synthetic_dt(d, SyntheticSpec("j", jitter=0.1), seed=1)
    b = make_synthetic_dt(d, SyntheticSpec("j", jitter=0.1), seed=1)
    assert [x["bbox"] for x in a] == [x["bbox"] for x in b]


def test_jitter_monotonically_lowers_ap():
    d = _toy_gt()
    gt = _coco(d)
    aps = []
    for eps in (0.0, 0.1, 0.3):
        dt = make_synthetic_dt(d, SyntheticSpec(f"j{eps}", jitter=eps), seed=3)
        aps.append(evaluate(gt, dt).metrics["AP"])
    assert aps[0] > aps[1] > aps[2]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def test_mean_valid_excludes_sentinel():
    assert _mean_valid(np.array([-1.0, -1.0, 0.5, 1.0])) == pytest.approx(0.75)
    assert np.isnan(_mean_valid(np.array([-1.0, -1.0])))


def test_bootstrap_ci_brackets_point_estimate():
    d = _toy_gt()
    gt = _coco(d)
    dt = make_synthetic_dt(d, SyntheticSpec("j", jitter=0.15), seed=5)
    pt, lo, hi = bootstrap_ci(gt, dt, metric="AP50", n_boot=20)
    assert lo <= pt <= hi or np.isnan(lo)
