"""切片评测器的回归测试。

用手工构造的极小 GT 做单元测试，与 s6_slice_eval.py 在真实
val_hard_pool（1,230 图 / 2,099 框）上的端到端验证互补。

重点守两条：
  * iscrowd 视图的守恒性（非成员被忽略，总数不变）
  * 残差切片的定义自洽（是基切片的真子集，且与 T1 无交集）
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logodet.eval.slice_eval import (  # noqa: E402
    SliceSpec,
    check_partition,
    default_slices,
    evaluate_slices,
    make_slice_gt,
    slice_members,
    slices_to_frame,
)


def _tables():
    """5 框跨三个尺寸档、两种密度，其中 2 个是截断框。"""
    images = pd.DataFrame(
        {
            "image_id": [1, 2, 3],
            "n_boxes": [2, 2, 1],
            "supercat": ["Food", "Food", "Clothes"],
        }
    )
    ann = pd.DataFrame(
        {
            "ann_id": [10, 11, 12, 13, 14],
            "image_id": [1, 1, 2, 2, 3],
            "is_hard_t1": [True, False, False, True, False],
            "area_bin": ["small", "large", "medium", "large", "large"],
        }
    )
    return images, ann


def _gt_dict() -> dict:
    return {
        "info": {},
        "licenses": [],
        "images": [
            {"id": 1, "file_name": "1.jpg", "width": 400, "height": 300},
            {"id": 2, "file_name": "2.jpg", "width": 400, "height": 300},
            {"id": 3, "file_name": "3.jpg", "width": 400, "height": 300},
        ],
        "categories": [{"id": 1, "name": "logo", "supercategory": "logo"}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1,
             "bbox": [10.0, 10.0, 20.0, 20.0], "area": 400.0, "iscrowd": 0},
            {"id": 11, "image_id": 1, "category_id": 1,
             "bbox": [100.0, 100.0, 150.0, 120.0], "area": 18000.0, "iscrowd": 0},
            {"id": 12, "image_id": 2, "category_id": 1,
             "bbox": [50.0, 50.0, 60.0, 60.0], "area": 3600.0, "iscrowd": 0},
            {"id": 13, "image_id": 2, "category_id": 1,
             "bbox": [200.0, 30.0, 150.0, 130.0], "area": 19500.0, "iscrowd": 0},
            {"id": 14, "image_id": 3, "category_id": 1,
             "bbox": [20.0, 20.0, 200.0, 150.0], "area": 30000.0, "iscrowd": 0},
        ],
    }


def _coco(d: dict) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO()
        c.dataset = d
        c.createIndex()
    return c


def _perfect_dt(d: dict) -> list[dict]:
    return [
        {"image_id": a["image_id"], "category_id": 1,
         "bbox": list(a["bbox"]), "score": 1.0}
        for a in d["annotations"]
    ]


# ---------------------------------------------------------------------------
# 成员计算
# ---------------------------------------------------------------------------


def test_slice_members_ann_level():
    images, ann = _tables()
    m = slice_members(ann, images, default_slices())
    assert m["T1_truncated"] == {10, 13}
    assert m["SIZE_small"] == {10}
    assert m["SIZE_medium"] == {12}
    assert m["SIZE_large"] == {11, 13, 14}


def test_slice_members_image_level_condition():
    """密度是图级属性，但 iscrowd 是框级 —— merge 后在框上求值。"""
    images, ann = _tables()
    m = slice_members(ann, images, default_slices())
    # image 1,2 各 2 框 → DENSITY_2；image 3 单框 → DENSITY_1
    assert m["DENSITY_2"] == {10, 11, 12, 13}
    assert m["DENSITY_1"] == {14}


def test_clean_excludes_all_hard_axes():
    images, ann = _tables()
    m = slice_members(ann, images, default_slices())
    # CLEAN = 非 T1 且非 small 且非 5+ → {11, 12, 14}
    assert m["CLEAN"] == {11, 12, 14}
    assert not (m["CLEAN"] & m["T1_truncated"])
    assert not (m["CLEAN"] & m["SIZE_small"])


def test_residual_slice_is_strict_subset_without_t1():
    images, ann = _tables()
    m = slice_members(ann, images, default_slices())
    assert m["SIZE_large_noT1"] <= m["SIZE_large"]
    assert not (m["SIZE_large_noT1"] & m["T1_truncated"])
    # SIZE_large={11,13,14}，其中 13 是 T1 → 残差={11,14}
    assert m["SIZE_large_noT1"] == {11, 14}


def test_residual_density_slice():
    images, ann = _tables()
    m = slice_members(ann, images, default_slices())
    assert m["DENSITY_1_noT1"] == {14}


# ---------------------------------------------------------------------------
# 切片视图（iscrowd 机制）
# ---------------------------------------------------------------------------


def test_make_slice_gt_preserves_total_count():
    """视图只改 iscrowd，绝不删框 —— 删了会让匹配非成员的 det 变假阳。"""
    d = _gt_dict()
    sl = make_slice_gt(d, {10, 13})
    assert len(sl["annotations"]) == len(d["annotations"])


def test_make_slice_gt_marks_non_members_ignored():
    d = _gt_dict()
    sl = make_slice_gt(d, {10, 13})
    flags = {a["id"]: a["iscrowd"] for a in sl["annotations"]}
    assert flags == {10: 0, 11: 1, 12: 1, 13: 0, 14: 1}


def test_make_slice_gt_does_not_mutate_input():
    d = _gt_dict()
    _ = make_slice_gt(d, {10})
    assert all(a["iscrowd"] == 0 for a in d["annotations"])


def test_empty_slice_marks_everything_ignored():
    d = _gt_dict()
    sl = make_slice_gt(d, set())
    assert all(a["iscrowd"] == 1 for a in sl["annotations"])


# ---------------------------------------------------------------------------
# 划分检查
# ---------------------------------------------------------------------------


def test_check_partition_accepts_valid_partition():
    images, ann = _tables()
    m = slice_members(ann, images, default_slices())
    all_ids = set(ann["ann_id"].tolist())
    ok, st = check_partition(m, all_ids, ["SIZE_small", "SIZE_medium", "SIZE_large"])
    assert ok
    assert st["missing"] == 0 and st["extra"] == 0 and st["pairwise_overlap"] == 0


def test_check_partition_detects_overlap():
    m = {"a": {1, 2, 3}, "b": {3, 4}}
    ok, st = check_partition(m, {1, 2, 3, 4}, ["a", "b"])
    assert not ok
    assert st["pairwise_overlap"] == 1


def test_check_partition_detects_missing():
    m = {"a": {1, 2}, "b": {3}}
    ok, st = check_partition(m, {1, 2, 3, 4}, ["a", "b"])
    assert not ok
    assert st["missing"] == 1


# ---------------------------------------------------------------------------
# 逐切片评测
# ---------------------------------------------------------------------------


def test_perfect_prediction_scores_one_on_every_slice():
    """切片视图本身不能引入偏差。"""
    d = _gt_dict()
    images, ann = _tables()
    specs = default_slices()
    m = slice_members(ann, images, specs)
    res = evaluate_slices(d, _perfect_dt(d), m, specs,
                          min_ann_for_reliable=1, ci_only_when_unreliable=True,
                          n_boot=0)
    for r in res:
        if r.n_ann:
            assert r.metrics["AP"] == pytest.approx(1.0, abs=1e-6), r.name


def test_slice_only_evaluates_images_with_members():
    """只在含成员框的图上评测，否则 AR 分母混入无关图。"""
    d = _gt_dict()
    images, ann = _tables()
    specs = [SliceSpec("only_img3", "test", lambda a: a["ann_id"] == 14)]
    m = slice_members(ann, images, specs)
    res = evaluate_slices(d, _perfect_dt(d), m, specs, min_ann_for_reliable=1,
                          n_boot=0)
    assert res[0].n_img == 1
    assert res[0].n_ann == 1


def test_unreliable_flag_and_ci():
    d = _gt_dict()
    images, ann = _tables()
    specs = default_slices()
    m = slice_members(ann, images, specs)
    # 门槛设 100 → 全部切片都不可靠 → 全部应带 CI
    res = evaluate_slices(d, _perfect_dt(d), m, specs,
                          min_ann_for_reliable=100, ci_metrics=("AP50",),
                          n_boot=5, ci_only_when_unreliable=True)
    for r in res:
        assert not r.reliable
        if r.n_ann:
            assert "AP50" in r.ci


def test_reliable_slices_skip_ci():
    d = _gt_dict()
    images, ann = _tables()
    specs = default_slices()
    m = slice_members(ann, images, specs)
    res = evaluate_slices(d, _perfect_dt(d), m, specs,
                          min_ann_for_reliable=1, ci_metrics=("AP50",),
                          n_boot=5, ci_only_when_unreliable=True)
    for r in res:
        if r.n_ann:
            assert r.reliable
            assert not r.ci  # 样本充足就不算 CI（很贵）


def test_empty_slice_returns_nan_not_crash():
    d = _gt_dict()
    images, ann = _tables()
    specs = [SliceSpec("nothing", "test", lambda a: a["ann_id"] < 0)]
    m = slice_members(ann, images, specs)
    res = evaluate_slices(d, _perfect_dt(d), m, specs, min_ann_for_reliable=1,
                          n_boot=0)
    assert res[0].n_ann == 0
    assert np.isnan(res[0].metrics["AP"])


def test_slices_to_frame_has_required_columns():
    d = _gt_dict()
    images, ann = _tables()
    specs = default_slices()
    m = slice_members(ann, images, specs)
    res = evaluate_slices(d, _perfect_dt(d), m, specs, min_ann_for_reliable=1,
                          n_boot=0)
    df = slices_to_frame(res)
    for c in ("slice", "axis", "n_ann", "n_img", "reliable", "AP", "AP50"):
        assert c in df.columns


# ---------------------------------------------------------------------------
# 分辨力：这是切片评测存在的理由
# ---------------------------------------------------------------------------


def test_slice_detects_targeted_degradation():
    """只让某个切片的预测变差，该切片的指标必须显著更低。

    没有这条性质，「难例上表现更差」这个结论就无从得出。
    """
    d = _gt_dict()
    images, ann = _tables()
    specs = default_slices()
    m = slice_members(ann, images, specs)
    t1 = m["T1_truncated"]

    # T1 成员的预测整体平移 40 px（IoU 大幅下降），其余完美
    dt = []
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        if int(a["id"]) in t1:
            x, y = x + 40, y + 40
        dt.append({"image_id": a["image_id"], "category_id": 1,
                   "bbox": [x, y, w, h], "score": 1.0})

    res = {r.name: r.metrics.get("AP", float("nan"))
           for r in evaluate_slices(d, dt, m, specs, min_ann_for_reliable=1,
                                    n_boot=0)}
    assert res["T1_truncated"] < res["CLEAN"]


def test_residual_slice_isolates_contamination():
    """残差切片必须能把交叉污染拆出来。

    真实数据上实测（S6-L8，val_hard_pool）：SIZE_large 含 37.5% 的 T1，
    只给 T1 加 ε=0.30 抖动时 SIZE_large 掉到 0.5808，
    而 SIZE_large_noT1 回到 0.9587（CLEAN 是 0.9583）—— 几乎完全恢复。

    这里只断言**方向**（残差 > 基切片），不断言恢复到满分。原因见
    test_displaced_detection_is_fp_not_absorbed：本测试用的是硬平移 40px，
    T1 的预测已完全离开原框，匹配不到任何 GT（包括被忽略的那个），
    于是变成纯假阳，会污染所有切片。真实数据里的相对抖动仍与原框重叠、
    能被 ignored GT 吸收，所以才能完全恢复。
    """
    d = _gt_dict()
    images, ann = _tables()
    specs = default_slices()
    m = slice_members(ann, images, specs)
    t1 = m["T1_truncated"]

    dt = []
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        if int(a["id"]) in t1:
            x, y = x + 40, y + 40
        dt.append({"image_id": a["image_id"], "category_id": 1,
                   "bbox": [x, y, w, h], "score": 1.0})

    res = {r.name: r.metrics.get("AP", float("nan"))
           for r in evaluate_slices(d, dt, m, specs, min_ann_for_reliable=1,
                                    n_boot=0)}
    assert res["SIZE_large"] < 1.0
    assert res["SIZE_large_noT1"] > res["SIZE_large"], "残差切片必须高于基切片"


def test_detection_matching_ignored_gt_is_absorbed():
    """匹配上被忽略 GT 的 det 会被吸收 —— 既不计 TP 也不计 FP。

    这是切片评测能成立的前提：检测器正确找到了一个不属于本切片的 logo，
    不该因此被罚。
    """
    d = _gt_dict()
    # 只把 ann 14 当成员，其余全忽略
    sl = make_slice_gt(d, {14})
    gt = _coco(sl)
    from logodet.eval.coco_eval import evaluate

    # 完美预测全部 5 个框：4 个匹配 ignored GT（应被吸收），1 个匹配成员
    r = evaluate(gt, _perfect_dt(d), image_ids=[3])
    assert r.metrics["AP"] == pytest.approx(1.0, abs=1e-6)


def test_displaced_detection_is_fp_not_absorbed():
    """**完全偏离**的 det 不会被 ignored GT 吸收，而是计为假阳。

    吸收的前提是 IoU 达到匹配阈值。det 若被整体平移到与原框不重叠的位置，
    它匹配不到任何 GT（哪怕附近有个被忽略的 GT），于是成为假阳 ——
    并且这个假阳会污染**每一个**包含该图的切片，不只是它"本该属于"的那个。

    这条对解读 S7 很重要：一个把框画到完全错误位置的模型，
    其错误会扩散到所有切片，而不是只体现在对应的难例切片上。
    """
    d = _gt_dict()
    from logodet.eval.coco_eval import evaluate

    # 只把 image 3 的 ann 14 当成员
    sl = make_slice_gt(d, {14})
    gt = _coco(sl)
    # image 3 上：给成员一个完美 det，再加一个完全偏离的 det
    dt = [
        {"image_id": 3, "category_id": 1, "bbox": [20.0, 20.0, 200.0, 150.0],
         "score": 1.0},
        {"image_id": 3, "category_id": 1, "bbox": [300.0, 250.0, 50.0, 40.0],
         "score": 0.9},
    ]
    r = evaluate(gt, dt, image_ids=[3])
    # 高分假阳排在真阳之后（0.9 < 1.0），按 S5-K6 的结论 AP 不受影响
    assert r.metrics["AP"] == pytest.approx(1.0, abs=1e-6)

    # 但若假阳分数更高，就会拉低 AP —— 证明它确实被计为 FP 而非被吸收
    dt_high = [
        {"image_id": 3, "category_id": 1, "bbox": [20.0, 20.0, 200.0, 150.0],
         "score": 0.5},
        {"image_id": 3, "category_id": 1, "bbox": [300.0, 250.0, 50.0, 40.0],
         "score": 0.9},
    ]
    r_high = evaluate(gt, dt_high, image_ids=[3])
    assert r_high.metrics["AP"] < 1.0
