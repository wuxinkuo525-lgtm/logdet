"""S3 划分逻辑的回归测试。

重点守住三处：
  * **簇不跨界** —— 早期版本用 groupby().indices（子组内位置）去
    .iloc 全量 Series，导致 val 只有 216 张而不是 15,865 张、
    2,995 个类在 val 缺席。这是本阶段最严重的一个 bug。
  * 每类至少 1 张 val（小类的 ceil 兜底）
  * 最大余数法配额精确等于目标（直接 round 会偏离）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logodet.splits.make_split import (  # noqa: E402
    assign_clusters,
    check_split_ratio,
    image_area_bin,
    make_split,
)
from logodet.splits.make_val_subsets import (  # noqa: E402
    _largest_remainder,
    make_val2k_repr,
    make_val_hard_pool,
)


def _fake_dataset(
    n_classes: int = 20, per_class: int = 20, dup_pairs: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """造一个小数据集。dup_pairs 指定制造多少对「字节大小相同」的重复图。"""
    rows = []
    inv_rows = []
    iid = 1
    for c in range(n_classes):
        brand = f"brand{c:03d}"
        for k in range(per_class):
            # 前 dup_pairs 对图共享同一 file_size，构成同一簇
            size = 1000 + (k // 2 if k // 2 < dup_pairs else 100 + k)
            rows.append(
                {
                    "image_id": iid,
                    "rel_path": f"Food/{brand}/{k}.jpg",
                    "brand_dir": brand,
                    "supercat": "Food",
                    "img_w": 500,
                    "img_h": 400,
                    "n_boxes": 1 + (k % 3),
                    "n_boxes_bin": ["1", "2", "3-4"][k % 3],
                }
            )
            inv_rows.append({"image_rel": f"Food/{brand}/{k}.jpg", "image_bytes": size})
            iid += 1
    return pd.DataFrame(rows), pd.DataFrame(inv_rows)


# ---------------------------------------------------------------------------
# 簇
# ---------------------------------------------------------------------------


def test_assign_clusters_groups_identical_size():
    img, inv = _fake_dataset(n_classes=1, per_class=6, dup_pairs=3)
    cid = assign_clusters(img, inv)
    sizes = pd.Series(cid).value_counts()
    # 3 对重复 → 3 个双图簇
    assert int((sizes == 2).sum()) == 3


def test_assign_clusters_raises_on_missing_bytes():
    img, inv = _fake_dataset(n_classes=1, per_class=4)
    inv = inv.iloc[:2]  # 故意缺两条
    with pytest.raises(ValueError, match="image_bytes"):
        assign_clusters(img, inv)


def test_clusters_never_span_splits():
    """本阶段最严重 bug 的回归测试。

    早期实现用 groupby().indices（子组内位置）配 .iloc（全量位置），
    结果每个类都往整表最前面几十行写。这条测试同时能抓住那类错误：
    若位置错乱，簇会被拆开、且 val 比例会严重偏离。
    """
    img, inv = _fake_dataset(n_classes=30, per_class=20, dup_pairs=5)
    out, rep = make_split(img, inv, val_ratio=0.10)
    nun = out.groupby("cluster_id")["split"].nunique()
    assert (nun == 1).all(), "存在跨界的簇"


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------


def test_split_is_a_partition():
    img, inv = _fake_dataset()
    out, _ = make_split(img, inv, val_ratio=0.10)
    assert set(out["split"].unique()) <= {"trainval", "val"}
    assert len(out) == len(img)
    assert out["image_id"].is_unique


def test_split_ratio_in_band():
    img, inv = _fake_dataset(n_classes=50, per_class=20)
    out, rep = make_split(img, inv, val_ratio=0.10)
    actual = rep.n_val / len(out)
    assert 0.08 <= actual <= 0.13, f"实际比例 {actual}"


def test_every_class_present_in_both_sides():
    img, inv = _fake_dataset(n_classes=25, per_class=20)
    out, rep = make_split(img, inv, val_ratio=0.10)
    assert rep.classes_absent_in_val == []
    assert rep.classes_absent_in_trainval == []


def test_tiny_class_still_gets_one_val_image():
    """4 张图的类，0.1*4=0.4，不兜底就会在 val 缺席。

    注意此时实际 val 比例是 25%（10 类各留 1 张 / 40 图），远高于 10% 目标。
    这不是 bug，而是「每类至少 1 张 val」这条约束的结构下限
    （n_classes/n_images = 10/40）。内部自检必须放过这种情况。
    """
    img, inv = _fake_dataset(n_classes=10, per_class=4)
    out, rep = make_split(img, inv, val_ratio=0.10)
    per = out.groupby("brand_dir")["split"].apply(lambda s: int((s == "val").sum()))
    assert (per >= 1).all(), "存在 val 数为 0 的类"
    assert rep.structural_floor_ratio == pytest.approx(10 / 40)


def test_structural_floor_is_reported():
    """结构下限要如实记录，便于报告解释为何实际比例高于目标。"""
    img, inv = _fake_dataset(n_classes=50, per_class=20)
    _, rep = make_split(img, inv, val_ratio=0.10)
    assert rep.structural_floor_ratio == pytest.approx(50 / 1000)


def test_split_is_reproducible():
    img, inv = _fake_dataset(n_classes=30, per_class=20)
    a, _ = make_split(img, inv, val_ratio=0.10)
    b, _ = make_split(img, inv, val_ratio=0.10)
    assert a["split"].tolist() == b["split"].tolist()


def test_check_split_ratio_accepts_normal_case():
    # 真实数据规模：3,000 类 / 158,654 图，val 17,216 → 10.85%
    ok, actual, upper = check_split_ratio(17216, 158654, 3000, 0.10)
    assert ok
    assert actual == pytest.approx(0.1085, abs=1e-4)


def test_check_split_ratio_accepts_structural_floor():
    """小类场景：10 类 × 4 图，每类留 1 张 → 25%，应被放过。"""
    ok, actual, upper = check_split_ratio(10, 40, 10, 0.10)
    assert ok, f"actual={actual} upper={upper}"
    assert actual == pytest.approx(0.25)


def test_check_split_ratio_rejects_too_low():
    # 类很大（结构下限 2%），但 val 只给了 1%，低于 0.10*0.8
    ok, actual, _ = check_split_ratio(10, 1000, 20, 0.10)
    assert not ok
    assert actual == pytest.approx(0.01)


def test_check_split_ratio_rejects_too_high():
    # 结构下限仅 2%，却把 60% 的图放进 val —— 纯逻辑错误，必须拦住
    ok, actual, upper = check_split_ratio(600, 1000, 20, 0.10)
    assert not ok
    assert actual == pytest.approx(0.60)
    assert upper < 0.60


# ---------------------------------------------------------------------------
# 配额分配
# ---------------------------------------------------------------------------


def test_largest_remainder_hits_target_exactly():
    for total in (100, 2000, 7):
        w = np.array([3.0, 1.0, 1.0, 5.0, 2.0])
        q = _largest_remainder(w, total)
        assert q.sum() == total, f"total={total} 得到 {q.sum()}"
        assert (q >= 0).all()


def test_largest_remainder_beats_naive_round():
    # 这组权重下直接 round 会偏离目标，最大余数法不会
    w = np.array([1.0] * 3)
    q = _largest_remainder(w, 100)
    assert q.sum() == 100
    assert sorted(q.tolist()) == [33, 33, 34]


# ---------------------------------------------------------------------------
# val 子集
# ---------------------------------------------------------------------------


def _val_frame(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "image_id": np.arange(1, n + 1),
            "supercat": rng.choice(["Food", "Clothes", "Sports"], n),
            "img_area_bin": rng.choice(["small", "medium", "large"], n, p=[0.05, 0.3, 0.65]),
            "n_boxes_bin": rng.choice(["1", "2", "3-4", "5+"], n, p=[0.85, 0.1, 0.03, 0.02]),
            "n_boxes": rng.integers(1, 4, n),
        }
    )


def test_val2k_hits_exact_target():
    val = _val_frame(600)
    sel, rep = make_val2k_repr(val, target=200)
    assert len(sel) == 200
    assert rep.n_images == 200


def test_val2k_caps_at_available():
    val = _val_frame(50)
    sel, _ = make_val2k_repr(val, target=200)
    assert len(sel) == 50


def test_val2k_is_reproducible():
    val = _val_frame(600)
    a, _ = make_val2k_repr(val, target=200)
    b, _ = make_val2k_repr(val, target=200)
    assert sorted(a["image_id"]) == sorted(b["image_id"])


def test_hard_pool_takes_all_when_short():
    val = _val_frame(200)
    ann = pd.DataFrame(
        {
            "image_id": np.repeat(val["image_id"].to_numpy()[:20], 2),
            "ann_id": np.arange(1, 41),
            "is_hard_t1": [True] * 10 + [False] * 30,
            "area_bin": ["small"] * 6 + ["large"] * 34,
        }
    )
    sel, rep = make_hard_pool_safe(val, ann)
    # 配额 400 远大于可用量 → 应记录 shortfall 并取全部
    assert rep.shortfalls
    assert len(sel) > 0
    assert set(sel["hard_pool_role"].unique()) <= {"hard", "clean_control"}


def make_hard_pool_safe(val, ann):
    return make_val_hard_pool(val, ann, quota_per_axis=400)


def test_hard_pool_has_clean_controls():
    val = _val_frame(400)
    ann = pd.DataFrame(
        {
            "image_id": np.repeat(val["image_id"].to_numpy()[:100], 2),
            "ann_id": np.arange(1, 201),
            "is_hard_t1": [True] * 60 + [False] * 140,
            "area_bin": ["large"] * 200,
        }
    )
    sel, _ = make_val_hard_pool(val, ann, quota_per_axis=20)
    n_hard = int((sel["hard_pool_role"] == "hard").sum())
    n_clean = int((sel["hard_pool_role"] == "clean_control").sum())
    assert n_hard > 0 and n_clean > 0
    assert n_clean == n_hard, "clean 对照应与难例等量"


# ---------------------------------------------------------------------------
# 图级尺寸标签
# ---------------------------------------------------------------------------


def test_image_area_bin_uses_largest_box():
    ann = pd.DataFrame(
        {
            "image_id": [1, 1, 2],
            "ann_id": [1, 2, 3],
            "area": [100.0, 50000.0, 900.0],
            "area_bin": ["small", "large", "small"],
        }
    )
    s = image_area_bin(ann)
    assert s.loc[1] == "large"  # 取最大框的档
    assert s.loc[2] == "small"
