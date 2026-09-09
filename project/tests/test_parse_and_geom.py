"""S2 解析与几何派生的回归测试。

重点守住三处曾经出错或容易出错的地方：
  * XML 实体反转义（regex 方案在这上面误判了 7.44% 的框）
  * area_bin 的边界闭合方向（pd.cut 曾在 area==1024 上偏 17 个框）
  * 清洗规则的执行顺序（先纠反序再 clip，否则归因会错）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logodet.curate.clean_rules import CleanStats, clean_box  # noqa: E402
from logodet.curate.derive_geom import (  # noqa: E402
    AREA_LARGE,
    AREA_SMALL,
    add_geometry,
    add_iof_occlusion,
    add_shape_outlier,
    add_truncation_label,
)
from logodet.ingest.voc_parse import normalize_class_name, parse_voc_xml  # noqa: E402


def _xml(objects: str, *, w: int = 500, h: int = 400, verified: str = "no") -> bytes:
    return (
        f'<?xml version="1.0" ?><annotation verified="{verified}">'
        f"<folder>VOC2007</folder><filename>1.jpg</filename>"
        f"<size><width>{w}</width><height>{h}</height><depth>3</depth></size>"
        f"<segmented>0</segmented>{objects}</annotation>"
    ).encode("utf-8")


def _obj(name: str, x1: int, y1: int, x2: int, y2: int, trunc: int = 0) -> str:
    return (
        f"<object><name>{name}</name><pose>Unspecified</pose>"
        f"<truncated>{trunc}</truncated><difficult>0</difficult>"
        f"<bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
        f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
    )


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def test_parse_basic_fields():
    ann = parse_voc_xml("Food/x/1.xml", _xml(_obj("brandA", 10, 20, 60, 70, trunc=1)))
    assert ann.width == 500 and ann.height == 400
    assert ann.verified == "no"
    assert len(ann.boxes) == 1
    b = ann.boxes[0]
    assert b.name_raw == "brandA"
    assert (b.xmin, b.ymin, b.xmax, b.ymax) == (10, 20, 60, 70)
    assert b.truncated == 1
    assert not ann.errors


def test_parse_unescapes_xml_entities():
    """关键回归：`&amp;` 必须还原成 `&`。

    侦察阶段用 regex 直接取字节流，导致目录名 'A. Favre & Fils' 与
    XML 里的 'A. Favre &amp; Fils' 被判为不一致，误报了 7.44% 的框。
    """
    ann = parse_voc_xml("C/x/1.xml", _xml(_obj("A. Favre &amp; Fils", 1, 1, 9, 9)))
    assert ann.boxes[0].name_raw == "A. Favre & Fils"


def test_parse_multiple_objects_keeps_order():
    objs = _obj("b", 1, 1, 9, 9) + _obj("b", 20, 20, 30, 30) + _obj("b", 40, 40, 50, 50)
    ann = parse_voc_xml("C/x/1.xml", _xml(objs))
    assert [b.order_in_file for b in ann.boxes] == [0, 1, 2]


def test_parse_missing_size_records_error():
    data = (
        b'<?xml version="1.0" ?><annotation><filename>1.jpg</filename>'
        b"<object><name>b</name><bndbox><xmin>1</xmin><ymin>1</ymin>"
        b"<xmax>9</xmax><ymax>9</ymax></bndbox></object></annotation>"
    )
    ann = parse_voc_xml("C/x/1.xml", data)
    assert ann.width is None
    assert any("size" in e for e in ann.errors)


def test_parse_malformed_does_not_raise():
    ann = parse_voc_xml("C/x/1.xml", b"<annotation><object><name>b</name>")
    assert isinstance(ann.errors, list)  # 不抛异常即达标


def test_normalize_class_name_preserves_case_and_suffix():
    # 不动大小写、不动 -N 后缀 —— P3 代理依赖后缀区分文字版/图形版
    assert normalize_class_name("  Lexus-2  ") == "Lexus-2"
    assert normalize_class_name("A   B") == "A B"
    assert normalize_class_name("ＡＢ") == "AB"  # NFKC 全角→半角


# ---------------------------------------------------------------------------
# 清洗规则
# ---------------------------------------------------------------------------


def test_clean_box_swaps_then_clips():
    st = CleanStats()
    out = clean_box(60, 70, 10, 20, 500, 400, st)
    assert out == (10, 20, 60, 70)
    assert st.counts["R1"] == 2  # x 与 y 各交换一次


def test_clean_box_clips_out_of_bounds():
    st = CleanStats()
    out = clean_box(-5, -5, 600, 500, 500, 400, st)
    assert out == (0, 0, 500, 400)
    assert st.counts["R2"] == 1


def test_clean_box_drops_degenerate():
    st = CleanStats()
    assert clean_box(10, 10, 10, 50, 500, 400, st) is None
    assert st.counts["R3"] == 1


def test_clean_box_order_matters_for_attribution():
    """反序框被 clip 成退化框时，归因必须是 R1 而非只有 R3。

    若顺序写反（先 clip 再纠反序），R1 的信息就丢了，报告里会归错因。
    """
    st = CleanStats()
    clean_box(400, 300, 100, 100, 500, 400, st)
    assert st.counts["R1"] == 2


# ---------------------------------------------------------------------------
# 几何派生
# ---------------------------------------------------------------------------


def _frame(boxes: list[tuple], w: int = 500, h: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ann_id": np.arange(1, len(boxes) + 1),
            "image_id": [b[0] for b in boxes],
            "x1": [float(b[1]) for b in boxes],
            "y1": [float(b[2]) for b in boxes],
            "x2": [float(b[3]) for b in boxes],
            "y2": [float(b[4]) for b in boxes],
            "img_w": w,
            "img_h": h,
        }
    )


def test_area_bin_boundary_is_exact():
    """关键回归：area 恰好等于 32² 与 96² 时的归档。

    约定：small = area < 32²；medium = 32² <= area <= 96²；large = area > 96²
    早期用 pd.cut(right=True) 让 area==1024 落进 small，与直算差 17 个框（G7 抓到）。
    """
    # 32x32=1024（恰好 = AREA_SMALL），96x96=9216（恰好 = AREA_LARGE）
    df = _frame([
        (1, 0, 0, 31, 31),      # 961  < 1024  → small
        (2, 0, 0, 32, 32),      # 1024 == 1024 → medium（不是 small）
        (3, 0, 0, 96, 96),      # 9216 == 9216 → medium（不是 large）
        (4, 0, 0, 97, 97),      # 9409 > 9216  → large
    ])
    out = add_geometry(df)
    assert list(out["area_bin"]) == ["small", "medium", "medium", "large"]

    # 与直算口径必须完全一致
    area = out["area"].to_numpy()
    assert int((area < AREA_SMALL).sum()) == int((out["area_bin"] == "small").sum())
    assert int((area > AREA_LARGE).sum()) == int((out["area_bin"] == "large").sum())


def test_geometry_fields():
    out = add_geometry(_frame([(1, 100, 50, 300, 150)]))
    r = out.iloc[0]
    assert r["bw"] == 200 and r["bh"] == 100
    assert r["area"] == 20000
    assert r["ar"] == pytest.approx(2.0)
    assert r["rel_area"] == pytest.approx(20000 / (500 * 400))
    assert r["cx_rel"] == pytest.approx(200 / 500)


def test_iof_uses_containment_not_iou():
    """大框被小框压住：IoU 很小但 IoF 很大。

    这正是不用 IoU 的理由 —— IoU 会漏掉"大目标被局部遮挡"这类。
    """
    # 大框 200x200=40000，小框 100x100=10000 完全落在大框内
    df = _frame([(1, 0, 0, 200, 200), (1, 50, 50, 150, 150)])
    out = add_iof_occlusion(add_geometry(df), threshold=0.2)
    big, small = out.iloc[0], out.iloc[1]

    # 小框被完全包含 → IoF(small) = 1.0
    assert small["p2_occ_iof"] == pytest.approx(1.0)
    # 大框被压住 1/4 → IoF(big) = 0.25；而 IoU 只有 10000/40000 = 0.25 也一样，
    # 换个更悬殊的例子看差异更明显，这里先确认 IoF 语义正确
    assert big["p2_occ_iof"] == pytest.approx(10000 / 40000)
    assert bool(small["is_hard_p2"]) and bool(big["is_hard_p2"])


def test_iof_zero_for_single_box_image():
    out = add_iof_occlusion(add_geometry(_frame([(1, 0, 0, 50, 50)])), threshold=0.3)
    assert out.iloc[0]["p2_occ_iof"] == 0.0
    assert not bool(out.iloc[0]["is_hard_p2"])


def test_iof_zero_for_disjoint_boxes():
    df = _frame([(1, 0, 0, 50, 50), (1, 100, 100, 150, 150)])
    out = add_iof_occlusion(add_geometry(df), threshold=0.3)
    assert (out["p2_occ_iof"] == 0).all()


def test_shape_outlier_needs_min_class_size():
    """类内样本不足时 p3_valid 必须为 False，否则会造出一堆假离群。"""
    df = _frame([(i, 0, 0, 100, 50) for i in range(1, 6)])
    df["class_id"] = 1
    out = add_shape_outlier(add_geometry(df), min_class_size=20)
    assert not out["p3_valid"].any()
    assert not out["is_hard_p3"].any()


def test_shape_outlier_flags_true_outlier():
    # 24 个长宽比 2.0 的框 + 1 个长宽比 10.0 的离群框
    rows = [(i, 0, 0, 100, 50) for i in range(1, 25)]
    rows.append((99, 0, 0, 500, 50))
    df = _frame(rows)
    df["class_id"] = 1
    out = add_shape_outlier(add_geometry(df), z_threshold=3.0, min_class_size=20)
    assert out["p3_valid"].all()
    assert int(out["is_hard_p3"].sum()) == 1
    assert bool(out.iloc[-1]["is_hard_p3"])


def test_shape_outlier_mad_floor_prevents_inf():
    """同类长宽比完全相同（MAD==0）时不能产生 inf。"""
    rows = [(i, 0, 0, 100, 50) for i in range(1, 26)]
    df = _frame(rows)
    df["class_id"] = 1
    out = add_shape_outlier(add_geometry(df), min_class_size=20)
    assert np.isfinite(out["p3_ar_z"]).all()
    assert not out["is_hard_p3"].any()


def test_truncation_uses_real_label():
    df = _frame([(1, 0, 0, 50, 50), (2, 0, 0, 50, 50)])
    df["truncated"] = [1, 0]
    out = add_truncation_label(df)
    assert list(out["is_hard_t1"]) == [True, False]
