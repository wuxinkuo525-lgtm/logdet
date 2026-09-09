"""S4 数据层的回归测试。

最重要的一条是 test_core_dataset_does_not_import_torch —— 它守的是
一条**架构约束**而非行为：核心 Dataset 不 import torch，
于是"核心与框架耦合"在物理上不可能发生，将来换框架时改动被限制在
adapters/ 目录内。这种约束不写测试就会在某次"顺手 import 一下"里悄悄失效。
"""

from __future__ import annotations

import ast
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from logodet.data.adapters.to_coco_json import (  # noqa: E402
    build_coco_dt,
    build_coco_gt,
)
from logodet.data.core_dataset import CoreDataset, ZipImageReader  # noqa: E402
from logodet.data.schema import Detection, Sample  # noqa: E402


# ---------------------------------------------------------------------------
# 架构约束
# ---------------------------------------------------------------------------


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_core_dataset_does_not_import_torch():
    """核心数据层不得依赖 torch —— 这是本项目的架构地基。

    用 AST 静态检查而不是 sys.modules：后者会被测试进程里
    其它模块的 import 干扰，测不准。
    """
    for mod in ("core_dataset.py", "schema.py"):
        p = SRC / "logodet" / "data" / mod
        imports = _imports_of(p)
        assert "torch" not in imports, f"{mod} 不应 import torch，实际导入了 {imports}"
        assert "torchvision" not in imports, f"{mod} 不应 import torchvision"


def test_only_adapters_and_loader_import_torch():
    """torch 只允许出现在 adapters/ 与 loader.py 里。"""
    data_dir = SRC / "logodet" / "data"
    allowed = {"loader.py", "to_torchvision.py"}
    for p in data_dir.rglob("*.py"):
        if p.name == "__init__.py":
            continue
        if "torch" in _imports_of(p):
            assert p.name in allowed, f"{p.relative_to(data_dir)} 不应 import torch"


# ---------------------------------------------------------------------------
# Sample 契约
# ---------------------------------------------------------------------------


def _sample(n: int = 2, w: int = 100, h: int = 80, with_image: bool = True) -> Sample:
    boxes = np.array([[10, 10, 30, 30], [40, 40, 90, 70]][:n], dtype="float32")
    return Sample(
        image_id=7,
        rel_path="Food/b/1.jpg",
        width=w,
        height=h,
        boxes_xyxy=boxes.reshape(n, 4),
        labels=np.ones(n, dtype="int32"),
        ann_ids=np.arange(1, n + 1, dtype="int64"),
        iscrowd=np.zeros(n, dtype="int8"),
        image=np.zeros((h, w, 3), dtype="uint8") if with_image else None,
    )


def test_sample_validate_passes_on_good_data():
    _sample().validate()


def test_sample_validate_catches_negative_coords():
    s = _sample()
    s.boxes_xyxy[0, 0] = -1
    with pytest.raises(ValueError, match="负坐标"):
        s.validate()


def test_sample_validate_catches_out_of_bounds():
    s = _sample(w=100, h=80)
    s.boxes_xyxy[1, 2] = 200
    with pytest.raises(ValueError, match="越界"):
        s.validate()


def test_sample_validate_catches_degenerate():
    s = _sample()
    s.boxes_xyxy[0] = [10, 10, 10, 30]
    with pytest.raises(ValueError, match="退化"):
        s.validate()


def test_sample_validate_catches_length_mismatch():
    s = _sample(2)
    bad = Sample(
        image_id=s.image_id, rel_path=s.rel_path, width=s.width, height=s.height,
        boxes_xyxy=s.boxes_xyxy, labels=np.ones(1, dtype="int32"),
        ann_ids=s.ann_ids, iscrowd=s.iscrowd, image=s.image,
    )
    with pytest.raises(ValueError, match="长度"):
        bad.validate()


def test_sample_validate_catches_image_size_mismatch():
    s = _sample(w=100, h=80)
    bad = Sample(
        image_id=s.image_id, rel_path=s.rel_path, width=s.width, height=s.height,
        boxes_xyxy=s.boxes_xyxy, labels=s.labels, ann_ids=s.ann_ids,
        iscrowd=s.iscrowd, image=np.zeros((10, 10, 3), dtype="uint8"),
    )
    with pytest.raises(ValueError, match="不一致"):
        bad.validate()


def test_sample_allows_zero_boxes():
    s = Sample(
        image_id=1, rel_path="a.jpg", width=10, height=10,
        boxes_xyxy=np.zeros((0, 4), dtype="float32"),
        labels=np.zeros(0, dtype="int32"),
        ann_ids=np.zeros(0, dtype="int64"),
        iscrowd=np.zeros(0, dtype="int8"),
    )
    s.validate()
    assert s.n_boxes == 0


# ---------------------------------------------------------------------------
# CoreDataset
# ---------------------------------------------------------------------------


def _tables():
    images = pd.DataFrame(
        {
            "image_id": [1, 2, 3],
            "rel_path": ["Food/b/1.jpg", "Food/b/2.jpg", "Food/b/3.jpg"],
            "img_w": [100, 100, 100],
            "img_h": [80, 80, 80],
            "supercat": ["Food"] * 3,
            "brand_dir": ["b"] * 3,
            "n_boxes": [2, 1, 0],
        }
    )
    ann = pd.DataFrame(
        {
            "ann_id": [10, 11, 12],
            "image_id": [1, 1, 2],
            "x1": [10.0, 40.0, 5.0],
            "y1": [10.0, 40.0, 5.0],
            "x2": [30.0, 90.0, 50.0],
            "y2": [30.0, 70.0, 50.0],
            "class_id": [7, 7, 9],
            "iscrowd": [0, 0, 0],
        }
    )
    return images, ann


def test_core_dataset_orders_by_image_id(tmp_path):
    images, ann = _tables()
    shuffled = images.iloc[[2, 0, 1]].reset_index(drop=True)
    core = CoreDataset(shuffled, ann, dataset_root=tmp_path, load_image=False)
    assert core.image_ids == [1, 2, 3], "必须按 image_id 升序，否则遍历不可复现"


def test_core_dataset_ann_ids_are_preserved(tmp_path):
    images, ann = _tables()
    core = CoreDataset(images, ann, dataset_root=tmp_path, load_image=False)
    s = core[0]
    assert s.ann_ids.tolist() == [10, 11], "ann_ids 必须端到端保留，切片评测靠它对齐"
    assert s.n_boxes == 2


def test_core_dataset_class_agnostic_labels(tmp_path):
    images, ann = _tables()
    agn = CoreDataset(images, ann, dataset_root=tmp_path, load_image=False,
                      class_agnostic=True)
    assert (agn[0].labels == 1).all()
    cls = CoreDataset(images, ann, dataset_root=tmp_path, load_image=False,
                      class_agnostic=False)
    assert cls[0].labels.tolist() == [7, 7]


def test_core_dataset_handles_image_with_no_boxes(tmp_path):
    images, ann = _tables()
    core = CoreDataset(images, ann, dataset_root=tmp_path, load_image=False)
    s = core[2]  # image_id=3 没有标注
    assert s.n_boxes == 0
    s.validate()


def test_core_dataset_filters_by_image_ids(tmp_path):
    images, ann = _tables()
    core = CoreDataset(images, ann, dataset_root=tmp_path, load_image=False,
                       image_ids=[3, 1])
    assert core.image_ids == [1, 3]


def test_core_dataset_raises_clear_error_on_bad_root(tmp_path):
    images, ann = _tables()
    core = CoreDataset(images, ann, dataset_root=tmp_path / "nope", load_image=True)
    with pytest.raises(FileNotFoundError, match="paths.yaml"):
        _ = core[0]


# ---------------------------------------------------------------------------
# ZipImageReader
# ---------------------------------------------------------------------------


def test_zip_reader_is_picklable(tmp_path):
    """spawn 要 pickle dataset。句柄不可 pickle，所以 __getstate__ 只传路径。"""
    import pickle

    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("Top/x/1.jpg", b"fake")
    r = ZipImageReader(z, "Top")
    r.build_index()
    blob = pickle.dumps(r)
    r2 = pickle.loads(blob)
    assert r2.read("x/1.jpg") == b"fake"


def test_zip_reader_restricted_index(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for i in range(5):
            zf.writestr(f"Top/x/{i}.jpg", b"d")
    r = ZipImageReader(z, "Top")
    assert r.build_index({"x/1.jpg", "x/3.jpg"}) == 2
    assert r.read("x/1.jpg") == b"d"
    with pytest.raises(KeyError):
        r.read("x/0.jpg")


# ---------------------------------------------------------------------------
# COCO adapter
# ---------------------------------------------------------------------------


def test_coco_gt_bbox_conversion_is_xywh():
    """内部 [x1,y1,x2,y2] → COCO [x, y, w, h]，不做任何平移。"""
    images, ann = _tables()
    coco = build_coco_gt(images, ann, class_agnostic=True)
    a0 = next(a for a in coco["annotations"] if a["id"] == 10)
    assert a0["bbox"] == [10.0, 10.0, 20.0, 20.0]  # x2-x1=20, y2-y1=20
    assert a0["area"] == 400.0
    assert a0["category_id"] == 1


def test_coco_gt_class_agnostic_has_single_category():
    images, ann = _tables()
    coco = build_coco_gt(images, ann, class_agnostic=True)
    assert len(coco["categories"]) == 1
    assert coco["categories"][0]["name"] == "logo"
    assert {a["category_id"] for a in coco["annotations"]} == {1}


def test_coco_gt_ignore_sets_iscrowd():
    """切片评测靠 iscrowd=1 让非切片成员被忽略。"""
    images, ann = _tables()
    coco = build_coco_gt(images, ann, ignore_ann_ids={11})
    m = {a["id"]: a["iscrowd"] for a in coco["annotations"]}
    assert m[10] == 0 and m[11] == 1 and m[12] == 0


def test_coco_gt_respects_image_id_filter():
    images, ann = _tables()
    coco = build_coco_gt(images, ann, image_ids=[1])
    assert [i["id"] for i in coco["images"]] == [1]
    assert {a["image_id"] for a in coco["annotations"]} == {1}


def test_coco_dt_conversion():
    pred = pd.DataFrame(
        {
            "image_id": [1, 1],
            "x1": [0.0, 5.0], "y1": [0.0, 5.0],
            "x2": [10.0, 25.0], "y2": [20.0, 15.0],
            "score": [0.9, 0.1],
            "category_id": [1, 1],
        }
    )
    dt = build_coco_dt(pred)
    assert dt[0]["bbox"] == [0.0, 0.0, 10.0, 20.0]
    assert dt[1]["bbox"] == [5.0, 5.0, 20.0, 10.0]
    assert dt[0]["score"] == 0.9


def test_detection_validate():
    d = Detection(
        image_id=1,
        boxes_xyxy=np.zeros((3, 4), dtype="float32"),
        scores=np.zeros(3, dtype="float32"),
        labels=np.ones(3, dtype="int32"),
    )
    d.validate()
    bad = Detection(
        image_id=1,
        boxes_xyxy=np.zeros((3, 4), dtype="float32"),
        scores=np.zeros(2, dtype="float32"),
        labels=np.ones(3, dtype="int32"),
    )
    with pytest.raises(ValueError, match="长度"):
        bad.validate()
