"""VOC XML 解析。

单个 XML → 一个 ParsedAnnotation。只做解析与容错，**不做清洗判断**
（那是 curate/clean_rules.py 的事），职责分开便于分别测试。

用 lxml 而不是 regex 的硬性理由：XML 实体必须反转义。
侦察阶段用 regex 直接从字节流取 `<name>`，结果 7.44% 的框被误判为
"类名与目录名不一致" —— 全部是 `A. Favre &amp; Fils` vs `A. Favre & Fils`
这类转义差异。lxml 会自动还原成 `&`。

recover=True 让 lxml 容忍轻微畸形（未闭合标签、非法字符），
避免一个坏文件炸掉整批解析。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from lxml import etree

# 复用同一个 parser 实例，避免每个文件都重建（158k 次的开销不可忽略）
_PARSER = etree.XMLParser(recover=True, huge_tree=False, resolve_entities=False)


@dataclass
class ParsedBox:
    name_raw: str  # XML 原文（已反转义，未归一）
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    truncated: int  # 真实标注，非代理。实测 10.43% 为 1
    difficult: int  # 实测全 0，保留字段以备上游更新
    pose: str
    order_in_file: int  # 在 XML 中的原始顺序，用于生成稳定 ann_id


@dataclass
class ParsedAnnotation:
    rel_path: str
    filename: str | None
    width: int | None
    height: int | None
    depth: int | None
    verified: str | None  # 根节点 verified 属性，部分文件没有
    segmented: int | None
    boxes: list[ParsedBox] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _text(node, tag: str) -> str | None:
    el = node.find(tag)
    if el is None or el.text is None:
        return None
    return el.text.strip()


def _int_or_none(node, tag: str) -> int | None:
    s = _text(node, tag)
    if s is None:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _num(node, tag: str) -> float | None:
    s = _text(node, tag)
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def normalize_class_name(raw: str) -> str:
    """类名归一：NFKC + 去首尾空白 + 折叠内部连续空白。

    **不动大小写，也不动 `-1` / `-2` 后缀。**

    为什么不合并 `-N` 后缀（S2 之后 P3 代理要用）：
    文字版 logo 的长宽比常在 3~6，图形版在 0.8~1.5。合并后类内长宽比的
    MAD 被人为放大，z-score 系统性偏小，P3 就筛不出任何离群点了。
    """
    s = unicodedata.normalize("NFKC", raw)
    return " ".join(s.split())


def parse_voc_xml(rel_path: str, data: bytes) -> ParsedAnnotation:
    """解析单个 VOC XML。任何局部失败都记进 errors，不抛异常。"""
    ann = ParsedAnnotation(
        rel_path=rel_path,
        filename=None,
        width=None,
        height=None,
        depth=None,
        verified=None,
        segmented=None,
    )

    try:
        root = etree.fromstring(data, _PARSER)
    except etree.XMLSyntaxError as e:
        ann.errors.append(f"XMLSyntaxError: {e}")
        return ann
    if root is None:
        ann.errors.append("解析后根节点为 None")
        return ann

    ann.verified = root.get("verified")
    ann.filename = _text(root, "filename")
    ann.segmented = _int_or_none(root, "segmented")

    size = root.find("size")
    if size is not None:
        ann.width = _int_or_none(size, "width")
        ann.height = _int_or_none(size, "height")
        ann.depth = _int_or_none(size, "depth")
    else:
        ann.errors.append("缺 <size> 节点")

    for i, obj in enumerate(root.findall("object")):
        name = _text(obj, "name")
        if name is None:
            ann.errors.append(f"object[{i}] 缺 <name>")
            continue

        bb = obj.find("bndbox")
        if bb is None:
            ann.errors.append(f"object[{i}] 缺 <bndbox>")
            continue

        coords = [_num(bb, t) for t in ("xmin", "ymin", "xmax", "ymax")]
        if any(c is None for c in coords):
            ann.errors.append(f"object[{i}] bndbox 坐标不完整")
            continue

        ann.boxes.append(
            ParsedBox(
                name_raw=name,
                xmin=coords[0],
                ymin=coords[1],
                xmax=coords[2],
                ymax=coords[3],
                truncated=_int_or_none(obj, "truncated") or 0,
                difficult=_int_or_none(obj, "difficult") or 0,
                pose=_text(obj, "pose") or "",
                order_in_file=i,
            )
        )

    return ann
