"""磁盘清点（inventory）。

遍历解压后的数据集目录，产出一张「每个 stem 一行」的清单表。

为什么以 **stem** 而不是文件为单位：LogoDet-3K 的每个样本是一对
`<name>.jpg` + `<name>.xml`。以 stem 聚合后，四象限一目了然：

    both       图与标注都在      ← 唯一合法状态
    only_img   有图无标注        ← 必须为 0
    only_xml   有标注无图        ← 必须为 0
    other      不成对的杂项      ← 必须为 0

目录结构假定为严格三层：`LogoDet-3K/{超类}/{品牌}/{文件}`。
任何偏离（比如文件直接躺在超类目录下）都会被单独记录为 depth_anomaly，
而不是被静默吞掉 —— 静默吞掉会让后面的图像数对账莫名其妙地差几个。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class InventoryReport:
    n_supercats: int = 0
    n_brand_dirs: int = 0
    n_files: int = 0
    quadrants: dict[str, int] = field(default_factory=dict)
    ext_counts: dict[str, int] = field(default_factory=dict)
    depth_anomalies: list[str] = field(default_factory=list)
    brand_case_collisions: dict[str, list[str]] = field(default_factory=dict)
    per_supercat: pd.DataFrame | None = None


def scan_dataset(
    root: Path,
    *,
    image_exts: set[str],
    ann_ext: str,
    ignore_names: set[str],
) -> tuple[pd.DataFrame, InventoryReport]:
    """扫描数据集目录，返回 (清单表, 报告)。"""
    rep = InventoryReport()
    ann_ext = ann_ext.lower()

    # (supercat, brand, stem) -> 累积信息
    rows: dict[tuple[str, str, str], dict] = {}
    ext_counts: dict[str, int] = defaultdict(int)
    brand_lower: dict[str, list[str]] = defaultdict(list)

    supercats = sorted(
        d for d in root.iterdir() if d.is_dir() and d.name not in ignore_names
    )
    rep.n_supercats = len(supercats)

    for sc_dir in supercats:
        # 超类目录下若直接躺着文件，属于结构异常
        for entry in sc_dir.iterdir():
            if entry.is_file() and entry.name not in ignore_names:
                rep.depth_anomalies.append(entry.relative_to(root).as_posix())

        brands = sorted(
            d for d in sc_dir.iterdir() if d.is_dir() and d.name not in ignore_names
        )
        rep.n_brand_dirs += len(brands)

        for br_dir in brands:
            brand_lower[f"{sc_dir.name}/{br_dir.name}".lower()].append(
                f"{sc_dir.name}/{br_dir.name}"
            )

            for f in br_dir.iterdir():
                if f.name in ignore_names:
                    continue
                if f.is_dir():
                    rep.depth_anomalies.append(f.relative_to(root).as_posix())
                    continue

                rep.n_files += 1
                ext = f.suffix.lower()
                ext_counts[ext] += 1

                key = (sc_dir.name, br_dir.name, f.stem)
                rec = rows.setdefault(
                    key,
                    {
                        "supercat": sc_dir.name,
                        "brand_dir": br_dir.name,
                        "stem": f.stem,
                        "image_rel": None,
                        "xml_rel": None,
                        "image_ext": None,
                        "image_bytes": 0,
                        "xml_bytes": 0,
                    },
                )
                # 用 as_posix() 而不是 str()：Windows 上 Path.relative_to()
                # 转字符串会用反斜杠，但 zip 内部路径规范强制用正斜杠 —— 两边
                # 分隔符对不上会导致 S2 从 zip 读 XML 时按 rel_path 查表全部
                # 失配（inventory 里存的是这份 rel_path，是下游唯一真源）。
                rel = f.relative_to(root).as_posix()
                size = f.stat().st_size

                if ext in image_exts:
                    rec["image_rel"] = rel
                    rec["image_ext"] = ext
                    rec["image_bytes"] = size
                elif ext == ann_ext:
                    rec["xml_rel"] = rel
                    rec["xml_bytes"] = size
                else:
                    # 既不是图也不是标注 —— 记进 other 象限
                    rec.setdefault("other_rel", []).append(rel)

    for key, variants in brand_lower.items():
        uniq = sorted(set(variants))
        if len(uniq) > 1:
            rep.brand_case_collisions[key] = uniq

    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        rep.quadrants = {"both": 0, "only_img": 0, "only_xml": 0, "other": 0}
        rep.ext_counts = dict(ext_counts)
        return df, rep

    has_img = df["image_rel"].notna()
    has_xml = df["xml_rel"].notna()
    df["quadrant"] = "other"
    df.loc[has_img & has_xml, "quadrant"] = "both"
    df.loc[has_img & ~has_xml, "quadrant"] = "only_img"
    df.loc[~has_img & has_xml, "quadrant"] = "only_xml"

    # 稳定的字典序排序 → image_id 才能可复现
    df = df.sort_values(["supercat", "brand_dir", "stem"], kind="mergesort").reset_index(
        drop=True
    )

    rep.quadrants = df["quadrant"].value_counts().to_dict()
    for q in ("both", "only_img", "only_xml", "other"):
        rep.quadrants.setdefault(q, 0)
    rep.ext_counts = dict(sorted(ext_counts.items(), key=lambda kv: -kv[1]))

    rep.per_supercat = (
        df.assign(is_img=has_img.values)
        .groupby("supercat")
        .agg(
            classes=("brand_dir", "nunique"),
            images=("is_img", "sum"),
            stems=("stem", "size"),
        )
        .reset_index()
    )

    return df, rep
