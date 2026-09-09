"""XML 数据源：从 zip 顺序读，而不是从磁盘逐文件读。

**这是 S2 最重要的一个工程决定，有实测支撑。**

本机的文件守卫对每次文件打开都有固定开销。基准测试（scripts/bench_xml_read.py，
3,000 个 XML 样本，字节数校验一致）：

    磁盘逐文件读   20 文件/秒   → 全量 158,654 需 131.7 分钟
    zip 顺序读  1,656 文件/秒   → 全量 158,654 需   1.6 分钟
                                              提速 82.5x

原因：zip 只需打开**一次**文件句柄，之后全是句柄内的顺序读；
逐文件路径要打开 158,654 次，每次都过一遍守卫。

附带好处：按 zip 内的物理顺序遍历，磁盘寻道也最少。

代价：解析阶段依赖 raw/LogoDet-3K.zip 存在。这不是问题 —— S1 本来就
刻意保留 zip 并留了 sha256，正是为了任何环节都能对回同一份源。
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class XmlEntry:
    """一条 XML 记录。rel_path 已剥掉 zip 内的顶层目录，与 inventory 表对齐。"""

    rel_path: str
    data: bytes


def iter_xml_from_zip(
    zip_path: Path,
    *,
    top_dir: str,
    ann_ext: str = ".xml",
    only: set[str] | None = None,
    progress_every: int = 40_000,
    verbose: bool = True,
) -> Iterator[XmlEntry]:
    """按 zip 内物理顺序产出 XML 条目。

    Args:
        top_dir: zip 内顶层目录名（如 "LogoDet-3K"），会从 rel_path 里剥掉
        only: 若给定，只产出 rel_path 在此集合内的条目
        progress_every: 每处理这么多条打印一次进度；0 表示不打印
    """
    prefix = f"{top_dir}/"
    ann_ext = ann_ext.lower()
    n = 0

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or not name.lower().endswith(ann_ext):
                continue
            if not name.startswith(prefix):
                continue
            rel = name[len(prefix):]
            if only is not None and rel not in only:
                continue

            yield XmlEntry(rel, zf.read(info))
            n += 1
            if verbose and progress_every and n % progress_every == 0:
                print(f"    ... 已读 {n:,} 个 XML", flush=True)


def count_xml_in_zip(zip_path: Path, *, top_dir: str, ann_ext: str = ".xml") -> int:
    """只数不读，用于进度条总量。只扫中央目录，约 1 秒。"""
    prefix = f"{top_dir}/"
    ann_ext = ann_ext.lower()
    with zipfile.ZipFile(zip_path) as zf:
        return sum(
            1
            for i in zf.infolist()
            if not i.is_dir()
            and i.filename.startswith(prefix)
            and i.filename.lower().endswith(ann_ext)
        )
