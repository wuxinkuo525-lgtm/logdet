"""格式无关的核心 Dataset。

**本模块刻意不 import torch。** 这不是风格洁癖 —— 它让"核心与框架耦合"
在物理上不可能发生。任何需要 torch 的转换都必须写成 adapter，
于是将来换 Ultralytics / MMDetection 时，改动被限制在 adapters/ 目录内。

图像读取走 **zip 通路**，有实测支撑（scripts/bench_image_read.py，300 张样本，
字节与解码结果均逐一校验一致）：

    A 磁盘逐文件      38.7 img/s   → 推理全集 3,079 图需 79.6s   未达 60 img/s 门槛
    B zip 随机访问 1,360.4 img/s   → 推理全集需  2.3s            快 35.2x

值得注意的是 zip **随机**访问也这么快，说明瓶颈完全来自每次 open() 过
文件守卫的固定开销，而不是磁盘寻道。这与 S2 在 XML 上的发现同源
（那里是 20 → 15,305 文件/秒）。

代价：依赖 raw/LogoDet-3K.zip 存在。这不是问题 —— S1 本就刻意保留 zip
并留了 sha256，正是为了任何环节都能对回同一份源。
若 zip 缺失，会自动回退到磁盘通路并打印警告，而不是直接失败。
"""

from __future__ import annotations

import io
import threading
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .schema import Sample


class ZipImageReader:
    """从 zip 里按需读图。每个进程/线程各持一个句柄。

    zipfile.ZipFile 不是线程安全的，也不能跨进程共享 —— DataLoader 用
    spawn 起 worker 时，每个 worker 必须自己打开。用 thread-local 存句柄，
    并且**不把句柄放进 __init__**，否则 spawn 时 pickle 会失败。
    """

    def __init__(self, zip_path: Path, top_dir: str) -> None:
        self.zip_path = Path(zip_path)
        self.top_dir = top_dir
        self._local = threading.local()
        self._index: dict[str, zipfile.ZipInfo] | None = None

    def __getstate__(self) -> dict:
        # spawn 时只传路径，句柄与索引在子进程里重建
        return {"zip_path": self.zip_path, "top_dir": self.top_dir}

    def __setstate__(self, state: dict) -> None:
        self.zip_path = state["zip_path"]
        self.top_dir = state["top_dir"]
        self._local = threading.local()
        self._index = None

    @property
    def _zf(self) -> zipfile.ZipFile:
        zf = getattr(self._local, "zf", None)
        if zf is None:
            zf = zipfile.ZipFile(self.zip_path)
            self._local.zf = zf
        return zf

    def build_index(self, rel_paths: Iterable[str] | None = None) -> int:
        """建立 rel_path → ZipInfo 映射。

        给定 rel_paths 时只保留这些条目，**只省内存、不省时间** ——
        实测全量索引 317,308 条目 1.36s，限制到 600 条目仍需 1.25s，
        因为无论如何都要扫完整个中央目录。
        省的是内存：全量 dict 约几十 MB，而推理只需 3,079 条。
        """
        prefix = f"{self.top_dir}/"
        wanted = set(rel_paths) if rel_paths is not None else None
        idx: dict[str, zipfile.ZipInfo] = {}
        for info in self._zf.infolist():
            if info.is_dir() or not info.filename.startswith(prefix):
                continue
            rel = info.filename[len(prefix):]
            if wanted is None or rel in wanted:
                idx[rel] = info
        self._index = idx
        return len(idx)

    def read(self, rel_path: str) -> bytes:
        if self._index is None:
            self.build_index()
        info = self._index.get(rel_path)
        if info is None:
            raise KeyError(f"zip 内找不到 {rel_path}")
        return self._zf.read(info)

    def close(self) -> None:
        zf = getattr(self._local, "zf", None)
        if zf is not None:
            zf.close()
            self._local.zf = None


class CoreDataset:
    """索引来自 parquet，图像按需解码，产出 Sample。

    Args:
        images: 图级表，需含 image_id / rel_path / img_w / img_h
        annotations: 框级表，需含 ann_id / image_id / x1..y2 等
        image_ids: 只保留这些图（如推理集合）。None 表示全部
        class_agnostic: True 时 labels 恒为 1（主轨）；False 时用 class_id
        load_image: False 时 Sample.image 为 None，只出标注（评测 GT 用得上）
        zip_reader: 给定则走 zip 通路；None 时自动尝试，失败回退磁盘
    """

    def __init__(
        self,
        images: pd.DataFrame,
        annotations: pd.DataFrame,
        *,
        dataset_root: Path,
        image_ids: Sequence[int] | None = None,
        class_agnostic: bool = True,
        load_image: bool = True,
        zip_reader: ZipImageReader | None = None,
        meta_cols: Sequence[str] = (),
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.class_agnostic = class_agnostic
        self.load_image = load_image
        self.zip_reader = zip_reader
        self.meta_cols = tuple(meta_cols)

        img = images
        if image_ids is not None:
            keep = set(int(i) for i in image_ids)
            img = img[img["image_id"].isin(keep)]
        # 稳定顺序：按 image_id 升序 —— 保证 shuffle=False 时遍历可复现
        self.images = img.sort_values("image_id", kind="mergesort").reset_index(drop=True)

        ids = set(self.images["image_id"])
        ann = annotations[annotations["image_id"].isin(ids)]
        # 按 (image_id, ann_id) 排序后分组，保证每个样本内框的顺序也确定
        ann = ann.sort_values(["image_id", "ann_id"], kind="mergesort")
        self._ann_by_img = {int(k): v for k, v in ann.groupby("image_id", sort=False)}

        self._rows = self.images.to_dict("records")

    # ---- 只读属性 ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def image_ids(self) -> list[int]:
        return [int(r["image_id"]) for r in self._rows]

    @property
    def rel_paths(self) -> list[str]:
        return [str(r["rel_path"]) for r in self._rows]

    # ---- 取样 -------------------------------------------------------------

    def _decode(self, rel_path: str) -> np.ndarray:
        raw: bytes | None = None
        if self.zip_reader is not None:
            try:
                raw = self.zip_reader.read(rel_path)
            except (KeyError, OSError):
                raw = None  # 落回磁盘
        if raw is None:
            p = self.dataset_root / rel_path
            if not p.is_file():
                raise FileNotFoundError(
                    f"图像不存在：{p}\n"
                    f"检查 configs/paths.yaml 的 dataset_root，或先跑 s1_download.py"
                )
            raw = p.read_bytes()
        with Image.open(io.BytesIO(raw)) as im:
            return np.asarray(im.convert("RGB"))

    def __getitem__(self, i: int) -> Sample:
        row = self._rows[i]
        image_id = int(row["image_id"])
        a = self._ann_by_img.get(image_id)

        if a is None or len(a) == 0:
            boxes = np.zeros((0, 4), dtype="float32")
            labels = np.zeros((0,), dtype="int32")
            ann_ids = np.zeros((0,), dtype="int64")
            iscrowd = np.zeros((0,), dtype="int8")
        else:
            boxes = a[["x1", "y1", "x2", "y2"]].to_numpy(dtype="float32")
            labels = (
                np.ones(len(a), dtype="int32")
                if self.class_agnostic
                else a["class_id"].to_numpy(dtype="int32")
            )
            ann_ids = a["ann_id"].to_numpy(dtype="int64")
            iscrowd = (
                a["iscrowd"].to_numpy(dtype="int8")
                if "iscrowd" in a.columns
                else np.zeros(len(a), dtype="int8")
            )

        meta: dict = {"supercat": row.get("supercat"), "brand_dir": row.get("brand_dir")}
        for c in self.meta_cols:
            if a is not None and c in a.columns:
                meta[c] = a[c].to_numpy()
            elif c in row:
                meta[c] = row[c]

        image = self._decode(str(row["rel_path"])) if self.load_image else None

        return Sample(
            image_id=image_id,
            rel_path=str(row["rel_path"]),
            width=int(row["img_w"]),
            height=int(row["img_h"]),
            boxes_xyxy=boxes,
            labels=labels,
            ann_ids=ann_ids,
            iscrowd=iscrowd,
            image=image,
            meta=meta,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
