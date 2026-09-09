"""内部数据契约。

**这是全项目唯一的内部样本格式定义。** 核心 Dataset 只产出 Sample，
框架相关的格式（torchvision / COCO / YOLO / MMDet）都由 adapter 从它转换。

三条写死的约定：

1. **坐标一律 0-based、半开区间 [x1, x2)、原图像素、float32**
   S2 的 G4b 门已实测确认数据本身是 0-based（存在 x1==0，且 x2 从不超过 W），
   所以解析时不做任何平移。任何 adapter 都不得偷偷改变这个约定。

2. **ann_ids 必须端到端保留**
   这是最容易被忽略却最关键的字段。切片评测要按 ann 精确对齐 ——
   若中途丢掉 ann_id，切片就只能退化到 image 级，
   「一张图里一个框是难例、另一个不是」这种情况就没法处理了。

3. **Sample 不含任何 torch 对象**
   core_dataset 物理上不 import torch（见该模块说明），
   所以 Sample 里只能是 numpy / Python 原生类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Sample:
    """一个样本 = 一张图 + 它的全部标注。"""

    image_id: int
    rel_path: str  # 相对 dataset_root；绝对路径不进数据结构，保证产物可跨机搬迁
    width: int
    height: int

    # (N, 4) float32，0-based 半开区间 [x1, x2)，原图像素坐标
    boxes_xyxy: np.ndarray
    # (N,) int32。主轨 class-agnostic 恒为 1；副轨才是 1..N
    labels: np.ndarray
    # (N,) int64 —— 见模块说明的第 2 条
    ann_ids: np.ndarray
    # (N,) int8。真源恒 0；切片视图生成时才按需置 1 表示"忽略"
    iscrowd: np.ndarray

    # 已解码的图像，(H, W, 3) uint8 RGB。惰性模式下为 None
    image: np.ndarray | None = None

    # 只读附加信息：supercat / class_ids / 难例标记 等
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_boxes(self) -> int:
        return int(len(self.boxes_xyxy))

    def validate(self) -> None:
        """自检。任何 adapter 在转换前都应先过一遍。"""
        n = len(self.boxes_xyxy)
        if self.boxes_xyxy.shape != (n, 4):
            raise ValueError(f"boxes 形状应为 ({n}, 4)，得到 {self.boxes_xyxy.shape}")
        for name, arr in (
            ("labels", self.labels),
            ("ann_ids", self.ann_ids),
            ("iscrowd", self.iscrowd),
        ):
            if len(arr) != n:
                raise ValueError(f"{name} 长度 {len(arr)} 与框数 {n} 不一致")

        if n:
            x1, y1, x2, y2 = self.boxes_xyxy.T
            if (x1 < 0).any() or (y1 < 0).any():
                raise ValueError("存在负坐标")
            if (x2 > self.width + 0.5).any() or (y2 > self.height + 0.5).any():
                raise ValueError("存在越界坐标")
            if (x2 <= x1).any() or (y2 <= y1).any():
                raise ValueError("存在退化框（x2<=x1 或 y2<=y1）")

        if self.image is not None:
            if self.image.ndim != 3 or self.image.shape[2] != 3:
                raise ValueError(f"image 形状应为 (H, W, 3)，得到 {self.image.shape}")
            h, w = self.image.shape[:2]
            if (h, w) != (self.height, self.width):
                raise ValueError(
                    f"image 尺寸 ({h}, {w}) 与表里的 ({self.height}, {self.width}) 不一致"
                )


@dataclass(frozen=True)
class Detection:
    """预测输出的统一格式。

    任何 baseline（RPN proposals / OWLv2 / COCO 检测器）都产这个，
    再由 adapter 转成 pycocotools 需要的 dict。这样评测器只需认一种格式。
    """

    image_id: int
    boxes_xyxy: np.ndarray  # (M, 4) float32
    scores: np.ndarray  # (M,) float32
    labels: np.ndarray  # (M,) int32

    def validate(self) -> None:
        m = len(self.boxes_xyxy)
        if self.boxes_xyxy.shape != (m, 4):
            raise ValueError(f"boxes 形状应为 ({m}, 4)")
        if len(self.scores) != m or len(self.labels) != m:
            raise ValueError("scores / labels 长度与框数不一致")
