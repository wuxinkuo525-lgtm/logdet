"""Sample → torchvision 检测格式。

**这是唯一 import torch 的数据层文件。** 核心 Dataset 与 Sample 都不碰
torch，所以将来换框架时改动被限制在本目录内。

torchvision 的检测模型接受 `List[Tensor]` + `List[Dict]`，图像尺寸可以不同、
不需要 pad —— 模型内部的 GeneralizedRCNNTransform 会自己做 resize 与 batch pad。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..schema import Sample


def to_torchvision(s: Sample) -> tuple[torch.Tensor, dict[str, Any]]:
    """返回 (image_tensor, target_dict)。

    image: (3, H, W) float32，值域 [0, 1]
    target: boxes / labels / image_id / area / iscrowd / ann_ids

    ann_ids 一路带到 target 里 —— 切片评测要按 ann 精确对齐，
    丢了它切片就只能退化到 image 级。
    """
    if s.image is None:
        raise ValueError(
            f"Sample(image_id={s.image_id}) 没有图像数据。"
            f"构造 CoreDataset 时需 load_image=True"
        )

    # np.asarray(PIL_image) 返回只读数组，torch.from_numpy 会警告
    # "given NumPy array is not writable"。用 np.array(copy=True) 拿可写副本。
    # 这里本来就要 copy 一次（permute + div_ 需要连续可写内存），所以不额外开销。
    img = torch.from_numpy(np.array(s.image, dtype=np.uint8, copy=True))
    img = img.permute(2, 0, 1).to(torch.float32).div_(255.0)

    boxes = torch.from_numpy(np.ascontiguousarray(s.boxes_xyxy)).to(torch.float32)
    if boxes.numel() == 0:
        boxes = boxes.reshape(0, 4)

    wh = boxes[:, 2:] - boxes[:, :2]
    area = wh[:, 0] * wh[:, 1] if boxes.shape[0] else torch.zeros(0, dtype=torch.float32)

    target: dict[str, Any] = {
        "boxes": boxes,
        "labels": torch.from_numpy(np.ascontiguousarray(s.labels)).to(torch.int64),
        "image_id": torch.tensor(s.image_id, dtype=torch.int64),
        "area": area,
        "iscrowd": torch.from_numpy(np.ascontiguousarray(s.iscrowd)).to(torch.int64),
        "ann_ids": torch.from_numpy(np.ascontiguousarray(s.ann_ids)).to(torch.int64),
        # 原图尺寸：预测框要映射回原图坐标时需要
        "orig_size": torch.tensor([s.height, s.width], dtype=torch.int64),
    }
    return img, target


def detections_to_records(
    image_id: int,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    baseline: str,
    run_id: str,
) -> list[dict[str, Any]]:
    """预测 → 统一的 predictions.parquet 行格式。

    任何 baseline 都产这个格式，COCO JSON 由它派生。这样评测器只认一种输入，
    新增 baseline 不需要动评测代码。
    """
    out = []
    for (x1, y1, x2, y2), sc, lb in zip(boxes_xyxy, scores, labels):
        out.append(
            {
                "image_id": int(image_id),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "score": float(sc),
                "category_id": int(lb),
                "baseline": baseline,
                "run_id": run_id,
            }
        )
    return out
