"""OWLv2 开放词表检测 baseline（L2）。

### 几何对齐：本模块最容易出错、也最容易被误读的地方

`Owlv2ImageProcessor` 的预处理顺序是：

    原图 W×H  →  **先 pad 成正方形** S×S（S=max(W,H)，右/下补 0.5 灰）
              →  再 resize 到 960×960

模型输出的框是**相对于那个正方形**的归一化坐标。所以调用
`post_process_object_detection(..., target_sizes=T)` 时：

    传 T=(H, W)      → 框被按原图长宽比拉伸，**系统性错位**
    传 T=(S, S)      → 框落在 padded 方形坐标系里，而由于 padding 补在
                       右/下、原图锚在左上角，这个坐标系与原图坐标系
                       **在原图范围内是一致的** → 正确

错位的后果是 AP/AR 接近 0 —— 而这与"模型确实找不到 logo"**在数字上无法区分**。
所以必须有独立的几何验证，不能只看指标。本模块提供两条：

  1. `synthetic_alignment_check()` —— 在纯色背景上贴一个已知位置的色块，
     用对应 prompt 检测，验证框中心落在色块内。这是决定性判据。
  2. 真实图目视抽检（由 scripts/s7b_probe_owlv2.py 落盘画框图）

### prompt 是超参，必须在 trainval 上选

OWLv2 对文本 prompt 高度敏感。若在 val2k_repr 上比较 prompt 再报同一集合的
指标，就是在报告用的集合上调超参 —— 数据泄漏。所以 prompt 选型只用
trainval 抽样（本项目从不在 trainval 上报任何数字）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

MODEL_ID = "google/owlv2-base-patch16-ensemble"

# 候选 prompt。逗号分隔的多个 query 会被 OWLv2 当作多个类别分别打分，
# 我们取所有 query 的最高分作为 class-agnostic 的 "logo" 得分。
PROMPT_CANDIDATES: dict[str, list[str]] = {
    "bare": ["logo"],
    "article": ["a logo"],
    "brand": ["a brand logo"],
    "photo": ["a photo of a logo"],
    "multi": ["a logo", "a brand name", "a trademark"],
    "signage": ["a logo", "a brand logo on a product", "a store sign"],
}


@dataclass
class Owlv2Output:
    image_id: int
    boxes_xyxy: np.ndarray  # (M, 4) float32，原图像素坐标
    scores: np.ndarray  # (M,) float32，已校准的检测置信度，可跨图比较
    query_idx: np.ndarray  # (M,) int32，命中的是第几个 query


def load_owlv2(device: str = "cpu"):
    """加载 processor 与模型。首次调用会下载约 1.6GB 权重。"""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID)
    model.eval().to(device)
    return processor, model


@torch.inference_mode()
def raw_forward(
    processor, model, image_rgb: np.ndarray, queries: Sequence[str], device: str
) -> tuple[np.ndarray, np.ndarray]:
    """底层前向：返回 (boxes_xyxy 原图像素, per_query_scores)。

    **全项目只有这一处做 OWLv2 的坐标换算。** 早期版本里
    `detect()` 走 `post_process_object_detection`、prompt 选型走手算，
    同一几何逻辑有两条路径 —— 这正是静默不一致的温床，已合并。

    不用 `post_process_object_detection` 的另一个原因：它会先对 query 取
    max，per-query 分数就丢了，而 prompt 选型需要按子集取 max
    （OWLv2 的 query 之间互不影响，所以一次前向能评所有候选）。
    """
    from PIL import Image

    H, W = image_rgb.shape[:2]
    side = float(max(H, W))  # padded 方形边长，见模块说明

    inputs = processor(text=[list(queries)], images=Image.fromarray(image_rgb),
                       return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)

    # logits (1, P, Q) → sigmoid 得分；pred_boxes (1, P, 4) 是相对 padded
    # 方形归一化的 cxcywh
    scores = torch.sigmoid(out.logits[0]).detach().to("cpu").numpy()
    cxcywh = out.pred_boxes[0].detach().to("cpu").numpy()

    cx, cy, w, h = cxcywh.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1) * side
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H)

    keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
    return boxes[keep].astype("float32"), scores[keep].astype("float32")


def detect(
    processor,
    model,
    image_rgb: np.ndarray,
    queries: Sequence[str],
    *,
    device: str = "cpu",
    threshold: float = 0.02,
    max_dets: int = 100,
    image_id: int = 0,
) -> Owlv2Output:
    """对单张图做开放词表检测。多个 query 取逐框最高分。

    threshold 取很低的 0.02：AP 是排序指标，低分框排在后面不伤 AP
    （S5-K6 已实测），砍掉只会白损召回。
    """
    boxes, per_q = raw_forward(processor, model, image_rgb, queries, device)
    if not len(boxes):
        return Owlv2Output(image_id, boxes, np.zeros(0, "float32"),
                           np.zeros(0, "int32"))

    best_q = per_q.argmax(axis=1).astype("int32")
    scores = per_q.max(axis=1)

    keep = scores >= threshold
    boxes, scores, best_q = boxes[keep], scores[keep], best_q[keep]

    if len(scores) > max_dets:
        order = np.argsort(-scores)[:max_dets]
        boxes, scores, best_q = boxes[order], scores[order], best_q[order]

    return Owlv2Output(image_id=image_id, boxes_xyxy=boxes,
                       scores=scores, query_idx=best_q)


def synthetic_alignment_check(
    processor, model, *, device: str = "cpu"
) -> tuple[bool, list[dict]]:
    """决定性几何验证：在已知位置贴色块，看框能不能落上去。

    构造三张**非正方形**的图（正方形图检不出 padding 错误，因为 S=W=H），
    每张在一个已知象限贴一个纯色块，用对应 prompt 检测，
    检查最高分框的中心是否落在色块内。

    若 target_sizes 传成了 (H, W)，框会沿长边被拉伸，中心明显偏移 → 检出。
    """
    cases = [
        # (W, H, 色块 x1,y1,x2,y2, 颜色, prompt)
        (640, 360, (40, 40, 200, 160), (220, 30, 30), "a red square"),
        (360, 640, (200, 420, 330, 590), (30, 90, 220), "a blue square"),
        (800, 400, (560, 240, 760, 380), (30, 180, 60), "a green square"),
    ]
    results: list[dict] = []
    all_ok = True

    for W, H, (bx1, by1, bx2, by2), color, prompt in cases:
        img = np.full((H, W, 3), 245, dtype="uint8")
        img[by1:by2, bx1:bx2] = color
        out = detect(processor, model, img, [prompt], device=device,
                     threshold=0.05, image_id=0)
        if not len(out.boxes_xyxy):
            results.append({"prompt": prompt, "wh": (W, H), "detected": False,
                            "center_inside": False})
            all_ok = False
            continue
        best = out.boxes_xyxy[int(np.argmax(out.scores))]
        cx, cy = (best[0] + best[2]) / 2, (best[1] + best[3]) / 2
        inside = (bx1 <= cx <= bx2) and (by1 <= cy <= by2)
        # 也报中心偏移量，便于判断错位模式
        tx, ty = (bx1 + bx2) / 2, (by1 + by2) / 2
        results.append({
            "prompt": prompt, "wh": (W, H), "detected": True,
            "gt_box": (bx1, by1, bx2, by2),
            "pred_box": tuple(round(float(v), 1) for v in best),
            "center_offset_px": (round(float(cx - tx), 1), round(float(cy - ty), 1)),
            "center_inside": bool(inside),
            "score": round(float(out.scores.max()), 4),
        })
        all_ok = all_ok and inside

    return all_ok, results
