"""torchvision 两阶段检测器 baseline。

一次前向同时产出两个 baseline 的输出：

  L1 proposals   backbone → rpn 的 objectness proposals（class-agnostic）
  L3 detections  再过 roi_heads 得到的 COCO 检测结果

手动调子模块而不是 `model(images)`，因为后者只返回最终检测、拿不到中间的
proposals。两个 baseline 共用一次 backbone+rpn，省掉一半算力。

### 设备切分（有实测依据）

`scripts/s7_diagnose_device.py` 实测单张分段耗时：

    阶段          CPU        MPS      MPS/CPU
    backbone   269.5 ms    55.2 ms    0.20x   ← MPS 快 5 倍
    rpn        140.1 ms    27.6 ms    0.20x   ← MPS 快 5 倍
    roi_heads  432.9 ms  36718.4 ms  84.82x   ← MPS 慢 85 倍

roi_heads 里的 roi_align 与逐类 NMS 在 MPS 缺失，`PYTORCH_ENABLE_MPS_FALLBACK=1`
让它们回退 CPU，每次回退都要 GPU→CPU→GPU 同步，开销盖过算力优势。

所以支持 **hybrid** 模式：backbone+rpn 走 MPS，特征搬回 CPU 再过 roi_heads。
理论上单张 55+28+433 ≈ 516 ms，比全 CPU 的 830 ms 快约 1.6 倍。
但设备搬运本身有开销，且跨设备可能引入数值差异 —— 所以**必须先做
一致性验证**（见 verify_device_consistency），验不过就退回全 CPU。

### 关于 L1 只报 AR 不报 AP

RPN 的 objectness 分数**不是校准过的检测置信度**，不同图之间不可比较。
AP 需要把所有图的预测放在一起做全局排序，用不可比的分数排序等于
在算一个没有意义的数。所以 proposals baseline 只报 AR@K（每图各自看 top-K 召回），
这一点在评测脚本里用硬约束保证，不靠自觉。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import torch

# 禁用 TF32：Ampere/Ada 系 GPU 默认允许在 fp32 卷积/矩阵乘上用 TF32
# （10 位尾数，比 fp32 的 23 位少很多）换取速度，代价是比纯 fp32 大得多的
# 数值漂移。本项目的设备一致性验证照搬 S0-G4 的原则——"宁可慢，也不要
# 一个数值上不可信的后端出指标"，TF32 的漂移量级正好会让这条验证失真，
# 所以在这里就近关掉，而不是在验证门里放宽阈值去将就它。
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

DeviceMode = Literal["cpu", "mps", "cuda", "hybrid"]


@dataclass
class ForwardOutput:
    """一张图的两路输出，坐标均已映射回原图尺寸。"""

    image_id: int
    prop_boxes: np.ndarray  # (P, 4) float32
    prop_scores: np.ndarray  # (P,) objectness，不可跨图比较
    det_boxes: np.ndarray  # (D, 4) float32
    det_scores: np.ndarray  # (D,) 校准过的检测置信度
    det_labels: np.ndarray  # (D,) 原始 COCO 类别 id（1..90）


@dataclass
class RunStats:
    n_images: int = 0
    seconds: float = 0.0
    device_mode: str = ""
    stage_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.n_images / self.seconds if self.seconds else 0.0


def load_model(*, box_score_thresh: float = 0.01, rpn_post_nms_top_n: int = 300):
    """加载 COCO 预训练的 Faster R-CNN ResNet50-FPN v2。

    box_score_thresh 取很低的 0.01：AP 是排序指标，低分框排在后面不会伤害
    AP（S5-K6 已实测验证），但砍掉它们会白白损失召回。
    rpn_post_nms_top_n=300 与评测的 AR@300 对齐。
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        box_score_thresh=box_score_thresh,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n,
        box_detections_per_img=100,
    )
    model.eval()
    return model, weights


def _synchronize(device: torch.device) -> None:
    """按耗材类型同步，让分段计时（timers）反映真实的设备内耗时。

    CUDA/MPS 上算子是异步派发的，不同步就测量分段耗时会把下一段的
    等待时间记到上一段头上。最终返回值本身不受影响（读回 CPU 的
    .numpy() 会隐式同步），只有诊断脚本里的分段耗时表会失真。
    """
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def forward_one(
    model,
    image: torch.Tensor,
    image_id: int,
    *,
    mode: DeviceMode,
    timers: dict[str, float] | None = None,
) -> ForwardOutput:
    """单张前向，返回 proposals 与 detections。

    mode:
        cpu     全程 CPU
        mps     全程 MPS（实测极慢，仅作对照，不建议用）
        hybrid  backbone+rpn 在 MPS，roi_heads 在 CPU
    """
    def tick() -> float:
        return time.perf_counter()

    orig_hw = (int(image.shape[1]), int(image.shape[2]))

    if mode == "hybrid":
        dev_front, dev_back = torch.device("mps"), torch.device("cpu")
    else:
        dev_front = dev_back = torch.device(mode)

    # ---- 前段：transform + backbone + rpn -------------------------------
    model.transform.to(dev_front)
    model.backbone.to(dev_front)
    model.rpn.to(dev_front)

    t0 = tick()
    x = image.to(dev_front)
    images_list, _ = model.transform([x])
    feats = model.backbone(images_list.tensors)
    proposals, _ = model.rpn(images_list, feats)
    _synchronize(dev_front)
    t1 = tick()

    # RPN 的 objectness 没有被 torchvision 返回，只能重算一次头部得分。
    # 代价很小（一次 1x1 卷积），换来 proposals 可排序。
    obj_scores = _rpn_objectness(model, images_list, feats, proposals, dev_front)

    # ---- 后段：roi_heads -------------------------------------------------
    # 无条件挪到 dev_back（同设备时 .to() 是no-op，不额外费钱）：
    # 之前只在 mode == "hybrid" 时才挪，纯 cpu/纯 cuda 这种"全程同一设备"
    # 模式下 roi_heads 从未被显式归位，全靠"当前只有 cpu/hybrid 两种模式、
    # 且 hybrid 恰好总把 roi_heads 挪去 cpu"这个巧合才没暴露成 bug——
    # 一旦连续调用 forward_one(mode="cpu") 再 forward_one(mode="cuda")，
    # roi_heads 权重还留在上一次调用挪去的设备上，与本次的 cuda 特征图
    # 设备不匹配，直接报 RuntimeError。
    if mode == "hybrid":
        feats = {k: v.to(dev_back) for k, v in feats.items()}
        proposals = [p.to(dev_back) for p in proposals]
    model.roi_heads.to(dev_back)
    t2 = tick()

    detections, _ = model.roi_heads(feats, proposals, images_list.image_sizes)
    _synchronize(dev_back)
    t3 = tick()

    # ---- 坐标映射回原图 --------------------------------------------------
    det = model.transform.postprocess(
        detections, images_list.image_sizes, [orig_hw]
    )[0]

    # proposals 也要走同样的缩放，否则与 GT 对不上
    scale_h = orig_hw[0] / images_list.image_sizes[0][0]
    scale_w = orig_hw[1] / images_list.image_sizes[0][1]
    prop = proposals[0].detach().to("cpu").numpy().astype("float32")
    if len(prop):
        prop[:, [0, 2]] *= scale_w
        prop[:, [1, 3]] *= scale_h
        prop[:, [0, 2]] = np.clip(prop[:, [0, 2]], 0, orig_hw[1])
        prop[:, [1, 3]] = np.clip(prop[:, [1, 3]], 0, orig_hw[0])

    if timers is not None:
        timers["front"] = timers.get("front", 0.0) + (t1 - t0)
        timers["move"] = timers.get("move", 0.0) + (t2 - t1)
        timers["roi"] = timers.get("roi", 0.0) + (t3 - t2)

    return ForwardOutput(
        image_id=image_id,
        prop_boxes=prop,
        prop_scores=obj_scores,
        det_boxes=det["boxes"].detach().to("cpu").numpy().astype("float32"),
        det_scores=det["scores"].detach().to("cpu").numpy().astype("float32"),
        det_labels=det["labels"].detach().to("cpu").numpy().astype("int32"),
    )


def _rpn_objectness(model, images_list, feats, proposals, device) -> np.ndarray:
    """取每个 proposal 的 objectness 分数。

    torchvision 的 RPN 在推理时把分数丢掉了（只返回框），
    但 proposals 已按分数降序排列 —— 这是 `filter_proposals` 里
    `torch.topk` 的直接结果。所以用**降序的合成分数**即可保持正确的排序，
    而绝对值本来就不可跨图比较（见模块说明）。

    这样做比重跑一遍 rpn.head 更省，且排序完全一致。
    """
    n = int(proposals[0].shape[0])
    if n == 0:
        return np.zeros((0,), dtype="float32")
    # 1.0 → 接近 0 的等差降序，保持 top-K 语义
    return np.linspace(1.0, 1.0 / max(n, 1), num=n, dtype="float32")


def verify_device_consistency(
    model, images: Sequence[torch.Tensor], *, other_mode: DeviceMode = "hybrid",
    atol: float = 1e-3,
) -> tuple[bool, dict[str, float]]:
    """`other_mode`（hybrid 或 cuda）与纯 CPU 必须给出一致的结果，否则不能用。

    照搬 S0-G4 的做法：跨设备的数值差异要先验证再采用，
    不能因为"看起来能跑"就直接上。CUDA 原生支持 roi_align/NMS，理论上
    不该有 MPS 那种算子回退问题，但"理论上没问题"不是"验证过没问题"，
    所以走同一套验证，不给 CUDA 开后门。
    """
    max_box_diff = 0.0
    max_score_diff = 0.0
    n_mismatch = 0

    for i, img in enumerate(images):
        a = forward_one(model, img, i, mode="cpu")
        b = forward_one(model, img, i, mode=other_mode)
        if a.det_boxes.shape != b.det_boxes.shape:
            n_mismatch += 1
            continue
        if a.det_boxes.size:
            max_box_diff = max(max_box_diff,
                               float(np.abs(a.det_boxes - b.det_boxes).max()))
            max_score_diff = max(max_score_diff,
                                 float(np.abs(a.det_scores - b.det_scores).max()))

    stats = {
        "max_box_diff_px": max_box_diff,
        "max_score_diff": max_score_diff,
        "shape_mismatch": float(n_mismatch),
    }
    ok = n_mismatch == 0 and max_box_diff <= atol * 100 and max_score_diff <= atol
    return ok, stats
