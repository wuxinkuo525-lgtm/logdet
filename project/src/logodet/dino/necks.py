"""高分辨率加权融合 neck。

    C2/C3/C4/C5 → FPN → P2/P3/P4/P5（全部 256 通道）
    P2_enh = α·P2 + β·Up(P3) + γ·Up(P4)      (α,β,γ) = softmax(3 个全局标量)
    Enhanced P2 = GN(Conv3×3(P2_enh))
    输出 [Enhanced P2, P3, P4, P5]

只增强 P2；P3/P4 原样透传；P5 不参与融合但保留给 DINO。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.necks import FPN
from mmdet.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class EnhancedP2FPN(BaseModule):
    def __init__(
        self,
        in_channels: Sequence[int] = (256, 512, 1024, 2048),
        out_channels: int = 256,
        norm_cfg: dict | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert len(in_channels) == 4, "需要 C2/C3/C4/C5 四级输入"
        norm_cfg = norm_cfg or dict(type="GN", num_groups=32)

        self.fpn = FPN(
            in_channels=list(in_channels),
            out_channels=out_channels,
            num_outs=4,
            norm_cfg=norm_cfg,
        )
        # 三个全局共享标量，初值相等 → softmax 后恰为 1/3、1/3、1/3
        self.fusion_logits = nn.Parameter(torch.zeros(3))
        self.fusion_conv = ConvModule(
            out_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=None
        )

    def fusion_weights(self) -> torch.Tensor:
        """softmax 归一化后的 (α, β, γ)。"""
        return self.fusion_logits.softmax(dim=0)

    def forward(self, inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        p2, p3, p4, p5 = self.fpn(inputs)
        size = p2.shape[-2:]
        w = self.fusion_weights()
        fused = (
            w[0] * p2
            + w[1] * F.interpolate(p3, size=size, mode="bilinear", align_corners=False)
            + w[2] * F.interpolate(p4, size=size, mode="bilinear", align_corners=False)
        )
        return self.fusion_conv(fused), p3, p4, p5
