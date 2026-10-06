"""DINO（MMDetection）侧的自定义模块。

import 本包即把两个模块注册进 mmdet 的 MODELS 注册表，配置文件里通过
`custom_imports = dict(imports=["logodet.dino"])` 触发。

    EnhancedP2FPN        FPN + 高分辨率加权融合，输出 [Enhanced P2, P3, P4, P5]
    ProtoHierDINOHead    DINOHead + Prototype 分支 + 9 超类 Hierarchy 分支
    FusionWeightLoggerHook  把融合权重 α/β/γ 写进训练日志
    ClassAgnosticCocoMetric  单类口径（只问「框没框到 logo」）的 AP / AR，与 baseline 表同口径

本包依赖 mmdet/mmcv，只在 DINO 训练环境（logodet_dino）里可用；
S0–S7b 的 baseline 环境不 import 它。
"""

from .heads import ProtoHierDINOHead
from .hooks import FinalEvaluationHook, FusionWeightLoggerHook, TrainingHealthHook
from .metrics import ClassAgnosticCocoMetric
from .necks import EnhancedP2FPN

__all__ = ["ClassAgnosticCocoMetric", "EnhancedP2FPN", "FinalEvaluationHook", "FusionWeightLoggerHook", "ProtoHierDINOHead"]
