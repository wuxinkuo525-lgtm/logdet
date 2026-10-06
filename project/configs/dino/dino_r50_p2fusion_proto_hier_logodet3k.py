# DINO-R50 + Enhanced P2 加权融合 + Prototype + Hierarchy，LogoDet-3K 3000 细类。
#
# 基座：MMDetection 原生 dino-4scale_r50_improved_8xb2-12e_coco（配置与权重都只用这一份）。
# 相对基座的改动只有四处：
#   1. backbone 多输出一级 C2
#   2. neck 换成 EnhancedP2FPN，喂给 DINO 的四个尺度变为 stride 4/8/16/32
#   3. bbox_head 换成 ProtoHierDINOHead，3000 类 + 两个辅助分支
#   4. 数据集换成 runs/coco/ 下已冻结 split 导出的两份 COCO 标注
#
# 路径只依赖环境变量 LOGDET_ROOT（与 configs/paths.yaml 同一约定），
# 换机器/上服务器不用改本文件。

# `_base_` 必须写在 import 之前：mmengine 靠第一条语句判断配置风格，
# 先看到 import 会把本文件当成 lazy-import 风格，与基座不匹配而报错。
_base_ = "mmdet::dino/dino-4scale_r50_improved_8xb2-12e_coco.py"

import json  # noqa: E402
import os  # noqa: E402

custom_imports = dict(imports=["logodet.dino"], allow_failed_imports=False)
custom_hooks = [dict(type="FusionWeightLoggerHook"), dict(type="TrainingHealthHook")]

logdet_root = os.path.expanduser(os.environ.get("LOGDET_ROOT", "~/Desktop/logdet")).replace("\\", "/")
data_root = os.environ.get("LOGDET_DATA_ROOT", logdet_root + "/data/LogoDet-3K").replace("\\", "/") + "/"
runs_root = os.environ.get("LOGDET_RUNS_ROOT", logdet_root + "/runs").replace("\\", "/")
coco_dir = runs_root + "/coco"

train_ann = coco_dir + "/instances_train_dino.json"
val_ann = coco_dir + "/instances_val_dino.json"
fine_to_super_file = coco_dir + "/fine_to_super_mapping.json"

# 类名顺序 = category_id 升序，于是 label = class_id - 1（head 里的 fine→super 查表依赖这一点）
with open(val_ann, encoding="utf-8") as f:
    classes = tuple(c["name"] for c in sorted(json.load(f)["categories"], key=lambda c: c["id"]))
del f
num_classes = len(classes)
metainfo = dict(classes=classes)

load_from = (
    runs_root
    + "/cache/checkpoints/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth"
)

model = dict(
    # init_cfg=None：backbone 权重全部来自 load_from，不再去下 torchvision 的 resnet50
    # （集群计算节点不一定能联网）
    backbone=dict(out_indices=(0, 1, 2, 3), init_cfg=None),
    neck=dict(
        _delete_=True,
        type="EnhancedP2FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_cfg=dict(type="GN", num_groups=32),
    ),
    bbox_head=dict(
        type="ProtoHierDINOHead",
        num_classes=num_classes,
        fine_to_super_file=fine_to_super_file,
        num_superclasses=9,
        proto_dim=128,
        proto_temperature=0.07,
        loss_prototype_weight=0.1,
        loss_hierarchy_weight=0.1,
    ),
)

# LogoDet-3K 原图约 480×400。stride-4 的 P2 进 encoder 后 token 数是基座的 ~4 倍，
# 所以不照搬 COCO 的 800×1333，短边在 384–608 之间多尺度。
train_scales = [(s, 800) for s in range(384, 609, 32)]
test_scale = (512, 800)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomChoiceResize", scales=train_scales, keep_ratio=True),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=test_scale, keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img=""),
        metainfo=metainfo,
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img=""),
        metainfo=metainfo,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

# 两个口径一起报：coco/* 是 3000 类（框对且品牌认对），agn/* 是单类（只问框没框到 logo，
# 与 BASELINE_EVALUATION 同口径）。基座的 val_evaluator 是 dict，这里整个换成 list。
val_evaluator = [
    dict(
        type="CocoMetric",
        ann_file=val_ann,
        metric="bbox",
        format_only=False,
        backend_args=None,
    ),
    dict(type="ClassAgnosticCocoMetric", ann_file=val_ann),
]
test_evaluator = val_evaluator
