"""class-agnostic 单类口径的 COCO 指标，训练过程中每次评测顺带算一遍。

3000 类 mAP 要「框对 + 品牌认对」才算对，训练早期（或每类只有一两张图的子集上）
一直贴着 0，分不出两组超参的好坏。这里把 3000 个品牌塌缩成一个类「logo」，
只问「框没框到」——与 BASELINE_EVALUATION 的三个零样本 baseline 同一口径、
同一套 summarize（logodet.eval.coco_eval），数字可以直接并排看。

预测取自 ProtoHierDINOHead 附带输出的 agn_bboxes / agn_scores
（每个 query 只保留最高分的类，再取 top-K 个 query）。
"""

from __future__ import annotations

import json
from pathlib import Path

from mmdet.registry import METRICS
from mmengine.evaluator import BaseMetric
from pycocotools.coco import COCO

from logodet.eval.coco_eval import DEFAULT_MAX_DETS, evaluate

REPORTED = ("AP", "AP50", "AP75", "AP_small", "AR@100", "AR@300", "AR50@300")


def collapse_to_single_class(coco: dict) -> dict:
    """把一份多类 COCO 标注塌缩成 categories=[logo] 的单类标注（不改原 dict）。"""
    return {
        **coco,
        "categories": [{"id": 1, "name": "logo", "supercategory": "logo"}],
        "annotations": [{**a, "category_id": 1} for a in coco["annotations"]],
    }


@METRICS.register_module()
class ClassAgnosticCocoMetric(BaseMetric):
    default_prefix = "agn"

    def __init__(self, ann_file: str, collect_device: str = "cpu", prefix: str | None = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file

    def process(self, data_batch: dict, data_samples: list[dict]) -> None:
        for sample in data_samples:
            pred = sample["pred_instances"]
            boxes = pred["agn_bboxes"].cpu().numpy()
            scores = pred["agn_scores"].cpu().numpy()
            # 每张图只 append 一项：mmengine 汇总时会把 self.results 截到「数据集图数」那么长，
            # 按框 append 的话只剩下前几张图的框
            self.results.append([
                dict(image_id=sample["img_id"], category_id=1, bbox=[x1, y1, x2 - x1, y2 - y1], score=s)
                for (x1, y1, x2, y2), s in zip(boxes.tolist(), scores.tolist())
            ])

    def compute_metrics(self, results: list[list[dict]]) -> dict:
        gt = COCO()
        gt.dataset = collapse_to_single_class(json.loads(Path(self.ann_file).read_text(encoding="utf-8")))
        gt.createIndex()
        detections = [det for per_image in results for det in per_image]
        metrics = evaluate(gt, detections, max_dets=DEFAULT_MAX_DETS).metrics
        # 评测集里某个面积档一个 GT 都没有时该项是 nan；按 COCO / mmdet CocoMetric 的惯例报 -1，
        # 否则 nan 会一路进到最终评测回执（JSON 不允许 nan），训练在最后一步报错
        return {k: round(metrics[k], 4) if metrics[k] == metrics[k] else -1.0 for k in REPORTED}
