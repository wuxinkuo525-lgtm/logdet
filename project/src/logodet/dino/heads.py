"""DINOHead + Prototype 分支 + Hierarchy 分支。

两个辅助分支并列，都从「最后一层 decoder、Hungarian 匹配上的正样本 query
特征 h (256-d)」出发：

    Prototype:  h → Linear 256→256 → ReLU → Linear 256→128 → L2 norm
                  → 与 3000×128 可学习 prototype 做 cosine / τ → CE
    Hierarchy:  h → Linear(256, 9) → CE（目标是细类对应的超类）

    L_total = L_DINO + λ_proto · L_prototype + λ_hier · L_hierarchy

background / no-object query 和 denoising query 都不参与辅助 loss。
一个 batch 没有任何正样本时，两个辅助 loss 安全返回 0（带计算图，不产生 NaN）。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.dense_heads import DINOHead
from mmdet.registry import MODELS


def load_fine_to_super(path: str | Path, num_classes: int, num_superclasses: int) -> torch.Tensor:
    """读 fine_to_super_mapping.json → 长度 num_classes 的 0-based 超类下标张量。

    文件里 class_id 是 1..3000、superclass_id 是 1..9；mmdet 的 label 是
    category_id 升序后的 0-based 下标，class_id 恰为连续 1..3000，所以
    label = class_id - 1，超类下标同理减 1。
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["fine_to_super"]
    if sorted(int(k) for k in raw) != list(range(1, num_classes + 1)):
        raise ValueError(f"{path} 的 class_id 不是完整的 1..{num_classes}")
    table = torch.tensor([int(raw[str(c)]) - 1 for c in range(1, num_classes + 1)], dtype=torch.long)
    if not (table.min() >= 0 and table.max() < num_superclasses):
        raise ValueError(f"{path} 的 superclass_id 超出 1..{num_superclasses}")
    return table


@MODELS.register_module()
class ProtoHierDINOHead(DINOHead):
    def __init__(
        self,
        *args,
        fine_to_super_file: str,
        num_superclasses: int = 9,
        proto_dim: int = 128,
        proto_temperature: float = 0.07,
        loss_prototype_weight: float = 0.1,
        loss_hierarchy_weight: float = 0.1,
        **kwargs,
    ) -> None:
        self.num_superclasses = num_superclasses
        self.proto_dim = proto_dim
        self.proto_temperature = proto_temperature
        self.loss_prototype_weight = loss_prototype_weight
        self.loss_hierarchy_weight = loss_hierarchy_weight
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "fine_to_super",
            load_fine_to_super(fine_to_super_file, self.num_classes, num_superclasses),
            persistent=False,
        )

    def _init_layers(self) -> None:
        super()._init_layers()
        self.proto_proj = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.proto_dim),
        )
        self.prototypes = nn.Parameter(torch.empty(self.num_classes, self.proto_dim))
        self.hier_fc = nn.Linear(self.embed_dims, self.num_superclasses)

    def init_weights(self) -> None:
        super().init_weights()
        nn.init.normal_(self.prototypes, std=1.0)
        for m in self.proto_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.hier_fc.weight)
        nn.init.zeros_(self.hier_fc.bias)

    # ------------------------------------------------------------------ aux
    def prototype_logits(self, h: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.proto_proj(h), dim=-1)
        protos = F.normalize(self.prototypes, dim=-1)
        return z @ protos.t() / self.proto_temperature

    def matched_positive_queries(
        self,
        hidden_states: torch.Tensor,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        batch_gt_instances: list,
        batch_img_metas: list,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """最后一层 matching query 里匹配上 GT 的那部分：(h [N,256], 细类 label [N])。

        用和主 loss 完全相同的 assigner、相同的输入重跑一次匈牙利匹配，
        结果与主 loss 最后一层的分配一致。
        """
        bs = cls_scores.size(0)
        with torch.no_grad():
            labels_list = self.get_targets(
                [cls_scores[i] for i in range(bs)],
                [bbox_preds[i] for i in range(bs)],
                batch_gt_instances,
                batch_img_metas,
            )[0]
        labels = torch.stack(labels_list)  # [bs, num_matching_queries]，背景 = num_classes
        pos = labels < self.num_classes
        return hidden_states[pos], labels[pos]

    def loss_aux_branches(self, h: torch.Tensor, labels: torch.Tensor) -> dict:
        if h.size(0) == 0:
            zero = (
                self.proto_proj[0].weight.sum() + self.prototypes.sum() + self.hier_fc.weight.sum()
            ) * 0.0
            return dict(loss_prototype=zero, loss_hierarchy=zero.clone())
        loss_proto = F.cross_entropy(self.prototype_logits(h), labels)
        loss_hier = F.cross_entropy(self.hier_fc(h), self.fine_to_super[labels])
        return dict(
            loss_prototype=self.loss_prototype_weight * loss_proto,
            loss_hierarchy=self.loss_hierarchy_weight * loss_hier,
        )

    # -------------------------------------------------------------- predict
    def _predict_by_feat_single(self, cls_score, bbox_pred, img_meta, rescale: bool = True):
        """在常规输出之外附带一份 class-agnostic 视图（agn_bboxes / agn_scores）。

        常规的 top-K 是在 query×3000 类上取的，同一个 query 会带着不同类别重复出现；
        单类口径下每个 query 只算一次，所以只保留它最高分的那个类再取 top-K 个 query。
        两份输出长度都是 max_per_img，可以放进同一个 InstanceData。
        """
        results = super()._predict_by_feat_single(cls_score, bbox_pred, img_meta, rescale)
        top, idx = cls_score.max(-1, keepdim=True)
        agnostic = super()._predict_by_feat_single(
            torch.full_like(cls_score, -1e4).scatter_(1, idx, top), bbox_pred, img_meta, rescale
        )
        results.agn_bboxes = agnostic.bboxes
        results.agn_scores = agnostic.scores
        return results

    # ----------------------------------------------------------------- loss
    def loss(
        self,
        hidden_states: torch.Tensor,
        references: list,
        enc_outputs_class: torch.Tensor,
        enc_outputs_coord: torch.Tensor,
        batch_data_samples: list,
        dn_meta: dict,
    ) -> dict:
        batch_gt_instances = [s.gt_instances for s in batch_data_samples]
        batch_img_metas = [s.metainfo for s in batch_data_samples]

        outs = self(hidden_states, references)
        losses = self.loss_by_feat(
            *outs, enc_outputs_class, enc_outputs_coord, batch_gt_instances, batch_img_metas, dn_meta
        )

        # denoising query 排在前面，辅助分支只看后面的 matching query
        num_dn = dn_meta["num_denoising_queries"] if dn_meta else 0
        all_cls, all_bbox = outs
        h, labels = self.matched_positive_queries(
            hidden_states[-1][:, num_dn:],
            all_cls[-1][:, num_dn:],
            all_bbox[-1][:, num_dn:],
            batch_gt_instances,
            batch_img_metas,
        )
        losses.update(self.loss_aux_branches(h, labels))
        return losses
