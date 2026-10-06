"""把 EnhancedP2FPN 的融合权重 α/β/γ 写进训练日志。

阶段总结第 9 节要求确认「softmax 后 α/β/γ 是否随训练发生变化」，
它们不属于任何 loss 项，默认不会出现在日志里。
"""

from __future__ import annotations

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmdet.registry import HOOKS
from logodet.dino_run import finish_evaluation, progress
import torch
import os


@HOOKS.register_module()
class TrainingHealthHook(Hook):
    priority = 'ABOVE_NORMAL'

    def before_train(self, runner):
        device = next(runner.model.parameters()).device
        if os.environ.get('LOGDET_REQUIRE_GPU') == '1' and device.type != 'cuda':
            raise RuntimeError('GPU job model is not on CUDA')
        progress('train_start', iteration=runner.iter, total=runner.max_iters, device=str(device))

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        for name, value in (outputs or {}).items():
            if ('loss' in name or 'grad' in name) and not torch.isfinite(torch.as_tensor(value)).all():
                raise FloatingPointError(f'Non-finite {name} at iteration {runner.iter + 1}')
        if runner.iter == 0 or (runner.iter + 1) % 50 == 0:
            progress('train', iteration=runner.iter + 1, total=runner.max_iters,
                     loss=float(outputs['loss']) if outputs and 'loss' in outputs else None,
                     gpu_allocated_mb=round(torch.cuda.memory_allocated() / 2**20) if torch.cuda.is_available() else None)

    def before_val(self, runner):
        progress('validation', iteration=runner.iter)

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        for sample in outputs or []:
            pred = sample.pred_instances
            for name in ('bboxes', 'scores', 'agn_bboxes', 'agn_scores'):
                if hasattr(pred, name) and not torch.isfinite(getattr(pred, name)).all():
                    raise FloatingPointError(f'Non-finite validation {name}')
        if batch_idx % 50 == 0:
            progress('validation', iteration=runner.iter, batches=batch_idx + 1)

    def after_val_epoch(self, runner, metrics):
        if 'agn/AP' in metrics and not torch.isfinite(torch.tensor(metrics['agn/AP'])):
            raise FloatingPointError('Non-finite validation agn/AP')
        progress('validation_complete', iteration=runner.iter)


@HOOKS.register_module()
class FinalEvaluationHook(Hook):
    """Only a completed final evaluation may be used by the sweep."""
    def __init__(self):
        self.latest = None

    def after_val_epoch(self, runner, metrics):
        self.latest = (runner.iter, dict(metrics))

    def after_train(self, runner):
        finish_evaluation(runner, self.latest)


@HOOKS.register_module()
class FusionWeightLoggerHook(Hook):
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        weights = model.neck.fusion_weights().detach().cpu().tolist()
        for name, value in zip(("alpha", "beta", "gamma"), weights):
            runner.message_hub.update_scalar(f"train/fusion_{name}", value)
