"""把 EnhancedP2FPN 的融合权重 α/β/γ 写进训练日志。

阶段总结第 9 节要求确认「softmax 后 α/β/γ 是否随训练发生变化」，
它们不属于任何 loss 项，默认不会出现在日志里。
"""

from __future__ import annotations

from collections import Counter

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
class ScheduleFromConfigHook(Hook):
    """续训后让「在哪一步降学习率」以当前配置为准。

    mmengine 续训时会把存档里的调度器状态整个恢复回来，连降学习率的位置（milestones）和
    调度器的作用区间（end）都是旧的——中途改了 epoch 数 / 衰减位置再续训，实际执行的仍是旧计划。
    这里在恢复之后把这两项改回配置里的值。已经降过学习率的不能反悔：那种情况直接报错。
    """

    def __init__(self, milestones: list[int], end: int) -> None:
        self.milestones = sorted(int(m) for m in milestones)
        self.end = int(end)

    def before_train(self, runner) -> None:
        schedulers = runner.param_schedulers
        if isinstance(schedulers, dict):
            schedulers = [s for group in schedulers.values() for s in group]
        for sched in schedulers:
            if type(sched).__name__ != "MultiStepLR":
                continue
            restored = sorted(sched.milestones.elements())
            if restored == self.milestones and sched.end == self.end:
                continue
            done_before = [m for m in restored if m <= runner.iter]
            done_now = [m for m in self.milestones if m <= runner.iter]
            if done_before != done_now:
                raise RuntimeError(
                    f"存档已在 iter {done_before} 降过学习率，新计划要求的是 {done_now}（当前 iter {runner.iter}）："
                    "已经执行过的降学习率改不回来，不能在这个存档上换计划。")
            runner.logger.info(
                f"学习率计划以当前配置为准：降 lr 的位置 {restored} → {self.milestones}，作用区间终点 {sched.end} → {self.end}")
            sched.milestones = Counter(self.milestones)
            sched.end = self.end


@HOOKS.register_module()
class FusionWeightLoggerHook(Hook):
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None) -> None:
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        weights = model.neck.fusion_weights().detach().cpu().tolist()
        for name, value in zip(("alpha", "beta", "gamma"), weights):
            runner.message_hub.update_scalar(f"train/fusion_{name}", value)
