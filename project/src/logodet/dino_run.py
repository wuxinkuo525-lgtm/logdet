"""Portable DINO artifacts and verified final evaluations (no MMDetection import)."""
from __future__ import annotations

import json
import math
import hashlib
import os
import time
import warnings
from pathlib import Path, PureWindowsPath


def scalar_records(path: Path):
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not line.endswith(('\n', '\r')):
                warnings.warn(f'Ignoring interrupted final JSON record: {path}', RuntimeWarning)
            else:
                raise


def progress(phase: str, **details) -> None:
    target = os.environ.get('LOGDET_PROGRESS_FILE')
    if target:
        write_json_atomic(Path(target), dict(time=time.time(), phase=phase, **details))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_prediction_bundle(pred_dir: Path, required_ids=None, *, allow_partial=False) -> dict:
    import numpy as np
    man = json.loads((pred_dir / 'manifest.json').read_text(encoding='utf-8'))
    if man.get('schema_version') != 2 or man.get('status') != 'complete':
        raise ValueError('Prediction bundle is unverified/legacy; regenerate it before reporting')
    if not allow_partial and man.get('partial', True):
        raise ValueError('Rehearsal/partial predictions cannot be used for a formal report')
    for name in ('predictions.npz', 'own_eval_ann.json'):
        if file_sha256(pred_dir / name) != man['sha256'][name]:
            raise ValueError(f'Prediction bundle changed or download incomplete: {name}')
    receipt = man['final_evaluation']
    step = receipt['eval_step']
    if step != receipt['max_iters'] or step != receipt['checkpoint_iter']:
        raise ValueError('Prediction checkpoint was not finally evaluated')
    if receipt['checkpoint'] != man['checkpoint_identity']:
        raise ValueError('Prediction checkpoint does not match its final evaluation')
    if not math.isfinite(float(receipt['metrics']['agn/AP'])):
        raise ValueError('Invalid final AP')
    with np.load(pred_dir / 'predictions.npz', allow_pickle=False) as z:
        ids = z['processed_image_ids'].tolist()
        if len(ids) != len(set(ids)) or len(ids) != man['n_images']:
            raise ValueError('Invalid processed image inventory')
        if required_ids is not None and not set(required_ids).issubset(ids):
            raise ValueError('Predictions do not cover every evaluation image')
        for kind in ('agnostic', 'classwise'):
            boxes, scores = z[f'{kind}_box'], z[f'{kind}_score']
            image_ids, labels = z[f'{kind}_image_id'], z[f'{kind}_category_id']
            if boxes.shape != (len(scores), 4) or len(image_ids) != len(scores) or len(labels) != len(scores):
                raise ValueError('Prediction arrays have inconsistent shapes')
            if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                raise ValueError('Non-finite prediction boxes/scores')
            if not set(image_ids.tolist()).issubset(ids) or (scores < 0).any() or (scores > 1).any():
                raise ValueError('Invalid prediction image IDs/scores')
            limit = 1 if kind == 'agnostic' else man['conditions']['num_classes']
            if (labels < 1).any() or (labels > limit).any() or (boxes[:, 2:] < boxes[:, :2]).any():
                raise ValueError('Invalid prediction labels/box extents')
    return man


def write_json_atomic(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def export_eval_ann(source: str | Path, pred_dir: Path) -> str:
    name = 'own_eval_ann.json'
    write_json_atomic(pred_dir / name, json.loads(Path(source).read_text(encoding='utf-8')))
    return name


def resolve_eval_ann(manifest: dict, pred_dir: Path) -> Path:
    raw = manifest['own_eval_ann']
    path = Path(raw)
    if not path.is_absolute() and not PureWindowsPath(raw).is_absolute():
        path = pred_dir / path
    if not path.is_file():
        raise FileNotFoundError(
            f'Evaluation annotation missing: {path}. Regenerate the prediction bundle '
            'with s9_dino_predict.py and download the entire run directory; '
            'legacy absolute paths require the original annotation file.')
    return path


def checkpoint_path(run_dir: Path) -> Path:
    raw = (run_dir / 'last_checkpoint').read_text(encoding='utf-8').strip()
    path = Path(raw)
    return path if path.is_absolute() else run_dir / path


def checkpoint_identity(path: Path) -> dict:
    stat = path.stat()
    return dict(name=path.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def validate_final_eval(run_dir: Path, metric: str = 'agn/AP') -> dict:
    receipt = json.loads((run_dir / 'final_eval.json').read_text(encoding='utf-8'))
    step = receipt['eval_step']
    if step != receipt['checkpoint_iter'] or step != receipt['max_iters']:
        raise ValueError(f'{run_dir}: final evaluation/checkpoint/plan steps do not match')
    if receipt['checkpoint'] != checkpoint_identity(checkpoint_path(run_dir)):
        raise ValueError(f'{run_dir}: final evaluation belongs to a different checkpoint')
    if receipt['metrics'][metric] is None or not math.isfinite(float(receipt['metrics'][metric])):
        raise ValueError(f'{run_dir}: non-finite final metric {metric}')
    return receipt


def checkpoint_iteration(path: Path) -> int:
    import torch
    return int(torch.load(path, map_location='cpu')['meta']['iter'])


def finish_evaluation(runner, latest: tuple | None) -> None:
    """Run before loggers close; recover validation interrupted at max_iters."""
    run_dir = Path(runner.work_dir)
    path = checkpoint_path(run_dir)
    step = checkpoint_iteration(path)
    if step != runner.iter or step != runner.max_iters:
        raise ValueError('Refusing to mark an unfinished checkpoint as finally evaluated')
    try:
        receipt = validate_final_eval(run_dir)
        if receipt['max_iters'] == runner.max_iters:
            return
    except (OSError, ValueError, KeyError):
        pass
    metrics = latest[1] if latest and latest[0] == step else runner.val_loop.run()
    if not math.isfinite(float(metrics['agn/AP'])):
        raise ValueError('Final agn/AP must be finite')
    write_json_atomic(run_dir / 'final_eval.json', dict(
        checkpoint=checkpoint_identity(path), checkpoint_iter=step,
        eval_step=step, max_iters=runner.max_iters, seed=runner.seed,
        metrics={k: float(v) if math.isfinite(float(v)) else None for k, v in metrics.items()},
    ))


class PlanMismatch(ValueError):
    """这次要求的实验设置与该输出目录里已有的实验不一致。"""


def check_plan(run_dir: Path, plan: dict, *, accept_change: bool = False) -> str:
    """把实验计划（解析后的超参）与 run_dir/plan.json 对账，返回 'new' / 'match' / 'legacy' / 'changed'。

    同名输出目录只能装同一个实验：续训或「已完成，跳过」之前先对账，设置变了就拒绝，
    否则改了 epoch 数 / 输入尺寸 / 辅助分支之后会悄悄沿用旧结果。
    目录里已有存档却没有 plan.json 的是加这个检查之前跑的旧实验，无从核对，按原样接收并补记。
    """
    plan = json.loads(json.dumps(plan))  # 元组 → 列表等，与读回来的形式一致
    path = run_dir / 'plan.json'
    if not path.is_file():
        legacy = run_dir.is_dir() and (any(run_dir.glob('*.pth')) or (run_dir / 'DONE').is_file())
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, {'plan': plan, 'adopted_without_check': legacy})
        return 'legacy' if legacy else 'new'
    old = json.loads(path.read_text(encoding='utf-8'))['plan']
    if old == plan:
        return 'match'
    diff = {k: (old.get(k), plan.get(k)) for k in sorted(set(old) | set(plan)) if old.get(k) != plan.get(k)}
    if not accept_change:
        lines = '\n'.join(f'    {k}: 已有 {a!r} -> 这次 {b!r}' for k, (a, b) in diff.items())
        raise PlanMismatch(
            f'{run_dir} 里已有的实验与这次要求的设置不同：\n{lines}\n'
            '换一个 --run-name 另起一组；确实要在原目录上改设置续跑（比如延长恒定学习率的长跑），'
            '加 --accept-plan-change，并自行删掉该目录下的 DONE。')
    write_json_atomic(path, {'plan': plan, 'adopted_without_check': False, 'changed_from': old})
    return 'changed'
