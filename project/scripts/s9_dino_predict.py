#!/usr/bin/env python
"""S9 步骤三（上）：用一次 DINO 训练的 checkpoint 出预测，并把当时的超参 / 条件一起留痕。

推理范围 = baseline 用的 infer_union（val2k_repr ∪ val_hard_pool，3,079 图）
          + 这次训练自己的评测集（shared 模式是 256 张 dev_eval）。
每张图存两份输出：
    classwise  常规 top-300（query×3000 类），用来算「品牌也要认对」的 3000 类指标
    agnostic   每个 query 只留最高分的类再取 top-300，用来算与 baseline 同口径的单类指标

产物在 runs/predictions/dino/<run-name>/：
    predictions.npz   两份预测
    manifest.json     checkpoint、超参、数据、环境、训练末尾的 loss
    own_eval_ann.json 本次评测标注快照，可随预测目录跨平台下载

本脚本只在 DINO 环境（logodet_dino）里跑；出表在 baseline 环境里跑 s9_dino_report.py
（切片评测要读 parquet 表，DINO 环境没装 pyarrow）。

用法：
    python scripts/s9_dino_predict.py --run-name shared
    python scripts/s9_dino_predict.py --run-name shared_lr2e-4 [--checkpoint iter_20000.pth]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
os.environ.setdefault("LOGDET_ROOT", PROJECT_DIR.parent.as_posix())

import mmcv  # noqa: E402
import mmdet  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmdet.apis import inference_detector, init_detector  # noqa: E402
from mmengine.config import Config  # noqa: E402
from logodet.dino_run import (export_eval_ann, scalar_records, progress, validate_final_eval,
                             checkpoint_identity, file_sha256, write_json_atomic,
                             validate_prediction_bundle)  # noqa: E402


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def last_scalars(run_dir: Path) -> tuple[dict, list[dict]]:
    """训练日志里最后一条 train 记录，以及全部评测记录（没有日志就返回空）。"""
    logs = sorted(run_dir.glob("*/vis_data/scalars.json"))
    train, evals = {}, []
    for log in logs:
        for rec in scalar_records(log):
            if "loss" in rec:
                train = rec
            elif any(k.startswith(("coco/", "agn/")) for k in rec):
                evals.append({k: None if isinstance(v, float) and not math.isfinite(v) else v
                              for k, v in rec.items()})
    return train, evals


def conditions(cfg: Config, ckpt_path: Path, ckpt_meta: dict, run_dir: Path) -> dict:
    """这次训练的超参和条件，全部从训练时落盘的配置里读（不是从当前代码的默认值）。"""
    train_ds = cfg.train_dataloader.dataset
    train_ann = read_json(train_ds.ann_file)
    scales = [s for step in train_ds.pipeline if step["type"] == "RandomChoiceResize" for s in step["scales"]]
    test_scale = [step["scale"] for step in cfg.test_dataloader.dataset.pipeline if step["type"] == "Resize"]
    head = cfg.model.bbox_head
    last_train, evals = last_scalars(run_dir)
    loss_keys = ("loss", "loss_cls", "loss_bbox", "loss_iou", "dn_loss_cls", "loss_prototype", "loss_hierarchy",
                 "fusion_alpha", "fusion_beta", "fusion_gamma", "grad_norm", "lr")
    return dict(
        checkpoint=ckpt_path.as_posix(),
        checkpoint_iter=ckpt_meta.get("iter"),
        seed=ckpt_meta.get("seed", cfg.get("randomness", {}).get("seed")),
        max_iters=cfg.train_cfg.get("max_iters"),
        val_interval=cfg.train_cfg.get("val_interval"),
        pretrained=Path(str(cfg.load_from)).name if cfg.load_from else None,
        train_ann=Path(train_ds.ann_file).name,
        train_images=len(train_ann["images"]),
        train_boxes=len(train_ann["annotations"]),
        train_classes_present=len({a["category_id"] for a in train_ann["annotations"]}),
        num_classes=head.num_classes,
        batch_size=cfg.train_dataloader.batch_size,
        optimizer=cfg.optim_wrapper.optimizer.type,
        lr=cfg.optim_wrapper.optimizer.lr,
        weight_decay=cfg.optim_wrapper.optimizer.weight_decay,
        backbone_lr_mult=cfg.optim_wrapper.paramwise_cfg.custom_keys.backbone.lr_mult,
        grad_clip=cfg.optim_wrapper.clip_grad.max_norm,
        amp=cfg.optim_wrapper.type == "AmpOptimWrapper",
        param_scheduler=[dict(s) for s in cfg.param_scheduler],
        train_short_sides=sorted({s[0] for s in scales}),
        train_long_side_cap=max((s[1] for s in scales), default=None),
        test_scale=list(test_scale[0]) if test_scale else None,
        num_queries=cfg.model.num_queries,
        max_per_img=cfg.model.test_cfg.max_per_img,
        loss_prototype_weight=head.loss_prototype_weight,
        loss_hierarchy_weight=head.loss_hierarchy_weight,
        proto_temperature=head.proto_temperature,
        neck=cfg.model.neck.type,
        head=head.type,
        last_train_log={k: last_train[k] for k in ("iter", *loss_keys) if k in last_train},
        eval_log=evals,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True, help="runs/dino/<run-name>")
    ap.add_argument("--checkpoint", default=None, help="run 目录下的文件名，默认取 last_checkpoint 指向的那个")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-images", type=int, default=None,
                    help="只取 infer_union 的前 N 张（彩排用；这样出的预测不能拿去出正式报告）")
    args = ap.parse_args()

    runs_root = Path(os.environ.get("LOGDET_RUNS_ROOT", os.environ["LOGDET_ROOT"] + "/runs"))
    run_dir = runs_root / "dino" / args.run_name
    cfg_path = next(run_dir.glob("*.py"))
    ckpt = run_dir / args.checkpoint if args.checkpoint else Path(
        (run_dir / "last_checkpoint").read_text(encoding="utf-8").strip())
    cfg = Config.fromfile(cfg_path)
    receipt = validate_final_eval(run_dir)
    identity = checkpoint_identity(ckpt)
    if identity != receipt['checkpoint']:
        raise ValueError('Only the verified final checkpoint can be exported')
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError('--max-images must be positive')
    progress('predict_initializing')

    # 评测图清单：baseline 的 infer_union + 这次训练自己的评测集
    union = read_json(runs_root / "eval" / "gt" / "infer_union.json")["images"][:args.max_images]
    own_ann = cfg.val_dataloader.dataset.ann_file
    images = {im["id"]: im["file_name"] for im in union}
    images.update({im["id"]: im["file_name"] for im in read_json(own_ann)["images"]})
    data_root = Path(cfg.data_root)

    model = init_detector(str(cfg_path), str(ckpt), device=args.device)
    meta = torch.load(ckpt, map_location="cpu").get("meta", {})

    cols = {name: {k: [] for k in ("image_id", "box", "score", "category_id")} for name in ("classwise", "agnostic")}
    t0 = time.time()
    for n, (image_id, file_name) in enumerate(sorted(images.items()), 1):
        with torch.no_grad():
            pred = inference_detector(model, str(data_root / file_name)).pred_instances
        for values in (pred.bboxes, pred.scores, pred.agn_bboxes, pred.agn_scores):
            if not torch.isfinite(values).all():
                raise ValueError(f'Non-finite prediction for image {image_id}')
        for name, boxes, scores, cats in (
            ("classwise", pred.bboxes, pred.scores, pred.labels + 1),  # label 0-based → category_id 1..3000
            ("agnostic", pred.agn_bboxes, pred.agn_scores, torch.ones_like(pred.labels)),
        ):
            c = cols[name]
            c["image_id"].append(np.full(len(scores), image_id, dtype=np.int64))
            c["box"].append(boxes.cpu().numpy().astype(np.float32))
            c["score"].append(scores.cpu().numpy().astype(np.float32))
            c["category_id"].append(cats.cpu().numpy().astype(np.int16))
        if n == 1 or n % 50 == 0 or n == len(images):
            progress('predict', completed=n, total=len(images))
            print(f"{n:,}/{len(images):,}", flush=True)
    elapsed = time.time() - t0

    out_dir = runs_root / "predictions" / "dino" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    bundled_ann = export_eval_ann(own_ann, out_dir)
    progress('saving_predictions', completed=len(images))
    np.savez_compressed(
        out_dir / "predictions.tmp.npz",
        processed_image_ids=np.asarray(sorted(images), dtype=np.int64),
        **{f"{name}_{k}": np.concatenate(v) for name, c in cols.items() for k, v in c.items()},
    )
    (out_dir / 'predictions.tmp.npz').replace(out_dir / 'predictions.npz')
    manifest = dict(
        schema_version=2, status='complete', partial=args.max_images is not None or args.run_name.startswith('rehearsal_'),
        final_evaluation=receipt, checkpoint_identity=identity,
        sha256={name: file_sha256(out_dir / name) for name in ('predictions.npz', 'own_eval_ann.json')},
        run_name=args.run_name,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_images=len(images),
        n_infer_union=len(union),
        own_eval_ann=bundled_ann,
        own_eval_name=Path(own_ann).stem,
        img_per_sec=round(len(images) / elapsed, 2),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        versions=dict(torch=torch.__version__, mmdet=mmdet.__version__, mmcv=mmcv.__version__),
        conditions=conditions(cfg, ckpt, meta, run_dir),
    )
    write_json_atomic(out_dir / 'manifest.json', manifest)
    validate_prediction_bundle(out_dir, images, allow_partial=True)
    progress('predictions_verified', completed=len(images))
    print(f"{len(images):,} 图，{elapsed / 60:.1f} 分钟 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
