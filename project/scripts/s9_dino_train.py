#!/usr/bin/env python
"""S9 步骤二：DINO 训练入口（阶段总结 Step 5–8）。

    --mode tiny    tiny overfit：从正式 train split 固定抽 40 张图，训 300 iter，
                   并在同一批图上评测（看 loss 是否明显下降）
    --mode smoke   smoke training：正式 train 全集上跑 300 iter，
                   在 val 固定抽的 200 张图上评测（查 NaN / OOM / 稳定性）
    --mode shared  交接清单子集：只用 runs/handoff/shared_image_ids.json 里的 train_ids
                   （5,000 张）训练，在 dev_eval_ids（256 张）上评测。两份清单都在
                   冻结的 trainval 内，与检测头 / ViT 那条线用的是同一批图
    --mode full    完整训练，给服务器用：完整 train / 完整 val，按 iter 存档，
                   配 --resume 可跨多个限时作业续训

tiny / smoke 用到的子集标注写在 runs/coco/debug/，是从两份正式 COCO 标注里
按固定种子抽的子集——只是临时视图，不改动也不重新划分 trainval/val。

用法（logodet_dino 环境）：
    python scripts/s9_dino_train.py --mode tiny
    python scripts/s9_dino_train.py --mode smoke
    python scripts/s9_dino_train.py --mode shared [--resume]
    python scripts/s9_dino_train.py --mode full --resume [--epochs 4] [--lr 1e-4]

超参（epoch 数、batch、学习率计划、输入尺寸、辅助 loss 权重……）放在 configs/dino/hparams/*.yaml，
用 --hparams 指定；shared 默认读 shared5k.yaml。换数据规模时复制一份改数字即可。
--hparams 可以给多个文件，后面的覆盖前面的——实验变体写成只含几行的小文件
（configs/dino/hparams/mods/），叠在基础预设上：
    --hparams configs/dino/hparams/shared5k.yaml configs/dino/hparams/mods/noaux.yaml
优先级：命令行参数 > 预设（靠后的文件优先）> 模型配置默认值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
os.environ.setdefault("LOGDET_ROOT", PROJECT_DIR.parent.as_posix())

import yaml  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402

from logodet.dino_run import PlanMismatch, check_plan  # noqa: E402

DEFAULT_CONFIG = PROJECT_DIR / "configs" / "dino" / "dino_r50_p2fusion_proto_hier_logodet3k.py"
HPARAMS_DIR = PROJECT_DIR / "configs" / "dino" / "hparams"
# 各模式默认读的超参预设；没列出的模式不读预设，只用模型配置的默认值 + 命令行
DEFAULT_HPARAMS = {"shared": HPARAMS_DIR / "shared5k.yaml", "full": HPARAMS_DIR / "full141k.yaml"}
HPARAM_KEYS = {
    "epochs", "batch_size", "ckpt_interval",
    "lr", "backbone_lr_mult", "weight_decay", "warmup_iters", "lr_decay_epochs", "lr_decay_gamma",
    "min_short_side", "max_short_side", "max_long_side", "test_short_side",
    "keep_ckpt_every_epochs",
    "loss_prototype_weight", "loss_hierarchy_weight", "proto_temperature",
    "amp", "seed",
}
SUBSET_SEED = 20260917


def write_subset(
    src_ann: str, out_path: Path, *, n_images: int | None = None, image_ids: list[int] | None = None
) -> str:
    """从一份 COCO 标注里取子集，categories 保持完整 3000 类。

    给 image_ids 就严格按清单取（清单里有源标注没有的 id 直接报错）；
    否则按固定种子随机抽 n_images 张。
    """
    coco = json.loads(Path(src_ann).read_text(encoding="utf-8"))
    ids = sorted(im["id"] for im in coco["images"])
    if image_ids is not None:
        keep = set(image_ids)
        missing = keep - set(ids)
        if missing:
            raise ValueError(f"清单里有 {len(missing)} 个 image_id 不在 {Path(src_ann).name} 中，"
                             f"例如 {sorted(missing)[:5]}")
    else:
        keep = set(random.Random(SUBSET_SEED).sample(ids, n_images))
    sub = {
        **coco,
        "images": [im for im in coco["images"] if im["id"] in keep],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
    }
    sub["info"] = {**coco.get("info", {}), "debug_subset_of": Path(src_ann).name, "subset_seed": SUBSET_SEED}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    return out_path.as_posix()


def set_val_ann(cfg: Config, ann_file: str) -> None:
    """评测换一份标注：dataloader 和每个 evaluator（3000 类 + 单类口径）都要跟着换。"""
    cfg.val_dataloader.dataset.ann_file = ann_file
    for evaluator in cfg.val_evaluator:
        evaluator.ann_file = ann_file


def set_input_scales(
    cfg: Config,
    *,
    min_short_side: int | None = None,
    max_short_side: int | None = None,
    max_long_side: int | None = None,
    test_short_side: int | None = None,
) -> None:
    """改训练多尺度范围和测试尺寸；没给的项保持模型配置里的值（训练 384–608 / 长边 800，测试 512）。

    训练短边在 [min, max] 之间每 32 取一档。显存跟 token 数（≈ 面积）走：
    batch 1 时短边 608 实测约 7.4 GB，736 约 10 GB。
    """
    for step in cfg.train_dataloader.dataset.pipeline:
        if step["type"] == "RandomChoiceResize":
            shorts = [s[0] for s in step["scales"]]
            lo, hi = min_short_side or min(shorts), max_short_side or max(shorts)
            long_side = max_long_side or step["scales"][0][1]
            step["scales"] = [(s, long_side) for s in range(lo, hi + 1, 32)]
    for step in cfg.val_dataloader.dataset.pipeline:
        if step["type"] == "Resize" and (test_short_side or max_long_side):
            step["scale"] = (test_short_side or step["scale"][0], max_long_side or step["scale"][1])


def to_iter_based(cfg: Config, max_iters: int, scales: dict) -> None:
    """短跑模式：按 iter 计数、恒定学习率、结束时评测一次、不存 checkpoint。

    输入尺寸完整套用预设（训练短边范围、长边、测试尺寸），只是没指定短边上限时压到 480——
    否则拿它测某个尺寸变体的显存 / 速度，测到的不是那个变体。
    """
    set_input_scales(cfg, **{**scales, "max_short_side": scales.get("max_short_side") or 480})
    cfg.train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=max_iters)
    cfg.param_scheduler = []
    cfg.train_dataloader.sampler = dict(type="InfiniteSampler", shuffle=True)
    cfg.default_hooks.logger = dict(type="LoggerHook", interval=10)
    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook", by_epoch=False, interval=10**9, save_last=False
    )
    cfg.log_processor = dict(type="LogProcessor", window_size=10, by_epoch=False)


def load_hparams(paths: list[str | Path]) -> dict:
    """读超参预设 YAML，多个文件按顺序叠加（后面的覆盖前面的）。

    键名写错直接报错，免得一个拼写错误悄悄退回默认值。
    """
    merged: dict = {}
    for path in paths:
        hp = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(hp) - HPARAM_KEYS
        if unknown:
            raise ValueError(f"{path} 里有不认识的超参：{sorted(unknown)}；可用的是 {sorted(HPARAM_KEYS)}")
        merged.update(hp)
    return {k: v for k, v in merged.items() if v is not None}


def apply_hparams(cfg: Config, hp: dict, amp: bool) -> None:
    """把预设里与训练计划无关的那部分写进配置（训练计划在 to_resumable_full 里折算）。"""
    cfg.randomness = {**cfg.get("randomness", {}), "seed": int(hp.get("seed", SUBSET_SEED))}
    if hp.get("batch_size"):
        cfg.train_dataloader.batch_size = int(hp["batch_size"])
    if hp.get("lr"):
        cfg.optim_wrapper.optimizer.lr = float(hp["lr"])
    if "weight_decay" in hp:
        cfg.optim_wrapper.optimizer.weight_decay = float(hp["weight_decay"])
    if "backbone_lr_mult" in hp:
        cfg.optim_wrapper.paramwise_cfg.custom_keys.backbone.lr_mult = float(hp["backbone_lr_mult"])
    for key in ("loss_prototype_weight", "loss_hierarchy_weight", "proto_temperature"):
        if key in hp:
            cfg.model.bbox_head[key] = float(hp[key])
    if amp or hp.get("amp"):
        cfg.optim_wrapper.type = "AmpOptimWrapper"


def to_resumable_full(
    cfg: Config,
    epochs: int,
    ckpt_interval: int,
    *,
    lr_decay_epochs: list[int] | None = None,
    lr_decay_gamma: float = 0.1,
    warmup_iters: int = 0,
    keep_ckpt_every_epochs: int = 0,
) -> None:
    """完整训练：epoch 数折算成 iter，按 iter 存 checkpoint。

    集群单个作业有墙钟上限（TC2 默认 6 小时），一个 epoch 跑不完；
    按 epoch 存档的话被杀掉就白跑。折算后的学习率计划与配置等价：
    默认最后一个 epoch 开始时 ×0.1（lr_decay_epochs 可改），每个 epoch 结束时评测一次。
    """
    train_ann = cfg.train_dataloader.dataset.ann_file
    n_images = len(json.loads(Path(train_ann).read_text(encoding="utf-8"))["images"])
    iters_per_epoch = math.ceil(n_images / cfg.train_dataloader.batch_size)
    max_iters = epochs * iters_per_epoch
    if lr_decay_epochs is None:
        lr_decay_epochs = [max(epochs - 1, 1)] if epochs > 1 else []
    cfg.train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=iters_per_epoch)
    cfg.custom_hooks = [*cfg.get("custom_hooks", []), dict(type="FinalEvaluationHook")]
    cfg.param_scheduler = [
        dict(
            type="MultiStepLR",
            begin=0,
            end=max_iters,
            by_epoch=False,
            milestones=[e * iters_per_epoch for e in lr_decay_epochs if e < epochs],
            gamma=lr_decay_gamma,
        )
    ]
    if warmup_iters:
        cfg.param_scheduler.insert(
            0, dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=warmup_iters)
        )
    cfg.train_dataloader.sampler = dict(type="InfiniteSampler", shuffle=True)
    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook", by_epoch=False, interval=ckpt_interval, max_keep_ckpts=2
    )
    if keep_ckpt_every_epochs:
        # 滚动存档只留最近 2 个；这里另存一份永久的里程碑，供之后从某个 epoch 分叉出去（比如试不同的衰减位置）。
        # 文件名不能和滚动存档撞车，否则会被滚动清理掉
        cfg.custom_hooks = [*cfg.get("custom_hooks", []), dict(
            type="CheckpointHook", by_epoch=False, interval=keep_ckpt_every_epochs * iters_per_epoch,
            max_keep_ckpts=-1, save_last=False, filename_tmpl="milestone_iter_{}.pth",
        )]
    cfg.log_processor = dict(type="LogProcessor", window_size=50, by_epoch=False)
    print(f"{n_images:,} 图，batch {cfg.train_dataloader.batch_size}，"
          f"{iters_per_epoch:,} iter/epoch × {epochs} epoch = {max_iters:,} iter，"
          f"每 {ckpt_interval:,} iter 存档")
    print(f"lr {cfg.optim_wrapper.optimizer.lr:g}，warmup {warmup_iters:,} iter，"
          f"在 iter {cfg.param_scheduler[-1]['milestones']} 处 ×{lr_decay_gamma:g}")


def experiment_plan(cfg: Config, mode: str, init_from: str) -> dict:
    """决定「这是哪个实验」的全部设置（解析后的值）。只影响运行方式的不算：存档间隔、worker 数、输出目录。"""
    train_ds, head, opt = cfg.train_dataloader.dataset, cfg.model.bbox_head, cfg.optim_wrapper

    def image_digest(ann: str) -> str:
        ids = sorted(im["id"] for im in json.loads(Path(ann).read_text(encoding="utf-8"))["images"])
        return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]

    return dict(
        mode=mode,
        train_images=image_digest(train_ds.ann_file),
        eval_images=image_digest(cfg.val_dataloader.dataset.ann_file),
        init_from=init_from,
        max_iters=cfg.train_cfg.max_iters,
        val_interval=cfg.train_cfg.val_interval,
        batch_size=cfg.train_dataloader.batch_size,
        lr=opt.optimizer.lr,
        weight_decay=opt.optimizer.weight_decay,
        backbone_lr_mult=opt.paramwise_cfg.custom_keys.backbone.lr_mult,
        amp=opt.type == "AmpOptimWrapper",
        param_scheduler=[{k: s[k] for k in ("type", "start_factor", "end", "milestones", "gamma") if k in s}
                         for s in cfg.param_scheduler],
        train_scales=[list(s) for step in train_ds.pipeline if step["type"] == "RandomChoiceResize"
                      for s in step["scales"]],
        test_scale=[list(step["scale"]) for step in cfg.val_dataloader.dataset.pipeline if step["type"] == "Resize"],
        loss_prototype_weight=head.loss_prototype_weight,
        loss_hierarchy_weight=head.loss_hierarchy_weight,
        proto_temperature=head.proto_temperature,
        seed=cfg.randomness["seed"],
        milestone_interval=[h["interval"] for h in cfg.custom_hooks if h.get("filename_tmpl")],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("tiny", "smoke", "shared", "full"), required=True)
    ap.add_argument("--shared-ids", default=None, help="shared 模式的清单，默认 runs/handoff/shared_image_ids.json")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--max-iters", type=int, default=300, help="tiny / smoke 的迭代数")
    ap.add_argument("--tiny-images", type=int, default=40)
    ap.add_argument("--smoke-val-images", type=int, default=200)
    ap.add_argument("--max-short-side", type=int, default=None,
                    help="训练短边上限；tiny / smoke 默认 480，full 默认不限（用配置里的 608）")
    ap.add_argument("--epochs", type=int, default=None, help="full 的 epoch 数，默认取配置的 max_epochs")
    ap.add_argument("--ckpt-interval", type=int, default=None, help="shared / full 每多少 iter 存一次 checkpoint，默认 2000")
    ap.add_argument("--hparams", nargs="+", default=None,
                    help="超参预设 YAML（configs/dino/hparams/），可给多个，后面的覆盖前面的；"
                         "shared 默认 shared5k.yaml。命令行参数优先于预设")
    ap.add_argument("--init-from", default=None,
                    help="不从 COCO 预训练起步，改从这个 checkpoint 的权重起步（只取权重，学习率计划从头算）")
    ap.add_argument("--val-subset", choices=("val2k_repr", "val_hard_pool", "full"), default="val2k_repr",
                    help="full 模式每个 epoch 评测用的 val 范围")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None, help="模型初始化、采样和增强的随机种子，默认 20260917")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--run-name", default=None,
                    help="输出目录名：runs/dino/<run-name>，默认用模式名。每组实验给一个不同的名字，互不覆盖")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--plan-only", action="store_true",
                    help="只核对这次的设置与输出目录里已有实验是否一致，不训练（调参脚本跳过已完成的组之前用）")
    ap.add_argument("--accept-plan-change", action="store_true",
                    help="设置与输出目录里已有实验不同时仍在原目录上继续，并改写 plan.json")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir or f"{cfg.runs_root}/dino/{args.run_name or args.mode}"
    debug_dir = Path(cfg.coco_dir) / "debug"

    default_hp = DEFAULT_HPARAMS.get(args.mode)
    hp = load_hparams(args.hparams or ([default_hp] if default_hp else []))
    for key in ("epochs", "batch_size", "ckpt_interval", "lr", "max_short_side", "seed"):
        if getattr(args, key) is not None:
            hp[key] = getattr(args, key)
    apply_hparams(cfg, hp, args.amp)
    cfg.optim_wrapper.clip_grad['error_if_nonfinite'] = True
    schedule = dict(
        lr_decay_epochs=hp.get("lr_decay_epochs"),
        lr_decay_gamma=float(hp.get("lr_decay_gamma", 0.1)),
        warmup_iters=int(hp.get("warmup_iters", 0)),
        keep_ckpt_every_epochs=int(hp.get("keep_ckpt_every_epochs", 0)),
    )
    scales = {k: hp.get(k) for k in ("min_short_side", "max_short_side", "max_long_side", "test_short_side")}
    if args.init_from:
        cfg.load_from = args.init_from
    if args.num_workers is not None:
        cfg.train_dataloader.num_workers = args.num_workers
    elif os.name == "nt":
        # Windows 的每个 worker 进程都要各自加载一份 torch 的 CUDA DLL，占掉几 GB 提交内存；
        # 4+2 个 worker 实测会在评测汇总时把系统提交额度耗尽。读图不是瓶颈（data_time < 0.01s）。
        cfg.train_dataloader.num_workers = 2
        cfg.val_dataloader.num_workers = 0
        cfg.val_dataloader.persistent_workers = False
    # 续训时 work_dir 里还没有 checkpoint（第一个作业）就从预训练权重正常起步
    cfg.resume = args.resume and any(Path(cfg.work_dir).glob("*.pth"))
    init_from = str(cfg.load_from)  # 续训时下面会清掉 load_from；计划核对要的是「最初从哪起步」
    if cfg.resume:
        cfg.load_from = None  # 否则 mmengine 会从 load_from（COCO 预训练）而不是 work_dir 最新存档续
    epochs = hp.get("epochs") or cfg.max_epochs
    ckpt_interval = hp.get("ckpt_interval") or 2000

    if args.mode == "tiny":
        sub = write_subset(cfg.train_ann, debug_dir / f"tiny_train_{args.tiny_images}.json",
                           n_images=args.tiny_images)
        cfg.train_dataloader.dataset.ann_file = sub
        set_val_ann(cfg, sub)
        to_iter_based(cfg, args.max_iters, scales)
    elif args.mode == "smoke":
        sub = write_subset(cfg.val_ann, debug_dir / f"smoke_val_{args.smoke_val_images}.json",
                           n_images=args.smoke_val_images)
        set_val_ann(cfg, sub)
        to_iter_based(cfg, args.max_iters, scales)
    elif args.mode == "shared":
        # train_ids / dev_eval_ids 都在冻结的 trainval 内，所以两份子集都从 train 标注里取；
        # val 完全不碰，留给最终评测
        shared = json.loads(Path(args.shared_ids or f"{cfg.runs_root}/handoff/shared_image_ids.json")
                            .read_text(encoding="utf-8"))
        # 换了清单（比如彩排用的小清单）就换一套文件名，不覆盖正式清单导出的那两份——
        # 已跑完的组之后出预测时还要按落盘配置里的路径读它们
        tag = Path(args.shared_ids).stem.removesuffix("_image_ids") if args.shared_ids else "shared"
        train_sub = write_subset(cfg.train_ann, debug_dir / f"{tag}_train.json", image_ids=shared["train_ids"])
        dev_sub = write_subset(cfg.train_ann, debug_dir / f"{tag}_dev_eval.json", image_ids=shared["dev_eval_ids"])
        cfg.train_dataloader.dataset.ann_file = train_sub
        set_val_ann(cfg, dev_sub)
        set_input_scales(cfg, **scales)
        to_resumable_full(cfg, epochs, ckpt_interval, **schedule)
    else:
        if args.val_subset != "full":
            # 训练中途的评测默认只用 S3 冻结的 val 子集：pycocotools 按 (图, 类) 逐对建表，
            # 完整 val × 3000 类是 5 千多万对，内存要十几 GB
            subset_gt = Path(f"{cfg.runs_root}/eval/gt/{args.val_subset}.json")
            if not subset_gt.is_file():
                raise SystemExit(f"FAIL: full 模式每个 epoch 在 {args.val_subset} 上评测，需要 {subset_gt}"
                                 f"（本机 runs/eval/gt/ 下有，上传到集群同一位置；它和 infer_union.json 不能互相替代）")
            ids = [im["id"] for im in json.loads(subset_gt.read_text(encoding="utf-8"))["images"]]
            sub = write_subset(cfg.val_ann, Path(cfg.coco_dir) / "subsets" / f"instances_{args.val_subset}_dino.json",
                               image_ids=ids)
            set_val_ann(cfg, sub)
        set_input_scales(cfg, **scales)
        to_resumable_full(cfg, epochs, ckpt_interval, **schedule)
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = cfg.val_evaluator

    if args.mode in ("shared", "full"):
        try:
            status = check_plan(Path(cfg.work_dir), experiment_plan(cfg, args.mode, init_from),
                                accept_change=args.accept_plan_change)
        except PlanMismatch as exc:
            raise SystemExit(f"FAIL: {exc}")
        print({"new": "实验计划已记录到 plan.json", "match": "实验计划与 plan.json 一致",
               "legacy": "WARN: 该目录是加入计划核对之前跑的，无从核对，按原样接收并补记 plan.json",
               "changed": "WARN: 按 --accept-plan-change 改写了 plan.json"}[status])
    if args.plan_only:
        return 0

    Runner.from_cfg(cfg).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
