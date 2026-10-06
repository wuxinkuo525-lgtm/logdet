#!/usr/bin/env python
"""S9 步骤一：DINO 单 batch sanity check（阶段总结 Step 4 + 第 9 节重点检查项）。

不训练，只取 2 张正式 train 图做一次 forward + backward，逐条核对：
    - Enhanced P2 / P3 / P4 / P5 的 shape（256 通道，stride 4/8/16/32）
    - 3000 类 logits、9 类 hierarchy logits、128-d embedding、3000×128 prototype
    - 所有 loss finite
    - fusion 标量、prototype、hierarchy head、decoder、backbone 都收到梯度
    - COCO 预训练权重哪些加载了、哪些重新初始化

用法（logodet_dino 环境）：
    python scripts/s9_dino_sanity.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
os.environ.setdefault("LOGDET_ROOT", PROJECT_DIR.parent.as_posix())

import torch  # noqa: E402
from mmengine.config import Config  # noqa: E402
from mmengine.dataset import pseudo_collate  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmengine.runner import load_checkpoint  # noqa: E402

DEFAULT_CONFIG = PROJECT_DIR / "configs" / "dino" / "dino_r50_p2fusion_proto_hier_logodet3k.py"


def check(ok: bool, msg: str, failures: list[str]) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        failures.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    from mmdet.registry import DATASETS, MODELS

    cfg = Config.fromfile(args.config)
    init_default_scope("mmdet")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.environ.get('LOGDET_REQUIRE_GPU') == '1' and device != 'cuda':
        raise RuntimeError('GPU job cannot fall back to CPU')
    failures: list[str] = []

    print("=" * 78)
    print(f" S9 步骤一：DINO 单 batch sanity check（device={device}）")
    print("=" * 78)

    # ---- 模型 + 预训练权重 ---------------------------------------------------
    model = MODELS.build(cfg.model)
    model.init_weights()
    ckpt = load_checkpoint(model, cfg.load_from, map_location="cpu", strict=False, logger="silent")
    after = model.state_dict()
    # key 存在且 shape 一致的才会被 load_state_dict(strict=False) 真正写入
    loaded = {k for k, v in ckpt["state_dict"].items() if k in after and after[k].shape == v.shape}
    print("\n[1/4] 预训练权重加载（按顶层模块统计 已加载/总数 的张量个数）")
    for prefix in ("backbone", "neck", "encoder", "decoder", "bbox_head", "query_embedding",
                   "level_embed", "memory_trans", "dn_query_generator"):
        keys = [k for k in after if k.startswith(prefix)]
        n = sum(k in loaded for k in keys)
        print(f"      {prefix:<20} {n:>4} / {len(keys):<4}")
    check(any(k.startswith("backbone") for k in loaded), "backbone 加载了 COCO 预训练权重", failures)
    check(any(k.startswith("decoder") for k in loaded), "DINO decoder 加载了 COCO 预训练权重", failures)
    del ckpt
    model.to(device).train()

    # ---- 一个 batch ----------------------------------------------------------
    ds_cfg = cfg.train_dataloader.dataset.copy()
    ds_cfg["indices"] = 64
    dataset = DATASETS.build(ds_cfg)
    batch = pseudo_collate([dataset[i * 16] for i in range(args.batch_size)])
    data = model.data_preprocessor(batch, training=True)
    inputs, samples = data["inputs"], data["data_samples"]
    B, _, H, W = inputs.shape
    n_gt = sum(len(s.gt_instances) for s in samples)
    print(f"\n[2/4] batch：inputs {tuple(inputs.shape)}，GT 框 {n_gt} 个，"
          f"细类 label {sorted({int(x) for s in samples for x in s.gt_instances.labels})}")

    # ---- shape ---------------------------------------------------------------
    print("\n[3/4] 结构与维度")
    with torch.no_grad():
        feats = model.extract_feat(inputs)
    check(len(feats) == 4, f"neck 输出 4 个尺度（实际 {len(feats)}）", failures)
    for name, f, s in zip(("Enhanced P2", "P3", "P4", "P5"), feats, (4, 8, 16, 32)):
        exp = (B, 256, -(-H // s), -(-W // s))
        check(tuple(f.shape) == exp, f"{name:<11} shape {tuple(f.shape)}，期望 {exp}（stride {s}）", failures)

    head = model.bbox_head
    C = cfg.num_classes
    check(C == 3000 and head.cls_branches[-1].out_features == 3000,
          f"细类分类头输出维度 = {head.cls_branches[-1].out_features}", failures)
    check(head.hier_fc.out_features == 9, f"hierarchy 头输出维度 = {head.hier_fc.out_features}", failures)
    check(tuple(head.prototypes.shape) == (3000, 128),
          f"prototype 矩阵 shape = {tuple(head.prototypes.shape)}", failures)
    with torch.no_grad():
        z = torch.nn.functional.normalize(head.proto_proj(torch.randn(5, 256, device=device)), dim=-1)
    check(z.shape[-1] == 128 and torch.allclose(z.norm(dim=-1), torch.ones(5, device=device), atol=1e-5),
          f"prototype embedding 为 {z.shape[-1]}-d 且已 L2 归一化", failures)
    f2s = head.fine_to_super
    check(len(f2s) == 3000 and int(f2s.min()) == 0 and int(f2s.max()) == 8 and len(f2s.unique()) == 9,
          "fine→super 查表覆盖 3000 细类、9 个超类", failures)
    w = model.neck.fusion_weights().detach().cpu().tolist()
    check(all(abs(x - 1 / 3) < 1e-6 for x in w), f"fusion 初始权重 α/β/γ = {[round(x, 4) for x in w]}", failures)

    # ---- loss + 梯度 ---------------------------------------------------------
    print("\n[4/4] loss 与梯度")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    losses = model.loss(inputs, samples)
    total, log_vars = model.parse_losses(losses)
    for k in ("loss_cls", "loss_bbox", "loss_iou", "dn_loss_cls", "loss_prototype", "loss_hierarchy"):
        print(f"      {k:<16} {float(log_vars[k]):.4f}")
    print(f"      {'total (全部项)':<16} {float(total):.4f}   共 {len(losses)} 个 loss 项")
    check(all(torch.isfinite(torch.as_tensor(v)).all() for v in log_vars.values()), "所有 loss 项 finite", failures)
    check("loss_prototype" in losses and "loss_hierarchy" in losses, "辅助 loss 已并入总 loss", failures)
    total.backward()

    def gnorm(p: torch.Tensor) -> float | None:
        return None if p.grad is None else float(p.grad.norm())

    grads = {
        "fusion 标量 w2/w3/w4": model.neck.fusion_logits,
        "fusion 3×3 conv": model.neck.fusion_conv.conv.weight,
        "prototype 矩阵": head.prototypes,
        "prototype projection": head.proto_proj[0].weight,
        "hierarchy head": head.hier_fc.weight,
        "3000 类分类头": head.cls_branches[-1].weight,
        "DINO decoder": next(model.decoder.parameters()),
        "DINO encoder": next(model.encoder.parameters()),
        "ResNet-50 layer4": model.backbone.layer4[-1].conv3.weight,
        "ResNet-50 layer2": model.backbone.layer2[0].conv1.weight,
    }
    for name, p in grads.items():
        g = gnorm(p)
        check(g is not None and g > 0 and g == g, f"{name:<22} grad norm = {g}", failures)
    print(f"      fusion 标量各自的梯度 = {model.neck.fusion_logits.grad.detach().cpu().tolist()}")
    if device == "cuda":
        print(f"      峰值显存（batch={B}，{H}×{W}）= {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

    print("\n" + ("全部通过。" if not failures else f"{len(failures)} 项未通过：\n  - " + "\n  - ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
