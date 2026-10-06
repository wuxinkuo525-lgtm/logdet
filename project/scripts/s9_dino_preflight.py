#!/usr/bin/env python
"""S9：开跑前自检。集群作业一启动先跑这个，几秒钟内把「跑到半夜才会暴露」的问题提前报出来。

查的东西：
    1. GPU 可用、mmcv 的编译算子能 import（DINO 的可变形注意力依赖它）
    2. 训练要读的文件都在：两份 COCO 标注、fine→super 映射、图清单、COCO 预训练权重
    3. 图清单（train + dev_eval）和 baseline 评测清单里的每一张图都在数据目录下
    4. runs/ 可写，剩余空间够放整套调参的存档

任何一项不过就以非 0 退出，并把缺什么写清楚。infer_union.json 缺失也会失败，
避免训练结束却无法导出完整预测。

用法（logodet_dino 环境）：
    python scripts/s9_dino_preflight.py [--shared-ids <清单>]
    python scripts/s9_dino_preflight.py --make-rehearsal-ids <正式清单> <输出>   # 彩排用的小清单
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("LOGDET_ROOT", PROJECT_DIR.parent.as_posix())

PRETRAINED = "dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth"
MIN_FREE_GB = 25  # 9 组 × 2 个滚动存档 + 4 个里程碑，每个约 0.63 GB，再留余量


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_rehearsal_ids(src: Path, out: Path, n_train: int = 20, n_dev: int = 16) -> None:
    ids = read_json(src)
    out.write_text(json.dumps({
        "train_ids": ids["train_ids"][:n_train],
        "dev_eval_ids": ids["dev_eval_ids"][:n_dev],
        "note": f"彩排用：{src.name} 的前 {n_train} / {n_dev} 张",
    }), encoding="utf-8")
    print(f"彩排清单 → {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared-ids", default=None)
    ap.add_argument("--make-rehearsal-ids", nargs=2, metavar=("SRC", "OUT"), default=None)
    args = ap.parse_args()

    if args.make_rehearsal_ids:
        make_rehearsal_ids(Path(args.make_rehearsal_ids[0]), Path(args.make_rehearsal_ids[1]))
        return 0

    root = Path(os.environ["LOGDET_ROOT"])
    data_root = Path(os.environ.get("LOGDET_DATA_ROOT", root / "data" / "LogoDet-3K"))
    runs_root = Path(os.environ.get("LOGDET_RUNS_ROOT", root / "runs"))
    ids_path = Path(args.shared_ids) if args.shared_ids else runs_root / "handoff" / "shared_image_ids.json"
    failures: list[str] = []
    warnings: list[str] = []

    # ---- 1. GPU 与编译算子 ----
    try:
        import torch
        from mmcv.ops import MultiScaleDeformableAttention  # noqa: F401

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU     {props.name}，{props.total_memory / 2**30:.1f} GB；torch {torch.__version__}")
            # 真在 GPU 上算一下：驱动与 CUDA 运行库不匹配时 is_available() 也可能是 True
            torch.zeros(8, device="cuda").sum().item()
        else:
            failures.append("torch.cuda.is_available() 为 False：这个节点上用不了 GPU")
    except Exception as exc:  # noqa: BLE001  自检要把任何原因都报出来
        failures.append(f"环境不可用：{type(exc).__name__}: {exc}")

    # ---- 2. 必需文件 ----
    required = {
        "训练标注": runs_root / "coco" / "instances_train_dino.json",
        "val 标注": runs_root / "coco" / "instances_val_dino.json",
        "fine→super 映射": runs_root / "coco" / "fine_to_super_mapping.json",
        "图清单": ids_path,
        "COCO 预训练权重": runs_root / "cache" / "checkpoints" / PRETRAINED,
    }
    for label, path in required.items():
        if path.is_file():
            print(f"文件    {label}：{path}（{path.stat().st_size / 2**20:.1f} MB）")
        else:
            failures.append(f"缺 {label}：{path}")
    union_path = runs_root / "eval" / "gt" / "infer_union.json"
    if not union_path.is_file():
        failures.append(f"没有 {union_path}：不能生成完整预测，停止扫描")

    # ---- 3. 图片都在 ----
    train_ann, val_ann = required["训练标注"], required["val 标注"]
    if train_ann.is_file() and val_ann.is_file() and ids_path.is_file():
        ids = read_json(ids_path)
        wanted = set(ids["train_ids"]) | set(ids["dev_eval_ids"])
        names = {im["id"]: im["file_name"] for im in read_json(train_ann)["images"] if im["id"] in wanted}
        unknown = wanted - set(names)
        if unknown:
            failures.append(f"图清单里有 {len(unknown)} 个 id 不在训练标注里，例如 {sorted(unknown)[:3]}")
        if union_path.is_file():
            names.update({im["id"]: im["file_name"] for im in read_json(union_path)["images"]})
        missing = [name for name in names.values() if not (data_root / name).is_file()]
        if missing:
            failures.append(f"数据目录 {data_root} 下缺 {len(missing):,} / {len(names):,} 张图，例如 {missing[:3]}")
        else:
            print(f"图片    {len(names):,} 张全部在 {data_root}")

    # ---- 4. 磁盘 ----
    try:
        (runs_root / "dino").mkdir(parents=True, exist_ok=True)
        probe = runs_root / "dino" / ".preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        free_gb = shutil.disk_usage(runs_root).free / 2**30
        print(f"磁盘    {runs_root} 可写，剩余 {free_gb:.0f} GB")
        if free_gb < MIN_FREE_GB:
            failures.append(f"{runs_root} 只剩 {free_gb:.0f} GB，整套调参至少要 {MIN_FREE_GB} GB")
    except OSError as exc:
        failures.append(f"{runs_root} 不可写：{exc}")

    for w in warnings:
        print(f"WARN    {w}")
    if failures:
        for f in failures:
            print(f"FAIL    {f}")
        print(f"自检未通过（{len(failures)} 项），不开跑。", file=sys.stderr)
        return 1
    print("自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
