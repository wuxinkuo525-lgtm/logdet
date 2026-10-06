#!/usr/bin/env python
"""S9：从几组已跑完的 run 里按最后一次评测的某个指标挑出最好的一组。

tc2_shared_sweep.sbatch.sh 在每个阶段结束时调用本脚本，
用选出的结果决定下一阶段怎么跑（哪个 lr、采不采用某个变体、最终哪组最好）。

选择结果写进 --out（JSON，含每组的指标，留痕用）。--out 已存在时直接沿用里面的结论、
不重新挑——后面的阶段被 6 小时墙钟打断后续跑，前面的结论不能中途变。

标准输出只有一行：最好那组的名字（给 shell 接）。其余信息走标准错误。

用法：
    python scripts/s9_dino_pick_best.py --runs shared_lr1e-4 shared_lr2e-4 --out runs/dino/sweep_lr.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
os.environ.setdefault("LOGDET_ROOT", PROJECT_DIR.parent.as_posix())
from logodet.dino_run import validate_final_eval, write_json_atomic, scalar_records

SHOWN = ("agn/AP", "agn/AP50", "agn/AR@300", "coco/bbox_mAP")


def last_records(run_dir: Path) -> tuple[dict, dict]:
    """(最后一次评测记录, 最后一条训练记录)。续训会产生多个时间戳目录，按时间顺序读。"""
    last_eval, last_train = {}, {}
    for log in sorted(run_dir.glob("*/vis_data/scalars.json")):
        for rec in scalar_records(log):
            if "loss" in rec:
                last_train = rec
            elif any(k.startswith(("coco/", "agn/")) for k in rec):
                last_eval = rec
    return last_eval, last_train


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="runs/dino/ 下的目录名")
    ap.add_argument("--metric", default="agn/AP", help="越大越好的评测指标")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    runs_root = Path(os.environ.get("LOGDET_RUNS_ROOT", os.environ["LOGDET_ROOT"] + "/runs"))
    table = {}
    for name in args.runs:
        run_dir = runs_root / "dino" / name
        try:
            receipt = validate_final_eval(run_dir, args.metric)
        except (OSError, ValueError, KeyError) as exc:
            print(f"FAIL: {name} 最终评测未经验证：{exc}。请用原训练命令 --resume 补齐评测。", file=sys.stderr)
            return 1
        ev = receipt["metrics"]
        _, tr = last_records(run_dir)
        table[name] = {**{k: ev[k] for k in (*SHOWN, args.metric) if k in ev},
                       "eval_step": receipt["eval_step"], "checkpoint": receipt["checkpoint"],
                       "loss_cls": tr.get("loss_cls"), "loss": tr.get("loss")}
        print(f"{name}: " + "  ".join(f"{k}={v}" for k, v in table[name].items()), file=sys.stderr)

    best = max(table, key=lambda n: table[n][args.metric])
    if out.is_file():
        previous = json.loads(out.read_text(encoding="utf-8"))
        if previous.get("metric") != args.metric or previous.get("runs") != table or previous.get("best") != best:
            print(f"FAIL: {out} 的旧选择与已验证结果不一致。请改用新的 --out，并检查依赖旧选择的实验。", file=sys.stderr)
            return 1
        print(best)
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, {"metric": args.metric, "best": best, "runs": table})
    print(f"按 {args.metric} 选中 {best}，已写入 {out}", file=sys.stderr)
    print(best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
