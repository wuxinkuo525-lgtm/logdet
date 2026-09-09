#!/usr/bin/env python
"""基准测试：XML 从磁盘逐文件读 vs 从 zip 顺序读。

背景：本机的文件守卫对**每次文件打开**都有固定开销。S1 的解压已经证明
逐文件写盘慢两个数量级；这里验证读取是否同样受影响，以决定 S2 的解析
数据源到底用哪个。

zip 里 158,654 个 XML 加起来才约 110MB，而 zip 只需要打开一次。
"""

from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.config import load_yaml  # noqa: E402
from logodet.paths import P  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


def main() -> int:
    inv = pd.read_parquet(P.artifact("tables") / "file_inventory.parquet")
    sub = inv.sample(n=N, random_state=6129)
    rels = sub["xml_rel"].tolist()
    print(f"样本数：{N:,}\n")

    # ---- 方式 A：磁盘逐文件 ----
    t0 = time.time()
    total_a = 0
    for i, rel in enumerate(rels, 1):
        total_a += len((P.dataset / rel).read_bytes())
        if i % 1000 == 0:
            print(f"  [磁盘] {i:,}/{N:,}  {i / (time.time() - t0):.0f} 文件/秒", flush=True)
    dt_a = time.time() - t0
    print(f"\nA 磁盘逐文件 : {dt_a:.1f}s  → {N / dt_a:.0f} 文件/秒  "
          f"（全量 158,654 需 {158654 / (N / dt_a) / 60:.1f} 分钟）")

    # ---- 方式 B：zip 顺序读 ----
    cfg = load_yaml("dataset.yaml")
    zip_path = P.raw / cfg["source"]["zip_name"]
    top = cfg["parse"]["archive_top_dir"]
    want = {f"{top}/{r}" for r in rels}

    t0 = time.time()
    total_b = 0
    hit = 0
    with zipfile.ZipFile(zip_path) as zf:
        # 按 zip 内物理顺序遍历，磁盘寻道最少
        for info in zf.infolist():
            if info.filename in want:
                total_b += len(zf.read(info))
                hit += 1
                if hit % 1000 == 0:
                    print(f"  [zip] {hit:,}/{N:,}  {hit / (time.time() - t0):.0f} 文件/秒",
                          flush=True)
    dt_b = time.time() - t0
    print(f"\nB zip 顺序读 : {dt_b:.1f}s  → {hit / dt_b:.0f} 文件/秒  "
          f"（全量 158,654 需 {158654 / (hit / dt_b) / 60:.1f} 分钟）")

    print(f"\n字节数一致性 : A={total_a:,}  B={total_b:,}  "
          f"{'一致' if total_a == total_b else '不一致！'}")
    print(f"提速倍数     : {dt_a / dt_b:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
