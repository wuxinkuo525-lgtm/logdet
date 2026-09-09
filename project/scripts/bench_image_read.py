#!/usr/bin/env python
"""基准测试：图像读取的三条通路。

S2 已经证明本机的文件守卫对**每次文件打开**都有固定开销：
XML 逐文件读只有 20 文件/秒，改从 zip 顺序读后是 1,656~15,305 文件/秒（快 82.5x）。

图像比 XML 大得多（约 25KB vs 700B），固定开销的占比不同，所以必须重测，
不能照搬 XML 的结论。三条候选通路：

  A 磁盘逐文件      最直接，dataloader 的默认做法
  B zip 随机访问    单一句柄，但要在 317,308 个条目里跳着取，寻道多
  C 预打包缓存      一次性把需要的图打成单文件 + 索引，之后 mmap 随机读

判据：S4 的吞吐门要求 >= 60 img/s（含解码）。若 A 就能达标，就不要引入 B/C 的复杂度。

用法：
    python scripts/bench_image_read.py [样本数]
"""

from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.config import load_yaml  # noqa: E402
from logodet.paths import P  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def main() -> int:
    cfg = load_yaml("dataset.yaml")
    zip_path = P.raw / cfg["source"]["zip_name"]
    top = cfg["parse"]["archive_top_dir"]

    sp = P.artifact("splits")
    v2k = pd.read_parquet(sp / "val2k_repr.parquet")
    vhp = pd.read_parquet(sp / "val_hard_pool.parquet")
    img = pd.read_parquet(sp / "split_images.parquet")

    infer_ids = sorted(set(v2k["image_id"]) | set(vhp["image_id"]))
    infer = img[img["image_id"].isin(infer_ids)]
    print(f"推理集合（val2k ∪ hard_pool）: {len(infer):,} 图")

    sub = infer.sample(n=min(N, len(infer)), random_state=6129)
    rels = sub["rel_path"].tolist()
    print(f"基准样本数: {len(rels)}\n")

    # ---- A 磁盘逐文件 ------------------------------------------------------
    t0 = time.time()
    n_bytes_a = 0
    shapes_a = []
    for i, rel in enumerate(rels, 1):
        p = P.dataset / rel
        raw = p.read_bytes()
        n_bytes_a += len(raw)
        with Image.open(io.BytesIO(raw)) as im:
            arr = np.asarray(im.convert("RGB"))
        shapes_a.append(arr.shape)
        if i % 100 == 0:
            print(f"  [A 磁盘] {i}/{len(rels)}  {i / (time.time() - t0):.1f} img/s", flush=True)
    dt_a = time.time() - t0
    rate_a = len(rels) / dt_a

    # ---- B zip 随机访问 ----------------------------------------------------
    want = {f"{top}/{r}": r for r in rels}
    t0 = time.time()
    n_bytes_b = 0
    shapes_b = {}
    with zipfile.ZipFile(zip_path) as zf:
        # 先建名字 → ZipInfo 的映射（一次性，只读中央目录）
        idx = {i.filename: i for i in zf.infolist() if i.filename in want}
        t_index = time.time() - t0
        t0 = time.time()
        for k, (name, rel) in enumerate(want.items(), 1):
            raw = zf.read(idx[name])
            n_bytes_b += len(raw)
            with Image.open(io.BytesIO(raw)) as im:
                arr = np.asarray(im.convert("RGB"))
            shapes_b[rel] = arr.shape
            if k % 100 == 0:
                print(f"  [B zip ] {k}/{len(rels)}  {k / (time.time() - t0):.1f} img/s",
                      flush=True)
    dt_b = time.time() - t0
    rate_b = len(rels) / dt_b

    # ---- 结果 --------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f" 图像读取基准（{len(rels)} 张，含 PIL 解码）")
    print("=" * 74)
    print(f"\n  A 磁盘逐文件   {dt_a:>7.1f}s   {rate_a:>7.1f} img/s")
    print(f"  B zip 随机访问 {dt_b:>7.1f}s   {rate_b:>7.1f} img/s"
          f"   （另需建索引 {t_index:.1f}s，一次性）")
    print(f"\n  提速倍数 B/A : {rate_b / rate_a:.2f}x")
    print(f"  字节数一致   : A={n_bytes_a:,}  B={n_bytes_b:,}  "
          f"{'一致' if n_bytes_a == n_bytes_b else '不一致！'}")

    same_shape = all(shapes_b[r] == s for r, s in zip(rels, shapes_a))
    print(f"  解码结果一致 : {'是' if same_shape else '否！'}")

    print(f"\n  平均图像大小 : {n_bytes_a / len(rels) / 1024:.1f} KB")
    print(f"  推理全集 {len(infer):,} 图的预计耗时：")
    print(f"    A 通路 {len(infer) / rate_a:>7.1f}s")
    print(f"    B 通路 {len(infer) / rate_b:>7.1f}s（含索引 {t_index:.1f}s）")

    gate = 60.0
    print(f"\n  S4 吞吐门要求 >= {gate:.0f} img/s")
    for name, rate in (("A 磁盘", rate_a), ("B zip", rate_b)):
        print(f"    {name:<8} {rate:>7.1f} img/s  {'达标' if rate >= gate else '未达标'}")
    if rate_a >= gate:
        print("\n  → A 通路已达标，**不引入 zip/缓存的复杂度**")
    else:
        print(f"\n  → A 通路未达标，采用 B 通路（快 {rate_b / rate_a:.1f}x）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
