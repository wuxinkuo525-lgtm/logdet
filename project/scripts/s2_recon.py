#!/usr/bin/env python
"""S2 前置侦察：在写解析器之前，先用真实数据回答五个设计问题。

不做任何假设。计划阶段曾根据"论文未记录字段"判断标注里没有
truncated / difficult，但实际 XML 是完整的 VOC 格式，这两个字段都在。
它们究竟有没有信号，只能实测 —— 若 truncated 有信号，P1 截断代理
就该直接用标注而不是自己算几何。

五个问题：
  Q1 truncated / difficult / pose 是否被真正填过？
  Q2 坐标是 1-based 还是 0-based？（最容易造成全局 AP 静默偏低的坑）
  Q3 脏数据规模：坐标反序、越界、退化框、缺 size
  Q4 类名与所在品牌目录名是否一致？
  Q5 XML 的 size 与 PIL 真实解码尺寸是否一致（EXIF 旋转探针）？

数据源走 zip 而非磁盘 —— 实测快 82.5 倍，见 ingest/xml_source.py 的说明。

用法：
    python scripts/s2_recon.py              # 全量，约 2 分钟
    python scripts/s2_recon.py --sample 8000
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.config import load_yaml  # noqa: E402
from logodet.ingest.xml_source import iter_xml_from_zip  # noqa: E402
from logodet.paths import P  # noqa: E402

RE_TRUNC = re.compile(rb"<truncated>(.*?)</truncated>")
RE_DIFF = re.compile(rb"<difficult>(.*?)</difficult>")
RE_POSE = re.compile(rb"<pose>(.*?)</pose>")
RE_VERIF = re.compile(rb'<annotation\s+verified="(.*?)"')
RE_NAME = re.compile(rb"<name>(.*?)</name>")
RE_SIZE = re.compile(rb"<width>(\d+)</width>\s*<height>(\d+)</height>", re.S)
RE_BOX = re.compile(
    rb"<xmin>(-?\d+)</xmin>\s*<ymin>(-?\d+)</ymin>"
    rb"\s*<xmax>(-?\d+)</xmax>\s*<ymax>(-?\d+)</ymax>",
    re.S,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="抽样数；0 表示全量")
    ap.add_argument("--decode-check", type=int, default=200, help="PIL 交叉验证的图像数")
    args = ap.parse_args()

    cfg = load_yaml("dataset.yaml")
    zip_path = P.raw / cfg["source"]["zip_name"]
    top = cfg["parse"]["archive_top_dir"]

    inv = pd.read_parquet(P.artifact("tables") / "file_inventory.parquet")
    brand_of = dict(zip(inv["xml_rel"], inv["brand_dir"]))

    only = None
    if args.sample:
        only = set(inv.sample(n=args.sample, random_state=6129)["xml_rel"])

    trunc, diff, pose, verif = (collections.Counter() for _ in range(4))
    n_xml = n_obj = 0
    min_x = min_y = 10**9
    cnt_x0 = cnt_y0 = cnt_x1 = cnt_y1 = 0
    over_w = over_h = eq_w = eq_h = 0
    swapped = degenerate = size_missing = 0
    name_mismatch = 0
    mismatch_samples: list[tuple[str, str, str]] = []

    print(f"从 zip 读取 XML（{zip_path.name}）...")
    t0 = time.time()
    for e in iter_xml_from_zip(zip_path, top_dir=top, only=only):
        b = e.data
        n_xml += 1
        trunc.update(RE_TRUNC.findall(b))
        diff.update(RE_DIFF.findall(b))
        pose.update(RE_POSE.findall(b))
        verif.update(RE_VERIF.findall(b))

        m = RE_SIZE.search(b)
        if not m:
            size_missing += 1
            W = H = 0
        else:
            W, H = int(m.group(1)), int(m.group(2))

        brand = brand_of.get(e.rel_path)
        if brand is not None:
            for nm in RE_NAME.findall(b):
                s = nm.decode("utf-8", "replace").strip()
                if s != brand:
                    name_mismatch += 1
                    if len(mismatch_samples) < 15:
                        mismatch_samples.append((e.rel_path, brand, s))

        for a, bb, c, d in RE_BOX.findall(b):
            x1, y1, x2, y2 = int(a), int(bb), int(c), int(d)
            n_obj += 1
            min_x, min_y = min(min_x, x1), min(min_y, y1)
            cnt_x0 += x1 == 0
            cnt_y0 += y1 == 0
            cnt_x1 += x1 == 1
            cnt_y1 += y1 == 1
            if W and x2 > W:
                over_w += 1
            if H and y2 > H:
                over_h += 1
            if W and x2 == W:
                eq_w += 1
            if H and y2 == H:
                eq_h += 1
            if x1 > x2 or y1 > y2:
                swapped += 1
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                degenerate += 1

    dt = time.time() - t0
    print(f"  完成：{n_xml:,} 个 XML / {n_obj:,} 个框，{dt:.1f}s "
          f"（{n_xml / max(dt, 1e-9):.0f} 文件/秒）\n")

    def dec(ctr: collections.Counter) -> dict[str, int]:
        return {k.decode("utf-8", "replace"): v for k, v in ctr.items()}

    print("=" * 72)
    print(f" S2 侦察结果（{n_xml:,} 个 XML / {n_obj:,} 个框）")
    print("=" * 72)

    print("\n--- Q1 VOC 属性字段是否有信号 ---")
    for label, ctr in (("truncated", trunc), ("difficult", diff),
                       ("pose", pose), ("verified", verif)):
        d = dec(ctr)
        total = sum(d.values()) or 1
        top5 = ", ".join(f"{k!r}={v:,}({v / total:.2%})"
                         for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:5])
        verdict = "有信号" if len(d) > 1 else "常量，无信号"
        print(f"  {label:<10} 取值数={len(d):<3} {verdict:<12} {top5}")

    print("\n--- Q2 坐标基准 ---")
    print(f"  min(xmin)={min_x}  min(ymin)={min_y}")
    print(f"  xmin==0 : {cnt_x0:,}      ymin==0 : {cnt_y0:,}")
    print(f"  xmin==1 : {cnt_x1:,}      ymin==1 : {cnt_y1:,}")
    print(f"  xmax >W : {over_w:,}      ymax >H : {over_h:,}")
    print(f"  xmax==W : {eq_w:,}        ymax==H : {eq_h:,}")
    basis = ("1-based" if (min_x >= 1 and min_y >= 1 and cnt_x0 == 0 and cnt_y0 == 0)
             else "0-based")
    print(f"  → 判定：{basis}")

    print("\n--- Q3 脏数据 ---")
    print(f"  xmin>xmax 或 ymin>ymax : {swapped:,}")
    print(f"  宽或高 <=0             : {degenerate:,}")
    print(f"  缺 <size> 节点          : {size_missing:,}")

    print("\n--- Q4 类名 vs 品牌目录名 ---")
    print(f"  不一致的框数 : {name_mismatch:,} / {n_obj:,} ({name_mismatch / max(n_obj, 1):.3%})")
    for rel, brand, nm in mismatch_samples[:10]:
        print(f"    {rel}  目录={brand!r}  XML={nm!r}")

    print("\n--- Q5 XML size vs PIL 真实尺寸（EXIF 旋转探针）---")
    from PIL import Image

    sub = inv.sample(n=min(args.decode_check, len(inv)), random_state=42)
    want = set(sub["xml_rel"])
    xml_sizes: dict[str, tuple[int, int]] = {}
    for e in iter_xml_from_zip(zip_path, top_dir=top, only=want, verbose=False):
        m = RE_SIZE.search(e.data)
        if m:
            xml_sizes[e.rel_path] = (int(m.group(1)), int(m.group(2)))

    mism = checked = 0
    for rel_img, rel_xml in zip(sub["image_rel"], sub["xml_rel"]):
        if rel_xml not in xml_sizes:
            continue
        xw, xh = xml_sizes[rel_xml]
        with Image.open(P.dataset / rel_img) as im:
            pw, ph = im.size
        checked += 1
        if (xw, xh) != (pw, ph):
            mism += 1
            if mism <= 5:
                print(f"    不一致 {rel_img}: XML=({xw},{xh}) PIL=({pw},{ph})")
    print(f"  校验 {checked} 张，不一致 {mism} 张")

    out = P.artifact("tables") / "s2_recon.json"
    out.write_text(
        json.dumps(
            {
                "scope": "full" if not args.sample else f"sample={args.sample}",
                "n_xml": n_xml,
                "n_obj": n_obj,
                "elapsed_sec": round(dt, 1),
                "attr_truncated": dec(trunc),
                "attr_difficult": dec(diff),
                "attr_pose": dec(pose),
                "attr_verified": dec(verif),
                "coord": {
                    "min_xmin": min_x, "min_ymin": min_y,
                    "xmin_eq_0": cnt_x0, "ymin_eq_0": cnt_y0,
                    "xmin_eq_1": cnt_x1, "ymin_eq_1": cnt_y1,
                    "xmax_gt_W": over_w, "ymax_gt_H": over_h,
                    "xmax_eq_W": eq_w, "ymax_eq_H": eq_h,
                    "verdict": basis,
                },
                "dirty": {
                    "swapped": swapped,
                    "degenerate": degenerate,
                    "size_missing": size_missing,
                },
                "name_mismatch_boxes": name_mismatch,
                "name_mismatch_samples": mismatch_samples[:15],
                "pil_checked": checked,
                "pil_mismatch": mism,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
