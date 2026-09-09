#!/usr/bin/env python
"""把核验裁剪图拼成带标签的联系表（contact sheet）。

为什么要拼版：60 张单独看要 60 次读图，效率极低。拼成每页 6 张、
每格带 ann_id / 切片 / 代理值标注的联系表，10 页即可看完，
且每格仍有约 480px，判断遮挡与形状离群的精度不受影响。

用法：
    python scripts/s2_make_contact_sheets.py
产出：
    runs/slices/sheets/sheet_01.jpg ... sheet_10.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.paths import P  # noqa: E402

COLS, ROWS = 3, 2
CELL = 480
LABEL_H = 46
PAD = 8
BG = (250, 250, 248)


def _font(size: int):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def main() -> int:
    slices_dir = P.artifact("slices")
    sheet_dir = slices_dir / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(slices_dir / "review_sheet.csv")
    # 按切片 + 代理值排序，同类相邻便于横向比较
    df = df.sort_values(["slice", "proxy_value"]).reset_index(drop=True)

    f_big = _font(24)
    f_small = _font(18)

    per_sheet = COLS * ROWS
    n_sheets = (len(df) + per_sheet - 1) // per_sheet
    sheet_w = COLS * (CELL + PAD) + PAD
    sheet_h = ROWS * (CELL + LABEL_H + PAD) + PAD + 34

    print(f"共 {len(df)} 张 → {n_sheets} 页（每页 {per_sheet} 张，每格 {CELL}px）")

    for s in range(n_sheets):
        chunk = df.iloc[s * per_sheet : (s + 1) * per_sheet]
        sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
        d = ImageDraw.Draw(sheet)
        d.text((PAD, 8), f"sheet {s + 1:02d} / {n_sheets}", fill=(60, 60, 60), font=f_big)

        for k, r in enumerate(chunk.itertuples()):
            cx = PAD + (k % COLS) * (CELL + PAD)
            cy = 34 + PAD + (k // COLS) * (CELL + LABEL_H + PAD)

            crop_path = P.artifacts / r.crop_path
            try:
                with Image.open(crop_path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((CELL, CELL))
                    ox = cx + (CELL - im.width) // 2
                    oy = cy + (CELL - im.height) // 2
                    sheet.paste(im, (ox, oy))
                    d.rectangle([ox - 1, oy - 1, ox + im.width, oy + im.height],
                                outline=(200, 200, 195), width=1)
            except Exception as e:
                d.text((cx + 10, cy + 10), f"读图失败\n{e}", fill=(200, 0, 0), font=f_small)

            ly = cy + CELL + 2
            d.text((cx + 2, ly),
                   f"#{r.ann_id}  {r.slice}  {r.stratum}  值={r.proxy_value}",
                   fill=(20, 20, 20), font=f_big)
            d.text((cx + 2, ly + 24),
                   f"{r.class_name[:26]}  {int(r.bw)}x{int(r.bh)}  ar={r.ar:.2f}  "
                   f"{r.area_bin}  trunc={r.truncated}",
                   fill=(90, 90, 90), font=f_small)

        out = sheet_dir / f"sheet_{s + 1:02d}.jpg"
        sheet.save(out, quality=90)
        ids = ", ".join(str(x) for x in chunk["ann_id"])
        print(f"  {out.name}  ann_id: {ids}")

    print(f"\n拼版图在 {sheet_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
