#!/usr/bin/env python
"""把 VLM 预标结果写回 review_sheet.csv 并计算代理 precision。

标注口径（对应 review_sheet 里的 question 列）：
    P2 → 该框是否**真的被另一个物体遮挡**
    P3 → 该框的形状离群是否**真由遮挡/截断/透视形变/标注缺陷导致**
    1 = 是   0 = 不是   ? = 不确定（不计入 precision 分母，但要报数量）

reason 里的分类标签用于归因统计：
    nesting      同一品牌标识的嵌套/相邻标注框（图形标 + 文字标分开标）
    z_order      几何重叠成立但该框是遮挡者而非被遮挡者
    bimodal      类内徽标/字标双模态导致的伪离群
    natural      自然字标形状差异
    too_small    目标过小无法判断
    unreadable   缩略图分辨率不足
    real_occl    真实实例间遮挡
    real_trunc   真实截断
    real_persp   真实透视形变
    ann_defect   标注缺陷（框过大跨多实例等）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.paths import P  # noqa: E402

# ann_id -> (label, category, 说明)
LABELS: dict[int, tuple[str, str, str]] = {
    # ---------------- P2 互遮挡（25 张）----------------
    82904: ("1", "real_occl", "口香糖包装堆叠，后排那盒 logo 被前排压住"),
    161042: ("0", "nesting", "周大生完整可见，重叠来自中文/英文分框"),
    57340: ("1", "real_occl", "驼鹿图案压在 BiGG MiXX 字母上，前景遮挡"),
    183240: ("0", "nesting", "雪佛龙徽标完整，重叠来自徽标框/整体框嵌套"),
    73597: ("?", "unreadable", "35x117 窄条，传单上多处标注重叠辨不清"),
    67607: ("?", "too_small", "39x29 过小"),
    44591: ("0", "nesting", "英文/中文分框，国美电完整"),
    73640: ("0", "nesting", "文字框与圆形徽标框相邻重叠，徽标完整"),
    73755: ("0", "nesting", "同招牌两框重叠，logo 完整（模糊属另一维度）"),
    88865: ("1", "real_occl", "两盒保鲜膜堆叠，下盒挡住上盒底部，trunc=1 一致"),
    73576: ("?", "unreadable", "缩略图里辨不出红框位置"),
    158733: ("0", "nesting", "百世汇通整体框与 BEST EXPRESS 文字框重叠"),
    21725: ("0", "nesting", "JL 图形标框与文字标框重叠，平铺 T 恤完整"),
    178044: ("0", "nesting", "Ω 标框嵌在整体 lockup 框内"),
    73736: ("0", "z_order", "横幅文字压在圆形徽标上，该框是遮挡者而非被遮挡者"),
    57038: ("0", "nesting", "Nestlé 标框与 Big Turk 文字框相邻，两者都完整"),
    128433: ("0", "nesting", "感叹号单独框，嵌在整体 logo 框内"),
    106646: ("?", "too_small", "9x5 px 共 45 像素，本身像标注质量问题"),
    13184: ("0", "nesting", "重复/嵌套标注，文字可见"),
    128493: ("0", "nesting", "感叹号单独框嵌在整体框内"),
    17517: ("0", "nesting", "中间 K 徽标框嵌在字标框内"),
    46551: ("0", "nesting", "中文框+英文框+合并框三重嵌套"),
    17109: ("0", "nesting", "黑桃符号框嵌在整体 lockup 内"),
    145030: ("0", "nesting", "IoF=0.97，文字子框几乎完全被品牌色块框包含"),
    43123: ("0", "nesting", "IoF=0.98，中英文框三重嵌套"),
    # ---------------- P3 形状离群（35 张）----------------
    160396: ("0", "natural", "ar=11.07 但字标本身极宽，完整清晰"),
    192871: ("0", "natural", "横幅字标完整"),
    176136: ("0", "natural", "SBS 文字完整无遮挡（类名与图内 logo 不符，R9 案例）"),
    77097: ("1", "real_persp", "椭圆招牌旋转 90 度，文字竖排导致 ar 反转"),
    105571: ("0", "natural", "餐车招牌字标完整"),
    145200: ("0", "natural", "细长文字条完整"),
    188588: ("0", "bimodal", "字标完整；类内混有徽标图与字标图，双模态伪离群"),
    51032: ("0", "natural", "字标完整（模糊属另一维度）"),
    188146: ("0", "bimodal", "同 188588，lexus 类内徽标/字标混杂"),
    59000: ("0", "natural", "瓶身轻微透视，标签完整"),
    182072: ("0", "bimodal", "圆形徽章完整；类内双模态所致"),
    64488: ("0", "natural", "两行排布致 ar=1.44，招牌完整"),
    162087: ("0", "natural", "字标完整"),
    88247: ("0", "natural", "包装文字完整（R9 案例）"),
    58248: ("0", "bimodal", "罐身小字完整；蓝瓶徽标未标框"),
    194070: ("0", "bimodal", "YUTONG 字标完整可读；弧形徽标未标框"),
    183319: ("0", "bimodal", "COLNAGO 字标完整；三叶徽标未标框"),
    119687: ("0", "nesting", "字标框嵌在椭圆整标内，完整"),
    26157: ("0", "bimodal", "字标完整；NB 花押徽标未标框"),
    193852: ("0", "bimodal", "字标完整"),
    107954: ("1", "ann_defect", "标注框过大横跨多个产品实例，形状异常确有实因"),
    83288: ("?", "unreadable", "127x13 细条，辨不清框住哪行文字"),
    54408: ("0", "bimodal", "Pure Baking Soda 完整；圆形徽标未标框"),
    26935: ("0", "bimodal", "字标完整；同图另有吉祥物+圆徽"),
    102405: ("0", "natural", "花体细长字标完整"),
    76964: ("1", "real_persp", "冰淇淋桶圆柱面，logo 环绕包裹，真透视形变"),
    59789: ("0", "bimodal", "字标完整；大 B + 冰淇淋徽标未标框"),
    53835: ("0", "natural", "字标完整"),
    84663: ("0", "bimodal", "字标完整；鱼形徽标未标框"),
    24811: ("0", "natural", "卫衣上字标完整"),
    67634: ("1", "real_trunc", "顶部 Flake 被画面上边缘裁掉只剩细缝，trunc=1 一致"),
    117790: ("1", "real_persp", "框住盒盖顶面字标，强透视压缩"),
    189591: ("0", "bimodal", "字标完整；M 徽标与 EYEWEAR 未标框"),
    24325: ("0", "natural", "字标完整"),
    28787: ("1", "real_trunc", "字标左右都贴画面边缘被截断，trunc=1 一致"),
}


def main() -> int:
    sheet_path = P.artifact("slices") / "review_sheet.csv"
    df = pd.read_csv(sheet_path)

    missing = set(df["ann_id"]) - set(LABELS)
    if missing:
        print(f"FAIL: {len(missing)} 个 ann_id 未标注：{sorted(missing)[:10]}")
        return 1

    df["vlm_label"] = df["ann_id"].map(lambda a: LABELS[a][0])
    df["vlm_category"] = df["ann_id"].map(lambda a: LABELS[a][1])
    df["vlm_reason"] = df["ann_id"].map(lambda a: LABELS[a][2])
    # human_label 留空待复核；final_label 先取 vlm_label
    df["final_label"] = df["vlm_label"]
    df.to_csv(sheet_path, index=False)

    print("=" * 76)
    print(" VLM 预标结果与代理 precision")
    print("=" * 76)

    for sl in ("P2", "P3"):
        sub = df[df["slice"] == sl]
        n1 = int((sub["vlm_label"] == "1").sum())
        n0 = int((sub["vlm_label"] == "0").sum())
        nq = int((sub["vlm_label"] == "?").sum())
        prec = n1 / (n1 + n0) if (n1 + n0) else float("nan")
        print(f"\n{sl}  共 {len(sub)} 张")
        print(f"  1 = 确实难例 : {n1}")
        print(f"  0 = 不是     : {n0}")
        print(f"  ? = 不确定   : {nq}（不计入分母）")
        print(f"  precision    = {n1}/({n1}+{n0}) = {prec:.3f}   门槛 0.60  "
              f"→ {'通过' if prec >= 0.6 else '未通过'}")
        print("  归因分布：")
        for cat, n in sub["vlm_category"].value_counts().items():
            print(f"    {cat:<12} {n:>3}")

    print("\n" + "=" * 76)
    print(" 待你复核的清单（vlm_label ∈ {0, ?}）")
    print("=" * 76)
    need = df[df["vlm_label"].isin(["0", "?"])]
    print(f"\n共 {len(need)} 张。但其中 {int((need['vlm_category'] == 'nesting').sum())} 张归因为")
    print("「嵌套标注」、", int((need["vlm_category"] == "bimodal").sum()),
          "张归因为「类内双模态」—— 这两类是**系统性**判断，")
    print("不是逐张的主观判断，抽查少量即可确认。")
    print("\n建议优先复核这些（判断最不确定或影响最大）：")
    prio = need[need["vlm_category"].isin(["unreadable", "too_small", "z_order", "ann_defect"])]
    for r in prio.itertuples():
        print(f"  #{r.ann_id:<7} {r.slice}  {r.vlm_label}  {r.vlm_category:<11} {r.vlm_reason}")
    print(f"\n共 {len(prio)} 张需重点复核，其余可抽查。")
    print(f"\n清单已更新：{sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
