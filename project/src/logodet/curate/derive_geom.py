"""几何派生与难例代理。

全部是**纯向量化计算，零图像解码**，对 194k 框只需秒级。

三个难例轴（相对计划有一处重要改动）：

  T1 截断  —— **用真实标注 truncated==1**，不是代理。
              侦察实测 20,270 框 (10.43%) 为 1。
              计划里的"框贴图像边界"几何测量降级为**校准诊断**：
              既然这里同时有几何测量和人工标注，就能直接算出
              「贴边」对「标注为截断」的 precision / recall。
              这是唯一能为 P2 / P3（无真值）的可信度提供旁证的机会。

  P2 互遮挡 —— IoF 包含比 ≥ 阈值。**不用 IoU**：大框被小框压住时
              IoU 很小但遮挡很严重，IoU 会漏掉这类。

  P3 形状离群 —— 类内长宽比的 MAD z-score ≥ 3。
              利用 logo 是刚性图案这一特性：同一类的长宽比分布应该很集中，
              离群者大概率是被遮挡/截断/透视形变，或标注有误。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# COCO 的面积分档，与论文 Fig.5D 的 small/medium/large 定义一致，
# 因此尺寸维度的结论可以直接与论文的分布事实对话。
AREA_SMALL = 32**2
AREA_LARGE = 96**2


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """派生框级几何字段。df 需含 x1,y1,x2,y2,img_w,img_h。"""
    out = df
    bw = out["x2"] - out["x1"]
    bh = out["y2"] - out["y1"]
    out["bw"] = bw.astype("float32")
    out["bh"] = bh.astype("float32")

    area = bw * bh
    out["area"] = area.astype("float32")
    out["sqrt_area"] = np.sqrt(area).astype("float32")
    out["rel_area"] = (area / (out["img_w"] * out["img_h"])).astype("float32")

    ar = bw / bh
    out["ar"] = ar.astype("float32")
    out["log_ar"] = np.log(ar).astype("float32")

    out["cx_rel"] = ((out["x1"] + out["x2"]) / 2 / out["img_w"]).astype("float32")
    out["cy_rel"] = ((out["y1"] + out["y2"]) / 2 / out["img_h"]).astype("float32")

    # 分档边界的闭合方向必须写死并与验证门里的直算保持一致：
    #     small  : area <  32²
    #     medium : 32² <= area <= 96²
    #     large  : area >  96²
    #
    # 不用 pd.cut：它的 right=True/False 只能统一控制所有边界，
    # 而这里需要「左开右闭 + 右开」的混合闭合，用 pd.cut 会在
    # area 恰好 == 1024 的框上产生 17 个偏差（G7 抓到过一次）。
    out["area_bin"] = np.select(
        [area < AREA_SMALL, area > AREA_LARGE],
        ["small", "large"],
        default="medium",
    )

    return out


def add_edge_distance(df: pd.DataFrame, *, min_px: float = 2.0, frac: float = 0.005) -> pd.DataFrame:
    """计算框到图像四条边的最小距离，以及贴边条数。

    这**不再是** T1 的判定依据（T1 用真实标注），而是用来校准
    「几何贴边」这个代理有多接近人工的截断判断。
    """
    d_left = df["x1"]
    d_top = df["y1"]
    d_right = df["img_w"] - df["x2"]
    d_bottom = df["img_h"] - df["y2"]

    df["edge_dist"] = np.minimum(
        np.minimum(d_left, d_top), np.minimum(d_right, d_bottom)
    ).astype("float32")

    thr = np.maximum(min_px, frac * np.minimum(df["img_w"], df["img_h"]))
    df["edge_thr"] = thr.astype("float32")
    df["edge_count"] = (
        (d_left <= thr).astype("int8")
        + (d_top <= thr).astype("int8")
        + (d_right <= thr).astype("int8")
        + (d_bottom <= thr).astype("int8")
    )
    df["touches_edge"] = df["edge_count"] >= 1
    return df


def add_iof_occlusion(df: pd.DataFrame, *, threshold: float = 0.3) -> pd.DataFrame:
    """P2 互遮挡：occ(a) = max_b area(a∩b) / area(a)。

    用 IoF（Intersection over Foreground，包含比）而不是 IoU：
    一个大框被小框压住时 IoU 很小（分母是并集）但遮挡很严重，
    IoU 判据会把这类漏掉。IoF 的分母是 a 自身面积，直接反映
    "a 有多大比例被别人压住"。

    只在 n_boxes >= 2 的图上算；单框图恒为 0。
    """
    df["p2_occ_iof"] = np.float32(0.0)
    df["p2_occ_partner"] = np.int64(-1)

    multi = df.groupby("image_id")["ann_id"].transform("size") >= 2
    sub = df[multi]
    if sub.empty:
        df["is_hard_p2"] = False
        return df

    occ = np.zeros(len(df), dtype="float32")
    partner = np.full(len(df), -1, dtype="int64")
    pos = {a: i for i, a in enumerate(df["ann_id"].to_numpy())}

    for _, g in sub.groupby("image_id", sort=False):
        x1 = g["x1"].to_numpy(dtype="float64")
        y1 = g["y1"].to_numpy(dtype="float64")
        x2 = g["x2"].to_numpy(dtype="float64")
        y2 = g["y2"].to_numpy(dtype="float64")
        area = (x2 - x1) * (y2 - y1)
        ids = g["ann_id"].to_numpy()

        # 两两相交面积
        ix1 = np.maximum(x1[:, None], x1[None, :])
        iy1 = np.maximum(y1[:, None], y1[None, :])
        ix2 = np.minimum(x2[:, None], x2[None, :])
        iy2 = np.minimum(y2[:, None], y2[None, :])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        np.fill_diagonal(inter, 0.0)

        ratio = inter / np.maximum(area[:, None], 1e-9)
        best = ratio.argmax(axis=1)
        best_val = ratio[np.arange(len(ids)), best]

        for k, aid in enumerate(ids):
            if best_val[k] > 0:
                j = pos[aid]
                occ[j] = best_val[k]
                partner[j] = ids[best[k]]

    df["p2_occ_iof"] = occ
    df["p2_occ_partner"] = partner
    df["is_hard_p2"] = df["p2_occ_iof"] >= threshold
    return df


def add_shape_outlier(
    df: pd.DataFrame,
    *,
    z_threshold: float = 3.0,
    min_class_size: int = 20,
    mad_floor_frac: float = 0.02,
) -> pd.DataFrame:
    """P3 类内长宽比离群：z = |ar - median_c| / (1.4826 * MAD_c)。

    1.4826 是把 MAD 换算成正态分布下标准差的一致性常数。

    两处保护：
      * 类内样本 < min_class_size → p3_valid=False，不参与该切片。
        样本太少时 MAD 极不稳定，会造出一堆假离群。
      * MAD == 0（刚性 logo 很常见，同类长宽比几乎恒定）→ 给一个
        与中位数成比例的地板，否则除零会让 z 爆成 inf。
    """
    g = df.groupby("class_id")["ar"]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median())
    n_c = g.transform("size")

    floor = mad_floor_frac * med.abs()
    mad_eff = np.maximum(mad, floor)

    z = (df["ar"] - med).abs() / (1.4826 * mad_eff + 1e-9)
    df["p3_ar_median"] = med.astype("float32")
    df["p3_ar_mad"] = mad.astype("float32")
    df["p3_ar_z"] = z.astype("float32")
    df["p3_valid"] = (n_c >= min_class_size) & (mad_eff > 0)
    df["is_hard_p3"] = df["p3_valid"] & (df["p3_ar_z"] >= z_threshold)
    return df


def add_truncation_label(df: pd.DataFrame) -> pd.DataFrame:
    """T1 截断：直接用真实标注，不是代理。"""
    df["is_hard_t1"] = df["truncated"].astype("int8") == 1
    return df


def summarize_slices(df: pd.DataFrame, *, hard_any_cols: list[str] | None = None) -> pd.DataFrame:
    """汇总各切片体量。

    Args:
        hard_any_cols: 参与 hard_any 并集的列。默认只有 T1 ——
            P2/P3 已因 precision 未达门槛而降级为探索性列
            （见 configs/slices.yaml 的 proxy_verdict）。
    """
    hard_any_cols = hard_any_cols or ["is_hard_t1"]
    # 三个难例代理列都统计，但只有 hard_any_cols 参与并集
    all_cols = [c for c in ("is_hard_t1", "is_hard_p2", "is_hard_p3") if c in df.columns]
    rows = []
    for c in all_cols:
        rows.append(
            {
                "slice": c.replace("is_hard_", "").upper(),
                "role": "主力" if c in hard_any_cols else "探索性",
                "n_ann": int(df[c].sum()),
                "n_img": int(df.loc[df[c], "image_id"].nunique()),
                "rate": float(df[c].mean()),
            }
        )

    # 尺寸轴（COCO 口径，无歧义，无需代理验证）
    if "area_bin" in df.columns:
        for b in ("small", "medium", "large"):
            m = df["area_bin"] == b
            rows.append(
                {
                    "slice": f"SIZE:{b}",
                    "role": "主力",
                    "n_ann": int(m.sum()),
                    "n_img": int(df.loc[m, "image_id"].nunique()),
                    "rate": float(m.mean()),
                }
            )

    hard_any = df[hard_any_cols].any(axis=1)
    rows.append(
        {
            "slice": "HARD_ANY",
            "role": "主力",
            "n_ann": int(hard_any.sum()),
            "n_img": int(df.loc[hard_any, "image_id"].nunique()),
            "rate": float(hard_any.mean()),
        }
    )
    rows.append(
        {
            "slice": "CLEAN",
            "role": "主力",
            "n_ann": int((~hard_any).sum()),
            "n_img": int(df.loc[~hard_any, "image_id"].nunique()),
            "rate": float((~hard_any).mean()),
        }
    )
    return pd.DataFrame(rows)


def overlap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """三个切片的两两重叠计数矩阵。"""
    cols = ["is_hard_t1", "is_hard_p2", "is_hard_p3"]
    names = [c.replace("is_hard_", "").upper() for c in cols]
    m = np.zeros((len(cols), len(cols)), dtype="int64")
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            m[i, j] = int((df[a] & df[b]).sum())
    return pd.DataFrame(m, index=names, columns=names)
