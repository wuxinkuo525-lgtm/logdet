"""难例切片评测器。

### 机制：靠 iscrowd=1 忽略非切片成员

评测切片 T1（截断框）时，把**所有非 T1 的 GT 置 iscrowd=1**，保留全图 GT。
pycocotools 对 iscrowd=1 的 GT：

  * 不计入 recall 分母 —— 于是指标只反映 T1 框的召回情况
  * 匹配到它的 det 被"吸收"，既不计 TP 也不计 FP

第二条很关键：检测器正确找到了一个非切片的 logo，不该因此被罚。
S5 的 K8 门已实测验证该机制（启用忽略 AP=1.0000 vs 不启用 0.5050）。

这个设计的好处是**预测只需跑一次**：3,079 张图前向一遍得到一份
predictions，所有切片都在同一份预测上做 GT 侧的视图变换。

### 与 COCO 原生 areaRng 的口径差异（重要）

尺寸切片有两种算法，**结果不同且都正确**，必须说清用的是哪个：

| | COCO 原生 `AP_small` | 本模块的 SIZE_small 切片视图 |
| --- | --- | --- |
| GT 侧 | 只保留 small GT | 非 small GT 置 iscrowd（保留但忽略）|
| DT 侧 | **只保留 small 范围内的 det** | 全部 det 参与 |
| 未匹配的大框 det | 不参与计算 | 计为 FP |

所以切片视图对假阳更严格。原生口径回答"在小目标这个尺度上模型表现如何"，
切片视图回答"在完整的检测输出下，小目标被召回得如何"。
后者更贴近"难例上表现怎样"这个问题，所以作为主口径；
两者都报，差异在报告里显式对账（S6-L6）。

### 可靠性

框数低于 min_ann_for_reliable（默认 200）的切片标 UNRELIABLE，
必须带 bootstrap CI，且禁止对该切片下强结论。
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from pycocotools.coco import COCO

from .coco_eval import DEFAULT_MAX_DETS, bootstrap_ci, evaluate


@dataclass(frozen=True)
class SliceSpec:
    """一个切片的定义。

    predicate 接收合并了图级列的 annotations 表，返回**框级**布尔掩码。
    统一到框级是因为 iscrowd 是框级属性；图级条件（如密度）
    通过 merge 图级列后在框上求值即可。
    """

    name: str
    axis: str
    predicate: Callable[[pd.DataFrame], pd.Series]
    note: str = ""


def default_slices() -> list[SliceSpec]:
    """S2/S3 定下的三轴 + 剔除 T1 的残差切片。

    **不含 P2/P3** —— 它们经 60 张核验 precision 仅 0.125/0.171，
    已降级为探索性列。

    ### 为什么需要 `*_noT1` 残差切片

    切片之间**不独立**。实测重叠（val_hard_pool，见 s6_diagnose_crosstalk.py）：

        T1 中有 87.3% 属于 SIZE_large     SIZE_large 中有 37.5% 是 T1
        T1 中有 71.1% 属于 DENSITY_1      DENSITY_1 中有 33.9% 是 T1

    原因很直接：被画面边缘裁掉的目标通常尺寸大，所以截断框绝大多数是大目标。

    后果是**一条会导致错误结论的解读陷阱**：模型若在截断框上差，
    SIZE_large 会跟着显得差，此时若直接写"模型对大目标不行"就是错的。

    残差切片把这两件事拆开：报 SIZE_large 时同时报 SIZE_large_noT1，
    若后者回到 CLEAN 水平，就证明前者的下降全部来自 T1 污染。
    """
    return [
        SliceSpec("T1_truncated", "truncation", lambda a: a["is_hard_t1"],
                  "真实标注 truncated==1"),
        SliceSpec("SIZE_small", "size", lambda a: a["area_bin"] == "small",
                  "area < 32²"),
        SliceSpec("SIZE_medium", "size", lambda a: a["area_bin"] == "medium",
                  "32² ≤ area ≤ 96²"),
        SliceSpec("SIZE_large", "size", lambda a: a["area_bin"] == "large",
                  "area > 96²"),
        SliceSpec("DENSITY_1", "density", lambda a: a["n_boxes"] == 1, "单框图"),
        SliceSpec("DENSITY_2", "density", lambda a: a["n_boxes"] == 2, ""),
        SliceSpec("DENSITY_3_4", "density", lambda a: a["n_boxes"].between(3, 4), ""),
        SliceSpec("DENSITY_5plus", "density", lambda a: a["n_boxes"] >= 5, "密集场景"),
        # 残差切片：剔除 T1 后的同一维度，用于拆解交叉污染
        SliceSpec("SIZE_large_noT1", "residual",
                  lambda a: (a["area_bin"] == "large") & ~a["is_hard_t1"],
                  "SIZE_large ∖ T1 —— T1 与它重叠 37.5%，必须一并报"),
        SliceSpec("DENSITY_1_noT1", "residual",
                  lambda a: (a["n_boxes"] == 1) & ~a["is_hard_t1"],
                  "DENSITY_1 ∖ T1 —— T1 与它重叠 33.9%"),
        # 受控对比：只在大目标内部比较截断与否。
        #
        # S7 首轮出现 T1(0.0510) > CLEAN(0.0085) 的反直觉结果，查下来是
        # **尺寸混淆**：T1 有 87.3% 是大目标，而大目标本身就是这个
        # COCO 预训练检测器唯一能找到的东西（SIZE_large 0.0376 vs
        # SIZE_small 0.0003）。直接比 T1 vs CLEAN 等于在比"大 vs 小"。
        #
        # T1_large 与 SIZE_large_noT1 同为大目标、只差截断与否，
        # 这才是能回答"截断到底难不难"的对照。
        SliceSpec("T1_large", "controlled",
                  lambda a: a["is_hard_t1"] & (a["area_bin"] == "large"),
                  "T1 ∩ SIZE_large —— 与 SIZE_large_noT1 构成受控对比"),
        SliceSpec(
            "CLEAN", "control",
            lambda a: ~(a["is_hard_t1"] | (a["area_bin"] == "small") | (a["n_boxes"] >= 5)),
            "三轴都不命中 —— 难例切片的对照组",
        ),
    ]


@dataclass
class SliceResult:
    name: str
    axis: str
    n_ann: int = 0
    n_img: int = 0
    reliable: bool = True
    metrics: dict[str, float] = field(default_factory=dict)
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    note: str = ""


def slice_members(
    annotations: pd.DataFrame,
    images: pd.DataFrame,
    specs: Sequence[SliceSpec],
) -> dict[str, set[int]]:
    """算出每个切片包含哪些 ann_id。

    先把图级列合并到框级 —— 密度这类条件是图的属性，但 iscrowd 是框的属性。
    """
    img_cols = [c for c in ("n_boxes", "supercat") if c in images.columns]
    a = annotations.merge(
        images[["image_id", *img_cols]], on="image_id", how="left",
        suffixes=("", "_img"),
    )
    out: dict[str, set[int]] = {}
    for s in specs:
        mask = s.predicate(a).fillna(False).to_numpy(dtype=bool)
        out[s.name] = set(a.loc[mask, "ann_id"].astype("int64").tolist())
    return out


def make_slice_gt(coco_gt_dict: dict, member_ann_ids: set[int]) -> dict:
    """生成切片视图：非成员的 GT 置 iscrowd=1。

    不删除非成员 —— 删了会让匹配到它们的 det 变成假阳，
    等于因为"正确检出了一个不属于本切片的 logo"而罚模型。
    """
    return {
        **coco_gt_dict,
        "annotations": [
            {**a, "iscrowd": 0 if int(a["id"]) in member_ann_ids else 1}
            for a in coco_gt_dict["annotations"]
        ],
    }


def _as_coco(d: dict) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO()
        c.dataset = d
        c.createIndex()
    return c


def evaluate_slices(
    coco_gt_dict: dict,
    detections: list[dict],
    members: dict[str, set[int]],
    specs: Sequence[SliceSpec],
    *,
    max_dets: Sequence[int] = DEFAULT_MAX_DETS,
    min_ann_for_reliable: int = 200,
    ci_metrics: Sequence[str] = ("AP50",),
    n_boot: int = 100,
    ci_only_when_unreliable: bool = True,
) -> list[SliceResult]:
    """逐切片评测。

    Args:
        ci_only_when_unreliable: 只对不可靠切片算 bootstrap CI。
            CI 很贵（每次要重跑 n_boot 遍评测），而样本充足的切片
            点估计本身就稳。
    """
    by_name = {s.name: s for s in specs}
    ann_img = {int(a["id"]): int(a["image_id"]) for a in coco_gt_dict["annotations"]}
    results: list[SliceResult] = []

    for name, ids in members.items():
        spec = by_name.get(name)
        if spec is None:
            continue
        n_ann = len(ids)
        n_img = len({ann_img[i] for i in ids if i in ann_img})
        reliable = n_ann >= min_ann_for_reliable

        r = SliceResult(
            name=name, axis=spec.axis, n_ann=n_ann, n_img=n_img,
            reliable=reliable, note=spec.note,
        )
        if n_ann == 0:
            r.metrics = {"AP": float("nan"), "AP50": float("nan")}
            r.note = (r.note + "；切片为空").strip("；")
            results.append(r)
            continue

        sl_dict = make_slice_gt(coco_gt_dict, ids)
        sl_gt = _as_coco(sl_dict)
        # 只在含成员框的图上评测 —— 其余图对该切片没有信息，
        # 纳入只会拖慢且让 AR 的分母混入无关图
        eval_imgs = sorted({ann_img[i] for i in ids if i in ann_img})
        res = evaluate(sl_gt, detections, image_ids=eval_imgs, max_dets=max_dets)
        r.metrics = res.metrics

        if (not reliable) or (not ci_only_when_unreliable):
            for m in ci_metrics:
                pt, lo, hi = bootstrap_ci(
                    sl_gt, detections, metric=m, n_boot=n_boot,
                    image_ids=eval_imgs, max_dets=max_dets,
                )
                r.ci[m] = (lo, hi)
        results.append(r)

    return results


def slices_to_frame(results: Sequence[SliceResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row: dict[str, Any] = {
            "slice": r.name,
            "axis": r.axis,
            "n_ann": r.n_ann,
            "n_img": r.n_img,
            "reliable": r.reliable,
        }
        for k in ("AP", "AP50", "AP75", "AR@100", "AR@300"):
            row[k] = r.metrics.get(k, float("nan"))
        for m, (lo, hi) in r.ci.items():
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        row["note"] = r.note
        rows.append(row)
    return pd.DataFrame(rows)


def check_partition(
    members: dict[str, set[int]],
    all_ann_ids: set[int],
    axis_slices: Sequence[str],
) -> tuple[bool, dict[str, int]]:
    """检查同一轴上的切片是否互斥且完备。

    尺寸三档、密度四档都必须构成对全体框的划分 —— 若不成立，
    说明分档逻辑有洞（某些框既不属于任何档，或同属两档），
    那么"各档指标"就无法拼回总体。
    """
    sets = [members[n] for n in axis_slices if n in members]
    union = set().union(*sets) if sets else set()
    pairwise_overlap = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            pairwise_overlap += len(sets[i] & sets[j])
    stats = {
        "union": len(union),
        "total": len(all_ann_ids),
        "missing": len(all_ann_ids - union),
        "extra": len(union - all_ann_ids),
        "pairwise_overlap": pairwise_overlap,
    }
    ok = stats["missing"] == 0 and stats["extra"] == 0 and pairwise_overlap == 0
    return ok, stats
