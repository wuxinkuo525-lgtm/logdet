"""L0 自校验：用合成预测验证评测器本身是对的。

**这是整个 setup 里最重要的一道保险。** 评测器错了，后面所有 baseline 数字
都是废的，而且这种错误在指标上表现为"效果偏低"，极易被误判成模型问题。

思路：拿 GT 自己造预测，此时**正确答案是已知的**：

  1. 完美预测（ε=0）        → AP 必须 ≈ 1.0
  2. 抖动阶梯（ε 逐级增大）  → AP 必须严格单调递减
  3. 空预测                → AP 必须 = 0
  4. 只报一半的框           → AR@100 必须 ≈ 0.5
  5. 完美 + 大量低分假阳     → AP 下降但 AR 基本不变

第 1 条顺带就是**坐标格式的探针**：如果 GT 与 DT 的 xyxy↔xywh 转换有任何
不一致，完美预测就得不到 1.0。所以不需要单独写格式测试。

### 断言策略

只有**两条硬断言**：
  * AP(ε) 随 ε 严格单调递减
  * AP(0) > 0.999 且 AP(0.40) < 0.20

中间档（ε=0.05/0.10/0.20）**不预设精确数字** —— 它们取决于 IoU 阈值网格
与抖动模型的具体交互，事先猜一个值再去凑，等于把测试写成了自证。
首轮实测后把数值固化进 configs/eval.yaml 作为回归基线，
之后任何偏离都必须能解释。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticSpec:
    """一组合成预测的描述。"""

    name: str
    jitter: float = 0.0  # 相对框宽高的抖动幅度
    keep_ratio: float = 1.0  # 保留多少比例的 GT 框
    n_false_per_image: int = 0  # 每图额外加多少假阳
    false_score: float = 0.05  # 假阳的分数
    score: float = 1.0  # 真阳的分数


def jitter_boxes(
    boxes: np.ndarray,
    eps: float,
    rng: np.random.Generator,
    *,
    img_w: float,
    img_h: float,
) -> np.ndarray:
    """按框自身尺寸的比例抖动四个坐标。

    用**相对**幅度而不是绝对像素：绝对抖动对大框几乎无影响、对小框是毁灭性的，
    会让 AP 曲线的形状被框尺寸分布主导，而不是反映抖动本身。
    """
    if eps <= 0 or len(boxes) == 0:
        return boxes.copy()

    out = boxes.astype("float64").copy()
    w = out[:, 2] - out[:, 0]
    h = out[:, 3] - out[:, 1]
    # 四个坐标独立抖动，幅度 ~ U(-eps, eps) × 对应边长
    dx1 = rng.uniform(-eps, eps, len(out)) * w
    dy1 = rng.uniform(-eps, eps, len(out)) * h
    dx2 = rng.uniform(-eps, eps, len(out)) * w
    dy2 = rng.uniform(-eps, eps, len(out)) * h
    out[:, 0] += dx1
    out[:, 1] += dy1
    out[:, 2] += dx2
    out[:, 3] += dy2

    # 夹回图内并保证非退化
    out[:, 0] = np.clip(out[:, 0], 0, img_w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, img_h - 1)
    out[:, 2] = np.clip(out[:, 2], out[:, 0] + 1, img_w)
    out[:, 3] = np.clip(out[:, 3], out[:, 1] + 1, img_h)
    return out


def make_synthetic_dt(
    coco_gt_dict: dict,
    spec: SyntheticSpec,
    *,
    seed: int = 6129,
) -> list[dict]:
    """从 COCO GT 字典造一组合成预测。

    只用 iscrowd==0 的 GT 造真阳 —— iscrowd==1 的是被忽略项，
    拿它造预测会让"完美预测"得不到满分，属于自己给自己下绊子。
    """
    rng = np.random.default_rng(seed)
    sizes = {im["id"]: (float(im["width"]), float(im["height"]))
             for im in coco_gt_dict["images"]}

    by_img: dict[int, list[dict]] = {}
    for a in coco_gt_dict["annotations"]:
        if a.get("iscrowd", 0) == 1:
            continue
        by_img.setdefault(int(a["image_id"]), []).append(a)

    out: list[dict] = []
    for image_id, (iw, ih) in sizes.items():
        anns = by_img.get(image_id, [])
        if anns:
            xywh = np.array([a["bbox"] for a in anns], dtype="float64")
            xyxy = np.column_stack(
                [xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]]
            )
            cats = [int(a["category_id"]) for a in anns]

            keep = np.arange(len(xyxy))
            if spec.keep_ratio < 1.0:
                # 逐框独立按概率保留，**不设"每图至少留 1 个"的下限**。
                #
                # 早期写成 k = max(1, round(n * ratio))，结果单框图
                # （占 84.6%）永远保留全部框，实测 AR@100 = 0.8515
                # 而不是期望的 0.5 —— 这条门于是测不出任何东西。
                # 逐框采样让"丢一半"在全局意义上成立，代价是部分图
                # 会一个预测都没有，而这正是真实检测器的行为。
                mask = rng.random(len(xyxy)) < spec.keep_ratio
                keep = np.where(mask)[0]
            if len(keep) == 0:
                continue
            jit = jitter_boxes(xyxy[keep], spec.jitter, rng, img_w=iw, img_h=ih)
            for (x1, y1, x2, y2), ci in zip(jit, [cats[i] for i in keep]):
                out.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(ci),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(spec.score),
                    }
                )

        for _ in range(spec.n_false_per_image):
            w = rng.uniform(0.05, 0.3) * iw
            h = rng.uniform(0.05, 0.3) * ih
            x = rng.uniform(0, max(1.0, iw - w))
            y = rng.uniform(0, max(1.0, ih - h))
            out.append(
                {
                    "image_id": int(image_id),
                    "category_id": 1,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "score": float(spec.false_score),
                }
            )
    return out


# L0 套件定义。ε 阶梯用于单调性检验。
JITTER_LEVELS = (0.0, 0.05, 0.10, 0.20, 0.40)

L0_SUITE = (
    SyntheticSpec("perfect", jitter=0.0),
    SyntheticSpec("jitter_005", jitter=0.05),
    SyntheticSpec("jitter_010", jitter=0.10),
    SyntheticSpec("jitter_020", jitter=0.20),
    SyntheticSpec("jitter_040", jitter=0.40),
    SyntheticSpec("half_recall", jitter=0.0, keep_ratio=0.5),
    SyntheticSpec("perfect_plus_fp", jitter=0.0, n_false_per_image=10, false_score=0.05),
)
