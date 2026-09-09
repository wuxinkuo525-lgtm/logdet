"""清洗规则集。

每条规则独立计数并带命中上限，超限就阻塞 —— 目的是让"数据里有多少脏东西"
成为一个可对账的数字，而不是被静默修掉。

侦察实测（全量 194,265 框）：
    坐标反序 (R1)   0
    坐标越界 (R2)   0    —— xmax 从不超过 W，ymax 从不超过 H
    退化框   (R3)   2    —— 宽或高 <= 0，丢弃
    缺 size  (R4)   0
所以本阶段实际只有 R3 会命中 2 次。其余规则保留，因为它们是针对
上游数据更新的保险，且实现成本已经付了。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 规则代号 → 人类可读说明。写进报告用。
RULE_DESC = {
    "R1": "xmin>xmax 或 ymin>ymax → 交换",
    "R2": "坐标越界 → clip 到 [0,W]/[0,H]",
    "R3": "clip 后宽或高 <=0 → 丢弃该框",
    "R4": "缺 width/height 或为 0 → 用 PIL 读真实尺寸补",
    "R5": "同图同类完全重复框 → 去重保留一个",
    "R6": "类名 NFKC + 空白归一",
    "R7": "XML 解析报错（lxml recover 已兜底）",
    "R8": "<filename> 与磁盘名不一致 → 以磁盘为准",
    "R9": "XML 类名与所在品牌目录名不一致 → 以 XML 为准",
}


@dataclass
class CleanStats:
    """每条规则的命中计数。"""

    counts: dict[str, int] = field(default_factory=lambda: {k: 0 for k in RULE_DESC})
    samples: dict[str, list[str]] = field(default_factory=lambda: {k: [] for k in RULE_DESC})

    def hit(self, rule: str, sample: str | None = None, *, max_samples: int = 20) -> None:
        self.counts[rule] += 1
        if sample and len(self.samples[rule]) < max_samples:
            self.samples[rule].append(sample)

    def as_rows(self, denom_boxes: int, denom_images: int) -> list[tuple[str, ...]]:
        # R4/R7/R8 是图像级，其余是框级
        img_level = {"R4", "R7", "R8"}
        out = []
        for rule, desc in RULE_DESC.items():
            n = self.counts[rule]
            d = denom_images if rule in img_level else denom_boxes
            rate = n / d if d else 0.0
            out.append((rule, f"{n:,}", f"{rate:.4%}  {desc}"))
        return out


def clean_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    stats: CleanStats,
    *,
    where: str = "",
) -> tuple[float, float, float, float] | None:
    """清洗单个框。返回 None 表示该框应被丢弃。

    顺序固定：先纠反序（R1），再 clip 越界（R2），最后判退化（R3）。
    这个顺序不能调 —— 若先 clip 再纠反序，反序框会被 clip 成退化框，
    丢失"这是反序"的信息，报告里就归错因了。
    """
    if x1 > x2:
        x1, x2 = x2, x1
        stats.hit("R1", where)
    if y1 > y2:
        y1, y2 = y2, y1
        stats.hit("R1", where)

    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(width), x2), min(float(height), y2)
    if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
        stats.hit("R2", where)
    x1, y1, x2, y2 = cx1, cy1, cx2, cy2

    if x2 - x1 <= 0 or y2 - y1 <= 0:
        stats.hit("R3", where)
        return None

    return x1, y1, x2, y2
