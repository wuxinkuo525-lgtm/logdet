"""collate 与 DataLoader 工厂。

三个关键决定：

**1. 绝不 pad 框。**
检测任务每张图的框数不同。pad 到统一长度会引入假框，需要额外的 mask
一路传播到损失/评测里，是个持续的出错来源。torchvision 的检测模型原生
接受 `List[Tensor]` + `List[Dict]`，所以直接返回列表最省事也最不易错。
代价是不能对 batch 做向量化运算 —— setup 阶段只做推理，无所谓。

**2. 默认 num_workers=0 —— 有实测支撑，不是偷懒。**

S4 吞吐门实测（scripts/s4_smoke_loader.py，600 图）：

    num_workers=0   572.6 img/s
    num_workers=4    23.6 img/s   慢 24x
    num_workers=6    17.5 img/s   慢 33x
    num_workers=8    14.0 img/s   慢 41x

**worker 越多越慢**，这排除了"启动开销一次性摊销"的解释。
根因诊断（scripts/s4_diagnose_workers.py）：

    单进程链路拆解        仅 zip 读取 3,217 img/s
                        + PIL 解码  1,517 img/s
                        + 转张量      813 img/s   ← 已远超 60 门槛

    多进程的额外开销      每 worker 反序列化 dataset      0.6 MB / 0.16s
                        每 worker 扫一遍 zip 中央目录         1.25s
                        每样本张量经管道回传             2.12 MB

关键在最后一项：**JPEG 解码后数据膨胀 111 倍**（19.6 KB → 2.12 MB float32），
600 张就是 1.24 GB 的 IPC 流量，而单进程是零拷贝。
多进程只在「单样本计算量 ≫ IPC 量」时划算，本场景恰好相反。

多 worker 的支持代码保留 —— 将来若换成大图输入或加重的预处理（增强、
多尺度），平衡点会移动，届时把 num_workers 调回来即可。

**3. MPS 下的 worker 约束（num_workers>0 时才生效，每条都有具体理由）。**

| 约束 | 理由 |
|---|---|
| worker 只产 CPU tensor | MPS 张量不能跨进程传递；collate 出口有断言兜底 |
| 保持 spawn 而非 fork | macOS 上 fork 后触碰 ObjC runtime 会随机崩溃 |
| persistent_workers=True | spawn 启动慢，不复用的话每个 epoch 都要重付一次 |
| pin_memory=False | MPS 不支持 pinned host memory，开了只会刷警告且无收益 |
| worker 内线程数设 1 | 否则 N 个 worker × 多线程解码会互相抢核，反而更慢 |
| 不在 __init__ 持有 zip 句柄 | spawn 要 pickle dataset；句柄不可 pickle（见 ZipImageReader.__getstate__）|
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .core_dataset import CoreDataset
from .schema import Sample


def detection_collate(batch: Sequence[tuple[torch.Tensor, dict]]) -> tuple[list, list]:
    """变长框的 collate。返回 (List[Tensor], List[Dict])，不做任何 pad。"""
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]

    # 出口断言：worker 里绝不能产出 MPS 张量。
    # 违反时错误会在主进程的某个不相关位置爆出来，极难定位，
    # 所以在这里当场拦住。
    for im in images:
        if im.device.type != "cpu":
            raise RuntimeError(
                f"collate 收到 {im.device.type} 张量。worker 只能产 CPU 张量，"
                f".to('mps') 必须在主进程的推理循环里做。"
            )
    return images, targets


class AdaptedDataset(Dataset):
    """把 CoreDataset 套上 adapter，成为 torch 认识的 Dataset。

    薄薄一层：唯一职责是调 adapter。核心逻辑全在 CoreDataset 里，
    而后者不 import torch。
    """

    def __init__(self, core: CoreDataset, adapt: Callable[[Sample], Any]) -> None:
        self.core = core
        self.adapt = adapt

    def __len__(self) -> int:
        return len(self.core)

    def __getitem__(self, i: int):
        s = self.core[i]
        s.validate()
        return self.adapt(s)


def _worker_init(worker_id: int) -> None:
    """限制 worker 内的线程数，避免 N 个 worker 互相抢核。"""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        import cv2

        cv2.setNumThreads(0)
    except ImportError:
        pass


def build_loader(
    core: CoreDataset,
    adapt: Callable[[Sample], Any],
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False,
    seed: int | None = None,
) -> DataLoader:
    """构造 DataLoader。

    默认 `num_workers=0`：本场景实测比多进程快 24-41 倍，根因见模块说明。
    默认 `shuffle=False`：推理要可复现的遍历顺序。
    """
    ds = AdaptedDataset(core, adapt)

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": detection_collate,
        # MPS 不支持 pinned host memory
        "pin_memory": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
        kwargs["worker_init_fn"] = _worker_init
        # macOS 上必须 spawn：fork 后触碰 ObjC runtime 会随机崩溃
        kwargs["multiprocessing_context"] = torch.multiprocessing.get_context("spawn")
    if shuffle and seed is not None:
        kwargs["generator"] = torch.Generator().manual_seed(seed)

    return DataLoader(ds, **kwargs)
