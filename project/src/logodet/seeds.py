"""随机流管理。

为什么不用一个全局 seed：如果划分、抽样、加噪都从同一个
np.random.default_rng(6129) 取数，那么**改动任何一处的取数次数**
都会让后面所有随机结果跟着变 —— 明明只改了 L0 的噪声代码，
数据划分却变了，这种耦合极难排查。

做法：每个用途派生一条独立的流。purpose 字符串参与哈希，
所以 seed_for("split") 与 seed_for("val2k") 互不干扰，
新增一个用途也不会扰动已有的任何一条。
"""

from __future__ import annotations

import hashlib

import numpy as np

# 项目基准 seed（课程代号）。改它会让所有随机结果整体变化，非必要不动。
BASE_SEED = 6129


def derive_seed(purpose: str, base: int = BASE_SEED) -> int:
    """把 (base, purpose) 映射成一个 64 位整数 seed。"""
    blob = f"{base}:{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def seed_for(purpose: str, base: int = BASE_SEED) -> np.random.Generator:
    """取某个用途专属的随机数发生器。

    >>> rng = seed_for("split_agnostic")
    >>> rng.integers(0, 100, 3)
    array([...])
    """
    return np.random.default_rng(np.random.PCG64(derive_seed(purpose, base)))
