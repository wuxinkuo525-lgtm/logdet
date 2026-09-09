"""配置加载与配置指纹（config fingerprint）。

指纹的用途：每个产物（parquet / coco json / metrics）都要带上生成它的
配置的 sha256。这样当两次跑出来的数字不一样时，第一件事就能确认
"是不是配置变了"，而不用靠回忆。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(name: str | Path) -> dict[str, Any]:
    """读 configs/ 下的 yaml。传相对名（如 "eval.yaml"）或绝对路径都可以。"""
    p = Path(name)
    if not p.is_absolute():
        p = CONFIG_DIR / p
    if not p.is_file():
        raise FileNotFoundError(f"找不到配置文件：{p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def fingerprint(obj: Any) -> str:
    """对任意可 JSON 化的配置对象算稳定 sha256。

    sort_keys=True 保证字典顺序不影响指纹；default=str 让 Path
    之类的对象也能进指纹而不抛错。
    """
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """算文件 sha256。分块读，3GB 的 zip 也不会吃满内存。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
