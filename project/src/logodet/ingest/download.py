"""S1 数据集下载。

用 curl 而不是 kaggle CLI：后者是依赖锁里唯一需要源码构建的包，
构建时的 mkdir 会被本机文件守卫拦截，并且会拖垮整个 pip 事务
（详见 project/README.md 的踩坑记录 #3）。

curl 还有两个实际优势：
    -C -      断点续传，3.3GB 断了不用重来
    --retry   网络抖动自动重试
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import file_sha256, load_yaml
from ..paths import P


class CredentialError(RuntimeError):
    """Kaggle 凭据缺失或格式不对。"""


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    size_bytes: int
    sha256: str
    skipped: bool  # True 表示复用了已有文件，没有真的下载


def read_kaggle_credentials() -> tuple[str, str]:
    """从 ~/.kaggle/kaggle.json 读凭据；环境变量优先（CI 场景）。"""
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return env_user, env_key

    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.is_file():
        raise CredentialError(
            f"找不到 {cred}。\n"
            "获取方式：Kaggle → Account → API → Create New Token，"
            "把下载的 kaggle.json 放到 ~/.kaggle/ 并 chmod 600。"
        )

    # 权限不对只警告不拦路 —— 这是安全建议，不是功能前提
    mode = cred.stat().st_mode & 0o777
    if mode & 0o077:
        print(f"  [warn] {cred} 权限为 {mode:o}，建议 chmod 600", file=sys.stderr)

    try:
        data = json.loads(cred.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CredentialError(f"{cred} 不是合法 JSON：{e}") from e

    user, key = data.get("username"), data.get("key")
    if not user or not key:
        raise CredentialError(f"{cred} 缺少 username 或 key 字段")
    return user, key


def download_dataset(*, force: bool = False, quiet: bool = False) -> DownloadResult:
    """下载 zip 到 raw/。已存在且尺寸合理时直接复用。

    **不解压**：先留 zip + 算 sha256，之后任何一次重解压都能对回同一份源。
    """
    cfg = load_yaml("dataset.yaml")
    src = cfg["source"]
    zip_path = P.raw / src["zip_name"]
    approx = int(src.get("approx_zip_bytes", 0))

    if zip_path.exists() and not force:
        size = zip_path.stat().st_size
        # 允许比标称小 10%（镜像重新打包会有出入），但不能是明显的半截文件
        if approx and size >= approx * 0.9:
            if not quiet:
                print(f"  已存在 {zip_path.name}（{size / 1e9:.2f} GB），跳过下载")
            return DownloadResult(zip_path, size, file_sha256(zip_path), skipped=True)
        if not quiet:
            print(f"  已存在但偏小（{size / 1e9:.2f} GB），断点续传 ...")

    user, key = read_kaggle_credentials()
    P.raw.mkdir(parents=True, exist_ok=True)

    cmd = [
        "curl",
        "-L",
        "-C", "-",              # 断点续传
        "--retry", "5",
        "--retry-delay", "3",
        "--fail-with-body",     # HTTP 4xx/5xx 时非零退出，而不是把错误页写进 zip
        "-u", f"{user}:{key}",
        "-o", str(zip_path),
        src["kaggle_api"],
    ]
    if quiet:
        cmd.insert(1, "-s")

    if not quiet:
        print(f"  curl → {zip_path}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"curl 退出码 {proc.returncode}。"
            "若为 22（HTTP 错误），检查凭据是否有效，或该数据集是否需要先在网页上接受条款。"
        )

    if not zip_path.exists():
        raise RuntimeError(f"curl 声称成功但 {zip_path} 不存在")

    size = zip_path.stat().st_size
    if not quiet:
        print(f"  计算 sha256（{size / 1e9:.2f} GB，约需 10-20 秒）...")
    digest = file_sha256(zip_path)

    meta = {
        "zip_name": src["zip_name"],
        "source_url": src["kaggle_api"],
        "size_bytes": size,
        "sha256": digest,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (P.raw / "download_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return DownloadResult(zip_path, size, digest, skipped=False)
