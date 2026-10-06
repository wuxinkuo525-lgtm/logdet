"""安全解压：先预检，再按需选择快慢路径。

系统 `unzip` / `ditto` 快得多，但对三类问题不设防：

  1. **文件名编码**  zip 的 general purpose bit 11 (0x800) 置位才表示文件名是
     UTF-8。未置位时按规范应为 CP437，但中文环境的打包工具常直接塞 GBK 字节，
     解出来就是乱码。
  2. **macOS 大小写不敏感冲突**  APFS 默认 case-insensitive，zip 里若同时有
     `Adidas/` 和 `adidas/`，后者会**静默覆盖**前者 —— 无声的数据损坏。
  3. **zip slip 路径穿越**  条目名含 `../` 时会写到目标目录之外。

早期实现的做法是「一律走 Python 逐文件解压」来规避这三点。代价极高：
LogoDet-3K 有 317k 个小文件，每次写盘都要过一遍系统的文件守卫，实测
**25 分钟只解出 381MB / 3GB**，还看不到头。

现在的做法：**先做一次预检**（只读 zip 中央目录，约 1 秒），逐条确认三类风险
是否真实存在。全部不存在 → 用原生 `ditto` 快速解压；任一存在 → 退回 Python
逐文件路径并对具体条目做修正。

对 LogoDet-3K 的实测预检结果：317,308 条目全部未置 UTF-8 标志，但**没有一条
含非 ASCII 字符**（纯 ASCII 用任何编码解都一样），大小写冲突 0，路径穿越 0
—— 所以走快路径，且完全无损失。
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PreflightReport:
    """预检结果。三个 risky_* 全为空才可以走快路径。"""

    total_entries: int = 0
    utf8_flagged: int = 0
    risky_names: list[str] = field(default_factory=list)
    case_collisions: dict[str, list[str]] = field(default_factory=dict)
    path_traversal: list[str] = field(default_factory=list)
    top_dirs: list[str] = field(default_factory=list)

    @property
    def can_use_fast_path(self) -> bool:
        return not (self.risky_names or self.case_collisions or self.path_traversal)

    def as_rows(self) -> list[tuple[str, ...]]:
        return [
            ("zip 条目总数", f"{self.total_entries:,}", ""),
            ("UTF-8 标志已置位", f"{self.utf8_flagged:,}", ""),
            ("未置位且含非 ASCII", f"{len(self.risky_names):,}", "仅这些有乱码风险"),
            ("大小写冲突组", f"{len(self.case_collisions):,}", "APFS 会静默覆盖"),
            ("路径穿越条目", f"{len(self.path_traversal):,}", "zip slip"),
            ("顶层目录", ", ".join(self.top_dirs[:3]), ""),
        ]


@dataclass
class UnzipReport:
    path_used: str = ""  # "ditto" | "python"
    total_entries: int = 0
    extracted_files: int = 0
    recovered_names: list[tuple[str, str]] = field(default_factory=list)
    preflight: PreflightReport | None = None

    def as_rows(self) -> list[tuple[str, str]]:
        return [
            ("解压路径", self.path_used),
            ("zip 条目总数", f"{self.total_entries:,}"),
            ("解出文件数", f"{self.extracted_files:,}"),
            ("编码还原的文件名", f"{len(self.recovered_names):,}"),
        ]


class CaseCollisionError(RuntimeError):
    """zip 内存在仅大小写不同的路径，在 APFS 上会互相覆盖。"""


class ZipSlipError(RuntimeError):
    """zip 条目试图写到目标目录之外。"""


def decode_entry_name(info: zipfile.ZipInfo) -> tuple[str, bool]:
    """还原条目名。返回 (名字, 是否做了编码还原)。

    仅在 UTF-8 标志位未置位时尝试还原 —— 置位的情况 Python 已经解对了，
    再去 encode('cp437') 反而可能抛异常或造成二次损坏。
    """
    name = info.filename
    if info.flag_bits & 0x800:
        return name, False
    if not is_risky_name(name, info.flag_bits):
        # 纯 ASCII 用 cp437 / gbk / utf-8 解码结果完全一致，无需还原
        return name, False

    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name, False

    for enc in ("utf-8", "gbk"):
        try:
            fixed = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        return (fixed, True) if fixed != name else (name, False)
    return name, False


def is_risky_name(filename: str, flag_bits: int) -> bool:
    """判断条目名是否存在乱码风险。

    抽成纯函数是为了能直接测 —— 用 zipfile 造不出"非 ASCII 且未置 UTF-8 标志"
    的样本，因为 Python 的 writestr 遇到非 ASCII 名会强制置位。

    风险成立的**充要条件**是两条同时满足：
      1. UTF-8 标志位未置位（置位则 Python 已经解对了）
      2. 名字含非 ASCII 字符（纯 ASCII 用 cp437/gbk/utf-8 解码结果完全一致）

    只看条件 1 会误判：LogoDet-3K 的 317,308 条目全部未置位，但全是纯 ASCII，
    实际零风险。误判的代价是白白退回慢路径（实测慢两个数量级）。
    """
    return not (flag_bits & 0x800) and not filename.isascii()


def _case_keys(name: str) -> list[str]:
    """产出该条目需要参与大小写冲突比对的所有路径键。

    既要比完整路径（两个文件同名不同大小写 → 互相覆盖），
    也要比每一级目录前缀（两个目录同名不同大小写 → 在 APFS 上合并）。

    目录合并对本项目尤其致命：品牌目录就是类别，`Adidas/` 与 `adidas/`
    合并会让 3000 类变成 2999，而两边的文件名若再撞上就直接丢数据。
    """
    p = name.rstrip("/")
    if not p:
        return []
    parts = p.split("/")
    return ["/".join(parts[: i + 1]) for i in range(len(parts))]


def preflight(zip_path: Path, dest: Path) -> PreflightReport:
    """只读中央目录，判断三类风险是否真实存在。约 1 秒。"""
    rep = PreflightReport()
    dest = dest.resolve()
    lower_map: dict[str, set[str]] = defaultdict(set)

    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        rep.total_entries = len(infos)
        tops: set[str] = set()

        for info in infos:
            if info.flag_bits & 0x800:
                rep.utf8_flagged += 1
            elif is_risky_name(info.filename, info.flag_bits):
                rep.risky_names.append(info.filename)

            name = info.filename
            if name.startswith("/") or ".." in name.split("/"):
                rep.path_traversal.append(name)
                continue

            target = (dest / name).resolve()
            if not str(target).startswith(str(dest)):
                rep.path_traversal.append(name)
                continue

            for key in _case_keys(name):
                lower_map[key.lower()].add(key)
            p = name.rstrip("/")
            if p:
                tops.add(p.split("/")[0])

        rep.top_dirs = sorted(tops)

    rep.case_collisions = {k: sorted(v) for k, v in lower_map.items() if len(v) > 1}
    return rep


def _extract_fast_native(zip_path: Path, dest: Path) -> str:
    """尽量用系统原生工具批量解压，比逐文件 Python 循环快一到两个数量级。

    优先级：
      1. ``ditto``   —— macOS 原生，本项目最初就是给它写的
      2. ``zipfile.extractall`` —— 标准库兜底，跨平台行为一致

    刻意不用系统 ``tar``：tar 格式与 zip 是两种不同的容器格式，GNU tar
    （Linux、以及 Windows 上 Git 自带的那份）读不了 zip，只有 macOS/BSD
    的 bsdtar（基于 libarchive）可以 —— 但同名命令在 PATH 上到底解析到
    哪一个因机器而异，`shutil.which("tar")` 赌不出来，一旦赌错就是直接
    运行时报错。zipfile 是标准库，行为在任何平台上都确定。

    返回实际用上的工具名，写进 ``UnzipReport.path_used`` 便于事后核对。
    """
    ditto = shutil.which("ditto")
    if ditto:
        proc = subprocess.run(
            [ditto, "-x", "-k", "--sequesterRsrc", str(zip_path), str(dest)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ditto 退出码 {proc.returncode}: {proc.stderr[:500]}")
        return "ditto"

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return "zipfile"


def _extract_with_python(
    zip_path: Path, dest: Path, rep: UnzipReport, *, progress_every: int
) -> None:
    """逐文件解压并修正条目名。只在预检发现风险时才走这条路。"""
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        for i, info in enumerate(infos, 1):
            name, recovered = decode_entry_name(info)
            if recovered:
                rep.recovered_names.append((info.filename, name))

            target = dest / name
            if name.endswith("/") or info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)
            rep.extracted_files += 1

            if progress_every and i % progress_every == 0:
                print(f"    ... {i:,}/{len(infos):,}")


def safe_extract(
    zip_path: Path,
    dest: Path,
    *,
    progress_every: int = 25_000,
    verbose: bool = True,
    force_slow_path: bool = False,
) -> UnzipReport:
    """预检 → 选路径 → 解压。

    Args:
        force_slow_path: 强制走 Python 逐文件路径（调试用）
    """
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    pf = preflight(zip_path, dest)
    rep = UnzipReport(total_entries=pf.total_entries, preflight=pf)

    if verbose:
        print("      预检：")
        for k, v, note in pf.as_rows():
            print(f"        {k:<20} {v:<12} {note}")

    if pf.path_traversal:
        raise ZipSlipError(
            f"{len(pf.path_traversal)} 个条目试图写到目标目录之外，样例："
            f"{pf.path_traversal[:3]}"
        )
    if pf.case_collisions:
        sample = list(pf.case_collisions.items())[:5]
        raise CaseCollisionError(
            f"发现 {len(pf.case_collisions)} 组仅大小写不同的路径。\n"
            f"APFS 默认大小写不敏感，直接解压会让它们互相静默覆盖。\n"
            f"样例：{sample}\n"
            f"处置：改到大小写敏感的卷上解压，或为冲突条目手工重命名。"
        )

    if pf.can_use_fast_path and not force_slow_path:
        if verbose:
            print("      预检三项均无风险 → 走原生快路径")
        rep.path_used = _extract_fast_native(zip_path, dest)
        if verbose:
            print(f"        实际使用   {rep.path_used}")
        # 原生工具不报文件数，落盘后现数
        top = dest / pf.top_dirs[0] if pf.top_dirs else dest
        rep.extracted_files = sum(1 for p in top.rglob("*") if p.is_file())
    else:
        reason = "强制指定" if force_slow_path else f"{len(pf.risky_names)} 个条目名需要编码还原"
        if verbose:
            print(f"      走 Python 逐文件路径（{reason}），317k 文件预计较慢")
        rep.path_used = "python"
        _extract_with_python(zip_path, dest, rep, progress_every=progress_every)

    return rep


def verify_zip(zip_path: Path) -> str | None:
    """CRC 校验。返回第一个损坏条目名，全部完好则返回 None。"""
    with zipfile.ZipFile(zip_path) as zf:
        return zf.testzip()
