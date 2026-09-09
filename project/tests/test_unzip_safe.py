"""解压预检的回归测试。

预检现在决定走 ditto 快路径还是 Python 慢路径。判断错的后果是静默数据损坏
（大小写冲突被覆盖、文件名乱码），所以三类风险的识别必须有测试守着。

用 zipfile 现场造小 zip，不依赖真实数据集。
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logodet.ingest.unzip_safe import (  # noqa: E402
    CaseCollisionError,
    ZipSlipError,
    decode_entry_name,
    is_risky_name,
    preflight,
    safe_extract,
)


def _make_zip(path: Path, entries: dict[str, bytes], *, utf8_flag: bool = True) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            info = zipfile.ZipInfo(name)
            if utf8_flag:
                info.flag_bits |= 0x800
            else:
                info.flag_bits &= ~0x800
            zf.writestr(info, data)
    return path


# ---------------------------------------------------------------------------
# 预检：三类风险的识别
# ---------------------------------------------------------------------------


def test_clean_zip_uses_fast_path(tmp_path):
    z = _make_zip(tmp_path / "a.zip", {"root/a.jpg": b"x", "root/a.xml": b"y"})
    pf = preflight(z, tmp_path / "out")
    assert pf.can_use_fast_path
    assert pf.total_entries == 2
    assert pf.top_dirs == ["root"]
    assert not pf.risky_names
    assert not pf.case_collisions
    assert not pf.path_traversal


def test_pure_ascii_without_utf8_flag_is_not_risky(tmp_path):
    """关键用例：LogoDet-3K 的 317,308 条目全部未置 UTF-8 标志，但都是纯 ASCII。

    纯 ASCII 名用 cp437 / gbk / utf-8 解码结果完全一致，不构成乱码风险。
    早期实现只看标志位就判定"有风险"，会白白退回慢路径（实测慢两个数量级）。
    """
    z = _make_zip(
        tmp_path / "b.zip",
        {"LogoDet-3K/Food/coca-cola/1.jpg": b"x"},
        utf8_flag=False,
    )
    pf = preflight(z, tmp_path / "out")
    assert pf.utf8_flagged == 0
    assert not pf.risky_names, "纯 ASCII 名不应被判为有风险"
    assert pf.can_use_fast_path


def test_non_ascii_without_utf8_flag_is_risky():
    """直接测纯函数：用 zipfile 造不出这个场景。

    Python 的 writestr 遇到非 ASCII 文件名会**强制置位** UTF-8 标志，
    所以"非 ASCII 且未置位"这个组合无法通过 zipfile 构造出来。
    风险判定因此被抽成纯函数，在这一层直接验。
    """
    # 未置位 + 非 ASCII → 有风险
    assert is_risky_name("root/测试.jpg", 0) is True
    # 未置位 + 纯 ASCII → 无风险（LogoDet-3K 全部 317,308 条目属于此类）
    assert is_risky_name("root/plain.jpg", 0) is False
    # 已置位 → 无论内容都无风险
    assert is_risky_name("root/测试.jpg", 0x800) is False
    assert is_risky_name("root/plain.jpg", 0x800) is False


def test_case_collision_on_directory_component(tmp_path):
    """目录层级的大小写冲突必须被检出。

    `Adidas/1.jpg` 与 `adidas/2.jpg` 的**完整路径不同**，只比完整路径会漏。
    但在 APFS 上这两个目录会合并 —— 品牌目录就是类别，合并意味着
    3000 类变 2999，两边文件名再撞上就直接丢数据。
    """
    z = _make_zip(tmp_path / "d.zip", {"root/Adidas/1.jpg": b"x", "root/adidas/2.jpg": b"y"})
    pf = preflight(z, tmp_path / "out")
    assert pf.case_collisions, "目录层级冲突必须检出"
    assert "root/adidas" in pf.case_collisions
    assert sorted(pf.case_collisions["root/adidas"]) == ["root/Adidas", "root/adidas"]
    assert not pf.can_use_fast_path
    with pytest.raises(CaseCollisionError):
        safe_extract(z, tmp_path / "out", verbose=False)


def test_case_collision_on_full_path(tmp_path):
    """完整路径同名不同大小写 → 直接互相覆盖。"""
    z = _make_zip(tmp_path / "d2.zip", {"root/b/A.jpg": b"x", "root/b/a.jpg": b"y"})
    pf = preflight(z, tmp_path / "out")
    assert "root/b/a.jpg" in pf.case_collisions
    assert not pf.can_use_fast_path


def test_path_traversal_detected_and_blocks(tmp_path):
    z = tmp_path / "e.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../escape.txt", b"x")
    pf = preflight(z, tmp_path / "out")
    assert pf.path_traversal
    with pytest.raises(ZipSlipError):
        safe_extract(z, tmp_path / "out", verbose=False)


# ---------------------------------------------------------------------------
# 文件名解码
# ---------------------------------------------------------------------------


def test_decode_keeps_name_when_utf8_flag_set():
    info = zipfile.ZipInfo("root/中文.jpg")
    info.flag_bits |= 0x800
    name, recovered = decode_entry_name(info)
    assert name == "root/中文.jpg"
    assert recovered is False


def test_decode_keeps_pure_ascii_untouched():
    info = zipfile.ZipInfo("root/plain.jpg")
    info.flag_bits &= ~0x800
    name, recovered = decode_entry_name(info)
    assert name == "root/plain.jpg"
    assert recovered is False


def test_decode_recovers_gbk_mojibake():
    # 模拟：打包工具写入 GBK 字节但未置 UTF-8 标志，
    # Python 的 zipfile 会用 cp437 解码成一串拉丁字符
    mojibake = "测试.jpg".encode("gbk").decode("cp437")
    info = zipfile.ZipInfo(mojibake)
    info.flag_bits &= ~0x800
    name, recovered = decode_entry_name(info)
    assert recovered is True
    assert name == "测试.jpg"


# ---------------------------------------------------------------------------
# 端到端解压
# ---------------------------------------------------------------------------


def test_extract_roundtrip(tmp_path):
    entries = {
        "root/Food/brand/1.jpg": b"image-bytes",
        "root/Food/brand/1.xml": b"<annotation/>",
        "root/Food/brand/2.jpg": b"more",
        "root/Food/brand/2.xml": b"<annotation/>",
    }
    z = _make_zip(tmp_path / "f.zip", entries)
    out = tmp_path / "out"
    rep = safe_extract(z, out, verbose=False)

    assert rep.path_used in ("ditto", "python")
    assert rep.total_entries == 4
    for rel, data in entries.items():
        assert (out / rel).read_bytes() == data


def test_slow_path_can_be_forced(tmp_path):
    z = _make_zip(tmp_path / "g.zip", {"root/a.jpg": b"x"})
    out = tmp_path / "out"
    rep = safe_extract(z, out, verbose=False, force_slow_path=True)
    assert rep.path_used == "python"
    assert rep.extracted_files == 1
    assert (out / "root/a.jpg").read_bytes() == b"x"
