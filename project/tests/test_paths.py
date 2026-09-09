"""paths.py 的回归测试。

重点是 test_nested_placeholder：S0-G5 曾因占位符正则写成 [^}]+ 而在
嵌套写法上截断，这条测试锁死修复后的行为。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logodet.paths import (  # noqa: E402
    CloudSyncPathError,
    _substitute,
    _UnresolvedRootRef,
    load_paths,
)


# ---------------------------------------------------------------------------
# 占位符展开
# ---------------------------------------------------------------------------


def test_env_placeholder_uses_default(monkeypatch):
    monkeypatch.delenv("LOGDET_TEST_VAR", raising=False)
    assert _substitute("${env:LOGDET_TEST_VAR|/fallback}", {}) == "/fallback"


def test_env_placeholder_prefers_environment(monkeypatch):
    monkeypatch.setenv("LOGDET_TEST_VAR", "/from/env")
    assert _substitute("${env:LOGDET_TEST_VAR|/fallback}", {}) == "/from/env"


def test_env_placeholder_without_default_raises(monkeypatch):
    monkeypatch.delenv("LOGDET_TEST_VAR", raising=False)
    with pytest.raises(KeyError):
        _substitute("${env:LOGDET_TEST_VAR}", {})


def test_roots_reference():
    assert _substitute("${roots.a}/sub", {"a": "/base"}) == "/base/sub"


def test_roots_reference_unresolved_signals_retry():
    # _substitute 层面遇到未解析的 root 会抛 _UnresolvedRootRef，
    # 这是给 load_paths 的"再等一轮"信号，不是最终错误。
    # 它是 KeyError 的子类，所以调用方按 KeyError 捕获也安全。
    with pytest.raises(_UnresolvedRootRef):
        _substitute("${roots.b}/sub", {"a": "/base"})


def test_nested_placeholder(monkeypatch):
    """S0-G5 抓到的回归：嵌套占位符不得在第一个 } 处截断。

    旧正则 [^}]+ 会把 "env:V|${roots.logdet" 当成一个完整 token，
    剩下 "/data}" 拼回去后变成 ${roots.logdet/data} → KeyError。
    """
    monkeypatch.delenv("LOGDET_NESTED", raising=False)
    got = _substitute("${env:LOGDET_NESTED|${roots.logdet}/data/LogoDet-3K}", {"logdet": "/L"})
    assert got == "/L/data/LogoDet-3K"


def test_nested_placeholder_env_wins(monkeypatch):
    monkeypatch.setenv("LOGDET_NESTED", "/override")
    got = _substitute("${env:LOGDET_NESTED|${roots.logdet}/data}", {"logdet": "/L"})
    assert got == "/override"


def test_unknown_placeholder_raises():
    with pytest.raises(KeyError):
        _substitute("${bogus.thing}", {})


# ---------------------------------------------------------------------------
# 云同步断言
# ---------------------------------------------------------------------------


def _write_paths_yaml(tmp_path: Path, root: str) -> Path:
    cfg = {
        "roots": {
            "logdet": root,
            "dataset": "${roots.logdet}/data/LogoDet-3K",
            "artifacts": "${roots.logdet}/runs",
            "raw": "${roots.logdet}/raw",
            "cache": "${roots.logdet}/cache",
        },
        "artifact_subdirs": {"tables": "tables", "logs": "logs"},
        "guards": {
            "forbid_path_substrings": ["Mobile Documents"],
            "check_roots": ["dataset", "artifacts", "raw", "cache"],
            "remediation": "关闭 iCloud 桌面同步",
        },
    }
    p = tmp_path / "paths.yaml"
    # 故意用 sort_keys=True（yaml 的默认值）：这会把 artifacts 排到 logdet
    # 之前，从而顺带验证 load_paths 的顺序无关性。
    # 早期实现要求"被引用的 root 必须写在前面"，正是这条测试抓出来的。
    p.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return p


def test_load_paths_is_order_independent(tmp_path):
    """roots 里被引用者写在后面也必须能解析。"""
    cfg_body = {
        "roots": {
            # artifacts 引用 logdet，但 logdet 写在最后
            "artifacts": "${roots.logdet}/runs",
            "dataset": "${roots.logdet}/data/LogoDet-3K",
            "raw": "${roots.logdet}/raw",
            "cache": "${roots.logdet}/cache",
            "logdet": str(tmp_path / "logdet"),
        },
        "artifact_subdirs": {"tables": "tables"},
        "guards": {"forbid_path_substrings": [], "check_roots": []},
    }
    p = tmp_path / "ordered.yaml"
    p.write_text(yaml.safe_dump(cfg_body, sort_keys=False), encoding="utf-8")
    paths = load_paths(p, create=False, check=False)
    assert paths.artifacts == tmp_path / "logdet" / "runs"
    assert paths.dataset == tmp_path / "logdet" / "data" / "LogoDet-3K"


def test_circular_reference_raises(tmp_path):
    cfg_body = {
        "roots": {
            "logdet": "${roots.dataset}/up",
            "dataset": "${roots.logdet}/down",
            "artifacts": "/tmp/a",
            "raw": "/tmp/r",
            "cache": "/tmp/c",
        },
        "guards": {"forbid_path_substrings": [], "check_roots": []},
    }
    p = tmp_path / "cyclic.yaml"
    p.write_text(yaml.safe_dump(cfg_body, sort_keys=False), encoding="utf-8")
    with pytest.raises(KeyError, match="循环引用"):
        load_paths(p, create=False, check=False)


def test_local_root_passes(tmp_path):
    cfg = _write_paths_yaml(tmp_path, str(tmp_path / "logdet"))
    p = load_paths(cfg, create=True)
    assert p.dataset.name == "LogoDet-3K"
    assert p.artifacts.is_dir()
    # dataset 故意不建：S1 要靠它是否存在来判断下载状态
    assert not p.dataset.exists()


def test_cloud_synced_root_rejected(tmp_path):
    fake_icloud = tmp_path / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "logdet"
    cfg = _write_paths_yaml(tmp_path, str(fake_icloud))
    with pytest.raises(CloudSyncPathError) as e:
        load_paths(cfg, create=False)
    assert "Mobile Documents" in str(e.value)
    assert "关闭 iCloud 桌面同步" in str(e.value)


def test_artifact_subdir_must_be_registered(tmp_path):
    cfg = _write_paths_yaml(tmp_path, str(tmp_path / "logdet"))
    p = load_paths(cfg, create=True)
    assert p.artifact("tables").is_dir()
    # 未登记的子目录必须报错，防止代码里散落硬编码目录名
    with pytest.raises(KeyError):
        p.artifact("not_registered")


def test_image_path_composition(tmp_path):
    cfg = _write_paths_yaml(tmp_path, str(tmp_path / "logdet"))
    p = load_paths(cfg, create=True)
    assert p.image("Food/coca-cola/1.jpg") == p.dataset / "Food/coca-cola/1.jpg"


# ---------------------------------------------------------------------------
# 随机流派生
# ---------------------------------------------------------------------------


def test_seed_streams_are_independent_and_reproducible():
    from logodet.seeds import seed_for

    a1 = seed_for("split_agnostic").integers(0, 10**6, 5).tolist()
    a2 = seed_for("split_agnostic").integers(0, 10**6, 5).tolist()
    b1 = seed_for("val2k").integers(0, 10**6, 5).tolist()

    assert a1 == a2, "同一 purpose 必须完全可复现"
    assert a1 != b1, "不同 purpose 必须是独立的流"
