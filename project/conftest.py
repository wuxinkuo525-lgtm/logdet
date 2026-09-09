"""pytest 全局配置。

唯一职责：把 basetemp 强制指到工作区内。

本机沙箱会拦截对系统临时目录 /private/var/folders/... 的 mkdir
（返回 Node 风格 EEXIST），导致所有用到 tmp_path 的用例集体 ERROR，
且报错指向 sitecustomize.py，极易误判成代码问题。

放在 conftest 里而不是 pytest.ini，是因为 --basetemp 只能走命令行 /
钩子，不能写进 ini 的 addopts（会与 pytest 自身的清理逻辑冲突）。
"""

from __future__ import annotations

from pathlib import Path

# 定位到 logdet/runs/.pytest_tmp（conftest 在 project/ 下，上溯一层到 logdet）
_LOGDET_ROOT = Path(__file__).resolve().parent.parent
_TMP = _LOGDET_ROOT / "runs" / ".pytest_tmp"


def pytest_configure(config) -> None:
    if config.option.basetemp is None:
        _TMP.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(_TMP)
