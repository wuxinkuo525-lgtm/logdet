"""路径解析与三区隔离断言（three-zone isolation）。

全项目只有 configs/paths.yaml 含绝对路径；本模块把它解析成 Path 对象，
并在任何路径被真正使用之前断言它不落在云同步目录内。

两条写死的设计约束（不要放宽）：

1. dataset / artifacts / raw / cache 四个 root 必须在本机盘。
   断言在 Path.resolve() 之后做，所以软链接绕不过去。

2. 数据表里只存 rel_path，绝对路径运行时现拼。
   这样 runs/ 下的产物可以整体搬到别的机器而不失效。

典型用法::

    from logodet.paths import P

    P.dataset                      # ~/Desktop/logdet/data/LogoDet-3K
    P.artifact("tables")           # ~/Desktop/logdet/runs/tables（已 mkdir）
    P.image("Food/coca-cola/1.jpg")  # 拼出图像绝对路径
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ${env:VAR|default} 或 ${roots.xxx}
#
# 字符类里排除 { 和 } 是关键：这样正则只会匹配**最内层**的占位符。
# 配合下面的循环展开，嵌套写法 ${env:VAR|${roots.logdet}/data} 会先把
# 内层的 ${roots.logdet} 换掉，再整体匹配外层的 ${env:...}。
# 若写成 [^}]+ 则会在第一个 } 处截断，把 "env:VAR|${roots.logdet" 当成一个
# 完整 token —— 这个 bug 在 S0-G5 上被抓到过一次。
_TOKEN = re.compile(r"\$\{([^{}]*)\}")

# paths.py 位于 project/src/logodet/paths.py
#   parents[0] = logodet, parents[1] = src, parents[2] = project
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_YAML = PROJECT_ROOT / "configs" / "paths.yaml"


class CloudSyncPathError(RuntimeError):
    """某个 root 解析后落在云同步目录内。"""


class _UnresolvedRootRef(KeyError):
    """引用了一个本轮还没解析出来的 root。

    这不是错误，只是"再等一轮"的信号 —— load_paths 靠它实现顺序无关的
    多轮解析。之所以要一个专门的异常类而不是靠比对错误文本：
    文本匹配会在改一句提示语时静默失效。
    """


def _substitute(raw: str, resolved: dict[str, str]) -> str:
    """把单个字符串里的 ${env:...} 与 ${roots.xxx} 展开。

    未定义的引用直接抛错，绝不静默留下字面量 —— 静默会导致
    程序在几百行之后才因为一个叫 "${roots.logdet}/data" 的
    目录不存在而失败，排查成本极高。
    """

    def repl(m: re.Match[str]) -> str:
        token = m.group(1).strip()

        if token.startswith("env:"):
            body = token[4:]
            var, sep, default = body.partition("|")
            val = os.environ.get(var.strip())
            if val:
                return val
            if not sep:
                raise KeyError(f"环境变量 {var.strip()} 未设置且无默认值：${{{token}}}")
            return default

        if token.startswith("roots."):
            key = token[6:]
            if key not in resolved:
                raise _UnresolvedRootRef(key)
            return resolved[key]

        raise KeyError(f"无法识别的路径占位符：${{{token}}}")

    # 循环展开，支持 ${env:A|${roots.b}/x} 这类嵌套
    out = raw
    for _ in range(8):
        new = _TOKEN.sub(repl, out)
        if new == out:
            return new
        out = new
    raise RecursionError(f"路径占位符展开超过 8 层，疑似循环引用：{raw}")


def _expand(raw: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(raw)))


@dataclass(frozen=True)
class Paths:
    """解析完成的路径集合。所有字段都是已 resolve 的绝对路径。"""

    project: Path
    logdet: Path
    dataset: Path
    artifacts: Path
    raw: Path
    cache: Path
    artifact_subdirs: dict[str, str] = field(default_factory=dict)
    source_yaml: Path | None = None

    # ---- 派生路径 ---------------------------------------------------------

    def artifact(self, name: str, *, mkdir: bool = True) -> Path:
        """取 runs/ 下的子目录。name 必须在 paths.yaml 的 artifact_subdirs 里登记过。"""
        if name not in self.artifact_subdirs:
            raise KeyError(
                f"未登记的产物子目录 {name!r}；"
                f"可用：{sorted(self.artifact_subdirs)}。"
                f"新增请改 configs/paths.yaml，不要在代码里硬编码。"
            )
        p = self.artifacts / self.artifact_subdirs[name]
        if mkdir:
            p.mkdir(parents=True, exist_ok=True)
        return p

    def image(self, rel_path: str | Path) -> Path:
        """把表里的 rel_path 拼成图像绝对路径。"""
        return self.dataset / rel_path

    def run_dir(self, run_id: str, *, mkdir: bool = True) -> Path:
        p = self.artifact("predictions") / run_id
        if mkdir:
            p.mkdir(parents=True, exist_ok=True)
        return p

    # ---- 自检 -------------------------------------------------------------

    def as_table(self) -> list[tuple[str, str]]:
        return [
            ("project", str(self.project)),
            ("logdet", str(self.logdet)),
            ("dataset", str(self.dataset)),
            ("artifacts", str(self.artifacts)),
            ("raw", str(self.raw)),
            ("cache", str(self.cache)),
        ]


def _check_guards(values: dict[str, Path], guards: dict[str, Any]) -> None:
    forbidden = guards.get("forbid_path_substrings") or []
    to_check = guards.get("check_roots") or []
    remediation = (guards.get("remediation") or "").rstrip()

    problems: list[str] = []
    for name in to_check:
        if name not in values:
            raise KeyError(f"guards.check_roots 里的 {name!r} 不是已定义的 root")
        # resolve() 之后再比对：软链接 / iCloud 桌面同步都绕不过去
        real = str(values[name].resolve())
        for mark in forbidden:
            if mark in real:
                problems.append(f"  {name:<10} → {real}\n             命中禁止片段：{mark!r}")

    if problems:
        raise CloudSyncPathError(
            "以下 root 落在云同步目录内，拒绝继续：\n"
            + "\n".join(problems)
            + ("\n\n" + remediation if remediation else "")
        )


def load_paths(
    config_path: str | Path | None = None,
    *,
    create: bool = True,
    check: bool = True,
) -> Paths:
    """读 paths.yaml 并解析。

    Args:
        config_path: 默认 project/configs/paths.yaml
        create: 是否 mkdir 四个 root（dataset 除外 —— 它由 S1 下载阶段建立，
            提前建空目录会让 S1 的"数据是否存在"判断失真）
        check: 是否执行云同步断言。只有在写单测时才该关掉。
    """
    cfg_path = Path(config_path) if config_path else DEFAULT_PATHS_YAML
    if not cfg_path.is_file():
        raise FileNotFoundError(f"找不到路径配置：{cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    # 顺序无关的多轮解析。
    #
    # 为什么不要求"被引用的 root 必须写在前面"：yaml 的键顺序很容易被
    # 改动（有人手动整理、或工具以 sort_keys=True 重写），一旦 artifacts
    # 排到 logdet 之前就会炸。多轮解析让 paths.yaml 里的书写顺序完全自由。
    pending: dict[str, str] = {k: str(v) for k, v in (cfg.get("roots") or {}).items()}
    resolved_str: dict[str, str] = {}
    resolved: dict[str, Path] = {}

    for _ in range(len(pending) + 1):
        if not pending:
            break
        progressed = False
        for key in list(pending):
            try:
                expanded = _substitute(pending[key], resolved_str)
            except _UnresolvedRootRef:
                continue  # 依赖还没就绪，下一轮再试
            p = _expand(expanded)
            resolved_str[key] = str(p)
            resolved[key] = p
            del pending[key]
            progressed = True
        if not progressed:
            raise KeyError(
                "以下 root 无法解析，疑似循环引用或引用了不存在的 root："
                f"{ {k: v for k, v in pending.items()} }（已解析：{sorted(resolved)}）"
            )

    missing = {"logdet", "dataset", "artifacts", "raw", "cache"} - set(resolved)
    if missing:
        raise KeyError(f"paths.yaml 缺少必需的 root：{sorted(missing)}")

    if check:
        _check_guards(resolved, cfg.get("guards") or {})

    if create:
        # dataset 故意不建：S1 要靠它是否存在来判断数据下载状态
        for name in ("logdet", "artifacts", "raw"):
            resolved[name].mkdir(parents=True, exist_ok=True)

    return Paths(
        project=PROJECT_ROOT,
        logdet=resolved["logdet"],
        dataset=resolved["dataset"],
        artifacts=resolved["artifacts"],
        raw=resolved["raw"],
        cache=resolved["cache"],
        artifact_subdirs=dict(cfg.get("artifact_subdirs") or {}),
        source_yaml=cfg_path,
    )


class _LazyPaths:
    """模块级单例。首次访问属性时才真正读 yaml 并跑断言。"""

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        self._inner: Paths | None = None

    def _get(self) -> Paths:
        if self._inner is None:
            self._inner = load_paths()
        return self._inner

    def __getattr__(self, item: str) -> Any:
        return getattr(self._get(), item)

    def reload(self, **kwargs: Any) -> Paths:
        self._inner = load_paths(**kwargs)
        return self._inner


P = _LazyPaths()
