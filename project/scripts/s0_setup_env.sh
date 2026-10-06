#!/usr/bin/env bash
# ============================================================================
# S0：创建虚拟环境并安装锁定版本依赖。
#
# 为什么用 venv 而不是 conda create：
#   本机的删除守卫（safe-delete）会拦截 conda 建 env 时对内部临时索引文件
#   （.tmp.index.json.*）的 unlink，导致 "Preparing transaction" 阶段必然失败。
#   venv 的创建过程只写不删，天然不触发该守卫。
#   附带好处：环境自包含在 logdet 根目录下，不再依赖 conda 的 env 管理。
#
# 基础解释器：需要 Python 3.11 或 3.12。
#   不用 3.13 —— torch 2.5.1 只提供到 cp312 的 arm64 wheel。
#   本机已探明可用：~/miniconda3/envs/ca6126/bin/python (3.11.15)
#
# 幂等：重复执行安全。venv 已存在且解释器可用时只重跑 pip install。
#
# 用法：
#   bash scripts/s0_setup_env.sh
#   BASE_PYTHON=/path/to/python3.11 bash scripts/s0_setup_env.sh   # 手动指定基础解释器
# ============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDET_ROOT="${LOGDET_ROOT:-$(cd "$PROJECT_DIR/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$LOGDET_ROOT/.venv}"
LOCK_FILE="$PROJECT_DIR/environment/requirements.lock.txt"

# venv 内部布局在 Windows（含 Git Bash/MSYS）上是 Scripts/python.exe，
# 在 macOS/Linux 上是 bin/python —— 这是 CPython venv 模块本身按平台
# 决定的，不是 shell 决定的。按 OSTYPE 探测一次，后面统一用这两个变量。
case "${OSTYPE:-}" in
  msys*|cygwin*|win32*)
    VENV_BINDIR="Scripts"
    VENV_PYEXE="python.exe"
    ;;
  *)
    VENV_BINDIR="bin"
    VENV_PYEXE="python"
    ;;
esac

# ---- 挑选基础解释器 --------------------------------------------------------

pick_base_python() {
  if [[ -n "${BASE_PYTHON:-}" ]]; then
    echo "$BASE_PYTHON"
    return
  fi
  local cands=(
    "$HOME/miniconda3/envs/ca6126/bin/python"
    "$HOME/miniconda3/envs/alfworld/bin/python"
    "$HOME/miniconda3/envs/class_py312/bin/python"
    "$(command -v python3.11 || true)"
    "$(command -v python3.12 || true)"
  )
  for c in "${cands[@]}"; do
    [[ -x "$c" ]] || continue
    # 必须是 3.11 或 3.12
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)' 2>/dev/null; then
      echo "$c"
      return
    fi
  done
  return 1
}

BASE_PY="$(pick_base_python)" || {
  echo "FAIL: 找不到 Python 3.11 或 3.12 作为基础解释器。" >&2
  echo "      torch 2.5.1 只有 cp311 / cp312 的 arm64 wheel，3.13 装不上。" >&2
  echo "      处置：BASE_PYTHON=/path/to/python3.11 bash scripts/s0_setup_env.sh" >&2
  exit 1
}
BASE_VER="$("$BASE_PY" -c 'import sys;print(sys.version.split()[0])')"

echo "=============================================================="
echo " S0 环境搭建"
echo "   基础解释器 : $BASE_PY  ($BASE_VER)"
echo "   venv       : $VENV_DIR"
echo "   依赖锁     : $LOCK_FILE"
echo "=============================================================="

[[ -f "$LOCK_FILE" ]] || { echo "FAIL: 找不到依赖锁 $LOCK_FILE" >&2; exit 1; }

# venv 绝不能落在云同步目录里 —— site-packages 有几万个小文件
VENV_PARENT_REAL="$(cd "$(dirname "$VENV_DIR")" && pwd -P)"
case "$VENV_PARENT_REAL" in
  *"Mobile Documents"*|*"Google Drive"*|*"Dropbox"*|*"OneDrive"*)
    echo "FAIL: venv 落在云同步目录内（$VENV_PARENT_REAL）" >&2
    echo "      处置：关闭 iCloud 桌面同步，或 VENV_DIR=~/logdet_venv 重跑" >&2
    exit 1
    ;;
esac

# ---- 建 venv ---------------------------------------------------------------

PY="$VENV_DIR/$VENV_BINDIR/$VENV_PYEXE"

if [[ -x "$PY" ]] && "$PY" -c "import sys" 2>/dev/null; then
  echo "[1/3] venv 已存在且解释器可用，跳过创建"
else
  # 判据用「能否真的启动」而不是「文件是否存在」：
  # 若上次创建被中途打断，会留下一个存在但一启动就 SIGKILL 的解释器
  # （macOS 对 adhoc 签名与内容不匹配的二进制直接杀，且没有任何 traceback）。
  if [[ -e "$VENV_DIR" ]]; then
    STALE="$VENV_DIR.stale.$(date +%s)"
    echo "[1/3] 检测到不可用的 venv，重命名为 $(basename "$STALE") 后重建 ..."
    # 用 mv 而不是 rm：删除守卫会拦大批量 unlink，重命名不受影响
    mv "$VENV_DIR" "$STALE"
  fi
  echo "[1/3] 创建 venv ..."
  "$BASE_PY" -m venv "$VENV_DIR"
fi

"$PY" -c "import sys" 2>/dev/null || {
  echo "FAIL: $PY 无法启动" >&2
  exit 1
}

# ---- 装依赖 ----------------------------------------------------------------

echo "[2/3] 安装锁定依赖 ..."
# pip 自升级失败不中断链路
"$PY" -m pip install --upgrade pip -q 2>/dev/null || echo "  (pip 自升级失败，用自带版本继续)"
"$PY" -m pip install -r "$LOCK_FILE"

# ---- 写 activate 钩子 ------------------------------------------------------

echo "[3/3] 写入 activate 环境变量 ..."
HOOK_MARK="# --- logodet env vars ---"
if ! grep -qF "$HOOK_MARK" "$VENV_DIR/$VENV_BINDIR/activate" 2>/dev/null; then
  cat >> "$VENV_DIR/$VENV_BINDIR/activate" <<EOF

$HOOK_MARK
# MPS 遇到未实现算子时回退 CPU，而不是直接抛 NotImplementedError
export PYTORCH_ENABLE_MPS_FALLBACK=1
# transformers 的 tokenizer 并行在 DataLoader worker 里会刷警告
export TOKENIZERS_PARALLELISM=false
# 主进程留够线程；worker 内另行设为 1（见 data/loader.py）
export OMP_NUM_THREADS=8
# 模型权重统一落本机 cache，不进项目目录
export HF_HOME="\${HF_HOME:-\$HOME/.cache/huggingface}"
EOF
fi

# 记录基础解释器来源，便于日后排查 venv 失效
cat > "$PROJECT_DIR/environment/venv_base.txt" <<EOF
# 由 scripts/s0_setup_env.sh 自动生成
# venv 通过符号链接依赖下面这个基础解释器；若它被删除，venv 会失效。
# 重建：BASE_PYTHON=<任意 python3.11 或 3.12> bash scripts/s0_setup_env.sh
base_python=$BASE_PY
base_version=$BASE_VER
venv_dir=$VENV_DIR
created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo
echo "完成。下一步跑验证门："
echo "  $PY $PROJECT_DIR/scripts/s0_verify_env.py"
