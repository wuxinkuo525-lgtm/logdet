#!/bin/bash
# CCDS GPU Cluster TC2：创建 DINO 训练用的 conda 环境。
#
# 在登录节点（head node）上跑一次即可，只装包、不跑计算：
#     bash scripts/cluster/tc2_setup_env.sh
#
# 使用本项目验证的版本组合；下载站同时提供 Linux/Windows wheel，pip 按当前平台选择。
# 不复制 Windows conda 环境或 win_amd64.whl 到集群。

set -euo pipefail

ENV_NAME="${ENV_NAME:-logodet_dino}"

# 集群的 conda/Qt 激活脚本会读取未定义变量；仅在环境初始化期间关闭 nounset。
# 保留 errexit 和 pipefail，安装/激活失败仍立即退出。
set +u
module load anaconda
eval "$(conda shell.bash hook)"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -y -n "${ENV_NAME}" python=3.11
fi
conda activate "${ENV_NAME}"
set -u

python - <<'EOF'
import platform, sys
if sys.platform != 'linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 11):
    raise SystemExit('This setup requires Linux x86_64 and Python 3.11 (MMCV wheel target).')
EOF

pip install "torch==2.1.0+cu121" "torchvision==0.16.0+cu121" \
    --index-url https://download.pytorch.org/whl/cu121
pip install "numpy==1.26.4" "opencv-python==4.10.0.84"
python -m pip install --only-binary=mmcv "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
pip install "mmengine==0.10.7" "mmdet==3.3.0"
# mmengine / mmdet 的依赖会把 numpy 拉回 2.x，torch 2.1 用不了，最后再钉一次
pip install "numpy==1.26.4"
# 新建的 conda 环境自带最新版 setuptools，已经删掉了 pkg_resources；mmengine 0.10.7 读
# `_base_ = "mmdet::..."` 这种配置时要 import 它，缺了就在加载配置的第一步报 ModuleNotFoundError
pip install "setuptools==69.5.1"

python - <<'EOF'
import mmcv, mmdet, mmengine, numpy, torch, torchvision
from mmcv.ops import MultiScaleDeformableAttention  # noqa: F401  编译算子能 import 才算装对
from mmengine.utils import is_installed
assert is_installed("mmdet")  # 走一遍 pkg_resources，训练脚本加载配置时就是这条路
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| numpy", numpy.__version__)
print("mmcv", mmcv.__version__, "| mmengine", mmengine.__version__, "| mmdet", mmdet.__version__)
EOF
echo "环境 ${ENV_NAME} 就绪。GPU 是否可用要在计算节点上验证：sbatch scripts/cluster/tc2_dino.sbatch.sh sanity"
