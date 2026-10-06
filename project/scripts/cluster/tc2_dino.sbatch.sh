#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --job-name=logodet_dino
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
#
# CCDS GPU Cluster TC2 上的 DINO 作业。必须在 project/ 目录下提交：
#
#     cd ~/logdet/project
#     sbatch scripts/cluster/tc2_dino.sbatch.sh sanity          # 单 batch 检查，几分钟
#     sbatch scripts/cluster/tc2_dino.sbatch.sh smoke           # 300 iter，拿真实速度和显存
#     sbatch scripts/cluster/tc2_dino.sbatch.sh shared --run-name shared_lr1e-4 --lr 1e-4
#     sbatch scripts/cluster/tc2_dino.sbatch.sh full --epochs 4 --lr 1e-4
#
# 上面的资源数是手册里 QoS "normal" 的上限（1 GPU / 10 核 / 30G / 6 小时）。
# 先用 `mytcinfo` 核对自己实际被分到的 QoS，不一致就改 #SBATCH 行。
#
# shared / full 模式跑不完 6 小时就会被杀，但每 2000 iter 存一次档，
# **原样再提交同一条命令**就从最新存档接着跑，直到训练结束。
#
# 目录约定（与本地一致，三区隔离）：
#     ~/logdet/project   代码
#     ~/logdet/data      数据集（LogoDet-3K/<超类>/<品牌>/*.jpg）
#     ~/logdet/runs      产物：coco/ 标注、cache/checkpoints/ 预训练权重、dino/ 训练输出

set -euo pipefail

MODE="${1:?用法: sbatch tc2_dino.sbatch.sh <sanity|tiny|smoke|shared|full> [传给训练脚本的其余参数]}"
shift

cd "${SLURM_SUBMIT_DIR}"
export LOGDET_ROOT="${LOGDET_ROOT:-$(dirname "$(pwd)")}"
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export LOGDET_REQUIRE_GPU=1
RUNS_ROOT="${LOGDET_RUNS_ROOT:-${LOGDET_ROOT}/runs}"
run_py() {
    python -u scripts/s9_dino_watch.py --status "${RUNS_ROOT}/dino/health/job_${SLURM_JOB_ID}.json" \
        --stale-seconds "${STALE_SECONDS:-900}" -- python -u "$@"
}

# 集群 conda/Qt 激活脚本不兼容 nounset；激活后恢复严格检查。
set +u
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME:-logodet_dino}"
set -u

echo "node=$(hostname)  job=${SLURM_JOB_ID}  mode=${MODE}  LOGDET_ROOT=${LOGDET_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "${MODE}" = "sanity" ]; then
    run_py scripts/s9_dino_sanity.py "$@"
elif [ "${MODE}" = "full" ] || [ "${MODE}" = "shared" ]; then
    run_py scripts/s9_dino_train.py --mode "${MODE}" --resume --num-workers 8 "$@"
else
    run_py scripts/s9_dino_train.py --mode "${MODE}" --num-workers 8 "$@"
fi
