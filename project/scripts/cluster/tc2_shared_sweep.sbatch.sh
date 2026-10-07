#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=logodet_sweep
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
#
# 一条命令把 shared 子集（5,000 图）上的整套调参依次跑完。必须在 project/ 目录下提交：
#
#     cd ~/logdet/project
#     sbatch scripts/cluster/tc2_shared_sweep.sbatch.sh
#
# **正式跑之前先彩排一遍**（同样的九组、同样的流程，但每组只用 20 张图，全程约半小时）：
#
#     sbatch --export=ALL,REHEARSAL=1 scripts/cluster/tc2_shared_sweep.sbatch.sh
#
# 彩排的输出都带 rehearsal_ 前缀，不会和正式结果混在一起；跑完保留存档供完整性校验。
# 看到日志末尾的「彩排通过」才算过。想连「墙钟到点自动续交」也一起彩排，把时限压短：
#
#     sbatch --time=00:12:00 --export=ALL,REHEARSAL=1 scripts/cluster/tc2_shared_sweep.sbatch.sh
#
# 四个阶段，后一个阶段用前一个阶段自动选出的结果：
#
#   1. 学习率      shared_lr<LR>       三组 lr，各 4 epoch          → 选出最好的 lr
#   2. 变体        shared_noaux        关掉两个辅助分支，4 epoch     → 比基准好才采用
#                  shared_hires        放大输入，4 epoch            → 比基准好才采用；任何错误都停止流程
#   3. 长跑        shared_long         用选出的 lr + 采用的变体，恒定 lr 跑 12 epoch，
#                                      每 4 epoch 留一个里程碑存档，给阶段 4 当起点
#   4. 衰减分叉    shared_decay_e<N>   从第 N 个 epoch 的里程碑起，lr ×0.1 再训 1 epoch
#                                      （近似「第 N epoch 降 lr、共 N+1 epoch」的计划），N = 8 / 12
#                                      → 与 4 epoch 的基准组一起，比出 4 / 9 / 13 epoch 哪个最好
#
# 「好」的标准：dev_eval（256 图）上最终评测的 agn/AP（单类口径，只看框没框到 logo）。
# 每次选择连同参与比较的各组指标都写进 runs/dino/sweep_*.json，可以事后复核。
#
# 每组的流程：训练 → 出预测和超参留痕（runs/predictions/dino/<组名>/）→ 写 DONE 标记 → 下一组。
#
# 墙钟（6 小时）到点前 5 分钟，作业会**尝试自己提交下一个作业**然后退出，下一个作业从最新存档接着跑。
# 这一步在 TC2 上第一次实跑时失败了（sbatch: Access/permission denied，原因未确认），
# 所以不要指望它：到点后看日志，没有「已自动提交下一个作业」就手动再提交同一条命令。
# 只有「到点」才会尝试续交；训练自己报错退出时不会（否则会无限重试）。
# 手动再提交永远是安全的：有 DONE 标记的组直接跳过，被打断的那组续训。
# **整套跑完之前不要动 runs/dino/ 下的文件**（不删、不移、不改名），选优会校验存档。
#
# 可调的环境变量（用 sbatch --export=ALL,名字=值 传；续跑时要和第一次一样）：
#     LRS="1e-4 2e-4 5e-5"   阶段 1 的学习率
#     BEST_LR=1e-4           不用自动选的 lr，自己指定
#     DECAY_AT="8 12"        阶段 4 从哪些 epoch 分叉（必须是 4 的倍数且不超过长跑的 epoch 数，对应里程碑存档）
#     STOP_AFTER=2           只跑到第几阶段（默认 4）
#     AUTO_RESUBMIT=0        关掉到点自动续交
#     CHAIN_LEFT=12          最多自动续交多少次（防止意外的无限续交）
#     REHEARSAL=1            彩排，见上
#     ROUND2="long20 aux"    第一轮跑完后接着跑第二轮的哪几组（默认不跑）：
#                              long20  长跑延到 20 epoch，从第 16 / 20 个 epoch 再分叉（约 5.5 小时）
#                              aux     两个辅助分支分别消融，4 组各 9 epoch（约 19 小时）
#                              auxw    辅助 loss 权重 0.1 对 0.5（基准组与 aux 共用，另加约 4.7 小时）
#                              wd      weight decay 1e-4 对 1e-3 / 1e-5（另加约 9.5 小时）
#     MIN_BASE_AP=0.20       阶段 1 选出的基准组 agn/AP 低于这个数就停下（退出码 3），不进入后面的阶段
#     SKIP_HIRES=1           不跑放大输入那组（它报显存不够时用：看过日志确认是 OOM 后，带上这个再提交）
#     BATCH=4                每步喂几张图（默认 1）。不同 batch 的组各自一套名字和选优留痕
#                            （shared_b4_lr1e-4、sweep_b4_*.json……），互不覆盖，可以并存对照。
#                            A40 实测 batch 4：1.40 s/iter、显存 28.4 GB，每张图比 batch 1 快 1.44 倍；
#                            混合精度（--amp）在这个模型上第一步就出现非有限梯度，不能用。
#
# 资源数是手册里 QoS "normal" 的上限，先用 `mytcinfo` 核对，不一致就改 #SBATCH 行。

set -euo pipefail

LRS="${LRS:-1e-4 2e-4 5e-5}"
DECAY_AT="${DECAY_AT:-8 12}"
STOP_AFTER="${STOP_AFTER:-4}"
REHEARSAL="${REHEARSAL:-0}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
CHAIN_LEFT="${CHAIN_LEFT:-12}"
BATCH="${BATCH:-1}"
SELF="scripts/cluster/tc2_shared_sweep.sbatch.sh"

cd "${SLURM_SUBMIT_DIR}"
export LOGDET_ROOT="${LOGDET_ROOT:-$(dirname "$(pwd)")}"
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export LOGDET_REQUIRE_GPU=1
RUNS_ROOT="${LOGDET_RUNS_ROOT:-${LOGDET_ROOT}/runs}"
mkdir -p "${RUNS_ROOT}/dino/health"
JOB_STATE=RUNNING
job_exit() {
    local RC=$?
    if [ "${JOB_STATE}" = RUNNING ]; then
        if [ "${RC}" -eq 0 ]; then JOB_STATE=COMPLETED; else JOB_STATE=FAILED; fi
    fi
    printf '%s state=%s exit=%s\n' "$(date -Is)" "${JOB_STATE}" "${RC}" | tee "${RUNS_ROOT}/dino/health/job_${SLURM_JOB_ID}.txt"
}
trap job_exit EXIT
HP="configs/dino/hparams"
BASE="${HP}/shared5k.yaml"

# 集群 conda/Qt 激活脚本不兼容 nounset；激活后恢复严格检查。
set +u
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME:-logodet_dino}"
set -u

echo "node=$(hostname)  job=${SLURM_JOB_ID}  LOGDET_ROOT=${LOGDET_ROOT}  rehearsal=${REHEARSAL}  batch=${BATCH}  chain_left=${CHAIN_LEFT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- 到点自动续交 ----
# python 都放后台再 wait：bash 只有在 wait 时才能立刻响应信号，前台子进程会把信号一直压到它结束。
CHILD=""
run_py() {
    python -u scripts/s9_dino_watch.py --status "${RUNS_ROOT}/dino/health/job_${SLURM_JOB_ID}.json" \
        --stale-seconds "${STALE_SECONDS:-900}" -- python -u "$@" &
    CHILD=$!
    local RC=0
    wait "${CHILD}" || RC=$?
    CHILD=""
    return "${RC}"
}
on_timeout() {
    trap '' USR1
    echo "=== 墙钟快到了（$(date '+%F %T')）"
    local RC=124
    JOB_STATE=NEEDS_RESUME
    if [ "${AUTO_RESUBMIT}" = "1" ] && [ "${CHAIN_LEFT}" -gt 0 ]; then
        export LRS DECAY_AT STOP_AFTER REHEARSAL AUTO_RESUBMIT BATCH SKIP_HIRES MIN_BASE_AP ROUND2
        if sbatch --dependency="afterany:${SLURM_JOB_ID}" --export=ALL,CHAIN_LEFT=$((CHAIN_LEFT - 1)) "${SELF}"; then
            JOB_STATE=RESUBMITTED
            RC=0
            echo "=== 已自动提交下一个作业（还可自动续交 $((CHAIN_LEFT - 1)) 次）"
        else
            JOB_STATE=RESUBMIT_FAILED
            RC=1
            echo "=== 自动续交失败：请手动执行 sbatch ${SELF}"
        fi
    else
        echo "=== 不自动续交（AUTO_RESUBMIT=${AUTO_RESUBMIT}，CHAIN_LEFT=${CHAIN_LEFT}）：请手动执行 sbatch ${SELF}"
    fi
    if [ -n "${CHILD}" ]; then
        kill -TERM "${CHILD}" 2>/dev/null || true
        wait "${CHILD}" 2>/dev/null || true
    fi
    exit "${RC}"
}
trap on_timeout USR1
on_terminated() {
    JOB_STATE=INTERRUPTED
    if [ -n "${CHILD}" ]; then
        kill -TERM "${CHILD}" 2>/dev/null || true
        wait "${CHILD}" 2>/dev/null || true
    fi
    exit 143
}
trap on_terminated TERM INT

# ---- 彩排：同样的流程，极小的数据 ----
P="shared"            # 组名前缀
SWEEP="sweep"         # 选择留痕文件前缀
TRAIN_EXTRA=()
PREDICT_EXTRA=()
VERIFY_EXTRA=()
N_TRAIN=5000          # 训练图数，用来算每个 epoch 多少 iter（里程碑存档按 iter 命名）
CKPT_INTERVAL=""      # 空 = 用预设里的 2000
IDS="${RUNS_ROOT}/handoff/shared_image_ids.json"
if [ "${REHEARSAL}" = "1" ]; then
    P="rehearsal_shared"
    SWEEP="rehearsal_sweep"
    N_TRAIN=20
    CKPT_INTERVAL=20
    IDS="${RUNS_ROOT}/handoff/rehearsal_image_ids.json"
    run_py scripts/s9_dino_preflight.py --make-rehearsal-ids "${RUNS_ROOT}/handoff/shared_image_ids.json" "${IDS}"
    TRAIN_EXTRA=(--shared-ids "${IDS}")
    PREDICT_EXTRA=(--max-images 30)
    VERIFY_EXTRA=(--allow-partial)
fi
if [ "${BATCH}" != "1" ]; then
    P="${P}_b${BATCH}"
    SWEEP="${SWEEP}_b${BATCH}"
    TRAIN_EXTRA+=(--batch-size "${BATCH}")
    # 存档间隔按图数折算：batch 4 时 2000 iter 是 47 分钟，被墙钟杀掉丢得太多
    [ -z "${CKPT_INTERVAL}" ] && CKPT_INTERVAL=$((2000 / BATCH))
fi
[ -n "${CKPT_INTERVAL}" ] && TRAIN_EXTRA+=(--ckpt-interval "${CKPT_INTERVAL}")
ITERS_PER_EPOCH=$(((N_TRAIN + BATCH - 1) / BATCH))

# ---- 开跑前自检：缺文件、缺图、没 GPU、磁盘不够，都在第一分钟报出来，而不是半夜 ----
run_py scripts/s9_dino_preflight.py --shared-ids "${IDS}"

# run_one <组名> <lr> [传给训练脚本的其余参数]
run_one() {
    local NAME="$1" LR="$2"
    shift 2
    local RUN_DIR="${RUNS_ROOT}/dino/${NAME}"
    if [ -f "${RUN_DIR}/DONE" ] && [ -f "${RUN_DIR}/final_eval.json" ]; then
        # 同名目录里已完成的实验必须就是这次要的那个：预设 / 变体 / batch 改过就在这里报错，不悄悄沿用旧结果
        run_py scripts/s9_dino_train.py --mode shared --resume --plan-only --run-name "${NAME}" --lr "${LR}" \
            ${TRAIN_EXTRA[@]+"${TRAIN_EXTRA[@]}"} "$@" || return $?
        verify_run "${NAME}" || return $?
        echo "=== ${NAME}：已完成，跳过"
        return 0
    fi

    echo "=== ${NAME}：开始 / 续训  $(date '+%F %T')"
    # 显式 return：本函数有时在「失败了也要继续」的上下文里被调用，那里 set -e 不生效
    run_py scripts/s9_dino_train.py --mode shared --resume --num-workers 8 --run-name "${NAME}" --lr "${LR}" \
        ${TRAIN_EXTRA[@]+"${TRAIN_EXTRA[@]}"} "$@" || return $?

    # 出表预测与超参留痕必须完整生成并验证后才能写 DONE
    if [ -f "${RUNS_ROOT}/eval/gt/infer_union.json" ]; then
        run_py scripts/s9_dino_predict.py --run-name "${NAME}" ${PREDICT_EXTRA[@]+"${PREDICT_EXTRA[@]}"} || return $?
    else
        echo "FAIL: 缺少 infer_union.json，不能把没有预测的组标记完成" >&2
        return 1
    fi

    verify_run "${NAME}" || return $?
    date '+%F %T' > "${RUN_DIR}/DONE"
    echo "=== ${NAME}：完成  $(date '+%F %T')"
}

verify_run() {
    run_py scripts/s9_dino_verify.py --run-dir "${RUNS_ROOT}/dino/$1" \
        --pred-dir "${RUNS_ROOT}/predictions/dino/$1" --union "${RUNS_ROOT}/eval/gt/infer_union.json" \
        ${VERIFY_EXTRA[@]+"${VERIFY_EXTRA[@]}"}
}

# pick <留痕名> <组名...>：打印其中最好的一组
pick() {
    local OUT="$1"
    shift
    python scripts/s9_dino_pick_best.py --runs "$@" --out "${RUNS_ROOT}/dino/${SWEEP}_${OUT}.json"
}

stage_done() {
    if [ "${STOP_AFTER}" -le "$1" ]; then
        echo "按 STOP_AFTER=${STOP_AFTER} 停在阶段 $1"
        JOB_STATE=STAGE_COMPLETED
        exit 0
    fi
}

# ---- 阶段 1：学习率 ----
LR_RUNS=()
for LR in ${LRS}; do
    LR_RUNS+=("${P}_lr${LR}")
    run_one "${P}_lr${LR}" "${LR}" --hparams "${BASE}"
done
if [ -n "${BEST_LR:-}" ]; then
    REF="${P}_lr${BEST_LR}"
else
    REF="$(pick lr "${LR_RUNS[@]}")"
    BEST_LR="${REF#"${P}_lr"}"
fi
echo "=== 阶段 1 结论：lr=${BEST_LR}（基准组 ${REF}）"
if [ -n "${MIN_BASE_AP:-}" ]; then
    BASE_AP="$(python -c "import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['metrics']['agn/AP'])" \
        "${RUNS_ROOT}/dino/${REF}/final_eval.json")"
    if awk "BEGIN { exit !(${BASE_AP} < ${MIN_BASE_AP}) }"; then
        echo "=== 停在阶段 1：基准组 agn/AP=${BASE_AP} 低于门槛 MIN_BASE_AP=${MIN_BASE_AP}，不往下跑。"
        echo "    确认要继续就去掉 MIN_BASE_AP（或调低）再提交。"
        JOB_STATE=GATE_STOPPED
        exit 3
    fi
    echo "=== 基准组 agn/AP=${BASE_AP}，达到门槛 MIN_BASE_AP=${MIN_BASE_AP}"
fi
stage_done 1

# ---- 阶段 2：变体，各自与同一个 lr 的基准组比 ----
run_one "${P}_noaux" "${BEST_LR}" --hparams "${BASE}" "${HP}/mods/noaux.yaml"
MODS=()
# 先赋值再比较：挑选失败（评测回执不合格）要让脚本停下，不能被当成「没被选中」
WIN="$(pick noaux "${REF}" "${P}_noaux")"
[ "${WIN}" = "${P}_noaux" ] && MODS+=("${HP}/mods/noaux.yaml")

# 任意错误都停止，避免把环境/评测/导出错误当作显存不足跳过。
HIRES_DIR="${RUNS_ROOT}/dino/${P}_hires"
if [ -f "${HIRES_DIR}/FAILED" ]; then
    echo "FAIL: 存在旧 FAILED 标记，请先检查旧日志、解决原因并归档标记后重试。" >&2
    exit 1
fi
if [ "${SKIP_HIRES:-0}" = "1" ]; then
    echo "=== ${P}_hires：按 SKIP_HIRES=1 跳过，不采用放大输入"
else
    run_one "${P}_hires" "${BEST_LR}" --hparams "${BASE}" "${HP}/mods/hires.yaml"
    WIN="$(pick hires "${REF}" "${P}_hires")"
    [ "${WIN}" = "${P}_hires" ] && MODS+=("${HP}/mods/hires.yaml")
fi
echo "=== 阶段 2 结论：采用的变体 = ${MODS[*]:-（无，保持基准）}"
stage_done 2

# ---- 阶段 3：长跑 ----
run_one "${P}_long" "${BEST_LR}" --hparams "${BASE}" ${MODS[@]+"${MODS[@]}"} "${HP}/mods/long.yaml"
stage_done 3

# ---- 阶段 4：衰减分叉 ----
DECAY_LR="$(awk "BEGIN { printf \"%g\", ${BEST_LR} * 0.1 }")"
# 最终候选 = 所有「降过学习率」的组：基准组、两个变体（都是 4 epoch）和各个分叉。
# 长跑本身全程恒定 lr，分数不可比，不参与。
FINAL_RUNS=("${REF}" "${P}_noaux")
[ "${SKIP_HIRES:-0}" = "1" ] || FINAL_RUNS+=("${P}_hires")
for E in ${DECAY_AT}; do
    MILESTONE="${RUNS_ROOT}/dino/${P}_long/milestone_iter_$((E * ITERS_PER_EPOCH)).pth"
    if [ ! -f "${MILESTONE}" ] && [ ! -f "${RUNS_ROOT}/dino/${P}_decay_e${E}/DONE" ]; then
        echo "FAIL: 找不到里程碑存档 ${MILESTONE}（DECAY_AT / ITERS_PER_EPOCH 设对了吗？）"
        exit 1
    fi
    FINAL_RUNS+=("${P}_decay_e${E}")
    run_one "${P}_decay_e${E}" "${DECAY_LR}" --hparams "${BASE}" ${MODS[@]+"${MODS[@]}"} "${HP}/mods/decay.yaml" \
        --init-from "${MILESTONE}"
done
FINAL="$(pick final "${FINAL_RUNS[@]}")"

echo "全部完成。lr=${BEST_LR}，变体=${MODS[*]:-无}，最终最好的一组：${FINAL}"
echo "各阶段的比较明细：${RUNS_ROOT}/dino/${SWEEP}_*.json"

# ---- 第二轮：补第一轮没探索到的地方（ROUND2 里列了哪几组就跑哪几组，按书写顺序）----
# 第一轮的教训是 4 个 epoch 时品牌分类还没学起来，所以除 long20 外都用 9 个 epoch 的计划（mods/sched9.yaml）。
# 这一轮只跑、只记录、不自动采用任何结果：每组各出一份 ${SWEEP}_r2_<组>.json 供人看。
S9=("${BASE}" ${MODS[@]+"${MODS[@]}"} "${HP}/mods/sched9.yaml")
for GROUP in ${ROUND2:-}; do
    echo "=== 第二轮 / ${GROUP}"
    case "${GROUP}" in
    long20)
        # 训练更久：长跑延到 20 epoch，再从第 16、20 个 epoch 分叉降 lr
        L20="${RUNS_ROOT}/dino/${P}_long20"
        if ! ls "${L20}"/*.pth >/dev/null 2>&1 && [ ! -f "${L20}/DONE" ]; then
            SEED_CKPT="${RUNS_ROOT}/dino/${P}_long/milestone_iter_$((12 * ITERS_PER_EPOCH)).pth"
            [ -f "${SEED_CKPT}" ] || { echo "FAIL: 找不到第一轮长跑第 12 个 epoch 的存档 ${SEED_CKPT}" >&2; exit 1; }
            mkdir -p "${L20}"
            cp "${SEED_CKPT}" "${L20}/iter_$((12 * ITERS_PER_EPOCH)).pth"
            # mmengine 续训只认 last_checkpoint 这个指针文件，不会自己去找目录里的 .pth；
            # 少了它会一声不吭地从头训（本机短测抓到过）
            echo "${L20}/iter_$((12 * ITERS_PER_EPOCH)).pth" > "${L20}/last_checkpoint"
            echo "已把 ${SEED_CKPT} 拷进 ${L20}，从第 12 个 epoch 续训"
        fi
        run_one "${P}_long20" "${BEST_LR}" --hparams "${BASE}" ${MODS[@]+"${MODS[@]}"} "${HP}/mods/long20.yaml"
        R2_RUNS=("${FINAL_RUNS[@]}")
        for E in 16 20; do
            R2_RUNS+=("${P}_decay_e${E}")
            run_one "${P}_decay_e${E}" "${DECAY_LR}" --hparams "${BASE}" ${MODS[@]+"${MODS[@]}"} "${HP}/mods/decay.yaml" \
                --init-from "${L20}/milestone_iter_$((E * ITERS_PER_EPOCH)).pth"
        done
        echo "=== 第二轮 / long20 最好：$(pick r2_long20 "${R2_RUNS[@]}")"
        ;;
    aux)
        # 两个辅助分支分别消融：都开 / 只 Prototype / 只 Hierarchy / 都关
        run_one "${P}_s9_base" "${BEST_LR}" --hparams "${S9[@]}"
        run_one "${P}_s9_proto" "${BEST_LR}" --hparams "${S9[@]}" "${HP}/mods/proto_only.yaml"
        run_one "${P}_s9_hier" "${BEST_LR}" --hparams "${S9[@]}" "${HP}/mods/hier_only.yaml"
        run_one "${P}_s9_noaux" "${BEST_LR}" --hparams "${S9[@]}" "${HP}/mods/noaux.yaml"
        echo "=== 第二轮 / aux 最好：$(pick r2_aux "${P}_s9_base" "${P}_s9_proto" "${P}_s9_hier" "${P}_s9_noaux")"
        ;;
    auxw)
        # 辅助 loss 的权重大小：0.1（基准）对 0.5
        run_one "${P}_s9_base" "${BEST_LR}" --hparams "${S9[@]}"
        run_one "${P}_s9_aux05" "${BEST_LR}" --hparams "${S9[@]}" "${HP}/mods/aux05.yaml"
        echo "=== 第二轮 / auxw 最好：$(pick r2_auxw "${P}_s9_base" "${P}_s9_aux05")"
        ;;
    wd)
        # weight decay：1e-4（基准）对 1e-3、1e-5
        run_one "${P}_s9_base" "${BEST_LR}" --hparams "${S9[@]}"
        run_one "${P}_s9_wd1e-3" "${BEST_LR}" --hparams "${S9[@]}" "${HP}/mods/wd1e-3.yaml"
        run_one "${P}_s9_wd1e-5" "${BEST_LR}" --hparams "${S9[@]}" "${HP}/mods/wd1e-5.yaml"
        echo "=== 第二轮 / wd 最好：$(pick r2_wd "${P}_s9_base" "${P}_s9_wd1e-3" "${P}_s9_wd1e-5")"
        ;;
    *)
        echo "FAIL: ROUND2 里有不认识的组 ${GROUP}（可用：long20 aux auxw wd）" >&2
        exit 1
        ;;
    esac
done
[ -z "${ROUND2:-}" ] || echo "第二轮完成：${ROUND2}"

if [ "${REHEARSAL}" = "1" ]; then
    echo "彩排通过。保留彩排存档供回执校验；彩排预测不能用于正式报告。"
fi
