# TC2 集群运行 DINO：操作步骤

编写日期：2026-10-05。适用于本项目当前的 Windows 本机 → TC2 Linux 集群 → Windows 本机出报告流程。

依据：同目录 `CCDSGPU-TC2-Masters-UserGuide.pdf`（2025-08-13 版）及当前 `scripts/cluster/`、`s9_dino_*.py`。
手册的页码在下文标明；项目路径和训练命令来自当前代码。实际账号配额以 `mytcinfo` 为准。

**先完成步骤 1–6，确认 sanity 和 smoke 都成功，再提交正式实验。本文中的集群命令在 SSH 终端执行，标明 PowerShell 的命令在自己的 Windows 电脑执行。**

## 1. 连接校园网，首次登录集群

手册第 3–5 页：校外先连接 NTU VPN（https://ntuvpn.ntu.edu.sg），再访问集群；必须先 SSH 登录一次，系统才会创建 home 目录。

在 Windows PowerShell 执行，把 `your_username` 换成自己的 NTU 网络账号，小写：

```powershell
ssh -p 22 xinkuo001@10.96.189.12
```

也可以使用 PuTTY，主机 `10.96.189.12`、端口 `22`、SSH。使用 NTU 网络账号密码，输入密码时终端不会显示字符。首次连接会提示确认服务器主机密钥。

登录后执行：

```bash
whoami
pwd
mytcinfo
squeue -u "$USER"
```

记录自己的 home 路径和 QoS。后文 `~` 就代表这个账号的 home，不需要照抄其他人的用户名。

## 2. 核对资源，建立目录

手册第 13、16–17 页给出的 `normal` 示例：

| 项目 | 手册示例上限 | 本项目脚本申请 |
| --- | --- | --- |
| Partition | MGPU-TC2 | MGPU-TC2 |
| QoS | normal，必须已分配给本人 | normal |
| GPU | 每用户合计 1 张 | 1 张 |
| CPU | 每用户合计 10 核 | 10 核 |
| 主机内存 | 每用户合计 30 GB | 28 GB |
| 单次作业时限 | 6 小时 | 6 小时 |
| 同时运行作业数 | 最多 2 个，仍受上述总资源约束 | 本流程一次运行一个 GPU 作业 |

28 GB 是系统内存，不是 GPU 显存。不要把“最多 2 个作业”理解成能同时申请两张 GPU。
如果 `mytcinfo` 的输出不同，先按自己的配额调整两个 `.sbatch.sh` 文件中的 `#SBATCH` 行，再提交。

如果 `mytcinfo` 不可用，可用手册第 13 页的命令查询：

```bash
sacctmgr show user "$USER" withassoc format=user,qos
sacctmgr -P show qos normal withassoc format=name,MaxTRESPU,MaxJobsPU,MaxWall
```

创建项目目录：

```bash
mkdir -p ~/logdet/project
mkdir -p ~/logdet/data
mkdir -p ~/logdet/runs/coco
mkdir -p ~/logdet/runs/handoff
mkdir -p ~/logdet/runs/cache/checkpoints
mkdir -p ~/logdet/runs/eval/gt
```

手册第 8 页说明 home 有磁盘配额（学生账号可能为 100 GB，按实际分配为准）；可用 `ncdu ~` 检查自己的目录，按 `q` 退出。预留环境、数据和多个训练 checkpoint 的空间，不要只按压缩包大小估算。

## 3. 上传代码、数据、标注和预训练权重（WinSCP 或 PowerShell 二选一）

手册第 5–7 页：打开 WinSCP，新建连接：

| 设置 | 值 |
| --- | --- |
| 文件协议 | SFTP |
| 主机 | 10.96.189.12 |
| 端口 | 22 |
| 用户名 | 小写 NTU 网络账号 |
| 密码 | NTU 网络账号密码 |

本机根目录：`E:\ntu\computer_vision\cvproject\logdet`。
远端定位到步骤 1 查到的 home，再进入 `logdet`。上传下面这些文件，保持目录层级：

| 本机相对于 logdet 的路径 | 集群目标 |
| --- | --- |
| `project\src\` | `~/logdet/project/src/` |
| `project\configs\` | `~/logdet/project/configs/` |
| `project\scripts\` | `~/logdet/project/scripts/` |
| `raw\LogoDet-3K.zip` | `~/logdet/data/LogoDet-3K.zip` |
| `runs\coco\instances_train_dino.json` | `~/logdet/runs/coco/instances_train_dino.json` |
| `runs\coco\instances_val_dino.json` | `~/logdet/runs/coco/instances_val_dino.json` |
| `runs\coco\fine_to_super_mapping.json` | `~/logdet/runs/coco/fine_to_super_mapping.json` |
| `runs\handoff\shared_image_ids.json` | `~/logdet/runs/handoff/shared_image_ids.json` |
| `runs\eval\gt\infer_union.json` | `~/logdet/runs/eval/gt/infer_union.json` |
| `runs\eval\gt\val2k_repr.json` | `~/logdet/runs/eval/gt/val2k_repr.json`（只有全量训练 `--mode full` 需要：每个 epoch 在这 2,000 张上评测；它和 `infer_union.json` 不能互相替代） |
| `runs\cache\checkpoints\dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth` | `~/logdet/runs/cache/checkpoints/` 下同名文件 |

**`infer_union.json` 必须上传。** 缺少它会自检失败；预测完成且校验通过之后才写 DONE。

不需要上传本机 `.venv`、Windows conda 环境、Windows wheel、旧训练目录或缓存。第一次实验使用新的远端 `runs/dino/`，避免把本机旧 DONE、选优结果和绝对路径一起带过去。

如果希望使用命令上传，可用下面的 PowerShell 命令代替 WinSCP。**在 Windows 本机新开一个 PowerShell 窗口执行，不要在集群 SSH 终端执行。** 把 `your_username` 改为自己的账号；需已完成首次 SSH 登录并连接校园网/VPN。每条 `scp` 成功后再执行下一条；上传失败时先处理错误，不要继续解压或提交训练。

```powershell
Set-Location 'E:\ntu\computer_vision\cvproject\logdet'
$tc2Target = 'your_username@10.96.189.12'

ssh -p 22 $tc2Target 'mkdir -p ~/logdet/project ~/logdet/data ~/logdet/runs/coco ~/logdet/runs/handoff ~/logdet/runs/eval/gt ~/logdet/runs/cache/checkpoints'

scp -P 22 -r .\project\src .\project\configs .\project\scripts "${tc2Target}:logdet/project/"
scp -P 22 .\raw\LogoDet-3K.zip "${tc2Target}:logdet/data/"
scp -P 22 .\runs\coco\instances_train_dino.json .\runs\coco\instances_val_dino.json .\runs\coco\fine_to_super_mapping.json "${tc2Target}:logdet/runs/coco/"
scp -P 22 .\runs\handoff\shared_image_ids.json "${tc2Target}:logdet/runs/handoff/"
scp -P 22 .\runs\eval\gt\infer_union.json "${tc2Target}:logdet/runs/eval/gt/"
scp -P 22 .\runs\cache\checkpoints\dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth "${tc2Target}:logdet/runs/cache/checkpoints/"
```

`scp` 的端口参数是大写 `-P`，`ssh` 是小写 `-p`。远端的 `logdet/...` 是相对于账号 home 的路径，对应 `~/logdet/...`。这里只上传项目所需目录，不上传 Windows 环境。代码更新时可以重传第一条 `scp`；数据和权重已上传完整则无需重复传。

**通过 WinSCP 或上述命令完成上传后**，再在集群 SSH 终端解压一次；下面的 `unzip` 和 `ls` 只负责解压、检查，不会上传文件：

```bash
cd ~/logdet/data
unzip -q LogoDet-3K.zip
ls -l LogoDet-3K/Clothes/2xist/1.jpg
cd ~/logdet/project
ls -l ../runs/coco/instances_train_dino.json
ls -l ../runs/coco/instances_val_dino.json
ls -l ../runs/coco/fine_to_super_mapping.json
ls -l ../runs/handoff/shared_image_ids.json
ls -l ../runs/eval/gt/infer_union.json
ls -lh ../runs/cache/checkpoints/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth
ls -l src/logodet/dino_run.py
```

本地已核对压缩包内部自带 `LogoDet-3K/` 顶层目录。正确结果是 `~/logdet/data/LogoDet-3K/Clothes/...`，不要多套一层同名目录。最后一行检查本次修复新增的辅助模块也已上传。

## 4. 在集群建立 Linux 环境（只需首次安装）

手册第 9–12 页：加载共享 Anaconda，在自己的环境中安装包；不要使用 sudo，也不要向只读 base 环境装包。

```bash
cd ~/logdet/project
module avail
bash scripts/cluster/tc2_setup_env.sh
```

该脚本会加载 `anaconda` 模块、创建/激活 `logodet_dino`，安装项目固定版本：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11 |
| PyTorch / torchvision | 2.1.0+cu121 / 0.16.0+cu121 |
| MMCV / MMDetection / MMEngine | 2.1.0 / 3.3.0 / 0.10.7 |
| NumPy | 1.26.4 |

**Windows 包不能直接用于 Linux，但脚本中的下载地址不是 Windows 专用地址。** pip 根据当前系统选择 Linux wheel；脚本检查 Linux x86_64 / Python 3.11，并要求 MMCV 使用预编译包，找不到匹配 wheel 就报错。

手册建议优先使用 conda；此项目为保持验证过的 PyTorch/MMCV 组合，在独立 conda 环境内使用 pip 固定版本，不照搬其他项目的安装命令。

结束时应看到版本信息和“环境 logodet_dino 就绪”。这是安装和导入检查，还不是 GPU 运行成功。
脚本在子 shell 中激活环境，不代表当前 SSH shell 已激活；后续 sbatch 脚本会自行激活，无需每次手动激活。

手册第 11 页的 `module load cuda/12.8.0` 是示例，本项目按 cu121 预编译环境配置，不需要为照抄示例而切换版本。实际驱动和 GPU 是否可用由下一步验证。

## 5. 提交单 batch 检查，先确认 GPU 能跑

手册第 2–3、13、18 页：登录节点没有计算 GPU，不在登录节点直接运行训练、预测、sanity 或 `nvidia-smi`。通过 `sbatch` 提交后，它们在分配到的计算节点执行。

```bash
cd ~/logdet/project
sbatch scripts/cluster/tc2_dino.sbatch.sh sanity --batch-size 1
```

使用 batch 1 与当前 shared 训练预设一致，也避免 sanity 默认 batch 2 带来额外显存需求。
提交成功会返回 `Submitted batch job 12345` 一类提示，记下真实 job ID。返回 ID 只说明提交成功，不代表已经运行成功。

把下文 `12345` 替换为刚拿到的 job ID：

```bash
squeue -u "$USER"
scontrol show jobid 39788
tail -f output_logodet_dino_12345.out
```

作业开始后日志才会出现。`Ctrl+C` 仅退出 `tail`，不会取消训练。
作业结束后检查：

```bash
tail -n 80 output_logodet_dino_12345.out
tail -n 80 error_logodet_dino_12345.err
sacct -j 12345 --format=JobID,JobName,State,ExitCode,Elapsed
```

通过标准：日志显示 `device=cuda`、GPU 型号和显存、各项检查通过，最后“全部通过”，作业为 `COMPLETED` / `ExitCode=0:0`。
若显示 `device=cpu`，即使检查跑完也不能视为 GPU 验证成功。

首次加载 COCO 权重时，3000 类分类头、新 neck 和辅助分支存在不匹配/未加载提示是预期现象；以实际 traceback、失败检查项和退出状态判断是否异常。

## 6. 提交 300 步 smoke，确认训练与评测链路

等 sanity 完成后再提交：

```bash
cd ~/logdet/project
sbatch scripts/cluster/tc2_dino.sbatch.sh smoke --batch-size 1 --max-iters 300
```

使用新 job ID 查看同格式日志。检查 loss 为有限值、没有 OOM/Traceback、结束时同时输出 `coco/*` 和 `agn/*` 指标，作业成功退出。
smoke 使用较小输入，不代表大尺寸变体一定放得下；它用于验证基本训练流程，不用于选择超参，也不保存续训 checkpoint。

## 7. 正式提交 shared 调参实验

当前代码与早期会话记录的“四组实验”不同，默认是四阶段、最多九组训练：

| 阶段 | 训练目录 | 内容 |
| --- | --- | --- |
| 1 | `shared_lr1e-4`、`shared_lr2e-4`、`shared_lr5e-5` | 三个学习率，每组 4 epoch |
| 2 | `shared_noaux`、`shared_hires` | 关闭辅助分支、放大输入，各 4 epoch |
| 3 | `shared_long` | 使用选出的学习率和变体，长跑 16 epoch |
| 4 | `shared_decay_e8`、`shared_decay_e12`、`shared_decay_e16` | 从长跑对应里程碑权重起步，低学习率再训 2 epoch |

阶段 4 目前只加载权重、重新初始化优化器；应解释为低学习率微调分叉，不能把结果直接视为保持 AdamW 状态的连续学习率衰减实验。

默认运行全流程：

```bash
cd ~/logdet/project
sbatch scripts/cluster/tc2_shared_sweep.sbatch.sh
```

如果这次只想先完成学习率和变体对比，使用下面这条替代上面的提交命令：

```bash
sbatch --export=ALL,STOP_AFTER=2 scripts/cluster/tc2_shared_sweep.sbatch.sh
```

两条命令二选一，不要同时提交。前两阶段完成后，再用默认命令会跳过已完成部分并继续后续阶段。
整个流程可能需要多个 6 小时作业；没有在 TC2 实测耗时，不应沿用早期四组实验的 14 小时估算。

默认种子 `20260917`。开始实验前可以修改 `configs/dino/hparams/shared5k.yaml`；同一组开始后续跑保持原配置、学习率、数据及输出目录不变。
扫描脚本通过环境变量控制阶段；不要假定给扫描脚本追加训练 CLI 参数会透传。

## 8. 看进度、识别状态、到时限后续跑

最新版本每分钟在 stderr 写 `[HEALTH]`：当前命令、训练/验证/预测阶段、最近进度、
距上次进度的秒数、GPU 利用率和显存（index、utilization.gpu、memory.used、memory.total，单位 %/MiB）。
GPU 统计是设备级，不能单凭利用率判断本进程正常；要同时看 iteration/batches/completed 是否推进。

```bash
cd ~/logdet/project
# 将 23456 换成自己的 job ID。
tail -f error_logodet_sweep_23456.err
cat ../runs/dino/health/job_23456.json
cat ../runs/dino/health/job_23456.txt
```

JSON 是最近一条命令的实时状态；TXT 是扫描作业退出时的总结，运行期间可能尚不存在。
子命令 SUCCEEDED 不代表整套实验完成，整套结束以 TXT 的 COMPLETED 为准。
默认 900 秒无进度变为 STALE_WARNING，日志会持续提示；不自动杀掉可能正在慢速评测或重放数据的任务。
可在提交时用 `--export=ALL,STALE_SECONDS=1800` 调整。若进程/节点被强制杀死，状态文件可能停留在 RUNNING，
必须同时核对时间戳和 `sacct`。这些提示写入集群日志，不会推送到手机或聊天。

异常 loss、非有限梯度和无效预测会停止作业。任意阶段错误（包括大尺寸 OOM）都停止，不再自动跳过。
扫描默认在墙钟到点前 5 分钟自动续交（最多 12 次，可用 AUTO_RESUBMIT=0 关闭）：
TXT 的 RESUBMITTED 表示已经提交下一作业，RESUBMIT_FAILED 表示续交失败，NEEDS_RESUME 表示需手动续跑。
续交失败返回非零，STAGE_COMPLETED 只表示 STOP_AFTER 要求的阶段完成。

把 `23456` 换成扫描作业的真实 ID：

```bash
cd ~/logdet/project
squeue -u "$USER"
tail -f output_logodet_sweep_23456.out
```

查看错误、指标和已完成组：

```bash
tail -n 80 error_logodet_sweep_23456.err
grep 'agn/AP' output_logodet_sweep_23456.out
find ../runs/dino -name DONE -print
find ../runs/dino -name final_eval.json -print
```

作业不在 `squeue` 中时，不要直接判断失败。查询历史：

```bash
sacct -j 23456 --format=JobID,JobName,State,ExitCode,Elapsed
seff 23456
myjobhistory
```

| 状态/现象 | 下一步 |
| --- | --- |
| PD / Priority / Resources | 排队中；查 `scontrol`，不要重复提交同一组 |
| PD / QOSMax… | 核对个人总资源和并发作业；按手册第 19–20 页处理，必要时取消不合配额的作业后修改再提交 |
| R | 正在运行，继续看日志 |
| TIMEOUT | 先查是否已有自动续交作业；没有时，确认旧作业结束再手动提交 |
| FAILED / OUT_OF_MEMORY | 先看 stderr；不能一律当成正常超时 |
| COMPLETED | 当前脚本成功退出；再看日志是 STOP_AFTER 阶段完成，还是整个扫描“全部完成” |

**需要手动续跑时，先确认队列中没有本流程的自动续交作业，再原样提交：**

```bash
cd ~/logdet/project
sbatch scripts/cluster/tc2_shared_sweep.sbatch.sh
```

若首次使用 `STOP_AFTER=2`，续跑也使用相同的 `--export=ALL,STOP_AFTER=2` 命令。
正常续跑行为：

1. 已有 DONE 的组会再次校验最终评测和预测包，校验成功才跳过。
2. 未完成的组从最近 checkpoint 继续；默认每 2,000 iter 保存，未保存部分需要重算。
3. 如果最后 checkpoint 已保存但最终评测没完成，会补评测后再选优。
4. 最终评测步数、checkpoint 步数及计划总步数一致后才用于自动选择。
5. 已有且一致的选优结果沿用；旧选择不匹配会报错，不能通过随便删文件掩盖设置变更。

如果在生成预测时超时，续跑可能重新生成该组预测；当前预测导出不按图像增量续跑。
如存在旧版本留下的 `shared_hires/FAILED`，新脚本会停止并要求检查原因；解决后归档该标记再续跑。

如需主动停止一个作业，仅取消自己的对应 ID：

```bash
scancel 23456
```

手册第 18 页说明 sbatch 提交后可断开 SSH，计算仍由 SLURM 执行；不需要让电脑一直开着，也不需要改用 `srun`。

## 9. 下载结果，在 Windows 本机出完整报告

使用 WinSCP 下载每个已完成组的整个目录：

```text
集群：~/logdet/runs/predictions/dino/<组名>/
本机：E:\ntu\computer_vision\cvproject\logdet\runs\predictions\dino\<组名>\
```

每组至少应有：

```text
predictions.npz
manifest.json
own_eval_ann.json
```

另外下载保存 `runs/dino/sweep_*.json`、各组 `final_eval.json`、训练配置、日志及需要保留的 checkpoint。
这些用于复核和归档，不必为了出报告下载所有训练 checkpoint。

在本机 PowerShell 中执行，以 `shared_lr1e-4` 为例：

```powershell
Set-Location 'E:\ntu\computer_vision\cvproject\logdet\project'
$env:LOGDET_ROOT = 'E:\ntu\computer_vision\cvproject\logdet'
& '..\.venv\Scripts\python.exe' scripts\s9_dino_report.py --run-name shared_lr1e-4
```

输出：`project\report\s9_dino_shared_lr1e-4_report.md`。
其他组替换 `--run-name` 即可。使用本机 baseline `.venv`，因为报告还要读 parquet。
本机需要保留原有的 `runs/metrics/s7_overall.parquet`、`s7_slices.parquet`、切片 tables/splits、
`runs/eval/gt/val2k_repr.json`、`val_hard_pool.json` 和 `runs/coco/instances_val_dino.json`。

新 manifest 的评测标注路径相对于预测目录，不依赖 Linux home 路径。报告要求 schema_version=2 的已验证预测包，
检查 checkpoint 评测信息、文件 SHA-256、预测数值及完整图片覆盖率；彩排和部分预测会拒绝出正式报告。
旧版预测包需要用新脚本重新导出（其训练目录须先补齐 final_eval.json），不能手工加字段冒充已验证数据。

## 10. 常见问题

| 问题 | 检查/处理 |
| --- | --- |
| SSH 连接超时 | 校园网或 NTU VPN、IP 和端口是否正确 |
| `module: command not found` | 确认在集群 SSH 登录环境；不要在本机 PowerShell 执行 Linux 安装脚本 |
| `conda activate` 报错 | 项目脚本已含 `module load anaconda` 和 `eval "$(conda shell.bash hook)"`；确认上传的是当前脚本 |
| pip 找不到 MMCV wheel | 检查 Linux x86_64、Python 3.11、网络和固定版本；不要拿 win_amd64 包替代或直接升级整套依赖 |
| `libGL.so.1` 等系统库缺失 | 保存完整报错，检查集群可用模块或联系支持；不使用 sudo 修改共享系统 |
| 文件找不到 | 检查上传表，尤其模型完整文件名、3 个 COCO JSON 和解压后的目录层级 |
| `bad interpreter` / DOS line breaks | 将 `.sh` 保存为 UTF-8、LF 换行再上传；当前仓库脚本已是 LF |
| sanity OOM | 确认用了 `--batch-size 1`，查看计算节点 GPU 实际显存 |
| 写 checkpoint 失败/空间不足 | `ncdu ~` 检查配额，先备份再整理自己的产物，保留继续训练及阶段 4 所需存档 |
| GPU/驱动不兼容 | 保存 sanity 的 stdout/stderr 和节点信息，交给集群支持确认环境；不要以登录节点查不到 GPU 作为证据 |

手册第 26–27 页：额外 QoS 或存储需要向 `ccdsgpu-tc@ntu.edu.sg` 申请并提供实际运行证据；集群不提供用户数据备份，应及时下载产物。配额未获批准前不要自行改成未授权 QoS。

## 最短执行清单

上传表内文件、解压、确认配额后，按顺序执行，每项成功后再进入下一项：

```bash
cd ~/logdet/project
bash scripts/cluster/tc2_setup_env.sh
sbatch scripts/cluster/tc2_dino.sbatch.sh sanity --batch-size 1
# 等 sanity 成功。
sbatch scripts/cluster/tc2_dino.sbatch.sh smoke --batch-size 1 --max-iters 300
# 等 smoke 成功。
sbatch scripts/cluster/tc2_shared_sweep.sbatch.sh
# 无自动续交作业且旧作业已结束：重复上一条 sbatch 命令。
```

本文是基于手册与当前代码整理的操作指南，尚未在 TC2 实机执行。
