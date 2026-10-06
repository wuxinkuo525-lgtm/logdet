# DINO 检查修复（2026-10-05）

本次只修复跨平台报告、最终评测恢复和训练种子；衰减分叉的优化器行为未改。

- 新预测目录包含 `own_eval_ann.json`，manifest 使用相对路径。下载整个
  `runs/predictions/dino/<组名>/` 即可在本机读取该组的评测标注。
  报告仍需要本机原有的 baseline 指标、切片表和 val 标注。
  旧 manifest 的本机绝对路径仍可用；旧集群预测需要重新导出，不能仅移动旧 manifest 修复。
- shared/full 训练新增 `FinalEvaluationHook`。最后一个 checkpoint 保存后若评测被中断，
  原命令加 `--resume` 会补评测；正常结束复用当次最终评测，不重复计算。
  成功后原子写入 `final_eval.json`，包含评测步数、checkpoint 元数据中的迭代数、计划总步数、
  checkpoint 文件标识和指标。选优只使用步数一致且对应当前 checkpoint 的结果。
  旧 DONE 没有最终评测文件时会重新进入训练入口补评测。
  旧选优 JSON 若不匹配新验证结果会报错，需改用新的输出路径并检查依赖旧结论的实验。
- 默认训练种子为 `20260917`，预设 `seed` 或命令行 `--seed` 可覆盖（包括 0）。
  配置、checkpoint 和新预测 manifest 记录种子，报告显示种子。
  这控制初始化、采样和增强，不保证跨 GPU/平台逐位相同。
  恢复旧 checkpoint 时 MMEngine 沿用 checkpoint 中的旧种子，不会重新初始化模型。

## Linux 集群环境

不要上传 Windows conda 环境、`.venv` 或 `win_amd64.whl` 作为 Linux 环境。
上传代码、数据和 `.pth` 权重，在集群重新执行：

```bash
cd ~/logdet/project
bash scripts/cluster/tc2_setup_env.sh
sbatch scripts/cluster/tc2_dino.sbatch.sh sanity
```

安装脚本检查 Linux x86_64 / Python 3.11，并要求 MMCV 使用预编译 wheel。
官方目录明确提供 `mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl`：
https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

PyTorch 2.1.0 的 CUDA 12.1 安装地址支持 Linux 和 Windows，pip 根据平台选包：
https://pytorch.org/get-started/previous-versions/

安装脚本的 import 检查不等于 GPU 验证。TC2 的驱动、GPU 架构、显存和计算节点系统库仍需
通过上述 sanity 作业确认；本次未登录或安装 TC2 环境。
