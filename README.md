# LogoDet-3K 上的 DINO 检测器

在 LogoDet-3K（158,654 张图、3,000 个品牌）上训练一个检测器：框出 logo，并认出是哪个品牌。
这页告诉你**结果在哪、先看什么**。接手这部分工作的同学从这里开始。

## 最终结果

用全部 141,438 张训练图训 4 遍，在模型没见过的 2,000 张验证图（val2k_repr）上：

| 模型 | 找 logo 的 AP | AP50 | 小目标 AP | AR@100 | 认品牌的 AP |
| --- | --- | --- | --- | --- | --- |
| **本项目的 DINO，全量训练** | **0.6439** | **0.9110** | **0.4862** | **0.8331** | **0.690** |
| 本项目的 DINO，只用 5,000 张 | 0.3745 | 0.6149 | 0.1142 | 0.7728 | 0.156 |
| 基线 OWLv2（零样本，不训练） | 0.1452 | 0.3061 | 0.0137 | 0.6043 | 不认品牌 |
| 基线 COCO 类塌缩（不训练） | 0.0055 | 0.0142 | 0.0001 | 0.1860 | 不认品牌 |

- "找 logo"只看框没框到，不管品牌，这样才能和不认品牌的基线比。"认品牌"要求框对且品牌也对。
- AP 是综合成绩；AP50 是宽松标准（大致框住就算对）；AR@100 是每张图给 100 个框的机会时能找回多少。都是 0 到 1，越高越好。
- 12 个难例切片（小目标、截断、密集等）上全部大幅超过 OWLv2。

## 结果在哪

**先看这三份**：

| 想知道 | 看哪个文件 |
| --- | --- |
| 最终模型的全部成绩，与基线并排 | [`project/report/s9_dino_full_report.md`](project/report/s9_dino_full_report.md) |
| 最终模型错在哪（逐个框的归类） | [`project/report/s9_dino_full_box_analysis.md`](project/report/s9_dino_full_box_analysis.md) |
| 整个工作做了什么、为什么这么做、还剩什么 | [`project/DINO_HANDOFF_2026-10-06.md`](project/DINO_HANDOFF_2026-10-06.md) |

**全部结果的位置**：

| 内容 | 位置 | 说明 |
| --- | --- | --- |
| 每组实验的对比报告 | `project/report/s9_dino_<组名>_report.md` | 10 份。`full` 是最终模型；`shared_*` 是 5,000 张上的调参实验 |
| 框级分析报告 | `project/report/s9_dino_<组名>_box_analysis.md` | `full` 和 `shared_b4_decay_e12`（5,000 张最好的一组）各一份 |
| 框级分析的例图 | `runs/analysis/box_<组名>/` | 每类错误的例子拼图，绿框是标注，红框是模型的预测。**原因需要人看图判断，这一步还没人做** |
| 指标表（程序读的） | `runs/metrics/` | `s7_*` 是基线，`s9_dino_*` 是各组实验，`s9_box_*` 是每个框的归类明细 |
| 最终模型的原始检测框 | `runs/predictions/dino/full/` | `predictions.npz`；`manifest.json` 里记着训练条件和模型存档的标识 |
| 5,000 张最好那组的原始检测框 | `runs/predictions/dino/shared_b4_decay_e12/` | 同上 |
| 基线的原始检测框 | `runs/predictions/L1_*.parquet`、`L2_*`、`L3_*` | 三个基线各一份 |
| 标注、数据划分、评测用的标准答案 | `runs/tables/`、`runs/splits/`、`runs/eval/gt/` | 所有模型共用同一套 `image_id` 和切片，数字才能互相对上 |
| 基线的评测方法和解读 | [`project/BASELINE_EVALUATION.md`](project/BASELINE_EVALUATION.md) | |

`.parquet` 和 `.npz` 是数据文件，网页上打不开，要用 Python 读（`pandas.read_parquet`、`numpy.load`）。
里面的数字都已经排进了对应的 `.md` 报告，只想看结果的话读报告就够了。

## 各组实验的名字

| 组名 | 是什么 |
| --- | --- |
| `full` | **最终模型**：全量 141,438 张，每步 4 张图，4 遍 |
| `shared_b4_decay_e12` | 5,000 张上最好的一组：先训 12 遍，降学习率再训 1 遍 |
| `shared_b4_decay_e8`、`shared_b4_long` | 同一系列：训 9 遍；训 12 遍不降学习率 |
| `shared_b4_lr1e-4`、`shared_b4_lr2e-4` | 比较两个学习率（每步 4 张图，4 遍） |
| `shared_b4_noaux` | 关掉两个辅助分支的对照 |
| `shared_lr1e-4`、`shared_lr2e-4`、`shared` | 更早的实验，每步 1 张图 |

## 其他文档

| 文档 | 内容 |
| --- | --- |
| [`project/DINO_HANDOFF_2026-10-06.md`](project/DINO_HANDOFF_2026-10-06.md) | 交接文档：任务、模型设计、实验结论、走过的弯路、怎么运行、还没做的事 |
| [`project/DINO_SESSION_2026-10-05.md`](project/DINO_SESSION_2026-10-05.md) | 按时间的工作记录。第 13 节是 5,000 张调参的结论，第 16 节是全量训练的结果 |
| [`project/TC2_DINO_RUN_GUIDE.md`](project/TC2_DINO_RUN_GUIDE.md) | 在学校 TC2 集群上从登录到出报告的逐步操作 |
| [`project/README.md`](project/README.md) | 更早的部分：数据处理、划分、基线评测体系 |

## 不在这个仓库里的东西

| 内容 | 在哪 | 原因 |
| --- | --- | --- |
| 模型权重 | TC2 集群 `~/logdet/runs/dino/full/milestone_iter_141440.pth`（约 726 MB） | 超过代码托管平台的单文件上限。集群不提供备份，需要的话找项目负责人要 |
| 数据集图片 | 从 Kaggle 下载 LogoDet-3K，步骤见 `project/README.md` | 约 3.9 GB |
| 其他 8 组调参实验的原始检测框 | 项目负责人本机 `runs/predictions/dino/` | 每组约 22 MB，按需索取 |

## 重新生成报告

不需要 GPU，也不需要模型权重，有这个仓库里的文件就够：

```
cd project
python scripts/s9_dino_report.py --run-name full          # 与基线并排的全套表格
python scripts/s9_dino_box_analysis.py --run-name full    # 框级分析和例图
```

第二条要画例图，需要本机有数据集图片；只要统计表的话，读 `runs/metrics/s9_box_full_*.parquet` 即可。
环境见 `project/environment/`。

## 已知的局限

- 所有结论只跑了一个随机种子，没有重复实验。
- 全量训练每遍结束在 val2k_repr 上评测，最终成绩也报在这批图上；训练计划是事先定的，没有根据评测结果调整过。
- 框级分析里"错误的原因"还没有人工复核，报告里的原因说明是初步印象。
- 辅助分支和 P2 融合这两个设计点的贡献，目前只有 5,000 张上一组不充分的对照，见交接文档第 6、11 节。
- 不能和 LogoDet-3K 原论文的数字直接比较：论文没有公开数据划分，这里用的是自己的划分。
