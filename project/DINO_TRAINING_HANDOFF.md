# DINO 训练准备 —— 会话交接记录

> 这不是前三份正式文档（README / FILES_AND_REPRODUCTION / BASELINE_EVALUATION）
> 那种长期维护的文档，是**换机器续聊用的工作记事**，把这次对话里"做了什么决定、
> 卡在哪、还没做"记下来。换新会话时把这份文件读一遍就能接上。

## 目标

用 MMDetection 的 DINO 在 LogoDet-3K 的 **3000 个细类logo** 上训练检测器，
严格基于已经冻结的 trainval/val 划分（不能重新划分）。9个superclass作为
辅助hierarchy标签（暂缓实现，先跑通基础3000类训练循环再加）。

## 已经做完、可以直接用的部分

### 1. Windows + CUDA 适配（整个 S0-S7b 原pipeline）

原项目在 macOS (Apple M4 Pro) 上写的，这次完整迁移到Windows+RTX 4070，
**63条门全部复现，几乎每个最终指标都跟Mac上的结果精确一致**（sha256级别）。
改了这几处真实bug（都已验证不影响任何已有逻辑，133条pytest全过）：

- `scripts/s0_setup_env.sh`：venv目录布局按OS区分（`bin/` vs `Scripts/`）
- `src/logodet/ingest/inventory.py`：路径分隔符统一用`.as_posix()`（Windows的`\`
  跟zip内部的`/`对不上，之前导致158,654张图被整体跳过）
- `src/logodet/ingest/unzip_safe.py`：去掉对macOS专属`ditto`的硬依赖，加
  `zipfile.extractall()`兜底
- `scripts/s0_verify_env.py`：torch/torchvision版本比对改为只比PEP440公开
  版本号，不被平台构建标签（`+cu124`）误判成版本不一致
- `src/logodet/baselines/torchvision_detector.py`：修复`roi_heads`在纯CPU/
  纯CUDA模式下从未被显式挪动设备的潜藏bug；加CUDA支持；**禁用TF32**
  （Ada架构GPU默认开TF32会让CUDA vs CPU一致性验证失败，box差0.14px，
  关掉后降到0.0005px）
- `src/logodet/baselines/owlv2_detector.py` + `scripts/s7b_probe_owlv2.py`：
  同样加CUDA支持+禁用TF32+补上设备一致性验证（原来直接信速度，不验证数值）

**这些改动目前只在本地，还没commit/push。**

### 2. DINO训练用的3000类COCO标注（已导出+交叉验证通过）

脚本：`scripts/s8_export_dino_coco.py`（新文件，未提交）

产物在 `runs/coco/`（这个目录被`.gitignore`排除，不会进git，**到新机器要
重新跑脚本生成**，前提是数据已经下载解压好）：

```
runs/coco/instances_train_dino.json   141,438图 / 173,088框 / 3000类 / 32.0MB
runs/coco/instances_val_dino.json      17,216图 /  21,174框 / 3000类 /  4.1MB
runs/coco/fine_to_super_mapping.json  class_id(1-3000) -> superclass_id(1-9)
runs/coco/manifest_dino.json          源表sha256留痕
```

已验证：train/val图片id无交集、并集=158,654全量、category_id两侧完全一致
(1..3000)、框数总和=194,262与annotations.parquet一致、每张图的split归属
与`runs/splits/split_images.parquet`逐一核对一致（无泄漏）。

**换机器复现这一步**：确认`data/LogoDet-3K`已解压（跑S1），`runs/tables/`
三张表和`runs/splits/split_images.parquet`已生成（跑S2+S3），然后：
```bash
cd project
python scripts/s8_export_dino_coco.py
```

### 3. 选定的DINO配置

MMDetection自带的 `configs/dino/dino-4scale_r50_improved_8xb2-12e_coco.py`
（R50 4scale 改进版，COCO box AP 50.1，是R50系列里最好的）。

权重下载地址：
```
https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth
```

用户明确要求：**只用这个MMDetection原生checkpoint**，不混用IDEA-Research
官方仓库的checkpoint（除非确认参数格式完全兼容）。

## 卡住的地方：mmcv 在 Windows 上没有预编译包

`mmdetection` 要求 `mmcv>=2.0.0rc4,<2.2.0`。实测探测了openmmlab官方wheel索引：

- cu118/cu121：有预编译包，但**全部是 `manylinux1_x86_64`（纯Linux）**，
  torch版本只到2.4.0为止（2.5.x完全没有）
- cu124：索引页都不存在

即：**mmcv从没发过Windows的预编译wheel**，Windows上装mmcv完整版（带CUDA
算子，DINO的可变形注意力依赖这个）只能从源码编译，需要本机装
Visual Studio Build Tools（C++工具链）+ CUDA Toolkit（跟显卡驱动是两码事）。
这台机器上**两样都没装**（`nvcc`、`cl`均找不到）。

### 已经讨论过的三条路线，用户还没拍板

| 路线 | 做法 | 风险 |
| --- | --- | --- |
| A. Windows原生编译 | 装VS Build Tools+CUDA Toolkit，源码编译mmcv | 下载量大（5-10GB），版本互相打架是这个生态最常见的报错源 |
| B. WSL2 | 开WSL2，Ubuntu里直接装Linux预编译wheel，不编译 | 更稳；但WSL读`/mnt/e/...`上31.7万个小文件可能很慢（跟我们S1/S2阶段踩过的"文件守卫开销"是同一类问题，没实测过） |
| C. 学校服务器 | 把图片+COCO标注（约4GB）传上去，大概率Linux，直接装 | 最干净；但用户还没确认服务器访问权限/能否自己装环境 |

用户最后一条消息是"先不要做，先描述下这个问题"——**没有选定任何一条路线**，
等用户确认服务器情况或决定要不要在本机啃A/B路线。

## 环境清单（这台机器上，供下次对照）

| 环境 | 路径 | 用途 | 关键版本 |
| --- | --- | --- | --- |
| 原pipeline用venv | `E:\ntu\computer_vision\cvproject\logdet\.venv` | S0-S7b全流程baseline | torch 2.5.1+cu124，numpy 1.26.4。**跑Python脚本必须设`PYTHONUTF8=1`**（否则Windows控制台GBK编码会导致中文输出乱码、特殊符号直接崩溃） |
| DINO训练用conda env | `logodet_dino`（`D:\condamini\envs\logodet_dino`） | 给MMDetection/DINO用 | torch 2.4.0+cu121，numpy 1.26.4（mmengine会把numpy拉回2.x，装完后要重新pin回1.26.4）。mmengine已装，mmcv卡住 |
| 纯解释器env | `logodet_base`（`D:\condamini\envs\logodet_base`） | 只是给venv当基础解释器用 | python 3.11.16，没装任何包 |

`LOGDET_ROOT` 环境变量设为 `E:/ntu/computer_vision/cvproject/logdet`
（项目根目录，`paths.yaml`靠这个找数据/产物位置）。

## 另一条暂未展开的支线：假logo/仿冒检测

中途讨论过"检测假logo"这个需求，澄清是**品牌保护/打假**场景。结论：
LogoDet-3K没有任何"真假"标签，直接训练不可行。讨论了两条技术路线：

1. **参照比对/异常检测**（推荐起点）：拿每个品牌的标准logo当参照，检测出来
   的logo跟参照比对，差异大就标可疑——不需要假货样本
2. **有监督分类**：需要真实的"真/假logo"标注数据，搜到几个候选公开数据集：
   - [Kaggle: Fake/Real Logo Detection Dataset](https://www.kaggle.com/datasets/prosperchuks/fakereal-logo-detection-dataset)（规模小，70x70缩略图，教学级别）
   - [Roboflow: Fake Logo Detection](https://universe.roboflow.com/dataset-0xg35/fake-logo-detection-nx1d0)（275张，带预训练模型）

**这条支线完全没有开始任何代码工作**，只是记录了讨论结论，跟DINO训练这条
主线是独立的，谁先做都不影响另一个。

## 下一步（任何一个都可以先捡起来）

1. 决定mmcv走A/B/C哪条路线，然后继续搭DINO训练环境
2. 先把本地这些真实bug修复commit+push到GitHub（`git status`能看到完整列表）
3. 如果要探索打假方向，先去看一眼Kaggle那个数据集具体长什么样，评估能不能用
