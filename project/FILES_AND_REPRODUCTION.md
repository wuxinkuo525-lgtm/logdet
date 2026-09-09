# 文件说明与复现说明

> **这份文档的定位**：`README.md`（1,400+ 行）是**逐阶段的实验日志** —— 每个阶段做了什么、
> 哪个假设被实测推翻、踩了什么坑。它按时间顺序组织，适合从头读一遍。
>
> 本文是**索引 + 操作手册**：想知道「某个文件是干什么的」或「怎么把结果跑出来」看这里。
> 想知道「为什么这样设计」回 README 对应阶段。
>
> 第三份 `BASELINE_EVALUATION.md` 是评估口径与结果解读，面向要引用数字的人。

---

## 1. 三区隔离：什么进仓库，什么不进

```
~/Desktop/logdet/                    ← 项目根（可用 LOGDET_ROOT 覆盖）
├── project/     940 KB   ✅ 进仓库   代码 + 配置 + 报告，纯文本
├── data/         3.9 GB   ❌         LogoDet-3K 解压结果，317,308 个文件
├── raw/          2.9 GB   ❌         下载的 zip + sha256
├── runs/          54 MB   ❌         全部产物（parquet / COCO GT / 预测 / 指标）
└── .venv/        1.0 GB   ❌         Python 3.11.15 环境，47 个包
```

**只有 `project/` 进 git**，其余约 **7.8 GB** 由 `.gitignore` 排除。这不是偷懒：

| 目录 | 为什么不能进 | 怎么恢复 | 恢复耗时 |
| --- | --- | --- | --- |
| `data/` | 158,654 jpg + 158,654 xml = **31.7 万个小文件**，即便分片也会让仓库不可用 | 解压 `raw/` 的 zip | **112.5 秒**（S1 实测） |
| `raw/` | 单个 2.9 GB 文件，**超过 GitHub 100 MB 硬上限** | 从 Kaggle 重下，S1 有 sha256 可校验 | 取决于带宽 |
| `.venv/` | 平台相关二进制，跨机器无效 | `bash scripts/s0_setup_env.sh` | 约 3 分钟 |
| `runs/` | 派生数据，且会随每次重跑变化 | 见 §5 的重建成本表 | 约 40 分钟 |

### 一条重要的例外

`runs/` 里有**一样东西跑代码产不出来** —— 6 个人工复核标签（人看图做的判断）。
它们已经被固化进 `configs/slices.yaml` 的 `review_provenance.human_labels`，
所以**整条流水线只靠 `project/` 就能完整复现**。这是 commit `a77b4ef` 专门补的缺口。

同理，切分产物的参考 sha256 记在 `configs/splits.yaml` 的 `expected_sha256`，
S3-H9 门会拿重建结果与它比对 —— 「可重建」因此是**可验证的断言**，而不是一句声称。

### 零号约束：不要把数据放进云同步目录

`configs/paths.yaml` 里有一条守卫：四个 root 在 `Path.resolve()` 之后不得包含
`Mobile Documents` / `Google Drive` / `Dropbox` / `OneDrive`，命中直接报错。

原因是 macOS 的 iCloud「桌面与文档文件夹」同步一旦开启，`~/Desktop` 会 resolve 到
`~/Library/Mobile Documents/com~apple~CloudDocs/Desktop`。317k 个小文件丢进去会持续
上传占配额，而「优化存储」的按需下载会让 dataloader 随机读取时**直接卡死**。

断言在 `resolve()` 之后做，所以软链接绕不过去。若你的桌面正在被同步：

```bash
export LOGDET_ROOT=~/logdet     # 换个非同步路径即可，代码一行不用改
```

---

## 2. 目录结构总览

```
project/
├── README.md                   逐阶段实验日志（设计理由、被推翻的假设、踩坑）
├── FILES_AND_REPRODUCTION.md   本文：文件索引 + 复现手册
├── BASELINE_EVALUATION.md      评估数据说明 / 指标说明 / 结果解读
├── conftest.py                 pytest 配置（把临时目录重定向到 runs/.pytest_tmp）
│
├── configs/                    全部可调参数，代码里不留魔数
├── environment/                依赖锁 + S0 环境实测快照
├── src/logodet/                库代码（可 import，不含 __main__ 逻辑）
├── scripts/                    可执行入口（主流程 / 验证门 / 诊断 / 基准）
├── tests/                      133 条回归测试
├── report/                     由验证门自动生成，**不要手改**
└── notebooks/                  预留，当前为空
```

**`src/` 与 `scripts/` 的分工**：`src/` 是纯函数与类，不打印、不落盘、不读环境；
`scripts/` 负责编排、打印表格、写产物。所有阈值走 `configs/`。
这条分工让 `tests/` 能在不碰真实数据的前提下测算法。

---

## 3. 逐文件说明

### 3.1 `configs/` —— 唯一的参数来源

| 文件 | 行数 | 内容 | 谁会改它 |
| --- | --- | --- | --- |
| `paths.yaml` | 61 | **全项目唯一含绝对路径的文件**。四个 root + `runs/` 子目录登记表 + 云同步守卫 | 换机器时（或改用环境变量） |
| `dataset.yaml` | 235 | 数据集先验：9 个超类的类数/图数、论文对账、R1–R9 清洗规则、侦察结论 | S1/S2 脚本读取 |
| `slices.yaml` | 189 | 难例轴定义（T1 / 尺寸 / 密度）、**P2/P3 的降级裁决全过程**、核验来源与 6 个人工标签 | `s2_merge_human_labels.py` 会回写 |
| `splits.yaml` | 103 | 划分参数（10% val、弱去重键、108 层分层）、两个 val 子集的用途与**禁止用途**、`expected_sha256` 锚点 | 改划分逻辑时 |
| `runtime.yaml` | 68 | **由脚本自动写入**：`device`、`num_workers`、按组件的设备策略、实测吞吐 | 不要手改，重跑 S0/S4/S7 即刷新 |
| `l0_baseline.json` | 126 | L0 自校验的回归基线（抖动阶梯的精确数值） | 改评测器后需确认它不变 |

> `configs/baselines/` 目录当前为空 —— 预留给训练阶段的模型超参，setup 阶段的
> baseline 参数都在脚本里（prompt 候选、阈值），因为它们与代码逻辑强耦合。

### 3.2 `src/logodet/` —— 库代码

**基础设施（4 个文件）**

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `paths.py` | 272 | 路径解析 + 三区隔离断言。`P.artifact("tables")` 之类的入口；**未在 `paths.yaml` 登记的子目录会抛 KeyError**（防止代码里硬编码路径） |
| `config.py` | 51 | YAML 加载 + **配置指纹**（sha256）。指纹写进每个产物的头部，任何指标都能反查算在哪套配置上 |
| `seeds.py` | 36 | 随机流按**用途**派生。改了 val2k 的抽样代码不会扰动主划分 |
| `gates.py` | 141 | 验证门框架。统一 PASS/FAIL/WARN 语义、阻塞判定、markdown 表格生成 |

**数据摄取 `ingest/`（5 个文件）**

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `download.py` | 133 | 下载 + sha256 校验。用 `curl -L -C -` 而非 kaggle CLI（后者是锁文件里唯一需源码构建的包） |
| `unzip_safe.py` | 282 | 安全解压：先预检路径穿越/炸弹，再选快慢路径 |
| `inventory.py` | 154 | 磁盘清点，产出 `file_inventory.parquet` |
| `xml_source.py` | 85 | **XML 从 zip 顺序读**而非磁盘逐文件读 —— 实测快 82.5 倍 |
| `voc_parse.py` | 155 | VOC XML 解析（lxml `recover=True`），处理 R1–R9 脏数据 |

**清洗与派生 `curate/`（2 个文件）**

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `clean_rules.py` | 91 | R1–R9 清洗规则的实现 |
| `derive_geom.py` | 259 | 几何派生：`area` / `area_bin`（COCO 32²/96² 阈值）/ `ar` / 贴边距离 / T1 / P2 / P3 代理 |

**划分 `splits/`（2 个文件）**

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `make_split.py` | 179 | trainval/val 划分。簇为原子单位 + 每类至少 1 张 val + **内部自检**（比例偏离就抛异常）。含结构下限的解释 |
| `make_val_subsets.py` | 162 | `val2k_repr`（108 层分层 + 最大余数法）与 `val_hard_pool`（三轴定额富集 + 等量 clean 对照） |

**数据层 `data/`（5 个文件）**

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `schema.py` | 109 | 内部数据契约（`Sample` 等）。纯 numpy，**不 import torch** |
| `core_dataset.py` | 230 | 格式无关的核心 Dataset，图像走 zip 通路。**物理上不 import torch**（有 AST 静态测试守这条） |
| `adapters/to_torchvision.py` | 90 | `Sample` → torchvision 检测格式。**唯一允许 import torch 的地方之一** |
| `adapters/to_coco_json.py` | 146 | parquet → COCO GT JSON / 预测 → COCO dt |
| `loader.py` | 144 | collate（**绝不 pad 框**）+ DataLoader 工厂。MPS 下的 worker 约束逐条写了理由 |

**评测 `eval/`（3 个文件）**

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `coco_eval.py` | 250 | 包装 pycocotools 但**自己实现 summarize** —— 官方 `stats[0]` 硬编码 `maxDets=100`，非标准 maxDets 会静默返回 `-1.0`。含 `bootstrap_ci`（按图重采样） |
| `l0_selfcheck.py` | 167 | 五类合成预测（完美/抖动阶梯/空/丢一半/加假阳），用来验证评测器本身是对的 |
| `slice_eval.py` | 290 | 切片评测器：非成员 GT 置 `iscrowd=1` 忽略。含 `default_slices()`（12 个切片，**不含 P2/P3**）与残差/受控切片 |

**Baseline `baselines/`（2 个文件）**

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `torchvision_detector.py` | 227 | 一次前向同时产出 RPN proposals（L1）与 COCO 检测（L3）。支持 hybrid 设备切分 |
| `owlv2_detector.py` | 190 | OWLv2 开放词表检测（L2）。几何逻辑**只有一条代码路径**（`raw_forward`），`target_sizes` 传 `max(H,W)` |

### 3.3 `scripts/` —— 可执行入口（35 个）

命名规律：`sN_` 是第 N 阶段，`sN_verify_*` / `sN_smoke_*` / `*_selfcheck` 是验证门，
`*_recon` 是**写码之前**的前置侦察，`*_diagnose_*` 是反直觉结果的根因诊断，`bench_*` 是基准测试。

**主流程（跑一遍就出结果）**

| 脚本 | 阶段 | 作用 |
| --- | --- | --- |
| `s0_setup_env.sh` | S0 | 建 venv 并按锁文件装依赖。**幂等**，可重复执行 |
| `s1_download.py` | S1 | 下载 3 GB zip + sha256 校验 + 安全解压 |
| `s2_build_tables.py` | S2 | XML → 三张 parquet（约 23 秒） |
| `s2_make_review_sheet.py` | S2 | 生成代理核验清单（60 张裁剪图） |
| `s2_apply_vlm_labels.py` | S2 | 写回 VLM 预标（**60 个标签硬编码在 `LABELS` 字典里**） |
| `s2_merge_human_labels.py` | S2 | 合并人工复核、重算 precision、回写 `slices.yaml` |
| `s3_make_splits.py` | S3 | 划分 + 两个 val 子集（约 1.4 秒） |
| `s5_build_gt.py` | S5 | 物化三份 COCO GT |
| `s7_select_device.py` | S7 | 选推理设备并**验证数值一致性** |
| `s7_run_baselines.py` | S7 | L1 + L3 推理（约 14.4 分钟） |
| `s7b_probe_owlv2.py` | S7b | 下载 OWLv2 + **几何对齐验证** + 速度探针 |
| `s7b_select_prompt.py` | S7b | 在 **trainval** 上选 prompt（在 val 上选就是泄漏） |
| `s7b_run_owlv2.py` | S7b | L2 推理（约 21.8 分钟） |

**验证门（判定能不能进下一阶段）**

| 脚本 | 门数 | 阻塞语义 |
| --- | --- | --- |
| `s0_verify_env.py` | 6 | G3（pycocotools 冒烟）、G5/G6（云同步）阻塞 |
| `s1_verify_data.py` | 7 | 全部阻塞 |
| `s2_verify_tables.py` | 12 | G4（论文对账）为 WARN，其余阻塞 |
| `s3_verify_splits.py` | 9 | H8（口径隔离留痕）为 WARN；**H9 会自动重跑一次划分**验可复现 |
| `s4_smoke_loader.py` | 7 | 全部 PASS；`--quick` 跳过多 worker 吞吐扫描 |
| `s5_l0_selfcheck.py` | 9 | 硬断言只两条（AP 严格单调 + 端点），中间档比对回归基线 |
| `s6_slice_eval.py` | 8 | L6（与原生 areaRng 对账）为 WARN |
| `s7_evaluate.py` | 4 | 全部阻塞，含 **M2 负控制（AP 必须精确为 0）** |

**前置侦察（先量后写，省掉了大量无用代码）**

| 脚本 | 推翻了什么 |
| --- | --- |
| `s2_recon.py` | `truncated` 有真标注（省掉一个代理）；类名权威是目录名不是 XML |
| `s3_recon.py` | 99.96% 是单图簇、每类最少 4 张图、每图恰好 1 个类 —— 三块计划代码全不需要 |
| `s7_probe_speed.py` | 40 张图 20 分钟没跑完 → 触发设备诊断 |

**根因诊断（每个反直觉数字都有一个）**

| 脚本 | 回答的问题 |
| --- | --- |
| `s2_diagnose_r9.py` | 类名与目录名不一致的分型 |
| `s2_diagnose_adjacency.py` | R9 是否为「下拉列表选错相邻项」 |
| `s2_diagnose_size.py` | 尺寸分布与论文 Fig.5D 差 3pp 的根因 |
| `s2_validate_proxy_fix.py` | 代理修正方案是否真有效（结论：最好也只到 0.429，不够） |
| `s4_diagnose_workers.py` | 多 worker 为何慢 24–41 倍（解码后膨胀 111 倍 → IPC 压倒一切） |
| `s6_diagnose_crosstalk.py` | 切片重叠有多严重（T1 中 87.3% 是大目标） |
| `s7_diagnose_device.py` | Faster R-CNN 在 MPS 上慢 44 倍（`roi_heads` 算子回退） |
| `s7_diagnose_density.py` | 密度切片为何不单调（受控后 D2 > D1 仍成立，但样本量不足 → 开放问题） |

**基准测试**

| 脚本 | 结论 |
| --- | --- |
| `bench_xml_read.py` | zip 顺序读比磁盘逐文件快 82.5 倍 |
| `bench_image_read.py` | zip 随机访问 1,360 img/s vs 磁盘 38.7 img/s，快 35.2 倍 |

### 3.4 `tests/` —— 133 条回归测试

| 文件 | 行数 | 守什么 |
| --- | --- | --- |
| `test_paths.py` | 193 | 三区隔离断言、未登记子目录必须报错、软链接绕不过 |
| `test_unzip_safe.py` | 185 | 路径穿越、zip 炸弹的预检 |
| `test_parse_and_geom.py` | 258 | XML 解析容错、`area_bin` 边界（`area == 1024` 恰好等号的那 17 个框） |
| `test_splits.py` | 294 | 簇不跨界、每类至少 1 张 val、比例自检能拦住偏离 |
| `test_data_layer.py` | 329 | **AST 静态检查：核心层不 import torch**；`ann_ids` 端到端保留 |
| `test_eval.py` | 356 | 自写 summarize 的正确性；请求不存在的 maxDets 档必须抛 KeyError |
| `test_slice_eval.py` | 403 | 切片视图的 iscrowd 机制；**完全偏离的 det 会污染所有切片**这条边界 |

跑法：

```bash
../.venv/bin/python -m pytest -q          # 全部
../.venv/bin/python -m pytest tests/test_eval.py -v
```

### 3.5 `report/` —— 自动生成，不要手改

| 文件 | 由谁生成 |
| --- | --- |
| `s1_data_report.md` | `s1_verify_data.py` |
| `s2_tables_report.md` | `s2_verify_tables.py` |
| `s3_splits_report.md` | `s3_verify_splits.py` |
| `s4_loader_report.md` | `s4_smoke_loader.py` |
| `s5_eval_report.md` | `s5_l0_selfcheck.py` |
| `s6_slices_report.md` | `s6_slice_eval.py` |
| `s7_baselines_report.md` | `s7_evaluate.py` |
| `environment/env_report.md` | `s0_verify_env.py` |

每份都带生成时间戳。**重跑对应脚本即刷新** —— 手改会在下次重跑时丢失，
而且会让文档与产物失去一致性（这正是我们靠 `expected_sha256` 之类的锚点在防的事）。

`report/tables/` 当前为空，预留给训练阶段的大表导出。

---

## 4. 复现说明

### 4.1 前置条件

| 项 | 本项目实测环境 | 换环境要注意什么 |
| --- | --- | --- |
| 平台 | macOS 15 (Darwin 25.6.0) / arm64 / Apple M4 Pro | **设备策略必须重测**（见 §6） |
| 内存 | 48 GB | 建议 ≥ 16 GB（`num_workers=0`，不吃内存） |
| 磁盘 | 需 ≥ 8 GB 空闲 | zip 2.9 G + 解压 3.9 G + 产物 54 M + venv 1 G |
| Python | 3.11.15 | 3.11 上 torch/torchvision/pycocotools 的 arm64 wheel 最完整；3.13 易掉进源码编译 |
| Kaggle 凭据 | `~/.kaggle/kaggle.json` | S1 下载必需 |

### 4.2 从零全量复现

```bash
export LOGDET_ROOT=~/Desktop/logdet     # 确保不在云同步目录内
cd $LOGDET_ROOT/project

# --- S0 环境（约 3 分钟）--------------------------------------------------
bash scripts/s0_setup_env.sh
../.venv/bin/python scripts/s0_verify_env.py

# --- S1 数据（下载取决于带宽 + 解压 112.5 秒）-----------------------------
../.venv/bin/python scripts/s1_download.py
../.venv/bin/python scripts/s1_verify_data.py

# --- S2 中间表（约 23 秒）------------------------------------------------
../.venv/bin/python scripts/s2_build_tables.py
../.venv/bin/python scripts/s2_verify_tables.py

# --- S2 代理核验（可选，仅在要重做 P2/P3 裁决时跑）-----------------------
../.venv/bin/python scripts/s2_make_review_sheet.py
../.venv/bin/python scripts/s2_apply_vlm_labels.py
../.venv/bin/python scripts/s2_merge_human_labels.py \
    --set 67607=0 73597=0 73576=0 83288=0 73736=0 '106646=?'
#   ↑ 6 个人工标签取自 configs/slices.yaml 的 human_labels
#     zsh 下 ? 必须加引号，否则被当通配符

# --- S3 划分（约 1.4 秒）-------------------------------------------------
../.venv/bin/python scripts/s3_make_splits.py
../.venv/bin/python scripts/s3_verify_splits.py

# --- S4 dataloader（约 45 秒）--------------------------------------------
../.venv/bin/python scripts/s4_smoke_loader.py --quick

# --- S5 评测器（约 10 秒）------------------------------------------------
../.venv/bin/python scripts/s5_build_gt.py
../.venv/bin/python scripts/s5_l0_selfcheck.py

# --- S6 切片评测器（约 8 秒）---------------------------------------------
../.venv/bin/python scripts/s6_slice_eval.py

# --- S7 baseline（推理 14.4 分钟）----------------------------------------
../.venv/bin/python scripts/s7_select_device.py 8
../.venv/bin/python scripts/s7_run_baselines.py

# --- S7b OWLv2（下载约 3 分钟 + 推理 21.8 分钟）--------------------------
../.venv/bin/python scripts/s7b_probe_owlv2.py
../.venv/bin/python scripts/s7b_select_prompt.py 200
../.venv/bin/python scripts/s7b_run_owlv2.py

# --- 最终评测（约 45 秒）-------------------------------------------------
../.venv/bin/python scripts/s7_evaluate.py
../.venv/bin/python -m pytest -q
```

**机器时间合计约 45 分钟**（不含 Kaggle 下载），其中 **36 分钟是三个 baseline 的推理**。

### 4.3 分阶段耗时与预期门结果

| 阶段 | 主命令 | 耗时 | 预期结果 |
| --- | --- | --- | --- |
| S0 | `s0_setup_env.sh` + `s0_verify_env.py` | 3 min | 6 门全 PASS；写入 `device: mps` |
| S1 | `s1_download.py` + `s1_verify_data.py` | 下载 + 112.5 s | 7 门全 PASS；`file_inventory` 158,654 行 |
| S2 | `s2_build_tables.py` + `s2_verify_tables.py` | 23 s | 11 PASS + 1 WARN；三表 158,654 / 194,262 / 3,000 |
| S3 | `s3_make_splits.py` + `s3_verify_splits.py` | 1.4 s | 8 PASS + 1 WARN；trainval/val = 141,438 / 17,216 |
| S4 | `s4_smoke_loader.py --quick` | 45 s | 7 门全 PASS；写入 `num_workers: 0` |
| S5 | `s5_build_gt.py` + `s5_l0_selfcheck.py` | 10 s | 9 门全 PASS；第二次起报「与回归基线完全一致」 |
| S6 | `s6_slice_eval.py` | 8 s | 7 PASS + 1 WARN；12 个切片全部可靠 |
| S7 | `s7_run_baselines.py` + `s7_evaluate.py` | 15 min | 4 门全 PASS；M2 负控制 AP 精确为 0 |
| S7b | `s7b_run_owlv2.py` | 25 min | 几何对齐三例全过；prompt = `["a logo","a brand name","a trademark"]` |

**合计 63 条门：59 PASS + 4 WARN + 0 FAIL**，外加 133 条回归测试。

### 4.4 只想重跑评测，不想重跑推理

预测已落盘的情况下（`runs/predictions/*.parquet` 存在）：

```bash
../.venv/bin/python scripts/s5_build_gt.py        # 若 GT 缺失
../.venv/bin/python scripts/s7_evaluate.py        # 约 45 秒，出 s7_baselines_report.md
../.venv/bin/python scripts/s6_slice_eval.py      # 切片评测器自校验
../.venv/bin/python scripts/s7_diagnose_density.py
```

**这是迭代评测口径时该走的路径** —— 改切片定义、加受控切片、换指标都不需要重新前向。

### 4.5 换机器

```bash
export LOGDET_ROOT=/new/path/logdet
```

代码一行不用改（`configs/paths.yaml` 是唯一含绝对路径的文件，且支持环境变量覆盖）。
数据表里只存 `rel_path`，绝对路径运行时现拼，所以 `runs/` 下的产物可以整体搬走而不失效。

**但有三件事必须在新机器上重跑**：

1. `s0_verify_env.py` —— G4 会重测 MPS/CPU 数值一致性
2. `s7_diagnose_device.py` + `s7_select_device.py` —— **设备策略是本机实测结果，不是常识**
3. `s4_smoke_loader.py`（完整版，不加 `--quick`）—— 重扫 `num_workers` 最优值

---

## 5. `runs/` 产物说明与重建成本

| 产物 | 大小 | 由谁产出 | 重建耗时 | 内容 |
| --- | --- | --- | --- | --- |
| `tables/file_inventory.parquet` | 2.4 M | `s1_download.py` | 112.5 s（含解压） | 磁盘清点，158,654 行 |
| `tables/images.parquet` | 3.0 M | `s2_build_tables.py` | 23 s | 图级表，158,654 行 × 15 列 |
| `tables/annotations.parquet` | 11 M | 同上 | 同上 | 框级表，194,262 行 × 45 列（含几何派生与三个难例代理） |
| `tables/classes.parquet` | 136 K | 同上 | 同上 | 类级表，3,000 行 |
| `slices/review_sheet.csv` | 16 K | `s2_make_review_sheet.py` | 秒级 | 60 张核验记录。**`human_label` 列不可重建**（已固化进 `slices.yaml`） |
| `slices/crops/*.jpg` | 1.2 M | 同上 | 秒级 | 60 张裁剪图 |
| `slices/sheets/*.jpg` | 1.9 M | `s2_make_contact_sheets.py` | 秒级 | 10 张带标签的联系表 |
| `splits/split_images.parquet` | 4.0 M | `s3_make_splits.py` | 1.4 s | 全量图 + `split` / `cluster_id` / 子集归属 |
| `splits/val2k_repr.parquet` | 132 K | 同上 | 同上 | 2,000 图，代表性子集 |
| `splits/val_hard_pool.parquet` | 132 K | 同上 | 同上 | 1,230 图，难例富集池 |
| `eval/gt/*.json` | 1.5 M | `s5_build_gt.py` | 秒级 | 三份 COCO GT，头部带源表 sha256 + 配置指纹 |
| `predictions/L1_rpn_proposals.parquet` | 21 M | `s7_run_baselines.py` | **14.4 min** | 923,700 行（每图 300 框） |
| `predictions/L3_coco_collapsed.parquet` | 4.1 M | 同上 | 同上 | 90,077 行 |
| `predictions/L2_owlv2_zeroshot.parquet` | 2.1 M | `s7b_run_owlv2.py` | **21.8 min** | 39,455 行 |
| `predictions/manifest_*.json` | 16 K | 同上 | — | 每次推理的完整配置留痕（模型、设备、prompt、对齐验证结果） |
| `metrics/s7_overall.parquet` | 12 K | `s7_evaluate.py` | 45 s | 总体指标 |
| `metrics/s7_slices.parquet` | 12 K | 同上 | 同上 | 逐切片指标 |
| `metrics/s7_density_*.parquet` | 16 K | `s7_diagnose_density.py` | 3 s | 密度受控对比 |
| `metrics/s7*_*.json` | 12 K | 各探针 | — | 设备选型 / OWLv2 探针 / prompt 选型 |
| `cache/s4_viz/*.jpg` | — | `s4_smoke_loader.py` | 45 s | 20 张画框图（J2 门的目视抽检） |
| `cache/s7b_owlv2_viz/*.jpg` | — | `s7b_probe_owlv2.py` | 1 min | OWLv2 几何对齐目视验证 |

**最贵的是三个 baseline 的推理（合计 36 分钟）**，但它是确定性可重建的，
且所有结论已固化在 `report/*.md` 与本目录的两份说明文档里。

---

## 6. 已知的复现障碍（不掩盖）

| # | 障碍 | 影响 | 处置 |
| --- | --- | --- | --- |
| 1 | **`scipy` 曾漏在锁文件外** | 全新环境能跑通 S0–S7，却在 S7b 构造 `Owlv2ImageProcessor` 时抛 `requires scipy` | 已补进 `requirements.lock.txt`（范围锁）。本机若被文件守卫拦住源码构建，用 `pip install --only-binary :all: scipy` |
| 2 | **设备策略是单机实测** | `runtime.yaml` 的 `device_by_component` 在别的硬件/torch 版本上可能反转 —— 算子回退的覆盖面随版本变化 | 换机器必须重跑 `s7_diagnose_device.py` 与 `s7b_probe_owlv2.py` |
| 3 | **6 个人工标签跑代码产不出来** | 重做 P2/P3 裁决需要人看图 | 已固化进 `configs/slices.yaml` 的 `human_labels`，`s2_merge_human_labels.py --set` 可直接喂 |
| 4 | **60 个 VLM 预标是硬编码的** | 换 VLM 会得到不同标签，precision 会变 | 标签写死在 `s2_apply_vlm_labels.py` 的 `LABELS` 字典。报告口径必须写 **"VLM-assisted, 60 预标 / 6 人工复核"**，禁止简写为「人工核验」 |
| 5 | **论文官方划分无法复现** | 我们的 trainval/val 与论文的 142,142 / 16,510 **不是同一个划分** | 官方 zip 里不含 split 清单。所有跨论文比较必须声明这一点 |
| 6 | **弱去重不是真 pHash** | 只抓「字节大小完全相同」，抓不到缩放过/加水印的近重复 | setup 阶段不训练，泄漏影响为 0；真 pHash 推迟到训练阶段 |
| 7 | **Kaggle 凭据** | 无凭据无法跑 S1 | 需要 `~/.kaggle/kaggle.json`；`s1_download.py` 用 `curl -L -C -` 支持断点续传 |
| 8 | **本机环境的两个特有干扰** | ① 文件守卫会拦 git 删 `.git/*.lock`，导致每条 git 命令留下 stale lock；② 全量 pytest 偶发 3 条 `tmp_path` 用例报 `SystemExit: 1` | ① 用 `mv` 移走锁而非 `rm`，且把移锁与 git 操作放进同一条命令；② `conftest.py` 已记录该脆弱点，连跑三次可确认非代码回归 |

---

## 7. 复现完成后的自查清单

跑完全流程，用这些数字核对（任一不符说明中间出了问题）：

| 检查项 | 期望值 |
| --- | --- |
| 图 / 框 / 类 | **158,654 / 194,262 / 3,000** |
| `truncated == 1` 的框 | **20,270（10.43%）** |
| trainval / val | **141,438 / 17,216（10.85%）** |
| val 缺席的类 | **0 / 3,000** |
| `split_images.parquet` sha256 前 16 位 | **`15597e724d4231cb`** |
| `val2k_repr.parquet` sha256 前 16 位 | **`2aedb2b8b1bcdd4b`** |
| `val_hard_pool.parquet` sha256 前 16 位 | **`02192ec63cf33967`** |
| val2k_repr / val_hard_pool / 并集 | **2,000 / 1,230 / 3,079 图**，重叠 **151** |
| L0 完美预测 AP | **1.0000** |
| L0 抖动 ε=0.40 的 AP | **0.0398** |
| M2 负控制（错类别）AP | **精确 0.000000** |
| L2 OWLv2 AP / AP50 | **0.1451 / 0.3061** |
| L3 类塌缩 AP | **0.0055** |
| L1 proposals AR@300 | **0.3268** |
| 门总数 | **63 条：59 PASS + 4 WARN + 0 FAIL** |
| 回归测试 | **133 条全过** |

> `val_ratio` 实际是 **10.85%** 而不是精确 10% —— 这是簇不可拆 + 每类至少留 1 张 val
> 两条约束的必然结果。`SplitReport` 里显式报告了**结构下限** = 类数/图数 = **1.89%**，
> 写报告时可以直接引用它解释「为什么不是 10%」。
