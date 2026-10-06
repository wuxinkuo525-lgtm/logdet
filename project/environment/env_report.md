# S0 环境实测快照

生成时间：2026-09-15 03:22:44
由 `scripts/s0_verify_env.py` 自动生成，**不要手改** —— 重跑脚本即可刷新。

**推荐 device：`cpu`**（已写入 `configs/runtime.yaml`）

## 验证门总览

| 门 | 名称 | 阻塞 | 结果 | 说明 |
| --- | --- | --- | --- | --- |
| G1 | 依赖版本对账 | 是 | PASS | 精确锁 9 项全部一致，范围锁 5 项已记录 |
| G2 | MPS 可用性 | 否 | WARN | MPS 不可用，全部推理将走 CPU（OWLv2 耗时约翻 2-3 倍，但结论不受影响） |
| G3 | pycocotools 冒烟 | 是 | PASS | 完美预测得满分，评测器与 numpy ABI 均正常 |
| G4 | MPS/CPU 数值一致 | 否 | SKIP | MPS 不可用，无需比对 |
| G5 | root 非云同步目录 | 是 | PASS | dataset / artifacts / raw / cache 四个 root 解析后均在本机盘 |
| G6 | 无 iCloud 占位符 | 是 | PASS | 项目目录内 *.icloud 数 == 0 |

## 硬件

| 项 | 值 |
| --- | --- |
| 平台 | Windows 10 / AMD64 |
| 芯片 | (读取失败) |
| 物理核数 | (读取失败) |
| 内存 | (读取失败) |
| Python | 3.11.16 |
| 解释器 | E:\ntu\computer_vision\cvproject\logdet\.venv\Scripts\python.exe |

## G1 依赖版本对账 明细

| 项 | 值 | 说明 |
| --- | --- | --- | --- |
| lxml | 5.3.0 | 5.3.0 | ok |
| numpy | 1.26.4 | 1.26.4 | ok |
| pandas | 2.2.3 | 2.2.3 | ok |
| pillow | 10.4.0 | 10.4.0 | ok |
| pyarrow | 17.0.0 | 17.0.0 | ok |
| pycocotools | 2.0.8 | 2.0.8 | ok |
| torch | 2.5.1 | 2.5.1+cu124 | ok |
| torchvision | 0.20.1 | 0.20.1+cu124 | ok |
| transformers | 4.46.3 | 4.46.3 | ok |
| matplotlib | >=3.9,<4 | 3.11.2 | 范围锁·仅记录 |
| pytest | >=8.3,<9 | 8.4.2 | 范围锁·仅记录 |
| pyyaml | >=6.0,<7 | 6.0.3 | 范围锁·仅记录 |
| scipy | >=1.11,<2 | 1.17.1 | 范围锁·仅记录 |
| tqdm | >=4.66,<5 | 4.70.1 | 范围锁·仅记录 |

## G2 MPS 可用性 明细

| 项 | 值 |
| --- | --- |
| torch.__version__ | 2.5.1+cu124 |
| mps.is_built() | False |
| mps.is_available() | False |
| PYTORCH_ENABLE_MPS_FALLBACK | 1 |

## G3 pycocotools 冒烟 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| numpy | 1.26.4 |  |
| GT 图数 / 框数 | 3 / 5 |  |
| AP@[.50:.95] | 1.000000 | 期望 == 1.000000 |
| AP50 | 1.000000 | 期望 == 1.000000 |
| AR@100 | 1.000000 | 期望 == 1.000000 |

## G5 root 非云同步目录 明细

| 项 | 值 |
| --- | --- |
| project | E:\ntu\computer_vision\cvproject\logdet\project |
| logdet | E:\ntu\computer_vision\cvproject\logdet |
| dataset | E:\ntu\computer_vision\cvproject\logdet\data\LogoDet-3K |
| artifacts | E:\ntu\computer_vision\cvproject\logdet\runs |
| raw | E:\ntu\computer_vision\cvproject\logdet\raw |
| cache | C:\Users\0\.cache |
| dataset 是否已存在 | 是 |
