# S0 环境实测快照

生成时间：2026-09-07 16:55:44
由 `scripts/s0_verify_env.py` 自动生成，**不要手改** —— 重跑脚本即可刷新。

**推荐 device：`mps`**（已写入 `configs/runtime.yaml`）

## 验证门总览

| 门 | 名称 | 阻塞 | 结果 | 说明 |
| --- | --- | --- | --- | --- |
| G1 | 依赖版本对账 | 是 | PASS | 精确锁 9 项全部一致，范围锁 4 项已记录 |
| G2 | MPS 可用性 | 否 | PASS | MPS 可用 |
| G3 | pycocotools 冒烟 | 是 | PASS | 完美预测得满分，评测器与 numpy ABI 均正常 |
| G4 | MPS/CPU 数值一致 | 否 | PASS | 一致（feat 1.42e-05，box 0.001px），推荐 device=mps |
| G5 | root 非云同步目录 | 是 | PASS | dataset / artifacts / raw / cache 四个 root 解析后均在本机盘 |
| G6 | 无 iCloud 占位符 | 是 | PASS | 项目目录内 *.icloud 数 == 0 |

## 硬件

| 项 | 值 |
| --- | --- |
| 平台 | Darwin 25.6.0 / arm64 |
| 芯片 | Apple M4 Pro |
| 物理核数 | 14 |
| 内存 | 48 GB |
| Python | 3.11.15 |
| 解释器 | /Users/qiuyuantang/Desktop/logdet/.venv/bin/python |

## G1 依赖版本对账 明细

| 项 | 值 | 说明 |
| --- | --- | --- | --- |
| lxml | 5.3.0 | 5.3.0 | ok |
| numpy | 1.26.4 | 1.26.4 | ok |
| pandas | 2.2.3 | 2.2.3 | ok |
| pillow | 10.4.0 | 10.4.0 | ok |
| pyarrow | 17.0.0 | 17.0.0 | ok |
| pycocotools | 2.0.8 | 2.0.8 | ok |
| torch | 2.5.1 | 2.5.1 | ok |
| torchvision | 0.20.1 | 0.20.1 | ok |
| transformers | 4.46.3 | 4.46.3 | ok |
| matplotlib | >=3.9,<4 | 3.11.1 | 范围锁·仅记录 |
| pytest | >=8.3,<9 | 8.4.2 | 范围锁·仅记录 |
| pyyaml | >=6.0,<7 | 6.0.3 | 范围锁·仅记录 |
| tqdm | >=4.66,<5 | 4.70.0 | 范围锁·仅记录 |

## G2 MPS 可用性 明细

| 项 | 值 |
| --- | --- |
| torch.__version__ | 2.5.1 |
| mps.is_built() | True |
| mps.is_available() | True |
| PYTORCH_ENABLE_MPS_FALLBACK | 1 |

## G3 pycocotools 冒烟 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| numpy | 1.26.4 |  |
| GT 图数 / 框数 | 3 / 5 |  |
| AP@[.50:.95] | 1.000000 | 期望 == 1.000000 |
| AP50 | 1.000000 | 期望 == 1.000000 |
| AR@100 | 1.000000 | 期望 == 1.000000 |

## G4 MPS/CPU 数值一致 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| backbone feat['0'] 最大绝对差 | 1.419e-05 | 阈值 < 5e-2 |
| RPN proposal 前 50 框最大坐标差(px) | 0.0009 | 阈值 < 1.0 |
| proposal 数 cpu / mps | 1000 / 1000 |  |

## G5 root 非云同步目录 明细

| 项 | 值 |
| --- | --- |
| project | /Users/qiuyuantang/Desktop/logdet/project |
| logdet | /Users/qiuyuantang/Desktop/logdet |
| dataset | /Users/qiuyuantang/Desktop/logdet/data/LogoDet-3K |
| artifacts | /Users/qiuyuantang/Desktop/logdet/runs |
| raw | /Users/qiuyuantang/Desktop/logdet/raw |
| cache | /Users/qiuyuantang/.cache |
| dataset 是否已存在 | 否（S1 尚未下载，正常） |
