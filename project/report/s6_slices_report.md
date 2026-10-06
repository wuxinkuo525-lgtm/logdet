# S6 切片评测报告（val_hard_pool）

生成时间：2026-09-15 00:15:08

由验证脚本自动生成，**不要手改** —— 重跑脚本即可刷新。

## 验证门总览

| 门 | 名称 | 阻塞 | 结果 | 说明 |
| --- | --- | --- | --- | --- |
| L1 | 切片 GT 守恒 | 是 | PASS | 每个切片视图的非 ignore 数都等于成员数，总数守恒 |
| L2 | 同轴互斥完备 | 是 | PASS | 尺寸三档与密度四档各自构成划分；残差切片定义自洽 |
| L3 | 完美预测逐切片 | 是 | PASS | 所有非空切片 AP 均 > 0.999 —— 切片视图本身不引入偏差 |
| L4 | 差异化抖动可检出 | 是 | PASS | T1 切片 AP=0.2155 显著低于 CLEAN=0.9583（差 +0.7429）—— 切片评测具备分辨力 |
| L5 | UNRELIABLE 与 CI | 是 | PASS | 不可靠切片全部带 CI，可靠切片不带 —— 符合规则 |
| L6 | 与原生 areaRng 对账 | 否 | WARN | 两种口径的数值差异已逐档记录；报告中必须注明用的是切片视图口径 |
| L7 | 报告完整性 | 否 | PASS | 12 个切片的表结构完整，可供 S7 直接复用 |
| L8 | 残差切片拆解污染 | 是 | PASS | 剔除 T1 后两个残差切片都回到 CLEAN 水平 —— 证明下降全部来自交叉污染 |

## 切片评测演示（合成预测：T1 成员 ε=0.25，其余 ε=0.05）

| 切片 | 轴 | 框数 | 图数 | 可靠 | AP | AP50 | AP50 CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1_truncated | truncation | 426 | 391 | 是 | 0.3119 | 0.8946 | — |
| SIZE_small | size | 409 | 207 | 是 | 0.8162 | 1.0000 | — |
| SIZE_medium | size | 698 | 376 | 是 | 0.7627 | 0.9863 | — |
| SIZE_large | size | 992 | 846 | 是 | 0.5791 | 0.9561 | — |
| DENSITY_1 | density | 894 | 894 | 是 | 0.6052 | 0.9600 | — |
| DENSITY_2 | density | 364 | 182 | 是 | 0.6780 | 0.9679 | — |
| DENSITY_3_4 | density | 286 | 84 | 是 | 0.7636 | 0.9881 | — |
| DENSITY_5plus | density | 555 | 70 | 是 | 0.7979 | 0.9892 | — |
| SIZE_large_noT1 | residual | 620 | 536 | 是 | 0.8329 | 1.0000 | — |
| DENSITY_1_noT1 | residual | 591 | 591 | 是 | 0.8383 | 1.0000 | — |
| T1_large | controlled | 372 | 341 | 是 | 0.3208 | 0.9050 | — |
| CLEAN | control | 908 | 766 | 是 | 0.8304 | 1.0000 | — |

## 口径声明

- **切片视图口径**：非成员 GT 置 `iscrowd=1`（保留但忽略），全部 det 参与。
- 与 COCO 原生 `AP_small` 等**不同**：后者同时过滤 GT 与 DT。
- 本子集为 `val_hard_pool`，**分布已被人为富集，禁止用它报总体指标**。
- 框数 < 200 的切片标 UNRELIABLE，必须带 bootstrap CI，禁止下强结论。

## L1 切片 GT 守恒 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| T1_truncated | 非ignore 426 / 成员 426 | ignore 1,673，合计 2,099 |
| SIZE_small | 非ignore 409 / 成员 409 | ignore 1,690，合计 2,099 |
| SIZE_medium | 非ignore 698 / 成员 698 | ignore 1,401，合计 2,099 |
| SIZE_large | 非ignore 992 / 成员 992 | ignore 1,107，合计 2,099 |
| DENSITY_1 | 非ignore 894 / 成员 894 | ignore 1,205，合计 2,099 |
| DENSITY_2 | 非ignore 364 / 成员 364 | ignore 1,735，合计 2,099 |
| DENSITY_3_4 | 非ignore 286 / 成员 286 | ignore 1,813，合计 2,099 |
| DENSITY_5plus | 非ignore 555 / 成员 555 | ignore 1,544，合计 2,099 |
| SIZE_large_noT1 | 非ignore 620 / 成员 620 | ignore 1,479，合计 2,099 |
| DENSITY_1_noT1 | 非ignore 591 / 成员 591 | ignore 1,508，合计 2,099 |
| T1_large | 非ignore 372 / 成员 372 | ignore 1,727，合计 2,099 |
| CLEAN | 非ignore 908 / 成员 908 | ignore 1,191，合计 2,099 |
| 全体框数 | 2,099 | 每个视图的总框数都必须等于它 |

## L2 同轴互斥完备 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| size 轴（3 档） | 并集 2,099 / 全体 2,099 | 缺 0 多 0 两两重叠 0 |
| density 轴（4 档） | 并集 2,099 / 全体 2,099 | 缺 0 多 0 两两重叠 0 |
| T1 ∩ CLEAN | 0 | 必须 0 —— CLEAN 定义为三轴都不命中 |
| T1 ∩ SIZE_small | 6 | 允许非 0 —— 不同轴之间不要求互斥 |
| SIZE_large_noT1 | 620 框，∩T1=0 | 是 SIZE_large 子集：True；与 T1 无交集：True |
| DENSITY_1_noT1 | 591 框，∩T1=0 | 是 DENSITY_1 子集：True；与 T1 无交集：True |

## L3 完美预测逐切片 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| T1_truncated | AP=1.000000 | n_ann=426 |
| SIZE_small | AP=1.000000 | n_ann=409 |
| SIZE_medium | AP=1.000000 | n_ann=698 |
| SIZE_large | AP=1.000000 | n_ann=992 |
| DENSITY_1 | AP=1.000000 | n_ann=894 |
| DENSITY_2 | AP=1.000000 | n_ann=364 |
| DENSITY_3_4 | AP=1.000000 | n_ann=286 |
| DENSITY_5plus | AP=1.000000 | n_ann=555 |
| SIZE_large_noT1 | AP=1.000000 | n_ann=620 |
| DENSITY_1_noT1 | AP=1.000000 | n_ann=591 |
| T1_large | AP=1.000000 | n_ann=372 |
| CLEAN | AP=1.000000 | n_ann=908 |

## L4 差异化抖动可检出 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| 抖动设置 | T1 成员 ε=0.30，其余 ε=0.02 |  |
| T1_truncated | AP=0.2155 | AP50=0.7570 |
| CLEAN 对照 | AP=0.9583 | AP50=1.0000 |
| 差距 CLEAN − T1 | +0.7429 | 阈值 > 0.30 |
| 总体（未切片） | AP=0.7427 | 应落在两者之间 —— 总体是混合结果 |
|   SIZE_small | AP=0.9443 | 其中 1.5% 是 T1 成员 → 被连带拉低的幅度与此成正比 |
|   SIZE_medium | AP=0.8635 | 其中 6.9% 是 T1 成员 → 被连带拉低的幅度与此成正比 |
|   SIZE_large | AP=0.5808 | 其中 37.5% 是 T1 成员 → 被连带拉低的幅度与此成正比 |
|   SIZE_large_noT1 | AP=0.9587 | 其中 0.0% 是 T1 成员 → 被连带拉低的幅度与此成正比 |
|   DENSITY_1 | AP=0.6190 | 其中 33.9% 是 T1 成员 → 被连带拉低的幅度与此成正比 |
|   DENSITY_1_noT1 | AP=0.9720 | 其中 0.0% 是 T1 成员 → 被连带拉低的幅度与此成正比 |
|   DENSITY_5plus | AP=0.9210 | 其中 3.8% 是 T1 成员 → 被连带拉低的幅度与此成正比 |
| 总体是否落在区间内 | True | 非阻塞判据，仅作合理性参考 |

## L5 UNRELIABLE 与 CI 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| T1_truncated | n_ann=426 | 无 CI（样本充足） |
| SIZE_small | n_ann=409 | 无 CI（样本充足） |
| SIZE_medium | n_ann=698 | 无 CI（样本充足） |
| SIZE_large | n_ann=992 | 无 CI（样本充足） |
| DENSITY_1 | n_ann=894 | 无 CI（样本充足） |
| DENSITY_2 | n_ann=364 | 无 CI（样本充足） |
| DENSITY_3_4 | n_ann=286 | 无 CI（样本充足） |
| DENSITY_5plus | n_ann=555 | 无 CI（样本充足） |
| SIZE_large_noT1 | n_ann=620 | 无 CI（样本充足） |
| DENSITY_1_noT1 | n_ann=591 | 无 CI（样本充足） |
| T1_large | n_ann=372 | 无 CI（样本充足） |
| CLEAN | n_ann=908 | 无 CI（样本充足） |
| 规则 | n_ann < 200 → 标 UNRELIABLE 且必须带 CI | 样本充足的切片不算 CI（每次要重跑 n_boot 遍评测，很贵） |

## L6 与原生 areaRng 对账 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| 口径差异 | 原生同时过滤 GT+DT；切片视图只过滤 GT | 切片视图对假阳更严格，数值一般更低或相等 |
| small | 原生 AP_small=0.4819 | 切片视图=0.4784  差 -0.0035 |
| medium | 原生 AP_medium=0.4759 | 切片视图=0.4758  差 -0.0001 |
| large | 原生 AP_large=0.5235 | 切片视图=0.5214  差 -0.0021 |
| 主口径 | 切片视图 | 更贴近「在完整检测输出下，该类目标被召回得如何」 |

## L7 报告完整性 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| 切片数 | 12 |  |
| 必需列齐全 | True |  |
| AP 为 nan 的切片 | 0 | 只允许空切片为 nan |
| 落盘 | metrics\s6_slice_demo_val_hard_pool.parquet | S7 会用同样的表结构 |

## L8 残差切片拆解污染 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| CLEAN 基准 | AP=0.9583 | 未含任何 T1 成员 |
| SIZE_large | AP=0.5808（含 37.5% T1） | 剔除后 SIZE_large_noT1=0.9587，回升 +0.3779 |
|   SIZE_large_noT1 vs CLEAN | 差 +0.0004 | 回到 CLEAN 水平（|差| < 0.10） |
| DENSITY_1 | AP=0.6190（含 33.9% T1） | 剔除后 DENSITY_1_noT1=0.9720，回升 +0.3530 |
|   DENSITY_1_noT1 vs CLEAN | 差 +0.0137 | 回到 CLEAN 水平（|差| < 0.10） |
| 解读规则 | 报 SIZE_large 必须同时报 SIZE_large_noT1 | 否则会把 T1 污染误读成「模型对大目标不行」 |
