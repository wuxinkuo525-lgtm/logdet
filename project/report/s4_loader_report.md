# S4 dataloader 验证报告

生成时间：2026-09-15 00:14:21

由验证脚本自动生成，**不要手改** —— 重跑脚本即可刷新。

## 验证门总览

| 门 | 名称 | 阻塞 | 结果 | 说明 |
| --- | --- | --- | --- | --- |
| J1 | 逐样本正确性 | 是 | PASS | 200 个样本的坐标、标签、ann_ids、尺寸全部正确 |
| J2 | 可视化抽检 | 否 | PASS | 20 张画框图已落盘到 cache\s4_viz，请人眼确认框位置正确（本项目唯一允许的主观检查） |
| J3 | collate 变长 | 是 | PASS | 变长框未被 pad，ann_ids 长度对齐 |
| J4 | 确定性 | 是 | PASS | 两次遍历的 id 序列与张量内容完全一致 |
| J5 | 吞吐 | 是 | PASS | 最优 num_workers=0，752.6 img/s（门槛 60） |
| J6 | worker 安全 | 是 | PASS | worker 只产 CPU 张量，且出口断言能拦住违规 |
| J7 | 路径解耦 | 是 | PASS | 错误路径会抛带处置建议的异常 |

## 图像读取通路基准（scripts/bench_image_read.py，300 张，字节与解码逐一校验一致）

| 通路 | 吞吐 | 说明 |
| --- | --- | --- |
| A 磁盘逐文件 | 38.7 img/s | 推理全集 3,079 图需 79.6s，未达 60 门槛 |
| B zip 随机访问 | 1,360.4 img/s | 推理全集需 2.3s（含 1.0s 建索引），快 35.2x |
| 结论 | 采用 B | 瓶颈是每次 open() 过文件守卫的固定开销，非磁盘寻道 |

## J1 逐样本正确性 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| 抽样样本数 | 200 |  |
| 累计框数 | 263 |  |
| 坐标越界/退化 | 0 | 必须 0 |
| labels != 1 | 0 | 必须 0（主轨 class-agnostic） |
| ann_ids 与表不一致 | 0 | 必须 0 —— 切片评测靠它对齐 |
| image 尺寸与表不符 | 0 | 必须 0 |
| image_id 错位 | 0 | 必须 0 |

## J2 可视化抽检 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| 落盘张数 | 20 |  |
| 目录 | E:\ntu\computer_vision\cvproject\logdet\runs\cache\s4_viz |  |

## J3 collate 变长 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| batch 内框数 | [1, 2, 65] |  |
| collate 后长度 | 3 / 3 | 应为 3 |
| 各 target 框数 | [1, 2, 65] | 应逐一对应 |
| 未被 pad | True | 不同框数应保持不同形状 |
| ann_ids 长度对齐 | True | 必须 True |
| 图像形状 | [(3, 383, 484), (3, 372, 515), (3, 353, 506)] | 允许各不相同 |

## J4 确定性 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| 遍历样本数 | 84 | 限 21 个 batch |
| id 序列一致 | True | 必须 True |
| 前 5 batch 张量 sha256 | a106003f5287f803… | 一致 |
| 是否按 image_id 升序 | True | CoreDataset 保证稳定顺序 |

## J5 吞吐 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| num_workers=0 | 752.6 img/s | 600 图 / 0.8s |
| 最优配置 | num_workers=0 | 752.6 img/s |
| 门槛 | >= 60 img/s | 达标 |
| 图像通路 | zip | 磁盘实测仅 38.7 img/s，见 bench_image_read.py |

## J6 worker 安全 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| 多 worker 出口 device | ['cpu'] | 只应有 cpu |
| spawn 上下文 | 已启用 | macOS 上 fork 会随机崩溃 |
| persistent_workers | 已启用 | 抵消 spawn 的启动开销 |
| pin_memory | False | MPS 不支持 pinned host memory |
| 非 CPU 张量被拦住 | True | collate 出口断言生效 |

## J7 路径解耦 明细

| 项 | 值 | 说明 |
| --- | --- | --- |
| 抛 FileNotFoundError | True | 必须 True，不能静默返回空 |
| 错误信息含 dataset_root 提示 | True | 必须 True |
| 错误信息首行 | 图像不存在：\nonexistent\logdet\data\LogoDet-3K\Clothes\2xist\1.jp |  |
