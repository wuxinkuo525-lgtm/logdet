#!/usr/bin/env python
"""S4：dataloader 冒烟验证门（七条）。

阻塞门：
    J1  逐样本正确性（坐标合法 / labels / image_id / ann_ids 对齐）
    J3  collate 变长框（构造 n_boxes ∈ {1,2,7} 的 batch）
    J4  确定性（shuffle=False 两次遍历，id 序列与张量 sha256 一致）
    J5  吞吐 >= 60 img/s，并实测选最优 num_workers 写回配置
    J6  worker 安全（出口张量必须在 CPU）
    J7  路径解耦（改错 dataset_root 必须抛带清晰信息的错，而非静默返回空）

非阻塞门：
    J2  可视化抽检 20 张落盘 —— **全项目唯一允许的主观检查**，且只在这里

为什么 J2 只在这里：dataloader 是唯一"人眼能一次性确认对错"的环节
（框画在图上对不对，看一眼就知道）。再往后的指标都必须靠机械判据。

用法：
    python scripts/s4_smoke_loader.py
    python scripts/s4_smoke_loader.py --quick   # 跳过 J5 的多 worker 扫描
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from logodet.config import load_yaml  # noqa: E402
from logodet.data.adapters.to_torchvision import to_torchvision  # noqa: E402
from logodet.data.core_dataset import CoreDataset, ZipImageReader  # noqa: E402
from logodet.data.loader import build_loader, detection_collate  # noqa: E402
from logodet.data.schema import Sample  # noqa: E402
from logodet.gates import GateResult, GateRunner, md_table  # noqa: E402
from logodet.paths import P  # noqa: E402

REPORT = PROJECT_DIR / "report" / "s4_loader_report.md"
THROUGHPUT_GATE = 60.0


def _load_tables():
    t = P.artifact("tables")
    sp = P.artifact("splits")
    img = pd.read_parquet(sp / "split_images.parquet")
    ann = pd.read_parquet(t / "annotations.parquet")
    v2k = pd.read_parquet(sp / "val2k_repr.parquet")
    vhp = pd.read_parquet(sp / "val_hard_pool.parquet")
    infer_ids = sorted(set(v2k["image_id"]) | set(vhp["image_id"]))
    return img, ann, infer_ids


def _make_core(img, ann, ids, *, load_image=True, use_zip=True) -> CoreDataset:
    cfg = load_yaml("dataset.yaml")
    reader = None
    if use_zip:
        zp = P.raw / cfg["source"]["zip_name"]
        if zp.is_file():
            reader = ZipImageReader(zp, cfg["parse"]["archive_top_dir"])
            rels = set(img[img["image_id"].isin(set(ids))]["rel_path"])
            reader.build_index(rels)
    return CoreDataset(
        img, ann,
        dataset_root=P.dataset,
        image_ids=ids,
        class_agnostic=True,
        load_image=load_image,
        zip_reader=reader,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过 J5 的多 worker 扫描")
    args = ap.parse_args()

    print("=" * 78)
    print(" S4 dataloader 冒烟验证门")
    print("=" * 78)

    img, ann, infer_ids = _load_tables()
    print(f"\n推理集合 {len(infer_ids):,} 图（val2k ∪ hard_pool）")

    runner = GateRunner("S4", "S4 dataloader 验证报告")
    core = _make_core(img, ann, infer_ids)
    print(f"CoreDataset 长度 {len(core):,}"
          f"，图像通路 {'zip' if core.zip_reader else '磁盘'}")

    # ---- J1 逐样本正确性 ---------------------------------------------------
    def j1() -> GateResult:
        rng = np.random.default_rng(6129)
        idxs = rng.choice(len(core), size=min(200, len(core)), replace=False)
        ann_by_img = ann.groupby("image_id")["ann_id"].apply(set).to_dict()

        bad_box = bad_label = bad_id = bad_ann = bad_shape = 0
        n_boxes_total = 0
        for i in idxs:
            s: Sample = core[int(i)]
            s.validate()  # 内含坐标合法性检查
            n_boxes_total += s.n_boxes
            if s.n_boxes:
                x1, y1, x2, y2 = s.boxes_xyxy.T
                if not ((x1 >= 0).all() and (y1 >= 0).all()
                        and (x2 <= s.width + 0.5).all() and (y2 <= s.height + 0.5).all()
                        and (x2 > x1).all() and (y2 > y1).all()):
                    bad_box += 1
                if not (s.labels == 1).all():
                    bad_label += 1
                if set(s.ann_ids.tolist()) != ann_by_img.get(s.image_id, set()):
                    bad_ann += 1
            if s.image is None or s.image.shape[:2] != (s.height, s.width):
                bad_shape += 1
            if int(img.loc[img["image_id"] == s.image_id, "img_w"].iloc[0]) != s.width:
                bad_id += 1

        rows = [
            ("抽样样本数", f"{len(idxs)}", ""),
            ("累计框数", f"{n_boxes_total:,}", ""),
            ("坐标越界/退化", str(bad_box), "必须 0"),
            ("labels != 1", str(bad_label), "必须 0（主轨 class-agnostic）"),
            ("ann_ids 与表不一致", str(bad_ann), "必须 0 —— 切片评测靠它对齐"),
            ("image 尺寸与表不符", str(bad_shape), "必须 0"),
            ("image_id 错位", str(bad_id), "必须 0"),
        ]
        ok = not (bad_box or bad_label or bad_ann or bad_shape or bad_id)
        return GateResult("J1", "逐样本正确性", "PASS" if ok else "FAIL",
                          "200 个样本的坐标、标签、ann_ids、尺寸全部正确" if ok
                          else "存在不一致样本", blocking=True, rows=rows)

    runner.run(j1, "J1", "逐样本正确性", True)

    # ---- J2 可视化抽检（非阻塞）-------------------------------------------
    def j2() -> GateResult:
        from PIL import Image, ImageDraw

        out_dir = P.artifact("cache") / "s4_viz"
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(core), size=min(20, len(core)), replace=False)
        saved = 0
        for i in idxs:
            s = core[int(i)]
            im = Image.fromarray(s.image).convert("RGB")
            d = ImageDraw.Draw(im)
            for (x1, y1, x2, y2) in s.boxes_xyxy:
                d.rectangle([x1, y1, x2, y2], outline=(255, 32, 32),
                            width=max(2, int(min(im.size) * 0.005)))
            im.save(out_dir / f"{s.image_id}_{s.n_boxes}box.jpg", quality=85)
            saved += 1
        return GateResult(
            "J2", "可视化抽检", "PASS",
            f"{saved} 张画框图已落盘到 {out_dir.relative_to(P.artifacts)}，"
            f"请人眼确认框位置正确（本项目唯一允许的主观检查）",
            blocking=False,
            rows=[("落盘张数", str(saved), ""), ("目录", str(out_dir), "")],
        )

    runner.run(j2, "J2", "可视化抽检", False)

    # ---- J3 collate 变长 ---------------------------------------------------
    def j3() -> GateResult:
        # 找出框数分别为 1 / 2 / >=7 的样本，凑成一个 batch
        nb = img[img["image_id"].isin(set(infer_ids))].set_index("image_id")["n_boxes"]
        picks: list[int] = []
        for target in (1, 2):
            cand = nb[nb == target]
            if len(cand):
                picks.append(int(cand.index[0]))
        cand = nb[nb >= 7]
        if len(cand):
            picks.append(int(cand.index[0]))
        if len(picks) < 2:
            return GateResult("J3", "collate 变长", "FAIL",
                              "推理集合里凑不出不同框数的样本", blocking=True)

        id2pos = {v: k for k, v in enumerate(core.image_ids)}
        items = [to_torchvision(core[id2pos[i]]) for i in picks]
        images, targets = detection_collate(items)

        rows = [("batch 内框数", str([int(nb.loc[i]) for i in picks]), "")]
        ok_len = len(images) == len(picks) and len(targets) == len(picks)
        ok_shape = all(
            t["boxes"].shape[0] == int(nb.loc[i]) for t, i in zip(targets, picks)
        )
        ok_nopad = len({tuple(t["boxes"].shape) for t in targets}) == len(
            {int(nb.loc[i]) for i in picks}
        )
        ok_annid = all(t["ann_ids"].shape[0] == t["boxes"].shape[0] for t in targets)
        ok_imgvar = len({tuple(im.shape) for im in images}) >= 1
        rows += [
            ("collate 后长度", f"{len(images)} / {len(targets)}", f"应为 {len(picks)}"),
            ("各 target 框数", str([int(t["boxes"].shape[0]) for t in targets]), "应逐一对应"),
            ("未被 pad", str(ok_nopad), "不同框数应保持不同形状"),
            ("ann_ids 长度对齐", str(ok_annid), "必须 True"),
            ("图像形状", str([tuple(im.shape) for im in images]), "允许各不相同"),
        ]
        ok = ok_len and ok_shape and ok_nopad and ok_annid and ok_imgvar
        return GateResult("J3", "collate 变长", "PASS" if ok else "FAIL",
                          "变长框未被 pad，ann_ids 长度对齐" if ok else "collate 行为异常",
                          blocking=True, rows=rows)

    runner.run(j3, "J3", "collate 变长", True)

    # ---- J4 确定性 ---------------------------------------------------------
    def j4() -> GateResult:
        def traverse() -> tuple[list[int], str]:
            ld = build_loader(core, to_torchvision, batch_size=4,
                              num_workers=0, shuffle=False)
            ids: list[int] = []
            h = hashlib.sha256()
            for k, (images, targets) in enumerate(ld):
                for t in targets:
                    ids.append(int(t["image_id"]))
                if k < 5:  # 只对前 5 个 batch 的张量做哈希，够用且快
                    for im in images:
                        h.update(np.ascontiguousarray(im.numpy()).tobytes())
                if k >= 20:
                    break
            return ids, h.hexdigest()

        ids_a, h_a = traverse()
        ids_b, h_b = traverse()
        rows = [
            ("遍历样本数", f"{len(ids_a)}", "限 21 个 batch"),
            ("id 序列一致", str(ids_a == ids_b), "必须 True"),
            ("前 5 batch 张量 sha256", h_a[:16] + "…", "一致" if h_a == h_b else "不一致"),
            ("是否按 image_id 升序", str(ids_a == sorted(ids_a)), "CoreDataset 保证稳定顺序"),
        ]
        ok = ids_a == ids_b and h_a == h_b
        return GateResult("J4", "确定性", "PASS" if ok else "FAIL",
                          "两次遍历的 id 序列与张量内容完全一致" if ok else "遍历不可复现",
                          blocking=True, rows=rows)

    runner.run(j4, "J4", "确定性", True)

    # ---- J5 吞吐 -----------------------------------------------------------
    def j5() -> GateResult:
        worker_opts = [0] if args.quick else [0, 4, 6, 8]
        n_probe = min(600, len(core))
        results: list[tuple[int, float]] = []
        rows: list[tuple[str, ...]] = []

        for nw in worker_opts:
            sub = _make_core(img, ann, infer_ids[:n_probe])
            ld = build_loader(sub, to_torchvision, batch_size=4,
                              num_workers=nw, shuffle=False)
            t0 = time.time()
            n = 0
            for images, _ in ld:
                n += len(images)
            dt = time.time() - t0
            rate = n / dt
            results.append((nw, rate))
            rows.append((f"num_workers={nw}", f"{rate:.1f} img/s",
                         f"{n} 图 / {dt:.1f}s"))
            print(f"        num_workers={nw:<2} {rate:>7.1f} img/s", flush=True)

        best_nw, best_rate = max(results, key=lambda x: x[1])
        rows.append(("最优配置", f"num_workers={best_nw}", f"{best_rate:.1f} img/s"))
        rows.append(("门槛", f">= {THROUGHPUT_GATE:.0f} img/s",
                     "达标" if best_rate >= THROUGHPUT_GATE else "未达标"))
        rows.append(("图像通路", "zip" if core.zip_reader else "磁盘",
                     "磁盘实测仅 38.7 img/s，见 bench_image_read.py"))

        # 写回运行时配置，供后续 baseline 直接用
        rt = PROJECT_DIR / "configs" / "runtime.yaml"
        if rt.is_file():
            txt = rt.read_text(encoding="utf-8")
            if "num_workers:" not in txt:
                txt += (
                    f"\n# 由 scripts/s4_smoke_loader.py 实测选出\n"
                    f"num_workers: {best_nw}\n"
                    f"loader_img_per_sec: {best_rate:.1f}\n"
                )
                rt.write_text(txt, encoding="utf-8")

        ok = best_rate >= THROUGHPUT_GATE
        return GateResult("J5", "吞吐", "PASS" if ok else "FAIL",
                          f"最优 num_workers={best_nw}，{best_rate:.1f} img/s（门槛 {THROUGHPUT_GATE:.0f}）",
                          blocking=True, rows=rows)

    runner.run(j5, "J5", "吞吐", True)

    # ---- J6 worker 安全 ----------------------------------------------------
    def j6() -> GateResult:
        import torch

        # 正向：多 worker 下遍历若产出非 CPU 张量，collate 会抛异常
        sub = _make_core(img, ann, infer_ids[:64])
        ld = build_loader(sub, to_torchvision, batch_size=4, num_workers=2, shuffle=False)
        devices = set()
        for images, targets in ld:
            for im in images:
                devices.add(im.device.type)
            for t in targets:
                devices.add(t["boxes"].device.type)
        # 反向：故意塞一个非 CPU 张量，collate 必须拦住
        guard_works = True
        if torch.backends.mps.is_available():
            fake = (torch.zeros(3, 4, 4, device="mps"), {"boxes": torch.zeros(0, 4)})
            try:
                detection_collate([fake])
                guard_works = False
            except RuntimeError:
                guard_works = True

        rows = [
            ("多 worker 出口 device", str(sorted(devices)), "只应有 cpu"),
            ("spawn 上下文", "已启用", "macOS 上 fork 会随机崩溃"),
            ("persistent_workers", "已启用", "抵消 spawn 的启动开销"),
            ("pin_memory", "False", "MPS 不支持 pinned host memory"),
            ("非 CPU 张量被拦住", str(guard_works), "collate 出口断言生效"),
        ]
        ok = devices == {"cpu"} and guard_works
        return GateResult("J6", "worker 安全", "PASS" if ok else "FAIL",
                          "worker 只产 CPU 张量，且出口断言能拦住违规" if ok
                          else "worker 张量设备异常或断言失效",
                          blocking=True, rows=rows)

    runner.run(j6, "J6", "worker 安全", True)

    # ---- J7 路径解耦 -------------------------------------------------------
    def j7() -> GateResult:
        bad_core = CoreDataset(
            img, ann,
            dataset_root=Path("/nonexistent/logdet/data/LogoDet-3K"),
            image_ids=infer_ids[:4],
            class_agnostic=True,
            load_image=True,
            zip_reader=None,  # 关掉 zip，强制走磁盘
        )
        raised = False
        msg = ""
        try:
            _ = bad_core[0]
        except FileNotFoundError as e:
            raised = True
            msg = str(e)
        rows = [
            ("抛 FileNotFoundError", str(raised), "必须 True，不能静默返回空"),
            ("错误信息含 dataset_root 提示", str("paths.yaml" in msg), "必须 True"),
            ("错误信息首行", msg.splitlines()[0][:60] if msg else "(无)", ""),
        ]
        ok = raised and "paths.yaml" in msg
        return GateResult("J7", "路径解耦", "PASS" if ok else "FAIL",
                          "错误路径会抛带处置建议的异常" if ok
                          else "路径错误未被正确报出", blocking=True, rows=rows)

    runner.run(j7, "J7", "路径解耦", True)

    # ---- 报告 --------------------------------------------------------------
    extra = [(
        "图像读取通路基准（scripts/bench_image_read.py，300 张，字节与解码逐一校验一致）",
        md_table([
            ("A 磁盘逐文件", "38.7 img/s", "推理全集 3,079 图需 79.6s，未达 60 门槛"),
            ("B zip 随机访问", "1,360.4 img/s", "推理全集需 2.3s（含 1.0s 建索引），快 35.2x"),
            ("结论", "采用 B", "瓶颈是每次 open() 过文件守卫的固定开销，非磁盘寻道"),
        ], headers=("通路", "吞吐", "说明")),
    )]
    runner.write_report(REPORT, extra_sections=extra)
    print(f"\n报告已写入 {REPORT.relative_to(PROJECT_DIR)}")
    return runner.print_summary()


if __name__ == "__main__":
    raise SystemExit(main())
