#!/usr/bin/env python
"""S0 验证门：六条机械判据，全部只输出 PASS / FAIL / WARN + 数字，无主观判断。

阻塞门（FAIL 则拒绝进入 S1）：
    G1  依赖版本与 requirements.lock.txt 精确锁逐项对账
    G3  pycocotools 玩具 COCO 冒烟：GT == DT 时 AP 必须 == 1.0
    G5  四个 root 解析后都不在云同步目录内
    G6  项目目录内无 .icloud 占位符

非阻塞门（FAIL 只降级 + 标红，不拦路）：
    G2  MPS 可用性
    G4  MPS 与 CPU 的数值一致性 —— 超差则把 device 降为 cpu 写回 runtime.yaml

为什么 G3 是最有价值的一条：它用一个 3 图 5 框的玩具数据集，
把"评测器能否对完美预测给出满分"这件事变成一个可断言的数字。
numpy ABI 不兼容、pycocotools 编译错、bbox 格式搞反 —— 这三类
最常见又最难查的问题，都会在这一条上当场暴露。

用法：
    python scripts/s0_verify_env.py
    python scripts/s0_verify_env.py --skip-heavy   # 跳过 G4（不下模型权重）
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

# 这些必须在 import torch 之前生效。脚本可能被直接用绝对路径调用
# （没走 venv 的 activate），所以在这里兜一层默认值，
# 而不是只依赖 activate 钩子 —— 否则 MPS 缺算子时会抛 NotImplementedError
# 而不是回退 CPU，表现为 G4 莫名失败。
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")

LOCK_FILE = PROJECT_DIR / "environment" / "requirements.lock.txt"
REPORT_FILE = PROJECT_DIR / "environment" / "env_report.md"
RUNTIME_FILE = PROJECT_DIR / "configs" / "runtime.yaml"


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    gate: str
    name: str
    status: str  # PASS / FAIL / WARN / SKIP
    detail: str
    blocking: bool
    rows: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


# ---------------------------------------------------------------------------
# G1 依赖版本对账
# ---------------------------------------------------------------------------


def _parse_lock(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """拆出精确锁与范围锁。注释里的备用组合会被 # 剥掉，不会误入。"""
    exact: dict[str, str] = {}
    ranged: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" in line:
            name, ver = line.split("==", 1)
            exact[name.strip().lower()] = ver.strip()
        else:
            m = re.match(r"^([A-Za-z0-9._-]+)", line)
            if m:
                ranged[m.group(1).lower()] = line[m.end() :].strip()
    return exact, ranged


def gate_versions() -> GateResult:
    from importlib.metadata import PackageNotFoundError, version

    exact, ranged = _parse_lock(LOCK_FILE)
    rows: list[tuple[str, ...]] = []
    bad: list[str] = []

    for name, want in sorted(exact.items()):
        try:
            got = version(name)
        except PackageNotFoundError:
            rows.append((name, want, "未安装", "FAIL"))
            bad.append(name)
            continue
        # 只比 PEP 440 的"公开版本号"（`+` 前面那段）。
        #
        # torch/torchvision 在不同平台上会给同一个发行版本追加不同的
        # 本地构建标签：Windows 上 CUDA 版是 "2.5.1+cu124"、CPU 版是
        # "2.5.1+cpu"，而 macOS/Linux 的 wheel 通常没有这个后缀。
        # 这个标签只说明装的是哪种硬件后端的构建，不代表换了发行版本——
        # 真正的"换后端会不会改变数值"由 S0-G4 / S7 的一致性验证负责，
        # 不该让 G1 在这里把构建标签误判成版本漂移。
        got_public = got.split("+", 1)[0]
        ok = got_public == want
        rows.append((name, want, got, "ok" if ok else "MISMATCH"))
        if not ok:
            bad.append(f"{name}(want {want}, got {got})")

    for name, spec in sorted(ranged.items()):
        try:
            got = version(name)
        except PackageNotFoundError:
            rows.append((name, spec, "未安装", "FAIL"))
            bad.append(name)
            continue
        rows.append((name, spec, got, "范围锁·仅记录"))

    status = "PASS" if not bad else "FAIL"
    detail = (
        f"精确锁 {len(exact)} 项全部一致，范围锁 {len(ranged)} 项已记录"
        if not bad
        else "不一致：" + "; ".join(bad)
    )
    return GateResult("G1", "依赖版本对账", status, detail, blocking=True, rows=rows)


# ---------------------------------------------------------------------------
# G2 MPS 可用性
# ---------------------------------------------------------------------------


def gate_mps() -> GateResult:
    import torch

    built = torch.backends.mps.is_built()
    avail = torch.backends.mps.is_available()
    rows = [
        ("torch.__version__", torch.__version__),
        ("mps.is_built()", str(built)),
        ("mps.is_available()", str(avail)),
        ("PYTORCH_ENABLE_MPS_FALLBACK", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "(未设置)")),
    ]
    if built and avail:
        return GateResult("G2", "MPS 可用性", "PASS", "MPS 可用", blocking=False, rows=rows)
    return GateResult(
        "G2",
        "MPS 可用性",
        "WARN",
        "MPS 不可用，全部推理将走 CPU（OWLv2 耗时约翻 2-3 倍，但结论不受影响）",
        blocking=False,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# G3 pycocotools 玩具 COCO 冒烟
# ---------------------------------------------------------------------------


def _toy_coco() -> tuple[dict, list[dict]]:
    """3 图 5 框的最小 COCO。bbox 为 [x, y, w, h]，与 COCO 规范一致。"""
    boxes = [
        (1, 10.0, 20.0, 50.0, 40.0),
        (1, 120.0, 30.0, 30.0, 30.0),
        (2, 5.0, 5.0, 100.0, 90.0),
        (3, 60.0, 40.0, 20.0, 20.0),
        (3, 150.0, 100.0, 40.0, 40.0),
    ]
    gt = {
        "info": {"description": "S0-G3 toy"},
        "licenses": [],
        "images": [
            {"id": i, "file_name": f"{i}.jpg", "width": 200, "height": 150} for i in (1, 2, 3)
        ],
        "categories": [{"id": 1, "name": "logo"}],
        "annotations": [
            {
                "id": k + 1,
                "image_id": img,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
            for k, (img, x, y, w, h) in enumerate(boxes)
        ],
    }
    dt = [
        {"image_id": img, "category_id": 1, "bbox": [x, y, w, h], "score": 1.0}
        for (img, x, y, w, h) in boxes
    ]
    return gt, dt


def gate_pycocotools() -> GateResult:
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt, dt = _toy_coco()
    rows: list[tuple[str, ...]] = []

    with tempfile.TemporaryDirectory() as td:
        gt_path = Path(td) / "gt.json"
        gt_path.write_text(json.dumps(gt), encoding="utf-8")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            coco_gt = COCO(str(gt_path))
            coco_dt = coco_gt.loadRes(dt)
            ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
            ev.evaluate()
            ev.accumulate()

        p = ev.params
        ai = list(p.areaRngLbl).index("all")
        mi = list(p.maxDets).index(100)

        prec = ev.eval["precision"][:, :, :, ai, mi]
        prec = prec[prec > -1]
        ap = float(np.mean(prec)) if prec.size else float("nan")

        prec50 = ev.eval["precision"][0, :, :, ai, mi]
        prec50 = prec50[prec50 > -1]
        ap50 = float(np.mean(prec50)) if prec50.size else float("nan")

        rec = ev.eval["recall"][:, :, ai, mi]
        rec = rec[rec > -1]
        ar100 = float(np.mean(rec)) if rec.size else float("nan")

    rows += [
        ("numpy", np.__version__),
        ("GT 图数 / 框数", "3 / 5"),
        ("AP@[.50:.95]", f"{ap:.6f}", "期望 == 1.000000"),
        ("AP50", f"{ap50:.6f}", "期望 == 1.000000"),
        ("AR@100", f"{ar100:.6f}", "期望 == 1.000000"),
    ]

    tol = 1e-6
    ok = (
        abs(ap - 1.0) < tol
        and abs(ap50 - 1.0) < tol
        and abs(ar100 - 1.0) < tol
    )
    if ok:
        return GateResult(
            "G3",
            "pycocotools 冒烟",
            "PASS",
            "完美预测得满分，评测器与 numpy ABI 均正常",
            blocking=True,
            rows=rows,
        )
    return GateResult(
        "G3",
        "pycocotools 冒烟",
        "FAIL",
        f"完美预测未得满分（AP={ap:.6f}）。优先怀疑 numpy 版本或 bbox 格式，"
        f"不要继续往下走，否则后面所有 AP 都不可信",
        blocking=True,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# G4 MPS / CPU 数值一致性
# ---------------------------------------------------------------------------


def _synthetic_images(n: int = 4):
    """确定性合成图。故意造出边缘和色块，避免纯噪声让特征退化。"""
    import numpy as np
    import torch

    rng = np.random.default_rng(6129)
    out = []
    for _ in range(n):
        h, w = 320, 400
        # 背景渐变
        yy = np.linspace(0.1, 0.7, h, dtype=np.float32)[:, None]
        xx = np.linspace(0.2, 0.9, w, dtype=np.float32)[None, :]
        img = np.stack([yy + xx * 0.3, yy * 0.6 + xx, yy * 0.9 + xx * 0.2]) % 1.0
        # 若干实心矩形，制造明确边缘
        for _ in range(6):
            x0, y0 = rng.integers(0, w - 60), rng.integers(0, h - 60)
            bw, bh = rng.integers(30, 60), rng.integers(30, 60)
            color = rng.random(3).astype(np.float32)
            img[:, y0 : y0 + bh, x0 : x0 + bw] = color[:, None, None]
        out.append(torch.from_numpy(img.astype(np.float32)))
    return out


def _forward_parts(model, imgs, device: str):
    import torch

    model = model.to(device).eval()
    batch = [im.to(device) for im in imgs]
    with torch.inference_mode():
        images, _ = model.transform(batch, None)
        feats = model.backbone(images.tensors)
        proposals, _ = model.rpn(images, feats, None)
    feat0 = feats["0"].detach().float().cpu()
    props = [p.detach().float().cpu() for p in proposals]
    return feat0, props


def gate_mps_cpu_parity() -> GateResult:
    import torch

    if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        return GateResult(
            "G4", "MPS/CPU 数值一致", "SKIP", "MPS 不可用，无需比对", blocking=False
        )

    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    imgs = _synthetic_images(4)
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model.rpn._post_nms_top_n = {"training": 1000, "testing": 1000}

    feat_cpu, props_cpu = _forward_parts(model, imgs, "cpu")
    feat_mps, props_mps = _forward_parts(model, imgs, "mps")

    d_feat = float((feat_cpu - feat_mps).abs().max())

    # proposal 数量可能因 NMS 边界抖动而略有差异，只比对公共前缀
    k = min(50, min(len(a) for a in props_cpu), min(len(a) for a in props_mps))
    d_box = max(
        float((a[:k] - b[:k]).abs().max()) for a, b in zip(props_cpu, props_mps)
    ) if k > 0 else float("nan")

    rows = [
        ("backbone feat['0'] 最大绝对差", f"{d_feat:.3e}", "阈值 < 5e-2"),
        (f"RPN proposal 前 {k} 框最大坐标差(px)", f"{d_box:.4f}", "阈值 < 1.0"),
        ("proposal 数 cpu / mps", f"{len(props_cpu[0])} / {len(props_mps[0])}", ""),
    ]

    ok = d_feat < 5e-2 and (k > 0 and d_box < 1.0)
    if ok:
        return GateResult(
            "G4",
            "MPS/CPU 数值一致",
            "PASS",
            f"一致（feat {d_feat:.2e}，box {d_box:.3f}px），推荐 device=mps",
            blocking=False,
            rows=rows,
        )
    return GateResult(
        "G4",
        "MPS/CPU 数值一致",
        "FAIL",
        f"超差（feat {d_feat:.2e}，box {d_box:.3f}px）→ 已把 device 降级为 cpu 写回 runtime.yaml",
        blocking=False,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# G5 / G6 路径卫生
# ---------------------------------------------------------------------------


def gate_paths() -> GateResult:
    from logodet.paths import CloudSyncPathError, load_paths

    try:
        p = load_paths()
    except CloudSyncPathError as e:
        return GateResult("G5", "root 非云同步目录", "FAIL", str(e), blocking=True)

    rows = [(name, val) for name, val in p.as_table()]
    rows.append(("dataset 是否已存在", "是" if p.dataset.exists() else "否（S1 尚未下载，正常）"))
    return GateResult(
        "G5",
        "root 非云同步目录",
        "PASS",
        "dataset / artifacts / raw / cache 四个 root 解析后均在本机盘",
        blocking=True,
        rows=rows,
    )


def gate_icloud_placeholders() -> GateResult:
    stubs = sorted(PROJECT_DIR.rglob("*.icloud"))
    if not stubs:
        return GateResult(
            "G6", "无 iCloud 占位符", "PASS", "项目目录内 *.icloud 数 == 0", blocking=True
        )
    rows = [(str(s.relative_to(PROJECT_DIR)),) for s in stubs[:20]]
    return GateResult(
        "G6",
        "无 iCloud 占位符",
        "FAIL",
        f"发现 {len(stubs)} 个未下载的占位符，先执行：brctl download {PROJECT_DIR}",
        blocking=True,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# 硬件快照 & 报告
# ---------------------------------------------------------------------------


def hardware_snapshot() -> list[tuple[str, str]]:
    def sh(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return "(读取失败)"

    mem = sh(["sysctl", "-n", "hw.memsize"])
    try:
        mem_gb = f"{int(mem) / 1024 ** 3:.0f} GB"
    except ValueError:
        mem_gb = mem
    return [
        ("平台", f"{platform.system()} {platform.release()} / {platform.machine()}"),
        ("芯片", sh(["sysctl", "-n", "machdep.cpu.brand_string"])),
        ("物理核数", sh(["sysctl", "-n", "hw.physicalcpu"])),
        ("内存", mem_gb),
        ("Python", sys.version.split()[0]),
        ("解释器", sys.executable),
    ]


def _md_table(rows: list[tuple[str, ...]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    padded = [tuple(list(r) + [""] * (width - len(r))) for r in rows]
    head = "| " + " | ".join(["项", "值", "说明"][:width]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in padded]
    return "\n".join([head, sep, *body])


def write_report(results: list[GateResult], hw: list[tuple[str, str]], device: str) -> None:
    lines = [
        "# S0 环境实测快照",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"由 `scripts/s0_verify_env.py` 自动生成，**不要手改** —— 重跑脚本即可刷新。",
        "",
        f"**推荐 device：`{device}`**（已写入 `configs/runtime.yaml`）",
        "",
        "## 验证门总览",
        "",
        "| 门 | 名称 | 阻塞 | 结果 | 说明 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.gate} | {r.name} | {'是' if r.blocking else '否'} | "
            f"{r.status} | {r.detail.splitlines()[0] if r.detail else ''} |"
        )

    lines += ["", "## 硬件", "", _md_table([(k, v) for k, v in hw])]

    for r in results:
        if not r.rows:
            continue
        lines += ["", f"## {r.gate} {r.name} 明细", "", _md_table(r.rows)]

    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runtime(device: str, results: list[GateResult]) -> None:
    g4 = next((r for r in results if r.gate == "G4"), None)
    reason = g4.detail if g4 else ""
    RUNTIME_FILE.write_text(
        "\n".join(
            [
                "# 由 scripts/s0_verify_env.py 自动生成，不要手改。",
                "# device 的取值由 S0-G4（MPS/CPU 数值一致性）决定：",
                "#   一致  → mps",
                "#   超差  → cpu（宁可慢，也不要用一个数值上不可信的后端出指标）",
                "",
                f"device: {device}",
                f"verified_at: {datetime.now().isoformat(timespec='seconds')}",
                f"g4_detail: {json.dumps(reason, ensure_ascii=False)}",
                "",
            ]
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="S0 环境验证门")
    ap.add_argument(
        "--skip-heavy",
        action="store_true",
        help="跳过 G4（不下载 Faster R-CNN 权重，用于快速重跑）",
    )
    args = ap.parse_args()

    print("=" * 78)
    print(" S0 环境验证门")
    print("=" * 78)

    results: list[GateResult] = []

    def run(fn, gate: str, name: str, blocking: bool):
        try:
            r = fn()
        except Exception as e:  # 门自身崩了也要变成一条 FAIL，而不是堆栈糊一屏
            r = GateResult(gate, name, "FAIL", f"{type(e).__name__}: {e}", blocking=blocking)
        results.append(r)
        mark = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "SKIP": "SKIP"}[r.status]
        print(f"\n[{r.gate}] {r.name} ... {mark}")
        print(f"      {r.detail}")
        for row in r.rows:
            cells = list(row) + [""] * (3 - len(row))
            print(f"        {cells[0]:<34} {str(cells[1]):<26} {cells[2]}")
        return r

    run(gate_versions, "G1", "依赖版本对账", True)
    run(gate_mps, "G2", "MPS 可用性", False)
    run(gate_pycocotools, "G3", "pycocotools 冒烟", True)
    if args.skip_heavy:
        results.append(
            GateResult("G4", "MPS/CPU 数值一致", "SKIP", "--skip-heavy 指定跳过", blocking=False)
        )
        print("\n[G4] MPS/CPU 数值一致 ... SKIP")
    else:
        run(gate_mps_cpu_parity, "G4", "MPS/CPU 数值一致", False)
    run(gate_paths, "G5", "root 非云同步目录", True)
    run(gate_icloud_placeholders, "G6", "无 iCloud 占位符", True)

    # device 决策：G2 不可用 → cpu；G4 超差 → cpu；否则 mps
    g2 = next(r for r in results if r.gate == "G2")
    g4 = next(r for r in results if r.gate == "G4")
    device = "mps" if (g2.status == "PASS" and g4.status in ("PASS", "SKIP")) else "cpu"

    hw = hardware_snapshot()
    write_runtime(device, results)
    write_report(results, hw, device)

    blocking_fails = [r for r in results if r.blocking and r.failed]
    print("\n" + "=" * 78)
    for r in results:
        flag = "阻塞" if r.blocking else "非阻塞"
        print(f"  {r.gate}  {r.name:<22} {flag:<6} {r.status}")
    print("=" * 78)
    print(f"  推荐 device : {device}   （已写入 configs/runtime.yaml）")
    print(f"  环境报告     : {REPORT_FILE.relative_to(PROJECT_DIR)}")

    if blocking_fails:
        print(f"\n  S0 未通过：{len(blocking_fails)} 条阻塞门失败 → 禁止进入 S1")
        return 1
    warns = [r for r in results if r.status == "WARN"]
    print(f"\n  S0 通过{'（含 ' + str(len(warns)) + ' 条 WARN）' if warns else ''} → 可进入 S1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
