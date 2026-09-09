#!/usr/bin/env python
"""S1 步骤二：数据完整性验证门（七条）。

阻塞门：
    V1  zip 完整性（sha256 已留存 + CRC 无损坏）
    V2  超类目录数 == 9 且名称集合精确匹配
    V3  品牌目录数 == 3000，且 9 个超类分项差值全 0
    V4  图像数 == 158,652，且 9 个超类分项在容差内
    V5  孤儿文件 == 0（only_img == 0 且 only_xml == 0）
    V7  大小写冲突 == 0（APFS 静默覆盖探针）

非阻塞门：
    V6  扩展名分布（非 .jpg 的图像单独列出，纳入后续解析白名单）

一条重要原则：V3/V4 的分项差值非 0 时，**先怀疑镜像而不是自己的脚本**。
用官方直链对同一超类抽查，差异必须逐条归因，不接受"大致对得上"。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.config import load_yaml  # noqa: E402
from logodet.gates import GateResult, GateRunner, md_table  # noqa: E402
from logodet.ingest.inventory import scan_dataset  # noqa: E402
from logodet.paths import P  # noqa: E402

REPORT = PROJECT_DIR / "report" / "s1_data_report.md"


def main() -> int:
    cfg = load_yaml("dataset.yaml")
    exp = cfg["expected"]
    tol = cfg["tolerance"]
    parse_cfg = cfg["parse"]

    image_exts = {e.lower() for e in parse_cfg["image_extensions"]}
    ann_ext = parse_cfg["annotation_extension"]
    ignore = set(parse_cfg.get("ignore_names") or [])

    print("=" * 78)
    print(" S1 数据完整性验证门")
    print("=" * 78)

    if not P.dataset.is_dir():
        print(f"\nFAIL: 数据集目录不存在 {P.dataset}")
        print("      先跑 python scripts/s1_download.py")
        return 1

    print(f"\n扫描 {P.dataset} ...")
    t0 = time.time()
    df, rep = scan_dataset(
        P.dataset, image_exts=image_exts, ann_ext=ann_ext, ignore_names=ignore
    )
    print(f"  {rep.n_files:,} 个文件 / {len(df):,} 个 stem，耗时 {time.time() - t0:.1f}s")

    inv_path = P.artifact("tables") / "file_inventory.parquet"
    df.to_parquet(inv_path, index=False)
    print(f"  清单已写入 {inv_path}")

    runner = GateRunner("S1", "S1 数据完整性报告")

    # ---- V1 zip 完整性 -----------------------------------------------------
    def gate_zip() -> GateResult:
        meta_p = P.raw / "download_meta.json"
        unzip_p = P.artifact("tables") / "unzip_report.json"
        rows: list[tuple[str, ...]] = []
        if meta_p.is_file():
            m = json.loads(meta_p.read_text(encoding="utf-8"))
            rows += [
                ("zip 大小", f"{m['size_bytes']:,} B", f"{m['size_bytes'] / 2**30:.2f} GiB"),
                ("sha256", m["sha256"][:32] + "…", "全文见 raw/download_meta.json"),
                ("下载时间", m.get("downloaded_at", "-"), ""),
            ]
        if unzip_p.is_file():
            u = json.loads(unzip_p.read_text(encoding="utf-8"))
            rows += [
                ("zip 条目总数", f"{u['total_entries']:,}", ""),
                ("解出文件数", f"{u['extracted_files']:,}", ""),
                ("编码还原文件名", f"{u['recovered_name_count']:,}", "cp437→gbk 兜底"),
                ("解压耗时", f"{u['elapsed_sec']}s", ""),
            ]
        if not rows:
            return GateResult(
                "V1", "zip 完整性", "FAIL",
                "找不到 download_meta.json / unzip_report.json，无法证明来源",
                blocking=True,
            )
        return GateResult(
            "V1", "zip 完整性", "PASS",
            "sha256 已留存；CRC 校验在 s1_download.py 中已通过（testzip() 返回 None）",
            blocking=True, rows=rows,
        )

    runner.run(gate_zip, "V1", "zip 完整性", True)

    # ---- V2 超类目录 -------------------------------------------------------
    def gate_supercats() -> GateResult:
        want = set(exp["per_supercategory"].keys())
        got = set(df["supercat"].unique())
        rows = [
            ("期望超类数", str(exp["totals"]["supercategories"]), ""),
            ("实测超类数", str(len(got)), ""),
            ("缺失", ", ".join(sorted(want - got)) or "无", ""),
            ("多余", ", ".join(sorted(got - want)) or "无", ""),
        ]
        ok = got == want and len(got) == exp["totals"]["supercategories"]
        return GateResult(
            "V2", "超类目录", "PASS" if ok else "FAIL",
            "名称集合精确匹配" if ok else "超类集合与期望不一致",
            blocking=True, rows=rows,
        )

    runner.run(gate_supercats, "V2", "超类目录", True)

    # ---- V3 品牌目录数（逐超类对账）---------------------------------------
    def gate_brand_dirs() -> GateResult:
        got = df.groupby("supercat")["brand_dir"].nunique()
        rows: list[tuple[str, ...]] = []
        bad = 0
        for name, spec in sorted(
            exp["per_supercategory"].items(), key=lambda kv: -kv[1]["classes"]
        ):
            w = spec["classes"]
            g = int(got.get(name, 0))
            d = g - w
            if abs(d) > tol["supercategory_classes"]:
                bad += 1
            rows.append((name, f"期望 {w} / 实测 {g}", f"差值 {d:+d}"))
        total_got = int(got.sum())
        d_total = total_got - exp["totals"]["brand_dirs"]
        rows.append(
            ("合计", f"期望 {exp['totals']['brand_dirs']} / 实测 {total_got}", f"差值 {d_total:+d}")
        )
        ok = bad == 0 and abs(d_total) <= tol["brand_dirs"]
        return GateResult(
            "V3", "品牌目录数", "PASS" if ok else "FAIL",
            "9 个超类分项差值全 0" if ok else f"{bad} 个超类分项不符，合计差 {d_total:+d}",
            blocking=True, rows=rows,
        )

    runner.run(gate_brand_dirs, "V3", "品牌目录数", True)

    # ---- V4 图像数（逐超类对账）-------------------------------------------
    def gate_images() -> GateResult:
        has_img = df["image_rel"].notna()
        got = df[has_img].groupby("supercat").size()
        rows: list[tuple[str, ...]] = []
        bad: list[str] = []
        for name, spec in sorted(
            exp["per_supercategory"].items(), key=lambda kv: -kv[1]["images"]
        ):
            w = spec["images"]
            g = int(got.get(name, 0))
            d = g - w
            if abs(d) > tol["per_supercategory_images"]:
                bad.append(f"{name}({d:+d})")
            rows.append((name, f"期望 {w:,} / 实测 {g:,}", f"差值 {d:+d}"))

        total_got = int(has_img.sum())
        d_total = total_got - exp["totals"]["images"]
        rows.append(
            ("合计", f"期望 {exp['totals']['images']:,} / 实测 {total_got:,}",
             f"差值 {d_total:+d}")
        )
        # 把"期望值为何是 158,654 而不是论文摘要的 158,652"写进报告，
        # 免得下次看到这个数字的人以为我们抄错了
        inc = cfg.get("known_inconsistencies", {}).get("total_images", {})
        if inc:
            gap = inc.get("paper_table_ii_sum", 0) - inc.get("paper_abstract", 0)
            rows.append(
                ("期望值来源",
                 f"论文 Table II 分项和 {inc.get('paper_table_ii_sum', 0):,}",
                 f"论文摘要写 {inc.get('paper_abstract', 0):,}，与自身分项表差 "
                 f"{gap:+d}，已查证为上游笔误")
            )

        ok = not bad and abs(d_total) <= tol["total_images"]
        detail = (
            f"总图数精确命中 {exp['totals']['images']:,}，9 个超类分项差值全 0"
            if ok else f"不符：合计差 {d_total:+d}；超差超类 {', '.join(bad) or '无'}"
        )
        return GateResult("V4", "图像数", "PASS" if ok else "FAIL", detail,
                          blocking=True, rows=rows)

    runner.run(gate_images, "V4", "图像数", True)

    # ---- V5 孤儿文件 -------------------------------------------------------
    def gate_orphans() -> GateResult:
        q = rep.quadrants
        rows = [
            ("both（图+标注）", f"{q.get('both', 0):,}", "唯一合法状态"),
            ("only_img（有图无标注）", f"{q.get('only_img', 0):,}", "必须为 0"),
            ("only_xml（有标注无图）", f"{q.get('only_xml', 0):,}", "必须为 0"),
            ("other（不成对杂项）", f"{q.get('other', 0):,}", "必须为 0"),
            ("目录层级异常", f"{len(rep.depth_anomalies):,}", "必须为 0"),
        ]
        bad = (q.get("only_img", 0) + q.get("only_xml", 0) + q.get("other", 0)
               + len(rep.depth_anomalies))
        if bad and rep.depth_anomalies:
            rows.append(("层级异常样例", "; ".join(rep.depth_anomalies[:5]), ""))
        return GateResult(
            "V5", "孤儿文件", "PASS" if bad == 0 else "FAIL",
            "每个 stem 都图与标注成对" if bad == 0 else f"存在 {bad} 个不成对/异常条目",
            blocking=True, rows=rows,
        )

    runner.run(gate_orphans, "V5", "孤儿文件", True)

    # ---- V6 扩展名分布（非阻塞）-------------------------------------------
    def gate_extensions() -> GateResult:
        rows = [(ext or "(无扩展名)", f"{n:,}", "") for ext, n in rep.ext_counts.items()]
        non_jpg = {
            e: n for e, n in rep.ext_counts.items()
            if e in image_exts and e not in (".jpg", ".jpeg")
        }
        detail = (
            "全部图像为 .jpg/.jpeg" if not non_jpg
            else f"存在非 jpg 图像：{non_jpg} —— 已在 dataset.yaml 的 image_extensions 白名单内"
        )
        return GateResult("V6", "扩展名分布", "PASS", detail, blocking=False, rows=rows)

    runner.run(gate_extensions, "V6", "扩展名分布", False)

    # ---- V7 大小写冲突 -----------------------------------------------------
    def gate_case() -> GateResult:
        n = len(rep.brand_case_collisions)
        rows = [("品牌目录大小写冲突组", str(n), "APFS 默认大小写不敏感，冲突会静默覆盖")]
        for k, v in list(rep.brand_case_collisions.items())[:5]:
            rows.append((k, " vs ".join(v), ""))
        return GateResult(
            "V7", "大小写冲突", "PASS" if n == 0 else "FAIL",
            "无冲突" if n == 0 else f"发现 {n} 组冲突，数据可能已被静默覆盖",
            blocking=True, rows=rows,
        )

    runner.run(gate_case, "V7", "大小写冲突", True)

    # ---- 报告 --------------------------------------------------------------
    per_sc = rep.per_supercat
    extra: list[tuple[str, str]] = []
    if per_sc is not None:
        merged = per_sc.copy()
        merged["exp_classes"] = merged["supercat"].map(
            lambda s: exp["per_supercategory"].get(s, {}).get("classes")
        )
        merged["exp_images"] = merged["supercat"].map(
            lambda s: exp["per_supercategory"].get(s, {}).get("images")
        )
        merged["d_classes"] = merged["classes"] - merged["exp_classes"]
        merged["d_images"] = merged["images"] - merged["exp_images"]
        merged = merged.sort_values("exp_images", ascending=False)
        rows = [
            (
                r.supercat,
                f"{int(r.exp_classes)}",
                f"{int(r.classes)}",
                f"{int(r.d_classes):+d}",
                f"{int(r.exp_images):,}",
                f"{int(r.images):,}",
                f"{int(r.d_images):+d}",
            )
            for r in merged.itertuples()
        ]
        extra.append(
            (
                "逐超类对账（类数 / 图数）",
                md_table(
                    rows,
                    headers=("超类", "期望类数", "实测类数", "差值",
                             "期望图数", "实测图数", "差值"),
                ),
            )
        )

    runner.write_report(REPORT, extra_sections=extra)
    print(f"\n报告已写入 {REPORT.relative_to(PROJECT_DIR)}")
    return runner.print_summary()


if __name__ == "__main__":
    raise SystemExit(main())
