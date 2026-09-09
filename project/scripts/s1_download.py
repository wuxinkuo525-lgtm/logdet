#!/usr/bin/env python
"""S1 步骤一：下载 + 校验 + 安全解压。

幂等：zip 已存在且尺寸合理就跳过下载；数据集目录已存在且非空就跳过解压。

用法：
    python scripts/s1_download.py                # 下载 + 解压
    python scripts/s1_download.py --skip-unzip   # 只下载
    python scripts/s1_download.py --force-unzip  # 强制重解（先把旧目录重命名让路）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from logodet.config import load_yaml  # noqa: E402
from logodet.ingest.download import download_dataset  # noqa: E402
from logodet.ingest.unzip_safe import safe_extract, verify_zip  # noqa: E402
from logodet.paths import P  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="S1 下载与解压")
    ap.add_argument("--skip-unzip", action="store_true", help="只下载，不解压")
    ap.add_argument("--force-unzip", action="store_true", help="即使已解压也重解一次")
    ap.add_argument("--force-download", action="store_true", help="忽略已有 zip 重新下载")
    ap.add_argument(
        "--skip-crc",
        action="store_true",
        help="跳过 CRC 校验。testzip() 会把 317k 个条目全部解压一遍验 CRC，"
        "很慢；ditto 解压时本身也会校验 CRC，故重跑时可跳过",
    )
    args = ap.parse_args()

    cfg = load_yaml("dataset.yaml")
    parse_cfg = cfg["parse"]
    ignore = set(parse_cfg.get("ignore_names") or [])
    top_dir = parse_cfg["archive_top_dir"]

    print("=" * 78)
    print(" S1 步骤一：下载 + 校验 + 解压")
    print("=" * 78)

    # ---- 下载 --------------------------------------------------------------
    print("\n[1/3] 获取 zip ...")
    t0 = time.time()
    res = download_dataset(force=args.force_download)
    print(f"      路径   : {res.path}")
    print(f"      大小   : {res.size_bytes:,} B ({res.size_bytes / 2**30:.2f} GiB)")
    print(f"      sha256 : {res.sha256}")
    print(f"      来源   : {'复用已有文件' if res.skipped else '本次下载'}")

    # ---- CRC 校验 ----------------------------------------------------------
    if args.skip_crc:
        print("\n[2/3] --skip-crc 指定，跳过 CRC 校验（ditto 解压时仍会自行校验）")
    else:
        print("\n[2/3] zip 完整性校验（CRC）...")
        print("      testzip() 会把全部 317k 条目解压一遍，预计数分钟")
        t1 = time.time()
        bad = verify_zip(res.path)
        if bad is not None:
            print(f"      FAIL: 损坏条目 {bad!r}")
            print("      处置：删除 zip 后重新下载（curl -C - 的续传可能拼接了坏块）")
            return 1
        print(f"      PASS: testzip() 无损坏条目（{time.time() - t1:.1f}s）")

    # ---- 解压 --------------------------------------------------------------
    # dataset root 是 .../data/LogoDet-3K，而 zip 内顶层目录也叫 LogoDet-3K，
    # 所以解压目标要用 dataset 的父目录，避免出现 data/LogoDet-3K/LogoDet-3K。
    dest = P.dataset.parent
    already = P.dataset.is_dir() and any(P.dataset.iterdir())

    if args.skip_unzip:
        print("\n[3/3] --skip-unzip 指定，跳过解压")
        return 0

    if already and not args.force_unzip:
        n = sum(1 for _ in P.dataset.iterdir())
        print(f"\n[3/3] {P.dataset} 已存在且非空（{n} 个一级条目），跳过解压")
        print("      如需重解：--force-unzip")
        return 0

    if already and args.force_unzip:
        stale = P.dataset.with_name(f"{P.dataset.name}.stale.{int(time.time())}")
        print(f"\n      已有数据重命名让路 → {stale.name}")
        # 用 mv 而不是 rm：本机删除守卫会拦大批量 unlink
        P.dataset.rename(stale)

    print(f"\n[3/3] 解压到 {dest} ...")
    t2 = time.time()
    rep = safe_extract(res.path, dest, verbose=True)
    dt = time.time() - t2

    print()
    for k, v in rep.as_rows():
        print(f"      {k:<18} {v}")
    print(f"      耗时               {dt:.1f}s")

    if rep.recovered_names:
        print("\n      编码还原样例（前 5 条）：")
        for old, new in rep.recovered_names[:5]:
            print(f"        {old!r} → {new!r}")

    pf = rep.preflight
    meta_path = P.artifact("tables") / "unzip_report.json"
    meta_path.write_text(
        json.dumps(
            {
                "zip_sha256": res.sha256,
                "zip_size_bytes": res.size_bytes,
                "dest": str(dest),
                "top_dir": top_dir,
                "path_used": rep.path_used,
                "elapsed_sec": round(dt, 1),
                "total_entries": rep.total_entries,
                "extracted_files": rep.extracted_files,
                "recovered_name_count": len(rep.recovered_names),
                "preflight": {
                    "utf8_flagged": pf.utf8_flagged if pf else None,
                    "risky_name_count": len(pf.risky_names) if pf else None,
                    "case_collision_count": len(pf.case_collisions) if pf else None,
                    "path_traversal_count": len(pf.path_traversal) if pf else None,
                    "top_dirs": pf.top_dirs if pf else None,
                },
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n      报告已写入 {meta_path}")
    print(f"\n总耗时 {time.time() - t0:.1f}s。下一步：python scripts/s1_verify_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
