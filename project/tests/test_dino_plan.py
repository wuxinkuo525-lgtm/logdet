"""实验计划对账（logodet.dino_run.check_plan）：同名输出目录只能装同一个实验。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logodet.dino_run import PlanMismatch, check_plan  # noqa: E402

PLAN = {"mode": "shared", "max_iters": 15000, "lr": 1e-4, "batch_size": 4,
        "train_scales": [(384, 800), (416, 800)], "param_scheduler": [{"type": "MultiStepLR", "milestones": []}]}


class CheckPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "shared_b4_long"

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_launch_records_plan(self):
        self.assertEqual(check_plan(self.run_dir, PLAN), "new")
        saved = json.loads((self.run_dir / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["plan"]["max_iters"], 15000)
        self.assertFalse(saved["adopted_without_check"])

    def test_same_plan_matches_even_with_tuples(self):
        check_plan(self.run_dir, PLAN)
        # 第二次传进来的还是元组，落盘读回来是列表——不能因此判成不一致
        self.assertEqual(check_plan(self.run_dir, dict(PLAN)), "match")

    def test_longer_schedule_on_finished_run_is_refused(self):
        check_plan(self.run_dir, PLAN)
        (self.run_dir / "DONE").write_text("x", encoding="utf-8")
        with self.assertRaises(PlanMismatch) as ctx:
            check_plan(self.run_dir, {**PLAN, "max_iters": 20000})
        self.assertIn("max_iters", str(ctx.exception))
        self.assertIn("15000", str(ctx.exception))
        # 被拒绝时不能改写原记录
        self.assertEqual(json.loads((self.run_dir / "plan.json").read_text(encoding="utf-8"))["plan"]["max_iters"], 15000)

    def test_changed_variant_is_refused(self):
        check_plan(self.run_dir, PLAN)
        with self.assertRaises(PlanMismatch):
            check_plan(self.run_dir, {**PLAN, "train_scales": [(512, 1000), (544, 1000)]})

    def test_explicit_accept_rewrites_and_keeps_history(self):
        check_plan(self.run_dir, PLAN)
        self.assertEqual(check_plan(self.run_dir, {**PLAN, "max_iters": 20000}, accept_change=True), "changed")
        saved = json.loads((self.run_dir / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["plan"]["max_iters"], 20000)
        self.assertEqual(saved["changed_from"]["max_iters"], 15000)
        self.assertEqual(check_plan(self.run_dir, {**PLAN, "max_iters": 20000}), "match")

    def test_run_from_before_the_check_is_adopted_and_flagged(self):
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "iter_2000.pth").write_bytes(b"x")
        self.assertEqual(check_plan(self.run_dir, PLAN), "legacy")
        self.assertTrue(json.loads((self.run_dir / "plan.json").read_text(encoding="utf-8"))["adopted_without_check"])
        # 补记之后就按正常规则核对
        with self.assertRaises(PlanMismatch):
            check_plan(self.run_dir, {**PLAN, "lr": 2e-4})


if __name__ == "__main__":
    unittest.main()
