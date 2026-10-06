"""CPU-only regression tests for portable artifacts and interrupted final validation."""
import ast
import contextlib
import io
import json
import os
import runpy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from logodet import dino_run as run


class DinoRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=PROJECT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ckpt = self.root / 'iter_20000.pth'
        self.ckpt.write_bytes(b'checkpoint fixture')
        (self.root / 'last_checkpoint').write_text(str(self.ckpt), encoding='utf-8')
        self.runner = SimpleNamespace(
            work_dir=str(self.root), iter=20000, max_iters=20000, seed=20260917,
            val_loop=SimpleNamespace(run=Mock(return_value={'agn/AP': .3})))
        self.loader = patch.object(run, 'checkpoint_iteration', return_value=20000)
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_annotation_bundle_survives_move_without_source(self):
        source = self.root / 'source.json'
        gt = {'images': [{'id': 3}], 'annotations': [], 'categories': []}
        source.write_text(json.dumps(gt), encoding='utf-8')
        bundle = self.root / 'cluster'; bundle.mkdir()
        name = run.export_eval_ann(source, bundle)
        local = self.root / 'download'; shutil.copytree(bundle, local)
        source.unlink()
        self.assertFalse(Path(name).is_absolute())
        self.assertEqual(json.loads(run.resolve_eval_ann({'own_eval_ann': name}, local).read_text()), gt)

    def test_legacy_local_absolute_annotation(self):
        source = self.root / 'old.json'; source.write_text('{}')
        self.assertEqual(run.resolve_eval_ann({'own_eval_ann': str(source)}, self.root), source)
        with self.assertRaises(FileNotFoundError):
            run.resolve_eval_ann({'own_eval_ann': '/home/cluster/missing.json'}, self.root)

    def test_interrupted_final_validation_is_recovered(self):
        run.finish_evaluation(self.runner, (15000, {'agn/AP': .9}))
        self.runner.val_loop.run.assert_called_once()
        receipt = run.validate_final_eval(self.root)
        self.assertEqual(receipt['eval_step'], 20000)
        self.assertEqual(receipt['metrics']['agn/AP'], .3)
        run.finish_evaluation(self.runner, None)
        self.runner.val_loop.run.assert_called_once()  # already certified: no duplicate

    def test_normal_training_does_not_repeat_validation(self):
        run.finish_evaluation(self.runner, (20000, {'agn/AP': .4}))
        self.runner.val_loop.run.assert_not_called()
        self.assertEqual(run.validate_final_eval(self.root)['metrics']['agn/AP'], .4)

    def test_failure_during_validation_never_certifies_run(self):
        self.runner.val_loop.run.side_effect = RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):
            run.finish_evaluation(self.runner, None)
        self.assertFalse((self.root / 'final_eval.json').exists())

    def test_stale_step_or_replaced_checkpoint_rejected(self):
        run.finish_evaluation(self.runner, None)
        path = self.root / 'final_eval.json'
        receipt = json.loads(path.read_text()); receipt['eval_step'] = 15000
        run.write_json_atomic(path, receipt)
        with self.assertRaises(ValueError): run.validate_final_eval(self.root)
        receipt['eval_step'] = 20000; run.write_json_atomic(path, receipt)
        self.ckpt.write_bytes(b'replaced checkpoint')
        with self.assertRaises(ValueError): run.validate_final_eval(self.root)

    def test_unfinished_checkpoint_rejected(self):
        with patch.object(run, 'checkpoint_iteration', return_value=18000):
            with self.assertRaises(ValueError): run.finish_evaluation(self.runner, None)
        self.runner.val_loop.run.assert_not_called()

    def test_undefined_auxiliary_metric_is_null(self):
        run.finish_evaluation(self.runner, (20000, {'agn/AP': .3, 'agn/AP_small': float('nan')}))
        self.assertIsNone(run.validate_final_eval(self.root)['metrics']['agn/AP_small'])
        with self.assertRaises(ValueError): run.validate_final_eval(self.root, 'agn/AP_small')

    def test_nonfinite_primary_metric_rejected(self):
        with self.assertRaises(ValueError):
            run.finish_evaluation(self.runner, (20000, {'agn/AP': float('nan')}))
        self.assertFalse((self.root / 'final_eval.json').exists())

    def test_truncated_log_tail_warns_but_corrupt_middle_fails(self):
        p = self.root / 'scalars.json'
        p.write_text('{"loss": 1}\n{"loss":', encoding='utf-8')
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(list(run.scalar_records(p)), [{'loss': 1}])
        p.write_text('{bad}\n{"loss": 1}\n', encoding='utf-8')
        with self.assertRaises(json.JSONDecodeError): list(run.scalar_records(p))

    def make_bundle(self):
        import numpy as np
        arrays = {'processed_image_ids': np.array([1, 2])}
        for kind in ('agnostic', 'classwise'):
            arrays.update({f'{kind}_box': np.array([[0., 0., 1., 1.]]),
                           f'{kind}_score': np.array([.5]), f'{kind}_image_id': np.array([1]),
                           f'{kind}_category_id': np.array([1])})
        np.savez(self.root / 'predictions.npz', **arrays)
        (self.root / 'own_eval_ann.json').write_text('{}')
        run.finish_evaluation(self.runner, None)
        receipt = run.validate_final_eval(self.root)
        man = dict(schema_version=2, status='complete', partial=False, n_images=2,
                   final_evaluation=receipt, checkpoint_identity=receipt['checkpoint'],
                   conditions={'num_classes': 3000},
                   sha256={n: run.file_sha256(self.root / n) for n in ('predictions.npz', 'own_eval_ann.json')})
        run.write_json_atomic(self.root / 'manifest.json', man)
        return man

    def test_report_inventory_includes_zero_detection_images(self):
        self.make_bundle()
        run.validate_prediction_bundle(self.root, {1, 2})
        with self.assertRaises(ValueError): run.validate_prediction_bundle(self.root, {1, 2, 3})

    def test_rehearsal_and_corrupt_download_are_rejected(self):
        man = self.make_bundle()
        man['partial'] = True
        run.write_json_atomic(self.root / 'manifest.json', man)
        with self.assertRaises(ValueError): run.validate_prediction_bundle(self.root)
        run.validate_prediction_bundle(self.root, allow_partial=True)
        (self.root / 'predictions.npz').write_bytes(b'truncated')
        with self.assertRaises(ValueError): run.validate_prediction_bundle(self.root, allow_partial=True)

    def test_watchdog_detects_stale_progress(self):
        ns = runpy.run_path(str(PROJECT / 'scripts/s9_dino_watch.py'))
        p = self.root / 'progress.json'
        self.assertTrue(ns['snapshot'](p, 100, 900, now=1001)['stale'])
        run.write_json_atomic(p, {'time': 999, 'phase': 'train', 'iteration': 50})
        self.assertFalse(ns['snapshot'](p, 100, 900, now=1001)['stale'])

    def test_health_hook_rejects_nan_and_cpu_gpu_job(self):
        import numpy as np
        tree = ast.parse((PROJECT / 'src/logodet/dino/hooks.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TrainingHealthHook')
        cls.decorator_list = []
        ns = {'Hook': object, 'os': os, 'progress': Mock(),
              'torch': SimpleNamespace(isfinite=np.isfinite, as_tensor=np.asarray, tensor=np.asarray)}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), '<health hook>', 'exec'), ns)
        hook = ns['TrainingHealthHook']()
        with self.assertRaises(FloatingPointError):
            hook.after_train_iter(self.runner, 0, outputs={'loss': float('nan')})
        pred = SimpleNamespace(scores=np.array([float('inf')]))
        with self.assertRaises(FloatingPointError):
            hook.after_val_iter(self.runner, 0, outputs=[SimpleNamespace(pred_instances=pred)])
        model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type='cpu'))]))
        self.runner.model = model
        with patch.dict(os.environ, LOGDET_REQUIRE_GPU='1'):
            with self.assertRaises(RuntimeError): hook.before_train(self.runner)

    def test_picker_requires_final_receipt_even_with_old_logs(self):
        runs = self.root / 'runs'
        work = runs / 'dino' / 'trial'; work.mkdir(parents=True)
        logdir = work / '20261005_120000' / 'vis_data'; logdir.mkdir(parents=True)
        (logdir / 'scalars.json').write_text('{"agn/AP": 0.9, "step": 15000}\n')
        out = self.root / 'best.json'
        argv = ['s9_dino_pick_best.py', '--runs', 'trial', '--out', str(out)]
        def invoke():
            with patch.dict(os.environ, LOGDET_RUNS_ROOT=str(runs)), patch.object(sys, 'argv', argv):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as exit_info:
                        runpy.run_path(str(PROJECT / 'scripts/s9_dino_pick_best.py'), run_name='__main__')
            return exit_info.exception.code
        self.assertEqual(invoke(), 1)
        self.assertFalse(out.exists())
        (work / 'last_checkpoint').write_text(str(self.ckpt))
        self.runner.work_dir = str(work)
        run.finish_evaluation(self.runner, None)
        self.assertEqual(invoke(), 0)
        self.assertEqual(json.loads(out.read_text())['runs']['trial']['agn/AP'], .3)
        self.assertEqual(invoke(), 0)  # frozen valid choice can be reused

    def test_seed_default_and_override_applied_to_runner_config(self):
        # Execute the real configuration function without importing the GPU stack.
        tree = ast.parse((PROJECT / 'scripts/s9_dino_train.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'apply_hparams')
        ns = {'Config': object, 'SUBSET_SEED': 20260917}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<apply_hparams>', 'exec'), ns)
        class Config(dict):
            def __setattr__(self, key, value): self[key] = value
        cfg = Config(randomness={'deterministic': True})
        ns['apply_hparams'](cfg, {}, False)
        self.assertEqual(cfg['randomness'], {'deterministic': True, 'seed': 20260917})
        ns['apply_hparams'](cfg, {'seed': 0}, False)
        self.assertEqual(cfg['randomness']['seed'], 0)


if __name__ == '__main__':
    unittest.main()
