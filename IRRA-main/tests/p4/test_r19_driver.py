"""Synthetic R19 runner checks. No datasets, checkpoints or GPU are read."""
from pathlib import Path
import collections
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).parent))
from tools.p4.r19 import driver


def make_s0_fixture(root, python=sys.executable):
    """Create the two *frozen* S0 references and a JSON input spec on demand.

    This utility is deliberately not run by the fast unit tests. The integration
    rehearsal calls it once, then launches p4_r19.sh S0 against the returned spec.
    """
    import test_r17 as fixture
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in ('p4_r15_out', 'p4_r17_out', 'p4_r18_out'):
        (root / name).mkdir(exist_ok=True)
    rng = np.random.default_rng(17)
    dim = 8
    blank = (rng.standard_normal(dim), rng.standard_normal(dim))
    feats = root / 'inputs'
    feats.mkdir()
    dev = fixture.make_archive(rng, 14, 100000, dim, blank)
    test = fixture.make_archive(rng, 10, 800000, dim, blank)
    train = fixture.make_archive(rng, 20, 900000, dim, blank)
    np.savez(feats / 'dev.npz', **dev)
    np.savez(feats / 'test.npz', **test)
    np.savez(feats / 'train_ir.npz', f_ir=train['clean/f_ir'], f_rgb=train['clean/f_rgb'],
             raw_pids=train['clean/query_raw_pids'][::2][train['clean/img_pids']])
    spec = {'seeds': [101], 'fits': [7001], 'assign_seeds': [7031], 'grid': 'dev',
            'no_identity_test': 'test_r18_eval.CameraTests', 'datasets': {}}
    for dataset, parent in [('rgbt', 'p4_r17_out'), ('rgbnt', 'p4_r18_out')]:
        reference = root / parent / 'const/s101_f7001'
        reference.mkdir(parents=True)
        script = CODE / 'tools/p4/transfer/r18_xcam.py'
        command = [str(python), '-B']
        if dataset == 'rgbnt':
            command += [str(CODE / 'tools/p4/r18/run_frozen.py'), '--expect_sha', driver.sha256(script),
                        '--log', str(reference / 'degenerate_fits.json'), '--']
        command += [str(script), '--b12', '.', '--feats', str(feats), '--base', 'dev', '--adapter', 'dev',
                    '--assign', 'none', '--out', str(reference), '--tag', 'reference_s101_f7001',
                    '--assign_seeds', '7031', '--mixes', 'M0', 'M4', '--skip_joint',
                    '--methods', 'B*', 'QE-DBA', 'QE+M', '--grid_preset', 'dev',
                    '--select_rule', 'constrained', '--r_max', '0.999999', '--tie_break', 'bstar',
                    '--test_npz', str(feats / 'test.npz'), '--link_map', 'train_lda',
                    '--train_ir_npz', str(feats / 'train_ir.npz'), '--test_seed', '7001', '--perq_ext']
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (reference / 'reference.log').open('wb') as log:
            subprocess.run(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        spec['datasets'][dataset] = {'101': {'dev': str(feats / 'dev.npz'),
            'test': str(feats / 'test.npz'), 'train_ir': str(feats / 'train_ir.npz'),
            'selection': str(driver.unique(reference, '*_selection.json')),
            'regression': str(reference), 'guarded': dataset == 'rgbnt', 'cams': None}}
    path = root / 's0_spec.json'
    driver.save(path, spec)
    return path


class ManifestTests(unittest.TestCase):
    def fixture(self, root):
        for name in ('p4_r15_out', 'p4_r17_out', 'p4_r18_out'):
            p = root / name
            p.mkdir()
            (p / 'protected.txt').write_text(name, encoding='utf-8')
        with patch.object(driver, 'source_signature', return_value={'fixture-code': 'unchanged'}):
            return driver.Runner(root)

    def test_byte_equal_manifest_reuses_and_changed_input_is_retained_stale(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner = self.fixture(root)
            source = root / 'input.txt'
            source.write_text('first', encoding='utf-8')
            output = runner.out / 's0/rgbt/unit'
            command = [sys.executable, '-B', '-c',
                       'from pathlib import Path;import sys;Path(sys.argv[1]).write_text(Path(sys.argv[2]).read_text())',
                       str(output / 'result.txt'), str(source)]
            runner.unit('one', output, [source], [command], {'seed': 101})
            before = (output / 'result.txt').stat().st_mtime_ns
            manifest = (output / 'manifest.json').read_bytes()
            runner.unit('one', output, [source], [command], {'seed': 101})
            self.assertEqual((output / 'result.txt').stat().st_mtime_ns, before)
            self.assertEqual((output / 'manifest.json').read_bytes(), manifest)
            self.assertEqual(runner.events[-1]['status'], 'reused')
            source.write_text('second', encoding='utf-8')
            runner.unit('one', output, [source], [command], {'seed': 101})
            self.assertEqual((output / 'result.txt').read_text(), 'second')
            stale = list((runner.out / '_stale').glob('one_*'))
            self.assertEqual(len(stale), 1)
            self.assertEqual((stale[0] / 'result.txt').read_text(), 'first')
            self.assertEqual((stale[0] / 'manifest.json').read_bytes(), manifest)

    def test_changed_output_is_not_reused_and_failure_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = self.fixture(Path(temp))
            output = runner.out / 's0/unit'
            command = [sys.executable, '-B', '-c', 'print("done")']
            runner.unit('one', output, [], [command])
            (output / 'driver.log').write_text('tampered', encoding='utf-8')
            runner.unit('one', output, [], [command])
            self.assertEqual(len(list((runner.out / '_stale').iterdir())), 1)
            with self.assertRaises(subprocess.CalledProcessError):
                runner.unit('fail', runner.out / 's0/fail', [],
                            [[sys.executable, '-B', '-c', 'raise SystemExit(4)'], command])
            self.assertFalse((runner.out / 's0/fail/.done').exists())
            self.assertEqual(json.loads((runner.out / 's0/fail/driver_run_record.json').read_text())['exit_codes'], [4])

    def test_guard_checks_all_three_old_outputs_and_write_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'must remain'):
                runner.inside(root / 'p4_r18_out/new')
            for directory in runner.protected:
                p = directory / 'protected.txt'
                original = p.read_bytes()
                p.write_bytes(original + b'changed')
                with self.assertRaisesRegex(ValueError, 'Protected output changed'):
                    runner.guard()
                p.write_bytes(original)
            runner.guard()
            p = runner.protected[0] / 'protected.txt'
            p.write_text('changed-at-finish', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Protected output changed'):
                runner.finish('S0', 'COMPLETE')
            self.assertEqual((runner.out / 'STATUS_S0').read_text(), 'STOPPED: PROTECTED_OUTPUT_CHANGED\n')
            self.assertTrue((runner.out / 'DIGEST19_S0.txt').is_file())


class PlanningTests(unittest.TestCase):
    def test_pretrained_cache_is_checked_without_creating_or_replacing_it(self):
        with tempfile.TemporaryDirectory() as temp:
            r = Path(temp)
            config = r / 'config.yaml'
            config.write_text('pretrain_choice: ViT-B/16\n', encoding='utf-8')
            cache = r / 'cache'
            content = b'synthetic pretrained bytes'
            import hashlib
            digest = hashlib.sha256(content).hexdigest()
            models = {'ViT-B/16': 'https://example.invalid/' + digest + '/ViT-B-16.pt'}
            with self.assertRaises(FileNotFoundError):
                driver.pretrained_input(config, cache=cache, models=models)
            self.assertFalse(cache.exists())
            cache.mkdir()
            path = cache / 'ViT-B-16.pt'
            path.write_bytes(content)
            result = driver.pretrained_input(config, cache=cache, models=models)
            self.assertEqual(result['sha256'], digest)
            self.assertEqual(path.read_bytes(), content)
            path.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                driver.pretrained_input(config, cache=cache, models=models)
            self.assertEqual(path.read_bytes(), b'bad')

    def test_plans_three_seeds_three_fits_and_expected_methods(self):
        spec = driver.default_specs('/root/autodl-tmp')
        self.assertEqual(spec['seeds'], [101, 102, 103])
        self.assertEqual(spec['fits'], [7001, 7002, 7003])
        self.assertEqual(spec['assign_seeds'], [7031, 7032, 7033, 7034, 7035])
        self.assertEqual(len(driver.MIXES), 13)
        self.assertEqual(set(spec['datasets']), {'rgbt', 'rgbnt', 'rgbnt_subset'})
        self.assertEqual(spec['no_identity_test'], 'test_r18_eval.CameraTests')
        self.assertIn('QE+SRC2f+pI', driver.METHODS4)

    def test_rgbnt_cameras_are_sliced_by_preserved_raw_identity_order(self):
        with tempfile.TemporaryDirectory() as temp:
            r = Path(temp)
            full = {'clean/txt_pids': np.array([0, 1, 2]), 'clean/query_raw_pids': np.array([20, 30, 40]),
                    'clean/img_pids': np.array([0, 0, 1, 2, 2])}
            subset = {'clean/txt_pids': np.array([0, 1]), 'clean/query_raw_pids': np.array([20, 40]),
                      'clean/img_pids': np.array([0, 0, 1, 1])}
            np.savez(r / 'full.npz', **full)
            np.savez(r / 'subset.npz', **subset)
            np.savez(r / 'cams.npz', dev_cams=['cam1', 'cam2'], test_cams=['cam1', 'cam2', 'cam3', 'cam4', 'cam5'])
            driver.subset_cameras(r / 'full.npz', r / 'subset.npz', r / 'cams.npz', r / 'out.npz')
            with np.load(r / 'out.npz') as z:
                np.testing.assert_array_equal(z['test_cams'], ['cam1', 'cam2', 'cam4', 'cam5'])
                np.testing.assert_array_equal(z['dev_cams'], ['cam1', 'cam2'])

    def test_m0_requires_exact_metrics_and_both_methods(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'r12_unit_cells.csv'
            head = 'evalset,mixture,aseed,method,mAP,R1\n'
            rows = 'test,M0,7031,QE-DBA,10.0,20.0\ntest,M0,7031,QE+M,10.0,20.0\n'
            p.write_text(head + rows, encoding='utf-8')
            self.assertEqual(driver.check_m0(p.parent), {'m0_pairs': 1, 'equal': True})
            p.write_text(head + rows.replace('QE+M,10.0', 'QE+M,10.1'), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'M0 mismatch'):
                driver.check_m0(p.parent)


if __name__ == '__main__':
    unittest.main(verbosity=2)
