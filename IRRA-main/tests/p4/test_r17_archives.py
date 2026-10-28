"""Acceptance cases of r17_check_archives.py on tiny synthetic archive pairs (no real data).

The archives follow the exporter's layout (pack_feats.py): a degraded-state array is stored only when it is not bit-identical to clean,
so a valid archive holds just the target modality of every degraded state. Each case changes one thing in a copy and checks the exit
status (0 = no problem, 2 = problem) and the part named in the message. Cases 2 and 3 are the ones the v2 check missed: it combined the
development and held-out parts with all(), so a non-target change in only one part passed.
Run: python -m pytest IRRA-main/tests/p4/test_r17_archives.py -q   (or python IRRA-main/tests/p4/test_r17_archives.py)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
CHECK = CODE_ROOT / "tools" / "p4" / "transfer" / "r17_check_archives.py"
TARGET = {'rgb_missing': 'f_rgb', 'ir_missing': 'f_ir', 'rgb_noise_010': 'f_rgb', 'ir_noise_010': 'f_ir', 'rgb_blur_k7': 'f_rgb', 'ir_blur_k7': 'f_ir'}


def archive(seed, raw0, n_ids=12, dim=16):
    rng = np.random.default_rng(seed)
    img = np.repeat(np.arange(n_ids), 3); txt = np.repeat(np.arange(n_ids), 2)
    f_rgb = rng.standard_normal((len(img), dim)).astype(np.float32); f_ir = rng.standard_normal((len(img), dim)).astype(np.float32)
    z = {'clean/text_features': rng.standard_normal((len(txt), dim)).astype(np.float32), 'clean/txt_pids': txt.astype(np.int64),
         'clean/img_pids': img.astype(np.int64), 'clean/query_raw_pids': (raw0 + txt).astype(np.int64), 'clean/f_rgb': f_rgb, 'clean/f_ir': f_ir}
    for s, k in TARGET.items():
        base = z[f'clean/{k}']
        z[f'{s}/{k}'] = (np.tile(rng.standard_normal(dim), (len(img), 1)) if 'missing' in s else base + 0.1 * rng.standard_normal(base.shape)).astype(np.float32)
    return z


def run(args, cwd):
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
    return subprocess.run([sys.executable, '-B'] + [str(a) for a in args], cwd=cwd, capture_output=True, text=True, encoding='utf-8',
                          errors='replace', env=env)


class R17Archives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.T = Path(cls.tmp.name); cls.n = 0

    @classmethod
    def tearDownClass(cls):
        try:
            cls.tmp.cleanup()
        except OSError:   # Windows can still hold a handle on a temporary file; leftovers in the temp folder are harmless
            pass

    def check(self, dev=None, test=None, extra_seed=False):
        """write one archive pair (edited by the dev/test callbacks), run the check, return (exit status, output, json report)"""
        R17Archives.n += 1; d = self.T / f'c{self.n}'; d.mkdir()
        zd, zt = archive(1, 1000), archive(2, 5000)
        if dev: dev(zd)
        if test: test(zt)
        np.savez(d / 'dev.npz', **zd); np.savez(d / 'test.npz', **zt)
        pairs = [f's101={d / "dev.npz"},{d / "test.npz"}']
        if extra_seed:   # a second, valid seed: a problem in one seed must still fail the whole check
            np.savez(d / 'dev2.npz', **archive(3, 1000)); np.savez(d / 'test2.npz', **archive(4, 5000))
            pairs.append(f's102={d / "dev2.npz"},{d / "test2.npz"}')
        p = run([CHECK, '--pairs'] + pairs + ['--assign_seeds', '7031', '--json', d / 'r.json'], CODE_ROOT)
        rep = json.loads((d / 'r.json').read_text(encoding='utf-8')) if (d / 'r.json').exists() else None
        return p.returncode, p.stdout + p.stderr, rep

    def status_line(self, out):
        return [x for x in out.splitlines() if x.startswith('[status]')][-1]

    def test_01_valid_pair(self):
        rc, out, rep = self.check()
        self.assertEqual(rc, 0, out); self.assertIn('[status] OK', out); self.assertEqual(rep['s101']['problems'], [])
        row = [r for r in rep['s101']['rows'] if r['key'] == 'rgb_missing/f_ir'][0]
        self.assertEqual(row['read_as'], 'clean/f_ir (fallback)'); self.assertEqual(row['parts']['test']['rows_changed'], 0.0)

    def test_02_unexpected_change_in_development_part_only(self):
        def dev(z): z['ir_noise_010/f_rgb'] = z['clean/f_rgb'] + np.float32(0.01)
        def test(z): z['ir_noise_010/f_rgb'] = z['clean/f_rgb'].copy()      # stored in both archives, so test mode reads it in both parts
        rc, out, rep = self.check(dev, test)
        self.assertEqual(rc, 2, out); line = self.status_line(out)
        self.assertIn('ir_noise_010/f_rgb: UNEXPECTED CHANGE (dev', line); self.assertNotIn('(test', line)

    def test_03_unexpected_change_in_heldout_part_only(self):
        def dev(z): z['rgb_blur_k7/f_ir'] = z['clean/f_ir'].copy()
        def test(z):
            z['rgb_blur_k7/f_ir'] = z['clean/f_ir'].copy(); z['rgb_blur_k7/f_ir'][4] += np.float32(1e-3)   # one row only
        rc, out, rep = self.check(dev, test)
        self.assertEqual(rc, 2, out); line = self.status_line(out)
        self.assertIn('rgb_blur_k7/f_ir: UNEXPECTED CHANGE (test: 1 of 36 rows differ', line); self.assertNotIn('(dev', line)

    def test_04_non_target_stored_but_identical_is_fine(self):
        def both(z): z['rgb_missing/f_ir'] = z['clean/f_ir'].copy()
        rc, out, rep = self.check(both, both)
        self.assertEqual(rc, 0, out)
        row = [r for r in rep['s101']['rows'] if r['key'] == 'rgb_missing/f_ir'][0]; self.assertTrue(row['in_merged'])

    def test_05_target_unchanged_in_heldout_part_only(self):
        def test(z): z['rgb_noise_010/f_rgb'] = z['clean/f_rgb'].copy()
        rc, out, rep = self.check(None, test)
        self.assertEqual(rc, 2, out); line = self.status_line(out)
        self.assertIn('rgb_noise_010/f_rgb: MISSING CHANGE (test: identical to clean)', line); self.assertNotIn('(dev', line)

    def test_06_target_key_missing_from_heldout_archive_falls_back_in_both_parts(self):
        def test(z): del z['ir_missing/f_ir']
        rc, out, rep = self.check(None, test)
        self.assertEqual(rc, 2, out); line = self.status_line(out)
        self.assertIn('ir_missing/f_ir: MISSING CHANGE (dev: identical to clean, key absent -> clean fallback)', line)
        self.assertIn('MISSING CHANGE (test: identical to clean, key absent -> clean fallback)', line)

    def test_07_partial_change_of_missing_or_noise_state(self):
        def dev(z): z['ir_missing/f_ir'][0] = z['clean/f_ir'][0]
        rc, out, rep = self.check(dev)
        self.assertEqual(rc, 2, out); self.assertIn('ir_missing/f_ir: PARTIAL CHANGE (dev: 1 of 36 rows equal clean)', self.status_line(out))

    def test_08_partial_change_of_blur_is_reported_not_failed(self):
        def test(z): z['rgb_blur_k7/f_rgb'][:5] = z['clean/f_rgb'][:5]
        rc, out, rep = self.check(None, test)
        self.assertEqual(rc, 0, out)
        row = [r for r in rep['s101']['rows'] if r['key'] == 'rgb_blur_k7/f_rgb'][0]
        self.assertAlmostEqual(row['parts']['test']['rows_changed'], 31 / 36)

    def test_09_non_finite_values(self):
        def test(z): z['rgb_noise_010/f_rgb'][2, 3] = np.nan
        rc, out, rep = self.check(None, test)
        self.assertEqual(rc, 2, out); self.assertIn('rgb_noise_010/f_rgb: NON-FINITE (test: 1 values)', self.status_line(out))
        def dev(z): z['clean/f_ir'][0, 0] = np.inf
        rc, out, rep = self.check(dev)
        self.assertEqual(rc, 2, out); self.assertIn('dev: clean/f_ir has non-finite values', self.status_line(out))

    def test_10_shape_mismatch(self):
        def dev(z): z['rgb_missing/f_rgb'] = z['rgb_missing/f_rgb'][:-1]
        rc, out, rep = self.check(dev)
        self.assertEqual(rc, 2, out); self.assertIn('rgb_missing/f_rgb: SHAPE [35, 16] vs clean [36, 16] (dev)', self.status_line(out))
        def test(z): z['clean/text_features'] = z['clean/text_features'][:, :8]
        rc, out, rep = self.check(None, test)
        self.assertEqual(rc, 2, out); self.assertIn('test: dimensions differ (text 8, f_rgb 16, f_ir 16)', self.status_line(out))

    def test_11_unused_state_is_not_required(self):
        def both(z): del z['ir_blur_k7/f_ir']
        rc, out, rep = self.check(both, both)
        self.assertEqual(rc, 0, out); self.assertIn('unused state, not required: MISSING CHANGE', out)

    def test_12_base_key_missing(self):
        def test(z): del z['clean/query_raw_pids']
        rc, out, rep = self.check(None, test)
        self.assertEqual(rc, 2, out); self.assertIn("test: base keys missing ['clean/query_raw_pids']", self.status_line(out))

    def test_13_problem_in_one_seed_fails_the_whole_check(self):
        def dev(z): z['ir_noise_010/f_rgb'] = z['clean/f_rgb'] + np.float32(0.01)
        def test(z): z['ir_noise_010/f_rgb'] = z['clean/f_rgb'].copy()
        rc, out, rep = self.check(dev, test, extra_seed=True)
        self.assertEqual(rc, 2, out); self.assertEqual(rep['s102']['problems'], []); self.assertTrue(rep['s101']['problems'])
        self.assertIn('[status] PROBLEMS (1): s101 ir_noise_010/f_rgb', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
