"""CPU acceptance tests of the round-17 scripts on tiny synthetic archives (no real data, no official-test file, no model).

1. r17_const.py with the round-15 method list and no new option reproduces the frozen r15_test.py outputs (cells, summary, diag cells,
   per-query records; selection entries of every method that both scripts select).
2. With --fixed_selection, the shared methods (B*, QE-DBA, QE+M, QE+CAET, QE+SRC2f, QE+EM) are reproduced exactly, and the new
   methods satisfy: QE+Const@0 == QE-DBA, QE+Const@1 == QE+NoGate, QE+Const == QE+Const@<chosen gain>; no top-1 change when the
   front end's top-1 item is not IR-only; explicit-key per-query records agree with the cells.
3. r17_judge.py, r17_reagg_r15.py and r17_check_archives.py run on these outputs and report VALID / exit 0; the judge stops with
   INVALID (exit 3) and prints no table when a shared cell differs from round 15, is NaN on one side only, or a run was made by another
   r17_const.py.
4. --timing: every stage has its boundary, the timed B* scoring and front end equal the posteriors every method uses, per-stage
   allocation peaks exist; the whole-process peak RSS is recorded where the POSIX `resource` module exists (not on Windows).
5. r17_judge.py --devsens_only on a development-sensitivity run: VALID, and INVALID when an expected directory is missing.
Run: python -m pytest IRRA-main/tests/p4/test_r17.py -q   (or python IRRA-main/tests/p4/test_r17.py)
Child processes use UTF-8 for their output, so a GBK console does not matter.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
TRANSFER = CODE_ROOT / "tools" / "p4" / "transfer"
SC = ['clean', 'rgb_missing', 'ir_missing', 'rgb_noise_010', 'ir_noise_010', 'rgb_blur_k7', 'ir_blur_k7']
MIXES = ['M0', 'M4', 'M1g', 'M5e', 'M4id']
R15_METHODS = ['B*', 'CAET', 'A-SRC-2f', 'A-2f-EM', 'A-2f-ctxn', 'A-2f-ctxn-g', 'GP-same', 'QE-DBA', 'FeatImp', 'FeatImp-g', 'QE+M', 'GP+M',
               'QE+CAET', 'QE+SRC2f', 'QE+EM', 'QE+Orc', 'DB', 'Oracle-A']
R17_METHODS = ['B*', 'QE-DBA', 'QE+M', 'QE+CAET', 'QE+SRC2f', 'QE+EM', 'QE+Const', 'QE+NoGate']
SHARED = ['B*', 'QE-DBA', 'QE+M', 'QE+CAET', 'QE+SRC2f', 'QE+EM']
HAS_RESOURCE = importlib.util.find_spec('resource') is not None   # POSIX only: process peak RSS (absent on Windows)


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def make_archive(rng, n_ids, raw0, dim=32, blank=None, full_keys=True):
    """identities with 2-4 images and 2 descriptions each; IR shares identity structure with RGB, text follows both"""
    id_rgb = rng.standard_normal((n_ids, dim)); id_ir = 0.6 * id_rgb + 0.8 * rng.standard_normal((n_ids, dim))
    img_ids = np.concatenate([[i] * int(rng.integers(2, 5)) for i in range(n_ids)])
    txt_ids = np.repeat(np.arange(n_ids), 2)
    f_rgb = l2n(id_rgb[img_ids] + 0.45 * rng.standard_normal((len(img_ids), dim))).astype(np.float32)
    f_ir = l2n(id_ir[img_ids] + 0.45 * rng.standard_normal((len(img_ids), dim))).astype(np.float32)
    text = l2n(0.8 * id_rgb[txt_ids] + 0.3 * id_ir[txt_ids] + 0.6 * rng.standard_normal((len(txt_ids), dim))).astype(np.float32)
    out = {'clean/text_features': text, 'clean/txt_pids': txt_ids.astype(np.int64), 'clean/img_pids': img_ids.astype(np.int64),
           'clean/query_raw_pids': (raw0 + txt_ids).astype(np.int64), 'clean/f_rgb': f_rgb, 'clean/f_ir': f_ir}
    noise = lambda x, s: (x + s * rng.standard_normal(x.shape)).astype(np.float32)
    out['rgb_missing/f_rgb'] = np.tile(blank[0], (len(img_ids), 1)).astype(np.float32)
    out['ir_missing/f_ir'] = np.tile(blank[1], (len(img_ids), 1)).astype(np.float32)
    out['rgb_noise_010/f_rgb'] = noise(f_rgb, 0.10); out['ir_noise_010/f_ir'] = noise(f_ir, 0.10)
    out['rgb_blur_k7/f_rgb'] = noise(f_rgb, 0.05); out['ir_blur_k7/f_ir'] = noise(f_ir, 0.05)
    return out


def jload(p):
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def pload(p):
    with open(p, 'rb') as fh:
        return pickle.load(fh)


def run(args, cwd):
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
    p = subprocess.run([sys.executable, '-B'] + [str(a) for a in args], cwd=cwd, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', env=env)
    return p


class R17(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); T = Path(cls.tmp.name); cls.T = T
        rng = np.random.default_rng(17); dim = 32; blank = (rng.standard_normal(dim), rng.standard_normal(dim))
        (T / 'feats').mkdir()
        np.savez(T / 'feats' / 'base_ref_s101.npz', **make_archive(rng, 40, 100000, dim, blank))
        np.savez(T / 'test_base_s101.npz', **make_archive(rng, 34, 800000, dim, blank))
        tr = make_archive(rng, 70, 900000, dim, blank)
        np.savez(T / 'train_ir.npz', f_ir=tr['clean/f_ir'], f_rgb=tr['clean/f_rgb'], raw_pids=tr['clean/query_raw_pids'][::2][tr['clean/img_pids']])
        common = ['--b12', '.', '--feats', T / 'feats', '--assign', 'none', '--base', 'base_ref_s101', '--adapter', 'base_ref_s101',
                  '--assign_seeds', '7031', '7032', '--mixes'] + MIXES + ['--skip_joint', '--grid_preset', 'dev', '--select_rule', 'constrained',
                  '--r_max', '0.999999', '--tie_break', 'bstar', '--test_npz', T / 'test_base_s101.npz', '--link_map', 'train_lda',
                  '--train_ir_npz', T / 'train_ir.npz', '--test_seed', '7001']
        cls.r15 = T / 'r15'; cls.r17a = T / 'r17a'; cls.r17 = T / 'r17'
        for d in (cls.r15, cls.r17a, cls.r17): d.mkdir()
        cls.p15 = run([TRANSFER / 'r15_test.py', '--out', cls.r15, '--tag', 't_s101_f7001', '--methods'] + R15_METHODS + common, CODE_ROOT)
        cls.p17a = run([TRANSFER / 'r17_const.py', '--out', cls.r17a, '--tag', 't_s101_f7001', '--methods'] + R15_METHODS + common, CODE_ROOT)
        sel = cls.r15 / 'r12_t_s101_f7001_selection.json'
        cls.p17 = run([TRANSFER / 'r17_const.py', '--out', cls.r17, '--tag', 'c_s101_f7001', '--methods'] + R17_METHODS + common +
                      ['--fixed_selection', sel, '--const_curve', '--perq_ext', '--timing'], CODE_ROOT)
        cls.dev = T / 'devsens' / 'ka20_s101'; cls.dev.mkdir(parents=True)
        cls.pdev = run([TRANSFER / 'r17_const.py', '--b12', '.', '--feats', T / 'feats', '--assign', 'none', '--out', cls.dev, '--base', 'base_ref_s101',
                        '--adapter', 'base_ref_s101', '--tag', 'd_ka20_s101', '--folds', '2', '--split_seeds', '2026', '--assign_seeds', '2026',
                        '--mixes', 'M4', 'M1g', 'M5e', 'M4id', '--skip_joint', '--methods', 'B*', 'A-2f-ctxn', '--grid_preset', 'dev',
                        '--select_rule', 'constrained', '--r_max', '0.999999', '--tie_break', 'bstar', '--link_map', 'train_lda',
                        '--train_ir_npz', T / 'train_ir.npz', '--kappa_a_fixed', '20'], CODE_ROOT)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.tmp.cleanup()
        except OSError:   # Windows can still hold a handle on a temporary file; leftovers in the temp folder are harmless
            pass

    def test_0_runs(self):
        for p in (self.p15, self.p17a, self.p17, self.pdev):
            self.assertEqual(p.returncode, 0, p.stdout[-2000:] + p.stderr[-4000:])

    def test_1_no_option_equals_r15(self):
        for suffix in ('cells.csv', 'summary.csv', 'diag_cells.csv'):
            a = pd.read_csv(self.r15 / f'r12_t_s101_f7001_{suffix}'); b = pd.read_csv(self.r17a / f'r12_t_s101_f7001_{suffix}')
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        pa = pload(self.r15 / 'r12_t_s101_f7001_perq.pkl'); pb = pload(self.r17a / 'r12_t_s101_f7001_perq.pkl')
        self.assertEqual(pa, pb)
        sa = {e['method']: e['chosen'] for e in jload(self.r15 / 'r12_t_s101_f7001_selection.json')}
        sb = {e['method']: e['chosen'] for e in jload(self.r17a / 'r12_t_s101_f7001_selection.json')}
        for m, v in sb.items():
            self.assertEqual(sa[m], v, m)
        self.assertTrue({'QE-DBA', 'GP-same', 'FeatImp-g', 'A-2f-ctxn-g'} <= set(sb))

    def test_2_fixed_selection_shared_methods(self):
        a = pd.read_csv(self.r15 / 'r12_t_s101_f7001_cells.csv'); b = pd.read_csv(self.r17 / 'r12_c_s101_f7001_cells.csv')
        a = a[a['method'].isin(SHARED)]; b = b[b['method'].isin(SHARED)]
        m = a.merge(b, on=['split', 'evalset', 'mixture', 'aseed', 'method'], suffixes=('', '_17'))
        self.assertEqual(len(m), len(a)); self.assertEqual(len(m), len(b))
        for c in ('mAP', 'R1', 'R5', 'R10', 'mINP', 'dmAP', 'dR1', 'disp_harm', 'disp_help'):
            self.assertEqual(float((m[c] - m[c + '_17']).abs().max()), 0.0, c)
        sel = jload(self.r17 / 'r12_c_s101_f7001_selection.json')
        q = [e for e in sel if e['method'] == 'QE-DBA']; self.assertEqual(len(q), 1); self.assertEqual(q[0].get('source'), 'fixed_selection')
        c = [e for e in sel if e['method'] == 'QE+Const']; self.assertEqual(len(c), 1)
        self.assertEqual([r['cfg'] for r in c[0]['table']], [0.0, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0])

    def test_3_new_method_identities(self):
        b = pd.read_csv(self.r17 / 'r12_c_s101_f7001_cells.csv'); cols = ['mAP', 'R1', 'R5', 'R10', 'mINP']
        cell = lambda m: b[b['method'] == m].set_index(['evalset', 'mixture', 'aseed'])[cols].sort_index().values
        g = [e for e in jload(self.r17 / 'r12_c_s101_f7001_selection.json') if e['method'] == 'QE+Const'][0]['chosen']
        self.assertEqual(np.abs(cell('QE+Const@0') - cell('QE-DBA')).max(), 0.0)
        self.assertEqual(np.abs(cell('QE+Const@1') - cell('QE+NoGate')).max(), 0.0)
        self.assertEqual(np.abs(cell('QE+Const') - cell(f'QE+Const@{float(g):g}')).max(), 0.0)

    def test_4_records_and_locality(self):
        P = pload(self.r17 / 'r17_c_s101_f7001_perq_ext.pkl'); R = P['records']
        b = pd.read_csv(self.r17 / 'r12_c_s101_f7001_cells.csv')
        self.assertEqual(len(R), len(MIXES) * 2 * len(R17_METHODS))
        for (ss, es, mx, asd, m), r in R.items():
            row = b[(b['evalset'] == es) & (b['mixture'] == mx) & (b['aseed'] == asd) & (b['method'] == m)]
            self.assertEqual(len(row), 1)
            self.assertAlmostEqual(float(r['AP'].mean()), float(row['mAP'].iloc[0]), places=9)
            self.assertAlmostEqual(float(r['R5'].mean()) * 100, float(row['R5'].iloc[0]), places=9)
            f = R[(ss, es, mx, asd, 'QE-DBA')]; self.assertTrue(np.array_equal(f['q'], r['q']))
            if m in ('QE+M', 'QE+Const', 'QE+NoGate', 'QE+CAET', 'QE+SRC2f', 'QE+EM'):
                keep = f['top1_av'] != 1
                self.assertEqual(int((r['top1'][keep] != f['top1'][keep]).sum()), 0, (mx, asd, m))

    def test_4b_timing_boundaries(self):
        st = jload(self.r17 / 'r17_c_s101_f7001_r17log.json')['times']['stages'][0]
        for k in ('online_link_graph', 'online_bstar_scoring', 'online_bstar_sigmoid_only', 'online_qedba_after_bstar', 'online_frontend_full',
                  'online_gate_and_assoc', 'online_assoc_only', 'online_transfer_gate', 'online_transfer_const', 'online_sort'):
            self.assertIn('median_s', st[k], k); self.assertIn(k, st['scope'], k)
        self.assertNotIn('online_frontend_bstar', st); self.assertNotIn('peak_rss_mb', st)
        self.assertIn('ONLY the final sigmoid', st['scope']['online_bstar_sigmoid_only'])
        self.assertLess(st['check_bstar_scoring_max_abs_diff'], 1e-9); self.assertLess(st['check_frontend_full_max_abs_diff'], 1e-9)
        for k in ('bstar_scoring', 'frontend_full', 'gate_and_assoc', 'assoc_only', 'transfer_gate', 'transfer_const', 'sort'):
            self.assertGreater(st['module_memory'][k]['alloc_peak_mb'], 0.0, k); self.assertGreater(st['module_memory'][k]['inputs_mb'], 0.0, k)
        self.assertEqual(st['process_peak_rss_mb'] is not None, HAS_RESOURCE)

    def test_5_reports(self):
        T = self.T
        rt = T / 'const'; (rt / 's101_f7001').mkdir(parents=True)
        for f in self.r17.iterdir(): (rt / 's101_f7001' / f.name).write_bytes(f.read_bytes())
        r15root = T / 'r15root'; (r15root / 's101_f7001').mkdir(parents=True)
        for f in self.r15.iterdir(): (r15root / 's101_f7001' / f.name).write_bytes(f.read_bytes())
        sha = hashlib.sha256((TRANSFER / 'r17_const.py').read_bytes()).hexdigest()
        judge = lambda r15, out, extra=(): run([TRANSFER / 'r17_judge.py', '--r17_root', rt, '--r15_root', r15, '--seeds', '101', '--fits', '7001',
                                               '--boot', '200', '--out', T / out] + list(extra), CODE_ROOT)
        p = judge(r15root, 'judge.json', ['--expect_script_sha', sha])
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:]); self.assertIn('[status] VALID', p.stdout)
        self.assertIn('[table B]', p.stdout); self.assertIn('online_bstar_scoring', p.stdout); self.assertIn('alloc peak', p.stdout)
        if HAS_RESOURCE:   # the whole-process line exists only where the peak RSS was recorded
            self.assertIn('NOT the memory of the transfer module', p.stdout)
        p = judge(r15root, 'judge_badsha.json', ['--expect_script_sha', '0' * 64])
        self.assertEqual(p.returncode, 3, p.stdout[-3000:]); self.assertIn('[stopped]', p.stdout); self.assertNotIn('[table B]', p.stdout)
        bad = T / 'r15bad'; (bad / 's101_f7001').mkdir(parents=True)                 # one round-15 cell of QE+M changed in a copy
        for f in self.r15.iterdir(): (bad / 's101_f7001' / f.name).write_bytes(f.read_bytes())
        cf = bad / 's101_f7001' / 'r12_t_s101_f7001_cells.csv'; c = pd.read_csv(cf)
        i = c.index[(c['method'] == 'QE+M') & (c['evalset'].astype(str) == 'test')][0]; c.loc[i, 'R10'] += 0.01; c.to_csv(cf, index=False)
        p = judge(bad, 'judge_bad.json')
        self.assertEqual(p.returncode, 3, p.stdout[-3000:]); self.assertIn('NOT REPRODUCED', p.stdout); self.assertIn('[status] INVALID', p.stdout)
        self.assertNotIn('[table B]', p.stdout); self.assertNotIn('[timing]', p.stdout)
        nan = T / 'r15nan'; (nan / 's101_f7001').mkdir(parents=True)                # one round-15 cell of QE+M set to NaN: a NaN on one side only
        for f in self.r15.iterdir(): (nan / 's101_f7001' / f.name).write_bytes(f.read_bytes())
        cf = nan / 's101_f7001' / 'r12_t_s101_f7001_cells.csv'; c = pd.read_csv(cf)
        i = c.index[(c['method'] == 'QE+M') & (c['evalset'].astype(str) == 'test')][0]; c.loc[i, 'mAP'] = np.nan; c.to_csv(cf, index=False)
        p = judge(nan, 'judge_nan.json')
        self.assertEqual(p.returncode, 3, p.stdout[-3000:]); self.assertIn('NaN on one side only 1', p.stdout); self.assertNotIn('[table B]', p.stdout)
        qr = f's101={T / "feats" / "base_ref_s101.npz"},{T / "test_base_s101.npz"}'
        p = run([TRANSFER / 'r17_reagg_r15.py', '--r15_root', r15root, '--qraw', qr, '--fits', '7001', '--boot', '200', '--json', T / 'reagg.json'], CODE_ROOT)
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:]); self.assertIn('[status] VALID', p.stdout)
        p = run([TRANSFER / 'r17_check_archives.py', '--pairs', qr, '--assign_seeds', '7031', '7032', '--json', T / 'arch.json'], CODE_ROOT)
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:])

    def test_6_devsens_report(self):
        T = self.T; root = T / 'devsens'
        p = run([TRANSFER / 'r17_judge.py', '--devsens_only', '--devsens_root', root, '--devsens_expect', 'ka20_s101', '--out', T / 'dev.json'], CODE_ROOT)
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:]); self.assertIn('[status] VALID', p.stdout)
        self.assertIn('kappa_a used [20.0]', p.stdout)
        p = run([TRANSFER / 'r17_judge.py', '--devsens_only', '--devsens_root', root, '--devsens_expect', 'ka20_s101', 'ka45_s101',
                 '--out', T / 'dev2.json'], CODE_ROOT)
        self.assertEqual(p.returncode, 3, p.stdout[-3000:]); self.assertIn('ka45_s101: directory missing', p.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
