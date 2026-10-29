# -*- coding: utf-8 -*-
"""r18_verdict.py 的测试：合成的三种子、全量与去重子集 R18J_core，经 r18_core.means 生成均值行，
再按 driver_support.finish 的格式拼成 DIGEST18_L2。不读任何真实结果。"""
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from tools.p4.r18 import r18_verdict as V          # noqa: E402
from tools.p4.r18.r18_core import means as core_means   # noqa: E402

SEEDS, FITS = V.SEEDS, V.FITS
ABS = ('B*', 'QE-DBA', 'QEir', 'KR', 'RVN', 'KRir', 'RVNir', 'GP-same', 'CAET', 'QE+CAET', 'QE+SRC2f', 'QE+EM',
       'FeatImp-g', 'A-2f-ctxn', 'QE+M', 'QE+Orc', 'Oracle-A')
GRID = ('0', '0.05', '0.1', '0.2', '0.25', '0.5', '0.75', '1')


def base_rows():
    rows = {}
    for scope in V.SCOPES:
        nid, nq = (30, 836) if scope == 'full' else (26, 712)
        for s in SEEDS:
            k = s - 101

            def put(src, key, v, s=s, scope=scope):
                rows[(src, scope, s, key)] = v

            put('status', 'judge', {'evalset': 'test', 'exit_status': 0})
            put('status', 'r17', {'valid': True, 'reproduction_ok': True, 'problems': []})
            put('status', 'selection_equal', {'SELECTION_EQUAL': True, 'differences': 0} if scope == 'subset'
                else {'SELECTION_EQUAL': None, 'differences': None})
            if scope == 'full':
                put('status', 'xcam', {'status': 'VALID', 'problems': []})
            for p in V.RUNNER_PARTS[scope]:
                for f in FITS:
                    put('status', f'runner|{p}/s{s}_f{f}', {'events': 1, 'heads': {'anyh': 1}})

            def jp(x, y, g, d, lo, hi, r1=0.0, rlo=-0.5, rhi=0.5):
                put('judge', f'pair|{x}|{y}|{g}', {'dmAP': d, 'lo': lo, 'hi': hi, 'dR1': r1, 'r_lo': rlo, 'r_hi': rhi,
                                                   'n_q': nq, 'n_id': nid})
            jp('QE+M', 'QE-DBA', 'A', 3.0 + 0.1 * k, 2.0, 4.0, 1.0, 0.2, 1.8)
            jp('QE+M', 'QE-DBA', 'B', 0.05, -0.10, 0.20, 0.0, -0.2, 0.2)
            for y in ('QEir', 'KRir', 'GP-same', 'QE+CAET', 'QE+SRC2f', 'QE+EM'):
                jp('QE+M', y, 'A', 1.5 + 0.01 * k, 0.5, 2.5)
                jp('QE+M', y, 'B', -0.1, -0.3, 0.1)
            for y in ('FeatImp-g', 'FeatImp', 'CAET', 'A-SRC-2f', 'A-2f-EM'):
                for g in 'AB':
                    jp('A-2f-ctxn', y, g, 0.3, 0.1, 0.5)
            for g in 'AB':
                jp('QE+Orc', 'QE+M', g, 0.5, 0.2, 0.8)
            put('judge', 'claims', {'C1': True, 'C2': True, 'C3': True, 'C4': False, 'C5': True})
            put('cells', 'strongest|A', {'method': 'KRir', 'dmAP': 1.5})
            put('cells', 'strongest|B', {'method': 'QEir', 'dmAP': 2.0})
            for i, m in enumerate(ABS):
                put('cells', f'abs|{m}|A', {'mAP': 40 + i + 0.1 * k, 'R1': 41 + i, 'R5': 60 + i, 'R10': 70 + i, 'mINP': 30 + i})
                put('cells', f'abs|{m}|B', {'mAP': 38 + i, 'R1': 36 + i, 'mINP': 33 + i})
                put('cells', f'abs|{m}|M0', {'mAP': 45.5 + 0.01 * k, 'R1': 44.0} if m in ('QE+M', 'QE-DBA')
                    else {'mAP': 44 + i, 'R1': 43 + i})
            for m in ('QE-DBA', 'QEir', 'KRir', 'GP-same', 'FeatImp-g', 'CAET', 'A-2f-ctxn', 'QE+CAET', 'QE+EM', 'QE+M', 'Oracle-A'):
                put('cells', f'mixture_dmAP|{m}', {mx: 1.0 + 0.1 * j + 0.01 * k for j, mx in enumerate(V.MIXES)})
            put('cells', 'mixture_mAP|B*', {mx: 30.0 + j for j, mx in enumerate(V.MIXES)})
            for m in ('QE+M', 'A-2f-ctxn', 'CAET', 'QE+CAET'):
                put('diag', f'mixture_diag|{m}', {mx: {'frac_src': 1.0 if mx in ('M4', 'M1g') else 0.6, 'coverage': 0.9,
                                                       'leak': float('nan') if mx in ('M4', 'M1g') else 0.2}
                                                  for mx in V.MIXES if mx != 'M0'})

            def cp(name, g, ap, r1=(0.0, -0.3, 0.3), extra=(0.1, -0.2, 0.4)):
                put('r17', f'pair|{name}|{g}', {'AP': list(ap), 'R1': list(r1), 'R5': list(extra), 'R10': list(extra),
                                               'mINP': list(extra)})
            cp('QE+M-QE+Const', 'A', (0.8, 0.3, 1.3), (0.2, -0.2, 0.6))
            cp('QE+M-QE+Const', 'B', (0.1, -0.05, 0.25), (0.0, -0.2, 0.2))
            for name in ('QE+M-QE+NoGate', 'QE+Const-QE-DBA', 'QE+NoGate-QE-DBA', 'QE+M-QE-DBA', 'QE+M-QE+CAET'):
                for g in ('A', 'B', 'M5e', 'M5f', 'M4id'):
                    cp(name, g, (0.4, 0.1, 0.7))
            for g in ('M5e', 'M5f', 'M4id'):
                cp('QE+M-QE+Const', g, (0.1, -0.1, 0.3))
            for f in FITS:
                put('r17', f'gamma|f{f}', {'gamma': '0.25'})
            put('r17', 'locality', {m: {'changed': 0, 'changed_ir': 10, 'n_front_ir': 100, 'n_front_not_ir': 500}
                                    for m in ('QE+M', 'QE+Const', 'QE+NoGate', 'QE+CAET', 'QE+EM', 'QE+SRC2f')})
            put('r17', 'curve', {gm: {'A_mAP': 40 + float(gm), 'A_R1': 41.0, 'B_mAP': 38 - float(gm), 'B_R1': 36.0} for gm in GRID})
            if scope == 'full':
                def xp(x, y, g, d, lo, hi):
                    put('xcam', f'pair|{x}|{y}|{g}', {'dmAP': d, 'lo': lo, 'hi': hi, 'dR1': 0.1, 'r_lo': -0.2, 'r_hi': 0.4,
                                                      'R5': {'mean': 0.1, 'lo': -0.3, 'hi': 0.5},
                                                      'R10': {'mean': 0.2, 'lo': -0.2, 'hi': 0.6},
                                                      'mINP': {'mean': 0.3, 'lo': -0.1, 'hi': 0.7}})
                xp('QE+M-xcam', 'QE-DBA', 'A', 1.2, 0.4, 2.0)
                xp('QE+M-xcam', 'QE-DBA', 'B', 0.0, -0.1, 0.1)
                xp('QE+M', 'QE-DBA', 'A', 3.0 + 0.1 * k, 2.0, 4.0)
                xp('QE+M', 'QE-DBA', 'B', 0.05, -0.10, 0.20)
                xp('QE+M', 'QE+M-xcam', 'A', 1.8, 1.0, 2.6)
                xp('QE+M', 'QE+M-xcam', 'B', 0.05, -0.05, 0.15)
                put('xcam', 'ratio', {'denominator': 3.0, 'numerator': 1.2, 'ratio': 0.4})
                for mx in V.MIXES:
                    put('xcam', f'coverage|{mx}', {'mean': {'ir_only_items': 100.0, 'same_identity_complete_any_camera': 100.0,
                                                            'same_identity_complete_other_camera': 40.0,
                                                            'same_identity_sources_only_same_camera': 60.0},
                                                   'ratio': {'ir_only_items': 1.0, 'same_identity_complete_any_camera': 1.0,
                                                             'same_identity_complete_other_camera': 0.4,
                                                             'same_identity_sources_only_same_camera': 0.6}})
    return rows


def fix_claims(rows):
    """按冻结 judge_test.py 的公式重算 C1–C3（ok_gain = 点估计 ≥ 门槛且下界 > 0），使夹具自洽"""
    for scope in V.SCOPES:
        for s in SEEDS:
            if ('judge', scope, s, 'claims') not in rows:
                continue
            P = lambda x, y, g: rows[('judge', scope, s, f'pair|{x}|{y}|{g}')]
            pa, pb = P('QE+M', 'QE-DBA', 'A'), P('QE+M', 'QE-DBA', 'B')
            zA = rows[('cells', scope, s, 'strongest|A')]['method']
            zB = rows[('cells', scope, s, 'strongest|B')]['method']
            sa, sb = P('QE+M', zA, 'A'), P('QE+M', zB, 'B')
            c = rows[('judge', scope, s, 'claims')]
            c['C1'] = pa['dmAP'] >= 1.0 and pa['lo'] > 0
            c['C2'] = pb['dmAP'] >= -0.3 and pb['dR1'] >= -0.3
            c['C3'] = sa['dmAP'] >= 1.0 and sa['lo'] > 0 and sb['dmAP'] >= -0.3
    return rows


def render(rows, status_l1='COMPLETE', status_l2='COMPLETE', with_means=True):
    lst = [dict(src=k[0], set=k[1], seed=k[2], key=k[3], v=v) for k, v in rows.items()]
    if with_means:
        lst += core_means(lst, list(SEEDS))
    core = '\n'.join('R18J ' + json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(',', ':')) for r in lst) + '\n'
    parts = [f'STATUS_L2={status_l2}', f'STATUS_L1={status_l1}', f'STATUS_L2={status_l2}',
             'PROTECTED_MANIFEST files=3 sha256=' + '0' * 64,
             '\n===== run_frozen auxiliary single-class fits by unit =====', 'const/s101_f7001 events=1 heads={"anyh": 1}',
             '\n===== R18J_core.txt =====', core, '\n===== stage manifests =====']
    body = '\n'.join(parts) + '\n'
    footer = ''
    while True:          # 与 driver_support.finish 相同的定点计算
        text = body + footer
        updated = f'DIGEST_STATS lines={text.count(chr(10))} bytes={len(text.encode("utf-8"))}\n'
        if updated == footer:
            break
        footer = updated
    return text


def run(rows, **kw):
    return V.analyze(render(fix_claims(rows), **kw))


def setp(rows, scope, seeds, x, y, g, **vals):
    for s in seeds:
        rows[('judge', scope, s, f'pair|{x}|{y}|{g}')].update(vals)


def setc(rows, scope, seeds, g, ap=None, r1=None):
    for s in seeds:
        v = rows[('r17', scope, s, f'pair|QE+M-QE+Const|{g}')]
        if ap is not None:
            v['AP'] = list(ap)
        if r1 is not None:
            v['R1'] = list(r1)


class TierTests(unittest.TestCase):
    def test_all_good_is_effective(self):
        out = run(base_rows())
        self.assertEqual(out['reasons'], [])
        for sc in V.SCOPES:
            r = out['scopes'][sc]
            self.assertEqual(r['invalid'], [], r['invalid'])
            self.assertEqual(r['tier'], '有效')
            self.assertTrue(r['E1'] and r['E2'] and r['E3'])
        self.assertEqual(out['final'], '有效')
        self.assertEqual(out['scopes']['full']['E4'], '①a')
        self.assertTrue(out['scopes']['full']['xcam_positive'])
        text = V.report(out, 'x', '0' * 64) + V.tables(out)
        self.assertIn('论文取较保守的一档：有效', text)
        self.assertIn('全部逐位相同', text)

    def test_limited_when_one_seed_below_gain(self):
        rows = base_rows()
        setp(rows, 'full', [102], 'QE+M', 'QE-DBA', 'A', dmAP=0.8)
        out = run(rows)
        self.assertEqual(out['scopes']['full']['tier'], '有限适用')
        self.assertEqual(out['scopes']['subset']['tier'], '有效')
        self.assertEqual(out['final'], '有限适用')

    def test_limited_when_interval_touches_zero(self):
        rows = base_rows()
        setp(rows, 'subset', [103], 'QE+M', 'QE-DBA', 'A', lo=0.0)
        out = run(rows)
        self.assertEqual(out['scopes']['subset']['tier'], '有限适用')
        self.assertEqual(out['final'], '有限适用')

    def test_boundary_values(self):
        rows = base_rows()
        setp(rows, 'full', SEEDS, 'QE+M', 'QE-DBA', 'A', dmAP=1.0, lo=1e-12)
        setp(rows, 'full', SEEDS, 'QE+M', 'QE-DBA', 'B', dmAP=-0.3, dR1=-0.3, lo=-0.31, r_lo=-0.29)
        out = run(rows)
        r = out['scopes']['full']
        self.assertTrue(r['E1'] and r['E2'])
        self.assertEqual(r['tier'], '有效')
        self.assertTrue(r['E2_wording']['mAP'].startswith('点估计在原容差内'))
        self.assertEqual(r['E2_wording']['R1'], '损害在原容差以内')

    def test_e2_fails_on_r1(self):
        rows = base_rows()
        setp(rows, 'full', [101], 'QE+M', 'QE-DBA', 'B', dR1=-0.31)
        out = run(rows)
        r = out['scopes']['full']
        self.assertFalse(r['E2'])
        self.assertEqual(r['tier'], '有限适用')
        self.assertEqual(r['E2_wording']['R1'], '超出原容差')

    def test_no_effect_when_all_points_nonpositive(self):
        rows = base_rows()
        for sc in V.SCOPES:
            setp(rows, sc, SEEDS, 'QE+M', 'QE-DBA', 'A', dmAP=0.0, lo=-0.5, hi=0.5)
        setp(rows, 'subset', [101], 'QE+M', 'QE-DBA', 'A', dmAP=-0.2)
        out = run(rows)
        self.assertEqual(out['scopes']['full']['tier'], '未见效果')
        self.assertEqual(out['scopes']['subset']['tier'], '未见效果')
        self.assertEqual(out['final'], '未见效果')

    def test_mixed_sign_is_limited_not_no_effect(self):
        rows = base_rows()
        setp(rows, 'full', SEEDS, 'QE+M', 'QE-DBA', 'A', dmAP=-0.1, lo=-0.5)
        setp(rows, 'full', [103], 'QE+M', 'QE-DBA', 'A', dmAP=0.1)
        out = run(rows)
        self.assertEqual(out['scopes']['full']['tier'], '有限适用')

    def test_final_takes_more_conservative(self):
        rows = base_rows()
        for sc in V.SCOPES:
            setp(rows, sc, SEEDS, 'QE+M', 'QE-DBA', 'A', dmAP=-0.1, lo=-0.5)
        setp(rows, 'full', [101], 'QE+M', 'QE-DBA', 'A', dmAP=0.4)
        out = run(rows)
        self.assertEqual(out['scopes']['full']['tier'], '有限适用')
        self.assertEqual(out['scopes']['subset']['tier'], '未见效果')
        self.assertEqual(out['final'], '未见效果')


class ValidityTests(unittest.TestCase):
    def assertProblem(self, rows, scope, text):
        out = run(rows)
        r = out['scopes'][scope]
        self.assertEqual(r['tier'], '实现或协议问题')
        self.assertTrue(any(text in p for p in r['invalid']), r['invalid'])
        self.assertEqual(out['final'], '实现或协议问题')
        return out

    def test_selection_false(self):
        rows = base_rows()
        for s in SEEDS:
            rows[('status', 'subset', s, 'selection_equal')] = {'SELECTION_EQUAL': False, 'differences': 2}
        out = self.assertProblem(rows, 'subset', 'SELECTION_EQUAL=false')
        self.assertTrue(out['scopes']['subset']['selection_false'])
        self.assertEqual(out['scopes']['full']['tier'], '有效')

    def test_selection_unrecorded(self):
        rows = base_rows()
        rows[('status', 'subset', 102, 'selection_equal')] = {'SELECTION_EQUAL': None, 'differences': None}
        self.assertProblem(rows, 'subset', '没有记录 SELECTION_EQUAL')

    def test_r17_reproduction(self):
        rows = base_rows()
        rows[('status', 'full', 101, 'r17')]['reproduction_ok'] = False
        self.assertProblem(rows, 'full', '复现不一致')

    def test_r17_problems_nonempty(self):
        rows = base_rows()
        rows[('status', 'subset', 103, 'r17')]['problems'] = ['x']
        self.assertProblem(rows, 'subset', 'r17_judge')

    def test_judge_exit_and_evalset(self):
        rows = base_rows()
        rows[('status', 'full', 102, 'judge')] = {'evalset': 'dev', 'exit_status': 0}
        self.assertProblem(rows, 'full', 'judge_test 的状态不对')

    def test_xcam_invalid(self):
        rows = base_rows()
        rows[('status', 'full', 101, 'xcam')] = {'status': 'INVALID', 'problems': ['p']}
        self.assertProblem(rows, 'full', '跨相机报告不是 VALID')

    def test_m0_invariant(self):
        rows = base_rows()
        rows[('cells', 'full', 103, 'abs|QE+M|M0')]['mAP'] += 1e-6
        self.assertProblem(rows, 'full', 'M0 上 QE+M 与 QE-DBA')

    def test_locality(self):
        rows = base_rows()
        rows[('r17', 'subset', 101, 'locality')]['QE+CAET']['changed'] = 1
        self.assertProblem(rows, 'subset', '改变了 top-1')

    def test_runner_head_outside_whitelist(self):
        rows = base_rows()
        rows[('status', 'full', 102, 'runner|const/s102_f7002')] = {'events': 2, 'heads': {'anyh': 1, 'cal': 1}}
        self.assertProblem(rows, 'full', '允许范围外')

    def test_runner_inconsistent_counts(self):
        rows = base_rows()
        rows[('status', 'full', 102, 'runner|xcam/s102_f7003')] = {'events': 3, 'heads': {'anyh': 1}}
        self.assertProblem(rows, 'full', '运行器记录格式不对')

    def test_identity_count(self):
        rows = base_rows()
        setp(rows, 'subset', [101], 'QE+M', 'QE-DBA', 'B', n_id=30)
        self.assertProblem(rows, 'subset', '评价身份数为 30')

    def test_query_count(self):
        rows = base_rows()
        setp(rows, 'full', [103], 'QE+M', 'QE-DBA', 'A', n_q=835)
        self.assertProblem(rows, 'full', '评价查询数为 835')

    def test_subset_query_count_must_shrink(self):
        rows = base_rows()
        setp(rows, 'subset', [103], 'QE+M', 'QE-DBA', 'A', n_q=836)
        self.assertProblem(rows, 'subset', '评价查询数为 836')

    def test_claims_c1_mismatch(self):
        rows = fix_claims(base_rows())
        rows[('judge', 'full', 101, 'claims')]['C1'] = False
        out = V.analyze(render(rows))
        self.assertEqual(out['scopes']['full']['tier'], '实现或协议问题')
        self.assertTrue(any('C1' in p for p in out['scopes']['full']['invalid']))

    def test_claims_c3_mismatch_is_only_a_warning(self):
        rows = fix_claims(base_rows())
        rows[('judge', 'full', 101, 'claims')]['C3'] = False
        out = V.analyze(render(rows))
        r = out['scopes']['full']
        self.assertEqual(r['tier'], '有效')
        self.assertTrue(any('C3' in w for w in r['warnings']))

    def test_strongest_not_in_pool_only_blocks_e3(self):
        rows = base_rows()
        rows[('cells', 'full', 101, 'strongest|A')] = {'method': 'QE+M', 'dmAP': 9.0}
        out = V.analyze(render(rows))
        r = out['scopes']['full']
        self.assertEqual(r['tier'], '有效')
        self.assertIsNone(r['seeds'][101]['E3'])
        self.assertIsNone(r['E3'])
        self.assertTrue(any('不在候选池内' in w for w in r['warnings']))

    def test_nonfinite_pair_value(self):
        rows = base_rows()
        setp(rows, 'full', [101], 'QE+M', 'QE-DBA', 'A', lo=float('nan'))
        out = V.analyze(render(rows))
        self.assertEqual(out['scopes']['full']['tier'], '实现或协议问题')

    def test_missing_pair_marked_missing_in_judge(self):
        rows = base_rows()
        rows[('judge', 'full', 101, 'pair|QE+M|QE-DBA|A')] = {'missing': True}
        out = V.analyze(render(rows))
        self.assertEqual(out['scopes']['full']['tier'], '实现或协议问题')

    def test_unreadable_gamma(self):
        rows = base_rows()
        rows[('r17', 'full', 101, 'gamma|f7002')] = {'gamma': None}
        out = run(rows)
        self.assertEqual(out['scopes']['full']['tier'], '实现或协议问题')


class InputTests(unittest.TestCase):
    def test_missing_subset_is_not_judged(self):
        rows = {k: v for k, v in base_rows().items() if k[1] != 'subset'}
        out = run(rows)
        self.assertIsNone(out['final'])
        self.assertEqual(out['scopes']['subset']['state'], 'missing')
        self.assertEqual(out['scopes']['full']['tier'], '有效')

    def test_incomplete_scope_is_not_judged(self):
        rows = base_rows()
        del rows[('r17', 'subset', 102, 'gamma|f7003')]
        out = run(rows)
        self.assertIsNone(out['final'])
        self.assertEqual(out['scopes']['subset']['state'], 'incomplete')

    def test_missing_runner_unit(self):
        rows = base_rows()
        del rows[('status', 'full', 101, 'runner|xcam/s101_f7001')]
        out = run(rows)
        self.assertIsNone(out['final'])

    def test_missing_strongest_pair_only_blocks_e3(self):
        rows = base_rows()
        rows[('cells', 'full', 101, 'strongest|A')] = {'method': 'KR', 'dmAP': 1.6}
        out = V.analyze(render(rows))
        r = out['scopes']['full']
        self.assertEqual(out['final'], '有效')
        self.assertIsNone(r['seeds'][101]['E3A'])
        self.assertIsNone(r['E3'])
        self.assertTrue(any('pair|QE+M|KR|A' in w for w in r['warnings']))
        self.assertIn('不可判', V.report(out, 'x', '0' * 64))

    def test_strongest_kr_with_pair(self):
        rows = base_rows()
        for s in SEEDS:
            rows[('cells', 'full', s, 'strongest|A')] = {'method': 'KR', 'dmAP': 1.6}
            rows[('judge', 'full', s, 'pair|QE+M|KR|A')] = {'dmAP': 0.9, 'lo': 0.2, 'hi': 1.6, 'dR1': 0.1, 'r_lo': -0.3,
                                                            'r_hi': 0.5, 'n_q': 836, 'n_id': 30}
        out = run(rows)
        r = out['scopes']['full']
        self.assertFalse(r['E3'])
        self.assertEqual(r['tier'], '有效')
        self.assertEqual(r['seeds'][101]['zA'], 'KR')

    def test_duplicate_rows(self):
        text = render(base_rows())
        lines = text.split('\n')
        dup = next(ln for ln in lines if ln.startswith('R18J '))
        body = text[:text.rindex('DIGEST_STATS')]
        body = body.replace(dup + '\n', dup + '\n' + dup + '\n', 1)
        out = V.analyze(body + self.footer(body))
        self.assertIsNone(out['final'])
        self.assertTrue(any('重复' in p for p in out['reasons']))

    @staticmethod
    def footer(body):
        footer = ''
        while True:
            text = body + footer
            upd = f'DIGEST_STATS lines={text.count(chr(10))} bytes={len(text.encode("utf-8"))}\n'
            if upd == footer:
                return footer
            footer = upd

    def test_truncated_digest(self):
        text = render(base_rows())
        cut = text[:len(text) // 2] + '\n' + text[text.rindex('DIGEST_STATS'):]
        out = V.analyze(cut)
        self.assertIsNone(out['final'])
        self.assertTrue(any('DIGEST_STATS' in p for p in out['reasons']))

    def test_two_digests_concatenated(self):
        text = render(base_rows())
        out = V.analyze(text + text)
        self.assertIsNone(out['final'])

    def test_crlf_is_rejected(self):
        out = V.analyze(render(base_rows()).replace('\n', '\r\n'))
        self.assertIsNone(out['final'])

    def test_status_not_complete(self):
        out = V.analyze(render(fix_claims(base_rows()), status_l2='STOPPED: x (exit=1)'))
        self.assertIsNone(out['final'])
        out = V.analyze(render(fix_claims(base_rows()), status_l1='STOPPED'))
        self.assertIsNone(out['final'])

    def test_unexpected_seed(self):
        rows = base_rows()
        rows[('status', 'full', 104, 'judge')] = {'evalset': 'test', 'exit_status': 0}
        out = V.analyze(render(fix_claims({k: v for k, v in rows.items() if k[2] != 104}) | {
            ('status', 'full', 104, 'judge'): {'evalset': 'test', 'exit_status': 0}}, with_means=False))
        self.assertIsNone(out['final'])
        self.assertTrue(any('种子不在' in p for p in out['reasons']))

    def test_no_r18j_rows(self):
        out = V.analyze(render({}, with_means=False))
        self.assertIsNone(out['final'])


class E4Tests(unittest.TestCase):
    def e4(self, A, B, A_r1=(0.0, -0.2, 0.2), B_r1=(0.0, -0.2, 0.2), seeds_override=None):
        rows = base_rows()
        setc(rows, 'full', SEEDS, 'A', ap=A, r1=A_r1)
        setc(rows, 'full', SEEDS, 'B', ap=B, r1=B_r1)
        if seeds_override:
            for (s, g), ap in seeds_override.items():
                setc(rows, 'full', [s], g, ap=ap)
        return run(rows)['scopes']['full']

    def test_1a_via_c4(self):
        r = self.e4((0.8, 0.3, 1.3), (0.0, -0.1, 0.1))
        self.assertEqual(r['E4'], '①a')

    def test_1b_when_a_gain_small(self):
        r = self.e4((0.4, 0.1, 0.7), (0.0, -0.1, 0.1))
        self.assertEqual(r['E4'], '①b')

    def test_1b_when_b_below_c4_tolerance(self):
        r = self.e4((0.8, 0.3, 1.3), (-0.2, -0.4, 0.0))
        self.assertEqual(r['E4'], '①b')

    def test_2_tradeoff(self):
        r = self.e4((0.8, 0.3, 1.3), (-0.2, -0.4, 0.0), B_r1=(-0.35, -0.6, -0.1))
        self.assertEqual(r['E4'], '②')

    def test_3_neither_repeatable(self):
        r = self.e4((0.8, 0.3, 1.3), (0.1, -0.1, 0.3), seeds_override={(102, 'A'): (0.5, -0.1, 1.1)})
        self.assertEqual(r['E4'], '③')

    def test_1a_via_c5(self):
        r = self.e4((0.0, -0.2, 0.2), (0.5, 0.2, 0.8))
        self.assertEqual(r['E4'], '①a')
        self.assertEqual([rt['gain'] for rt in r['E4_routes']], ['B'])

    def test_uncovered(self):
        r = self.e4((0.8, 0.3, 1.3), (0.2, 0.05, 0.35), A_r1=(-0.4, -0.7, -0.1))
        self.assertEqual({rt['out'] for rt in r['E4_routes']}, {'①a', '②'})
        self.assertEqual(r['E4'], '未覆盖')

    def test_both_routes_1a_and_1b_read_as_1a(self):
        r = self.e4((0.8, 0.3, 1.3), (0.2, 0.05, 0.35))
        self.assertEqual({rt['out'] for rt in r['E4_routes']}, {'①a', '①b'})
        self.assertEqual(r['E4'], '①a')

    def test_zero_lower_bound_not_positive(self):
        r = self.e4((0.8, 0.0, 1.3), (0.0, -0.1, 0.1))
        self.assertEqual(r['E4'], '③')

    def test_opposite_direction_and_gamma_zero(self):
        rows = base_rows()
        for s in SEEDS:
            rows[('r17', 'full', s, 'pair|QE+M-QE+Const|A')]['R10'] = [-0.2, -0.5, 0.1]
        rows[('r17', 'full', 102, 'gamma|f7001')] = {'gamma': '0.0'}
        r = run(rows)['scopes']['full']
        self.assertTrue(any('R10' in x for x in r['E4_opposite']))
        self.assertEqual(r['gamma_zero'], ['s102 f7001'])

    def test_other_group_wording(self):
        r = self.e4((0.8, 0.3, 1.3), (-0.1, -0.35, 0.1))
        self.assertTrue(r['E4_other']['B']['AP'].startswith('点估计在原容差内'))
        r = self.e4((0.8, 0.3, 1.3), (0.1, 0.05, 0.2))
        self.assertIn('也都可重复为正', r['E4_other']['B']['AP'])
        r = self.e4((0.8, 0.3, 1.3), (0.0, -0.1, 0.1))
        self.assertIn('都不为负', r['E4_other']['B']['AP'])


class XcamAndOutputTests(unittest.TestCase):
    def test_xcam_wording(self):
        rows = base_rows()
        rows[('xcam', 'full', 102, 'pair|QE+M-xcam|QE-DBA|A')]['lo'] = -0.01
        out = run(rows)
        self.assertFalse(out['scopes']['full']['xcam_positive'])
        self.assertIn('收益主要来自同相机供体', V.report(out, 'x', '0' * 64))

    def test_tables_and_mean_check(self):
        out = run(base_rows())
        t = V.tables(out)
        for tag in ('表 R1', '表 R2', '表 R3', '表 R4', '表 R5', 'R6 去重子集', '常数对照'):
            self.assertIn(tag, t)
        self.assertIn('全部逐位相同', t)
        self.assertNotIn('nan', t.lower().replace('mINP'.lower(), ''))

    def test_mean_check_flags_altered_mean(self):
        text = render(fix_claims(base_rows()))
        lines = text.split('\n')
        for i, ln in enumerate(lines):
            if ln.startswith('R18J ') and '"key":"abs|QE+M|A"' in ln and '"seed":"mean"' in ln:
                row = json.loads(ln[5:])
                row['v']['mAP']['mean'] += 1e-9
                lines[i] = 'R18J ' + json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        body = '\n'.join(lines)
        body = body[:body.rindex('DIGEST_STATS')]
        out = V.analyze(body + InputTests.footer(body))
        self.assertEqual(out['final'], '有效')
        self.assertIn('abs|QE+M|A|mAP', V.tables(out))

    def test_main_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            good, bad = Path(d) / 'good.txt', Path(d) / 'bad.txt'
            good.write_text(render(fix_claims(base_rows())), encoding='utf-8', newline='\n')
            rows = {k: v for k, v in base_rows().items() if k[1] != 'subset'}
            bad.write_text(render(fix_claims(rows)), encoding='utf-8', newline='\n')
            script = CODE / 'tools/p4/r18/r18_verdict.py'
            p = subprocess.run([sys.executable, '-B', str(script), str(good)], capture_output=True)
            self.assertEqual(p.returncode, 0, p.stdout.decode('utf-8', 'replace')[-2000:])
            self.assertIn('论文取较保守的一档：有效', p.stdout.decode('utf-8'))
            p = subprocess.run([sys.executable, '-B', str(script), str(bad)], capture_output=True)
            self.assertEqual(p.returncode, 2)
            p = subprocess.run([sys.executable, '-B', str(script), str(Path(d) / 'none.txt')], capture_output=True)
            self.assertEqual(p.returncode, 1)
            p = subprocess.run([sys.executable, '-B', str(script)], capture_output=True)
            self.assertEqual(p.returncode, 1)

    def test_deterministic_output(self):
        out1 = V.analyze(render(fix_claims(base_rows())))
        out2 = V.analyze(render(fix_claims(base_rows())))
        self.assertEqual(V.report(out1, 'x', 'y') + V.tables(out1), V.report(out2, 'x', 'y') + V.tables(out2))



class RevisionTests(unittest.TestCase):
    """独立审查后补的行为：E4 写法与三条限制、两个范围的措辞取较保守者、类型检查、健壮性、门槛附近的显示"""

    def report_of(self, rows):
        out = run(rows)
        return out, V.report(out, 'x', '0' * 64) + V.tables(out)

    def e4rows(self, A, B, A_r1=(0.0, -0.2, 0.2), B_r1=(0.0, -0.2, 0.2), scopes=('full', 'subset')):
        rows = base_rows()
        for sc in scopes:
            setc(rows, sc, SEEDS, 'A', ap=A, r1=A_r1)
            setc(rows, sc, SEEDS, 'B', ap=B, r1=B_r1)
        return rows

    def test_e4_wording_for_each_outcome(self):
        cases = {'①a': ((0.8, 0.3, 1.3), (0.0, -0.1, 0.1), (0.0, -0.2, 0.2)),
                 '①b': ((0.4, 0.1, 0.7), (0.0, -0.1, 0.1), (0.0, -0.2, 0.2)),
                 '②': ((0.8, 0.3, 1.3), (-0.2, -0.4, 0.0), (-0.35, -0.6, -0.1)),
                 '③': ((0.0, -0.2, 0.2), (0.0, -0.1, 0.1), (0.0, -0.2, 0.2))}
        for outcome, (A, B, B_r1) in cases.items():
            out, text = self.report_of(self.e4rows(A, B, B_r1=B_r1))
            self.assertEqual(out['scopes']['full']['E4'], outcome)
            self.assertIn(V.E4_WORDING[outcome], text)
            self.assertEqual(V.SECOND_LIMIT in text, outcome in ('②', '①b'), outcome)
        out, text = self.report_of(self.e4rows((0.8, 0.3, 1.3), (0.2, 0.05, 0.35), A_r1=(-0.4, -0.7, -0.1)))
        self.assertEqual(out['combined']['E4'], '未覆盖')
        self.assertIn(V.E4_WORDING['未覆盖'], text)

    def test_other_group_wording_only_for_route_other_groups(self):
        out, text = self.report_of(self.e4rows((0.8, 0.3, 1.3), (0.0, -0.1, 0.1)))
        self.assertIn('另一组 B 的写法', text)
        self.assertNotIn('另一组 A 的写法', text)
        out, text = self.report_of(self.e4rows((0.0, -0.2, 0.2), (0.0, -0.1, 0.1)))
        self.assertIn('另一组 A 的写法', text)
        self.assertIn('另一组 B 的写法', text)

    def test_curve_third_limitation(self):
        out, text = self.report_of(base_rows())
        self.assertEqual(out['scopes']['full']['curve']['hits'], [])
        self.assertIn('已测的 8 个常数取值都没有同时在两组上超过 QE+M', text)
        rows = base_rows()
        rows[('r17', 'full', 102, 'curve')]['0.75'] = {'A_mAP': 60.0, 'A_R1': 41.0, 'B_mAP': 60.0, 'B_R1': 36.0}
        out, text = self.report_of(rows)
        self.assertEqual(out['scopes']['full']['curve']['hits'], [(102, 0.75)])
        self.assertEqual(out['combined']['curve_hits'], [('full', 102, 0.75)])
        self.assertIn('s102 γ=0.75', text)
        self.assertIn('必须如实写出', text)
        self.assertEqual(out['final'], '有效')

    def test_curve_near_tie_and_incomplete_grid(self):
        rows = base_rows()
        qa = rows[('cells', 'subset', 103, 'abs|QE+M|A')]['mAP']
        qb = rows[('cells', 'subset', 103, 'abs|QE+M|B')]['mAP']
        rows[('r17', 'subset', 103, 'curve')]['0.5'].update(A_mAP=qa + 5e-7, B_mAP=qb - 5e-7)   # 容差内可能翻转
        rows[('r17', 'subset', 103, 'curve')]['0.25']['A_mAP'] = qa + 5e-7                  # 另一组远低于 QE+M：不算
        out, text = self.report_of(rows)
        r = out['scopes']['subset']
        self.assertEqual(r['curve']['near'], [(103, 0.5)])
        self.assertEqual(r['curve']['hits'], [])
        self.assertIn('需核对', text)
        self.assertIn('第三条限制须人工核对', text)
        self.assertNotIn('写"已测的 8 个常数取值都没有同时在两组上超过 QE+M"', text.split('======== 结论')[1])
        self.assertEqual(r['tier'], '有效')
        rows = base_rows()
        del rows[('r17', 'subset', 101, 'curve')]['0.05']
        out, text = self.report_of(rows)
        r = out['scopes']['subset']
        self.assertTrue(any('不是 8 个已测点' in w for w in r['warnings']))
        self.assertFalse(r['curve']['complete'])
        self.assertIn('第三条限制须人工核对', text.split('======== 结论')[1])
        rows = base_rows()
        rows[('r17', 'full', 102, 'curve')]['0.0'] = dict(rows[('r17', 'full', 102, 'curve')]['0'])
        r = run(rows)['scopes']['full']
        self.assertTrue(any('出现不止一次' in w for w in r['warnings']))
        self.assertFalse(r['curve']['complete'])

    def test_combine_takes_conservative_wording(self):
        rows = base_rows()
        setc(rows, 'full', SEEDS, 'A', ap=(0.8, 0.3, 1.3))
        setc(rows, 'full', SEEDS, 'B', ap=(0.0, -0.1, 0.1))
        setc(rows, 'subset', SEEDS, 'A', ap=(0.8, 0.3, 1.3))
        setc(rows, 'subset', [102], 'A', ap=(0.5, -0.1, 1.1))
        setp(rows, 'subset', [101], 'QE+M', 'QE-DBA', 'B', lo=-0.45)
        setp(rows, 'subset', [103], 'QE+M', 'QEir', 'B', dmAP=-0.5)
        out, text = self.report_of(rows)
        f, s, c = out['scopes']['full'], out['scopes']['subset'], out['combined']
        self.assertEqual((f['E4'], s['E4'], c['E4']), ('①a', '③', '③'))
        self.assertEqual(c['E2']['mAP'], '点估计在原容差内，但不能排除超出容差的损害')
        self.assertEqual((f['E3'], s['E3'], c['E3']), (True, False, False))
        self.assertEqual(set(c['differ']), {'E2 写法', 'E3', 'E4', 'E4 得益组'})
        self.assertEqual(c['E4_gain'], {'full': ['A'], 'subset': []})
        self.assertIn('全量与子集不一致的项', text)
        self.assertEqual(out['final'], '有效')

    def test_type_checks(self):
        cases = [
            (('status', 'full', 101, 'judge'), {'evalset': 'test', 'exit_status': False}, 'judge_test 的状态不对'),
            (('r17', 'full', 102, 'gamma|f7001'), {'gamma': 'nan'}, '8 点网格'),
            (('r17', 'full', 102, 'gamma|f7002'), {'gamma': True}, '8 点网格'),
            (('r17', 'subset', 103, 'gamma|f7003'), {'gamma': '0.3'}, '8 点网格'),
        ]
        for key, value, text in cases:
            rows = base_rows()
            rows[key] = value
            out = run(rows)
            r = out['scopes'][key[1]]
            self.assertEqual(r['tier'], '实现或协议问题', key)
            self.assertTrue(any(text in p for p in r['invalid']), r['invalid'])
        rows = base_rows()
        del rows[('r17', 'full', 101, 'locality')]['QE+EM']
        self.assertTrue(any('局部性记录缺' in p for p in run(rows)['scopes']['full']['invalid']))
        rows = base_rows()
        rows[('status', 'subset', 102, 'runner|const/s102_f7009')] = {'events': 0, 'heads': {}}
        self.assertTrue(any('多出' in p for p in run(rows)['scopes']['subset']['invalid']))

    def test_gamma_grid_spellings(self):
        rows = base_rows()
        for s, g in ((101, '0'), (102, '0.0'), (103, '1')):
            rows[('r17', 'full', s, 'gamma|f7001')] = {'gamma': g}
        rows[('r17', 'full', 101, 'gamma|f7002')] = {'gamma': 0.05}
        r = run(rows)['scopes']['full']
        self.assertEqual(r['tier'], '有效')
        self.assertEqual(r['gamma_zero'], ['s101 f7001', 's102 f7001'])

    def test_robust_formatting_and_tables(self):
        self.assertEqual(V.h2(float('inf')), 'inf')
        V.h2(1e300)
        V.sgx(1e300)
        rows = base_rows()
        rows[('r17', 'full', 101, 'gamma|f7001')] = ['x']
        rows[('xcam', 'full', 102, 'ratio')] = ['y']
        out = run(rows)
        self.assertEqual(out['scopes']['full']['tier'], '实现或协议问题')
        V.tables(out)

    def test_main_returns_3_on_internal_error(self):
        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / 'good.txt'
            good.write_text(render(fix_claims(base_rows())), encoding='utf-8', newline='\n')
            orig = V.tables
            try:
                V.tables = lambda out: 1 / 0
                self.assertEqual(V.main(['r18_verdict.py', str(good)]), 3)
            finally:
                V.tables = orig
            orig = V.analyze
            try:
                V.analyze = lambda text: {}['boom']
                self.assertEqual(V.main(['r18_verdict.py', str(good)]), 3)
            finally:
                V.analyze = orig

    def test_threshold_annotation_only_in_readings(self):
        rows = base_rows()
        setp(rows, 'full', [101], 'QE+M', 'QE-DBA', 'A', dmAP=0.99999)
        out, text = self.report_of(rows)
        self.assertIn('+1.00(0.99999) [+2.00, +4.00] ✗', text)
        self.assertEqual(out['scopes']['full']['tier'], '有限适用')
        self.assertNotIn('(0.99999)', V.tables(out))
        self.assertEqual(V.sgx(0.9999996), '+1.00(0.9999996)')
        self.assertEqual(V.sgx(-0.3000004), '-0.30(-0.3000004)')

    def test_conclusion_sentences(self):
        out, text = self.report_of(base_rows())
        self.assertIn(V.TIER_SENTENCE['有效'], text)
        self.assertIn('不写"独立"', text)
        self.assertIn('不写成跨视角泛化', text)
        rows = base_rows()
        rows[('status', 'full', 101, 'r17')]['valid'] = False
        out, text = self.report_of(rows)
        self.assertEqual(out['final'], '实现或协议问题')
        self.assertIn(V.TIER_SENTENCE['实现或协议问题'], text)
        self.assertNotIn('措辞（全量与子集取较保守者', text)


class RoundTwoMinorTests(unittest.TestCase):
    def test_gain_group_difference_flagged(self):
        rows = base_rows()
        setc(rows, 'full', SEEDS, 'A', ap=(0.8, 0.3, 1.3))
        setc(rows, 'full', SEEDS, 'B', ap=(0.0, -0.1, 0.1))
        setc(rows, 'subset', SEEDS, 'A', ap=(0.0, -0.2, 0.2))
        setc(rows, 'subset', SEEDS, 'B', ap=(0.5, 0.2, 0.8))
        out = run(rows)
        c = out['combined']
        self.assertEqual((out['scopes']['full']['E4'], out['scopes']['subset']['E4'], c['E4']), ('①a', '①a', '①a'))
        self.assertEqual(c['E4_gain'], {'full': ['A'], 'subset': ['B']})
        self.assertIn('E4 得益组', c['differ'])
        self.assertIn('得益组：全量 A；子集 B', V.report(out, 'x', '0' * 64))

    def test_curve_hit_named_in_conclusion(self):
        rows = base_rows()
        rows[('r17', 'subset', 101, 'curve')]['1'] = {'A_mAP': 99.0, 'A_R1': 41.0, 'B_mAP': 99.0, 'B_R1': 36.0}
        out = run(rows)
        concl = V.report(out, 'x', '0' * 64).split('======== 结论')[1]
        self.assertIn('必须如实写出', concl)
        self.assertIn('子集 s101 γ=1', concl)


if __name__ == '__main__':
    unittest.main()
