"""CPU-only synthetic records for R19 judgment alignment and frozen rule reuse."""
import csv
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import numpy as np

CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE))
from tools.p4.r19 import r19_judge as judge


METHODS=('B*','QE-DBA','QEir','QE+CtxOnly','QE+M','QE+M-lam','QE+M-lam2',
         'A-2f-ctxn','FeatImp-r','FeatImp-g','QE+CAET','QE+Const')


def synthetic_units(root,seed=101):
    """Six queries, unequal identity sizes, explicit fit/assignment effects."""
    q=np.arange(6,dtype=np.int32)
    qid=np.array([11,11,11,12,12,13])
    for fit in judge.FITS:
        d=Path(root)/f's{seed}_f{fit}'
        d.mkdir(parents=True)
        records={};cells=[]
        for mx_index,mx in enumerate(judge.A+judge.B):
            for aseed in judge.ASSIGN_SEEDS:
                for method in METHODS:
                    delta={'B*':0,'QE-DBA':1,'QEir':.5,'QE+M':3,'QE+CtxOnly':2,
                           'QE+M-lam':2.5,'QE+M-lam2':2.4,'A-2f-ctxn':3,
                           'FeatImp-r':2,'FeatImp-g':1.5,'QE+CAET':2,'QE+Const':2}[method]
                    ap=30+np.arange(6)*.1+delta+mx_index*.01+(fit-7001)*.03+(aseed-7031)*.001
                    record=dict(q=q.copy(),qid=qid.copy(),AP=ap,
                                R1=np.array([1,0,1,0,1,0],dtype=np.uint8),
                                R5=np.array([1,1,1,0,1,0],dtype=np.uint8),
                                R10=np.ones(6,dtype=np.uint8),mINP=ap*.6)
                    records[(fit,'test',mx,aseed,method)]=record
                    cells.append(dict(split=fit,fold=0,evalset='test',mixture=mx,aseed=aseed,method=method,
                                      mAP=ap.mean(),R1=50.,R5=400./6,R10=100.,mINP=record['mINP'].mean()))
        with (d/'r12_fixture_cells.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
        with (d/'r12_fixture_diag_cells.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['evalset','mixture','aseed','method','coverage','wrong_mass'])
            w.writeheader();w.writerow(dict(evalset='test',mixture='M4',aseed=7031,method='QE+M',coverage=.8,wrong_mass=.1))
        (d/'r12_fixture_selection.json').write_text(json.dumps([dict(method='QE-DBA',chosen='(1, 1, 1)',feasible=True)]),encoding='utf-8')
        with (d/'r17_fixture_perq_ext.pkl').open('wb') as f:
            pickle.dump(dict(meta=dict(assign_seeds=list(judge.ASSIGN_SEEDS)),records=records),f)


class JudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='r19_judge_')
        cls.root=Path(cls.temp.name)
        synthetic_units(cls.root)
        cls.data=judge.SeedData(cls.root,101)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_15_cells_then_group_and_alignment(self):
        d=self.data
        expected=np.zeros(6,dtype=np.float64)
        for mixture in judge.A:
            for fit in judge.FITS:
                for seed in judge.ASSIGN_SEEDS:
                    expected+=d.units[fit][(mixture,seed,'QE+M')]['AP']
        expected/=75
        np.testing.assert_array_equal(d.means('QE+M',judge.A,'AP'),expected)
        pair=d.pair('QE+M','QE-DBA','A')
        self.assertEqual(pair['n_cells'],75)
        self.assertEqual(pair['n_q'],6)
        self.assertEqual(pair['n_id'],3)
        self.assertAlmostEqual(pair['dmAP'],2.)

    def test_bootstrap_is_exact_frozen_function_and_rng_unchanged(self):
        x=np.array([.1,.4,.7,-.2,.3,.6]);ids=np.array([8,8,8,9,9,10])
        state=np.random.get_state()
        scope={'np':np,'a':judge.SimpleNamespace(boot=4000)}
        fn=judge.extract_function(judge.TRANSFER/'r17_judge.py','boot_ci',scope)
        self.assertEqual(judge.frozen_boot(x,ids),fn(x,ids,seed=7))
        newer=np.random.get_state()
        self.assertEqual(state[0],newer[0]);np.testing.assert_array_equal(state[1],newer[1]);self.assertEqual(state[2:],newer[2:])

    def test_cell_mismatch_and_missing_assignment_fail(self):
        with tempfile.TemporaryDirectory(prefix='r19_bad_') as t:
            root=Path(t);synthetic_units(root)
            p=root/'s101_f7001/r17_fixture_perq_ext.pkl'
            with p.open('rb') as f:env=pickle.load(f)
            key=(7001,'test','M4',7031,'QE+M')
            original=env['records'][key]['AP'].copy()
            env['records'][key]['AP']+=1
            with p.open('wb') as f:pickle.dump(env,f)
            with self.assertRaisesRegex(ValueError,'Cell/per-query disagreement'):judge.SeedData(root,101)
            env['records'][key]['AP']=original;del env['records'][key]
            with p.open('wb') as f:pickle.dump(env,f)
            with self.assertRaisesRegex(ValueError,'Missing perq_ext'):judge.SeedData(root,101)

    def test_query_identity_changes_fail(self):
        with tempfile.TemporaryDirectory(prefix='r19_alignment_') as t:
            root=Path(t);synthetic_units(root)
            p=root/'s101_f7002/r17_fixture_perq_ext.pkl'
            with p.open('rb') as f:env=pickle.load(f)
            env['records'][(7002,'test','M4',7031,'QE+M')]['qid'][0]=999
            with p.open('wb') as f:pickle.dump(env,f)
            with self.assertRaisesRegex(ValueError,'Query/identity alignment differs'):judge.SeedData(root,101)

    def test_curve_cells_need_no_perquery_records(self):
        with tempfile.TemporaryDirectory(prefix='r19_curve_') as t:
            root=Path(t);synthetic_units(root)
            for fit in judge.FITS:
                p=root/f's101_f{fit}/r12_fixture_cells.csv'
                with p.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
                extra=dict(rows[0]);extra['method']='QE+M@0.5';rows.append(extra)
                with p.open('w',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            d=judge.SeedData(root,101)
            self.assertNotIn('QE+M@0.5',d.methods)
            self.assertEqual(len(d.means('QE+M',judge.A,'AP')),6)

    def test_frozen_claims_and_e_forms_reuse(self):
        data={s:self.data for s in judge.SEEDS}
        claims={s:judge.frozen_claims(d) for s,d in data.items()}
        self.assertEqual(set(claims[101]['claims']),{'C1','C2','C3','C4','C5'})
        self.assertTrue(claims[101]['claims']['C1'])
        self.assertTrue(claims[101]['claims']['C2'])
        forms=judge.frozen_e_forms(data,claims)
        self.assertTrue(forms['post_hoc'])
        self.assertTrue(forms['form_only'])
        self.assertTrue(forms['E1']);self.assertTrue(forms['E2'])
        self.assertIn(forms['E4'],('①a','①b','②','③','未覆盖'))
        self.assertEqual(len(forms['source_sha256']),64)

    def test_rules_are_fixed_and_v1_thresholds_strict(self):
        class Fake:
            def pair(self,left,right,group):
                if right=='QE+CtxOnly':value=.3 if group=='A' else .09
                elif right=='FeatImp-r':value=1 if group=='A' else .1
                else:value=2 if left=='QE+M' else 1.5
                return dict(dmAP=value,lo=.01,metrics={'R10':{'mean':0.}})
        out=judge.s1_rules({s:Fake() for s in judge.SEEDS})
        self.assertFalse(out['V1a'])  # exact .3 is not < .3
        self.assertFalse(out['V1b'])
        self.assertTrue(out['V6_dataset'])
        self.assertEqual(judge.ASSIGN_SEEDS,(7031,7032,7033,7034,7035))

    def test_v2_A_loss_zero_boundary_and_lambda2_is_separate(self):
        class Fake:
            def __init__(self,loss):self.loss=loss
            def pair(self,left,right,group):
                if right=='QE+CtxOnly':value=.2
                elif right=='FeatImp-r':value=.1
                else:value=2. if left=='QE+M' else 1.4 if left=='QE+M-lam' else 0.
                r10=0. if left=='QE+M' else -self.loss
                if group=='B':r10=-100.  # V2 must use only group A.
                return dict(dmAP=value,lo=.01,metrics={'R10':{'mean':r10}})
        good=judge.s1_rules({s:Fake(0.) for s in judge.SEEDS})
        self.assertTrue(good['V2']['holds'])
        self.assertFalse(good['V2_lam2_descriptive']['holds'])
        self.assertTrue(all(r['zero_main_loss_requires_zero_lambda_loss'] for r in good['V2']['per_seed']))
        bad=judge.s1_rules({s:Fake(1e-12) for s in judge.SEEDS})
        self.assertFalse(bad['V2']['holds'])

    def test_v4_absolute_boundary_and_nonpositive_full_has_no_flag(self):
        gains=dict(rgbnt=[2.,2.,2.],rgbnt_subset=[2.,2.,2.],rgbnt_nested=[1.5,1.5,1.5],rgbnt_sparse=[.9,.9,.9])
        report=judge.v4_rules(gains)
        self.assertTrue(report['V4a']);self.assertTrue(report['V4b_repeated_observation_dependence'])
        gains['rgbnt_nested']=[1.49]*3
        self.assertFalse(judge.v4_rules(gains)['V4a'])
        gains['rgbnt']=[0.]*3
        result=judge.v4_rules(gains)
        self.assertFalse(result['V4b_applicable'])
        self.assertIsNone(result['V4b_repeated_observation_dependence'])

    def test_markdown_has_tables_and_posthoc_label(self):
        report=dict(rules=judge.RULES,datasets={},verdicts={'V1a':False})
        text=judge.markdown(report)
        for x in ('Table V1','Table V2','Table V3','Table V4','post-hoc'):
            self.assertIn(x,text)

    def test_s4_decomposition_is_additive_and_in_markdown(self):
        pairs=[('QE-DBA','B*'),('QE+Const','QE-DBA'),('QE+M','QE+Const'),('QE+M','B*')]
        summary=judge.summary_dataset({101:self.data},pairs)
        report=dict(rules=judge.RULES,datasets={'synthetic':summary},verdicts={},V5_post_hoc={})
        text=judge.markdown(report)
        self.assertIn('S4 decomposition: front-end / constant transfer / learned gate',text)
        self.assertIn('QE+Const - QE-DBA',text)
        for group in ('A','B'):
            values=[summary[101]['pairs'][left+'|'+right+'|'+group]['dmAP'] for left,right in pairs]
            self.assertAlmostEqual(sum(values[:3]),values[3],places=12)


if __name__=='__main__':
    unittest.main(verbosity=2)
