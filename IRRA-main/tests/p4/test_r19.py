"""Synthetic-only R19 acceptance; no image, weight or official archive is read."""
from pathlib import Path
import csv
import importlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import unittest
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE), str(Path(__file__).parent)]
from tools.p4.r19.core import (nested_indices, capped_indices, camera_states,
    select_lambda, refit_weights, fit_xcam, xcam_structure, compare_cells)
from test_r17 import make_archive
T = CODE/'tools/p4/transfer'


def softmax(x, mask):
    x = np.where(mask, x, -np.inf)
    mx = x.max(1,keepdims=True); mx = np.where(np.isfinite(mx),mx,0)
    e = np.exp(x-mx)*mask
    return e/np.maximum(e.sum(1,keepdims=True),1e-300)


class HelperTests(unittest.TestCase):
    def test_lambda_feasible_tie_and_fallback(self):
        rows = [dict(**{'lambda':v}, dmAP=[gain]*5,dR1=[r]*5,dR10=[r10]*5)
                for v,gain,r,r10 in [(0,0,0,0),(.2,2,-.2,-.8),(.4,2,-.3,-.7),(.6,3,-.3,-1.5)]]
        self.assertEqual(select_lambda(rows,-1),(0.2,True))
        self.assertEqual(select_lambda(rows,-2),(0.6,True))
        for row in rows: row['dR1']=[-2]*5
        self.assertEqual(select_lambda(rows,-1),(0.,False))

    def test_nested_assignment_and_unambiguous_order(self):
        rng=np.random.default_rng(7);blank=(np.ones(8),np.ones(8))
        full=make_archive(rng,6,100,8,blank)
        qm=full['clean/txt_pids']!=2;gm=full['clean/img_pids']!=2
        qkeys={'clean/text_features','clean/txt_pids','clean/query_raw_pids'}
        sub={k:v[qm if k in qkeys else gm] for k,v in full.items()}
        ix=nested_indices(full,sub);np.testing.assert_array_equal(ix,np.flatnonzero(gm))
        states=np.random.default_rng(7031).choice(['clean','rgb_missing'],len(gm))
        np.testing.assert_array_equal(states[ix],states[gm])
        sub['clean/f_ir']=sub['clean/f_ir'][::-1]
        with self.assertRaises(ValueError):nested_indices(full,sub)

    def test_sparse_queries_unchanged_and_rng_private(self):
        pids=np.repeat(np.arange(5),[8,4,3,2,1]);before=np.random.get_state()
        ix=capped_indices(pids,3,9101);again=capped_indices(pids,3,9101)
        np.testing.assert_array_equal(ix,again)
        self.assertTrue(all((pids[ix]==p).sum()<=3 for p in np.unique(pids)))
        after=np.random.get_state()
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(before,after)))

    def test_camera_mixtures_do_not_use_seed(self):
        cams=np.array(['cam1','cam2','cam3','cam1'])
        self.assertEqual(camera_states(cams,'cam1').tolist(),['rgb_missing','clean','clean','rgb_missing'])
        self.assertEqual(camera_states(cams,'cam2').tolist(),['clean','rgb_missing','clean','clean'])

    def test_refit_inference_never_requires_pid_and_blocks_same_camera(self):
        g=dict(J=np.array([0,1]),Fi=np.array([2,3]),CF=np.array([[.8,.4],[.7,.3]]),X=np.zeros((2,9)))
        g['X'][:,3]=.2
        class Head:
            def predict_proba(self,x):return np.tile([.2,.8],(len(x),1))
        fit=dict(mu=np.zeros(9),sd=np.ones(9),context_mu=np.zeros(2),context_sd=np.ones(2),head=Head(),kappa=20)
        cams=np.array(['cam1','cam2','cam1','cam1'])
        w,available,r=refit_weights(g,cams,[0,.3],fit,softmax)
        self.assertFalse(available[0]);self.assertTrue(available[1]);np.testing.assert_array_equal(w[0],0)
        self.assertAlmostEqual(w[1].sum(),.8)
        g['SF']=np.array([[True,False],[True,False]]);g['tm']=np.ones(2,bool)
        _,_,mask=xcam_structure(g,cams)
        self.assertEqual((g['SF']&mask).any(1).tolist(),[False,True])
        from sklearn.linear_model import LogisticRegression
        # Repeated controlled fitting episodes exercise labels and masked kappa.
        episodes=[('ind',np.arange(4),None,None,g,None)]*2
        result=fit_xcam(episodes,np.arange(4),cams,[[0,.3]]*2,softmax,LogisticRegression)
        self.assertEqual(result['labels'].tolist(),[False,True,False,True])
        empty=dict(J=np.array([0]),Fi=np.array([],dtype=int),CF=np.full((1,1),-1.),
                   SF=np.zeros((1,1),bool),X=np.zeros((1,9)),tm=np.ones(1,bool))
        episodes.append(('ind',np.arange(4),None,None,empty,None))
        result=fit_xcam(episodes,np.arange(4),cams,[[0,.3]]*3,softmax,LogisticRegression)
        self.assertEqual(result['labels'].tolist(),[False,True,False,True,False])
        self.assertEqual(result['rows'],5)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='r19_tests_');cls.root=Path(cls.temp.name)
        rng=np.random.default_rng(17);dim=8;blank=(rng.standard_normal(dim),rng.standard_normal(dim))
        cls.dev=make_archive(rng,14,100000,dim,blank);cls.test=make_archive(rng,10,800000,dim,blank)
        (cls.root/'feats').mkdir();np.savez(cls.root/'feats/dev.npz',**cls.dev);np.savez(cls.root/'test.npz',**cls.test)
        tr=make_archive(rng,20,900000,dim,blank)
        np.savez(cls.root/'train.npz',f_ir=tr['clean/f_ir'],f_rgb=tr['clean/f_rgb'],raw_pids=tr['clean/query_raw_pids'][::2][tr['clean/img_pids']])
        np.savez(cls.root/'cams.npz',dev_cams=np.array(['cam1','cam2'])[np.arange(len(cls.dev['clean/img_pids']))%2],test_cams=np.array(['cam1','cam2'])[np.arange(len(cls.test['clean/img_pids']))%2])
        cls.common=['--b12','.', '--feats',str(cls.root/'feats'),'--base','dev','--adapter','dev','--assign','none','--assign_seeds','7031','--mixes','M0','M4','--skip_joint','--grid_preset','dev','--select_rule','constrained','--r_max','0.999999','--tie_break','bstar','--test_npz',str(cls.root/'test.npz'),'--link_map','train_lda','--train_ir_npz',str(cls.root/'train.npz'),'--test_seed','7001','--tag','fixture','--perq_ext']
        cls.run_script('ref','r18_xcam.py',['B*','QE-DBA','QE+M','QE+Const','A-2f-ctxn'])
        cls.run_script('default','r19_variants.py',['B*','QE-DBA','QE+M','QE+Const','A-2f-ctxn'])
        selection=str(cls.root/'ref/r12_fixture_selection.json')
        cls.methods=['B*','QE-DBA','QE+M','QE+Const','QE+NoGate','A-2f-ctxn','FeatImp-r','QE+CtxOnly','QE+SRC2f+kap','QE+SRC2f+pI','QE+M-lam','QE+M-lam2','QE+M-xcam-refit']
        cls.run_script('new','r19_variants.py',cls.methods,['--fixed_selection',selection,'--lam_curve','--lam_grid','0','.5','1','--item_cams',str(cls.root/'cams.npz'),'--max_per_id','3','--regress_against',str(cls.root/'ref'),'--nogate_rmax'])

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    @classmethod
    def run_script(cls,name,script,methods,extra=(),check=True):
        out=cls.root/name;out.mkdir()
        env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        wrapper="import runpy,sys,pickle,pathlib; script=sys.argv[1]; sys.argv=sys.argv[1:]; ns=runpy.run_path(script,run_name='__main__'); pickle.dump((ns['res'],ns['g']),open(pathlib.Path(ns['args'].out)/'test_last_gallery.pkl','wb')); pickle.dump(ns['ASSIGN'],open(pathlib.Path(ns['args'].out)/'test_assignments.pkl','wb'))"
        cmd=[sys.executable,'-B','-c',wrapper,str(T/script),'--out',str(out),'--methods',*methods,*cls.common,*extra]
        p=subprocess.run(cmd,cwd=CODE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        if check and p.returncode:raise AssertionError(p.stdout.decode('utf-8',errors='replace')[-8000:])
        return p

    def test_defaults_are_byte_identical_to_frozen(self):
        for suffix in ['cells.csv','selection.json','diag_cells.csv','summary.csv']:
            self.assertEqual((self.root/'ref'/('r12_fixture_'+suffix)).read_bytes(),(self.root/'default'/('r12_fixture_'+suffix)).read_bytes(),suffix)

    def test_curves_and_perquery_storage(self):
        with (self.root/'new/r17_fixture_perq_ext.pkl').open('rb') as f:records=pickle.load(f)['records']
        self.assertFalse(any(k[-1].startswith('QE+M@') for k in records))
        self.assertTrue(any(k[-1]=='QE+M-lam' for k in records))
        for k,r in records.items():
            if k[-1] not in ('QE-DBA','QE+M','QE+Const'):self.assertNotIn('top10',r)
        c=pd.read_csv(self.root/'new/r12_fixture_cells.csv')
        for x,y in [('QE+M@0','QE-DBA'),('QE+M@1','QE+M')]:
            a=c[c.method==x].set_index(['evalset','mixture','aseed']);b=c[c.method==y].set_index(['evalset','mixture','aseed'])
            pd.testing.assert_frame_equal(a[['mAP','R1','R5','R10']],b[['mAP','R1','R5','R10']],check_exact=True)
        for k,r in records.items():
            if k[1]=='k3':self.assertEqual(len(r['q']),len(self.test['clean/txt_pids']))

    def test_ctxonly_equal_gate_and_feature_gate_assertion(self):
        # These are actual weights from the last synthetic gallery, captured by
        # runpy after the unmodified CLI evaluation, not reimplemented formulas.
        with (self.root/'new/test_last_gallery.pkl').open('rb') as f:res,g=pickle.load(f)
        r=res['QE+CtxOnly'][1].sum(1)
        np.testing.assert_allclose(r,np.repeat(r[0],len(r)),rtol=0,atol=1e-16)
        np.testing.assert_array_equal(res['FeatImp-r'][1],res['A-2f-ctxn'][1])
        diag=pd.read_csv(self.root/'new/r12_fixture_diag_cells.csv')
        self.assertTrue({'QE+CtxOnly','FeatImp-r','QE+M-xcam-refit'}<=set(diag.method))
        self.assertIn('auc_xcam',diag.columns)
        log=json.loads((self.root/'new/r17_fixture_r17log.json').read_text())
        self.assertIn('xcam_refit',log['r17log'][0])
        association=softmax(log['r17log'][0]['ka_used']*g['CF'],np.ones_like(g['CF'],bool))
        np.testing.assert_array_equal(res['QE+NoGate'][1],.999999*association)

    def test_cli_nested_and_camera_assignments(self):
        qmask=self.test['clean/txt_pids']!=2;gmask=self.test['clean/img_pids']!=2
        qkeys={'clean/text_features','clean/txt_pids','clean/query_raw_pids'}
        subset={k:v[qmask if k in qkeys else gmask] for k,v in self.test.items()}
        np.savez(self.root/'subset.npz',**subset)
        with np.load(self.root/'cams.npz') as z:
            np.savez(self.root/'subset_cams.npz',dev_cams=z['dev_cams'],test_cams=z['test_cams'][gmask])
        self.run_script('nested','r19_variants.py',['B*','QE-DBA','QE+M'],[
            '--fixed_selection',str(self.root/'ref/r12_fixture_selection.json'),
            '--test_npz',str(self.root/'subset.npz'),'--assign_nested_from',str(self.root/'test.npz'),
            '--item_cams',str(self.root/'subset_cams.npz'),'--mixes','M4','Mcam1','Mcam2'])
        with (self.root/'new/test_assignments.pkl').open('rb') as f:full=pickle.load(f)
        with (self.root/'nested/test_assignments.pkl').open('rb') as f:nested=pickle.load(f)
        d=len(self.dev['clean/img_pids'])
        np.testing.assert_array_equal(nested[('M4',7031)][d:],full[('M4',7031)][d:][gmask])
        with np.load(self.root/'subset_cams.npz') as z:
            for target in ('cam1','cam2'):
                np.testing.assert_array_equal(nested[('M'+target,7031)][d:],camera_states(z['test_cams'],target))

    def test_regression_mismatch_exit_four(self):
        ref=self.root/'bad_reference';ref.mkdir()
        c=pd.read_csv(self.root/'ref/r12_fixture_cells.csv');c.loc[0,'mAP']+=1
        c.to_csv(ref/'r12_bad_cells.csv',index=False)
        p=self.run_script('mismatch','r19_variants.py',['B*','QE-DBA','QE+M'],['--fixed_selection',str(self.root/'ref/r12_fixture_selection.json'),'--regress_against',str(ref)],check=False)
        self.assertEqual(p.returncode,4,p.stdout.decode('utf-8',errors='replace')[-2000:])


def load_tests(loader, tests, pattern):
    # One requested entry point includes the independently authored components.
    for name in ('test_r19_training','test_r19_judge','test_r19_driver'):
        if (Path(__file__).parent/(name+'.py')).is_file():
            tests.addTests(loader.loadTestsFromModule(importlib.import_module(name)))
    return tests


if __name__=='__main__':unittest.main(verbosity=2)
