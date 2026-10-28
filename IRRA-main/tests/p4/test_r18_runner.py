"""R18 guarded execution: exact binary-fit parity and narrow single-class scope."""
from pathlib import Path
from collections import Counter
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE));sys.path.insert(0,str(Path(__file__).parent))
from tools.p4.r18.run_frozen import guarded_class,run,GUARDED_HEADS,FORBIDDEN_METHODS
from tools.p4.r18.r18_exact import runner_records
import test_r17 as fixtures
from test_r18_eval import deep_equal
T=CODE/'tools/p4/transfer';RUNNER=CODE/'tools/p4/r18/run_frozen.py'
WORK=Path(os.environ.get('R18_WORK',r'E:\论文4\2过程文档\p4_rgbnt201\r18'))
ENV=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

class GuardTests(unittest.TestCase):
    def test_two_classes_are_bitwise_parent_fit(self):
        rng=np.random.default_rng(77);X=rng.standard_normal((80,7));y=np.arange(80)%2==0;events=[]
        a=LogisticRegression(C=1.0,max_iter=2000).fit(X,y)
        b=guarded_class(__file__,events)(C=1.0,max_iter=2000).fit(X,y)
        for name in ('coef_','intercept_','classes_','n_iter_'):np.testing.assert_array_equal(getattr(a,name),getattr(b,name),strict=True)
        np.testing.assert_array_equal(a.predict_proba(X),b.predict_proba(X),strict=True)
        self.assertFalse(events)

    def script(self,directory,name,head,label,foreign=False):
        target=directory/(name+'.py')
        source='import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nX=np.arange(24,dtype=float).reshape(8,3)\ny=np.full(8,'+str(label)+',dtype=bool)\n'
        source+=("exec(compile('anyh = LogisticRegression().fit(X, y)', 'foreign.py', 'exec'))\n" if foreign else head+' = LogisticRegression(C=1.0,max_iter=2000).fit(X, y)\n')
        source+="np.savez(__file__+'.npz',classes="+head+".classes_,coef="+head+".coef_,intercept="+head+".intercept_,prob="+head+".predict_proba(X),pred="+head+".predict(X))\n"
        target.write_text(source,encoding='utf-8',newline='\n');return target

    def test_allowed_true_and_false_constants_and_restoration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for head in GUARDED_HEADS:
                for label in (False,True):
                    p=self.script(root,head+str(label),head,label);log=root/(p.stem+'.json');argv=sys.argv
                    with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(run(str(p),[],sha(p),log),0)
                    import sklearn.linear_model as lm
                    self.assertIs(lm.LogisticRegression,LogisticRegression);self.assertIs(sys.argv,argv)
                    record=json.loads(log.read_text());self.assertEqual(len(record['events']),1)
                    self.assertEqual(record['events'][0],{'head':head,'file':str(p),'line':5,'samples':8,'label':label})
                    with np.load(str(p)+'.npz') as z:
                        np.testing.assert_array_equal(z['classes'],[False,True]);np.testing.assert_array_equal(z['coef'],np.zeros((1,3)))
                        np.testing.assert_array_equal(z['intercept'],[20. if label else -20.])
                        np.testing.assert_array_equal(z['prob'],np.tile([float(not label),float(label)],(8,1)))
                        np.testing.assert_array_equal(z['pred'],np.full(8,label))

    def test_nonwhitelisted_head_and_foreign_file_still_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name,head,foreign in [('other','src2N',False),('foreign','anyh',True)]:
                p=self.script(root,name,head,True,foreign);log=root/(name+'.json')
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(ValueError):run(str(p),[],sha(p),log)
                self.assertEqual(json.loads(log.read_text())['events'],[])

    def test_hash_and_forbidden_methods_exit_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);p=root/'script.py';p.write_text("raise RuntimeError('must not run')\n",encoding='utf-8',newline='\n')
            cases=[('0'*64,[])]+[(sha(p),['--methods','B*',m]) for m in sorted(FORBIDDEN_METHODS)]
            for index,(expected,args) in enumerate(cases):
                log=root/f'{index}.json'
                result=subprocess.run([sys.executable,'-B',str(RUNNER),'--expect_sha',expected,'--log',str(log),'--',str(p),*args],env=ENV,capture_output=True)
                self.assertEqual(result.returncode,2,result.stderr);self.assertEqual(json.loads(log.read_text())['events'],[])

    def test_target_exit_code_is_preserved_and_logged(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'exit.py';p.write_text('raise SystemExit(7)\n',encoding='utf-8',newline='\n');log=p.with_suffix('.json')
            result=subprocess.run([sys.executable,'-B',str(RUNNER),'--expect_sha',sha(p),'--log',str(log),'--',str(p)],env=ENV,capture_output=True)
            self.assertEqual(result.returncode,7);self.assertEqual(json.loads(log.read_text())['events'],[])

    def test_exact_runner_record_is_one_row_per_unit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'const';unit=root/'s101_f7001';unit.mkdir(parents=True)
            (unit/'degenerate_fits.json').write_text(json.dumps({'script':'frozen.py','script_sha256':'a'*64,'events':[{'head':'anyh'},{'head':'anyh'},{'head':'gateC'}]}),encoding='utf-8',newline='\n')
            records=[];runner_records(records,['full='+str(root)])
            self.assertEqual(len(records),1);self.assertEqual(records[0]['src'],'runner');self.assertEqual(records[0]['seed'],'101')
            self.assertEqual(records[0]['v']['events'],3);self.assertEqual(records[0]['v']['heads'],{'anyh':2,'gateC':1})

class FrozenParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(tempfile.mkdtemp(prefix='runner_',dir=WORK));root=cls.root;(root/'feats').mkdir()
        rng=np.random.default_rng(17);dim=32;blank=(rng.standard_normal(dim),rng.standard_normal(dim))
        dev=fixtures.make_archive(rng,40,100000,dim,blank);test=fixtures.make_archive(rng,24,800000,dim,blank);tr=fixtures.make_archive(rng,70,900000,dim,blank)
        np.savez(root/'feats/dev.npz',**dev);np.savez(root/'test.npz',**test)
        np.savez(root/'train.npz',f_ir=tr['clean/f_ir'],f_rgb=tr['clean/f_rgb'],raw_pids=tr['clean/query_raw_pids'][::2][tr['clean/img_pids']])
        cls.common=['--b12','.','--feats',root/'feats','--base','dev','--adapter','dev','--assign','none','--assign_seeds','7031','7032','--mixes','M0','M4','M5f','--skip_joint','--grid_preset','dev','--select_rule','constrained','--r_max','0.999999','--tie_break','bstar','--test_npz',root/'test.npz','--link_map','train_lda','--train_ir_npz',root/'train.npz','--test_seed','7001','--tag','t_s101_f7001']
        cls.methods=['B*','QE-DBA','QE+M','QE+CAET','QE+SRC2f','QE+EM'];cls.saved={}
        for script in ('r15_test.py','r17_const.py','r18_xcam.py'):
            for wrapped in (False,True):
                output=root/'same_output';output.mkdir();extra=[]
                if script!='r15_test.py':extra=['--perq_ext','--fixed_selection',next(cls.saved[('r15_test.py',False)].glob('*_selection.json'))]
                command=[sys.executable,'-B']
                if wrapped:command += [str(RUNNER),'--expect_sha',sha(T/script),'--log',str(output/'degenerate_fits.json'),'--']
                command += [str(T/script),'--out',str(output),'--methods',*cls.methods,*map(str,cls.common),*map(str,extra)]
                cls.execute(script+str(wrapped),command)
                dest=root/(script+'_'+str(wrapped));shutil.copytree(output,dest);cls.saved[(script,wrapped)]=dest
                assert output.resolve().is_relative_to(root.resolve());shutil.rmtree(output)

    @classmethod
    def execute(cls,name,command):
        result=subprocess.run(command,cwd=CODE,env=ENV,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (cls.root/(name+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8',newline='\n')
        if result.returncode:raise RuntimeError(name+': '+result.stdout[-1500:]+result.stderr[-5000:])
        return result

    def test_all_three_scripts_direct_and_wrapped_equal(self):
        for script in ('r15_test.py','r17_const.py','r18_xcam.py'):
            with self.subTest(script=script):
                a,b=self.saved[(script,False)],self.saved[(script,True)]
                self.assertEqual(json.loads((b/'degenerate_fits.json').read_text())['events'],[])
                for suffix in ('cells.csv','summary.csv','diag_cells.csv'):
                    pd.testing.assert_frame_equal(pd.read_csv(next(a.glob('*_'+suffix))),pd.read_csv(next(b.glob('*_'+suffix))),check_exact=True)
                for suffix in (('perq.pkl','perq_ext.pkl') if script!='r15_test.py' else ('perq.pkl',)):
                    with next(a.glob('*_'+suffix)).open('rb') as f:left=pickle.load(f)
                    with next(b.glob('*_'+suffix)).open('rb') as f:right=pickle.load(f)
                    deep_equal(self,left,right)
                self.assertEqual(json.loads(next(a.glob('*_selection.json')).read_text()),json.loads(next(b.glob('*_selection.json')).read_text()))
                left=json.loads(next(a.glob('*_run_record.json')).read_text());right=json.loads(next(b.glob('*_run_record.json')).read_text())
                for record in (left,right):
                    for key in ('wall_seconds','finished','times'):record.pop(key,None)
                self.assertEqual(left,right)

    def test_dense_identity_fixture_fails_direct_and_runs_guarded(self):
        root=self.root;source=root/'feats/dev.npz'
        with np.load(source) as z:arrays={k:z[k] for k in z.files}
        for key in arrays:
            if key not in ('clean/text_features','clean/txt_pids','clean/query_raw_pids'):arrays[key]=np.repeat(arrays[key],6,axis=0)
        np.savez(root/'feats/dense.npz',**arrays)
        args=list(map(str,self.common));args[args.index('dev')]='dense';args[args.index('dev')]='dense'
        direct=root/'dense_direct';direct.mkdir()
        command=[sys.executable,'-B',str(T/'r17_const.py'),'--out',str(direct),'--methods',*self.methods,*args]
        result=subprocess.run(command,cwd=CODE,env=ENV,capture_output=True,text=True,encoding='utf-8')
        (root/'dense_direct.log').write_text(result.stdout+result.stderr,encoding='utf-8',newline='\n')
        self.assertNotEqual(result.returncode,0);self.assertIn('anyh = LogisticRegression',result.stderr)
        output=root/'dense_guarded';output.mkdir();command[command.index(str(direct))]=str(output)
        wrapped=[sys.executable,'-B',str(RUNNER),'--expect_sha',sha(T/'r17_const.py'),'--log',str(output/'degenerate_fits.json'),'--',*command[2:]]
        self.execute('dense_guarded',wrapped)
        events=json.loads((output/'degenerate_fits.json').read_text())['events']
        self.assertTrue(events);self.assertIn('anyh',{e['head'] for e in events});self.assertLessEqual({e['head'] for e in events},set(GUARDED_HEADS))

if __name__=='__main__':unittest.main(verbosity=2)
