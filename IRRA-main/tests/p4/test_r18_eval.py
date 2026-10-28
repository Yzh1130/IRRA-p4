"""Synthetic-only R18 protocol, camera restriction, copied statistics and subset checks."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import hashlib
import importlib.util
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
CODE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CODE));sys.path.insert(0,str(Path(__file__).parent))
from tools.p4.r18.xcam_core import restricted_weights
from tools.p4.r18.r18_subset import subset_archive
import test_r17 as fixtures
T=CODE/'tools/p4/transfer'

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def softmax(z,mask):
    z=np.where(mask,z,-np.inf);mx=z.max(1,keepdims=True);mx=np.where(np.isfinite(mx),mx,0);e=np.exp(z-mx)*mask
    return e/np.maximum(e.sum(1,keepdims=True),1e-300)

def deep_equal(case,a,b):
    if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b,strict=True)
    elif isinstance(a,dict):
        case.assertEqual(set(a),set(b))
        for k in a:deep_equal(case,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        case.assertEqual(len(a),len(b))
        for x,y in zip(a,b):deep_equal(case,x,y)
    else:case.assertEqual(a,b)

class CameraTests(unittest.TestCase):
    def fixture(self):
        return {'J':np.array([0,1]),'Fi':np.array([2,3]),'CF':np.array([[.8,.5],[.7,.3]]),'X':np.array([[.8,.5,.3,.2,.1,.6,0,0,.5],[.7,.3,.4,.4,.1,.3,0,0,.5]])}

    def test_same_camera_sources_have_zero_weights(self):
        g=self.fixture();before=g['CF'].copy()
        w,available=restricted_weights(g,np.array(['cam1']*4),20,lambda x:np.tile([.2,.8],(len(x),1)),lambda x:x,np.array([0.,.4]),softmax)
        self.assertFalse(available.any());self.assertTrue(np.array_equal(w,np.zeros((2,2))));np.testing.assert_array_equal(g['CF'],before)

    def test_all_foreign_camera_preserves_original_weights_bitwise(self):
        g=self.fixture();original=np.array([[.123,.234],[.345,.456]])
        w,available=restricted_weights(g,np.array(['cam1','cam1','cam2','cam2']),20,None,None,None,softmax,original)
        self.assertEqual(w.tobytes(),original.tobytes());self.assertTrue(available.all())

    def test_partial_camera_mask_and_only_two_gate_features_recomputed(self):
        g=self.fixture();before={k:v.copy() for k,v in g.items()};seen=[]
        def predict(x):seen.append(x.copy());return np.tile([.2,.8],(len(x),1))
        w,_=restricted_weights(g,np.array(['cam1','cam2','cam1','cam2']),20,predict,lambda x:x,np.array([.7,.4]),softmax)
        self.assertEqual(w[0,0],0);self.assertEqual(w[1,1],0)
        np.testing.assert_array_equal(seen[0][:,0],[.5,.7]);np.testing.assert_allclose(seen[0][:,1],[.3,.3],rtol=0,atol=1e-15)
        for k in g:np.testing.assert_array_equal(g[k],before[k])

class SubsetTests(unittest.TestCase):
    def test_unequal_query_gallery_counts_and_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);q=np.array([0,0,1,1,2]);g=np.array([0,0,0,1,2,2,2]);raw=np.array([10,10,20,20,30])
            arrays={'clean/text_features':np.arange(15,dtype=np.float32).reshape(5,3),'clean/txt_pids':q,'clean/img_pids':g,'clean/query_raw_pids':raw}
            for key in ['clean/f_rgb','clean/f_ir','rgb_missing/f_rgb','ir_missing/f_ir','rgb_noise_010/f_rgb','ir_noise_010/f_ir','rgb_blur_k7/f_rgb','ir_blur_k7/f_ir']:arrays[key]=np.arange(21,dtype=np.float32).reshape(7,3)
            source=root/'in.npz';np.savez(source,**arrays);overlap=root/'overlap.json'
            overlap.write_text(json.dumps({'overlap_test_ids':[20]}),encoding='utf-8');result=subset_archive(source,overlap,root/'out.npz')
            self.assertEqual(result['after'],{'identities':2,'queries':3,'items':6});self.assertIn('少于 10',result['status'])
            with np.load(root/'out.npz') as z:
                self.assertEqual(set(z.files),set(arrays))
                np.testing.assert_array_equal(z['clean/text_features'],arrays['clean/text_features'][[0,1,4]])
                np.testing.assert_array_equal(z['clean/f_ir'],arrays['clean/f_ir'][[0,1,2,4,5,6]])
            overlap.write_text(json.dumps({'overlap_test_ids':[]}),encoding='utf-8');self.assertEqual(subset_archive(source,overlap,root/'empty.npz')['exit_code'],0);self.assertFalse((root/'empty.npz').exists())
            overlap.write_text(json.dumps({'overlap_test_ids':[10,20,30]}),encoding='utf-8');self.assertEqual(subset_archive(source,overlap,root/'none.npz')['exit_code'],2)
            overlap.write_text(json.dumps({'overlap_test_ids':[99]}),encoding='utf-8')
            with self.assertRaises(ValueError):subset_archive(source,overlap,root/'bad.npz')

class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work=Path(os.environ.get('R18_WORK',r'E:\论文4\2过程文档\p4_rgbnt201\r18'));cls.root=Path(tempfile.mkdtemp(prefix='eval_',dir=work));root=cls.root
        rng=np.random.default_rng(17);dim=32;blank=(rng.standard_normal(dim),rng.standard_normal(dim));(root/'feats').mkdir()
        dev=fixtures.make_archive(rng,40,100000,dim,blank);test=fixtures.make_archive(rng,24,800000,dim,blank);train=fixtures.make_archive(rng,70,900000,dim,blank)
        np.savez(root/'feats/dev.npz',**dev);np.savez(root/'test.npz',**test)
        np.savez(root/'train.npz',f_ir=train['clean/f_ir'],f_rgb=train['clean/f_rgb'],raw_pids=train['clean/query_raw_pids'][::2][train['clean/img_pids']])
        cams=root/'cams.npz';np.savez(cams,dev_cams=np.array(['cam1','cam2'])[np.arange(len(dev['clean/img_pids']))%2],test_cams=np.array(['cam1','cam2'])[np.arange(len(test['clean/img_pids']))%2])
        common=['--b12','.','--feats',root/'feats','--base','dev','--adapter','dev','--assign','none','--assign_seeds','7031','7032','--mixes','M0','M4','M5f','--skip_joint','--grid_preset','dev','--select_rule','constrained','--r_max','0.999999','--tie_break','bstar','--test_npz',root/'test.npz','--link_map','train_lda','--train_ir_npz',root/'train.npz','--test_seed','7001','--tag','t_s101_f7001']
        cls.methods=['B*','QE-DBA','QE+M','QE+CAET','QE+SRC2f','QE+EM']
        cls.reference=root/'reference/s101_f7001';cls.reference.mkdir(parents=True)
        cls.run_cli('reference','r15_test.py',['--out',cls.reference,'--methods',*cls.methods,*common])
        selection=next(cls.reference.glob('*_selection.json'));options=['--perq_ext','--fixed_selection',selection]
        cls.saved={}
        for name,script,extra in [('r17','r17_const.py',[]),('default','r18_xcam.py',[]),('cams_without_method','r18_xcam.py',['--item_cams',cams])]:
            directory=root/'same_output';directory.mkdir()
            cls.run_cli(name,script,['--out',directory,'--methods',*cls.methods,*common,*options,*extra])
            dest=root/name;shutil.copytree(directory,dest);cls.saved[name]=dest
            assert directory.resolve().is_relative_to(root.resolve());shutil.rmtree(directory)
        cls.active=root/'xcam/s101_f7001';cls.active.mkdir(parents=True)
        cls.run_cli('active','r18_xcam.py',['--out',cls.active,'--methods','B*','QE-DBA','QE+M','QE+M-xcam',*common,*options,'--item_cams',cams])

    @classmethod
    def run_cli(cls,name,script,args):
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
        p=subprocess.run([sys.executable,'-B',str(T/script),*map(str,args)],cwd=CODE,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (cls.root/(name+'.log')).write_text(p.stdout+p.stderr,encoding='utf-8',newline='\n')
        if p.returncode:raise RuntimeError(f'{name} failed {p.returncode}: '+p.stdout[-3000:]+p.stderr[-4000:])
        return p

    def test_default_and_unused_camera_argument_equal_frozen_protocol(self):
        for candidate in ('default','cams_without_method'):
            a,b=self.saved['r17'],self.saved[candidate]
            for suffix in ('cells.csv','summary.csv','diag_cells.csv'):
                pd.testing.assert_frame_equal(pd.read_csv(next(a.glob('*_'+suffix))),pd.read_csv(next(b.glob('*_'+suffix))),check_exact=True)
            for suffix in ('perq.pkl','perq_ext.pkl'):
                with next(a.glob('*_'+suffix)).open('rb') as f:left=pickle.load(f)
                with next(b.glob('*_'+suffix)).open('rb') as f:right=pickle.load(f)
                deep_equal(self,left,right)
            self.assertEqual(json.loads(next(a.glob('*_selection.json')).read_text()),json.loads(next(b.glob('*_selection.json')).read_text()))
            left=json.loads(next(a.glob('*_run_record.json')).read_text());right=json.loads(next(b.glob('*_run_record.json')).read_text())
            for record in (left,right):
                for key in ('script','script_sha256','wall_seconds','finished','times'):record.pop(key,None)
                record['args'].pop('item_cams',None)
            self.assertEqual(left,right)

    def test_judge_reproduction_and_expected_script_hash(self):
        sha=hashlib.sha256((T/'r18_xcam.py').read_bytes()).hexdigest()
        p=self.run_cli('judge','r18_xcam_judge.py',['--xcam_root',self.root/'xcam','--r15_root',self.root/'reference','--seeds','101','--fits','7001','--expect_script_sha',sha,'--boot','200','--out',self.root/'judge.json'])
        self.assertIn('[status] VALID',p.stdout)
        result=json.loads((self.root/'judge.json').read_text());self.assertEqual(result['status'],'VALID');self.assertTrue(result['reproduction']['s101_f7001']['ok'])
        p=subprocess.run([sys.executable,'-B',str(T/'r18_xcam_judge.py'),'--xcam_root',str(self.root/'xcam'),'--r15_root',str(self.root/'reference'),'--seeds','101','--fits','7001','--expect_script_sha','0'*64,'--out',str(self.root/'invalid.json')],capture_output=True)
        self.assertEqual(p.returncode,3)

    def test_copied_statistics_functions_are_verbatim_and_identical(self):
        judge=module(T/'r18_xcam_judge.py','r18_test_judge');scope={'np':np,'a':SimpleNamespace(evalset='test',boot=4000)}
        for source,functions in [('judge_test.py',['group_pair']),('r17_judge.py',['boot_ci','per_query'])]:
            original=(T/source).read_text(encoding='utf-8');copy=(T/'r18_xcam_judge.py').read_text(encoding='utf-8')
            for name in functions:
                node=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name==name);text=ast.get_source_segment(original,node)
                copied=next(n for n in ast.parse(copy).body if isinstance(n,ast.FunctionDef) and n.name==name)
                self.assertEqual(text,ast.get_source_segment(copy,copied));exec(text,scope)
        pq={('test','M4',m):{0:[(v,100.)],1:[(v+1,0.)]} for m,v in [('x',3.),('y',1.)]};ids=np.array([8,9])
        judge.a=scope['a'];self.assertEqual(judge.group_pair(pq,'x','y',['M4'],ids),scope['group_pair'](pq,'x','y',['M4'],ids))
        vector=np.array([1.,2.,4.]);identity=np.array([1,1,2]);self.assertEqual(judge.boot_ci(vector,identity),scope['boot_ci'](vector,identity))
        rec=json.loads(next(self.active.glob('*_run_record.json')).read_text());ext=pickle.load(next(self.active.glob('*_perq_ext.pkl')).open('rb'))
        data={101:{7001:{'rec':rec,'pq':ext}}};judge.DATA=data;scope['DATA']=data
        x=judge.per_query(101,'QE+M',['M4'],'R5');y=scope['per_query'](101,'QE+M',['M4'],'R5');deep_equal(self,x,y)

if __name__=='__main__':unittest.main(verbosity=2)
