"""Synthetic execution-layer checks: layered caches, resources, digest and strict IO."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import contextlib,io,json,re,shutil,sys,tarfile,tempfile,unittest
import numpy as np
CODE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CODE))
from tools.p4.r18 import workflow as wf,driver_support as support

class ExecutionTests(unittest.TestCase):
    def test_pack12_requires_every_key_and_preserves_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);arrays={k:np.arange(6,dtype=np.float32).reshape(2,3) for k in wf.KEYS12}
            missing=dict(arrays);missing.pop('ir_missing/f_ir');np.savez(p/'missing.npz',**missing)
            with self.assertRaisesRegex(ValueError,'ir_missing/f_ir'):wf.pack12(p/'missing.npz',p/'failed.npz')
            self.assertFalse((p/'failed.npz').exists());np.savez(p/'all.npz',**arrays);wf.pack12(p/'all.npz',p/'out.npz')
            with np.load(p/'out.npz') as z:
                self.assertEqual(set(z.files),set(wf.KEYS12))
                for k in arrays:np.testing.assert_array_equal(z[k],arrays[k],strict=True)

    def test_cpu_and_memory_budget_respect_cgroup_and_affinity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def put(name,value):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value,encoding='utf-8',newline='\n')
            put('proc/meminfo',f'MemAvailable: {64*2**20} kB\n');put('sys/fs/cgroup/memory.max',str(32*2**30));put('sys/fs/cgroup/memory.current',str(8*2**30));put('sys/fs/cgroup/cpu.max','400000 100000')
            with patch.object(support.os,'sched_getaffinity',return_value=set(range(16)),create=True):v=support.resource_summary(root)
            self.assertEqual((v['available_memory_GiB'],v['nproc'],v['PAR']),(24,4,2))
            put('sys/fs/cgroup/memory.max','max');put('sys/fs/cgroup/cpu.max','1250000 100000')
            with patch.object(support.os,'sched_getaffinity',return_value=set(range(32)),create=True):v=support.resource_summary(root)
            self.assertEqual((v['nproc'],v['PAR']),(13,7))
            put('proc/meminfo',f'MemAvailable: {512*2**20} kB\n');put('sys/fs/cgroup/cpu.max','max 100000')
            with patch.object(support.os,'sched_getaffinity',return_value=set(range(64)),create=True):self.assertEqual(support.resource_summary(root)['PAR'],9)
            with patch.object(support.os,'sched_getaffinity',return_value={0},create=True):self.assertEqual(support.resource_summary(root)['PAR'],1)

    def test_compact_digest_excludes_large_payloads_and_counts_itself(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/'out';checks=out/'checks';checks.mkdir(parents=True)
            def write(name,value):
                p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value),encoding='utf-8',newline='\n');return p
            protected=write('checks/protected.json',{'PROTECTED_FILE_'+str(i):'a'*64 for i in range(2000)})
            write('checks/smoke.json',{'status':'smoke_completed','steps':[{'finite':True,'L_total':i,'seconds':.1,'peak_memory_bytes':10} for i in range(30)]})
            write('checks/archives_test.json',{'s101':{'problems':[],'rows':['ARCHIVE_BIG_PAYLOAD']*500}})
            write('checks/parallel.json',{'nproc':16,'PAR':7,'available_memory_GiB':64})
            write('verdict_full.json',{'sentinel':'DO_NOT_INCLUDE_VERDICT_JSON'})
            (out/'verdict_full.txt').write_text('JUDGE_TEXT_BODY\n',encoding='utf-8');(out/'report_const_full.txt').write_text('R17_TEXT_BODY\n',encoding='utf-8');(out/'report_xcam.txt').write_text('XCAM_TEXT_BODY\n',encoding='utf-8')
            (out/'R18J.txt').write_text('DO_NOT_INCLUDE_FULL_R18J\n',encoding='utf-8');(out/'R18J_core.txt').write_text('R18J {"core":true}\n',encoding='utf-8')
            (out/'STATUS_L1').write_text('COMPLETE\n',encoding='utf-8');(out/'STATUS_L2').write_text('STOPPED: deliberate fixture\n',encoding='utf-8')
            write('test/s101_f7001/degenerate_fits.json',{'events':[{'head':'anyh'}]})
            with contextlib.redirect_stdout(io.StringIO()):support.finish(out,root,'L2','STOPPED: deliberate fixture')
            p=out/'DIGEST18_L2.txt';text=p.read_text(encoding='utf-8')
            for banned in ('PROTECTED_FILE_','DO_NOT_INCLUDE_VERDICT_JSON','DO_NOT_INCLUDE_FULL_R18J','ARCHIVE_BIG_PAYLOAD'):self.assertNotIn(banned,text)
            for required in ('files=2000',wf.sha256(protected),'JUDGE_TEXT_BODY','R17_TEXT_BODY','XCAM_TEXT_BODY','R18J {"core":true}','events=1 heads={"anyh": 1}','deliberate fixture'):self.assertIn(required,text)
            lines,size=map(int,re.search(r'DIGEST_STATS lines=(\d+) bytes=(\d+)',text).groups())
            self.assertEqual(lines,text.count('\n'));self.assertEqual(size,len(p.read_bytes()));self.assertLess(size,20000)
            with tarfile.open(root/'p4_r18_L2_results.tar.gz') as tf:self.assertTrue(any(n.endswith('verdict_full.json') for n in tf.getnames()))

    def test_report_changes_reuse_both_layers_but_scoring_changes_do_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);code=root/'code';code.mkdir();paths=set(wf.signature_paths('l1')+wf.signature_paths('eval'))
            paths.update(CODE/p for p in wf.REPORT_FILES if (CODE/p).is_file())
            for p in paths:
                q=code/p.relative_to(CODE);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
            data=root/'input.txt';data.write_text('input',encoding='utf-8')
            def cli(kind):
                out=root/(kind+'.txt')
                return SimpleNamespace(manifest=root/(kind+'.json'),done=root/(kind+'.done'),inputs=[str(data)],outputs=[str(out)],command=[sys.executable,'-B','-c','from pathlib import Path;import sys;Path(sys.argv[1]).write_text("output")',str(out)],signature_kind=kind,rerun=False,verify_only=False,mkdir=False)
            a,b=cli('l1'),cli('eval')
            with patch.object(wf,'CODE',code),contextlib.redirect_stdout(io.StringIO()):
                wf.stage(a);wf.stage(b)
                manifest=root/'L1_manifest.json';wf.write_json(manifest,{'kind':'l1','code':wf.code_signature('l1'),'files':wf.files_of([data])})
                report=code/'tools/p4/r18/r18_exact.py';report.write_text(report.read_text(encoding='utf-8')+'\n# report change\n',encoding='utf-8',newline='\n')
                with patch.object(wf.subprocess,'run',side_effect=AssertionError('Completed work was rerun')):
                    wf.stage(a);wf.stage(b)
                    with patch.object(sys,'argv',['workflow.py','manifest','--out',str(manifest),'--verify']):wf.main()
                common=code/'tools/p4/r18/common.py';old=common.read_bytes();common.write_bytes(old+b'\n# changed\n')
                with self.assertRaisesRegex(ValueError,'differs'):wf.stage(a)
                common.write_bytes(old)
                eval_file=code/'tools/p4/r18/xcam_core.py';eval_file.write_bytes(eval_file.read_bytes()+b'\n# scoring change\n')
                with self.assertRaisesRegex(ValueError,'differs'):wf.stage(b)
                with patch.object(wf.subprocess,'run',side_effect=AssertionError('L1 was rerun')):wf.stage(a)
                a.command.append('changed command')
                with self.assertRaisesRegex(ValueError,'differs'):wf.stage(a)

    def test_driver_shutdown_guard_cadence_and_smoke_environment(self):
        source=(CODE/'tools/p4/r18/p4_r18.sh').read_text(encoding='utf-8')
        self.assertIn('sync; /usr/bin/shutdown;',source);self.assertNotIn('/usr/sbin/shutdown',source)
        self.assertIn('[ ! -e /usr/bin/shutdown ]',source);self.assertIn('sleep 180',source);self.assertIn('p4_r18_no_shutdown',source)
        unit=source.split('unit() {',1)[1].split('\n}\nfind_run',1)[0]
        self.assertNotRegex(unit,r'\bguard\b');self.assertIn('--signature_kind "$kind"',unit)
        self.assertIn('sha256sum -c --quiet',source)
        self.assertIn('env -u OPENBLAS_NUM_THREADS -u OMP_NUM_THREADS -u MKL_NUM_THREADS TRAIN_SEED=101',source)
        self.assertIn('上次运行可能被中断',source)

if __name__=='__main__':unittest.main(verbosity=2)
