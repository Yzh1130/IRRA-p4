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


# Scheduling-only acceptance additions. All workers, locks and Git fixtures are
# private to TemporaryDirectory; no real feature, weight or protected tree is used.
import concurrent.futures
import hashlib
import socket
import time


def _protected(root):
    for name in ('p4_r15_out', 'p4_r17_out', 'p4_r18_out'):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'unchanged.txt').write_text(name, encoding='utf-8')


def _parallel_runner(root, par=3, stage='S1', min_free_gb=0):
    _protected(root)
    return driver.Runner(root, stage=stage, par=par, min_free_gb=min_free_gb)


def _worker(directory, delay=0.0, code=0, required=None, protected=None):
    program = (
        'import json,os,sys,time;from pathlib import Path;'
        'd=Path(sys.argv[1]);req=sys.argv[4];'
        'assert not req or Path(req).is_file(), "dependency not ready";'
        '(d/"started").write_text(str(time.monotonic()));'
        '(d/"temporary.json").write_text(json.dumps({k:os.environ[k] for k in ("TMP","TEMP","TMPDIR")}));'
        'time.sleep(float(sys.argv[2]));'
        'bad=sys.argv[5];'
        'Path(bad).write_text("changed") if bad else None;'
        '(d/"finished").write_text(str(time.monotonic()));'
        'sys.exit(int(sys.argv[3]))'
    )
    return [sys.executable, '-B', '-c', program, str(directory), str(delay), str(code),
            str(required or ''), str(protected or '')]


def _unit(runner, label, delay=0.0, code=0, dependencies=(), required=None, protected=None):
    name = runner.stage + '/' + label
    directory = runner.out / runner.stage.lower() / label
    return driver.Unit(name, directory, [], [_worker(directory, delay, code, required, protected)],
                       {'synthetic_worker': label}, tuple(dependencies))


class ParallelSchedulingTests(unittest.TestCase):
    def test_dependencies_are_explicit_and_temporary_directories_are_isolated(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = _parallel_runner(Path(temp))
            camera = _unit(runner, 'cameras', .08)
            subset = _unit(runner, 'subset', dependencies=(camera.name,), required=camera.directory/'finished')
            independent = _unit(runner, 'independent', .05)
            driver.run_units(runner, [camera, subset, independent])
            self.assertLessEqual(float((camera.directory/'finished').read_text()),
                                 float((subset.directory/'started').read_text()))
            roots = []
            for unit in (camera, subset, independent):
                values = json.loads((unit.directory/'temporary.json').read_text())
                self.assertEqual(values['TMP'], values['TEMP'])
                self.assertEqual(values['TMP'], values['TMPDIR'])
                self.assertTrue(Path(values['TMP']).is_relative_to(runner.out/'_tmp'/'S1'))
                roots.append(values['TMP'])
                record = json.loads((unit.directory/'driver_run_record.json').read_text())
                self.assertEqual(record['par'], 3)
                self.assertIn('started_at', record)
                self.assertIn('finished_at', record)
                self.assertIn('peak_rss_bytes', record)
                if not hasattr(os, 'wait4'):
                    self.assertIsNone(record['peak_rss_bytes'])
                else:
                    self.assertGreater(record['peak_rss_bytes'], 0)
            self.assertEqual(len(roots), len(set(roots)))

    def test_first_failure_stops_dispatch_and_running_work_finishes(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = _parallel_runner(Path(temp), par=2)
            bad = _unit(runner, 'bad', .10, 4)
            running = _unit(runner, 'running', .35)
            pending = _unit(runner, 'must_not_start')
            with self.assertRaises(Exception):
                driver.run_units(runner, [bad, running, pending])
            self.assertTrue((running.directory/'finished').is_file())
            self.assertFalse(pending.directory.exists())
            self.assertFalse((bad.directory/'.done').exists())
            runner.finish('S1', 'STOPPED', RuntimeError('synthetic failure'))
            self.assertEqual((runner.out/'STATUS_S1').read_text(), 'STOPPED\n')
            # A restarted stage has a new Runner; completed work is reused.
            runner = _parallel_runner(Path(temp), par=2)
            resumed = _unit(runner, 'bad', code=0)
            driver.run_units(runner, [resumed, running, pending])
            self.assertTrue((pending.directory/'.done').is_file())
            self.assertTrue(any((runner.out/'_stale').iterdir()))

    def test_guard_failure_stops_pending_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner = _parallel_runner(root, par=2)
            modifier = _unit(runner, 'modifier', .05, protected=runner.protected[0]/'unchanged.txt')
            running = _unit(runner, 'running', .25)
            pending = _unit(runner, 'must_not_start')
            with self.assertRaisesRegex(Exception, 'Protected'):
                driver.run_units(runner, [modifier, running, pending])
            self.assertTrue((running.directory/'finished').is_file())
            self.assertFalse(pending.directory.exists())
            with self.assertRaisesRegex(Exception, 'Protected'):
                runner.finish('S1', 'STOPPED')
            self.assertEqual((runner.out/'STATUS_S1').read_text(), 'STOPPED\n')

    def test_wait4_branch_uses_exact_child_pid_and_rss_kib_conversion(self):
        from types import SimpleNamespace
        calls=[]
        class FakePopen:
            def __init__(self,*args,**kwargs):
                self.pid=31337
                self.returncode=None
                calls.append((args,kwargs,self))
            def wait(self):
                raise AssertionError('wait4 branch must not call cumulative/wait fallback')
        with tempfile.TemporaryDirectory() as temp:
            runner=_parallel_runner(Path(temp),par=3)
            unit=_unit(runner,'mock_posix')
            with patch.object(driver.subprocess,'Popen',FakePopen), \
                 patch.object(driver.os,'wait4',create=True,side_effect=[InterruptedError(),(31337,0,SimpleNamespace(ru_maxrss=123))]) as wait4, \
                 patch.object(driver.os,'waitstatus_to_exitcode',create=True,return_value=0) as decode:
                runner.unit(unit.name,unit.directory,unit.inputs,unit.commands,unit.parameters)
            self.assertEqual(wait4.call_count,2)
            self.assertTrue(all(c.args==(31337,0) for c in wait4.call_args_list))
            decode.assert_called_once_with(0)
            self.assertEqual(calls[0][2].returncode,0)
            record=json.loads((unit.directory/'driver_run_record.json').read_text())
            self.assertEqual(record['peak_rss_bytes'],123*1024)
            self.assertEqual(record['exit_codes'],[0])
            self.assertEqual(record['par'],3)
            self.assertIn('started_at',record);self.assertIn('finished_at',record)

    def test_s0_s3_ignore_positive_parallel_setting_and_digest_is_sorted(self):
        for stage in ('S0','S3'):
            with tempfile.TemporaryDirectory() as temp:
                runner=_parallel_runner(Path(temp),par=3,stage=stage)
                self.assertEqual(runner.par,1)
        with tempfile.TemporaryDirectory() as temp:
            runner=_parallel_runner(Path(temp),par=3)
            tasks=[_unit(runner,'z',.03),_unit(runner,'a',.01)]
            driver.run_units(runner,tasks)
            runner.finish('S1','COMPLETE')
            lines=(runner.out/'DIGEST19_S1.txt').read_text(encoding='utf-8').splitlines()
            events=json.loads(next(x[7:] for x in lines if x.startswith('events ')))
            self.assertEqual([e['unit'] for e in events],sorted(e['unit'] for e in events))
            summary=json.loads(next(x[8:] for x in lines if x.startswith('SUMMARY ')))
            self.assertEqual(summary['unit_count'],2)
            self.assertGreaterEqual(summary['wall_seconds'],0)
            self.assertIn('free_gb_start',summary);self.assertIn('free_gb_end',summary)
            self.assertIn('peak_rss_bytes',summary);self.assertIn('peak_rss_unit',summary)

    def test_disk_failure_before_any_worker_and_invalid_parallelism(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _protected(root)
            args = ['--stage','S0','--root',str(root),'--repo',str(CODE.parent),
                    '--min_free_gb','1e100']
            with self.assertRaises(Exception):
                driver.main(args)
            self.assertEqual((root/'p4_r19_out/STATUS_S0').read_text(), 'STOPPED\n')
            digest = (root/'p4_r19_out/DIGEST19_S0.txt').read_text(encoding='utf-8')
            self.assertIn('磁盘不足', digest)
            self.assertIn('GB', digest)
            self.assertFalse((root/'p4_r19_out/s0').exists())
        for value in ('0','-1','bad'):
            with self.subTest(par=value), self.assertRaises(SystemExit):
                driver.main(['--stage','S0','--par',value])

    def test_guard_calls_are_serialized_inside_one_process(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = _parallel_runner(Path(temp))
            original = driver.files_of
            active = peak = 0
            lock = __import__('threading').Lock()
            def observed(paths):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    time.sleep(.03)
                    return original(paths)
                finally:
                    with lock: active -= 1
            with patch.object(driver, 'files_of', observed):
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(lambda _: runner.guard(), range(4)))
            self.assertEqual(peak, 1)

    def test_stage_input_collection_never_hashes_other_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            for stage, relative in [('S1','s1'),('S2','s2'),('S4','mlm1/eval')]:
                directory = out/relative/'unit'
                directory.mkdir(parents=True)
                for suffix in ('cells.csv','diag_cells.csv','selection.json','perq_ext.pkl','run_record.json'):
                    (directory/('r12_fixture_'+suffix)).write_text(stage)
            for stage,relative in [('S1','s1'),('S2','s2'),('S4','mlm1/eval')]:
                paths = driver.stage_inputs(out,stage)
                self.assertEqual(len(paths), 5)
                self.assertTrue(all(Path(p).is_relative_to(out/relative) for p in paths))


class StageLockAndGateTests(unittest.TestCase):
    def test_s0_gate_uses_only_the_explicit_evaluation_closure(self):
        paths=set(driver.calculation_paths(CODE))
        self.assertIn('IRRA-main/tools/p4/transfer/r19_variants.py',paths)
        self.assertIn('IRRA-main/tools/p4/r19/core.py',paths)
        self.assertIn('IRRA-main/tools/p4/r19/r19_judge.py',paths)
        self.assertTrue({'IRRA-main/'+p for p in driver.FROZEN} <= paths)
        for path in ('tools/p4/r19/training.py','configs/p4/r19_mlm1_override.yaml',
                     'tools/p4/r19/driver.py','tools/p4/r19/p4_r19.sh','tools/p4/r19/README_r19.md'):
            self.assertNotIn('IRRA-main/'+path,paths)

    def test_same_stage_rejected_with_owner_metadata_other_stage_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            commit = 'a'*40
            with driver.stage_lock(out,'S1',commit):
                contents = json.loads((out/'.driver_S1.lock').read_text(encoding='utf-8'))
                self.assertEqual(contents['pid'],os.getpid())
                self.assertEqual(contents['hostname'],socket.gethostname())
                self.assertEqual(contents['git_commit'],commit)
                self.assertIn('started_at',contents)
                with self.assertRaisesRegex(Exception, str(os.getpid())):
                    with driver.stage_lock(out,'S1',commit):
                        self.fail('duplicate owner lock acquired')
                with driver.stage_lock(out,'S2',commit):
                    self.assertTrue((out/'.driver_S1.lock').exists())
                    self.assertTrue((out/'.driver_S2.lock').exists())
            self.assertFalse((out/'.driver_S1.lock').exists())

    def test_s0_gate_and_calculation_change_but_scheduler_only_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root/'repo'; code = repo/'IRRA-main'; out = root/'p4_r19_out'
            code.mkdir(parents=True); out.mkdir()
            calc = code/'calculation.py'; calc.write_text('value=1\n')
            scheduler = code/'tools/p4/r19/driver.py'; scheduler.parent.mkdir(parents=True)
            scheduler.write_text('# scheduling v1\n')
            def git(*args):
                return subprocess.check_output(['git',*args],cwd=repo,stderr=subprocess.DEVNULL).decode().strip()
            git('init');git('add','.')
            git('-c','user.name=R19 synthetic','-c','user.email=r19@example.invalid','commit','-m','fixture')
            commit=git('rev-parse','HEAD')
            with patch.object(driver,'calculation_paths',return_value=['IRRA-main/calculation.py']):
                with self.assertRaises(Exception):driver.require_prerequisites(out,'S1',code)
                (out/'STATUS_S0').write_text('STOPPED\n')
                with self.assertRaises(Exception):driver.require_prerequisites(out,'S1',code)
                (out/'STATUS_S0').write_text('COMPLETE\n')
                (out/'DIGEST19_S0.txt').write_text('git_commit '+commit+'\n')
                driver.require_prerequisites(out,'S1',code)
                calc.write_text('value=2\n')
                with self.assertRaises(Exception):driver.require_prerequisites(out,'S1',code)
                calc.write_text('value=1\n');scheduler.write_text('# scheduling v2\n')
                training=code/'tools/p4/r19/training.py';training.write_text('# training-only change\n')
                git('add','.');git('-c','user.name=R19 synthetic','-c','user.email=r19@example.invalid','commit','-m','scheduler only')
                driver.require_prerequisites(out,'S1',code)
                with self.assertRaises(Exception):driver.require_prerequisites(out,'S4',code)
                (out/'STATUS_S3').write_text('STOPPED\n')
                with self.assertRaises(Exception):driver.require_prerequisites(out,'S4',code)
                (out/'STATUS_S3').write_text('COMPLETE\n')
                (out/'mlm1').mkdir();(out/'mlm1/evaluation_inputs.json').write_text('{}')
                driver.require_prerequisites(out,'S4',code)
                (out/'.driver_S3.lock').write_text('{}')
                with self.assertRaises(Exception):driver.require_prerequisites(out,'S4',code)

    def test_cross_process_stages_share_one_atomic_baseline_and_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);_protected(root)
            (root/'p4_r19_out').mkdir()
            original_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE).decode().strip()
            (root/'p4_r19_out/STATUS_S0').write_text('COMPLETE\n')
            (root/'p4_r19_out/DIGEST19_S0.txt').write_text('git_commit '+original_commit+'\n')
            shim=root/'scheduler_test_shim.py'
            shim.write_text(
                'import sys,time,json;from pathlib import Path\n'
                'sys.path.insert(0,sys.argv.pop(1))\n'
                'from tools.p4.r19 import driver\n'
                'def run(r,s):\n'
                ' d=r.out/r.stage.lower()/"one"\n'
                ' c=[sys.executable,"-B","-c","import time;time.sleep(.4)"]\n'
                ' driver.run_units(r,[driver.Unit(r.stage+"/one",d,[],[c],{},())])\n'
                'driver.s1=run;driver.s2=run\n'
                'raise SystemExit(driver.main(sys.argv[1:]))\n',encoding='utf-8')
            env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            processes=[]
            for stage in ('S1','S2'):
                args=[sys.executable,'-B',str(shim),str(CODE),'--stage',stage,'--root',str(root),
                      '--repo',str(CODE.parent),'--par','2','--min_free_gb','0']
                processes.append(subprocess.Popen(args,cwd=CODE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT))
            for p in processes:
                output,_=p.communicate(timeout=45)
                self.assertEqual(p.returncode,0,output.decode('utf-8',errors='replace'))
            out=root/'p4_r19_out'
            self.assertEqual((out/'STATUS_S1').read_text(),'COMPLETE\n')
            self.assertEqual((out/'STATUS_S2').read_text(),'COMPLETE\n')
            baseline=json.loads((out/'checks/protected.json').read_text())
            self.assertEqual(baseline,driver.files_of([root/n for n in ('p4_r15_out','p4_r17_out','p4_r18_out')]))
            self.assertEqual(len(list((out/'checks').glob('protected.json'))),1)


# The production judge requires all 3 x 3 x 5 cells, so this *test-only* shim
# aggregates the reduced fixture's cells. It exercises scheduler inputs/order
# without changing the production judge, bootstrap, or any scoring function.
_PARALLEL_SHIM = r"""
import sys,os,json
from pathlib import Path
sys.path.insert(0,sys.argv.pop(1))
from tools.p4.r19 import driver

def tiny_judge(runner,stage,dependencies):
    directory=runner.out/'reports'/stage
    command=[sys.executable,'-B',__file__,str(driver.CODE),'aggregate',str(runner.out),stage,str(directory/'report.json')]
    return driver.Unit('judge_'+stage,directory,lambda:driver.stage_inputs(runner.out,stage),[command],
                       {'test_only_reduced_judge':True},tuple(dependencies),True)

if sys.argv[1:2]==['aggregate']:
    import csv,math
    out=Path(sys.argv[2]);stage=sys.argv[3];destination=Path(sys.argv[4])
    relative={'S1':'s1','S2':'s2'}[stage]
    rows=[]
    for path in sorted((out/relative).rglob('r12_*_cells.csv')):
        if path.name.endswith('_diag_cells.csv'):continue
        assert (path.parent/'.done').is_file(), 'judge ran before a unit completed'
        with path.open(encoding='utf-8',newline='') as stream:
            for row in csv.DictReader(stream):
                values={}
                for key,value in row.items():
                    try:
                        x=float(value);values[key]=x if math.isfinite(x) else None
                    except (ValueError,TypeError):values[key]=value
                rows.append({'unit':str(path.parent.relative_to(out/relative)),**values})
    rows.sort(key=lambda r:json.dumps(r,sort_keys=True))
    destination.write_text(json.dumps(rows,sort_keys=True,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    raise SystemExit(0)

driver.MIXES=['M0','M4']
driver.METHODS1=['B*','QE-DBA','QE+M']
driver.METHODS2=['B*','QE-DBA','QE+M','QE+M-xcam']
driver.METHODS_NESTED=['B*','QE-DBA','QE+M']
driver.judge_unit=tiny_judge
args=sys.argv[1:]
args+=['--par',os.environ.get('R19_PAR','1'),'--min_free_gb','0']
raise SystemExit(driver.main(args))
"""


def _make_parallel_features(root):
    import test_r17 as fixture
    root=Path(root);root.mkdir()
    rng=np.random.default_rng(17);dim=8
    blank=(rng.standard_normal(dim),rng.standard_normal(dim))
    dev=fixture.make_archive(rng,14,100000,dim,blank)
    test=fixture.make_archive(rng,10,800000,dim,blank)
    training=fixture.make_archive(rng,20,900000,dim,blank)
    np.savez(root/'dev.npz',**dev);np.savez(root/'test.npz',**test)
    qmask=test['clean/txt_pids']!=2;gmask=test['clean/img_pids']!=2
    qkeys={'clean/text_features','clean/txt_pids','clean/query_raw_pids'}
    subset={k:v[qmask if k in qkeys else gmask] for k,v in test.items()}
    np.savez(root/'subset.npz',**subset)
    np.savez(root/'train.npz',f_ir=training['clean/f_ir'],f_rgb=training['clean/f_rgb'],
             raw_pids=training['clean/query_raw_pids'][::2][training['clean/img_pids']])
    np.savez(root/'cams.npz',dev_cams=np.array(['cam1','cam2'])[np.arange(len(dev['clean/img_pids']))%2],
             test_cams=np.array(['cam1','cam2'])[np.arange(len(test['clean/img_pids']))%2])
    selections=root/'selections';selections.mkdir()
    for fit in (7001,7002):
        directory=selections/f's101_f{fit}';directory.mkdir()
        (directory/'r12_fixture_selection.json').write_text(json.dumps([
            {'split':fit,'fold':0,'method':'QE-DBA','chosen':'(1, 3, 0.6)','feasible':True}]))
    datasets={}
    for name in ('rgbt','rgbnt','rgbnt_subset'):
        datasets[name]={'101':dict(dev=str(root/'dev.npz'),
            test=str(root/('subset.npz' if name.endswith('subset') else 'test.npz')),
            train_ir=str(root/'train.npz'),selection_root=str(selections),
            guarded=name!='rgbt',cams=str(root/'cams.npz'))}
    spec={'seeds':[101],'fits':[7001,7002],'assign_seeds':[7031],'grid':'dev','datasets':datasets}
    path=root/'spec.json';path.write_text(json.dumps(spec),encoding='utf-8')
    return path


class ParallelFeatureEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='r19_parallel_features_')
        cls.root=Path(cls.tmp.name)
        cls.spec=_make_parallel_features(cls.root/'features')
        cls.shim=cls.root/'scheduling_only_shim.py';cls.shim.write_text(_PARALLEL_SHIM,encoding='utf-8')
        cls.commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE).decode().strip()
        cls.runs={}
        for stage in ('S1','S2'):
            for par in (1,3):
                root=cls.root/(stage+'_p'+str(par));root.mkdir();_protected(root)
                out=root/'p4_r19_out';out.mkdir()
                (out/'STATUS_S0').write_text('COMPLETE\n')
                (out/'DIGEST19_S0.txt').write_text('git_commit '+cls.commit+'\n')
                env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',
                    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',R19_PAR=str(par))
                command=[sys.executable,'-B',str(cls.shim),str(CODE),'--stage',stage,'--root',str(root),
                         '--repo',str(CODE.parent),'--spec',str(cls.spec)]
                with (root/'acceptance.log').open('wb') as log:
                    result=subprocess.run(command,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                if result.returncode:
                    raise AssertionError((root/'acceptance.log').read_text(encoding='utf-8',errors='replace')[-8000:])
                cls.runs[(stage,par)]=root

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_serial_parallel_actual_calculation_bytes_and_report_numbers(self):
        for stage in ('S1','S2'):
            left=self.runs[(stage,1)]/'p4_r19_out';right=self.runs[(stage,3)]/'p4_r19_out'
            relative='s1' if stage=='S1' else 's2'
            def hashes(root):
                return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (root/relative).rglob('*') if p.is_file() and
                        p.name.endswith(('_cells.csv','_diag_cells.csv','_selection.json','_perq_ext.pkl'))}
            self.assertEqual(hashes(left),hashes(right),stage)
            self.assertGreaterEqual(len([p for p in (left/relative).rglob('r12_*_cells.csv')
                                         if not p.name.endswith('_diag_cells.csv')]),4)
            a=(left/'reports'/stage/'report.json').read_bytes()
            b=(right/'reports'/stage/'report.json').read_bytes()
            self.assertEqual(hashlib.sha256(a).hexdigest(),hashlib.sha256(b).hexdigest())
            self.assertEqual((left/('STATUS_'+stage)).read_text(),'COMPLETE\n')
            self.assertEqual((right/('STATUS_'+stage)).read_text(),'COMPLETE\n')

    def test_camera_dependency_and_judge_last_from_actual_run_records(self):
        from datetime import datetime
        out=self.runs[('S2',3)]/'p4_r19_out'
        record_files=list(out.rglob('driver_run_record.json'))
        records={record['unit']:record for p in record_files
                 if (record:=json.loads(p.read_text(encoding='utf-8')))['unit'].startswith('S2/')}
        print('S2 actual run records:',[(p.relative_to(out).as_posix(),
              json.loads(p.read_text(encoding='utf-8'))['unit']) for p in record_files])
        self.assertIn('S2/cameras/s101',records)
        camera=records['S2/cameras/s101']
        for fit in (7001,7002):
            row=records[f'S2/rgbnt_subset/s101_f{fit}']
            self.assertGreaterEqual(datetime.fromisoformat(row['started_at']),datetime.fromisoformat(camera['finished_at']))
        judge=json.loads((out/'reports/S2/driver_run_record.json').read_text(encoding='utf-8'))
        self.assertGreaterEqual(datetime.fromisoformat(judge['started_at']),
                                max(datetime.fromisoformat(r['finished_at']) for r in records.values()))


class HistoricalRunSelectionTests(unittest.TestCase):
    @staticmethod
    def history(root, dataset="rgbt"):
        paths = {}
        for seed in driver.SEEDS:
            name = ("base_clean" if dataset == "rgbt" else "r201_base") + f"_s{seed}"
            prefix = "p4_preexp" if dataset == "rgbt" else "p4_r18_out"
            label = "RGBT-PEDES" if dataset == "rgbt" else "RGBNT201-PEDES"
            run = root / prefix / "runs" / name / label / ("20260101_000000_" + name)
            run.mkdir(parents=True)
            (run / "last.pth").write_bytes(b"synthetic checkpoint marker")
            (run / "configs.yaml").write_text("root_dir: synthetic_data\n", encoding="utf-8")
            paths[seed] = run
        return paths

    def test_three_completed_histories_exclude_aborted_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for dataset in ("rgbt", "rgbnt"):
                paths = self.history(root, dataset)
                for seed, run in paths.items():
                    aborted = run.with_name(run.name.replace("000000", "000001"))
                    aborted.mkdir()
                    (aborted / "configs.yaml").write_text("root_dir: placeholder\n")
                self.assertEqual(driver.historical_run_configs(root, dataset),
                                 {seed: run / "configs.yaml" for seed, run in paths.items()})

    def test_duplicate_completed_or_missing_config_or_seed_fails(self):
        for problem in ("duplicate", "config", "checkpoint"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); paths = self.history(root)
                if problem == "duplicate":
                    run = paths[101].with_name("20260102_000000_base_clean_s101")
                    run.mkdir(); (run / "last.pth").write_bytes(b"another")
                else:
                    (paths[101] / ("configs.yaml" if problem == "config" else "last.pth")).unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    driver.historical_run_configs(root, "rgbt")

    def test_rgbnt_base_yaml_uses_completed_rgbt_s102_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths = self.history(root)
            base = root / "original.yaml"; base.write_text("dataset_name: RGBT-PEDES\n")
            (paths[102] / "configs.yaml").write_text(
                "config_file: " + base.as_posix() + "\n", encoding="utf-8")
            aborted = paths[102].with_name("20260102_000000_base_clean_s102")
            aborted.mkdir(); (aborted / "configs.yaml").write_text("config_file: absent\n")
            from types import SimpleNamespace
            self.assertEqual(driver.training_base(SimpleNamespace(root=root), "rgbnt"), base)


# These stubs preserve the tools' output-existence contracts, not their model
# computation. They run in child processes after the real Runner has created
# its manifest/log, so a directory/file collision is observable at execution.
_S3_OUTPUT_CONTRACT_STUB = r"""
import json,sys
from pathlib import Path
kind,encoded,directory,events=sys.argv[1:]
argv=json.loads(encoded);directory=Path(directory)
def arg(flag):return argv[argv.index(flag)+1]
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n',encoding='utf-8')
assert (directory/'manifest.json').is_file(), 'Runner manifest not pre-created'
assert (directory/'driver.log').is_file(), 'Runner log not pre-created'
if kind in ('derive','runtime'):
    target=Path(arg('--output'))
    sidecar=target.with_suffix('.derivation.json' if kind=='derive' else '.runtime.json')
    targets=[(target,True),(sidecar,True)]
elif kind=='train':
    config=load(arg('--config_file'));target=Path(config['output_dir']);targets=[(target,False)]
elif kind=='health':target=Path(arg('--out'));targets=[(target,False)]
elif kind=='export':target=Path(arg('--output_dir'));targets=[(target,True)]
elif kind=='pack':target=Path(arg('--out_dir'));targets=[(target,False)]
else:target=Path(arg('--output'));targets=[(target,True)]
event={'kind':kind,'unit_dir':str(directory),'argv':argv,
       'manifest_exists':True,'driver_log_exists':True,
       'targets':[{'path':str(p),'refuses_existing':refuse,'existed':p.exists()} for p,refuse in targets]}
with Path(events).open('a',encoding='utf-8') as stream:stream.write(json.dumps(event)+'\n')
for path,refuse in targets:
    if refuse and path.exists():raise FileExistsError('Refusing existing '+kind+' output: '+str(path))
if kind=='derive':
    seed=int(arg('--seed'));dataset=arg('--dataset');smoke='--smoke' in argv
    config=load(arg('--base'))
    config.update(dataset_name=dataset,train_seed=seed,output_dir=arg('--output_dir'),
                  root_dir=arg('--root_dir'),mlm_single_count=True)
    if '--dev_split_file' in argv:config['dev_split_file']=arg('--dev_split_file')
    config['name']=('base_mlm1_rgbnt' if dataset=='RGBNT201-PEDES' else 'base_mlm1')+f'_s{seed}'+('_smoke' if smoke else '')
    write(target,config);write(sidecar,{'synthetic':True})
elif kind=='runtime':write(target,load(arg('--config')));write(sidecar,{'synthetic':True})
elif kind=='train':
    run=target/config['dataset_name']/('20000101_000000_'+config['name']);run.mkdir(parents=True)
    write(run/'configs.yaml',config);write(run/'run_record.json',{'synthetic':True})
    (run/'last.pth').write_bytes(b'synthetic marker, not a trained checkpoint')
    (run/'train_log.txt').write_text('synthetic training command\n',encoding='utf-8')
elif kind=='health':write(target,{'ok':True,'synthetic':True})
elif kind=='export':
    target.mkdir(parents=True)
    (target/'metrics.tsv').write_text('scene\tR1\nclean\t1\n',encoding='utf-8')
    (target/'features').mkdir();(target/'features/payload.txt').write_text('synthetic export payload',encoding='utf-8')
elif kind=='pack':
    source=Path(arg('--export_dirs'));assert (source/'features/payload.txt').is_file()
    target.mkdir(parents=True,exist_ok=True)
    (target/(arg('--names')+'.npz')).write_bytes(b'synthetic packed marker')
else:
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(b'synthetic '+kind.encode()+b' marker')
"""


class _S3OutputContractRunner(driver.Runner):
    def __init__(self, root, stub, event_path):
        super().__init__(root, stage='S3', min_free_gb=0)
        self.stub, self.event_path, self.planned = stub, event_path, []

    def unit(self, name, directory, inputs, commands, parameters=None):
        commands = [[str(value) for value in command] for command in commands]
        self.planned.append(dict(name=name, directory=Path(directory),
                                 inputs=list(inputs), commands=commands))
        if name.startswith('derive_'):kind = 'derive'
        elif name.startswith('runtime_health_base_'):kind = 'runtime'
        elif name.startswith('train_ir_'):kind = 'train_ir'
        elif name.startswith('train_'):kind = 'train'
        elif name.startswith('health_'):kind = 'health'
        elif name.startswith('export_'):kind = 'export'
        elif name.startswith('pack12_'):kind = 'pack12'
        elif name.startswith('pack_'):kind = 'pack'
        elif name.startswith('subset_'):kind = 'subset'
        else:raise AssertionError('Uncovered S3 tool: ' + name)
        rewritten = [[sys.executable, '-B', str(self.stub), kind,
                      json.dumps(command), str(directory), str(self.event_path)] for command in commands]
        # Full original arguments remain in the manifest's command signature.
        return super().unit(name, directory, inputs, rewritten, parameters)


def _run_s3_output_contract(root, legacy=False):
    from tools.p4.r19 import training
    root = Path(root);root.mkdir()
    _protected(root)
    histories = {}
    for dataset in ('rgbt', 'rgbnt'):
        histories[dataset] = {}
        for seed in driver.SEEDS:
            path = root/'history'/dataset/f's{seed}.yaml';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({'root_dir':str(root/'data'),'train_seed':seed}),encoding='utf-8')
            histories[dataset][seed] = path
    base = root/'base.yaml';split = root/'dev.json';annotation = root/'annotation.json';weights = root/'weights.marker'
    split.write_text('{}');annotation.write_text('[]');weights.write_bytes(b'not model weights')
    base.write_text(json.dumps({'dataset_name':'RGBT-PEDES','dev_split_file':str(split)}),encoding='utf-8')
    stub = root/'output_contract_stub.py';stub.write_text(_S3_OUTPUT_CONTRACT_STUB,encoding='utf-8')
    runner = _S3OutputContractRunner(root,stub,root/'events.jsonl')
    spec = {'seeds':list(driver.SEEDS),'fits':[7001],'assign_seeds':[7031]}
    def fingerprint(dataset, root_dir, output_path=None):
        record = {'annotation_path':str(annotation),'synthetic':True}
        driver.save(output_path,record)
        return record
    def compare(expected, historical, seed, output_path=None, smoke=False):
        driver.save(output_path,{'ok':True,'synthetic':True})
    error = None
    with patch.object(driver,'historical_run_configs',side_effect=lambda root,dataset:histories[dataset]), \
         patch.object(driver,'training_base',return_value=base), \
         patch.object(driver,'pretrained_input',return_value={'path':str(weights),'sha256':driver.sha256(weights)}), \
         patch.object(training,'data_fingerprint',side_effect=fingerprint), \
         patch.object(training,'compare_recipe_history',side_effect=compare):
        plan = driver.s3
        if legacy:
            import ast
            old = subprocess.check_output(['git','show','9dfe09bbc67c8ea2233cf33c1c73d9b545d78f97:IRRA-main/tools/p4/r19/driver.py'],cwd=CODE.parent).decode('utf-8')
            node = next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name=='s3')
            scope = vars(driver).copy()
            exec(compile(ast.Module(body=[node],type_ignores=[]),'git:9dfe09bb/driver.py','exec'),scope)
            plan = scope['s3']
        try:plan(runner,spec)
        except subprocess.CalledProcessError as exc:error = exc
        runner.finish('S3','STOPPED' if error else 'COMPLETE',error)
    return runner,error


class S3OutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='r19_s3_output_contract_')
        cls.root = Path(cls.temp.name)
        cls.runner,error = _run_s3_output_contract(cls.root/'current')
        if error:
            log = cls.runner.planned[-1]['directory']/'driver.log'
            raise AssertionError(log.read_text(encoding='utf-8'))
        cls.events = [json.loads(line) for line in cls.runner.event_path.read_text(encoding='utf-8').splitlines()]

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_legacy_s3_plan_fails_on_precreated_export_directory(self):
        runner,error = _run_s3_output_contract(self.root/'legacy',legacy=True)
        self.assertIsInstance(error,subprocess.CalledProcessError)
        self.assertNotEqual(error.returncode,0)
        self.assertEqual(runner.planned[-1]['name'],'export_rgbt_dev_s101')
        directory = runner.planned[-1]['directory']
        self.assertIn('FileExistsError',(directory/'driver.log').read_text(encoding='utf-8'))
        self.assertFalse((directory/'.done').exists())
        self.assertEqual((runner.out/'STATUS_S3').read_text(),'STOPPED\n')
        self.assertEqual(runner.records['health_rgbt_s101']['record']['status'],'complete')
        self.assertEqual(runner.records['health_rgbt_s101_smoke']['record']['status'],'complete')

    def test_current_s3_chains_pack_input_to_export_payload_and_hashes_all_files(self):
        planned = {unit['name']:unit for unit in self.runner.planned}
        self.assertEqual(len(planned),77)
        self.assertEqual((self.runner.out/'STATUS_S3').read_text(),'COMPLETE\n')
        for dataset in ('rgbt','rgbnt'):
            for seed in driver.SEEDS:
                for split in ('dev','test'):
                    label = f'{dataset}_{split}_s{seed}'
                    export,pack = planned['export_'+label],planned['pack_'+label]
                    command = export['commands'][0]
                    target = Path(command[command.index('--output_dir')+1])
                    self.assertEqual(target,export['directory']/'export')
                    command = pack['commands'][0]
                    self.assertEqual(Path(command[command.index('--export_dirs')+1]),target)
                    manifest = json.loads((pack['directory']/'manifest.json').read_text(encoding='utf-8'))
                    for path in target.rglob('*'):
                        if path.is_file():
                            item = manifest['inputs'][str(driver.resolved_path(path))]
                            self.assertEqual(item['sha256'],driver.sha256(path))
        self.assertEqual(set(json.loads((self.runner.out/'mlm1/evaluation_inputs.json').read_text())['datasets']),
                         {'rgbt','rgbnt','rgbnt_subset'})

    def test_all_refusing_outputs_are_absent_after_runner_creates_metadata(self):
        self.assertEqual({event['kind'] for event in self.events},
                         {'derive','runtime','train','health','export','pack','pack12','train_ir','subset'})
        for event in self.events:
            self.assertTrue(event['manifest_exists'])
            self.assertTrue(event['driver_log_exists'])
            for target in event['targets']:
                if target['refuses_existing']:
                    self.assertFalse(target['existed'],event)
            if event['kind'] in ('train','pack'):
                self.assertTrue(event['targets'][0]['existed'],event)

    def test_file_output_stubs_refuse_existing_files_without_overwriting(self):
        for kind in ('train_ir','pack12','subset'):
            with self.subTest(kind=kind):
                directory = self.root/'negative'/kind;directory.mkdir(parents=True)
                (directory/'manifest.json').write_text('{}');(directory/'driver.log').touch()
                target = directory/'existing.npz';target.write_bytes(b'keep existing bytes')
                command = [sys.executable,'-B',str(self.runner.stub),kind,
                           json.dumps(['stub','--output',str(target)]),str(directory),str(directory/'event.jsonl')]
                result = subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(b'FileExistsError',result.stdout)
                self.assertEqual(target.read_bytes(),b'keep existing bytes')


if __name__ == '__main__':
    unittest.main(verbosity=2)
