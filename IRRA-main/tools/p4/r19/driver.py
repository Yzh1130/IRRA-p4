"""R19 staged runner: immutable inputs, byte-identical manifests, retained stale work.

No stage launches automatically. S0/S1/S2/S4 are CPU evaluation; S3 explicitly
launches GPU training. R19_SPEC_JSON can replace input paths for a synthetic S0.
"""
from __future__ import annotations

import argparse
import ast
import collections
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import csv
from dataclasses import dataclass, field
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import quote
import uuid

CODE = Path(__file__).resolve().parents[3]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

SEEDS = (101, 102, 103)
FITS = (7001, 7002, 7003)
ASSIGN_SEEDS = (7031, 7032, 7033, 7034, 7035)
MIXES = ('M0 M4 M6c M1g M4c60 M9 M5e M5f M4id M10 M4c25 M4c50 M4c75').split()
METHODS1 = ('B* QE-DBA QEir QE+Const QE+SRC2f QE+SRC2f+kap QE+SRC2f+pI '
            'QE+CtxOnly QE+M QE+EM QE+Orc QE+M-lam QE+M-lam2 A-2f-ctxn FeatImp-g FeatImp-r').split()
METHODS2 = 'QE-DBA QE+M QE+M-xcam QE+M-xcam-refit QE+Const'.split()
METHODS_NESTED = 'B* QE-DBA QEir QE+Const QE+SRC2f QE+M QE+Orc'.split()
METHODS4 = ('B* CAET A-SRC-2f A-2f-EM A-2f-ctxn A-2f-ctxn-g GP-same KR RVN QE-DBA '
            'KRir RVNir QEir FeatImp FeatImp-g KR+M RVN+M QE+M KRir+M RVNir+M QEir+M '
            'GP+M QE+CAET QE+SRC2f QE+EM QE+Orc DB Raw Oracle-A '
            'QE+CtxOnly QE+SRC2f+kap QE+SRC2f+pI QE+Const QE+NoGate').split()
FROZEN = {
    'tools/p4/transfer/r15_test.py': '670a89bed5c5a7a367382bc3d9484c1f38455f828826e6f41711db4674fdb82c',
    'tools/p4/transfer/r17_const.py': '0871f83d715f64fe1099969d0918f9e7348739c8b7a51f71e4c74d34f8163d95',
    'tools/p4/transfer/r18_xcam.py': 'ea2951d63034244ac407418a43b8b8237d2a7e36c1a2eb6052b1800e82b0e61d',
    'tools/p4/transfer/judge_test.py': '1375f6add366f72e2ff9677d16b6a84fab950a2ece4d99a94494dd5c123e126a',
    'tools/p4/transfer/r17_judge.py': '3a5d73fcec8cb448cd5bbe2a345a3f35c3c4bddf67c61e78cf69fd41bbc74c57',
    'tools/p4/r18/r18_verdict.py': 'ab7884c49f8826320d4663ca8a0dbac103b8ad9dd9f872a6bbfec14a9e7ea881',
    'tools/p4/r18/run_frozen.py': 'bbb245ff9a0f142eb529c2ecf9328eae517d7f4931fe6cccddabfc5e4ccc0f99',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(value))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resolved_path(path):
    path = Path(path).resolve()
    if os.name == 'nt':
        text = str(path)
        if text.startswith('\\\\?\\UNC\\'):
            return Path('\\\\' + text[8:])
        if text.startswith('\\\\?\\'):
            return Path(text[4:])
    return path


@contextmanager
def stage_lock(out, stage, commit):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ('.driver_' + stage + '.lock')
    metadata = dict(pid=os.getpid(), hostname=socket.gethostname(), started_at=now_iso(), git_commit=commit)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # A concurrent creator may not have finished the very short JSON write.
        for _ in range(50):
            try:
                previous = json.loads(path.read_text(encoding='utf-8'))
                break
            except (ValueError, OSError):
                previous = {'metadata': 'unavailable; inspect this lock file'}
                time.sleep(0.01)
        raise RuntimeError('Stage already locked; not deleting ' + str(path) + ': ' + json.dumps(previous, ensure_ascii=False))
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(metadata))
        yield metadata
    finally:
        path.unlink(missing_ok=True)


def protected_baseline(path, snapshot):
    """Serialize first publication across stages, then publish one atomic JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    claim = path.parent / '.protected_init.lock'
    deadline = time.monotonic() + 30
    while True:
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError('Protected baseline initialization is locked; inspect ' + str(claim))
            time.sleep(0.02)
    try:
        if path.exists():
            if path.read_bytes() != canonical(snapshot):
                raise ValueError('Protected directories differ from the recorded baseline')
        else:
            save(path, snapshot)
    finally:
        claim.unlink(missing_ok=True)


def calculation_paths(code):
    code = Path(code).resolve()
    from tools.p4.r18 import workflow
    paths = set(FROZEN) | {'tools/p4/transfer/r19_variants.py',
                          'tools/p4/r19/core.py', 'tools/p4/r19/r19_judge.py'}
    paths.update(Path(p).resolve().relative_to(CODE).as_posix()
                 for p in workflow.signature_paths('eval'))
    return sorted('IRRA-main/' + p for p in paths)


def require_prerequisites(out, stage, code):
    out, code = Path(out), Path(code).resolve()
    if stage == 'S0':
        return
    status = out / 'STATUS_S0'
    if not status.is_file() or status.read_text(encoding='utf-8').strip() != 'COMPLETE':
        raise ValueError('S0 prerequisite: STATUS_S0 must be COMPLETE')
    digest = out / 'DIGEST19_S0.txt'
    if not digest.is_file():
        raise ValueError('S0 prerequisite: DIGEST19_S0.txt is missing')
    import re
    commits = re.findall(r'^git_commit ([0-9a-f]{40})$', digest.read_text(encoding='utf-8'), re.M)
    if len(commits) != 1:
        raise ValueError('S0 prerequisite: expected one full git_commit in DIGEST19_S0.txt')
    paths = calculation_paths(code)
    for refs in ((commits[0], 'HEAD'), ('HEAD',)):
        result = subprocess.run(['git', 'diff', '--quiet', *refs, '--', *paths], cwd=code.parent,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            reason = 'calculation-related files changed' if result.returncode == 1 else 'cannot compare S0 git commit'
            raise ValueError('S0 prerequisite: ' + reason + ': ' + result.stderr.decode('utf-8', errors='replace').strip())
    if stage == 'S4':
        s3 = out / 'STATUS_S3'
        if not s3.is_file() or s3.read_text(encoding='utf-8').strip() != 'COMPLETE':
            raise ValueError('S4 prerequisite: STATUS_S3 must be COMPLETE')
        if (out / '.driver_S3.lock').exists():
            raise ValueError('S4 prerequisite: S3 is still running')
        if not (out / 'mlm1/evaluation_inputs.json').is_file():
            raise FileNotFoundError('S4 requires mlm1/evaluation_inputs.json')
    if stage == 'S3' and (out / '.driver_S4.lock').exists():
        raise ValueError('S3 cannot start while S4 is running')


@dataclass(frozen=True)
class Unit:
    name: str
    directory: Path
    inputs: object
    commands: object
    parameters: object = None
    dependencies: tuple = field(default_factory=tuple)
    serial: bool = False


def stage_inputs(out, stage):
    relative = {'S1': 's1', 'S2': 's2', 'S4': 'mlm1/eval'}[stage]
    return sorted(p for p in (Path(out) / relative).rglob('*') if p.is_file() and
                  p.name.endswith(('_cells.csv', '_diag_cells.csv', '_selection.json', '_perq_ext.pkl', '_run_record.json')))


def run_units(runner, units):
    """Bounded DAG dispatch: no queued excess work; drain on the first failure."""
    units = list(units)
    names = [u.name for u in units]
    if len(names) != len(set(names)) or any(set(u.dependencies) - set(names) for u in units):
        raise ValueError('Duplicate unit names or unknown dependencies')
    pending, active, completed = units.copy(), {}, set()
    error = None
    def execute(unit):
        try:
            inputs = unit.inputs() if callable(unit.inputs) else unit.inputs
            return runner.unit(unit.name, unit.directory, inputs, unit.commands, unit.parameters)
        except BaseException as exc:
            runner.fail(exc)
            raise
    with ThreadPoolExecutor(max_workers=runner.par) as pool:
        while pending or active:
            if runner.stop_event.is_set() and error is None:
                error = runner.first_error or RuntimeError('Unit failed; stopping dispatch')
            if error is None:
                for unit in list(pending):
                    if len(active) >= runner.par or runner.stop_event.is_set():
                        break
                    if not set(unit.dependencies) <= completed:
                        continue
                    if unit.serial and active:
                        continue
                    try:
                        with runner.dispatch_lock:
                            if runner.stop_event.is_set():
                                break
                            runner.check_disk()
                            pending.remove(unit)
                            if unit.serial:
                                execute(unit)
                                completed.add(unit.name)
                            else:
                                active[pool.submit(execute, unit)] = unit
                    except BaseException as exc:
                        error = exc
                        runner.fail(exc)
                        break
            if active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    unit = active.pop(future)
                    try:
                        future.result()
                        completed.add(unit.name)
                    except BaseException as exc:
                        if error is None:
                            error = exc
                        runner.fail(exc)
            elif error is not None:
                break
            elif pending and not any(set(u.dependencies) <= completed for u in pending):
                error = ValueError('Cyclic or unsatisfied unit dependencies')
                runner.fail(error)
    if error is not None:
        raise error
    return completed


def files_of(paths):
    result = {}
    for path in paths:
        path = resolved_path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = sorted(path.rglob('*')) if path.is_dir() else [path]
        for p in candidates:
            if p.is_symlink():
                raise ValueError('Symlink is not a read-only input: ' + str(p))
            if p.is_file():
                result[str(p)] = {'bytes': p.stat().st_size, 'sha256': sha256(p)}
    return result


def unique(root, pattern):
    paths = sorted(Path(root).glob(pattern))
    if pattern == '*_cells.csv':
        paths = [p for p in paths if not p.name.endswith('_diag_cells.csv')]
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one {root}/{pattern}; found {len(paths)}')
    return paths[0]


def source_signature(code=CODE):
    """Include the actual local import closure plus every new R19 scoring file."""
    from tools.p4.r18 import workflow
    paths = set(workflow.signature_paths('l1') + workflow.signature_paths('eval'))
    for directory in (code / 'tools/p4/r19',):
        paths.update(p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.py', '.sh', '.md'))
    paths.add(code / 'tools/p4/transfer/r19_variants.py')
    override = code / 'configs/p4/r19_mlm1_override.yaml'
    if override.is_file():
        paths.add(override)
    paths.update(code / p for p in FROZEN)
    return files_of(sorted(paths))


def gallery_keys(archive):
    """The frozen 12-key archives preserve raw identity and within-ID order."""
    import numpy as np
    with np.load(archive, allow_pickle=False) as z:
        qpid, raw = z['clean/txt_pids'], z['clean/query_raw_pids']
        mapping = {}
        for a, b in zip(qpid.tolist(), raw.tolist()):
            if int(a) in mapping and mapping[int(a)] != int(b):
                raise ValueError('One archive PID maps to more than one raw identity')
            mapping[int(a)] = int(b)
        counts = collections.Counter()
        keys = []
        for pid in z['clean/img_pids']:
            identity = mapping[int(pid)]
            keys.append((identity, counts[identity]))
            counts[identity] += 1
    return keys


def subset_cameras(full, subset, cams, output):
    import numpy as np
    whole = gallery_keys(full)
    smaller = gallery_keys(subset)
    index = {key: i for i, key in enumerate(whole)}
    if len(index) != len(whole) or any(k not in index for k in smaller):
        raise ValueError('Subset camera mapping is not unique')
    with np.load(cams, allow_pickle=False) as z:
        dev, test = z['dev_cams'], z['test_cams']
        if len(test) != len(whole):
            raise ValueError('Full test camera count does not match gallery')
        values = {'dev_cams': dev, 'test_cams': test[[index[k] for k in smaller]]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        np.savez_compressed(f, **values)


def check_m0(directory):
    path = unique(directory, '*_cells.csv')
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    by = collections.defaultdict(dict)
    for row in rows:
        if row['mixture'] == 'M0' and row['method'] in ('QE+M', 'QE-DBA'):
            by[(row.get('evalset', ''), row['aseed'])][row['method']] = row
    if not by:
        raise ValueError('M0 cells are missing')
    compared = 0
    for key, methods in by.items():
        if set(methods) != {'QE+M', 'QE-DBA'}:
            raise ValueError('M0 method is missing for ' + str(key))
        left, right = methods['QE+M'], methods['QE-DBA']
        for field in left:
            if field not in ('method',):
                if left[field] != right[field]:
                    raise ValueError(f'M0 mismatch: {key} {field}: {left[field]} != {right[field]}')
        compared += 1
    return {'m0_pairs': compared, 'equal': True}


class Runner:
    def __init__(self, root, code=CODE, synthetic=False, par=1, min_free_gb=10, stage=None):
        self.root = resolved_path(root)
        self.out = self.root / 'p4_r19_out'
        self.code = resolved_path(code)
        self.synthetic = synthetic
        self.legacy_finish_status = stage is None
        self.stage = stage or 'S0'
        self.par = 1 if self.stage in ('S0', 'S3') else par
        if isinstance(par, bool) or not isinstance(par, int) or par < 1:
            raise ValueError('par must be a positive integer')
        if not math.isfinite(min_free_gb) or min_free_gb < 0:
            raise ValueError('min_free_gb must be finite and nonnegative')
        self.min_free_gb = min_free_gb
        self.started_at = now_iso()
        self.started_clock = time.perf_counter()
        self.guard_lock = threading.RLock()
        self.dispatch_lock = threading.RLock()
        self.event_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.first_error = None
        self.events, self.records = [], {}
        self.disk_start_gb = self.check_disk()
        self.protected = [self.root / n for n in ('p4_r15_out', 'p4_r17_out', 'p4_r18_out')]
        for path in self.protected:
            if self.out.is_relative_to(path) or path.is_relative_to(self.out):
                raise ValueError('Output overlaps a protected directory')
            if not path.is_dir():
                raise FileNotFoundError(path)
        self.out.mkdir(parents=True, exist_ok=True)
        self.before = files_of(self.protected)
        protected_baseline(self.out / 'checks/protected.json', self.before)
        self.env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                        PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        self.env['PYTHONPATH'] = os.pathsep.join([str(self.code / 'tests/p4'),
                                                str(self.code), self.env.get('PYTHONPATH', '')])
        self.code_hashes = source_signature(self.code)
        for relative, expected in FROZEN.items():
            if sha256(self.code / relative) != expected:
                raise ValueError('Frozen code hash mismatch: ' + relative)
        self.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.code,
                                              text=True).strip()

    def check_disk(self):
        available = shutil.disk_usage(self.root).free / (1024 ** 3)
        if available < self.min_free_gb:
            raise RuntimeError(f'磁盘不足: available={available:.3f} GB; minimum={self.min_free_gb:g} GB')
        return available

    def fail(self, error):
        with self.dispatch_lock:
            if self.first_error is None:
                self.first_error = error
            self.stop_event.set()

    def remember(self, name, directory, status, record=None):
        with self.event_lock:
            self.events.append({'unit': name, 'status': status})
            self.records[name] = {'directory': str(directory), 'record': record or {}}

    def guard(self):
        with self.guard_lock:
            current = files_of(self.protected)
            if current != self.before:
                save(self.out / 'checks/protected_changed.json', current)
                raise ValueError('Protected output changed; stopping')
            print('[R19_GUARD] unchanged', len(current), flush=True)

    def inside(self, path):
        path = resolved_path(path)
        if not path.is_relative_to(self.out) or path == self.out:
            raise ValueError('R19 writes must remain under p4_r19_out: ' + str(path))
        return path

    def unit(self, name, directory, inputs, commands, parameters=None):
        try:
            self.check_disk()
            return self._unit(name, directory, inputs, commands, parameters)
        except BaseException as exc:
            self.fail(exc)
            raise

    def _unit(self, name, directory, inputs, commands, parameters):
        directory = self.inside(directory)
        signature = canonical({'schema': 1, 'git_commit': self.commit,
                               'code': self.code_hashes, 'inputs': files_of(inputs),
                               'commands': [[str(x) for x in c] for c in commands],
                               'parameters': parameters or {},
                               'threads': {k: self.env[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}})
        digest = hashlib.sha256(signature).hexdigest()
        manifest, done = directory / 'manifest.json', directory / '.done'
        if directory.exists():
            same = manifest.is_file() and manifest.read_bytes() == signature and done.is_file() and done.read_text().strip() == digest
            record_path = directory / 'driver_run_record.json'
            if same and record_path.is_file():
                previous = json.loads(record_path.read_text(encoding='utf-8'))
                same = previous.get('outputs') == self.output_hashes(directory)
            else:
                same = False
            if same:
                self.guard()
                print('[R19_REUSE]', name, flush=True)
                self.remember(name, directory, 'reused', previous)
                return directory
            stale = self.out / '_stale' / (quote(name, safe='') + '_' + str(os.getpid()) + '_' + uuid.uuid4().hex)
            self.inside(stale)
            stale.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(directory), str(stale))
            print('[R19_STALE]', directory, '->', stale, flush=True)
        directory.mkdir(parents=True)
        manifest.write_bytes(signature)
        temporary = self.inside(self.out / '_tmp' / self.stage / quote(name, safe=''))
        temporary.mkdir(parents=True, exist_ok=True)
        env = dict(self.env, TMP=str(temporary), TEMP=str(temporary), TMPDIR=str(temporary))
        start, started_at = time.perf_counter(), now_iso()
        codes, peaks, error = [], [], None
        try:
            with (directory / 'driver.log').open('wb') as log:
                for command in commands:
                    command = [str(x) for x in command]
                    print('[R19_RUN]', name, ' '.join(command), flush=True)
                    process = subprocess.Popen(command, cwd=self.code, env=env,
                                               stdout=log, stderr=subprocess.STDOUT)
                    if hasattr(os, 'wait4'):
                        while True:
                            try:
                                _, status, usage = os.wait4(process.pid, 0)
                                break
                            except InterruptedError:
                                continue
                        process.returncode = os.waitstatus_to_exitcode(status)
                        peaks.append(int(usage.ru_maxrss * 1024))
                    else:
                        process.wait()
                    codes.append(process.returncode)
                    if process.returncode:
                        raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException as exc:
            error = exc
            self.fail(exc)
        try:
            self.guard()
        except BaseException as exc:
            if error is None:
                error = exc
            self.fail(exc)
        record = {'status': 'failed' if error else 'complete', 'unit': name,
                  'git_commit': self.commit, 'exit_codes': codes,
                  'wall_seconds': time.perf_counter() - start, 'manifest_sha256': digest,
                  'peak_gpu_memory_bytes': None, 'par': self.par,
                  'started_at': started_at, 'finished_at': now_iso(),
                  'peak_rss_bytes': max(peaks) if peaks else None}
        if error is None:
            record.update(peak_memory_note='CPU driver; GPU peaks remain in training run_record',
                          outputs=self.output_hashes(directory))
        save(directory / 'driver_run_record.json', record)
        if error is not None:
            self.remember(name, directory, 'failed', record)
            raise error
        done.write_text(digest + '\n', encoding='utf-8', newline='\n')
        self.remember(name, directory, 'complete', record)
        return directory

    @staticmethod
    def output_hashes(directory):
        excluded = {'manifest.json', '.done', 'driver_run_record.json'}
        return {str(p.relative_to(directory)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                for p in sorted(Path(directory).rglob('*')) if p.is_file() and p.name not in excluded}

    def finish(self, stage, status, error=None):
        guard_error = None
        try:
            self.guard()
        except BaseException as exc:
            guard_error, error = exc, exc
            status = 'STOPPED: PROTECTED_OUTPUT_CHANGED' if self.legacy_finish_status else 'STOPPED'
        (self.out / ('STATUS_' + stage)).write_text(status + '\n', encoding='utf-8', newline='\n')
        events = sorted(self.events, key=lambda x: x['unit'])
        digest = ['R19 ' + stage + ' post-hoc validation', 'STATUS ' + status,
                  'git_commit ' + self.commit, 'error ' + (str(error) if error else 'none'),
                  'events ' + json.dumps(events, ensure_ascii=False)]
        for _, entry in sorted(self.records.items()):
            path = Path(entry['directory']) / 'driver_run_record.json'
            if path.is_file():
                digest.append('UNIT ' + str(path.relative_to(self.out)) + ' sha256=' + sha256(path))
        for path in sorted((self.out / 'reports' / stage).glob('*.md')) if (self.out / 'reports' / stage).exists() else []:
            digest += ['REPORT ' + str(path.relative_to(self.out)), path.read_text(encoding='utf-8')]
        rss = [(entry['record'].get('peak_rss_bytes'), name) for name, entry in self.records.items()
               if entry['record'].get('peak_rss_bytes') is not None]
        maximum = max(rss) if rss else (None, None)
        summary = dict(unit_count=len(events), wall_seconds=time.perf_counter()-self.started_clock,
                       peak_rss_bytes=maximum[0], peak_rss_unit=maximum[1],
                       started_at=self.started_at, finished_at=now_iso(),
                       free_gb_start=self.disk_start_gb,
                       free_gb_end=shutil.disk_usage(self.root).free/(1024**3))
        digest.append('SUMMARY ' + json.dumps(summary, ensure_ascii=False, sort_keys=True))
        (self.out / ('DIGEST19_' + stage + '.txt')).write_text('\n'.join(digest) + '\n', encoding='utf-8', newline='\n')
        if guard_error is not None:
            raise guard_error


def default_specs(root, seeds=SEEDS):
    root = Path(root)
    result = {'seeds': list(seeds), 'fits': list(FITS), 'assign_seeds': list(ASSIGN_SEEDS),
              'grid': 'v3', 'datasets': {}, 'no_identity_test': 'test_r18_eval.CameraTests'}
    for dataset in ('rgbt', 'rgbnt', 'rgbnt_subset'):
        by_seed = {}
        for seed in seeds:
            if dataset == 'rgbt':
                dev = root / ('p4_r15_out/feats' if seed == 103 else 'r13_inputs') / f'base_ref_s{seed}.npz'
                test = root / f'p4_r15_out/feats/test_base_s{seed}.npz'
                train = root / ('p4_r15_out/train_ir' if seed == 103 else 'p4_r13_out/train_ir') / f'base_ref_s{seed}_train_ir.npz'
                parent = root / 'p4_r15_out/test/constrained'
                reference = root / 'p4_r17_out/const'
                cams = None
            else:
                dev = root / f'p4_r18_out/feats/r201_dev_s{seed}.npz'
                test = root / f'p4_r18_out/feats/r201_{"testnov" if dataset.endswith("subset") else "test"}_s{seed}.npz'
                train = root / f'p4_r18_out/train_ir/r201_s{seed}_train_ir.npz'
                parent = root / ('p4_r18_out/subset/test/constrained' if dataset.endswith('subset') else 'p4_r18_out/test/constrained')
                reference = root / ('p4_r18_out/subset/const' if dataset.endswith('subset') else 'p4_r18_out/const')
                cams = root / f'p4_r18_out/feats/cams_s{seed}.npz'
            by_seed[str(seed)] = {'dev': str(dev), 'test': str(test), 'train_ir': str(train),
                'selection_root': str(parent), 'regression_root': str(reference), 'guarded': dataset != 'rgbt',
                'cams': str(cams) if cams else None}
        result['datasets'][dataset] = by_seed
    return result


def evaluation_commands(runner, cfg, seed, fit, directory, methods, mixes,
                        assign_seeds, grid='v3', fixed=True, lam_curve=False,
                        extra=(), regress=False, cams=None):
    script = runner.code / 'tools/p4/transfer/r19_variants.py'
    command = [sys.executable, '-B']
    if cfg.get('guarded'):
        command += [str(runner.code / 'tools/p4/r18/run_frozen.py'), '--expect_sha', sha256(script),
                    '--log', str(directory / 'degenerate_fits.json'), '--']
    command += [str(script), '--b12', '.', '--feats', str(Path(cfg['dev']).parent),
                '--base', Path(cfg['dev']).stem, '--adapter', Path(cfg['dev']).stem,
                '--assign', 'none', '--out', str(directory), '--tag', f'r19_s{seed}_f{fit}',
                '--assign_seeds', *map(str, assign_seeds), '--mixes', *mixes,
                '--skip_joint', '--methods', *methods, '--grid_preset', grid,
                '--select_rule', 'constrained', '--r_max', '0.999999', '--tie_break', 'bstar',
                '--test_npz', cfg['test'], '--link_map', 'train_lda', '--train_ir_npz', cfg['train_ir'],
                '--test_seed', str(fit), '--perq_ext']
    inputs = [cfg['dev'], cfg['test'], cfg['train_ir']]
    if fixed:
        selection = cfg.get('selection') or str(unique(Path(cfg['selection_root']) / f's{seed}_f{fit}', '*_selection.json'))
        command += ['--fixed_selection', selection]
        inputs.append(selection)
    if lam_curve:
        command += ['--lam_curve']
    if cams:
        command += ['--item_cams', str(cams)]
        inputs.append(cams)
    if regress:
        old = cfg.get('regression') or str(Path(cfg['regression_root']) / f's{seed}_f{fit}')
        command += ['--regress_against', old]
        inputs.append(unique(old, '*_cells.csv'))
    command += list(map(str, extra))
    return command, inputs


def evaluation_unit(runner, spec, stage, dataset, seed, fit, *, methods, mixes=None,
              fixed=True, lam_curve=False, extra=(), regress=False, cams=None):
    cfg = spec['datasets']['rgbnt' if dataset == 'rgbnt_sparse' else 'rgbnt_subset' if dataset == 'rgbnt_nested' else dataset][str(seed)]
    directory = runner.out / stage.lower() / dataset / f's{seed}_f{fit}'
    command, inputs = evaluation_commands(runner, cfg, seed, fit, directory, methods,
                        mixes or MIXES, spec['assign_seeds'], spec.get('grid', 'v3'), fixed,
                        lam_curve, extra, regress, cams)
    if '--assign_nested_from' in extra:
        inputs.append(extra[list(extra).index('--assign_nested_from') + 1])
    return Unit(f'{stage}/{dataset}/s{seed}_f{fit}', directory, inputs, [command],
                dict(stage=stage, dataset=dataset, seed=seed, fit=fit, methods=methods,
                     mixes=mixes or MIXES, assign_seeds=spec['assign_seeds'], post_hoc=True))


def eval_unit(*args, **kwargs):
    unit = evaluation_unit(*args, **kwargs)
    args[0].unit(unit.name, unit.directory, unit.inputs, unit.commands, unit.parameters)
    return unit.directory


def s0(runner, spec):
    for dataset in ('rgbt', 'rgbnt'):
        tiny = dict(spec, assign_seeds=[7031])
        directory = eval_unit(runner, tiny, 'S0', dataset, 101, 7001,
                              methods=['B*', 'QE-DBA', 'QE+M'], mixes=['M0', 'M4'], regress=True)
        save(runner.out / f'checks/S0_{dataset}_m0.json', check_m0(directory))
        runner.guard()
    # The exact existing no-identity node is supplied by the checked spec; no
    # spelling-based substitute is silently accepted as the historical test.
    node = spec.get('no_identity_test')
    if not node:
        raise ValueError('S0 requires the verified existing no_identity_test unittest node')
    directory = runner.out / 's0/no_identity_test'
    runner.unit('S0/no_identity_test', directory,
                [runner.code / 'tests/p4/test_r18_eval.py', runner.code / 'tests/p4/test_r17.py'],
                [[sys.executable, '-B', '-m', 'unittest', node, '-v']], {'test': node})


def judge_unit(runner, stage, dependencies=()):
    directory = runner.out / 'reports' / stage
    command = [sys.executable, '-B', runner.code / 'tools/p4/r19/r19_judge.py', '--stage', stage, '--root', runner.out]
    return Unit('judge_' + stage, directory, lambda: stage_inputs(runner.out, stage), [command],
                {'post_hoc': True}, tuple(dependencies), serial=True)


def judge(runner, stage):
    unit = judge_unit(runner, stage)
    return runner.unit(unit.name, unit.directory, unit.inputs(), unit.commands, unit.parameters)


def plan_s1(runner, spec):
    units = []
    for dataset in ('rgbt', 'rgbnt', 'rgbnt_subset'):
        for seed in spec['seeds']:
            for fit in spec['fits']:
                units.append(evaluation_unit(runner, spec, 'S1', dataset, seed, fit,
                                             methods=METHODS1, lam_curve=True))
    units.append(judge_unit(runner, 'S1', [u.name for u in units]))
    return units


def s1(runner, spec):
    return run_units(runner, plan_s1(runner, spec))


def plan_s2(runner, spec):
    units = []
    for seed in spec['seeds']:
        full = spec['datasets']['rgbnt'][str(seed)]
        subset = spec['datasets']['rgbnt_subset'][str(seed)]
        directory = runner.out / 's2/cameras' / f's{seed}'
        target = directory / 'subset.npz'
        name = f'S2/cameras/s{seed}'
        cmd = [sys.executable, '-B', str(Path(__file__).resolve()), 'subset-cams',
               '--full', full['test'], '--subset', subset['test'], '--cams', full['cams'], '--output', str(target)]
        units.append(Unit(name, directory, [full['test'], subset['test'], full['cams']], [cmd]))
        for fit in spec['fits']:
            for dataset, cams in [('rgbnt', full['cams']), ('rgbnt_subset', target)]:
                unit = evaluation_unit(runner, spec, 'S2', dataset, seed, fit, methods=METHODS2,
                                       mixes=MIXES + ['Mcam1', 'Mcam2'], cams=cams)
                if dataset == 'rgbnt_subset':
                    unit = Unit(unit.name, unit.directory, unit.inputs, unit.commands,
                                unit.parameters, (name,))
                units.append(unit)
            units.append(evaluation_unit(runner, spec, 'S2', 'rgbnt_nested', seed, fit,
                                          methods=METHODS_NESTED,
                                          extra=['--assign_nested_from', full['test']]))
            units.append(evaluation_unit(runner, spec, 'S2', 'rgbnt_sparse', seed, fit,
                                          methods=METHODS_NESTED,
                                          extra=['--max_per_id', '3', '--max_per_id_seed', '9101']))
    units.append(judge_unit(runner, 'S2', [u.name for u in units]))
    return units


def s2(runner, spec):
    return run_units(runner, plan_s2(runner, spec))


def training_base(runner, dataset):
    if dataset == 'rgbt':
        path = runner.root / 'p4_preexp/configs/p4/base_clean.yaml'
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    # This is exactly R18 L1's BASE_YAML provenance, without changing that file.
    from utils.config import load_config_mapping
    cfg = unique(runner.root / 'p4_preexp/runs/base_clean_s102/RGBT-PEDES', '*_base_clean_s102/configs.yaml')
    path = Path(load_config_mapping(cfg)['config_file'])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def pretrained_input(base, code=CODE, cache=None, models=None):
    """Read-only: fail before training can download/write a missing CLIP cache."""
    from utils.config import load_config_mapping
    name = load_config_mapping(base).get('pretrain_choice', 'ViT-B/16')
    if models is None:
        tree = ast.parse((Path(code) / 'model/clip_model.py').read_text(encoding='utf-8-sig'))
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and
                          any(isinstance(t, ast.Name) and t.id == '_MODELS' for t in n.targets))
        models = ast.literal_eval(assignment.value)
    expected = None
    if name in models:
        url = models[name]
        path = (Path(cache) if cache is not None else Path.home() / '.cache/clip') / url.split('/')[-1]
        expected = url.split('/')[-2]
    else:
        path = Path(name)
    if not path.is_file():
        raise FileNotFoundError('CLIP weights must already exist; R19 will not download: ' + str(path))
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError('Existing CLIP cache SHA256 differs; no download or replacement: ' + str(path))
    return {'pretrain_choice': name, 'path': str(path.resolve()), 'sha256': actual,
            'expected_sha256': expected, 'bytes': path.stat().st_size, 'read_only': True}


def s3(runner, spec):
    """Explicit GPU stage; one two-epoch smoke per dataset before six formal runs."""
    from tools.p4.r18 import workflow
    fresh = {'seeds': list(spec['seeds']), 'fits': list(spec['fits']),
             'assign_seeds': list(spec['assign_seeds']), 'grid': 'v3',
             'datasets': {'rgbt': {}, 'rgbnt': {}, 'rgbnt_subset': {}}}
    r18 = runner.code / 'tools/p4/r18'
    py = [sys.executable, '-B']
    data_root = runner.code / 'datasets/TextReID'
    for dataset, dataset_name in [('rgbt', 'RGBT-PEDES'), ('rgbnt', 'RGBNT201-PEDES')]:
        base = training_base(runner, dataset)
        pretrained = pretrained_input(base, runner.code)
        save(runner.out / ('checks/pretrained_' + dataset + '.json'), pretrained)
        for smoke, seeds in [(True, [101]), (False, spec['seeds'])]:
            for seed in seeds:
                suffix = f'{dataset}_s{seed}' + ('_smoke' if smoke else '')
                name = ('base_mlm1_rgbnt' if dataset == 'rgbnt' else 'base_mlm1') + f'_s{seed}' + ('_smoke' if smoke else '')
                cfgdir = runner.out / 'mlm1/configs' / suffix
                config = cfgdir / 'config.yaml'
                run_parent = runner.out / ('mlm1/smoke' if smoke else 'mlm1/runs') / suffix
                command = [*py, runner.code / 'tools/p4/r19/training.py', 'derive',
                           '--base', base, '--dataset', dataset_name, '--seed', str(seed),
                           '--output_dir', run_parent, '--output', config]
                cfg_inputs = [base]
                if dataset == 'rgbnt':
                    split = runner.code / 'splits/rgbnt201_dev30.json'
                    command += ['--root_dir', data_root, '--dev_split_file', split]
                    cfg_inputs.append(split)
                if smoke:
                    command += ['--smoke']
                runner.unit('derive_' + suffix, cfgdir, cfg_inputs, [command],
                            {'dataset': dataset_name, 'train_seed': seed, 'smoke': smoke,
                             'num_epoch_override': 2 if smoke else None, 'mlm_single_count': True})
                expected_dir = runner.out / 'mlm1/runtime_configs' / suffix
                expected_config = expected_dir / 'expected.yaml'
                runner.unit('runtime_health_base_' + suffix, expected_dir, [config],
                            [[*py, runner.code / 'tools/p4/r19/training.py', 'runtime-health-base',
                              '--config', config, '--output', expected_config]],
                            {'parser_only': True, 'train_seed': seed, 'smoke': smoke})
                # TRAIN_SEED is scoped to this subprocess only; no stage-global RNG.
                train_command = [*py, '-c',
                    'import os,runpy,sys;os.environ["TRAIN_SEED"]=sys.argv.pop(1);'
                    'script=sys.argv.pop(1);sys.argv[0]=script;runpy.run_path(script,run_name="__main__")',
                    str(seed), runner.code / 'train.py', '--config_file', config]
                runner.unit('train_' + suffix, run_parent, [config, pretrained['path'], *cfg_inputs[1:]], [train_command],
                            {'dataset': dataset_name, 'train_seed': seed, 'smoke': smoke,
                             'mlm_single_count': True})
                run = workflow.find_run(run_parent / dataset_name, name)
                health_dir = runner.out / 'mlm1/health' / suffix
                runner.unit('health_' + suffix, health_dir,
                            [run / 'run_record.json', run / 'configs.yaml', run / 'train_log.txt', run / 'last.pth', config, expected_config],
                            [[*py, r18 / 'workflow.py', 'health', '--run', run, '--base', expected_config,
                              '--out', health_dir / 'health.json']],
                            {'smoke': smoke, 'post_hoc': True,
                             'base_recipe_validation': str(config.with_suffix('.derivation.json'))})
                if smoke:
                    continue
                from utils.config import load_config_mapping
                active_config = load_config_mapping(config)
                export_split = str(active_config['dev_split_file'])
                export_root = str(active_config['root_dir'])
                files = {}
                for split_name in ('dev', 'test'):
                    label = f'{dataset}_{split_name}_s{seed}'
                    export = runner.out / 'mlm1/exports' / label
                    cmd = [*py, runner.code / 'tools/p4/export_eval_split.py', '--split', split_name,
                           '--checkpoint', run / 'last.pth', '--config_file', run / 'configs.yaml',
                           '--root_dir', export_root, '--dev_split_file', export_split,
                           '--eval_degrade_seed', '2026', '--device', 'cuda',
                           '--save_modality_features', '--save_item_meta', '--output_dir', export]
                    runner.unit('export_' + label, export, [run / 'last.pth', run / 'configs.yaml'], [cmd])
                    packed = runner.out / 'mlm1/packed' / label
                    runner.unit('pack_' + label, packed, [export],
                                [[*py, runner.code / 'tools/p4/pack_feats.py', '--export_dirs', export,
                                  '--names', label, '--out_dir', packed]])
                    keys_dir = runner.out / 'mlm1/feats' / label
                    archive = keys_dir / (label + '.npz')
                    runner.unit('pack12_' + label, keys_dir, [packed / (label + '.npz')],
                                [[*py, r18 / 'workflow.py', 'pack12', '--input', packed / (label + '.npz'),
                                  '--output', archive]])
                    files[split_name] = str(archive)
                train_ir_dir = runner.out / 'mlm1/train_ir' / f'{dataset}_s{seed}'
                train_ir = train_ir_dir / 'train_ir.npz'
                runner.unit('train_ir_' + suffix, train_ir_dir,
                            [run / 'last.pth', run / 'configs.yaml', files['dev']],
                            [[*py, runner.code / 'tools/p4/export_train_ir_feats.py', '--checkpoint', run / 'last.pth',
                              '--config_file', run / 'configs.yaml', '--output', train_ir,
                              '--root_dir', export_root, '--dev_split_file', export_split,
                              '--verify_dev_npz', files['dev'], '--device', 'cuda']])
                fresh['datasets'][dataset][str(seed)] = {'dev': files['dev'], 'test': files['test'],
                    'train_ir': str(train_ir), 'guarded': dataset == 'rgbnt', 'cams': None}
                if dataset == 'rgbnt':
                    subset_dir = runner.out / 'mlm1/subset' / f's{seed}'
                    subset = subset_dir / f'rgbnt_testnov_s{seed}.npz'
                    overlap = runner.code / 'splits/rgbnt201_overlap.json'
                    runner.unit('subset_' + suffix, subset_dir, [files['test'], overlap],
                                [[*py, r18 / 'r18_subset.py', '--input', files['test'], '--overlap', overlap,
                                  '--output', subset]])
                    fresh['datasets']['rgbnt_subset'][str(seed)] = dict(fresh['datasets'][dataset][str(seed)], test=str(subset))
    save(runner.out / 'mlm1/evaluation_inputs.json', fresh)


def plan_s4(runner, spec):
    generated = runner.out / 'mlm1/evaluation_inputs.json'
    if not generated.is_file():
        raise FileNotFoundError('S4 requires completed S3: ' + str(generated))
    fresh = json.loads(generated.read_text(encoding='utf-8'))
    units = []
    for dataset in ('rgbt', 'rgbnt', 'rgbnt_subset'):
        for seed in fresh['seeds']:
            for fit in fresh['fits']:
                units.append(evaluation_unit(runner, fresh, 'mlm1/eval', dataset, seed, fit,
                                              methods=METHODS4, fixed=False, extra=['--nogate_rmax']))
    units.append(judge_unit(runner, 'S4', [u.name for u in units]))
    return units


def s4(runner, spec):
    return run_units(runner, plan_s4(runner, spec))


def positive_integer(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('par must be a positive integer')
    if number < 1:
        raise argparse.ArgumentTypeError('par must be a positive integer')
    return number


def minimum_free(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('min_free_gb must be finite and nonnegative')
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('min_free_gb must be finite and nonnegative')
    return number


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'subset-cams':
        ap = argparse.ArgumentParser()
        for name in ('full', 'subset', 'cams', 'output'):
            ap.add_argument('--' + name, required=True)
        a = ap.parse_args(argv[1:])
        subset_cameras(a.full, a.subset, a.cams, a.output)
        return 0
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['S0', 'S1', 'S2', 'S3', 'S4'], required=True)
    ap.add_argument('--root', default='/root/autodl-tmp')
    ap.add_argument('--repo', default=None)
    ap.add_argument('--spec')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--par', type=positive_integer, default=1)
    ap.add_argument('--min_free_gb', type=minimum_free, default=10)
    a = ap.parse_args(argv)
    code = Path(a.repo).resolve() / 'IRRA-main' if a.repo else CODE
    out = Path(a.root).resolve() / 'p4_r19_out'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip()
    # Lock refusal must not overwrite the active instance's STATUS or digest.
    with stage_lock(out, a.stage, commit):
        error, runner = None, None
        started = time.perf_counter()
        disk_start = shutil.disk_usage(Path(a.root)).free / (1024**3)
        try:
            if disk_start < a.min_free_gb:
                raise RuntimeError(f'磁盘不足: available={disk_start:.3f} GB; minimum={a.min_free_gb:g} GB')
            require_prerequisites(out, a.stage, code)
            if a.synthetic and a.stage == 'S3':
                raise ValueError('Synthetic rehearsal cannot launch GPU training')
            runner = Runner(a.root, code, a.synthetic, par=a.par,
                            min_free_gb=a.min_free_gb, stage=a.stage)
            spec = json.loads(Path(a.spec).read_text(encoding='utf-8')) if a.spec else default_specs(a.root)
            save(runner.out / ('checks/inputs_' + a.stage + '.json'), spec)
            {'S0': s0, 'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4}[a.stage](runner, spec)
        except BaseException as exc:
            error = exc
            raise
        finally:
            if runner is not None:
                runner.finish(a.stage, 'COMPLETE' if error is None else 'STOPPED', error)
            else:
                (out / ('STATUS_' + a.stage)).write_text('STOPPED\n', encoding='utf-8', newline='\n')
                summary = dict(unit_count=0, wall_seconds=time.perf_counter()-started,
                               peak_rss_bytes=None, peak_rss_unit=None, free_gb_start=disk_start,
                               free_gb_end=shutil.disk_usage(Path(a.root)).free/(1024**3))
                (out / ('DIGEST19_' + a.stage + '.txt')).write_text(
                    'R19 ' + a.stage + '\nSTATUS STOPPED\ngit_commit ' + commit + '\nerror ' + str(error) +
                    '\nSUMMARY ' + json.dumps(summary, ensure_ascii=False, sort_keys=True) + '\n',
                    encoding='utf-8', newline='\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
