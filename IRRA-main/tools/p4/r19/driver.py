"""R19 staged runner: immutable inputs, byte-identical manifests, retained stale work.

No stage launches automatically. S0/S1/S2/S4 are CPU evaluation; S3 explicitly
launches GPU training. R19_SPEC_JSON can replace input paths for a synthetic S0.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

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
    path.write_bytes(canonical(value))


def files_of(paths):
    result = {}
    for path in paths:
        path = Path(path).resolve()
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
    def __init__(self, root, code=CODE, synthetic=False):
        self.root = Path(root).resolve()
        self.out = self.root / 'p4_r19_out'
        self.code = Path(code).resolve()
        self.synthetic = synthetic
        self.protected = [self.root / n for n in ('p4_r15_out', 'p4_r17_out', 'p4_r18_out')]
        for p in self.protected:
            if self.out.is_relative_to(p) or p.is_relative_to(self.out):
                raise ValueError('Output overlaps a protected directory')
            if not p.is_dir():
                raise FileNotFoundError(p)
        self.out.mkdir(parents=True, exist_ok=True)
        self.before = files_of(self.protected)
        guard = self.out / 'checks/protected.json'
        if guard.exists() and guard.read_bytes() != canonical(self.before):
            raise ValueError('Protected directories differ from the recorded baseline')
        if not guard.exists():
            save(guard, self.before)
        self.events = []
        self.env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                        PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        tmp = self.out / '_tmp'
        tmp.mkdir(exist_ok=True)
        self.env.update(TMP=str(tmp), TEMP=str(tmp), TMPDIR=str(tmp))
        self.env['PYTHONPATH'] = os.pathsep.join([str(self.code / 'tests/p4'),
                                                str(self.code), self.env.get('PYTHONPATH', '')])
        self.code_hashes = source_signature(self.code)
        for relative, expected in FROZEN.items():
            if sha256(self.code / relative) != expected:
                raise ValueError('Frozen code hash mismatch: ' + relative)
        self.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.code,
                                              text=True).strip()

    def guard(self):
        current = files_of(self.protected)
        if current != self.before:
            save(self.out / 'checks/protected_changed.json', current)
            raise ValueError('Protected output changed; stopping')
        print('[R19_GUARD] unchanged', len(current), flush=True)

    def inside(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.out) or path == self.out:
            raise ValueError('R19 writes must remain under p4_r19_out: ' + str(path))
        return path

    def unit(self, name, directory, inputs, commands, parameters=None):
        directory = self.inside(directory)
        signature = canonical({'schema': 1, 'git_commit': self.commit,
                               'code': self.code_hashes, 'inputs': files_of(inputs),
                               'commands': [[str(x) for x in c] for c in commands],
                               'parameters': parameters or {},
                               'threads': {k: self.env[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}})
        digest = hashlib.sha256(signature).hexdigest()
        manifest = directory / 'manifest.json'
        done = directory / '.done'
        if directory.exists():
            same = manifest.is_file() and manifest.read_bytes() == signature and done.is_file() and done.read_text().strip() == digest
            records = directory / 'driver_run_record.json'
            if same and records.is_file():
                previous = json.loads(records.read_text(encoding='utf-8'))
                output_files = self.output_hashes(directory)
                same = previous.get('outputs') == output_files
            else:
                same = False
            if same:
                print('[R19_REUSE]', name, flush=True)
                self.events.append({'unit': name, 'status': 'reused'})
                self.guard()
                return directory
            stale = self.out / '_stale' / (name.replace('/', '_') + '_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            self.inside(stale)
            stale.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(directory), str(stale))
            print('[R19_STALE]', directory, '->', stale, flush=True)
        directory.mkdir(parents=True)
        manifest.write_bytes(signature)
        start = time.perf_counter()
        codes = []
        with (directory / 'driver.log').open('wb') as log:
            for command in commands:
                command = [str(x) for x in command]
                print('[R19_RUN]', name, ' '.join(command), flush=True)
                result = subprocess.run(command, cwd=self.code, env=self.env,
                                        stdout=log, stderr=subprocess.STDOUT)
                codes.append(result.returncode)
                self.guard()
                if result.returncode:
                    save(directory / 'driver_run_record.json', {'status': 'failed', 'unit': name,
                         'git_commit': self.commit, 'exit_codes': codes,
                         'wall_seconds': time.perf_counter() - start,
                         'manifest_sha256': digest, 'peak_gpu_memory_bytes': None})
                    raise subprocess.CalledProcessError(result.returncode, command)
        save(directory / 'driver_run_record.json', {'status': 'complete', 'unit': name,
             'git_commit': self.commit, 'exit_codes': codes,
             'wall_seconds': time.perf_counter() - start, 'manifest_sha256': digest,
             'peak_gpu_memory_bytes': None, 'peak_memory_note': 'CPU driver; GPU peaks remain in training run_record',
             'outputs': self.output_hashes(directory)})
        done.write_text(digest + '\n', encoding='utf-8', newline='\n')
        self.events.append({'unit': name, 'status': 'complete'})
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
            guard_error = exc
            status = 'STOPPED: PROTECTED_OUTPUT_CHANGED'
            error = exc
        (self.out / ('STATUS_' + stage)).write_text(status + '\n', encoding='utf-8', newline='\n')
        digest = ['R19 ' + stage + ' post-hoc validation', 'STATUS ' + status,
                  'git_commit ' + self.commit, 'error ' + (str(error) if error else 'none'),
                  'events ' + json.dumps(self.events, ensure_ascii=False)]
        digest.extend('UNIT ' + str(p.relative_to(self.out)) + ' sha256=' + sha256(p)
                      for p in sorted(self.out.rglob('driver_run_record.json')))
        for p in sorted((self.out / 'reports' / stage).glob('*.md')) if (self.out / 'reports' / stage).exists() else []:
            if p.suffix == '.md':
                digest += ['REPORT ' + str(p.relative_to(self.out)), p.read_text(encoding='utf-8')]
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


def eval_unit(runner, spec, stage, dataset, seed, fit, *, methods, mixes=None,
              fixed=True, lam_curve=False, extra=(), regress=False, cams=None):
    cfg = spec['datasets']['rgbnt' if dataset == 'rgbnt_sparse' else 'rgbnt_subset' if dataset == 'rgbnt_nested' else dataset][str(seed)]
    directory = runner.out / stage.lower() / dataset / f's{seed}_f{fit}'
    command, inputs = evaluation_commands(runner, cfg, seed, fit, directory, methods,
                        mixes or MIXES, spec['assign_seeds'], spec.get('grid', 'v3'), fixed,
                        lam_curve, extra, regress, cams)
    if '--assign_nested_from' in extra:
        inputs.append(extra[list(extra).index('--assign_nested_from') + 1])
    runner.unit(f'{stage}/{dataset}/s{seed}_f{fit}', directory, inputs, [command],
                dict(stage=stage, dataset=dataset, seed=seed, fit=fit, methods=methods,
                     mixes=mixes or MIXES, assign_seeds=spec['assign_seeds'], post_hoc=True))
    return directory


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


def judge(runner, stage):
    directory = runner.out / 'reports' / stage
    inputs = [p for base in ('s1', 's2', 'mlm1/eval') for p in (runner.out / base).rglob('*')
              if p.is_file() and (p.name.endswith(('_cells.csv', '_diag_cells.csv', '_selection.json', '_perq_ext.pkl', '_run_record.json')))]
    command = [sys.executable, '-B', runner.code / 'tools/p4/r19/r19_judge.py', '--stage', stage, '--root', runner.out]
    runner.unit('judge_' + stage, directory, inputs, [command], {'post_hoc': True})


def s1(runner, spec):
    for dataset in ('rgbt', 'rgbnt', 'rgbnt_subset'):
        for seed in spec['seeds']:
            for fit in spec['fits']:
                eval_unit(runner, spec, 'S1', dataset, seed, fit, methods=METHODS1, lam_curve=True)
    judge(runner, 'S1')


def s2(runner, spec):
    for seed in spec['seeds']:
        full = spec['datasets']['rgbnt'][str(seed)]
        subset = spec['datasets']['rgbnt_subset'][str(seed)]
        directory = runner.out / 's2/cameras' / f's{seed}'
        target = directory / 'subset.npz'
        cmd = [sys.executable, '-B', str(Path(__file__).resolve()), 'subset-cams',
               '--full', full['test'], '--subset', subset['test'], '--cams', full['cams'], '--output', str(target)]
        runner.unit(f'S2/cameras/s{seed}', directory, [full['test'], subset['test'], full['cams']], [cmd])
        for fit in spec['fits']:
            for dataset, cams in [('rgbnt', full['cams']), ('rgbnt_subset', target)]:
                eval_unit(runner, spec, 'S2', dataset, seed, fit, methods=METHODS2,
                          mixes=MIXES + ['Mcam1', 'Mcam2'], cams=cams)
            eval_unit(runner, spec, 'S2', 'rgbnt_nested', seed, fit, methods=METHODS_NESTED,
                      extra=['--assign_nested_from', full['test']])
            eval_unit(runner, spec, 'S2', 'rgbnt_sparse', seed, fit, methods=METHODS_NESTED,
                      extra=['--max_per_id', '3', '--max_per_id_seed', '9101'])
    judge(runner, 'S2')


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


def s4(runner, spec):
    # A fresh spec is emitted by S3; the original input spec is never mutated.
    generated = runner.out / 'mlm1/evaluation_inputs.json'
    if not generated.is_file():
        raise FileNotFoundError('S4 requires completed S3: ' + str(generated))
    fresh = json.loads(generated.read_text(encoding='utf-8'))
    for dataset in ('rgbt', 'rgbnt', 'rgbnt_subset'):
        for seed in fresh['seeds']:
            for fit in fresh['fits']:
                # eval_unit's output stage can be a path relative to p4_r19_out.
                eval_unit(runner, fresh, 'mlm1/eval', dataset, seed, fit, methods=METHODS4,
                          fixed=False, extra=['--nogate_rmax'])
    judge(runner, 'S4')


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
    a = ap.parse_args(argv)
    code = Path(a.repo) / 'IRRA-main' if a.repo else CODE
    out = Path(a.root).resolve() / 'p4_r19_out'
    out.mkdir(parents=True, exist_ok=True)
    lock = out / '.driver.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('R19 driver is already running or was interrupted; inspect lock before proceeding')
    os.close(fd)
    error = None
    runner = None
    try:
        runner = Runner(a.root, code, a.synthetic)
        spec = json.loads(Path(a.spec).read_text(encoding='utf-8')) if a.spec else default_specs(a.root)
        if a.synthetic and a.stage != 'S0':
            raise ValueError('Synthetic shell rehearsal is limited to S0')
        save(runner.out / ('checks/inputs_' + a.stage + '.json'), spec)
        {'S0': s0, 'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4}[a.stage](runner, spec)
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            if runner is not None:
                runner.finish(a.stage, 'COMPLETE' if error is None else 'STOPPED', error)
            else:
                (out / ('STATUS_' + a.stage)).write_text('STOPPED\n', encoding='utf-8', newline='\n')
                (out / ('DIGEST19_' + a.stage + '.txt')).write_text('R19 ' + a.stage + '\nSTOPPED\n' + str(error) + '\n', encoding='utf-8', newline='\n')
        finally:
            lock.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
