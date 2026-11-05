"""Post-hoc R19 judgments from explicit per-query records, never from identities at inference.

The thresholds, groups, 4000 identity-cluster bootstrap draws and seed 7 are
fixed here. Curve rows are descriptive; all judgments use named selected methods.
Only trusted local perq_ext pickles produced by the frozen pipeline are inputs.
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np


CODE = Path(__file__).resolve().parents[3]
TRANSFER = CODE / 'tools/p4/transfer'
SEEDS = (101, 102, 103)
FITS = (7001, 7002, 7003)
ASSIGN_SEEDS = (7031, 7032, 7033, 7034, 7035)
A = ('M4', 'M6c', 'M1g', 'M4c60', 'M9')
B = ('M5e', 'M5f', 'M4id')
GROUPS = {'A': A, 'B': B}
METRICS = ('AP', 'R1', 'R5', 'R10', 'mINP')
POOL = ('KR', 'RVN', 'QE-DBA', 'KRir', 'RVNir', 'QEir', 'GP-same')
RULES = {
    'V1a': 'All three seeds: A mAP(QE+M - QE+CtxOnly) < 0.3; abs(B difference) < 0.1.',
    'V1b': 'V1a false; all seeds: abs(B difference) < 0.1 and A difference >= 0.5.',
    'V2': 'All three seeds, group A relative to QE-DBA: gain_lambda >= 0.70*gain_QE+M; loss_lambda <= 0.50*loss_QE+M, loss=max(0,-dR10). If main loss=0, selected lambda loss must also be 0. Relaxed lambda2 is descriptive and not part of V2.',
    'V3': 'All three seed A mAP lower CI bounds of QE+M-xcam-refit - QE-DBA > 0.',
    'V4': 'gain_S=three-seed mean group-A mAP(QE+M-QE-DBA). V4a: abs(gain_nested-gain_full)<=0.5. V4b: flag repeated-observation dependence if gain_sparse<0.50*gain_full, only when gain_full>0; otherwise report without a flag.',
    'V6': 'Both datasets, all three seeds: A lower mAP bound(A-2f-ctxn - FeatImp-r) > 0; abs(B difference) < 0.3.',
    'bootstrap': 'Per query: mean over the 15 (fit, assignment seed) cells within each mixture, then equal mean over group mixtures; identity-cluster resampling 4000 times, seed 7; unrounded values.',
    'V5': 'Post-hoc only. RGBT uses unchanged frozen judge C1-C5; RGBNT uses frozen E1/E2 point forms, E3/C3, and frozen E4 outcome function. Not a new preregistered validity verdict.',
}
STAGE_PATHS = {
    'S1': ('s1', ('rgbt', 'rgbnt', 'rgbnt_subset')),
    'S2': ('s2', ('rgbnt', 'rgbnt_subset', 'rgbnt_nested', 'rgbnt_sparse')),
    'S4': ('mlm1/eval', ('rgbt', 'rgbnt', 'rgbnt_subset')),
}
_BOOT = None
_VERDICT = None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def one(directory, pattern):
    hits = sorted(Path(directory).glob(pattern))
    if pattern == 'r12_*_cells.csv':
        hits = [p for p in hits if not p.name.endswith('_diag_cells.csv')]
    if len(hits) != 1:
        raise ValueError(f'{directory}: expected one {pattern}, got {len(hits)}')
    return hits[0]


def extract_function(path, name, namespace):
    """Compile the exact function AST without running the frozen CLI module."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def frozen_boot(per_query, qid):
    global _BOOT
    if _BOOT is None:
        _BOOT = extract_function(TRANSFER / 'r17_judge.py', 'boot_ci',
                                 {'np': np, 'a': SimpleNamespace(boot=4000)})
    return _BOOT(np.asarray(per_query, dtype=np.float64), np.asarray(qid), seed=7)


def frozen_verdict():
    global _VERDICT
    if _VERDICT is None:
        path = CODE / 'tools/p4/r18/r18_verdict.py'
        spec = importlib.util.spec_from_file_location('_r19_frozen_verdict', path)
        _VERDICT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_VERDICT)
    return _VERDICT


def metric_values(record, metric):
    values = np.asarray(record[metric], dtype=np.float64)
    if metric in ('R1', 'R5', 'R10'):
        values = values * 100.0
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f'Invalid per-query {metric} array')
    return values


class SeedData:
    """Strict 3x5 alignment; aggregate cells are checked against their own query records."""
    def __init__(self, root, seed, evalset='test'):
        self.root, self.seed, self.evalset = Path(root), int(seed), evalset
        self.units = {}
        self.inputs = {}
        self._means = {}
        self._pairs = {}
        self.selections = []
        self.diagnostics = []
        self.cells = []
        self.curve_cells = []
        qref = idref = None
        for fit in FITS:
            directory = self.root / f's{seed}_f{fit}'
            cellpath = one(directory, 'r12_*_cells.csv')
            diagpath = one(directory, 'r12_*_diag_cells.csv')
            selpath = one(directory, 'r12_*_selection.json')
            perpath = one(directory, 'r17_*_perq_ext.pkl')
            for path in (cellpath, diagpath, selpath, perpath):
                self.inputs[str(path)] = sha256(path)
            with cellpath.open(encoding='utf-8-sig', newline='') as f:
                cells = list(csv.DictReader(f))
            with diagpath.open(encoding='utf-8-sig', newline='') as f:
                diag = list(csv.DictReader(f))
            sel = json.loads(selpath.read_text(encoding='utf-8'))
            with perpath.open('rb') as f:
                envelope = pickle.load(f)
            records = envelope['records']
            meta = envelope.get('meta', {})
            if meta and sorted(map(int, meta.get('assign_seeds', ASSIGN_SEEDS))) != list(ASSIGN_SEEDS):
                raise ValueError(f'{perpath}: expected exactly the five assignment seeds')
            filtered = {}
            seen = set()
            for row in cells:
                if row['evalset'] != evalset:
                    continue
                mx, method, aseed = row['mixture'], row['method'], int(row['aseed'])
                if int(row['split']) != fit or int(row['fold']) != 0 or aseed not in ASSIGN_SEEDS:
                    raise ValueError(f'Unexpected fitting/assignment unit in {cellpath}')
                key = (mx, aseed, method)
                if key in seen:
                    raise ValueError(f'Duplicate cells key {key} in {cellpath}')
                seen.add(key)
                if '@' in method:
                    self.curve_cells.append(row)
                    continue  # curves intentionally carry no perq_ext records
                record = records.get((fit, evalset, mx, aseed, method))
                if record is None:
                    raise ValueError(f'Missing perq_ext record: {directory}, {key}')
                q, qid = np.asarray(record['q']), np.asarray(record['qid'])
                if q.ndim != 1 or not len(q) or len(q) != len(set(q.tolist())) or q.shape != qid.shape:
                    raise ValueError(f'Invalid query alignment in {perpath}')
                if qref is None:
                    qref, idref = q.copy(), qid.copy()
                elif not np.array_equal(qref, q) or not np.array_equal(idref, qid):
                    raise ValueError(f'Query/identity alignment differs at {perpath}')
                for metric in METRICS:
                    values = metric_values(record, metric)
                    if values.shape != q.shape:
                        raise ValueError(f'Query metric length differs at {perpath}')
                    column = 'mAP' if metric == 'AP' else metric
                    if column not in row or not math.isclose(float(row[column]), float(values.mean()), abs_tol=1e-6, rel_tol=0):
                        raise ValueError(f'Cell/per-query disagreement: {cellpath}, {key}, {column}')
                filtered[key] = record
                self.cells.append(row)
            if not filtered:
                raise ValueError(f'No records for evalset {evalset}: {directory}')
            self.units[fit] = filtered
            self.selections.extend(dict(x, fitting_seed=fit) for x in sel)
            self.diagnostics.extend(dict(x, fitting_seed=fit) for x in diag if x.get('evalset') == evalset)
        self.q, self.qid = qref, idref
        self.methods = set.intersection(*[{k[2] for k in records} for records in self.units.values()])

    def means(self, method, mixes, metric):
        """Preserve the frozen per_query addition order: mixture, fit, assignment."""
        key = (method, tuple(mixes), metric)
        if key in self._means:
            return self._means[key]
        acc = None
        n = 0
        for mixture in mixes:
            for fit in FITS:
                for aseed in ASSIGN_SEEDS:
                    rowkey = (mixture, aseed, method)
                    if rowkey not in self.units[fit]:
                        raise ValueError(f'Missing a 15-cell entry: seed={self.seed}, fit={fit}, {rowkey}')
                    value = metric_values(self.units[fit][rowkey], metric)
                    if acc is None:
                        acc = np.zeros_like(value)
                    acc += value
                    n += 1
        if n != 15 * len(mixes):
            raise AssertionError('Not exactly 15 cells per mixture')
        self._means[key] = acc / n
        return self._means[key]

    def pair(self, left, right, group):
        key = (left, right, group)
        if key in self._pairs:
            return self._pairs[key]
        mixes = GROUPS.get(group, (group,))
        metrics = {}
        for metric in METRICS:
            per = self.means(left, mixes, metric) - self.means(right, mixes, metric)
            point, low, high = frozen_boot(per, self.qid)
            metrics[metric] = {'mean': point, 'lo': low, 'hi': high}
        ap, r1 = metrics['AP'], metrics['R1']
        value = dict(dmAP=ap['mean'], lo=ap['lo'], hi=ap['hi'], dR1=r1['mean'],
                     r_lo=r1['lo'], r_hi=r1['hi'], n_q=len(self.q), n_id=len(np.unique(self.qid)),
                     n_cells=15 * len(mixes), metrics=metrics)
        self._pairs[key] = value
        return value

    def absolute(self, method, group, metric='AP'):
        return float(self.means(method, GROUPS.get(group, (group,)), metric).mean())

    def diagnostic_means(self):
        groups = {}
        for row in self.diagnostics:
            for group, mixes in GROUPS.items():
                if row['mixture'] not in mixes:
                    continue
                key = row['method'] + '|' + group
                dest = groups.setdefault(key, {})
                for column, value in row.items():
                    if column in ('split', 'fold', 'aseed', 'fitting_seed'):
                        continue
                    try:
                        number = float(value)
                    except (ValueError, TypeError):
                        continue
                    if math.isfinite(number):
                        dest.setdefault(column, []).append(number)
        return {k: {c: float(np.mean(v)) for c, v in columns.items()} for k, columns in groups.items()}

    def curve_means(self):
        """Descriptive cell means only: curve methods deliberately have no query CI."""
        result = {}
        for method in sorted({r['method'] for r in self.curve_cells}):
            result[method] = {}
            for group,mixes in GROUPS.items():
                per_mix = []
                for mixture in mixes:
                    rows=[r for r in self.curve_cells if r['method']==method and r['mixture']==mixture]
                    if len(rows)!=15:
                        raise ValueError(f'Curve {method} has {len(rows)} cells for {mixture}; expected 15')
                    per_mix.append({metric:float(np.mean([float(r['mAP' if metric=='AP' else metric]) for r in rows])) for metric in METRICS})
                result[method][group]={metric:float(np.mean([r[metric] for r in per_mix])) for metric in METRICS}
        return result


def s1_rules(data):
    """All seed tests use unrounded paired point estimates; V6 pools no seeds."""
    pa = [data[s].pair('QE+M', 'QE+CtxOnly', 'A')['dmAP'] for s in SEEDS]
    pb = [data[s].pair('QE+M', 'QE+CtxOnly', 'B')['dmAP'] for s in SEEDS]
    v1a = all(a < 0.3 and abs(b) < 0.1 for a, b in zip(pa, pb))
    v1b = not v1a and all(a >= 0.5 and abs(b) < 0.1 for a, b in zip(pa, pb))
    gain = [data[s].pair('QE+M', 'QE-DBA', 'A')['dmAP'] for s in SEEDS]
    variants = {}
    for method in ('QE+M-lam','QE+M-lam2'):
        rows = []
        for i, seed in enumerate(SEEDS):
            base_pair = data[seed].pair('QE+M', 'QE-DBA', 'A')
            lam_pair = data[seed].pair(method, 'QE-DBA', 'A')
            base = base_pair['metrics']['R10']['mean']
            lam = lam_pair['metrics']['R10']['mean']
            loss, lam_loss = max(0.0, -base), max(0.0, -lam)
            point=lam_pair['dmAP']
            rows.append(dict(seed=seed,main_gain=gain[i],lambda_gain=point,
                             main_gain_ci=base_pair['metrics'].get('AP'),lambda_gain_ci=lam_pair['metrics'].get('AP'),
                             main_R10_loss=loss,lambda_R10_loss=lam_loss,
                             main_R10_difference_ci=base_pair['metrics']['R10'],lambda_R10_difference_ci=lam_pair['metrics']['R10'],
                             zero_main_loss_requires_zero_lambda_loss=loss==0.0,
                             holds=point >= 0.70 * gain[i] and lam_loss <= 0.50 * loss))
        variants[method] = dict(per_seed=rows, holds=all(r['holds'] for r in rows))
    v6 = all(data[s].pair('A-2f-ctxn', 'FeatImp-r', 'A')['lo'] > 0 and
             abs(data[s].pair('A-2f-ctxn', 'FeatImp-r', 'B')['dmAP']) < 0.3 for s in SEEDS)
    return dict(V1a=v1a,V1b=v1b,V1_points=dict(A=pa,B=pb),
                V2=variants['QE+M-lam'],V2_lam2_descriptive=variants['QE+M-lam2'],V6_dataset=v6)


def v4_rules(gains, intervals=None):
    if any(len(values)!=3 or not np.isfinite(values).all() for values in gains.values()):
        raise ValueError('V4 requires three finite base-seed gains per scope')
    means={name:float(np.mean(values)) for name,values in gains.items()}
    difference=means['rgbnt_nested']-means['rgbnt']
    positive=means['rgbnt']>0
    return dict(V4a=abs(difference)<=0.5,nested_minus_full=difference,absolute_difference=abs(difference),
                per_seed_gains=gains,per_seed_gain_ci=intervals or {},mean_gains=means,
                V4b_repeated_observation_dependence=means['rgbnt_sparse']<0.50*means['rgbnt'] if positive else None,
                V4b_applicable=positive,
                V4b_note='Full gain > 0: apply 50% criterion' if positive else 'Full gain <= 0: report values without dependence flag')


def frozen_claims(data):
    """Execute the original C1-C5 AST body, with paired values from explicit records."""
    path = TRANSFER / 'judge_test.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    outer = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'spec' and
                 any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'have' for t in x.targets) for x in ast.walk(n)))
    block = next(n for n in outer.body if isinstance(n, ast.If) and
                 isinstance(n.test, ast.UnaryOp) and isinstance(n.test.operand, ast.Attribute) and n.test.operand.attr == 'no_claims')
    # Everything before the first print is the exact frozen numerical C1-C5 logic.
    body = []
    for node in block.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            break
        body.append(node)
    pairs = {}
    pairs_to_preload = [('QE+M','QE-DBA'), ('A-2f-ctxn','FeatImp-g'), ('QE+M','QE+CAET')]
    for left,right in pairs_to_preload:
        if left in data.methods and right in data.methods:
            for group in GROUPS:
                pairs[(left,right,group)] = data.pair(left,right,group)
    def gm(method, mixes):
        group = 'A' if tuple(mixes) == A else 'B'
        return data.absolute(method,group) - data.absolute('B*',group)
    def group_pair(_pq,left,right,mixes,_ids,nboot):
        if nboot != 4000:
            raise AssertionError('Frozen draw count changed')
        return data.pair(left,right,'A' if tuple(mixes) == A else 'B')
    namespace = dict(C={},P=pairs,have=data.methods,gm=gm,POOL=POOL,A=A,B=B,OURS='QE+M',FRONT='QE-DBA',MAIN='A-2f-ctxn',
                     group_pair=group_pair,pq=None,ids=None,a=SimpleNamespace(boot=4000),fmt=lambda p:str(p),np=np)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
    return dict(claims={k:v['holds'] for k,v in namespace['C'].items()},details=namespace['C'],
                strongest=dict(A=namespace['zA'],B=namespace['zB']),source_sha256=sha256(path))


def frozen_e_forms(data, claims):
    """Use the frozen E1/E2 AST assignments and E4 function; no R18 protocol verdict."""
    verdict = frozen_verdict()
    source = CODE / 'tools/p4/r18/r18_verdict.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'analyze_scope')
    nodes = []
    for node in ast.walk(function):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='S' and
                                             isinstance(t.slice,ast.Constant) and t.slice.value in ('E1','E2') for t in node.targets):
            nodes.append(node)
    if len(nodes) != 2:
        raise ValueError('Frozen E1/E2 source shape changed')
    per = {}
    for seed in SEEDS:
        pa = data[seed].pair('QE+M','QE-DBA','A')
        pb = data[seed].pair('QE+M','QE-DBA','B')
        namespace = dict(S={},pa=pa,pb=pb,E1_GAIN=verdict.E1_GAIN,TOL=verdict.TOL)
        exec(compile(ast.Module(body=sorted(nodes,key=lambda n:n.lineno),type_ignores=[]),str(source),'exec'),namespace)
        per[seed] = dict(**namespace['S'],E3=claims[seed]['claims']['C3'])
    def trip(seed,group,metric):
        p = data[seed].pair('QE+M','QE+Const',group)['metrics'][metric]
        return p['mean'],p['lo'],p['hi']
    outcome,routes,trace = verdict.e4_outcome(trip)
    e2 = {metric:verdict.tol_wording([data[s].pair('QE+M','QE-DBA','B')['metrics'][metric]['mean'] for s in SEEDS],
                                   [data[s].pair('QE+M','QE-DBA','B')['metrics'][metric]['lo'] for s in SEEDS]) for metric in ('AP','R1')}
    return dict(post_hoc=True,form_only=True,per_seed=per,E1=all(v['E1'] for v in per.values()),
                E2=all(v['E2'] for v in per.values()),E3=all(v['E3'] for v in per.values()),
                E4=outcome,E4_routes=routes,E4_trace=trace,E2_wording=e2,source_sha256=sha256(source))


def summary_dataset(data, wanted_pairs):
    result = {}
    for seed, d in data.items():
        absolute = {m:{g:{metric: d.absolute(m,g,metric) for metric in METRICS} for g in GROUPS}
                    for m in sorted(d.methods)}
        pairs = {}
        for left,right in wanted_pairs:
            if left not in d.methods or right not in d.methods:
                continue
            for group in GROUPS:
                pairs[left+'|'+right+'|'+group] = d.pair(left,right,group)
        result[seed] = dict(absolute=absolute,pairs=pairs,selection=d.selections,
                            diagnostics=d.diagnostic_means(),curve_cell_means=d.curve_means(),
                            n_q=len(d.q),n_id=len(np.unique(d.qid)))
    return result


def stage_report(root, stage):
    if stage not in STAGE_PATHS:
        raise ValueError('Judge supports S1, S2, S4')
    relative,datasets = STAGE_PATHS[stage]
    loaded = {}
    for dataset in datasets:
        evalset = 'k3' if dataset == 'rgbnt_sparse' else 'test'
        loaded[dataset] = {seed:SeedData(Path(root)/relative/dataset,seed,evalset) for seed in SEEDS}
    report = dict(stage=stage,post_hoc=True,rules=RULES,groups=GROUPS,seeds=SEEDS,fits=FITS,assign_seeds=ASSIGN_SEEDS,
                  bootstrap=dict(draws=4000,seed=7,cluster='raw identity',aggregation='15 cells per mixture then equal group mean'),
                  datasets={},verdicts={},inputs={})
    for data in loaded.values():
        for d in data.values():
            report['inputs'].update(d.inputs)
    if stage in ('S1','S4'):
        pairlist = [('QE+M','QE-DBA'),('QE+M','QE+CtxOnly'),('QE+M','QE+SRC2f'),('QE+M','QE+SRC2f+kap'),
                    ('QE+M','QE+SRC2f+pI'),('QE+M-lam','QE-DBA'),('QE+M-lam2','QE-DBA'),('QE+M','QE+Const'),
                    ('A-2f-ctxn','FeatImp-r'),('A-2f-ctxn','FeatImp-g'),('QE+M','QE+CAET')]
        if stage=='S4':
            pairlist += [('QE-DBA','B*'),('QE+Const','QE-DBA'),('QE+M','B*')]
        for name,data in loaded.items():
            report['datasets'][name] = summary_dataset(data,pairlist)
        if stage == 'S1':
            per = {name:s1_rules(data) for name,data in loaded.items()}
            report['verdicts'] = dict(per_dataset=per,V6=all(per[n]['V6_dataset'] for n in ('rgbt','rgbnt')))
        else:
            claims = {name:{seed:frozen_claims(d) for seed,d in data.items()} for name,data in loaded.items()}
            rgbt = claims['rgbt']
            report['V5_post_hoc'] = dict(rgbt=dict(per_seed=rgbt,all_seeds={key:all(rgbt[s]['claims'][key] for s in SEEDS) for key in ('C1','C2','C3','C4','C5')}),
                                          rgbnt=frozen_e_forms(loaded['rgbnt'],claims['rgbnt']),
                                          rgbnt_subset=frozen_e_forms(loaded['rgbnt_subset'],claims['rgbnt_subset']))
            report['verdicts'] = dict(V5='post-hoc; see V5_post_hoc; original archived verdicts unchanged')
    else:
        pairlist = [('QE+M','QE-DBA'),('QE+M-xcam','QE-DBA'),('QE+M-xcam-refit','QE-DBA'),
                    ('QE+M-xcam-refit','QE+M'),('QE+M-xcam-refit','QE+M-xcam'),('QE+M','QE+Const')]
        for name,data in loaded.items():
            report['datasets'][name] = summary_dataset(data,pairlist)
        v3 = {name:all(data[s].pair('QE+M-xcam-refit','QE-DBA','A')['lo']>0 for s in SEEDS)
              for name,data in loaded.items() if name in ('rgbnt','rgbnt_subset')}
        gains = {name:[data[s].pair('QE+M','QE-DBA','A')['dmAP'] for s in SEEDS] for name,data in loaded.items()}
        intervals={name:{seed:data[seed].pair('QE+M','QE-DBA','A')['metrics']['AP'] for seed in SEEDS} for name,data in loaded.items()}
        report['verdicts'] = dict(V3=v3,V4=v4_rules(gains,intervals))
    report['code'] = {str(Path(__file__).relative_to(CODE)):sha256(__file__),
                      'tools/p4/transfer/r17_judge.py':sha256(TRANSFER/'r17_judge.py'),
                      'tools/p4/transfer/judge_test.py':sha256(TRANSFER/'judge_test.py'),
                      'tools/p4/r18/r18_verdict.py':sha256(CODE/'tools/p4/r18/r18_verdict.py')}
    return report


def markdown(report):
    lines = ['# R19 post-hoc validation', '', 'All judgments are post-hoc; original archived verdicts are not rewritten.', '',
             'Fixed rules:', '', *[f'- **{key}**: {value}' for key,value in report['rules'].items()], '']
    lines += ['## Table V1: gate-input comparisons', '', '| dataset | seed | comparison | A mAP [95% CI] | B mAP [95% CI] |',
              '|---|---|---|---|---|']
    gate_methods = ('QE+CtxOnly','QE+SRC2f','QE+SRC2f+kap','QE+SRC2f+pI')
    def ci(pair,metric='AP'):
        v=pair['metrics'][metric]
        return f"{v['mean']:.6f} [{v['lo']:.6f}, {v['hi']:.6f}]"
    for dataset,seeds in report['datasets'].items():
        for seed,row in seeds.items():
            for method in gate_methods:
                a=row['pairs'].get('QE+M|'+method+'|A');b=row['pairs'].get('QE+M|'+method+'|B')
                if a and b:lines.append(f'| {dataset} | {seed} | QE+M - {method} | {ci(a)} | {ci(b)} |')
    lines += ['', '## Table V2: selected lambda trade-off', '', '| dataset | seed | method - QE-DBA | A mAP [95% CI] | A R10 [95% CI] | B R10 [95% CI] |', '|---|---|---|---|---|---|']
    for dataset,seeds in report['datasets'].items():
        for seed,row in seeds.items():
            for method in ('QE+M','QE+M-lam','QE+M-lam2'):
                a=row['pairs'].get(method+'|QE-DBA|A');b=row['pairs'].get(method+'|QE-DBA|B')
                if a and b:lines.append(f'| {dataset} | {seed} | {method} | {ci(a)} | {ci(a,"R10")} | {ci(b,"R10")} |')
    lines += ['', '## Table V3: cross-camera refit', '', '| dataset | seed | comparison | A mAP [95% CI] | B mAP [95% CI] |', '|---|---|---|---|---|']
    for dataset,seeds in report['datasets'].items():
        for seed,row in seeds.items():
            for method in ('QE+M-xcam','QE+M-xcam-refit'):
                a=row['pairs'].get(method+'|QE-DBA|A');b=row['pairs'].get(method+'|QE-DBA|B')
                if a and b:lines.append(f'| {dataset} | {seed} | {method} - QE-DBA | {ci(a)} | {ci(b)} |')
    lines += ['', '## Table V4: nested/sparse galleries and feature control', '', '| dataset | seed | comparison | A mAP [95% CI] | B mAP [95% CI] |', '|---|---|---|---|---|']
    for dataset,seeds in report['datasets'].items():
        for seed,row in seeds.items():
            methods = [('QE+M','QE-DBA'),('A-2f-ctxn','FeatImp-r')]
            for left,right in methods:
                a=row['pairs'].get(left+'|'+right+'|A');b=row['pairs'].get(left+'|'+right+'|B')
                if a and b:lines.append(f'| {dataset} | {seed} | {left} - {right} | {ci(a)} | {ci(b)} |')
    lines += ['', '## Mechanical outcomes', '', '```json',json.dumps(report['verdicts'],ensure_ascii=False,indent=2),'```']
    if 'V5_post_hoc' in report:
        lines += ['', '## S4 decomposition: front-end / constant transfer / learned gate', '',
                  '| dataset | seed | group | QE-DBA - B* | QE+Const - QE-DBA | QE+M - QE+Const | QE+M - B* |',
                  '|---|---|---|---|---|---|---|']
        for dataset,seeds in report['datasets'].items():
            for seed,row in seeds.items():
                for group in GROUPS:
                    parts=[row['pairs'][left+'|'+right+'|'+group] for left,right in
                           [('QE-DBA','B*'),('QE+Const','QE-DBA'),('QE+M','QE+Const'),('QE+M','B*')]]
                    if not math.isclose(sum(p['dmAP'] for p in parts[:3]),parts[3]['dmAP'],rel_tol=0,abs_tol=1e-10):
                        raise ValueError('S4 decomposition does not sum to total gain')
                    lines.append(f'| {dataset} | {seed} | {group} | '+ ' | '.join(ci(p) for p in parts)+' |')
        lines += ['', '## V5 post-hoc frozen forms', '', '```json',json.dumps(report['V5_post_hoc'],ensure_ascii=False,indent=2),'```']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=tuple(STAGE_PATHS))
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--out_dir',type=Path,help='Default: ROOT/reports/STAGE; existing reports are never overwritten')
    args=parser.parse_args(argv)
    out=args.out_dir or args.root/'reports'/args.stage
    names=['report.json','tables.md']
    if args.stage=='S4':
        names += [f'V5_{dataset}_posthoc.json' for dataset in ('rgbt','rgbnt','rgbnt_subset')]
    if any((out/name).exists() for name in names):
        raise FileExistsError(f'Refusing to overwrite reports: {out}')
    report=stage_report(args.root,args.stage)
    out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    for dataset,value in report.get('V5_post_hoc',{}).items():
        path=out/f'V5_{dataset}_posthoc.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(dict(post_hoc=True,dataset=dataset,stage='S4',result=value),
                                   ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    text=markdown(report)
    (out/'tables.md').write_text(text,encoding='utf-8',newline='\n')
    print(text,end='')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
