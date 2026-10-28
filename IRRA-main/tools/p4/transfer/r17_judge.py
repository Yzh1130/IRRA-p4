"""Round 17: report of the post-hoc, non-blind constant-gain control on the official test split (project doc "论文4_第十七轮_独立裁决与收尾计划").
The round-15 verdicts (C1-C5) are not touched. This script only reads the outputs of r17_const.py and, for the reproduction check, the
round-15 cells; it computes no new score.

Inputs per base seed: <r17_root>/s<seed>_f<fit>/ (three fits), each written by r17_const.py with --perq_ext.
Sections:
  [structure]     every (fit, mixture, assignment seed, method) cell exactly once; per-query records keyed explicitly
  [reproduction]  B*, QE-DBA, QE+M, QE+CAET, QE+SRC2f, QE+EM against the round-15 cells (same fit, mixture, assignment seed)
  [selection]     chosen constant gain per fit and the full development table (mean dmAP over the five types, worst-type dR1)
  [table A]       absolute metrics (cell means) per seed, groups A and B, and every B mixture
  [table B]       paired differences, per query averaged over the 15 cells and the group's mixtures, 95% identity-cluster bootstrap
                  (4000 resamples, seed 7, as judge_test.py); metrics AP, R1, R5, R10, mINP
  [curve]         report-only: QE+Const at every grid value (cell means); never used to choose the gain
  [locality]      top-1 changes relative to QE-DBA, split by the availability of QE-DBA's top-1 item
  [harm/help]     rank-1 flips relative to QE-DBA with explicit denominators
  [top-5]         R5 losses of QE+M relative to QE-DBA and the composition of the new top-5 (IR-only negatives)
  [timing]        stage timing from --timing (medians over fits)
usage: python -B r17_judge.py --r17_root DIR --r15_root DIR --seeds 101 102 103 --out JSON [--devsens_root DIR]
"""
import argparse, glob, json, os, pickle, re
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--r17_root', required=True); ap.add_argument('--r15_root', required=True)
ap.add_argument('--seeds', type=int, nargs='+', default=[101, 102, 103]); ap.add_argument('--fits', type=int, nargs='+', default=[7001, 7002, 7003])
ap.add_argument('--boot', type=int, default=4000); ap.add_argument('--out', required=True); ap.add_argument('--devsens_root', default=None)
ap.add_argument('--tol', type=float, default=1e-6)
a = ap.parse_args()

A_GRP = ['M4', 'M6c', 'M1g', 'M4c60', 'M9']; B_GRP = ['M5e', 'M5f', 'M4id']
SHARED = ['B*', 'QE-DBA', 'QE+M', 'QE+CAET', 'QE+SRC2f', 'QE+EM']
MAIN = ['QE-DBA', 'QE+M', 'QE+Const', 'QE+NoGate', 'QE+SRC2f', 'QE+CAET', 'QE+EM']
PAIRS = [('QE+M', 'QE+Const'), ('QE+M', 'QE+NoGate'), ('QE+Const', 'QE-DBA'), ('QE+M', 'QE-DBA'), ('QE+NoGate', 'QE-DBA'),
         ('QE+M', 'QE+SRC2f'), ('QE+M', 'QE+CAET'), ('QE+M', 'QE+EM')]
METRICS = ['AP', 'R1', 'R5', 'R10', 'mINP']
CELLCOL = {'AP': 'mAP', 'R1': 'R1', 'R5': 'R5', 'R10': 'R10', 'mINP': 'mINP'}
REPORT = {}; PROBLEMS = []


def one(pattern):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1: raise SystemExit(f'expected exactly one file for {pattern}, found {len(hits)}')
    return hits[0]


def prefix(d):
    """r12_<tag>_ prefix of the single run in directory d (found through its summary file; 'cells' alone would also match diag_cells)"""
    return one(os.path.join(d, 'r12_*_summary.csv'))[:-len('summary.csv')]


def load_fit(seed, fit):
    d = os.path.join(a.r17_root, f's{seed}_f{fit}')
    cells = pd.read_csv(prefix(d) + 'cells.csv'); cells = cells[cells['evalset'].astype(str) == 'test'].copy()
    sel = json.load(open(prefix(d) + 'selection.json'))
    rec = json.load(open(prefix(d) + 'run_record.json'))
    log = json.load(open(one(os.path.join(d, 'r17_*_r17log.json'))))
    pq = pickle.load(open(one(os.path.join(d, 'r17_*_perq_ext.pkl')), 'rb'))
    return dict(dir=d, cells=cells, sel=sel, rec=rec, log=log, pq=pq)


def boot_ci(per_q, qid, seed=7):
    u = np.unique(qid); idx = [np.where(qid == v)[0] for v in u]
    s = np.array([per_q[i].sum() for i in idx]); c = np.array([len(i) for i in idx])
    rb = np.random.default_rng(seed); b = np.empty(a.boot)
    for k in range(a.boot):
        x = rb.integers(0, len(u), len(u)); b[k] = s[x].sum() / c[x].sum()
    return float(per_q.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def star(lo, hi): return '*' if (lo > 0 or hi < 0) else ' '


out_lines = []
def P(*x): out_lines.append(' '.join(str(v) for v in x))


P('################ 论文4 第十七轮 常数迁移对照（事后补充对照，同一测试集，非盲；C1-C5 原判定不变）################')
DATA = {}
for s in a.seeds:
    DATA[s] = {f: load_fit(s, f) for f in a.fits}
AVAIL = sorted(set.intersection(*[set(D['cells']['mixture']) for s in a.seeds for D in DATA[s].values()]))
A_GRP = [m for m in A_GRP if m in AVAIL]; B_GRP = [m for m in B_GRP if m in AVAIL]   # groups restricted to the mixtures that were run
if not A_GRP or not B_GRP: raise SystemExit(f'groups A/B need at least one mixture each; available {AVAIL}')

# ---------------------------------------------------------------- structure
P('\n[structure]')
for s in a.seeds:
    for f, D in DATA[s].items():
        args_ = D['rec']['args']; aseeds = [int(x) for x in args_['assign_seeds']]; mixes = list(args_['mixes'])
        if [int(x) for x in args_['test_seed']] != [f]: PROBLEMS.append(f's{s} f{f}: test_seed {args_["test_seed"]}')
        c = D['cells']; dup = c.duplicated(['split', 'mixture', 'aseed', 'method']).sum()
        methods = sorted(set(c['method']))
        want = len(aseeds) * len(mixes) * len(methods)   # every round-17 method is written for every cell (QE variants fall back to the front end)
        if dup or len(c) != want: PROBLEMS.append(f's{s} f{f}: {len(c)} test cells, expected {want}, duplicates {dup}')
        R = D['pq']['records']; nrec = 0; qref = None
        for (ss, es, mx, asd, m), r in R.items():
            if es != 'test': continue
            nrec += 1
            if ss != f: PROBLEMS.append(f's{s} f{f}: record with fit {ss}')
            if qref is None: qref = r['q']
            elif not np.array_equal(qref, r['q']): PROBLEMS.append(f's{s} f{f}: query index mismatch at {(mx, asd, m)}')
        want_rec = len(aseeds) * len(mixes) * len([m for m in methods if not m.startswith('QE+Const@')])
        if nrec != want_rec: PROBLEMS.append(f's{s} f{f}: {nrec} per-query records, expected {want_rec}')
        P(f'  s{s} f{f}: {len(c)} cells ({len(mixes)} mixtures x {len(aseeds)} assignment seeds x {len(methods)} methods), {nrec} per-query records, '
          f'{0 if qref is None else len(qref)} queries; script {D["rec"].get("script_sha256", "")[:12]}; fixed selection {os.path.basename(str(args_.get("fixed_selection")))}')
P('  problems: ' + ('; '.join(PROBLEMS) if PROBLEMS else 'none'))

# ---------------------------------------------------------------- reproduction against round 15
P('\n[reproduction] round-17 recomputation vs round-15 cells (same fit, mixture, assignment seed); max |difference|')
REPRO_OK = True; REPORT['reproduction'] = {}
for s in a.seeds:
    for f, D in DATA[s].items():
        f15 = prefix(os.path.join(a.r15_root, f's{s}_f{f}')) + 'cells.csv'
        c15 = pd.read_csv(f15); c15 = c15[c15['evalset'].astype(str) == 'test']
        m = D['cells'][D['cells']['method'].isin(SHARED)].merge(c15[c15['method'].isin(SHARED)], on=['split', 'mixture', 'aseed', 'method'], suffixes=('', '_15'))
        n17 = D['cells']['method'].isin(SHARED).sum()
        cols = [c for c in ('mAP', 'R1', 'R5', 'R10', 'mINP', 'dmAP', 'dR1') if c in m.columns and c + '_15' in m.columns]
        mx = max(float((m[c] - m[c + '_15']).abs().max()) for c in cols) if len(m) else float('nan')
        ok = len(m) == n17 and n17 > 0 and mx <= a.tol
        REPRO_OK &= ok; REPORT['reproduction'][f's{s}_f{f}'] = dict(matched=int(len(m)), cells=int(n17), max_abs_diff=mx, ok=bool(ok))
        P(f'  s{s} f{f}: matched {len(m)}/{n17} cells, max |diff| over {cols} = {mx:.3g} -> {"reproduced" if ok else "NOT REPRODUCED"}')
P('  verdict: ' + ('all shared methods reproduced within tolerance %.0e' % a.tol if REPRO_OK else 'REPRODUCTION FAILED: the new control must not be paired with the frozen results'))

# ---------------------------------------------------------------- selection
P('\n[selection] QE+Const gain chosen on the development fit galleries (frozen main rule; differences relative to B* as for every selected method)')
REPORT['selection'] = {}
for s in a.seeds:
    for f, D in DATA[s].items():
        e = [x for x in D['sel'] if x['method'] == 'QE+Const']
        if len(e) != 1: PROBLEMS.append(f's{s} f{f}: {len(e)} QE+Const selection entries'); continue
        e = e[0]; REPORT['selection'][f's{s}_f{f}'] = e
        tbl = ' | '.join(f"{r['cfg']:g}: {r['mean_dmAP']:+.2f}/{r['min_dR1']:+.2f}{'' if r['feasible'] else 'x'}" for r in e['table'])
        P(f"  s{s} f{f}: chosen g = {e['chosen']} (feasible {e['feasible']}) | g: mean dmAP / worst-type dR1 (x = infeasible): {tbl}")
        qe_ = [x for x in D['sel'] if x['method'] == 'QE-DBA']
        P(f"           front end QE-DBA {qe_[0]['chosen'] if qe_ else '?'} ({qe_[0].get('source', '') if qe_ else ''}); kappa_a {D['log']['r17log'][0]['ka_used']}; "
          f"selection time {D['log']['times'].get('select_const_s', ['?'])[0]} s")

# ---------------------------------------------------------------- absolute metrics (cell means)
P('\n[table A] absolute metrics (%), cell means over 3 fits x assignment seeds; groups are means over their mixtures')
REPORT['absolute'] = {}
hdr = '  %-10s | %-34s | %-34s | %s' % ('method', 'A: mAP / R1 / R5 / R10 / mINP', 'B: mAP / R1 / R5 / R10 / mINP', 'B mixtures M5e M5f M4id: mAP / R1 / mINP')
for s in a.seeds:
    c = pd.concat([D['cells'] for D in DATA[s].values()])
    P(f'  ---- s{s}'); P(hdr)
    for m in MAIN:
        cm = c[c['method'] == m]
        if cm.empty: continue
        g = lambda mixes, k: float(np.mean([cm[cm['mixture'] == x][k].mean() for x in mixes]))
        Av = [g(A_GRP, k) for k in ('mAP', 'R1', 'R5', 'R10', 'mINP')]; Bv = [g(B_GRP, k) for k in ('mAP', 'R1', 'R5', 'R10', 'mINP')]
        Bm = ['%.2f/%.2f/%.2f' % tuple(cm[cm['mixture'] == x][k].mean() for k in ('mAP', 'R1', 'mINP')) for x in B_GRP]
        REPORT['absolute'][f's{s}:{m}'] = dict(A=Av, B=Bv, Bmix=Bm)
        P('  %-10s | %s | %s | %s' % (m, ' '.join('%6.2f' % v for v in Av), ' '.join('%6.2f' % v for v in Bv), '  '.join(Bm)))


# ---------------------------------------------------------------- paired differences with explicit alignment
def per_query(seed, method, mixes, metric):
    """per-query value averaged over all (fit, assignment seed) cells and over the given mixtures; explicit keys"""
    acc = None; qid = None; n = 0
    for mx in mixes:
        for f, D in DATA[seed].items():
            R = D['pq']['records']; aseeds = [int(x) for x in D['rec']['args']['assign_seeds']]
            for asd in aseeds:
                r = R[(f, 'test', mx, asd, method)]
                v = r[metric].astype(np.float64) * (100.0 if metric in ('R1', 'R5', 'R10') else 1.0)
                if acc is None: acc = np.zeros_like(v); qref = r['q']; qid = r['qid']
                elif not np.array_equal(r['q'], qref): raise SystemExit(f'query alignment failed at s{seed} f{f} {mx} {asd} {method}')
                acc += v; n += 1
    return acc / n, qid, n


P('\n[table B] paired differences X - Y (per query: mean over 15 cells and the group\'s mixtures), 95% identity-cluster bootstrap; * = excludes 0')
REPORT['paired'] = {}
for s in a.seeds:
    P(f'  ---- s{s}')
    for X, Y in PAIRS:
        for gname, mixes in (('A', A_GRP), ('B', B_GRP)) + tuple((x, [x]) for x in B_GRP):
            parts = []
            for met in METRICS:
                vx, qid, n1 = per_query(s, X, mixes, met); vy, _, n2 = per_query(s, Y, mixes, met)
                mu, lo, hi = boot_ci(vx - vy, qid); REPORT['paired'][f's{s}:{X}-{Y}:{gname}:{met}'] = (mu, lo, hi, n1)
                parts.append(f'{met} {mu:+6.2f} [{lo:+5.2f},{hi:+5.2f}]{star(lo, hi)}')
            P(f'  {X + " - " + Y:22s} {gname:5s} ' + '  '.join(parts))

# ---------------------------------------------------------------- curve (report-only)
P('\n[curve] report-only: QE+Const at every grid value on the test galleries (cell means); NOT used to choose the gain')
REPORT['curve'] = {}
for s in a.seeds:
    c = pd.concat([D['cells'] for D in DATA[s].values()])
    names = sorted({m for m in c['method'] if m.startswith('QE+Const@')}, key=lambda m: float(m.split('@')[1]))
    rows = []
    for m in names:
        cm = c[c['method'] == m]; g = lambda mixes, k: float(np.mean([cm[cm['mixture'] == x][k].mean() for x in mixes]))
        vals = dict(A_mAP=g(A_GRP, 'mAP'), A_R1=g(A_GRP, 'R1'), B_mAP=g(B_GRP, 'mAP'), B_R1=g(B_GRP, 'R1')); REPORT['curve'][f's{s}:{m}'] = vals
        rows.append(f"{m.split('@')[1]}: A {vals['A_mAP']:.2f}/{vals['A_R1']:.2f} B {vals['B_mAP']:.2f}/{vals['B_R1']:.2f}")
    for m in ('QE+M', 'QE-DBA'):
        cm = c[c['method'] == m]; g = lambda mixes, k: float(np.mean([cm[cm['mixture'] == x][k].mean() for x in mixes]))
        rows.append(f"[{m}] A {g(A_GRP, 'mAP'):.2f}/{g(A_GRP, 'R1'):.2f} B {g(B_GRP, 'mAP'):.2f}/{g(B_GRP, 'R1'):.2f}")
    P(f'  s{s} (g: A mAP/R1  B mAP/R1): ' + ' | '.join(rows))

# ---------------------------------------------------------------- locality, harm/help, top-5 composition
P('\n[locality] top-1 changes relative to QE-DBA over (query, cell) pairs, split by the availability of QE-DBA\'s top-1 item')
P('           (exact-arithmetic proposition: no change when QE-DBA\'s top-1 is not IR-only; floating-point ties are the only exception)')
REPORT['locality'] = {}; REPORT['harmhelp'] = {}; REPORT['top5'] = {}
for s in a.seeds:
    for X in ('QE+M', 'QE+Const', 'QE+NoGate', 'QE+CAET', 'QE+SRC2f', 'QE+EM'):
        n_c = ch_c = n_i = ch_i = 0
        for f, D in DATA[s].items():
            R = D['pq']['records']
            for (ss, es, mx, asd, m), r in R.items():
                if m != X or es != 'test': continue
                fr = R[(ss, es, mx, asd, 'QE-DBA')]; nc = fr['top1_av'] != 1
                n_c += nc.sum(); ch_c += (r['top1'][nc] != fr['top1'][nc]).sum(); n_i += (~nc).sum(); ch_i += (r['top1'][~nc] != fr['top1'][~nc]).sum()
        REPORT['locality'][f's{s}:{X}'] = dict(n_front_not_ir=int(n_c), changed=int(ch_c), n_front_ir=int(n_i), changed_ir=int(ch_i))
        P(f'  s{s} {X:9s}: front top-1 not IR-only {n_c:7d} pairs, top-1 changed {ch_c:5d} | front top-1 IR-only {n_i:7d} pairs, changed {ch_i:6d}')
P('\n[harm/help] rank-1 flips relative to QE-DBA (harm: right -> wrong, help: wrong -> right), in % of the stated denominator')
for s in a.seeds:
    for X in ('QE+M', 'QE+Const', 'QE+NoGate'):
        for gname, mixes in (('A', A_GRP), ('B', B_GRP)):
            tot = irt = h = g_ = hi = gi = 0
            for f, D in DATA[s].items():
                R = D['pq']['records']
                for mx in mixes:
                    for asd in [int(x) for x in D['rec']['args']['assign_seeds']]:
                        fr = R[(f, 'test', mx, asd, 'QE-DBA')]; r = R[(f, 'test', mx, asd, X)]; ir = fr['top1_av'] == 1
                        hh = (fr['R1'] == 1) & (r['R1'] == 0); gg = (fr['R1'] == 0) & (r['R1'] == 1)
                        tot += len(ir); irt += ir.sum(); h += hh.sum(); g_ += gg.sum(); hi += hh[ir].sum(); gi += gg[ir].sum()
            d = dict(pairs=int(tot), harm_all=100.0 * h / tot, help_all=100.0 * g_ / tot, pairs_front_ir=int(irt),
                     harm_front_ir=100.0 * hi / max(irt, 1), help_front_ir=100.0 * gi / max(irt, 1))
            REPORT['harmhelp'][f's{s}:{X}:{gname}'] = d
            P(f"  s{s} {X:9s} {gname}: all pairs n={tot}: harm {d['harm_all']:.2f} help {d['help_all']:.2f} | pairs whose QE-DBA top-1 is IR-only "
              f"n={irt}: harm {d['harm_front_ir']:.2f} help {d['help_front_ir']:.2f}")
P('\n[top-5] QE+M vs QE-DBA on group A: R5 losses/gains and the number of IR-only NEGATIVE items in the top-5 (mean per query-cell pair)')
for s in a.seeds:
    for X in ('QE+M', 'QE+Const'):
        lose = gain = tot = 0; neg_ir_f = neg_ir_x = neg_ir_f_l = neg_ir_x_l = 0.0
        for f, D in DATA[s].items():
            R = D['pq']['records']
            for mx in A_GRP:
                for asd in [int(x) for x in D['rec']['args']['assign_seeds']]:
                    fr = R[(f, 'test', mx, asd, 'QE-DBA')]; r = R[(f, 'test', mx, asd, X)]
                    if 'top10_av' not in r or 'top10_av' not in fr: continue
                    l_ = (fr['R5'] == 1) & (r['R5'] == 0); lose += l_.sum(); gain += ((fr['R5'] == 0) & (r['R5'] == 1)).sum(); tot += len(l_)
                    nf = ((fr['top10_av'][:, :5] == 1) & (fr['top10_pos'][:, :5] == 0)).sum(1); nx = ((r['top10_av'][:, :5] == 1) & (r['top10_pos'][:, :5] == 0)).sum(1)
                    neg_ir_f += nf.sum(); neg_ir_x += nx.sum(); neg_ir_f_l += nf[l_].sum(); neg_ir_x_l += nx[l_].sum()
        if tot == 0: continue
        d = dict(pairs=int(tot), r5_loss=100.0 * lose / tot, r5_gain=100.0 * gain / tot, ir_neg_top5_front=neg_ir_f / tot, ir_neg_top5_x=neg_ir_x / tot,
                 ir_neg_top5_front_on_loss=neg_ir_f_l / max(lose, 1), ir_neg_top5_x_on_loss=neg_ir_x_l / max(lose, 1))
        REPORT['top5'][f's{s}:{X}'] = d
        P(f"  s{s} {X:8s}: R5 loss {d['r5_loss']:.2f}% gain {d['r5_gain']:.2f}% of {tot} pairs | IR-only negatives in top-5: QE-DBA {d['ir_neg_top5_front']:.2f} -> {X} {d['ir_neg_top5_x']:.2f}; "
          f"on R5-loss pairs {d['ir_neg_top5_front_on_loss']:.2f} -> {d['ir_neg_top5_x_on_loss']:.2f}")

# ---------------------------------------------------------------- timing
P('\n[timing] medians over fits of the per-fit medians (warm-up + repeats); online stages on one test gallery; float64')
stages = [st for s in a.seeds for D in DATA[s].values() for st in D['log']['times'].get('stages', [])]
REPORT['timing'] = {}
if stages:
    keys = [k for k, v in stages[0].items() if isinstance(v, dict) and 'median_s' in v]
    nq = np.median([st['shape']['n_queries'] for st in stages]); ni = np.median([st['shape']['n_items'] for st in stages])
    for k in keys:
        v = float(np.median([st[k]['median_s'] for st in stages if k in st])); REPORT['timing'][k] = v
        per_q = f' ({v / nq * 1e3:.4f} ms per query)' if k.startswith('online') else ''
        P(f'  {k:28s} {v * 1e3:10.2f} ms{per_q}')
    sel = [st.get('fit_const_selection_s') for st in stages if st.get('fit_const_selection_s') is not None]
    if sel: P(f'  QE+Const gain selection (development fit galleries, whole grid): median {np.median(sel):.2f} s per fit'); REPORT['timing']['const_selection_s'] = float(np.median(sel))
    inc = float(np.median([st['online_gate_and_assoc']['median_s'] - st['online_assoc_only']['median_s'] for st in stages]))
    P(f'  incremental online cost of the adaptive gate over the constant gain (gate features + logistic + gallery statistics): {inc * 1e3:.2f} ms per gallery')
    P(f"  gallery: {int(ni)} items, {int(nq)} queries; peak RSS median {np.median([st['peak_rss_mb'] for st in stages]):.0f} MB; env {stages[0]['env']}")
    REPORT['timing']['incremental_gate_s'] = inc

# ---------------------------------------------------------------- development sensitivity (optional)
if a.devsens_root:
    P('\n[dev sensitivity] development half galleries, B* front end, MAIN (A-2f-ctxn) - B*, cell means over 5 splits x 2 folds x assignment seeds')
    REPORT['devsens'] = {}
    for d in sorted(glob.glob(os.path.join(a.devsens_root, '*'))):
        hits = glob.glob(os.path.join(d, 'r12_*_summary.csv'))
        if len(hits) != 1: continue
        c = pd.read_csv(hits[0][:-len('summary.csv')] + 'cells.csv'); c = c[c['method'] == 'A-2f-ctxn']
        lg = json.load(open(one(os.path.join(d, 'r17_*_r17log.json'))))
        g = lambda mixes, k: float(np.mean([c[c['mixture'] == x][k].mean() for x in mixes if (c['mixture'] == x).any()]))
        ka = sorted({round(x['ka_used'], 3) for x in lg['r17log']}); kf = sorted({round(x['ka_fitted'], 3) for x in lg['r17log']})
        v = dict(A_dmAP=g(A_GRP, 'dmAP'), A_dR1=g(A_GRP, 'dR1'), B_dmAP=g(B_GRP, 'dmAP'), B_dR1=g(B_GRP, 'dR1'), ka_used=ka, ka_fitted=kf,
                 wccn_shrink=lg['wccn_shrink'], wccn_shrink_eval=lg['wccn_shrink_eval'], n_cells=int(len(c)))
        REPORT['devsens'][os.path.basename(d)] = v
        P(f"  {os.path.basename(d):28s} A dmAP {v['A_dmAP']:+6.2f} dR1 {v['A_dR1']:+6.2f} | B dmAP {v['B_dmAP']:+6.2f} dR1 {v['B_dR1']:+6.2f} | "
          f"kappa_a used {ka} (fitted {kf}) | shrink fit {v['wccn_shrink']} eval {v['wccn_shrink_eval']} | {v['n_cells']} cells")

P('\n[status] ' + ('VALID: structure and reproduction checks passed' if (REPRO_OK and not PROBLEMS) else 'INVALID: ' + '; '.join(PROBLEMS + ([] if REPRO_OK else ['reproduction failed']))))
REPORT['problems'] = PROBLEMS; REPORT['reproduction_ok'] = bool(REPRO_OK)
json.dump(REPORT, open(a.out, 'w'), indent=1, default=str)
print('\n'.join(out_lines))
raise SystemExit(0 if (REPRO_OK and not PROBLEMS) else 3)
