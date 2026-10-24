"""Round-15 official-test judge (project doc "论文4_第十五轮_官方测试集预注册"): claims C1-C5 per base seed, printed tables.
Reads the outputs of r15_test.py. Several output directories of one base seed (one per fitting seed) are pooled: their cells are
concatenated and their per-query records are concatenated query by query, in the order the directories are given.
Group means are means over cells; paired differences use the identity-cluster bootstrap of the evaluation identities
(4000 resamples, seed 7), applied jointly to all mixtures of a group: per query, the difference is averaged over the cells
and then over the group's mixtures.
usage: python -B judge_test.py --runs TAG=DIR[,DIR2,...] [...] --qraw TAG=NPZ[,NPZ2] [...] [--evalset test] [--out verdict.json]
  each DIR holds exactly one r12_<name>_summary.csv (any <name>);
  --qraw: archives whose clean/query_raw_pids, concatenated in this order, index the per-query records
          (test mode: the development archive, then the held-out archive; development mode: the development archive)"""
import argparse, glob, json, os, pickle
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--qraw', nargs='+', required=True)
ap.add_argument('--evalset', default='test'); ap.add_argument('--out', default=None)
ap.add_argument('--boot', type=int, default=4000)
ap.add_argument('--no_claims', action='store_true')   # scale-analysis sub-galleries: tables only
a = ap.parse_args()

A = ['M4', 'M6c', 'M1g', 'M4c60', 'M9']          # sources + partial sources
B = ['M5e', 'M5f', 'M4id']                       # no source
OURS, FRONT, MAIN = 'QE+M', 'QE-DBA', 'A-2f-ctxn'
POOL = ['KR', 'RVN', 'QE-DBA', 'KRir', 'RVNir', 'QEir', 'GP-same']   # generic post-processing (C3)
SHOW = ['Raw', 'DB', 'B*', 'KR', 'RVN', 'QE-DBA', 'KRir', 'RVNir', 'QEir', 'GP-same', 'FeatImp', 'FeatImp-g', 'CAET', 'A-SRC-2f', 'A-2f-EM',
        MAIN, 'A-2f-ctxn-g', 'KR+M', 'RVN+M', 'KRir+M', 'RVNir+M', 'QEir+M', 'GP+M', 'QE+CAET', 'QE+SRC2f', 'QE+EM', OURS, 'QE+Orc', 'Oracle-A']
PAIRS = [(OURS, FRONT), (OURS, 'QEir'), (OURS, 'KRir'), (OURS, 'GP-same'), (OURS, 'QE+CAET'), (OURS, 'QE+SRC2f'), (OURS, 'QE+EM'),
         (MAIN, 'FeatImp-g'), (MAIN, 'FeatImp'), (MAIN, 'CAET'), (MAIN, 'A-SRC-2f'), (MAIN, 'A-2f-EM'), ('QE+Orc', OURS)]

qraws = {}
for spec in a.qraw:
    tag, files = spec.split('=', 1)
    qraws[tag] = np.concatenate([np.load(f)['clean/query_raw_pids'] for f in files.split(',')])

def prefix_of(d):
    hits = sorted(glob.glob(os.path.join(d, 'r12_*_summary.csv')))
    if len(hits) != 1: raise SystemExit(f'{d}: expected exactly one r12_*_summary.csv, found {len(hits)}')
    return hits[0][:-len('summary.csv')]

def load_pooled(dirs):
    cells, diag, sel, ml, cx, recs, pq = [], [], [], [], [], [], {}
    for d in dirs:
        p = prefix_of(d)
        c = pd.read_csv(p + 'cells.csv'); c['evalset'] = c['evalset'].astype(str); cells.append(c)
        dg = pd.read_csv(p + 'diag_cells.csv'); dg['evalset'] = dg['evalset'].astype(str); diag.append(dg)
        sel += json.load(open(p + 'selection.json')); recs.append(json.load(open(p + 'run_record.json')))
        if os.path.exists(p + 'maplog.csv'): ml.append(pd.read_csv(p + 'maplog.csv'))
        if os.path.exists(p + 'ctxlog.csv'): cx.append(pd.read_csv(p + 'ctxlog.csv'))
        part = pickle.load(open(p + 'perq.pkl', 'rb'))
        for k, dct in part.items():
            tgt = pq.setdefault(k, {})
            for q, lst in dct.items(): tgt.setdefault(q, []).extend(lst)
    return (pd.concat(cells, ignore_index=True), pd.concat(diag, ignore_index=True), sel,
            pd.concat(ml, ignore_index=True) if ml else None, pd.concat(cx, ignore_index=True) if cx else None, recs, pq)

def group_pair(pq, X, Y, mixes, ids, rng_seed=7, nboot=4000):
    """mean over queries of the per-query difference averaged over cells and mixtures; identity-cluster bootstrap"""
    qs = None
    for mx in mixes:
        k1, k2 = (a.evalset, mx, X), (a.evalset, mx, Y)
        if k1 not in pq or k2 not in pq: return None
        s_ = set(pq[k1]) & set(pq[k2]); qs = s_ if qs is None else qs & s_
    qs = sorted(qs); dm = np.zeros(len(qs)); dr = np.zeros(len(qs))
    for mx in mixes:
        P1, P2 = pq[(a.evalset, mx, X)], pq[(a.evalset, mx, Y)]
        dm += np.array([np.mean([u[0] - v[0] for u, v in zip(P1[q], P2[q])]) for q in qs])
        dr += np.array([np.mean([u[1] - v[1] for u, v in zip(P1[q], P2[q])]) for q in qs])
    dm /= len(mixes); dr /= len(mixes)
    qid = ids[np.array(qs)]; u = np.unique(qid); idx = [np.where(qid == v)[0] for v in u]
    sm = np.array([dm[i].sum() for i in idx]); sr = np.array([dr[i].sum() for i in idx]); c = np.array([len(i) for i in idx])
    rb = np.random.default_rng(rng_seed); bm, br = np.empty(nboot), np.empty(nboot)
    for b in range(nboot):
        x = rb.integers(0, len(u), len(u)); bm[b] = sm[x].sum() / c[x].sum(); br[b] = sr[x].sum() / c[x].sum()
    return dict(dmAP=float(dm.mean()), lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)),
                dR1=float(dr.mean()), r_lo=float(np.percentile(br, 2.5)), r_hi=float(np.percentile(br, 97.5)), n_q=len(qs), n_id=len(u))

def fmt(p):
    if p is None: return '        n/a        '
    star = '*' if (p['lo'] > 0 or p['hi'] < 0) else ' '
    return f"{p['dmAP']:+6.2f} [{p['lo']:+5.2f},{p['hi']:+5.2f}]{star} R1 {p['dR1']:+5.2f}"

res = {}
for spec in a.runs:
    tag, ds = spec.split('=', 1); dirs = ds.split(',')
    cells, diag, sel, ml, cx, recs, pq = load_pooled(dirs)
    c = cells[cells.evalset.map(lambda e: e == a.evalset or (e.startswith(a.evalset + '_') and a.evalset[0] == 'p'))]
    if not len(c): raise SystemExit(f'{tag}: no cells for evalset {a.evalset}')
    ids = qraws[tag]; have = set(c.method)
    mean_tab = c.groupby(['method', 'mixture']).mean(numeric_only=True); n_tab = c.groupby(['method', 'mixture']).size()
    def v(m, mx, k='dmAP'):
        return float(mean_tab.loc[(m, mx), k]) if (m, mx) in mean_tab.index and k in mean_tab.columns else float('nan')
    def gm(m, grp, k='dmAP'): return float(np.mean([v(m, mx, k) for mx in grp]))
    def has_m0(m): return (m, 'M0') in mean_tab.index
    r0 = recs[0]
    print(f'\n######## {tag} | evalset {a.evalset} | fits pooled: {len(dirs)} | script {r0.get("script_sha256", "?")[:12]} | '
          f'selection rule {r0["args"].get("select_rule")} | grid {r0["args"].get("grid_preset")} | fitting seeds {[r["args"].get("test_seed") for r in recs]}')
    print(f'cells per mixture: {int(n_tab.max())} | queries with records: {len(set().union(*[set(pq[k]) for k in pq if k[0] == a.evalset]))}')
    ext = all(k in c.columns for k in ('R5', 'R10', 'mINP'))
    print('\n[table 1] absolute metrics (%), mean over the group\'s mixtures | A = M4 M6c M1g M4c60 M9, B = M5e M5f M4id, M0 = all complete')
    print(f'{"method":11s} | {"A mAP":>6s} {"R1":>6s}' + (f' {"R5":>6s} {"R10":>6s} {"mINP":>6s}' if ext else '') + f' | {"B mAP":>6s} {"R1":>6s}'
          + (f' {"mINP":>6s}' if ext else '') + f' | {"M0 mAP":>6s} {"R1":>6s} | {"M10 mAP":>7s}')
    for m in SHOW:
        if m not in have: continue
        m0m = m if has_m0(m) else 'B*'
        row = f'{m:11s} | {gm(m, A, "mAP"):6.2f} {gm(m, A, "R1"):6.2f}' + (f' {gm(m, A, "R5"):6.2f} {gm(m, A, "R10"):6.2f} {gm(m, A, "mINP"):6.2f}' if ext else '')
        row += f' | {gm(m, B, "mAP"):6.2f} {gm(m, B, "R1"):6.2f}' + (f' {gm(m, B, "mINP"):6.2f}' if ext else '')
        row += f' | {v(m0m, "M0", "mAP"):6.2f} {v(m0m, "M0", "R1"):6.2f} | {v(m, "M10", "mAP"):7.2f}'
        print(row)
    print('\n[table 2] change vs B* (dmAP / dR1), group means; harm/help = queries whose B* top-1 is not IR-only and whose rank-1 turned right->wrong / wrong->right (%, group A)')
    print(f'{"method":11s} | {"A dmAP":>6s} {"dR1":>6s} | {"B dmAP":>6s} {"dR1":>6s} | {"M0":>6s} | {"M10":>6s} | {"harm":>5s} {"help":>5s}')
    for m in SHOW:
        if m not in have or m == 'B*': continue
        m0 = f'{v(m, "M0"):+6.2f}' if has_m0(m) else '  =B* '
        print(f'{m:11s} | {gm(m, A):+6.2f} {gm(m, A, "dR1"):+6.2f} | {gm(m, B):+6.2f} {gm(m, B, "dR1"):+6.2f} | {m0} | {v(m, "M10"):+6.2f} | '
              f'{gm(m, A, "disp_harm"):5.2f} {gm(m, A, "disp_help"):5.2f}')
    MX = [m for m in ['M0', 'M4', 'M6c', 'M1g', 'M4c60', 'M9', 'M5e', 'M5f', 'M4id', 'M10', 'M4c25', 'M4c50', 'M4c75'] if m in set(c.mixture)]
    KEY = [m for m in ['QE-DBA', 'QEir', 'KRir', 'GP-same', 'FeatImp-g', 'CAET', MAIN, 'QE+CAET', 'QE+EM', OURS, 'Oracle-A'] if m in have]
    print('\n[table 3] dmAP vs B* per mixture  (B* absolute mAP in the first row)')
    print(f'{"":11s} ' + ' '.join(f'{x:>6s}' for x in MX))
    print(f'{"B* mAP":11s} ' + ' '.join(f'{v("B*", x, "mAP"):6.2f}' for x in MX))
    for m in KEY:
        print(f'{m:11s} ' + ' '.join((f'{v(m, x):+6.2f}' if not np.isnan(v(m, x)) else '   =B*') for x in MX))
    print('\n[table 4] paired differences X - Y, group level, 95% identity-cluster bootstrap (* = interval excludes 0)')
    P = {}
    for X, Y in PAIRS:
        if X not in have or Y not in have: continue
        for gname, grp in (('A', A), ('B', B)):
            P[(X, Y, gname)] = group_pair(pq, X, Y, grp, ids, nboot=a.boot)
        print(f'{X:>9s} - {Y:<10s} A: {fmt(P[(X, Y, "A")])} | B: {fmt(P[(X, Y, "B")])}')
    C = {}
    if not a.no_claims:
        gA = {m: gm(m, A) for m in POOL if m in have}; gB = {m: gm(m, B) for m in POOL if m in have}
        zA = max(gA, key=gA.get); zB = max(gB, key=gB.get)
        for X, Y in ((OURS, zA), (OURS, zB)):
            for gname, grp in (('A', A), ('B', B)):
                if (X, Y, gname) not in P: P[(X, Y, gname)] = group_pair(pq, X, Y, grp, ids, nboot=a.boot)
        def ok_gain(p, thr): return p is not None and p['dmAP'] >= thr and p['lo'] > 0
        p1 = P.get((OURS, FRONT, 'A')); C['C1'] = dict(holds=ok_gain(p1, 1.0), detail=f'Ours - QE-DBA on A: {fmt(p1)} (needs >= +1.00 and lower bound > 0)')
        p2 = P.get((OURS, FRONT, 'B')); C['C2'] = dict(holds=p2 is not None and p2['dmAP'] >= -0.3 and p2['dR1'] >= -0.3,
                                                        detail=f'Ours - QE-DBA on B: {fmt(p2)} (needs dmAP >= -0.30 and dR1 >= -0.30)')
        p3a, p3b = P[(OURS, zA, 'A')], P[(OURS, zB, 'B')]
        C['C3'] = dict(holds=ok_gain(p3a, 1.0) and p3b is not None and p3b['dmAP'] >= -0.3,
                       detail=f'strongest generic on A = {zA} ({gA[zA]:+.2f}): Ours - {zA} on A {fmt(p3a)} (needs >= +1.00, lower bound > 0); '
                              f'strongest generic on B = {zB} ({gB[zB]:+.2f}): Ours - {zB} on B {fmt(p3b)} (needs >= -0.30)')
        p4a, p4b = P.get((MAIN, 'FeatImp-g', 'A')), P.get((MAIN, 'FeatImp-g', 'B'))
        C['C4'] = dict(holds=ok_gain(p4a, 0.5) and p4b is not None and p4b['dmAP'] >= -0.1,
                       detail=f'transfer alone - FeatImp-g: A {fmt(p4a)} (needs >= +0.50, lower bound > 0) | B {fmt(p4b)} (needs >= -0.10)')
        p5a, p5b = P.get((OURS, 'QE+CAET', 'A')), P.get((OURS, 'QE+CAET', 'B'))
        C['C5'] = dict(holds=p5b is not None and ok_gain(p5b, 0.3) and p5a is not None and p5a['dmAP'] >= -0.3,
                       detail=f'gated vs ungated transfer on the same front end: B {fmt(p5b)} (needs >= +0.30, lower bound > 0) | A {fmt(p5a)} (needs >= -0.30)')
        print('\n[claims]')
        for k, q in C.items(): print(f'  {k}: {"HOLDS" if q["holds"] else "does not hold"} | {q["detail"]}')
    print('\n[selection] chosen configuration per fit:')
    for m in ['KR', 'RVN', 'QE-DBA', 'KRir', 'RVNir', 'QEir', 'GP-same', 'FeatImp-g', 'A-2f-ctxn-g']:
        ch = [(x['split'], x['chosen'], x['feasible']) for x in sel if x['method'] == m]
        if ch: print(f'  {m:11s} ' + '; '.join(f'fit {sp}: {cfg}{"" if fe else " (no feasible config)"}' for sp, cfg, fe in ch))
    if r0.get('wccn_diag'): print('[wccn_diag]', json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r0['wccn_diag'].items()}))
    if ml is not None and len(ml):
        ml['grp'] = ml.evalset.astype(str).str.replace(r'_\d+$', '', regex=True)
        print('[link NN identity accuracy on the evaluation galleries, mean]'); print(ml.groupby('grp')[['n_items', 'acc_raw_cosine', 'acc_train_id_wccn', 'acc_fitfold_wccn']].mean().round(3).to_string())
    if cx is not None and len(cx):
        cx = cx[cx.src == 'eval'].copy(); cx['es'] = cx.type.str.split(':').str[0].str.replace(r'_\d+$', '', regex=True); cx['mx'] = cx.type.str.split(':').str[1]
        cx = cx[cx.es == a.evalset]
        if len(cx): print('[kappa_hat and p_I on the evaluation galleries, mean]'); print(cx.groupby('mx')[['kappa_all', 'pI', 'n_ironly']].mean().round(3).to_string())
    dg = diag[(diag.evalset.map(lambda e: e == a.evalset or e.startswith(a.evalset + '_'))) & (diag.method.isin([MAIN, OURS, 'CAET', 'QE+CAET']))]
    if len(dg):
        print('[leak = transfer mass on IR-only items without a same-identity complete item; coverage = valid mass where a source exists]')
        print(dg.groupby(['method', 'mixture'])[['leak', 'coverage', 'wrong_mass', 'auc']].mean().round(3).unstack(0).to_string())
    print(f'[run] wall seconds per fit {[r.get("wall_seconds") for r in recs]} | versions {r0.get("versions")}')
    res[tag] = dict(claims={k: q['holds'] for k, q in C.items()}, detail={k: q['detail'] for k, q in C.items()},
                    pairs={f'{k[0]} - {k[1]} | {k[2]}': q for k, q in P.items()},
                    group_means={m: dict(A=gm(m, A), B=gm(m, B), A_R1=gm(m, A, 'dR1'), B_R1=gm(m, B, 'dR1')) for m in SHOW if m in have and m != 'B*'})

final = {}
if not a.no_claims:
    print('\n######## claims over base seeds (a claim holds only if it holds on every base seed)')
    for k in ('C1', 'C2', 'C3', 'C4', 'C5'):
        per = {t: r['claims'][k] for t, r in res.items()}; final[k] = all(per.values())
        print(f'  {k}: {"HOLDS" if final[k] else "does not hold"}  ' + '  '.join(f'{t}={"Y" if x else "N"}' for t, x in per.items()))
if a.out: json.dump(dict(evalset=a.evalset, final=final, per_seed=res), open(a.out, 'w'), indent=1)
