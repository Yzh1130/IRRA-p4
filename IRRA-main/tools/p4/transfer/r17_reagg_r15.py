"""Round 17, read-only re-aggregation of the saved round-15 official-test outputs (main selection rule). No score is recomputed.

Inputs per base seed: <r15_root>/s<seed>_f<fit>/r12_*_{cells.csv, perq.pkl, run_record.json} for the three fits.
The round-15 per-query records are lists ordered by the evaluation loop (assignment seeds in the order of --assign_seeds); they carry no
cell key. This script therefore VERIFIES the alignment explicitly before using it: for every (fit, mixture, method) and list position i,
the mean over queries of the stored (dAP, dR1) must equal the cells.csv row of (fit, mixture, assign_seeds[i], method). Only then are the
records indexed by explicit keys (fit, assignment seed, mixture, query) and compared across methods.
Reports (evaluation set 'test'):
  [structure]   3 fits x 5 assignment seeds x 13 mixtures x 29 methods, no duplicates; record lengths; alignment check
  [cells]       mAP / R1 / R5 / R10 / mINP per mixture and seed (cell means) for the front end, the method and the gate variants
  [paired]      paired dmAP and dR1 with 95% identity-cluster bootstrap (4000, seed 7) per mixture and group; R5 / R10 / mINP
                differences as cell-mean point estimates only (the round-15 records keep no per-query R5 / R10 / mINP)
  [flips]       rank-1 flips of QE+M relative to QE-DBA over (query, cell) pairs; the records do not hold the top-1 item, so this is
                not a test of the locality proposition
usage: python -B r17_reagg_r15.py --r15_root <p4_r15_out/test/constrained> --qraw s101=<dev.npz>,<test.npz> ... --json out.json
"""
import argparse, glob, json, os, pickle
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--r15_root', required=True); ap.add_argument('--qraw', nargs='+', required=True)
ap.add_argument('--fits', type=int, nargs='+', default=[7001, 7002, 7003]); ap.add_argument('--boot', type=int, default=4000)
ap.add_argument('--json', required=True); ap.add_argument('--tol', type=float, default=1e-9)
a = ap.parse_args()

A = ['M4', 'M6c', 'M1g', 'M4c60', 'M9']; B = ['M5e', 'M5f', 'M4id']; OTHER = ['M0', 'M10', 'M4c25', 'M4c50', 'M4c75']
SHOW = ['QE-DBA', 'QE+M', 'QE+EM', 'QE+CAET', 'QE+SRC2f', 'QEir', 'QE+Orc']
PAIRS = [('QE+M', 'QE-DBA'), ('QE+M', 'QE+EM'), ('QE+M', 'QE+CAET'), ('QE+M', 'QE+SRC2f'), ('QE+M', 'QEir')]
REPORT = {}; PROBLEMS = []


def one(pattern):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1: raise SystemExit(f'expected exactly one file for {pattern}, found {len(hits)}')
    return hits[0]


def prefix(d):
    """r12_<tag>_ prefix of the single run in directory d (found through its summary file; 'cells' alone would also match diag_cells)"""
    return one(os.path.join(d, 'r12_*_summary.csv'))[:-len('summary.csv')]


def boot(per_q, qid, seed=7):
    u = np.unique(qid); idx = [np.where(qid == v)[0] for v in u]
    s = np.array([per_q[i].sum() for i in idx]); c = np.array([len(i) for i in idx])
    rb = np.random.default_rng(seed); b = np.empty(a.boot)
    for k in range(a.boot):
        x = rb.integers(0, len(u), len(u)); b[k] = s[x].sum() / c[x].sum()
    return float(per_q.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def st(lo, hi): return '*' if (lo > 0 or hi < 0) else ' '


for spec in a.qraw:
    tag, paths = spec.split('=', 1); seed = int(tag.lstrip('s')); dev, test = paths.split(',')
    qraw = np.concatenate([np.load(dev)['clean/query_raw_pids'], np.load(test)['clean/query_raw_pids']])
    print(f'\n######## {tag}')
    cells_all = []; REC = {}   # explicit keys: (fit, aseed, mixture, method) -> {query: (dAP, dR1)}
    for fit in a.fits:
        d = os.path.join(a.r15_root, f's{seed}_f{fit}')
        rec = json.load(open(prefix(d) + 'run_record.json')); args_ = rec['args']
        aseeds = [int(x) for x in args_['assign_seeds']]
        if [int(x) for x in args_['test_seed']] != [fit]: PROBLEMS.append(f'{tag} f{fit}: test_seed {args_["test_seed"]}')
        c = pd.read_csv(prefix(d) + 'cells.csv'); c = c[c['evalset'].astype(str) == 'test'].copy(); cells_all.append(c)
        dup = int(c.duplicated(['split', 'mixture', 'aseed', 'method']).sum()); mixes = sorted(set(c['mixture'])); methods = sorted(set(c['method']))
        # every (mixture, assignment seed) must carry the same method set within a mixture (a gallery without IR-only items, e.g. M0,
        # has no transfer-only rows); no duplicates; the assignment seeds are exactly those of the run record
        msets = {mx: {asd: frozenset(c[(c['mixture'] == mx) & (c['aseed'] == asd)]['method']) for asd in aseeds} for mx in mixes}
        uneven = [mx for mx, v in msets.items() if len(set(v.values())) != 1]
        if dup or uneven or sorted(set(c['aseed'])) != sorted(aseeds):
            PROBLEMS.append(f'{tag} f{fit}: duplicates {dup}, mixtures with uneven method sets {uneven}, aseeds {sorted(set(c["aseed"]))}')
        pq = pickle.load(open(prefix(d) + 'perq.pkl', 'rb'))
        worst = 0.0; nq = None
        for (es, mx, m), dct in pq.items():
            if es != 'test': continue
            qs = sorted(dct)
            if nq is None: nq = len(qs)
            if len(qs) != nq: PROBLEMS.append(f'{tag} f{fit} {mx} {m}: {len(qs)} queries, expected {nq}')
            L = np.array([dct[q] for q in qs], dtype=np.float64)          # queries x positions x 2
            if L.shape[1] != len(aseeds): PROBLEMS.append(f'{tag} f{fit} {mx} {m}: {L.shape[1]} entries per query, expected {len(aseeds)}'); continue
            for i, asd in enumerate(aseeds):
                row = c[(c['mixture'] == mx) & (c['aseed'] == asd) & (c['method'] == m)]
                if len(row) != 1: PROBLEMS.append(f'{tag} f{fit} {mx} {asd} {m}: {len(row)} cell rows'); continue
                dif = max(abs(L[:, i, 0].mean() - float(row['dmAP'].iloc[0])), abs(L[:, i, 1].mean() - float(row['dR1'].iloc[0])))
                worst = max(worst, dif)
                REC[(fit, asd, mx, m)] = (np.array(qs), L[:, i, 0], L[:, i, 1])
        ok = worst <= a.tol
        if not ok: PROBLEMS.append(f'{tag} f{fit}: alignment check failed (max |per-query mean - cell| = {worst:.3g})')
        print(f'  [structure] f{fit}: {len(c)} cells ({len(mixes)} mixtures x {len(aseeds)} assignment seeds, up to {len(methods)} methods per mixture), duplicates {dup}; '
              f'{nq} queries per record list; alignment position i <-> assignment seed {aseeds}: max |mean of records - cell| = {worst:.2e} -> {"verified" if ok else "FAILED"}')
    C = pd.concat(cells_all)
    REPORT[tag] = dict(cells={}, paired={}, point={}, flips={})
    print('  [cells] cell means over all fits x assignment seeds: mAP / R1 / R5 / R10 / mINP')
    for mx in A + B + OTHER:
        parts = []
        for m in SHOW:
            r = C[(C['mixture'] == mx) & (C['method'] == m)]
            if r.empty: continue
            v = [float(r[k].mean()) for k in ('mAP', 'R1', 'R5', 'R10', 'mINP')]; REPORT[tag]['cells'][f'{mx}:{m}'] = v
            parts.append(f'{m} ' + '/'.join(f'{x:.2f}' for x in v))
        print(f'    {mx:6s} ' + ' | '.join(parts))

    def perq(m, mixes, col):
        acc = None; qref = None; n = 0
        for mx in mixes:
            for fit in a.fits:
                for asd in sorted({k[1] for k in REC if k[0] == fit}):
                    qs, dap, dr1 = REC[(fit, asd, mx, m)]
                    v = dap if col == 0 else dr1
                    if acc is None: acc = np.zeros_like(v); qref = qs
                    elif not np.array_equal(qs, qref): raise SystemExit('query alignment failed')
                    acc += v; n += 1
        return acc / n, qraw[qref], n
    print('  [paired] X - Y: dmAP [95% CI] | dR1 [95% CI] | R5 / R10 / mINP cell-mean differences (point estimates, no interval)')
    for X, Y in PAIRS:
        for gname, mixes in [('A', A), ('B', B)] + [(x, [x]) for x in A + B + OTHER]:
            try:
                vx, qid, n = perq(X, mixes, 0); vy, _, _ = perq(Y, mixes, 0); mu, lo, hi = boot(vx - vy, qid)
                rx, _, _ = perq(X, mixes, 1); ry, _, _ = perq(Y, mixes, 1); mr, lr, hr = boot(rx - ry, qid)
            except KeyError:
                continue
            pt = []
            for k in ('R5', 'R10', 'mINP'):
                xs = np.mean([C[(C['mixture'] == x) & (C['method'] == X)][k].mean() for x in mixes]); ys = np.mean([C[(C['mixture'] == x) & (C['method'] == Y)][k].mean() for x in mixes])
                pt.append(float(xs - ys))
            REPORT[tag]['paired'][f'{X}-{Y}:{gname}'] = dict(dmAP=(mu, lo, hi), dR1=(mr, lr, hr), R5=pt[0], R10=pt[1], mINP=pt[2], cells=n)
            print(f'    {X + " - " + Y:18s} {gname:6s} dmAP {mu:+6.2f} [{lo:+5.2f},{hi:+5.2f}]{st(lo, hi)} | dR1 {mr:+6.2f} [{lr:+5.2f},{hr:+5.2f}]{st(lr, hr)} | '
                  f'R5 {pt[0]:+6.2f} R10 {pt[1]:+6.2f} mINP {pt[2]:+6.2f}')
    print('  [flips] QE+M vs QE-DBA, % of (query, cell) pairs (denominator = all pairs of the group); not a locality test')
    for gname, mixes in (('A', A), ('B', B)):
        n = h = g = 0
        for mx in mixes:
            for (fit, asd, mx_, m), (qs, dap, dr1) in REC.items():
                if mx_ != mx or m != 'QE+M': continue
                f_ = REC[(fit, asd, mx, 'QE-DBA')]; d = np.round(dr1 - f_[2])
                n += len(d); h += int((d == -100).sum()); g += int((d == 100).sum())
        REPORT[tag]['flips'][gname] = dict(pairs=n, harm=100.0 * h / max(n, 1), help=100.0 * g / max(n, 1))
        print(f'    {gname}: pairs {n}, harm {100.0 * h / max(n, 1):.2f}%, help {100.0 * g / max(n, 1):.2f}%')
print('\n[status] ' + ('VALID: structure and alignment verified' if not PROBLEMS else 'PROBLEMS: ' + '; '.join(PROBLEMS)))
REPORT['problems'] = PROBLEMS
json.dump(REPORT, open(a.json, 'w'), indent=1)
raise SystemExit(0 if not PROBLEMS else 3)
