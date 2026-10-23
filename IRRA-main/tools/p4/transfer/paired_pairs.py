"""Paired per-query differences for arbitrary method pairs (identity-cluster bootstrap, 4000 resamples, seed 7).
usage: paired_pairs.py <run_dir> <tag> <feats_npz> <out_csv> X:Y [X:Y ...]"""
import pickle, sys, numpy as np, pandas as pd
run, tag, feats, outc = sys.argv[1:5]; pairs = [p.split(':') for p in sys.argv[5:]]
pq = pickle.load(open(f'{run}/r12_{tag}_perq.pkl', 'rb')); qraw = np.load(feats)['clean/query_raw_pids']; rows = []
ORDER = ['M1', 'M3', 'M4', 'M4c25', 'M4c50', 'M4c75', 'M4id', 'M6c', 'M1g', 'M4c60', 'M9', 'M5e', 'M5f', 'M10']
MIXES = sorted(set(k[1] for k in pq), key=lambda m: (ORDER.index(m) if m in ORDER else 99, m))
for es in sorted(set(k[0] for k in pq)):
    for mx in MIXES:
        for X, Y in pairs:
            if (es, mx, X) not in pq or (es, mx, Y) not in pq: continue
            A, B = pq[(es, mx, X)], pq[(es, mx, Y)]; qs = sorted(set(A) & set(B))
            dm = np.array([np.mean([a[0] - b[0] for a, b in zip(A[q], B[q])]) for q in qs]); dr = np.array([np.mean([a[1] - b[1] for a, b in zip(A[q], B[q])]) for q in qs])
            ids = np.array([qraw[q] for q in qs]); u = np.unique(ids); idx = [np.where(ids == v)[0] for v in u]
            sm = np.array([dm[i].sum() for i in idx]); sr = np.array([dr[i].sum() for i in idx]); c = np.array([len(i) for i in idx])
            rb = np.random.default_rng(7); bm, br = [], []
            for _ in range(4000):
                x = rb.integers(0, len(u), len(u)); bm.append(sm[x].sum() / c[x].sum()); br.append(sr[x].sum() / c[x].sum())
            rows.append(dict(tag=tag, evalset=es, mixture=mx, X=X, Y=Y, dmAP=dm.mean(), dmAP_lo=np.percentile(bm, 2.5), dmAP_hi=np.percentile(bm, 97.5),
                             dR1=dr.mean(), dR1_lo=np.percentile(br, 2.5), dR1_hi=np.percentile(br, 97.5)))
P = pd.DataFrame(rows); P.to_csv(outc, index=False)
for (X, Y), d in P.groupby(['X', 'Y'], sort=False):
    print(f'{tag} {X} - {Y}: ' + '  '.join(f"{r.mixture}:{r.dmAP:+.2f}[{r.dmAP_lo:+.2f},{r.dmAP_hi:+.2f}]/{r.dR1:+.2f}" for r in d.itertuples()))
