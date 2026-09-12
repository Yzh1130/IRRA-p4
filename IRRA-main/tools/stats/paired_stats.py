#!/usr/bin/env python3
# paired_stats.py — reproduce every seed-paired statistic reported in the paper and its reproducibility notes.
# Read-only. Inputs: metrics.tsv files (columns: group seed scenario mode strength kernel dseed split R1 R5 R10 mAP mINP).
# Usage:
#   python paired_stats.py --main MAIN_GRID/metrics.tsv --dcalign stage4/metrics.tsv \
#                          --dblur DBLUR_GRID/metrics.tsv --dnonoise DNONOISE_GRID/metrics.tsv
# Conventions (fixed before the runs they analyse; see docs/protocol/ANALYSIS_PLAN.md and docs/dblur_followup_protocol_20260911.md):
#   split=test, dseed=2026, checkpoint=last (rows are written by the runners from last.pth only; the stage-4 file also
#   contains 'best' checkpoint rows for the clean scenario under groups ending in '-best' — they are excluded by group name).
#   Core scenarios: the 11 listed below; extended scenarios are ignored here (their values are not used in any test).
#   Paired difference = later configuration minus earlier configuration (e.g. D−A). Improved count = number of seeds with
#   difference > 0 (for relative drop, difference < 0 is counted as improved). 95% CI = mean ± t(0.975, n−1)·sd/√n (uncorrected).
#   Wilcoxon = exact two-sided signed-rank test; zero differences dropped, ties get average ranks.
#   Holm within each fixed family: F1 A−D (8 endpoints), F2 D−E (9), F3 D−U (9), F4 D−F (8), F5 Dblur (4), F6 DnoNoise (4).
#   Duplicate (group, seed, scenario) keys raise an error; nothing is silently overwritten.
import argparse, csv, itertools, collections, sys

CORE = ['clean', 'rgb_missing', 'ir_missing', 'rgb_noise_005', 'rgb_noise_010', 'ir_noise_005', 'ir_noise_010',
        'rgb_blur_k7', 'ir_blur_k7', 'both_noise_005', 'both_blur_k7']
T975 = {10: 2.262157, 5: 2.776445}

def load(path, groups, seeds):
    d = collections.defaultdict(dict); seen = set()
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['split'] != 'test' or int(float(r['dseed'])) != 2026: continue
            if r['group'] not in groups or int(r['seed']) not in seeds or r['scenario'] not in CORE: continue
            key = (r['group'], int(r['seed']), r['scenario'])
            if key in seen: raise SystemExit(f'duplicate row for {key} in {path}')
            seen.add(key)
            d[(r['group'], int(r['seed']))][r['scenario']] = {m: float(r[m]) for m in ('R1', 'mAP', 'mINP')}
    for g in groups:
        for s in seeds:
            miss = [n for n in CORE if n not in d.get((g, s), {})]
            if miss: raise SystemExit(f'{g} seed {s}: missing scenarios {miss} in {path}')
    return d

def summ(sc):
    names = [n for n in CORE if n != 'clean']
    vis = [n for n in names if n.startswith('rgb_')]; ir = [n for n in names if n.startswith('ir_')]; both = [n for n in names if n.startswith('both_')]
    c = sc['clean']['R1']; m = lambda L, k='R1': sum(sc[n][k] for n in L) / len(L)
    out = {'clean R1': c, 'ir R1': m(ir), 'vis R1': m(vis), 'both R1': m(both), 'all10 R1': m(names),
           'all10 mAP': m(names, 'mAP'), 'all10 mINP': m(names, 'mINP'),
           'reldrop%': sum((c - sc[n]['R1']) / c * 100 for n in names) / 10,
           'P1': (sc['rgb_noise_005']['R1'] + sc['rgb_noise_010']['R1']) / 2, 'P2': sc['rgb_blur_k7']['R1']}
    for n in names: out['sc:' + n] = sc[n]['R1']
    return out

def wilcoxon(x):
    x = [v for v in x if v != 0]; n = len(x)
    if n == 0: return 1.0
    a = sorted(range(n), key=lambda i: abs(x[i])); rk = [0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(x[a[j + 1]]) == abs(x[a[i]]): j += 1
        for k in range(i, j + 1): rk[a[k]] = (i + j + 2) / 2
        i = j + 1
    wp = sum(r for r, v in zip(rk, x) if v > 0); wm = sum(r for r, v in zip(rk, x) if v < 0); w = min(wp, wm)
    cnt = sum(1 for s in itertools.product((0, 1), repeat=n) if sum(r for r, b in zip(rk, s) if b) <= w + 1e-9)
    return min(1.0, 2 * cnt / 2 ** n)

def paired(S, a, b, met, seeds):
    x = [S[(b, s)][met] - S[(a, s)][met] for s in seeds]; n = len(x); mu = sum(x) / n
    sd = (sum((v - mu) ** 2 for v in x) / (n - 1)) ** 0.5; h = T975[n] * sd / n ** 0.5
    better = sum(v < 0 for v in x) if met == 'reldrop%' else sum(v > 0 for v in x)
    return {'cmp': f'{a}->{b}', 'met': met, 'diff': mu, 'sd': sd, 'lo': mu - h, 'hi': mu + h, 'up': better, 'n': n, 'p': wilcoxon(x)}

def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0] * m; run = 0
    for k, i in enumerate(order): run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj

def fmt(r, q=None):
    s = f"{r['cmp']:14s} {r['met']:18s} {r['diff']:+.3f}±{r['sd']:.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] better={r['up']}/{r['n']} p={r['p']:.4f}"
    return s + (f' holm={q:.4f}' if q is not None else '')

FAMILIES = {  # family id -> (a, b, endpoints, table)
    'F1': ('A', 'D', ['clean R1', 'ir R1', 'vis R1', 'both R1', 'all10 R1', 'all10 mAP', 'all10 mINP', 'reldrop%'], 'paper Table 7'),
    'F2': ('D', 'E', ['clean R1', 'ir R1', 'vis R1', 'both R1', 'all10 R1', 'all10 mAP', 'all10 mINP', 'sc:rgb_noise_010', 'reldrop%'], 'paper Table 9'),
    'F3': ('D', 'U', ['clean R1', 'ir R1', 'vis R1', 'both R1', 'all10 R1', 'all10 mAP', 'all10 mINP', 'sc:rgb_noise_010', 'reldrop%'], 'appendix Table A9'),
    'F4': ('D', 'F', ['clean R1', 'ir R1', 'vis R1', 'both R1', 'all10 R1', 'all10 mAP', 'all10 mINP', 'reldrop%'], 'appendix Table A10'),
}
CI_ONLY = [('A', 'E', 'appendix Table A8 / paper Table 10'), ('A', 'U', 'paper 4.2.2/4.5.3'), ('E', 'F', 'paper 4.5.2'), ('E', 'U', 'paper 4.9 context')]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', required=True); ap.add_argument('--dcalign'); ap.add_argument('--dblur'); ap.add_argument('--dnonoise')
    a = ap.parse_args()
    seeds = list(range(101, 111))
    S = {k: summ(v) for k, v in load(a.main, {'A', 'C', 'D', 'E', 'F', 'U', 'H2'}, seeds).items()}
    for fid, (x, y, mets, tab) in FAMILIES.items():
        print(f'== {fid} {x}->{y} ({tab}); Holm over {len(mets)} endpoints')
        rows = [paired(S, x, y, m, seeds) for m in mets]
        for r, q in zip(rows, holm([r['p'] for r in rows])): print(fmt(r, q))
    print('== CI-only comparisons (no significance claims; p printed for completeness only)')
    for x, y, tab in CI_ONLY:
        for m in ['clean R1', 'ir R1', 'vis R1', 'both R1', 'all10 R1', 'all10 mAP', 'all10 mINP', 'reldrop%']:
            print(fmt(paired(S, x, y, m, seeds)), f'  <{tab}>')
    print('== ablation (raw p, paper 4.6 / appendix Table A20)')
    for x, y in (('A', 'H2'), ('D', 'H2'), ('A', 'C')):
        for m in ['ir R1', 'vis R1', 'all10 R1']: print(fmt(paired(S, x, y, m, seeds)))
    print('== per-config means (paper Tables 5/6)')
    for g in ('A', 'C', 'D', 'E', 'F', 'U', 'H2'):
        print(g, {k: round(sum(S[(g, s)][k] for s in seeds) / 10, 3) for k in ('clean R1', 'ir R1', 'vis R1', 'both R1', 'all10 R1', 'all10 mAP', 'all10 mINP', 'reldrop%')})
    if a.dcalign:
        seeds5 = list(range(101, 106))
        D = {k: summ(v) for k, v in load(a.dcalign, {'DC-add', 'DC-dyn'}, seeds5).items()}
        print('== DCAlign (paper Table 13; n=5, raw p, no Holm; groups ending in -best excluded by name)')
        for m in ['clean R1', 'ir R1', 'vis R1', 'both R1', 'all10 R1']: print(fmt(paired(D, 'DC-add', 'DC-dyn', m, seeds5)))
    for fid, path, g in (('F5', a.dblur, 'Dblur'), ('F6', a.dnonoise, 'DnoNoise')):
        if not path: continue
        S.update({k: summ(v) for k, v in load(path, {g}, seeds).items()})
        print(f'== {fid} {g} (paper Table 11); Holm over 4 tests')
        rows = [paired(S, 'A', g, 'P1', seeds), paired(S, 'A', g, 'P2', seeds), paired(S, 'D', g, 'P1', seeds), paired(S, 'D', g, 'P2', seeds)]
        for r, q in zip(rows, holm([r['p'] for r in rows])): print(fmt(r, q))
        print(f'-- {g} descriptive vs A / vs D (no test)')
        for x in ('A', 'D'):
            for m in ['clean R1', 'vis R1', 'ir R1', 'all10 R1', 'sc:rgb_noise_005', 'sc:rgb_noise_010', 'sc:rgb_blur_k7', 'sc:rgb_missing']:
                print(fmt(paired(S, x, g, m, seeds)).rsplit(' p=', 1)[0])
    print('== done')

if __name__ == '__main__':
    main()
