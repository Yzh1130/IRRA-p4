"""Apply the stage-1c v2 decision rules (written in round 13, before any stage-1c run) to summary CSVs.

usage:
  python3 -I judge_v2.py --runs TAG@MODE=DIR [...] --primary TAG [TAG ...] [--mode stage1c|dev] [--out verdict.json]
    TAG  : run tag (summary file DIR/r12_<TAG>_summary.csv)
    MODE : half (2-fold, 239 strict) | three (3-fold: 'small' ~159 and 'large' ~319, strict) | full (2-fold, 478 stress test)
All rules use point estimates; identity-cluster bootstrap intervals are reported separately and are not part of the gate.

Gate rules (each primary setting, on strict galleries 'half' and 'large'):
  S1  sources     : every mixture  dmAP(main) >= dmAP(CAET) - 0.3
  S2  no source   : every mixture  dmAP(main) >= -0.3 and dR1(main) >= -0.3, and if dmAP(CAET) <= -0.3:
                    dmAP(main) - dmAP(CAET) >= 0.5 * |dmAP(CAET)|            (v1 used a fixed +0.5; reported as S2_v1)
  S3  partial     : mean over mixtures of dmAP(main) - dmAP(CAET) >= -0.1   (v1: every mixture >= 0; reported as S3_v1)
  L   leak        : leak(main) on M4id <= 0.08 on 'large' (strict) and 'full' (stress)
  R   scale       : 3-fold, large vs small (same fitted models): no-source advantage ratio >= 0.7, source gain ratio >= 0.8
Necessity rules (each primary setting, 'half' only; means over mixtures):
  Na  main - A-SRC-2f over no-source mixtures             >= +0.2
  Nb  main - N1 (A-ctxonlyN) over source + partial         >= +0.3
  Nc  main - N3 (CAET-hardN) over source + partial         >= +0.3
  Nd  N2 (CAET-recalN) - main over all non-probe mixtures  <= +0.1
  Ne  (information only) main - A-2f-EM on the multi-state mixture and over all mixtures
"""
import argparse, json
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='+', required=True)
ap.add_argument('--primary', nargs='+', required=True)
ap.add_argument('--mode', choices=['stage1c', 'dev'], default='stage1c')
ap.add_argument('--out', default=None)
a = ap.parse_args()
if a.mode == 'stage1c':
    G = dict(sources=['M4', 'M6c', 'M1g'], partial=['M4c60', 'M9'], nosource=['M5e', 'M5f'], leak=['M4id'], multistate=['M1g'], probe=['M10'])
else:
    G = dict(sources=['M1', 'M3', 'M4'], partial=['M4c25', 'M4c50', 'M4c75'], nosource=['M4id'], leak=['M4id'], multistate=['M1'], probe=[])
ALLM = G['sources'] + G['partial'] + G['nosource'] + [m for m in G['leak'] if m not in G['nosource']]
MAIN, N1, N2, N3 = 'A-2f-ctxn', 'A-ctxonlyN', 'CAET-recalN', 'CAET-hardN'

S = {}   # (tag, evalset) -> {(mixture, method): row}
for spec in a.runs:
    left, d = spec.split('=', 1); tag, mode = left.split('@')
    df = pd.read_csv(f'{d}/r12_{tag}_summary.csv')
    for r in df.itertuples():
        S.setdefault((tag, r.evalset), {})[(r.mixture, r.method)] = r

def v(tag, es, mx, m, k='dmAP'):
    r = S.get((tag, es), {}).get((mx, m)); return None if r is None else float(getattr(r, k))
def mean_diff(tag, es, mxs, x, y):
    vals = [v(tag, es, mx, x) - v(tag, es, mx, y) for mx in mxs if v(tag, es, mx, x) is not None and v(tag, es, mx, y) is not None]
    return (sum(vals) / len(vals)) if len(vals) == len(mxs) else None

res = {}
for (tag, es) in sorted(S):
    out = {}
    out['S1'] = {mx: dict(main=v(tag, es, mx, MAIN), caet=v(tag, es, mx, 'CAET')) for mx in G['sources']}
    for x in out['S1'].values(): x['ok'] = None if None in (x['main'], x['caet']) else x['main'] >= x['caet'] - 0.3
    s2 = {}
    for mx in G['nosource']:
        x, r1, y = v(tag, es, mx, MAIN), v(tag, es, mx, MAIN, 'dR1'), v(tag, es, mx, 'CAET')
        if None in (x, r1, y): s2[mx] = dict(ok=None, ok_v1=None); continue
        ok = x >= -0.3 and r1 >= -0.3 and (y > -0.3 or x - y >= 0.5 * abs(y))
        s2[mx] = dict(main=x, main_dR1=r1, caet=y, ok=ok, ok_v1=(x >= -0.3 and r1 >= -0.3 and x - y >= 0.5))
    out['S2'] = s2
    md = mean_diff(tag, es, G['partial'], MAIN, 'CAET')
    out['S3'] = dict(mean_main_minus_caet=md, ok=None if md is None else md >= -0.1,
                     ok_v1=all(v(tag, es, mx, MAIN) is not None and v(tag, es, mx, 'CAET') is not None and v(tag, es, mx, MAIN) >= v(tag, es, mx, 'CAET') for mx in G['partial']),
                     per_mixture={mx: (v(tag, es, mx, MAIN), v(tag, es, mx, 'CAET')) for mx in G['partial']})
    out['leak'] = {mx: v(tag, es, mx, MAIN, 'leak') for mx in G['leak']}
    out['N'] = dict(Na=mean_diff(tag, es, G['nosource'], MAIN, 'A-SRC-2f'), Nb=mean_diff(tag, es, G['sources'] + G['partial'], MAIN, N1),
                    Nc=mean_diff(tag, es, G['sources'] + G['partial'], MAIN, N3), Nd=mean_diff(tag, es, ALLM, N2, MAIN),
                    Ne_multistate=mean_diff(tag, es, G['multistate'], MAIN, 'A-2f-EM'), Ne_all=mean_diff(tag, es, ALLM, MAIN, 'A-2f-EM'))
    out['source_gain'] = mean_diff(tag, es, G['sources'], MAIN, 'B*')
    out['nosource_adv'] = mean_diff(tag, es, G['nosource'], MAIN, 'CAET')
    out['probe'] = {mx: dict(main=v(tag, es, mx, MAIN), caet=v(tag, es, mx, 'CAET'), asrc2f=v(tag, es, mx, 'A-SRC-2f'),
                             main_dR1=v(tag, es, mx, MAIN, 'dR1'), leak=v(tag, es, mx, MAIN, 'leak')) for mx in G['probe']}
    res[f'{tag}|{es}'] = out

def ok_all(d): return all(x.get('ok') for x in d.values())
def f2(x): return 'NA' if x is None else f'{x:+.2f}'
verdict = {}
for tag in sorted(set(t for t, _ in S)):
    prim = tag in a.primary; gates = {}; nec = {}
    for es in ('half', 'large'):
        o = res.get(f'{tag}|{es}')
        if o is None: continue
        gates[f'S1@{es}'] = ok_all(o['S1']); gates[f'S2@{es}'] = ok_all(o['S2']); gates[f'S3@{es}'] = bool(o['S3']['ok'])
        print(f'[{tag}{" primary" if prim else ""}] {es}: S1 {gates[f"S1@{es}"]}  S2 {gates[f"S2@{es}"]} (v1 {all(x.get("ok_v1") for x in o["S2"].values())})  '
              f'S3 {gates[f"S3@{es}"]} (v1 {o["S3"]["ok_v1"]}, mean main-CAET {f2(o["S3"]["mean_main_minus_caet"])})  leak {o["leak"]}')
        print('    S1 ' + '  '.join(f'{mx}: {f2(x["main"])} vs CAET {f2(x["caet"])}' for mx, x in o['S1'].items()))
        print('    S2 ' + '  '.join(f'{mx}: {f2(x.get("main"))} (dR1 {f2(x.get("main_dR1"))}) vs CAET {f2(x.get("caet"))}' for mx, x in o['S2'].items()))
        if o['probe']: print('    probe ' + '  '.join(f'{mx}: main {f2(x["main"])} CAET {f2(x["caet"])} A-SRC-2f {f2(x["asrc2f"])} dR1 {f2(x["main_dR1"])} leak {x["leak"]}' for mx, x in o['probe'].items()))
    for es in ('large', 'full'):
        o = res.get(f'{tag}|{es}')
        if o is not None:
            lk = list(o['leak'].values())[0]; gates[f'L@{es}'] = lk is not None and lk <= 0.08
    sm, lg = res.get(f'{tag}|small'), res.get(f'{tag}|large')
    if sm and lg:
        ok_den = all(x is not None and x > 0 for x in (sm['nosource_adv'], sm['source_gain'])) and all(x is not None for x in (lg['nosource_adv'], lg['source_gain']))
        if ok_den:
            r_ns, r_s = lg['nosource_adv'] / sm['nosource_adv'], lg['source_gain'] / sm['source_gain']
            gates['R'] = r_ns >= 0.7 and r_s >= 0.8
            print(f'[{tag}] scale large/small: no-source advantage ratio {r_ns:.2f}, source gain ratio {r_s:.2f}')
        else:
            gates['R'] = False   # undefined ratio (non-positive advantage or gain at ~159 items) counts as a failure
            print(f'[{tag}] scale large/small: ratio undefined (small-gallery advantage or gain <= 0) -> R fails')
    h = res.get(f'{tag}|half')
    if h:
        n = h['N']
        nec = dict(Na=None if n['Na'] is None else n['Na'] >= 0.2, Nb=None if n['Nb'] is None else n['Nb'] >= 0.3,
                   Nc=None if n['Nc'] is None else n['Nc'] >= 0.3, Nd=None if n['Nd'] is None else n['Nd'] <= 0.1)
        print(f'[{tag}] necessity (half): Na {f2(n["Na"])} {nec["Na"]} | Nb {f2(n["Nb"])} {nec["Nb"]} | Nc {f2(n["Nc"])} {nec["Nc"]} | Nd {f2(n["Nd"])} {nec["Nd"]} | '
              f'Ne(info) multistate {f2(n["Ne_multistate"])}, all {f2(n["Ne_all"])}')
    if a.mode == 'stage1c' and prim:   # every required check must exist; a missing run counts as a failure
        for k in ('S1@half', 'S2@half', 'S3@half', 'S1@large', 'S2@large', 'S3@large', 'L@large', 'L@full', 'R'):
            if k not in gates: gates[k] = False; print(f'[{tag}] required check {k} missing -> counts as failure')
        for k in ('Na', 'Nb', 'Nc', 'Nd'):
            if nec.get(k) is None: nec[k] = False; print(f'[{tag}] required necessity check {k} missing -> counts as failure')
    verdict[tag] = dict(primary=prim, gates=gates, necessity=nec)
prim_ok_g = all(all(vv for vv in verdict[t]['gates'].values()) and len(verdict[t]['gates']) > 0 for t in a.primary if t in verdict)
prim_ok_n = all(all(vv for vv in verdict[t]['necessity'].values()) and len(verdict[t]['necessity']) > 0 for t in a.primary if t in verdict)
final = ('PASS: proceed to stage 2' if prim_ok_g and prim_ok_n else
         'GATES PASS, NECESSITY PARTIAL: narrow the claim per failed N-rule (Na failing = stop as main method)' if prim_ok_g else
         'FAIL: holds only under the development protocol; stop as the main method of paper 4')
print('FINAL (primary settings):', final)
if a.out:
    json.dump(dict(final=final, verdict=verdict, details=res), open(a.out, 'w'), indent=1, default=lambda o: None)
