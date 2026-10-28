"""Round 17, read-only check of the feature archives that the round-15 test runs used (no retrieval metric is computed).

For every base seed it opens the development archive and the held-out archive exactly as r15_test.py does: test mode merges them and keeps
only the keys present in BOTH; a state key missing from either archive silently falls back to the clean array for BOTH parts. For every
degraded state and modality it then checks the array the script actually reads, SEPARATELY in the development part and the held-out part
(v2 combined the two parts with all() and could miss a change that appeared in only one part):
  - target modality (the one the state degrades: rgb_* -> f_rgb, ir_* -> f_ir) must be changed in BOTH parts. Missing and noise states
    degrade every item by construction, so every row must differ from clean; blur may leave a flat image unchanged, so at least one row
    must differ and the fraction is reported;
  - non-target modality must be unchanged in BOTH parts: exactly equal to clean, row by row (the exporter stores a degraded-state array
    only when it is not bit-identical to clean, so a stored non-target array that differs is a real change);
  - every array read has the clean array's shape and only finite values; f_rgb, f_ir and the text features share one dimension; the base
    keys exist in both archives with consistent lengths.
States that no round-15 test mixture uses (ir_blur_k7) are checked and reported but not required.
Reported only (never a problem): the key sets, which arrays enter the link graph (IR feature of every IR-bearing state) and the scores
(drop-blank embedding), whether query identities of the two parts overlap, and the held-out item count per state of mixture M1g for every
assignment seed (the generator of r15_test.py, own random stream).
Exit status 0 = no problem in any seed; 2 = at least one problem (listed per seed and in the final [status] line).
usage: python -B r17_check_archives.py --pairs s101=<dev.npz>,<test.npz> s102=... s103=... --assign_seeds 7031 7032 7033 7034 7035 --json out.json
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--pairs', nargs='+', required=True); ap.add_argument('--assign_seeds', type=int, nargs='+', default=[7031, 7032, 7033, 7034, 7035])
ap.add_argument('--json', required=True)
a = ap.parse_args()

SC = ['clean', 'rgb_missing', 'ir_missing', 'rgb_noise_010', 'ir_noise_010', 'rgb_blur_k7', 'ir_blur_k7']
AV = {'clean': 0, 'rgb_noise_010': 0, 'ir_noise_010': 0, 'rgb_blur_k7': 0, 'ir_blur_k7': 0, 'rgb_missing': 1, 'ir_missing': 2}
TARGET = {'rgb_missing': 'f_rgb', 'rgb_noise_010': 'f_rgb', 'rgb_blur_k7': 'f_rgb', 'ir_missing': 'f_ir', 'ir_noise_010': 'f_ir', 'ir_blur_k7': 'f_ir'}
ALL_ROWS = {'rgb_missing', 'ir_missing', 'rgb_noise_010', 'ir_noise_010'}       # degrade every item by construction
USED = {'rgb_missing', 'ir_missing', 'rgb_noise_010', 'ir_noise_010', 'rgb_blur_k7'}  # states of the 13 round-15 test mixtures (M1g: six states)
BASE = ['clean/text_features', 'clean/txt_pids', 'clean/img_pids', 'clean/query_raw_pids', 'clean/f_rgb', 'clean/f_ir']
SIX = np.array(['clean', 'rgb_missing', 'ir_missing', 'rgb_noise_010', 'ir_noise_010', 'rgb_blur_k7'])
PARTS = ('dev', 'test')


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def base_checks(z, part, probs):
    """base keys of one archive: presence, lengths, dimensions, finite values; None when the state checks cannot run"""
    miss = [k for k in BASE if k not in z.files]
    if miss:
        probs.append(f'{part}: base keys missing {miss}'); return None
    t, tp, ip, qr, fr, fi = (z[k] for k in BASE)
    n_t, n_i = len(tp), len(ip); bad = []
    if t.ndim != 2 or len(t) != n_t or len(qr) != n_t: bad.append(f'text features {list(t.shape)} vs {n_t} text pids / {len(qr)} raw pids')
    for nm, x in (('clean/f_rgb', fr), ('clean/f_ir', fi)):
        if x.ndim != 2 or len(x) != n_i: bad.append(f'{nm} shape {list(x.shape)}, expected ({n_i}, d)')
    if not bad and not (fr.shape[1] == fi.shape[1] == t.shape[1]):
        bad.append(f'dimensions differ (text {t.shape[1]}, f_rgb {fr.shape[1]}, f_ir {fi.shape[1]})')
    for nm, x in (('clean/text_features', t), ('clean/f_rgb', fr), ('clean/f_ir', fi)):
        if not np.isfinite(x).all(): bad.append(f'{nm} has non-finite values')
    probs += [f'{part}: {b}' for b in bad]
    return None if bad else dict(n_txt=n_t, n_items=n_i, dim=int(fr.shape[1]))


def part_row(Z, p, s, k, in_m, target):
    """statistics of the array part p actually reads for state s and modality k, and the problems found in that part"""
    c = Z[p][f'clean/{k}']; x = Z[p][f'{s}/{k}'] if in_m else c
    d = dict(shape=list(x.shape))
    if x.shape != c.shape: return d, [f'SHAPE {list(x.shape)} vs clean {list(c.shape)} ({p})']
    if not np.isfinite(x).all(): return d, [f'NON-FINITE ({p}: {int((~np.isfinite(x)).sum())} values)']
    rowdiff = (x != c).any(axis=1); x64, c64 = x.astype(np.float64), c.astype(np.float64)
    d.update(rows_changed=float(rowdiff.mean()), n_rows=int(len(x)), max_abs_diff=float(np.abs(x64 - c64).max()) if x.size else 0.0,
             mean_cos=float((l2n(x64) * l2n(c64)).sum(1).mean()) if x.size else 1.0)
    if target:
        if not rowdiff.any(): return d, [f'MISSING CHANGE ({p}: identical to clean' + ('' if in_m else ', key absent -> clean fallback') + ')']
        if s in ALL_ROWS and not rowdiff.all(): return d, [f'PARTIAL CHANGE ({p}: {int((~rowdiff).sum())} of {len(x)} rows equal clean)']
    elif rowdiff.any():
        return d, [f'UNEXPECTED CHANGE ({p}: {int(rowdiff.sum())} of {len(x)} rows differ, max |d| {d["max_abs_diff"]:.3g})']
    return d, []


report = {}; all_probs = []
for pr in a.pairs:
    tag, paths = pr.split('=', 1); dev, test = paths.split(',')
    Z = {'dev': np.load(dev), 'test': np.load(test)}
    keys = {p: set(Z[p].files) for p in PARTS}; merged = sorted(keys['dev'] & keys['test'])
    probs = []
    info = {p: base_checks(Z[p], p, probs) for p in PARTS}
    print(f'######## {tag} | development {dev} ({len(keys["dev"])} keys) | held-out {test} ({len(keys["test"])} keys)')
    print(f'  keys only in the development archive: {sorted(keys["dev"] - keys["test"]) or "none"}; '
          f'only in the held-out archive: {sorted(keys["test"] - keys["dev"]) or "none"}')
    rows = []; overlap = None
    if info['dev'] and info['test']:
        print(f"  base: dev {info['dev']['n_txt']} queries / {info['dev']['n_items']} items, test {info['test']['n_txt']} queries / "
              f"{info['test']['n_items']} items, dim {info['dev']['dim']} / {info['test']['dim']}")
        if info['dev']['dim'] != info['test']['dim']: probs.append(f"feature dimension differs between parts ({info['dev']['dim']} vs {info['test']['dim']})")
        overlap = len(set(Z['dev']['clean/query_raw_pids'].tolist()) & set(Z['test']['clean/query_raw_pids'].tolist()))
        print(f'  query identities shared by the two parts (report only): {overlap}')
        for s in SC[1:]:
            for k in ('f_rgb', 'f_ir'):
                key = f'{s}/{k}'; in_m = key in merged; target = TARGET[s] == k; used = s in USED
                parts, flags = {}, []
                for p in PARTS:
                    parts[p], f = part_row(Z, p, s, k, in_m, target); flags += f
                use = []
                if k == 'f_ir' and AV[s] != 2: use.append('link graph')
                if (AV[s] == 0) or (AV[s] == 1 and k == 'f_ir') or (AV[s] == 2 and k == 'f_rgb'): use.append('score')
                status = 'ok' if not flags else ('; '.join(flags) if used else 'unused state, not required: ' + '; '.join(flags))
                if flags and used: probs.append(f'{key}: ' + '; '.join(flags))
                rows.append(dict(key=key, target=target, used_by_test_mixtures=used, in_dev=key in keys['dev'], in_test=key in keys['test'],
                                 in_merged=in_m, read_as=key if in_m else f'clean/{k} (fallback)', used_for=use, status=status, parts=parts))
                fmt = lambda p: (f"{p} rows changed {100 * parts[p]['rows_changed']:5.1f}% max|d| {parts[p]['max_abs_diff']:.3g} cos {parts[p]['mean_cos']:.4f}"
                                 if 'rows_changed' in parts[p] else f'{p} shape {parts[p]["shape"]}')
                print(f"  {key:22s} {'target    ' if target else 'non-target'} merged={'yes' if in_m else 'NO '} read as {rows[-1]['read_as']:28s} "
                      f"used for {'+'.join(use) or '-':18s} {fmt('dev')} | {fmt('test')} -> {status}")
    m1g = {}
    if info['test']:
        nt = info['test']['n_items']
        for sd in a.assign_seeds:
            st = SIX[np.random.default_rng(sd + 96000).choice(6, nt, p=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1])]
            m1g[sd] = {s: int((st == s).sum()) for s in SIX}
        print('  M1g on the held-out gallery, items per state: ' + ' | '.join(f'{sd}: ' + ', '.join(f'{s} {n}' for s, n in v.items()) for sd, v in m1g.items()))
    print(f"  verdict {tag}: " + ('in both parts every used degraded state changes its own modality and leaves the other one unchanged; shapes and values valid'
                                 if not probs else 'PROBLEMS: ' + ' | '.join(probs)))
    report[tag] = dict(dev=dev, test=test, merged_keys=merged, base=info, query_identity_overlap=overlap, rows=rows, m1g_states=m1g, problems=probs)
    all_probs += [f'{tag} {x}' for x in probs]
print('\n[status] ' + ('OK: no problem in any seed' if not all_probs else f'PROBLEMS ({len(all_probs)}): ' + ' | '.join(all_probs)))
with open(a.json, 'w') as fh:
    json.dump(report, fh, indent=1)
raise SystemExit(0 if not all_probs else 2)
