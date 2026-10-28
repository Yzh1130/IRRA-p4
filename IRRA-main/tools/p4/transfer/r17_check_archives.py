"""Round 17, read-only check of the feature archives that the round-15 test runs used (no retrieval metric is computed).

For every base seed it opens the development archive and the held-out archive exactly as r15_test.py does (test mode merges them and
keeps only the keys present in BOTH; a missing state key silently falls back to the clean array) and reports:
  1. the key sets of the two archives and of the merge;
  2. for every state and modality, whether the array the script actually reads (after the fallback) differs from the clean array,
     separately for the development rows and the held-out rows, with the maximum absolute difference and the mean cosine;
  3. which of these arrays enter the link graph (IR feature of every IR-bearing state) and the scores (drop-blank embedding);
  4. the number of held-out items in each state of mixture M1g for every assignment seed (the generator of r15_test.py, re-run on the
     held-out item count only; it draws from its own random stream).
Expected (base model, independent RGB / IR encoders): each degraded state changes exactly its own modality, i.e. rgb_missing,
rgb_noise_010, rgb_blur_k7 change f_rgb; ir_missing, ir_noise_010, ir_blur_k7 change f_ir; nothing else changes.
Exit status 0 = every expected change is present in the merged arrays of every seed; 2 = at least one expected change is missing.
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
EXPECT = {('rgb_missing', 'f_rgb'), ('rgb_noise_010', 'f_rgb'), ('rgb_blur_k7', 'f_rgb'), ('ir_missing', 'f_ir'), ('ir_noise_010', 'f_ir'), ('ir_blur_k7', 'f_ir')}
SIX = np.array(['clean', 'rgb_missing', 'ir_missing', 'rgb_noise_010', 'ir_noise_010', 'rgb_blur_k7'])
def l2n(x): return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)

report = {}; ok_all = True
for pr in a.pairs:
    tag, paths = pr.split('=', 1); dev, test = paths.split(',')
    zd, zt = np.load(dev), np.load(test); kd, kt = set(zd.files), set(zt.files); merged = sorted(kd & kt)
    nd, nt = len(zd['clean/img_pids']), len(zt['clean/img_pids'])
    print(f'######## {tag} | development {dev} ({nd} items, {len(kd)} keys) | held-out {test} ({nt} items, {len(kt)} keys)')
    print(f'  keys only in the development archive: {sorted(kd - kt) or "none"}; only in the held-out archive: {sorted(kt - kd) or "none"}')
    rows = []; missing = []
    for s in SC[1:]:
        for k in ('f_rgb', 'f_ir'):
            key = f'{s}/{k}'; in_m = key in merged
            parts = {}
            for part, z in (('dev', zd), ('test', zt)):
                x = (z[key] if in_m else z[f'clean/{k}']).astype(np.float64); c = z[f'clean/{k}'].astype(np.float64)
                parts[part] = dict(max_abs_diff=float(np.abs(x - c).max()), mean_cos=float((l2n(x) * l2n(c)).sum(1).mean()))
            changed = all(parts[p]['max_abs_diff'] > 0 for p in parts)
            expected = (s, k) in EXPECT
            use = []
            if k == 'f_ir' and AV[s] != 2: use.append('link graph')
            if (AV[s] == 0) or (AV[s] == 1 and k == 'f_ir') or (AV[s] == 2 and k == 'f_rgb'): use.append('score')
            status = 'ok' if changed == expected else ('MISSING CHANGE' if expected else 'UNEXPECTED CHANGE')
            if status != 'ok': ok_all = False; missing.append(key)
            rows.append(dict(key=key, in_dev=key in kd, in_test=key in kt, in_merged=in_m, read_as=key if in_m else f'clean/{k} (fallback)',
                             used_for=use, expected_change=expected, changed=changed, status=status, **{f'{p}_{m}': v[m] for p, v in parts.items() for m in v}))
            print(f"  {key:22s} merged={'yes' if in_m else 'NO '} read as {rows[-1]['read_as']:28s} used for {'+'.join(use) or '-':18s} "
                  f"dev max|d| {parts['dev']['max_abs_diff']:.3g} cos {parts['dev']['mean_cos']:.4f} | test max|d| {parts['test']['max_abs_diff']:.3g} "
                  f"cos {parts['test']['mean_cos']:.4f} | expected change {'yes' if expected else 'no '} -> {status}")
    m1g = {}
    for sd in a.assign_seeds:
        st = SIX[np.random.default_rng(sd + 96000).choice(6, nt, p=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1])]
        m1g[sd] = {s: int((st == s).sum()) for s in SIX}
    print('  M1g on the held-out gallery, items per state: ' + ' | '.join(f'{sd}: ' + ', '.join(f'{s} {n}' for s, n in v.items()) for sd, v in m1g.items()))
    print(f"  verdict {tag}: {'every expected degraded array is present in the merged archive and differs from clean' if not missing else 'PROBLEM with ' + ', '.join(missing)}")
    report[tag] = dict(dev=dev, test=test, merged_keys=merged, rows=rows, m1g_states=m1g, problems=missing)
json.dump(report, open(a.json, 'w'), indent=1)
raise SystemExit(0 if ok_all else 2)
