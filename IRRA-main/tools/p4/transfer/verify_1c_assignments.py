"""Check that a protocol script's mixture generator reproduces the frozen stage-1c assignment tables.
Only assignments are generated; no retrieval is evaluated.

usage: python3 -I verify_1c_assignments.py --script <r13_simcf.py or tools/p4/transfer/r13_protocol.py> --inputs_dir <inputs>
exit code 0 = identical for both base seeds; 1 = any difference.
"""
import argparse, ast, csv, os, sys, types
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--script', required=True); ap.add_argument('--inputs_dir', required=True)
a = ap.parse_args()
MIXES = ['M4', 'M6c', 'M1g', 'M4c60', 'M9', 'M5e', 'M5f', 'M4id', 'M10']; SEEDS = [5031, 5032, 5033]
def l2n(x): return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)
src = open(a.script, encoding='utf-8').read(); tree = ast.parse(src)
fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_eval_assign']
if not fn: sys.exit('make_eval_assign not found at module level')
fn_src = ast.get_source_segment(src, fn[0]); bad = 0
for b in ('s102', 's101'):
    zb = np.load(os.path.join(a.inputs_dir, f'base_ref_{b}.npz'))
    tp, gp, qraw = zb['clean/txt_pids'], zb['clean/img_pids'], zb['clean/query_raw_pids']
    tp2raw = dict(zip(tp.tolist(), qraw.tolist())); item_raw = np.array([tp2raw[p] for p in gp.tolist()])
    ns = dict(np=np, csv=csv, l2n=l2n, item_raw=item_raw, FR0=l2n(zb['clean/f_rgb'].astype(np.float64)),
              args=types.SimpleNamespace(assign=os.path.join(a.inputs_dir, f'assignments_{b}.csv'), assign_seeds=SEEDS))
    exec(fn_src, ns); asg = ns['make_eval_assign'](len(gp))
    frozen = {}
    for r in csv.DictReader(open(os.path.join(a.inputs_dir, f'stage1c_assignments_{b}.csv'))):
        frozen.setdefault((r['mixture'], int(r['seed'])), {})[int(r['gallery_index'])] = r['state']
    for m in MIXES:
        for sd in SEEDS:
            fz = frozen[(m, sd)]; ref = np.array([fz[i] for i in range(len(fz))]); got = asg[(m, sd)]
            if not np.array_equal(ref, got.astype(str)):
                bad += 1; print(f'MISMATCH base {b} mixture {m} seed {sd}: {int((ref != got.astype(str)).sum())} items differ')
    print(f'base {b}: checked {len(MIXES) * len(SEEDS)} (mixture, seed) tables')
print('ALL IDENTICAL' if bad == 0 else f'{bad} tables differ'); sys.exit(0 if bad == 0 else 1)
