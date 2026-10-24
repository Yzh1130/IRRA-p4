"""Equivalence check of the test mode: split the development archive by the split-seed-2026 identity folds into a pseudo
development archive (fold 0) and a pseudo held-out archive (fold 1), and export the development-mode assignments of the fold-1
items as CSV mixtures D_<name>. Running the test mode on these files must reproduce the development-mode rows of
(split 2026, fold 0, half) exactly (same fitted models, same galleries)."""
import ast, csv, sys, types
import numpy as np

src_script, feats, base, assign_csv, out_dir = sys.argv[1:6]
z = np.load(f'{feats}/{base}.npz'); keys = z.files
tp, gp, qraw = z['clean/txt_pids'], z['clean/img_pids'], z['clean/query_raw_pids']
tp2raw = dict(zip(tp.tolist(), qraw.tolist())); item_raw = np.array([tp2raw[p] for p in gp.tolist()])
uq = np.unique(qraw); perm = np.random.default_rng(2026).permutation(uq); fold = {int(p): i % 2 for i, p in enumerate(perm)}
qf = np.array([fold[int(p)] for p in qraw]); gf = np.array([fold[int(p)] for p in item_raw])
QK = {'clean/text_features', 'clean/txt_pids', 'clean/query_raw_pids'}
for f, name in ((0, 'pseudo_dev'), (1, 'pseudo_test')):
    qi, gi = np.where(qf == f)[0], np.where(gf == f)[0]
    np.savez(f'{out_dir}/{name}.npz', **{k: (z[k][qi] if k in QK else z[k][gi]) for k in keys})
    print(name, len(qi), 'queries', len(gi), 'items')
# development-mode assignments (exact function from the script, development arrays)
tree = ast.parse(open(src_script).read()); ns = {'np': np, 'csv': csv}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('l2n', 'make_eval_assign'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'r14', 'exec'), ns)
ns['args'] = types.SimpleNamespace(assign_seeds=[2026])
FR0 = ns['l2n'](z['clean/f_rgb'].astype(np.float64))
A = ns['make_eval_assign'](len(gp), item_raw, FR0, assign_csv)
g1 = np.where(gf == 1)[0]
with open(f'{out_dir}/pseudo_assign.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['mixture', 'seed', 'gallery_index', 'state'])
    for mx in ('M4', 'M5f', 'M1g', 'M4id'):
        st = A[(mx, 2026)][g1]
        for i, s in enumerate(st): w.writerow([f'D_{mx}', 2026, i, s])
print('assignments written for', len(g1), 'held-out items')
