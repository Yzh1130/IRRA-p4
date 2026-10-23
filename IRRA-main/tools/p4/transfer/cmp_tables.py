"""Compare two result folders table by table (numeric max |diff|, text equality, NaN masks)."""
import sys, json
import numpy as np, pandas as pd
A, B, tag = sys.argv[1], sys.argv[2], sys.argv[3]
for kind in ('summary', 'cells', 'diag_cells', 'neardup'):
    try:
        a = pd.read_csv(f'{A}/r12_{tag}_{kind}.csv'); b = pd.read_csv(f'{B}/r12_{tag}_{kind}.csv')
    except FileNotFoundError as e:
        print(kind, 'missing', e); continue
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        print(kind, 'SHAPE/COLUMNS DIFFER', a.shape, b.shape); continue
    num = a.select_dtypes('number').columns; txt = [c for c in a.columns if c not in num]
    na_eq = bool((a[num].isna().values == b[num].isna().values).all())
    d = np.abs(a[num].values - b[num].values); md = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
    print(f'{kind:10s} shape {a.shape}  max|diff| {md:.3g}  text equal {all((a[c].astype(str) == b[c].astype(str)).all() for c in txt)}  NaN masks equal {na_eq}')
try:
    sa = json.load(open(f'{A}/r12_{tag}_selection.json')); sb = json.load(open(f'{B}/r12_{tag}_selection.json'))
    print('selection equal:', sa == sb, len(sa), len(sb))
except FileNotFoundError as e:
    print('selection missing', e)
