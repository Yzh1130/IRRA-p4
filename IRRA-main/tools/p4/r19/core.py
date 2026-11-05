"""Opt-in R19 helpers. Inference helpers accept camera/availability, never PID."""
from pathlib import Path
import csv
import numpy as np


def raw_gallery(archive):
    mapping = {}
    for label, raw in zip(archive['clean/txt_pids'], archive['clean/query_raw_pids']):
        label, raw = int(label), int(raw)
        if label in mapping and mapping[label] != raw:
            raise ValueError('Ambiguous query label/raw PID mapping')
        mapping[label] = raw
    return np.asarray([mapping[int(p)] for p in archive['clean/img_pids']])


def nested_indices(full, subset):
    """Match identity-deletion subsets by raw PID + original within-ID order.

    Full per-ID counts and clean modality arrays must match; partial deletion
    inside a retained identity is ambiguous and is deliberately rejected.
    """
    fr, sr = raw_gallery(full), raw_gallery(subset)
    index = np.empty(len(sr), dtype=np.int64)
    for pid in np.unique(sr):
        src, dst = np.flatnonzero(fr == pid), np.flatnonzero(sr == pid)
        if len(src) != len(dst) or not len(src):
            raise ValueError('Subset cannot be uniquely matched by raw PID/ordinal')
        for key in ('clean/f_rgb', 'clean/f_ir'):
            if not np.array_equal(full[key][src], subset[key][dst]):
                raise ValueError('Within-identity gallery order/features disagree')
        index[dst] = src
    return index


def capped_indices(raw, k, seed):
    if k < 1:
        raise ValueError('max_per_id must be positive')
    rng = np.random.default_rng(seed)
    keep = []
    for pid in np.unique(raw):
        idx = np.flatnonzero(raw == pid)
        keep.extend(idx if len(idx) <= k else rng.choice(idx, k, replace=False))
    return np.sort(np.asarray(keep, dtype=np.int64))


def camera_states(cameras, target):
    cameras = np.asarray(cameras).astype(str)
    if target not in ('cam1', 'cam2'):
        raise ValueError('Unsupported camera mixture')
    return np.where(cameras == target, 'rgb_missing', 'clean')


def select_lambda(table, r10_limit):
    """R-lambda: feasibility per type; stable ascending-lambda tie break."""
    if not table:
        raise ValueError('Empty lambda grid')
    ordered = sorted(table, key=lambda r: r['lambda'])
    def feasible(row):
        return min(row['dR1']) >= -0.5 and min(row['dR10']) >= r10_limit
    feasible_rows = [r for r in ordered if feasible(r)]
    chosen = max(feasible_rows, key=lambda r: np.mean(r['dmAP'])) if feasible_rows else max(ordered, key=lambda r: min(r['dR10']))
    return float(chosen['lambda']), bool(feasible_rows)


def xcam_structure(g, cameras):
    """Same masking and f1F/Delta recomputation as R18 restricted_weights."""
    mask = np.asarray(cameras)[g['J'], None] != np.asarray(cameras)[g['Fi']][None, :]
    x, cf = g['X'].copy(), g['CF'].copy()
    if not len(g['Fi']):
        # gallery_struct uses one sentinel column when no donor exists. It is
        # not a candidate: retain the original sentinel features but expose a
        # genuinely empty candidate matrix for labels and association fitting.
        cf = np.empty((len(g['J']), 0), dtype=cf.dtype)
    if len(g['J']) and len(g['Fi']):
        cf[~mask] = -9.0
        x[:, 0] = cf.max(axis=1)
        x[:, 5] = x[:, 0] - x[:, 3]
    return x, cf, mask


def fit_xcam(episodes, global_fit_items, cameras, contexts, softmax, logistic):
    blocks = []
    for (_, local_items, _, _, g, _), context in zip(episodes, contexts):
        if not len(g['J']):
            continue
        x, cf, mask = xcam_structure(g, np.asarray(cameras)[global_fit_items[local_items]])
        target = g['tm']
        same = g['SF'] if len(g['Fi']) else np.zeros_like(mask)
        labels = (same & mask).any(1)
        blocks.append((x, cf, mask, same, labels, target, context))
    if not blocks:
        raise ValueError('No xcam-refit training rows')
    x = np.concatenate([b[0][b[5]] for b in blocks])
    ctx = np.concatenate([np.tile(b[6], (int(b[5].sum()), 1)) for b in blocks])
    y = np.concatenate([b[4][b[5]] for b in blocks])
    mu, sd = x.mean(0), x.std(0) + 1e-9
    cmu, csd = ctx.mean(0), ctx.std(0) + 1e-9
    head = logistic(C=1.0, max_iter=2000).fit(np.c_[((x-mu)/sd)[:, [0, 5]], (ctx-cmu)/csd], y)
    best = None
    for k in (5, 10, 20, 30, 45, 60, 80, 120):
        values = []
        for _, cf, mask, same, labels, target, _ in blocks:
            rows = labels & target
            if rows.any():
                a = softmax(k*cf[rows], mask[rows])
                values.append(np.log((a*same[rows]).sum(1)+1e-12))
        score = np.concatenate(values).mean() if values else -1e9
        if best is None or score > best[0]:
            best = (score, k)
    return dict(head=head, mu=mu, sd=sd, context_mu=cmu, context_sd=csd, kappa=best[1],
                labels=y, rows=len(y), positives=int(y.sum()))


def refit_weights(g, cameras, context, fit, softmax):
    n, m = len(g['J']), len(g['Fi'])
    if not n or not m:
        return np.zeros((n, m)), np.zeros(n, bool), np.zeros(n)
    x, cf, mask = xcam_structure(g, cameras)
    z = ((x-fit['mu'])/fit['sd'])[:, [0, 5]]
    c = (np.asarray(context)-fit['context_mu'])/fit['context_sd']
    r = fit['head'].predict_proba(np.c_[z, np.tile(c, (n, 1))])[:, 1]
    w = r[:, None]*softmax(fit['kappa']*cf, mask)
    available = mask.any(1)
    w[~available] = 0.0
    return w, available, r


def refit_log(fit):
    return {k: np.asarray(fit[k]).tolist() for k in ('mu','sd','context_mu','context_sd')} | dict(
        kappa_a=fit['kappa'], coef=fit['head'].coef_.tolist(), intercept=fit['head'].intercept_.tolist(),
        rows=fit['rows'], positives=fit['positives'])


def compare_cells(current, reference, tolerance=1e-6):
    """Compare all common cell keys; missing expected common rows are errors."""
    key = ('evalset','mixture','aseed','method')
    def load(path):
        with Path(path).open(encoding='utf-8', newline='') as f:
            out = {}
            for row in csv.DictReader(f):
                k = tuple(row[c] for c in key)
                if k in out:
                    raise ValueError('Duplicate regression cell key')
                out[k] = row
            return out
    a, b = load(current), load(reference)
    common = a.keys() & b.keys()
    errors = []
    if not common:
        return ['No common regression cells']
    for k in sorted(common):
        for field in a[k].keys() & b[k].keys() - set(key):
            try:
                x, y = float(a[k][field]), float(b[k][field])
            except ValueError:
                if a[k][field] != b[k][field]: errors.append(f'{k} {field}: unequal text')
                continue
            if not (np.isnan(x) and np.isnan(y)) and (not np.isfinite(x-y) or abs(x-y)>tolerance):
                errors.append(f'{k} {field}: {x!r} != {y!r}')
    return errors
