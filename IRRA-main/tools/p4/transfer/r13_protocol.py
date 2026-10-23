# Source: R13/r13_scripts/r13_simcf.py (verbatim below).
# Original SHA256: 0af9a18c9c1480af0d58b9c31e39b247dc50439d60673d240aef2a844e58ca00
"""Round 13 = round-12 cross-fitted reference + same-information controls N1-N3 (A-ctxonlyN, CAET-recalN, CAET-hardN),
an optional target-rows-only episode statistic (--ctx_rows tgt), protocol diagnostics, and a guard that refuses the frozen stage-1c conditions.

Round 12 (CROSS-FITTED variant): unified offline simulation of cross-observation evidence transfer rules (frozen features, dev split only).

Protocol (identity-isolated):
  --folds 2 : P-ind halves (fit on one half of the dev identities, evaluate on the other half), 5 split seeds.
  --folds 3 : scale test. Fit on one third; evaluate on each other third separately ("small" galleries, ~159 items)
              and on the union of the two other thirds ("large", ~319 items). Same fitted models for both sizes.
Everything that is fitted or selected uses ONLY fit-fold identities:
  per-availability Platt (B*), LDA link whitening, link/existence/gate heads, association temperatures, and every
  scalar hyper-parameter. Scalars are selected ONCE per (split, fold) with a pooled objective over five simulated
  missingness types on the fit fold (independent 0.2/0.5/0.8, half identity-correlated, identity-level), never
  per evaluation mixture: maximise the mean fit-fold dmAP subject to every type's fit-fold dR1 >= -0.5
  (if no configuration is feasible, maximise the worst type's dR1).
  The only exception is 'GP-same-perMix', which reproduces the ChatGPT audit (gamma chosen per mixture on a fit-fold
  gallery with the same mixture type and seed) and is reported as a fairness reference.

Evaluation mixtures on the eval fold: M1, M3 (B12 assignments), M4 (Bernoulli 0.5), M4c50, M4id (identity-level),
plus the correlation sweep M4c25 / M4c75 when requested.

Diagnostics for IR-only gallery items (cross-availability weights only):
  y_src = 1[some complete item of the same identity is in the gallery]
  wrong mass  = sum of transfer weight on complete items of other identities
  leak        = transfer mass on items with y_src = 0;  coverage = valid mass on items with y_src = 1
  AUC         = how well the total transfer mass separates y_src = 1 from y_src = 0
"""
import argparse, csv, json, math, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--b12', required=True); ap.add_argument('--feats', required=True); ap.add_argument('--assign', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--base', required=True); ap.add_argument('--adapter', required=True)
ap.add_argument('--tag', required=True); ap.add_argument('--folds', type=int, default=2)
ap.add_argument('--split_seeds', type=int, nargs='+', default=[2026, 2027, 2028, 2029, 2030])
ap.add_argument('--assign_seeds', type=int, nargs='+', default=[2026, 2027, 2028])
ap.add_argument('--mixes', nargs='+', default=['M1', 'M3', 'M4', 'M4c50', 'M4id'])
ap.add_argument('--skip_joint', action='store_true'); ap.add_argument('--joint_steps', type=int, default=150)
ap.add_argument('--methods', nargs='*', default=None)
ap.add_argument('--no_idlevel_episodes', action='store_true')  # sensitivity: train without identity-level (c=1) episodes
ap.add_argument('--ctx_rows', choices=['all', 'tgt'], default='all')  # tgt: fitting-episode gallery statistics use only out-of-sample (target-half) IR-only rows
ap.add_argument('--stage1c_final', action='store_true')  # only for the single pre-registered stage-1c run; never used in round 13
ap.add_argument('--link_map', choices=['crossfit', 'train_lda'], default='crossfit')  # train_lda: shrinkage-WCCN link map fitted on the 1076 training identities
ap.add_argument('--train_ir_npz', default=None)  # train_lda only: npz with f_ir (N x d, base clean IR features) and raw_pids (N,)
ap.add_argument('--r_max', type=float, default=1.0)  # numerical safeguard: cap each IR-only item's total transfer mass (stage 1c: 1 - 1e-6)
ap.add_argument('--tie_break', choices=['index', 'bstar'], default='index')  # bstar: equal updated scores keep the B* order (stage 1c)
ap.add_argument('--eval_gallery', choices=['default', 'full'], default='default')  # full (2-fold only): all 478 items, held-out queries only = stress test
args = ap.parse_args()
STAGE1C_MIXES = {'M6c', 'M4c60', 'M9', 'M5e', 'M5f', 'M1g', 'M10'}; STAGE1C_ASSIGN = {5031, 5032, 5033}; STAGE1C_SPLIT = {5031, 5032, 5033, 5034, 5035}
if not args.stage1c_final and (STAGE1C_MIXES & set(args.mixes) or STAGE1C_ASSIGN & set(args.assign_seeds) or STAGE1C_SPLIT & set(args.split_seeds)):
    raise SystemExit('refusing to run: stage-1c mixtures/seeds are frozen and must stay unevaluated before the one-shot stage-1c run')
if args.eval_gallery == 'full' and args.folds != 2: raise SystemExit('--eval_gallery full requires --folds 2')
if args.link_map == 'train_lda' and not args.train_ir_npz: raise SystemExit('--link_map train_lda requires --train_ir_npz')
torch.set_num_threads(1)

SC = ['clean', 'rgb_missing', 'ir_missing', 'rgb_noise_010', 'ir_noise_010', 'rgb_blur_k7', 'ir_blur_k7']
AV = {'clean': 0, 'rgb_noise_010': 0, 'ir_noise_010': 0, 'rgb_blur_k7': 0, 'ir_blur_k7': 0, 'rgb_missing': 1, 'ir_missing': 2}
def l2n(x): return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)
def sig(x): return 1.0 / (1.0 + np.exp(-x))
def lgt(p): p = np.clip(p, 1e-9, 1 - 1e-9); return np.log(p) - np.log1p(-p)

zb = np.load(f'{args.feats}/{args.base}.npz')
def gb(s, k): return zb[f'{s}/{k}'] if f'{s}/{k}' in zb.files else zb[f'clean/{k}']
T = l2n(zb['clean/text_features'].astype(np.float64)); tp, gp, qraw = zb['clean/txt_pids'], zb['clean/img_pids'], zb['clean/query_raw_pids']
POS = tp[:, None] == gp[None, :]
tp2raw = dict(zip(tp.tolist(), qraw.tolist())); item_raw = np.array([tp2raw[p] for p in gp.tolist()])
IRF = {s: (l2n(gb(s, 'f_ir').astype(np.float64)) if AV[s] != 2 else None) for s in SC}
za = np.load(f'{args.feats}/{args.adapter}.npz'); ga = lambda s, k: za[f'{s}/{k}'] if f'{s}/{k}' in za.files else za[f'clean/{k}']
Sst = {}
for s in SC:
    a = AV[s]; fr, fi = ga(s, 'f_rgb').astype(np.float64), ga(s, 'f_ir').astype(np.float64)
    Sst[s] = T @ (l2n(fi) if a == 1 else (l2n(fr) if a == 2 else l2n(fr + fi))).T
FUSED = l2n(gb('clean', 'f_rgb').astype(np.float64) + gb('clean', 'f_ir').astype(np.float64))  # analysis only (near-duplicates)
FR_ST = {s: ga(s, 'f_rgb').astype(np.float64) for s in SC if AV[s] != 1}   # RGB feature of complete-state items (adapter)
FI_CLEAN = ga('clean', 'f_ir').astype(np.float64)                           # clean-route IR feature (computable from the IR image alone)
PROJ_TR = None; WCCN_DIAG = {}

# ---------------------------------------------------------------- basic tools
def fit_affine(x, y, iters=80, l2=1e-6):
    mu, sd = x.mean(), x.std() + 1e-12; X = np.c_[np.ones(len(x)), (x - mu) / sd]; w = np.zeros(2)
    for _ in range(iters):
        p = sig(X @ w); H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(2)
        w = w + np.linalg.solve(H, X.T @ (y - p))
    return (w[0], w[1], mu, sd)
def cal_logit(c, x): return c[0] + c[1] * (x - c[2]) / c[3]

def order_rows(S, S0=None):
    if args.tie_break == 'bstar' and S0 is not None: return np.lexsort((-S0, -S), axis=1)
    return np.argsort(-S, 1, kind='stable')

def ap_r1(S, pos, S0=None):
    o = order_rows(S, S0); m = np.take_along_axis(pos, o, 1)
    ap_ = ((np.cumsum(m, 1) / np.arange(1, S.shape[1] + 1)) * m).sum(1) / np.maximum(m.sum(1), 1)
    return ap_ * 100, m[:, 0] * 100.0

def lda_map(X, ids, shrink=0.5):
    mu = X.mean(0); Sw = np.zeros((X.shape[1], X.shape[1]))
    for u in np.unique(ids):
        Z = X[ids == u] - X[ids == u].mean(0); Sw += Z.T @ Z
    Sw /= len(X); Sw = (1 - shrink) * Sw + shrink * np.trace(Sw) / X.shape[1] * np.eye(X.shape[1])
    ev, U = np.linalg.eigh(Sw); Wm = U @ np.diag(ev ** -0.5) @ U.T
    return lambda Y: l2n((Y - mu) @ Wm)

def comp_fit(C, ids):
    """CAET's existing link fit: event = 'has ANY other item of the same identity' among all items."""
    same = ids[:, None] == ids[None, :]; np.fill_diagonal(same, False); has_sib = same.any(1); best = None
    for kappa in (10, 20, 30, 45, 60, 80):
        for theta in np.linspace(np.percentile(C, 90), np.percentile(C, 99.9), 25):
            L = kappa * (C - theta); np.fill_diagonal(L, -np.inf)
            mx = np.maximum(L.max(1, keepdims=True), 0); E = np.exp(L - mx); den = E.sum(1) + np.exp(-mx[:, 0])
            ll = np.where(has_sib, np.log((E * same).sum(1) / den + 1e-12), np.log(np.exp(-mx[:, 0]) / den + 1e-12)).mean()
            if best is None or ll > best[0]: best = (ll, kappa, theta)
    return best[1], best[2]

def comp_fit_rows(blocks):
    """CAET's link fit with the same event ('has ANY other item of the same identity') but only over the target rows of
    cross-fitted link matrices: blocks = [(C, ids, rowmask), ...]"""
    offd = np.concatenate([C[np.triu_indices(len(C), 1)] for C, _, _ in blocks]); best = None
    pre = []
    for C, ids, rm in blocks:
        same = ids[:, None] == ids[None, :]; np.fill_diagonal(same, False); pre.append((C, same, same.any(1), rm))
    for kappa in (10, 20, 30, 45, 60, 80):
        for theta in np.linspace(np.percentile(offd, 90), np.percentile(offd, 99.9), 25):
            lls = []
            for C, same, has_sib, rm in pre:
                L = kappa * (C - theta); np.fill_diagonal(L, -np.inf)
                mx = np.maximum(L.max(1, keepdims=True), 0); E = np.exp(L - mx); den = E.sum(1) + np.exp(-mx[:, 0])
                ll = np.where(has_sib, np.log((E * same).sum(1) / den + 1e-12), np.log(np.exp(-mx[:, 0]) / den + 1e-12)); lls.append(ll[rm])
            v = np.concatenate(lls).mean()
            if best is None or v > best[0]: best = (v, kappa, theta)
    return best[1], best[2]

def comp_weights_all(C, has, kappa, theta):
    L = kappa * (C - theta); L[~(has[:, None] & has[None, :])] = -np.inf; np.fill_diagonal(L, -np.inf)
    mx = np.maximum(L.max(1, keepdims=True), 0); E = np.exp(L - mx)
    return E / (E.sum(1, keepdims=True) + np.exp(-mx))

def softmax_rows(Z, mask):
    Z = np.where(mask, Z, -np.inf); mx = Z.max(1, keepdims=True); mx = np.where(np.isfinite(mx), mx, 0)
    E = np.exp(Z - mx) * mask; return E / np.maximum(E.sum(1, keepdims=True), 1e-300)

# ---------------------------------------------------------------- gallery structure features
FEAT_NAMES = ['f1F', 'f2F', 'margF', 'f1I', 'f2I', 'f1F-f1I', 'recip', 'log1p_rank', 'fracF']
def gallery_struct(C, av, ids=None):
    """C: link cosine matrix among gallery items (rows/cols = gallery items), av: availability per item.
    Returns per IR-only item: set features X (n_J x 9) and, if ids given, labels and candidate masks."""
    n = len(av); hasIR = av != 2; F = (av == 0) & hasIR; I = av == 1
    J = np.where(I)[0]; Fi = np.where(F)[0]; Ii = np.where(I)[0]
    Cm = C.copy(); np.fill_diagonal(Cm, -9.0)
    CF = Cm[np.ix_(J, Fi)] if len(Fi) else np.full((len(J), 1), -1.0)
    CI = Cm[np.ix_(J, Ii)] if len(Ii) else np.full((len(J), 1), -1.0)
    def top2(M):
        if M.shape[1] >= 2:
            p = -np.sort(-M, 1)[:, :2]; return p[:, 0], p[:, 1]
        return M[:, 0], np.full(M.shape[0], -1.0)
    f1F, f2F = top2(CF); f1I, f2I = top2(CI)
    recip = np.zeros(len(J)); lrank = np.zeros(len(J))
    if len(Fi) and len(Ii):
        kstar = Fi[np.argmax(CF, 1)]
        best_I_of_k = Ii[np.argmax(Cm[np.ix_(kstar, Ii)], 1)]
        recip = (best_I_of_k == J).astype(float)
        rows = Cm[np.ix_(kstar, np.where(hasIR)[0])]
        lrank = np.log1p((rows > Cm[kstar, J][:, None]).sum(1))
    fracF = np.full(len(J), F.mean())
    X = np.c_[f1F, f2F, f1F - f2F, f1I, f2I, f1F - f1I, recip, lrank, fracF]
    out = dict(J=J, Fi=Fi, Ii=Ii, X=X, CF=CF, CI=CI)
    if ids is not None:
        same = ids[:, None] == ids[None, :]; np.fill_diagonal(same, False)
        out['SF'] = same[np.ix_(J, Fi)] if len(Fi) else np.zeros((len(J), 1), bool)
        out['SI'] = same[np.ix_(J, Ii)] if len(Ii) else np.zeros((len(J), 1), bool)
        out['y_src'] = out['SF'].any(1); out['y_same'] = out['SI'].any(1); out['y_any'] = out['y_src'] | out['y_same']
    return out

def gal_ctx(C, av):
    hasIR = av != 2; J = np.where(av == 1)[0]; H = np.where(hasIR)[0]
    if len(J) == 0 or len(H) < 2: return 0.0
    Cm = C[np.ix_(J, H)].copy(); Cm[H[None, :] == J[:, None]] = -9
    nn = H[np.argmax(Cm, 1)]
    return float((av[nn] == 1).mean() - (len(J) - 1) / max(len(H) - 1, 1))

def gal_ctx2(C, av):
    """normalised homophily: rho_hat / (1 - p_I), with p_I the IR-only share among the other IR-bearing items.
    Under identity-level missingness E[rho_hat] ~ a (1 - p_I) (a = top-1 link accuracy), so the ratio ~ a*c does not depend on p_I."""
    hasIR = av != 2; J = np.where(av == 1)[0]; H = np.where(hasIR)[0]
    if len(J) == 0 or len(H) < 2: return 0.0, 0.0
    pI = (len(J) - 1) / max(len(H) - 1, 1)
    return gal_ctx(C, av) / max(1.0 - pI, 0.05), pI

def gal_ctx_rows(C, av, rows=None):
    """as gal_ctx, but only IR-only items with rows[j] = True contribute h_I (the baseline p_I still uses all items)"""
    if rows is None: return gal_ctx(C, av)
    hasIR = av != 2; J = np.where(av == 1)[0]; H = np.where(hasIR)[0]
    if len(J) == 0 or len(H) < 2: return 0.0
    Jr = J[rows[J]]
    if len(Jr) == 0: return 0.0
    Cm = C[np.ix_(Jr, H)].copy(); Cm[H[None, :] == Jr[:, None]] = -9
    nn = H[np.argmax(Cm, 1)]
    return float((av[nn] == 1).mean() - (len(J) - 1) / max(len(H) - 1, 1))

def gal_ctx2_rows(C, av, rows=None):
    if rows is None: return gal_ctx2(C, av)
    hasIR = av != 2; J = np.where(av == 1)[0]; H = np.where(hasIR)[0]
    if len(J) == 0 or len(H) < 2: return 0.0, 0.0
    pI = (len(J) - 1) / max(len(H) - 1, 1)
    return gal_ctx_rows(C, av, rows) / max(1.0 - pI, 0.05), pI

def em_prior(p, pi_tr, iters=100):
    """Saerens-style EM re-estimation of the source-existence base rate on one unlabeled gallery (label-shift assumption)"""
    pi = pi_tr; p = np.clip(p, 1e-6, 1 - 1e-6)
    for _ in range(iters):
        a = (pi / pi_tr) * p; b = ((1 - pi) / (1 - pi_tr)) * (1 - p); r = a / (a + b); new = float(r.mean())
        if abs(new - pi) < 1e-7: break
        pi = new
    return r, pi

def complete_struct(C, av, ids=None):
    """features for complete items (for the same-availability aggregation channel of branch B)."""
    hasIR = av != 2; F = np.where((av == 0) & hasIR)[0]; I = np.where(av == 1)[0]
    Cm = C.copy(); np.fill_diagonal(Cm, -9.0)
    CC = Cm[np.ix_(F, F)]
    p = -np.sort(-CC, 1)[:, :2] if len(F) >= 2 else np.c_[CC.max(1), np.full(len(F), -1.0)]
    g1I = Cm[np.ix_(F, I)].max(1) if len(I) else np.full(len(F), -1.0)
    X = np.c_[p[:, 0], p[:, 1], p[:, 0] - p[:, 1], g1I, np.full(len(F), (av == 0).mean())]
    out = dict(F=F, X=X, CC=CC)
    if ids is not None:
        same = ids[:, None] == ids[None, :]; np.fill_diagonal(same, False)
        out['SC'] = same[np.ix_(F, F)]; out['y'] = out['SC'].any(1)
    return out

if args.link_map == 'train_lda':
    ztr = np.load(args.train_ir_npz); Xtr = l2n(ztr['f_ir'].astype(np.float64)); ytr = np.asarray(ztr['raw_pids'])
    ov = set(np.unique(ytr).tolist()) & set(np.unique(item_raw).tolist())
    if ov: raise SystemExit(f'training identities overlap the development identities: {sorted(ov)[:10]}')
    PROJ_TR = lda_map(Xtr, ytr)
    def _sw_per_item(X, ids):
        tot = 0.0
        for u in np.unique(ids):
            Z = X[ids == u] - X[ids == u].mean(0); tot += float((Z ** 2).sum())
        return tot / len(X)
    Xdev = l2n(gb('clean', 'f_ir').astype(np.float64))
    WCCN_DIAG = dict(n_train_items=int(len(Xtr)), n_train_ids=int(len(np.unique(ytr))), sw_per_item_train=_sw_per_item(Xtr, ytr),
                     sw_per_item_dev=_sw_per_item(Xdev, item_raw))
    WCCN_DIAG['dev_over_train'] = WCCN_DIAG['sw_per_item_dev'] / max(WCCN_DIAG['sw_per_item_train'], 1e-12)
    print('[train_lda] within-class scatter per item (raw clean IR): train %.4f  dev %.4f  ratio dev/train %.3f' % (WCCN_DIAG['sw_per_item_train'], WCCN_DIAG['sw_per_item_dev'], WCCN_DIAG['dev_over_train']), flush=True)

# ---------------------------------------------------------------- availability patterns
def gen_states(rng, ids, kind, pi=0.5, c=0.0):
    """returns state per item: 'clean' or 'rgb_missing'"""
    n = len(ids)
    if kind == 'ind':
        miss = rng.random(n) < pi
    else:  # identity-correlated with probability c
        uid = np.unique(ids); shared = {u: (rng.random() < c, rng.random() < pi) for u in uid}; own = rng.random(n) < pi
        miss = np.array([shared[u][1] if shared[u][0] else own[i] for i, u in enumerate(ids)])
    return np.where(miss, 'rgb_missing', 'clean')

FIT_TYPES = [('ind', 0.2, 0.0), ('ind', 0.5, 0.0), ('ind', 0.8, 0.0), ('cor', 0.5, 0.5), ('cor', 0.5, 1.0)]
if args.no_idlevel_episodes: FIT_TYPES = [('ind', 0.2, 0.0), ('ind', 0.5, 0.0), ('ind', 0.8, 0.0), ('cor', 0.5, 0.5), ('cor', 0.5, 0.25)]

def make_eval_assign(n):
    asg = {}
    for r in csv.DictReader(open(args.assign)):
        asg.setdefault((r['mixture'], int(r['seed'])), {})[int(r['gallery_index'])] = r['state']
    asg = {k: np.array([v[i] for i in range(len(v))]) for k, v in asg.items()}
    for sd in args.assign_seeds:
        r4 = np.random.default_rng(sd + 40000); asg[('M4', sd)] = np.where(r4.random(n) < 0.5, 'rgb_missing', 'clean')
        asg[('M0', sd)] = np.array(['clean'] * n)   # fully complete gallery: isolates generic same-availability re-ranking
        uid = np.unique(item_raw); r6 = np.random.default_rng(sd + 60000)
        asg[('M6a', sd)] = np.where(r6.random(n) < 0.65, 'rgb_missing', 'clean')                         # independent, pi = 0.65
        for name, pi_ in (('M5a', 0.3), ('M5b', 0.7)):                                                    # identity-level, new pi
            r5b = np.random.default_rng(sd + 70000 + int(pi_ * 10)); miss_u = {u: r5b.random() < pi_ for u in uid}
            asg[(name, sd)] = np.array(['rgb_missing' if miss_u[u] else 'clean' for u in item_raw])
        r7 = np.random.default_rng(sd + 80000); q_u = {u: r7.beta(0.5, 0.5) for u in uid}                   # per-identity Beta(0.5,0.5) missing rate
        asg[('M7', sd)] = np.array(['rgb_missing' if r7.random() < q_u[u] else 'clean' for u in item_raw])
        # confirmation-2 mixtures (fresh)
        r6b = np.random.default_rng(sd + 61000); asg[('M6b', sd)] = np.where(r6b.random(n) < 0.35, 'rgb_missing', 'clean')
        r1f = np.random.default_rng(sd + 62000); six = np.array(['clean', 'rgb_missing', 'ir_missing', 'rgb_noise_010', 'ir_noise_010', 'rgb_blur_k7'])
        asg[('M1f', sd)] = six[r1f.integers(0, 6, n)]
        r8 = np.random.default_rng(sd + 81000); q8 = {u: r8.beta(2.0, 2.0) for u in uid}
        asg[('M8', sd)] = np.array(['rgb_missing' if r8.random() < q8[u] else 'clean' for u in item_raw])
        for name, pi_ in (('M5c', 0.85), ('M5d', 0.45)):
            r5c = np.random.default_rng(sd + 71000 + int(pi_ * 100)); mu_ = {u: r5c.random() < pi_ for u in uid}
            asg[(name, sd)] = np.array(['rgb_missing' if mu_[u] else 'clean' for u in item_raw])
        for c, name in [(0.25, 'M4c25'), (0.5, 'M4c50'), (0.75, 'M4c75'), (1.0, 'M4id')]:
            r5 = np.random.default_rng(sd + 50000 + int(c * 10)); uid = np.unique(item_raw)
            shared = {u: (r5.random() < c, r5.random() < 0.5) for u in uid}; own = r5.random(n) < 0.5
            asg[(name, sd)] = np.array(['rgb_missing' if (shared[u][1] if shared[u][0] else own[i]) else 'clean' for i, u in enumerate(item_raw)])
        # ---- stage-1c mixtures: defined 2026-10-23 after all round-12 runs; NEVER evaluated in round 12.
        # Each uses its own RNG stream, so every mixture above is unchanged.
        uid = np.unique(item_raw)
        r6c = np.random.default_rng(sd + 91000); asg[('M6c', sd)] = np.where(r6c.random(n) < 0.55, 'rgb_missing', 'clean')   # independent, pi = 0.55
        r46 = np.random.default_rng(sd + 92000); sh6 = {u: (r46.random() < 0.6, r46.random() < 0.5) for u in uid}; own6 = r46.random(n) < 0.5
        asg[('M4c60', sd)] = np.array(['rgb_missing' if (sh6[u][1] if sh6[u][0] else own6[i]) else 'clean' for i, u in enumerate(item_raw)])  # c = 0.6
        r9 = np.random.default_rng(sd + 93000); q9 = {u: r9.beta(1.0, 1.0) for u in uid}                                    # per-identity Beta(1,1)
        asg[('M9', sd)] = np.array(['rgb_missing' if r9.random() < q9[u] else 'clean' for u in item_raw])
        for name, pi_, off in (('M5e', 0.6, 94000), ('M5f', 0.25, 95000)):                                                  # identity-level, new pi
            r5e = np.random.default_rng(sd + off); m5 = {u: r5e.random() < pi_ for u in uid}
            asg[(name, sd)] = np.array(['rgb_missing' if m5[u] else 'clean' for u in item_raw])
        r1g = np.random.default_rng(sd + 96000)                                                                             # non-uniform six-state
        asg[('M1g', sd)] = six[r1g.choice(6, n, p=[0.3, 0.3, 0.1, 0.1, 0.1, 0.1])]
        # informative boundary probe (not a pass/fail criterion): appearance-correlated missingness, not identity-coded.
        # A random direction u in the base clean RGB feature space; the half of the gallery with the larger projection loses RGB.
        r10 = np.random.default_rng(sd + 97000); u10 = l2n(r10.standard_normal(FR0.shape[1])[None, :])[0]; s10 = FR0 @ u10
        asg[('M10', sd)] = np.where(s10 >= np.quantile(s10, 0.5), 'rgb_missing', 'clean')
    return asg
FR0 = l2n(gb('clean', 'f_rgb').astype(np.float64))   # base clean RGB features, used only by the M10 generator
ASSIGN = make_eval_assign(len(gp))

# ---------------------------------------------------------------- transfer rules
def post_update(P, J, Fi, W):
    """W: n_J x n_F cross weights (row sums <= 1). P' = (1 - sum W) P_j + W P_F"""
    Pn = P.copy()
    if len(J) and len(Fi):
        s = W.sum(1)
        if args.r_max < 1.0:
            W = W * np.minimum(1.0, args.r_max / np.maximum(s, 1e-300))[:, None]; s = W.sum(1)
        Pn[:, J] = (1 - s)[None, :] * P[:, J] + P[:, Fi] @ W.T
    return Pn

def inc_update(P, L, LI, J, Fi, W):
    """branch-increment mixture: P'_j = (1 - sum W) P_j + sum_k W_jk sigma(l_j + l_Fk - l_Ik)"""
    Pn = P.copy(); D = L[:, Fi] - LI[:, Fi]
    for t, j in enumerate(J):
        ks = np.where(W[t] > 1e-6)[0]
        if len(ks) == 0: continue
        Pn[:, j] = (1 - W[t, ks].sum()) * P[:, j] + sig(L[:, [j]] + D[:, ks]) @ W[t, ks]
    return Pn

def avg_update(P, L, LI, J, Fi, W):
    """average branch increment in logit space == precomputed feature increment (see report section on equivalence)"""
    Ln = L.copy()
    if len(J) and len(Fi): Ln[:, J] = L[:, J] + (L[:, Fi] - LI[:, Fi]) @ W.T
    return sig(Ln)

def lat_update(P, Wall, av, gamma):
    """symmetric smoothing restricted to pairs of the SAME availability (IR-only<->IR-only, complete<->complete)"""
    W = Wall * (av[:, None] == av[None, :]) * gamma
    return (1 - W.sum(1))[None, :] * P + P @ W.T

def gp_update(P, Wall, gamma):
    W = Wall * gamma; return (1 - W.sum(1))[None, :] * P + P @ W.T

def latI_update(P, Wall, av, gamma):
    """same-availability smoothing restricted to IR-only <-> IR-only pairs; complete items are never changed"""
    W = Wall * ((av[:, None] == 1) & (av[None, :] == 1)) * gamma
    return (1 - W.sum(1))[None, :] * P + P @ W.T

# ---------------------------------------------------------------- learned link head (identity softmax + triplet) on frozen IR features
def train_link_head(X, ids, dim=128, steps=300, seed=0):
    torch.manual_seed(seed)
    uid, y = np.unique(ids, return_inverse=True)
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y)
    W = torch.nn.Parameter(torch.randn(X.shape[1], dim) * 0.02); Wc = torch.nn.Parameter(torch.randn(dim, len(uid)) * 0.02)
    opt = torch.optim.Adam([W, Wc], lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        Z = torch.nn.functional.normalize(Xt @ W, dim=1); Cn = torch.nn.functional.normalize(Wc, dim=0)
        logits = 16.0 * (Z @ Cn); lid = torch.nn.functional.cross_entropy(logits, yt)
        D = 1 - Z @ Z.T; same = (yt[:, None] == yt[None, :]).float(); eye = torch.eye(len(yt))
        dpos = (D * (same - eye)).max(1).values; dneg = (D + 4 * same).min(1).values
        has = ((same - eye).sum(1) > 0).float()
        ltri = (torch.relu(dpos - dneg + 0.2) * has).sum() / has.sum().clamp(min=1)
        loss = lid + ltri; opt.zero_grad(); loss.backward(); opt.step()
    Wn = W.detach().numpy()
    return lambda Y: l2n(Y @ Wn)

# ---------------------------------------------------------------- branch B (gated two-channel update), torch
class BranchB(torch.nn.Module):
    def __init__(self, gI, gC, ka, kb, kc):
        super().__init__()
        self.WI = torch.nn.Parameter(torch.tensor(gI[0], dtype=torch.float32)); self.bI = torch.nn.Parameter(torch.tensor(gI[1], dtype=torch.float32))
        self.WC = torch.nn.Parameter(torch.tensor(gC[0], dtype=torch.float32)); self.bC = torch.nn.Parameter(torch.tensor(gC[1], dtype=torch.float32))
        self.lk = torch.nn.Parameter(torch.tensor(np.log([ka, kb, kc]), dtype=torch.float32))
    def forward(self, P, g, XIs, XCs):
        """P: queries x items (torch); g: dict from gallery_struct/complete_struct; returns updated P and channel weights."""
        ka, kb, kc = torch.exp(self.lk)
        Pn = P.clone()
        J, Fi, Ii = g['J'], g['Fi'], g['Ii']
        gate = torch.softmax(torch.tensor(XIs, dtype=torch.float32) @ self.WI + self.bI, 1) if len(J) else None
        if len(J):
            CF = torch.tensor(g['CF'], dtype=torch.float32); CI = torch.tensor(g['CI'], dtype=torch.float32)
            A = torch.softmax(ka * CF, 1) if len(Fi) else torch.zeros_like(CF)
            selfmask = torch.tensor(Ii[None, :] == J[:, None])
            B = torch.softmax((kb * CI).masked_fill(selfmask, -1e9), 1) if len(Ii) > 1 else torch.zeros_like(CI)
            cross = P[:, Fi] @ A.T if len(Fi) else torch.zeros(P.shape[0], len(J))
            same = 0.5 * (P[:, J] + (P[:, Ii] @ B.T)) if len(Ii) > 1 else P[:, J]
            Pn[:, J] = gate[:, 0][None, :] * P[:, J] + gate[:, 1][None, :] * cross + gate[:, 2][None, :] * same
        F = g['cF']
        if len(F) >= 2:
            gc = torch.softmax(torch.tensor(XCs, dtype=torch.float32) @ self.WC + self.bC, 1)
            CC = torch.tensor(g['CC'], dtype=torch.float32); eye = torch.eye(len(F), dtype=torch.bool)
            Cw = torch.softmax((kc * CC).masked_fill(eye, -1e9), 1)
            Pn[:, F] = gc[:, 0][None, :] * P[:, F] + gc[:, 1][None, :] * 0.5 * (P[:, F] + P[:, F] @ Cw.T)
        return Pn, gate

# ---------------------------------------------------------------- main loop
uq = np.unique(qraw)
rows_out = []; diag_out = []; perq = {}; sel_log = []; ctxlog = []; nnlog = []; maplog = []
METHODS_ALL = ['B*', 'CAET', 'CAET-g', 'GP-same', 'GP-same-perMix', 'LAT-only', 'A-SRC+LAT', 'THR', 'A-noR', 'A-SRC-post', 'A-SRC-inc', 'A-SRC-avg',
               'A-ANY-post', 'A-SELF-post', 'A-SRCind-post', 'A-SRC-post-head', 'CAET-head', 'GP-same-head',
               'B-sup', 'B-sup-crossonly', 'B-joint', 'B-joint-nopen', 'Oracle-A', 'Random-A', 'LAT-I', 'A-SRC+LAT-I', 'B-sup-noC', 'A-SRC-feat', 'CAET-recal', 'A-SRC-1f', 'A-SRC-2f', 'A-LGctx-post', 'A-SRC-lowe', 'A-SRC-dI', 'A-QV', 'A-SRC-EM', 'A-2f-EM', 'A-SRC-ctx', 'A-2f-ctx', 'A-ctxonly', 'A-2f-ctxn', 'A-ctxonlyN', 'CAET-recalN', 'CAET-hardN', 'A-1f-ctxn', 'A-2f-kap']
METHODS = args.methods or METHODS_ALL
if args.link_map == 'train_lda' and set(METHODS) & {'A-SRC-post-head', 'CAET-head', 'GP-same-head', 'B-sup', 'B-sup-crossonly', 'B-joint', 'B-joint-nopen', 'B-sup-noC', 'GP-same-perMix'}:
    raise SystemExit('train_lda mode supports only the link-map-based rules; remove head / branch-B methods')
t0 = time.time()

def split_folds(ss):
    perm = np.random.default_rng(ss).permutation(uq); return {int(p): i % args.folds for i, p in enumerate(perm)}

for ss in args.split_seeds:
    fold = split_folds(ss)
    qf_all = np.array([fold[int(p)] for p in qraw]); gf_all = np.array([fold[int(p)] for p in item_raw])
    for f in range(args.folds):
        qf, gf = np.where(qf_all == f)[0], np.where(gf_all == f)[0]
        if args.folds == 2 and args.eval_gallery == 'full':
            eval_sets = [('full', np.where(qf_all != f)[0], np.arange(len(gp)))]   # stress test: fit-fold items stay in the gallery as distractors
        elif args.folds == 2:
            eval_sets = [('half', np.where(qf_all != f)[0], np.where(gf_all != f)[0])]
        else:
            others = [o for o in range(args.folds) if o != f]
            eval_sets = [(f'small{o}', np.where(qf_all == o)[0], np.where(gf_all == o)[0]) for o in others]
            eval_sets.append(('large', np.where(qf_all != f)[0], np.where(gf_all != f)[0]))
        rng = np.random.default_rng(ss * 10 + f)
        # ---- calibration (fit fold only)
        cal = {a: fit_affine(Sst[st_][np.ix_(qf, gf)].ravel(), POS[np.ix_(qf, gf)].ravel().astype(float))
               for a, st_ in {0: 'clean', 1: 'rgb_missing', 2: 'ir_missing'}.items()}
        idf = item_raw[gf]; Xf = IRF['clean'][gf]
        if args.link_map == 'train_lda':
            proj = PROJ_TR; head = PROJ_TR                        # learned link head not used in this mode
        else:
            proj = lda_map(Xf, idf)                               # eval-time map (shrinkage WCCN): all fit identities
            head = train_link_head(Xf, idf, seed=ss * 10 + f)     # eval-time ID + triplet link head
        # ---- two-way cross-fitting inside the fit fold: link maps fitted on one half of the fit identities are applied to
        #      the other half, so every link-dependent quantity (CAET kappa/theta, heads, temperatures, thresholds, scalar
        #      selection) is learned from out-of-sample links, as at test time
        rngc = np.random.default_rng(ss * 1000 + f * 17 + 5)
        ufit = np.unique(idf); halfA = set(rngc.permutation(ufit)[: len(ufit) // 2].tolist())
        inA = np.array([u in halfA for u in idf])
        DIRS = []
        if args.link_map == 'train_lda':                         # every development item is out-of-sample for the link map: no cross-fitting
            Ud = PROJ_TR(Xf); DIRS.append(dict(proj=PROJ_TR, head=PROJ_TR, U=Ud, C=Ud @ Ud.T, CH=Ud @ Ud.T, tgt=np.ones(len(gf), bool)))
        else:
            for d, (fitm, tgt) in enumerate(((inA, ~inA), (~inA, inA))):
                pj = lda_map(Xf[fitm], idf[fitm]); hd = train_link_head(Xf[fitm], idf[fitm], seed=ss * 10 + f + 100 * (d + 1))
                Ud = pj(Xf); Hd = hd(Xf)
                DIRS.append(dict(proj=pj, head=hd, U=Ud, C=Ud @ Ud.T, CH=Hd @ Hd.T, tgt=tgt))
        if args.link_map == 'train_lda':   # diagnostics only: NN identity accuracy on strict evaluation galleries under three link maps
            for (ename_, qe_, ge_) in eval_sets:
                if ename_ == 'full': continue
                Xe_ = IRF['clean'][ge_]; ide_ = item_raw[ge_]
                def _acc(U_):
                    Cm_ = U_ @ U_.T; np.fill_diagonal(Cm_, -9); nn_ = np.argmax(Cm_, 1); sm_ = ide_[:, None] == ide_[None, :]; np.fill_diagonal(sm_, False); hs_ = sm_.any(1)
                    return float((ide_[nn_] == ide_)[hs_].mean())
                maplog.append(dict(split=ss, fold=f, evalset=ename_, n_items=len(ge_), acc_raw_cosine=_acc(Xe_), acc_train_id_wccn=_acc(PROJ_TR(Xe_)),
                                   acc_fitfold_wccn=_acc(lda_map(Xf, idf)(Xe_))))
        kap, th = comp_fit_rows([(D['C'], idf, D['tgt']) for D in DIRS])
        for d_, D in enumerate(DIRS):
            Cm_ = D['C'].copy(); np.fill_diagonal(Cm_, -9); nn_ = np.argmax(Cm_, 1); sm_ = idf[:, None] == idf[None, :]; np.fill_diagonal(sm_, False); hs_ = sm_.any(1)
            ok_ = idf[nn_] == idf
            _m1, _m2 = hs_ & ~D['tgt'], hs_ & D['tgt']
            nnlog.append(dict(split=ss, fold=f, dir=d_, acc_fit_rows=float(ok_[_m1].mean()) if _m1.any() else float('nan'), acc_tgt_rows=float(ok_[_m2].mean()) if _m2.any() else float('nan'),
                              n_fit=int((hs_ & ~D['tgt']).sum()), n_tgt=int((hs_ & D['tgt']).sum())))
        kap_h, th_h = comp_fit_rows([(D['CH'], idf, D['tgt']) for D in DIRS])
        # ---- episodes (pooled types x both directions); only target-half IR-only items supply training rows
        EP = []
        for e in range(60):
            kind, pi, c = FIT_TYPES[e % len(FIT_TYPES)]; D = DIRS[e % len(DIRS)]
            keep = rng.random(len(gf)) >= rng.uniform(0.0, 0.3)
            items = np.where(keep)[0]
            st = gen_states(rng, idf[items], kind, pi, c); av = np.array([AV[s] for s in st])
            Ce = D['C'][np.ix_(items, items)]
            g = gallery_struct(Ce, av, idf[items]); gc_ = complete_struct(Ce, av, idf[items])
            g['tm'] = D['tgt'][items][g['J']]; g['dir'] = e % len(DIRS); gc_['tm'] = D['tgt'][items][gc_['F']]
            EP.append((kind, items, st, av, g, gc_))
        Xall = np.concatenate([g['X'][g['tm']] for *_, g, _ in EP]); ysrc = np.concatenate([g['y_src'][g['tm']] for *_, g, _ in EP])
        yany = np.concatenate([g['y_any'][g['tm']] for *_, g, _ in EP]); ysame = np.concatenate([g['y_same'][g['tm']] for *_, g, _ in EP])
        indmask = np.concatenate([[k == 'ind'] * int(g['tm'].sum()) for k, _, _, _, g, _ in EP]).astype(bool)
        mu_x, sd_x = Xall.mean(0), Xall.std(0) + 1e-9; Z = lambda X: (X - mu_x) / sd_x
        src = LogisticRegression(C=1.0, max_iter=2000).fit(Z(Xall), ysrc)
        anyh = LogisticRegression(C=1.0, max_iter=2000).fit(Z(Xall), yany)
        srci = LogisticRegression(C=1.0, max_iter=2000).fit(Z(Xall[indmask]), ysrc[indmask])
        # descriptor-based heads use the raw IR space (fixed unsupervised PCA), identical at training and test time
        pca_mu = Xf.mean(0); _, _, Vt = np.linalg.svd(Xf - pca_mu, full_matrices=False); Pc = Vt[:16].T
        own = np.concatenate([(Xf[items][g['J']][g['tm']] - pca_mu) @ Pc for _, items, _, _, g, _ in EP])
        selfh = LogisticRegression(C=1.0, max_iter=2000).fit(own, ysrc)
        sC = []
        for kind_, items_, st_e, av_e, g_e, _ in EP:
            Wl = comp_weights_all(DIRS[g_e['dir']]['C'][np.ix_(items_, items_)], av_e != 2, kap, th)
            s_ = Wl[np.ix_(g_e['J'], g_e['Fi'])].sum(1) if len(g_e['J']) and len(g_e['Fi']) else np.zeros(len(g_e['J']))
            sC.append(s_[g_e['tm']])
        sC = np.concatenate(sC)
        recal = LogisticRegression(C=1.0, max_iter=2000).fit(lgt(np.clip(sC, 1e-6, 1 - 1e-6))[:, None], ysrc)
        src1 = LogisticRegression(C=1.0, max_iter=2000).fit(Z(Xall)[:, [0]], ysrc)
        src2 = LogisticRegression(C=1.0, max_iter=2000).fit(Z(Xall)[:, [0, 5]], ysrc)
        srcL = LogisticRegression(C=1.0, max_iter=2000).fit(Z(Xall)[:, [0, 2]], ysrc)
        srcD = LogisticRegression(C=1.0, max_iter=2000).fit(Z(Xall)[:, [5]], ysrc)
        pi_tr = float(ysrc.mean())
        RW = lambda g_e, items_: (DIRS[g_e['dir']]['tgt'][items_] if args.ctx_rows == 'tgt' else None)
        ctx_all = np.concatenate([[gal_ctx_rows(DIRS[g_e['dir']]['C'][np.ix_(items_, items_)], av_e, RW(g_e, items_))] * int(g_e['tm'].sum()) for _, items_, _, av_e, g_e, _ in EP])
        mu_ctx, sd_ctx = ctx_all.mean(), ctx_all.std() + 1e-9
        srcC = LogisticRegression(C=1.0, max_iter=2000).fit(np.c_[Z(Xall), (ctx_all - mu_ctx) / sd_ctx], ysrc)
        src2C = LogisticRegression(C=1.0, max_iter=2000).fit(np.c_[Z(Xall)[:, [0, 5]], (ctx_all - mu_ctx) / sd_ctx], ysrc)
        pi_tr2 = pi_tr
        srcCO = LogisticRegression(C=1.0, max_iter=2000).fit(((ctx_all - mu_ctx) / sd_ctx)[:, None], ysrc)
        ctx2_all = np.concatenate([[gal_ctx2_rows(DIRS[g_e['dir']]['C'][np.ix_(items_, items_)], av_e, RW(g_e, items_))] * int(g_e['tm'].sum()) for _, items_, _, av_e, g_e, _ in EP]).reshape(-1, 2)
        mu_c2, sd_c2 = ctx2_all.mean(0), ctx2_all.std(0) + 1e-9
        src2N = LogisticRegression(C=1.0, max_iter=2000).fit(np.c_[Z(Xall)[:, [0, 5]], (ctx2_all - mu_c2) / sd_c2], ysrc)
        ctx_std = (ctx2_all - mu_c2) / sd_c2
        srcCON = LogisticRegression(C=1.0, max_iter=2000).fit(ctx_std, ysrc)                                              # N1
        srcCRN = LogisticRegression(C=1.0, max_iter=2000).fit(np.c_[lgt(np.clip(sC, 1e-6, 1 - 1e-6)), ctx_std], ysrc)    # N2
        # minimality ablations of the main rule (same rows, same regularisation): drop Delta, or drop p_I
        src1N = LogisticRegression(C=1.0, max_iter=2000).fit(np.c_[Z(Xall)[:, [0]], ctx_std], ysrc)                       # f1F + kappa + p_I
        src2K = LogisticRegression(C=1.0, max_iter=2000).fit(np.c_[Z(Xall)[:, [0, 5]], ctx_std[:, [0]]], ysrc)            # f1F + Delta + kappa
        # protocol diagnostics: per-episode statistic with all rows vs target rows only
        for e_, (kind_, items_, _, av_e, g_e, _) in enumerate(EP):
            Ce_ = DIRS[g_e['dir']]['C'][np.ix_(items_, items_)]; ka_all = gal_ctx2(Ce_, av_e); ka_tgt = gal_ctx2_rows(Ce_, av_e, DIRS[g_e['dir']]['tgt'][items_])
            ctxlog.append(dict(split=ss, fold=f, src='episode', ep=e_, type=f'{kind_}{FIT_TYPES[e_ % len(FIT_TYPES)][1]}c{FIT_TYPES[e_ % len(FIT_TYPES)][2]}', dir=g_e['dir'],
                               kappa_all=ka_all[0], kappa_tgt=ka_tgt[0], pI=ka_all[1], n_ironly=int((av_e == 1).sum()), n_tgt_rows=int(g_e['tm'].sum())))
        def lg_ctx(Rloc, g_e, kap_a):
            # own raw-IR descriptor + attention readouts over the complete set and over the other IR-only items (PCA-16 each)
            J_, Fi_, Ii_ = g_e['J'], g_e['Fi'], g_e['Ii']
            own_ = (Rloc[J_] - pca_mu) @ Pc
            if len(Fi_): rF = softmax_rows(kap_a * g_e['CF'], np.ones_like(g_e['CF'], bool)) @ Rloc[Fi_]
            else: rF = np.zeros((len(J_), Rloc.shape[1])) + pca_mu
            if len(Ii_) > 1: rI = softmax_rows(kap_a * g_e['CI'], ~(Ii_[None, :] == J_[:, None])) @ Rloc[Ii_]
            else: rI = np.zeros((len(J_), Rloc.shape[1])) + pca_mu
            pF, pI = (rF - pca_mu) @ Pc, (rI - pca_mu) @ Pc
            return np.c_[own_, pF, pI, own_ * pF, own_ * pI]
        y3 = np.where(ysrc, 1, np.where(ysame, 2, 0))
        gateI = LogisticRegression(C=1.0, max_iter=3000).fit(Z(Xall), y3)
        WI = np.zeros((9, 3)); bI = np.zeros(3)
        for ci, cl in enumerate(gateI.classes_): WI[:, cl] = gateI.coef_[ci] if gateI.coef_.shape[0] == 3 else 0; bI[cl] = gateI.intercept_[ci] if gateI.coef_.shape[0] == 3 else 0
        XC = np.concatenate([gc_['X'][gc_['tm']] for *_, gc_ in EP]); yC = np.concatenate([gc_['y'][gc_['tm']] for *_, gc_ in EP])
        mu_c, sd_c = XC.mean(0), XC.std(0) + 1e-9; ZC = lambda X: (X - mu_c) / sd_c
        gateC = LogisticRegression(C=1.0, max_iter=2000).fit(ZC(XC), yC)
        WC = np.c_[np.zeros(5), gateC.coef_[0]]; bC = np.array([0.0, gateC.intercept_[0]])
        def fit_kappa(key_C, key_S, ykey):
            best = None
            for k in (5, 10, 20, 30, 45, 60, 80, 120):
                ll = []
                for *_, g, _ in EP:
                    m = g[ykey] & g['tm']
                    if m.sum() == 0 or g[key_C].shape[1] < 1: continue
                    Cc = g[key_C][m]; Sm = g[key_S][m]
                    if key_C == 'CI':
                        selfm = g['Ii'][None, :] == g['J'][m][:, None]; Cc = np.where(selfm, -9, Cc)
                    Aw = softmax_rows(k * Cc, np.ones_like(Cc, bool)); ll.append(np.log((Aw * Sm).sum(1) + 1e-12))
                v = np.concatenate(ll).mean() if ll else -1e9
                if best is None or v > best[0]: best = (v, k)
            return best[1]
        ka = fit_kappa('CF', 'SF', 'y_src'); kb = fit_kappa('CI', 'SI', 'y_same')
        lgh = LogisticRegression(C=1.0, max_iter=3000).fit(np.concatenate([lg_ctx(Xf[items_], g_e, ka)[g_e['tm']] for _, items_, _, _, g_e, _ in EP]), ysrc)
        kc_best = None
        for k in (5, 10, 20, 30, 45, 60, 80, 120):
            ll = []
            for *_, gc_ in EP:
                if len(gc_['F']) < 2: continue
                rows = gc_['y'] & gc_['tm']
                if rows.sum() == 0: continue
                CCm = np.where(np.eye(len(gc_['F']), dtype=bool), -9, gc_['CC'])[rows]
                Aw = softmax_rows(k * CCm, np.ones_like(CCm, bool)); ll.append(np.log((Aw * gc_['SC'][rows]).sum(1) + 1e-12))
            v = np.concatenate(ll).mean() if ll else -1e9
            if kc_best is None or v > kc_best[0]: kc_best = (v, k)
        kc = kc_best[1]
        # agreement of the shared-modality (IR-only) calibrated logits: d = l^I_qj - l^I_qk, same-id vs different-id pairs
        LIf = cal_logit(cal[1], Sst['rgb_missing'][np.ix_(qf, gf)])
        sameF = idf[:, None] == idf[None, :]; np.fill_diagonal(sameF, False); iu_ = np.triu_indices(len(gf), 1)
        ps, pd_ = np.where(sameF[iu_])[0], np.where(~sameF[iu_])[0]
        rq = np.random.default_rng(ss * 10 + f + 99); pd_ = rq.choice(pd_, min(len(pd_), 4000), replace=False)
        dS = (LIf[:, iu_[0][ps]] - LIf[:, iu_[1][ps]]).ravel(); dD = (LIf[:, iu_[0][pd_]] - LIf[:, iu_[1][pd_]]).ravel()
        vS, vD = float(np.mean(dS ** 2)), float(np.mean(dD ** 2))
        def qv_update(P, LI, J, Fi, r, A, beta):
            if beta == 0: return post_update(P, J, Fi, r[:, None] * A)
            d2 = (LI[:, J][:, :, None] - LI[:, Fi][:, None, :]) ** 2
            llr = beta * (0.5 * np.log(vD / vS) - 0.5 * d2 * (1.0 / vS - 1.0 / vD))
            E = A[None, :, :] * np.exp(np.minimum(llr, 30.0))                     # q x nJ x nF
            num = (1 - r)[None, :] * P[:, J] + r[None, :] * np.einsum('qjk,qk->qj', E, P[:, Fi])
            den = (1 - r)[None, :] + r[None, :] * E.sum(2)
            Pn = P.copy(); Pn[:, J] = num / den; return Pn
        # ---- helpers to build a scored gallery (any item set + states) for queries qi
        def build(qi, items, st, proj_=None, head_=None):
            proj_ = proj_ or proj; head_ = head_ or head
            av = np.array([AV[s] for s in st]); L = np.zeros((len(qi), len(items)))
            for s in np.unique(st):
                cols = np.where(st == s)[0]; L[:, cols] = cal_logit(cal[AV[s]], Sst[s][np.ix_(qi, items[cols])])
            LI = cal_logit(cal[1], Sst['rgb_missing'][np.ix_(qi, items)])  # each item's IR-only view (analysis + increments)
            X = np.zeros((len(items), Xf.shape[1])); hasIR = av != 2
            for t, (j, s) in enumerate(zip(items, st)):
                if hasIR[t]: X[t] = IRF[s][j]
            U = np.where(hasIR[:, None], proj_(X), 0); C = U @ U.T
            H = np.where(hasIR[:, None], head_(X), 0); CH = H @ H.T
            return dict(L=L, P=sig(L), LI=LI, av=av, has=hasIR, C=C, CH=CH, items=items, st=st, qi=qi, U=U, Xraw=X)
        def methods_on(G, ids_for_diag=None, chosen=None, which=None):
            """returns {method: (P_new, Wcross or None)}"""
            P, L, LI, av, has, C, CH = G['P'], G['L'], G['LI'], G['av'], G['has'], G['C'], G['CH']
            g = gallery_struct(C, av, ids_for_diag); J, Fi = g['J'], g['Fi']
            out = {'B*': (P, np.zeros((len(J), len(Fi))))}
            Wall = comp_weights_all(C, has, kap, th)
            Wc = Wall[np.ix_(J, Fi)] if len(J) and len(Fi) else np.zeros((len(J), len(Fi)))
            want = lambda m: (which is None or m in which) and m in METHODS
            if want('CAET'): out['CAET'] = (post_update(P, J, Fi, Wc), Wc)
            if want('CAET-g'): gm = chosen['CAET-g']; out['CAET-g'] = (post_update(P, J, Fi, Wc * gm), Wc * gm)
            if want('GP-same'): gm = chosen['GP-same']; out['GP-same'] = (gp_update(P, Wall, gm), Wc * gm)
            if want('GP-same-perMix') and chosen.get('GP-same-perMix') is not None:
                gm = chosen['GP-same-perMix']; out['GP-same-perMix'] = (gp_update(P, Wall, gm), Wc * gm)
            if want('LAT-only'): out['LAT-only'] = (lat_update(P, Wall, av, chosen['LAT-only']), np.zeros((len(J), len(Fi))))
            if want('A-SRC+LAT') and len(J) and len(Fi):
                A_ = softmax_rows(ka * g['CF'], np.ones_like(g['CF'], bool)); W_ = src.predict_proba(Z(g['X']))[:, 1][:, None] * A_
                out['A-SRC+LAT'] = (lat_update(post_update(P, J, Fi, W_), Wall, av, chosen['A-SRC+LAT']), W_)
            if want('LAT-I'): out['LAT-I'] = (latI_update(P, Wall, av, chosen['LAT-I']), np.zeros((len(J), len(Fi))))
            if want('A-SRC+LAT-I') and len(J) and len(Fi):
                A_ = softmax_rows(ka * g['CF'], np.ones_like(g['CF'], bool)); W_ = src.predict_proba(Z(g['X']))[:, 1][:, None] * A_
                out['A-SRC+LAT-I'] = (latI_update(post_update(P, J, Fi, W_), Wall, av, chosen['A-SRC+LAT-I']), W_)
            if want('B-sup-noC') and len(J) and len(Fi):
                # three-way supervised gate (keep / cross / same), complete items never changed
                Zg = Z(g['X']) @ WI + bI; gt = np.exp(Zg - Zg.max(1, keepdims=True)); gt /= gt.sum(1, keepdims=True)
                A_ = softmax_rows(ka * g['CF'], np.ones_like(g['CF'], bool)); cross = P[:, Fi] @ A_.T
                Ii = g['Ii']; selfm = Ii[None, :] == J[:, None]
                if len(Ii) > 1:
                    B_ = softmax_rows(kb * g['CI'], ~selfm); same = 0.5 * (P[:, J] + P[:, Ii] @ B_.T)
                else:
                    same = P[:, J]
                Pn = P.copy(); Pn[:, J] = gt[:, 0][None, :] * P[:, J] + gt[:, 1][None, :] * cross + gt[:, 2][None, :] * same
                out['B-sup-noC'] = (Pn, gt[:, 1][:, None] * A_)
            if want('THR') and len(J) and len(Fi):
                tau, dl = chosen['THR']; W = np.zeros((len(J), len(Fi))); k1 = np.argmax(g['CF'], 1)
                ok = (g['X'][:, 0] > tau) & (g['X'][:, 2] > dl); W[np.arange(len(J))[ok], k1[ok]] = 1.0
                out['THR'] = (post_update(P, J, Fi, W), W)
            if len(J) and len(Fi):
                A = softmax_rows(ka * g['CF'], np.ones_like(g['CF'], bool))
                rS = src.predict_proba(Z(g['X']))[:, 1]
                if want('A-noR'): out['A-noR'] = (post_update(P, J, Fi, A), A)
                if want('A-SRC-post'): W = rS[:, None] * A; out['A-SRC-post'] = (post_update(P, J, Fi, W), W)
                if want('A-SRC-inc'): W = rS[:, None] * A; out['A-SRC-inc'] = (inc_update(P, L, LI, J, Fi, W), W)
                if want('A-SRC-EM'):
                    rE, _ = em_prior(rS, pi_tr); W = rE[:, None] * A; out['A-SRC-EM'] = (post_update(P, J, Fi, W), W)
                if want('A-2f-EM'):
                    rE, _ = em_prior(src2.predict_proba(Z(g['X'])[:, [0, 5]])[:, 1], pi_tr2); W = rE[:, None] * A; out['A-2f-EM'] = (post_update(P, J, Fi, W), W)
                if want('A-SRC-ctx') or want('A-2f-ctx'):
                    cz = (gal_ctx(C, av) - mu_ctx) / sd_ctx
                    if want('A-SRC-ctx'):
                        W = srcC.predict_proba(np.c_[Z(g['X']), np.full(len(J), cz)])[:, 1][:, None] * A; out['A-SRC-ctx'] = (post_update(P, J, Fi, W), W)
                    if want('A-2f-ctx'):
                        W = src2C.predict_proba(np.c_[Z(g['X'])[:, [0, 5]], np.full(len(J), cz)])[:, 1][:, None] * A; out['A-2f-ctx'] = (post_update(P, J, Fi, W), W)
                if want('A-2f-ctxn'):
                    c2 = (np.array(gal_ctx2(C, av)) - mu_c2) / sd_c2
                    W = src2N.predict_proba(np.c_[Z(g['X'])[:, [0, 5]], np.tile(c2, (len(J), 1))])[:, 1][:, None] * A; out['A-2f-ctxn'] = (post_update(P, J, Fi, W), W)
                if want('A-ctxonly'):
                    cz1 = (gal_ctx(C, av) - mu_ctx) / sd_ctx
                    W = srcCO.predict_proba(np.full((len(J), 1), cz1))[:, 1][:, None] * A; out['A-ctxonly'] = (post_update(P, J, Fi, W), W)
                if want('A-ctxonlyN'):
                    c2 = (np.array(gal_ctx2(C, av)) - mu_c2) / sd_c2
                    rr = srcCON.predict_proba(np.tile(c2, (len(J), 1)))[:, 1]
                    W = rr[:, None] * A; out['A-ctxonlyN'] = (post_update(P, J, Fi, W), W)
                if want('CAET-recalN'):
                    c2 = (np.array(gal_ctx2(C, av)) - mu_c2) / sd_c2
                    rr = srcCRN.predict_proba(np.c_[lgt(np.clip(Wc.sum(1), 1e-6, 1 - 1e-6)), np.tile(c2, (len(J), 1))])[:, 1]
                    W = rr[:, None] * A; out['CAET-recalN'] = (post_update(P, J, Fi, W), W)
                if want('A-1f-ctxn') or want('A-2f-kap'):
                    c2 = (np.array(gal_ctx2(C, av)) - mu_c2) / sd_c2
                    if want('A-1f-ctxn'):
                        W = src1N.predict_proba(np.c_[Z(g['X'])[:, [0]], np.tile(c2, (len(J), 1))])[:, 1][:, None] * A; out['A-1f-ctxn'] = (post_update(P, J, Fi, W), W)
                    if want('A-2f-kap'):
                        W = src2K.predict_proba(np.c_[Z(g['X'])[:, [0, 5]], np.full((len(J), 1), c2[0])])[:, 1][:, None] * A; out['A-2f-kap'] = (post_update(P, J, Fi, W), W)
                if want('CAET-hardN'):
                    W = Wc * float(gal_ctx2(C, av)[0] < chosen['CAET-hardN'])
                    out['CAET-hardN'] = (post_update(P, J, Fi, W), W)
                if want('A-QV'): out['A-QV'] = (qv_update(P, LI, J, Fi, rS, A, chosen['A-QV']), rS[:, None] * A)
                if want('A-SRC-avg'): W = rS[:, None] * A; out['A-SRC-avg'] = (avg_update(P, L, LI, J, Fi, W), W)
                if want('CAET-recal'):
                    s_ = Wc.sum(1); rr = recal.predict_proba(lgt(np.clip(s_, 1e-6, 1 - 1e-6))[:, None])[:, 1]
                    W = rr[:, None] * (Wc / np.maximum(s_[:, None], 1e-12)); out['CAET-recal'] = (post_update(P, J, Fi, W), W)
                if want('A-SRC-1f'): W = src1.predict_proba(Z(g['X'])[:, [0]])[:, 1][:, None] * A; out['A-SRC-1f'] = (post_update(P, J, Fi, W), W)
                if want('A-SRC-2f'): W = src2.predict_proba(Z(g['X'])[:, [0, 5]])[:, 1][:, None] * A; out['A-SRC-2f'] = (post_update(P, J, Fi, W), W)
                if want('A-SRC-lowe'): W = srcL.predict_proba(Z(g['X'])[:, [0, 2]])[:, 1][:, None] * A; out['A-SRC-lowe'] = (post_update(P, J, Fi, W), W)
                if want('A-SRC-dI'): W = srcD.predict_proba(Z(g['X'])[:, [5]])[:, 1][:, None] * A; out['A-SRC-dI'] = (post_update(P, J, Fi, W), W)
                if want('A-LGctx-post'):
                    W = lgh.predict_proba(lg_ctx(G['Xraw'], g, ka))[:, 1][:, None] * A; out['A-LGctx-post'] = (post_update(P, J, Fi, W), W)
                if want('A-SRC-feat'):
                    # feature-level alternative: impute the missing RGB feature as the a-weighted mean of the sources' RGB features,
                    # re-score the IR-only item with the complete branch and its calibrator, then mix with the same r_j
                    its, sts = G['items'], G['st']
                    FRs = np.stack([FR_ST[sts[k]][its[k]] for k in Fi]); fhat = A @ FRs
                    fused = l2n(fhat + FI_CLEAN[its[J]]); lhat = cal_logit(cal[0], T[G['qi']] @ fused.T)
                    W = rS[:, None] * A; Pn = P.copy(); Pn[:, J] = (1 - rS)[None, :] * P[:, J] + rS[None, :] * sig(lhat)
                    out['A-SRC-feat'] = (Pn, W)
                if want('A-ANY-post'): W = anyh.predict_proba(Z(g['X']))[:, 1][:, None] * A; out['A-ANY-post'] = (post_update(P, J, Fi, W), W)
                if want('A-SRCind-post'): W = srci.predict_proba(Z(g['X']))[:, 1][:, None] * A; out['A-SRCind-post'] = (post_update(P, J, Fi, W), W)
                if want('A-SELF-post'):
                    W = selfh.predict_proba((G['Xraw'][J] - pca_mu) @ Pc)[:, 1][:, None] * A; out['A-SELF-post'] = (post_update(P, J, Fi, W), W)
                if want('A-SRC-post-head'):
                    gh = gallery_struct(CH, av); Ah = softmax_rows(ka * gh['CF'], np.ones_like(gh['CF'], bool))
                    W = rS[:, None] * Ah; out['A-SRC-post-head'] = (post_update(P, J, Fi, W), W)
                if want('CAET-head'):
                    Wh = comp_weights_all(CH, has, kap_h, th_h)[np.ix_(J, Fi)]; out['CAET-head'] = (post_update(P, J, Fi, Wh), Wh)
                if want('GP-same-head'):
                    gm = chosen['GP-same-head']; Wallh = comp_weights_all(CH, has, kap_h, th_h)
                    out['GP-same-head'] = (gp_update(P, Wallh, gm), Wallh[np.ix_(J, Fi)] * gm)
                if want('Oracle-A') and ids_for_diag is not None:
                    Sm = g['SF'].astype(float); W = Sm / np.maximum(Sm.sum(1, keepdims=True), 1); out['Oracle-A'] = (post_update(P, J, Fi, W), W)
                if want('Random-A'):
                    rr = np.random.default_rng(123); Ar = rr.dirichlet(np.ones(len(Fi)), size=len(J)); W = rS[:, None] * Ar
                    out['Random-A'] = (post_update(P, J, Fi, W), W)
            if any(want(m) for m in ('B-sup', 'B-sup-crossonly', 'B-joint', 'B-joint-nopen')):
                gcs = complete_struct(C, av); g2 = dict(g); g2['cF'] = gcs['F']; g2['CC'] = gcs['CC']
                XIs = Z(g['X']) if len(J) else np.zeros((0, 9)); XCs = ZC(gcs['X']) if len(gcs['F']) else np.zeros((0, 5))
                Pt = torch.tensor(P, dtype=torch.float32)
                for name, mdl in (('B-sup', Bsup), ('B-joint', Bjoint), ('B-joint-nopen', Bjoint_np)):
                    if not want(name) or mdl is None: continue
                    with torch.no_grad():
                        Pn, gate = mdl(Pt, g2, XIs, XCs)
                    Wx = (gate[:, 1].numpy()[:, None] * softmax_rows(float(torch.exp(mdl.lk[0])) * g['CF'], np.ones_like(g['CF'], bool))) if len(J) and len(Fi) else np.zeros((len(J), len(Fi)))
                    out[name] = (Pn.numpy().astype(np.float64), Wx)
                if want('B-sup-crossonly') and Bsup is not None and len(J) and len(Fi):
                    with torch.no_grad():
                        gate = torch.softmax(torch.tensor(XIs, dtype=torch.float32) @ Bsup.WI + Bsup.bI, 1).numpy()
                    W = gate[:, 1][:, None] * softmax_rows(ka * g['CF'], np.ones_like(g['CF'], bool))
                    out['B-sup-crossonly'] = (post_update(P, J, Fi, W), W)
            return out, g
        # ---- fit-fold simulated galleries for selection (one per pooled type)
        SELG = []
        for (kind, pi, c) in FIT_TYPES:
            st = gen_states(rng, idf, kind, pi, c); per = []
            for D in DIRS:
                Gs_ = build(qf, gf, st, proj_=D['proj'], head_=D['head']); Gs_['rows'] = D['tgt']
                per.append((Gs_, np.isin(qraw[qf], idf[D['tgt']])))
            SELG.append(per)
        pf = POS[np.ix_(qf, gf)]
        base_fit = [[ap_r1(G['P'], pf) for G, _ in per] for per in SELG]
        def pooled_select(name, grid, fn):
            best = None
            for cfg in grid:
                dm, dr = [], []
                for per, bases in zip(SELG, base_fit):
                    A_, R_, BA, BR = [], [], [], []
                    for (G, qm), (b_ap, b_r1) in zip(per, bases):
                        a_, r_ = ap_r1(fn(G, cfg), pf, G['P']); A_.append(a_[qm]); R_.append(r_[qm]); BA.append(b_ap[qm]); BR.append(b_r1[qm])
                    dm.append(np.concatenate(A_).mean() - np.concatenate(BA).mean()); dr.append(np.concatenate(R_).mean() - np.concatenate(BR).mean())
                feas = min(dr) >= -0.5; key = (feas, np.mean(dm) if feas else min(dr))
                if best is None or key > best[0]: best = (key, cfg)
            sel_log.append(dict(split=ss, fold=f, method=name, chosen=str(best[1]), feasible=bool(best[0][0])))
            return best[1]
        GAM = [0.0, 0.25, 0.5, 0.75, 1.0]
        chosen = {}
        def caet_g(G, gm):
            g = gallery_struct(G['C'], G['av']); Wall = comp_weights_all(G['C'], G['has'], kap, th)
            return post_update(G['P'], g['J'], g['Fi'], Wall[np.ix_(g['J'], g['Fi'])] * gm) if len(g['J']) and len(g['Fi']) else G['P']
        chosen['CAET-g'] = pooled_select('CAET-g', GAM, caet_g)
        chosen['GP-same'] = pooled_select('GP-same', GAM, lambda G, gm: gp_update(G['P'], comp_weights_all(G['C'], G['has'], kap, th), gm))
        chosen['LAT-only'] = pooled_select('LAT-only', GAM, lambda G, gm: lat_update(G['P'], comp_weights_all(G['C'], G['has'], kap, th), G['av'], gm))
        def asrc_lat(G, gm):
            g = gallery_struct(G['C'], G['av']); Wall = comp_weights_all(G['C'], G['has'], kap, th)
            if not (len(g['J']) and len(g['Fi'])): return lat_update(G['P'], Wall, G['av'], gm)
            A_ = softmax_rows(ka * g['CF'], np.ones_like(g['CF'], bool)); W_ = src.predict_proba(Z(g['X']))[:, 1][:, None] * A_
            return lat_update(post_update(G['P'], g['J'], g['Fi'], W_), Wall, G['av'], gm)
        chosen['A-SRC+LAT'] = pooled_select('A-SRC+LAT', GAM, asrc_lat) if 'A-SRC+LAT' in METHODS else 0.0
        if 'LAT-I' in METHODS:
            chosen['LAT-I'] = pooled_select('LAT-I', GAM, lambda G, gm: latI_update(G['P'], comp_weights_all(G['C'], G['has'], kap, th), G['av'], gm))
        def asrc_latI(G, gm):
            g = gallery_struct(G['C'], G['av']); Wall = comp_weights_all(G['C'], G['has'], kap, th)
            if not (len(g['J']) and len(g['Fi'])): return latI_update(G['P'], Wall, G['av'], gm)
            A_ = softmax_rows(ka * g['CF'], np.ones_like(g['CF'], bool)); W_ = src.predict_proba(Z(g['X']))[:, 1][:, None] * A_
            return latI_update(post_update(G['P'], g['J'], g['Fi'], W_), Wall, G['av'], gm)
        if 'A-SRC+LAT-I' in METHODS: chosen['A-SRC+LAT-I'] = pooled_select('A-SRC+LAT-I', GAM, asrc_latI)
        chosen['GP-same-head'] = pooled_select('GP-same-head', GAM, lambda G, gm: gp_update(G['P'], comp_weights_all(G['CH'], G['has'], kap_h, th_h), gm))
        same_cos = np.concatenate([D['C'][np.ix_(np.where(D['tgt'])[0], np.where(D['tgt'])[0])][np.triu(idf[D['tgt']][:, None] == idf[D['tgt']][None, :], 1)] for D in DIRS])
        taus = list(np.quantile(same_cos, [0.1, 0.3, 0.5, 0.7, 0.9])); dls = [0.0, 0.02, 0.05, 0.1, 0.2]
        def thr_fn(G, cfg):
            tau, dl = cfg; g = gallery_struct(G['C'], G['av'])
            if not (len(g['J']) and len(g['Fi'])): return G['P']
            W = np.zeros((len(g['J']), len(g['Fi']))); k1 = np.argmax(g['CF'], 1); ok = (g['X'][:, 0] > tau) & (g['X'][:, 2] > dl)
            W[np.arange(len(g['J']))[ok], k1[ok]] = 1.0; return post_update(G['P'], g['J'], g['Fi'], W)
        chosen['THR'] = pooled_select('THR', [(t, d) for t in taus for d in dls], thr_fn)
        def aqv_fn(G, beta):
            g = gallery_struct(G['C'], G['av'])
            if not (len(g['J']) and len(g['Fi'])): return G['P']
            A_ = softmax_rows(ka * g['CF'], np.ones_like(g['CF'], bool)); r_ = src.predict_proba(Z(g['X']))[:, 1]
            return qv_update(G['P'], G['LI'], g['J'], g['Fi'], r_, A_, beta)
        if 'A-QV' in METHODS: chosen['A-QV'] = pooled_select('A-QV', [0.0, 0.25, 0.5, 1.0], aqv_fn)
        if 'CAET-hardN' in METHODS:
            def hardN_fn(G, threshold):
                g = gallery_struct(G['C'], G['av'])
                if not len(g['J']) or not len(g['Fi']): return G['P']
                Wall = comp_weights_all(G['C'], G['has'], kap, th)
                kap_hat = gal_ctx2_rows(G['C'], G['av'], G.get('rows') if args.ctx_rows == 'tgt' else None)[0]
                W = Wall[np.ix_(g['J'], g['Fi'])] * float(kap_hat < threshold)
                return post_update(G['P'], g['J'], g['Fi'], W)
            chosen['CAET-hardN'] = pooled_select('CAET-hardN', [0.0, 0.25, 0.5, 0.75, 1.0], hardN_fn)
        # ---- branch B models
        Bsup = BranchB((WI, bI), (WC, bC), ka, kb, kc); Bjoint = None; Bjoint_np = None
        if not args.skip_joint and any(m in METHODS for m in ('B-joint', 'B-joint-nopen')):
            def train_joint(penalty):
                mdl = BranchB((WI, bI), (WC, bC), ka, kb, kc); opt = torch.optim.Adam(mdl.parameters(), lr=0.02)
                pfT = torch.tensor(pf); prep = []
                for per in SELG:
                    for (G, qm), D in zip(per, DIRS):
                        g = gallery_struct(G['C'], G['av'], idf); gcs = complete_struct(G['C'], G['av'], idf); g2 = dict(g); g2['cF'] = gcs['F']; g2['CC'] = gcs['CC']
                        y3l = np.where(g['y_src'], 1, np.where(g['y_same'], 2, 0)); tmJ = D['tgt'][g['J']]
                        prep.append((torch.tensor(G['P'], dtype=torch.float32), g2, Z(g['X']) if len(g['J']) else np.zeros((0, 9)),
                                     ZC(gcs['X']) if len(gcs['F']) else np.zeros((0, 5)), torch.tensor(y3l), torch.tensor(qm), torch.tensor(tmJ)))
                def rank_loss(Pm, qm):
                    S = torch.log(Pm.clamp(1e-6, 1 - 1e-6)) - torch.log1p(-Pm.clamp(1e-6, 1 - 1e-6))
                    lse_all = torch.logsumexp(S, 1); lse_pos = torch.logsumexp(S.masked_fill(~pfT, -1e9), 1)
                    return (lse_all - lse_pos)[qm]
                base_rl = [rank_loss(e_[0], e_[5]).detach() for e_ in prep]
                for step in range(args.joint_steps):
                    k = step % len(prep); Pm, g2, XIs, XCs, y3l, qm, tmJ = prep[k]
                    Pn, gate = mdl(Pm, g2, XIs, XCs); rl = rank_loss(Pn, qm)
                    loss = rl.mean()
                    if penalty: loss = loss + 1.0 * torch.relu(rl - base_rl[k]).mean()
                    if gate is not None and int(tmJ.sum()): loss = loss + 0.5 * torch.nn.functional.nll_loss(torch.log(gate[tmJ] + 1e-9), y3l[tmJ])
                    opt.zero_grad(); loss.backward(); opt.step()
                return mdl
            if 'B-joint' in METHODS: Bjoint = train_joint(True)
            if 'B-joint-nopen' in METHODS: Bjoint_np = train_joint(False)
        # ---- evaluation
        for (ename, qe, ge) in eval_sets:
            pe = POS[np.ix_(qe, ge)]; ide = item_raw[ge]
            for mx in args.mixes:
                for asd in args.assign_seeds:
                    st = ASSIGN[(mx, asd)][ge]; G = build(qe, ge, st)
                    G['U'] = np.where(G['has'][:, None], proj(np.stack([IRF[s][j] if AV[s] != 2 else np.zeros(Xf.shape[1]) for j, s in zip(ge, st)])), 0)
                    if 'GP-same-perMix' in METHODS:  # ChatGPT-audit selection: same mixture type and seed on the fit fold, mAP only
                        stf = ASSIGN[(mx, asd)][gf]; Gf = build(qf, gf, stf); Wf = comp_weights_all(Gf['C'], Gf['has'], kap, th)
                        chosen['GP-same-perMix'] = max(GAM, key=lambda gm: ap_r1(gp_update(Gf['P'], Wf, gm), pf)[0].mean())
                    res, g = methods_on(G, ide, chosen)
                    k2_ = gal_ctx2(G['C'], G['av']); ctxlog.append(dict(split=ss, fold=f, src='eval', ep=-1, type=f'{ename}:{mx}:{asd}', dir=-1, kappa_all=k2_[0], kappa_tgt=np.nan, pI=k2_[1],
                                                                      n_ironly=int((G['av'] == 1).sum()), n_tgt_rows=-1))
                    b_ap, b_r1 = ap_r1(res['B*'][0], pe)
                    ysrc_e = g['y_src'] if len(g['J']) else np.zeros(0, bool)
                    top0 = np.argmax(res['B*'][0], 1); irtop = G['av'][top0] == 1
                    # near-duplicate level of each query's identity: max fused cosine among its items in this gallery (analysis only)
                    for name, (Pn, W) in res.items():
                        a_, r_ = ap_r1(Pn, pe, res['B*'][0])
                        top1 = order_rows(Pn, res['B*'][0])[:, 0]
                        rows_out.append(dict(split=ss, fold=f, evalset=ename, mixture=mx, aseed=asd, method=name,
                                             mAP=a_.mean(), R1=r_.mean(), dmAP=a_.mean() - b_ap.mean(), dR1=r_.mean() - b_r1.mean(),
                                             disp_c=float((top1 != top0)[~irtop].mean() * 100) if (~irtop).any() else np.nan,
                                             share_irtop=float(irtop.mean() * 100),
                                             dR1_irtop=float((r_ - b_r1)[irtop].mean()) if irtop.any() else np.nan,
                                             dR1_ctop=float((r_ - b_r1)[~irtop].mean()) if (~irtop).any() else np.nan))
                        for qi_, d_ap, d_r1 in zip(qe, a_ - b_ap, r_ - b_r1):
                            perq.setdefault((ename.rstrip('012') if ename.startswith('small') else ename, mx, name), {}).setdefault(int(qi_), []).append((d_ap, d_r1))
                        if len(g['J']) and len(g['Fi']) and W is not None and W.shape == g['SF'].shape:
                            tot = W.sum(1); val = (W * g['SF']).sum(1); wrong = tot - val
                            auc = np.nan
                            if 0 < ysrc_e.sum() < len(ysrc_e):
                                from sklearn.metrics import roc_auc_score
                                auc = roc_auc_score(ysrc_e, tot + 1e-12 * np.random.default_rng(0).random(len(tot)))
                            diag_out.append(dict(split=ss, fold=f, evalset=ename, mixture=mx, aseed=asd, method=name,
                                                 n_ironly=len(g['J']), frac_src=float(ysrc_e.mean()),
                                                 wrong_mass=float(wrong.mean()), leak=float(tot[~ysrc_e].mean()) if (~ysrc_e).any() else np.nan,
                                                 coverage=float(val[ysrc_e].mean()) if ysrc_e.any() else np.nan,
                                                 cover50=float((val[ysrc_e] >= 0.5).mean()) if ysrc_e.any() else np.nan, auc=auc))
        print(f'[{args.tag}] split {ss} fold {f} done  kappa_a={ka} kb={kb} kc={kc}  chosen={ {k: v for k, v in chosen.items() if k != "GP-same-perMix"} }  {time.time() - t0:.0f}s', flush=True)

# ---------------------------------------------------------------- summaries
import collections
def summarize():
    out = []
    keys = sorted(set((r['evalset'] if not r['evalset'].startswith('small') else 'small', r['mixture'], r['method']) for r in rows_out))
    for es, mx, m in keys:
        rr = [r for r in rows_out if (r['evalset'] if not r['evalset'].startswith('small') else 'small') == es and r['mixture'] == mx and r['method'] == m]
        dm = np.array([r['dmAP'] for r in rr]); dr = np.array([r['dR1'] for r in rr])
        # identity-cluster bootstrap over per-query mean differences (evaluation-sampling uncertainty only)
        pq = perq.get((es, mx, m), {}); qs = sorted(pq); md = np.array([np.mean([v[0] for v in pq[q]]) for q in qs]); mr = np.array([np.mean([v[1] for v in pq[q]]) for q in qs])
        qid = np.array([qraw[q] for q in qs]); uid_ = np.unique(qid); idx = [np.where(qid == u)[0] for u in uid_]
        sums_m = np.array([md[i].sum() for i in idx]); sums_r = np.array([mr[i].sum() for i in idx]); cnt = np.array([len(i) for i in idx])
        rb = np.random.default_rng(2026); bm, br = [], []
        for _ in range(2000):
            x = rb.integers(0, len(uid_), len(uid_)); bm.append(sums_m[x].sum() / cnt[x].sum()); br.append(sums_r[x].sum() / cnt[x].sum())
        dd = [d for d in diag_out if (d['evalset'] if not d['evalset'].startswith('small') else 'small') == es and d['mixture'] == mx and d['method'] == m]
        dg = {k: float(np.nanmean([d[k] for d in dd])) if dd else np.nan for k in ('frac_src', 'wrong_mass', 'leak', 'coverage', 'cover50', 'auc')}
        out.append(dict(evalset=es, mixture=mx, method=m, n_cells=len(rr), mAP=dm.mean() + np.mean([r['mAP'] - r['dmAP'] for r in rr]),
                        dmAP=dm.mean(), dmAP_ci_lo=np.percentile(bm, 2.5), dmAP_ci_hi=np.percentile(bm, 97.5),
                        dR1=dr.mean(), dR1_ci_lo=np.percentile(br, 2.5), dR1_ci_hi=np.percentile(br, 97.5),
                        pos_cells=int((dm > 0).sum()), **dg))
    return out
summ = summarize()
with open(f'{args.out}/r12_{args.tag}_summary.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
with open(f'{args.out}/r12_{args.tag}_cells.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
with open(f'{args.out}/r12_{args.tag}_diag_cells.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(diag_out[0].keys())); w.writeheader(); w.writerows(diag_out)
json.dump(sel_log, open(f'{args.out}/r12_{args.tag}_selection.json', 'w'), indent=0)
for nm_, lst_ in (('ctxlog', ctxlog), ('nnlog', nnlog), ('maplog', maplog)):
    if not lst_: continue
    with open(f'{args.out}/r12_{args.tag}_{nm_}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(lst_[0].keys())); w.writeheader(); w.writerows(lst_)
import hashlib, platform, sklearn, os as _os
def _sha(p_): return hashlib.sha256(open(p_, 'rb').read()).hexdigest()
json.dump(dict(args=vars(args), script=_os.path.abspath(__file__), script_sha256=_sha(__file__),
               inputs={_os.path.basename(x): _sha(x) for x in (f'{args.feats}/{args.base}.npz', f'{args.feats}/{args.adapter}.npz', args.assign)},
               versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, torch=torch.__version__),
               threads={k: _os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
               wall_seconds=round(time.time() - t0, 1), finished=time.strftime('%Y-%m-%d %H:%M:%S %Z'), wccn_diag=WCCN_DIAG),
          open(f'{args.out}/r12_{args.tag}_run_record.json', 'w'), indent=1)
import pickle; pickle.dump({k: v for k, v in perq.items()}, open(f'{args.out}/r12_{args.tag}_perq.pkl', 'wb'))
# ---- near-duplicate strata (analysis only): per identity, max raw IR cosine over pairs of its own gallery items
IRraw = l2n(gb('clean', 'f_ir').astype(np.float64)); Craw = IRraw @ IRraw.T
nd = {}
for u in np.unique(item_raw):
    ix = np.where(item_raw == u)[0]
    nd[int(u)] = np.nan if len(ix) < 2 else max(Craw[a, b] for a in ix for b in ix if a < b)
vals = np.array([v for v in nd.values() if not np.isnan(v)]); t1, t2 = np.quantile(vals, [1 / 3, 2 / 3])
def stratum(u):
    v = nd[int(u)]
    return 'single' if np.isnan(v) else ('low' if v < t1 else ('mid' if v < t2 else 'high'))
nd_rows = []
for (es, mx, m), pq in perq.items():
    for sname in ('single', 'low', 'mid', 'high'):
        qs = [q for q in pq if stratum(qraw[q]) == sname]
        if not qs: continue
        nd_rows.append(dict(evalset=es, mixture=mx, method=m, stratum=sname, n_queries=len(qs),
                            dmAP=float(np.mean([np.mean([v[0] for v in pq[q]]) for q in qs])), dR1=float(np.mean([np.mean([v[1] for v in pq[q]]) for q in qs]))))
with open(f'{args.out}/r12_{args.tag}_neardup.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(nd_rows[0].keys())); w.writeheader(); w.writerows(nd_rows)
print('near-dup tercile cut points (max same-id raw IR cosine):', round(t1, 3), round(t2, 3), '| identities with >=2 items:', len(vals), 'of', len(nd))
print('\n%-6s %-6s %-16s %6s %15s %6s %15s %5s | %6s %6s %6s %6s %6s %5s' % ('eval', 'mix', 'method', 'dmAP', '95%CI(id-boot)', 'dR1', '95%CI', '+cel', 'srcfr', 'wrong', 'leak', 'cover', 'cov50', 'AUC'))
for r in summ:
    print('%-6s %-6s %-16s %+6.2f [%+5.2f,%+5.2f] %+6.2f [%+5.2f,%+5.2f] %2d/%2d | %6.2f %6.3f %6.3f %6.3f %6.3f %5.2f' % (
        r['evalset'], r['mixture'], r['method'], r['dmAP'], r['dmAP_ci_lo'], r['dmAP_ci_hi'], r['dR1'], r['dR1_ci_lo'], r['dR1_ci_hi'],
        r['pos_cells'], r['n_cells'], r['frac_src'], r['wrong_mass'], r['leak'], r['coverage'], r['cover50'], r['auc']))
print('total time %.0fs' % (time.time() - t0))
