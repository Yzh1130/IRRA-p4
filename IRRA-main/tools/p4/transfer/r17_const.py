# Round 17 (project doc "论文4_第十七轮_独立裁决与收尾计划") = the frozen round-15 script r15_test.py (sha256 670a89be...) plus:
#   QE+Const  : p'_j = (1 - g) p_j + g sum_k A_jk p_k on the frozen QE-DBA front end, with the main rule's association A, the same
#               availability mask and r_max; one global g selected on the development fit galleries with the frozen main rule
#               (grid --const_grid, locked before any round-17 test run; ties keep the first, i.e. the smaller, value)
#   QE+NoGate : the same with g = 1 (capped to r_max inside post_update)
#   --fixed_selection : the round-15 front-end configuration is reused (never re-selected); only that configuration is calibrated
#   --perq_ext        : per-query records with explicit keys (fit seed, evaluation set, mixture, assignment seed, method)
#   --timing          : stage timing with warm-up and repeats on cached features (no retrieval metric inside the timed code)
#   --kappa_a_fixed / --wccn_shrink / --wccn_shrink_eval : development sensitivity of the association temperature and the WCCN map
#   selections whose methods are not run are skipped (selection consumes no random numbers, so every other output is unchanged)
# With none of the new options and the round-15 method list, the outputs are those of r15_test.py (tests/p4/test_r17.py checks this).
"""Round 15 = round-14 reference r14_app.py (sha256 78465684...) + the official-test additions of the round-15 pre-registration
(project doc "论文4_第十五轮_官方测试集预注册"): grid preset v3 (widened baseline grids), several fitting seeds in test mode,
identity-disjoint test sub-galleries for the scale analysis (report-only), and one helper that groups evaluation-set names.
Development-mode outputs with the presets dev / v2 are unchanged (checked bit for bit against r14_app.py).

--- round-14 docstring ---
Round 14 stage-2 development comparison (application-type paper route) = the round-13 reference (r13_simcf.py, sha256 0af9a18c...)
plus generic post-processing baselines and compositions, all under the same identity-isolated protocol and selection rule:
  KR      : k-reciprocal re-ranking (Zhong et al. 2017) adapted to text queries (gallery k-reciprocal sets in the joint embedding space,
            query soft vector from the B* posterior), final score 1 - [(1-lam) d_Jaccard + lam (1 - P)]
  RVN     : re-ranking by visual neighbours (NAFS, Gao et al. 2021): hard-set Jaccard between the query's top-l list and each item's
            top-l visual neighbours, averaged with the original posterior (weight w)
  QE-DBA  : alpha-DBA on gallery embeddings + text query expansion with the top-k_q items, per-availability Platt refitted per configuration
  FeatImp : ungated graph imputation of the missing RGB feature of IR-only items (link-weighted mean of complete items' RGB features)
  Y+M     : the main rule's directed transfer (A-2f-ctxn weights, which depend only on links and availability) applied on top of method Y
The round-13 guard against the frozen stage-1c mixtures is removed: stage 1c ran once on 2026-10-24 and is closed.
Spec: project doc "论文4_第十四轮_应用型路线与第二阶段对比规格" (written before any run).

--- round-13 docstring ---
Round 13 = round-12 cross-fitted reference + same-information controls N1-N3 (A-ctxonlyN, CAET-recalN, CAET-hardN),
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
# ---- round 14 test mode (official test split): fit everything on ALL development identities, evaluate once on a held-out archive
ap.add_argument('--test_npz', default=None)        # held-out archive in the same 12-key format as the development archives
ap.add_argument('--test_seed', type=int, nargs='+', default=[0])   # plays the role of the split seed in test mode (RNG streams only); several = several fits, pooled
ap.add_argument('--test_parts', type=int, nargs='*', default=[])   # round 15, test mode only: also evaluate on K identity-disjoint parts of the held-out gallery (scale analysis)
ap.add_argument('--test_part_seed', type=int, default=9001)        # RNG of the identity partition into parts
ap.add_argument('--allow_crossfit_test', action='store_true')   # equivalence checks only: test mode with the cross-fitted link map
ap.add_argument('--grid_preset', choices=['dev', 'v2', 'v3'], default='dev')   # 'v2' (review): KR / RVN / QE grids with mild settings and identity options
ap.add_argument('--select_rule', choices=['constrained', 'dmap'], default='constrained')   # 'dmap': sensitivity analysis without the dR1 >= -0.5 feasibility rule
# ---- round 17
ap.add_argument('--fixed_selection', default=None)      # r15 selection.json of the same base archive and fitting seed: frozen QE-DBA / QEir configuration reused
ap.add_argument('--const_grid', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0])   # locked before any r17 test run
ap.add_argument('--const_curve', action='store_true')    # report-only: QE+Const at every grid value (never used to choose g)
ap.add_argument('--perq_ext', action='store_true')       # explicit-key per-query records: AP, R1, R5, R10, mINP, top-1 item and its availability
ap.add_argument('--timing', action='store_true')         # stage timing on cached features (warm-up + repeats)
ap.add_argument('--kappa_a_fixed', type=float, default=None)     # dev sensitivity: fixed association temperature (gate unchanged)
ap.add_argument('--wccn_shrink', type=float, default=0.5)        # dev sensitivity: shrinkage of the training-identity WCCN map; everything downstream refitted
ap.add_argument('--wccn_shrink_eval', type=float, default=None)  # dev sensitivity: inference-only perturbation (fit with --wccn_shrink, evaluate with this map)
args = ap.parse_args()
if args.fixed_selection and not args.test_npz: raise SystemExit('--fixed_selection is defined for test mode only')
if (args.wccn_shrink != 0.5 or args.wccn_shrink_eval is not None) and args.link_map != 'train_lda': raise SystemExit('--wccn_shrink(_eval) needs --link_map train_lda')
if sorted(args.const_grid) != list(args.const_grid) or len(set(args.const_grid)) != len(args.const_grid): raise SystemExit('--const_grid must be strictly increasing')
if args.eval_gallery == 'full' and args.folds != 2: raise SystemExit('--eval_gallery full requires --folds 2')
if args.link_map == 'train_lda' and not args.train_ir_npz: raise SystemExit('--link_map train_lda requires --train_ir_npz')
torch.set_num_threads(1)

SC = ['clean', 'rgb_missing', 'ir_missing', 'rgb_noise_010', 'ir_noise_010', 'rgb_blur_k7', 'ir_blur_k7']
AV = {'clean': 0, 'rgb_noise_010': 0, 'ir_noise_010': 0, 'rgb_blur_k7': 0, 'ir_blur_k7': 0, 'rgb_missing': 1, 'ir_missing': 2}
def l2n(x): return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)
def sig(x): return 1.0 / (1.0 + np.exp(-x))
def lgt(p): p = np.clip(p, 1e-9, 1 - 1e-9); return np.log(p) - np.log1p(-p)
def es_group(name):
    """evaluation-set group used in summaries and per-query records: small0/small1/... -> small; test parts pK_k -> pK; others unchanged"""
    if name.startswith('small'): return 'small'
    if name.startswith('p') and '_' in name and name[1:name.index('_')].isdigit(): return name[:name.index('_')]
    return name

class _Merged:
    """test mode: the development archive followed by the held-out archive (queries and items concatenated);
    contiguous pid labels of the held-out part are offset so that the two label spaces stay disjoint"""
    def __init__(self, zd, zt):
        self.zd, self.zt = zd, zt; self.files = sorted(set(zd.files) & set(zt.files))
        self.off = int(max(zd['clean/txt_pids'].max(), zd['clean/img_pids'].max())) + 1
    def __getitem__(self, k):
        a, b = self.zd[k], self.zt[k]
        if k in ('clean/txt_pids', 'clean/img_pids'): b = b + self.off
        return np.concatenate([a, b], 0)
if args.test_npz:
    if args.adapter != args.base: raise SystemExit('test mode is defined for base features only (--adapter must equal --base)')
    if args.link_map != 'train_lda' and not args.allow_crossfit_test: raise SystemExit('test mode requires --link_map train_lda')
    _zd = np.load(f'{args.feats}/{args.base}.npz'); _zt = np.load(args.test_npz)
    _ov = set(_zd['clean/query_raw_pids'].tolist()) & set(_zt['clean/query_raw_pids'].tolist())
    if _ov: raise SystemExit(f'held-out identities overlap the development identities: {sorted(_ov)[:10]}')
    zb = _Merged(_zd, _zt); N_DEV_Q, N_DEV_G = len(_zd['clean/txt_pids']), len(_zd['clean/img_pids'])
else:
    zb = np.load(f'{args.feats}/{args.base}.npz')
def gb(s, k): return zb[f'{s}/{k}'] if f'{s}/{k}' in zb.files else zb[f'clean/{k}']
T = l2n(zb['clean/text_features'].astype(np.float64)); tp, gp, qraw = zb['clean/txt_pids'], zb['clean/img_pids'], zb['clean/query_raw_pids']
POS = tp[:, None] == gp[None, :]
tp2raw = dict(zip(tp.tolist(), qraw.tolist())); item_raw = np.array([tp2raw[p] for p in gp.tolist()])
IRF = {s: (l2n(gb(s, 'f_ir').astype(np.float64)) if AV[s] != 2 else None) for s in SC}
za = zb if args.test_npz else np.load(f'{args.feats}/{args.adapter}.npz'); ga = lambda s, k: za[f'{s}/{k}'] if f'{s}/{k}' in za.files else za[f'clean/{k}']
Sst = {}; EMB = {}
for s in SC:
    a = AV[s]; fr, fi = ga(s, 'f_rgb').astype(np.float64), ga(s, 'f_ir').astype(np.float64)
    EMB[s] = l2n(fi) if a == 1 else (l2n(fr) if a == 2 else l2n(fr + fi))   # drop-blank joint-space embedding of each item in state s
    Sst[s] = T @ EMB[s].T
SRAW = {}
if args.methods and 'Raw' in args.methods:
    # original scoring of the base model, without drop-blank or calibration (motivation baseline): for a missing state the archive
    # holds the blank-image feature of the missing modality, so l2n(f_rgb + f_ir) is the blank-embedding fused feature
    for s in SC:
        SRAW[s] = T @ l2n(ga(s, 'f_rgb').astype(np.float64) + ga(s, 'f_ir').astype(np.float64)).T
FUSED = l2n(gb('clean', 'f_rgb').astype(np.float64) + gb('clean', 'f_ir').astype(np.float64))  # analysis only (near-duplicates)
FR_ST = {s: ga(s, 'f_rgb').astype(np.float64) for s in SC if AV[s] != 1}   # RGB feature of complete-state items (adapter)
FI_CLEAN = ga('clean', 'f_ir').astype(np.float64)                           # clean-route IR feature (computable from the IR image alone)
PROJ_TR = None; WCCN_DIAG = {}; PROJ_EVAL = None

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

def ext_metrics(S, pos, S0=None):
    """R5, R10 and mINP (mean inverse negative penalty: |positives| / rank of the hardest positive), in %"""
    o = order_rows(S, S0); m = np.take_along_axis(pos, o, 1); npos = m.sum(1)
    last = S.shape[1] - np.argmax(m[:, ::-1], 1)          # 1-based rank of the last positive
    return m[:, :5].any(1) * 100.0, m[:, :10].any(1) * 100.0, np.where(npos > 0, npos / last, 0.0) * 100.0

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
    PROJ_TR = lda_map(Xtr, ytr, shrink=args.wccn_shrink)   # r17: shrinkage is a parameter (default 0.5 = round 15)
    if args.wccn_shrink_eval is not None: PROJ_EVAL = lda_map(Xtr, ytr, shrink=args.wccn_shrink_eval)   # r17 dev sensitivity (inference only)
    def _sw_per_item(X, ids):
        tot = 0.0
        for u in np.unique(ids):
            Z = X[ids == u] - X[ids == u].mean(0); tot += float((Z ** 2).sum())
        return tot / len(X)
    Xdev = l2n(gb('clean', 'f_ir').astype(np.float64))
    WCCN_DIAG = dict(n_train_items=int(len(Xtr)), n_train_ids=int(len(np.unique(ytr))), sw_per_item_train=_sw_per_item(Xtr, ytr),
                     sw_per_item_dev=_sw_per_item(Xdev, item_raw))
    WCCN_DIAG['dev_over_train'] = WCCN_DIAG['sw_per_item_dev'] / max(WCCN_DIAG['sw_per_item_train'], 1e-12)
    if args.test_npz:   # test mode: the line above pools development and held-out items; report the two parts separately as well
        WCCN_DIAG['sw_per_item_devonly'] = _sw_per_item(Xdev[:N_DEV_G], item_raw[:N_DEV_G])
        WCCN_DIAG['sw_per_item_test'] = _sw_per_item(Xdev[N_DEV_G:], item_raw[N_DEV_G:])
        WCCN_DIAG['test_over_train'] = WCCN_DIAG['sw_per_item_test'] / max(WCCN_DIAG['sw_per_item_train'], 1e-12)
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

def make_eval_assign(n, item_raw, FR0, csv_path):
    """round 14: the item identities, base RGB features (M10) and CSV are parameters, so that test mode can generate every mixture
    on the held-out gallery alone; with the development arrays the RNG draws are exactly those of round 13"""
    asg = {}
    for r in (csv.DictReader(open(csv_path)) if csv_path else []):
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
if args.test_npz:   # mixtures generated on the held-out gallery alone (development rows: placeholders, never evaluated)
    _te = np.arange(N_DEV_G, len(gp))
    _A = make_eval_assign(len(_te), item_raw[_te], FR0[_te], None if args.assign == 'none' else args.assign)
    ASSIGN = {k: np.concatenate([np.array(['clean'] * N_DEV_G), v]) for k, v in _A.items()}
else:
    ASSIGN = make_eval_assign(len(gp), item_raw, FR0, None if args.assign == 'none' else args.assign)   # r17: 'none' allowed in development mode

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

# ---------------------------------------------------------------- r17: stage timing (cached features, frozen settings; no retrieval metric is computed here)
def _bench(fn, warm=2, reps=10):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        t_ = time.perf_counter(); fn(); ts.append(time.perf_counter() - t_)
    ts = np.array(ts)
    return dict(median_s=float(np.median(ts)), p25_s=float(np.percentile(ts, 25)), p75_s=float(np.percentile(ts, 75)), reps=reps, warmup=warm)

def r17_timing(ss, f):
    """fitting stages on the development identities, then the online stages on one evaluation gallery (first mixture other than M0,
    first assignment seed). Uses the module-level state of the current fit (the main loop runs at module level)."""
    import os, platform, resource
    out = dict(split=int(ss), fold=int(f))
    out['fit_platt'] = _bench(lambda: {a: fit_affine(Sst[st_][np.ix_(qf, gf)].ravel(), POS[np.ix_(qf, gf)].ravel().astype(float))
                                       for a, st_ in {0: 'clean', 1: 'rgb_missing', 2: 'ir_missing'}.items()}, 1, 3)
    if args.link_map == 'train_lda': out['fit_wccn'] = _bench(lambda: lda_map(Xtr, ytr, shrink=args.wccn_shrink), 1, 3)
    def _eps():
        for kind_, items_, st_e, av_e, g_e, _ in EP:
            Ce_ = DIRS[g_e['dir']]['C'][np.ix_(items_, items_)]; gallery_struct(Ce_, av_e, idf[items_]); gal_ctx2(Ce_, av_e)
    out['fit_episode_features'] = _bench(_eps, 1, 3)
    Xg_ = np.c_[Z(Xall)[:, [0, 5]], (ctx2_all - mu_c2) / sd_c2]
    out['fit_gate_logreg'] = _bench(lambda: LogisticRegression(C=1.0, max_iter=2000).fit(Xg_, ysrc), 1, 5)
    out['fit_kappa_a'] = _bench(lambda: fit_kappa('CF', 'SF', 'y_src'), 1, 3)
    if 'select_const_s' in TIMES: out['fit_const_selection_s'] = TIMES['select_const_s'][-1]
    ename, qe, ge = eval_sets[0]
    mx = next(m for m in args.mixes if m != 'M0'); asd = args.assign_seeds[0]
    st = ASSIGN[(mx, asd)][ge]; G = build(qe, ge, st); av = G['av']; hasIR = G['has']; C = G['C']
    X = np.zeros((len(ge), Xf.shape[1]))
    for t_, (j_, s_) in enumerate(zip(ge, st)):
        if hasIR[t_]: X[t_] = IRF[s_][j_]
    def _link():
        U = np.where(hasIR[:, None], proj(X), 0); return U @ U.T
    out['online_link_graph'] = _bench(_link, 2, 10)
    def _gate_assoc():
        g_ = gallery_struct(C, av); c2_ = (np.array(gal_ctx2(C, av)) - mu_c2) / sd_c2
        r_ = src2N.predict_proba(np.c_[Z(g_['X'])[:, [0, 5]], np.tile(c2_, (len(g_['J']), 1))])[:, 1]
        return g_, r_, softmax_rows(ka * g_['CF'], np.ones_like(g_['CF'], bool))
    def _const_assoc():
        J_ = np.where(av == 1)[0]; F_ = np.where((av == 0) & hasIR)[0]; CF_ = C[np.ix_(J_, F_)]
        return softmax_rows(ka * CF_, np.ones_like(CF_, bool))
    out['online_gate_and_assoc'] = _bench(_gate_assoc, 2, 10)
    out['online_assoc_only'] = _bench(_const_assoc, 2, 10)
    g_, r_, A_ = _gate_assoc(); J_, Fi_ = g_['J'], g_['Fi']
    out['online_frontend_bstar'] = _bench(lambda: sig(G['L']), 2, 10)
    Py = G['P']
    if 'QE-DBA' in chosen:
        calq_ = cal_qe[('emb', chosen['QE-DBA'])]
        out['online_frontend_qedba'] = _bench(lambda: qedba_post(G, chosen['QE-DBA'], calq_), 2, 5)
        Py = qedba_post(G, chosen['QE-DBA'], calq_)
    W_ = r_[:, None] * A_; gm_ = float(chosen.get('QE+Const', 0.5))
    out['online_transfer_gate'] = _bench(lambda: post_update(Py, J_, Fi_, W_), 3, 20)
    out['online_transfer_const'] = _bench(lambda: post_update(Py, J_, Fi_, gm_ * A_), 3, 20)
    Pn = post_update(Py, J_, Fi_, W_)
    out['online_sort'] = _bench(lambda: order_rows(Pn, G['P']), 2, 10)
    out['shape'] = dict(evalset=ename, mixture=mx, aseed=int(asd), n_queries=int(len(qe)), n_items=int(len(ge)), n_ir_only=int(len(J_)), n_complete=int(len(Fi_)))
    out['peak_rss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    out['env'] = dict(platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
                      affinity=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None, numpy=np.__version__,
                      dtype='float64', torch_threads=torch.get_num_threads(),
                      threads={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
    TIMES.setdefault('stages', []).append(out)

# ---------------------------------------------------------------- round 14: generic post-processing baselines
def topk_idx(S, k):
    """indices of the k largest entries per row, in decreasing order (stable for ties)"""
    return np.argsort(-S, 1, kind='stable')[:, :k]

def ir_affinity(G):
    """gallery-gallery affinity of the IR variants (round 14, review item 1): IR-bearing pairs use the link cosine G['C'] (the shared IR
    modality, same link map as the main rule); a pair with one RGB-only item is never a neighbour (-1); RGB-only pairs use the
    joint-space cosine. Label-free at evaluation time; cached per gallery."""
    if '_iraff' not in G:
        has = G['has']; A = np.where(has[:, None] & has[None, :], G['C'], -1.0); ro = ~has
        if ro.any(): A[np.ix_(ro, ro)] = G['E'][ro] @ G['E'][ro].T
        np.fill_diagonal(A, 1.0); G['_iraff'] = A
    return G['_iraff']

def kr_gallery(E, k1, Aff=None):
    """gallery side of k-reciprocal encoding (Zhong et al. 2017, released re_ranking code), on the joint-space embeddings E (unit rows),
    or on a given cosine affinity Aff (IR variants). Distances: squared Euclidean 2 - 2cos, each row divided by its maximum (as
    original_dist in the released code). Returns the unexpanded soft vectors V (n x n, unit row sums) and the initial ranking (self first)."""
    n = len(E); Cg = (E @ E.T) if Aff is None else Aff
    D = np.maximum(2.0 - 2.0 * Cg, 0.0); D = D / np.maximum(D.max(1, keepdims=True), 1e-12)
    order = np.argsort(D, 1, kind='stable')
    k1_ = min(k1, n - 1); kh = min(int(np.around(k1 / 2.0)), n - 1)
    def recip_sets(k):
        Nk = order[:, :k + 1]; inN = np.zeros((n, n), bool); inN[np.arange(n)[:, None], Nk] = True
        M = inN & inN.T                                       # j in N(i,k) and i in N(j,k)
        return [np.where(M[i])[0] for i in range(n)]
    R = recip_sets(k1_); Rh = recip_sets(kh)
    V = np.zeros((n, n))
    for i in range(n):
        ext = R[i]
        for j in R[i]:
            cand = Rh[j]
            if len(np.intersect1d(cand, R[i])) > 2.0 / 3.0 * len(cand):
                ext = np.union1d(ext, cand)
        w = np.exp(-D[i, ext]); V[i, ext] = w / w.sum()
    return V, order

def kr_query(P, Vraw, k1, k2):
    """text-query soft vector: B* posterior mass on the query's top-k1 items (no reciprocity exists across modalities);
    local query expansion with the unexpanded vectors of its top-(k2-1) items"""
    n = P.shape[1]; o = topk_idx(P, min(k1, n)); Vq = np.zeros_like(P); r = np.arange(P.shape[0])[:, None]
    Vq[r, o] = P[r, o]; Vq /= np.maximum(Vq.sum(1, keepdims=True), 1e-300)
    if k2 > 1:
        o2 = topk_idx(P, min(k2 - 1, n)); Vq = (Vq + Vraw[o2].sum(1)) / (1 + o2.shape[1])
    return Vq

def jaccard_min_sum(Vq, Vg, chunk=48):
    """m[q,i] = sum_k min(Vq[q,k], Vg[i,k]); with unit-sum vectors the Jaccard distance is 1 - m/(2 - m)"""
    m = np.zeros((Vq.shape[0], Vg.shape[0]))
    for a in range(0, Vq.shape[0], chunk):
        m[a:a + chunk] = np.minimum(Vq[a:a + chunk, None, :], Vg[None, :, :]).sum(2)
    return m

def jaccard_min_sum_sparse(Vq, Vg):
    """same quantity column by column over the supports (the released code's invIndex loop); equal to jaccard_min_sum up to rounding"""
    m = np.zeros((Vq.shape[0], Vg.shape[0]))
    for k in range(Vg.shape[1]):
        qk = np.nonzero(Vq[:, k])[0]
        if not len(qk): continue
        gk = np.nonzero(Vg[:, k])[0]
        if not len(gk): continue
        m[np.ix_(qk, gk)] += np.minimum(Vq[qk, k][:, None], Vg[gk, k][None, :])
    return m

def kr_scores(G, cfg, space='emb'):
    """final score 1 - [(1 - lam) * d_Jaccard + lam * (1 - P)], P = B* posterior (comparable across availabilities)"""
    k1, k2, lam = cfg; P = G['P']; cache = G.setdefault('_kr' if space == 'emb' else '_krir', {})
    if (k1, k2) not in cache:
        if ('V', k1) not in cache: cache[('V', k1)] = kr_gallery(G['E'], k1, None if space == 'emb' else ir_affinity(G))
        Vraw, order = cache[('V', k1)]
        Vg = Vraw if k2 <= 1 else Vraw[order[:, :min(k2, len(Vraw))]].mean(1)   # local query expansion (k2 neighbours incl. itself)
        Vq = kr_query(P, Vraw, k1, k2)
        m = jaccard_min_sum(Vq, Vg) if len(Vg) <= 600 else jaccard_min_sum_sparse(Vq, Vg)   # large (official test) galleries: inverted index
        cache[(k1, k2)] = 1.0 - m / np.maximum(2.0 - m, 1e-12)
    dJ = cache[(k1, k2)]
    return 1.0 - ((1.0 - lam) * dJ + lam * (1.0 - P))

def rvn_scores(G, cfg, space='emb'):
    """NAFS re-ranking by visual neighbours: D_J = 1 - |N_t2i(q,l) & N_i2i(i,l)| / |N_t2i(q,l) | N_i2i(i,l)|"""
    l, w = cfg; P, E = G['P'], G['E']; n = E.shape[0]; l_ = min(l, n)
    Bg = np.zeros((n, n)); Bg[np.arange(n)[:, None], topk_idx((E @ E.T) if space == 'emb' else ir_affinity(G), l_)] = 1.0
    Bq = np.zeros(P.shape); Bq[np.arange(P.shape[0])[:, None], topk_idx(P, l_)] = 1.0
    inter = Bq @ Bg.T; DJ = 1.0 - inter / (2.0 * l_ - inter)
    return w * P + (1.0 - w) * (1.0 - DJ)

def qedba_raw(Tq, P, E, kd, kq, beta, alpha=3.0, Aff=None):
    """alpha-DBA on the gallery embeddings (top-k_d neighbours incl. itself, weights max(cos,0)^alpha), then text query expansion
    t' = norm(t + beta * sum_j w_j e'_j) over the query's top-k_q items by B* posterior, w_j ~ P^alpha; returns raw cosines t' . e'.
    Aff (IR variant): neighbours and weights of the DBA step come from this affinity; the averaged vectors stay joint-space embeddings."""
    n = E.shape[0]
    if kd > 1:
        Cg = (E @ E.T) if Aff is None else Aff; og = topk_idx(Cg, min(kd, n)); r = np.arange(n)[:, None]
        Wd = np.zeros((n, n)); Wd[r, og] = np.maximum(Cg[r, og], 0.0) ** alpha; E2 = l2n(Wd @ E)
    else:
        E2 = E
    if kq > 0:
        oq = topk_idx(P, min(kq, n)); r = np.arange(P.shape[0])[:, None]
        Wq = np.zeros(P.shape); Wq[r, oq] = P[r, oq] ** alpha; Wq /= np.maximum(Wq.sum(1, keepdims=True), 1e-300)
        T2 = l2n(Tq + beta * (Wq @ E2))
    else:
        T2 = Tq
    return T2 @ E2.T

def qedba_post(G, cfg, calq, space='emb'):
    """QE-DBA scores calibrated per availability with the configuration's own Platt parameters"""
    S = qedba_raw(T[G['qi']], G['P'], G['E'], *cfg, Aff=None if space == 'emb' else ir_affinity(G)); L = np.zeros_like(S)
    for a in np.unique(G['av']):
        cols = G['av'] == a; L[:, cols] = cal_logit(calq[a], S[:, cols])
    return sig(L)

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
               'B-sup', 'B-sup-crossonly', 'B-joint', 'B-joint-nopen', 'Oracle-A', 'Random-A', 'LAT-I', 'A-SRC+LAT-I', 'B-sup-noC', 'A-SRC-feat', 'CAET-recal', 'A-SRC-1f', 'A-SRC-2f', 'A-LGctx-post', 'A-SRC-lowe', 'A-SRC-dI', 'A-QV', 'A-SRC-EM', 'A-2f-EM', 'A-SRC-ctx', 'A-2f-ctx', 'A-ctxonly', 'A-2f-ctxn', 'A-ctxonlyN', 'CAET-recalN', 'CAET-hardN', 'A-1f-ctxn', 'A-2f-kap',
               'KR', 'RVN', 'QE-DBA', 'FeatImp', 'KR+M', 'RVN+M', 'QE+M', 'GP+M', 'QE+CAET', 'QE+SRC2f', 'QE+EM', 'QE+Orc', 'DB', 'Raw',
               'KRir', 'RVNir', 'QEir', 'KRir+M', 'RVNir+M', 'QEir+M', 'FeatImp-g', 'A-2f-ctxn-g', 'QE+Const', 'QE+NoGate']
METHODS = args.methods or METHODS_ALL
# ---- r17: frozen round-15 selections, explicit-key records, timing
import ast as _ast
FIXED_ALL = {}
if args.fixed_selection:
    for e_ in json.load(open(args.fixed_selection)):
        if e_['method'] in ('QE-DBA', 'QEir'):
            k_ = (int(e_['split']), int(e_['fold']), e_['method'])
            if k_ in FIXED_ALL: raise SystemExit(f'--fixed_selection: duplicate entry {k_}')
            FIXED_ALL[k_] = _ast.literal_eval(e_['chosen'])
PQX = {}; R17LOG = []; TIMES = {}
TOP10_METHODS = ('QE-DBA', 'QE+M', 'QE+Const')
if args.link_map == 'train_lda' and set(METHODS) & {'A-SRC-post-head', 'CAET-head', 'GP-same-head', 'B-sup', 'B-sup-crossonly', 'B-joint', 'B-joint-nopen', 'B-sup-noC', 'GP-same-perMix'}:
    raise SystemExit('train_lda mode supports only the link-map-based rules; remove head / branch-B methods')
t0 = time.time()

def split_folds(ss):
    perm = np.random.default_rng(ss).permutation(uq); return {int(p): i % args.folds for i, p in enumerate(perm)}

for ss in (args.test_seed if args.test_npz else args.split_seeds):
    if args.test_npz:   # fold 0 = all development identities (fit), 1 = held-out archive (evaluate once)
        qf_all = (np.arange(len(qraw)) >= N_DEV_Q).astype(int); gf_all = (np.arange(len(gp)) >= N_DEV_G).astype(int)
    else:
        fold = split_folds(ss)
        qf_all = np.array([fold[int(p)] for p in qraw]); gf_all = np.array([fold[int(p)] for p in item_raw])
    for f in ([0] if args.test_npz else range(args.folds)):
        qf, gf = np.where(qf_all == f)[0], np.where(gf_all == f)[0]
        if args.test_npz:
            te_q, te_g = np.where(qf_all == 1)[0], np.where(gf_all == 1)[0]
            eval_sets = [('test', te_q, te_g)]
            for K in args.test_parts:   # report-only scale analysis: K identity-disjoint parts, each with its own queries
                perm_ = np.random.default_rng(args.test_part_seed + K).permutation(np.unique(item_raw[te_g]))
                part_ = {int(p_): i_ % K for i_, p_ in enumerate(perm_)}
                pq_ = np.array([part_[int(qraw[q_])] for q_ in te_q]); pg_ = np.array([part_[int(item_raw[g_])] for g_ in te_g])
                eval_sets += [(f'p{K}_{k_}', te_q[pq_ == k_], te_g[pg_ == k_]) for k_ in range(K)]
        elif args.folds == 2 and args.eval_gallery == 'full':
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
        ka_fitted = ka
        if args.kappa_a_fixed is not None: ka = float(args.kappa_a_fixed)   # r17 dev sensitivity: association temperature fixed (the gate is unchanged)
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
            E = np.zeros((len(items), EMB['clean'].shape[1]))                  # round 14: joint-space embedding of each item in its state
            S0 = np.zeros((len(qi), len(items))); SR = np.zeros((len(qi), len(items))) if SRAW else None
            for s in np.unique(st):
                cols = np.where(st == s)[0]; E[cols] = EMB[s][items[cols]]; S0[:, cols] = Sst[s][np.ix_(qi, items[cols])]
                if SRAW: SR[:, cols] = SRAW[s][np.ix_(qi, items[cols])]
            return dict(L=L, P=sig(L), LI=LI, av=av, has=hasIR, C=C, CH=CH, items=items, st=st, qi=qi, U=U, Xraw=X, E=E, S=S0, SRAW=SR)
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
            # ---- round 14: generic post-processing baselines, ungated feature imputation, and compositions Y+M
            nz = np.zeros((len(J), len(Fi)))
            W_M = None   # the main rule's transfer weights (A-2f-ctxn); they depend only on links and availability, not on the scores
            if len(J) and len(Fi):
                A_m = softmax_rows(ka * g['CF'], np.ones_like(g['CF'], bool))
                c2m = (np.array(gal_ctx2(C, av)) - mu_c2) / sd_c2
                W_M = src2N.predict_proba(np.c_[Z(g['X'])[:, [0, 5]], np.tile(c2m, (len(J), 1))])[:, 1][:, None] * A_m
                if 'A-2f-ctxn' in out:   # self-check: the composition weights are the main rule's weights, and Y+M with Y = B* is the main rule
                    assert np.array_equal(W_M, out['A-2f-ctxn'][1]) and np.array_equal(post_update(P, J, Fi, W_M), out['A-2f-ctxn'][0])
                if want('FeatImp'): out['FeatImp'] = (featimp_P(G, g, 1.0), A_m)
                if want('FeatImp-g'): gm = chosen['FeatImp-g']; out['FeatImp-g'] = (featimp_P(G, g, gm), gm * A_m)
                if want('A-2f-ctxn-g'): gm = chosen['A-2f-ctxn-g']; out['A-2f-ctxn-g'] = (post_update(P, J, Fi, gm * W_M), gm * W_M)
            def with_main(Py):
                return (post_update(Py, J, Fi, W_M), W_M) if W_M is not None else (Py, nz)
            for name, fn_, sp_ in (('KR', kr_scores, 'emb'), ('RVN', rvn_scores, 'emb'), ('KRir', kr_scores, 'ir'), ('RVNir', rvn_scores, 'ir')):
                if want(name) or want(name + '+M'):
                    Py = fn_(G, chosen[name], sp_)
                    if want(name): out[name] = (Py, nz)
                    if want(name + '+M'): out[name + '+M'] = with_main(Py)
            QEX = ('QE+CAET', 'QE+SRC2f', 'QE+EM', 'QE+Orc', 'QE+Const', 'QE+NoGate')   # gate variants on top of QE-DBA (r17: constant gain, no gate)
            if any(want(m) for m in ('QE-DBA', 'QE+M') + QEX):
                Py = qedba_post(G, chosen['QE-DBA'], cal_qe[('emb', chosen['QE-DBA'])])
                if want('QE-DBA'): out['QE-DBA'] = (Py, nz)
                if want('QE+M'): out['QE+M'] = with_main(Py)
                if W_M is not None:
                    if want('QE+CAET'): out['QE+CAET'] = (post_update(Py, J, Fi, Wc), Wc)
                    r2 = src2.predict_proba(Z(g['X'])[:, [0, 5]])[:, 1]
                    if want('QE+SRC2f'): W_ = r2[:, None] * A_m; out['QE+SRC2f'] = (post_update(Py, J, Fi, W_), W_)
                    if want('QE+EM'): rE, _ = em_prior(r2, pi_tr2); W_ = rE[:, None] * A_m; out['QE+EM'] = (post_update(Py, J, Fi, W_), W_)
                    if want('QE+Orc') and ids_for_diag is not None:
                        Sm = g['SF'].astype(float); W_ = Sm / np.maximum(Sm.sum(1, keepdims=True), 1); out['QE+Orc'] = (post_update(Py, J, Fi, W_), W_)
                    if want('QE+Const'): W_ = chosen['QE+Const'] * A_m; out['QE+Const'] = (post_update(Py, J, Fi, W_), W_)
                    if want('QE+NoGate'): out['QE+NoGate'] = (post_update(Py, J, Fi, A_m), A_m)   # g = 1, capped to r_max inside post_update
                    if args.const_curve and want('QE+Const'):
                        for gm_ in args.const_grid: W_ = gm_ * A_m; out[f'QE+Const@{gm_:g}'] = (post_update(Py, J, Fi, W_), W_)
                else:
                    for m_ in QEX:
                        if want(m_): out[m_] = (Py, nz)
                    if args.const_curve and want('QE+Const'):
                        for gm_ in args.const_grid: out[f'QE+Const@{gm_:g}'] = (Py, nz)
            if want('QEir') or want('QEir+M'):
                Py = qedba_post(G, chosen['QEir'], cal_qe[('ir', chosen['QEir'])], 'ir')
                if want('QEir'): out['QEir'] = (Py, nz)
                if want('QEir+M'): out['QEir+M'] = with_main(Py)
            if want('GP+M'):
                out['GP+M'] = with_main(gp_update(P, Wall, chosen['GP-same']))
            if want('DB'): out['DB'] = (G['S'], nz)              # drop-blank cosine, no calibration
            if want('Raw') and G['SRAW'] is not None: out['Raw'] = (G['SRAW'], nz)   # blank-embedding cosine, no calibration
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
                feas = (min(dr) >= -0.5) if args.select_rule == 'constrained' else True; key = (feas, np.mean(dm) if feas else min(dr))
                if best is None or key > best[0]: best = (key, cfg)
            sel_log.append(dict(split=ss, fold=f, method=name, chosen=str(best[1]), feasible=bool(best[0][0])))
            return best[1]
        def pooled_select_table(name, grid, fn):
            """r17: the frozen main rule (same objective, feasibility rule, fallback and tie order as pooled_select), full table logged.
            Differences are relative to B* on the fit galleries, as for every method selected with pooled_select."""
            best = None; table = []
            for cfg in grid:
                dm, dr = [], []
                for per, bases in zip(SELG, base_fit):
                    A_, R_, BA, BR = [], [], [], []
                    for (G, qm), (b_ap, b_r1) in zip(per, bases):
                        a_, r_ = ap_r1(fn(G, cfg), pf, G['P']); A_.append(a_[qm]); R_.append(r_[qm]); BA.append(b_ap[qm]); BR.append(b_r1[qm])
                    dm.append(np.concatenate(A_).mean() - np.concatenate(BA).mean()); dr.append(np.concatenate(R_).mean() - np.concatenate(BR).mean())
                feas = (min(dr) >= -0.5) if args.select_rule == 'constrained' else True; key = (feas, np.mean(dm) if feas else min(dr))
                table.append(dict(cfg=cfg, dmAP_by_type=[round(float(x), 6) for x in dm], dR1_by_type=[round(float(x), 6) for x in dr],
                                  mean_dmAP=round(float(np.mean(dm)), 6), min_dR1=round(float(min(dr)), 6), feasible=bool(feas)))
                if best is None or key > best[0]: best = (key, cfg)
            sel_log.append(dict(split=ss, fold=f, method=name, chosen=str(best[1]), feasible=bool(best[0][0]), table=table,
                                types=[f'{k}{p}c{c}' for k, p, c in FIT_TYPES]))
            return best[1]
        NEED = lambda *ms: any(m in METHODS for m in ms)   # r17: selections of methods that are not run are skipped (no random numbers involved)
        GAM = [0.0, 0.25, 0.5, 0.75, 1.0]
        chosen = {}
        def caet_g(G, gm):
            g = gallery_struct(G['C'], G['av']); Wall = comp_weights_all(G['C'], G['has'], kap, th)
            return post_update(G['P'], g['J'], g['Fi'], Wall[np.ix_(g['J'], g['Fi'])] * gm) if len(g['J']) and len(g['Fi']) else G['P']
        if NEED('CAET-g'): chosen['CAET-g'] = pooled_select('CAET-g', GAM, caet_g)
        if NEED('GP-same', 'GP+M'): chosen['GP-same'] = pooled_select('GP-same', GAM, lambda G, gm: gp_update(G['P'], comp_weights_all(G['C'], G['has'], kap, th), gm))
        if NEED('LAT-only'): chosen['LAT-only'] = pooled_select('LAT-only', GAM, lambda G, gm: lat_update(G['P'], comp_weights_all(G['C'], G['has'], kap, th), G['av'], gm))
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
        if NEED('GP-same-head'): chosen['GP-same-head'] = pooled_select('GP-same-head', GAM, lambda G, gm: gp_update(G['P'], comp_weights_all(G['CH'], G['has'], kap_h, th_h), gm))
        same_cos = np.concatenate([D['C'][np.ix_(np.where(D['tgt'])[0], np.where(D['tgt'])[0])][np.triu(idf[D['tgt']][:, None] == idf[D['tgt']][None, :], 1)] for D in DIRS])
        taus = list(np.quantile(same_cos, [0.1, 0.3, 0.5, 0.7, 0.9])); dls = [0.0, 0.02, 0.05, 0.1, 0.2]
        def thr_fn(G, cfg):
            tau, dl = cfg; g = gallery_struct(G['C'], G['av'])
            if not (len(g['J']) and len(g['Fi'])): return G['P']
            W = np.zeros((len(g['J']), len(g['Fi']))); k1 = np.argmax(g['CF'], 1); ok = (g['X'][:, 0] > tau) & (g['X'][:, 2] > dl)
            W[np.arange(len(g['J']))[ok], k1[ok]] = 1.0; return post_update(G['P'], g['J'], g['Fi'], W)
        if NEED('THR'): chosen['THR'] = pooled_select('THR', [(t, d) for t in taus for d in dls], thr_fn)
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
        # ---- round 14: scalars of the generic baselines, selected with the same pooled fit-fold objective as GP-same
        if args.grid_preset == 'dev':   # grids of the stage-2 development comparison (spec section 2)
            KR_GRID = [(k1, k2, lam) for k1 in (5, 10, 20) for k2 in (1, 3) for lam in (0.3, 0.6)]
            RVN_GRID = [(l, w) for l in (5, 10, 20) for w in (0.3, 0.5, 0.7)]
            QE_B = (0.3, 0.6)
        elif args.grid_preset == 'v2':  # 'v2' (review items 2): the development selections sat on the grid edges; identity options included
            KR_GRID = [(k1, k2, lam) for k1 in (2, 3, 5, 10) for k2 in (1, 3) for lam in (0.3, 0.6, 0.8, 0.9, 1.0)]
            RVN_GRID = [(l, w) for l in (2, 3, 5, 10) for w in (0.3, 0.5, 0.7, 0.85, 0.95, 1.0)]
            QE_B = (0.1, 0.2, 0.3, 0.6)
        else:                           # 'v3' (round-15 pre-registration): widened past every edge the v2 selections reached
            KR_GRID = [(k1, k2, lam) for k1 in (1, 2, 3, 5, 10) for k2 in (1, 3) for lam in (0.0, 0.1, 0.2, 0.3, 0.6, 0.8, 0.9, 1.0)]
            RVN_GRID = [(l, w) for l in (2, 3, 5, 10) for w in (0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0)]   # l = 1 ranks exactly as w = 1
            QE_B = (0.1, 0.2, 0.3, 0.6, 1.0, 1.5, 2.5)   # 2.5 added after the v3 development look selected 1.5 (the edge) once
        if args.grid_preset == 'v3':   # k_d = 8 and k_q = 20 added after a v3 development look selected k_d = 5 and k_q = 10 (the edges) in some fits
            QE_GRID = [(kd, 0, 0.0) for kd in (1, 2, 3, 5, 8)] + [(kd, kq, b) for kd in (1, 2, 3, 5, 8) for kq in (2, 3, 5, 10, 20) for b in QE_B]
        else:
            QE_GRID = [(kd, 0, 0.0) for kd in (1, 3, 5)] + [(kd, kq, b) for kd in (1, 3, 5) for kq in (3, 5) for b in QE_B]
        for sp_, sfx in (('emb', ''), ('ir', 'ir')):   # joint-space versions and IR-link versions (review item 1)
            if any(m in METHODS for m in (f'KR{sfx}', f'KR{sfx}+M')):
                chosen[f'KR{sfx}'] = pooled_select(f'KR{sfx}', KR_GRID, lambda G, cfg, sp_=sp_: kr_scores(G, cfg, sp_))
            if any(m in METHODS for m in (f'RVN{sfx}', f'RVN{sfx}+M')):
                chosen[f'RVN{sfx}'] = pooled_select(f'RVN{sfx}', RVN_GRID, lambda G, cfg, sp_=sp_: rvn_scores(G, cfg, sp_))
        cal_qe = {}   # (space, configuration) -> per-availability Platt of the transformed scores
        QE_USERS = {'emb': ('QE-DBA', 'QE+M', 'QE+CAET', 'QE+SRC2f', 'QE+EM', 'QE+Orc', 'QE+Const', 'QE+NoGate'), 'ir': ('QEir', 'QEir+M')}
        FIXED = {m_: v_ for (s_, f_, m_), v_ in FIXED_ALL.items() if s_ == ss and f_ == f}   # r17: frozen configurations of this fit
        for sp_, users in QE_USERS.items():
            if not any(m in METHODS for m in users): continue
            Ghom = build(qf, gf, np.array(['ir_missing'] * len(gf)))   # RGB-only items never occur in the simulated fit galleries
            nm_ = 'QE-DBA' if sp_ == 'emb' else 'QEir'
            if args.fixed_selection and nm_ not in FIXED: raise SystemExit(f'--fixed_selection has no {nm_} entry for split {ss} fold {f}')
            if nm_ in FIXED and FIXED[nm_] not in QE_GRID: raise SystemExit(f'frozen {nm_} configuration {FIXED[nm_]} is not in the {args.grid_preset} grid')
            for cfg in ([FIXED[nm_]] if nm_ in FIXED else QE_GRID):   # r17: with --fixed_selection only the frozen configuration is calibrated
                xs = {0: [], 1: [], 2: []}; ys = {0: [], 1: [], 2: []}
                for per in SELG:                                          # transformed scores on the five simulated mixed fit galleries
                    for Gs_, _qm in (per[:1] if sp_ == 'emb' else per):   # joint space: E and P do not depend on the link direction
                        S_ = qedba_raw(T[qf], Gs_['P'], Gs_['E'], *cfg, Aff=None if sp_ == 'emb' else ir_affinity(Gs_))
                        for a_ in (0, 1):
                            cols = Gs_['av'] == a_
                            if cols.any(): xs[a_].append(S_[:, cols].ravel()); ys[a_].append(pf[:, cols].ravel().astype(float))
                S_ = qedba_raw(T[qf], Ghom['P'], Ghom['E'], *cfg, Aff=None if sp_ == 'emb' else ir_affinity(Ghom))
                xs[2].append(S_.ravel()); ys[2].append(pf.ravel().astype(float))
                cal_qe[(sp_, cfg)] = {a_: fit_affine(np.concatenate(xs[a_]), np.concatenate(ys[a_])) for a_ in (0, 1, 2)}
            if nm_ in FIXED:
                chosen[nm_] = FIXED[nm_]; sel_log.append(dict(split=ss, fold=f, method=nm_, chosen=str(FIXED[nm_]), feasible=None, source='fixed_selection'))
            else:
                chosen[nm_] = pooled_select(nm_, QE_GRID, lambda G, cfg, sp_=sp_: qedba_post(G, cfg, cal_qe[(sp_, cfg)], sp_))
        # ---- r17: one global constant gain on the frozen front end and the main rule's association A (decisive control)
        if 'QE+Const' in METHODS:
            cfg_c = chosen['QE-DBA']; cal_c = cal_qe[('emb', cfg_c)]
            def const_fn(G, gm):
                if '_r17_py' not in G: G['_r17_py'] = qedba_post(G, cfg_c, cal_c)
                Py_ = G['_r17_py']; g_ = gallery_struct(G['C'], G['av'])
                if not (len(g_['J']) and len(g_['Fi'])): return Py_
                A_ = softmax_rows(ka * g_['CF'], np.ones_like(g_['CF'], bool))
                return post_update(Py_, g_['J'], g_['Fi'], gm * A_)
            _t0c = time.perf_counter()
            chosen['QE+Const'] = pooled_select_table('QE+Const', [float(x) for x in args.const_grid], const_fn)
            TIMES.setdefault('select_const_s', []).append(round(time.perf_counter() - _t0c, 4))
        # ---- review items 3-4: back-off strength for feature imputation, and the same selection rule applied to the main rule
        def main_W(G):
            g_ = gallery_struct(G['C'], G['av'])
            if not (len(g_['J']) and len(g_['Fi'])): return g_, None
            A_ = softmax_rows(ka * g_['CF'], np.ones_like(g_['CF'], bool))
            c2_ = (np.array(gal_ctx2(G['C'], G['av'])) - mu_c2) / sd_c2
            return g_, src2N.predict_proba(np.c_[Z(g_['X'])[:, [0, 5]], np.tile(c2_, (len(g_['J']), 1))])[:, 1][:, None] * A_
        def main_g_scores(G, gm):
            g_, W_ = main_W(G)
            return G['P'] if W_ is None else post_update(G['P'], g_['J'], g_['Fi'], gm * W_)
        def featimp_P(G, g_, gm):
            J_, Fi_ = g_['J'], g_['Fi']; A_ = softmax_rows(ka * g_['CF'], np.ones_like(g_['CF'], bool)); its, sts = G['items'], G['st']
            FRs = np.stack([FR_ST[sts[k]][its[k]] for k in Fi_]); fhat = A_ @ FRs
            lhat = cal_logit(cal[0], T[G['qi']] @ l2n(fhat + FI_CLEAN[its[J_]]).T)
            Pn = G['P'].copy(); Pn[:, J_] = (1 - gm) * G['P'][:, J_] + gm * sig(lhat); return Pn
        def featimp_scores(G, gm):
            g_ = gallery_struct(G['C'], G['av'])
            return G['P'] if not (len(g_['J']) and len(g_['Fi'])) else featimp_P(G, g_, gm)
        if 'A-2f-ctxn-g' in METHODS: chosen['A-2f-ctxn-g'] = pooled_select('A-2f-ctxn-g', GAM, main_g_scores)
        if 'FeatImp-g' in METHODS: chosen['FeatImp-g'] = pooled_select('FeatImp-g', GAM, featimp_scores)
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
        if PROJ_EVAL is not None: proj = PROJ_EVAL   # r17 dev sensitivity: inference-only perturbation of the link map (every fitted quantity kept)
        R17LOG.append(dict(split=int(ss), fold=int(f), ka_fitted=float(ka_fitted), ka_used=float(ka), chosen={k: str(v) for k, v in chosen.items()}))
        if args.timing: r17_timing(ss, f)
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
                        rows_out[-1].update(disp_harm=float(((b_r1 == 100) & (r_ == 0))[~irtop].mean() * 100) if (~irtop).any() else np.nan,
                                            disp_help=float(((b_r1 == 0) & (r_ == 100))[~irtop].mean() * 100) if (~irtop).any() else np.nan)
                        if args.test_npz:   # paper metrics (test mode only, so development outputs keep the round-13 columns)
                            r5_, r10_, minp_ = ext_metrics(Pn, pe, res['B*'][0])
                            rows_out[-1].update(R5=r5_.mean(), R10=r10_.mean(), mINP=minp_.mean())
                        if args.perq_ext and not name.startswith('QE+Const@'):   # r17: explicit keys (fit seed, evaluation set, mixture, assignment seed, method)
                            if not args.test_npz: r5_, r10_, minp_ = ext_metrics(Pn, pe, res['B*'][0])
                            rec_ = dict(q=np.asarray(qe, np.int32), qid=np.asarray(qraw[qe]), AP=np.asarray(a_, np.float64), R1=(r_ > 0).astype(np.uint8),
                                        R5=(r5_ > 0).astype(np.uint8), R10=(r10_ > 0).astype(np.uint8), mINP=np.asarray(minp_, np.float64),
                                        top1=np.asarray(ge[top1], np.int32), top1_av=np.asarray(G['av'][top1], np.int8))
                            if name in TOP10_METHODS:   # top-10 composition (item, availability, positive) for the R5/R10 analysis
                                o10_ = order_rows(Pn, res['B*'][0])[:, :10]; rec_['top10'] = np.asarray(ge[o10_], np.int32)
                                rec_['top10_av'] = np.asarray(G['av'][o10_], np.int8); rec_['top10_pos'] = np.take_along_axis(pe, o10_, 1).astype(np.uint8)
                            key_ = (int(ss), ename, mx, int(asd), name)
                            if key_ in PQX: raise SystemExit(f'duplicate per-query record {key_}')
                            PQX[key_] = rec_
                        for qi_, d_ap, d_r1 in zip(qe, a_ - b_ap, r_ - b_r1):
                            perq.setdefault((es_group(ename), mx, name), {}).setdefault(int(qi_), []).append((d_ap, d_r1))
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
    keys = sorted(set((es_group(r['evalset']), r['mixture'], r['method']) for r in rows_out))
    for es, mx, m in keys:
        rr = [r for r in rows_out if es_group(r['evalset']) == es and r['mixture'] == mx and r['method'] == m]
        dm = np.array([r['dmAP'] for r in rr]); dr = np.array([r['dR1'] for r in rr])
        # identity-cluster bootstrap over per-query mean differences (evaluation-sampling uncertainty only)
        pq = perq.get((es, mx, m), {}); qs = sorted(pq); md = np.array([np.mean([v[0] for v in pq[q]]) for q in qs]); mr = np.array([np.mean([v[1] for v in pq[q]]) for q in qs])
        qid = np.array([qraw[q] for q in qs]); uid_ = np.unique(qid); idx = [np.where(qid == u)[0] for u in uid_]
        sums_m = np.array([md[i].sum() for i in idx]); sums_r = np.array([mr[i].sum() for i in idx]); cnt = np.array([len(i) for i in idx])
        rb = np.random.default_rng(2026); bm, br = [], []
        for _ in range(2000):
            x = rb.integers(0, len(uid_), len(uid_)); bm.append(sums_m[x].sum() / cnt[x].sum()); br.append(sums_r[x].sum() / cnt[x].sum())
        dd = [d for d in diag_out if es_group(d['evalset']) == es and d['mixture'] == mx and d['method'] == m]
        dg = {k: float(np.nanmean([d[k] for d in dd])) if dd else np.nan for k in ('frac_src', 'wrong_mass', 'leak', 'coverage', 'cover50', 'auc')}
        ext = {k: float(np.mean([r[k] for r in rr])) for k in ('R1', 'R5', 'R10', 'mINP')} if args.test_npz else {}
        out.append(dict(evalset=es, mixture=mx, method=m, n_cells=len(rr), mAP=dm.mean() + np.mean([r['mAP'] - r['dmAP'] for r in rr]),
                        dmAP=dm.mean(), dmAP_ci_lo=np.percentile(bm, 2.5), dmAP_ci_hi=np.percentile(bm, 97.5),
                        dR1=dr.mean(), dR1_ci_lo=np.percentile(br, 2.5), dR1_ci_hi=np.percentile(br, 97.5),
                        pos_cells=int((dm > 0).sum()), **dg, **ext))
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
               inputs={_os.path.basename(x): _sha(x) for x in (f'{args.feats}/{args.base}.npz', f'{args.feats}/{args.adapter}.npz', args.assign, args.test_npz, args.train_ir_npz, args.fixed_selection)
                       if x and x != 'none'},   # round 15: test mode may run with --assign none; held-out and training-identity archives are recorded too
               versions=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, torch=torch.__version__),
               threads={k: _os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
               wall_seconds=round(time.time() - t0, 1), finished=time.strftime('%Y-%m-%d %H:%M:%S %Z'), wccn_diag=WCCN_DIAG),
          open(f'{args.out}/r12_{args.tag}_run_record.json', 'w'), indent=1)
import pickle; pickle.dump({k: v for k, v in perq.items()}, open(f'{args.out}/r12_{args.tag}_perq.pkl', 'wb'))
if args.perq_ext:   # r17
    pickle.dump(dict(meta=dict(tag=args.tag, fits=[int(s) for s in (args.test_seed if args.test_npz else args.split_seeds)], assign_seeds=[int(a) for a in args.assign_seeds],
                               mixes=list(args.mixes), methods=list(METHODS), test_mode=bool(args.test_npz), n_dev_q=int(N_DEV_Q) if args.test_npz else None),
                     records=PQX), open(f'{args.out}/r17_{args.tag}_perq_ext.pkl', 'wb'))
json.dump(dict(r17log=R17LOG, times=TIMES, const_grid=list(args.const_grid), fixed_selection=args.fixed_selection, kappa_a_fixed=args.kappa_a_fixed,
               wccn_shrink=args.wccn_shrink, wccn_shrink_eval=args.wccn_shrink_eval), open(f'{args.out}/r17_{args.tag}_r17log.json', 'w'), indent=1)
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
