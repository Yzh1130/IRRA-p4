"""P4 calibration protocol for mixed-availability galleries (论文4, 2026-10-15).

Protocols
  P-ind   identity-isolated cross-fitting. Dev identities are split 2-fold (repeated over split seeds).
          Calibrators are fitted ONLY on (fit-fold queries x fit-fold gallery items).
          Evaluation uses (eval-fold queries x eval-fold gallery items), i.e. half-size galleries.
          Every method is scored on exactly the same eval galleries; absolute numbers are NOT comparable
          with the 478-item gallery used before.
  P-trans legacy protocol of calib_exp3/calib_exp4: queries are split by identity, but Platt is fitted on
          (fit-fold queries x ALL 478 gallery items) and evaluated on (eval-fold queries x ALL items).
          Reported separately; it is not identity-independent (eval-fold gallery items enter the fit as negatives).

Score input
  s = raw drop-blank cosine  <t, norm(available-modality feature sum)>  for every method except raw_add.
  raw_add = legacy blank-embedding fusion  <t, norm(f_rgb + f_ir)>  with the encoded blank for the missing modality.

Methods
  raw_add, drop          uncalibrated
  bstar                  drop + per-availability Platt (logistic; F fitted on clean, I on rgb_missing, R on ir_missing,
                         homogeneous fit matrices restricted per protocol)
  rk_affine              per-availability positive slope + bias, trained with the listwise ranking loss
  rk_affine_hub          rk_affine - beta_a * h(g)                     (same loss)
  rk_pwl                 per-availability monotone piecewise-linear map (same loss)
  rk_item                rk_affine_hub + omega_a(s) * u(x_g)   omega_a(s) = -relu(gamma_a - kappa_a s),
                         u = softplus(MLP(x_g)) = "per-item score correction"  (RSC head; same loss)
  Ranking-loss methods share: identity split, sampler (item states drawn uniformly from the M1 state set each step),
  loss, optimiser (Adam lr, steps), init seed. No per-method tuning (equal tuning budget = 0).

Per-item inputs (rk_item, rk_affine_hub) use only modalities that are actually available for the item:
  x_g = [onehot(F,I,R), h(g), |f_rgb| if RGB present else 0, |f_ir| if IR present else 0, cos(f_rgb,f_ir) if F else 0].
  Degradation type (noise/blur) is never an input. h(g) = mean of the top-k cosines between the item embedding and a
  reference text bank = fit-fold query texts (texts of the item's own identity excluded) -- no eval-fold text is used.

Outputs (out_dir): splits.json, params.jsonl, assignments_M4.csv, perq_<model>.npz, summary.csv, comparisons.csv,
          branch_mass.csv (log Z_I/Z_F per query at tau=50 and IR-only softmax share in M3/M4), meta.json
"""
import argparse, json, os, time
import numpy as np
import torch

SC = ['clean', 'rgb_missing', 'ir_missing', 'rgb_noise_010', 'ir_noise_010', 'rgb_blur_k7', 'ir_blur_k7']
AVAIL = {'clean': 0, 'rgb_noise_010': 0, 'ir_noise_010': 0, 'rgb_blur_k7': 0, 'ir_blur_k7': 0, 'rgb_missing': 1, 'ir_missing': 2}
AV_NAME = ['F', 'I', 'R']
M1_STATES = ['clean', 'rgb_noise_010', 'ir_noise_010', 'rgb_missing', 'ir_missing', 'rgb_blur_k7']
FIT_STATE = {0: 'clean', 1: 'rgb_missing', 2: 'ir_missing'}
MIXES = ['M1', 'M2', 'M3', 'M4']
RANK_METHODS = ['rk_affine', 'rk_affine_hub', 'rk_pwl', 'rk_item']
ALL_METHODS = ['raw_add', 'drop', 'bstar'] + RANK_METHODS
HOMOG = ['clean', 'rgb_missing', 'ir_missing']


def l2n(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


class Feats:
    def __init__(self, path):
        z = np.load(path)
        get = lambda s, k: z[f'{s}/{k}'] if f'{s}/{k}' in z.files else z[f'clean/{k}']
        self.t = l2n(z['clean/text_features'].astype(np.float64))
        self.tp = z['clean/txt_pids']; self.gp = z['clean/img_pids']; self.qraw = z['clean/query_raw_pids']
        self.pos = self.tp[:, None] == self.gp[None, :]
        self.fr = {s: get(s, 'f_rgb').astype(np.float64) for s in SC}
        self.fi = {s: get(s, 'f_ir').astype(np.float64) for s in SC}
        self.emb_drop, self.S_drop, self.S_add = {}, {}, {}
        for s in SC:
            a = AVAIL[s]
            e = l2n(self.fi[s]) if a == 1 else (l2n(self.fr[s]) if a == 2 else l2n(self.fr[s] + self.fi[s]))
            self.emb_drop[s] = e
            self.S_drop[s] = self.t @ e.T
            self.S_add[s] = self.t @ l2n(self.fr[s] + self.fi[s]).T
        # per-item raw inputs from available modalities only
        self.xraw = {}
        for s in SC:
            a = AVAIL[s]
            nr = np.linalg.norm(self.fr[s], axis=1) if a in (0, 2) else np.zeros(len(self.gp))
            ni = np.linalg.norm(self.fi[s], axis=1) if a in (0, 1) else np.zeros(len(self.gp))
            cs = (l2n(self.fr[s]) * l2n(self.fi[s])).sum(1) if a == 0 else np.zeros(len(self.gp))
            self.xraw[s] = np.stack([nr, ni, cs], 1)


def eval_rows(S, pos):
    order = np.argsort(-S, axis=1, kind='stable')
    m = np.take_along_axis(pos, order, axis=1)
    cum = np.cumsum(m, axis=1)
    prec = cum / np.arange(1, S.shape[1] + 1)
    AP = (prec * m).sum(1) / np.maximum(m.sum(1), 1)
    R1 = (m[:, 0]).astype(np.float64)
    return R1 * 100, AP * 100


def fit_logistic(x, y, iters=60):
    mu, sd = x.mean(), x.std() + 1e-12
    z = (x - mu) / sd
    w = np.zeros(2); X = np.c_[np.ones_like(z), z]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        w = w + np.linalg.solve(H, X.T @ (y - p))
    return float(w[1] / sd), float(w[0] - w[1] * mu / sd)


def hub_scores(F, items, bank_q, k):
    """h_s(g) for every state s, items g in `items`; bank = query texts bank_q (own-identity texts excluded)."""
    out = {}
    tb = F.t[bank_q]; bank_pid = F.tp[bank_q]
    for s in SC:
        sim = F.emb_drop[s][items] @ tb.T                         # items x bank
        own = F.gp[items][:, None] == bank_pid[None, :]
        sim = np.where(own, -np.inf, sim)
        kk = min(k, sim.shape[1] - 1)
        top = -np.sort(-sim, axis=1)[:, :kk]
        out[s] = top.mean(1)
    return out


def inv_softplus(y):
    return float(np.log(np.expm1(y)))


class Calib(torch.nn.Module):
    def __init__(self, kind, knots=None, x_mu=None, x_sd=None):
        super().__init__()
        self.kind = kind
        P = torch.nn.Parameter
        self.a = P(torch.full((3,), inv_softplus(50.0), dtype=torch.float64))
        self.b = P(torch.zeros(3, dtype=torch.float64))
        if kind in ('rk_affine_hub', 'rk_item'):
            self.beta = P(torch.zeros(3, dtype=torch.float64))
        if kind == 'rk_pwl':
            self.knots = torch.tensor(knots, dtype=torch.float64)              # 5 knots -> 6 segments
            self.w = P(torch.full((3, 6), inv_softplus(50.0), dtype=torch.float64))
        if kind == 'rk_item':
            self.kappa = P(torch.full((3,), inv_softplus(1.0), dtype=torch.float64))
            self.gamma = P(torch.full((3,), 0.35, dtype=torch.float64))
            self.mlp = torch.nn.Sequential(torch.nn.Linear(7, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)).double()
            with torch.no_grad():
                self.mlp[2].bias.fill_(-2.0)
            self.x_mu = torch.tensor(x_mu); self.x_sd = torch.tensor(x_sd)

    def forward(self, s, av, h=None, xr=None):
        # s: Q x G scores; av: G availability ids; h: G hub; xr: G x 3 raw item inputs
        sp = torch.nn.functional.softplus
        if self.kind == 'rk_pwl':
            e = self.knots
            w = sp(self.w)[av]                                                   # G x 6
            seg = [torch.clamp(s - e[0], max=0.0)]                               # (-inf, e0]
            for j in range(len(e) - 1):
                seg.append(torch.clamp(s - e[j], min=0.0, max=float(e[j + 1] - e[j])))
            seg.append(torch.clamp(s - e[-1], min=0.0))                          # (e_last, inf)
            out = sum(w[None, :, j] * seg[j] for j in range(6))
            return out + self.b[av][None, :]
        out = sp(self.a)[av][None, :] * s + self.b[av][None, :]
        if self.kind in ('rk_affine_hub', 'rk_item'):
            out = out - (self.beta[av] * h)[None, :]
        if self.kind == 'rk_item':
            onehot = torch.nn.functional.one_hot(av, 3).double()
            x = torch.cat([onehot, ((h - self.x_mu[0]) / self.x_sd[0])[:, None], (xr - self.x_mu[1:]) / self.x_sd[1:]], 1)
            u = sp(self.mlp(x)).squeeze(1)                                       # per-item score correction >= 0
            omega = -torch.relu(self.gamma[av][None, :] - sp(self.kappa)[av][None, :] * s)
            out = out + 50.0 * omega * u[None, :]
        return out


def train_rank(kind, F, qf, gf, hub_f, steps, lr, seed, knots=None):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    Sst = {s: torch.tensor(F.S_drop[s][np.ix_(qf, gf)]) for s in M1_STATES}
    Hst = {s: torch.tensor(hub_f[s]) for s in M1_STATES}
    Xst = {s: torch.tensor(F.xraw[s][gf]) for s in M1_STATES}
    allx = np.concatenate([np.c_[hub_f[s], F.xraw[s][gf]] for s in M1_STATES])
    x_mu = allx.mean(0); x_sd = allx.std(0) + 1e-6
    model = Calib(kind, knots=knots, x_mu=x_mu, x_sd=x_sd)
    pos = torch.tensor(F.pos[np.ix_(qf, gf)])
    npos = pos.sum(1).clamp(min=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    st_idx = np.arange(len(M1_STATES))
    for _ in range(steps):
        choice = rng.choice(st_idx, size=len(gf))
        S = torch.zeros(len(qf), len(gf), dtype=torch.float64); h = torch.zeros(len(gf), dtype=torch.float64)
        xr = torch.zeros(len(gf), 3, dtype=torch.float64); av = torch.zeros(len(gf), dtype=torch.long)
        for j, s in enumerate(M1_STATES):
            cols = torch.tensor(np.where(choice == j)[0])
            if len(cols) == 0:
                continue
            S[:, cols] = Sst[s][:, cols]; h[cols] = Hst[s][cols]; xr[cols] = Xst[s][cols]; av[cols] = AVAIL[s]
        L = model(S, av, h, xr)
        logp = L - torch.logsumexp(L, 1, keepdim=True)
        loss = -((logp * pos).sum(1) / npos).mean()
        if kind == 'rk_item':
            loss = loss + 1e-4 * sum((p ** 2).sum() for p in model.mlp.parameters())
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def params_of(model):
    return {k: v.detach().cpu().numpy().round(6).tolist() for k, v in model.state_dict().items()}


def make_m4(n, seeds):
    rows = {}
    for sd in seeds:
        r = np.random.default_rng(sd + 40000)
        rows[sd] = np.where(r.random(n) < 0.5, 'rgb_missing', 'clean')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats_dir', required=True)
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--assignments', required=True, help='B12 assignments.csv (M1/M2/M3, seeds 2026-2028)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--split_seeds', nargs='+', type=int, default=[2026, 2027, 2028, 2029, 2030])
    ap.add_argument('--assign_seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    ap.add_argument('--k_hub', type=int, default=16)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--init_seed', type=int, default=20261015)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--skip_rank', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=False)
    import pandas as pd
    A = pd.read_csv(args.assignments)
    first = Feats(os.path.join(args.feats_dir, args.models[0] + '.npz'))
    n_items = len(first.gp)
    assign = {}
    for (mx, sd), g in A.groupby(['mixture', 'seed']):
        assign[(mx, int(sd))] = g.sort_values('gallery_index').state.values
    m4 = make_m4(n_items, args.assign_seeds)
    for sd in args.assign_seeds:
        assign[('M4', sd)] = m4[sd]
    pd.DataFrame([{'mixture': 'M4', 'seed': sd, 'gallery_index': j, 'state': m4[sd][j]} for sd in args.assign_seeds for j in range(n_items)]) \
        .to_csv(os.path.join(args.out_dir, 'assignments_M4.csv'), index=False)

    # identity folds (by query_raw_pid; txt/img pid map 1:1)
    uq = np.unique(first.qraw)
    tp2raw = dict(zip(first.tp.tolist(), first.qraw.tolist()))
    item_raw = np.array([tp2raw[p] for p in first.gp.tolist()])
    splits = {}
    for ss in args.split_seeds:
        perm = np.random.default_rng(ss).permutation(uq)
        fold_of = {int(p): i % 2 for i, p in enumerate(perm)}
        splits[ss] = fold_of
    json.dump({str(ss): {'fold0': sorted(int(p) for p, f in fo.items() if f == 0), 'fold1': sorted(int(p) for p, f in fo.items() if f == 1)}
               for ss, fo in splits.items()}, open(os.path.join(args.out_dir, 'splits.json'), 'w'))

    plog = open(os.path.join(args.out_dir, 'params.jsonl'), 'w')
    summary_rows, comp_rows, branch_rows = [], [], []
    rng_boot = np.random.default_rng(2026)
    for mname in args.models:
        t0 = time.time()
        F = first if mname == args.models[0] else Feats(os.path.join(args.feats_dir, mname + '.npz'))
        nq = len(F.tp)
        # per-query accumulators: protocol -> mixture -> method -> [split, assign, query]
        R = {p: {mx: {m: np.full((len(args.split_seeds), len(args.assign_seeds), nq), np.nan) for m in ALL_METHODS} for mx in MIXES + HOMOG}
             for p in ['P-ind', 'P-trans']}
        AP = {p: {mx: {m: np.full_like(R[p][mx][m], np.nan) for m in ALL_METHODS} for mx in MIXES + HOMOG} for p in R}
        GRP = {mx: np.full((len(args.split_seeds), len(args.assign_seeds), nq), -1, dtype=np.int8) for mx in MIXES}
        for si, ss in enumerate(args.split_seeds):
            fold_of = splits[ss]
            qfold = np.array([fold_of[int(p)] for p in F.qraw]); gfold = np.array([fold_of[int(p)] for p in item_raw])
            for f in (0, 1):
                qf = np.where(qfold != f)[0]; qe = np.where(qfold == f)[0]
                gf = np.where(gfold != f)[0]; ge = np.where(gfold == f)[0]
                fit_ids = set(F.qraw[qf].tolist()) | set(item_raw[gf].tolist())
                eval_ids = set(F.qraw[qe].tolist()) | set(item_raw[ge].tolist())
                assert not (fit_ids & eval_ids), 'P-ind: fit and eval identities overlap'
                allg = np.arange(n_items)
                # Platt: P-ind (fit queries x fit items) and P-trans (fit queries x all items)
                platt = {}
                for proto, gfit in [('P-ind', gf), ('P-trans', allg)]:
                    pp = {}
                    for a, fs in FIT_STATE.items():
                        x = F.S_drop[fs][np.ix_(qf, gfit)].ravel(); y = F.pos[np.ix_(qf, gfit)].ravel().astype(float)
                        pp[a] = fit_logistic(x, y)
                    platt[proto] = pp
                    plog.write(json.dumps({'model': mname, 'protocol': proto, 'split_seed': ss, 'eval_fold': f, 'method': 'bstar',
                                           'params': {AV_NAME[a]: pp[a] for a in pp}}) + '\n')
                # ranking calibrators (P-ind only)
                models = {}
                if not args.skip_rank:
                    hub_f = hub_scores(F, gf, qf, args.k_hub)
                    hub_e = hub_scores(F, ge, qf, args.k_hub)
                    fitS = np.concatenate([F.S_drop[s][np.ix_(qf, gf)].ravel() for s in M1_STATES])
                    knots = np.quantile(fitS, [0.1, 0.3, 0.5, 0.7, 0.9]).tolist()
                    for kind in RANK_METHODS:
                        mdl = train_rank(kind, F, qf, gf, hub_f, args.steps, args.lr, args.init_seed + 97 * si + f, knots=knots)
                        models[kind] = mdl
                        plog.write(json.dumps({'model': mname, 'protocol': 'P-ind', 'split_seed': ss, 'eval_fold': f, 'method': kind,
                                               'params': params_of(mdl), 'knots': knots if kind == 'rk_pwl' else None}) + '\n')
                # evaluation
                for proto, geval in [('P-ind', ge), ('P-trans', allg)]:
                    pos_e = F.pos[np.ix_(qe, geval)]
                    for mx in MIXES + HOMOG:
                        for ai, asd in enumerate(args.assign_seeds):
                            if mx in HOMOG:
                                if ai > 0:
                                    continue
                                st = np.array([mx] * n_items)
                            else:
                                st = assign[(mx, asd)]
                            ste = st[geval]
                            av = np.array([AVAIL[s] for s in ste])
                            Sd = np.zeros((len(qe), len(geval))); Sa = np.zeros_like(Sd)
                            for s in np.unique(ste):
                                cols = np.where(ste == s)[0]
                                Sd[:, cols] = F.S_drop[s][np.ix_(qe, geval[cols])]
                                Sa[:, cols] = F.S_add[s][np.ix_(qe, geval[cols])]
                            if mx in MIXES:
                                g = np.full(len(qe), -1, dtype=np.int8)
                                for qi in range(len(qe)):
                                    avs = np.unique(av[pos_e[qi]])
                                    g[qi] = avs[0] if len(avs) == 1 else 3
                                if proto == 'P-ind':
                                    GRP[mx][si, ai, qe] = g
                            scores = {'raw_add': Sa, 'drop': Sd}
                            pp = platt[proto]
                            scores['bstar'] = np.stack([pp[a][0] * Sd[:, j] + pp[a][1] for j, a in enumerate(av)], 1)
                            if proto == 'P-ind' and models:
                                # hub/x for eval items in their states
                                pos_in_ge = {int(v): k for k, v in enumerate(ge)}
                                hcol = np.array([hub_e[s][pos_in_ge[int(geval[j])]] for j, s in enumerate(ste)])
                                xcol = np.stack([F.xraw[s][geval[j]] for j, s in enumerate(ste)])
                                with torch.no_grad():
                                    for kind, mdl in models.items():
                                        scores[kind] = mdl(torch.tensor(Sd), torch.tensor(av), torch.tensor(hcol), torch.tensor(xcol)).numpy()
                            for m, Sm in scores.items():
                                r1, apv = eval_rows(Sm, pos_e)
                                R[proto][mx][m][si, ai, qe] = r1; AP[proto][mx][m][si, ai, qe] = apv
        # save per-query arrays
        np.savez_compressed(os.path.join(args.out_dir, f'perq_{mname}.npz'),
                            **{f'{p}|{mx}|{m}|R1': R[p][mx][m].astype(np.float32) for p in R for mx in R[p] for m in ALL_METHODS},
                            **{f'{p}|{mx}|{m}|AP': AP[p][mx][m].astype(np.float32) for p in R for mx in R[p] for m in ALL_METHODS},
                            **{f'group|{mx}': GRP[mx] for mx in MIXES}, query_pid=F.qraw)
        # summaries
        for p in R:
            for mx in MIXES + HOMOG:
                for m in ALL_METHODS:
                    r = R[p][mx][m]; a = AP[p][mx][m]
                    if np.isnan(r).all():
                        continue
                    row = {'model': mname, 'protocol': p, 'mixture': mx, 'method': m, 'group': 'all',
                           'R1': float(np.nanmean(r)), 'mAP': float(np.nanmean(a)), 'n_query_evals': int((~np.isnan(r)).sum())}
                    summary_rows.append(row)
                    if mx in MIXES and p == 'P-ind':
                        for gi, gname in enumerate(['F', 'I', 'R', 'mixed']):
                            msk = (GRP[mx] == gi) & ~np.isnan(r)
                            if msk.sum() == 0:
                                continue
                            summary_rows.append({'model': mname, 'protocol': p, 'mixture': mx, 'method': m, 'group': gname,
                                                 'R1': float(r[msk].mean()), 'mAP': float(a[msk].mean()),
                                                 'n_query_evals': int(msk.sum())})
        # paired comparisons (P-ind; per-query mean over split x assign, PID-cluster bootstrap)
        pairs = [('bstar', 'drop'), ('drop', 'raw_add'), ('bstar', 'raw_add'),
                 ('rk_affine', 'bstar'), ('rk_affine_hub', 'rk_affine'), ('rk_pwl', 'rk_affine'),
                 ('rk_item', 'rk_affine_hub'), ('rk_item', 'rk_pwl'), ('rk_item', 'bstar')]
        pid = F.qraw; up = np.unique(pid); idx = [np.where(pid == u)[0] for u in up]
        for mx in MIXES + ['rgb_missing', 'clean']:
            for a_m, b_m in pairs:
                ra, rb = R['P-ind'][mx][a_m], R['P-ind'][mx][b_m]
                if np.isnan(ra).all() or np.isnan(rb).all():
                    continue
                for met, X, Y in [('R1', ra, rb), ('mAP', AP['P-ind'][mx][a_m], AP['P-ind'][mx][b_m])]:
                    d = np.nanmean(X.reshape(-1, nq), 0) - np.nanmean(Y.reshape(-1, nq), 0)
                    sums = np.array([d[i].sum() for i in idx]); cnt = np.array([len(i) for i in idx])
                    bs = [sums[s].sum() / cnt[s].sum() for s in (rng_boot.integers(0, len(up), len(up)) for _ in range(args.n_boot))]
                    per_split = [float(np.nanmean(X[si]) - np.nanmean(Y[si])) for si in range(len(args.split_seeds))]
                    comp_rows.append({'model': mname, 'protocol': 'P-ind', 'mixture': mx, 'a': a_m, 'b': b_m, 'metric': met,
                                      'diff': float(d.mean()), 'ci_lo': float(np.percentile(bs, 2.5)), 'ci_hi': float(np.percentile(bs, 97.5)),
                                      'per_split_min': min(per_split), 'per_split_max': max(per_split)})
        # branch-mass diagnostic (full dev gallery, raw drop-blank, tau=50): Z_I/Z_F per query, actual IR-only softmax share
        SF = F.S_drop['clean']; SI = F.S_drop['rgb_missing']
        mxs = np.maximum(SF.max(1), SI.max(1))[:, None]
        lr = np.log(np.exp(50 * (SI - mxs)).sum(1) / np.exp(50 * (SF - mxs)).sum(1))
        bm = {'model': mname, 'median_log_ZI_over_ZF': float(np.median(lr)), 'q25': float(np.percentile(lr, 25)), 'q75': float(np.percentile(lr, 75))}
        for mx in ['M3', 'M4']:
            sh, fr = [], []
            for asd in args.assign_seeds:
                isI = assign[(mx, asd)] == 'rgb_missing'
                eI = np.exp(50 * (SI[:, isI] - mxs)).sum(1); eF = np.exp(50 * (SF[:, ~isI] - mxs)).sum(1)
                sh.append(float((eI / (eI + eF)).mean())); fr.append(float(isI.mean()))
            bm[f'{mx}_ir_fraction'] = float(np.mean(fr))
            bm[f'{mx}_ir_softmax_share'] = float(np.mean(sh))
        branch_rows.append(bm)
        print(f'{mname}: done in {time.time() - t0:.0f}s', flush=True)
    plog.close()
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, 'summary.csv'), index=False)
    pd.DataFrame(comp_rows).to_csv(os.path.join(args.out_dir, 'comparisons.csv'), index=False)
    pd.DataFrame(branch_rows).to_csv(os.path.join(args.out_dir, 'branch_mass.csv'), index=False)
    json.dump({'protocols': {'P-ind': 'fit on fit-fold queries x fit-fold items; evaluate eval-fold queries x eval-fold items (half gallery)',
                             'P-trans': 'legacy: fit on fit-fold queries x all items; evaluate eval-fold queries x all items (478 gallery)'},
               'score_input': 'raw drop-blank cosine (raw_add uses blank-embedding fusion)',
               'hub_bank': 'fit-fold query texts, own-identity texts excluded; k=%d' % args.k_hub,
               'rank_training': {'sampler': 'item states uniform over M1 states each step', 'steps': args.steps, 'lr': args.lr,
                                 'optimizer': 'Adam', 'init_seed': args.init_seed, 'tuning': 'none (equal budget)'},
               'split_seeds': args.split_seeds, 'assign_seeds': args.assign_seeds,
               'M4': 'Bernoulli(0.5) rgb_missing vs clean per item, rng=default_rng(seed+40000); saved in assignments_M4.csv',
               'groups': 'positive-availability group of a query within its eval gallery: F, I, R (all positives one group) or mixed'},
              open(os.path.join(args.out_dir, 'meta.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
