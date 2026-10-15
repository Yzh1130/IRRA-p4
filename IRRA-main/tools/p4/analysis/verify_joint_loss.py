"""Independent check of the ACD joint t2i loss decomposition (论文4, 2026-10-15).
Setting: one text row, B gallery items, each item has a frozen full column (logit zF) and a student IR-only column (zI).
Positive set P (same PID) is identical in both branches. Joint target is uniform over the 2|P| positive columns.
"""
import math, torch
torch.set_default_dtype(torch.float64)
g = torch.Generator().manual_seed(20261015)
lse = torch.logsumexp

def l_branch(z, P):                      # multi-positive CE, uniform target over P
    return -(z[P].mean()) + lse(z, 0)
def l_joint(zF, zI, P):                  # uniform target over 2|P| positives, one softmax over 2B columns
    return -(0.5 * zF[P].mean() + 0.5 * zI[P].mean()) + lse(torch.cat([zF, zI]), 0)
def g_term(zF, zI):
    d = lse(zI, 0) - lse(zF, 0)
    return torch.log(2 * torch.cosh(d / 2))

print('== 1. identity  l_joint = 0.5 lF + 0.5 lI + log 2cosh((logZI-logZF)/2)  (random multi-positive cases)')
worst = 0.0
for trial in range(2000):
    B = int(torch.randint(4, 33, (1,), generator=g)); m = int(torch.randint(1, 4, (1,), generator=g))
    P = torch.randperm(B, generator=g)[:m]
    zF = torch.randn(B, generator=g) * 6; zI = torch.randn(B, generator=g) * 6 + torch.randn(1, generator=g) * 4
    lhs = l_joint(zF, zI, P); rhs = 0.5 * l_branch(zF, P) + 0.5 * l_branch(zI, P) + g_term(zF, zI)
    worst = max(worst, abs(float(lhs - rhs)))
print(f'   max |lhs-rhs| over 2000 cases (|P| in 1..3, B in 4..32): {worst:.2e}')

print('== 2. gradient wrt student logits: d/dzI_k = -c_pos*1[k in P]/|P| + c_neg*softmax_I(k)')
B, P = 8, torch.tensor([1, 5])
zF = torch.tensor([3.0, 6.0, 4.5, 2.0, 1.0, 3.5, 0.5, 2.5]); zI0 = torch.tensor([1.0, 3.0, 3.5, 0.0, -1.0, 2.0, 0.0, 1.0])
for name, f in [('A0     (lI)', lambda z: l_branch(z, P)), ('A0-bal (0.5 lI)', lambda z: 0.5 * l_branch(z, P)), ('A1     (joint)', lambda z: l_joint(zF, z, P))]:
    z = zI0.clone().requires_grad_(True); f(z).backward(); gr = z.grad
    sm = torch.softmax(zI0, 0); onehot = torch.zeros(B); onehot[P] = 1 / len(P)
    # solve c_pos, c_neg from two components (one negative, one positive)
    c_neg = float(gr[0] / sm[0]); c_pos = float((c_neg * sm[1] - gr[1]) / onehot[1])
    resid = (gr - (-c_pos * onehot + c_neg * sm)).abs().max()
    print(f'   {name}: c_pos={c_pos:.4f}  c_neg={c_neg:.4f}  (fit residual {float(resid):.1e})')
ZF, ZI = float(torch.exp(zF).sum()), float(torch.exp(zI0).sum())
print(f'   pi_I = ZI/(ZF+ZI) = {ZI/(ZF+ZI):.4f}  -> A1 negative-push coefficient equals pi_I; A0-bal fixes it at 0.5, A0 at 1')
z = zI0.clone().requires_grad_(True); g_term(zF, z).backward()
expect = 0.5 * math.tanh((math.log(ZI) - math.log(ZF)) / 2) * torch.softmax(zI0, 0)
print(f'   g-term gradient = 0.5*tanh(dlogZ/2)*softmax_I : max err {float((z.grad - expect).abs().max()):.1e}; '
      f'here ZI<ZF so it raises every student logit in proportion to softmax_I (hard negatives included)')

print('== 3. free-logit optimum: student positives do NOT generally equal the full positives')
def free_opt(zF, P, steps=20000, lr=0.5):
    z = torch.zeros(len(zF), requires_grad=True); opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad(); l = l_joint(zF, z, P); l.backward(); opt.step()
    return z.detach()
cases = [('|P|=1, confident full row', torch.tensor([8.0, 2.0, 1.0, 0.5, 0.0, -1.0]), torch.tensor([0])),
         ('|P|=1, full row with strong negative', torch.tensor([5.0, 4.8, 3.0, 1.0, 0.0, -1.0]), torch.tensor([0])),
         ('|P|=2, unequal full positives', torch.tensor([5.0, 3.0, 4.0, 1.0, 0.0, -1.0]), torch.tensor([0, 1]))]
for name, zF_, P_ in cases:
    zs = free_opt(zF_, P_); ZF_ = float(torch.exp(zF_).sum())
    pred = math.log(ZF_ / len(P_)); negmax = float(zs[[i for i in range(len(zF_)) if i not in P_.tolist()]].max())
    print(f'   {name}: full pos {[round(float(v),2) for v in zF_[P_]]}, student pos {[round(float(v),2) for v in zs[P_]]}, '
          f'analytic log(ZF/|P|)={pred:.2f}, student max-neg {negmax:.1f} (diverging), log(ZI/ZF)={float(lse(zs,0))-math.log(ZF_):+.3f}')

print('== 4. unit-vector encoder: the free-logit optimum is not reachable (shared item embeddings across rows, |logit|<=tau)')
torch.manual_seed(7)
tau, d, n_id, per = 50.0, 16, 24, 2               # 24 identities x 2 images = 48 items; 48 texts (one per item)
B = n_id * per
pid = torch.arange(n_id).repeat_interleave(per)
proto = torch.nn.functional.normalize(torch.randn(n_id, d), dim=1)
T = torch.nn.functional.normalize(proto[pid] + 0.35 * torch.randn(B, d), dim=1)            # texts
CF = torch.nn.functional.normalize(proto[pid] + 0.45 * torch.randn(B, d), dim=1)           # frozen full items
W = torch.nn.Parameter(torch.nn.functional.normalize(proto[pid] + 0.9 * torch.randn(B, d), dim=1).clone())  # student IR items (free unit vectors)
opt = torch.optim.Adam([W], lr=0.01)
posm = (pid[:, None] == pid[None, :])
def joint_t2i_rows(CI):
    zF = tau * T @ CF.T; zI = tau * T @ CI.T
    allz = torch.cat([zF, zI], 1); pm = torch.cat([posm, posm], 1).double()
    return -(allz * pm).sum(1) / pm.sum(1) + lse(allz, 1), zF, zI
def i2t_student(CI):
    zI = tau * T @ CI.T; pm = posm.double()
    return (-(zI * pm).sum(0) / pm.sum(0) + lse(zI, 0)).mean()
for step in range(4000):
    opt.zero_grad(); CI = torch.nn.functional.normalize(W, dim=1)
    rows, zF, zI = joint_t2i_rows(CI); loss = rows.mean() + i2t_student(CI); loss.backward(); opt.step()
with torch.no_grad():
    CI = torch.nn.functional.normalize(W, dim=1); rows, zF, zI = joint_t2i_rows(CI)
    dlog = lse(zI, 1) - lse(zF, 1)
    posF = zF[posm].view(B, per); posI = zI[posm].view(B, per)
    target = (lse(zF, 1) - math.log(per))[:, None]
    print(f'   per-row log(ZI/ZF): mean {float(dlog.mean()):+.3f}, sd {float(dlog.std()):.3f}, range [{float(dlog.min()):+.2f}, {float(dlog.max()):+.2f}]  (free-logit optimum: all 0)')
    print(f'   student pos - full pos (logit units): mean {float((posI-posF).mean()):+.2f}, sd {float((posI-posF).std()):.2f}; '
          f'student pos - log(ZF/|P|): mean {float((posI-target).mean()):+.2f}, sd {float((posI-target).std()):.2f}')
    print(f'   student negative logits min {float(zI[~posm].min()):.1f} (bounded below by -tau={-tau:.0f}; free optimum needs -inf)')

print('== 5. general split: target mass w_I on the IR branch  =>  l = w_F lF + w_I lI + [log(ZF+ZI) - w_F logZF - w_I logZI]')
print('      extra term = H(w) + KL(w || pi), pi_I = ZI/(ZF+ZI); stationary point ZI/ZF = w_I/w_F')
for wI in [1/6, 0.2, 0.5]:
    zF_ = torch.tensor([5.0, 4.0, 2.0, 1.0, 0.0]); P_ = torch.tensor([0])
    z = torch.zeros(5, requires_grad=True); opt = torch.optim.Adam([z], lr=0.3)
    for _ in range(15000):
        opt.zero_grad()
        l = -((1 - wI) * zF_[P_].mean() + wI * z[P_].mean()) + lse(torch.cat([zF_, z]), 0)
        l.backward(); opt.step()
    r = float(torch.exp(lse(z.detach(), 0) - lse(zF_, 0)))
    print(f'   w_I={wI:.3f}: optimised ZI/ZF={r:.4f}, predicted w_I/w_F={wI/(1-wI):.4f}')
print('   Deployment: gallery with IR-only fraction rho, item-wise mass parity ZI=ZF  =>  expected IR-only softmax share = rho r/(rho r+1-rho) = rho when r=1.')
print('   So the 50/50 training split encodes availability-neutral mass for every rho, not a 50% deployment mix.')
