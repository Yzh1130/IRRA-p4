"""Synthetic archives for timing and code-path tests only (no official-test data): a held-out archive with the official test
split's size (553 identities, ~1,470 items, ~2,490 queries) and a training-identity IR archive (1076 identities, 2772 items),
both built by perturbing replicated development identities. Retrieval numbers computed on them mean nothing."""
import sys, numpy as np
src, out_test, out_train = sys.argv[1:4]
z = np.load(src); rng = np.random.default_rng(424242)
tp, gp, qraw = z['clean/txt_pids'], z['clean/img_pids'], z['clean/query_raw_pids']
tp2raw = dict(zip(tp.tolist(), qraw.tolist())); item_raw = np.array([tp2raw[p] for p in gp.tolist()])
uid = np.unique(qraw); D = z['clean/f_ir'].shape[1]
SC = ['clean', 'rgb_missing', 'ir_missing', 'rgb_noise_010', 'ir_noise_010', 'rgb_blur_k7', 'ir_blur_k7']
def build(n_ids, raw0):
    picks = rng.choice(uid, n_ids, replace=True); Q, G = [], []
    for new, u in enumerate(picks):
        Q.append((new, np.where(qraw == u)[0])); G.append((new, np.where(item_raw == u)[0]))
    off_id = {new: rng.standard_normal(D) * 0.02 for new in range(n_ids)}
    out = {}
    qi = np.concatenate([q for _, q in Q]); qn = np.concatenate([[n] * len(q) for n, q in Q])
    gi = np.concatenate([g for _, g in G]); gn = np.concatenate([[n] * len(g) for n, g in G])
    T = z['clean/text_features'][qi].astype(np.float32) + np.stack([off_id[n] for n in qn]).astype(np.float32) + rng.standard_normal((len(qi), D)).astype(np.float32) * 0.005
    out['clean/text_features'] = T.astype(z['clean/text_features'].dtype)
    out['clean/txt_pids'] = qn.astype(z['clean/txt_pids'].dtype); out['clean/query_raw_pids'] = (raw0 + qn).astype(z['clean/query_raw_pids'].dtype)
    out['clean/img_pids'] = gn.astype(z['clean/img_pids'].dtype)
    item_noise = rng.standard_normal((len(gi), D)).astype(np.float32) * 0.005; idoff = np.stack([off_id[n] for n in gn]).astype(np.float32)
    for k in z.files:
        sc, key = k.split('/')
        if key in ('f_rgb', 'f_ir'):
            a = z[k][gi].astype(np.float32)
            if not (sc == 'rgb_missing' and key == 'f_rgb') and not (sc == 'ir_missing' and key == 'f_ir'): a = a + idoff + item_noise
            out[k] = a.astype(z[k].dtype)
    return out, len(qi), len(gi)
t, nq, ng = build(553, 800000); np.savez(out_test, **t); print('synthetic held-out archive:', nq, 'queries', ng, 'items', 553, 'identities')
# training-identity IR archive: 1076 identities
picks = rng.choice(uid, 1076, replace=True); F, R = [], []
for new, u in enumerate(picks):
    ix = np.where(item_raw == u)[0]; off = rng.standard_normal(D) * 0.02
    F.append(z['clean/f_ir'][ix].astype(np.float32) + off + rng.standard_normal((len(ix), D)).astype(np.float32) * 0.005); R += [900000 + new] * len(ix)
F = np.concatenate(F); np.savez(out_train, f_ir=F, f_rgb=F, raw_pids=np.array(R)); print('synthetic training-identity archive:', F.shape, len(set(R)), 'identities')
