"""Round-18 xcam reproduction, paired identity bootstrap, ratio and coverage.
Functions group_pair/load_pooled/prefix_of copied verbatim from frozen judge_test.py
(1375f6add366f72e2ff9677d16b6a84fab950a2ece4d99a94494dd5c123e126a).
per_query/boot_ci copied verbatim from frozen r17_judge.py
(3a5d73fcec8cb448cd5bbe2a345a3f35c3c4bddf67c61e78cf69fd41bbc74c57).
"""
import argparse, glob, json, os, pickle, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
CODE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(CODE))
from tools.p4.r18.common import protocol_prefix, write_json
A=['M4','M6c','M1g','M4c60','M9']
B=['M5e','M5f','M4id']
a=SimpleNamespace(evalset='test',boot=4000)
DATA={}

def prefix_of(d):
    hits = sorted(glob.glob(os.path.join(d, 'r12_*_summary.csv')))
    if len(hits) != 1: raise SystemExit(f'{d}: expected exactly one r12_*_summary.csv, found {len(hits)}')
    return hits[0][:-len('summary.csv')]


def load_pooled(dirs):
    cells, diag, sel, ml, cx, recs, pq = [], [], [], [], [], [], {}
    for d in dirs:
        p = prefix_of(d)
        c = pd.read_csv(p + 'cells.csv'); c['evalset'] = c['evalset'].astype(str); cells.append(c)
        dg = pd.read_csv(p + 'diag_cells.csv'); dg['evalset'] = dg['evalset'].astype(str); diag.append(dg)
        sel += json.load(open(p + 'selection.json')); recs.append(json.load(open(p + 'run_record.json')))
        if os.path.exists(p + 'maplog.csv'): ml.append(pd.read_csv(p + 'maplog.csv'))
        if os.path.exists(p + 'ctxlog.csv'): cx.append(pd.read_csv(p + 'ctxlog.csv'))
        part = pickle.load(open(p + 'perq.pkl', 'rb'))
        for k, dct in part.items():
            tgt = pq.setdefault(k, {})
            for q, lst in dct.items(): tgt.setdefault(q, []).extend(lst)
    return (pd.concat(cells, ignore_index=True), pd.concat(diag, ignore_index=True), sel,
            pd.concat(ml, ignore_index=True) if ml else None, pd.concat(cx, ignore_index=True) if cx else None, recs, pq)


def group_pair(pq, X, Y, mixes, ids, rng_seed=7, nboot=4000):
    """mean over queries of the per-query difference averaged over cells and mixtures; identity-cluster bootstrap"""
    qs = None
    for mx in mixes:
        k1, k2 = (a.evalset, mx, X), (a.evalset, mx, Y)
        if k1 not in pq or k2 not in pq: return None
        s_ = set(pq[k1]) & set(pq[k2]); qs = s_ if qs is None else qs & s_
    qs = sorted(qs); dm = np.zeros(len(qs)); dr = np.zeros(len(qs))
    for mx in mixes:
        P1, P2 = pq[(a.evalset, mx, X)], pq[(a.evalset, mx, Y)]
        dm += np.array([np.mean([u[0] - v[0] for u, v in zip(P1[q], P2[q])]) for q in qs])
        dr += np.array([np.mean([u[1] - v[1] for u, v in zip(P1[q], P2[q])]) for q in qs])
    dm /= len(mixes); dr /= len(mixes)
    qid = ids[np.array(qs)]; u = np.unique(qid); idx = [np.where(qid == v)[0] for v in u]
    sm = np.array([dm[i].sum() for i in idx]); sr = np.array([dr[i].sum() for i in idx]); c = np.array([len(i) for i in idx])
    rb = np.random.default_rng(rng_seed); bm, br = np.empty(nboot), np.empty(nboot)
    for b in range(nboot):
        x = rb.integers(0, len(u), len(u)); bm[b] = sm[x].sum() / c[x].sum(); br[b] = sr[x].sum() / c[x].sum()
    return dict(dmAP=float(dm.mean()), lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)),
                dR1=float(dr.mean()), r_lo=float(np.percentile(br, 2.5)), r_hi=float(np.percentile(br, 97.5)), n_q=len(qs), n_id=len(u))


def boot_ci(per_q, qid, seed=7):
    u = np.unique(qid); idx = [np.where(qid == v)[0] for v in u]
    s = np.array([per_q[i].sum() for i in idx]); c = np.array([len(i) for i in idx])
    rb = np.random.default_rng(seed); b = np.empty(a.boot)
    for k in range(a.boot):
        x = rb.integers(0, len(u), len(u)); b[k] = s[x].sum() / c[x].sum()
    return float(per_q.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def per_query(seed, method, mixes, metric):
    """per-query value averaged over all (fit, assignment seed) cells and over the given mixtures; explicit keys"""
    acc = None; qid = None; n = 0
    for mx in mixes:
        for f, D in DATA[seed].items():
            R = D['pq']['records']; aseeds = [int(x) for x in D['rec']['args']['assign_seeds']]
            for asd in aseeds:
                r = R[(f, 'test', mx, asd, method)]
                v = r[metric].astype(np.float64) * (100.0 if metric in ('R1', 'R5', 'R10') else 1.0)
                if acc is None: acc = np.zeros_like(v); qref = r['q']; qid = r['qid']
                elif not np.array_equal(r['q'], qref): raise SystemExit(f'query alignment failed at s{seed} f{f} {mx} {asd} {method}')
                acc += v; n += 1
    return acc / n, qid, n


def judge(cli):
    global a, DATA
    a=cli;DATA={};report={'reproduction':{},'paired':{},'ratio':{},'coverage':{}}
    problems=[]
    for seed in a.seeds:
        DATA[seed]={};directories=[];cells=[];covs=[]
        for fit in a.fits:
            directory=Path(a.xcam_root)/f's{seed}_f{fit}';prefix=protocol_prefix(directory);directories.append(str(directory))
            rec=json.loads(Path(prefix+'run_record.json').read_text(encoding='utf-8'))
            if rec.get('script_sha256')!=a.expect_script_sha:problems.append(f's{seed}_f{fit}: unexpected script SHA256')
            actual=pd.read_csv(prefix+'cells.csv');reference=pd.read_csv(protocol_prefix(Path(a.r15_root)/f's{seed}_f{fit}')+'cells.csv')
            actual=actual[actual.evalset=='test'];reference=reference[reference.evalset=='test']
            shared=['B*','QE-DBA','QE+M'];x=actual[actual.method.isin(shared)];y=reference[reference.method.isin(shared)]
            keys=['split','fold','evalset','mixture','aseed','method']
            merged=x.merge(y,on=keys,suffixes=('', '_ref'),validate='one_to_one')
            ok=len(x)==len(y)==len(merged)>0 and set(shared)<=set(x.method) and set(shared)<=set(y.method)
            maximum=0.0
            for col in ('mAP','R1','R5','R10','mINP','dmAP','dR1'):
                if col not in merged or col+'_ref' not in merged:ok=False;continue
                av=merged[col].to_numpy(float);bv=merged[col+'_ref'].to_numpy(float)
                ok &= bool(np.allclose(av,bv,atol=1e-6,rtol=0,equal_nan=True))
                good=np.isfinite(av)&np.isfinite(bv)
                if good.any():maximum=max(maximum,float(np.abs(av[good]-bv[good]).max()))
            report['reproduction'][f's{seed}_f{fit}']={'ok':bool(ok),'matched_cells':len(merged),'max_abs_difference':maximum,'script_sha256':rec.get('script_sha256')}
            if not ok:problems.append(f's{seed}_f{fit}: reproduction differs')
            ext=list(directory.glob('r17_*_perq_ext.pkl'))
            if len(ext)!=1:raise ValueError('Expected one explicit-key per-query file')
            with ext[0].open('rb') as f:pq=pickle.load(f)
            DATA[seed][fit]={'rec':rec,'pq':pq}
            cells.append(actual)
            covs.append(pd.read_csv(prefix+'xcam_cov.csv'))
        if problems:continue
        _,_,_,_,_,_,pq=load_pooled(directories)
        records=list(DATA[seed][a.fits[0]]['pq']['records'].values())
        n=max(int(r['q'].max()) for r in records)+1;ids=np.full(n,-1,dtype=np.int64)
        for r in records:
            old=ids[r['q']]
            if ((old!=-1)&(old!=r['qid'])).any():raise ValueError('Query/raw identity alignment mismatch')
            ids[r['q']]=r['qid']
        available=set(pd.concat(cells).mixture);groups={'A':[m for m in A if m in available],'B':[m for m in B if m in available]}
        if not all(groups.values()):raise ValueError('A and B each require at least one mixture')
        pairs=[('QE+M-xcam','QE-DBA'),('QE+M','QE-DBA'),('QE+M','QE+M-xcam')]
        report['paired'][str(seed)]={}
        for x,y in pairs:
            entry={}
            for group,mixes in groups.items():
                main=group_pair(pq,x,y,mixes,ids,nboot=a.boot)
                extra={}
                for metric in ('R5','R10','mINP'):
                    vx,qid,_=per_query(seed,x,mixes,metric);vy,qid2,_=per_query(seed,y,mixes,metric)
                    if not np.array_equal(qid,qid2):raise ValueError('Extended query identities disagree')
                    mean,lo,hi=boot_ci(vx-vy,qid)
                    extra[metric]={'mean':mean,'lo':lo,'hi':hi}
                entry[group]={'mAP_R1':main,**extra}
            report['paired'][str(seed)][x+' - '+y]=entry
        numerator=report['paired'][str(seed)]['QE+M-xcam - QE-DBA']['A']['mAP_R1']['dmAP']
        denominator=report['paired'][str(seed)]['QE+M - QE-DBA']['A']['mAP_R1']['dmAP']
        report['ratio'][str(seed)]={'numerator':numerator,'denominator':denominator,'ratio':numerator/denominator if denominator>0 else 'n/a'}
        cov=pd.concat(covs,ignore_index=True);coverage_out={}
        for mixture,frame in cov[cov.evalset=='test'].groupby('mixture',sort=True):
            fields=['ir_only_items','same_identity_complete_any_camera','same_identity_complete_other_camera','same_identity_sources_only_same_camera']
            means={k:float(frame[k].mean()) for k in fields};denom=means['ir_only_items']
            coverage_out[mixture]={'mean':means,'ratio':{k:means[k]/denom if denom else None for k in fields}}
        report['coverage'][str(seed)]=coverage_out
    report['status']='INVALID' if problems else 'VALID';report['problems']=problems
    write_json(a.out,report)
    sections=['[status] '+report['status']]
    for key in ('reproduction','paired','ratio','coverage'):
        sections.append('['+key+']\n'+json.dumps(report[key],ensure_ascii=False,indent=2))
    text='\n'.join(sections)+'\n'
    Path(a.txt or str(Path(a.out).with_suffix('.txt'))).write_text(text,encoding='utf-8',newline='\n')
    print(text)
    return 3 if problems else 0

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--xcam_root',required=True);ap.add_argument('--r15_root',required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=[101,102,103]);ap.add_argument('--fits',nargs='+',type=int,default=[7001,7002,7003])
    ap.add_argument('--expect_script_sha',required=True);ap.add_argument('--out',required=True);ap.add_argument('--txt');ap.add_argument('--boot',type=int,default=4000);ap.add_argument('--evalset',default='test')
    args=ap.parse_args()
    try:sys.exit(judge(args))
    except (ValueError,FileNotFoundError,KeyError) as exc:
        write_json(args.out,{'status':'INVALID','problems':[str(exc)]});print('[status] INVALID: '+str(exc));sys.exit(3)


