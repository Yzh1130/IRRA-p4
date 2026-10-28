"""Identity-disjoint slices of an existing 12-key archive; never encode images."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

def subset_archive(source, overlap, output):
    excluded=set(map(int,json.loads(Path(overlap).read_text(encoding='utf-8'))['overlap_test_ids']))
    if not excluded:
        return {'status':'empty exclusion list; no file generated','exit_code':0}
    with np.load(source,allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    q=np.asarray(arrays['clean/txt_pids']);raw=np.asarray(arrays['clean/query_raw_pids']);g=np.asarray(arrays['clean/img_pids'])
    mapping={}
    for label,pid in zip(q,raw):
        if int(label) in mapping and mapping[int(label)]!=int(pid):raise ValueError('Ambiguous label/raw PID mapping')
        mapping[int(label)]=int(pid)
    if not excluded<=set(map(int,raw)):raise ValueError('An overlap identity is absent from archive')
    if set(map(int,g))-mapping.keys():raise ValueError('Gallery identity has no raw PID mapping')
    grow=np.asarray([mapping[int(p)] for p in g]);qm=~np.isin(raw,sorted(excluded));gm=~np.isin(grow,sorted(excluded))
    before=dict(identities=len(set(map(int,raw))),queries=len(q),items=len(g))
    if not qm.any() or not gm.any():return {'status':'no identities remain; no file generated','before':before,'exit_code':2}
    if set(q[qm])!=set(g[gm]):raise ValueError('A remaining identity lacks a query or item')
    query_keys={'clean/text_features','clean/txt_pids','clean/query_raw_pids'}
    result={}
    for key,value in arrays.items():
        mask=qm if key in query_keys else gm
        if value.shape[0]!=len(mask):raise ValueError(f'Unknown row semantics: {key} {value.shape}')
        result[key]=value[mask]
        if result[key].dtype!=value.dtype:raise AssertionError('dtype changed')
    if set(result)!=set(arrays):raise AssertionError('keys changed')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:np.savez_compressed(f,**result)
    after=dict(identities=len(set(map(int,raw[qm]))),queries=int(qm.sum()),items=int(gm.sum()))
    return {'status':'少于 10 个身份，只作描述' if after['identities']<10 else 'written','before':before,'after':after,'exit_code':0}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',required=True);ap.add_argument('--overlap',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();r=subset_archive(a.input,a.overlap,a.output);print(json.dumps(r,ensure_ascii=False));sys.exit(r['exit_code'])
