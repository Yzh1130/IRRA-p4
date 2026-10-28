"""Unrounded R18J records, group absolute metrics, and HALF_UP seed summaries."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import argparse
from collections import defaultdict,Counter
import json
import math
import re
import sys
import numpy as np
import pandas as pd
CODE=Path(__file__).resolve().parents[3];sys.path.insert(0,str(CODE))
from tools.p4.r18.common import protocol_prefix
A=['M4','M6c','M1g','M4c60','M9'];B=['M5e','M5f','M4id'];POOL=['KR','RVN','QE-DBA','KRir','RVNir','QEir','GP-same']

def gm(cells,method,mixes,key):
    return float(np.mean([cells[(cells.method==method)&(cells.mixture==mx)][key].mean() for mx in mixes]))

def m0m(cells,method,key):
    rows=cells[(cells.method==method)&(cells.mixture=='M0')]
    if rows.empty:rows=cells[(cells.method=='B*')&(cells.mixture=='M0')]
    return float(rows[key].mean())

def emit(records,src,scope,seed,key,value):
    records.append(dict(src=src,set=scope,seed=str(seed),key=key,v=value if isinstance(value,dict) else {'value':value}))

def flatten_json(records,src,scope,node,seeds,path=(),inherited=None):
    if path:
        match=re.fullmatch(r's?(\d+)(?:_f\d+)?(?::.*)?',str(path[-1]))
        if match and int(match.group(1)) in seeds:inherited=int(match.group(1))
    if isinstance(node,dict):
        for key,value in sorted(node.items()):flatten_json(records,src,scope,value,seeds,path+(str(key),),inherited)
    else:
        for seed in ([inherited] if inherited is not None else seeds):emit(records,src,scope,seed,'.'.join(path),node)

def runner_records(records,roots):
    seen=set()
    for spec in roots:
        scope,directory=spec.split('=',1);root=Path(directory)
        if not root.is_dir():raise FileNotFoundError(root)
        for log in sorted(root.rglob('degenerate_fits.json')):
            if log.resolve() in seen:continue
            seen.add(log.resolve());data=json.loads(log.read_text(encoding='utf-8'))
            match=re.search(r'(?:^|/)s(\d+)(?:_f\d+)?(?:/|$)',log.parent.as_posix())
            seed=match.group(1) if match else 'unknown'
            counts=Counter(event['head'] for event in data['events'])
            emit(records,'runner',scope,seed,root.name+'/'+log.parent.relative_to(root).as_posix(),
                 {'events':len(data['events']),'heads':dict(sorted(counts.items())),
                  'script':data['script'],'script_sha256':data['script_sha256']})

def build(cli):
    records=[]
    runner_records(records,getattr(cli,'runner_roots',[]))
    for src,items in [('judge',cli.judge),('r17',cli.r17)]:
        for spec in items:
            scope,path=spec.split('=',1);flatten_json(records,src,scope,json.loads(Path(path).read_text(encoding='utf-8')),cli.seeds)
    if cli.xcam:flatten_json(records,'xcam','full',json.loads(Path(cli.xcam).read_text(encoding='utf-8')),cli.seeds)
    for spec in cli.cells:
        scope,root=spec.split('=',1)
        for seed in cli.seeds:
            dirs=sorted(Path(root).glob(f's{seed}_f*'))
            if not dirs:raise ValueError(f'No cells for seed {seed}: {root}')
            c=pd.concat([pd.read_csv(protocol_prefix(d)+'cells.csv') for d in dirs],ignore_index=True);c=c[c.evalset=='test']
            dg=pd.concat([pd.read_csv(protocol_prefix(d)+'diag_cells.csv') for d in dirs],ignore_index=True);dg=dg[dg.evalset=='test']
            groups={'A':[m for m in A if (c.mixture==m).any()],'B':[m for m in B if (c.mixture==m).any()]}
            for method in sorted(set(c.method)):
                for group,mixes in groups.items():
                    metrics=['mAP','R1','R5','R10','mINP'] if group=='A' else ['mAP','R1','mINP']
                    emit(records,'cells',scope,seed,f'absolute.{group}.{method}',{k:gm(c,method,mixes,k) for k in metrics if k in c})
                emit(records,'cells',scope,seed,'absolute.M0.'+method,{k:m0m(c,method,k) for k in ('mAP','R1')})
            for group,mixes in groups.items():
                pool={m:gm(c,m,mixes,'dmAP') for m in POOL if m in set(c.method)};pool={k:v for k,v in pool.items() if math.isfinite(v)}
                strongest=max(pool,key=pool.get) if pool else None
                emit(records,'cells',scope,seed,'strongest_generic.'+group,{'method':strongest,'dmAP':pool.get(strongest)})
            for (mixture,method),frame in c.groupby(['mixture','method']):emit(records,'cells',scope,seed,'mixture.'+mixture+'.'+method,{'dmAP':float(frame.dmAP.mean())})
            for (mixture,method),frame in dg.groupby(['mixture','method']):emit(records,'diag_cells',scope,seed,'mixture.'+mixture+'.'+method,{k:float(frame[k].mean()) for k in ('frac_src','leak','coverage') if k in frame})
    grouped=defaultdict(dict)
    for row in records:
        if row['src']=='runner':continue  # exactly one raw row per execution unit
        for field,value in row['v'].items():
            if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):grouped[(row['src'],row['set'],row['key'],field)][row['seed']]=float(value)
    def rounded(value):return str(Decimal(str(value)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))
    for (src,scope,key,field),values in sorted(grouped.items()):
        numbers=[values[str(s)] for s in cli.seeds if str(s) in values]
        mean=float(np.mean(numbers));sd=float(np.std(numbers,ddof=1)) if len(numbers)>1 else None
        emit(records,src,scope,'mean',key+'.'+field,dict(mean=mean,std_ddof1=sd,n=len(numbers),mean_2dp_HALF_UP=rounded(mean),std_2dp_HALF_UP=rounded(sd) if sd is not None else None))
    text='\n'.join('R18J '+json.dumps(r,sort_keys=True,ensure_ascii=False,separators=(',',':')) for r in records)+'\n'
    Path(cli.out).parent.mkdir(parents=True,exist_ok=True);Path(cli.out).write_text(text,encoding='utf-8',newline='\n');print(text,end='');return records

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--judge',nargs='*',default=[]);ap.add_argument('--r17',nargs='*',default=[]);ap.add_argument('--xcam');ap.add_argument('--cells',nargs='+',required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[101,102,103]);ap.add_argument('--out',required=True);ap.add_argument('--runner_roots',nargs='*',default=[])
    build(ap.parse_args())
