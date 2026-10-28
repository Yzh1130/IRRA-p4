"""Compact, source-traceable R18J comparisons; never read feature archives."""
from collections import defaultdict
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
import json,math,re
import numpy as np
import pandas as pd
from tools.p4.r18.common import protocol_prefix

ABS_METHODS=('B*','QE-DBA','QEir','KR','RVN','KRir','RVNir','GP-same','CAET',
             'QE+CAET','QE+SRC2f','QE+EM','FeatImp-g','A-2f-ctxn','QE+M','QE+Orc','Oracle-A')
TABLE3=('QE-DBA','QEir','KRir','GP-same','FeatImp-g','CAET','A-2f-ctxn','QE+CAET','QE+EM','QE+M','Oracle-A')
POOL=('KR','RVN','QE-DBA','KRir','RVNir','QEir','GP-same')
A=['M4','M6c','M1g','M4c60','M9'];B=['M5e','M5f','M4id']

def numeric_fields(node,prefix=''):
    if isinstance(node,dict):
        for k,v in node.items():yield from numeric_fields(v,prefix+'.'+str(k) if prefix else str(k))
    elif isinstance(node,(list,tuple)):
        for i,v in enumerate(node):yield from numeric_fields(v,prefix+'.'+str(i))
    elif isinstance(node,(int,float)) and not isinstance(node,bool) and math.isfinite(node):yield prefix,float(node)

def point_fields(row):
    if row['src']=='status' or row['src']=='r17' and row['key']=='locality':return
    for field,value in numeric_fields(row['v']):
        leaf=field.rsplit('.',1)[-1]
        if leaf in ('lo','hi','r_lo','r_hi','n_q','n_id','n'):continue
        if row['src']=='r17' and row['key'].startswith('pair|') and leaf!='0':continue
        if row['src']=='xcam' and row['key'].startswith('coverage|') and field.startswith('mean.'):continue
        yield field,value

def means(rows,seeds):
    grouped=defaultdict(lambda:defaultdict(dict))
    for row in rows:
        if row['seed'] not in seeds:continue
        for field,value in point_fields(row):grouped[(row['src'],row['set'],row['key'])][field][row['seed']]=value
    out=[]
    def two(x):return str(Decimal(str(x)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))
    for (src,scope,key),fields in sorted(grouped.items()):
        stats={}
        for field,values in sorted(fields.items()):
            numbers=[values[s] for s in seeds if s in values];mu=float(np.mean(numbers));sd=float(np.std(numbers,ddof=1)) if len(numbers)>1 else None
            stats[field]={'n':len(numbers),'mean':mu,'sd':sd,'mean_2dp':two(mu),'sd_2dp':two(sd) if sd is not None else None}
        out.append({'src':src,'set':scope,'seed':'mean','key':key,'v':stats})
    return out

def build_core(cli,full_rows):
    from tools.p4.r18.r18_exact import gm,m0m
    rows=[];seeds=[int(s) for s in cli.seeds]
    def emit(src,scope,seed,key,value):rows.append({'src':src,'set':scope,'seed':int(seed),'key':key,'v':value})
    load=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
    exits={scope:int(value) for scope,value in (v.split('=',1) for v in getattr(cli,'judge_exit_status',[]))}
    selections={scope:load(path) for scope,path in (v.split('=',1) for v in getattr(cli,'selection_equal',[]))}
    scopes=[v.split('=',1)[0] for v in cli.cells]
    for scope in scopes:
        selection=selections.get(scope)
        for seed in seeds:
            emit('status',scope,seed,'selection_equal',{'SELECTION_EQUAL':selection.get('SELECTION_EQUAL') if selection else None,
                 'differences':len(selection.get('differences',{})) if selection else None})
    for spec in cli.judge:
        scope,path=spec.split('=',1);data=load(path)
        for seed in seeds:
            emit('status',scope,seed,'judge',{'evalset':data['evalset'],'exit_status':exits.get(scope)})
            per=data['per_seed'].get('s'+str(seed),data['per_seed'].get(str(seed)))
            if per is None:raise ValueError('Missing judge seed '+str(seed))
            for name,value in sorted(per['pairs'].items()):
                pair,group=name.rsplit(' | ',1);left,right=pair.split(' - ',1)
                emit('judge',scope,seed,'pair|'+left+'|'+right+'|'+group,{k:value[k] for k in ('dmAP','lo','hi','dR1','r_lo','r_hi','n_q','n_id')} if value is not None else {'missing':True})
            emit('judge',scope,seed,'claims',per['claims'])
    for spec in cli.r17:
        scope,path=spec.split('=',1);data=load(path)
        for seed in seeds:
            emit('status',scope,seed,'r17',{k:data.get(k) for k in ('valid','reproduction_ok','problems')})
            grouped=defaultdict(dict)
            for name,value in data.get('paired',{}).items():
                s,pair,group,metric=name.split(':',3)
                if s=='s'+str(seed):grouped[(pair,group)][metric]=value[:3]
            for (pair,group),value in sorted(grouped.items()):emit('r17',scope,seed,'pair|'+pair+'|'+group,value)
            for name,value in sorted(data.get('selection',{}).items()):
                match=re.fullmatch(r's(\d+)_f(\d+)',name)
                if match and int(match.group(1))==seed:emit('r17',scope,seed,'gamma|f'+match.group(2),{'gamma':value['chosen']})
            locality={k.split(':',1)[1]:v for k,v in data.get('locality',{}).items() if k.startswith('s'+str(seed)+':')}
            curve={k.split('@',1)[1]:v for k,v in data.get('curve',{}).items() if k.startswith('s'+str(seed)+':QE+Const@')}
            emit('r17',scope,seed,'locality',locality);emit('r17',scope,seed,'curve',curve)
    if cli.xcam:
        data=load(cli.xcam)
        for seed in seeds:
            emit('status','full',seed,'xcam',{k:data.get(k) for k in ('status','problems')})
            for name,groups in sorted(data['paired'][str(seed)].items()):
                left,right=name.split(' - ',1)
                for group,value in sorted(groups.items()):
                    v={k:value['mAP_R1'][k] for k in ('dmAP','lo','hi','dR1','r_lo','r_hi')}
                    v.update({k:value[k] for k in ('R5','R10','mINP')})
                    emit('xcam','full',seed,'pair|'+left+'|'+right+'|'+group,v)
            emit('xcam','full',seed,'ratio',data['ratio'][str(seed)])
            for mix,value in sorted(data['coverage'][str(seed)].items()):emit('xcam','full',seed,'coverage|'+mix,value)
    for spec in cli.cells:
        scope,root=spec.split('=',1)
        for seed in seeds:
            dirs=sorted(Path(root).glob(f's{seed}_f*'))
            cells=pd.concat([pd.read_csv(protocol_prefix(p)+'cells.csv') for p in dirs]);cells=cells[cells.evalset=='test']
            diag=pd.concat([pd.read_csv(protocol_prefix(p)+'diag_cells.csv') for p in dirs]);diag=diag[diag.evalset=='test']
            have=set(cells.method);mixes=sorted(set(cells.mixture))
            groups={'A':[m for m in A if m in mixes],'B':[m for m in B if m in mixes]}
            for group,items in groups.items():
                pool={m:gm(cells,m,items,'dmAP') for m in POOL if m in have};pool={k:v for k,v in pool.items() if math.isfinite(v)}
                best=max(pool,key=pool.get) if pool else None
                emit('cells',scope,seed,'strongest|'+group,{'method':best,'dmAP':pool.get(best)})
            for method in ABS_METHODS:
                if method not in have:continue
                for group,items in groups.items():
                    keys=('mAP','R1','R5','R10','mINP') if group=='A' else ('mAP','R1','mINP')
                    emit('cells',scope,seed,'abs|'+method+'|'+group,{k:gm(cells,method,items,k) for k in keys if k in cells})
                emit('cells',scope,seed,'abs|'+method+'|M0',{k:m0m(cells,method,k) for k in ('mAP','R1')})
            for method in TABLE3:
                if method in have:emit('cells',scope,seed,'mixture_dmAP|'+method,{mx:float(cells[(cells.method==method)&(cells.mixture==mx)].dmAP.mean()) for mx in mixes if ((cells.method==method)&(cells.mixture==mx)).any()})
            emit('cells',scope,seed,'mixture_mAP|B*',{mx:float(cells[(cells.method=='B*')&(cells.mixture==mx)].mAP.mean()) for mx in mixes})
            for method in ('QE+M','A-2f-ctxn','CAET','QE+CAET'):
                frame=diag[diag.method==method]
                if not frame.empty:emit('diag',scope,seed,'mixture_diag|'+method,{mx:{k:float(v[k].mean()) for k in ('frac_src','leak','coverage')} for mx,v in frame.groupby('mixture')})
    for row in full_rows:
        if row['src']=='runner':emit('status',row['set'],int(row['seed']),'runner|'+row['key'],{'events':row['v']['events'],'heads':row['v']['heads']})
    rows.extend(means(rows,seeds))
    return rows

def write_core(cli,full_rows):
    rows=build_core(cli,full_rows)
    text='\n'.join('R18J '+json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':')) for row in rows)+'\n'
    target=Path(cli.core_out);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text,encoding='utf-8',newline='\n')
    print('[R18J_CORE]',target,'lines='+str(len(rows)),'bytes='+str(len(text.encode('utf-8'))))
    return rows
