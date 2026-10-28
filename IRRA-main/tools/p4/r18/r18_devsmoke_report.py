"""Development-only flow checks; random R1 is calculated for each actual fold gallery."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
CODE=Path(__file__).resolve().parents[3];sys.path.insert(0,str(CODE))
from tools.p4.r18.common import protocol_prefix,write_json
A=['M4','M6c','M1g','M4c60','M9'];B=['M5e','M5f','M4id']

def report(run,archive,output,allow_below_random_threshold=False):
    prefix=protocol_prefix(run);cells=pd.read_csv(prefix+'cells.csv');record=json.loads(Path(prefix+'run_record.json').read_text(encoding='utf-8'))
    if record['args'].get('test_npz'):raise ValueError('Development smoke refuses a held-out archive')
    with np.load(archive,allow_pickle=False) as z:
        qraw=z['clean/query_raw_pids'];qp=z['clean/txt_pids'];gp=z['clean/img_pids']
    mapping=dict(zip(qp.tolist(),qraw.tolist()));grow=np.asarray([mapping[int(p)] for p in gp]);unique=np.unique(qraw)
    random_rows=[]
    for split,fold in sorted(set(zip(cells['split'],cells['fold']))):
        permutation=np.random.default_rng(int(split)).permutation(unique);assignment={int(p):i%int(record['args']['folds']) for i,p in enumerate(permutation)}
        query=np.asarray([assignment[int(p)]!=int(fold) for p in qraw])
        gallery=np.ones(len(grow),bool) if record['args'].get('eval_gallery')=='full' else np.asarray([assignment[int(p)]!=int(fold) for p in grow])
        expectation=float(np.mean([(grow[gallery]==p).sum()/gallery.sum() for p in qraw[query]])*100)
        random_rows.append(dict(split=int(split),fold=int(fold),queries=int(query.sum()),items=int(gallery.sum()),random_R1_percent=expectation))
    result={'groups':{},'random_R1_per_gallery':random_rows,'process_wall_seconds':record.get('wall_seconds')}
    for name,mixes in [('A',A),('B',B),('M0',['M0'])]:
        actual=[m for m in mixes if (cells.mixture==m).any()]
        result['groups'][name]={}
        for method in ('B*','QE-DBA','QE+M'):
            vals={metric:float(np.mean([cells[(cells.method==method)&(cells.mixture==m)][metric].mean() for m in actual])) for metric in ('mAP','R1')}
            result['groups'][name][method]=vals
        for metric in ('mAP','R1'):
            result['groups'][name]['QE+M_minus_QE-DBA_'+metric]=result['groups'][name]['QE+M'][metric]-result['groups'][name]['QE-DBA'][metric]
            result['groups'][name]['QE+M_minus_B*_'+metric]=result['groups'][name]['QE+M'][metric]-result['groups'][name]['B*'][metric]
    m0=cells[cells.mixture=='M0'];left=m0[m0.method=='QE+M'];right=m0[m0.method=='QE-DBA'];keys=['split','fold','evalset','mixture','aseed']
    joined=left.merge(right,on=keys,suffixes=('_m','_q'),validate='one_to_one')
    equal=len(joined)==len(left)==len(right)>0 and all(np.array_equal(joined[k+'_m'].values,joined[k+'_q'].values) for k in ('mAP','R1'))
    random_mean=float(np.mean([r['random_R1_percent'] for r in random_rows]));baseline=result['groups']['M0']['B*']['R1']
    result['mechanical_checks']={'M0_QE_M_equals_front':bool(equal),'random_R1_mean_percent':random_mean,'M0_Bstar_R1_percent':baseline,'Bstar_at_least_3x_random':bool(baseline>=3*random_mean)}
    result['mechanical_checks']['random_threshold_required']=not allow_below_random_threshold
    result['status']='VALID' if equal and (baseline>=3*random_mean or allow_below_random_threshold) else 'INVALID'
    write_json(output,result);print(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True);ap.add_argument('--archive',required=True);ap.add_argument('--out',required=True);ap.add_argument('--allow_below_random_threshold',action='store_true',help='Local smoke checkpoint only: report the 3x-random check without requiring it')
    a=ap.parse_args();r=report(a.run,a.archive,a.out,a.allow_below_random_threshold);sys.exit(0 if r['status']=='VALID' else 3)
