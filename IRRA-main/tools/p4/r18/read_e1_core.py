"""Example: read E1 dmAP and lower bounds for QE+M versus QE-DBA on A.

python -B tools/p4/r18/read_e1_core.py --input R18J_core.txt
Use --seeds 101 for a one-seed local rehearsal; defaults require 101 102 103.
"""
from pathlib import Path
import argparse,json

def read_e1(path,seeds=(101,102,103),scope='full'):
    rows=[json.loads(line[5:]) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.startswith('R18J ')]
    result=[]
    for seed in seeds:
        matches=[r for r in rows if r['src']=='judge' and r['set']==scope and r['seed']==seed and r['key']=='pair|QE+M|QE-DBA|A']
        if len(matches)!=1:raise ValueError(f'Expected one E1 comparison for seed {seed}, found {len(matches)}')
        v=matches[0]['v'];result.append({'seed':seed,'dmAP':v['dmAP'],'lo':v['lo']})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--seeds',type=int,nargs='+',default=[101,102,103]);p.add_argument('--set',default='full',choices=['full','subset']);a=p.parse_args()
    print(json.dumps(read_e1(a.input,a.seeds,a.set),ensure_ascii=False,indent=2))
