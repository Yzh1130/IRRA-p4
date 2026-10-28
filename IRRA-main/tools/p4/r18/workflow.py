"""Fail-closed R18 manifests, configuration health, packing and camera metadata."""
from pathlib import Path
import argparse
import ast
import fnmatch
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
import numpy as np
CODE=Path(__file__).resolve().parents[3];sys.path.insert(0,str(CODE))
from tools.p4.r18.common import read_json,write_json,sha256
ALLOWED_CONFIG={'name','output_dir','train_seed','deg_rng_seed','dataset_name','root_dir','dev_split_file'}
KEYS12=['clean/text_features','clean/txt_pids','clean/img_pids','clean/query_raw_pids','clean/f_rgb','clean/f_ir',
        'rgb_missing/f_rgb','ir_missing/f_ir','rgb_noise_010/f_rgb','ir_noise_010/f_ir','rgb_blur_k7/f_rgb','ir_blur_k7/f_ir']

def files_of(paths,exclude=()):
    result={}
    for pattern in paths:
        p=Path(pattern)
        if not p.exists():raise FileNotFoundError(p)
        files=[p] if p.is_file() else sorted(q for q in p.rglob('*') if q.is_file() and '.ipynb_checkpoints' not in q.parts)
        for q in files:
            if q.name in exclude:continue
            result[str(q.resolve())]=sha256(q)
    return result

REPORT_FILES={'tools/p4/r18/p4_r18.sh','tools/p4/r18/driver_support.py',
              'tools/p4/r18/r18_exact.py','tools/p4/r18/r18_devsmoke_report.py',
              'tools/p4/transfer/r18_xcam_judge.py','tools/p4/transfer/judge_test.py',
              'tools/p4/transfer/r17_judge.py'}
EVAL_FILES=('tools/p4/transfer/r15_test.py','tools/p4/transfer/r17_const.py',
            'tools/p4/transfer/r17_check_archives.py','tools/p4/transfer/r18_xcam.py',
            'tools/p4/r18/xcam_core.py','tools/p4/r18/run_frozen.py',
            'tools/p4/r18/r18_subset.py','tools/p4/r18/common.py')

def signature_paths(kind='l1'):
    if kind=='eval':return [CODE/p for p in EVAL_FILES]
    if kind!='l1':raise ValueError('Unknown code signature kind: '+kind)
    paths={CODE/'train.py',CODE/'tools/p4/pack_feats.py',CODE/'tools/p4/r18/common.py',CODE/'tools/p4/r18/workflow.py'}
    for directory in ('datasets','model','utils','processor'):
        paths.update((CODE/directory).rglob('*.py'))
    paths.update((CODE/'tools/p4').glob('export_*.py'))
    # Resolve the repository-local import closure, including imports inside
    # optional training/export branches, without importing or running them.
    todo=list(paths);seen=set()
    def candidates(parts):
        base=CODE.joinpath(*parts)
        return [base.with_suffix('.py'),base/'__init__.py']
    while todo:
        path=todo.pop()
        if path in seen or not path.is_file():continue
        seen.add(path)
        if path.relative_to(CODE).as_posix() in REPORT_FILES:continue
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names=[]
            if isinstance(node,ast.Import):names=[alias.name.split('.') for alias in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    package=list(path.relative_to(CODE).parts[:-1]);prefix=package[:len(package)-node.level+1]
                else:prefix=[]
                module=prefix+(node.module.split('.') if node.module else [])
                names=[module]+[module+alias.name.split('.') for alias in node.names if alias.name!='*']
            for parts in names:
                for imported in candidates(parts):
                    if imported.is_file() and imported not in paths:
                        paths.add(imported);todo.append(imported)
    return sorted(p for p in paths if p.is_file() and p.relative_to(CODE).as_posix() not in REPORT_FILES)

def code_signature(kind='l1'):
    return files_of(signature_paths(kind))

def guard(paths,manifest):
    current=files_of(paths)
    if Path(manifest).is_file():
        if read_json(manifest)!=current:raise ValueError('Protected source tree changed')
    else:write_json(manifest,current)
    print('[R18_GUARD] unchanged',len(current))

def stage(cli):
    manifest=Path(cli.manifest);done=Path(cli.done)
    command=cli.command[1:] if cli.command and cli.command[0]=='--' else cli.command
    kind=getattr(cli,'signature_kind','l1')
    signature={'kind':kind,'command':command,'inputs':files_of(cli.inputs),'code':code_signature(kind)}
    if manifest.is_file() and done.is_file():
        record=read_json(manifest)
        if record['signature']!=signature:raise ValueError('Completed stage manifest/input/code differs; refusing reuse')
        if record['outputs']!=files_of(cli.outputs,exclude={manifest.name,done.name}):raise ValueError('Completed stage outputs changed')
        if not cli.rerun:
            print('[R18_REUSE]',cli.manifest);return
    elif cli.verify_only:
        raise ValueError('L2R requires every score unit to have .done and a matching manifest')
    elif done.exists() or manifest.exists():
        raise ValueError('Incomplete stage markers; refusing to overwrite')
    if cli.verify_only:return
    for pattern in cli.outputs:
        p=Path(pattern)
        if p.exists() and not cli.rerun:
            if p.is_file() or any(p.iterdir()):raise FileExistsError(f'Incomplete stage has existing output: {p}')
    done.parent.mkdir(parents=True,exist_ok=True)
    for pattern in cli.outputs:
        p=Path(pattern)
        p.parent.mkdir(parents=True,exist_ok=True)
        if cli.mkdir:p.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();p=subprocess.run(command)
    if p.returncode:raise SystemExit(p.returncode)
    outputs=files_of(cli.outputs,exclude={manifest.name,done.name})
    write_json(manifest,{'signature':signature,'outputs':outputs,'elapsed_seconds':time.perf_counter()-start})
    done.write_text('COMPLETE\n',encoding='utf-8',newline='\n')

def verify_manifest(path):
    record=read_json(path);done=Path(path).with_suffix('.done')
    if not done.is_file():raise ValueError('Completed unit has no .done: '+str(path))
    sig=record['signature']
    if sig['code']!=code_signature(sig.get('kind','l1')) or sig['inputs']!=files_of(sig['inputs']):raise ValueError('Completed unit input/code changed')
    if record['outputs']!=files_of(record['outputs']):raise ValueError('Completed unit output changed')
    print('[R18_VERIFIED]',path)

def config_paths(config):
    from utils.config import load_config_mapping
    cfg=load_config_mapping(config);allowed={'root_dir','dev_split_file','output_dir','config_file'};rows=[];bad=[]
    controls={'synth_caption_bank_file':'use_synth_caption_bank','mga_pseudo_json':'use_mga_pseudo','offline_hardneg_index_file':'use_offline_hardneg','proto_init_file':None,'load_ckpt_file':None}
    for key,value in cfg.items():
        if key.endswith(('_file','_path','_json','_dir')) or isinstance(value,str) and value and Path(value).exists():
            flag=controls.get(key);active=bool(cfg.get(flag)) if flag else bool(value)
            row={'key':key,'value':value,'empty':not bool(value),'control':flag,'control_enabled':bool(cfg.get(flag)) if flag else None,'would_be_active':active}
            rows.append(row)
            if value and key not in allowed and (key.endswith(('_file','_path','_json')) or any(s in key for s in ('caption','pseudo','hardneg','anno','split'))):bad.append(row)
    return {'config':str(Path(config).resolve()),'sha256':sha256(config),'paths':rows,'forbidden_dataset_file_parameters':bad}

def config_compare(first,second):
    from utils.config import load_config_mapping
    a,b=load_config_mapping(first),load_config_mapping(second)
    differences={k:{'first':a.get(k),'second':b.get(k),'missing_first':k not in a,'missing_second':k not in b} for k in sorted(a.keys()|b.keys()) if k not in a or k not in b or a[k]!=b[k]}
    forbidden={k:v for k,v in differences.items() if k not in ALLOWED_CONFIG}
    return {'allowed_keys':sorted(ALLOWED_CONFIG),'differences':differences,'forbidden_differences':forbidden,'ok':not forbidden}

def health(run,base_config):
    run=Path(run);cfg=run/'configs.yaml'
    comparison=config_compare(base_config,cfg)
    if not comparison['ok']:raise ValueError('Training recipe changed outside the allowed keys: '+str(comparison['forbidden_differences']))
    record=read_json(run/'run_record.json')
    from utils.config import load_config_mapping
    config=load_config_mapping(cfg)
    if record.get('status')!='completed':raise ValueError('Training record is not completed')
    steps=record.get('steps',[]);epochs=record.get('epochs',[])
    if not steps or not all(s.get('finite',False) and all(math.isfinite(float(s[k])) for k in ('L_clean','L_deg','L_total') if k in s) for s in steps):raise ValueError('Missing or nonfinite iteration records')
    seen={int(e['epoch']) for e in epochs}
    if seen!=set(range(1,int(config['num_epoch'])+1)):raise ValueError('Iteration/epoch records do not cover every epoch')
    if {int(s['epoch']) for s in steps}!=seen:raise ValueError('Iteration records do not cover every epoch')
    if not (run/'last.pth').is_file():raise FileNotFoundError(run/'last.pth')
    log=(run/'train_log.txt').read_text(encoding='utf-8')
    validations=list(re.finditer(r'Validation Results - Epoch:\s*(\d+)',log))
    if not validations or int(validations[-1].group(1))!=int(config['num_epoch']):raise ValueError('Final development evaluation missing')
    table=re.search(r'\|\s*t2i\s*\|\s*([^\n]+)',log[validations[-1].end():])
    if table is None:raise ValueError('Final development metric table missing')
    metrics=[float(x.strip()) for x in table.group(1).split('|') if x.strip()]
    if len(metrics)!=5 or not all(math.isfinite(v) for v in metrics):raise ValueError('Invalid final development metrics')
    return {'comparison':comparison,'steps':len(steps),'epochs':len(seen),'last_sha256':sha256(run/'last.pth'),'last_epoch':epochs[-1],
            'last_dev_metrics':dict(zip(('R1','R5','R10','mAP','mINP'),metrics)),
            'elapsed_seconds':(datetime.fromisoformat(record['ended_at'])-datetime.fromisoformat(record['started_at'])).total_seconds(),
            'iteration_seconds':sum(float(s.get('seconds',0)) for s in steps),
            'peak_memory_bytes':max(int(s.get('peak_memory_bytes',0)) for s in steps)}

def find_run(parent,name,expected_sha=None):
    matches=[]
    for p in Path(parent).glob('*_'+name):
        if (p/'last.pth').is_file() and (not expected_sha or sha256(p/'last.pth')==expected_sha):matches.append(p)
    if len(matches)!=1:raise ValueError(f'Expected one matching run: {parent}/{name}, got {len(matches)}')
    return matches[0]

def pack12(source,output):
    with np.load(source,allow_pickle=False) as z:
        missing=[key for key in KEYS12 if key not in z]
        if missing:raise ValueError('Missing mandatory KEYS12 entries: '+', '.join(missing))
        arrays={key:z[key] for key in KEYS12}
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:np.savez_compressed(f,**arrays)
    return {'keys':list(arrays),'sha256':sha256(output)}

def compare_npz(first,second,tolerance=1e-5):
    maximum=0.;details={}
    with np.load(first,allow_pickle=False) as a,np.load(second,allow_pickle=False) as b:
        if set(a.files)!=set(b.files):raise ValueError('Archive keys differ')
        for key in a.files:
            if a[key].shape!=b[key].shape or a[key].dtype!=b[key].dtype:raise ValueError(f'Archive shape/dtype mismatch: {key}')
            if a[key].dtype.kind in 'fiu':
                d=float(np.max(np.abs(a[key].astype(np.float64)-b[key].astype(np.float64))))
                if not math.isfinite(d):raise ValueError('Nonfinite archive comparison')
            else:d=0. if np.array_equal(a[key],b[key]) else float('inf')
            maximum=max(maximum,d);details[key]=d
    if maximum>tolerance:raise ValueError(f'Archive max difference {maximum} exceeds {tolerance}')
    return {'max_abs_difference':maximum,'per_key':details}

def camera_file(dev_meta,test_meta,dev_archive,test_archive,output,dev_full=None,test_full=None):
    result={}
    for split,meta_path,archive in [('dev',dev_meta,dev_archive),('test',test_meta,test_archive)]:
        meta=read_json(meta_path)
        with np.load(archive,allow_pickle=False) as z:
            if len(meta['gallery_filenames'])!=len(z['clean/img_pids']):raise ValueError('Camera gallery length mismatch')
            mapping=dict(zip(z['clean/txt_pids'].tolist(),z['clean/query_raw_pids'].tolist()))
            if [mapping[int(p)] for p in z['clean/img_pids']]!=meta['gallery_raw_pids']:raise ValueError('Camera raw-PID order mismatch')
            if meta['query_raw_pids']!=z['clean/query_raw_pids'].tolist():raise ValueError('Camera query raw-PID order mismatch')
        if any(c is None for c in meta['gallery_cameras']):raise ValueError('RGBNT201 camera labels cannot be null')
        full=dev_full if split=='dev' else test_full
        if full is None:raise ValueError('Full packed archive is required to verify within-PID filename order')
        with np.load(full,allow_pickle=False) as z:
            names=[Path(str(v)).name for v in z['clean/gallery_file_paths']]
            if names!=meta['gallery_filenames']:raise ValueError('Camera gallery filename order mismatch')
            if z['clean/gallery_raw_pids'].tolist()!=meta['gallery_raw_pids']:raise ValueError('Full gallery raw PID order mismatch')
            if z['clean/query_raw_pids'].tolist()!=meta['query_raw_pids']:raise ValueError('Full query raw PID order mismatch')
        result[split+'_cams']=np.asarray(meta['gallery_cameras'],dtype=str)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    with output.open('xb') as f:np.savez_compressed(f,**result)

def selection_equal(first,second,output):
    def normalized(root):
        return {str(p.relative_to(root)):{r['method']:(r['chosen'],r['feasible']) for r in read_json(p)} for p in Path(root).rglob('*_selection.json')}
    left,right=normalized(Path(first)),normalized(Path(second))
    # Tags/prefixes differ, match by s<seed>_f<fit> parent and method.
    def by_unit(items):return {str(Path(k).parent):v for k,v in items.items()}
    a,b=by_unit(left),by_unit(right);result={'SELECTION_EQUAL':a==b,'differences':{k:{'full':a.get(k),'subset':b.get(k)} for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}}
    write_json(output,result);return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='operation',required=True)
    p=sub.add_parser('guard');p.add_argument('--manifest',required=True);p.add_argument('paths',nargs='+')
    p=sub.add_parser('stage');p.add_argument('--manifest',required=True);p.add_argument('--done',required=True);p.add_argument('--inputs',nargs='*',default=[]);p.add_argument('--outputs',nargs='+',required=True);p.add_argument('--rerun',action='store_true');p.add_argument('--verify_only',action='store_true');p.add_argument('--mkdir',action='store_true');p.add_argument('command',nargs=argparse.REMAINDER)
    p.add_argument('--signature_kind',choices=('l1','eval'),default='l1')
    p=sub.add_parser('verify-manifest');p.add_argument('--manifest',required=True)
    p=sub.add_parser('find-run');p.add_argument('--parent',required=True);p.add_argument('--name',required=True);p.add_argument('--sha')
    p=sub.add_parser('config-paths');p.add_argument('--config',required=True);p.add_argument('--out')
    p=sub.add_parser('config-compare');p.add_argument('--first',required=True);p.add_argument('--second',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('health');p.add_argument('--run',required=True);p.add_argument('--base',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('pack12');p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('compare-npz');p.add_argument('--first',required=True);p.add_argument('--second',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('cams');p.add_argument('--dev_meta',required=True);p.add_argument('--test_meta',required=True);p.add_argument('--dev_archive',required=True);p.add_argument('--test_archive',required=True);p.add_argument('--output',required=True);p.add_argument('--dev_full',required=True);p.add_argument('--test_full',required=True)
    p=sub.add_parser('manifest');p.add_argument('--out',required=True);p.add_argument('--verify',action='store_true');p.add_argument('paths',nargs='*')
    p=sub.add_parser('selection');p.add_argument('--full',required=True);p.add_argument('--subset',required=True);p.add_argument('--out',required=True)
    a=ap.parse_args();result=None
    if a.operation=='guard':guard(a.paths,a.manifest)
    elif a.operation=='stage':stage(a)
    elif a.operation=='verify-manifest':verify_manifest(a.manifest)
    elif a.operation=='find-run':print(find_run(a.parent,a.name,a.sha));return
    elif a.operation=='config-paths':
        result=config_paths(a.config)
        if a.out:write_json(a.out,result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if result['forbidden_dataset_file_parameters']:raise ValueError('Dataset-specific recipe file parameter is nonempty; stopped')
        return
    elif a.operation=='config-compare':
        result=config_compare(a.first,a.second);write_json(a.out,result)
        if not result['ok']:raise ValueError('Configuration differences outside allowed list')
    elif a.operation=='health':result=health(a.run,a.base);write_json(a.out,result)
    elif a.operation=='pack12':result=pack12(a.input,a.output)
    elif a.operation=='compare-npz':result=compare_npz(a.first,a.second);write_json(a.out,result)
    elif a.operation=='cams':camera_file(a.dev_meta,a.test_meta,a.dev_archive,a.test_archive,a.output,a.dev_full,a.test_full)
    elif a.operation=='selection':result=selection_equal(a.full,a.subset,a.out)
    elif a.operation=='manifest':
        if a.verify:
            record=read_json(a.out)
            if record['code']!=code_signature('l1') or record['files']!=files_of(record['files']):raise ValueError('L1 manifest changed')
        else:write_json(a.out,{'kind':'l1','code':code_signature('l1'),'files':files_of(a.paths)})
    if result is not None:print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
