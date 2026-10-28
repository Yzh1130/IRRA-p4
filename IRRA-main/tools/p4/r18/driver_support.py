"""Read-only checks and result packaging used by the R18 shell driver."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,os,platform,sys,tarfile
CODE=Path(__file__).resolve().parents[3];sys.path.insert(0,str(CODE))
from tools.p4.r18.common import read_json,write_json,sha256

def prepare_data(root,upload):
    root,upload=Path(root),Path(upload);manifest=upload/'RGBNT201_files.sha256'
    rows=[line.split('  ',1) for line in manifest.read_text(encoding='utf-8').splitlines()]
    if len(rows)!=9575:raise ValueError('Data manifest must contain exactly 9575 files')
    if not (root/'RGBNT201').exists():
        expected,name=(upload/'RGBNT201_p4data.tar.sha256').read_text(encoding='utf-8').strip().split('  ',1)
        if name!='RGBNT201_p4data.tar' or sha256(upload/name)!=expected:raise ValueError('Upload archive hash mismatch')
        with tarfile.open(upload/name,'r:') as tf:tf.extractall(root,filter='data')
    for expected,name in rows:
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or not name.startswith('RGBNT201/') or '\\' in name:raise ValueError('Unsafe data manifest path')
        if sha256(root/rel)!=expected:raise ValueError('Data file hash mismatch: '+name)
    print('DATA_OK 9575 files')

def construct(root,split):
    from tools.p4.r18.common import dataset_class
    ds=dataset_class('RGBNT201-PEDES')(root=str(root),dev_split_file=str(split),verbose=False)
    counts={}
    for name in ('train','val','test'):
        annos=getattr(ds,name+'_annos')
        counts[name]={'identities':len({r['id'] for r in annos}),'pairs':len(annos),'captions':sum(bool(c.strip()) for r in annos for c in r['captions'])}
    expected={'train':{'identities':141,'pairs':3269,'captions':3267},'val':{'identities':30,'pairs':682,'captions':681},'test':{'identities':30,'pairs':836,'captions':836}}
    if counts!=expected:raise ValueError('Unexpected dataset counts: '+str(counts))
    sets=[{r['id'] for r in getattr(ds,k+'_annos')} for k in ('train','val','test')]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('Identity split overlap')
    print(json.dumps(counts,ensure_ascii=False))

def environment(synthetic=False):
    import numpy as np,torch
    print('Python',sys.version,'numpy',np.__version__,'torch',torch.__version__,'CUDA',torch.version.cuda,'platform',platform.platform())
    np.show_config()
    if not synthetic:
        if not torch.cuda.is_available():raise ValueError('CUDA is required for a real L1/L2 run')
        print('GPU',torch.cuda.get_device_name(0))
    available=0
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):available=int(line.split()[1])*1024
        try:
            limit=Path('/sys/fs/cgroup/memory.max').read_text().strip();used=int(Path('/sys/fs/cgroup/memory.current').read_text())
            if limit.isdigit():available=min(available,max(0,int(limit)-used))
        except (OSError,ValueError):pass
    # Same conservative memory budget as R17: 12 GiB per fit plus 6 GiB reserve.
    jobs=max(1,min(3,(available//2**30-6)//12)) if available else 1
    print('AVAILABLE_BYTES',available,'PARALLEL_FITS',jobs)
    return jobs

def validate_reports(paths):
    for path in paths:
        data=read_json(path)
        if data.get('status')=='INVALID' or data.get('status') not in (None,'VALID'):raise ValueError('INVALID report: '+str(path))
        if 'valid' in data:
            if data['valid'] is not True or data.get('problems'):raise ValueError('INVALID frozen R17 report: '+str(path))
        elif data.get('status') is None and not {'final','per_seed','evalset'}<=data.keys():raise ValueError('Unexpected frozen judge JSON')
    print('REPORTS_VALID',len(paths))

def smoke_health(directory,output):
    records=list(Path(directory).rglob('run_record.json'))
    if len(records)!=1:raise ValueError('Expected one smoke run record')
    record=read_json(records[0]);steps=record['steps']
    if record['status']!='smoke_completed' or len(steps)!=30 or not all(s['finite'] for s in steps):raise ValueError('Smoke incomplete/nonfinite')
    write_json(output,{'steps':steps,'record':str(records[0]),'status':record['status']})

def finish(out,root,phase,status):
    out,root=Path(out),Path(root);digest=out/('DIGEST18_'+phase+'.txt')
    parts=['STATUS_'+phase+'='+status]
    parts.append('\n===== run_frozen auxiliary single-class fits by unit =====')
    for log in sorted(out.rglob('degenerate_fits.json')):
        events=read_json(log)['events'];counts=Counter(e['head'] for e in events)
        parts.append(str(log.parent.relative_to(out))+' events='+str(len(events))+' heads='+json.dumps(dict(sorted(counts.items())),sort_keys=True))
    for p in sorted(out.glob('checks/*.json'))+sorted(out.glob('report*.json'))+sorted(out.glob('verdict*.json'))+sorted(out.glob('R18J*.txt')):
        parts.extend(['\n===== '+str(p.relative_to(out))+' =====',p.read_text(encoding='utf-8')])
    parts.extend(['\n===== stage manifests =====']+[str(p.relative_to(out))+' '+sha256(p) for p in sorted((out/'.manifests').glob('*.json'))])
    digest.write_text('\n'.join(parts)+'\n',encoding='utf-8',newline='\n')
    target=root/('p4_r18_'+phase+'_results.tar.gz')
    with tarfile.open(target,'w:gz') as tf:
        for p in sorted(out.rglob('*')):
            if not p.is_file() or p.suffix in ('.pth','.npz','.pt'):continue
            if p.suffix=='.pkl' and 'perq' in p.name:continue
            tf.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
    print('RESULT_PACKAGE',target,'SHA256',sha256(target))

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='op',required=True)
    p=sub.add_parser('data');p.add_argument('--root',required=True);p.add_argument('--upload',required=True)
    p=sub.add_parser('construct');p.add_argument('--root',required=True);p.add_argument('--split',required=True)
    p=sub.add_parser('environment');p.add_argument('--synthetic',action='store_true');p.add_argument('--parallel_file')
    p=sub.add_parser('reports');p.add_argument('paths',nargs='+')
    p=sub.add_parser('finish');p.add_argument('--out',required=True);p.add_argument('--root',required=True);p.add_argument('--phase',required=True);p.add_argument('--status',required=True)
    p=sub.add_parser('base-config');p.add_argument('--config',required=True)
    p=sub.add_parser('smoke');p.add_argument('--directory',required=True);p.add_argument('--out',required=True)
    a=ap.parse_args()
    if a.op=='data':prepare_data(a.root,a.upload)
    elif a.op=='construct':construct(a.root,a.split)
    elif a.op=='environment':
        jobs=environment(a.synthetic)
        if a.parallel_file:Path(a.parallel_file).write_text(str(jobs)+'\n',encoding='utf-8',newline='\n')
    elif a.op=='reports':validate_reports(a.paths)
    elif a.op=='smoke':smoke_health(a.directory,a.out)
    elif a.op=='finish':finish(a.out,a.root,a.phase,a.status)
    elif a.op=='base-config':
        from utils.config import load_config_mapping
        p=Path(load_config_mapping(a.config)['config_file'])
        if not p.is_file():raise FileNotFoundError(p)
        print(p)

if __name__=='__main__':main()
