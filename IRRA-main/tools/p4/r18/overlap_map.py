"""Read-only pair/identity overlap mapping. Perceptual hash functions copied verbatim
from audit_rgbnt201.py, SHA256 6eaa52375212e5ee5ed946efcd47650b12ad98794abc37d5b4e1af1d98c91af2.
"""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import re
import sys
import numpy as np
from PIL import Image
CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(CODE))
from tools.p4.r18.common import read_json, write_json, write_csv, sha256

_C = np.cos(np.pi * (np.arange(32)[None, :] + .5) * np.arange(32)[:, None] / 32)
_C[0] *= np.sqrt(1 / 32)
_C[1:] *= np.sqrt(2 / 32)
assert np.max(np.abs(_C @ _C.T - np.eye(32))) < 1e-12


def bits_to_hex(bits):
    return np.packbits(np.asarray(bits).reshape(-1), bitorder="big").tobytes().hex()


def image_hashes(image):
    gray = image.convert("L")
    small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.uint8)
    dhash = bits_to_hex(small[:, 1:] > small[:, :-1])
    block = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = (_C @ block @ _C.T)[:8, :8]
    phash = bits_to_hex(low > np.median(low))
    return dhash, phash


def popcounts(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    lookup = np.asarray([i.bit_count() for i in range(256)], dtype=np.uint8)
    return lookup[x.view(np.uint8).reshape(*x.shape, 8)].sum(axis=-1, dtype=np.uint8)


RULE = """图像对：同名的 RGB 与 T 构成一个对，两边都这样配。RGBNT201 的对 a 与 PEDES 的对 b\"匹配\"，当且仅当满足下面任一条：
- 任一模态的 SHA-256 相同；
- RGB 与 T 两个模态都满足 dHash ≤ 6 且 pHash ≤ 6：a 的 RGB 对 b 的 RGB，a 的 T 对 b 的 T。
只有一个模态近重复的，记为\"待人工查看\"，不算匹配。
重叠身份：至少有一个对匹配的 RGBNT201 身份。"""

def fingerprint(path):
    with Image.open(path) as im:
        im.load();dh,ph=image_hashes(im);size=(im.width,im.height)
    return dict(sha=sha256(path),dhash=dh,phash=ph,width=size[0],height=size[1])

def load_pairs(root, annotation, workers):
    annos=read_json(annotation);names={}
    for record in annos:
        name=Path(record['file_path']).name
        if name in names:raise ValueError(f'Duplicate pair record: {name}')
        names[name]=record
    paths=[Path(root)/m/name for name in names for m in ('RGB','T')]
    if not all(p.is_file() for p in paths):raise ValueError('Annotation pair references a missing RGB/T image')
    with ThreadPoolExecutor(max_workers=workers) as pool:values=list(pool.map(fingerprint,paths))
    pairs=[]
    for i,(name,a) in enumerate(names.items()):
        pairs.append(dict(name=name,pid=name.split('_')[0],id=int(a['id']),split=a['split'],captions=a['captions'],RGB=values[2*i],T=values[2*i+1]))
    return pairs

def run(rgbnt, pedes, dev_split, out, output_json, workers=4):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    a=load_pairs(rgbnt,Path(rgbnt)/'captionform0_8B.json',workers)
    b=load_pairs(pedes,Path(pedes)/'Caption.json',workers)
    split=read_json(dev_split);dev=set(map(int,split['dev_ids']));train=set(map(int,split['train_reduced_ids']))
    bvals={(m,h):np.asarray([int(p[m][h],16) for p in b],np.uint64) for m in ('RGB','T') for h in ('dhash','phash')}
    bsha={m:np.asarray([p[m]['sha'] for p in b]) for m in ('RGB','T')}
    matches=[];manual=[];equal_source={m:set() for m in ('RGB','T')}
    for start in range(0,len(a),128):
        chunk=a[start:start+128];dist={}
        for m in ('RGB','T'):
            for h in ('dhash','phash'):
                aa=np.asarray([int(p[m][h],16) for p in chunk],np.uint64)
                dist[(m,h)]=popcounts(aa[:,None]^bvals[(m,h)][None,:])
        sha={m:np.asarray([p[m]['sha'] for p in chunk])[:,None]==bsha[m][None,:] for m in ('RGB','T')}
        near={m:(dist[(m,'dhash')]<=6)&(dist[(m,'phash')]<=6) for m in ('RGB','T')}
        matched=sha['RGB']|sha['T']|(near['RGB']&near['T'])
        pending=(near['RGB']^near['T'])&~matched
        for mask,destination in ((matched,matches),(pending,manual)):
            for i,j in zip(*np.nonzero(mask)):
                aa,bb=chunk[int(i)],b[int(j)]
                equal=[m for m in ('RGB','T') if bool(sha[m][i,j])]
                for m in equal:equal_source[m].add(aa['name'])
                same=set(aa['captions'])&set(bb['captions'])
                destination.append(dict(rgbnt_filename=aa['name'],rgbnt_pid=aa['pid'],rgbnt_id=aa['id'],rgbnt_split=aa['split'],
                   pedes_filename=bb['name'],pedes_id=bb['id'],pedes_split=bb['split'],pedes_in_dev=bb['id'] in dev,pedes_in_train_reduced=bb['id'] in train,
                   basis='sha' if equal else 'near' if destination is matches else 'manual_single_modality',sha_modalities='|'.join(equal),
                   rgb_dhash=int(dist[('RGB','dhash')][i,j]),rgb_phash=int(dist[('RGB','phash')][i,j]),t_dhash=int(dist[('T','dhash')][i,j]),t_phash=int(dist[('T','phash')][i,j]),
                   identical_caption_strings=len(same),identical_nonempty_caption_strings=sum(bool(t.strip()) for t in same)))
    order=lambda r:(r['rgbnt_id'],r['rgbnt_filename'],r['pedes_id'],r['pedes_filename'])
    matches.sort(key=order);manual.sort(key=order)
    fields=['rgbnt_filename','rgbnt_pid','rgbnt_id','rgbnt_split','pedes_filename','pedes_id','pedes_split','pedes_in_dev','pedes_in_train_reduced','basis','sha_modalities','rgb_dhash','rgb_phash','t_dhash','t_phash','identical_caption_strings','identical_nonempty_caption_strings']
    write_csv(out/'overlap_pairs.csv',matches,fields);write_csv(out/'manual_review.csv',manual,fields)
    by_a=defaultdict(set);by_b=defaultdict(set)
    for r in matches:by_a[r['rgbnt_id']].add(r['pedes_id']);by_b[r['pedes_id']].add(r['rgbnt_id'])
    info={p['id']:p for p in a};id_rows=[]
    for aid,pids in sorted(by_a.items()):
        for pid in sorted(pids):
            bp=[p for p in b if p['id']==pid];mp=[r for r in matches if r['rgbnt_id']==aid and r['pedes_id']==pid]
            id_rows.append(dict(rgbnt_id=aid,rgbnt_pid=info[aid]['pid'],rgbnt_split=info[aid]['split'],pedes_id=pid,pedes_split='|'.join(sorted({p['split'] for p in bp})),
                pedes_total_pairs=len(bp),pedes_RGB_128x256=sum((p['RGB']['width'],p['RGB']['height'])==(128,256) for p in bp),
                pedes_T_128x256=sum((p['T']['width'],p['T']['height'])==(128,256) for p in bp),matched_pairs=len({r['pedes_filename'] for r in mp}),
                rgbnt_to_pedes_count=len(pids),pedes_to_rgbnt_count=len(by_b[pid])))
    id_fields=list(id_rows[0]) if id_rows else ['rgbnt_id','rgbnt_pid','rgbnt_split','pedes_id','pedes_split','pedes_total_pairs','pedes_RGB_128x256','pedes_T_128x256','matched_pairs','rgbnt_to_pedes_count','pedes_to_rgbnt_count']
    write_csv(out/'overlap_ids.csv',id_rows,id_fields)
    test_ids=sorted(i for i in by_a if info[i]['split']=='test');train_ids=sorted(i for i in by_a if info[i]['split']=='train')
    total_test=len({p['id'] for p in a if p['split']=='test'})
    result=dict(rule=RULE,thresholds={'dhash':6,'phash':6},pedes_dir='RGBT-PEDES-Final',overlap_test_ids=test_ids,
        overlap_test_pids=sorted({info[i]['pid'] for i in test_ids}),overlap_train_ids=train_ids,n_test_ids=total_test,n_test_ids_after_exclusion=total_test-len(test_ids),
        pedes_ids_of_overlap_test_ids={str(i):sorted(by_a[i]) for i in test_ids},
        pedes_split_of_overlap_test_ids={str(i):{str(pid):sorted({p['split'] for p in b if p['id']==pid}) for pid in sorted(by_a[i])} for i in test_ids},
        evidence_sha256=sha256(out/'overlap_pairs.csv'),script_sha256=sha256(Path(__file__)))
    write_json(output_json,result)
    report=dict(rgbnt_pairs=len(a),pedes_pairs=len(b),byte_identical_RGB_images=len(equal_source['RGB']),byte_identical_T_images=len(equal_source['T']),
        match_records=len(matches),matched_rgbnt_pairs=len({r['rgbnt_filename'] for r in matches}),manual_review_records=len(manual),
        overlap_test_ids=test_ids,overlap_test_pids=result['overlap_test_pids'],overlap_train_identity_count=len(train_ids),
        one_to_many={str(k):sorted(v) for k,v in by_a.items() if len(v)>1},many_to_one={str(k):sorted(v) for k,v in by_b.items() if len(v)>1},
        matched_pair_caption_equal_records=sum(r['identical_caption_strings']>0 for r in matches),matched_pair_nonempty_caption_equal_records=sum(r['identical_nonempty_caption_strings']>0 for r in matches),
        identity_mapping=id_rows,manual_review=manual,overlap_json=result)
    (out/'overlap_report.md').write_text('# RGBNT201 / RGBT-PEDES overlap: read-only evidence\n\n```json\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8',newline='\n')
    write_json(out/'overlap_summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('manual_review','identity_mapping','overlap_json')},ensure_ascii=False))
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--rgbnt',type=Path,default=CODE/'datasets/TextReID/RGBNT201');ap.add_argument('--pedes',type=Path,default=CODE/'datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final')
    ap.add_argument('--dev_split',type=Path,default=CODE/'splits/dev_190.json');ap.add_argument('--out',required=True);ap.add_argument('--output_json',type=Path,default=CODE/'splits/rgbnt201_overlap.json');ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args();run(a.rgbnt,a.pedes,a.dev_split,a.out,a.output_json,a.workers)


