"""Create and verify a deterministic upload tar; source data are read-only."""
from pathlib import Path
import argparse
import hashlib
import sys
import tarfile
CODE=Path(__file__).resolve().parents[3];sys.path.insert(0,str(CODE))
from tools.p4.r18.common import sha256

def package(data,out):
    data,out=Path(data),Path(out);out.mkdir(parents=True,exist_ok=True)
    files=sorted([*data.joinpath('RGB').glob('*'),*data.joinpath('T').glob('*'),data/'captionform0_8B.json'],key=lambda p:p.relative_to(data).as_posix())
    if len(files)!=9575 or not all(p.is_file() for p in files):raise ValueError('Expected 9575 source files')
    expected={'RGBNT201/'+p.relative_to(data).as_posix():sha256(p) for p in files}
    target=out/'RGBNT201_p4data.tar'
    if target.exists():raise FileExistsError(target)
    with tarfile.open(target,'w',format=tarfile.USTAR_FORMAT) as tar:
        for p in files:
            info=tarfile.TarInfo('RGBNT201/'+p.relative_to(data).as_posix());info.size=p.stat().st_size;info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o644
            with p.open('rb') as f:tar.addfile(info,f)
    with tarfile.open(target,'r:') as tar:
        members=tar.getmembers()
        if [m.name for m in members]!=list(expected):raise ValueError('Tar order/member mismatch')
        for member in members:
            if '\\' in member.name or member.mtime!=0 or member.uid!=0 or member.gid!=0:raise ValueError('Tar metadata mismatch')
            with tar.extractfile(member) as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=expected[member.name]:raise ValueError('Tar member checksum mismatch: '+member.name)
    (out/'RGBNT201_files.sha256').write_text(''.join(h+'  '+n+'\n' for n,h in expected.items()),encoding='utf-8',newline='\n')
    (out/'RGBNT201_p4data.tar.sha256').write_text(sha256(target)+'  RGBNT201_p4data.tar\n',encoding='utf-8',newline='\n')
    print('VERIFIED',len(expected),'members',target.stat().st_size,'bytes',sha256(target))
    return expected

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,default=CODE/'datasets/TextReID/RGBNT201');ap.add_argument('--out',required=True)
    a=ap.parse_args();package(a.data,a.out)
