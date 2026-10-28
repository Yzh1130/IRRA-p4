"""Bounded existing train.py run; save its in-memory smoke model for dev-only acceptance.

This wrapper adds no parser/config keys to train.py. The existing smoke return
still disables evaluation; only this explicit local wrapper saves a checkpoint.
"""
from pathlib import Path
import argparse
import hashlib
import os
import runpy
import sys
CODE=Path(__file__).resolve().parents[3]

def main(cli):
    sys.path.insert(0,str(CODE));os.chdir(CODE)
    from model import clip_model
    def cached_only(url,root):
        path=Path(root)/url.rsplit('/',1)[-1]
        if not path.is_file():raise FileNotFoundError(f'Offline smoke needs cached pretrained CLIP: {path}')
        expected=url.split('/')[-2]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Cached pretrained CLIP hash mismatch; network disabled')
        return str(path)
    clip_model._download=cached_only
    import urllib.request
    def offline(*args,**kwargs):raise RuntimeError('Network prohibited during local acceptance')
    urllib.request.urlopen=offline
    os.environ['TRAIN_SEED']=str(cli.seed)
    sys.argv=['train.py','--config_file',str(Path(cli.config_file).resolve()),'--dataset_name',cli.dataset_name,
       '--root_dir',str(Path(cli.root_dir).resolve()),'--dev_split_file',str(Path(cli.dev_split_file).resolve()),
       '--name',cli.name,'--output_dir',str(Path(cli.output_dir).resolve()),'--batch_size','2','--num_workers','0','--smoke_steps','20','--MLM']
    namespace=runpy.run_path(str(CODE/'train.py'),run_name='__main__')
    if cli.save_smoke_checkpoint:
        import torch
        checkpoint=Path(namespace['args'].output_dir)/'last.pth'
        if checkpoint.exists():raise FileExistsError(checkpoint)
        torch.save({'model':namespace['model'].state_dict(),'epoch':1,'r18_local_smoke_checkpoint':True},checkpoint)
        print('[R18_SMOKE_CHECKPOINT]',checkpoint)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config_file',required=True);ap.add_argument('--dataset_name',required=True,choices=['RGBT-PEDES','RGBNT201-PEDES']);ap.add_argument('--root_dir',required=True);ap.add_argument('--dev_split_file',required=True);ap.add_argument('--name',required=True);ap.add_argument('--output_dir',required=True);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--save_smoke_checkpoint',action='store_true')
    main(ap.parse_args())
