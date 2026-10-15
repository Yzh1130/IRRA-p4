"""Pack exported modality features (export_dev_records.py --save_modality_features) into one npz per checkpoint.

Each export dir must contain features/<scenario>.pt for the 7 scenarios below. Arrays identical to the clean
scenario are not stored again (the reader falls back to clean/<key>); fused features are skipped.
Usage:
  python -B tools/p4/pack_feats.py --export_dirs DIR1 DIR2 --names NAME1 NAME2 --out_dir OUT [--zip OUT.zip]
"""
import argparse, json, os, zipfile
import numpy as np
import torch

SCEN = ['clean', 'rgb_missing', 'ir_missing', 'rgb_noise_010', 'ir_noise_010', 'rgb_blur_k7', 'ir_blur_k7']


def flatten(obj, prefix):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            items.update(flatten(v, f'{prefix}.{k}' if prefix else str(k)))
    elif torch.is_tensor(obj):
        t = obj.detach().cpu()
        items[prefix] = (t.float() if t.is_floating_point() else t).numpy()
    elif isinstance(obj, np.ndarray):
        items[prefix] = obj
    elif isinstance(obj, (list, tuple)) and obj and all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in obj):
        items[prefix] = np.asarray(obj, dtype=np.int64)
    elif isinstance(obj, (list, tuple)) and obj and all(isinstance(x, str) for x in obj):
        items[prefix] = np.asarray(obj)
    else:
        items[prefix] = np.asarray(json.dumps(obj, default=str))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export_dirs', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--zip', default=None)
    a = ap.parse_args()
    assert len(a.export_dirs) == len(a.names)
    os.makedirs(a.out_dir, exist_ok=True)
    for d, name in zip(a.export_dirs, a.names):
        arrays, manifest, clean = {}, {}, None
        for scen in SCEN:
            p = os.path.join(d, 'features', f'{scen}.pt')
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
            flat = flatten(torch.load(p, map_location='cpu', weights_only=False), '')
            if scen == 'clean':
                clean = flat
            for k, v in flat.items():
                if 'fused' in k.lower():
                    manifest[f'{scen}/{k}'] = 'skipped_fused'; continue
                if scen != 'clean' and k in clean and clean[k].shape == v.shape and clean[k].dtype == v.dtype and np.array_equal(clean[k], v):
                    manifest[f'{scen}/{k}'] = 'same_as_clean'; continue
                arrays[f'{scen}/{k}'] = v
                manifest[f'{scen}/{k}'] = [list(v.shape), str(v.dtype)]
        np.savez_compressed(os.path.join(a.out_dir, f'{name}.npz'), **arrays)
        json.dump(manifest, open(os.path.join(a.out_dir, f'{name}_manifest.json'), 'w'), indent=1)
        print(name, round(os.path.getsize(os.path.join(a.out_dir, f'{name}.npz')) / 1e6, 1), 'MB', len(arrays), 'arrays')
    if a.zip:
        with zipfile.ZipFile(a.zip, 'w', zipfile.ZIP_STORED) as z:
            for f in sorted(os.listdir(a.out_dir)):
                z.write(os.path.join(a.out_dir, f), os.path.join(os.path.basename(a.out_dir.rstrip('/')), f))
        print(a.zip, round(os.path.getsize(a.zip) / 1e6, 1), 'MB')


if __name__ == '__main__':
    main()
