"""R18 real annotation parity plus synthetic exporter regression, never real test inference."""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType
from unittest.mock import patch

CODE=Path(__file__).resolve().parents[2]
WORK=Path(os.environ.get('R18_WORK',r'E:\论文4\2过程文档\p4_rgbnt201\r18'))

def capture(root, output):
    sys.path.insert(0,str(root))
    from datasets.rgbtpedes import RGBTPEDES
    from tools.p4.export_dev_records import validate_dev_split, _resolve_annotation
    from tools.p4.export_eval_split import build_test_data
    data=CODE/'datasets/TextReID';split=CODE/'splits/dev_190.json'
    ds=RGBTPEDES(root=str(data),dev_split_file=str(split),verbose=False)
    anno=_resolve_annotation(data)
    result={'train':ds.train,'val':ds.val,'test':ds.test,'dev':validate_dev_split(anno,split),'test_meta':build_test_data(anno,split),'annotation':str(anno)}
    Path(output).write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8',newline='\n')

if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='--capture':
    capture(Path(sys.argv[2]),Path(sys.argv[3]));sys.exit(0)

sys.path.insert(0,str(CODE));sys.path.insert(0,str(Path(__file__).parent))
from tools.p4 import export_eval_split as exporter
from tools.p4.export_dev_records import validate_dev_split
from tools.p4.r18.common import item_meta,read_json
from datasets.build import __factory as DATASET_FACTORY

class DataTests(unittest.TestCase):
    def test_rgbt_real_annotations_exactly_equal_to_git_archive_snapshot(self):
        snapshot=WORK/'base_snapshot/IRRA-main'
        self.assertTrue(snapshot.is_dir(),'Create WORK/base_snapshot with git archive first')
        with tempfile.TemporaryDirectory() as temporary:
            results=[]
            for index,root in enumerate((snapshot,CODE)):
                dest=Path(temporary)/f'{index}.json'
                p=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--capture',str(root),str(dest)],capture_output=True,text=True,encoding='utf-8')
                self.assertEqual(p.returncode,0,p.stdout+p.stderr);results.append(read_json(dest))
            self.assertEqual(results[0],results[1])

    def test_rgbnt_factory_counts_nonblank_and_identity_partition(self):
        ds=DATASET_FACTORY['RGBNT201-PEDES'](root=str(CODE/'datasets/TextReID'),dev_split_file=str(CODE/'splits/rgbnt201_dev30.json'),verbose=False)
        self.assertEqual((len(ds.train_id_container),len(ds.train_annos),len(ds.train)),(141,3269,3267))
        self.assertEqual((len(ds.val_id_container),len(ds.val_annos),len(ds.val['captions'])),(30,682,681))
        self.assertEqual((len(ds.test_id_container),len(ds.test_annos),len(ds.test['captions'])),(30,836,836))
        groups=[{a['id'] for a in annos} for annos in (ds.train_annos,ds.val_annos,ds.test_annos)]
        for i in range(3):
            for j in range(i):self.assertFalse(groups[i]&groups[j])
        self.assertEqual(len(ds.blank_caption_files),3)
        self.assertTrue(all(s[-1].strip() for s in ds.train))
        self.assertTrue(all('NI' not in Path(p).parts for p in ds.val['img_paths']+ds.val['img_paths_t']))

    def test_both_query_and_gallery_orders_and_item_meta(self):
        ds=DATASET_FACTORY['RGBNT201-PEDES'](root=str(CODE/'datasets/TextReID'),dev_split_file=str(CODE/'splits/rgbnt201_dev30.json'),verbose=False)
        annos=read_json(ds.anno_path);split=CODE/'splits/rgbnt201_dev30.json';dev=set(read_json(split)['dev_ids'])
        for mode in ('dev','test'):
            meta=validate_dev_split(ds.anno_path,split,'RGBNT201-PEDES') if mode=='dev' else exporter.build_test_data(ds.anno_path,split,'RGBNT201-PEDES')
            expected=ds.val if mode=='dev' else ds.test
            self.assertEqual([q['caption'] for q in meta['queries']],expected['captions'])
            self.assertEqual(meta['gallery_pids'],expected['image_pids'])
            selected=[i for i,a in enumerate(annos) if (mode=='test' and a['split']=='test') or (mode=='dev' and a['split']=='train' and a['id'] in dev)]
            meta['gallery_annotation_indices']=selected
            info=item_meta(ds.anno_path,meta,'RGBNT201-PEDES',mode)
            self.assertEqual(info['gallery_filenames'],[Path(p).name for p in expected['img_paths']])
            self.assertEqual(info['query_raw_pids'],[q['pid'] for q in meta['queries']])
            self.assertEqual(len(info['gallery_cameras']),len(expected['img_paths']))
            self.assertTrue(all(c.startswith('cam') for c in info['gallery_cameras']))
            self.assertEqual(len(info['query_filenames']),len(expected['captions']))

    def test_dataset_schema_rejects_without_reading_NI(self):
        from datasets.rgbnt201pedes import RGBNT201PEDES
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'RGBNT201';(data/'RGB').mkdir(parents=True);(data/'T').mkdir();(data/'NI').mkdir()
            rows=[dict(split='train',id=1,file_path='RGBT/000001_cam1_0_00.jpg',captions=['existing']),dict(split='test',id=2,file_path='RGBT/000002_cam2_0_00.jpg',captions=[''])]
            path=data/'captionform0_8B.json';path.write_text(json.dumps(rows),encoding='utf-8')
            ds=RGBNT201PEDES(root=str(root),verbose=False)
            self.assertEqual(len(ds.test['img_paths']),1);self.assertEqual(ds.test['captions'],[])
            for mutation in ('split','captions'):
                changed=[dict(r) for r in rows]
                changed[0][mutation]='val' if mutation=='split' else ['a','b']
                path.write_text(json.dumps(changed),encoding='utf-8')
                with self.assertRaises(ValueError):RGBNT201PEDES(root=str(root),verbose=False)

    def test_default_rgbt_export_bytes_equal_to_snapshot_and_meta_optin(self):
        import test_modality_features as fixtures
        source=WORK/'base_snapshot/IRRA-main/tools/p4/export_eval_split.py'
        legacy=ModuleType('r18_snapshot_exporter');legacy.__file__=str(source)
        exec(compile(source.read_text(encoding='utf-8'),str(source),'exec'),legacy.__dict__)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cli=fixtures._setup_export(root);cli.save_modality_features=True;cli.split='dev'
            directory=Path(cli.output_dir)
            results=[]
            for module in (legacy,exporter):
                with contextlib.redirect_stdout(io.StringIO()),patch.object(module.time,'perf_counter',return_value=0.0),patch.object(module,'datetime') as clock:
                    clock.now.return_value.isoformat.return_value='2000-01-01T00:00:00+00:00'
                    fixtures._run_export(cli,module.export_split)
                results.append({p.relative_to(directory).as_posix():p.read_bytes() for p in directory.rglob('*') if p.is_file() and p.name!='test_log.txt'})
                self.assertTrue(directory.resolve().is_relative_to(root.resolve()))
                shutil.rmtree(directory)
            self.assertEqual(results[0],results[1])
            self.assertEqual(len([k for k in results[0] if k.startswith('features/')]),11)
            self.assertNotIn('item_meta.json',results[1])
            cli.save_item_meta=True
            with contextlib.redirect_stdout(io.StringIO()):fixtures._run_export(cli,exporter.export_split)
            info=read_json(directory/'item_meta.json')
            self.assertEqual(info['gallery_filenames'],[f'{i}.jpg' for i in range(10)])
            self.assertEqual(info['gallery_cameras'],[None]*10)

if __name__=='__main__':unittest.main(verbosity=2)
