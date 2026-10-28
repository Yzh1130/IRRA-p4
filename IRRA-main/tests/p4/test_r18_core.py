"""Core report arithmetic and an explicit audit against synthetic L2 source tables."""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import contextlib,io,json,math,os,sys,tempfile,unittest
from decimal import Decimal,ROUND_HALF_UP
import numpy as np
import pandas as pd
CODE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(CODE))
from tools.p4.r18.r18_core import means,numeric_fields,point_fields
from tools.p4.r18.r18_exact import build
from tools.p4.r18.read_e1_core import read_e1
from tools.p4.r18.common import protocol_prefix
WORK=Path(os.environ.get('R18_WORK',r'E:\论文4\2过程文档\p4_rgbnt201\r18'))
# Full R18J has no per-mixture baseline mAP record. Only these raw
# fields may require the second source, cells.csv; all other finite values
# must occur in full R18J. Audit output lists the actually missing fields.
CSV_ONLY_MIXES=('M4','M6c','M1g','M4c60','M9','M5e','M5f','M4id','M10','M4c25','M4c50','M4c75')
CSV_ONLY_FIELDS=frozenset((101,'cells','mixture_mAP|B*',mix) for mix in CSV_ONLY_MIXES)

class CoreMathTests(unittest.TestCase):
    def test_three_seed_means_ddof1_and_half_up(self):
        rows=[{'src':'judge','set':'full','seed':seed,'key':'pair','v':{'dmAP':value,'nested':{'lo':value-.5},'flag':True,'method':'a'}} for seed,value in zip((101,102,103),(1.,2.,3.))]
        result=means(rows,[101,102,103]);self.assertEqual(len(result),1)
        v=result[0]['v'];self.assertEqual(v['dmAP'],{'n':3,'mean':2.,'sd':1.,'mean_2dp':'2.00','sd_2dp':'1.00'})
        self.assertNotIn('nested.lo',v);self.assertNotIn('flag',v);self.assertNotIn('method',v)
        rows[0]['v']={'value':2.675};single=means(rows[:1],[101])[0]['v']['value'];self.assertEqual(single['mean_2dp'],'2.68');self.assertIsNone(single['sd'])
        self.assertEqual(single['n'],1)

    def test_every_three_seed_point_field_recomputed_and_intervals_excluded(self):
        rows=[]
        for seed,x in zip((101,102,103),(.17,.29,.83)):
            for src,key,v in [('judge','pair|QE+M|QE-DBA|A',{'dmAP':x,'dR1':2*x,'lo':x-1,'hi':x+1,'r_lo':-2,'r_hi':2,'n_q':90,'n_id':15}),
                              ('r17','pair|QE+M-QE+Const|A',{'AP':[x,x-1,x+1],'R1':[2*x,2*x-1,2*x+1]}),
                              ('cells','abs|QE+M|A',{'mAP':x,'R1':2*x,'R5':3*x,'R10':4*x,'mINP':5*x}),
                              ('xcam','pair|QE+M|QE-DBA|A',{'dmAP':x,'lo':x-1,'R5':{'mean':x,'lo':x-1,'hi':x+1}}),
                              ('xcam','coverage|M4',{'mean':{'ir_only_items':20},'ratio':{'same_identity_complete_any_camera':x}}),
                              ('status','runner|x',{'events':2,'heads':{'anyh':2}})]:
                rows.append({'src':src,'set':'full','seed':seed,'key':key,'v':v})
        result=means(rows,[101,102,103]);audit=[]
        for row in result:
            originals=[r for r in rows if (r['src'],r['set'],r['key'])==(row['src'],row['set'],row['key'])]
            self.assertEqual(len(originals),3)
            for field,stats in row['v'].items():
                values=[dict(point_fields(r))[field] for r in originals]
                mu=float(np.mean(values));sd=float(np.std(values,ddof=1))
                self.assertEqual(stats['n'],3);self.assertEqual(np.float64(stats['mean']).tobytes(),np.float64(mu).tobytes());self.assertEqual(np.float64(stats['sd']).tobytes(),np.float64(sd).tobytes())
                for name,value in [('mean',mu),('sd',sd)]:self.assertEqual(stats[name+'_2dp'],str(Decimal(str(value)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)))
                audit.append({'src':row['src'],'key':row['key'],'field':field,'seeds':[101,102,103],'raw_values':values,'n':3})
        actual={(r['src'],r['key']):set(r['v']) for r in result}
        self.assertEqual(actual[('r17','pair|QE+M-QE+Const|A')],{'AP.0','R1.0'})
        self.assertEqual(actual[('judge','pair|QE+M|QE-DBA|A')],{'dmAP','dR1'})
        self.assertNotIn(('status','runner|x'),actual)
        self.assertEqual(actual[('xcam','coverage|M4')],{'ratio.same_identity_complete_any_camera'})
        (WORK/'step3/core_three_seed_mean_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8',newline='\n')

    def test_e1_reader_three_seeds_and_missing_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'core.txt'
            rows=[{'src':'judge','set':'full','seed':s,'key':'pair|QE+M|QE-DBA|A','v':{'dmAP':i+.5,'lo':i-.2}} for i,s in enumerate((101,102,103))]
            p.write_text(''.join('R18J '+json.dumps(r)+'\n' for r in rows),encoding='utf-8',newline='\n')
            self.assertEqual(read_e1(p),[{'seed':s,'dmAP':i+.5,'lo':i-.2} for i,s in enumerate((101,102,103))])
            with self.assertRaises(ValueError):read_e1(p,[104])

class CoreSyntheticOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ.get('R18_SYNTHETIC_OUT',str(WORK/'step3/driver_fixture/p4_r18_out')))
        if not (cls.root/'R18J_core.txt').is_file():raise RuntimeError('Run the synthetic step-3 L2/L2R fixture before this acceptance test')
        cls.full=[json.loads(s[5:]) for s in (cls.root/'R18J.txt').read_text(encoding='utf-8').splitlines()]
        cls.core=[json.loads(s[5:]) for s in (cls.root/'R18J_core.txt').read_text(encoding='utf-8').splitlines()]

    def test_all_actual_core_keys_in_a_three_seed_report_fixture(self):
        # Pure report fixture: replicate generated synthetic raw rows. These
        # are not three model runs and are never reported as experimental data.
        raw=[r for r in self.core if r['seed']==101]
        three=[dict(r,seed=seed) for seed in (101,102,103) for r in raw]
        actual=means(three,[101,102,103]);audit=[]
        def expected_points(r):
            v=r['v'];src,key=r['src'],r['key']
            if src=='judge' and key.startswith('pair|'):return {k:v[k] for k in ('dmAP','dR1')}
            if src=='r17' and key.startswith('pair|'):return {k+'.0':value[0] for k,value in v.items()}
            if src=='r17' and key=='curve':return {g+'.'+k:value for g,row in v.items() for k,value in row.items()}
            if src=='cells':return {k:value for k,value in v.items() if isinstance(value,(int,float)) and not isinstance(value,bool)}
            if src=='diag':return {mix+'.'+k:value for mix,row in v.items() for k,value in row.items()}
            if src=='xcam' and key.startswith('pair|'):
                return {**{k:v[k] for k in ('dmAP','dR1')},**{k+'.mean':v[k]['mean'] for k in ('R5','R10','mINP')}}
            if src=='xcam' and key=='ratio':return {k:value for k,value in v.items() if isinstance(value,(int,float))}
            if src=='xcam' and key.startswith('coverage|'):return {'ratio.'+k:value for k,value in v['ratio'].items() if isinstance(value,(int,float))}
            return {}
        expected={}
        for row in raw:
            fields={k:v for k,v in expected_points(row).items() if math.isfinite(v)}
            if fields:expected[(row['src'],row['set'],row['key'])]=fields
        self.assertEqual({(r['src'],r['set'],r['key']) for r in actual},set(expected))
        for row in actual:
            key=(row['src'],row['set'],row['key']);self.assertEqual(set(row['v']),set(expected[key]))
            source=[r for r in three if (r['src'],r['set'],r['key'])==key]
            self.assertEqual([r['seed'] for r in source],[101,102,103])
            for field,stats in row['v'].items():
                values=[expected_points(r)[field] for r in source];mu=float(np.mean(values));sd=float(np.std(values,ddof=1))
                self.assertEqual(stats['n'],3);self.assertEqual(np.float64(stats['mean']).tobytes(),np.float64(mu).tobytes());self.assertEqual(np.float64(stats['sd']).tobytes(),np.float64(sd).tobytes())
                for name,value in [('mean',mu),('sd',sd)]:self.assertEqual(stats[name+'_2dp'],str(Decimal(str(value)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)))
                audit.append({'src':key[0],'set':key[1],'key':key[2],'field':field,'seeds':[101,102,103],'n':3,'raw_values':values})
        (WORK/'step3/core_all_keys_three_seed_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8',newline='\n')

    def test_full_same_with_and_without_optional_core_and_core_is_small(self):
        out=self.root
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);args=SimpleNamespace(judge=['full='+str(out/'verdict_full.json'),'subset='+str(out/'verdict_subset.json')],r17=['full='+str(out/'report_const_full.json'),'subset='+str(out/'report_const_subset.json')],xcam=str(out/'report_xcam.json'),cells=['full='+str(out/'test/constrained'),'subset='+str(out/'subset/test/constrained')],runner_roots=['full='+str(out/'test/constrained'),'full='+str(out/'const'),'full='+str(out/'xcam'),'subset='+str(out/'subset/test/constrained'),'subset='+str(out/'subset/const')],seeds=[101],out=str(p/'full.txt'),core_out=None)
            with contextlib.redirect_stdout(io.StringIO()):build(args)
            self.assertEqual((p/'full.txt').read_bytes(),(out/'R18J.txt').read_bytes())
        self.assertLessEqual(len(self.core),.15*len(self.full))
        identities=[(r['src'],r['set'],r['seed'],r['key']) for r in self.core];self.assertEqual(len(identities),len(set(identities)))

    def test_every_seed_value_has_full_or_explicit_csv_provenance(self):
        # Recursive extraction includes raw interval lists in full R18J.
        full_values={v for row in self.full for _,v in numeric_fields(row['v'])}
        audit=[];frames={}
        for row in self.core:
            if row['seed']=='mean':continue
            for field,value in numeric_fields(row['v']):
                if value in full_values:continue
                allowed=(row['seed'],row['src'],row['key'],field)
                self.assertIn(allowed,CSV_ONLY_FIELDS,repr((row['set'],allowed,value)))
                scope=row['set'];root=self.root/('test/constrained' if scope=='full' else 'subset/test/constrained')
                if scope not in frames:frames[scope]=pd.concat([pd.read_csv(protocol_prefix(p)+'cells.csv') for p in sorted(root.glob('s101_f*'))])
                mix=field.split('.')[0];frame=frames[scope];expected=float(frame[(frame.evalset=='test')&(frame.method=='B*')&(frame.mixture==mix)].mAP.mean())
                self.assertEqual(np.float64(value).tobytes(),np.float64(expected).tobytes())
                audit.append({'set':scope,'seed':row['seed'],'key':row['key'],'field':field,'value':value,'source':'cells.csv B* evalset=test per-mixture all-unit mean'})
        p=WORK/'step3/core_csv_provenance.json';p.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
        self.assertTrue(audit,'Fixture should exercise values absent from the full report')

    def test_all_core_mean_rows_use_only_their_unrounded_seed_points(self):
        raw=[r for r in self.core if isinstance(r['seed'],int)];audit=[]
        for row in [r for r in self.core if r['seed']=='mean']:
            source=sorted([r for r in raw if (r['src'],r['set'],r['key'])==(row['src'],row['set'],row['key'])],key=lambda r:r['seed'])
            for field,stats in row['v'].items():
                values=[dict(point_fields(r))[field] for r in source if field in dict(point_fields(r))]
                self.assertEqual(stats['n'],len(values));self.assertEqual(stats['n'],1,'Local synthetic fixture deliberately has one seed')
                self.assertEqual(np.float64(stats['mean']).tobytes(),np.float64(np.mean(values)).tobytes());self.assertIsNone(stats['sd'])
                self.assertEqual(stats['mean_2dp'],str(Decimal(str(float(np.mean(values)))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)))
                audit.append({'set':row['set'],'src':row['src'],'key':row['key'],'field':field,'seeds':[r['seed'] for r in source],'n':len(values)})
        (WORK/'step3/core_local_mean_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8',newline='\n')

    def test_all_comparisons_and_example_match_source_json(self):
        for scope,suffix in [('full','full'),('subset','subset')]:
            source=json.loads((self.root/f'verdict_{suffix}.json').read_text())['per_seed']['s101']
            actual={(r['key']):r['v'] for r in self.core if r['src']=='judge' and r['set']==scope and r['seed']==101}
            for name,value in source['pairs'].items():
                pair,group=name.rsplit(' | ',1);x,y=pair.split(' - ',1)
                self.assertEqual(actual[f'pair|{x}|{y}|{group}'],value)
            self.assertEqual(actual['claims'],source['claims'])
        e1=read_e1(self.root/'R18J_core.txt',[101])[0]
        original=json.loads((self.root/'verdict_full.json').read_text())['per_seed']['s101']['pairs']['QE+M - QE-DBA | A']
        self.assertEqual((e1['dmAP'],e1['lo']),(original['dmAP'],original['lo']))

if __name__=='__main__':unittest.main(verbosity=2)
