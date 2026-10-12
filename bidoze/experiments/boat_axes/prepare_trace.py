"""Append a frame-pooled axis-trace channel while preserving all baseline arrays and labels."""
import argparse,json,hashlib,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from pisces_lite.proc import ProcessingConfig
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE
from arcasleep.data.batching import recording_input,pad_pooled
from arcasleep.data.prepare import load_dataset,prepare_subject,clean_accelerometer_rows
from boat_evidence.prepare import PSG_MAP


def main(a):
    (a.out/'nights').mkdir(parents=True,exist_ok=True);(a.out/'bundles').mkdir(exist_ok=True)
    if not (a.out/'bundles/seed0').exists():shutil.copytree(a.data/'bundles/seed0',a.out/'bundles/seed0')
    for f in ('splits.json','inventory.csv'):shutil.copy(a.data/f,a.out/f)
    cfg=json.loads((a.data/'bundles/seed0/preprocessing.json').read_text());proc=ProcessingConfig.from_dict(dict(cfg,spectral_channels=['x','y','z']))
    meta=pd.read_csv(a.data/'inventory.csv');plans=json.loads((a.data/'splits.json').read_text())
    if a.smoke:
        d=plans['D'];names=[d['fit'][0],d['fit'][-1],d['eval'][0],d['eval'][-1],d['test'][0],d['guard'][0]];meta=meta[meta.name.isin(names)]
    clinical=None;audits=[]
    for r in meta.to_dict('records'):
        target=a.out/'nights'/f"{r['name']}.npz"
        if target.exists():continue
        with np.load(a.data/'nights'/target.name) as z:fields={k:z[k].copy() for k in z.files}
        n=int(fields['n']);stem=r['name'].split('_subject_',1)[1]
        if r['role'] in ('fit','eval'):
            file=a.fused/'nights'/f"{r['name']}_night.npz"
            with np.load(file) as z:
                assert len(z['y'])==n and np.array_equal(z['y'],fields['y'][:n]),'Fused labels differ from frozen preparation'
                raw=np.column_stack([z['t'][:len(z['xyz'])],z['xyz']]).astype(float)
            provenance='aligned fused export; float32 raw axes/times; labels audited exactly'
        elif r['cohort']=='SleepAccel-Clinical':
            if clinical is None:clinical=load_dataset(a.root/'SleepAccel-Clinical')
            item=prepare_subject(clinical,stem);raw=item.accel_values.copy();raw[:,0]-=raw[0,0]
            mapped=np.array([PSG_MAP.get(v,v) for v in item.labels[:n]])
            assert len(mapped)==n and np.array_equal(mapped,fields['y'][:n]),'Clinical labels differ from frozen preparation'
            provenance='source prepared/aligned with original cache protocol; labels audited exactly'
        else:
            raw=clean_accelerometer_rows(pd.read_csv(a.root/r['cohort']/'cleaned_accelerometer'/f'{stem}.csv').to_numpy(float))
            provenance='original boat clock, origin 0; unchanged frozen labels'
        axis=proc.apply(raw,normalize=False,padding=False,origin=0.)
        valid=~np.all(axis==SPECTROGRAM_PADDING_VALUE,axis=(1,2))
        power=np.maximum(np.exp(axis)-np.finfo(float).eps,0).sum(2,keepdims=True)
        trace=np.log(power+np.finfo(float).eps);trace[~valid]=SPECTROGRAM_PADDING_VALUE
        epochs=len(trace)//15;trace=trace[:epochs*15]
        extra=pad_pooled(recording_input(trace,epochs,1),3000).astype(np.float32)
        assert np.allclose(extra[:n,-1,0],fields['x'][:n,-1,0],atol=1e-6),'Trace and original frame coverage differ'
        extra[:,-1,0]=fields['x'][:,-1,0]
        fields['x_trace']=np.concatenate([fields['x'],extra],axis=2)
        assert np.array_equal(fields['x_trace'][:,:,:2],fields['x'])
        assert np.isfinite(fields['x_trace']).all()
        np.savez_compressed(target,**fields)
        audits.append(dict(name=r['name'],n=n,raw_provenance=provenance,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        print('trace prepared',r['name'],n,flush=True)
    (a.out/'trace-input-recipe.json').write_text(json.dumps(dict(code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        channels=['original magnitude','original difference-norm jerk','axis trace: log(sum linear per-axis PSD) then frame mean'],
        covariance='constant-axis rotation invariant spectrum; not a full directional representation',
        baseline='original x and labels copied bitwise; same frame/gap coverage',
        preprocessing=cfg,context=3000,audits=audits,smoke=a.smoke),indent=2))
    print('trace preparation complete',len(meta),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','data','fused','out'):p.add_argument(k,type=Path)
    p.add_argument('--smoke',action='store_true');main(p.parse_args())
