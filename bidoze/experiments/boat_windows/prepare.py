"""Reuse exact native frames; isolate epoch-contained overlapping and nonoverlapping windows."""
import argparse,csv,hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from pisces_lite.proc import ProcessingConfig
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE as PAD
from arcasleep.data.batching import recording_input,pad_pooled
from arcasleep.data.prepare import clean_accelerometer_rows

LAYOUTS={'native':list(range(15)),'contained':list(range(11)),'nonoverlap':[0,5,10]}

def select(frames,n,layout):
    padded=np.full((n*15,*frames.shape[1:]),PAD,dtype=np.float32)
    take=min(len(frames),n*15);padded[:take]=frames[:take]
    return padded.reshape(n,15,*frames.shape[1:])[:,LAYOUTS[layout]]

def main(a):
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'nights').mkdir(exist_ok=True)
    for f in ('splits.json','inventory.csv'):shutil.copy(a.data/f,a.out/f)
    if not (a.out/'bundles').exists():shutil.copytree(a.data/'bundles',a.out/'bundles')
    cfg=json.loads((a.data/'bundles/seed0/preprocessing.json').read_text());assert cfg['window_seconds']==10 and cfg['window_step_seconds']==2
    proc=ProcessingConfig.from_dict(cfg);plans=json.loads((a.out/'splits.json').read_text());meta=list(csv.DictReader((a.out/'inventory.csv').open()))
    if a.smoke:
        d=plans[a.fold];names=[d['fit'][0],d['fit'][-1],d['eval'][0],d['eval'][-1],d['test'][0],d['guard'][0]];meta=[r for r in meta if r['name'] in names]
    audits=[]
    for row in meta:
        dest=a.out/'nights'/f"{row['name']}.npz"
        with np.load(a.data/'nights'/dest.name) as z:fields={k:z[k].copy() for k in ('x','y','n','still','anglez','truth') if k in z}
        n=int(fields['n']);cache=a.cache/f"{row['name']}_features.npz"
        if cache.exists():
            with np.load(cache) as z:frames=z['X'][0].astype(np.float32)
            provenance='original cached native frames'
        else:
            stem=row['name'].split('_subject_',1)[1]
            raw=clean_accelerometer_rows(pd.read_csv(a.root/row['cohort']/'cleaned_accelerometer'/f'{stem}.csv').to_numpy(float))
            frames=proc.apply(raw,normalize=False,padding=False,origin=0.).astype(np.float32)
            provenance='regenerated native frames; unchanged origin and raw data'
        # The frozen preparation discards incomplete trailing groups of 15.
        # Preserve that night-end rule before applying the window-layout contrast.
        frames=frames[:(len(frames)//15)*15]
        original_missing=fields['x'][:n,-1,0]>=1;common_missing=original_missing.copy();maxerr=0.;mederr=0.
        for layout in LAYOUTS:
            chosen=select(frames,n,layout);pooled=recording_input(chosen.reshape(-1,*frames.shape[1:]),n,1)
            missing=pooled[:,-1,0]>=1;common_missing|=missing
            fields[f'x_{layout}']=pad_pooled(pooled,3000).astype(np.float32)
            if layout=='native':
                assert np.array_equal(missing,original_missing),f"Native coverage changed: {row['name']}"
                diff=np.abs(pooled[~missing,:-1]-fields['x'][:n][~missing,:-1]);maxerr=float(diff.max());mederr=float(np.median(diff))
                assert mederr<1e-3,'Native input regeneration changed typical power'
        fields['frames']=select(frames,n,'native').reshape(-1,*frames.shape[1:]);fields['common_missing']=common_missing
        fields['y'][:n][common_missing]=-2
        assert np.isfinite(fields['frames']).all()
        np.savez_compressed(dest,**fields)
        audits.append(dict(name=row['name'],n=n,source=provenance,native_max_abs_error=maxerr,native_median_abs_error=mederr,original_missing=int(original_missing.sum()),common_missing=int(common_missing.sum())))
        print('frames prepared',row['name'],n,'max error',maxerr,flush=True)
    recipe=dict(layouts=LAYOUTS,starts_seconds={k:[2*i for i in v] for k,v in LAYOUTS.items()},window_seconds=10,cnn_slots=15,cnn_features=64,structural_slots='padding sentinel; excluded from physical missing fraction',coverage='same union of unavailable epochs across all layouts and frozen reference',context=3000,preprocessing=cfg,audits=audits,smoke=a.smoke,code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (a.out/'window-input-recipe.json').write_text(json.dumps(recipe,indent=2));(a.out/'complete.json').write_text(json.dumps(dict(recordings=len(meta),smoke=a.smoke)))
    print('preparation complete',len(meta),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','data','cache','out'):p.add_argument(k,type=Path)
    p.add_argument('--smoke',action='store_true');p.add_argument('--fold',default='D');main(p.parse_args())
