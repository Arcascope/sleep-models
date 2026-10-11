"""Freeze conservative vessel folds and prepare release-compatible inputs and raw-axis probes."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from arcasleep.data.batching import recording_input, pad_pooled
from arcasleep.data.bundle import CacheBundle
from arcasleep.data.prepare import clean_accelerometer_rows
from arcasleep.predict import load_predictor
from arcasleep.ggir import anglez_bins

CONTEXT = 3000
GROUPS = {'boat01':'A','boat02':'A','boat03':'A','boat04':'B','boat05':'B',
          'boat06':'B','boat07':'C','boat08':'C','boat09':'D','boat10':'D','boat11':'D'}
PSG_MAP = {0:0,1:1,2:1,3:2,4:3,5:3}


def motion_features(a, n):
    """Label-free raw-axis moments on the fixed PSG clock; no floor estimated from sleep labels."""
    e = np.floor(a[:,0] / 30).astype(int)
    good = (e >= 0) & (e < n)
    e, xyz = e[good], a[good,1:]
    counts = np.bincount(e,minlength=n)
    means = np.stack([np.bincount(e,weights=xyz[:,i],minlength=n) for i in range(3)],axis=1)/np.maximum(counts[:,None],1)
    centered = xyz - means[e]
    cov = np.empty((n,3,3))
    for i in range(3):
        for j in range(3):
            cov[:,i,j] = np.bincount(e,weights=centered[:,i]*centered[:,j],minlength=n)/np.maximum(counts,1)
    eig = np.linalg.eigvalsh(cov)
    trace = eig.sum(axis=1)
    enmo = np.maximum(np.linalg.norm(xyz,axis=1)-1,0)
    enmo_max = np.zeros(n)
    np.maximum.at(enmo_max,e,enmo)
    enmo_max[counts==0] = np.nan
    dt = np.diff(a[good,0]); same = (np.diff(e)==0) & (dt>0) & (dt<0.2)
    diff = np.diff(xyz,axis=0)
    diff_power = np.bincount(e[1:][same],weights=np.sum(diff[same]**2,axis=1),minlength=n)/np.maximum(np.bincount(e[1:][same],minlength=n),1)
    lag = max(1,int(round(0.25/np.median(dt[dt>0]))))
    pair = (e[lag:]==e[:-lag]) & (a[good,0][lag:]-a[good,0][:-lag]<0.5)
    ac = np.bincount(e[lag:][pair],weights=np.sum(centered[lag:][pair]*centered[:-lag][pair],axis=1),minlength=n)/np.maximum(np.bincount(e[lag:][pair],minlength=n),1)
    return dict(axis_trace=trace,axis_concentration=eig[:,-1]/np.maximum(trace,1e-12),
                axis_acf_250ms=ac/np.maximum(trace,1e-12),diff_power=diff_power,
                enmo_max=enmo_max,sample_count=counts,still=enmo_max<0.1)


def main(a):
    out=a.out; (out/'nights').mkdir(parents=True,exist_ok=True); (out/'bundles').mkdir(exist_ok=True)
    for seed in range(3):
        dest=out/'bundles'/f'seed{seed}'
        if not dest.exists(): shutil.copytree(a.release/f'seed{seed}',dest)
    predictor=load_predictor(out/'bundles/seed0')
    cache=a.source/'bidoze/results/bidoze-adult-devices-0/setup/cache/matched/12Hz_log_PSD_aligned'
    split=json.loads((a.source/'build/data/nights_matched/splits/final.json').read_text())
    land_roles={tuple(r):role for role in ('fit','eval') for r in split[role]}
    records=[]
    for f in sorted(cache.glob('*_features.npz')):
        dataset,stem=f.name.removesuffix('_features.npz').split('_subject_',1)
        role=land_roles.get((dataset,stem))
        if role is None and dataset!='SleepAccel-Clinical': continue
        role=role or 'guard'; name=f'{dataset}_subject_{stem}'
        target=out/'nights'/f'{name}.npz'
        if not target.exists():
            b=CacheBundle.from_npz(f); y=b.y_for_n_classes(4)[0]; n=len(y)
            if n>CONTEXT: raise ValueError(f'{f}: {n} exceeds fixed context')
            x=recording_input(b.X[0],n,1)
            np.savez_compressed(target,x=pad_pooled(x,CONTEXT).astype(np.float32),
                y=np.pad(y,(0,CONTEXT-n),constant_values=-2),n=n,
                still=np.zeros(n,bool),anglez=b.anglez_5s[0] if b.anglez_5s is not None else np.full((n,6),np.nan))
        records.append(dict(name=name,cohort=dataset,participant=stem.split('_')[0],group='land',role=role))
    boats=pd.read_csv(a.source/'bidoze/results/boat-baseline-2026-10-09/boats.csv').set_index('participant').boat.to_dict()
    jobs=[]
    for cohort,pattern in [('SURF','*_GalaxyWatch.csv'),('Somerset','*.csv')]:
        for f in sorted((a.root/cohort/'cleaned_psg').glob(pattern)):
            if (a.root/cohort/'cleaned_accelerometer'/f.name).exists(): jobs.append((cohort,f))
    if a.limit: jobs=jobs[:a.limit]
    for cohort,f in jobs:
        stem=f.stem; participant=stem.split('_')[0]; group=GROUPS[boats[participant]] if cohort=='SURF' else 'E'
        name=f'{cohort}_subject_{stem}'; target=out/'nights'/f'{name}.npz'
        if not target.exists():
            raw=clean_accelerometer_rows(pd.read_csv(a.root/cohort/'cleaned_accelerometer'/f.name).to_numpy(float))
            frames,origin=predictor.features(raw,origin=0.0)
            if origin!=0: raise ValueError('Grid origin changed')
            psg=pd.read_csv(f); k=np.rint(psg.TIMESTAMP.to_numpy()/30).astype(int)
            if not np.allclose(k*30,psg.TIMESTAMP): raise ValueError(f'PSG off grid: {f}')
            n=max(len(frames)//15,int(k.max())+1)
            if n>CONTEXT: raise ValueError(f'{f}: {n} exceeds fixed context')
            x=recording_input(frames,len(frames)//15,1); x=pad_pooled(x,n)
            y=np.full(n,-1,np.int32); y[k]=[PSG_MAP.get(v,-1) for v in psg.PSG]
            truth=y.copy()
            missing=x[:,-1,0]>=1; y[missing]=-2
            features=motion_features(raw,n)
            bin_t,bin_a=anglez_bins(raw)
            grid=np.full(n*6,np.nan); bi=np.rint(bin_t/5).astype(int)
            keep=(bi>=0)&(bi<len(grid)); grid[bi[keep]]=bin_a[keep]
            angles=grid.reshape(n,6)
            np.savez_compressed(target,x=pad_pooled(x,CONTEXT).astype(np.float32),
                y=np.pad(y,(0,CONTEXT-n),constant_values=-2),truth=np.pad(truth,(0,CONTEXT-n),constant_values=-1),
                n=n,anglez=angles,**features)
            print(f'prepared {name}: {n} epochs',flush=True)
        records.append(dict(name=name,cohort=cohort,participant=participant,group=group,role='boat'))
    pd.DataFrame(records).to_csv(out/'inventory.csv',index=False)
    plans={}
    for outer in 'ABCDE':
        inner=next(g for g in 'ABCDE' if g!=outer)
        fit=[r['name'] for r in records if r['role']=='fit' or (r['role']=='boat' and r['group'] not in (outer,inner))]
        held=[r['name'] for r in records if r['role']=='eval' or (r['role']=='boat' and r['group']==inner)]
        test=[r['name'] for r in records if r['role']=='boat' and r['group']==outer]
        guard=[r['name'] for r in records if r['role']=='guard']
        assert not set(fit)&set(held) and not set(fit+held)&set(test)
        participants=[{(r['cohort'],r['participant']) for r in records if r['name'] in names} for names in (fit,held,test)]
        assert not participants[0]&participants[1] and not (participants[0]|participants[1])&participants[2]
        plans[outer]=dict(fit=fit,eval=held,test=test,guard=guard,inner_group=inner)
    (out/'splits.json').write_text(json.dumps(plans,indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(context=CONTEXT,records=len(records),boat_records=len(jobs),code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),groups=GROUPS),indent=2))
    print(f'{len(records)} nights; folds frozen',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path); p.add_argument('source',type=Path); p.add_argument('release',type=Path); p.add_argument('out',type=Path); p.add_argument('--limit',type=int)
    main(p.parse_args())
