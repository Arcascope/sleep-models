"""Measure HR availability on each PSG clock; nearest measurements within 150 s, without interpolation."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main(a):
    sources=pd.read_csv(a.run/'diagnostics/physiology_inventory.csv');meta=pd.read_csv(a.run/'setup/data/inventory.csv');rows=[];epochs=[]
    for src in sources.to_dict('records'):
        p=a.root/src['cohort']/src['file']
        if p.suffix!='.csv':continue
        data=pd.read_csv(p,sep=None,engine='python')
        if not {'timestamp','bpm'}.issubset(data.columns):continue
        participant=next((v for v in p.parts if v.startswith('SURF') and '_OuraData' in v),'').split('_')[0]
        stamp=pd.to_datetime(data.timestamp,utc=True).to_numpy(dtype='datetime64[ns]').astype(np.int64)/1e9;hr=pd.to_numeric(data.bpm,errors='coerce').to_numpy();valid=np.isfinite(stamp)&np.isfinite(hr)&(hr>=30)&(hr<=220)
        stamp,hr=stamp[valid],hr[valid];order=np.argsort(stamp);stamp,hr=stamp[order],hr[order]
        for rec in meta[(meta.cohort==src['cohort'])&(meta.participant==participant)].to_dict('records'):
            stem=rec['name'].split('_subject_',1)[1];t0=float(stem.split('_')[1]);d=np.load(a.run/'setup/data/nights'/f"{rec['name']}.npz");n=int(d['n']);clock=t0+np.arange(n)*30+15
            right=np.minimum(np.searchsorted(stamp,clock),len(stamp)-1);left=np.maximum(right-1,0);ix=np.where(np.abs(stamp[left]-clock)<np.abs(stamp[right]-clock),left,right);distance=np.abs(stamp[ix]-clock);available=distance<=150
            truth=d['truth'][:n];covered=d['x'][:n,-1,0]<1;scored=truth>=0;stillwake=(truth==0)&d['still'][:n]&covered
            rows.append(dict(name=rec['name'],participant=participant,source='Oura measured HR',psg_epochs=int(scored.sum()),hr_coverage=float(available[scored].mean()),still_wake_epochs=int(stillwake.sum()),still_wake_hr_coverage=float(available[stillwake].mean()) if stillwake.any() else np.nan,tolerance_seconds=150))
            epochs.append(pd.DataFrame(dict(name=rec['name'],epoch=np.arange(n),hr=np.where(available,hr[ix],np.nan),nearest_measurement_seconds=distance,available=available)))
            d.close()
    out=a.run/'diagnostics';pd.DataFrame(rows).to_csv(out/'physiology_coverage.csv',index=False)
    if epochs:pd.concat(epochs,ignore_index=True).to_csv(out/'physiology_epochs.csv.gz',index=False)
    print(f'{len(rows)} matched nights; {len(set(r["participant"] for r in rows))} participants; coverage only, alignment remains unverified',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('run',type=Path);main(p.parse_args())
