"""Describe retained axis/model PSDs on a shared clock; no fitted floor or threshold."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pisces_lite.proc import ProcessingConfig
from arcasleep.data.prepare import clean_accelerometer_rows
from boat_evidence.prepare import motion_features, PSG_MAP
from boat_observation.sampling import CHANNELS, BANDS, epoch_power


def prepare_land(root, data, out):
    config=json.loads((data/'bundles/seed0/preprocessing.json').read_text())
    proc=ProcessingConfig.from_dict(dict(config,spectral_channels=CHANNELS))
    meta=pd.read_csv(data/'inventory.csv');meta=meta[meta.cohort=='SleepAccel-Clinical']
    for r in meta.to_dict('records'):
        dest=out/'land'/r['name'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():continue
        stem=r['name'].split('_subject_',1)[1]
        psg=pd.read_csv(root/r['cohort']/'cleaned_psg'/f'{stem}.csv')
        origin=float(psg.TIMESTAMP.min()); k=np.rint((psg.TIMESTAMP-origin)/30).astype(int)
        assert np.allclose(psg.TIMESTAMP-origin,30*k)
        n=int(k.max())+1;truth=np.full(n,-1);truth[k]=[PSG_MAP.get(v,-1) for v in psg.PSG]
        raw=clean_accelerometer_rows(pd.read_csv(root/r['cohort']/'cleaned_accelerometer'/f'{stem}.csv').to_numpy(float))
        raw=raw[(raw[:,0]>=origin)&(raw[:,0]<origin+30*n)].copy();raw[:,0]-=origin
        frames=proc.apply(raw,normalize=False,padding=False,origin=0.)
        power,count=epoch_power(frames,n);motion=motion_features(raw,n)
        np.savez_compressed(dest/'native_axis_psd.npz',epoch_linear_psd=power.astype(np.float32),
            frame_count=count,frequencies=np.arange(1,61)/10,channels=np.array(CHANNELS),
            truth=truth,still=motion['still'],gap=count==0)
        (dest/'complete.json').write_text(json.dumps(dict(cohort=r['cohort'],participant=r['participant'],
            grid_origin='first source PSG timestamp; no optimized shift',epochs=n,
            sampling_hz=float(1/np.median(np.diff(raw[:,0]))))))
        print('land PSD',r['name'],flush=True)


def describe(out,data):
    meta=pd.read_csv(data/'inventory.csv').set_index('name').to_dict('index');rows=[];spectra=[]
    for path in sorted(out.glob('*/**/*_axis_psd.npz')):
        if 'checks' in path.parts:continue
        name=path.parent.name
        if name not in meta:continue
        r=meta[name];variant=path.name.removesuffix('_axis_psd.npz')
        d=np.load(path);p=d['epoch_linear_psd'];freq=d['frequencies'];truth=d['truth'];still=d['still'];covered=~d['gap']
        # Sampling-arm metrics use common coverage; use it for spectral comparisons too.
        pred=path.parent/'probabilities.npz'
        if pred.exists():
            with np.load(pred) as v:covered &= ~v['common_missing']
        p=np.concatenate([p,p[:,:,:3].sum(axis=2,keepdims=True)],axis=2)
        groups=dict(all_covered=covered,still_wake=covered&still&(truth==0),still_sleep=covered&still&(truth>0))
        for label,take in groups.items():
            if not take.any():continue
            for c,channel in enumerate(CHANNELS+['trace']):
                median=np.nanmedian(p[take,:,c],axis=0)
                for lo,hi in BANDS:
                    band=(freq>=lo)&(freq<=hi);values=np.nanmean(p[take,:,c][:,band],axis=1)
                    rows.append(dict(name=name,cohort=r['cohort'],participant=r['participant'],variant=variant,
                        label=label,channel=channel,band=f'{lo:g}-{hi:g}',epochs=int(take.sum()),
                        median_epoch_band_db=float(10*np.log10(max(np.nanmedian(values),1e-30))),
                        p10_epoch_band_db=float(10*np.log10(max(np.nanquantile(values,.1),1e-30)))))
                for f,v in zip(freq,median):
                    spectra.append(dict(name=name,cohort=r['cohort'],participant=r['participant'],variant=variant,
                        label=label,channel=channel,frequency_hz=f,median_epoch_power_db=10*np.log10(max(v,1e-30))))
        d.close()
    a=out/'analysis';a.mkdir(exist_ok=True)
    table=pd.DataFrame(rows);table.to_csv(a/'power_nights.csv',index=False)
    if len(table):
        keys=['cohort','participant','variant','label','channel','band']
        subjects=table.groupby(keys).median_epoch_band_db.mean().reset_index()
        subjects.to_csv(a/'power_subjects.csv',index=False)
        subjects.groupby([k for k in keys if k!='participant']).median_epoch_band_db.mean().to_csv(a/'power_summary.csv')
    pd.DataFrame(spectra).to_csv(a/'per_bin_nights.csv.gz',index=False)
    (a/'power_protocol.json').write_text(json.dumps(dict(statistic='linear PSD averaged over valid frames within epoch; median epochs within label stratum; dB; average recordings then participants',
        labels='PSG labels for descriptive contrast only, never filter/floor estimation',
        interpretation='observed levels and lower percentiles, not a measured sensor-only noise floor',
        land='32 Clinical reference recordings, native only; source PSG clock; no model inference'),indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('root',type=Path);a.add_argument('data',type=Path);a.add_argument('out',type=Path)
    a.add_argument('--prepare-land',action='store_true');p=a.parse_args()
    if p.prepare_land:prepare_land(p.root,p.data,p.out)
    describe(p.out,p.data)
