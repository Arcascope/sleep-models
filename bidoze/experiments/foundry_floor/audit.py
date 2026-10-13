"""Audit Foundry mean-axis floors against the raw-timestamp, summed-axis device statistic.

No PSG outcomes: Foundry PSG filenames contain synthetic GGIR labels.
All paths are supplied by CLI; source CSVs are read only.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
import senpy

def fingerprint(path):
    s=path.stat();return {"size":s.st_size,"mtime_ns":s.st_mtime_ns}

def legacy(t,xyz,epochs):
    fs=50.0;tu=np.arange(t[0],t[-1],1/fs)
    uni=np.column_stack([np.interp(tu,t,xyz[:,k]) for k in range(3)])
    rows=[]
    for epoch in epochs:
        rel=epoch-tu[0]
        if rel<0 or rel+30>tu[-1]-tu[0]:continue
        levels=[];rms=[];stds=[]
        for k in range(3):
            lo=int((rel+k*10)*fs);segment=uni[lo:lo+500]
            if len(segment)!=500:continue
            std=np.std(np.linalg.norm(segment,axis=1))*1000
            if std>200 or std<=1e-9:continue
            f,power=welch(segment,fs=fs,nperseg=500,noverlap=0,detrend="linear",axis=0)
            mean_axis=power.mean(axis=1);trace=power.sum(axis=1)
            keep=(f>=2)&(f<=6)
            assert np.allclose(trace,3*mean_axis)
            levels.append(10*np.log10(np.mean(mean_axis[keep])))
            rms.append(np.sqrt(np.trapezoid(mean_axis[keep],f[keep]))*1000)
            stds.append(std)
        if not levels:continue
        i=int(np.argmin(levels))
        rows.append(dict(epoch=epoch,legacy_mean_axis_db=levels[i],legacy_mean_axis_rms_mg=rms[i],
            corrected_sum_mean_bin_db=levels[i]+10*np.log10(3),
            corrected_sum_rms_mg=rms[i]*np.sqrt(3),selected_mag_std_mg=stds[i]))
    return pd.DataFrame(rows)

def exact(t,xyz,epochs):
    # Same transform/2-s frame step and statistic as boat_noise/devices.py.
    relative=t-t[0]
    total=None
    for k in range(3):
        spec=senpy.compute_nufft_spectrogram(relative,xyz[:,k],10,8,kind="psd",
            empty_windows="keep",origin_s=0.0)
        total=spec.Sxx if total is None else total+spec.Sxx
    centres=spec.times+(spec.origin_s or 0)+t[0]
    f=spec.frequencies;keep=(f>=2)&(f<=6)
    db=10*np.log10(total+1e-14)
    level=np.median(db[:,keep],axis=1)
    trace_rms=np.sqrt(np.trapezoid(total[:,keep],f[keep],axis=1))*1000
    rows=[]
    for epoch in epochs:
        inside=np.flatnonzero((centres>=epoch)&(centres<epoch+30))
        if len(inside)<14 or not np.isfinite(level[inside]).all():continue
        i=inside[np.argmin(level[inside])]
        rows.append(dict(epoch=epoch,nufft_sum_median_bin_db=level[i],
            nufft_sum_rms_at_quietest_mg=trace_rms[i],frames=len(inside)))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--legacy-summary",type=Path,required=True);p.add_argument("--limit-per-device",type=int)
    p.add_argument("--legacy-only",action="store_true")
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    original=pd.read_csv(a.legacy_summary).set_index("window")
    rows=[]
    for device in ("Eric-Galaxy","Eric-iPhone"):
        files=sorted(p for p in (a.root/"cleaned_accelerometer").glob(device+"-*_accelerometer.csv") if p.name in original.index)
        if a.limit_per_device:files=files[:a.limit_per_device]
        for path in files:
            dest=a.out/"windows"/path.stem;dest.mkdir(parents=True,exist_ok=True)
            marker=dest/"complete.json"
            if marker.exists():
                rows.append(json.loads(marker.read_text()));continue
            psg=a.root/"cleaned_psg"/path.name.replace("_accelerometer.csv","_psg.csv")
            before={str(x):fingerprint(x) for x in (path,psg)}
            frame=pd.read_csv(path,usecols=["timestampMs","x","y","z"])
            t=frame.timestampMs.to_numpy(float)/1000;xyz=frame[["x","y","z"]].to_numpy(float)
            assert np.isfinite(t).all() and np.isfinite(xyz).all()
            magnitude=np.linalg.norm(xyz,axis=1);gravity=float(np.median(magnitude))
            assert .8<gravity<1.2, (device,gravity)
            labels=pd.read_csv(psg);epochs=labels.query("PSG == 1").TIMESTAMP.to_numpy(float)
            audit=legacy(t,xyz,epochs);assert len(audit)>=50
            record=dict(device=device,window=path.name,median_magnitude_g=gravity,
                median_fs=float(1/np.median(np.diff(t))),legacy_epochs=len(audit),
                source_backward_steps=int((np.diff(t)<0).sum()),source_repeated_steps=int((np.diff(t)==0).sum()))
            for column in ("legacy_mean_axis_db","legacy_mean_axis_rms_mg",
                           "corrected_sum_mean_bin_db","corrected_sum_rms_mg"):
                record[column+"_q10"]=float(audit[column].quantile(.1))
            record["selected_frames_below_3mg_fraction"]=float((audit.selected_mag_std_mg<3).mean())
            prior=original.loc[path.name]
            assert abs(record["legacy_mean_axis_db_q10"]-prior.floor_db_q10)<1e-7
            assert abs(record["legacy_mean_axis_rms_mg_q10"]-prior.rms_mg_q10)<1e-7
            if not a.legacy_only:
                order=np.argsort(t,kind="stable");clean_t=t[order];clean_xyz=xyz[order]
                unique=np.r_[True,np.diff(clean_t)>0]
                record["matched_duplicate_rows_removed"]=int((~unique).sum())
                matched=exact(clean_t[unique],clean_xyz[unique],epochs);assert len(matched)>=50
                matched.to_csv(dest/"matched_epochs.csv.gz",index=False)
                record.update(nufft_epochs=len(matched),
                    nufft_sum_median_bin_db_q10=float(matched.nufft_sum_median_bin_db.quantile(.1)),
                    nufft_sum_rms_at_quietest_mg_q10=float(matched.nufft_sum_rms_at_quietest_mg.quantile(.1)))
            audit.to_csv(dest/"legacy_audit_epochs.csv.gz",index=False)
            assert before=={str(x):fingerprint(x) for x in (path,psg)}
            record["inputs_unchanged"]=True
            marker.write_text(json.dumps(record,indent=2)+"\n")
            rows.append(record);pd.DataFrame(rows).to_csv(a.out/"summary.csv",index=False)
            print("AUDITED",device,path.name,record,flush=True)
    table=pd.DataFrame(rows);table.to_csv(a.out/"summary.csv",index=False)
    values=[c for c in table if c.endswith("_q10") or c=="selected_frames_below_3mg_fraction"]
    table.groupby("device")[values].agg(["median","min","max"]).to_csv(a.out/"device_summary.csv")
    (a.out/"protocol.json").write_text(json.dumps(dict(reference="foundry floor PR10 d2e4672",
        units="cleaned g, stationary magnitude empirically checked",labels="GGIR SPT; not PSG",
        legacy="50 Hz interpolation, 3 nonoverlapping 10 s frames per epoch, linear detrend, mean axes and mean bins",
        corrected_legacy="sum axes; otherwise identical legacy calculation",
        matched="senpy raw-timestamp NUFFT; 10 s windows, 2 s step, mean detrend; sum axes, median 2-6 Hz bins; min frame, q10 epoch",
        matched_cleaning="sort and drop duplicate timestamps as boat_noise/survey.load_accel; legacy retains original row order",
        same_night_comparison=False,stage_outcomes=False,source_files_modified=False),indent=2)+"\n")
    (a.out/"complete.json").write_text(json.dumps(dict(complete=True,windows=len(table),
        devices=table.device.value_counts().to_dict(),finished=time.time()))+"\n")
if __name__=="__main__":main()
