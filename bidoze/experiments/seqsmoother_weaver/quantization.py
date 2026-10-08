"""Round SleepAccel coordinates before interpolation; retain October 6 windows."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sleepaccel import peaks50


def subject(job):
    sid, data, original = job
    raw=np.loadtxt(data/'cleaned_accelerometer'/(sid+'_cleaned_motion.out'))
    rows=[]
    for step in [0., 1/4096, .001, .003, 1/256]:
        rounded=raw.copy()
        if step:
            rounded[:,1:]=np.round(rounded[:,1:]/step)*step
        prediction=peaks50(rounded,original.endpoint.to_numpy())
        e=(prediction.peak50_narrow.to_numpy()-original.hr_est.to_numpy())
        if step == 0 and np.max(np.abs(e)) > 1e-5:
            raise ValueError('Original peak finding did not reproduce for '+sid)
        for subset, mask in [('all',np.ones(len(original),bool)),('original_still_clear',original.still_clear.to_numpy()),
                             ('original_still_clear_in_band',(original.still_clear & original.watch_hr.between(54,78)).to_numpy())]:
            error=np.abs(prediction.peak50_narrow.to_numpy()[mask]-original.watch_hr.to_numpy()[mask])
            rows.extend(dict(participant=sid, coordinate_step=step,subset=subset,error=float(x)) for x in error)
    print('Quantization diagnostic:',sid,flush=True)
    return rows


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data',type=Path,required=True)
    a.add_argument('--old-epochs',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--workers',type=int,default=2)
    args=a.parse_args()
    old=pd.read_csv(args.old_epochs,dtype={'subject_id':str})
    old=old[old.cohort.eq('SleepAccel')].copy()
    old['endpoint']=45.+30.*old.groupby('subject_id').cumcount()
    old=old.dropna(subset=['hr_est','watch_hr'])
    jobs=[(sid,args.data,part) for sid,part in old.groupby('subject_id')]
    with ProcessPoolExecutor(args.workers) as pool:
        data=pd.DataFrame([row for result in pool.map(subject,jobs) for row in result])
    rows=[]
    for (step,subset),part in data.groupby(['coordinate_step','subset']):
        rows.append(dict(coordinate_step=step,subset=subset,participants=part.participant.nunique(),windows=len(part),
                         participant_macro_mae=part.groupby('participant').error.mean().mean(),pooled_mae=part.error.mean(),
                         pooled_median_abs_error=part.error.median()))
    data.groupby(['participant','coordinate_step','subset']).error.mean().to_csv(args.output/'quantization_participant_errors.csv')
    summary=pd.DataFrame(rows)
    summary.to_csv(args.output/'quantization_sensitivity.csv',index=False)
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
