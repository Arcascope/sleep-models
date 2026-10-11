"""Small label-blind shared-motion test with held-out time blocks and displaced/unrelated controls."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def seconds(root,cohort,stem):
    a=pd.read_csv(root/cohort/'cleaned_accelerometer'/f'{stem}.csv').to_numpy(float)
    a=a[np.all(np.isfinite(a),axis=1)];a=a[np.argsort(a[:,0],kind='stable')];a=a[np.r_[True,np.diff(a[:,0])>0]]
    i=np.floor(a[:,0]).astype(int); n=int(i.max())+1
    mag=np.linalg.norm(a[:,1:],axis=1);count=np.bincount(i,minlength=n)
    mean=np.bincount(i,weights=mag,minlength=n)/np.maximum(count,1)
    variance=np.maximum(np.bincount(i,weights=mag**2,minlength=n)/np.maximum(count,1)-mean**2,0)
    mean[count==0]=np.nan
    return mean,variance,float(stem.split('_')[1])


def main(a):
    meta=pd.read_csv(a.data/'inventory.csv');meta=meta[meta.cohort=='SURF']
    boats=pd.read_csv(a.boats).set_index('participant'); rows=[]; epochs=[]; cached={}; used=set()
    def load(stem):
        if stem not in cached: cached[stem]=seconds(a.root,'SURF',stem)
        return cached[stem]
    for r in meta.to_dict('records'):
        if len(used)>=a.max_pairs:break
        mates=str(boats.loc[r['participant'],'boat_mates']).split()
        if mates==['nan'] or not mates:continue
        stem=r['name'].split('_subject_',1)[1]
        t,tvar,t0=load(stem); best=None
        for ref in meta[meta.participant.isin(mates)].to_dict('records'):
            rst=ref['name'].split('_subject_',1)[1];start=float(rst.split('_')[1]);length=max(0,min(t0+len(t),start+30000)-max(t0,start))
            if best is None or length>best[0]:best=(length,rst)
        if best is None or best[0]<7200:continue
        rst=best[1];pair=tuple(sorted([stem,rst]))
        if pair in used:continue
        ref,rv,r0=load(rst);shift=int(round(t0-r0)); ti=np.arange(len(t));ri=ti+shift
        common=(ri>=0)&(ri<len(ref));ti=ti[common];ri=ri[common]
        valid=np.isfinite(t[ti])&np.isfinite(ref[ri]);ti=ti[valid];ri=ri[valid]
        if len(ti)<3600:continue
        wrong=meta[~meta.participant.isin(mates+[r['participant']])].iloc[0]['name'].split('_subject_',1)[1]
        w,_,_=load(wrong)
        controls={'mate':ref[ri],'displaced_mate':ref[(ri+1800)%len(ref)],'unrelated':w[np.arange(len(ti))%len(w)]}
        midpoint=np.median(ti);fit_time=ti<midpoint;test_time=~fit_time
        for control,x in controls.items():
            # Fit on observed low-movement seconds, selected without PSG, in the first block only.
            fit=fit_time & np.isfinite(x) & (tvar[ti]<0.0004)
            test=test_time & np.isfinite(x)
            if fit.sum()<300 or test.sum()<300:continue
            design=np.column_stack([np.ones(fit.sum()),x[fit]])
            beta=np.linalg.lstsq(design,t[ti[fit]],rcond=None)[0]
            residual=t[ti[test]]-(beta[0]+beta[1]*x[test]); target=t[ti[test]]
            rows.append(dict(target=stem,reference=rst,control=control,fit_seconds=int(fit.sum()),test_seconds=int(test.sum()),slope=beta[1],test_variance_ratio=np.var(residual)/max(np.var(target),1e-12),test_correlation=np.corrcoef(target,x[test])[0,1]))
            ep=(ti[test]//30).astype(int)
            for e in np.unique(ep):
                take=ep==e
                if take.sum()<20:continue
                epochs.append(dict(name=r['name'],participant=r['participant'],group=r['group'],epoch=int(e),control=control,target_sd=float(np.std(target[take])),residual_sd=float(np.std(residual[take])),seconds=int(take.sum())))
        used.add(pair);print(f'{stem}/{rst}: {len(ti)} shared seconds',flush=True)
    a.out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.out/'common_motion_pairs.csv',index=False);pd.DataFrame(epochs).to_csv(a.out/'common_motion_epochs.csv.gz',index=False)
    print(f'{len(used)} diagnostic pairs; reduction of variance alone is not a wake result',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('data',type=Path);p.add_argument('boats',type=Path);p.add_argument('out',type=Path);p.add_argument('--max-pairs',type=int,default=8);main(p.parse_args())
