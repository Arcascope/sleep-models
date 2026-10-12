"""Score common-coverage window comparisons and transition/stable epoch strata."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from boat_axes.score_training import main as score_all,metrics,METRICS,DIVISOR
from arcasleep_bidoze.adult import hybrid_calls
from arcasleep.ggir import evaluate_epoch_grid

def main(a):
    score_all(a)
    summary=pd.read_csv(a.out/'summary.csv');subjects=pd.read_csv(a.out/'subjects.csv');checks={};matched=[];interactions=[];strata=[]
    for pooling in ('mean','cnn'):
        configs=list((a.run/'training').glob(f'*_{pooling}/{a.fold}/seed0/config.json'))
        if configs:
            hashes={json.loads(p.read_text())['initial_parameter_sha256'] for p in configs}
            assert len(hashes)==1,f'{pooling} arms have different initialization'
            checks[pooling]=dict(initial_hash=next(iter(hashes)),arms=[p.parent.parent.parent.name for p in configs])
    for (cohort,role,mode,covered,metric),g in subjects.melt(id_vars=[c for c in subjects.columns if c not in METRICS],value_vars=METRICS,var_name='metric').query('model=="adapted"').groupby(['cohort','role','mode','covered','metric']):
        for lhs,rhs in [('contained_mean','native_mean'),('nonoverlap_mean','native_mean'),('nonoverlap_mean','contained_mean'),('contained_cnn','native_cnn'),('nonoverlap_cnn','native_cnn'),('nonoverlap_cnn','contained_cnn'),('native_cnn','native_mean'),('contained_cnn','contained_mean'),('nonoverlap_cnn','nonoverlap_mean')]:
            pair=g[g.arm.isin([lhs,rhs])].pivot(index='participant',columns='arm',values='value').dropna()
            if {lhs,rhs}.issubset(pair.columns):matched.append(dict(cohort=cohort,role=role,mode=mode,covered=covered,metric=metric,lhs=lhs,rhs=rhs,participants=len(pair),difference=float((pair[lhs]-pair[rhs]).mean())))
    for (cohort,role,mode,covered),g in subjects.query('model=="adapted"').groupby(['cohort','role','mode','covered']):
        for layout in ('contained','nonoverlap'):
            needed={f'{layout}_cnn',f'{layout}_mean','native_cnn','native_mean'}
            if not needed.issubset(set(g.arm)):continue
            for metric in METRICS:
                wide=g[g.arm.isin(needed)].pivot(index='participant',columns='arm',values=metric).dropna()
                delta=(wide[f'{layout}_cnn']-wide.native_cnn)-(wide[f'{layout}_mean']-wide.native_mean)
                interactions.append(dict(cohort=cohort,role=role,mode=mode,covered=covered,layout=layout,metric=metric,participants=len(wide),difference=float(delta.mean())))
    pd.DataFrame(interactions).to_csv(a.out/'window_interactions.csv',index=False)
    meta=pd.read_csv(a.run/'setup/data/inventory.csv').set_index('name').to_dict('index')
    for complete in (a.run/'training').glob('**/complete.json'):
        folder=complete.parent;cfg=json.loads((folder/'config.json').read_text())
        for file in (folder/'predictions').glob('*.npz'):
            with np.load(file) as z:
                if str(z['role']) not in ('test','guard'):continue
                y=z['y'];missing=z['missing'];proba=z['probabilities'];calls=hybrid_calls(proba[:,:4],proba[:,:4],DIVISOR);calls[missing]=4
                ggir=evaluate_epoch_grid(z['anglez']);calls[(calls>0)&(calls<4)&~ggir.is_sib&~ggir.is_spt]=0
                change=(y[:-1]>=0)&(y[1:]>=0)&((y[:-1]>0)!=(y[1:]>0))
                near=np.r_[False,change]|np.r_[change,False]
                for name,take in [('sleep_wake_boundary_adjacent',near),('stable',~near)]:
                    truth=y.copy();truth[~take]=-1
                    vals=metrics(truth,proba,calls,z['still'],missing,True)
                    strata.append(dict(arm=cfg['arm'],cohort=meta[file.stem]['cohort'],participant=meta[file.stem]['participant'],name=file.stem,stratum=name)|vals)
    pd.DataFrame(matched).to_csv(a.out/'window_matched.csv',index=False)
    if strata:
        nights=pd.DataFrame(strata);nights.to_csv(a.out/'strata_nights.csv',index=False)
        subjects2=nights.groupby(['arm','cohort','participant','stratum'])[[k for k in METRICS if k not in ('tst_error_minutes','waso_error_minutes')]].mean().reset_index()
        subjects2.to_csv(a.out/'strata_subjects.csv',index=False);subjects2.groupby(['arm','cohort','stratum'])[[k for k in METRICS if k not in ('tst_error_minutes','waso_error_minutes')]].mean().to_csv(a.out/'strata_summary.csv')
    (a.out/'initialization_control.json').write_text(json.dumps(checks,indent=2))
    (a.out/'window_protocol.json').write_text(json.dumps(dict(coverage='common missing mask across all three layouts and frozen reference',boundary='epoch immediately before or after a scored sleep/wake transition; PSG diagnostic only, never model input',comparison='matched scratch arms; reference is frozen seed0, not scratch epoch0',cnn='fixed 15 slots and 64 features, same initial parameters across layouts; excluded slots masked; gap divisor is selected windows',uncertainty='single seed discovery; no general gain or noninferiority claim'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('out',type=Path);p.add_argument('--fold',default='D');main(p.parse_args())
