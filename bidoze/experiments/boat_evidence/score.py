"""Score complete adaptation runs with frozen adult decoding, GGIR, and participant aggregation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score,roc_curve,cohen_kappa_score
from arcasleep_bidoze.adult import hybrid_calls
from arcasleep.ggir import evaluate_epoch_grid

DIVISOR=np.array([1,0.001989208747999981,0.004094626206317153,0.0027204850048531725])
METRICS=['sleep_sensitivity','wake_specificity','still_wake_specificity','moving_wake_specificity','kappa_sleep_wake','kappa_4stage','tst_error_minutes','waso_error_minutes','auroc','descriptive_spec95']


def metrics(y,p,call,still,missing,covered):
    good=y>=0
    if covered:good &= ~missing
    y,p,call,still=y[good],p[good],call[good],still[good]
    wake=y==0;sleep=y>0;predsleep=(call>0)&(call<4)
    sw=np.where(call==4,2,predsleep.astype(int))
    out=dict(epochs=len(y),sleep_sensitivity=np.mean(predsleep[sleep]) if sleep.any() else np.nan,
        wake_specificity=np.mean(call[wake]==0) if wake.any() else np.nan,
        still_wake_specificity=np.mean(call[wake&still]==0) if (wake&still).any() else np.nan,
        moving_wake_specificity=np.mean(call[wake&~still]==0) if (wake&~still).any() else np.nan,
        kappa_sleep_wake=cohen_kappa_score(sleep.astype(int),sw),kappa_4stage=cohen_kappa_score(y,call),
        tst_error_minutes=0.5*(predsleep.sum()-sleep.sum()))
    # WASO uses the PSG first/last sleep span as a diagnostic window, shared by both models.
    if sleep.any():
        lo,hi=np.flatnonzero(sleep)[[0,-1]];window=np.arange(len(y));inside=(window>=lo)&(window<=hi)
        out['waso_error_minutes']=0.5*(np.sum((call==0)&inside)-np.sum(wake&inside))
    else:out['waso_error_minutes']=np.nan
    valid=~missing[good]
    if len(np.unique(y[valid]>0))==2:
        ps=1-p[valid,0]/np.maximum(p[valid,:4].sum(axis=1),1e-12)
        fpr,tpr,_=roc_curve(y[valid]>0,ps);out.update(auroc=roc_auc_score(y[valid]>0,ps),descriptive_spec95=1-np.interp(.95,tpr,fpr))
    else:out.update(auroc=np.nan,descriptive_spec95=np.nan)
    return out


def main(a):
    meta=pd.read_csv(a.run/'setup/data/inventory.csv').set_index('name').to_dict('index');rows=[]
    for complete in sorted(a.run.glob('**/complete.json')):
        folder=complete.parent
        if 'checks' in folder.parts:continue
        config=json.loads((folder/'config.json').read_text());split=json.loads((folder/'split.json').read_text())
        for f in sorted((folder/'predictions').glob('*.npz')):
            r=meta[f.stem];d=np.load(f);y=d['y'];missing=d['missing'];still=d['still'];role=str(d['role'])
            if role not in ('test','guard'):continue
            ggir=evaluate_epoch_grid(d['anglez']); n=len(y)
            for model,key in [('adapted','probabilities'),('initial','reference')]:
                proba=d[key];raw=hybrid_calls(proba,proba,DIVISOR);raw[missing]=4
                filtered=raw.copy();demote=(raw>0)&(raw<4)&~ggir.is_sib&~ggir.is_spt;filtered[demote]=0
                for mode,call in [('raw',raw),('ggir',filtered)]:
                    for covered in (False,True):
                        rows.append(dict(run=str(folder.relative_to(a.run)),arm=config['arm'],fold=config['fold'],seed=config['seed'],name=f.stem,cohort=r['cohort'],participant=r['participant'],group=r['group'],role=role,model=model,mode=mode,covered=covered)|metrics(y,proba,call,still,missing,covered))
            d.close()
    a.out.mkdir(parents=True,exist_ok=True)
    if not rows:return
    nights=pd.DataFrame(rows);nights.to_csv(a.out/'nights.csv',index=False)
    keys=['run','arm','fold','seed','cohort','participant','group','role','model','mode','covered']
    subjects=nights.groupby(keys)[METRICS].mean().reset_index();subjects.to_csv(a.out/'subjects.csv',index=False)
    summary=subjects.groupby([k for k in keys if k not in ('participant','group')])[METRICS].mean();summary.to_csv(a.out/'summary.csv')
    paired=[];rng=np.random.default_rng(11)
    for keys2,g in subjects.groupby(['run','cohort','role','mode','covered']):
        for metric in METRICS:
            wide=g.pivot(index='participant',columns='model',values=metric).dropna()
            if len(wide)==0:continue
            delta=(wide.adapted-wide.initial).to_numpy();ci=np.percentile(delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(axis=1),[2.5,97.5])
            paired.append(dict(zip(['run','cohort','role','mode','covered'],keys2))|dict(metric=metric,participants=len(delta),difference=float(delta.mean()),ci_low=ci[0],ci_high=ci[1],p=wilcoxon(delta).pvalue if np.any(delta!=0) else 1))
    pd.DataFrame(paired).to_csv(a.out/'paired.csv',index=False)
    (a.out/'protocol.json').write_text(json.dumps(dict(stage_divisor=DIVISOR.tolist(),aggregation='recording within participant then participant macro',ci='participant bootstrap; vessel-cluster confirmation still required',threshold='descriptive_spec95 interpolates each test night ROC; not deployable',comparison='same seed initial vs adapted; not the 3-seed released ensemble',waso='diagnostic uses PSG-defined first-to-last sleep window'),indent=2))
    print(f'{len(rows)} scored recording/model/mode rows',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('out',type=Path);main(p.parse_args())
