"""Fixed vessel-held-out logistic probes of raw-axis information beyond released wake scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

FEATURES=['axis_trace','axis_concentration','axis_acf_250ms','diff_power']


def main(a):
    meta=pd.read_csv(a.data/'inventory.csv');meta=meta[meta.role=='boat'];rows=[]
    for r in meta.to_dict('records'):
        stem=r['name'].split('_subject_',1)[1]
        pred=a.baseline/'predictions/bidoze-adult'/f'{stem}.csv'
        if not pred.exists():raise FileNotFoundError(pred)
        p=pd.read_csv(pred); d=np.load(a.data/'nights'/f"{r['name']}.npz");n=int(d['n'])
        epochs=np.rint(p.start_s.to_numpy()/30).astype(int); keep=(epochs>=0)&(epochs<n)
        wake=np.full(n,np.nan);wake[epochs[keep]]=p.wake.to_numpy()[keep]
        part=pd.DataFrame({k:d[k][:n] for k in FEATURES});part['axis_trace']=np.log(np.maximum(part.axis_trace,1e-12));part['diff_power']=np.log(np.maximum(part.diff_power,1e-12))
        part['wake_logit']=np.log(np.clip(wake,1e-6,1-1e-6)/(1-np.clip(wake,1e-6,1-1e-6)))
        part['y']=d['truth'][:n];part['still']=d['still'][:n];part['missing']=d['x'][:n,-1,0]>=1
        for k in ('cohort','participant','group','name'):part[k]=r[k]
        rows.append(part);d.close()
    table=pd.concat(rows,ignore_index=True);table=table[(table.y>=0)&~table.missing].dropna(subset=FEATURES+['wake_logit']);table['wake']=(table.y==0).astype(int)
    outputs=[]
    for outer in 'ABCDE':
        inner=next(g for g in 'ABCDE' if g!=outer);fit=table[~table.group.isin([outer,inner])];held=table[table.group==inner];test=table[table.group==outer]
        for arm,features in [('score',['wake_logit']),('score_raw',['wake_logit']+FEATURES)]:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=500,class_weight='balanced',random_state=0))
            model.fit(fit[features],fit.wake)
            ps=model.predict_proba(held[features])[:,1]
            # Threshold selected on inner-eval sleep only; no outer-test label enters this decision.
            threshold=float(np.quantile(ps[held.wake==0],.95))
            got=test[['name','cohort','participant','group','y','still']].copy();got['p_wake']=model.predict_proba(test[features])[:,1];got['call_wake']=got.p_wake>threshold;got['arm']=arm;got['eval_threshold']=threshold;outputs.append(got)
            print(f'{outer}/{arm}: {len(fit)} fit, {len(held)} eval, {len(test)} test',flush=True)
    predicted=pd.concat(outputs,ignore_index=True);a.out.mkdir(parents=True,exist_ok=True);predicted.to_csv(a.out/'probe_epochs.csv.gz',index=False)
    nightly=[]
    for (name,arm),d in predicted.groupby(['name','arm']):
        wake=d.y==0;sleep=d.y>0;still_wake=wake&d.still
        fpr,tpr,_=roc_curve(sleep,1-d.p_wake)
        nightly.append(dict(name=name,arm=arm,cohort=d.cohort.iloc[0],participant=d.participant.iloc[0],group=d.group.iloc[0],sleep_sensitivity=d.call_wake[sleep].eq(False).mean(),wake_specificity=d.call_wake[wake].mean(),still_wake_specificity=d.call_wake[still_wake].mean(),auroc=roc_auc_score(sleep,1-d.p_wake),descriptive_spec95=1-np.interp(.95,tpr,fpr)))
    pd.DataFrame(nightly).to_csv(a.out/'probe_nights.csv',index=False)
    subjects=pd.DataFrame(nightly).groupby(['cohort','participant','group','arm']).mean(numeric_only=True).reset_index();subjects.to_csv(a.out/'probe_subjects.csv',index=False)
    subjects.groupby(['cohort','arm']).mean(numeric_only=True).to_csv(a.out/'probe_summary.csv')
    (a.out/'protocol.json').write_text(json.dumps(dict(seed=0,features=FEATURES,selection='fixed C=1; inner group held out; threshold on inner sleep only',aggregation='recording within participant then participant macro',diagnostic='descriptive_spec95 uses test ROC per night; not a deployable threshold',status='screening probe, not a trained deployment model'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('baseline',type=Path);p.add_argument('out',type=Path);main(p.parse_args())
