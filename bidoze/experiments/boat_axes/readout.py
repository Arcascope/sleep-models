"""Fixed vessel-held-out probes of retained axes beyond scores and existing mag/jerk inputs."""
from __future__ import annotations
import argparse,json,hashlib,subprocess,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.exceptions import ConvergenceWarning
from arcasleep.ggir import evaluate_epoch_grid

BANDS=[(.1,.5),(.5,2),(2,6)]
BASE=[f'{ch}_band{i}' for ch in ('mag','jerk') for i in range(3)]
AXES=[f'trace_band{i}' for i in range(3)]+['cov_log_trace','cov_principal_share','cov_smallest_share']
ARMS={'score':['wake_logit'],'score_magjerk':['wake_logit']+BASE,
      'score_axes':['wake_logit']+AXES,'score_magjerk_axes':['wake_logit']+BASE+AXES}
METRICS=['sleep_sensitivity','wake_specificity','still_wake_specificity','auroc','descriptive_spec95','still_auroc','still_spec95']


def person_night_weights(frame):
    nights=frame.groupby('participant')['name'].transform('nunique').to_numpy()
    epochs=frame.groupby(['participant','name'])['name'].transform('size').to_numpy()
    w=1/(nights*epochs);return w/w.sum()


def threshold(scores,held,forced):
    sleep=held.y.to_numpy()>0
    h=held.loc[sleep];w=person_night_weights(h)
    effective=scores[sleep].copy();effective[forced[sleep]]=1.
    order=np.argsort(effective,kind='stable');cdf=np.cumsum(w[order])
    cutoff=float(effective[order[min(np.searchsorted(cdf,.95),len(order)-1)]])
    calls=(scores[sleep]>cutoff)|forced[sleep]
    achieved=float(np.sum(w*~calls))
    return cutoff,achieved,bool(achieved<.95-1e-8)


def ranking(y,p):
    if len(np.unique(y))<2:return np.nan,np.nan
    fpr,tpr,_=roc_curve(y,1-p)
    return roc_auc_score(y,1-p),1-np.interp(.95,tpr,fpr)


def main(a):
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'models').mkdir(exist_ok=True)
    meta=pd.read_csv(a.data/'inventory.csv');meta=meta[meta.role=='boat'];rows=[];coverage=[]
    for r in meta.to_dict('records'):
        folder=a.observations/'nights'/r['name']
        with np.load(folder/'native_axis_psd.npz') as d,np.load(folder/'probabilities.npz') as p,np.load(a.data/'nights'/f"{r['name']}.npz") as source:
            psd=d['epoch_linear_psd'];freq=d['frequencies'];y=p['truth'];n=len(y)
            model=d['model_epoch_log_psd'][:n,:-1];eigen=np.linalg.eigvalsh(np.nan_to_num(d['axis_covariance']))
            trace=eigen.sum(1);wake=p['native'][:,0]/np.maximum(p['native'][:,:4].sum(1),1e-12)
            ggir=evaluate_epoch_grid(source['anglez'][:n]);forced=~ggir.is_sib&~ggir.is_spt
            feats={'wake_logit':np.log(np.clip(wake,1e-6,1-1e-6)/(1-np.clip(wake,1e-6,1-1e-6))),
                'cov_log_trace':np.log(np.maximum(trace,1e-30)),
                'cov_principal_share':eigen[:,-1]/np.maximum(trace,1e-30),
                'cov_smallest_share':eigen[:,0]/np.maximum(trace,1e-30)}
            for i,(lo,hi) in enumerate(BANDS):
                band=(freq>=lo)&(freq<=hi)
                feats[f'trace_band{i}']=np.log(np.maximum(np.mean(psd[:,band,:3].sum(2),axis=1),1e-30))
                for j,ch in enumerate(('mag','jerk')):feats[f'{ch}_band{i}']=np.mean(model[:,band,j],axis=1)
            frame=pd.DataFrame(feats);frame['epoch']=np.arange(n);frame['y']=y;frame['still']=p['still'];frame['forced_wake']=forced
            for k in ('name','cohort','participant','group'):frame[k]=r[k]
            eligible=(y>=0)&~p['common_missing'];finite=np.isfinite(frame[list(dict.fromkeys(sum(ARMS.values(),[])))].to_numpy()).all(1)&np.isfinite(d['axis_covariance']).all((1,2))
            coverage.append(dict(name=r['name'],cohort=r['cohort'],scored_covered=int(eligible.sum()),retained=int((eligible&finite).sum())))
            rows.append(frame[eligible&finite])
    table=pd.concat(rows,ignore_index=True);pd.DataFrame(coverage).to_csv(a.out/'coverage.csv',index=False)
    table.to_csv(a.out/'features.csv.gz',index=False)
    protocol=dict(code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arms=ARMS,bands=BANDS,seed=0,C=1,solver='lbfgs',max_iter=1000,
        reference='frozen three-seed released-ensemble wake score',
        fit='equal participant and recording mass, then equal weighted class mass; scaler fitted on fit only',
        split='A-D conservative held-out vessel groups; E Somerset transfer; inner B for outer A, A otherwise',
        threshold='95% recording-within-person then person-macro sensitivity on inner sleep epochs; separate raw/GGIR thresholds',
        ranking='test ROC per night is descriptive, not a deployment threshold',
        interpretation='discovery readout; fixed feature sets, no HPO; global spectral contrasts previously observed',
        continuation='matched seed-0 BiDoze training may proceed even after a linear-readout null')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2));outputs=[];fits=[];splits={}
    for outer in 'ABCDE':
        inner='B' if outer=='A' else 'A'
        fit=table[~table.group.isin([outer,inner])];held=table[table.group==inner];test=table[table.group==outer]
        persons=[set(zip(g.cohort,g.participant)) for g in (fit,held,test)]
        assert not persons[0]&persons[1] and not (persons[0]|persons[1])&persons[2]
        splits[outer]={role:sorted(g.name.unique()) for role,g in [('fit',fit),('eval',held),('test',test)]}
        weights=person_night_weights(fit);target=(fit.y.to_numpy()==0).astype(int)
        balanced=weights.copy()
        for c in (0,1):balanced[target==c]*=.5/weights[target==c].sum()
        balanced*=len(fit)
        for arm,features in ARMS.items():
            scaler=StandardScaler().fit(fit[features],sample_weight=weights)
            model=LogisticRegression(C=1,max_iter=1000,random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model.fit(scaler.transform(fit[features]),target,sample_weight=balanced)
            joblib.dump(dict(scaler=scaler,model=model,features=features),a.out/'models'/f'{outer}_{arm}.joblib')
            ph=model.predict_proba(scaler.transform(held[features]))[:,1];pt=model.predict_proba(scaler.transform(test[features]))[:,1]
            for mode in ('raw','ggir'):
                fh=held.forced_wake.to_numpy() if mode=='ggir' else np.zeros(len(held),bool)
                ft=test.forced_wake.to_numpy() if mode=='ggir' else np.zeros(len(test),bool)
                cut,achieved,unattained=threshold(ph,held,fh)
                got=test[['epoch','name','cohort','participant','group','y','still']].copy();got['p_wake']=pt;got['call_wake']=(pt>cut)|ft
                got['arm']=arm;got['mode']=mode;got['outer']=outer;got['threshold']=cut;outputs.append(got)
                fits.append(dict(outer=outer,inner=inner,arm=arm,mode=mode,fit_epochs=len(fit),eval_epochs=len(held),test_epochs=len(test),threshold=cut,inner_sleep_sensitivity=achieved,target_unattained=unattained,iterations=int(model.n_iter_[0]),features=features,coefficients=model.coef_[0].tolist()))
            print(outer,arm,'fit/eval/test',len(fit),len(held),len(test),flush=True)
    (a.out/'splits.json').write_text(json.dumps(splits,indent=2));(a.out/'fit.json').write_text(json.dumps(fits,indent=2))
    predicted=pd.concat(outputs,ignore_index=True);predicted.to_csv(a.out/'epochs.csv.gz',index=False);nightly=[]
    for (name,arm,mode),d in predicted.groupby(['name','arm','mode']):
        wake=d.y==0;sleep=d.y>0;still_wake=wake&d.still
        auc,spec=ranking(sleep.to_numpy(),d.p_wake.to_numpy());sa,ss=ranking(sleep[d.still].to_numpy(),d.p_wake[d.still].to_numpy())
        nightly.append(dict(name=name,arm=arm,mode=mode,cohort=d.cohort.iloc[0],participant=d.participant.iloc[0],group=d.group.iloc[0],
            sleep_sensitivity=(~d.call_wake[sleep]).mean(),wake_specificity=d.call_wake[wake].mean(),still_wake_specificity=d.call_wake[still_wake].mean(),auroc=auc,descriptive_spec95=spec,still_auroc=sa,still_spec95=ss))
    nights=pd.DataFrame(nightly);nights.to_csv(a.out/'nights.csv',index=False)
    subjects=nights.groupby(['cohort','participant','group','arm','mode'])[METRICS].mean().reset_index();subjects.to_csv(a.out/'subjects.csv',index=False)
    subjects.groupby(['cohort','arm','mode'])[METRICS].mean().to_csv(a.out/'summary.csv')
    comparisons=[];rng=np.random.default_rng(12)
    for (cohort,mode),g in subjects.groupby(['cohort','mode']):
        for comparator in ('score','score_magjerk'):
            baseline=g[g.arm==comparator].set_index(['participant','group'])[METRICS]
            for arm in ('score_axes','score_magjerk_axes'):
                delta=g[g.arm==arm].set_index(['participant','group'])[METRICS]-baseline
                for metric in METRICS:
                    v=delta[metric].dropna();groups=v.index.get_level_values('group').unique().tolist()
                    if cohort=='SURF':
                        boot=[pd.concat([v.xs(group,level='group') for group in rng.choice(groups,len(groups),replace=True)]).mean() for _ in range(2000)]
                        unit='conservative outer-group cluster bootstrap; only four clusters'
                    else:
                        values=v.to_numpy();boot=values[rng.integers(len(values),size=(2000,len(values)))].mean(1);unit='participant bootstrap; not vessel uncertainty'
                    lo,hi=np.percentile(boot,[2.5,97.5]);comparisons.append(dict(cohort=cohort,mode=mode,arm=arm,comparator=comparator,metric=metric,participants=len(v),difference=v.mean(),ci_low=lo,ci_high=hi,uncertainty=unit))
    pd.DataFrame(comparisons).to_csv(a.out/'paired.csv',index=False)
    (a.out/'complete.json').write_text(json.dumps(dict(nights=len(meta),epochs=len(table),models=20,complete=True)))
    print('readout complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('observations',type=Path);p.add_argument('out',type=Path);main(p.parse_args())
