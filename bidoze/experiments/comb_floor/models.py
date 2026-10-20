"""Participant-separated descriptor ablations and within-person matched contrasts."""
from pathlib import Path
import json,warnings
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,cohen_kappa_score,balanced_accuracy_score,roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from scipy.special import softmax
from scipy.stats import wasserstein_distance
from extract import OUT

BASE=['mag_logpower','mag_floor','mag_flatness','mag_slope','valid_fraction']
FLOOR=['jerk_floor','jerk_flatness','jerk_slope','mag_floor_relative','jerk_floor_relative']
COMB=['jerk_comb_excess_null','mag_comb_excess_null','jerk_trackscore','jerk_comb_visible','jerk_fundamental','jerk_fundamental_iqr','mag_trackscore','mag_comb_visible','jerk_comb_x_floor','jerk_comb_x_flatness']
BURST=['burst_fraction','quiet_fraction','mag_floor_low','jerk_floor_low','mag_power_range','jerk_power_range']
TEMP=[f'{x}_{q}' for x in ['mag_floor','jerk_floor','jerk_trackscore','jerk_comb_visible','jerk_fundamental'] for q in ['context','variability']]
RESP=[f'{x}_{q}' for x in ['mag','jerk'] for q in ['resp_power','resp_fraction','resp_peak']]+['mag_resp_peak_context','mag_resp_peak_variability']
SETS={'A magnitude':BASE,'B + jerk background':BASE+FLOOR,'C + comb':BASE+FLOOR+COMB,'D + bursts':BASE+FLOOR+COMB+BURST,'E + temporal':BASE+FLOOR+COMB+BURST+TEMP,'F + respiration proxy':BASE+FLOOR+COMB+BURST+TEMP+RESP}

def true_labels(g,task):
 y=g.stage.to_numpy();return (y>0).astype(int) if task=='sleep/wake' else (y==3).astype(int) if task=='NREM/REM' else np.where(y==0,0,np.where(y==3,2,1))

def stats(truth,pred,task):
 row=dict(kappa=float(cohen_kappa_score(truth,pred)) if len(np.unique(truth))>1 else np.nan,accuracy=float(np.mean(truth==pred)),balanced_accuracy=float(balanced_accuracy_score(truth,pred)))
 for cls in np.unique(truth):row['recall_'+str(cls)]=float(np.mean(pred[truth==cls]==cls))
 if task=='sleep/wake':row.update(sleep_sensitivity=row.get('recall_1',np.nan),wake_specificity=row.get('recall_0',np.nan),tst_bias_minutes=.5*(np.sum(pred==1)-np.sum(truth==1)),tst_mae_minutes=abs(.5*(np.sum(pred==1)-np.sum(truth==1))))
 return row

def main():
 d=pd.read_csv(OUT/'epochs.csv',dtype={'subject_id':str});d=d[(d.stage>=0)&(d.stage<4)].copy();rng=np.random.default_rng(20261020);records=[];candidates=[];coeffs=[];predictions=[]
 for task in ['sleep/wake','three-stage','NREM/REM']:
  use=d[d.stage>0].copy() if task=='NREM/REM' else d.copy();fit=use[use.partition=='fit'];val=use[use.partition=='validation'];test=use[use.partition=='test']
  chosen=np.concatenate([np.sort(rng.choice(ix,min(len(ix),1200),replace=False)) for ix in fit.groupby(['cohort','subject_id']).indices.values()]);fit=fit.iloc[chosen]
  fit=fit.reset_index(drop=True);fy=true_labels(fit,task);vy=true_labels(val,task);ty=true_labels(test,task)
  weights=fit.groupby(['cohort','subject_id']).subject_id.transform('size').to_numpy();weights=1/weights;weights*=len(weights)/weights.sum()
  for name,features in SETS.items():
   best=None
   for c in [.01,.1,1,10]:
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=c,max_iter=2500,solver='lbfgs'))
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always');model.fit(fit[features],fy,logisticregression__sample_weight=weights)
    vp=model.predict_proba(val[features]);loss=[]
    for _,ix in val.groupby(['cohort','subject_id']).indices.items():loss.append(log_loss(vy[ix],vp[ix],labels=model[-1].classes_))
    ce=float(np.mean(loss));conv=not any(issubclass(w.category,ConvergenceWarning) for w in caught);candidates.append(dict(task=task,model=name,C=c,validation_ce=ce,converged=conv))
    if best is None or ce<best[0]:best=(ce,model,c,conv,vp)
   ce,model,c,conv,vp=best;t=.5
   if task in ['sleep/wake','NREM/REM']:
    scores=[]
    for cut in np.linspace(.05,.95,181):
     vals=[cohen_kappa_score(vy[ix],(vp[ix,1]>=cut).astype(int)) for _,ix in val.groupby(['cohort','subject_id']).indices.items() if len(np.unique(vy[ix]))>1];scores.append(np.mean(vals))
    t=float(np.linspace(.05,.95,181)[int(np.argmax(scores))])
   prob=model.predict_proba(test[features]);call=(prob[:,1]>=t).astype(int) if task in ['sleep/wake','NREM/REM'] else prob.argmax(1)
   for key,ix in test.groupby(['cohort','subject_id']).indices.items():
    extra={'auroc':float(roc_auc_score(ty[ix],prob[ix,1])) if len(np.unique(ty[ix]))==2 else np.nan} if task in ['sleep/wake','NREM/REM'] else {}
    records.append(dict(task=task,model=name,cohort=key[0],subject_id=key[1],C=c,threshold=t,validation_ce=ce,converged=conv,epochs=len(ix),**stats(ty[ix],call[ix],task),**extra))
   pr=test[['cohort','subject_id','epoch','stage']].copy();pr['task']=task;pr['model']=name;pr['truth']=ty;pr['call']=call
   for j in range(prob.shape[1]):pr['p'+str(j)]=prob[:,j]
   predictions.append(pr)
   feat=model[0].get_feature_names_out(features);coef=model[-1].coef_/model[1].scale_[None,:]
   for j in range(len(coef)):
    for f,w in zip(feat,coef[j]):coeffs.append(dict(task=task,model=name,class_index=j,feature=f,coefficient=w,standardized_coefficient=model[-1].coef_[j,list(feat).index(f)]))
   # Exact selected weights and preprocessing parameters, without pickled objects.
   np.savez_compressed(OUT/'features'/f'model-{task.replace("/","_")}-{name[0]}.npz',imputer_statistics=model[0].statistics_,indicator_features=model[0].indicator_.features_,imputer_retained=np.isfinite(model[0].statistics_),scaler_mean=model[1].mean_,scaler_scale=model[1].scale_,coefficient=model[-1].coef_,intercept=model[-1].intercept_,classes=model[-1].classes_,threshold=t,features=np.array(features))
   print(task,name,'validation CE',round(ce,4),'C',c,'converged',conv,flush=True)
 pd.DataFrame(records).to_csv(OUT/'model_subjects.csv',index=False);pd.DataFrame(candidates).to_csv(OUT/'model_selection.csv',index=False);pd.DataFrame(coeffs).to_csv(OUT/'coefficients.csv',index=False);pd.concat(predictions).to_csv(OUT/'heldout_predictions.csv.gz',index=False)
 # Staged descriptor differences, each participant first contributes one median per class.
 measures=['mag_floor','jerk_floor','jerk_flatness','jerk_trackscore','jerk_comb_visible','burst_fraction','jerk_floor_low','jerk_fundamental_variability','mag_resp_fraction','mag_resp_peak_variability']
 med=d.groupby(['partition','cohort','subject_id','stage'])[measures].median();med.to_csv(OUT/'stage_subject_medians.csv')
 # A common movement band and comb-strength band creates within-person comparisons without using candidate outcomes.
 matched=[];counts=[]
 for comparison,first,second in [('quiet wake/sleep',lambda y:y==0,lambda y:y>0),('NREM/REM',lambda y:(y>0)&(y<3),lambda y:y==3)]:
  for (cohort,sid),g in d[d.partition=='test'].groupby(['cohort','subject_id']):
   g=g[g.burst_fraction<=.2].copy()
   if comparison=='quiet wake/sleep':g['bin1']=np.floor(g.mag_logpower/2);g['bin2']=np.floor(g.jerk_trackscore/2)
   else:g['bin1']=np.floor(g.jerk_floor/2);g['bin2']=np.floor(g.jerk_trackscore/2)
   differences=[];total=0
   for _,h in g.groupby(['bin1','bin2']):
    a=h[first(h.stage)];b=h[second(h.stage)]
    if min(len(a),len(b))<2:continue
    n=min(len(a),len(b));total+=n
    differences.append((n,b[measures].mean()-a[measures].mean()))
   counts.append(dict(comparison=comparison,cohort=cohort,subject_id=sid,matched_epochs_per_class=total,strata=len(differences)))
   if total:
    out=sum(n*v for n,v in differences)/total
    matched.append(dict(comparison=comparison,cohort=cohort,subject_id=sid,matched_epochs_per_class=total,**out.to_dict()))
 pd.DataFrame(matched).to_csv(OUT/'matched_differences.csv',index=False);pd.DataFrame(counts).to_csv(OUT/'matched_counts.csv',index=False)
 print('Finished model and matched-contrast analyses',flush=True)
if __name__=='__main__':main()
