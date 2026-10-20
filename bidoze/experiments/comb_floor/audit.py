"""Independent artifact, split, label, prediction and matched-window audits."""
from pathlib import Path
import os,json,hashlib,importlib.metadata as metadata
import numpy as np,pandas as pd
from scipy.special import expit,softmax
from sklearn.metrics import roc_auc_score
from pisces_lite.datasets.constants import PSG_5C_MAPPING_TO_WLDR
from extract import OUT,CACHE


def replay(z,x):
 x=x.copy();missing=np.isnan(x);x=np.where(missing,z['imputer_statistics'],x);x=x[:,z['imputer_retained']];x=np.column_stack([x,missing[:,z['indicator_features']]]) if len(z['indicator_features']) else x;x=(x-z['scaler_mean'])/z['scaler_scale'];logit=x@z['coefficient'].T+z['intercept'];return np.column_stack([1-expit(logit[:,0]),expit(logit[:,0])]) if logit.shape[1]==1 else softmax(logit,axis=1)


def kappa(y,p,n):
 if len(np.unique(y))<2:return np.nan
 cm=np.zeros((n,n));np.add.at(cm,(y,p),1);po=np.trace(cm)/cm.sum();pe=(cm.sum(0)*cm.sum(1)).sum()/cm.sum()**2;return (po-pe)/(1-pe)


def main():
 d=pd.read_csv(OUT/'epochs.csv',dtype={'subject_id':str});inventory=pd.read_csv(OUT/'inventory.csv',dtype={'subject_id':str});assert not inventory.duplicated(['cohort','subject_id']).any();assert inventory.groupby('partition').size().to_dict()=={'fit':21,'test':30,'validation':9};assert not np.isinf(d.select_dtypes('number').to_numpy()).any()
 label_checks=0
 for r in inventory.to_dict('records'):
  z=np.load(CACHE/f"{r['cohort']}_subject_{r['subject_id']}_features.npz");y=np.vectorize(lambda a:PSG_5C_MAPPING_TO_WLDR.get(int(a),int(a)),otypes=[int])(z['y'][0]);g=d[(d.cohort==r['cohort'])&(d.subject_id==r['subject_id'])].sort_values('epoch');np.testing.assert_array_equal(y,g.stage);assert hashlib.sha256((CACHE/f"{r['cohort']}_subject_{r['subject_id']}_features.npz").read_bytes()).hexdigest()==r['cache_sha256'];label_checks+=len(y)
  if r['partition']=='test':
   ds='testClinical' if r['cohort']=='SleepAccel-Clinical' else 'testSleepAccel';p=np.load(Path.cwd()/f"bidoze/results/cooperating-arms-2026-10-19/final7/draw0_seed0/predictions/{ds}_{r['subject_id']}.npz");np.testing.assert_array_equal(p['labels'],y)
 saved=pd.read_csv(OUT/'heldout_predictions.csv.gz',dtype={'subject_id':str});subjects=pd.read_csv(OUT/'model_subjects.csv',dtype={'subject_id':str});checked=0
 for (task,model),pr in saved.groupby(['task','model']):
  z=np.load(OUT/'features'/f'model-{task.replace("/","_")}-{model[0]}.npz');g=d[(d.partition=='test')&(d.stage>=0)&(d.stage<4)].copy()
  if task=='NREM/REM':g=g[g.stage>0]
  g=g.set_index(['cohort','subject_id','epoch']).loc[pd.MultiIndex.from_frame(pr[['cohort','subject_id','epoch']])];prob=replay(z,g[list(z['features'])].to_numpy(float));cols=['p'+str(i) for i in range(prob.shape[1])];np.testing.assert_allclose(prob,pr[cols].to_numpy(),atol=1e-10,rtol=1e-9)
  calls=(prob[:,1]>=float(z['threshold'])).astype(int) if task!='three-stage' else prob.argmax(1);np.testing.assert_array_equal(calls,pr.call);checked+=len(pr)
  for (cohort,sid),ix in pr.groupby(['cohort','subject_id']).indices.items():
   row=subjects[(subjects.task==task)&(subjects.model==model)&(subjects.cohort==cohort)&(subjects.subject_id==sid)].iloc[0];truth=pr.iloc[ix].truth.to_numpy(int);pred=pr.iloc[ix].call.to_numpy(int);np.testing.assert_allclose(kappa(truth,pred,prob.shape[1]),row.kappa,atol=1e-10,equal_nan=True)
   if task=='sleep/wake':np.testing.assert_allclose(.5*(np.sum(pred==1)-np.sum(truth==1)),row.tst_bias_minutes,atol=1e-10)
   if task!='three-stage' and len(np.unique(truth))==2:np.testing.assert_allclose(roc_auc_score(truth,prob[ix,1]),row.auroc,atol=1e-10)
 high=pd.read_csv(OUT/'highres_matched.csv',dtype={'subject_id':str});assert high.groupby('pair_id').size().eq(2).all();assert high.groupby('pair_id').binary_stage.sum().eq(1).all();assert high.groupby('pair_id').subject_id.nunique().eq(1).all()
 for (c,sid),g in high.groupby(['cohort','subject_id']):
  assert np.diff(np.sort(g.epoch)).min()>=2;orig=d[(d.cohort==c)&(d.subject_id==sid)].set_index('epoch')
  for row in g.to_dict('records'):
   y=orig.loc[row['epoch']-1:row['epoch']+1].stage.to_numpy();assert np.all(y==3) if row['binary_stage']==1 else np.all((y>0)&(y<3))
   for f in ['jerk_floor','jerk_trackscore','jerk_fundamental_variability','mag_resp_fraction','mag_resp_peak_variability']:np.testing.assert_allclose(row['coarse_'+f],orig.loc[row['epoch'],f],equal_nan=True)
 for _,g in high.groupby('pair_id'):
  for f in ['coarse_jerk_floor','coarse_jerk_trackscore']:assert np.floor(g[f]/2).nunique()==1
 hp=pd.read_csv(OUT/'highres_predictions.csv',dtype={'subject_id':str});hs=pd.read_csv(OUT/'highres_model_subjects.csv',dtype={'subject_id':str});high_checked=0
 for model,pr in hp.groupby('model'):
  z=np.load(OUT/'features'/('highres-model-'+model.replace(' ','_')+'.npz'));assert 'watch_bpm' not in z['features'];g=high.set_index(['cohort','subject_id','epoch']).loc[pd.MultiIndex.from_frame(pr[['cohort','subject_id','epoch']])];prob=replay(z,g[list(z['features'])].to_numpy(float))[:,1];np.testing.assert_allclose(prob,pr.p_rem,atol=1e-10,rtol=1e-9);np.testing.assert_array_equal(prob>=float(z['threshold']),pr.call.astype(bool));high_checked+=len(pr)
  for (cohort,sid),ix in pr.groupby(['cohort','subject_id']).indices.items():
   a=pr.iloc[ix];r=hs[(hs.model==model)&(hs.cohort==cohort)&(hs.subject_id==sid)].iloc[0];np.testing.assert_allclose(roc_auc_score(a.binary_stage,a.p_rem),r.auroc,atol=1e-10);np.testing.assert_allclose(kappa(a.binary_stage.to_numpy(int),a.call.to_numpy(int),2),r.kappa,atol=1e-10)
 assert pd.read_csv(OUT/'highres_model_selection.csv').iterations.max()<2500;assert subjects.converged.all();assert (pd.read_csv(OUT/'rotation_audit.csv').trace_rotation_relative_l2_error<1e-8).all()
 # Raw file hashes are saved by cohort and basename; no raw paths or raw samples enter the artifacts.
 data=Path(os.environ['SLEEP_ANALYSIS_DATA']);raw=[]
 for r in pd.read_csv(OUT/'highres_input_audit.csv',dtype={'subject_id':str}).to_dict('records'):
  c,sid=r['cohort'],r['subject_id'];files=[data/c/'cleaned_accelerometer'/(f'{sid}_cleaned_motion.out' if c=='SleepAccel' else f'{sid}.csv'),data/c/'cleaned_psg'/(f'{sid}_cleaned_psg.out' if c=='SleepAccel' else f'{sid}.csv')]
  if c=='SleepAccel':files.append(data/c/'cleaned_heartrate'/f'{sid}_cleaned_hr.out')
  for p in files:
   if p.exists():
    sha=hashlib.sha256()
    with p.open('rb') as f:
     while block:=f.read(1024*1024):sha.update(block)
    raw.append(dict(cohort=c,subject_id=sid,source=p.parent.name+'/'+p.name,sha256=sha.hexdigest()))
 pd.DataFrame(raw).to_csv(OUT/'raw_input_hashes.csv',index=False)
 versions={}
 for name in ['numpy','scipy','pandas','matplotlib','scikit-learn','pisces-lite','arcascope-senpy']:
  try:versions[name]=metadata.version(name)
  except metadata.PackageNotFoundError:pass
 (OUT/'environment.json').write_text(json.dumps(versions,indent=2));report=dict(all_checks_passed=True,cache_label_epochs_checked=label_checks,recording_count=len(inventory),heldout_prediction_rows_replayed=checked,highres_prediction_rows_replayed=high_checked,matched_raw_windows=len(high),raw_source_files_hashed=len(raw),figure_count=len(list((OUT/'figures').glob('*.png'))));(OUT/'audit.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':main()
