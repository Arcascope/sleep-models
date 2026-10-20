"""Matched, stage-pure 60-s raw windows with finer harmonic and respiratory descriptors."""
from pathlib import Path
import os,json,warnings
import numpy as np,pandas as pd,senpy
from scipy.ndimage import median_filter
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,cohen_kappa_score,log_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract import OUT,CACHE,DB
from raw import spectrum
DATA=Path(os.environ['SLEEP_ANALYSIS_DATA']);RNG=np.random.default_rng(20261020)

def feature(f,p,prefix):
 keep=(f>=.5)&(f<=5.5);db=10*np.log10(np.maximum(p,1e-30));df=np.median(np.diff(f));width=max(3,int(round(.5/df))//2*2+1);res=db-median_filter(db,size=width,mode='nearest');grid=np.arange(.6,2.001,.005)
 score=np.mean([np.interp(h*grid,f,res) for h in [1,2,3]],axis=0)-.25*(np.interp(1.5*grid,f,res)+np.interp(2.5*grid,f,res));best=int(np.argmax(score));f0=grid[best];low=(f>=.12)&(f<=.5);lp=p[low];lf=f[low];pk=lf[np.argmax(lp)];prob=lp/np.maximum(lp.sum(),1e-30)
 return {prefix+'_floor':float(np.quantile(db[keep],.2)),prefix+'_flatness':float(np.exp(np.mean(np.log(np.maximum(p[keep],1e-30))))/np.mean(p[keep])),prefix+'_comb':float(score[best]),prefix+'_f0':float(f0),prefix+'_resp_frequency':float(pk),prefix+'_resp_concentration':float(p[abs(f-pk)<=.03].sum()/max(lp.sum(),1e-30)),prefix+'_resp_entropy':float(-np.sum(prob*np.log(np.maximum(prob,1e-30)))/np.log(len(prob))),prefix+'_resp_fraction':float(lp.sum()/max(p.sum(),1e-30))}

def select(d):
 selected=[]
 for (c,sid),g in d.groupby(['cohort','subject_id']):
  g=g.sort_values('epoch').copy();macro=np.where(g.stage==3,2,np.where((g.stage>0)&(g.stage<3),1,0));pure=(macro==np.r_[0,macro[:-1]])&(macro==np.r_[macro[1:],0])&(macro>0)
  g=g[pure&(g.burst_fraction<=.2)&(g.valid_fraction==1)].copy();g['bin1']=np.floor(g.jerk_floor/2);g['bin2']=np.floor(g.jerk_trackscore/2);groups=[]
  for key,q in g.groupby(['bin1','bin2']):
   a=q[q.stage.isin([1,2])];b=q[q.stage==3]
   if min(len(a),len(b))<1:continue
   ai=RNG.permutation(a.index);bi=RNG.permutation(b.index)
   for x,y in zip(ai,bi):groups.append((g.loc[x],g.loc[y]))
  order=RNG.permutation(len(groups));used=[];count=0
  for k in order:
   a,b=groups[k]
   if any(abs(a.epoch-e)<2 or abs(b.epoch-e)<2 for e in used):continue
   pair=f'{c}-{sid}-{count}'
   for row in [a,b]:z=row.to_dict();z['pair_id']=pair;z['binary_stage']=int(row.stage==3);selected.append(z)
   used.extend([a.epoch,b.epoch]);count+=1
   if count>=10:break
 return pd.DataFrame(selected)

def analyze():
 d=pd.read_csv(OUT/'epochs.csv',dtype={'subject_id':str});sel=select(d);sel.to_csv(OUT/'highres_selection.csv',index=False);rows=[];audits=[];spectra=[]
 for (c,sid),g in sel.groupby(['cohort','subject_id']):
  if c=='SleepAccel':
   a=np.loadtxt(DATA/c/'cleaned_accelerometer'/f'{sid}_cleaned_motion.out');psg=np.loadtxt(DATA/c/'cleaned_psg'/f'{sid}_cleaned_psg.out');hrfile=DATA/c/'cleaned_heartrate'/f'{sid}_cleaned_hr.out';hr=np.loadtxt(hrfile) if hrfile.exists() else None
  else:a=pd.read_csv(DATA/c/'cleaned_accelerometer'/f'{sid}.csv').to_numpy(float);psg=pd.read_csv(DATA/c/'cleaned_psg'/f'{sid}.csv').to_numpy(float);hr=None
  a=a[np.isfinite(a).all(1)];a=a[np.argsort(a[:,0],kind='stable')];a=a[np.r_[True,np.diff(a[:,0])>0]];phase=psg[0,0]%30;start=max(phase+30*np.ceil((a[0,0]-phase)/30),psg[0,0]);end=min(phase+30*np.floor((a[-1,0]-phase)/30),psg[-1,0]);a=a[(a[:,0]>=start)&(a[:,0]<=end)];zero=a[0,0];t=a[:,0]-zero;ax=a[:,1:];mag=np.linalg.norm(ax,axis=1);j=senpy.compute_jerk(t,*ax.T);tj=np.asarray(j.timestamps_s);jerk=np.asarray(j.jerk)
  raw=np.load(CACHE/f'{c}_subject_{sid}_features.npz')['y'][0];grid=start+np.arange(len(raw))*30;indices=np.rint((psg[:,0]-start)/30).astype(int);mapped=np.full(len(raw),-2);seen=set()
  for index,label in zip(indices,psg[:,1]):
   if 0<=index<len(raw) and index not in seen:mapped[index]=int(label);seen.add(index)
  if c=='SleepAccel-Clinical':mapped[mapped==4]=5
  mask=raw>=0;np.testing.assert_array_equal(mapped[mask],raw[mask]);audits.append(dict(cohort=c,subject_id=sid,epoch0_raw_seconds=zero,scored_label_match=True,median_fs_hz=1/np.median(np.diff(t))))
  for r in g.to_dict('records'):
   at=r['epoch']*30+15;row={k:r[k] for k in ['cohort','subject_id','partition','epoch','pair_id','binary_stage']};row.update({'coarse_'+k:r[k] for k in ['jerk_floor','jerk_trackscore','jerk_fundamental_variability','mag_resp_fraction','mag_resp_peak_variability']});good=True;axis_desc=[];axis_spectra=[]
   for name,tt,sig in [('mag',t,mag),('jerk',tj,jerk),('x',t,ax[:,0]),('y',t,ax[:,1]),('z',t,ax[:,2])]:
    spec=spectrum(tt,sig,at,60)
    if spec is None:good=False;break
    f,p,gap,n=spec;m=(f>=.05)&(f<=6);f=f[m];p=p[m];good &=gap<=.1 and n>=2400
    desc=feature(f,p,name)
    if name in ['mag','jerk']:
     row.update(desc)
     if name=='jerk':
      halves=[]
      for centre in [at-15,at+15]:
       sub=spectrum(tt,sig,centre,30)
       if sub is not None:halves.append(feature(sub[0],sub[1],'half')['half_f0'])
      row['jerk_half_f0_change']=abs(halves[1]-halves[0]) if len(halves)==2 else np.nan
      if hr is not None:
       h=(hr[:,0]>=zero+at-30)&(hr[:,0]<zero+at+30);row['watch_bpm']=float(np.median(hr[h,1])) if h.any() else np.nan
      spectra.append(dict(cohort=c,subject_id=sid,epoch=r['epoch'],stage=r['binary_stage'],f=f.copy(),jerk_db=10*np.log10(np.maximum(p,1e-30))))
    else:axis_desc.append(desc);axis_spectra.append(p)
   if not good:continue
   # Select one axis without labels by its respiratory-band power fraction.
   axis_names=['x','y','z'];ratios=[z[n+'_resp_fraction'] for z,n in zip(axis_desc,axis_names)];best=int(np.argmax(ratios));name=axis_names[best]
   for q in ['resp_frequency','resp_concentration','resp_entropy','resp_fraction']:row['axis_'+q]=axis_desc[best][name+'_'+q]
   row.update(feature(f,np.sum(axis_spectra,axis=0),'trace'))
   row['median_sample_rate']=float(1/np.median(np.diff(t[(t>=at-30)&(t<at+30)])));rows.append(row)
  pd.DataFrame(rows).to_csv(OUT/'highres_windows.csv',index=False);print('highres',c,sid,len(g),'windows',flush=True)
 pd.DataFrame(audits).to_csv(OUT/'highres_input_audit.csv',index=False)
 # Retain only complete matched pairs after raw-window quality exclusions.
 out=pd.DataFrame(rows);counts=out.groupby('pair_id').size();out=out[out.pair_id.isin(counts[counts==2].index)];out.to_csv(OUT/'highres_matched.csv',index=False)
 # Selected raw spectra are derived artifacts, not raw acceleration.
 chosen=[s for s in spectra if s['cohort']=='SleepAccel-Clinical' and s['subject_id']=='aws021'];fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
 for stage,label,color in [(0,'NREM','#467ac1'),(1,'REM','#28b9bd')]:
  ss=[s for s in chosen if s['stage']==stage][:5]
  for s in ss:axes[stage].plot(s['f'],s['jerk_db'],lw=.8,label=f"{s['epoch']/120:.2f} h")
  axes[stage].set(ylabel='dB raw PSD',title='aws021: '+label+' matched, stage-pure 60-s windows');axes[stage].legend(fontsize=8,ncol=5);axes[stage].grid(alpha=.15)
 axes[-1].set(xlabel='Hz',xlim=(.1,6));fig.tight_layout();fig.savefig(OUT/'figures/highres-aws021-matched-stages.png',dpi=150);plt.close(fig)
 return out

def summarize(d):
 features=[('jerk_f0','Comb spacing (Hz)'),('jerk_half_f0_change','Spacing change across 30-s halves (Hz)'),('mag_resp_concentration','Magnitude breathing-peak concentration'),('axis_resp_entropy','Best-axis respiratory spectral entropy'),('axis_resp_frequency','Best-axis respiratory peak (Hz)'),('axis_resp_fraction','Best-axis respiratory power fraction'),('trace_resp_entropy','Total-axis respiratory spectral entropy'),('trace_resp_fraction','Total-axis respiratory power fraction')]
 pair=d.set_index(['pair_id','binary_stage']);paired=[]
 for pid,g in d.groupby('pair_id'):
  a=g[g.binary_stage==0].iloc[0];b=g[g.binary_stage==1].iloc[0];row=dict(cohort=a.cohort,subject_id=a.subject_id,partition=a.partition,pair_id=pid)
  for f,_ in features:row[f]=b[f]-a[f]
  paired.append(row)
 delta=pd.DataFrame(paired);person=delta.groupby(['partition','cohort','subject_id']).mean(numeric_only=True).reset_index();person.to_csv(OUT/'highres_person_differences.csv',index=False)
 fig,axes=plt.subplots(2,4,figsize=(21,8))
 for ax,(f,label) in zip(axes.flat,features):
  for i,c in enumerate(['SleepAccel','SleepAccel-Clinical']):
   vals=person[(person.partition=='test')&(person.cohort==c)][f].dropna();ax.scatter(vals,np.full(len(vals),i),s=22,alpha=.7)
   if len(vals):ax.scatter(vals.mean(),i,color='black',marker='D',s=35)
  ax.axvline(0,color='grey',lw=.8);ax.set_yticks([0,1],['SleepAccel','Clinical']);ax.set_xlabel('REM − NREM');ax.set_title(label);ax.grid(axis='x',alpha=.15)
 fig.suptitle('Higher-resolution matched stage differences; participant means on held-out adults');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'figures/highres-matched-stage-features.png',dpi=150);plt.close(fig)
 # Fit on fit participants only; choose C and decision threshold on validation participants.
 rawfeatures=[c for c in d.columns if c.startswith(('mag_','jerk_','axis_','trace_'))];coarse=['coarse_'+q for q in ['jerk_floor','jerk_trackscore','jerk_fundamental_variability','mag_resp_fraction','mag_resp_peak_variability']];sets={'coarse matched':coarse,'highres cardiac':['jerk_f0','jerk_comb','jerk_half_f0_change','jerk_flatness'],'highres respiratory':['mag_resp_frequency','mag_resp_concentration','mag_resp_entropy','axis_resp_frequency','axis_resp_concentration','axis_resp_entropy','axis_resp_fraction'],'highres trace respiratory':['trace_resp_frequency','trace_resp_concentration','trace_resp_entropy','trace_resp_fraction'],'highres combined':rawfeatures};rows=[];selection=[];predictions=[]
 for name,features in sets.items():
  parts={k:d[d.partition==k].copy() for k in ['fit','validation','test']};fit,val,test=[parts[k] for k in ['fit','validation','test']];best=None
  for c in [.01,.1,1,10]:
   model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=c,max_iter=2500));model.fit(fit[features],fit.binary_stage);v=model.predict_proba(val[features])[:,1];ce=np.mean([log_loss(g.binary_stage,v[ix],labels=[0,1]) for _,ix in val.groupby(['cohort','subject_id']).indices.items() for g in [val.iloc[ix]]])
   selection.append(dict(model=name,C=c,validation_ce=ce,iterations=int(model[-1].n_iter_.max())));
   if best is None or ce<best[0]:best=(ce,model,c,v)
  ce,model,c,v=best;cuts=np.linspace(.1,.9,161);scores=[np.mean([cohen_kappa_score(val.iloc[ix].binary_stage,(v[ix]>=cut).astype(int)) for ix in val.groupby(['cohort','subject_id']).indices.values()]) for cut in cuts];cut=cuts[int(np.argmax(scores))];prob=model.predict_proba(test[features])[:,1]
  pr=test[['cohort','subject_id','epoch','pair_id','binary_stage']].copy();pr['model']=name;pr['p_rem']=prob;pr['call']=(prob>=cut).astype(int);predictions.append(pr)
  np.savez_compressed(OUT/'features'/('highres-model-'+name.replace(' ','_')+'.npz'),features=np.array(features),coefficient=model[-1].coef_,intercept=model[-1].intercept_,classes=model[-1].classes_,imputer_statistics=model[0].statistics_,indicator_features=model[0].indicator_.features_,imputer_retained=np.isfinite(model[0].statistics_),scaler_mean=model[1].mean_,scaler_scale=model[1].scale_,threshold=cut)
  for (cohort,sid),ix in test.groupby(['cohort','subject_id']).indices.items():
   y=test.iloc[ix].binary_stage;rows.append(dict(model=name,cohort=cohort,subject_id=sid,epochs=len(ix),auroc=roc_auc_score(y,prob[ix]),kappa=cohen_kappa_score(y,(prob[ix]>=cut).astype(int)),C=c,threshold=cut,validation_ce=ce))
 pd.DataFrame(selection).to_csv(OUT/'highres_model_selection.csv',index=False);pd.concat(predictions).to_csv(OUT/'highres_predictions.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'highres_model_subjects.csv',index=False);fig,axes=plt.subplots(1,2,figsize=(14,5))
 table=pd.DataFrame(rows)
 for ax,c in zip(axes,['SleepAccel','SleepAccel-Clinical']):
  for i,name in enumerate(sets):
   a=table[(table.model==name)&(table.cohort==c)].auroc;ax.scatter(a,np.full(len(a),i),s=18,alpha=.4);ax.scatter(a.mean(),i,marker='D',color='black');ax.plot([a.min(),a.max()],[i,i],alpha=.4)
  ax.axvline(.5,color='grey',lw=.8);ax.set_yticks(range(len(sets)),list(sets));ax.set(xlabel='Participant AUROC',title=c,xlim=(0,1));ax.grid(axis='x',alpha=.15)
 fig.suptitle('NREM vs REM on the same quality-filtered, matched raw windows; ranges are participant spread');fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(OUT/'figures/highres-stage-auroc.png',dpi=150);plt.close(fig)
 print('High-resolution matched analysis complete',len(d),'windows',d.groupby('partition').subject_id.nunique().to_dict(),flush=True)
if __name__=='__main__':
 if (OUT/'highres_windows.csv').exists() and '--reuse' in __import__('sys').argv:d=pd.read_csv(OUT/'highres_matched.csv',dtype={'subject_id':str})
 else:d=analyze()
 summarize(d)
