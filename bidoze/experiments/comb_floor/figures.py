"""Render the four investigations from saved features and held-out scores."""
from pathlib import Path
import os,json
import numpy as np,pandas as pd
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from extract import OUT,CACHE,COLORS,STAGES,DB
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
D=pd.read_csv(OUT/'epochs.csv',dtype={'subject_id':str});SCORED=D[(D.stage>=0)&(D.stage<4)];TEST=SCORED[SCORED.partition=='test'];CMAP=ListedColormap(COLORS);CMAP.set_under('#b6b6b6');CMAP.set_over('#b6b6b6');NORM=BoundaryNorm(np.arange(-.5,4.5),4)

def save(fig,name):
 fig.savefig(FIG/name,dpi=150,bbox_inches='tight');plt.close(fig);print('figure',name,flush=True)

def shade(ax,g):
 t=g.hours.to_numpy();y=g.stage.to_numpy()
 for c in range(4):
  starts=np.flatnonzero((y==c)&~np.r_[False,(y[:-1]==c)]);ends=np.flatnonzero((y==c)&~np.r_[(y[1:]==c),False])
  for a,b in zip(starts,ends):ax.axvspan(t[a],t[b]+1/120,color=COLORS[c],alpha=.09,lw=0)

def overview(cohort,sid,tag=None):
 g=D[(D.cohort==cohort)&(D.subject_id==sid)].sort_values('epoch');z=np.load(CACHE/f'{cohort}_subject_{sid}_features.npz');n=len(g);x=z['X'][0].reshape(n,15,60,2);valid=~np.all(x==-1,axis=(2,3));x=np.where(valid[:,:,None,None],x,np.nan);p=np.nanmean(x,axis=1)*DB
 fig,axes=plt.subplots(8,1,figsize=(15,15),sharex=True,gridspec_kw={'height_ratios':[.45,1.4,1.4,1,1,1,1,1]});h=g.hours.to_numpy()
 axes[0].imshow(g.stage.to_numpy()[None],aspect='auto',extent=[0,n/120,0,1],cmap=CMAP,norm=NORM);axes[0].set_yticks([]);axes[0].set_title(f'{cohort} {sid} | PSG and separated spectral components')
 for ch in range(2):
  lo,hi=np.nanquantile(p[:,:,ch],[.03,.98]);im=axes[1+ch].imshow(p[:,:,ch].T,origin='lower',aspect='auto',extent=[0,n/120,.1,6],cmap='viridis',vmin=lo,vmax=hi);axes[1+ch].set_ylabel(('Magnitude' if ch==0 else 'Jerk')+'\nHz');fig.colorbar(im,ax=axes[1+ch],label='dB, cached PSD',pad=.01,fraction=.02)
 axes[3].plot(h,g.mag_floor,label='Magnitude');axes[3].plot(h,g.jerk_floor,label='Jerk');axes[3].set_ylabel('Background\nproxy (dB)');axes[3].legend(loc='upper right',ncol=2)
 axes[4].plot(h,g.jerk_trackscore,label='Harmonic prominence');axes[4].plot(h,g.jerk_comb_excess_null,label='Above scrambled null');axes[4].axhline(0,color='grey',lw=.7);axes[4].set_ylabel('Comb score\n(dB)');axes[4].legend(loc='upper right',ncol=2)
 axes[5].scatter(h,g.jerk_fundamental*60,c=g.stage,cmap=CMAP,norm=NORM,s=8);axes[5].set_ylabel('Visible comb\nspacing (bpm)');axes[5].set_ylim(30,125)
 axes[6].plot(h,g.burst_fraction,label='Burst fraction');axes[6].plot(h,g.jerk_flatness,label='Jerk flatness');axes[6].set_ylabel('Fraction');axes[6].legend(loc='upper right',ncol=2)
 axes[7].plot(h,g.mag_resp_fraction,label='Magnitude 0.1–0.5 Hz power fraction');axes[7].set_ylabel('Low-band\nfraction');axes[7].legend(loc='upper right');axes[7].set_xlabel('Hours from model recording start')
 for ax in axes[3:]:shade(ax,g);ax.grid(alpha=.15)
 handles=[plt.Line2D([],[],color=c,lw=5,label=s) for c,s in zip(COLORS,STAGES)];fig.legend(handles=handles,loc='lower center',ncol=4);fig.tight_layout(rect=[0,.025,1,1]);save(fig,(tag or f'overview-{cohort}-{sid}')+'.png')

def aws_diagnostics():
 cohort='SleepAccel-Clinical';sid='aws021';g=D[(D.cohort==cohort)&(D.subject_id==sid)].sort_values('epoch');z=np.load(OUT/f'features/{cohort}_{sid}.npz');t=z['time']/3600
 fig,axes=plt.subplots(3,1,figsize=(15,10),sharex=True)
 im=axes[0].pcolormesh(t,z['fundamental_grid']*60,z['jerk_bank'].T,shading='auto',cmap='magma',vmin=0,vmax=np.nanquantile(z['jerk_bank'],.99));axes[0].set_ylabel('Candidate spacing\n(bpm)');fig.colorbar(im,ax=axes[0],label='Comb score (dB)')
 vis=z['jerk_comb_visible']>0;axes[1].scatter(t,z['jerk_fundamental']*60,c=np.repeat(g.stage.to_numpy(),15),cmap=CMAP,norm=NORM,s=2,alpha=.7);axes[1].set_ylabel('Visible tracked\nspacing (bpm)');axes[1].set_ylim(30,125)
 axes[2].plot(t,z['jerk_trackscore'],alpha=.5,label='Tracked comb prominence');axes[2].plot(t,z['jerk_comb_excess_null'],alpha=.5,label='Prominence − scrambled null threshold');axes[2].set_ylabel('dB');axes[2].axhline(0,color='grey');axes[2].legend();axes[2].set_xlabel('Hours');fig.suptitle('aws021: comb spacing can persist while its prominence changes');fig.tight_layout(rect=[0,0,1,.96]);save(fig,'aws021-moving-comb.png')
 # Frame-level distributions distinguish burst averages from quiet-frame background.
 frame_y=np.repeat(g.stage.to_numpy(),15);groups=[(t>=2.5)&(t<3)&(frame_y>0)&(frame_y<4),(t>=3.5)&(t<4)&(frame_y==0)]
 fig,axes=plt.subplots(2,3,figsize=(15,8))
 for row,quiet in enumerate([False,True]):
  for col,(name,label) in enumerate([('mag_floor','Magnitude background (dB)'),('jerk_floor','Jerk background (dB)'),('jerk_trackscore','Comb prominence (dB)')]):
   vals=[z[name][m&(z['quiet'] if quiet else z['valid'])] for m in groups];axes[row,col].boxplot(vals,tick_labels=['Sleep 2.5–3 h','Wake 3.5–4 h'],showfliers=False);axes[row,col].set_ylabel(label);axes[row,col].set_title('Only quieter frames' if quiet else 'All valid frames');axes[row,col].grid(axis='y',alpha=.15)
 fig.suptitle('aws021: within-recording frame distributions; quiet = lower half of magnitude power');fig.tight_layout(rect=[0,0,1,.96]);save(fig,'aws021-quiet-frame-check.png')
 rows=[]
 for label,m in zip(['sleep','wake'],groups):
  for quiet in [False,True]:
   mm=m&(z['quiet'] if quiet else z['valid']);rows.append(dict(state=label,quiet_only=quiet,frames=int(mm.sum()),mag_floor=float(np.median(z['mag_floor'][mm])),jerk_floor=float(np.median(z['jerk_floor'][mm])),comb_score=float(np.median(z['jerk_trackscore'][mm])),flatness=float(np.median(z['jerk_flatness'][mm]))))
 pd.DataFrame(rows).to_csv(OUT/'aws021_frame_contrasts.csv',index=False)


def population():
 # Each point is a participant-stage median, avoiding pooling long recordings as independent samples.
 med=TEST.groupby(['cohort','subject_id','stage']).median(numeric_only=True).reset_index();fig,axes=plt.subplots(2,3,figsize=(16,10))
 for i,c in enumerate(['SleepAccel','SleepAccel-Clinical']):
  g=med[med.cohort==c]
  for stage,color,name in zip(range(4),COLORS,STAGES):
   q=g[g.stage==stage];axes[i,0].scatter(q.jerk_floor,q.jerk_trackscore,color=color,label=name,s=25,alpha=.75);axes[i,1].scatter(q.jerk_flatness,q.jerk_trackscore,color=color,s=25,alpha=.75);axes[i,2].scatter(q.burst_fraction,q.jerk_floor,color=color,s=25,alpha=.75)
  axes[i,0].set(xlabel='Jerk background proxy (dB)',ylabel='Comb prominence (dB)',title=c);axes[i,1].set(xlabel='Jerk flatness',ylabel='Comb prominence (dB)',title=c);axes[i,2].set(xlabel='Movement burst fraction',ylabel='Jerk background proxy (dB)',title=c)
 for ax in axes.flat:ax.grid(alpha=.15)
 axes[0,0].legend();fig.suptitle('Held-out adults: background, harmonic structure and bursts vary partly independently');fig.tight_layout(rect=[0,0,1,.96]);save(fig,'population-components.png')
 features=[('jerk_floor','Jerk background (dB)'),('jerk_trackscore','Comb prominence (dB)'),('jerk_comb_visible','Null-qualified comb fraction'),('burst_fraction','Movement burst fraction'),('jerk_fundamental_variability','Spacing variation (Hz)'),('mag_resp_fraction','0.1–0.5 Hz magnitude fraction')]
 fig,axes=plt.subplots(2,6,figsize=(23,9))
 for i,c in enumerate(['SleepAccel','SleepAccel-Clinical']):
  for j,(f,label) in enumerate(features):
   ax=axes[i,j];vals=[med[(med.cohort==c)&(med.stage==s)][f].dropna().to_numpy() for s in range(4)];box=ax.boxplot(vals,tick_labels=['W','L','D','R'],showfliers=False,patch_artist=True)
   for b,color in zip(box['boxes'],COLORS):b.set_facecolor(color);b.set_alpha(.3)
   for s,v in enumerate(vals):ax.scatter(np.full(len(v),s+1)+np.linspace(-.12,.12,len(v)),v,color=COLORS[s],s=12,alpha=.6)
   ax.set_title(label);ax.grid(axis='y',alpha=.15)
   if j==0:ax.set_ylabel(c)
 fig.suptitle('One point per held-out participant and PSG stage; missing stage/quality estimates remain absent');fig.tight_layout(rect=[0,0,1,.95]);save(fig,'population-stage-features.png')
 # Matched contrasts. Sleep-minus-wake, and REM-minus-NREM; per-participant averages over shared strata.
 mat=pd.read_csv(OUT/'matched_differences.csv',dtype={'subject_id':str});fig,axes=plt.subplots(2,4,figsize=(17,8));measures=[('jerk_floor','Jerk background (dB)'),('jerk_trackscore','Comb prominence (dB)'),('jerk_fundamental_variability','Spacing variation (Hz)'),('mag_resp_fraction','Low-band power fraction')]
 for i,comparison in enumerate(['quiet wake/sleep','NREM/REM']):
  for j,(f,label) in enumerate(measures):
   ax=axes[i,j]
   for k,c in enumerate(['SleepAccel','SleepAccel-Clinical']):
    a=mat[(mat.comparison==comparison)&(mat.cohort==c)][f].dropna();ax.scatter(a,np.full(len(a),k),s=22,alpha=.65,label=f'{c}, n={len(a)}');
    if len(a):ax.scatter(a.mean(),k,marker='D',color='black',s=35)
   ax.axvline(0,color='grey',lw=.8);ax.set_yticks([0,1],['SleepAccel','Clinical']);ax.set_xlabel(label+' difference');ax.set_title('Sleep − wake' if i==0 else 'REM − NREM');ax.grid(axis='x',alpha=.15)
 fig.suptitle('Within-person matched comparisons; dots are participants, diamonds are means');fig.tight_layout(rect=[0,0,1,.95]);save(fig,'matched-class-contrasts.png')
 # Before/after waking, all eligible transitions, with a wide separation from mixed transition epochs.
 rows=[]
 for (c,sid),g in TEST.groupby(['cohort','subject_id']):
  g=g.sort_values('epoch');y=g.stage.to_numpy();sleep=(y>0).astype(int)
  for k in range(10,len(g)-10):
   if np.mean(sleep[k-10:k])>=.9 and np.mean(sleep[k:k+10])<=.1:
    if rows and rows[-1]['subject_id']==sid and k-rows[-1]['transition_epoch']<10:continue
    for offset in range(-10,11):
     a=g.iloc[k+offset];rows.append(dict(cohort=c,subject_id=sid,transition_epoch=k,minutes=offset*.5,jerk_floor=a.jerk_floor,comb=a.jerk_trackscore,burst=a.burst_fraction))
 trans=pd.DataFrame(rows);trans.to_csv(OUT/'wake_transition_windows.csv',index=False)
 if len(trans):
  fig,axes=plt.subplots(1,3,figsize=(15,4))
  for j,(f,label) in enumerate([('jerk_floor','Background change (dB)'),('comb','Comb prominence change (dB)'),('burst','Burst fraction change')]):
   for c,color in zip(['SleepAccel','SleepAccel-Clinical'],['#387cb3','#d98532']):
    q=trans[trans.cohort==c].copy();base=q[q.minutes<=-2].groupby(['subject_id','transition_epoch'])[f].mean();q['relative']=[v-base.loc[(sid,k)] for v,sid,k in zip(q[f],q.subject_id,q.transition_epoch)];a=q.groupby(['subject_id','minutes']).relative.mean().unstack(0);mu=a.mean(1);axes[j].plot(mu.index,mu,color=color,label=f'{c}, {a.shape[1]} people');axes[j].fill_between(mu.index,a.quantile(.25,axis=1),a.quantile(.75,axis=1),color=color,alpha=.15)
   axes[j].axvline(0,color='grey',lw=.8);axes[j].set(xlabel='Minutes from sleep → wake',ylabel=label);axes[j].grid(alpha=.15)
  axes[0].legend(fontsize=8);fig.suptitle('Other held-out sleep → wake transitions; mean and participant interquartile range');fig.tight_layout(rect=[0,0,1,.93]);save(fig,'wake-transition-alignment.png')


def ablations():
 d=pd.read_csv(OUT/'model_subjects.csv',dtype={'subject_id':str});tasks=['sleep/wake','three-stage','NREM/REM'];order=list(dict.fromkeys(d.model));rng=np.random.default_rng(20261020);fig,axes=plt.subplots(2,3,figsize=(17,9))
 for i,c in enumerate(['SleepAccel','SleepAccel-Clinical']):
  for j,task in enumerate(tasks):
   ax=axes[i,j]
   for k,model in enumerate(order):
    a=d[(d.cohort==c)&(d.task==task)&(d.model==model)].kappa.dropna().to_numpy();boot=rng.choice(a,(1500,len(a)),replace=True).mean(1);lo,hi=np.quantile(boot,[.025,.975]);ax.plot([lo,hi],[k,k],color='#7f8a93',lw=2);ax.scatter(a.mean(),k,s=32,color='#3278a6');ax.scatter(a,np.full(len(a),k),s=8,alpha=.15,color='#3278a6')
   ax.set_yticks(range(len(order)),order if j==0 else []);ax.set_ylim(len(order)-.5,-.5);ax.axvline(0,color='grey',lw=.7);ax.set_title(c+' | '+task);ax.set_xlabel('Mean recording κ');ax.grid(axis='x',alpha=.15)
 fig.suptitle('Descriptor ablations: 21 fit, 9 validation, 30 held-out adults; bootstrap participant intervals');fig.tight_layout(rect=[0,0,1,.95]);save(fig,'heldout-ablation-kappa.png')
 if 'auroc' in d:
  fig,axes=plt.subplots(1,2,figsize=(14,5))
  for ax,task in zip(axes,['sleep/wake','NREM/REM']):
   for c,color in zip(['SleepAccel','SleepAccel-Clinical'],['#387cb3','#d98532']):
    q=d[(d.task==task)&(d.cohort==c)].groupby('model').auroc.mean().reindex(order);ax.plot(range(len(order)),q,marker='o',color=color,label=c)
   ax.axhline(.5,color='grey',lw=.8);ax.set_xticks(range(len(order)),[x[0] for x in order]);ax.set(xlabel='Ablation A → F',ylabel='Mean participant AUROC',title=task,ylim=(.35,1));ax.grid(alpha=.15);ax.legend()
  fig.tight_layout();save(fig,'heldout-ablation-auroc.png')
 # The user-selected difficult night, preserving its full scored denominator.
 pred=pd.read_csv(OUT/'heldout_predictions.csv.gz',dtype={'subject_id':str});g=D[D.subject_id=='aws021'];p=pred[(pred.subject_id=='aws021')&(pred.task=='sleep/wake')];fig,axes=plt.subplots(5,1,figsize=(15,10),sharex=True)
 axes[0].imshow(g.stage.to_numpy()[None],extent=[0,len(g)/120,0,1],aspect='auto',cmap=CMAP,norm=NORM);axes[0].set_yticks([]);axes[0].set_title('aws021: PSG and held-out sleep probabilities')
 for ax,model in zip(axes[1:4],['A magnitude','B + jerk background','F + respiration proxy']):
  q=p[p.model==model].set_index('epoch').reindex(range(len(g)));ax.plot(q.index/120,q.p1,label=model);t=d[(d.model==model)&(d.task=='sleep/wake')].threshold.iloc[0];ax.axhline(t,color='grey',ls='--',label=f'Validation threshold {t:.3f}');ax.set_ylim(0,1);ax.set_ylabel('P(sleep)');ax.legend(loc='upper right');shade(ax,g)
 q=np.load(Path.cwd()/'bidoze/results/cooperating-arms-2026-10-19/final7/draw0_seed0/predictions/testClinical_aws021.npz')['probabilities'].mean(0);axes[4].plot(np.arange(len(q))/120,q[:,1:4].sum(1),label='BiDoze final7: sum of sleep probabilities');axes[4].set_ylim(0,1);axes[4].set_ylabel('P(sleep)');axes[4].legend();axes[4].set_xlabel('Hours');shade(axes[4],g);fig.tight_layout();save(fig,'aws021-heldout-predictions.png')
 # Contribution diagrams use standardized coefficients of the full fit, not causal attribution.
 coef=pd.read_csv(OUT/'coefficients.csv');fig,axes=plt.subplots(1,2,figsize=(15,6))
 for ax,task in zip(axes,['sleep/wake','NREM/REM']):
  q=coef[(coef.task==task)&coef.model.str.startswith('F ')].copy();q=q.iloc[np.argsort(-abs(q.standardized_coefficient))[:15]].sort_values('standardized_coefficient');ax.barh(q.feature,q.standardized_coefficient,color=np.where(q.standardized_coefficient>0,'#3e88b5','#ce9655'));ax.axvline(0,color='grey');ax.set_title(task);ax.set_xlabel('Coefficient per training SD')
 fig.tight_layout();save(fig,'feature-coefficients.png')


def harmonic_alignment():
 target=np.arange(.5,3.501,.025);by=[]
 for (c,sid),g in TEST.groupby(['cohort','subject_id']):
  z=np.load(CACHE/f'{c}_subject_{sid}_features.npz');x=z['X'][0][:,:,1]*DB;bg=median_filter(x,size=(1,9),mode='nearest');res=x-bg;feat=np.load(OUT/f'features/{c}_{sid}.npz');freq=z['freq'];y=np.repeat(g.sort_values('epoch').stage.to_numpy(),15)
  # g only contains scored rows; use full labels on the original frame grid instead.
  full=D[(D.cohort==c)&(D.subject_id==sid)].sort_values('epoch');y=np.repeat(full.stage.to_numpy(),15);f0=feat['jerk_fundamental'];vis=feat['jerk_comb_visible']>0
  for stage in range(4):
   ix=np.flatnonzero((y==stage)&vis&np.isfinite(f0));ix=ix[::max(1,len(ix)//200)][:200]
   if len(ix)<5:continue
   aligned=np.array([np.interp(target*f0[i],freq,res[i],left=np.nan,right=np.nan) for i in ix]);by.append(dict(cohort=c,subject_id=sid,stage=stage,curve=np.nanmean(aligned,axis=0)))
 fig,axes=plt.subplots(1,2,figsize=(15,5))
 for ax,c in zip(axes,['SleepAccel','SleepAccel-Clinical']):
  for stage,color,name in zip(range(4),COLORS,STAGES):
   vals=[r['curve'] for r in by if r['cohort']==c and r['stage']==stage]
   if not vals:continue
   a=np.array(vals);ax.plot(target,np.nanmean(a,axis=0),color=color,label=f'{name}, {len(a)} people')
  for h in [1,2,3]:ax.axvline(h,color='grey',ls='--',lw=.7)
  ax.set(xlabel='Frequency / detected comb spacing',ylabel='Power above local background (dB)',title=c);ax.legend(fontsize=8);ax.grid(alpha=.15)
 fig.suptitle('Spacing-aligned spectra: conditional on passing the comb-quality gate; equal person weights');fig.tight_layout(rect=[0,0,1,.94]);save(fig,'harmonic-aligned-stage-profiles.png')


def heart_rate_reference():
 data=os.environ.get('SLEEP_ANALYSIS_DATA')
 if not data:return
 sid='759667';cohort='SleepAccel';hr=np.loadtxt(Path(data)/cohort/'cleaned_heartrate'/f'{sid}_cleaned_hr.out');psg=np.loadtxt(Path(data)/cohort/'cleaned_psg'/f'{sid}_cleaned_psg.out');rawt=np.loadtxt(Path(data)/cohort/'cleaned_accelerometer'/f'{sid}_cleaned_motion.out',usecols=0);phase=psg[0,0]%30;snapped=max(phase+30*np.ceil((rawt[0]-phase)/30),psg[0,0]);start=rawt[np.searchsorted(rawt,snapped)];z=np.load(OUT/f'features/{cohort}_{sid}.npz');g=D[(D.cohort==cohort)&(D.subject_id==sid)];t=z['time']/3600;fig,axes=plt.subplots(2,1,figsize=(14,7),sharex=True)
 axes[0].plot((hr[:,0]-start)/3600,hr[:,1],color='black',lw=.9,label='Watch heart rate');axes[0].scatter(t,z['jerk_fundamental']*60,s=3,color='#3d80b1',alpha=.6,label='Quality-gated comb spacing × 60');axes[0].set(ylabel='bpm',ylim=(30,140));axes[0].legend();axes[1].plot(t,z['jerk_trackscore'],label='Comb prominence');axes[1].plot(t,z['jerk_comb_excess_null'],label='Excess over shuffled null');axes[1].set(xlabel='Hours',ylabel='dB');axes[1].legend();fig.suptitle('759667 reference: check harmonic spacing against independent watch heart rate');fig.tight_layout(rect=[0,0,1,.95]);save(fig,'759667-comb-vs-watch-heart-rate.png')
 # Match at coarse 30-s bins; these are spacing comparisons, not beat-level HRV estimates.
 rows=[]
 for e in range(len(g)):
  m=(hr[:,0]>=start+e*30)&(hr[:,0]<start+(e+1)*30);ff=z['jerk_fundamental'][e*15:(e+1)*15]
  if m.any() and np.isfinite(ff).sum()>=5:rows.append(dict(epoch=e,watch_bpm=np.median(hr[m,1]),comb_bpm=np.nanmedian(ff)*60))
 pd.DataFrame(rows).to_csv(OUT/'759667_heart_rate_matches.csv',index=False)


def main():
 overview('SleepAccel-Clinical','aws021','aws021-components');aws_diagnostics();population();ablations();harmonic_alignment();heart_rate_reference()
 # Fixed reference plus development-only contrasts; no other test cases are selected for apparent classifier success.
 inventory=pd.read_csv(OUT/'inventory.csv',dtype={'subject_id':str});names=[('SleepAccel','759667')]
 dev=SCORED[SCORED.partition.isin(['fit','validation'])].groupby(['cohort','subject_id','stage']).median(numeric_only=True).reset_index()
 for c in ['SleepAccel','SleepAccel-Clinical']:
  for stage in [0,3]:
   q=dev[(dev.cohort==c)&(dev.stage==stage)].sort_values('jerk_trackscore',ascending=False)
   if len(q):names.append((c,q.iloc[0].subject_id))
 for c,sid in dict.fromkeys(names):overview(c,sid)
 pd.DataFrame(names,columns=['cohort','subject_id']).to_csv(OUT/'gallery_selection.csv',index=False)
 print('Figures complete',flush=True)
if __name__=='__main__':main()
