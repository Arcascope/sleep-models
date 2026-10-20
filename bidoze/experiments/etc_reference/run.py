"""CPU-only ETC reference and smooth linear heads. Run from the sleep-models root.
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python bidoze/results/etc-reference-2026-10-20/setup/run.py
"""
from pathlib import Path
import os,json,hashlib,sys,importlib.util,time
import numpy as np
import pandas as pd
from pisces_lite.datasets.constants import PSG_5C_MAPPING_TO_WLDR
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path.cwd(); OUT=ROOT/'bidoze/results/etc-reference-2026-10-20'; SETUP=Path(__file__).resolve().parent
COOP=ROOT/'bidoze/results/cooperating-arms-2026-10-19'
TRI=ROOT.parent/'sleep-models-tridoze/bidoze/results'
CACHE=TRI/'ensemble-size-2026-10-17/setup/cache'
spec=importlib.util.spec_from_file_location('etc_existing',SETUP/'etc_existing.py');etc=importlib.util.module_from_spec(spec);sys.modules[spec.name]=etc;spec.loader.exec_module(etc)
OUT.mkdir(exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)
METRICS=['kappa_sleep_wake','sleep_sensitivity','wake_specificity','tst_bias_minutes','tst_mae_minutes']
NAMES=['Sleep/wake κ','Sleep sensitivity','Wake specificity','TST bias (min)','TST absolute error (min)']
COHORT={'testSleepAccel':'SleepAccel','testClinical':'SleepAccel-Clinical','trainSleepAccel':'SleepAccel','trainClinical':'SleepAccel-Clinical'}
records={}
def read(draw,ds,sid):
 key=(draw,ds,str(sid))
 if key in records:return records[key]
 p=CACHE/f'draw{draw}/12Hz_log_PSD_aligned/{ds}_subject_{sid}_features.npz'
 z=np.load(p);x=z['X'][0];raw_y=z['y'][0];y=np.vectorize(lambda a: PSG_5C_MAPPING_TO_WLDR.get(int(a),int(a)),otypes=[np.int64])(raw_y); n=len(y);r=len(x)//n
 assert len(x)==n*r and x.shape[1]==60
 present=~np.all(x==-1,axis=(1,2));finite=np.isfinite(x).all(axis=(1,2));present &=finite
 count=present.reshape(n,r).sum(1);valid=count>0
 # Use channel 0 only. Undo the producer's natural log before physical power sum.
 power=np.maximum(np.exp(x[:,:,0])-np.finfo(np.float64).eps,0)
 power[~present]=0
 linear=power.sum(1).reshape(n,r).sum(1)/np.maximum(count,1)
 spectral=np.where(present[:,None],x[:,:,0],0).reshape(n,r,60).sum(1)/np.maximum(count[:,None],1)
 for arr in [linear,spectral]:
  arr[~valid]=np.median(arr[valid],axis=0) if valid.any() else 0
 historical=etc.wake_score(etc.epoch_energy(x,n))
 physical=etc.wake_score(linear)
 result=dict(y=y,raw_y=raw_y,valid=valid,activity=linear[:,None],spectral=spectral,historical=historical,physical=physical,source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 records[key]=result;return result

def metrics(y,pred):
 valid=(y>=0)&(y<4);y=y[valid];pred=pred[valid];ys=y>0;ps=(pred>0)&(pred<4)
 # Gap predictions count as errors in kappa, and as neither sleep nor wake in recalls.
 yt=np.where(y>0,1,0);pt=np.where(pred==4,2,np.where(pred>0,1,0));cm=np.zeros((3,3));np.add.at(cm,(yt,pt),1)
 po=np.trace(cm)/len(y);pe=(cm.sum(0)*cm.sum(1)).sum()/len(y)**2
 bias=.5*(ps.sum()-ys.sum())
 return dict(kappa_sleep_wake=(po-pe)/(1-pe) if pe<1 and len(np.unique(yt))>1 else np.nan,sleep_sensitivity=float(np.mean(ps[ys])) if ys.any() else np.nan,wake_specificity=float(np.mean(pred[~ys]==0)) if (~ys).any() else np.nan,tst_bias_minutes=bias,tst_mae_minutes=abs(bias))

def threshold(recs,kind):
 vals=np.concatenate([r[kind][(r['y']>=0)&(r['y']<4)&r['valid']] for r in recs])
 choices=np.unique(np.quantile(vals,np.linspace(0,1,501)))
 score=[]
 for t in choices:
  score.append(np.nanmean([metrics(r['y'],np.where(r[kind]>=t,0,1))['kappa_sleep_wake'] for r in recs]))
 j=int(np.nanargmax(score));return float(choices[j]),float(score[j])

def etc_reference():
 rows=[];cal=[]
 for d in range(5):
  split=json.loads((COOP/f'control1/draw{d}_seed0/split.json').read_text())
  keys={part:{(x['dataset'].replace('train','').replace('test',''),x['subject_id']) for x in split[part] if x['dataset']!='weaverDraw'} for part in ['fit','validation','test']}
  assert not(keys['fit']&keys['validation'] or keys['fit']&keys['test'] or keys['validation']&keys['test'])
  val=[read(d,x['dataset'],x['subject_id']) for x in split['validation'] if x['dataset'] in COHORT]
  for kind in ['historical','physical']:
   t,k=threshold(val,kind);cal.append(dict(draw=d,kind=kind,threshold=t,validation_kappa=k,validation_recordings=len(val)))
   for x in split['test']:
    if x['dataset'] not in COHORT:continue
    r=read(d,x['dataset'],x['subject_id']);calls=np.where(r[kind]>=t,0,1)
    rows.append(dict(draw=d,cohort=COHORT[x['dataset']],subject_id=str(x['subject_id']),kind=kind,n_scored=int(((r['y']>=0)&(r['y']<4)).sum()),missing_feature_epochs=int((~r['valid']).sum()),**metrics(r['y'],calls)))
  print('ETC completed draw',d,flush=True)
 pd.DataFrame(cal).to_csv(OUT/'thresholds.csv',index=False)
 df=pd.DataFrame(rows);df.to_csv(OUT/'etc_subjects.csv',index=False)
 return df

def figure(df,name,order=None):
 order=order or list(dict.fromkeys(df.model));fig,axes=plt.subplots(2,5,figsize=(22,max(6,len(order)*.28+2)),squeeze=False)
 for i,c in enumerate(['SleepAccel','SleepAccel-Clinical']):
  for j,m in enumerate(METRICS):
   ax=axes[i,j]
   for y,model in enumerate(order):
    v=df[(df.model==model)&(df.cohort==c)][m].dropna().to_numpy()
    if not len(v):continue
    ax.plot([v.min(),v.max()],[y,y],color='#8b9299');ax.scatter(v,np.full(len(v),y),s=12,alpha=.55);ax.scatter(v.mean(),y,color='black',marker='D',s=20)
   ax.axvline(0,color='grey',lw=.8);ax.set_yticks(range(len(order)),order if j==0 else []);ax.set_ylim(len(order)-.5,-.5);ax.set_title(c+'\n'+NAMES[j]);ax.grid(axis='x',alpha=.18)
 fig.suptitle('Model − ETC (historical Gaussian): same draw and test recordings',fontsize=15)
 fig.text(.5,.012,'Points: draw/subset means; diamond: their mean; lines: min–max. Participants overlap; ranges are descriptive, not confidence intervals. Rates are fractions.',ha='center',fontsize=10)
 fig.tight_layout(rect=[0,.035,1,.95]);fig.savefig(OUT/'figures'/name,dpi=140);plt.close(fig)

def comparisons(ref):
 base=ref[ref.kind=='historical']; b=base.groupby(['draw','cohort'])[METRICS].mean()
 allframes=[]
 frames=[]
 for p in sorted(COOP.glob('*/draw*_seed0/analysis/subjects.csv')):
  d=int(p.parents[1].name.split('_')[0][4:]);x=pd.read_csv(p,dtype={'subject_id':str});x['draw']=d;x['model']=p.parents[2].name;frames.append(x)
 x=pd.concat(frames);x['experiment']='cooperating-arms';allframes.append(x)
 for name,p in [('joint-ensemble',ROOT/'bidoze/results/bidoze-joint-ensemble-2026-10-15/analysis/subjects.csv'),('tridoze',TRI/'tridoze-2026-10-16/analysis/full-report/subjects.csv')]:
  x=pd.read_csv(p,dtype={'subject_id':str});x['experiment']=name;x['tst_bias_minutes']=x.tst_error_minutes;x['tst_mae_minutes']=abs(x.tst_error_minutes);allframes.append(x)
 merged=pd.concat(allframes,ignore_index=True)
 paired=merged.merge(base[['draw','cohort','subject_id',*METRICS]],on=['draw','cohort','subject_id'],suffixes=('','_etc'),validate='many_to_one',how='left',indicator=True)
 assert (paired._merge=='both').all(),paired[paired._merge!='both'][['experiment','draw','cohort','subject_id']]
 for m in METRICS:paired['delta_'+m]=paired[m]-paired[m+'_etc']
 paired.to_csv(OUT/'paired_subjects.csv',index=False)
 for exp,g in paired.groupby('experiment'):
  means=g.groupby(['model','draw','cohort'])[['delta_'+m for m in METRICS]].mean().rename(columns={'delta_'+m:m for m in METRICS}).reset_index()
  means.to_csv(OUT/f'{exp}_deltas.csv',index=False)
  # Keep the overview readable; exhaustive rows remain in the CSV.
  if exp=='cooperating-arms':order=['control1','final3','final5','final7','exchange7','shared7','control1_5m','final5_5m','exchange5_5m','shared5_5m']
  else:order=[m for m in means.model.unique() if 'member' not in m and not m.startswith('independent seed')]
  figure(means,f'{exp}-etc-effects.png',order)
 # Oct 17 has saved per-draw aggregate rows for every subset; subtraction commutes with their per-recording means.
 p=TRI/'ensemble-size-2026-10-17/analysis/subsets.csv';s=pd.read_csv(p)
 s['model']=s.apply(lambda r: f"{r['family']} {r['kind']} K={r['K']}"+(f" {r['label']}" if pd.notna(r['label']) else ''),axis=1)
 s.to_csv(OUT/'ensemble-size-source.csv',index=False)
 for m in METRICS:s[m]=[v-b.loc[(d,c),m] for v,d,c in zip(s[m],s.draw,s.cohort)]
 s.to_csv(OUT/'ensemble-size_deltas.csv',index=False)
 figure(s,'ensemble-size-etc-effects.png')
 # Descriptive subject-level paired spread after averaging each person's repeated appearances.
 person=paired.groupby(['experiment','model','cohort','subject_id'])[['delta_'+m for m in METRICS]].mean()
 stats=person.groupby(['experiment','model','cohort']).agg(['mean','std','count'])
 for m in METRICS:
  stats[('delta_'+m,'loa_low')]=stats[('delta_'+m,'mean')]-1.96*stats[('delta_'+m,'std')]
  stats[('delta_'+m,'loa_high')]=stats[('delta_'+m,'mean')]+1.96*stats[('delta_'+m,'std')]
 stats.to_csv(OUT/'person_delta_spread.csv')
 print('paired subjects',len(paired),'models',paired.groupby('experiment').model.nunique().to_dict(),flush=True)

# Linear heads use 31 epochs (15.5 min), centred and zero padded after training-only standardization.
K=31

def design(r,kind):
 a=r[kind]
 if kind=='logactivity':a=np.log(np.maximum(r['activity'],1e-15))
 return a

def windows(a):
 z=np.pad(a,((K//2,K//2),(0,0)),mode='constant');return np.lib.stride_tricks.sliding_window_view(z,K,axis=0).transpose(0,2,1).reshape(len(a),-1)

def fit_heads():
 split=json.loads((COOP/'control1/draw0_seed0/split.json').read_text())
 parts={k:[read(0,x['dataset'],x['subject_id']) for x in split[k] if x['dataset'] in COHORT] for k in ['fit','validation','test']}
 assert [len(parts[k]) for k in parts]==[21,9,30]
 results=[]; rng=np.random.default_rng(20261020)
 for kind,classes in [('activity',2),('logactivity',2),('activity',3),('spectral',3)]:
  # Set the scalar/log feature shape before accessing it.
  for group in parts.values():
   for r in group:r['logactivity']=np.log(np.maximum(r['activity'],1e-15))
  train=np.concatenate([r[kind][r['valid']] for r in parts['fit']]);mu=train.mean(0);sd=train.std(0);sd=np.maximum(sd,1e-12)
  def pack(rs,cap=None):
   xx=[];yy=[]
   for r in rs:
    a=(r[kind]-mu)/sd; a[~r['valid']]=0
    x=windows(a);v=(r['y']>=0)&(r['y']<4)&r['valid']
    # Do not let fitted temporal windows cross a feature gap.
    vv=np.convolve((~r['valid']).astype(int),np.ones(K,dtype=int),'same')==0;v &=vv
    ids=np.flatnonzero(v)
    if cap and len(ids)>cap:ids=np.sort(rng.choice(ids,cap,replace=False))
    y=r['y'][ids];y=np.where(y==0,0,np.where(y==3,2,1)) if classes==3 else np.where(y>0,1,0)
    xx.append(x[ids]);yy.append(y)
   return np.ascontiguousarray(np.concatenate(xx)),np.concatenate(yy)
  x,y=pack(parts['fit'],1600);v,vy=pack(parts['validation'],1200)
  n,p=x.shape;f=len(mu); count=np.bincount(y,minlength=classes);weights=n/(classes*count);sw=weights[y];sw/=sw.mean()
  previous=np.zeros(classes*p+classes)
  # Smooth absolute TV approximates |d| with sqrt(d²+1e-6); L2 fixes scale and softmax gauge.
  for lam in [0,.001,.01,.1,1.0]:
   started=time.monotonic()
   def objective(theta):
    w=theta[:classes*p].reshape(classes,K,f);bias=theta[classes*p:];flat=w.reshape(classes,p)
    logits=x@flat.T+bias;lse=logsumexp(logits,axis=1)
    loss=np.sum(sw*(lse-logits[np.arange(n),y]))/n
    pr=softmax(logits,axis=1);pr[np.arange(n),y]-=1;pr*=sw[:,None]/n
    grad=(pr.T@x).reshape(classes,K,f);gb=pr.sum(0)
    loss+=.0001*np.sum(w*w)/2;grad+=.0001*w
    for axis,scale in [(1,1),(2,.25)]:
     if w.shape[axis]<2:continue
     diff=np.diff(w,axis=axis);den=np.sqrt(diff*diff+1e-6)
     factor=lam*scale/classes
     loss+=factor*np.sum(den-1e-3);g=factor*diff/den
     low=[slice(None)]*3;high=low.copy();low[axis]=slice(None,-1);high[axis]=slice(1,None)
     grad[tuple(low)]-=g;grad[tuple(high)]+=g
    return loss,np.r_[grad.ravel(),gb]
   opt=minimize(objective,previous,jac=True,method='L-BFGS-B',options={'maxiter':350,'ftol':1e-10,'gtol':2e-6,'maxcor':12});previous=opt.x
   w=opt.x[:classes*p].reshape(classes,K,f);bias=opt.x[classes*p:];w-=w.mean(0,keepdims=True);bias-=bias.mean()
   vp=softmax(v@w.reshape(classes,p).T+bias,axis=1);vloss=float(np.mean(-np.log(np.clip(vp[np.arange(len(vy)),vy],1e-12,1))))
   tag=f'{kind}-{classes}class-tv{lam:g}'
   raw=w/sd[None,None,:]
   np.savez_compressed(OUT/f'{tag}.npz',standardized_kernel=w,kernel=raw,bias=bias,mean=mu,scale=sd,classes=classes,lambda_tv=lam)
   predrows=[]
   for i,r in enumerate(parts['test']):
    a=(r[kind]-mu)/sd;a[~r['valid']]=0;pr=softmax(windows(a)@w.reshape(classes,p).T+bias,axis=1);calls=pr.argmax(1)
    # Logistic class 2 is REM, and class 1 is NREM; both count as sleep.
    m=metrics(r['y'],calls);truth=np.where(r['y']==0,0,np.where(r['y']==3,2,1)) if classes==3 else (r['y']>0).astype(int)
    mask=(r['y']>=0)&(r['y']<4);accuracy=float(np.mean(calls[mask]==truth[mask]));cm=np.zeros((classes,classes));np.add.at(cm,(truth[mask],calls[mask]),1);po=np.trace(cm)/cm.sum();pe=(cm.sum(0)*cm.sum(1)).sum()/cm.sum()**2
    predrows.append(dict(recording=i,accuracy=accuracy,stage_kappa=(po-pe)/(1-pe),**m))
   pd.DataFrame(predrows).to_csv(OUT/f'{tag}-test.csv',index=False)
   contrast=w[0]-w[1] if classes==2 else w[0]-w[1]-w[2]
   hf=float(np.sum(np.diff(contrast,n=2,axis=0)**2)/(np.sum(contrast**2)+1e-15))
   row=dict(tag=tag,kind=kind,classes=classes,lambda_tv=lam,train_epochs=n,validation_epochs=len(vy),validation_ce=vloss,success=bool(opt.success),iterations=opt.nit,message=str(opt.message),seconds=time.monotonic()-started,roughness=hf,**pd.DataFrame(predrows)[METRICS+['stage_kappa','accuracy']].mean().to_dict());results.append(row)
   pd.DataFrame(results).to_csv(OUT/'logreg_sweep.csv',index=False)
   print(tag,'val CE',round(vloss,4),'roughness',round(hf,5),'iters',opt.nit,opt.message,flush=True)
  # All test metrics above are descriptive; select exclusively by validation cross entropy.
  candidates=[r for r in results if r['kind']==kind and r['classes']==classes];best=min(candidates,key=lambda r:r['validation_ce']);(OUT/f'{kind}-{classes}class-selected.json').write_text(json.dumps(best,indent=2))
 plot_kernels(pd.DataFrame(results))

def plot_kernels(table):
 for (kind,classes),g in table.groupby(['kind','classes']):
  names=['wake','sleep'] if classes==2 else ['wake','nrem','rem'];best=g.sort_values('validation_ce').iloc[0]
  fig,axes=plt.subplots(2,classes,figsize=(5*classes,8),squeeze=False)
  for _,r in g.iterrows():
   z=np.load(OUT/f'{r.tag}.npz');w=z['kernel'];contrasts=np.array([w[i]-np.sum(np.delete(w,i,axis=0),axis=0) for i in range(classes)])
   for row,arr in enumerate([w,contrasts]):
    for j in range(classes):
     # For 60-bin kernels, the line is the common additive log-power direction.
     # Show all frequency-specific temporal vectors separately in the supplementary line panel.
     y=arr[j].sum(1);axes[row,j].plot(np.arange(K),y,label=f'TV {r.lambda_tv:g}'+(' (selected)' if r.tag==best.tag else ''),lw=2 if r.tag==best.tag else 1)
     axes[row,j].set_title(names[j] if row==0 else names[j]+' − ('+' + '.join(n for n in names if n!=names[j])+')');axes[row,j].axhline(0,color='grey',lw=.5);axes[row,j].set_xlabel('Kernel index');axes[row,j].set_ylabel('Kernel vector' if kind!='spectral' else 'Σ frequency coefficients');axes[row,j].grid(alpha=.15)
  axes[0,0].legend(fontsize=8);fig.suptitle(f'{kind}, {classes} classes; sum-to-zero class gauge; original feature units');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'figures'/f'{kind}-{classes}class-kernels.png',dpi=150);plt.close(fig)
  if kind=='spectral':
   z=np.load(OUT/f'{best.tag}.npz');w=z['kernel'];arr=np.r_[w,np.array([w[i]-np.sum(np.delete(w,i,axis=0),axis=0) for i in range(classes)])]
   fig,axes=plt.subplots(2,3,figsize=(16,8));vmax=np.max(abs(arr))
   for i,ax in enumerate(axes.flat):
    im=ax.imshow(arr[i].T,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax);ax.set_title((names[i] if i<3 else names[i-3]+' − others'));ax.set_xlabel('Kernel index');ax.set_ylabel('Frequency index')
   fig.colorbar(im,ax=axes.ravel().tolist(),label='Coefficient per unit natural-log PSD',shrink=.7);fig.suptitle(f'Selected 31 × 60 spectral head; TV={best.lambda_tv:g}');fig.savefig(OUT/'figures'/'spectral-3class-heatmaps.png',dpi=150);plt.close(fig)
   fig,axes=plt.subplots(2,3,figsize=(16,8))
   for i,ax in enumerate(axes.flat):
    for f in range(60):ax.plot(np.arange(K),arr[i,:,f],color=plt.cm.viridis(f/59),alpha=.55,lw=.7)
    ax.set_title(names[i] if i<3 else names[i-3]+' − others');ax.set_xlabel('Kernel index');ax.set_ylabel('Kernel vector');ax.axhline(0,color='grey',lw=.5)
   fig.suptitle('Every frequency: 60 temporal kernel vectors per panel (selected spectral head)');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'figures'/'spectral-3class-all-frequency-lines.png',dpi=150);plt.close(fig)
   # For log(P+epsilon), the derivative wrt linear power is W/(P+epsilon).
   reference=np.exp(z['mean']);jac=w/reference[None,None,:]
   np.savez_compressed(OUT/'spectral-power-jacobian.npz',kernel=jac,reference_power=reference)
   fig,axes=plt.subplots(1,3,figsize=(15,4))
   for i,ax in enumerate(axes):ax.plot(np.arange(K),jac[i].sum(1));ax.set_title(names[i]);ax.set_xlabel('Kernel index');ax.set_ylabel('Σ ∂logit / ∂linear PSD')
   fig.suptitle('Local linear-power derivative at training geometric-mean PSD; state dependent');fig.tight_layout();fig.savefig(OUT/'figures'/'spectral-power-jacobian.png',dpi=150);plt.close(fig)

if __name__=='__main__':
 if '--heads-only' not in sys.argv:
  ref=etc_reference();comparisons(ref)
 if '--reference-only' not in sys.argv:fit_heads()
 (OUT/'provenance.json').write_text(json.dumps(dict(etc_source_commit='7e18ba76dbeb9ff3d6764e694e5bb821fae5bd61',etc_source_sha256=hashlib.sha256((SETUP/'etc_existing.py').read_bytes()).hexdigest(),kernel=etc.DEFAULT_KERNEL.__dict__,inputs=[dict(draw=k[0],dataset=k[1],subject_id=k[2],sha256=v['sha256']) for k,v in records.items()]),indent=2))
