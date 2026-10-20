"""Finish scalar convergence, render final kernel diagnostics and audit input/metric contracts."""
import run as r
import numpy as np,pandas as pd,json,hashlib,sys
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize
OUT=r.OUT
# Continuation uses the same deterministic sample as the third design in run.py.
split=json.loads((r.COOP/'control1/draw0_seed0/split.json').read_text());parts={k:[r.read(0,a['dataset'],a['subject_id']) for a in split[k] if a['dataset'] in r.COHORT] for k in ['fit','validation','test']};rng=np.random.default_rng(20261020)
for rs in parts.values():
 for a in rs:a['logactivity']=np.log(np.maximum(a['activity'],1e-15))
def pack(kind,rs,limit,mu,sd):
 xx=[];yy=[]
 for a in rs:
  z=(a[kind]-mu)/sd;z[~a['valid']]=0;x=r.windows(z);ok=(a['y']>=0)&(a['y']<4)&a['valid'];ok &= np.convolve((~a['valid']).astype(int),np.ones(31,dtype=int),'same')==0;ix=np.flatnonzero(ok)
  if len(ix)>limit:ix=np.sort(rng.choice(ix,limit,replace=False))
  xx.append(x[ix]);yy.append(a['y'][ix])
 return np.concatenate(xx),np.concatenate(yy)
for kind in ['activity','logactivity','activity']:
 a=np.concatenate([a[kind][a['valid']] for a in parts['fit']]);mu=a.mean(0);sd=np.maximum(a.std(0),1e-12);x,y=pack(kind,parts['fit'],1600,mu,sd);v,vy=pack(kind,parts['validation'],1200,mu,sd)
y=np.where(y==0,0,np.where(y==3,2,1));vy=np.where(vy==0,0,np.where(vy==3,2,1));c=3;p=31;n=len(y);sw=n/(3*np.bincount(y))[y];sw/=sw.mean();z=np.load(OUT/'activity-3class-tv0.1.npz');start=np.r_[z['standardized_kernel'].ravel(),z['bias']]
def obj(t):
 w=t[:3*p].reshape(3,p);b=t[3*p:];log=x@w.T+b;loss=np.sum(sw*(logsumexp(log,axis=1)-log[np.arange(n),y]))/n;prob=softmax(log,axis=1);prob[np.arange(n),y]-=1;prob*=sw[:,None]/n;g=prob.T@x;gb=prob.sum(0);loss+=.0001*np.sum(w*w)/2;g+=.0001*w;d=np.diff(w,axis=1);den=np.sqrt(d*d+1e-6);loss+=.1/3*np.sum(den-.001);q=.1/3*d/den;g[:,:-1]-=q;g[:,1:]+=q;return loss,np.r_[g.ravel(),gb]
opt=minimize(obj,start,jac=True,method='L-BFGS-B',options={'maxiter':6000,'ftol':1e-10,'gtol':2e-6,'maxcor':20});w=opt.x[:3*p].reshape(3,p,1);b=opt.x[3*p:];w-=w.mean(0);b-=b.mean();np.savez_compressed(OUT/'activity-3class-tv0.1-converged.npz',kernel=w/sd[None,None,:],standardized_kernel=w,bias=b,mean=mu,scale=sd)
log=v@w.reshape(3,p).T+b;ce=float(np.mean(logsumexp(log,axis=1)-log[np.arange(len(vy)),vy]));info=dict(success=bool(opt.success),iterations=opt.nit,message=str(opt.message),validation_ce=ce);(OUT/'activity-continuation.json').write_text(json.dumps(info,indent=2));print(info,flush=True)
# Six requested scalar curves for the converged smooth model, with the unregularized fit as reference.
fig,axes=r.plt.subplots(2,3,figsize=(15,7));names=['wake','nrem','rem']
for tag,label in [('activity-3class-tv0','No TV'),('activity-3class-tv0.1-converged','TV=0.1 (converged)')]:
 w=np.load(OUT/f'{tag}.npz')['kernel'][:,:,0];arr=np.r_[w,np.array([w[i]-np.sum(np.delete(w,i,axis=0),axis=0) for i in range(3)])]
 for i,ax in enumerate(axes.flat):ax.plot(np.arange(31),arr[i],label=label);ax.set_title(names[i] if i<3 else names[i-3]+' − others');ax.set_xlabel('Kernel index');ax.set_ylabel('Kernel vector');ax.axhline(0,color='grey',lw=.5)
axes[0,0].legend();fig.suptitle('Scalar linear-power activity: individual class kernels and requested contrasts');fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'figures'/'activity-3class-converged.png',dpi=150);r.plt.close(fig)
# Ensure every selected spectral figure uses the converged continuation rather than the earlier, numerically similar iterate.
table=pd.read_csv(OUT/'logreg_sweep.csv');new=table[(table.kind=='spectral')&(table.lambda_tv==.1)].copy();new['tag']='spectral-3class-tv0.1-converged';new['validation_ce']=json.loads((OUT/'spectral-continuation.json').read_text())['validation_ce'];table=table[table.tag!='spectral-3class-tv0.1'];r.plot_kernels(pd.concat([table,new],ignore_index=True))
# Exact one-vs-rest log odds has a local derivative, not a fixed A-B-C kernel.
z=np.load(OUT/'spectral-3class-tv0.1-converged.npz');w=z['kernel'];b=z['bias'];power=np.exp(z['mean']);true=[]
for i in range(3):
 others=[j for j in range(3) if j!=i];q=softmax(b[others]);true.append(w[i]-np.sum(w[others]*q[:,None,None],axis=0))
true=np.array(true);np.savez_compressed(OUT/'spectral-local-one-vs-rest.npz',log_psd_kernel=true,power_jacobian=true/power[None,None,:],reference_power=power)
fig,axes=r.plt.subplots(1,3,figsize=(15,4))
for i,ax in enumerate(axes):
 for f in range(60):ax.plot(np.arange(31),true[i,:,f],alpha=.5,lw=.7,color=r.plt.cm.viridis(f/59))
 ax.set_title(names[i]+' vs rest (local log odds)');ax.set_xlabel('Kernel index');ax.set_ylabel('Kernel vector')
fig.tight_layout();fig.savefig(OUT/'figures'/'spectral-local-one-vs-rest.png',dpi=150);r.plt.close(fig)
# Subject-level paired descriptive limits after averaging repeated appearances of each participant.
paired=pd.read_csv(OUT/'paired_subjects.csv',dtype={'subject_id':str});g=paired[paired.experiment=='cooperating-arms'];person=g.groupby(['model','cohort','subject_id'])[['delta_'+m for m in r.METRICS]].mean().reset_index();order=['control1','final3','final5','final7','exchange7','shared7','control1_5m','final5_5m','exchange5_5m','shared5_5m'];fig,axes=r.plt.subplots(2,5,figsize=(22,8))
for i,cohort in enumerate(['SleepAccel','SleepAccel-Clinical']):
 for j,m in enumerate(r.METRICS):
  ax=axes[i,j]
  for k,model in enumerate(order):
   a=person[(person.model==model)&(person.cohort==cohort)]['delta_'+m].dropna();mean=a.mean();span=1.96*a.std();ax.plot([mean-span,mean+span],[k,k],color='#8b9299');ax.scatter(a,np.full(len(a),k),alpha=.2,s=9,color='#3274a1');ax.scatter(mean,k,color='black',marker='D',s=20)
  ax.axvline(0,color='grey',lw=.7);ax.set_yticks(range(len(order)),order if j==0 else []);ax.set_ylim(len(order)-.5,-.5);ax.set_title(cohort+'\n'+r.NAMES[j]);ax.grid(axis='x',alpha=.15)
fig.suptitle('Model − historical ETC: between-participant paired differences',fontsize=15);fig.text(.5,.01,'Each point averages that participant’s test appearances; lines: mean ± 1.96 SD of those differences. Descriptive limits, not a CI or prediction interval for a new fit.',ha='center');fig.tight_layout(rect=[0,.035,1,.95]);fig.savefig(OUT/'figures'/'cooperating-arms-etc-person-effects.png',dpi=150);r.plt.close(fig)
# Validate saved prediction labels against ETC cache labels and independently reproduce the repository's metrics for all 50 fits.
from arcasleep_bidoze.adult import hybrid_calls
from arcasleep.training.config import load_metrics_config
from pisces_lite.metrics.logger import MetricsLogger
import arcasleep.evaluation
import tempfile
logger=MetricsLogger(load_metrics_config(r.ROOT/'bidoze/config/metrics.json')[0]);audit=[]
for path in sorted(r.COOP.glob('*/draw*_seed0/analysis/subjects.csv')):
 d=int(path.parents[1].name.split('_')[0][4:]);out=path.parents[1];saved=pd.read_csv(path,dtype={'subject_id':str});div=np.load(out/'stage_divisor.npy')
 for row in saved.to_dict('records'):
  q=np.load(out/'predictions'/f"{row['dataset']}_{row['subject_id']}.npz");truth=q['labels'];a=r.read(d,row['dataset'],row['subject_id']);np.testing.assert_array_equal(truth,a['y']);p=q['probabilities'].mean(0);calls=hybrid_calls(p,p,div);ours=r.metrics(truth,calls)
  for m in r.METRICS:np.testing.assert_allclose(ours[m],row[m],atol=1e-10,equal_nan=True)
  audit.append(dict(draw=d,model=row['arm'],subject_id=row['subject_id'],label_match=True,metric_match=True))
 if out.parent.name!='control1':continue
 # ETC metrics independently passed through the actual metric registry on one complete test cohort per draw.
 for _,rr in saved.iterrows():
  a=r.read(d,rr.dataset,rr.subject_id);cal=pd.read_csv(OUT/'thresholds.csv');t=cal[(cal.draw==d)&(cal.kind=='historical')].threshold.iloc[0];calls=np.where(a['historical']>=t,0,1);p=np.eye(5)[calls]
  with tempfile.TemporaryDirectory() as tmp:official=logger.record(r.Path(tmp)/'m.csv',y_true=a['y'],y_pred=calls,y_pred_proba=p,run_id='',fold='',epoch=0,subject_id='',dataset='',training_dataset='')
  off={m['metric']:m['value'] for m in official};ours=r.metrics(a['y'],calls)
  for m in r.METRICS[:3]:np.testing.assert_allclose(ours[m],off[m],atol=1e-10,equal_nan=True)
  # TST is evaluated only on the identical PSG-scored epochs, as in the source per-recording reports.
  np.testing.assert_allclose(ours['tst_bias_minutes'],off['tst_error_minutes'],atol=1e-10,equal_nan=True)

pd.DataFrame(audit).to_csv(OUT/'metric_audit.csv',index=False)
# Verify the older draw identities against each source cache, not just the subject IDs.
older=r.ROOT/'bidoze/results/bidoze-joint-ensemble-2026-10-15/setup/cache'
cacheaudit=[]
for d in range(5):
 for a in json.loads((r.COOP/f'control1/draw{d}_seed0/split.json').read_text())['test']:
  expected=r.read(d,a['dataset'],a['subject_id']);p=older/f"draw{d}/12Hz_log_PSD_aligned/{a['dataset']}_subject_{a['subject_id']}_features.npz";z=np.load(p);np.testing.assert_array_equal(z['y'][0],expected['raw_y']);np.testing.assert_array_equal(z['X'],np.load(expected['source'])['X']);cacheaudit.append(dict(draw=d,subject_id=a['subject_id'],features_match=True,labels_match=True))
pd.DataFrame(cacheaudit).to_csv(OUT/'cross_run_cache_audit.csv',index=False)
(OUT/'audit.json').write_text(json.dumps(dict(cooperating_rows_checked=len(audit),cross_run_cache_matches=len(cacheaudit),all_checks_passed=True),indent=2));print('audits complete',len(audit),len(cacheaudit),flush=True)
