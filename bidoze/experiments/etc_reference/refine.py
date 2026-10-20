"""Convergence continuation and symmetric 15-epoch activity-head experiments.
Reflection tying is an explicit time-symmetry assumption, not a Gaussian parameterization.
"""
import run as r
import numpy as np,pandas as pd,json,time
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize,curve_fit
from pathlib import Path
OUT=r.OUT
split=json.loads((r.COOP/'control1/draw0_seed0/split.json').read_text())
parts={k:[r.read(0,x['dataset'],x['subject_id']) for x in split[k] if x['dataset'] in r.COHORT] for k in ['fit','validation','test']}
rng=np.random.default_rng(20261020)
for group in parts.values():
 for z in group:z['logactivity']=np.log(np.maximum(z['activity'],1e-15));z['rmsactivity']=np.sqrt(z['activity'])

def arrays(kind,K,cap=1600):
 train=np.concatenate([a[kind][a['valid']] for a in parts['fit']]);mu=train.mean(0);sd=np.maximum(train.std(0),1e-12)
 def pack(rs,limit):
  xx=[];yy=[]
  for a in rs:
   z=(a[kind]-mu)/sd;z[~a['valid']]=0
   x=np.lib.stride_tricks.sliding_window_view(np.pad(z,((K//2,K//2),(0,0))),K,axis=0).transpose(0,2,1)
   ok=(a['y']>=0)&(a['y']<4)&a['valid'];ok &=np.convolve((~a['valid']).astype(int),np.ones(K,dtype=int),'same')==0
   ix=np.flatnonzero(ok)
   if limit and len(ix)>limit:ix=np.sort(rng.choice(ix,limit,replace=False))
   xx.append(x[ix]);yy.append(a['y'][ix])
  return np.concatenate(xx),np.concatenate(yy)
 return mu,sd,pack(parts['fit'],cap),pack(parts['validation'],1200)

# Continue the validation-selected spectral model until the solver reports convergence.
# Recreate the exact original deterministic training rows (activity, logactivity, activity before spectral).
# The original RNG sampled all four designs sequentially; reproduce its consumed draws exactly.
rng=np.random.default_rng(20261020)
for kind in ['activity','logactivity','activity']:
 arrays(kind,31)
mu,sd,(xt,y),(vt,vy)=arrays('spectral',31)
y=np.where(y==0,0,np.where(y==3,2,1));vy=np.where(vy==0,0,np.where(vy==3,2,1));x=np.ascontiguousarray(xt.reshape(len(xt),-1));v=vt.reshape(len(vt),-1)
c=3;K=31;F=60;p=x.shape[1];n=len(x);sw=n/(c*np.bincount(y,minlength=c))[y];sw/=sw.mean();tag='spectral-3class-tv0.1';z=np.load(OUT/f'{tag}.npz');start=np.r_[z['standardized_kernel'].ravel(),z['bias']]
assert np.allclose(mu,z['mean']) and np.allclose(sd,z['scale'])
def obj(t):
 w=t[:c*p].reshape(c,K,F);b=t[c*p:];logits=x@w.reshape(c,p).T+b
 loss=np.sum(sw*(logsumexp(logits,axis=1)-logits[np.arange(n),y]))/n
 prob=softmax(logits,axis=1);prob[np.arange(n),y]-=1;prob*=sw[:,None]/n
 g=(prob.T@x).reshape(c,K,F);gb=prob.sum(0);loss+=.0001*np.sum(w*w)/2;g+=.0001*w
 for axis,scale in [(1,1),(2,.25)]:
  d=np.diff(w,axis=axis);den=np.sqrt(d*d+1e-6);fac=.1*scale/c;loss+=fac*np.sum(den-.001);q=fac*d/den
  lo=[slice(None)]*3;hi=lo.copy();lo[axis]=slice(None,-1);hi[axis]=slice(1,None);g[tuple(lo)]-=q;g[tuple(hi)]+=q
 return loss,np.r_[g.ravel(),gb]
t=time.monotonic();fit=minimize(obj,start,jac=True,method='L-BFGS-B',options={'maxiter':4000,'ftol':1e-9,'gtol':2e-6,'maxcor':20})
w=fit.x[:c*p].reshape(c,K,F);b=fit.x[c*p:];w-=w.mean(0);b-=b.mean()
np.savez_compressed(OUT/f'{tag}-converged.npz',standardized_kernel=w,kernel=w/sd[None,None,:],bias=b,mean=mu,scale=sd)
validation_ce=float(np.mean(logsumexp(v@w.reshape(c,p).T+b,axis=1)-(v@w.reshape(c,p).T+b)[np.arange(len(vy)),vy]))
info=dict(tag=tag+'-converged',success=bool(fit.success),iterations=fit.nit,message=str(fit.message),validation_ce=validation_ce,seconds=time.monotonic()-t)
(OUT/'spectral-continuation.json').write_text(json.dumps(info,indent=2));print(info,flush=True)
# Keep original sweep immutable; produce converged plots using a separate plotting table and file prefix.
table=pd.read_csv(OUT/'logreg_sweep.csv');new=table[(table.kind=='spectral')&(table.lambda_tv==.1)].copy();new['tag']=tag+'-converged';new['validation_ce']=validation_ce;table=pd.concat([table,new],ignore_index=True);r.plot_kernels(table)
rows=[]
for i,a in enumerate(parts['test']):
 z=(a['spectral']-mu)/sd;z[~a['valid']]=0;calls=(r.windows(z)@w.reshape(c,p).T+b).argmax(1);mask=(a['y']>=0)&(a['y']<4);truth=np.where(a['y']==0,0,np.where(a['y']==3,2,1));cm=np.zeros((3,3));np.add.at(cm,(truth[mask],calls[mask]),1);po=np.trace(cm)/cm.sum();pe=(cm.sum(0)*cm.sum(1)).sum()/cm.sum()**2
 rows.append(dict(recording=i,subject_id=split['test'][i]['subject_id'],cohort=r.COHORT[split['test'][i]['dataset']],stage_kappa=(po-pe)/(1-pe),**r.metrics(a['y'],calls),recall_wake=cm[0,0]/cm[0].sum(),recall_nrem=cm[1,1]/cm[1].sum(),recall_rem=cm[2,2]/cm[2].sum()))
pd.DataFrame(rows).to_csv(OUT/'spectral-converged-test.csv',index=False)

# Narrow, reflection-tied, free-shape kernels: TV suppresses noise; quadratic variation avoids flat TV plateaus.
rows=[];kernels={};K=15;h=K//2;idx=np.minimum(np.arange(K),np.arange(K)[::-1]);c=2
for kind in ['activity','rmsactivity','logactivity']:
 mu,sd,(xt,y),(vt,vy)=arrays(kind,K);y=(y>0).astype(int);vy=(vy>0).astype(int);x=np.zeros((len(xt),h+1));v=np.zeros((len(vt),h+1))
 for i in range(K):x[:,idx[i]]+=xt[:,i,0];v[:,idx[i]]+=vt[:,i,0]
 n=len(x);sw=n/(2*np.bincount(y))[y];sw/=sw.mean()
 for tv,qv in [(0,0),(.001,.01),(.01,.1),(.03,.3),(.1,1)]:
  def obj(t):
   coef=t[:-1];w=coef[idx];logit=x@coef+t[-1];loss=np.mean(sw*(np.logaddexp(0,logit)-y*logit));res=(1/(1+np.exp(-np.clip(logit,-50,50)))-y)*sw/n;g=x.T@res;gb=res.sum()
   # w represents sleep − wake. Sign is reversed in the plotted wake contrast.
   dw=np.diff(w);den=np.sqrt(dw*dw+1e-6);loss+=tv*np.sum(den-.001)+qv*np.sum(dw*dw)/2+.0001*np.sum(w*w)/2
   gw=.0001*w;dgrad=tv*dw/den+qv*dw;gw[:-1]-=dgrad;gw[1:]+=dgrad
   np.add.at(g,idx,gw);return loss,np.r_[g,gb]
  fit=minimize(obj,np.zeros(h+2),jac=True,method='L-BFGS-B',options={'maxiter':2500,'ftol':1e-12,'gtol':1e-7});coef=fit.x[:-1];b=fit.x[-1];logit=v@coef+b;ce=float(np.mean(np.logaddexp(0,logit)-vy*logit));wake=-coef[idx]/sd[0]
  xidx=np.arange(K)-h
  def gauss(t,amp,sigma,offset):return offset+amp*np.exp(-.5*(t/sigma)**2)
  pars,_=curve_fit(gauss,xidx,wake,p0=[max(wake)-min(wake),1,min(wake)],bounds=([0,.1,-np.inf],[np.inf,20,np.inf]),maxfev=5000);fitted=gauss(xidx,*pars);r2=1-np.sum((wake-fitted)**2)/np.sum((wake-wake.mean())**2)
  tag=f'{kind}-symmetric-tv{tv:g}-qv{qv:g}';np.savez_compressed(OUT/f'{tag}.npz',wake_minus_sleep=wake,gaussian_fit=fitted,fit_parameters=pars,mean=mu,scale=sd,bias=b,coefficient=coef)
  kernels[tag]=wake;test=[]
  for a in parts['test']:
   z=(a[kind]-mu)/sd;z[~a['valid']]=0;z=np.lib.stride_tricks.sliding_window_view(np.pad(z,((h,h),(0,0))),K,axis=0).transpose(0,2,1)[:,:,0];logit=z@coef[idx]+b;test.append(r.metrics(a['y'],np.where(logit>0,1,0)))
  pd.DataFrame(test).to_csv(OUT/f'{tag}-test.csv',index=False)
  row=dict(tag=tag,kind=kind,tv=tv,qv=qv,success=bool(fit.success),iterations=fit.nit,validation_ce=ce,gaussian_r2=r2,gaussian_sigma_epochs=pars[1],roughness=np.sum(np.diff(wake,n=2)**2)/np.sum(wake*wake),**pd.DataFrame(test)[r.METRICS].mean().to_dict());rows.append(row);print(row,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'symmetric_sweep.csv',index=False)
fig,axes=r.plt.subplots(1,3,figsize=(16,4))
for ax,kind in zip(axes,['activity','rmsactivity','logactivity']):
 g=pd.DataFrame(rows);g=g[g.kind==kind]
 # Display the smoothest kernel within .01 nats of its best validation CE.
 selected=g[g.validation_ce<=g.validation_ce.min()+.01].sort_values('roughness').iloc[0]
 (OUT/f'{kind}-symmetric-selected.json').write_text(json.dumps(selected.to_dict(),indent=2))
 for _,a in g.iterrows():ax.plot(np.arange(K),kernels[a.tag],alpha=.35,label=f'TV={a.tv:g}, QV={a.qv:g}')
 z=np.load(OUT/f'{selected.tag}.npz');ax.plot(np.arange(K),z['wake_minus_sleep'],color='black',lw=2,label='Selected smooth head');ax.plot(np.arange(K),z['gaussian_fit'],'--',color='red',label=f'Gaussian fit R²={selected.gaussian_r2:.3f}');ax.set_title(kind);ax.set_xlabel('Kernel index');ax.set_ylabel('wake − sleep kernel');ax.legend(fontsize=6)
fig.suptitle('Free-shape 15-epoch logistic heads with explicit reflection tying; Gaussian fitted afterward');fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'figures'/'symmetric-gaussian-recovery.png',dpi=160)
