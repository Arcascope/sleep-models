"""Separate broadband background, moving harmonic combs and intermittent motion.
Run from the original sleep-models checkout; no raw samples are exported.
"""
from pathlib import Path
import json,hashlib,argparse
import numpy as np,pandas as pd
from scipy.ndimage import median_filter,uniform_filter1d
from pisces_lite.datasets.constants import PSG_5C_MAPPING_TO_WLDR
ROOT=Path.cwd();OUT=ROOT/'bidoze/results/comb-floor-2026-10-20';CACHE=ROOT/'feature_cache/12Hz_log_PSD_aligned'
SPLIT=ROOT/'bidoze/results/cooperating-arms-2026-10-19/control1/draw0_seed0/split.json'
COLORS=['#eaa12d','#467ac1','#324080','#28b9bd'];STAGES=['Wake','Light','Deep','REM']
DB=10/np.log(10);GRID=np.arange(.6,2.001,.025)

def comb_bank(logp,f):
 """Local frequency background plus a search over three harmonics, with inter-peak controls."""
 db=logp*DB;bg=median_filter(db,size=(1,9),mode='nearest');res=db-bg
 def sample(freq):
  # Linear interpolation along frequency, applied to every time row.
  hi=np.searchsorted(f,freq).clip(1,len(f)-1);lo=hi-1;w=(freq-f[lo])/(f[hi]-f[lo]);return res[:,lo]*(1-w)+res[:,hi]*w
 peaks=np.stack([sample(GRID*h) for h in [1,2,3]],axis=2)
 between=(sample(GRID*1.5)+sample(GRID*2.5))/2
 score=peaks.mean(2)-.5*between
 best=score.argmax(1);bestscore=score[np.arange(len(score)),best];peakcount=(peaks[np.arange(len(score)),best]>=3).sum(1)
 # Viterbi ridge follows slowly varying spacing but can switch if spectral evidence warrants it.
 penalty=.01*((GRID[:,None]-GRID[None,:])*60)**2
 value=score[0].copy();back=np.zeros(score.shape,dtype=np.int16)
 for t in range(1,len(score)):
  candidates=value[:,None]-penalty;back[t]=candidates.argmax(0);value=candidates.max(0)+score[t]
 ids=np.zeros(len(score),int);ids[-1]=value.argmax()
 for t in range(len(score)-1,0,-1):ids[t-1]=back[t,ids[t]]
 trackscore=score[np.arange(len(score)),ids];trackpeaks=peaks[np.arange(len(score)),ids];visible=(trackscore>=3)&((trackpeaks>=3).sum(1)>=2)
 # Frequency-scrambled residuals preserve each frame's residual distribution but remove harmonic placement.
 rng=np.random.default_rng(20261020);sel=np.linspace(0,len(res)-1,min(160,len(res))).astype(int);null=[]
 for _ in range(32):
  shuffled=np.take_along_axis(res[sel],rng.random(res[sel].shape).argsort(1),axis=1)
  def interpnull(freq):
   hi=np.searchsorted(f,freq).clip(1,len(f)-1);lo=hi-1;w=(freq-f[lo])/(f[hi]-f[lo]);return shuffled[:,lo]*(1-w)+shuffled[:,hi]*w
  nullscore=np.stack([interpnull(GRID*h) for h in [1,2,3]],axis=2).mean(2)-.25*(interpnull(GRID*1.5)+interpnull(GRID*2.5));null.append(nullscore.max(1))
 cutoff=float(np.quantile(np.concatenate(null),.95));visible &=trackscore>=cutoff
 return dict(null_cutoff=cutoff,bg=bg,residual=res,bank=score,best=GRID[best],bestscore=bestscore,track=GRID[ids],trackscore=trackscore,visible=visible,peakcount=peakcount)

def rolling(a,n=11,mode='median'):
 return pd.Series(a).rolling(n,center=True,min_periods=2).median().to_numpy() if mode=='median' else pd.Series(a).rolling(n,center=True,min_periods=3).std().to_numpy()

def one(cohort,sid,part):
 p=CACHE/f'{cohort}_subject_{sid}_features.npz';z=np.load(p);raw=z['y'][0];y=np.vectorize(lambda a:PSG_5C_MAPPING_TO_WLDR.get(int(a),int(a)),otypes=[int])(raw)
 x=z['X'][0].astype(np.float64);n=len(y);ratio=len(x)//n;assert ratio==15 and x.shape[1:]==(60,2)
 freq=z['freq'];valid=np.isfinite(x).all((1,2))&~np.all(x==-1,axis=(1,2));x[~valid]=np.nan
 # Do not bridge feature gaps in tracking: each uninterrupted segment is processed independently.
 filled=np.where(valid[:,None,None],x,np.nanmedian(x[valid],axis=0));mask=(freq>=.5)&(freq<=5.5)
 frame={};banks={}
 for ch,name in enumerate(['mag','jerk']):
  a=filled[:,:,ch];pwr=np.maximum(np.exp(a)-np.finfo(float).eps,0);band=pwr[:,mask]
  frame[name+'_floor']=np.quantile(a[:,mask],.2,axis=1)*DB
  frame[name+'_flatness']=np.exp(a[:,mask].mean(1))/np.maximum(band.mean(1),1e-30)
  frame[name+'_logpower']=np.log(np.maximum(pwr.sum(1)*(freq[1]-freq[0]),1e-30))*DB
  lf=np.log(freq[mask]);lf-=lf.mean();frame[name+'_slope']=(a[:,mask]@lf)/np.sum(lf*lf)
  parts=[];starts=np.flatnonzero(valid&~np.r_[False,valid[:-1]]);ends=np.flatnonzero(valid&~np.r_[valid[1:],False])+1
  bank=comb_bank(a,freq)
  for lo,hi in zip(starts,ends):
   local=bank if lo==0 and hi==len(a) else comb_bank(a[lo:hi],freq)
   for k in ['track','trackscore','visible']:bank[k][lo:hi]=local[k]
  bank['visible'][~valid]=False;banks[name]=bank
  for k in ['bestscore','trackscore']:frame[name+'_'+k]=bank[k]
  frame[name+'_comb_visible']=bank['visible'].astype(float)
  frame[name+'_comb_excess_null']=bank['trackscore']-bank['null_cutoff']
  frame[name+'_fundamental']=np.where(bank['visible'],bank['track'],np.nan)
  # Coarse respiratory descriptors are explicitly proxies at 0.1-Hz resolution.
  low=(freq>=.1)&(freq<=.5);frame[name+'_resp_power']=np.log(np.maximum(pwr[:,low].sum(1)*(freq[1]-freq[0]),1e-30))*DB
  frame[name+'_resp_fraction']=pwr[:,low].sum(1)/np.maximum(pwr.sum(1),1e-30)
  frame[name+'_resp_peak']=freq[low][a[:,low].argmax(1)]
 # Burst threshold is unsupervised, recording-specific median + 3 MAD of magnitude log power.
 energy=frame['mag_logpower'];med=np.median(energy[valid]);mad=np.median(abs(energy[valid]-med));burst=energy>med+3*max(mad,.1);quiet=energy<=med
 rows=[]
 for e in range(n):
  sl=slice(e*ratio,(e+1)*ratio);ok=valid[sl]
  row=dict(cohort=cohort,subject_id=sid,partition=part,epoch=e,hours=e/120,stage=int(y[e]),valid_fraction=ok.mean())
  for name,a in frame.items():
   arr=a[sl][ok];row[name]=(float(np.nanmean(arr)) if name.endswith('_comb_visible') else float(np.nanmedian(arr))) if np.isfinite(arr).any() else np.nan
  for ch in ['mag','jerk']:
   a=frame[ch+'_floor'][sl][ok];row[ch+'_floor_low']=np.quantile(a,.2) if len(a) else np.nan
   a=frame[ch+'_logpower'][sl][ok];row[ch+'_power_range']=np.quantile(a,.9)-np.quantile(a,.1) if len(a) else np.nan
   a=frame[ch+'_fundamental'][sl][ok];row[ch+'_fundamental_iqr']=np.nanquantile(a,.75)-np.nanquantile(a,.25) if np.isfinite(a).sum()>1 else np.nan
   q=quiet[sl]&ok;row[ch+'_quiet_floor']=float(np.median(frame[ch+'_floor'][sl][q])) if q.any() else np.nan
  row['burst_fraction']=float(burst[sl][ok].mean()) if ok.any() else np.nan;row['quiet_fraction']=float(quiet[sl][ok].mean()) if ok.any() else np.nan
  rows.append(row)
 df=pd.DataFrame(rows)
 for name in ['mag_floor','jerk_floor','jerk_trackscore','jerk_comb_visible','jerk_fundamental','mag_resp_peak']:
  df[name+'_context']=rolling(df[name].to_numpy());df[name+'_variability']=rolling(df[name].to_numpy(),mode='std')
 df['jerk_comb_x_floor']=df.jerk_trackscore*df.jerk_floor
 df['jerk_comb_x_flatness']=df.jerk_trackscore*df.jerk_flatness
 # Per-recording unsupervised offsets supplement, rather than replace, absolute levels.
 for name in ['mag_floor','jerk_floor']:df[name+'_relative']=df[name]-df[name].quantile(.2)
 target=OUT/'features'/f'{cohort}_{sid}.npz'
 # Derived descriptors and small diagnostic arrays only; raw acceleration and raw frame PSD are not copied.
 np.savez_compressed(target,fundamental_grid=GRID,time=np.arange(len(x))*2,
   mag_bank=banks['mag']['bank'].astype('f4'),jerk_bank=banks['jerk']['bank'].astype('f4'),
   **{k:v.astype('f4') for k,v in frame.items()},valid=valid,burst=burst,quiet=quiet)
 return df,dict(cohort=cohort,subject_id=sid,partition=part,cache_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),epochs=n,valid_frames=int(valid.sum()),scored_epochs=int(((y>=0)&(y<4)).sum()))

def main():
 split=json.loads(SPLIT.read_text());allrows=[];manifest=[];seen=set()
 for part in ['fit','validation','test']:
  for a in split[part]:
   ds=a['dataset'];sid=a['subject_id']
   if ds not in ['trainClinical','testClinical','trainSleepAccel','testSleepAccel']:continue
   cohort='SleepAccel-Clinical' if 'Clinical' in ds else 'SleepAccel';assert (cohort,sid) not in seen;seen.add((cohort,sid))
   df,meta=one(cohort,sid,part);allrows.append(df);manifest.append(meta);print(part,cohort,sid,len(df),flush=True)
 result=pd.concat(allrows,ignore_index=True);result.to_csv(OUT/'epochs.csv',index=False);pd.DataFrame(manifest).to_csv(OUT/'inventory.csv',index=False)
 assert [sum(m['partition']==p for m in manifest) for p in ['fit','validation','test']]==[21,9,30]
 (OUT/'method.json').write_text(json.dumps(dict(f0_grid_hz=GRID.tolist(),harmonics=[1,2,3],background_median_bins=9,visibility_score_db=3,visibility_harmonics=2,null_frequency_permutations=32,null_sample_frames=160,null_quantile=.95,viterbi_penalty_bpm_squared=.01,sampling_seed=20261020,frame_seconds=10,frame_step_seconds=2,epoch_seconds=30),indent=2));print('Wrote',len(result),'epochs from',len(manifest),'participants',flush=True)
if __name__=='__main__':main()
