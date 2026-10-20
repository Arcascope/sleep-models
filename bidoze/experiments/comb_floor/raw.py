"""Raw-timestamp 30/60-s spectra, sample-rate checks and physical amplitudes for aws021.
Raw data paths stay in invocation/environment rather than the published log.
"""
from pathlib import Path
import os,json,warnings
import numpy as np,pandas as pd
import senpy
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract import OUT,CACHE,DB
DATA=Path(os.environ['SLEEP_ANALYSIS_DATA']);SID='aws021';COHORT='SleepAccel-Clinical'

def spectrum(t,sig,at,win):
 m=(t>=at-win/2)&(t<at+win/2)
 if m.sum()<100:return None
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');s=senpy.compute_nufft_spectrogram(t[m],sig[m],win,0,kind='psd',origin_s=at-win/2,empty_windows='keep')
 f=np.asarray(s.frequencies);p=np.asarray(s.Sxx)[0];return f,p,float(np.max(np.diff(t[m]))),int(m.sum())

def main():
 a=pd.read_csv(DATA/COHORT/'cleaned_accelerometer'/f'{SID}.csv').to_numpy(float);psg=pd.read_csv(DATA/COHORT/'cleaned_psg'/f'{SID}.csv').to_numpy(float)
 a=a[np.isfinite(a).all(1)];a=a[np.argsort(a[:,0],kind='stable')];a=a[np.r_[True,np.diff(a[:,0])>0]]
 # Match the official preparation contract: snap overlapping sources to the PSG grid, then use the first retained accelerometer sample as NUFFT origin.
 phase=psg[0,0]%30;start=max(phase+30*np.ceil((a[0,0]-phase)/30),psg[0,0]);end=min(phase+30*np.floor((a[-1,0]-phase)/30),psg[-1,0]);a=a[(a[:,0]>=start)&(a[:,0]<=end)]
 zero=a[0,0];t=a[:,0]-zero;axes=a[:,1:];mag=np.linalg.norm(axes,axis=1);j=senpy.compute_jerk(t,*axes.T);tj=np.asarray(j.timestamps_s);jerk=np.asarray(j.jerk)
 cached=np.load(CACHE/f'{COHORT}_subject_{SID}_features.npz');raw_y=psg[:len(cached['y'][0]),1].astype(int);raw_y[raw_y==4]=5;scored=cached['y'][0]>=0;assert np.array_equal(raw_y[scored],cached['y'][0][scored])
 meta=dict(raw_psg_epoch0_seconds=float(start),first_retained_sample_seconds=float(zero),raw_seconds_minus_plot_hours_offset=float(zero),raw_psg_mapping='0 W,1 N1,2 N2,3 N3,4 REM; dataset maps 4 to 5 before WLDR',cache_label_match=True,median_fs_hz=float(1/np.median(np.diff(t))))
 rates=[]
 for lo in np.arange(0,5.75,.25):
  m=(t>=lo*3600)&(t<(lo+.25)*3600);dt=np.diff(t[m])
  if len(dt):rates.append(dict(hours=lo,median_fs_hz=1/np.median(dt),mean_fs_hz=len(dt)/900,max_gap_seconds=dt.max(),samples=len(dt)+1))
 pd.DataFrame(rates).to_csv(OUT/'raw_aws021_sample_rates.csv',index=False)
 # Confirm senpy's jerk is the per-sample vector difference, rather than divide-by-dt.
 expected=np.r_[0,np.linalg.norm(np.diff(axes,axis=0),axis=1)];np.testing.assert_allclose(jerk,expected,rtol=1e-10,atol=1e-12);meta['jerk_verified_unit']='g/sample'
 # Normalize the native PSD's integrated power on synthetic 1-mg sinusoids at this recording's timestamps.
 at=2.8*3600;sel=(t>=at-30)&(t<at+30);cal=[]
 for hz in [1.1,3.3]:
  ff,pp,_,_=spectrum(t[sel],.001*np.sin(2*np.pi*hz*t[sel]),at,60);band=abs(ff-hz)<.15;ratio=pp[band].sum()*np.diff(ff).mean()/(.001**2/2);cal.append(dict(hz=hz,power_normalization=ratio,recovered_rms_mg=np.sqrt(pp[band].sum()*np.diff(ff).mean()/ratio)*1000))
 norm=float(np.mean([x['power_normalization'] for x in cal]));meta['normalization']=cal
 # Time-frequency surfaces from individual windows; no spectra combined across windows.
 matrices={};windowrows=[]
 for win in [30,60]:
  centres=np.arange(2*3600+win/2,4.5*3600,30);surfaces=[]
  for cname,tt,sig in [('mag',t,mag),('jerk',tj,jerk)]:
   allp=[];allvalid=[]
   for centre in centres:
    result=spectrum(tt,sig,float(centre),win)
    if result is None:
     allp.append(None);allvalid.append(False);continue
    f,p,gap,n=result;keep=(f>=.05)&(f<=6);p=p[keep]/norm
    good=gap<=.1 and n>=win*40
    allp.append(p if good else np.full_like(p,np.nan));allvalid.append(good)
    if win==60:
     b=(f[keep]>=.5)&(f[keep]<=5.5);floor=np.quantile(p[b],.2);windowrows.append(dict(hours=centre/3600,channel=cname,good=good,max_gap_seconds=gap,samples=n,band_rms_mg=np.sqrt(np.sum(p[b])*np.diff(f).mean())*1000,floor_equivalent_rms_mg=np.sqrt(floor*5)*1000))
   ref=next(q for q in allp if q is not None);matrices[f'{cname}_{win}']=np.stack([q if q is not None else np.full_like(ref,np.nan) for q in allp]);matrices[f'freq_{win}']=f[keep];matrices[f'time_{win}']=centres/3600
  print('raw surfaces',win,flush=True)
 np.savez_compressed(OUT/'features/raw-aws021-spectra.npz',**matrices);pd.DataFrame(windowrows).to_csv(OUT/'raw_aws021_windows.csv',index=False);(OUT/'raw_aws021_audit.json').write_text(json.dumps(meta,indent=2))
 fig,axesfig=plt.subplots(4,1,figsize=(15,11),sharex=True)
 for i,(c,win) in enumerate([('mag',30),('jerk',30),('mag',60),('jerk',60)]):
  p=matrices[f'{c}_{win}'];db=10*np.log10(np.maximum(p,1e-25));vmin,vmax=np.nanquantile(db,[.05,.98]);im=axesfig[i].pcolormesh(matrices[f'time_{win}'],matrices[f'freq_{win}'],db.T,shading='auto',cmap='viridis',vmin=vmin,vmax=vmax);axesfig[i].set_ylabel('Hz');axesfig[i].set_title(f'aws021 | {c} | {win}-s windows, 30-s step');fig.colorbar(im,ax=axesfig[i],label='dB calibrated PSD')
 axesfig[-1].set_xlabel('Hours from retained recording start (same origin as model plot)');fig.tight_layout();fig.savefig(OUT/'figures/aws021-raw-30s-60s.png',dpi=150);plt.close(fig)
 # Single-window spectra before and after awakening, both logarithmic and linear physical scales.
 examples=[('Sleep',2.8),('Wake',3.7)];fig,axs=plt.subplots(2,2,figsize=(14,9));amplitudes=[]
 for label,hour in examples:
  for ch,(c,tt,sig,unit) in enumerate([('mag',t,mag,'mg'),('jerk',tj,jerk,'mg/sample')]):
   f,p,gap,n=spectrum(tt,sig,hour*3600,60);p=p/norm;keep=(f>=.1)&(f<=6);f=f[keep];p=p[keep];db=10*np.log10(np.maximum(p,1e-25));base=median_filter(db,size=31,mode='nearest')
   axs[ch,0].plot(f,db,label=f'{label}, {hour:.1f} h');axs[ch,1].plot(f,np.sqrt(p)*1000,label=f'{label}, {hour:.1f} h');axs[ch,0].set_ylabel('dB PSD');axs[ch,1].set_ylabel(f'{unit}/√Hz');axs[ch,0].set_title(c+' | one 60-s spectrum');axs[ch,1].set_title(c+' | amplitude spectral density')
   mask=(f>=.5)&(f<=5.5);floor=float(np.quantile(p[mask],.2));amplitudes.append(dict(state=label,hours=hour,channel=c,max_gap_seconds=gap,total_band_rms_mg=np.sqrt(p[mask].sum()*np.diff(f).mean())*1000,floor_equivalent_rms_mg=np.sqrt(floor*5)*1000))
 for ax in axs.flat:ax.set_xlabel('Hz');ax.legend();ax.grid(alpha=.15)
 fig.tight_layout();fig.savefig(OUT/'figures/aws021-single-window-physical-spectra.png',dpi=160);plt.close(fig);pd.DataFrame(amplitudes).to_csv(OUT/'raw_aws021_physical.csv',index=False)
 # Raw small movements and jerk impulses, same scales before/after awakening.
 fig,axs=plt.subplots(2,2,figsize=(14,7),sharex='col')
 for col,(label,hour) in enumerate(examples):
  centre=hour*3600;m=(t>=centre)&(t<centre+10);mj=(tj>=centre)&(tj<centre+10);axs[0,col].plot(t[m]-centre,(mag[m]-np.median(mag[m]))*1000,lw=.7);axs[1,col].plot(tj[mj]-centre,jerk[mj]*1000,lw=.7);axs[0,col].set_title(f'{label}, {hour} h');axs[0,col].set_ylabel('Demeaned |a| (mg)');axs[1,col].set_ylabel('Jerk (mg/sample)');axs[1,col].set_xlabel('Seconds')
 for row in axs:
  lims=np.array([a.get_ylim() for a in row]);lo,hi=lims[:,0].min(),lims[:,1].max()
  for ax in row:ax.set_ylim(lo,hi);ax.grid(alpha=.15)
 fig.tight_layout();fig.savefig(OUT/'figures/aws021-raw-motion.png',dpi=160);plt.close(fig)
 print('Raw analysis complete',meta,flush=True)
if __name__=='__main__':main()
