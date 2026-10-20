"""Pictures of directional low-frequency information and a rotation-invariance check."""
from pathlib import Path
import os,json
import numpy as np,pandas as pd,senpy
from scipy.signal import periodogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract import OUT
from raw import spectrum
DATA=Path(os.environ['SLEEP_ANALYSIS_DATA'])

def examples():
 d=pd.read_csv(OUT/'highres_matched.csv',dtype={'subject_id':str});selections=[];rotation=[]
 for c,sid in [('SleepAccel','759667'),('SleepAccel-Clinical','aws002')]:
  g=d[(d.cohort==c)&(d.subject_id==sid)];pairs=[]
  for pid,h in g.groupby('pair_id'):
   a=h[h.binary_stage==0].iloc[0];b=h[h.binary_stage==1].iloc[0];pairs.append((float(b.trace_resp_entropy-a.trace_resp_entropy),pid))
  target=np.median([q[0] for q in pairs]);pid=min(pairs,key=lambda q:abs(q[0]-target))[1];selected=g[g.pair_id==pid].sort_values('binary_stage');selections.extend(selected[['cohort','subject_id','epoch','binary_stage','pair_id','trace_resp_entropy','trace_resp_fraction']].to_dict('records'))
  if c=='SleepAccel':a=np.loadtxt(DATA/c/'cleaned_accelerometer'/f'{sid}_cleaned_motion.out');psg=np.loadtxt(DATA/c/'cleaned_psg'/f'{sid}_cleaned_psg.out')
  else:a=pd.read_csv(DATA/c/'cleaned_accelerometer'/f'{sid}.csv').to_numpy(float);psg=pd.read_csv(DATA/c/'cleaned_psg'/f'{sid}.csv').to_numpy(float)
  phase=psg[0,0]%30;start=max(phase+30*np.ceil((a[0,0]-phase)/30),psg[0,0]);a=a[a[:,0]>=start];t=a[:,0]-a[0,0];ax=a[:,1:];fig,axes=plt.subplots(2,3,figsize=(17,8),sharex='col')
  for row,r in enumerate(selected.to_dict('records')):
   at=r['epoch']*30+15;pp=[];ff=None
   for k,name in enumerate(['x','y','z']):
    f,p,_,_=spectrum(t,ax[:,k],at,60);m=(f>=.05)&(f<=6);f=f[m];p=p[m];pp.append(p);ff=f;axes[row,0].plot(f,10*np.log10(np.maximum(p,1e-30)),label=name,lw=.8)
   trace=np.sum(pp,axis=0);f,mag,_,_=spectrum(t,np.linalg.norm(ax,axis=1),at,60);m=(f>=.05)&(f<=6);mag=mag[m];f=f[m];axes[row,0].plot(f,10*np.log10(np.maximum(mag,1e-30)),label='|a|',color='black',lw=1.3)
   band=(ff>=.12)&(ff<=.5);axes[row,1].plot(ff[band],trace[band]/trace[band].sum(),label='Sum of axis PSDs');axes[row,1].plot(f[band],mag[band]/mag[band].sum(),label='PSD of |a|');axes[row,2].plot(ff,10*np.log10(np.maximum(trace,1e-30)),label='Total-axis PSD');axes[row,2].plot(f,10*np.log10(np.maximum(mag,1e-30)),label='Magnitude PSD',color='black');label='NREM' if r['binary_stage']==0 else 'REM'
   axes[row,0].set_title(f'{label}, {r["epoch"]/120:.2f} h: axes and magnitude');axes[row,1].set_title(f'0.12–0.5 Hz shape | trace entropy {r["trace_resp_entropy"]:.3f}');axes[row,2].set_title('Total-axis power preserves directional motion')
   # A fixed rotation mixes the raw axes; the summed PSD should remain unchanged.
   angle=.713;Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])@np.array([[1,0,0],[0,np.cos(.431),-np.sin(.431)],[0,np.sin(.431),np.cos(.431)]])
   rotated=ax@Q.T;qq=[]
   for k in range(3):
    fr,pr,_,_=spectrum(t,rotated[:,k],at,60);qq.append(pr[(fr>=.05)&(fr<=6)])
   error=float(np.linalg.norm(np.sum(qq,axis=0)-trace)/np.linalg.norm(trace));rotation.append(dict(cohort=c,subject_id=sid,epoch=r['epoch'],trace_rotation_relative_l2_error=error));assert error<1e-8
  for row in axes:
   row[0].set(xlim=(.05,.6),ylabel='dB raw PSD');row[1].set(xlim=(.12,.5),ylabel='Fraction per frequency bin');row[2].set(xlim=(.05,2),ylabel='dB raw PSD')
   for a in row:a.set_xlabel('Hz');a.legend(fontsize=7);a.grid(alpha=.15)
  fig.suptitle(f'{c} {sid}: one matched development pair nearest that person’s median entropy difference');fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'figures'/f'low-frequency-axes-{c}-{sid}.png',dpi=150);plt.close(fig)
 pd.DataFrame(selections).to_csv(OUT/'representation_examples.csv',index=False);pd.DataFrame(rotation).to_csv(OUT/'rotation_audit.csv',index=False)


def synthetic():
 fs=50;t=np.arange(0,60,1/fs);f0=.25;A=.001;parallel=np.column_stack([np.zeros(len(t)),np.zeros(len(t)),1+A*np.sin(2*np.pi*f0*t)]);perp=np.column_stack([A*np.sin(2*np.pi*f0*t),np.zeros(len(t)),np.ones(len(t))]);fig,axes=plt.subplots(2,3,figsize=(17,8));audit=[]
 for row,(name,a) in enumerate([('Along gravity',parallel),('Across gravity',perp)]):
  mag=np.linalg.norm(a,axis=1);jerk=np.r_[0,np.linalg.norm(np.diff(a,axis=0),axis=1)];ff,pm=periodogram(mag,fs=fs,scaling='density');trace=sum(periodogram(a[:,k],fs=fs,scaling='density')[1] for k in range(3));fj,pj=periodogram(jerk,fs=fs,scaling='density');sel=t<12
  axes[row,0].plot(t[sel],(mag[sel]-mag.mean())*1e6);axes[row,0].set(title=name+' | magnitude',xlabel='Seconds',ylabel='µg deviation in |a|')
  axes[row,1].plot(ff,10*np.log10(np.maximum(trace,1e-30)),label='Sum of axis PSDs');axes[row,1].plot(ff,10*np.log10(np.maximum(pm,1e-30)),label='PSD of |a|');axes[row,1].set(xlim=(.1,.8),ylim=(-180,-40),xlabel='Hz',ylabel='dB PSD');axes[row,1].legend()
  axes[row,2].plot(fj,10*np.log10(np.maximum(pj,1e-30)));axes[row,2].set(xlim=(.1,1.2),ylim=(-150,-70),title='Scalar jerk rectifies this sinusoid',xlabel='Hz',ylabel='dB jerk PSD')
  for ax in axes[row]:ax.grid(alpha=.15)
  audit.append(dict(direction=name,input_amplitude_mg=1,frequency_hz=f0,magnitude_half_peak_to_peak_mg=float((mag.max()-mag.min())/2*1000)))
 fig.suptitle('Same 1-mg, 0.25-Hz directional oscillation: magnitude can suppress and double its frequency');fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'figures/directional-signal-representation.png',dpi=150);plt.close(fig);(OUT/'synthetic_representation.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':examples();synthetic()
