import sys, time, numpy as np, warnings
warnings.simplefilter("ignore")
from arcasleep_watchtree.cohorts import COHORTS, subject_ids, load_night
for name in ["wearanize","aauwss","sleepaccel","weaver_garmin","weaver_fitbit"]:
    c=COHORTS[name]; ids,rep=subject_ids(c)
    t0=time.time(); n=load_night(c,ids[0],rep[ids[0]])
    dt=np.diff(n.hr_t); b=n.hr_bpm
    print(name,len(ids),"load s",round(time.time()-t0,1),"hr n",b.size,"epochs",n.labels.size,"dt q",np.percentile(dt,[0,1,5,50,95,99,100]).round(3),"int frac",np.mean(b==np.round(b)),"bpm q",np.percentile(b,[0,1,50,99,100]),flush=True)
from senpy import compute_nufft_spectrogram
t=np.arange(0,1200.)+np.random.rand(1200)*0.0; x=np.random.randn(1200)
r=compute_nufft_spectrogram(t,x,300.,0.,kind="psd",origin_s=0.0,empty_windows="keep")
print(r.frequencies[:5],r.frequencies[-1],len(r.frequencies),r.Sxx.shape,r.sample_count,r.valid, r.Sxx.mean(1))
