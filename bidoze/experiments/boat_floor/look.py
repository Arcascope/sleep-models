"""One-night look at in-band versus high-band magnitude/jerk spectra; exploratory."""
import json,sys
from pathlib import Path
import numpy as np, pandas as pd
from pisces_lite.proc import ProcessingConfig
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE
from arcasleep.data.prepare import clean_accelerometer_rows, median_sampling_rate
root=Path('/media/eric/Extreme SSD/pisces2-data');data=Path(sys.argv[1]);name=sys.argv[2]
cohort,stem=name.split('_subject_',1)
cfg=json.loads((data/'bundles/seed0/preprocessing.json').read_text())
raw=clean_accelerometer_rows(pd.read_csv(root/cohort/'cleaned_accelerometer'/f'{stem}.csv').to_numpy(float))
fs=median_sampling_rate(raw[:,0]);print('fs',fs)
rel=ProcessingConfig.from_dict(cfg).apply(raw,normalize=False,padding=False,origin=0.)
full=ProcessingConfig.from_dict(dict(cfg,fs=float(np.floor(fs*0.999*100)/100),fmax=None)).apply(raw,normalize=False,padding=False,origin=0.)
print(rel.shape,full.shape)
n=min(len(rel),len(full));rel,full=rel[:n],full[:n]
ok=~np.all(rel==SPECTROGRAM_PADDING_VALUE,axis=(1,2))&~np.all(full==SPECTROGRAM_PADDING_VALUE,axis=(1,2))
fr=np.arange(1,rel.shape[1]+1)/10; ff=np.arange(1,full.shape[1]+1)/10
for c,ch in enumerate(['mag','jerk']):
    R=np.exp(rel[ok,:,c]);Fu=np.exp(full[ok,:,c])
    band=(fr>=2)&(fr<=6);bandf=(ff>=2)&(ff<=6)
    k=np.median(R[:,band].mean(1)/Fu[:,bandf].mean(1))
    q=np.percentile(Fu,10,axis=0)
    print(ch,'scale release/native',k)
    for lo,hi in [(0.1,0.5),(0.5,2),(2,4),(4,6),(6,8),(8,12),(12,18),(18,24.5)]:
        t=(ff>=lo)&(ff<hi)
        if t.any():print(f'  {lo}-{hi} Hz p10 {10*np.log10(np.median(q[t])*k):.1f} dB  median {10*np.log10(np.median(np.median(Fu[:,t],0))*k):.1f}')
