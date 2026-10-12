"""Plot saved single-channel ablation responses without modifying hypnograms."""
import argparse,html,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(a):
    a.out.mkdir(parents=True,exist_ok=True);records=[]
    for name in ['BiDozeSleepAccelFold0_subject_1449548','weaverMatched_subject_6q40wasn_fitbit_n1']:
        z=np.load(a.analysis/'predictions'/f'{name}.npz');features=np.load(a.run/'hypnograms/cnn-inspection/features'/f'{name}.npz');t=np.arange(len(z['y']))/120
        fig,ax=plt.subplots(4,1,figsize=(15,9),sharex=True)
        ax[0].plot(t,features['nonoverlap_cnn'][:,16],lw=.6);ax[0].set_ylabel('Raw channel 16\npost-GELU')
        for label,color in [('baseline','black'),('zero','#d95f02'),('fit_mean','#1b9e77')]:
            ax[1].plot(t,z[label][:,0],label=label,color=color,lw=.7)
        ax[1].set_ylabel('Wake probability');ax[1].legend(ncol=3)
        for label,color in [('zero','#d95f02'),('fit_mean','#1b9e77')]:ax[2].plot(t,z[label][:,0]-z['baseline'][:,0],label=label,color=color,lw=.7)
        ax[2].axhline(0,color='black',lw=.4);ax[2].set_ylabel('Change in\nwake probability')
        ax[3].step(t,z['gap_fraction'],where='mid',lw=.7);ax[3].set_ylabel('Missing-window\nfraction');ax[3].set_xlabel('Elapsed hours from epoch origin')
        for aa in ax:
            aa.grid(alpha=.2);aa.set_xlim(t[0],t[-1]);aa.fill_between(t,0,1,where=z['missing'],transform=aa.get_xaxis_transform(),color='#888888',alpha=.2)
        fig.suptitle(name+' | nonoverlap CNN | original channel 16 (display row 60)\nInference-only: zero or fit-mean replacement; grey shading = common unavailable epochs')
        fig.tight_layout();fig.savefig(a.out/f'{name}.png',dpi=140);plt.close(fig);records.append(name)
    paired=pd.read_csv(a.analysis/'paired.csv');g=paired[(paired['mode']=='ggir')&paired.role.isin(['test','guard'])&paired.metric.isin(['auroc','descriptive_spec95','kappa_4stage'])]
    fig,axs=plt.subplots(1,3,figsize=(12,4));metrics=['auroc','descriptive_spec95','kappa_4stage'];groups=list(dict.fromkeys(zip(g.cohort,g.role)))
    for ax,metric in zip(axs,metrics):
        for j,var in enumerate(['zero','fit_mean']):
            vals=[float(g[(g.cohort==c)&(g.role==r)&(g.variant==var)&(g.metric==metric)].difference.iloc[0]) for c,r in groups]
            ax.bar(np.arange(len(groups))+(j-.5)*.3,vals,width=.3,label=var)
        ax.set_xticks(np.arange(len(groups)),[f'{c}\n{r}' for c,r in groups]);ax.axhline(0,color='black',lw=.7);ax.set_title(metric);ax.set_ylabel('Change versus fresh baseline');ax.grid(axis='y',alpha=.2)
    axs[0].legend();fig.suptitle('Single-channel ablation: participant-macro paired changes\nCommon available epochs; release decoding + GGIR; one seed, exploratory');fig.tight_layout();fig.savefig(a.out/'summary.png',dpi=150);plt.close(fig)
    links=''.join(f'<p><a href="{html.escape(n)}.png">{html.escape(n)}</a></p>' for n in records)
    (a.out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Channel 16 ablation</title><style>body{font:16px system-ui;margin:25px}img{max-width:100%}</style><a href="../cnn-inspection/">CNN inspection</a><h1>Channel 16 ablation</h1><p>Nonoverlap CNN original channel 16, display row 60. Inference only; other channels and model parameters remain unchanged. Zero and fit-mean replacements compared with fresh baseline. Exploratory channel selection used an evaluation recording.</p><a href="summary.png"><img src="summary.png"></a>'+links)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('analysis',type=Path);p.add_argument('out',type=Path);main(p.parse_args())
