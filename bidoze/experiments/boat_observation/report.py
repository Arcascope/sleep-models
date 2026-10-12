"""Paired night-to-person contrasts and a descriptive spectral comparison figure."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from boat_evidence.score import METRICS


def main(out):
    analysis=out/'analysis'
    power=pd.read_csv(analysis/'power_nights.csv')
    keys=['name','cohort','participant','variant','channel','band']
    paired=power.pivot(index=keys,columns='label',values='median_epoch_band_db').dropna(subset=['still_wake','still_sleep'])
    paired['wake_minus_sleep_db']=paired.still_wake-paired.still_sleep
    paired.reset_index().to_csv(analysis/'contrast_nights.csv',index=False)
    subjects=paired.reset_index().groupby(keys[1:]).wake_minus_sleep_db.mean().reset_index()
    subjects.to_csv(analysis/'contrast_subjects.csv',index=False)
    summary=subjects.groupby(['cohort','variant','channel','band']).wake_minus_sleep_db.agg(['mean','count']).reset_index()
    summary.to_csv(analysis/'contrast_summary.csv',index=False)
    table=pd.read_csv(analysis/'nights.csv');night_keys=['name','cohort','participant','group','mode']
    baseline=table[table.variant=='native'].set_index(night_keys)[METRICS]
    differences=[]
    for variant,g in table[table.variant!='native'].groupby('variant'):
        delta=g.set_index(night_keys)[METRICS]-baseline
        delta=delta.dropna(how='all').reset_index();delta['variant']=variant;differences.append(delta)
    differences=pd.concat(differences,ignore_index=True)
    differences.to_csv(analysis/'paired_nights.csv',index=False)
    persons=differences.groupby(['cohort','participant','group','mode','variant'])[METRICS].mean().reset_index()
    persons.to_csv(analysis/'paired_subjects.csv',index=False)
    persons.groupby(['cohort','mode','variant'])[METRICS].mean().to_csv(analysis/'paired_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(11,3.7),sharey=True)
    colors={'trace':'#2878a8','mag':'#ce7838','jerk':'#687579'}
    bands=['0.1-0.5','0.5-2','2-6']
    for ax,cohort in zip(axes,['SURF','Somerset','SleepAccel-Clinical']):
        for i,channel in enumerate(['trace','mag','jerk']):
            rows=summary[(summary.cohort==cohort)&(summary.variant=='native')&(summary.channel==channel)].set_index('band')
            ax.bar([j+(i-1)*.24 for j in range(3)],rows.loc[bands,'mean'],width=.23,color=colors[channel],label={'trace':'Axis trace','mag':'Magnitude','jerk':'Difference-norm jerk'}[channel])
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),[b+' Hz' for b in bands]);ax.set_title(cohort);ax.set_xlabel('Frequency band')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Still-wake minus still-sleep level (dB)')
    axes[-1].legend(fontsize=8,loc='upper right')
    fig.suptitle('Native spectra: paired nightly contrast, then participant mean',fontsize=11)
    fig.tight_layout();fig.savefig(analysis/'native_axis_contrast.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);main(p.parse_args().out)
