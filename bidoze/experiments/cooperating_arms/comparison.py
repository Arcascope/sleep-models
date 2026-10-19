"""Snapshot diagrams comparing only completed, scored cooperating-arm fits."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def comparison(run, path):
    frames=[]
    for file in sorted(run.glob('*/draw*_seed*/analysis/summary.csv')):
        out=file.parent.parent
        config=json.loads((out/'config.json').read_text())
        done=json.loads((out/'complete.json').read_text())
        if done.get('time_capped'):continue
        frame=pd.read_csv(file)
        frame['label']=f"{config['arm']} d{config['draw']} s{config['seed']}"
        frame['parameters']=config['parameters']
        frame['seconds']=done['train_seconds']
        frames.append(frame)
    if not frames:return
    df=pd.concat(frames,ignore_index=True)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for row,cohort in enumerate(['SleepAccel','SleepAccel-Clinical']):
        sub=df[df.cohort==cohort]
        for col,(metric,label) in enumerate([('kappa_4stage','Four-stage κ'),('kappa_sleep_wake','Sleep/wake κ'),('wake_spec_at_95_sens','Wake specificity at 95% sleep sensitivity (%)')]):
            ax=axes[row,col]
            ax.bar(sub.label,sub[metric],color=['#5b9bd5' if 'final' in x else '#70ad47' if 'exchange' in x else '#edb45c' if 'shared' in x else '#a5a5a5' for x in sub.label])
            reference=sub['baseline_'+metric].iloc[0]
            ax.axhline(reference,color='#555555',ls='--',label='Historical G ×3 (6.71M weights)')
            ax.set_title(f'{cohort}: {label}',fontsize=11)
            ax.tick_params(axis='x',rotation=25)
            ax.legend(fontsize=7)
    costs=df.drop_duplicates('label')
    details='; '.join(f'{r.label}: {r.parameters/1e6:.2f}M / {r.seconds/60:.1f} min' for _,r in costs.iterrows())
    fig.suptitle('Completed fits: combined predictions on all test recordings',fontsize=16)
    fig.text(.5,.015,details,ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(path,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();comparison(a.run,a.output)
