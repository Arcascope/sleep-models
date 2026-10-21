"""Compare complete fits on matched draws; expose coverage instead of pooling repeats."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

COHORTS=['SleepAccel','SleepAccel-Clinical']
METRICS=[('kappa_4stage','Four-stage κ'),('kappa_sleep_wake','Sleep/wake κ'),
         ('wake_spec_at_95_sens','Wake specificity at 95% sleep sensitivity'),
         ('tst_bias_minutes','TST bias (minutes; closer to zero is better)')]


def collect(run):
    frames=[]
    for file in sorted(run.glob('*/draw*_seed*/analysis/summary.csv')):
        out=file.parent.parent
        if not (out/'analysis/report_complete.json').exists():continue
        config=json.loads((out/'config.json').read_text());done=json.loads((out/'complete.json').read_text())
        if done.get('time_capped') or config['seed']!=0:continue
        frame=pd.read_csv(file);frame['experiment']=config['arm'];frame['draw']=config['draw']
        frame['parameters']=config['parameters'];frame['seconds']=done['train_seconds'];frames.append(frame)
    if not frames:return pd.DataFrame(),[],{}
    df=pd.concat(frames,ignore_index=True)
    coverage={name:sorted(g.draw.unique().tolist()) for name,g in df.groupby('experiment')}
    common=sorted(set.intersection(*(set(v) for v in coverage.values())))
    if not common:raise ValueError('no common completed draw across experiments')
    return df[df.draw.isin(common)],common,coverage


def comparison(run,path):
    df,draws,coverage=collect(run)
    if df.empty:return
    experiments=sorted(coverage,key=lambda x:('_5m' in x,not x.startswith('control'),x))
    labels=[x.replace('_5m','\n≈5M') for x in experiments]
    colors=['#a5a5a5' if x.startswith('control') else '#5b9bd5' if x.startswith('final') else '#70ad47' if x.startswith('exchange') else '#edb45c' for x in experiments]
    fig,axes=plt.subplots(2,4,figsize=(19,max(9,6+len(experiments)*.3)))
    for row,cohort in enumerate(COHORTS):
        sub=df[df.cohort==cohort]
        for col,(metric,label) in enumerate(METRICS):
            ax=axes[row,col];scale=.01 if metric=='wake_spec_at_95_sens' else 1
            groups=[sub[sub.experiment==e][metric].to_numpy()*scale for e in experiments]
            means=np.array([v.mean() for v in groups]);ax.bar(range(len(experiments)),means,color=colors,width=.72)
            if len(draws)>1:
                ax.errorbar(range(len(experiments)),means,yerr=[means-np.array([v.min() for v in groups]),np.array([v.max() for v in groups])-means],fmt='none',ecolor='#333333',capsize=3,lw=1)
            for i,v in enumerate(means):ax.annotate(f'{v:.3f}' if col<2 else f'{v:.1%}' if col==2 else f'{v:+.1f}',(i,v),xytext=(0,4 if v>=0 else -13),textcoords='offset points',ha='center',fontsize=8)
            ax.set_xticks(range(len(experiments)),labels,rotation=40,ha='right',fontsize=9)
            ax.set_title(f'{cohort}\n{label}',fontsize=10);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if col<3:
                ax.set_ylim(0,1)
                if col==2:ax.yaxis.set_major_formatter(PercentFormatter(1))
                ref=sub.groupby('draw')['baseline_'+metric].first().mean()*scale
                ax.axhline(ref,color='#444444',ls='--',lw=1,label='Historical G ×3')
                if 'split7_'+metric in sub and sub['split7_'+metric].notna().all():
                    ax.axhline(sub.groupby('draw')['split7_'+metric].first().mean()*scale,color='#2878b5',ls=':',lw=1,label='Historical split7 (own losses)')
                ax.legend(fontsize=7,loc='upper right')
            else:
                extent=max(30,max(abs(np.concatenate(groups)))*1.35);ax.set_ylim(-extent,extent);ax.axhline(0,color='#444444',lw=1)
    details='; '.join(f'{name}: {df[df.experiment==name].parameters.iloc[0]/1e6:.2f}M, draws {"/".join(map(str,coverage[name]))}' for name in experiments)
    import textwrap
    fig.suptitle('Cooperating-arm experiments: combined model predictions\nMatched completed draws '+', '.join(map(str,draws))+'; seed 0; each cohort scored separately',fontsize=16)
    fig.text(.5,.015,'\n'.join(textwrap.wrap(details,170))+'\nBars: mean of per-draw cohort metrics. Whiskers: draw min–max when available; overlapping draws are not independent. No uncertainty estimate from one draw.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.10,1,.93]);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=160);plt.close(fig)
    df.to_csv(path.with_suffix('.csv'),index=False)
    path.with_suffix('.json').write_text(json.dumps(dict(matched_draws=draws,coverage=coverage,metrics=[m for m,_ in METRICS]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();comparison(a.run,a.output)
