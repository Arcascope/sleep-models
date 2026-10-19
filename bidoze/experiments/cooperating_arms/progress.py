"""Show arm contributions at recorded milestones on two fixed validation nights."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_contributions(out,path,title):
    snapshots=sorted(out.glob('validation_probe_epoch*.npz'))
    if not snapshots:raise ValueError('no recorded validation milestones')
    steps_per_epoch=(len(json.loads((out/'split.json').read_text())['fit'])+1)//2
    first=np.load(snapshots[0]);members=first['probabilities'].shape[0]
    refs=json.loads(str(first['validation_refs']))
    fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True,sharey=True)
    for row,cohort in enumerate(['SleepAccel','SleepAccel-Clinical']):
        for stage,label in enumerate(['Wake','Light','Deep','REM']):
            ax=axes[row,stage]
            for arm in range(members):
                xx=[];yy=[]
                for file in snapshots:
                    z=np.load(file);on=z['labels'][row]==stage
                    xx.append(int(file.stem[-4:])*steps_per_epoch)
                    yy.append(z['probabilities'][arm,row,on,stage].mean() if on.any() else np.nan)
                ax.plot(xx,yy,marker='.',label=f'Arm {arm}')
            if not np.any(first['labels'][row]==stage):
                ax.text(.5,.5,'No PSG epochs of this stage',ha='center',va='center',transform=ax.transAxes,color='#666666')
            ax.set_title(f'{cohort} {refs[row]["subject_id"]}: {label}',fontsize=10);ax.set_ylim(0,1)
            if row==1:ax.set_xlabel('Optimizer updates')
            if stage==0:ax.set_ylabel('Mean probability of PSG stage')
    axes[0,-1].legend(ncol=2,fontsize=8)
    fig.suptitle(title+'\nMilestone snapshots on validation data; not cohort estimates',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(path,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();draw_contributions(a.fit,a.output,a.fit.parent.name+': contributions through training')
