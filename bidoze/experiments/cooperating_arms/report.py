"""Score all saved test predictions and render selected official hypnograms.

Figure selection is descriptive: two fixed reference IDs plus the largest
improvement and regression in each cohort. Every test ID remains in the tables.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'tridoze'))
import evaluate as scoring
from arcasleep.data.bundle import CacheBundle
from arcasleep.data.batching import recording_input
from arcasleep.reporting.hypnogram import render_subject_hypnogram

COHORTS={'testSleepAccel':'SleepAccel','testClinical':'SleepAccel-Clinical'}
METRICS=['kappa_4stage','kappa_sleep_wake','sleep_sensitivity','wake_specificity']


def architecture(path):
    fig,axes=plt.subplots(1,3,figsize=(16,7))
    colors=['#dceaf7','#dff0e2','#f9e7cb']
    for ax,kind,col in zip(axes,['A: final mean loss','B: intermediate exchange','C: shared CNN'],colors):
        ax.set(xlim=(0,10),ylim=(-1,13));ax.axis('off');ax.set_title(kind,fontweight='bold',fontsize=15)
        def box(x,y,w,h,label,c=col):
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',facecolor=c,edgecolor='#455a64'))
            ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
        def arrow(x,y,x2,y2):ax.annotate('',(x2,y2),(x,y),arrowprops=dict(arrowstyle='->',color='#455a64',lw=1.5))
        box(.4,11.4,9.2,.8,'Same spectrogram / same examples','#eeeeee')
        if kind.startswith('C'):
            box(.4,9.8,9.2,.9,'One trainable CNN');arrow(5,11.4,5,10.7)
        for x in [1,4,7]:
            if not kind.startswith('C'):
                box(x,9.8,2,.9,'CNN');arrow(x+1,11.4,x+1,10.7)
            else:arrow(x+1,9.8,x+1,9.3)
            box(x,7.9,2,1.4,'Layers\n1–2');arrow(x+1,9.8,x+1,9.3)
            box(x,4.5,2,1.4,'Layers\n3–4')
            if kind.startswith('B'):
                arrow(x+1,7.9,x+1,7.2);arrow(x+1,6.4,x+1,5.9)
            else:arrow(x+1,7.9,x+1,5.9)
            box(x,2.8,2,.8,'Head');arrow(x+1,4.5,x+1,3.6);arrow(x+1,2.8,5,2.1)
        if kind.startswith('B'):box(.5,6.4,9,.8,'Residual exchange: write → mean others → read','#f8dbd8')
        box(.4,1.3,9.2,.8,'Mean of ALL arm probabilities','#e3dcf2');box(.4,-.0,9.2,.8,'ONE class-balanced supervised loss','#e3dcf2');arrow(5,1.3,5,.8)
        ax.text(5,-.65,'3 representative arms shown; initial fits use 7',ha='center',fontsize=9)
    fig.suptitle('Cooperating arms: sharing progressively earlier',fontsize=19)
    fig.tight_layout(rect=[0,.03,1,.94]);fig.savefig(path,dpi=150);plt.close(fig)


def baseline_predictions(source,draw):
    paths=[source/('bidoze_g' if i==0 else f'bidoze_g_s{i}')/f'draw{draw}'/'predictions'/f'draw{draw}_seed_{i}' for i in range(3)]
    result={}
    for p in sorted(paths[0].glob('*.csv')):
        ds=next((d for d in COHORTS if p.name.startswith(d+'_')),None)
        if ds is None:continue
        loaded=[scoring.load_subject(root/p.name) for root in paths]
        for y,_ in loaded[1:]:np.testing.assert_array_equal(y,loaded[0][0])
        result[(ds,p.stem[len(ds)+1:])]=(loaded[0][0],np.mean([x[1] for x in loaded],axis=0))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True,type=Path)
    ap.add_argument('--source-run',required=True,type=Path)
    ap.add_argument('--baseline-run',required=True,type=Path)
    ap.add_argument('--arm',required=True)
    ap.add_argument('--draw',default=0,type=int)
    ap.add_argument('--seed',default=0,type=int)
    a=ap.parse_args();out=a.run/a.arm/f'draw{a.draw}_seed{a.seed}'
    tag=f'{a.arm}-d{a.draw}-s{a.seed}'
    assert (out/'complete.json').exists(),'fit predictions are not complete'
    analysis=out/'analysis';analysis.mkdir(exist_ok=True)
    figs=analysis/'figures';figs.mkdir(exist_ok=True)
    logger=scoring.MetricsLogger(scoring.load_metrics_config(HERE.parents[1]/'config/metrics.json')[0])
    divisor=np.load(out/'stage_divisor.npy');baseline=baseline_predictions(a.baseline_run,a.draw)
    rows=[];full={};pooled={ds:{} for ds in COHORTS};basepool={ds:{} for ds in COHORTS}
    splitpool={ds:{} for ds in COHORTS}
    for f in sorted((out/'predictions').glob('*.npz')):
        ds=next(d for d in COHORTS if f.stem.startswith(d+'_'));sid=f.stem[len(ds)+1:]
        z=np.load(f);q=z['probabilities'];y=z['labels'];p=q.mean(0)
        if not np.isfinite(q).all():raise ValueError(f'nonfinite probabilities in {f.name}')
        np.testing.assert_allclose(q.sum(-1),1,atol=1e-5)
        assert q.shape[1:]==(len(y),5)
        by,bp=baseline[(ds,sid)];np.testing.assert_array_equal(np.where(y<0,-1,y),by)
        calls=scoring.hybrid_calls(p,p,divisor);bc=scoring.hybrid_calls(bp,bp,divisor)
        m=scoring.subject_metrics(logger,y,p,calls);bm=scoring.subject_metrics(logger,y,bp,bc)
        splitfile=a.source_run/'split7_1x'/f'draw{a.draw}'/'predictions'/f'draw{a.draw}_seed_0'/f'{ds}_{sid}.csv'
        if splitfile.exists():
            sy,sp=scoring.load_subject(splitfile);np.testing.assert_array_equal(sy,by)
            sm=scoring.subject_metrics(logger,sy,sp,scoring.hybrid_calls(sp,sp,divisor))
            splitpool[ds][sid]=(sy,sp)
        else:sm=None
        valid=(y>=0)&(y<4)
        r=dict(arm=a.arm,cohort=COHORTS[ds],dataset=ds,subject_id=sid,n_scored=int(valid.sum()),
               **{k:float(m[k]) for k in METRICS},
               **{f'baseline_{k}':float(bm[k]) for k in METRICS},
               delta_kappa_4stage=float(m['kappa_4stage']-bm['kappa_4stage']))
        if sm is not None:r.update({f'split7_{k}':float(sm[k]) for k in METRICS})
        for stage,label in enumerate(['wake','light','deep','rem']):
            on=valid&(y==stage);r[f'recall_{label}']=float(np.mean(calls[on]==stage)) if on.any() else np.nan
        mins=scoring.sleep_minutes(y[valid],calls[valid]);r['tst_bias_minutes']=mins['pred_tst_minutes']-mins['psg_tst_minutes']
        r['tst_mae_minutes']=abs(r['tst_bias_minutes'])
        r['waso_bias_minutes']=mins['pred_waso_minutes']-mins['psg_waso_minutes']
        r['waso_mae_minutes']=abs(r['waso_bias_minutes'])
        # Leave out a readout only: intermediate paths remain present in connected variants.
        for i in range(len(q)):
            if len(q)>1:
                leave=(q.sum(0)-q[i])/(len(q)-1)
                lc=scoring.hybrid_calls(leave,leave,divisor)
                r[f'omit_readout_{i}_delta_accuracy']=float(np.mean(calls[valid]==y[valid])-np.mean(lc[valid]==y[valid]))
        rows.append(r);full[(ds,sid)]=(y,p,calls,q,bp,bc)
        pooled[ds][sid]=(y,p);basepool[ds][sid]=(y,bp)
    assert set(full)==set(baseline),'incomplete or unexpected test population'
    df=pd.DataFrame(rows);df.to_csv(analysis/'subjects.csv',index=False)
    summary=[]
    for ds,cohort in COHORTS.items():
        sub=df[df.dataset==ds];r=dict(arm=a.arm,cohort=cohort,recordings=len(sub))
        for c in METRICS+['tst_bias_minutes','tst_mae_minutes','waso_bias_minutes','waso_mae_minutes','delta_kappa_4stage']:
            r[c]=float(sub[c].mean());r[c+'_sd']=float(sub[c].std())
        r['wake_spec_at_95_sens']=100*scoring.wake_specificity_at(pooled[ds],.95)
        r['baseline_wake_spec_at_95_sens']=100*scoring.wake_specificity_at(basepool[ds],.95)
        for c in METRICS:
            r['baseline_'+c]=float(sub['baseline_'+c].mean())
            r['baseline_'+c+'_sd']=float(sub['baseline_'+c].std())
        if len(splitpool[ds])==len(sub):
            for c in METRICS:
                r['split7_'+c]=float(sub['split7_'+c].mean())
                r['split7_'+c+'_sd']=float(sub['split7_'+c].std())
            r['split7_wake_spec_at_95_sens']=100*scoring.wake_specificity_at(splitpool[ds],.95)
        summary.append(r)
    pd.DataFrame(summary).to_csv(analysis/'summary.csv',index=False)
    fixed=a.run/f'reference_ids_draw{a.draw}.json'
    if fixed.exists():references=json.loads(fixed.read_text())
    else:
        references={ds:sorted([sid for d,sid in full if d==ds])[0] for ds in COHORTS}
        # Known deep/light disagreement from the October 17 diagnosis.
        if ('testSleepAccel','5132496') in full:references['testSleepAccel']='5132496'
        fixed.write_text(json.dumps(references,indent=2))
    selected=[]
    for ds,cohort in COHORTS.items():
        sub=df[df.dataset==ds].sort_values('delta_kappa_4stage')
        candidates=[(references[ds],'fixed reference'),(sub.iloc[0].subject_id,'largest regression'),
                    (sub.iloc[-1].subject_id,'largest improvement')]
        used=set()
        for sid,reason in candidates:
            if sid in used:continue
            used.add(sid);selected.append(dict(dataset=ds,subject_id=sid,reason=reason))
            y,p,calls,q,bp,bc=full[(ds,sid)]
            path=a.source_run/'setup/cache'/f'draw{a.draw}'/'12Hz_log_PSD_aligned'/f'{ds}_subject_{sid}_features.npz'
            b=CacheBundle.from_npz(path);inputs=recording_input(b.X[0],len(y),1)
            if reason=='fixed reference':
                reference=a.run/'setup'/f'reference-G3-d{a.draw}-{ds}-{sid}.png'
                if not reference.exists():
                    render_subject_hypnogram(reference,inputs,bp,y,
                        f'Historical G x3 | {cohort} {sid} | adult decoder; raw mean probabilities',
                        channel_names=['Magnitude','Jerk'],epoch_seconds=30,predicted_stages=bc)
                import shutil
                shutil.copy2(reference,figs/reference.name)
            render_subject_hypnogram(figs/f'{tag}-{ds}-{sid}.png',inputs,p,y,
                f'{a.arm} | {cohort} {sid} | {reason} | adult decoder; raw mean probabilities',
                channel_names=['Magnitude','Jerk'],epoch_seconds=30,predicted_stages=calls)
    (analysis/'selected_ids.json').write_text(json.dumps(selected,indent=2))
    history=json.loads((out/'history.json').read_text())
    steps_per_epoch=(len(json.loads((out/'split.json').read_text())['fit'])+1)//2
    steps=np.arange(1,len(history['eval_kappa'])+1)*steps_per_epoch
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(steps,history['train_loss']);axes[0].set(xlabel='Optimizer updates',ylabel='Balanced loss of mean',yscale='log')
    axes[1].plot(steps,history['eval_kappa'],alpha=.4,label='Validation κ');axes[1].plot(steps,history['smoothed_eval_kappa'],label='Smoothed')
    axes[1].legend();axes[1].set(xlabel='Optimizer updates',ylabel='Combined validation κ')
    fig.suptitle(f'{a.arm}: training trajectory (selected epoch {history["best_epoch"]})')
    fig.tight_layout();fig.savefig(figs/f'{tag}-learning.png',dpi=150);plt.close(fig)
    from progress import draw_contributions
    draw_contributions(out,figs/f'{tag}-contributions.png',f'{a.arm}: contributions through training')
    contribution_cols=[c for c in df if c.startswith('omit_readout_')]
    if contribution_cols:
        fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for ax,cohort in zip(axes,COHORTS.values()):
            values=100*df[df.cohort==cohort][contribution_cols].mean().to_numpy()
            ax.bar(range(len(values)),values,color=['#70ad47' if x>=0 else '#d47771' for x in values])
            ax.axhline(0,color='black',lw=.7)
            ax.set(title=cohort,xlabel='Readout arm',xticks=range(len(values)))
        axes[0].set_ylabel('Accuracy gained by retaining readout (points)')
        fig.suptitle(f'{a.arm}: does each readout help the combined prediction?')
        fig.tight_layout();fig.savefig(figs/f'{tag}-readouts.png',dpi=150);plt.close(fig)
    architecture(figs/'cooperating-architecture.png')
    from comparison import comparison
    comparison(a.run,figs/f'{tag}-comparison.png')
    (analysis/'report_complete.json').write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(),selected=selected,report_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
    print(pd.DataFrame(summary).to_string(index=False));print('figures',figs)


if __name__=='__main__':main()
