"""Paired SleepAccel errors and a standalone scientific comparison figure."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from report import bootstrap

LABELS={'peak50_narrow':'Peak 54–78 bpm, 50 Hz', 'peak50_wide':'Peak 30–120 bpm, 50 Hz',
        'peak_narrow':'Peak 54–78 bpm, 100 Hz', 'peak_wide':'Peak 30–120 bpm, 100 Hz',
        'seqsmoother_weaver_transfer':'SeqSmoother: Weaver transfer', 'seqsmoother':'SeqSmoother: SleepAccel CV',
        'training_mean':'Constant HR: SleepAccel training mean', 'nightbeat':'Nightbeat',
        'peak_median':'Wide peak + 5-min median', 'nb_anchor':'Nightbeat intermediate anchor',
        'anchor_median':'Intermediate anchor + 5-min median',
        'constant_weaver_training_mean':'Constant HR: Weaver training mean'}


def finalize(root):
    status=json.loads((root/'status.json').read_text())
    if status['status']!='complete':
        raise ValueError('Complete all SleepAccel folds before finalizing')
    frame=pd.read_csv(root/'predictions.csv.gz',dtype={'participant':str})
    summary=pd.read_csv(root/'summary.csv')
    rng=np.random.default_rng(20261008)
    common=frame.eligible & frame.model_context & np.isfinite(frame.seqsmoother)
    paired=[]
    for subset,mask in [('all',common),('still_clear',common & frame.still_clear50),
                         ('still_clear_in_band',common & frame.still_clear50 & frame.reference.between(54,78))]:
        for method, baseline in [('seqsmoother','peak50_narrow'),('seqsmoother','peak50_wide'),('seqsmoother','peak_median'),('seqsmoother','nightbeat'),('seqsmoother','anchor_median'),('seqsmoother','nb_anchor'),
                                  ('seqsmoother','training_mean'),('seqsmoother_weaver_transfer','peak50_narrow'),
                                  ('seqsmoother_weaver_transfer','constant_weaver_training_mean'),
                                  ('peak_narrow','peak50_narrow')]:
            selected=mask & np.isfinite(frame[method]) & np.isfinite(frame[baseline])
            p=frame.loc[selected].copy()
            p['error_method']=(p[method]-p.reference).abs()
            p['error_baseline']=(p[baseline]-p.reference).abs()
            means=p.groupby(['participant','recording'])[['error_method','error_baseline']].mean().groupby('participant').mean()
            delta=means.error_method-means.error_baseline
            lo,hi=bootstrap(delta,rng)
            paired.append(dict(subset=subset,method=method,baseline=baseline,participants=len(means),windows=len(p),
                               difference_macro_mae=float(delta.mean()),ci_low=lo,ci_high=hi))
    paired=pd.DataFrame(paired)
    paired.to_csv(root/'paired.csv',index=False)
    subsets=['matched_model_windows','matched_still_clear','matched_still_clear_in_band']
    methods=['peak50_narrow','peak_median','peak_narrow','seqsmoother_weaver_transfer','seqsmoother','training_mean']
    fig, axes=plt.subplots(1,2,figsize=(15,6))
    for i, method in enumerate(methods):
        p=summary[summary.method.eq(method)].set_index('subset').reindex(subsets)
        x=np.arange(len(subsets))+(i-2.5)*.12
        axes[0].bar(x,p.participant_macro_mae,width=.115,label=LABELS[method])
        axes[1].bar(x,p.pooled_median_abs_error,width=.115,label=LABELS[method])
    for ax in axes:
        ax.set_xticks(np.arange(3),['All matched windows','Still and clear','Still, clear and HR 54–78'])
        ax.tick_params(axis='x',labelsize=9)
        ax.set_ylabel('Absolute error (bpm)')
        ax.grid(axis='y',alpha=.2)
    axes[0].set_title('Participant-macro mean absolute error')
    axes[1].set_title('Pooled median absolute error: October 6 metric')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    fig.suptitle('SleepAccel: 28 participants, Apple optical HR reference; identical windows per group')
    fig.tight_layout(rect=(0, .13, 1, .95))
    fig.savefig(root/'sleepaccel_comparison.png',dpi=180)
    plt.close(fig)
    summary['method_label']=summary.method.map(LABELS)
    summary.to_csv(root/'summary_labeled.csv',index=False)
    print(summary[summary.subset.isin(subsets)][['subset','method_label','participants','windows','availability','participant_macro_mae','pooled_median_abs_error']].to_string(index=False))
    print('\nPaired differences, negative means method has lower MAE than baseline:\n'+paired.to_string(index=False))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--run',type=Path,required=True)
    finalize(a.parse_args().run)
