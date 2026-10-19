"""Append one completed arm's artifact-backed section to the daily repository log.

Fetch/fast-forward main before calling, then review, commit and push text and figures
as one publication. This script never commits, pushes, or replaces other sections.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--log-root',required=True,type=Path)
    p.add_argument('--arm',required=True)
    p.add_argument('--draw',default=0,type=int)
    p.add_argument('--seed',default=0,type=int)
    a=p.parse_args();out=a.run/a.arm/f'draw{a.draw}_seed{a.seed}'
    tag=f'{a.arm}-d{a.draw}-s{a.seed}'
    assert (out/'analysis/report_complete.json').exists()
    config=json.loads((out/'config.json').read_text());done=json.loads((out/'complete.json').read_text())
    history=json.loads((out/'history.json').read_text());summary=pd.read_csv(out/'analysis/summary.csv')
    selected=json.loads((out/'analysis/selected_ids.json').read_text())
    day=a.log_root/'2026/October/19';log=day/'log.md';body=log.read_text()
    heading=f'### {a.arm}, draw {a.draw}, seed {a.seed}'
    assert heading not in body, 'this result already has a publication; edit it deliberately'
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
    images=[]
    for file in sorted((out/'analysis/figures').glob('*.png')):
        if file.name=='cooperating-architecture.png':continue
        shutil.copy2(file,day/'figures'/file.name);images.append(file.name)
    capped=' **Session capped: provisional, not converged.**' if done['time_capped'] else ''
    section=f'''{heading}

**Publication {stamp}.**{capped} Training code `{config['code_commit'][:7]}`; reporting `bidoze/experiments/cooperating_arms/report.py` writes this fit’s `analysis/subjects.csv`, `summary.csv`, `selected_ids.json`, and figures. Artifacts: `bidoze/results/{a.run.name}/{a.arm}/draw{a.draw}_seed{a.seed}/`. {done['parameters']:,} weights; training {done['train_seconds']:.1f} s, {len(history['eval_kappa']):,} epochs; best combined validation κ {history['best_eval_kappa']:.4f} at epoch {history['best_epoch']}.

Mean over all test recordings within each cohort; ± is SD across recordings, not seed uncertainty. Four-stage and sleep/wake κ use the adult release decoder. Wake specificity at 95% sleep sensitivity is pooled over the cohort, a descriptive ROC point using test labels rather than a calibrated deployment threshold. Gap calls on PSG-scored stage epochs count as misses. No participant test is claimed from this one fit.

| Cohort | Recordings | Model | κ 4-stage | κ sleep/wake | Wake specificity at 95% sensitivity (%) | TST bias (min) |
|:---|--:|:---|--:|--:|--:|--:|
'''
    for _,r in summary.iterrows():
        section+=f"| {r.cohort} | {int(r.recordings)} | {a.arm} | {r.kappa_4stage:.3f} ± {r.kappa_4stage_sd:.3f} | {r.kappa_sleep_wake:.3f} ± {r.kappa_sleep_wake_sd:.3f} | {r.wake_spec_at_95_sens:.1f} | {r.tst_bias_minutes:+.1f} |\n"
        section+=f"| {r.cohort} | {int(r.recordings)} | Historical G ×3 | {r.baseline_kappa_4stage:.3f} ± {r.baseline_kappa_4stage_sd:.3f} | {r.baseline_kappa_sleep_wake:.3f} ± {r.baseline_kappa_sleep_wake_sd:.3f} | {r.baseline_wake_spec_at_95_sens:.1f} | — |\n"
    section+='\n'
    for _,r in summary.iterrows():
        section+=f"- **{r.cohort}:** four-stage κ difference to G ×3 {r.delta_kappa_4stage:+.3f}; sleep/wake κ difference {r.kappa_sleep_wake-r.baseline_kappa_sleep_wake:+.3f}; wake specificity at matched sensitivity difference {r.wake_spec_at_95_sens-r.baseline_wake_spec_at_95_sens:+.1f} points.\n"
    section+=f'''\n![{a.arm} learning trajectory](figures/{tag}-learning.png)

*Combined training loss and validation κ against optimizer updates. Selection uses the unsmoothed validation maximum; the plateau schedule uses the smoothed curve.*

![{a.arm} stage contributions through training](figures/{tag}-contributions.png)

*Each line is one arm’s probability of the PSG stage on that stage’s epochs, for one fixed validation night per cohort. These two nights illustrate learning dynamics; they are not cohort performance estimates or requirements that every arm learn every stage.*

Selected hypnograms below show actual adult-decoder calls; their probability panels retain the raw mean probabilities. PSG gaps/unscored epochs are excluded from the numerical scoring. The fixed reference is retained across arms; largest improvement/regression is selected retrospectively against G ×3 and is illustrative.

'''
    subjects=pd.read_csv(out/'analysis/subjects.csv')
    for item in selected:
        ds,sid=item['dataset'],item['subject_id'];r=subjects[(subjects.dataset==ds)&(subjects.subject_id.astype(str)==str(sid))].iloc[0]
        section+=f"![{a.arm}, {r.cohort} {sid}, {item['reason']}](figures/{tag}-{ds}-{sid}.png)\n\n*{r.cohort} {sid}: {item['reason']}; κ 4-stage {r.kappa_4stage:.3f} versus G ×3 {r.baseline_kappa_4stage:.3f} (Δ {r.delta_kappa_4stage:+.3f}).*\n\n"
    marker='## 2. Open'
    assert marker in body
    body=body.replace(marker,section+marker,1)
    body=body.replace('- A7 is training; B7 and C7 follow serially on the local 5090. Publish figures and results after each completes, with a new publication timestamp.',
                      '- Local fits and reports run serially under the GPU lock; publish each completed result above with its own timestamp.')
    log.write_text(body)
    print('Prepared publication',stamp,a.arm)


if __name__=='__main__':main()
