"""Append one completed arm's artifact-backed section to the daily repository log.

Fetch/fast-forward main before calling, then review, commit and push text and figures
as one publication. This script never commits, pushes, or replaces other sections.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import re
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
    analysis=out/'analysis'
    assert (out/'analysis/report_complete.json').exists()
    config=json.loads((out/'config.json').read_text());done=json.loads((out/'complete.json').read_text())
    report_commit=json.loads((analysis/'report_complete.json').read_text()).get('report_commit','unknown')
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

**Publication {stamp}.**{capped} Fit and test predictions completed {done['finished']}; publication time is recorded separately. Training code `{config['code_commit'][:7]}`; reporting `bidoze/experiments/cooperating_arms/report.py` (`{report_commit[:7]}`) writes this fit’s `analysis/subjects.csv`, `summary.csv`, `selected_ids.json`, and figures. Artifacts: `bidoze/results/{a.run.name}/{a.arm}/draw{a.draw}_seed{a.seed}/`. {done['parameters']:,} weights; training {done['train_seconds']:.1f} s, {len(history['eval_kappa']):,} epochs; best combined validation κ {history['best_eval_kappa']:.4f} at epoch {history['best_epoch']}.

Mean over all test recordings within each cohort; ± is SD across recordings, not seed uncertainty. Four-stage and sleep/wake κ use the adult release decoder. Wake specificity at 95% sleep sensitivity is pooled over the cohort, a descriptive ROC point using test labels rather than a calibrated deployment threshold. Gap calls on PSG-scored stage epochs count as misses. No participant test is claimed from this one fit.

| Cohort | Recordings | Model | κ 4-stage | κ sleep/wake | Wake specificity at 95% sensitivity (%) | TST bias (min) |
|:---|--:|:---|--:|--:|--:|--:|
'''
    for _,r in summary.iterrows():
        section+=f"| {r.cohort} | {int(r.recordings)} | {a.arm} | {r.kappa_4stage:.3f} ± {r.kappa_4stage_sd:.3f} | {r.kappa_sleep_wake:.3f} ± {r.kappa_sleep_wake_sd:.3f} | {r.wake_spec_at_95_sens:.1f} | {r.tst_bias_minutes:+.1f} |\n"
        section+=f"| {r.cohort} | {int(r.recordings)} | Historical G ×3 | {r.baseline_kappa_4stage:.3f} ± {r.baseline_kappa_4stage_sd:.3f} | {r.baseline_kappa_sleep_wake:.3f} ± {r.baseline_kappa_sleep_wake_sd:.3f} | {r.baseline_wake_spec_at_95_sens:.1f} | — |\n"
    for _,r in summary.iterrows():
        if 'split7_kappa_4stage' in r.index:
            section+=f'| {r.cohort} | {int(r.recordings)} | Historical split7 (own losses) | {r.split7_kappa_4stage:.3f} ± {r.split7_kappa_4stage_sd:.3f} | {r.split7_kappa_sleep_wake:.3f} ± {r.split7_kappa_sleep_wake_sd:.3f} | {r.split7_wake_spec_at_95_sens:.1f} | — |\n'
    section+='\nHistorical split7 is rescored from the October 17 saved predictions on these same test IDs and decoder. Comparing it with the new fits changes the objective and adds feature dropout; it is a historical contrast rather than an isolated objective ablation.\n\n'
    for _,r in summary.iterrows():
        section+=f"- **{r.cohort}:** four-stage κ difference to G ×3 {r.delta_kappa_4stage:+.3f}; sleep/wake κ difference {r.kappa_sleep_wake-r.baseline_kappa_sleep_wake:+.3f}; wake specificity at matched sensitivity difference {r.wake_spec_at_95_sens-r.baseline_wake_spec_at_95_sens:+.1f} points.\n"
    if (analysis/'assessment.md').exists():
        section+='\n'+(analysis/'assessment.md').read_text()+'\n'
    section+=f'''\n![Completed fits through {a.arm}](figures/{tag}-comparison.png)

*All test recordings are scored; the dashed reference is G ×3. Each bar is one fit, with parameter count and GPU training time below. This figure is a snapshot at this publication.*

![{a.arm} learning trajectory](figures/{tag}-learning.png)

*Combined training loss and validation κ against optimizer updates. Selection uses the unsmoothed validation maximum; the plateau schedule uses the smoothed curve.*

![{a.arm} stage contributions through training](figures/{tag}-contributions.png)

*Each line is one arm’s probability of the PSG stage on that stage’s epochs, for one fixed validation night per cohort. These two nights illustrate learning dynamics; they are not cohort performance estimates or requirements that every arm learn every stage.*

Selected hypnograms below show actual adult-decoder calls; their probability panels retain the raw mean probabilities. PSG gaps/unscored epochs are excluded from the numerical scoring. The fixed reference is retained across arms; largest improvement/regression is selected retrospectively against G ×3 and is illustrative.

'''
    if (day/'figures'/f'{tag}-readouts.png').exists():
        section+=f'![Final readout contributions](figures/{tag}-readouts.png)\n\n*Mean per-recording accuracy difference between the full mean and a mean omitting one readout. Positive values mean retaining that readout helps. All internal paths remain active: this is a readout diagnostic, not removal of the branch or an individual-training requirement.*\n\n'
    subjects=pd.read_csv(out/'analysis/subjects.csv')
    for item in selected:
        ds,sid=item['dataset'],item['subject_id'];r=subjects[(subjects.dataset==ds)&(subjects.subject_id.astype(str)==str(sid))].iloc[0]
        refname=f'reference-G3-d{a.draw}-{ds}-{sid}.png'
        if item['reason']=='fixed reference' and refname not in body:
            section+=f'![Historical G ×3 reference for {sid}](figures/{refname})\n\n*G ×3 reference on the same recording, for comparison with the new model below.*\n\n'
        section+=f"![{a.arm}, {r.cohort} {sid}, {item['reason']}](figures/{tag}-{ds}-{sid}.png)\n\n*{r.cohort} {sid}: {item['reason']}; κ 4-stage {r.kappa_4stage:.3f} versus G ×3 {r.baseline_kappa_4stage:.3f} (Δ {r.delta_kappa_4stage:+.3f}).*\n\n"
    marker='## 2. Open'
    assert marker in body
    section=re.sub(r'(?<![\w/])-(?=\d)', '−', section)
    body=body.replace(marker,section+marker,1)
    if 'cooperating-capacity.png' not in body:
        shutil.copy2(a.run/'setup/capacity.png',day/'figures/cooperating-capacity.png')
        budget='![Measured parameter allocation](figures/cooperating-capacity.png)\n\n*Counts from instantiated tensors (`capacity.py`, `setup/parameter_counts.json`): C7 assigns 1.005M weights to transformers versus A7/B7’s 0.605M while using 2.4% fewer total weights than the one-path control. C7 changes both encoder sharing and capacity allocation.*\n\n'
        body=body.replace('**Interpretation.**',budget+'**Interpretation.**',1)
    body=body.replace('Three representative arms are drawn; the initial fits use seven. Every path ends in the same single loss.', 'Three representative arms are drawn; the initial fits use seven. Every path ends in the same single loss. Arrows show forward data flow; gradients follow the paths in reverse. Local CNN-to-fusion skips and per-arm input projections are omitted for clarity.')
    body=body.replace('- A7 is training; B7 and C7 follow serially on the local 5090. Publish figures and results after each completes, with a new publication timestamp.',
                      '- Local fits and reports run serially under the GPU lock; publish each completed result above with its own timestamp.')
    body=re.sub(r'\n\*\*Local batch status[^\n]*\n', '\n', body)
    initial=['final7','exchange7','shared7','control1']
    complete=[name for name in initial if (a.run/name/'draw0_seed0/analysis/report_complete.json').exists()]
    remaining=[name for name in initial if name not in complete]
    status=f'\n\n**Local batch status {stamp}.** Reported: {", ".join(complete)}. Remaining initial fits: {", ".join(remaining) if remaining else "none"}.\n'
    body=body.replace('## 2. Open','## 2. Open'+status,1)
    if 'control1' in complete:
        body=body.replace('Run a current one-path control, then take a promising architecture', 'Take a promising architecture')
    follow=a.run/'setup/count_followup.json'
    if follow.exists():
        decision=json.loads(follow.read_text())
        if decision.get('action')=='run':
            completed=[name for name in decision['arms'] if (a.run/name/'draw0_seed0/analysis/report_complete.json').exists()]
            pending=[name for name in decision['arms'] if name not in completed]
            body=re.sub(r'\n\*\*Count batch status[^\n]*\n', '\n', body)
            line=f'\n\n**Count batch status {stamp}.** Reported: {", ".join(completed) if completed else "none"}; remaining: {", ".join(pending) if pending else "none"}.\n'
            body=body.replace('## 2. Open','## 2. Open'+line,1)
            if not pending:
                body=body.replace('- Take a promising architecture to 3 and 5 arms and another seed/split. Try more arms or modest capacity growth only if that progression supports it.', '- The 3/5-arm comparison is complete. Assess another initialization and participant split before extending the count or growing capacity.')
    log.write_text(body)
    print('Prepared publication',stamp,a.arm)


if __name__=='__main__':main()
