"""After the initial four fits, take one promising joint architecture to 3/5 arms.

Selection uses repeatedly inspected development cohorts, not an untouched test.
The cohort-equal macro is for scheduling only; both cohorts remain separate in reports.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd
from publish_completed import publish,git


def choose(run):
    scores={}
    for arm in ['final7','exchange7','shared7','control1']:
        fit=run/arm/'draw0_seed0'
        if json.loads((fit/'complete.json').read_text()).get('time_capped'):
            return dict(action='review',reason=f'{arm} is session-capped, not converged')
        df=pd.read_csv(fit/'analysis/summary.csv').set_index('cohort')
        scores[arm]={c:float(df.loc[c,'kappa_4stage']) for c in ['SleepAccel','SleepAccel-Clinical']}
    control=scores['control1'];eligible=[]
    for arm in ['final7','exchange7','shared7']:
        delta=[scores[arm][c]-control[c] for c in control]
        if sum(delta)/2>=-.02 and min(delta)>=-.03:eligible.append(arm)
    if not eligible:return dict(action='review',reason='no seven-arm fit meets the predeclared near-control continuation rule',scores=scores)
    macro={arm:sum(scores[arm].values())/2 for arm in eligible};best=max(macro.values())
    # Prefer earlier sharing within one hundredth, below the known single-fit spread.
    selected=next(arm for arm in ['shared7','exchange7','final7'] if arm in macro and macro[arm]>=best-.01)
    prefix=selected.rstrip('0123456789')
    return dict(action='run',selected=selected,arms=[prefix+'3',prefix+'5'],scores=scores,
                rule='mean cohort stage kappa delta >= -0.02 and each cohort >= -0.03 versus control1; prefer shared, then exchange, within 0.01 of best eligible macro')


def post_decision(root,run,decision):
    if git(root,'branch','--show-current')!='main' or git(root,'diff','--name-only') or git(root,'diff','--cached','--name-only'):
        raise RuntimeError('research-log tracked tree needs review')
    git(root,'fetch','origin');git(root,'merge','--ff-only','origin/main')
    log=root/'2026/October/19/log.md';body=log.read_text()
    marker='**Count follow-up decision'
    if marker in body:return
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
    if decision['action']=='run':
        text=f"**Count follow-up decision {stamp}.** `{decision['selected']}` supplies the next jointly supervised 3- and 5-arm fits (`{'`, `'.join(decision['arms'])}`), keeping about one model’s weight budget. Selection uses the predeclared near-control rule below and the two development cohorts equally; it is a scheduling choice, not a replicated winner. Scores and the decision are saved in `bidoze/results/{run.name}/setup/count_followup.json`.\n\n"
    else:
        text=f"**Count follow-up decision {stamp}.** Further count fits need assessment: {decision['reason']}. Decision artifact: `bidoze/results/{run.name}/setup/count_followup.json`.\n\n"
    body=body.replace('## 2. Open','## 2. Open\n\n'+text,1);log.write_text(body)
    git(root,'diff','--check');git(root,'add','2026/October/19/log.md')
    git(root,'commit','-m','Record the next transformer-arm count comparison.')
    try:git(root,'push','origin','main')
    except subprocess.CalledProcessError:
        git(root,'pull','--rebase','origin','main');git(root,'push','origin','main')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','source-run','labels-root','baseline-run','log-root']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();deadline=time.monotonic()+28800
    receipt=a.run/'control1/draw0_seed0/analysis/publication.json'
    while not receipt.exists():
        if time.monotonic()>deadline:raise TimeoutError('initial batch publication did not finish')
        time.sleep(15)
    decision=choose(a.run);decision['decided']=datetime.now().astimezone().isoformat()
    (a.run/'setup/count_followup.json').write_text(json.dumps(decision,indent=2))
    print(json.dumps(decision),flush=True);post_decision(a.log_root,a.run,decision)
    if decision['action']!='run':return
    here=Path(__file__).resolve().parent
    for arm in decision['arms']:
        subprocess.run([sys.executable,str(here/'run_local.py'),'--run',str(a.run),
            '--source-run',str(a.source_run),'--labels-root',str(a.labels_root),
            '--baseline-run',str(a.baseline_run),'--arms',arm],check=True)
        analysis=a.run/arm/'draw0_seed0/analysis'
        (analysis/'assessment.md').write_text(f"**Count comparison.** This {arm} fit follows `{decision['selected']}` using the predeclared development-data scheduling rule. Both cohorts and total capacity remain explicit in the table; this single fit does not establish an optimal branch count.\n")
        publish(a.run,a.log_root,arm)
        if json.loads((analysis.parent/'complete.json').read_text()).get('time_capped'):
            print('session capped; further count fit needs review',flush=True);return


if __name__=='__main__':main()
