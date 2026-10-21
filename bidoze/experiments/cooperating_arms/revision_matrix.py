"""Run the declared five-draw comparisons without selecting from draw zero.

Wait for the already running count worker, use the common GPU lock, and publish
one completed report at a time. Publication failures are recorded and retried,
but do not stop subsequent GPU fits. Capped fits still stop for review.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from publish_completed import publish

BASE=['control1','final3','final5','final7','exchange7','shared7']
LARGE=['control1_5m','final5_5m','exchange5_5m','shared5_5m']


def jobs():
    return [dict(arm=arm,draw=draw,seed=0) for arms in [BASE,LARGE] for draw in range(5) for arm in arms]


def publish_or_defer(run,root,arm,draw):
    analysis=run/arm/f'draw{draw}_seed0/analysis'
    failure=analysis/'publication_error.json'
    try:
        publish(run,root,arm,draw=draw)
    except Exception as exc:
        error=dict(at=datetime.now().astimezone().isoformat(),arm=arm,draw=draw,
                   error=f'{type(exc).__name__}: {exc}')
        failure.write_text(json.dumps(error,indent=2))
        print('PUBLICATION DEFERRED; continuing fits',json.dumps(error),flush=True)
        return False
    failure.unlink(missing_ok=True)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','source-run','labels-root','baseline-run','log-root']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    manifest=dict(created=datetime.now().astimezone().isoformat(),jobs=jobs(),selection='No draw-0 architecture exclusion. All five draws for each specified configuration.',large_target_parameters=5_000_000)
    # Read-only preflight: existing mixed-cohort caches and all historical comparison predictions.
    from arcasleep.data.cache import discover_cached_subjects,participant_id
    from arcasleep.training.balanced import split_train_eval
    from run import TRAIN,TEST,CACHE
    from report import baseline_predictions
    audit={}
    for draw in range(5):
        found=discover_cached_subjects(TRAIN+TEST,a.source_run/'setup/cache'/f'draw{draw}'/CACHE)
        fit,val=split_train_eval(sorted(s.ref for ds in TRAIN for s in found[ds]),.3,0)
        tests=sorted(s.ref for ds in TEST for s in found[ds])
        groups=[set(participant_id(r.subject_id) for r in refs) for refs in [fit,val,tests]]
        assert not any(groups[i]&groups[j] for i in range(3) for j in range(i)),f'participant leakage draw {draw}'
        for refs,ds_names in [(fit,TRAIN[1:]),(val,TRAIN[1:]),(tests,TEST)]:
            assert all(any(r.dataset_name==ds for r in refs) for ds in ds_names)
        base=baseline_predictions(a.baseline_run,draw)
        assert set(base)=={(r.dataset_name,r.subject_id) for r in tests},f'baseline population draw {draw}'
        audit[str(draw)]=dict(fit=len(fit),validation=len(val),test=len(tests))
    manifest['preflight']=audit
    path=a.run/'setup/revision_matrix.json'
    if not path.exists():path.write_text(json.dumps(manifest,indent=2))
    else:assert json.loads(path.read_text())['jobs']==manifest['jobs'],'refuse changed matrix on restart'
    print('Preflight passed',json.dumps(audit),flush=True)
    if a.prepare_only:return
    while subprocess.run(['systemctl','--user','is-active','--quiet','arcasleep-arms-counts-20261019.service']).returncode==0:
        time.sleep(15)
    here=Path(__file__).resolve().parent
    for job in manifest['jobs']:
        arm,draw=job['arm'],job['draw'];out=a.run/arm/f'draw{draw}_seed0'
        if (out/'analysis/publication.json').exists():continue
        print(datetime.now().astimezone().isoformat(),job,flush=True)
        if not (out/'analysis/report_complete.json').exists():
            subprocess.run([sys.executable,str(here/'run_local.py'),'--run',str(a.run),
                '--source-run',str(a.source_run),'--labels-root',str(a.labels_root),
                '--baseline-run',str(a.baseline_run),'--arms',arm,'--draw',str(draw)],check=True)
        if json.loads((out/'complete.json').read_text()).get('time_capped'):
            raise RuntimeError(f'{job} reached its time cap; review/resume before continuing')
        (out/'analysis/assessment.md').write_text('**Five-draw comparison.** This fit belongs to the declared revision matrix. Retain every architecture across draws 0–4; assess paired differences to the capacity-matched control per cohort after complete coverage. Draws overlap in participants and are not independent samples. The 5M panel tests capacity and sharing at five arms, without changing the mean-only supervised objective.\n')
        publish_or_defer(a.run,a.log_root,arm,draw)
    pending=[j for j in manifest['jobs'] if not (a.run/j['arm']/f"draw{j['draw']}_seed0/analysis/publication.json").exists()]
    for job in pending:publish_or_defer(a.run,a.log_root,job['arm'],job['draw'])
    pending=[j for j in manifest['jobs'] if not (a.run/j['arm']/f"draw{j['draw']}_seed0/analysis/publication.json").exists()]
    print('Declared fits complete; pending publications:',json.dumps(pending),flush=True)


if __name__=='__main__':main()
