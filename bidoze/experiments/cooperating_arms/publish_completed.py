"""Publish explicitly authorized per-arm reports to the research-log main branch.

Wait only for the specified reports. Preserve unrelated tracked/untracked state;
stop on a dirty tracked tree or publication conflict. Never force-push.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def publish(run,root,arm):
    out=run/arm/'draw0_seed0';analysis=out/'analysis';record=analysis/'publication.json'
    if record.exists():return
    if git(root,'branch','--show-current')!='main':raise RuntimeError('log worktree is not main')
    if git(root,'diff','--name-only') or git(root,'diff','--cached','--name-only'):
        raise RuntimeError('tracked log changes need review; nothing staged or discarded')
    git(root,'fetch','origin')
    git(root,'merge','--ff-only','origin/main')
    here=Path(__file__).resolve().parent
    subprocess.run([sys.executable,str(here/'log_update.py'),'--run',str(run),
                    '--log-root',str(root),'--arm',arm],check=True)
    day=Path('2026/October/19');tag=f'{arm}-d0-s0'
    files=[str(day/'log.md')]
    files += [str(day/'figures'/f.name) for f in (analysis/'figures').glob(f'{tag}-*.png')]
    # The first result may add the measured architecture-budget diagram.
    cap=day/'figures/cooperating-capacity.png'
    if (root/cap).exists():files.append(str(cap))
    git(root,'diff','--check')
    # Verify every referenced local image in the changed log before publishing.
    import re
    for target in re.findall(r'!\[[^\]]*\]\((figures/[^)]+)\)',(root/day/'log.md').read_text()):
        if not (root/day/target).exists():raise RuntimeError(f'missing image {target}')
    git(root,'add','--',*files)
    git(root,'commit','-m',f'Add {arm} results and selected hypnograms.')
    try:git(root,'push','origin','main')
    except subprocess.CalledProcessError:
        # A concurrent publication may have advanced main; preserve both commits.
        git(root,'pull','--rebase','origin','main')
        git(root,'push','origin','main')
    result=dict(published=datetime.now().astimezone().isoformat(),commit=git(root,'rev-parse','HEAD'),arm=arm)
    record.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--log-root',type=Path,required=True)
    p.add_argument('--arms',nargs='+',required=True);p.add_argument('--wait-seconds',type=int,default=14400)
    a=p.parse_args();deadline=time.monotonic()+a.wait_seconds
    for arm in a.arms:
        ready=a.run/arm/'draw0_seed0/analysis/report_complete.json'
        while not ready.exists():
            if time.monotonic()>deadline:raise TimeoutError(f'waiting for {arm}')
            time.sleep(15)
        publish(a.run,a.log_root,arm)


if __name__=='__main__':main()
