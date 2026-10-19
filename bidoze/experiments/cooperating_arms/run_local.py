"""Serial local fits under the shared GPU lock; report every completed arm."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','source-run','labels-root','baseline-run']:
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--draw',type=int,default=0);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--arms',nargs='+',default=['exchange7','shared7','control1'])
    a=p.parse_args();here=Path(__file__).resolve().parent
    env={**os.environ,'PYTHONUNBUFFERED':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
    for arm in a.arms:
        out=a.run/arm/f'draw{a.draw}_seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
        if not (out/'complete.json').exists():
            args=[sys.executable,str(here/'run.py'),'--run',str(a.run),'--source-run',str(a.source_run),
                  '--labels-root',str(a.labels_root),'--arm',arm,'--draw',str(a.draw),'--seed',str(a.seed)]
            print(datetime.now().astimezone().isoformat(), 'waiting/running',arm,flush=True)
            with (out/'training.log').open('a') as log:
                subprocess.run(['flock',str(Path.home()/'.cache/arcasleep-gpu.lock'),*args],
                               env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        args=[sys.executable,str(here/'report.py'),'--run',str(a.run),'--source-run',str(a.source_run),
              '--baseline-run',str(a.baseline_run),'--arm',arm,'--draw',str(a.draw),'--seed',str(a.seed)]
        with (out/'report.log').open('a') as log:
            subprocess.run(args,env={**env,'JAX_PLATFORMS':'cpu'},stdout=log,stderr=subprocess.STDOUT,check=True)
        print(datetime.now().astimezone().isoformat(),'reported',arm,flush=True)


if __name__=='__main__':main()
