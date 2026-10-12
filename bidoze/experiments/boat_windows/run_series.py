"""Finite local six-arm window and pooling screen with per-arm scoring."""
import argparse,json,os,subprocess,time
from pathlib import Path

def main(a):
    setup=a.run/'setup';setup.mkdir(parents=True,exist_ok=True);watch=setup/'queue';watch.mkdir(exist_ok=True)
    (watch/'config.json').write_text(json.dumps(dict(pid=os.getpid(),arm='window_pooling_queue',fold=a.fold,seed=0)))
    data=setup/'data';base=Path(__file__).resolve().parent
    def execute(script,*args):
        command=[a.python,str(base/script),*map(str,args)];print('RUN',script,*args,flush=True);subprocess.run(command,check=True)
    execute('prepare.py',a.root,a.source_data,a.cache,data)
    for pooling in ('mean','cnn'):
        for layout in ('native','contained','nonoverlap'):
            dest=a.run/'training'/f'{layout}_{pooling}'/a.fold/'seed0'
            execute('train.py',data,dest,'--fold',a.fold,'--seed','0','--layout',layout,'--pooling',pooling,'--max-epochs',a.max_epochs)
            execute('score.py',a.run,a.run/'analysis','--fold',a.fold)
    (watch/'complete.json').write_text(json.dumps(dict(arms=6,fold=a.fold,seed=0,completed_unix=time.time())))
    print('ALL SIX ARMS COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','source-data','cache','run'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--python',required=True);p.add_argument('--fold',default='D');p.add_argument('--max-epochs',type=int,default=100)
    main(p.parse_args())
