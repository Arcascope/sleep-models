"""Bounded collector for this study: retrieve only completed tasks from the confidential prefix."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import boto3

PREFIX='research_log_artifacts/bidoze/results/boat-evidence-2026-10-11/'
BUCKET='arcasleep-conf-sleep-models-data'


def main(a):
    s3=boto3.client('s3',region_name='us-east-1');ecs=boto3.client('ecs',region_name='us-east-1');deadline=time.time()+a.hours*3600;downloaded=set();here=Path(__file__).parent
    while time.time()<deadline:
        path=a.run/'setup/aws/aws-tasks.json';tasks=json.loads(path.read_text()) if path.exists() else [];status=[]
        if tasks:
            described=ecs.describe_tasks(cluster='arcasleep-train',tasks=[r['task'] for r in tasks])['tasks']
            for task in described:
                record=next(r for r in tasks if r['task']==task['taskArn']);phase=record['phase'];last=task['lastStatus'];status.append(dict(phase=phase,status=last,exit_code=task.get('containers',[{}])[0].get('exitCode'),reason=task.get('stoppedReason')))
                if last!='STOPPED' or phase in downloaded:continue
                key=PREFIX+f'aws/training/{phase}/';got=0
                for page in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET,Prefix=key):
                    for obj in page.get('Contents',[]):
                        target=a.run/obj['Key'][len(PREFIX):];target.parent.mkdir(parents=True,exist_ok=True);s3.download_file(BUCKET,obj['Key'],str(target));got+=1
                if got:downloaded.add(phase);print(f'fetched {phase}: {got} files',flush=True)
        (a.run/'setup/task-status.json').write_text(json.dumps(status,indent=2))
        subprocess.run([a.python,str(here/'score.py'),str(a.run),str(a.run/'analysis')],check=True)
        local=all((a.run/f'local/full/{f}/seed0/complete.json').exists() for f in 'ED')
        if local and len(downloaded)==9:
            (a.run/'setup/collection-complete.json').write_text(json.dumps(dict(downloaded=sorted(downloaded),status='screening jobs complete; check task-status.json for failures')));return
        time.sleep(30)
    (a.run/'setup/collection-time-limit.json').write_text(json.dumps(dict(downloaded=sorted(downloaded),status='collector stopped at bounded time limit')))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--hours',type=float,default=3);p.add_argument('--python',default='/home/eric/venvs/.bidoze/bin/python');main(p.parse_args())
