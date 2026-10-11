"""Launch finite screening jobs on free ECS slots, using a role limited to this confidential run."""
import argparse
import copy
import json
import time
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

BUCKET='arcasleep-conf-sleep-models-data'
PREFIX='research_log_artifacts/bidoze/results/boat-evidence-2026-10-11'
ROLE='arcasleep-boat-evidence-oct11'
PHASES=['head/E/seed0','land/E/seed0','full/A/seed0','full/B/seed0','full/C/seed0',
        'head/A/seed0','head/B/seed0','head/C/seed0','head/D/seed0']


def main(a):
    session=boto3.Session(region_name='us-east-1'); iam=session.client('iam');ecs=session.client('ecs');cfn=session.client('cloudformation')
    outputs={r['OutputKey']:r['OutputValue'] for r in cfn.describe_stacks(StackName='arcasleep-train')['Stacks'][0]['Outputs']}
    cluster=outputs['ClusterName'];capacity=outputs['CapacityProviderName'];limit=int(outputs['MaxInstances'])
    try:role=iam.get_role(RoleName=ROLE)['Role']
    except iam.exceptions.NoSuchEntityException:
        role=iam.create_role(RoleName=ROLE,AssumeRolePolicyDocument=json.dumps({'Version':'2012-10-17','Statement':[{'Effect':'Allow','Principal':{'Service':'ecs-tasks.amazonaws.com'},'Action':'sts:AssumeRole'}]}),Description='October 11 confidential study; only its artifact prefix')['Role']
    policy={'Version':'2012-10-17','Statement':[
        {'Effect':'Allow','Action':['s3:PutObject','s3:GetObject','s3:AbortMultipartUpload'],'Resource':f'arn:aws:s3:::{BUCKET}/{PREFIX}/*'},
        {'Effect':'Allow','Action':'s3:ListBucket','Resource':f'arn:aws:s3:::{BUCKET}','Condition':{'StringLike':{'s3:prefix':[f'{PREFIX}/*']}}}]}
    iam.put_role_policy(RoleName=ROLE,PolicyName='this-run-only',PolicyDocument=json.dumps(policy))
    base=ecs.describe_task_definition(taskDefinition='arcasleep-train-bidoze:15')['taskDefinition']
    fields=['networkMode','executionRoleArn','cpu','memory','requiresCompatibilities','containerDefinitions','volumes']
    td={k:copy.deepcopy(base[k]) for k in fields if k in base};td.update(family='arcasleep-boat-evidence-oct11',taskRoleArn=role['Arn'])
    container=td['containerDefinitions'][0];container['image']=a.image
    env={r['name']:r['value'] for r in container.get('environment',[])}
    env.update(BIDOZE_RUN_ID='aws',BIDOZE_RESULTS_S3_PREFIX=f's3://{BUCKET}/{PREFIX}',BIDOZE_EXPERIMENT='experiments/boat_evidence',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2')
    container['environment']=[{'name':k,'value':v} for k,v in env.items()]
    container['entryPoint']=['timeout','--kill-after=10m','2h','bash','scripts/harness.sh']
    container['logConfiguration']['options']['awslogs-stream-prefix']='boat-evidence'
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'task-definition.json').write_text(json.dumps(td,indent=2))
    (a.out/'role-policy.json').write_text(json.dumps(policy,indent=2))
    if a.dry_run: print(json.dumps({'slots':limit,'phases':PHASES,'image':a.image,'role':role['Arn'],'destination':env['BIDOZE_RESULTS_S3_PREFIX']},indent=2));return
    arn=ecs.register_task_definition(**td)['taskDefinition']['taskDefinitionArn']
    record=a.out/'aws-tasks.json'; tasks=json.loads(record.read_text()) if record.exists() else []
    launched={r['phase'] for r in tasks}
    for phase in PHASES:
        if phase in launched:continue
        while len(ecs.list_tasks(cluster=cluster,desiredStatus='RUNNING')['taskArns'])>=limit:
            print('all slots occupied; checking in 30 seconds',flush=True);time.sleep(30)
        response=ecs.run_task(cluster=cluster,taskDefinition=arn,capacityProviderStrategy=[dict(capacityProvider=capacity,weight=1)],overrides={'containerOverrides':[{'name':container['name'],'command':[phase]}]},tags=[dict(key='run-id',value='boat-evidence-2026-10-11'),dict(key='phase',value=phase)])
        if response.get('failures') or not response.get('tasks'):raise RuntimeError(response)
        task=response['tasks'][0];tasks.append(dict(phase=phase,task=task['taskArn'],definition=arn,image=a.image,launch_epoch=time.time()))
        record.write_text(json.dumps(tasks,indent=2));print(f"launched {phase}: {task['taskArn']}",flush=True);time.sleep(5)
    (a.out/'aws-queue-complete.json').write_text(json.dumps(dict(launched=len(tasks),status='all submitted; not all training complete')))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('image');p.add_argument('out',type=Path);p.add_argument('--dry-run',action='store_true');main(p.parse_args())
