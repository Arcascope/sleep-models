#!/usr/bin/env bash
# Run one training job on the stack's GPU capacity.
usage() {
    cat <<'USAGE'
usage: scripts/run_training_task.sh IMAGE_URI

Run one training job on the stack's GPU capacity using IMAGE_URI, e.g.

  scripts/run_training_task.sh 904277341831.dkr.ecr.us-east-1.amazonaws.com/bidoze/train:12Hz_log_PSD-bcbb92714773-2d5cd52-dirty

Everything else comes from the CloudFormation stack's outputs.

Environment overrides:
  BIDOZE_STACK               stack name            (default bidoze-train)
  AWS_REGION                 region                (default us-east-1)
  BIDOZE_TRAINING_CONFIG     recipe                (default config/training.json)
  BIDOZE_RUN_ID              run id                (default <image tag>/<UTC launch time>)
USAGE
}
set -euo pipefail

if [[ $# -ne 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 1
fi
IMAGE="$1"

STACK="${BIDOZE_STACK:-bidoze-train}"
REGION="${AWS_REGION:-us-east-1}"
TRAINING_CONFIG="${BIDOZE_TRAINING_CONFIG:-config/training.json}"

out() {
    aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
        --query "Stacks[0].Outputs" --output text
}

# Capture the output once
STACK_DETAILS=$(out)

# The returned values are 1 per line. Use Grep to find the line.
CLUSTER=$(echo "$STACK_DETAILS" | grep ClusterName | cut -f2)
CAPACITY=$(echo "$STACK_DETAILS" | grep CapacityProviderName | cut -f2)
EXECUTION_ROLE_ARN=$(echo "$STACK_DETAILS" | grep ExecutionRoleArn | cut -f2)
TASK_ROLE_ARN=$(echo "$STACK_DETAILS" | grep TaskRoleArn | cut -f2)
LOG_GROUP=$(echo "$STACK_DETAILS" | grep LogGroupName | cut -f2)
RESULTS_URI=$(echo "$STACK_DETAILS" | grep ResultsUri | cut -f2)

RUN_ID="${BIDOZE_RUN_ID:-${IMAGE##*:}/$(date -u +%Y-%m-%dT%Hh%MmUTC)}"

export EXECUTION_ROLE_ARN TASK_ROLE_ARN IMAGE RUN_ID RESULTS_URI LOG_GROUP TRAINING_CONFIG AWS_REGION="$REGION"
TD=$(mktemp)
trap 'rm -f "$TD"' EXIT
envsubst < "$(dirname "$0")/../deploy/aws/train-task-definition.json" > "$TD"

echo "==> image   $IMAGE"
echo "    recipe  $TRAINING_CONFIG"
echo "    run id  $RUN_ID"

ARN=$(aws ecs register-task-definition --cli-input-json "file://$TD" --region "$REGION" \
      --query 'taskDefinition.taskDefinitionArn' --output text)
echo "==> registered $ARN"

# bridge network mode: no --network-configuration. The task reaches ECR, S3
# and CloudWatch through the instance, which sits in a public subnet; the
# stack's security group is attached to the instance and allows egress only.
TASK=$(aws ecs run-task \
    --cluster "$CLUSTER" \
    --task-definition "$ARN" \
    --capacity-provider-strategy "capacityProvider=$CAPACITY,weight=1" \
    --region "$REGION" --query 'tasks[0].taskArn' --output text)
echo "==> running $TASK"

echo
echo "capacity starts at zero, so the first task waits for an instance to boot."
echo "follow:    aws logs tail $LOG_GROUP --follow --region $REGION"
echo "status:    aws ecs describe-tasks --cluster $CLUSTER --tasks $TASK --region $REGION --query 'tasks[0].[lastStatus,stoppedReason]'"
echo "results:   $RESULTS_URI/$RUN_ID/"
