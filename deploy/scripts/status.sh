#!/usr/bin/env bash
# Show the stack's GPU slots and the tasks holding them.
#
#   deploy/scripts/status.sh
#
# Read-only. ARCASLEEP_STACK and AWS_REGION as for run_task.sh.
set -euo pipefail

STACK="${ARCASLEEP_STACK:-arcasleep-train}"
REGION="${AWS_REGION:-us-east-1}"

OUTPUTS=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
    --query "Stacks[0].Outputs[].[OutputKey,OutputValue]" --output text)
output() { awk -v k="$1" '$1 == k { print $2 }' <<<"$OUTPUTS"; }
CLUSTER=$(output ClusterName)
MAX_INSTANCES=$(output MaxInstances)

read -r INSTANCES DESIRED < <(aws autoscaling describe-auto-scaling-groups \
    --auto-scaling-group-names "$(output AutoScalingGroupName)" --region "$REGION" \
    --query 'AutoScalingGroups[0].[length(Instances), DesiredCapacity]' --output text)
mapfile -t TASKS < <(aws ecs list-tasks --cluster "$CLUSTER" --desired-status RUNNING \
    --region "$REGION" --query 'taskArns[]' --output text | tr '\t' '\n' | sed '/^$/d')
# The account-wide on-demand G/VT vCPU quota; other stacks draw on it too.
QUOTA=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA \
    --region "$REGION" --query 'Quota.Value' --output text 2>/dev/null || echo unknown)

echo "stack       $STACK ($(output InstanceType))"
echo "slots       $((MAX_INSTANCES - ${#TASKS[@]})) free of $MAX_INSTANCES"
echo "instances   $INSTANCES running, $DESIRED desired"
echo "G quota     $QUOTA vCPUs on-demand, account-wide"
echo

if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "no active tasks"
    exit 0
fi
printf '%-34s %-13s %-8s %s\n' TASK STATUS PHASE RUN-ID
aws ecs describe-tasks --cluster "$CLUSTER" --tasks "${TASKS[@]}" --include TAGS \
    --region "$REGION" --output json \
    | jq -r '.tasks[] | [(.taskArn | split("/")[-1]), .lastStatus,
        ((.tags // []) | map(select(.key == "phase"))[0].value // "-"),
        ((.tags // []) | map(select(.key == "run-id"))[0].value // "-")] | @tsv' \
    | awk -F'\t' '{ printf "%-34s %-13s %-8s %s\n", $1, $2, $3, $4 }'
