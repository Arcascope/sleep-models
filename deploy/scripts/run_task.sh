#!/usr/bin/env bash
# Run training on the arcasleep-train stack's GPU capacity: one ECS task per
# phase, or one task for the whole recipe when no phase is named.
usage() {
    cat <<'USAGE'
usage: deploy/scripts/run_task.sh [options] FAMILY IMAGE_URI [PHASE ...]

FAMILY is a model family (bidoze, unet); IMAGE_URI is a training image in that
family's repository on the stack. Each PHASE (fold0, fold1, ..., final) runs as
its own task on its own GPU, all under one run id. With no PHASE, one task
runs every phase in order.

Every task needs a free GPU slot (the stack's MaxInstances, less the tasks
already active). A launch needing more slots than are free is refused unless
--queue is given.

options:
  --config PATH     recipe inside the image, relative to /opt/FAMILY
                    (default: bidoze config/training-balanced.json,
                    unet config/training.json)
  --run-id ID       run id (default <image tag>/<UTC launch time>)
  --env KEY=VALUE   extra container environment; repeatable
  --max-hours N     stop a task after N hours (default 24); the harness
                    still uploads what it has
  --queue           wait for slots and launch each phase as one frees up
  --dry-run         print the task definition and the plan; launch nothing

environment:
  ARCASLEEP_STACK   stack name (default arcasleep-train)
  AWS_REGION        region     (default us-east-1)
USAGE
}
set -euo pipefail

STACK="${ARCASLEEP_STACK:-arcasleep-train}"
REGION="${AWS_REGION:-us-east-1}"
TRAINING_CONFIG=""
RUN_ID=""
MAX_HOURS=24
QUEUE=0
DRY_RUN=0
EXTRA_ENV=()
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) TRAINING_CONFIG=$2; shift 2 ;;
        --run-id) RUN_ID=$2; shift 2 ;;
        --env) [[ "$2" == *=* ]] || { echo "--env takes KEY=VALUE" >&2; exit 2; }
               EXTRA_ENV+=("$2"); shift 2 ;;
        --max-hours) MAX_HOURS=$2; shift 2 ;;
        --queue) QUEUE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) usage >&2; exit 2 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
[[ ${#POSITIONAL[@]} -ge 2 ]] || { usage >&2; exit 2; }
MODEL_FAMILY=${POSITIONAL[0]}
IMAGE=${POSITIONAL[1]}
PHASES=("${POSITIONAL[@]:2}")

case "$MODEL_FAMILY" in
    bidoze) ENV_PREFIX=BIDOZE; REPO_KEY=BidozeRepositoryUri
            : "${TRAINING_CONFIG:=config/training-balanced.json}" ;;
    unet)   ENV_PREFIX=UNET; REPO_KEY=UnetRepositoryUri
            : "${TRAINING_CONFIG:=config/training.json}" ;;
    *) echo "unknown model family: $MODEL_FAMILY" >&2; exit 2 ;;
esac
[[ "$MAX_HOURS" =~ ^[0-9]+$ && "$MAX_HOURS" -ge 1 ]] || { echo "--max-hours takes a whole number ≥ 1" >&2; exit 2; }

OUTPUTS=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
    --query "Stacks[0].Outputs[].[OutputKey,OutputValue]" --output text)
output() { awk -v k="$1" '$1 == k { print $2 }' <<<"$OUTPUTS"; }

CLUSTER=$(output ClusterName)
CAPACITY=$(output CapacityProviderName)
MAX_INSTANCES=$(output MaxInstances)
REPO=$(output "$REPO_KEY")
EXECUTION_ROLE_ARN=$(output ExecutionRoleArn)
TASK_ROLE_ARN=$(output TaskRoleArn)
LOG_GROUP=$(output LogGroupName)
RESULTS_URI=$(output ResultsUri)

# An image from another family's repository would run that family's harness
# under this family's environment names.
if [[ "$IMAGE" != "$REPO:"* && "$IMAGE" != "$REPO@"* ]]; then
    echo "$IMAGE is not in the $MODEL_FAMILY repository ($REPO)" >&2
    exit 2
fi

IMAGE_TAG=${IMAGE%@*}
IMAGE_TAG=${IMAGE_TAG##*:}
[[ -n "$RUN_ID" ]] || RUN_ID="$IMAGE_TAG/$(date -u +%Y-%m-%dT%Hh%MmUTC)"
TASK_FAMILY="$STACK-$MODEL_FAMILY"

export TASK_FAMILY MODEL_FAMILY ENV_PREFIX EXECUTION_ROLE_ARN TASK_ROLE_ARN IMAGE \
    RUN_ID RESULTS_URI LOG_GROUP TRAINING_CONFIG AWS_REGION="$REGION"
TD=$(mktemp)
trap 'rm -f "$TD"' EXIT
EXTRA_JSON=$(printf '%s\n' "${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}" | jq -R -s -c \
    '[split("\n")[] | select(length > 0) | index("=") as $i | {name: .[:$i], value: .[$i+1:]}]')
# The time limit sits in the entrypoint because RunTask can override a
# container's command but not its entrypoint. timeout signals the whole
# process group; the harness's EXIT trap then uploads what the run produced.
envsubst < "$(dirname "$0")/../aws/task-definition.json" \
    | jq --argjson extra "$EXTRA_JSON" --arg hours "${MAX_HOURS}h" '
        .containerDefinitions[0].environment += $extra
        | .containerDefinitions[0].entryPoint =
            ["timeout", "--kill-after=10m", $hours, "bash", "scripts/harness.sh"]' > "$TD"

# One task per named phase; a single task with no arguments runs them all.
if [[ ${#PHASES[@]} -eq 0 ]]; then
    TASK_ARGS=("")
else
    TASK_ARGS=("${PHASES[@]}")
fi

active_tasks() {
    aws ecs list-tasks --cluster "$CLUSTER" --desired-status RUNNING --region "$REGION" \
        --query 'length(taskArns)' --output text
}

ACTIVE=$(active_tasks)
FREE=$((MAX_INSTANCES - ACTIVE))
echo "==> $MODEL_FAMILY  $IMAGE"
echo "    recipe   $TRAINING_CONFIG"
echo "    run id   $RUN_ID"
echo "    phases   ${PHASES[*]:-all, in one task}"
echo "    slots    $FREE free of $MAX_INSTANCES ($ACTIVE active)"
echo "    limit    $MAX_HOURS h per task"

if [[ $DRY_RUN -eq 1 ]]; then
    echo
    cat "$TD"
    exit 0
fi

if [[ ${#TASK_ARGS[@]} -gt $FREE && $QUEUE -eq 0 ]]; then
    echo "needs ${#TASK_ARGS[@]} slots and $FREE are free; pass --queue to wait for them," >&2
    echo "or name fewer phases" >&2
    exit 1
fi

ARN=$(aws ecs register-task-definition --cli-input-json "file://$TD" --region "$REGION" \
      --query 'taskDefinition.taskDefinitionArn' --output text)
echo "==> registered $ARN"

TASKS=()
for arg in "${TASK_ARGS[@]}"; do
    while [[ $(active_tasks) -ge $MAX_INSTANCES ]]; do
        echo "    all $MAX_INSTANCES slots busy; checking again in 60 s"
        sleep 60
    done
    OVERRIDES='{}'
    if [[ -n "$arg" ]]; then
        OVERRIDES=$(jq -n -c --arg name "$TASK_FAMILY" --arg phase "$arg" \
            '{containerOverrides: [{name: $name, command: [$phase]}]}')
    fi
    TASK=$(aws ecs run-task \
        --cluster "$CLUSTER" \
        --task-definition "$ARN" \
        --capacity-provider-strategy "capacityProvider=$CAPACITY,weight=1" \
        --overrides "$OVERRIDES" \
        --tags "key=run-id,value=$RUN_ID" "key=phase,value=${arg:-all}" \
        --region "$REGION" --query 'tasks[0].taskArn' --output text)
    [[ "$TASK" == arn:* ]] || { echo "run-task returned no task for ${arg:-all}" >&2; exit 1; }
    echo "==> ${arg:-all}  $TASK"
    TASKS+=("${TASK##*/}")
    # Give ECS a moment to count the new task before the next slot check.
    sleep 5
done

echo
echo "capacity starts at zero, so each task waits a few minutes for an instance to boot."
echo "follow:    aws logs tail $LOG_GROUP --follow --log-stream-name-prefix $MODEL_FAMILY --region $REGION"
echo "status:    aws ecs describe-tasks --cluster $CLUSTER --region $REGION --tasks ${TASKS[*]} --query 'tasks[].[taskArn,lastStatus,stoppedReason,containers[0].exitCode]' --output text"
echo "results:   $RESULTS_URI/$RUN_ID/training/<phase>/"
