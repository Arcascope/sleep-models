# Autofish training on ECS/EC2

Everything a training run needs, defined here and created by one stack: its own
VPC, cluster, GPU capacity, roles, repository and log group. Nothing is shared
with anything else in the account, and `delete-stack` removes all of it.

## Files

| file | what it is |
|---|---|
| `infra.yaml` | CloudFormation: the whole environment, 21 resources |
| `train-task-definition.json` | task definition template; every value is substituted from stack outputs |
| `run_training_task.sh` | register a task definition for one image and run it once |

## 1. Create the stack

    aws cloudformation deploy \
      --stack-name autofish-train \
      --template-file deploy/aws/infra.yaml \
      --capabilities CAPABILITY_IAM \
      --region us-east-1

Parameters worth knowing:

- `InstanceType` (default `g6.2xlarge`) -- see **GPU sizing**.
- `ResultsBucket` / `ResultsPrefix` (default `arcascope-sleep-models-data` /
  `training_outputs`) -- the task role is scoped to exactly this prefix.
- `GpuAmiId` resolves the ECS-optimized GPU AMI from SSM at deploy time, so the
  driver and container runtime are already on the instance.

The ASG is created at **desired 0**. ECS managed scaling starts an instance when
a task is placed and returns to zero when it ends, so no GPU idles between runs.
The first task of the day therefore waits a few minutes for an instance to boot.

## 2. Push an image

    REPO=$(aws cloudformation describe-stacks --stack-name autofish-train \
      --query "Stacks[0].Outputs[?OutputKey=='RepositoryUri'].OutputValue" --output text)

    aws ecr get-login-password --region us-east-1 \
      | docker login --username AWS --password-stdin "${REPO%%/*}"

    scripts/build_train_image.sh \
      autofish-data:12Hz_log_PSD-<digest> --push "$REPO"

The repository is created with **immutable tags**, so a pushed tag cannot later
point at different bytes. `build_train_image.sh --push` prints the digest.

## 3. Run

    deploy/aws/run_training_task.sh

No arguments. It takes the most recently pushed image in the stack's
repository, resolves it to its image digest, and pins the task definition to
that digest -- a run's fingerprint records the code that produced it, and a
tag is a pointer that can be repointed, so the task definition names bytes.

Overrides, all optional:

    AUTOFISH_TRAINING_CONFIG=/path/to/training.json      # default: config/training.json, 30 epochs
    AUTOFISH_RUN_ID=my-run-name                          # default: run-<digest>
    AUTOFISH_STACK=autofish-train
    AWS_REGION=us-east-1

It prints the image tag and digest it chose, then commands to follow the logs
and check task status, and the S3 prefix the results will land under.

## GPU sizing

The job uses one GPU. Host requirements measured from a full six-model run:
**8.7 GiB RSS and one busy core**, so CPU and RAM are not the constraint.

| instance | GPU | VRAM | vCPU | RAM |
|---|---|---|---|---|
| g4dn.2xlarge | T4 | 16 GB | 8 | 32 GiB |
| g6.2xlarge *(default)* | L4 | 24 GB | 8 | 32 GiB |
| g5.2xlarge | A10G | 24 GB | 8 | 32 GiB |
| g6e.2xlarge | L40S | 48 GB | 8 | 64 GiB |

Measured peak with `XLA_PYTHON_CLIENT_PREALLOCATE=false` is around 22 GB, so
the 24 GB tier is the right one; the ~24.4 GB a 32 GB card reports by default is
XLA's 75% preallocation, not demand.

The task reserves 7168 CPU units and 28672 MiB, leaving headroom for the ECS
agent on an 8 vCPU / 32 GiB host. Changing `InstanceType` to a different size
means revisiting those two numbers in `train-task-definition.json`.

## Results

`harness.sh` uploads `$AUTOFISH_RESULTS_DIR/$AUTOFISH_RUN_ID` to
`$AUTOFISH_RESULTS_S3_PREFIX/$AUTOFISH_RUN_ID` from an `EXIT` trap, so a failed
run still ships its log and any partial checkpoints -- which is when they are
most worth having. `run-manifest.jsonl` is written last, so its presence means
everything before it arrived.

The upload refuses a destination that already holds objects unless
`AUTOFISH_UPLOAD_OVERWRITE` is set. The bucket is unversioned and a rerun of the
same data and code produces the same fingerprint, so a collision is expected
rather than exotic, and silently replacing results would be unrecoverable.

## Teardown

    aws cloudformation delete-stack --stack-name autofish-train

The ECR repository must be empty first, or the delete stalls on it.
