# Training on ECS/EC2

One CloudFormation stack, `arcasleep-train`, holds everything a training run
needs for every model family: a VPC, an ECS cluster, GPU capacity, roles, an
image repository per family and a log group. Nothing is shared with anything
else in the account, and `delete-stack` removes all of it.

## Files

| file | what it is |
|---|---|
| `aws/infra.yaml` | CloudFormation: the whole environment |
| `aws/task-definition.json` | task definition template; `run_task.sh` fills it from stack outputs |
| `docker/data.Dockerfile` | data image: one feature-cache prefix, data only |
| `scripts/build_data_image.sh` | build (and push) a data image from a cache directory |
| `scripts/build_train_image.sh` | build (and push) a family's training image on a data image |
| `scripts/run_task.sh` | launch training: one task per phase |
| `scripts/status.sh` | free GPU slots and the tasks holding the others |

Each family's training image is built from `<family>/docker/train.Dockerfile`;
its entrypoint is `<family>/scripts/harness.sh`.

## Account limits

These limits apply to the whole AWS account, not just this stack:

- **On-demand G/VT instances: 32 vCPUs.** One 8-vCPU GPU instance uses 8, so at
  most four can run at once across every stack.
- **Spot G/VT instances: 0.** The stack uses on-demand capacity only.
- **VPCs in us-east-1: 5.** The stack makes one.

`MaxInstances` (default 3) caps how many GPU instances the stack runs at once;
each instance runs one task.

## 1. Create the stack

    aws cloudformation deploy \
      --stack-name arcasleep-train \
      --template-file deploy/aws/infra.yaml \
      --capabilities CAPABILITY_IAM \
      --region us-east-1

Parameters worth knowing:

- `InstanceType` (default `g6.2xlarge`): see **GPU sizing**.
- `MaxInstances` (default 3): see **Account limits**.
- `ResultsBucket` / `ResultsPrefix` (default `arcascope-sleep-models-data` /
  `training_outputs`): the task role can write only under this prefix.

The autoscaling group starts at **0 instances**. ECS starts one instance per
task placed, up to `MaxInstances`, and returns to zero after the tasks end;
no GPU runs while idle. Each task therefore waits a few minutes for its
instance to boot.

Adding a model family means adding its repository and output to
`infra.yaml`, its case to `run_task.sh`, and redeploying.

## 2. Push images

Images go in two steps: a data image holding one feature-cache prefix, then a
training image per family built on it. Every family that reads the same cache
prefix (BiDoze and UNet both read `12Hz_log_PSD_aligned`) shares one data
image.

    out() { aws cloudformation describe-stacks --stack-name arcasleep-train \
      --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
    aws ecr get-login-password --region us-east-1 \
      | docker login --username AWS --password-stdin "$(out DataRepositoryUri | cut -d/ -f1)"

    deploy/scripts/build_data_image.sh bidoze/feature_cache/12Hz_log_PSD_aligned \
      --push "$(out DataRepositoryUri)"
    # prints <data repo>:12Hz_log_PSD_aligned-<digest>@sha256:...

    deploy/scripts/build_train_image.sh unet <data image reference> --push "$(out UnetRepositoryUri)"
    # prints the run_task.sh command for the pushed image

Data images are tagged by the cache manifest's digest, and training images by
`<data tag>-<git commit>`, with `-dirty` added when the working tree has
uncommitted changes. Tags are **immutable**: pushing a tag that already exists
fails, which means that data or that commit is already published.

The training recipe and split plans are copied into the image, so a changed
config needs a new commit and a new training image.

## 3. Run

    deploy/scripts/status.sh
    deploy/scripts/run_task.sh unet <image> fold0 fold1 fold2 fold3 fold4 final

Each named phase runs as its own task on its own GPU, all under one run id.
With no phase named, one task runs every phase in order.

A launch that needs more GPU slots than are free is refused. With `--queue`,
the script stays in the foreground and launches each phase as a slot frees.

Options:

    --config PATH       recipe inside the image (default: the family's release recipe)
    --run-id ID         default <image tag>/<UTC launch time>
    --env KEY=VALUE     extra container environment, repeatable
    --max-hours N       stop the task after N hours (default 24)
    --queue             wait for free slots instead of refusing
    --dry-run           print the task definition; launch nothing

A task that reaches `--max-hours` is stopped. Its harness still uploads what
the run produced.

## Results

Each phase is uploaded to `s3://<bucket>/<prefix>/<run id>/training/<phase>/`
when the harness exits, including after a crash or a timeout, so a failed run
still ships its log and any checkpoints. `run-manifest.jsonl` is written last
in each phase, so its presence means the rest of that phase arrived.

The upload refuses a destination that already holds objects unless
`BIDOZE_UPLOAD_OVERWRITE` or `UNET_UPLOAD_OVERWRITE` is set (pass it with
`--env`). The bucket is unversioned, so replaced results cannot be recovered.

## GPU sizing

Each task uses one GPU. The task reserves 7168 CPU units and 28672 MiB, which
leaves headroom for the ECS agent on an 8 vCPU host with 32 GiB or more.
Changing `InstanceType` to a different size means revisiting those two numbers
in `aws/task-definition.json`.

| instance | GPU | VRAM | vCPU | RAM |
|---|---|---|---|---|
| g6.2xlarge *(default)* | L4 | 24 GB | 8 | 32 GiB |
| g5.2xlarge | A10G | 24 GB | 8 | 32 GiB |
| g6e.2xlarge | L40S | 48 GB | 8 | 64 GiB |

Tasks run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so JAX allocates GPU
memory as it needs it rather than claiming 75% of the card at start.

Both families fit on a 24 GB card:

- **BiDoze:** peak around 22 GB.
- **UNet:** a training phase completes with JAX limited to 22 GB; it logs
  failed allocation attempts of 4–8 GB, which do not stop training. On a
  32 GB card with no limit, UNet allocates up to 30 GB. Host memory peaks at
  9 GB.

g6e.2xlarge is roughly twice the hourly price of g6.2xlarge; switch to it only
for a model that does not fit in 24 GB.

## Teardown

    aws cloudformation delete-stack --stack-name arcasleep-train

The ECR repositories must be empty first, or the deletion stalls on them.
