---
name: aws-gpu-runs
description: Train arcasleep models (BiDoze, UNet) on AWS GPUs. Covers building a data image for a feature cache and a training image on it, pushing both to ECR, launching one ECS task per CV phase on the shared arcasleep-train stack, monitoring, and fetching results from S3, all within the account's GPU limits. Use when an experiment needs more GPUs than the workstation has, or when asked to run, queue, check or collect a training run on AWS/ECS.
---

# GPU training runs on AWS

The `arcasleep-train` CloudFormation stack (us-east-1) runs training images on
GPU instances: one ECS task per phase, one GPU per task. Its reference is
`deploy/README.md`; read it once before a first run. Every script named here is
under `deploy/scripts/` and runs from the repository root.

## Limits

The GPU capacity is small, and other agents share it. Follow these rules
every time:

- **Never create, update or delete the stack, change its parameters
  (`MaxInstances`, `InstanceType`), request quota changes, or create AWS
  resources outside these scripts.** If a run needs any of these, stop and ask
  the user.
- **Check `deploy/scripts/status.sh` before every launch.** It shows free
  slots, and the run id and phase of every active task.
- **Launch no more tasks than there are free slots** unless the user has
  approved `--queue` for this run. With other agents' tasks active, leave them
  slots rather than queueing a full sweep behind them.
- **Never stop a task you did not launch.** To stop your own task:
  `aws ecs stop-task --cluster arcasleep-train --task <id> --region us-east-1`.
- **Set `--max-hours`** to about 1.5 × the expected duration. The default
  24 h limit is a backstop, not an estimate.
- **Run a wiring check before a sweep:** one phase, `--config
  config/training-smoke.json` (2 epochs) and `--max-hours 1`. A sweep that
  fails on its first epoch wastes six instance boots.
- An instance costs money from boot to scale-in, about 15 minutes after its
  task ends. Group work into fewer, larger launches rather than many short
  ones.

## Image layout

An image has two parts:

| image | repository | changes when | built by |
|---|---|---|---|
| data | `arcasleep-train/data` | the feature cache changes: a new `cache_prefix` in `config/inputs.json` or new recordings | `build_data_image.sh` |
| training | `arcasleep-train/<family>` | code, dependencies or anything in `<family>/config/` or `<family>/scripts/` | `build_train_image.sh` |

The training image copies the data image's cache in as one layer, then adds
the package, the family's `config/` and `scripts/`, and the harness as its
entrypoint. The recipe and split plans are therefore inside the image:
**changing a config means a new commit and a new training image.** No data
image is needed for that.

Decide which images you need:

1. **Same cache, new code or config**: reuse the existing data image and
   build a new training image. This is the usual case.
2. **New cache prefix** (different spectrogram settings, new datasets): build
   the cache locally with `arcasleep process`, then build a data image from
   it and a training image on top.
3. **New dependency** for a family: edit `<family>/docker/train.Dockerfile`,
   keeping pinned versions (`jax` is pinned exactly because it brings in the
   CUDA libraries).
4. **New model family**: needs a repository in the stack. Ask the user.

## Workflow

### 1. Log in to ECR, find the repositories

```bash
out() { aws cloudformation describe-stacks --stack-name arcasleep-train --region us-east-1 \
  --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
DATA_REPO=$(out DataRepositoryUri)
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin "${DATA_REPO%%/*}"
```

### 2. The data image: reuse one if it exists

```bash
aws ecr describe-images --repository-name arcasleep-train/data --region us-east-1 \
  --query 'sort_by(imageDetails,&imagePushedAt)[].[imageTags[0],imageDigest,imagePushedAt]' --output text
```

A tag is `<cache_prefix>-<manifest digest[:12]>`. If one matches your cache
prefix and the cache has not changed, use `"$DATA_REPO:<tag>@<digest>"`.
Otherwise:

```bash
deploy/scripts/build_data_image.sh <path>/feature_cache/<cache_prefix> --push "$DATA_REPO"
```

Pushing a tag that already exists fails. That failure means the identical
cache is already published, so use the existing image.

### 3. The training image

Commit first: the tag is `<data tag>-<git commit>`, and an image built from
uncommitted changes is tagged `-dirty`. A `-dirty` image cannot be traced back
to its exact code; use it only for wiring checks.

```bash
deploy/scripts/build_train_image.sh <family> "<data image reference>" --push "$(out <Family>RepositoryUri)"
```

`<Family>RepositoryUri` is `BidozeRepositoryUri` or `UnetRepositoryUri`. The
script prints the exact `run_task.sh` command, with the image pinned by digest.
A training image is about 6 GB. Pushing a new commit only uploads its changed
layers, and the data layer is unchanged.

### 4. Launch

```bash
deploy/scripts/status.sh
deploy/scripts/run_task.sh --dry-run <family> <image> fold0            # check the task definition
deploy/scripts/run_task.sh --max-hours 6 <family> <image> fold0 fold1 fold2
```

- Each phase named is one task. Phases are `fold0`…`fold4` and `final` for the
  committed 5-fold plans.
- `--run-id` defaults to `<image tag>/<UTC time>`. Name it after the
  experiment when it belongs in the research log.
- `--config config/<recipe>.json` selects another recipe already in the image.
- `--env KEY=VALUE` passes harness variables, such as
  `UNET_SPLIT_PLAN=config/splits/<plan>.json` or `BIDOZE_CV_CONFIG=...`.
- Phases of one run can be launched in separate calls with the same
  `--run-id`, as slots free up.

### 5. Monitor

```bash
deploy/scripts/status.sh
aws logs tail /ecs/arcasleep-train --follow --log-stream-name-prefix <family> --region us-east-1
aws ecs describe-tasks --cluster arcasleep-train --tasks <id> --region us-east-1 \
  --query 'tasks[].[lastStatus,stoppedReason,containers[0].exitCode]' --output text
```

Each task waits a few minutes in `PROVISIONING` for an instance to boot. If it
stays there for more than about 15 minutes, check the scaling activity:

```bash
aws autoscaling describe-scaling-activities --auto-scaling-group-name arcasleep-train-asg \
  --region us-east-1 --max-items 5 --query 'Activities[].[StatusCode,StatusMessage]' --output text
```

`InsufficientInstanceCapacity` means AWS has no instance of that type free in
any of the four zones; wait and try later. `VcpuLimitExceeded` means the
account's GPU quota is in use elsewhere: report it to the user.

Exit code 124 means `--max-hours` stopped the task. The phases still upload.

### 6. Collect results

Each phase uploads to `s3://arcascope-sleep-models-data/training_outputs/<run id>/training/<phase>/`.
`run-manifest.jsonl` is written last, so a phase whose prefix has no
`run-manifest.jsonl` has not finished uploading. Download into the family's
`results/` directory:

```bash
aws s3 sync s3://arcascope-sleep-models-data/training_outputs/<run id>/ <family>/results/<run id>/
```

Then pool and compare as for a local run (`arcasleep summarize`, the family's
`evaluate`). Record the run id, the image digest and the recipe in the
research log.

## Cleaning up

ECR keeps the 10 most recent data images and the 20 most recent training
images per family; older ones expire on their own. Do not delete images by
hand: another agent's run may be pinned to them.
