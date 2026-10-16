---
name: aws-gpu-runs
description: Train arcasleep models (BiDoze, UNet) on AWS GPUs. Covers building feature caches from the raw datasets in S3, building data and training images in CodeBuild, launching one ECS task per CV phase on the shared arcasleep-train stack, monitoring, and fetching results from S3, all within the account's GPU limits. Use when an experiment needs more GPUs than the workstation has, or when asked to run, queue, check or collect a training run on AWS/ECS.
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
- **Build images with `remote_build.sh`, not on the workstation.** Builds
  cost about $0.02 per minute and at most two run at once; don't start a
  build whose image is already in ECR (step 2 shows how to check).

## Images

| image | changes when | built by |
|---|---|---|
| deps | the jax or pisces-lite pin, or `deploy/docker/deps.Dockerfile` | `remote_build.sh deps` |
| prep | code that `arcasleep process` runs | `remote_build.sh prep` |
| data | the feature cache: a new `cache_prefix` in `config/inputs.json`, or new subjects | a prep task, then `remote_build.sh data <cache_prefix>` |
| training | code, or anything in `<family>/config/` or `<family>/scripts/` | `remote_build.sh train <family> <data image>` |

The training image is deps + code + the family's `config/` and `scripts/` +
the data image's cache. The recipe and split plans are therefore inside the
image: **changing a config means a new commit and a new training image.** No
data image is needed for that.

`remote_build.sh` builds `HEAD` as committed: **commit first** (a local commit
is enough; no push needed). Uncommitted changes are not in the image.

Decide which images you need:

1. **Same cache, new code or config**: build a new training image on the
   existing data image. This is the usual case.
2. **New cache prefix or new datasets**: run a prep task to build the cache in
   S3, then build a data image, then a training image on it.
3. **New dependency** for a family: edit `<family>/docker/train.Dockerfile`,
   pinning the version. A change to `deps.Dockerfile` or a pin rebuilds deps,
   which takes the longest; ask the user first.
4. **New model family**: needs a repository in the stack. Ask the user.

## Workflow

Run from the repository root. Stack outputs:

```bash
out() { aws cloudformation describe-stacks --stack-name arcasleep-train --region us-east-1 \
  --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
```

### 1. Find existing images

```bash
for r in data unet bidoze prep deps; do echo "== $r"
  aws ecr describe-images --repository-name arcasleep-train/$r --region us-east-1 \
    --query 'sort_by(imageDetails,&imagePushedAt)[-5:].[imageTags[0],imageDigest,imagePushedAt]' --output text
done
```

A data tag is `<cache_prefix>-<manifest digest[:12]>`; a training tag is
`<data tag>-<commit>`. Pin images as `<repo>:<tag>@<digest>`. If the training
image for your data tag and commit is already there, skip to step 4.

### 2. A new feature cache (only for case 2)

Raw datasets are directories under `s3://arcascope-sleep-models-data/data/`
(`aws s3 ls` it to see which exist). Cached subjects are skipped, so a prep
task only processes new ones:

```bash
deploy/scripts/remote_build.sh prep        # prints the prep image
deploy/scripts/status.sh
deploy/scripts/run_task.sh --max-hours 6 prep <prep image> <family>/config/inputs.json <dataset> [<dataset> ...]
```

When the task has stopped with exit code 0, check its log: each dataset says
whether it was cached under pseudonyms or source ids. Then compare the
cache's subject ids with the split plan before building a data image; a plan
id with no cache file means the cache was built from a different copy of the
data:

```bash
PLAN=<family>/config/splits/<plan>.json
COHORT=$(jq -r .cohort "$PLAN")
aws s3 ls s3://arcascope-sleep-models-data/feature_cache/<cache_prefix>/ | awk '{print $4}' \
  | sed -nE "s/^${COHORT}_subject_(.*)_features\.npz$/\1/p" | sort > /tmp/cache-ids
jq -r '.folds[][]' "$PLAN" | sort -u > /tmp/plan-ids
comm -13 /tmp/cache-ids /tmp/plan-ids | head    # plan ids missing from the cache
```

If ids are missing, stop and tell the user; do not build the data image.

Then:

```bash
deploy/scripts/remote_build.sh data <cache_prefix>   # prints the data image
```

### 3. The training image

```bash
deploy/scripts/remote_build.sh train <family> <data image>   # prints the training image
```

`<data image>` must be a full ECR URI, including the registry hostname:

```text
<aws-account-id>.dkr.ecr.<region>.amazonaws.com/arcasleep-train/data:<tag>@sha256:<digest>
```

A short name such as `arcasleep-train/data:<tag>` resolves to Docker Hub
inside the build and fails at `COPY --from=data`. Use the full pinned URI
printed by the data build.

For a run split across clouds, an existing training image can be copied
whole with `copy_image.sh`, preserving one digest across AWS and GCP.
See the `copy-images` skill and its federation setup.

`--no-wait` returns once the build has started; follow it with
`aws logs tail /aws/codebuild/arcasleep-train --follow --region us-east-1`.

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

ECR keeps the 5 most recent deps and prep images, the 10 most recent data
images and the 20 most recent training images per family; older ones expire
on their own. Do not delete images by
hand: another agent's run may be pinned to them.
