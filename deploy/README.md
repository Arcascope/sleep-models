# Training on ECS/EC2

One CloudFormation stack, `arcasleep-train`, holds everything a training run
needs for every model family: a VPC, an ECS cluster, GPU capacity, roles,
image repositories, a CodeBuild project that builds the images, and log
groups. Nothing is shared with anything
else in the account, and `delete-stack` removes all of it.

## Files

| file | what it is |
|---|---|
| `aws/infra.yaml` | CloudFormation: the whole environment |
| `aws/task-definition.json`, `aws/prep-task-definition.json` | task definition templates; `run_task.sh` fills them from stack outputs |
| `aws/buildspec.yml` | the CodeBuild spec; it runs `scripts/codebuild.sh` |
| `docker/deps.Dockerfile` | dependency image: Python, pisces-lite, CUDA jax |
| `docker/prep.Dockerfile` | feature-cache build image |
| `docker/data.Dockerfile` | data image: one feature-cache prefix, data only |
| `scripts/remote_build.sh` | build and push any image in CodeBuild |
| `scripts/build_*_image.sh` | build (and push) one kind of image on this machine |
| `scripts/run_task.sh` | launch training (one task per phase) or a cache build |
| `scripts/status.sh` | free GPU slots and the tasks holding the others |

Each family's training image is built from `<family>/docker/train.Dockerfile`;
its entrypoint is `<family>/scripts/harness.sh`.

## Images

| image | repository | holds | rebuilt when |
|---|---|---|---|
| deps | `arcasleep-train/deps` | Python, pisces-lite, CUDA jax | the jax pin (`scripts/deps_tag.sh`), the pisces-lite pin (`pyproject.toml`) or `deps.Dockerfile` changes |
| prep | `arcasleep-train/prep` | deps + pisces-lite's NUFFT + AWS CLI | the code changes and a cache must be built |
| data | `arcasleep-train/data` | one feature-cache prefix | the cache changes |
| training | `arcasleep-train/<family>` | deps + code + configs + data | the code, a config or the data changes |

Tags name their inputs, and are immutable:

- **deps:** `jax<version>-pisces<version>-<Dockerfile digest>`.
- **prep:** the commit.
- **data:** `<cache prefix>-<manifest digest>`.
- **training:** `<data tag>-<commit>`.

The same inputs always give the same tag, so an image whose tag is already in
ECR is never rebuilt.

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
- `ResultsBucket` (default `arcascope-sleep-models-data`) and its prefixes:
  `ResultsPrefix` (`training_outputs`), where training tasks write;
  `DataPrefix` (`data`), the raw datasets prep tasks read;
  `FeatureCachePrefix` (`feature_cache`), where prep tasks write caches;
  `BuildSourcePrefix` (`codebuild/source`), the source archives builds read.
  Each role can reach only the prefixes it needs.

The autoscaling group starts at **0 instances**. ECS starts one instance per
task placed, up to `MaxInstances`, and returns to zero after the tasks end;
no GPU runs while idle. Each task therefore waits a few minutes for its
instance to boot.

Adding a model family means adding its repository, output and build
variable to `infra.yaml`, its case to `run_task.sh`, and redeploying.

## 2. Build images

`remote_build.sh` builds in CodeBuild (8 vCPU, about $0.02 per minute; at most
two builds at once, each stopped after 60 minutes). It builds `HEAD` as
committed, so every image names a commit; uncommitted changes are not
included. It waits for the build and prints the pushed image, pinned by
digest.

    deploy/scripts/remote_build.sh deps                      # once per version bump
    deploy/scripts/remote_build.sh prep                      # the cache builder
    deploy/scripts/remote_build.sh data 12Hz_log_PSD_aligned # from the cache in S3
    deploy/scripts/remote_build.sh train unet <data image>

A train or prep build builds the deps image first if it is not in ECR yet.

### Feature caches

A cache is built on a GPU task from the raw datasets under
`s3://<bucket>/data/`, one directory each, with a `data_set.json` or
`pisces_lite_adapter.py` at its root:

    deploy/scripts/run_task.sh --max-hours 6 prep <prep image> unet/config/inputs.json weaver SleepAccel bidsleep

The task syncs the cache already at `s3://<bucket>/feature_cache/<cache
prefix>/` first, so only subjects not yet cached are processed. It then syncs
the result back; nothing in S3 is deleted. `remote_build.sh data <cache
prefix>` then packages the whole prefix as a data image.

### Building on this machine

The `build_*_image.sh` scripts build the same images locally; see each
script's header. A data image built from a local cache is pushed with
`build_data_image.sh <cache dir> --push <data repo>`.

## 3. Run

    deploy/scripts/status.sh
    deploy/scripts/run_task.sh unet <image> fold0 fold1 fold2 fold3 fold4 final

Each named phase runs as its own task on its own GPU, all under one run id.
The recipe and split plans are copied into the image, so a changed config
needs a new commit and a new training image.
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

## Google Cloud smoke run

The GCP scripts use project `shiftapp-arcascope`, Artifact Registry repository
`us-central1-docker.pkg.dev/shiftapp-arcascope/arcasleep-train`, and Cloud Build.
Like CodeBuild, the build uploads committed `HEAD`; commit changes first.

    deploy/scripts/remote_build_gcp.sh train unet <data image>
    python3 deploy/scripts/run_spot_smoke_gcp.py <training image>

The spot launcher resolves the image to a digest, discovers GPU-enabled zones,
and tries them sequentially (US, then Europe, then other regions). Restrict
that list with `--zones ZONE [ZONE ...]`. It provisions one `g2-standard-8` (L4, 32 GiB host
memory), runs `fold0` with `config/training-smoke.json` (two epochs), uploads
results to GCS, downloads them into `unet/results/<run id>/`, and deletes the
VM. A one-hour VM lifetime with automatic deletion provides a backstop if
the launcher is interrupted. Spot preemption can interrupt training or upload.

The local run directory also holds `gcp-run.json` (image digest, zone, exit
codes, cleanup status) and `serial.log`. COS installs the GPU driver on boot;
Docker credentials and results use writable directories under `/var/lib`.
Use `--provisioning-model STANDARD` for on-demand capacity. The default
`g2-standard-8` has 32 GiB host RAM; `--machine-type g2-standard-4` selects
16 GiB, and `--machine-type a2-highgpu-1g` selects a single A100 with 40 GB
GPU memory. Each type requires existing quota and regional capacity.

### On-demand Cloud Batch alternative

    bash deploy/scripts/run_smoke_gcp.sh <training image> us-central1

This submitter runs the same two-epoch phase on one on-demand L4. The task
has a one-hour limit and no retries; queued jobs can wait for GPU capacity
indefinitely. Delete a queued job if it is no longer needed.

Results are written to a shared directory on the VM. A separate container
uploads that directory to
`gs://arcascope-sleep-models-data/training_outputs/<run id>/`, including partial
outputs after a training failure. Training has a 55 minute runnable limit to
leave time for upload; reaching the task's overall limit can prevent upload.
The Batch service account needs Artifact Registry read and bucket write access.

    gcloud batch jobs describe <run id> --location=us-central1 --project=shiftapp-arcascope
    gcloud logging read 'labels.job_uid="<job uid>"' --project=shiftapp-arcascope
    gcloud storage rsync --recursive gs://arcascope-sleep-models-data/training_outputs/<run id>/ unet/results/<run id>/

The training harness's S3 upload is disabled for these smoke jobs.
