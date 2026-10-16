---
name: gcp-gpu-runs
description: Run arcasleep GPU training (BiDoze, UNet) on Google Cloud L4 GPUs, and choose between GCP and AWS when asked to "kick off a training job on GPUs" or to use one cloud with overflow on the other. Covers building images in Cloud Build, launching one VM per CV phase, monitoring, collecting results from GCS, and falling back to AWS on stockout. Use when asked to run, queue, check or collect a training run on GCP, or when a run should prefer GCP.
---

# GPU training runs on GCP, with AWS overflow

Project `shiftapp-arcascope`. Images live in Artifact Registry
`us-central1-docker.pkg.dev/shiftapp-arcascope/arcasleep-train/<family>`;
results go to `gs://arcascope-sleep-models-data/training_outputs/<run id>/`.
Scripts are under `deploy/scripts/` and run from the repository root. The
reference is the GCP section of `deploy/README.md`; the AWS counterpart of
everything here is the `aws-gpu-runs` skill, whose image and cache rules also
apply (read its "Images" section once).

`run_task_gcp.py` mirrors `run_task.sh`: `FAMILY IMAGE [PHASE ...]`, with
`--config`, `--run-id`, `--env` and `--max-hours`. Each phase runs on its own
L4 VM under one run id, and the layout of the results matches S3
(`training/<phase>/`).

## Dispatch: GCP first, AWS overflow

1. Launch on GCP (below). Each phase tries zones in turn (US, then Europe,
   then the rest) and is reported as `no-capacity` if none has an L4.
2. **Overflow to AWS only the phases reported `no-capacity`.** Confirm no VM
   was left behind (Cleanup), then follow `aws-gpu-runs` from its step 1
   with the same `--run-id`, naming only those phases, on an ECR image built
   from the same commit (`remote_build.sh`). Its `status.sh` and slot limits
   apply.
3. A phase that failed, timed out or was interrupted is **not** overflowed:
   report it and ask. Rerunning could repeat the same failure on a second bill.
4. The results of an overflowed run are split: collect GCS phases with the
   launcher (it does so itself) and S3 phases with `aws s3 sync` into the same
   `<family>/results/<run id>/`.
5. Say in the report which cloud ran each phase.

An Artifact Registry image is not usable by ECS, and an ECR image is not
usable here. A data image is copied between the registries with
`deploy/scripts/copy_image.sh` (the `copy-images` skill); a training image
is built on each cloud from the same commit.

## Limits

- **Never change quota, create resources outside these scripts, or delete a VM
  you did not launch.** Quota is 16 L4 per region; stockout, not quota, has
  been the constraint.
- **The launcher caps training VMs project-wide at 4 (`--max-vms`).** It
  refuses a launch needing more free slots unless `--queue` is approved, and
  counts other agents' VMs. Leave them slots.
- **Set `--max-hours`** to about 1.5 × the expected duration. A phase that hits
  it still uploads (exit 124, stage `timeout`).
- **Run a wiring check before a sweep:** one phase, `--config
  config/training-smoke.json`, `--max-hours 1`.
- A VM costs money from creation until the launcher deletes it. Group work into
  fewer launches.
- Defaults are STANDARD (on-demand) `g2-standard-8` (L4, 32 GiB RAM). `SPOT`
  is cheaper but can be preempted mid-phase, losing it, and spot L4 has been
  stocked out in most zones. `g2-standard-4` has 16 GiB, enough for the
  smoke recipe only; the full recipe has not been run on it.
- **Do not use Cloud Batch** (`run_smoke_gcp.sh`): it cannot schedule SPOT VMs,
  and queued jobs hold GPU requests indefinitely.
- Commit before building: Cloud Build uses `git archive HEAD`.
- Batch cloud calls; do not ask for approval for each status check.

## Workflow

### 1. Find the image

```bash
gcloud artifacts docker images list \
  us-central1-docker.pkg.dev/shiftapp-arcascope/arcasleep-train/<family> \
  --include-tags --sort-by=~CREATE_TIME --limit=5
```

A training tag is `<data tag>-<commit>`; pin by digest. If the image for your
commit exists, skip to step 3.

### 2. Build (only if no image for this commit)

```bash
deploy/scripts/remote_build_gcp.sh train <family> \
  us-central1-docker.pkg.dev/shiftapp-arcascope/arcasleep-train/data:<data tag>
```

The data image for UNet's current cache is
`data:12Hz_log_PSD_aligned-f9f44233ba28` (per `unet/config/inputs.json`, not
the `jerkdt` cache). A new cache needs a prep build first, which exists only
on AWS (`aws-gpu-runs` step 2); copy the resulting ECR data image here
with the `copy-images` skill. `remote_build_gcp.sh data` fails (its build
step has no `arcasleep`). `deps`
rebuilds only when the jax or pisces-lite pin changes; ask the user first.
Builds run in us-central1 and are unaffected by GPU stockout.

### 3. Launch

```bash
python3 deploy/scripts/run_task_gcp.py --dry-run <family> <image> fold0 fold1
python3 deploy/scripts/run_task_gcp.py --max-hours 6 <family> <image> fold0 fold1 fold2
```

- Phases are `fold0`…`fold4` and `final` for the committed 5-fold plans; with
  none named, one VM runs all of them in order.
- `--env KEY=VALUE` passes harness variables, such as
  `UNET_SPLIT_PLAN=config/splits/<plan>.json` or `BIDOZE_CV_CONFIG=...`.
- `--zones` restricts the search; `--machine-type a2-highgpu-1g` selects an A100
  (never run).
- The command blocks until every VM has finished, collected its results and
  deleted it. **Run it in the background** and wait for the notification.

### 4. Monitor

The launcher prints each phase's recent serial output. Elsewhere:

```bash
gcloud compute instances list --filter='labels.purpose=training' --project=shiftapp-arcascope
cat <family>/results/<run id>/gcp-run.json
```

### 5. Results

Downloaded into `<family>/results/<run id>/`, with `gcp-run.json` (image
digest, per-phase zone, exit codes, `vm_deleted`) and `gcp/<phase>/serial.log`.
A phase is good only if its stage is `succeeded` (training and upload exit 0)
and `vm_deleted` is `true`. Partial outputs of a failed phase are for
diagnosis only. Then pool and compare as for a local run, and record the run id,
image digest and recipe in the research log.

## Cleanup

After every launch, and before overflowing to AWS:

```bash
gcloud compute instances list --project=shiftapp-arcascope
gcloud compute disks list --project=shiftapp-arcascope
```

Both must be empty unless another agent has a run in flight. Delete only your
own (`gcloud compute instances delete <name> --zone=<zone> --quiet`). A VM the
launcher failed to delete is removed by GCE 30 minutes after `--max-hours`.

## Known pitfalls

- COS `/root/.docker` is read-only and `/mnt` is a small tmpfs; the startup
  template uses `/var/lib/arcasleep-docker` and `/var/lib/arcasleep-results`.
  Do not move them.
- Both registries hold `deps:jax0.11.2-pisces4.2.0-e5e67063`, built
  separately, so the two digests differ. `deploy/scripts/deps_tag.sh` prints the tag
  that the current pins require.
- BiDoze is supported by the launcher but has not been run on GCP, and has no
  image in Artifact Registry yet.
