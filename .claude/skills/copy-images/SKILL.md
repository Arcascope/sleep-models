---
name: copy-images
description: Copy a container image between AWS ECR and GCP Artifact Registry (either direction) for arcasleep training, e.g. to run on GCP a data or training image built on AWS. Use when an image exists in one cloud's registry and a run needs it in the other's, or when asked to "copy", "mirror" or "move" an image between ECR and AR.
---

# Copying images between ECR and Artifact Registry

```bash
deploy/scripts/copy_image.sh --dry-run <source> <destination>
deploy/scripts/copy_image.sh <source> <destination>
```

- `<source>` is pinned by digest (`<repo>@sha256:...`); `<destination>` is `<repo>:<tag>`, normally
  the source's own tag. One is ECR (`<aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/arcasleep-train/<repo>`),
  the other Artifact Registry (`<location>-docker.pkg.dev/<gcp-project>/<ar-repository>/<image>`).
  Resolve `<aws-account-id>` from the active AWS credentials with
  `aws sts get-caller-identity --query Account --output text`.
- The script starts an `e2-standard-4` VM in `us-east4-a` that runs `skopeo copy
  --preserve-digests`, waits, checks that the destination digest equals the source's, and
  deletes the VM. The bytes move cloud to cloud; nothing passes through this machine.
- A destination tag that already exists is not copied; the script only checks its digest.
- Run it in the background and wait for it to exit; it prints the pinned destination
  (`<repo>:<tag>@sha256:...`) last.
- Expect about 5 minutes for a data image: the 14.5 GB BiDoze data image copied from
  ECR to Artifact Registry in 198 s, plus about 1 minute to boot and delete the VM.

## Why not the obvious ways

- **Not `docker pull` + `docker push` on the workstation.** A 13 GB data image downloads and
  uploads through the office link: two slow copies instead of one fast one.
- **Not a GCP data build from a GCS copy of the cache.** `remote_build_gcp.sh data` runs
  `arcasleep manifest-local` in a build step where arcasleep is not installed, and fails with
  exit 127. Copying the ECR data image is also the only way to get the same bytes, and the
  same tag, on both clouds.
- `--preserve-digests` keeps the manifest as it is, so the image has one digest in both
  registries and a run on either cloud can be pinned to it.

## Rules

- Copy what a run needs, pinned by digest; do not mirror whole repositories.
- Copying AR → ECR pays GCP egress; ECR → AR pays AWS egress. Both are small next to GPU time.
- Follow `deploy/gcp/image-copy-auth.md` for one-time federation setup. Set
  `ARCASLEEP_PROJECT`, `COPY_SERVICE_ACCOUNT` and `COPY_AWS_ROLE_ARN` explicitly
  for the copy direction. Use the dedicated ECR-pull / AR-writer identity for
  ECR to AR, and the separate ECR-push / AR-reader identity for AR to ECR.
- The VM obtains temporary AWS credentials using its Google identity; no
  password is supplied in metadata. Never fall back to the default service
  account or forward workstation credentials when federation fails.
- The VM is deleted when the copy ends; GCE deletes it after 2 h regardless.
  After a failure, check only this project's `copy-image-*` VMs:
  `gcloud compute instances list --filter='labels.purpose=image-copy' --project="$ARCASLEEP_PROJECT"`.
  Delete only VMs you started.
- A training image is per cloud (it is built on the data image in that cloud's registry), so a
  run on both clouds usually copies the data image and builds the training image on each side
  from the same commit (`remote_build.sh train`, `remote_build_gcp.sh train`).
