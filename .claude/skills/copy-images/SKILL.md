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

## Choosing a copy or a build

Copy either a data image or a complete training image. For a run split
between AWS and GCP, building the training image once and copying it gives
both clouds one manifest digest, including identical dependency and data
layers. Pass each launcher its own registry's URI, pinned to that digest.

`--preserve-digests` keeps the manifest unchanged and the launcher verifies
the destination digest. Independent builds can produce different digests,
even from the same commit. Copying an existing data image also preserves
its exact bytes and tag; the GCP data-build path now installs arcasleep under
Python 3.12 but still needs a complete build verification.

The VM transfers bytes directly between clouds. Pulling and pushing through
the workstation adds both a download and an upload through its connection.

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
- Cleanup retries deletion up to three times and confirms that the exact VM
  launched by this invocation is absent. A failed or unverifiable cleanup
  makes the command fail; GCE's 2 h lifetime remains the backstop.
  After a failure, use the reported VM name and zone to inspect that VM.
  This inventory can help identify it:
  `gcloud compute instances list --filter='labels.purpose=image-copy' --project="$ARCASLEEP_PROJECT"`.
  Delete only VMs you started.
- For one digest across clouds, copy the complete training image. For separate
  builds, copy the data image and build on each cloud from the same commit
  (`remote_build.sh train`, `remote_build_gcp.sh train`); record both digests.
