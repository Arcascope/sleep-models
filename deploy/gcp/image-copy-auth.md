# Federated authentication for image copies

The launcher requires an explicit project, service account and AWS role. It
never falls back to the default Compute Engine account or forwards a password
from the launching machine.

Use a different identity for each direction:

| Direction | Google service account | Artifact Registry grant | AWS role |
| --- | --- | --- | --- |
| ECR to Artifact Registry | `arcasleep-copy-ecr-to-ar` | Writer on the one registry repository | ECR pull on the approved repositories |
| Artifact Registry to ECR | `arcasleep-copy-ar-to-ecr` | Reader on the one registry repository | ECR push on the approved repositories |

The Google ID token is exchanged through `AssumeRoleWithWebIdentity`. AWS
checks the service account's numeric subject, authorized party and audience.
The VM requests a 15-minute AWS session and obtains its own ECR password.
ECR documents a 12-hour token lifetime; do not assume the VM's deletion or
the requested STS duration revokes an already issued registry password.

JWTs, AWS credentials and registry passwords live in a mode-0700 directory
under `/run` (tmpfs on COS), with mode-0600 files. The AWS CLI reads the JWT
and AWS credentials from files. Skopeo reads a registry auth file. Secret
values are absent from custom metadata and command arguments. Helper containers
disable Docker logging so credential output is not persisted by the daemon.
Files are removed on success and failure; the VM has a two-hour deletion backstop.

Processes that can use the VM's service account can still exercise its
permissions. Restrict who can attach or impersonate these accounts; run only
the copy workload on these VMs.

## One-time setup

Review the resource names and repository list before running these mutating
commands. They create two new Google accounts, add bindings on the selected
Artifact Registry repository and create a separate AWS stack containing only
two IAM roles. An optional account binding lets a launcher attach the accounts
if it lacks that permission. They do not change training identities.

Set these values locally; do not commit a populated environment file:

```bash
export ARCASLEEP_PROJECT='<gcp-project>'
export AR_LOCATION=us-central1
export AR_REPOSITORY='<artifact-registry-repository>'
export LAUNCHER_MEMBER='user:<your-google-account>'
export AWS_REGION=us-east-1
export COPY_AWS_AUDIENCE=arcasleep-image-copy
ECR_TO_AR=arcasleep-copy-ecr-to-ar@$ARCASLEEP_PROJECT.iam.gserviceaccount.com
AR_TO_ECR=arcasleep-copy-ar-to-ecr@$ARCASLEEP_PROJECT.iam.gserviceaccount.com
```

Create the two accounts and grant repository-scoped access. Neither account
needs Service Account Token Creator or a service account key.

```bash
gcloud iam service-accounts create arcasleep-copy-ecr-to-ar \
  --project="$ARCASLEEP_PROJECT" --display-name="Image copy: ECR to AR"
gcloud iam service-accounts create arcasleep-copy-ar-to-ecr \
  --project="$ARCASLEEP_PROJECT" --display-name="Image copy: AR to ECR"

gcloud artifacts repositories add-iam-policy-binding "$AR_REPOSITORY" \
  --project="$ARCASLEEP_PROJECT" --location="$AR_LOCATION" \
  --member="serviceAccount:$ECR_TO_AR" --role=roles/artifactregistry.writer
gcloud artifacts repositories add-iam-policy-binding "$AR_REPOSITORY" \
  --project="$ARCASLEEP_PROJECT" --location="$AR_LOCATION" \
  --member="serviceAccount:$AR_TO_ECR" --role=roles/artifactregistry.reader

```

If the launcher already has `iam.serviceAccounts.actAs` (for example, through
an existing Owner role), no additional launcher grant is needed. Otherwise,
grant Service Account User on these two accounts only:

```bash
for service_account in "$ECR_TO_AR" "$AR_TO_ECR"; do
  gcloud iam service-accounts add-iam-policy-binding "$service_account" \
    --project="$ARCASLEEP_PROJECT" --member="$LAUNCHER_MEMBER" \
    --role=roles/iam.serviceAccountUser
done
```

Read the numeric IDs after creation. AWS uses the IDs, not email addresses,
so deleting and recreating a Google account does not silently restore trust.

```bash
PULL_SUBJECT=$(gcloud iam service-accounts describe "$ECR_TO_AR" \
  --project="$ARCASLEEP_PROJECT" --format='value(uniqueId)')
PUSH_SUBJECT=$(gcloud iam service-accounts describe "$AR_TO_ECR" \
  --project="$ARCASLEEP_PROJECT" --format='value(uniqueId)')
```

The AWS template is [image-copy-auth.json](../aws/image-copy-auth.json).
Supply an explicit comma-separated list of existing repository ARNs in
`ECR_REPOSITORY_ARNS`; wildcards are rejected. The repositories must be in
the account and region used for the copy. The pull role has only layer and
manifest reads. The push role has layer checks, manifest reads and uploads.
Both can request a regional ECR token (`Resource: "*"` is required for that
one API); neither has deletion, IAM, S3 or repository administration access.

```bash
export ECR_REPOSITORY_ARNS='<comma-separated-approved-repository-arns>'
aws cloudformation validate-template --region "$AWS_REGION" \
  --template-body file://deploy/aws/image-copy-auth.json
aws cloudformation deploy --region "$AWS_REGION" \
  --stack-name arcasleep-image-copy-auth \
  --template-file deploy/aws/image-copy-auth.json --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    GoogleEcrToArSubject="$PULL_SUBJECT" GoogleArToEcrSubject="$PUSH_SUBJECT" \
    Audience="$COPY_AWS_AUDIENCE" EcrRegion="$AWS_REGION" \
    EcrRepositoryArns="$ECR_REPOSITORY_ARNS"
```

Before a real copy, inspect the stack's role trust/permission policies and
the Google repository/account bindings. Check for inherited grants to the
new accounts as well. Validate that the pull identity cannot push, the
identities cannot access unrelated repositories, and a wrong audience or
opposite-direction subject cannot assume the role.

## Running a copy

For ECR to Artifact Registry:

```bash
export COPY_SERVICE_ACCOUNT="$ECR_TO_AR"
export COPY_AWS_ROLE_ARN=$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" --stack-name arcasleep-image-copy-auth \
  --query 'Stacks[0].Outputs[?OutputKey==`EcrPullRoleArn`].OutputValue | [0]' --output text)
deploy/scripts/copy_image.sh --dry-run <source> <destination>
deploy/scripts/copy_image.sh <source> <destination>
```

For Artifact Registry to ECR, use `$AR_TO_ECR` and output `EcrPushRoleArn`.
The caller still needs read-only image lookup permissions and permission to
create/delete the GCP VM. Those launcher permissions are separate from the
VM's restricted registry permissions. Every GCP call specifies the project.

The source tag is pinned before VM creation. A destination with a matching
digest is skipped; a different existing digest is rejected. Unexpected
lookup errors abort before launch. The launcher verifies the destination
digest and deletes its own VM even after failure.

## References

- [AWS: native Google workload identity federation](https://aws.amazon.com/blogs/security/access-aws-using-a-google-cloud-platform-native-workload-identity/)
- [AWS: Google OIDC claim mapping](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_iam-condition-keys.html#condition-keys-wif)
- [AWS: ECR authentication scope and lifetime](https://docs.aws.amazon.com/AmazonECR/latest/userguide/registry_auth.html)
- [AWS: repository-scoped push permissions](https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-push-iam.html)
- [Google: repository-scoped Artifact Registry roles](https://docs.cloud.google.com/artifact-registry/docs/access-control)
