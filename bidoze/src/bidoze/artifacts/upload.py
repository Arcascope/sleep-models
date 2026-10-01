"""Ship a finished run's artifacts to S3.

A run that leaves its outputs inside a container has not finished. This
uploads the run directory and, beside it, a manifest of what was uploaded --
the same canonical JSONL used for the feature cache, so a run's outputs are
content-addressed exactly like its inputs.

boto3 is imported lazily and is an optional dependency (``bidoze[aws]``), so
importing this module never forces it on a training image that does not upload.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from bidoze.artifacts.provenance import (
    ManifestEntry,
    build_local_manifest,
    manifest_payload,
    sha256_bytes,
)

MANIFEST_NAME = "run-manifest.jsonl"


@dataclass(frozen=True)
class UploadTarget:
    """Where a run's artifacts go: ``s3://<bucket>/<prefix>/<run>/``."""

    bucket: str
    prefix: str

    @classmethod
    def from_uri(cls, uri: str) -> "UploadTarget":
        if not uri.startswith("s3://"):
            raise ValueError(f"expected an s3:// URI, got {uri!r}")
        rest = uri[len("s3://") :].strip("/")
        bucket, _, prefix = rest.partition("/")
        if not bucket:
            raise ValueError(f"no bucket in {uri!r}")
        return cls(bucket=bucket, prefix=prefix)

    def key_for(self, relative: str) -> str:
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def uri_for(self, relative: str) -> str:
        return f"s3://{self.bucket}/{self.key_for(relative)}"


def _existing_keys(client, target: UploadTarget, limit: int = 1) -> List[str]:
    response = client.list_objects_v2(
        Bucket=target.bucket, Prefix=f"{target.prefix}/", MaxKeys=limit
    )
    return [item["Key"] for item in response.get("Contents", [])]


def upload_run(
    run_dir: Path | str,
    destination: str,
    *,
    client=None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> List[str]:
    """Upload ``run_dir`` under ``destination``; return the URIs written.

    Refuses a destination that already holds objects unless ``overwrite``.
    The bucket is not versioned, so a second run reusing a run id would
    otherwise replace results in place with nothing to recover them from --
    and the fingerprint that names the destination is deliberately stable
    across reruns of the same data and code, which makes the collision likely
    rather than exotic.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    entries: List[ManifestEntry] = build_local_manifest(run_dir)
    if not entries:
        raise FileNotFoundError(f"run directory holds no files: {run_dir}")
    target = UploadTarget.from_uri(destination)

    if client is None and not dry_run:
        import boto3

        client = boto3.client("s3")

    if not dry_run and not overwrite:
        clash = _existing_keys(client, target)
        if clash:
            raise FileExistsError(
                f"{destination} already holds objects (e.g. {clash[0]}); pass "
                "--overwrite to replace them, or use a different run id"
            )

    written: List[str] = []
    for entry in entries:
        if not dry_run:
            client.upload_file(str(run_dir / entry.path), target.bucket, target.key_for(entry.path))
        written.append(target.uri_for(entry.path))

    # Last, so its presence means the rest arrived.
    payload = manifest_payload(entries)
    if not dry_run:
        client.put_object(
            Bucket=target.bucket,
            Key=target.key_for(MANIFEST_NAME),
            Body=payload,
            ContentType="application/x-ndjson",
        )
    written.append(target.uri_for(MANIFEST_NAME))
    return written


def manifest_digest_for(run_dir: Path | str) -> str:
    """Digest of the manifest this run would upload."""
    return sha256_bytes(manifest_payload(build_local_manifest(Path(run_dir))))
