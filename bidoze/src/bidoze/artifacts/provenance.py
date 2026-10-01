"""Deterministic dataset manifests and run-provenance primitives.

The manifest hash identifies an ordered set of byte-addressed inputs.  S3 ETags
are deliberately excluded: multipart ETags are not portable content hashes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


def canonical_json(value: object) -> str:
    """Canonical JSON used for manifest lines and their aggregate digest."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size_bytes: int
    sha256: str
    object_version_id: str | None = None

    def canonical_line(self) -> str:
        return canonical_json(asdict(self))


def manifest_payload(entries: Iterable[ManifestEntry]) -> bytes:
    lines = [entry.canonical_line() for entry in sorted(entries, key=lambda item: item.path)]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def manifest_digest(entries: Iterable[ManifestEntry]) -> str:
    return sha256_bytes(manifest_payload(entries))


def write_manifest(entries: Iterable[ManifestEntry], output: Path) -> str:
    """Write deterministic UTF-8 JSONL and return its SHA-256 digest."""
    payload = manifest_payload(entries)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return sha256_bytes(payload)


def build_local_manifest(root: Path) -> list[ManifestEntry]:
    """Hash every regular file beneath ``root`` using normalized relative paths."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    entries: list[ManifestEntry] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        entries.append(
            ManifestEntry(path=relative, size_bytes=path.stat().st_size, sha256=sha256_file(path))
        )
    return entries


def _s3_checksum_to_hex(value: str | None) -> str | None:
    if not value:
        return None
    return base64.b64decode(value).hex()


def build_s3_manifest(
    bucket: str, prefix: str, *, require_sha256: bool = True, s3_client=None
) -> list[ManifestEntry]:
    """Build an immutable S3 object inventory from version IDs and SHA-256 checksums.

    Objects must have had an S3 SHA-256 checksum supplied when written.  This is
    the only S3 metadata accepted as a byte-content identity; an ETag is never
    substituted.
    """
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")
    prefix = prefix.lstrip("/")
    paginator = s3_client.get_paginator("list_objects_v2")
    entries: list[ManifestEntry] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for object_summary in page.get("Contents", []):
            key = object_summary["Key"]
            if key.endswith("/"):
                continue
            head = s3_client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
            checksum = _s3_checksum_to_hex(head.get("ChecksumSHA256"))
            if require_sha256 and checksum is None:
                raise ValueError(
                    f"s3://{bucket}/{key} has no S3 SHA-256 checksum; re-upload it with "
                    "ChecksumAlgorithm=SHA256 or create a verified local manifest."
                )
            if checksum is None:
                continue
            rel_key = key[len(prefix) :].lstrip("/") if prefix else key
            entries.append(
                ManifestEntry(
                    path=rel_key,
                    size_bytes=int(head["ContentLength"]),
                    sha256=checksum,
                    object_version_id=head.get("VersionId"),
                )
            )
    return entries


DEFAULT_MANIFEST_PATH = Path("build/manifest/manifest.jsonl")


def run_fingerprint(*, data_digest: str, code_digest: str, length: int = 12) -> str:
    """Short, stable identifier for the pair (input bytes, code that ran).

    Two runs share a fingerprint exactly when they saw the same feature cache
    and ran the same code, which is the property that makes it worth recording
    on every row of a results CSV: rows from reproducible runs collide, and
    rows that differ in either half do not.
    """
    payload = canonical_json({"code": code_digest, "data": data_digest})
    return sha256_bytes(payload.encode("utf-8"))[:length]


def data_digest(manifest_path: Path | str | None = None) -> str | None:
    """SHA-256 of the feature manifest, or None when it is not present.

    The manifest, not the image tag: a tag is a claim about the bytes, the
    manifest is a checkable record of them.
    """
    path = Path(manifest_path or os.environ.get("BIDOZE_MANIFEST_PATH") or DEFAULT_MANIFEST_PATH)
    return sha256_file(path) if path.is_file() else None


def code_digest() -> str | None:
    """Identify the running code: ``GIT_COMMIT`` if baked in, else ask git.

    Images carry GIT_COMMIT because they have neither git nor a .git dir. A
    working tree with uncommitted changes gets a ``-dirty`` suffix, so the
    digest never claims a commit whose tree is not what actually ran.
    """
    baked = os.environ.get("GIT_COMMIT")
    if baked:
        return baked
    import subprocess

    def _git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = _git("rev-parse", "--short", "HEAD")
    if commit is None:
        return None
    return f"{commit}-dirty" if _git("status", "--porcelain") else commit


def run_provenance(
    *,
    package_version: str,
    raw_manifest_sha256: str,
    split_manifest_sha256: str,
    derived_manifest_sha256: str | None,
    processing_config_sha256: str,
    training_config_sha256: str,
    model_config_sha256: str,
    seed: int,
    git_commit: str | None = None,
    image_digest: str | None = None,
) -> dict[str, object]:
    """Structured provenance shared by training, evaluation, and publishing."""
    return {
        "schema_version": 1,
        "package_version": package_version,
        "git_commit": git_commit or os.environ.get("GIT_COMMIT"),
        "container_image_digest": image_digest or os.environ.get("CONTAINER_IMAGE_DIGEST"),
        "raw_dataset_manifest_sha256": raw_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "derived_feature_manifest_sha256": derived_manifest_sha256,
        "processing_config_sha256": processing_config_sha256,
        "training_config_sha256": training_config_sha256,
        "model_config_sha256": model_config_sha256,
        "seed": int(seed),
    }
