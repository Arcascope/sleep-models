"""Copy a directory of checkpoints, local or ``s3://bucket/prefix``, into a local directory.

Usage: python fetch.py <source> <destination>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main(source: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if not source.startswith("s3://"):
        for path in Path(source).glob("*.pisces"):
            shutil.copy(path, destination / path.name)
        return
    import boto3

    bucket, _, prefix = source[len("s3://"):].partition("/")
    s3 = boto3.client("s3")
    pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/")
    for page in pages:
        for item in page.get("Contents", []):
            if item["Key"].endswith(".pisces"):
                s3.download_file(bucket, item["Key"], str(destination / Path(item["Key"]).name))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
