"""Keyed pseudonyms for subject ids.

A dataset directory holding :data:`PSEUDONYM_KEY_FILENAME` is cached under an
HMAC-SHA256 of each subject id rather than the id itself. The key is kept with
the raw data; a dataset without one keeps its ids.
"""
from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Dict, Iterable

PSEUDONYM_KEY_FILENAME = "arcasleep_pseudonym.key"
#: Hex characters kept from the digest.
PSEUDONYM_LENGTH = 16


def load_pseudonym_key(dataset_dir: Path | str) -> bytes | None:
    """The hex key in ``dataset_dir``, or None when the dataset has none."""
    path = Path(dataset_dir) / PSEUDONYM_KEY_FILENAME
    if not path.is_file():
        return None
    key = bytes.fromhex(path.read_text().strip())
    if len(key) < 16:
        raise ValueError(f"{path}: key is {len(key)} bytes; use at least 16")
    return key


def pseudonym(key: bytes, subject_id: str) -> str:
    return hmac.new(key, subject_id.encode("utf-8"), hashlib.sha256).hexdigest()[
        :PSEUDONYM_LENGTH
    ]


def cache_ids(key: bytes | None, subject_ids: Iterable[str]) -> Dict[str, str]:
    """Map each source id to the id the cache records it under."""
    ids = list(subject_ids)
    if key is None:
        return {sid: sid for sid in ids}
    mapped = {sid: pseudonym(key, sid) for sid in ids}
    if len(set(mapped.values())) != len(mapped):
        raise ValueError("pseudonym collision; lengthen PSEUDONYM_LENGTH")
    return mapped
