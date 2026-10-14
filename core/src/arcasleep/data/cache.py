"""Feature-cache discovery and loading.

Training reads a prebuilt NUFFT feature cache laid out under
``<cache_root>/<cache_prefix>/``, one file per subject named ``<dataset>``,
:data:`SUBJECT_SEPARATOR`, ``<subject_id>``, :data:`CACHE_SUFFIX`.

``cache_prefix`` names the feature surface the cache was built for, so one
root can hold several. Each NPZ is a
:class:`pisces_lite.model_io.ModelIOBundle` with ``X`` of shape ``(1, T, F)``
and ``y`` of shape ``(1, T // frames_per_epoch)``.

Reading the cache is all training does, so a training host needs neither senpy
nor a NUFFT backend. Building the cache is a separate concern.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from pisces_lite.cv import SubjectRef
from pisces_lite.model_io import ModelIOBundle

from arcasleep.data.bundle import CacheBundle


CACHE_SUFFIX = "_features.npz"
SUBJECT_SEPARATOR = "_subject_"
# A subject id is ``<participant>`` or ``<participant>_<recording>``, e.g.
# ``adult01_n1`` or ``child01_apple_n1``: one participant's nights and devices.
PARTICIPANT_SEPARATOR = "_"


def participant_id(subject_id: str) -> str:
    """The participant a subject id's recording belongs to."""
    return subject_id.split(PARTICIPANT_SEPARATOR, 1)[0]


@dataclass(frozen=True, order=True)
class CachedSubject:
    """One cached subject and the NPZ it was discovered in."""

    dataset: str
    subject_id: str
    path: Path

    @property
    def ref(self) -> SubjectRef:
        return SubjectRef(self.dataset, self.subject_id)


def resolve_feature_cache_dir(cache_root: Path | str, cache_prefix: str) -> Path:
    """``<cache_root>/<cache_prefix>`` — the directory the NPZs land in."""
    return Path(cache_root) / cache_prefix


def discover_cached_subjects(
    dataset_names: Sequence[str],
    cache_dir: Path | str,
) -> Dict[str, List[CachedSubject]]:
    """Group the cache's NPZ files by dataset name.

    Filenames are the dataset name, :data:`SUBJECT_SEPARATOR`, the subject ID and
    :data:`CACHE_SUFFIX`. The *first* separator splits the dataset from the
    subject ID, so subject IDs that themselves contain the separator round-trip
    correctly.

    A requested dataset with no cached subjects raises: training on a silent
    subset of the configured cohorts is never what the caller wanted.
    """
    cache_dir = Path(cache_dir)
    found: Dict[str, List[CachedSubject]] = {name: [] for name in dataset_names}

    for npz in sorted(cache_dir.glob(f"*{SUBJECT_SEPARATOR}*{CACHE_SUFFIX}")):
        core = npz.name[: -len(CACHE_SUFFIX)]
        index = core.find(SUBJECT_SEPARATOR)
        if index == -1:
            continue
        dataset = core[:index]
        if dataset in found:
            found[dataset].append(
                CachedSubject(dataset, core[index + len(SUBJECT_SEPARATOR) :], npz)
            )

    missing = [name for name in dataset_names if not found[name]]
    if missing:
        raise FileNotFoundError(f"no cached subjects for {missing} under {cache_dir}")
    return found


def load_cached_features(
    subjects: Iterable[CachedSubject],
) -> Dict[SubjectRef, ModelIOBundle]:
    """Load every cached subject into the ``{ref: bundle}`` map training uses.

    Each is a :class:`~arcasleep.data.bundle.CacheBundle`, so a held-out
    subject's ``anglez_5s`` reaches its saved prediction.
    """
    return {subject.ref: CacheBundle.from_npz(subject.path) for subject in subjects}


def load_bundles(paths: Iterable[Path | str]) -> Dict[SubjectRef, ModelIOBundle]:
    """Load an explicit list of cache NPZs, keyed by what they say they are.

    Where :func:`discover_cached_subjects` scans a cache directory for named
    datasets, this takes the files themselves -- an evaluation cohort is
    whatever set of recordings the caller points at, discovered by glob rather
    than by dataset name.

    The cohort and subject names come from inside each NPZ, not from its
    filename: the file is named for the directory the processing run read,
    which need not be the cohort it holds. Sorted by ref so write order does
    not depend on how the shell expanded a glob.
    """
    found: Dict[SubjectRef, ModelIOBundle] = {}
    sources: Dict[SubjectRef, Path] = {}
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"no such feature cache file: {path}")
        bundle = CacheBundle.from_npz(path)
        ref = SubjectRef(str(bundle.datasets[0]), str(bundle.subject_ids[0]))
        if ref in found:
            raise ValueError(
                f"{path} and {sources[ref]} both hold "
                f"{ref.dataset_name}/{ref.subject_id}; scoring one recording twice "
                "would weight it twice"
            )
        found[ref] = bundle
        sources[ref] = path
    if not found:
        raise ValueError("load_bundles: no paths given")
    return {ref: found[ref] for ref in sorted(found)}
