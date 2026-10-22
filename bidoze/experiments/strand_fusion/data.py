"""The draws of October 16 §8: cached recordings, the fit/eval split and recording predictions."""
from collections import Counter
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from arcasleep.data.batching import pad_pooled, recording_input
from arcasleep.data.bundle import CacheBundle
from arcasleep.data.cache import discover_cached_subjects, participant_id
from arcasleep.training.balanced import split_train_eval
from arcasleep_bidoze.adult import adult_stage_divisor, stage_counts

CACHE = "12Hz_log_PSD_aligned"
TRAIN = ["weaverDraw", "trainSleepAccel", "trainClinical"]
TEST = ["testSleepAccel", "testClinical"]
ADULT_EVAL = ("trainSleepAccel", "trainClinical")
PAD = 3000
RESULTS = Path("/home/eric/Engineering/Work/autopisces/sleep-models/bidoze/results")
SOURCE_RUN = Path("/home/eric/Engineering/Work/autopisces/sleep-models-tridoze/bidoze/results/ensemble-size-2026-10-17")
LABELS_ROOT = RESULTS / "bidoze-adult-devices-0/setup/labels"
G3_RUN = SOURCE_RUN.parent / "tridoze-2026-10-16"


def draw_data(source_run: Path, draw: int):
    """``(found, fit, validation, tests)``; participants never cross the three groups."""
    found = discover_cached_subjects(TRAIN + TEST, source_run / "setup/cache" / f"draw{draw}" / CACHE)
    refs = sorted(s.ref for ds in TRAIN for s in found[ds])
    fit, val = split_train_eval(refs, 0.3, 0)
    tests = sorted(s.ref for ds in TEST for s in found[ds])
    groups = [{participant_id(r.subject_id) for r in rs} for rs in (fit, val, tests)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]), "participant leakage"
    return found, fit, val, tests


def split_record(fit, val, tests):
    split = {name: [dict(dataset=r.dataset_name, subject_id=r.subject_id) for r in rs]
             for name, rs in [("fit", fit), ("validation", val), ("test", tests)]}
    split["counts"] = {name: dict(Counter(r.dataset_name for r in rs))
                       for name, rs in [("fit", fit), ("validation", val), ("test", tests)]}
    return split


def recording_batch(path: Path):
    """``(x (1, PAD, F + 1, C), labels (n,))`` of one cached recording."""
    bundle = CacheBundle.from_npz(Path(path))
    n = bundle.y.shape[1]
    if n > PAD:
        raise ValueError("recording longer than the 3000-epoch window")
    inputs = recording_input(bundle.X[0], n, 1)
    return jnp.asarray(pad_pooled(inputs, PAD)[None]), bundle.y_for_n_classes(4)[0]


def save_recording(out: Path, labels, strand_probabilities, fusion_probabilities=None):
    arrays = dict(labels=np.asarray(labels), strand_probabilities=np.asarray(strand_probabilities, np.float32))
    if fusion_probabilities is not None:
        arrays["fusion_probabilities"] = np.asarray(fusion_probabilities, np.float32)
    np.savez_compressed(out, **arrays)


def stage_divisor(found, labels_root: Path):
    """The draw's adult stage divisor: all nominal training nights against the BIDsleep reference."""
    counts_train = np.array([stage_counts(np.load(s.path)["y"]) for ds in TRAIN for s in found[ds]])
    adult_paths = sorted((labels_root / CACHE).glob("bidsleep_subject_*"))
    assert adult_paths, "BIDsleep reference missing"
    counts_adult = np.array([stage_counts(np.load(p)["y"]) for p in adult_paths])
    return adult_stage_divisor(counts_train, counts_adult, 0.999)
