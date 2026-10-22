"""Rebuild the test adults' model inputs under four accelerometer conditions.

Adapted from the October 21 §2 quantization experiment (private branch research/quantization-oct21,
commit 861ac7b, ``bidoze/experiments/quantization/run.py``); the feature rebuild is unchanged: the
cached processing config (senpy NUFFT, 12-Hz grid, 0.1-6 Hz, 10-s windows, 2-s hops), jerk as the norm
of consecutive x/y/z differences without division by the sample interval, 15 log-spectrum frames averaged per
30-s epoch, padded to 3,000 epochs.

Conditions (stored in this order along the first axis of ``inputs``):
  native, quantized_256, stride2, quantized_256_stride2
- quantized: ``rint(256 * axis_g) / 256`` independently on x/y/z (nearest, ties to even, no offset)
- stride2: the aligned raw array ``[::2]`` with its original timestamps; no interpolation or filter

A condition whose last window ends 1-2 frames earlier than the cache's (thinning) is padded with absent
frames; the count is in ``rates.csv`` (``padded_frames``).

Test adults are the union of the draws' ``testSleepAccel`` / ``testClinical`` participants. Writes
``<out>/<cohort>/<subject>.npz`` (inputs, y, pooled, native-vs-cache fidelity) and
``<out>/<cohort>/rates.csv``. Participants with no raw accelerometer file are listed in ``missing.txt``.

Usage: python build_features.py <cohort: SleepAccel|SleepAccel-Clinical> <raw data root> <feature cache root> <out>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from arcasleep.data.constants import SPECTROGRAM_PADDING_VALUE
from arcasleep.data.batching import pad_recording_input, recording_input
from arcasleep.data.prepare import load_dataset, prepare_subject
from arcasleep.processing import processing_config

HERE = Path(__file__).resolve().parent
BIDOZE = HERE.parents[2]
CONDITIONS = ["native", "quantized_256", "stride2", "quantized_256_stride2"]
DRAW_KEY = {"SleepAccel": "testSleepAccel", "SleepAccel-Clinical": "testClinical"}
CACHE_NAME = "12Hz_log_PSD_aligned"
MAX_PADDED_FRAMES = 2


def test_ids(cohort: str) -> list[str]:
    draws = json.loads((HERE.parent / "draws.json").read_text())
    return sorted({sid for d in draws.values() for sid in d[DRAW_KEY[cohort]]})


def quantize(raw: np.ndarray) -> np.ndarray:
    out = raw.copy()
    out[:, 1:] = np.rint(out[:, 1:] * 256) / 256
    assert np.max(np.abs(out[:, 1:] - raw[:, 1:])) <= 1 / 512 + 1e-12
    return out


def main(cohort: str, data: Path, cache_root: Path, out: Path) -> None:
    cfg = processing_config(json.loads((BIDOZE / "config/inputs.json").read_text()), "inputs")
    dataset = load_dataset(data / cohort)
    folder = out / cohort
    folder.mkdir(parents=True, exist_ok=True)
    ids = test_ids(cohort)
    available = {p.name.split("_")[0].removesuffix(".csv") for p in (data / cohort / "cleaned_accelerometer").iterdir()}
    missing = [s for s in ids if s not in available]
    (folder / "missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    rates = []
    for i, sid in enumerate(s for s in ids if s not in missing):
        dest = folder / f"{sid}.npz"
        subject = prepare_subject(dataset, sid)
        aligned = subject.accel_values
        with np.load(cache_root / CACHE_NAME / f"{cohort}_subject_{sid}_features.npz") as z:
            cached, y, freq = z["X"][0], z["y"][0], z["freq"]
        assert np.array_equal(y, subject.labels[: len(y)]), sid
        assert len(y) <= 3000
        pooled, fidelity, padded = [], None, {}
        for condition in CONDITIONS:
            raw = aligned[::2] if "stride2" in condition else aligned
            acc = quantize(raw) if "quantized" in condition else raw
            frames = cfg.apply(acc, normalize=False, padding=False, origin=float(raw[0, 0]))[: len(y) * 15]
            short = cached.shape[0] - frames.shape[0]
            if 0 < short <= MAX_PADDED_FRAMES:
                # Thinning can end the last 10-s window one hop earlier; the frames it would have held are absent.
                pad = np.full((short, *frames.shape[1:]), SPECTROGRAM_PADDING_VALUE, dtype=frames.dtype)
                frames = np.concatenate([frames, pad])
                padded[condition] = short
            assert frames.shape == cached.shape, (sid, frames.shape, cached.shape)
            if condition == "native":
                diff = np.abs(frames - cached)
                fidelity = dict(p99=float(np.percentile(diff, 99)), max=float(diff.max()))
                assert fidelity["p99"] < 0.01, (sid, fidelity)
            pooled.append(recording_input(frames, len(y), 1))
        assert np.array_equal(pooled[0][:, -1], pooled[1][:, -1]), "gap coverage changed"
        inputs = np.stack([pad_recording_input(v, 3000, 1) for v in pooled]).astype(np.float32)
        np.savez_compressed(dest, inputs=inputs, y=y, freq=freq, pooled=np.stack(pooled))
        rates.append(dict(subject_id=sid, native_median_hz=float(1 / np.median(np.diff(aligned[:, 0]))),
                          retained_median_hz=float(1 / np.median(np.diff(aligned[::2, 0]))), fidelity_p99=fidelity["p99"], padded_frames=json.dumps(padded)))
        print("FEATURES", cohort, sid, i + 1, len(ids), fidelity, flush=True)
    pd.DataFrame(rates).to_csv(folder / "rates.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
