"""Tree inputs computed only from Garmin enhanced actigraphy, as a CIRQA band logs them.

The CIRQA logs through the Garmin Health Standard SDK, which cannot run our code and cannot sync the
raw 25 Hz stream fast enough. What it can log is up to three enhanced actigraphy instances
(``ActigraphyConfiguration``). Each runs on the 25 Hz milli-g x/y/z stream with at most two
second-order IIR sections, the same on every axis:

* FMPre (``filterBeforeMagnitude = true``): ``a' = |(f(x), f(y), f(z))|``, never negative;
* FMPost (``filterBeforeMagnitude = false``): ``a' = f(|(x, y, z)|)``, signed.

Every processing interval reports the zero crossings of ``a' - zc``, the total energy (sum of
``|a'|`` over samples at or above the energy threshold, as an integer) and the time above the
threshold (samples with ``a' >= tat``, in seconds).

FMPre sees a rotation of the wrist, because each axis's share of gravity changes, and linear
acceleration. FMPost sees only the change in the length of the acceleration vector; with gravity
dominant that is the linear acceleration along gravity, and a rotation leaves it unchanged.

The CIRQA tree uses:

* a jerk instance: FMPre with ``1 - z^-1``, 1-s intervals, energy threshold 0. Its energy is the
  sum of ``||a[n] - a[n-1]||`` in milli-g, so it gives ``jerk`` and ``jerk_peak`` (and through
  them ``act``, ``tsb_min``, ``tnb_min``);
* one or two band-pass instances (5-s intervals) in place of ``lfm`` and the angle features, which
  need the raw stream.

Simulation choices: the input is rounded to whole milli-g; each filter starts at the steady state of
the first sample (the watch's filter runs continuously); float64 arithmetic; intervals aligned to
epoch starts.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from arcasleep_tree import features

FS = features.FS
BAND_INTERVAL_S = 5
JERK_INTERVAL_S = 1
#: Threshold of time above threshold and of zero crossings of the band instances, in milli-g.
THRESHOLD_MG = 10.0
#: Added to the FMPre energy before dividing, in milli-g.
RATIO_FLOOR_MG = 0.5
JERK_SOS = np.array([[1.0, -1.0, 0.0, 1.0, 0.0, 0.0]])  # H(z) = 1 - z^-1

BANDS = {
    "bp002": (0.02, 0.2),
    "bp005": (0.05, 0.5),
    "bp025": (0.25, 2.5),
}

#: Per-epoch signals of one band instance and their integer units.
INSTANCE_SCALE = {"e": 16, "emax": 16, "tat": 64, "zc": 1, "still": 1024}
RATIO_SCALE = 1024
INSTANCE_WINDOWS = {
    "e": {
        "b_spans": {"mean": [5, 15, 30, 60, 120], "max": [15, 60]},
        "f_spans": {"mean": [5, 15, 45], "max": [15, 45]},
    },
    "tat": {"b_spans": {"mean": [15, 60]}, "f_spans": {"mean": [15, 45]}},
    "zc": {"b_spans": {"mean": [15, 60]}, "f_spans": {"mean": [15]}},
    "still": {"b_spans": {"mean": [5, 15, 60]}, "f_spans": {"mean": [15]}},
}  # fmt: skip
RATIO_WINDOWS = {"b_spans": {"mean": [5, 15, 60]}, "f_spans": {"mean": [15, 45]}}
#: Inputs of the tree that need the raw stream.
RAW_ONLY_AT_TIME = {"lfm", "angle", "angle_step", "daz", "sib"}
RAW_ONLY_SIGNALS = {"lfm", "angle", "angle_step", "daz", "sib"}


def instance_name(band: str, pre: bool) -> str:
    return f"g{'pre' if pre else 'post'}_{band}"


def ratio_name(band: str) -> str:
    return f"gratio_{band}"


def register() -> None:
    """Adds every band instance's signals to the tree's known signals and integer units.

    Runs on import, so a worker process that imports this module knows them too.
    """
    for band in BANDS:
        for pre in (True, False):
            for sig, scale in INSTANCE_SCALE.items():
                features.SIGNAL_SCALE[f"{instance_name(band, pre)}_{sig}"] = scale
                features.AT_TIME_SCALE[f"{instance_name(band, pre)}_{sig}"] = scale
        features.SIGNAL_SCALE[ratio_name(band)] = RATIO_SCALE
        features.AT_TIME_SCALE[ratio_name(band)] = RATIO_SCALE


def cirqa_inputs(base: dict, instances: list[tuple[str, bool]], ratio: bool) -> dict:
    """``base`` without the inputs that need the raw stream, plus the band ``instances``
    (``(band, pre)`` pairs) and, with ``ratio``, the FMPost/FMPre energy ratio of their band."""
    import copy

    inp = copy.deepcopy(base)
    f = inp["features"]
    f["at_time"] = [n for n in f["at_time"] if n not in RAW_ONLY_AT_TIME]
    for s in RAW_ONLY_SIGNALS:
        f.pop(s, None)
    for band, pre in instances:
        name = instance_name(band, pre)
        f["at_time"] += [f"{name}_{sig}" for sig in INSTANCE_SCALE]
        f.update({f"{name}_{sig}": copy.deepcopy(w) for sig, w in INSTANCE_WINDOWS.items()})
    if ratio:
        bands = sorted({b for b, _ in instances})
        if len(bands) != 1 or len(instances) != 2:
            raise ValueError("the ratio needs one FMPre and one FMPost instance of one band")
        f["at_time"].append(ratio_name(bands[0]))
        f[ratio_name(bands[0])] = copy.deepcopy(RATIO_WINDOWS)
    return inp


#: Band of the released CIRQA trees' FMPre and FMPost instances.
RELEASE_BAND = "bp025"
#: Per-epoch columns of the released CIRQA trees, besides heart rate.
RELEASE_COLUMNS = (
    "jerk",
    f"gpre_{RELEASE_BAND}_e",
    f"gpre_{RELEASE_BAND}_tat",
    f"gpre_{RELEASE_BAND}_zc",
    f"gpre_{RELEASE_BAND}_still",
    f"gpost_{RELEASE_BAND}_e",
    f"gpost_{RELEASE_BAND}_tat",
    f"gratio_{RELEASE_BAND}",
)
#: Value of an inputs dict's ``signals`` key when its per-epoch signals come from :func:`cirqa_signals`.
SIGNALS_KEY = "enhanced_actigraphy"
HR_AT_TIME = {"hr", "hr_std", "hr_n", "hr_missing", "hr_base", "hr_rel"}
HR_SIGNALS = {"hr", "hr_rel", "hr_valid"}


def release_inputs(heart_rate: bool) -> dict:
    """Inputs of the released CIRQA trees: :data:`RELEASE_COLUMNS` with their windows, 120 min
    back and 45 min forward, and with ``heart_rate`` the tree's heart-rate columns."""
    import copy

    base = copy.deepcopy(features.INPUTS)
    base["back_min"] = 120
    base["baseline_blocks"] = 12
    inp = cirqa_inputs(base, [(RELEASE_BAND, True), (RELEASE_BAND, False)], True)
    f = inp["features"]
    keep = set(RELEASE_COLUMNS) | (HR_AT_TIME if heart_rate else set())
    f["at_time"] = [n for n in f["at_time"] if n in keep]
    signals = set(RELEASE_COLUMNS) | (HR_SIGNALS if heart_rate else set())
    for name in [k for k in f if k != "at_time" and k not in signals]:
        f.pop(name)
    inp["signals"] = SIGNALS_KEY
    return inp


def filtered_magnitude(xyz_mg: np.ndarray, sos: np.ndarray, pre: bool) -> np.ndarray:
    """``a'`` of an instance on ``(n, 3)`` milli-g samples, filter at the first sample's steady state."""
    x = np.asarray(xyz_mg, np.float64)
    zi = signal.sosfilt_zi(sos)
    if pre:
        y, _ = signal.sosfilt(sos, x, axis=0, zi=zi[:, :, None] * x[0][None, None, :])
        return np.sqrt((y * y).sum(1))
    m = np.sqrt((x * x).sum(1))
    return signal.sosfilt(sos, m, zi=zi * m[0])[0]


def interval_metrics(a: np.ndarray, interval_s: int, threshold: float = THRESHOLD_MG):
    """Per interval: zero crossings of ``a - threshold``, total energy (energy threshold 0, as an
    integer) and time above ``threshold`` in seconds."""
    per = interval_s * FS
    n = a.size // per
    a = a[: n * per]
    above = a - threshold >= 0
    flips = np.r_[False, above[1:] != above[:-1]]
    return (
        flips.reshape(n, per).sum(1).astype(float),
        np.floor(np.abs(a).reshape(n, per).sum(1)),
        above.reshape(n, per).sum(1) / FS,
    )


def cirqa_signals(acc25_g: np.ndarray, valid25: np.ndarray) -> dict[str, np.ndarray]:
    """Per-epoch signals of the jerk instance and of every band instance (FMPre and FMPost of every
    band) and ratio, for ``features.compute_features(signals=...)``. NaN where too few samples are
    valid."""
    n_epochs = acc25_g.shape[0] // features.SAMPLES_PER_EPOCH
    n = n_epochs * features.SAMPLES_PER_EPOCH
    present = valid25[:n].reshape(n_epochs, -1).mean(axis=1) >= features.MIN_VALID_FRACTION
    xyz = np.rint(np.asarray(acc25_g[:n]) * 1000.0)

    def masked(v):
        return np.where(present, v, np.nan)

    _, jerk_e, _ = interval_metrics(filtered_magnitude(xyz, JERK_SOS, True), JERK_INTERVAL_S)
    jerk_e = jerk_e.reshape(n_epochs, -1)
    out = {
        "present": present,
        "jerk": masked(jerk_e.sum(1) / features.SAMPLES_PER_EPOCH / 1000.0),
        "jerk_peak": masked(jerk_e.max(1) / FS / 1000.0),
    }
    for band, freq in BANDS.items():
        sos = signal.butter(2, freq, btype="bandpass", fs=FS, output="sos")
        for pre in (True, False):
            zc, energy, tat = (
                v.reshape(n_epochs, -1)
                for v in interval_metrics(filtered_magnitude(xyz, sos, pre), BAND_INTERVAL_S)
            )
            name = instance_name(band, pre)
            out[f"{name}_e"] = masked(energy.sum(1) / features.SAMPLES_PER_EPOCH)
            out[f"{name}_emax"] = masked(energy.max(1) / (BAND_INTERVAL_S * FS))
            out[f"{name}_tat"] = masked(tat.sum(1))
            out[f"{name}_zc"] = masked(zc.sum(1))
            out[f"{name}_still"] = masked((tat == 0).mean(1))
        out[ratio_name(band)] = out[f"{instance_name(band, False)}_e"] / (
            out[f"{instance_name(band, True)}_e"] + RATIO_FLOOR_MG
        )
    return out


register()
