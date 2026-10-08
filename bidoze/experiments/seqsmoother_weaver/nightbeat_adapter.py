"""Call unchanged numerical functions from a pinned official Nightbeat checkout.

The upstream files import Polars, Torch, and dataset-specific IO at module import.
Loading their function definitions avoids those unused dependencies; no numerical
function body is modified. No upstream dataset loader or ECG reference is called.
"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import ShortTimeFFT, find_peaks, find_peaks_cwt, medfilt
from scipy.stats import iqr, kurtosis, norm


def load_functions(vendor: Path) -> dict:
    namespace = dict(np=np, signal=signal, ShortTimeFFT=ShortTimeFFT,
                     find_peaks=find_peaks, find_peaks_cwt=find_peaks_cwt,
                     medfilt=medfilt, iqr=iqr, kurtosis=kurtosis, norm=norm)
    for name in ["shared_functionalities", "curve_tracing", "peak_detection", "nightbeat"]:
        path = vendor / "BCGAlgorithms" / (name + ".py")
        tree = ast.parse(path.read_text(), filename=str(path))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def rolling_median(times: np.ndarray, values: np.ndarray, seconds: float = 300) -> np.ndarray:
    out = np.full(len(times), np.nan)
    for i, t in enumerate(times):
        a, b = np.searchsorted(times, [t - seconds / 2, t + seconds / 2], side="left")
        b = np.searchsorted(times, t + seconds / 2, side="right")
        v = values[a:b]
        if np.isfinite(v).any():
            out[i] = np.nanmedian(v)
    return out


def predict(xyz: np.ndarray, valid: np.ndarray, origin: float, vendor: Path,
            progress=None) -> pd.DataFrame:
    """180-s chunks, 60-s stride; 60-s prediction windows, 15-s step, at 100 Hz.

    Upstream timestamps are starts; returned timestamps are endpoints in seconds.
    Reference arrays are empty deliberately: predictions cannot see HR labels.
    """
    fn = load_functions(vendor)
    fs = 100
    main = fn["get_butter_sos"](0.5, 3.5, fs, order=4)
    bcg = fn["get_butter_sos"](5, 14, fs, order=5)
    records = []
    starts = np.arange(np.ceil(origin / 60) * 60, origin + len(xyz) / fs - 180 + 1e-7, 60)
    for index, start in enumerate(starts):
        lo = int(round((start - origin) * fs))
        hi = lo + 180 * fs
        if lo < 0 or hi > len(xyz) or not valid[lo:hi].all():
            continue
        segment = xyz[lo:hi]
        processed = fn["nightbeat_process_window"](
            dict(acc_a_x=segment[:, 0], acc_a_y=segment[:, 1], acc_a_z=segment[:, 2]),
            fs=fs, sos_main=main, sos_bcg=bcg)
        row = {k: [v] for k, v in processed.items()}
        row.update(time=[float(start)], hr_vals=[pd.Series(dtype=float)],
                   hr_times=[pd.Series(dtype=float)])
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            times, anchors, labels, motion, peaks = fn["nightbeat_predict_whole_window"](
                row, fs=fs, win_size=60, overlap=0.75, motion_artifacts=True,
                curve_tracing=True, peak_detection=True)
        for t, anchor, quality, peak in zip(times, anchors, motion, peaks):
            records.append(dict(endpoint=t + 60, nb_anchor=anchor,
                                nb_motion=quality, nb_peak=peak))
        if progress and (index % 20 == 0 or index == len(starts) - 1):
            progress(index + 1, len(starts))
    if not records:
        return pd.DataFrame({name: pd.Series(dtype=float) for name in
                             ("endpoint", "nb_anchor", "nb_motion", "nb_peak", "nightbeat")})
    frame = pd.DataFrame(records).sort_values("endpoint").drop_duplicates("endpoint")
    # Gates in the official visualize_results.ipynb, with timestamps in seconds.
    retained = (frame.nb_motion == 0) & np.isfinite(frame.nb_anchor) & np.isfinite(frame.nb_peak)
    retained &= (frame.nb_anchor - frame.nb_peak).abs() < 10
    candidates = frame.loc[retained]
    median = rolling_median(candidates.endpoint.to_numpy(), candidates.nb_anchor.to_numpy())
    retained.loc[candidates.index] &= np.abs(candidates.nb_anchor.to_numpy() - median) < 10
    frame["nightbeat"] = frame.nb_anchor.where(retained)
    return frame
