"""Per-sensor audit and wrist-convention PSDs for the fixed Somerset environmental accelerometers.

Timestamps in the source CSVs are naive ISO strings; they are read here as if UTC
and expressed as seconds since T0 (2025-06-01 00:00, same naive clock). Epochs are
30-s bins on that clock (T0 is a multiple of 30 s), frames are 10-s windows every
2 s, matching the wrist ProcessingConfig (log_psd, detrend, jerk_diff, NUFFT/jax).
"""
import json
import sys
from pathlib import Path

import numpy as np
from pisces_lite.proc import ProcessingConfig
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE

T0_US = int(np.datetime64('2025-06-01T00:00:00', 'us').astype(np.int64))
CHANNELS = ['x', 'y', 'z', 'mag', 'jerk']
BANDS = {'0.1-0.5': (0.1, 0.5), '0.5-2': (0.5, 2.0), '2-6': (2.0, 6.0)}
CHUNK = 6 * 3600


def stamp(t):
    return str(np.datetime64(int(round(t * 1e6)) + T0_US, 'us').astype('datetime64[s]'))


def audit(t, xyz):
    dt = np.diff(t)
    same = np.all(xyz[1:] == xyz[:-1], axis=1)
    gaps = dt > 1.0
    big = np.flatnonzero(dt > 60)
    bounds = np.r_[0, big + 1, len(t)]
    segments = [[stamp(t[a]), stamp(t[b - 1]), float((t[b - 1] - t[a]) / 3600)] for a, b in zip(bounds[:-1], bounds[1:])]
    logged = float(np.sum(dt[~gaps]))
    return dict(
        rows=int(len(t)), start=stamp(t[0]), end=stamp(t[-1]), span_h=float((t[-1] - t[0]) / 3600),
        logged_h=logged / 3600, nonincreasing_steps=int(np.sum(dt <= 0)),
        median_dt_ms=float(np.median(dt) * 1e3), p05_dt_ms=float(np.quantile(dt, .05) * 1e3),
        p95_dt_ms=float(np.quantile(dt, .95) * 1e3), rate_within_logging_hz=float(np.sum(~gaps) / logged),
        repeated_row_fraction=float(same.mean()),
        value_change_rate_within_logging_hz=float(np.sum(~same & ~gaps) / logged),
        gaps_over_1s=int(gaps.sum()), gap_total_h=float(dt[gaps].sum() / 3600),
        longest_gap_s=float(dt.max()), segments_split_at_gaps_over_60s=segments,
    )


def main(src, out):
    d = np.load(src)
    t = (d['t_us'] - T0_US) / 1e6
    xyz = d['xyz'].astype(np.float64)
    order = np.argsort(t, kind='mergesort'); t, xyz = t[order], xyz[order]
    info = audit(t, xyz)
    mag = np.linalg.norm(xyz, axis=1)
    info['counts_magnitude_median'] = float(np.median(mag))
    info['counts_magnitude_p01_p99'] = [float(np.quantile(mag, .01)), float(np.quantile(mag, .99))]
    info['axis_median_counts'] = [float(v) for v in np.median(xyz, axis=0)]
    # Scale: counts read as mm/s^2 (value steps are ~9.8 counts = 1 mg); 1 g = 9806.65 counts.
    scale = 9806.65
    info['counts_per_g_assumed'] = scale
    info['lsb_g_approx'] = 9.8 / scale
    keep = np.concatenate([[True], np.diff(t) > 0])
    raw = np.column_stack([t[keep], xyz[keep] / scale])
    fmax = 6.0  # the sensor's own output rate is ~12.5 Hz (values change every ~80 ms)
    proc = ProcessingConfig.from_dict(dict(type='nufft', fs=12.0, window_seconds=10, window_step_seconds=2,
                                           fmin=0.1, fmax=fmax, detrend=True, jerk_diff=True,
                                           spectrogram_kind='log_psd', nufft_backend='jax',
                                           nufft_backend_kwargs={'eps': 1e-6}, normalization_mode='none',
                                           spectral_channels=CHANNELS))
    freq = proc.get_frequencies()
    freq = freq[freq >= 0.1 - 1e-9]
    first = int(np.ceil(raw[0, 0] / 30) * 30); last = int(np.floor(raw[-1, 0] / 30) * 30)
    n_ep = (last - first) // 30
    n_fr = n_ep * 15
    band_idx = {k: (freq >= lo - 1e-9) & (freq <= hi + 1e-9) for k, (lo, hi) in BANDS.items()}
    frame_band = np.full((n_fr, len(BANDS), 3), np.nan, np.float32)  # bands x [trace, mag, jerk]
    epoch_psd = np.full((n_ep, 60, 5), np.nan, np.float32)  # 0.1..6.0 Hz, x,y,z,mag,jerk
    epoch_frames = np.zeros(n_ep, np.int16)
    lo6 = (freq <= 6.0 + 1e-9)
    assert lo6.sum() == 60, lo6.sum()
    for c0 in range(first, last, CHUNK):
        c1 = min(c0 + CHUNK, last)
        sel = (raw[:, 0] >= c0) & (raw[:, 0] < c1 + 10)
        if sel.sum() < 100:
            continue
        seg = raw[sel].copy(); seg[:, 0] -= c0
        frames = proc.apply(seg, normalize=False, padding=False, origin=0.)
        nf = (c1 - c0) // 2
        frames = frames[:nf]
        valid = ~np.all(frames == SPECTROGRAM_PADDING_VALUE, axis=(1, 2))
        lin = np.maximum(np.exp(frames.astype(np.float64)) - np.finfo(np.float64).eps, 0)
        lin[~valid] = np.nan
        assert lin.shape[1] == len(freq), (lin.shape, len(freq))
        f0 = (c0 - first) // 2
        trace = lin[:, :, :3].sum(axis=2)
        for b, take in enumerate(band_idx.values()):
            frame_band[f0:f0 + len(lin), b, 0] = trace[:, take].mean(axis=1)
            frame_band[f0:f0 + len(lin), b, 1] = lin[:, take, 3].mean(axis=1)
            frame_band[f0:f0 + len(lin), b, 2] = lin[:, take, 4].mean(axis=1)
        ne = len(lin) // 15
        e0 = (c0 - first) // 30
        blk = lin[:ne * 15, lo6].reshape(ne, 15, 60, 5)
        cnt = valid[:ne * 15].reshape(ne, 15).sum(axis=1)
        with np.errstate(invalid='ignore'):
            epoch_psd[e0:e0 + ne] = np.nanmean(blk, axis=1)
        epoch_frames[e0:e0 + ne] = cnt
        print(Path(src).stem, c0, flush=True)
    stem = Path(src).stem
    np.savez_compressed(Path(out) / f'{stem}_spectra.npz', epoch_psd=epoch_psd, epoch_frames=epoch_frames,
                        frame_band=frame_band, first_epoch_s=first, t0=str(np.datetime64(T0_US, 'us')),
                        frequencies=np.arange(1, 61) / 10, channels=np.array(CHANNELS),
                        bands=np.array(list(BANDS)), frame_channels=np.array(['trace', 'mag', 'jerk']))
    info.update(fmax_hz=fmax, config_fs=12.0, epochs=int(n_ep), first_epoch_s_since_T0=first)
    (Path(out) / f'{stem}_audit.json').write_text(json.dumps(info, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
