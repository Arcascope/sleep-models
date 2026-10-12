"""Check the NUFFT PSD level for white noise at the wrist and environmental sampling rates, and the effect of 12.5 Hz sample-and-hold."""
import json, sys
import numpy as np
from pisces_lite.proc import ProcessingConfig
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE

proc = ProcessingConfig.from_dict(dict(type='nufft', fs=12.0, window_seconds=10, window_step_seconds=2, fmin=0.1, fmax=6.0,
    detrend=True, jerk_diff=True, spectrogram_kind='log_psd', nufft_backend='jax', nufft_backend_kwargs={'eps': 1e-6},
    normalization_mode='none', spectral_channels=['x', 'y', 'z', 'mag', 'jerk']))
rng = np.random.default_rng(0); sigma = 1e-3; dur = 1800; out = {}
def run(t, xyz):
    f = proc.apply(np.column_stack([t, xyz]), normalize=False, padding=False, origin=0.)
    v = ~np.all(f == SPECTROGRAM_PADDING_VALUE, axis=(1, 2)); p = np.exp(f[v].astype(float))
    freq = np.arange(1, 61) / 10
    return {b: float(np.median(p[:, (freq >= lo) & (freq <= hi), 0].mean(axis=1))) for b, (lo, hi) in {'0.1-0.5': (.1, .5), '2-6': (2, 6)}.items()}
for name, fs in [('wrist_16.7Hz', 16.7), ('env_48.8Hz', 48.8), ('uniform_12.5Hz', 12.5)]:
    t = np.arange(0, dur, 1 / fs) + rng.uniform(-0.2, 0.2, int(dur * fs)) / fs
    xyz = rng.normal(0, sigma, (len(t), 3)) + [0, 0, 1]
    out[name] = run(t, xyz) | {'sigma2_over_fs': sigma**2 / fs, 'band_white_level_one_sided_2sigma2_over_fs_dB': 10*np.log10(2*sigma**2/fs)}
# 12.5 Hz white noise held (zero-order hold) and polled at 48.8 Hz, as the environmental loggers do
t_true = np.arange(0, dur, 0.08); v = rng.normal(0, sigma, (len(t_true), 3)) + [0, 0, 1]
t_poll = np.sort(rng.uniform(0, dur - 0.1, int(dur * 48.8)))
held = v[np.searchsorted(t_true, t_poll, side='right') - 1]
out['held_12.5Hz_polled_48.8Hz'] = run(t_poll, held) | {'sigma2_over_12.5': sigma**2 / 12.5}
out['true_12.5Hz_same_noise'] = run(t_true, v)
for k, d in out.items():
    d.update({b + '_dB': float(10 * np.log10(d[b])) for b in ['0.1-0.5', '2-6']})
print(json.dumps(out, indent=1)); open(sys.argv[1], 'w').write(json.dumps(out, indent=1))
