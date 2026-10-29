"""Analytic watch cost per estimator from the code (no device runs). Counting rules: biquad (sosfilt section, transposed direct form II) =
5 multiply-adds (MAC) and 2 state floats; 25 Hz; window 750 samples x 3 axes; autocorrelation lags 1..38 (lag grid of 40-180 bpm, interpolated to 141
bins: 2 MAC per bin); tracker step = forward and backward running-max passes over 141 bins plus emission, about 10 operations per bin; float32 (4 B).
Hilbert: zero-padded radix-2 1024-point complex FFT, (N/2)log2(N) butterflies x 4 MAC, forward + inverse per axis."""
import numpy as np, pandas as pd
WL, AX, NL, NB = 750, 3, 38, 141
lag_mac = lambda n_ax: sum((WL - L) for L in range(1, NL + 1)) * n_ax
bp = 2 * 5 * AX                      # band-pass: 2 biquads per axis
interp_norm = NB * 2 + NL
track = NB * 10; track_state = NB * 4
fft = (1024 // 2) * 10 * 4
rows = []
def add(name, per_sample, window_mac, extra, state_f):
    rows.append(dict(estimator=name, per_sample_mac=per_sample, per_window_mac=window_mac, per_window_other=extra,
                     amortized_mac_per_sample=per_sample + window_mac / WL, amortized_mac_per_second=25 * (per_sample + window_mac / WL),
                     state_floats=state_f, state_bytes=state_f * 4))
# current: band-pass, 3-axis autocorrelation over the window buffer
add("current (0.67-8 Hz, 3-axis autocorrelation)", bp, lag_mac(AX) + 2 * WL * AX + interp_norm + track, "tracker 10 ops/bin x141 included",
    2 * 2 * AX + WL * AX + track_state // 4)
# squared + low-pass
add("squared + low-pass envelope", bp + AX + 5 * AX + 2, lag_mac(1) + 2 * WL + interp_norm + track, "-",
    2 * 2 * AX + 2 * AX + WL + track_state // 4)
# Hilbert
add("Hilbert envelope (window FFT)", bp, AX * 2 * fft + AX * WL * 2 + WL * 2 + lag_mac(1) + 2 * WL + interp_norm + track,
    f"{AX * WL} square roots", 2 * 2 * AX + WL * AX + 2 * 1024 + WL + track_state // 4)
d = pd.DataFrame(rows).round(1); d.to_csv(__file__.replace("setup/watch_cost.py", "watch_cost.csv"), index=False)
pd.set_option("display.width", 250); print(d.T.to_string())
