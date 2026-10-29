"""Context arms of the WatchTree stager: which windows, caps and baselines an arm keeps.

An arm is (B, F): backward reach B minutes, forward reach F minutes (F = 0: fully causal).
  - backward windows longer than B and forward windows longer than F are dropped (b120 -> none when B < 120 etc.)
  - tsb_min is capped at B (240 at the shipped B = 120); tnb_min is capped at F and dropped when F = 0
  - every baseline that looks back 2 h (hr_base / hr_rel, and the jerk floor that defines the bout indicator `act`,
    hence act_mean_* windows, tsb_min, tnb_min) looks back B minutes instead (see `baseline_shape`).
(120, 45) reproduces the shipped features.
"""
import re
import numpy as np
import pandas as pd
from arcasleep_watchtree import constants as C
from arcasleep_watchtree.features import EPOCHS_PER_MIN, WINDOWS, window_name
from arcasleep_watchtree.windows import window_stat

CURRENT = (120, 45)
ARMS = [(120, 45), (60, 45), (30, 45), (15, 45), (5, 45),
        (120, 15), (120, 5), (120, 0),
        (60, 15), (30, 15), (30, 5), (15, 0)]


def arm_name(b, f):
    return f"b{b}_f{f}"


def baseline_shape(b_min):
    """(block epochs, blocks kept, lowest blocks averaged, min epochs per block) for a baseline that looks back b_min.

    B = 120 gives the shipped 12 blocks of 20 epochs, mean of the lowest 3, block needs 5 epochs. Shorter reaches keep
    about the same share (lowest quarter of the blocks): block length = reach / 12 epochs (at least 1 epoch), as many
    blocks as fit in the reach, lowest round(blocks / 4), a block needs a quarter of its epochs (at least 1).
    """
    n = b_min * EPOCHS_PER_MIN
    be = max(1, n // 12)
    nb = n // be
    return be, nb, max(1, int(round(nb / 4))), max(1, be // 4)


def low_baseline(x, b_min):
    be, nb, lowest, min_ep = baseline_shape(b_min)
    n = x.size
    base = np.full(n, np.nan)
    blocks = []
    bs, bc, ps, pc = 0.0, 0, 0.0, 0
    for k in range(n):
        if np.isfinite(x[k]):
            bs += x[k]; bc += 1; ps += x[k]; pc += 1
        if blocks:
            base[k] = float(np.mean(sorted(blocks[-nb:])[:lowest]))
        elif pc:
            base[k] = ps / pc
        if (k + 1) % be == 0:
            if bc >= min_ep:
                blocks.append(bs / bc)
            bs, bc = 0.0, 0
    return base


def bout_timing(act, b, f):
    n = act.size
    since = np.empty(n); until = np.empty(n)
    last = -(10**9)
    reach = f * EPOCHS_PER_MIN
    cap = C.TSB_CAP_MIN if b >= 120 else b
    for k in range(n):
        if act[k] == 1:
            last = k
        since[k] = min((k - last) / EPOCHS_PER_MIN, cap)
        if reach:
            ahead = np.flatnonzero(act[k + 1: k + 1 + reach] == 1)
            until[k] = ahead[0] + 1 if ahead.size else reach
        else:
            until[k] = np.nan
    return since, until / EPOCHS_PER_MIN


def derive_night(jerk, hr, b, f):
    """Columns that change with the arm for one night: hr_base, hr_rel (+windows), act windows, tsb_min, tnb_min."""
    present = np.isfinite(jerk)
    base = low_baseline(hr, b)
    hr_rel = hr - base
    act = np.where(present, (jerk >= C.BOUT_RATIO * low_baseline(jerk, b)).astype(float), np.nan)
    since, until = bout_timing(np.nan_to_num(act, nan=0.0), b, f)
    cols = {}
    if b < 120:  # at B = 120 the shipped baseline-derived columns are kept as stored
        cols = {"hr_base": base, "hr_rel": hr_rel, "tsb_min": since}
    if f > 0 and (f != 45 or b < 120):
        cols["tnb_min"] = until
    series = {"act": act, "hr_rel": hr_rel}
    for signal, stat, d, minutes in WINDOWS:
        if b < 120 and signal in series:
            for m in minutes:
                if m <= (b if d == "b" else f):
                    cols[window_name(signal, stat, d, m)] = window_stat(series[signal], stat, d, m * EPOCHS_PER_MIN)
    return cols


def derive(table, b, f):
    """Per-arm changed columns for a whole table (sorted by subject, epoch), as float32."""
    parts = []
    for subject, ix in table.groupby("subject", sort=False).indices.items():
        g = table.iloc[ix]
        cols = derive_night(g.jerk.to_numpy(np.float64), g.hr.to_numpy(np.float64), b, f)
        parts.append(pd.DataFrame({"subject": subject, "epoch": g.epoch.to_numpy(), **cols}))
    return pd.concat(parts, ignore_index=True).astype({c: np.float32 for c in parts[0].columns if c not in ("subject", "epoch")})


def arm_features(base_list, b, f):
    out = []
    for name in base_list:
        m = re.search(r"_([bf])(\d+)$", name)
        if m:
            if int(m.group(2)) > (b if m.group(1) == "b" else f):
                continue
        elif name == "tnb_min" and f == 0:
            continue
        out.append(name)
    return out


def ctx_mask(epoch, n_rows, b, f):
    return (epoch >= b * EPOCHS_PER_MIN) & (epoch + f * EPOCHS_PER_MIN <= n_rows - 1)
