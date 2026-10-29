"""Part 1: synthetic signals sampled at the real timestamps of every SleepAccel / Wearanize+ night (no real acceleration values).
Per night, writes $N/p1/<cohort>/<night>.pkl: rows of spectral artefact levels (truth = the signal evaluated exactly on the 25 Hz grid, vs
(a) to_25hz, (b1) NUFFT PSD of the raw samples on their own timestamps, (b2) least-squares band-limited reconstruction on the 25 Hz grid) and
pulse-rate errors of the autocorrelation estimator. Never written with participant ids."""
import os, sys, warnings, pickle, zlib
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np
import ahr2 as A, ratefree as RF, synth as S
from senpy import compute_nufft_spectrogram
from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, subject_ids
from arcasleep_watchtree.preprocess import to_25hz, nominal_rate
from loadnight import load_with_rows
N = Path(os.environ["N"]); _S = {}
NWIN = 20
BANDS = [(0.67, 2), (2, 3), (3, 5), (5, 8), (8, 11), (11, 12.5)]
ACF_EPOCHS = 240
PULSES = (("pulse60", 60.0, 0.0), ("pulse67", 67.0, 0.0), ("pulse67_noise", 67.0, 1.0))


def _init(name):
    warnings.simplefilter("ignore")
    from arcasleep.data.prepare import load_dataset
    c = COHORTS[name]; sync = None
    if name == "wearanize":
        from arcasleep_watchtree.uncropped import sync_table, RAW_ROOT; sync = sync_table(DATA_ROOT / RAW_ROOT)
    _S.update(name=name, ds=load_dataset(DATA_ROOT / c.directory), sync=sync)


def psd(tt, x, a):
    """PSD (3 sums not needed: one axis) of samples tt, x over the 30 s window starting at a; bins j/30 Hz up to 12.5 Hz"""
    r = compute_nufft_spectrogram(tt, x, 30.0, 0.0, kind="psd", origin_s=a, empty_windows="keep", min_samples=100, target_fs=25.0)
    return (r.Sxx[0], r.frequencies) if r.valid[0] else (None, None)


def images(f, fn):
    out = []
    for k in (1, 2, 3):
        for s in (-1, 1):
            x = abs(f + s * k * fn) % 25.0; out.append(25.0 - x if x > 12.5 else x)
    return np.array(out)


def line_stats(P, Pt, fr, f, fn):
    """P, Pt: PSD of the method and of the truth. Line = bins within 4 of f. Returns dict of dB levels relative to the truth line power."""
    jl = np.abs(fr - f) <= 4 / 30.0
    base = Pt[jl].sum(); out = {"line_dB": 10 * np.log10(max(P[jl].sum(), 1e-300) / base)}
    for lab, sel in (("band", (fr >= 0.67) & (fr <= 8.0) & ~jl), ("all", (fr >= 0.1) & ~jl)):
        if not sel.any():
            out |= {f"spur_{lab}_total_dB": np.nan, f"spur_{lab}_peak_dB": np.nan, f"spur_{lab}_peak_hz": np.nan}; continue
        i = np.argmax(np.where(sel, P, -1)); pk = P[i]
        out[f"spur_{lab}_total_dB"] = 10 * np.log10(max(P[sel].sum(), 1e-300) / base); out[f"spur_{lab}_peak_dB"] = 10 * np.log10(max(pk, 1e-300) / base)
        out[f"spur_{lab}_peak_hz"] = fr[i]
    out["peak_at_image"] = float(np.abs(images(f, fn) - out["spur_all_peak_hz"]).min() <= 0.1) if np.isfinite(out["spur_all_peak_hz"]) else np.nan
    return out


def _one(args):
    subject, rep, name = args
    dst = N / "p1" / name / f"{rep}.pkl"
    if dst.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    night, rows = load_with_rows(name, subject, rep, _S["ds"], _S["sync"])
    ne = night.labels.size; t0 = night.t0; t = rows[:, 0]; fn = nominal_rate(t); u = t - t0
    rng = np.random.default_rng(zlib.crc32(rep.encode()))
    n25 = ne * 750; u25 = np.arange(n25) / 25.0
    sigs = {}   # key -> (x at raw (n,3), truth at grid (n25,3), meta)
    for g in range(3):
        fs3 = S.SIN_F[3 * g:3 * g + 3]
        sigs[f"sin{g}"] = (S.sines(u, fs3), S.sines(u25, fs3), fs3)
    fe = 1.0 / np.median(np.diff(t))   # white noise of equal PSD level on both grids: grid variance = raw variance x 25 / fs
    sigs["noise"] = (rng.standard_normal((u.size, 3)), np.sqrt(25.0 / fe) * rng.standard_normal((n25, 3)), None)
    sigs["pulse60s"] = (S.pulse(u, 60.0), S.pulse(u25, 60.0), None)
    rows_out, bandrows = [], []
    # to_25hz outputs (also the valid mask)
    ya = {}
    for k, (x, xt, meta) in sigs.items():
        ya[k], valid25 = to_25hz(t, x, t0, ne)
    ep_ok = valid25.reshape(ne, -1).mean(axis=1) >= 0.8
    nyq, edges = RF.window_nyquist(t, t0, ne)
    ok = np.flatnonzero(ep_ok & np.isfinite(nyq) & (np.diff(edges) >= 100))
    pick = ok[np.linspace(0, ok.size - 1, min(NWIN, ok.size)).astype(int)]
    # spectral
    acc = {}   # (key, axis, method) -> list of per-window dicts
    for k, (x, xt, meta) in sigs.items():
        for kw in pick:
            a = t0 + 30.0 * kw; a_, b_ = edges[kw], edges[kw + 1]
            m25 = np.arange(kw * 750, (kw + 1) * 750); m25 = m25[valid25[m25]]
            if m25.size < 750:
                continue
            fm = min(8.0, 0.9 * nyq[kw]); coef, ff = RF.ls_fit(t[a_:b_] - a, x[a_:b_], fm); rec = RF.ls_recon(coef, ff, 0.67, fm)
            for ax in range(3):
                P = {}
                P["truth"], fr = psd(t0 + m25 / 25.0, xt[m25, ax], a)
                P["a"], _ = psd(t0 + m25 / 25.0, ya[k][m25, ax], a)
                P["b1"], _ = psd(t[a_:b_], x[a_:b_, ax], a)
                P["b2"], _ = psd(a + RF.U, rec[:, ax], a)
                if any(v is None for v in P.values()):
                    continue
                if k == "noise" or k == "pulse60s":
                    if k == "pulse60s" and ax != 0:
                        continue
                    acc.setdefault((k, ax), []).append((kw, P, fr, fm))
                else:
                    acc.setdefault((k, ax), []).append((kw, P, fr, fm))
    spec = []
    for (k, ax), lst in acc.items():
        fr = lst[0][2]
        if k.startswith("sin"):
            f = sigs[k][2][ax]
            for meth in ("truth", "a", "b1", "b2"):
                ws = []
                for kw, P, _, fm in lst:
                    d = line_stats(P[meth], P["truth"], fr, f, fn)
                    if meth == "b2" and f > fm:
                        d["line_dB"] = np.nan
                    ws.append(d)
                d = {kk: float(np.nanmedian([w[kk] for w in ws])) for kk in ws[0]}
                d["spur_all_peak_dB_max"] = float(np.nanmax([w["spur_all_peak_dB"] for w in ws])); d["spur_band_peak_dB_max"] = float(np.nanmax([w["spur_band_peak_dB"] for w in ws]))
                d["peak_at_image"] = float(np.nanmean([w["peak_at_image"] for w in ws]))
                spec.append(dict(cohort=name, nominal=fn, kind="sine", f_hz=f, method=meth, windows=len(ws), **d))
        elif k == "noise":
            for meth in ("a", "b1", "b2"):
                for lo, hi in BANDS:
                    s = (fr >= lo) & (fr < hi)
                    r = [P[meth][s].sum() / P["truth"][s].sum() for _, P, _, _ in lst]
                    bandrows.append(dict(cohort=name, nominal=fn, method=meth, band=f"{lo}-{hi}", ratio=float(np.median(r)), windows=len(r)))
    # pulse PSD example: x axis, window nearest the middle of the night
    ex = None
    lst = acc.get(("pulse60s", 0))
    if lst:
        kw, P, fr, fm = lst[len(lst) // 2]; ex = dict(f=fr, **P)
    # autocorrelation estimator on the first ACF_EPOCHS epochs
    ne2 = min(ACF_EPOCHS, ne); m = t < t0 + 30 * ne2; t2 = t[m]; u2 = t2 - t0; n2 = ne2 * 750; ug = np.arange(n2) / 25.0
    est = []
    for lab, bpm, nz in PULSES:
        x = S.pulse(u2, bpm); xt = S.pulse(ug, bpm)
        if nz:
            sd = x.std(axis=0); x = x + nz * sd * rng.standard_normal(x.shape); xt = xt + nz * sd * rng.standard_normal(xt.shape)
        y, v = to_25hz(t2, x, t0, ne2); vep = RF.coverage(t2, t0, ne2) >= 0.8
        cur = {"A": A.acf_curves(y, v, ne2, 30, A._SOS_8), "truth": A.acf_curves(xt, np.ones(n2, bool), ne2, 30, A._SOS_8)}
        c, _ = RF.curves_all(t2, x, t0, ne2, vep, with_control=(t0 + np.flatnonzero(v) / 25.0, y[v]))
        cur.update(c)
        for arm, cc in cur.items():
            good = np.isfinite(cc).all(axis=1)
            if not good.any():
                continue
            pk = A.GRID_BPM[np.argmax(cc[good], axis=1)]
            tr = A.viterbi_filter(10.0 * cc, 1.0)[0]; tr = tr[good]
            est.append(dict(cohort=name, nominal=fn, signal=lab, bpm=bpm, arm=arm, windows=int(good.sum()), peak_mae=float(np.abs(pk - bpm).mean()), peak_within3=float((np.abs(pk - bpm) <= 3).mean()),
                            track_mae=float(np.abs(tr - bpm).mean()), track_bias=float((tr - bpm).mean())))
    meta = dict(cohort=name, nominal=fn, effective_hz=float(1 / np.median(np.diff(t))), nights=1)
    with open(dst.with_suffix(".tmp"), "wb") as fh:
        pickle.dump(dict(meta=meta, spec=spec, bands=bandrows, example=ex, est=est), fh)
    dst.with_suffix(".tmp").rename(dst)
    return 1


if __name__ == "__main__":
    from bpm_common import source_tables
    for name in sys.argv[1].split(","):
        sel = set(source_tables()[name].subject)
        ids, rep = subject_ids(COHORTS[name])
        todo = [(s, rep[s], name) for s in ids if rep[s] in sel]
        if len(sys.argv) > 2: todo = todo[:int(sys.argv[2])]
        with ProcessPoolExecutor(int(os.environ.get("NPROC", 12)), mp_context=multiprocessing.get_context("spawn"), initializer=_init, initargs=(name,)) as pool:
            list(pool.map(_one, todo))
        print(name, len(todo), flush=True)
