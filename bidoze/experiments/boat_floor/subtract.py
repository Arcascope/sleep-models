"""Frozen-release floor subtraction in the input power domain; confidential research only.

Each night's additive floor F is estimated without PSG from the magnitude/jerk spectrum
above the model band (half Nyquist to 0.95 Nyquist, never below 6.5 Hz): the night median of
each frame's median high-band power. Every release frame bin P becomes
ln(max(P - alpha * F, 0) + F_land), with F_land the median Clinical floor per channel, so a
fully removed floor is replaced by a land-level pedestal instead of an unbounded log.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from pisces_lite.proc import ProcessingConfig
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE
from arcasleep.data.batching import recording_input, pad_pooled
from arcasleep.data.bundle import CacheBundle
from arcasleep.data.prepare import clean_accelerometer_rows, median_sampling_rate
from arcasleep.predict import load_predictor
from arcasleep.ggir import evaluate_epoch_grid
from arcasleep_bidoze.adult import hybrid_calls
from boat_evidence.score import DIVISOR, METRICS, metrics

ALPHAS = [0.5, 1.0, 1.5, 2.0]
COHORTS = ['SURF', 'Somerset', 'SleepAccel-Clinical']
LAND_CACHE = 'bidoze/results/bidoze-adult-devices-0/setup/cache/matched/12Hz_log_PSD_aligned'


def raw_night(root, name):
    cohort, stem = name.split('_subject_', 1)
    path = root / cohort / 'cleaned_accelerometer' / f'{stem}.csv'
    return clean_accelerometer_rows(pd.read_csv(path).to_numpy(float)), path


def floor_estimate(raw, config):
    """Night median of per-frame median power above the model band, per channel."""
    fs = median_sampling_rate(raw[:, 0])
    nyquist = np.floor(fs * 0.999 * 100) / 200
    lo, hi = max(6.5, nyquist / 2), 0.95 * nyquist
    frames = ProcessingConfig.from_dict(dict(config, fs=2 * nyquist, fmax=None)).apply(
        raw, normalize=False, padding=False, origin=0.)
    keep = ~np.all(frames == SPECTROGRAM_PADDING_VALUE, axis=(1, 2))
    freq = np.arange(1, frames.shape[1] + 1) / 10
    band = (freq >= lo) & (freq <= hi)
    per_frame = np.median(np.exp(frames[keep][:, band]), axis=1)
    return np.median(per_frame, axis=0), dict(fs=float(fs), band_hz=[float(lo), float(hi)],
        frames=int(keep.sum()), frame_floor_iqr_db=(10 * np.log10(np.percentile(per_frame, [25, 75], axis=0))).tolist())


def subtracted(frames, floor, land, alpha):
    pad = np.all(frames == SPECTROGRAM_PADDING_VALUE, axis=(1, 2))
    power = np.exp(frames.astype(np.float64))
    out = np.log(np.maximum(power - alpha * floor, 0) + land)
    out[pad] = SPECTROGRAM_PADDING_VALUE
    clamped = float(np.mean(power[~pad] <= alpha * floor))
    return out.astype(frames.dtype), clamped


def release_frames(r, raw, models, source):
    if r['cohort'] == 'SleepAccel-Clinical':
        stem = r['name'].split('_subject_', 1)[1]
        return CacheBundle.from_npz(source / LAND_CACHE / f'SleepAccel-Clinical_subject_{stem}_features.npz').X[0]
    frames, origin = models[0].features(raw, origin=0.)
    assert origin == 0
    return frames


def summarize(out):
    nights = pd.concat([pd.read_csv(p) for p in (out / 'nights').glob('*/metrics.csv')], ignore_index=True)
    nights.to_csv(out / 'analysis/nights.csv', index=False)
    keys = ['cohort', 'participant', 'group', 'variant', 'mode']
    subjects = nights.groupby(keys)[METRICS + ['clamped_fraction']].mean().reset_index()
    subjects.to_csv(out / 'analysis/subjects.csv', index=False)
    summary = subjects.groupby(['cohort', 'variant', 'mode'])[METRICS + ['clamped_fraction']].mean()
    summary['participants'] = subjects.groupby(['cohort', 'variant', 'mode']).size()
    summary.reset_index().to_csv(out / 'analysis/summary.csv', index=False)


def main(a):
    for d in ('setup', 'nights', 'analysis'):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    config = json.loads((a.data / 'bundles/seed0/preprocessing.json').read_text())
    meta = pd.read_csv(a.data / 'inventory.csv')
    meta = meta[meta.cohort.isin(COHORTS)]
    if a.limit:
        meta = meta.groupby('cohort').head(a.limit)
    protocol = dict(code=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), alphas=ALPHAS,
        transform='ln(max(P - alpha*F, 0) + F_land) per frame bin, both channels, before epoch pooling',
        floor='night median of per-frame median power over [max(6.5, Nyquist/2), 0.95 Nyquist] Hz; no PSG',
        land_pedestal='median Clinical floor per channel', ensemble='mean five-class softmax, three frozen release seeds',
        decoding='stage-conditional, frozen divisor, physical gaps forced to gap; raw and GGIR',
        aggregation='recording within participant then participant macro',
        selection='alpha chosen on the inner boat group only (analyze.py); outer curves are descriptive')
    (a.out / 'setup/protocol.json').write_text(json.dumps(protocol, indent=2))
    floors_path = a.out / 'setup/floors.csv'
    floors = pd.read_csv(floors_path).set_index('name') if floors_path.exists() else pd.DataFrame()
    rows = []
    for r in meta.to_dict('records'):
        if r['name'] in floors.index:
            continue
        raw, _ = raw_night(a.root, r['name'])
        f, info = floor_estimate(raw, config)
        rows.append(dict(name=r['name'], cohort=r['cohort'], participant=r['participant'], group=r['group'],
            mag_floor=f[0], jerk_floor=f[1], mag_floor_db=10 * np.log10(f[0]), jerk_floor_db=10 * np.log10(f[1]),
            fs=info['fs'], band_lo=info['band_hz'][0], band_hi=info['band_hz'][1], frames=info['frames']))
        print('floor', r['name'], f'{rows[-1]["mag_floor_db"]:.1f} {rows[-1]["jerk_floor_db"]:.1f} dB', flush=True)
    if rows:
        floors = pd.concat([floors.reset_index(), pd.DataFrame(rows)], ignore_index=True)
        floors.to_csv(floors_path, index=False)
        floors = floors.set_index('name')
    land = floors[floors.cohort == 'SleepAccel-Clinical'][['mag_floor', 'jerk_floor']].median().to_numpy()
    (a.out / 'setup/land-pedestal.json').write_text(json.dumps(dict(mag=land[0], jerk=land[1],
        mag_db=10 * np.log10(land[0]), jerk_db=10 * np.log10(land[1]), nights=int((floors.cohort == 'SleepAccel-Clinical').sum()))))
    import jax
    models = [load_predictor(a.data / f'bundles/seed{i}') for i in range(3)]

    def ensemble(pooled):
        x = pad_pooled(pooled, 3000)[None].astype(np.float32)
        return np.mean([np.asarray(jax.nn.softmax(m.model.infer(x), axis=-1))[0] for m in models], axis=0)

    for r in meta.to_dict('records'):
        folder = a.out / 'nights' / r['name']
        if (folder / 'complete.json').exists():
            continue
        folder.mkdir(exist_ok=True)
        raw = None if r['cohort'] == 'SleepAccel-Clinical' else raw_night(a.root, r['name'])[0]
        with np.load(a.data / 'nights' / f"{r['name']}.npz") as d:
            n = int(d['n']); prepared = d['x'][:n].copy(); angles = d['anglez'][:n]
            truth = (d['truth'] if 'truth' in d else d['y'])[:n]; still = d['still'][:n]
        frames = release_frames(r, raw, models, a.source)
        missing = prepared[:, -1, 0] >= 1
        native = pad_pooled(recording_input(frames, len(frames) // 15, 1), n)
        audit = dict(max_input_diff=float(np.abs(native - prepared).max()), median_input_diff=float(np.median(np.abs(native - prepared))))
        assert audit['median_input_diff'] < 1e-5 and np.array_equal(native[:, -1, 0] >= 1, missing), audit
        floor = floors.loc[r['name'], ['mag_floor', 'jerk_floor']].to_numpy(float)
        outputs = {'native': ensemble(prepared)[:n]}; clamped = {'native': 0.0}
        for alpha in ALPHAS:
            changed, clamped[f'alpha{alpha:g}'] = subtracted(frames, floor, land, alpha)
            pooled = pad_pooled(recording_input(changed, len(changed) // 15, 1), n)
            assert np.array_equal(pooled[:, -1], native[:, -1])
            outputs[f'alpha{alpha:g}'] = ensemble(pooled)[:n]
        ggir = evaluate_epoch_grid(angles); results = []
        for variant, p in outputs.items():
            call = hybrid_calls(p[:, :4], p[:, :4], DIVISOR); call[missing] = 4
            filtered = call.copy(); filtered[(call > 0) & (call < 4) & ~ggir.is_sib & ~ggir.is_spt] = 0
            for mode, c in [('raw', call), ('ggir', filtered)]:
                results.append({k: r[k] for k in ('name', 'cohort', 'participant', 'group')} |
                    dict(variant=variant, mode=mode, clamped_fraction=clamped[variant]) | metrics(truth, p, c, still, missing, True))
        pd.DataFrame(results).to_csv(folder / 'metrics.csv', index=False)
        np.savez_compressed(folder / 'probabilities.npz', **outputs, truth=truth, still=still, missing=missing)
        (folder / 'complete.json').write_text(json.dumps(audit | dict(clamped=clamped, covered=int((~missing).sum()))))
        print(r['name'], audit, flush=True)
    summarize(a.out)
    (a.out / 'complete.json').write_text(json.dumps(dict(nights=len(meta), smoke=bool(a.limit))))
    print('floor subtraction complete', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path); p.add_argument('data', type=Path); p.add_argument('source', type=Path); p.add_argument('out', type=Path)
    p.add_argument('--limit', type=int)
    main(p.parse_args())
