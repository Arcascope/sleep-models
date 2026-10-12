"""Frozen-release sampling controls with retained axis PSDs; confidential research only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import firwin, freqz, kaiserord, fftconvolve
from pisces_lite.proc import ProcessingConfig
from pisces_lite.proc.constants import SPECTROGRAM_PADDING_VALUE
from arcasleep.data.prepare import clean_accelerometer_rows, median_sampling_rate
from arcasleep.data.batching import recording_input, pad_pooled
from arcasleep.predict import load_predictor
from arcasleep_bidoze.adult import hybrid_calls
from arcasleep.ggir import evaluate_epoch_grid
from boat_evidence.score import DIVISOR, METRICS, metrics

CHANNELS = ['x', 'y', 'z', 'mag', 'jerk']
VARIANTS = ['native', 'decimate3', 'lowpass_native', 'lowpass_decimate3']
BANDS = [(0.1, 0.5), (0.5, 2), (2, 6)]


def filter_axes(raw):
    """Symmetric FIR per continuous segment; never interpolate over a recording gap."""
    fs = median_sampling_rate(raw[:, 0])
    # 6 Hz pass edge, 8 Hz stop edge, explicitly below the decimated Nyquist.
    ntaps, beta = kaiserord(80, 2 / (fs / 2))
    ntaps += 1 - ntaps % 2
    taps = firwin(ntaps, 7, window=('kaiser', beta), fs=fs)
    f, h = freqz(taps, worN=16384, fs=fs)
    response = dict(fs=fs, taps=ntaps, pass_edge_hz=6, stop_edge_hz=8,
                    stop_max_db=float(20 * np.log10(np.max(np.abs(h[f >= 8])))),
                    pass_min_db=float(20 * np.log10(np.min(np.abs(h[f <= 6])))))
    assert fs / 2 > 8, 'Filter stop band must fit below native Nyquist'
    assert response['stop_max_db'] < -60 and response['pass_min_db'] > -0.1
    result = raw.copy()
    boundaries = np.r_[0, np.flatnonzero(np.diff(raw[:, 0]) > 0.2) + 1, len(raw)]
    short = 0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        t = raw[lo:hi, 0]
        if len(t) < 2:
            short += len(t)
            continue
        grid = np.arange(t[0], t[-1] + 0.1 / fs, 1 / fs)
        if len(grid) < 2:
            short += len(t)
            continue
        # Interpolation within a segment is part of the documented filter arm.
        xyz = np.column_stack([np.interp(grid, t, raw[lo:hi, c]) for c in (1, 2, 3)])
        pad = min(ntaps, len(grid) - 1)
        for c in range(3):
            extended = np.pad(xyz[:, c], (pad, pad), mode='reflect')
            smooth = fftconvolve(extended, taps, mode='same')[pad:-pad]
            result[lo:hi, c + 1] = np.interp(t, grid, smooth)
        short += len(t) if len(grid) < ntaps else 0
    response.update(segments=len(boundaries) - 1, short_segment_samples=int(short),
                    boundary='reflection; short segments retained and counted',
                    jitter='linear interpolation to within-segment regular grid, then back')
    return result, response


def variants(raw, cohort):
    low, response = filter_axes(raw)
    if cohort == 'Somerset':
        return {'native': raw, 'lowpass_native': low}, response
    assert response['fs'] / 6 > 8, 'Stop band must fit below decimated Nyquist'
    return dict(zip(VARIANTS, [raw, raw[::3], low, low[::3]])), response


def epoch_axis_covariance(raw, n):
    index=np.floor(raw[:,0]/30).astype(int);keep=(index>=0)&(index<n)
    index=index[keep];xyz=raw[keep,1:];count=np.bincount(index,minlength=n)
    mean=np.column_stack([np.bincount(index,weights=xyz[:,c],minlength=n) for c in range(3)])/np.maximum(count[:,None],1)
    centered=xyz-mean[index];cov=np.zeros((n,3,3))
    for i in range(3):
        for j in range(3):
            cov[:,i,j]=np.bincount(index,weights=centered[:,i]*centered[:,j],minlength=n)/np.maximum(count,1)
    mean[count==0]=np.nan;cov[count==0]=np.nan
    return mean,cov,count


def epoch_power(frames, n):
    aligned = np.full((n * 15, *frames.shape[1:]), SPECTROGRAM_PADDING_VALUE)
    aligned[:min(len(frames), len(aligned))] = frames[:len(aligned)]
    a = aligned.reshape(n, 15, *frames.shape[1:])
    valid = ~np.all(a == SPECTROGRAM_PADDING_VALUE, axis=(2, 3))
    linear = np.maximum(np.exp(a) - np.finfo(np.float64).eps, 0)
    mean = np.sum(np.where(valid[..., None, None], linear, 0), axis=1) / np.maximum(valid.sum(axis=1)[:, None, None], 1)
    mean[valid.sum(axis=1) == 0] = np.nan
    return mean, valid.sum(axis=1)


def band_rows(frames, **metadata):
    # Scalar PSDs retain the backend's own two-sided normalization (no invented +3 dB).
    keep = ~np.all(frames == SPECTROGRAM_PADDING_VALUE, axis=(1, 2))
    power = np.maximum(np.exp(frames[keep]) - np.finfo(np.float64).eps, 0)
    freq = np.arange(1, frames.shape[1] + 1) / 10
    channels = np.concatenate([power, power[:, :, :3].sum(axis=2, keepdims=True)], axis=2)
    rows = []
    for lo, hi in BANDS:
        take = (freq >= lo) & (freq <= hi)
        for c, channel in enumerate(CHANNELS + ['trace']):
            values = np.mean(channels[:, take, c], axis=1)
            rows.append(metadata | dict(channel=channel, band=f'{lo:g}-{hi:g}',
                median_frame_power_db=float(10*np.log10(max(np.median(values), 1e-30))),
                p10_frame_power_db=float(10*np.log10(max(np.quantile(values, .1), 1e-30))),
                mean_power_db=float(10*np.log10(max(np.mean(values), 1e-30)))))
    return rows


def summarize(out):
    parts = list((out / 'nights').glob('*/metrics.csv'))
    if not parts:
        return
    table = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    table.to_csv(out / 'analysis/nights.csv', index=False)
    keys = ['cohort', 'participant', 'group', 'variant', 'mode']
    subjects = table.groupby(keys)[METRICS].mean().reset_index()
    subjects.to_csv(out / 'analysis/subjects.csv', index=False)
    summary = subjects.groupby(['cohort', 'variant', 'mode'])[METRICS].mean().reset_index()
    counts = subjects.groupby(['cohort','variant','mode']).size().rename('participants').reset_index()
    summary.merge(counts).to_csv(out / 'analysis/summary.csv', index=False)


def main(a):
    for directory in ('setup', 'nights', 'analysis', 'calibration'):
        (a.out / directory).mkdir(parents=True, exist_ok=True)
    config = json.loads((a.data / 'bundles/seed0/preprocessing.json').read_text())
    axis_config = dict(config, spectral_channels=CHANNELS)
    processor = ProcessingConfig.from_dict(axis_config)
    protocol = dict(code=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        preprocessing=config, retained_channels=CHANNELS, variants=VARIANTS,
        release_seeds=[0,1,2], filtering='Kaiser FIR: 6 Hz pass, 8 Hz stop, 80 dB design; symmetric convolution',
        ensemble='mean five-class softmax across three frozen release seeds',
        coverage='intersection of original prepared and all variant covered epochs',
        decoding='frozen adult divisor, original still/anglez/GGIR; raw and GGIR',
        aggregation='recording within participant then participant macro',
        threshold='spec95 is descriptive per-night ROC, not a test-calibrated deployment threshold',
        jerk='norm of raw sample differences, no division by elapsed time',
        psd='backend power convention unchanged, natural log input; dB = 10 log10(linear power)',
        status='sampling perturbation, not an isolated estimate of a causal sensor floor')
    protocol['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol['native_baseline'] = 'Exact frozen prepared cache for native scoring; regeneration audited; two-channel release extraction for all perturbations'
    protocol['native_audit'] = 'Recomputed vs prepared input: median log-power diff <1e-5; max low-power-bin discrepancy recorded, prediction equivalence checked; ensemble max probability diff <0.001, at most one changed call per night'
    (a.out/'setup/protocol.json').write_text(json.dumps(protocol, indent=2))
    if a.calibration:
        rows = []; rng = np.random.default_rng(12); t = np.arange(600 * 50)/50
        for case in ('static', 'coherent2Hz', 'vibration24Hz'):
            clean = np.column_stack([t, np.zeros_like(t), np.zeros_like(t), np.ones_like(t)])
            if case != 'static':
                clean[:, 1] = .01*np.sin(2*np.pi*(2 if case == 'coherent2Hz' else 24)*t)
            for sigma in (0, 2.5, 3.7, 5):
                raw = clean.copy(); raw[:,1:] += rng.normal(0,sigma/1000,(len(raw),3))
                arms, response = variants(raw, 'SURF')
                for name, signal in arms.items():
                    f = processor.apply(signal, normalize=False, padding=False, origin=0.)
                    rows += band_rows(f[5:-5], case=case, sigma_axis_mg=sigma, variant=name)
        pd.DataFrame(rows).to_csv(a.out/'calibration/power.csv', index=False)
        (a.out/'calibration/filter.json').write_text(json.dumps(response,indent=2))
        print('synthetic calibration complete', flush=True)
        if a.calibration_only:
            return
    import jax
    from importlib.metadata import version
    (a.out/'setup/environment.json').write_text(json.dumps({k: version(k) for k in ['numpy','scipy','pandas','jax','jaxlib','pisces-lite']},indent=2))
    models = [load_predictor(a.data / f'bundles/seed{i}') for i in range(3)]
    meta = pd.read_csv(a.data/'inventory.csv')
    meta = meta[meta.cohort.isin(a.cohorts.split(','))]
    if a.limit:
        meta = meta.head(a.limit)
    (a.out/'setup/inventory.csv').write_text(meta.to_csv(index=False))
    hashes = {f'seed{i}': hashlib.sha256((a.data/f'bundles/seed{i}/model.safetensors').read_bytes()).hexdigest() for i in range(3)}
    (a.out/'setup/bundle-sha256.json').write_text(json.dumps(hashes,indent=2))
    for r in meta.to_dict('records'):
        folder = a.out/'nights'/r['name']
        if (folder/'complete.json').exists():
            continue
        folder.mkdir(exist_ok=True); start = time.monotonic()
        stem = r['name'].split('_subject_',1)[1]
        raw_path = a.root/r['cohort']/'cleaned_accelerometer'/f'{stem}.csv'
        raw = clean_accelerometer_rows(pd.read_csv(raw_path).to_numpy(float))
        with np.load(a.data/'nights'/f"{r['name']}.npz") as d:
            n = int(d['n']); truth = d['truth'][:n]; still = d['still'][:n]; angles = d['anglez'][:n]
            initial_missing = d['x'][:n,-1,0] >= 1
            prepared = d['x'][:n].copy()
        arms, response = variants(raw, r['cohort']); outputs = {}; missing = {}; power_rows = []
        
        for name, signal in arms.items():
            frames = processor.apply(signal, normalize=False, padding=False, origin=0.)
            assert frames.shape[1:] == (60,5), frames.shape
            # Model scoring uses the release's unchanged two-channel extraction;
            # axes are retained in a separate diagnostic branch.
            model_frames, origin = models[0].features(signal, origin=0.)
            assert origin == 0
            pooled = pad_pooled(recording_input(model_frames,len(model_frames)//15,1),n)
            if name == 'native':
                delta = np.abs(pooled-prepared)
                audit = dict(max_diff=float(delta.max()), median_diff=float(np.median(delta)), changed=int((delta>2e-4).sum()), shape=list(pooled.shape), gap_diff=float(np.abs(pooled[:,-1]-prepared[:,-1]).max()))
                (folder/'native-input-audit.json').write_text(json.dumps(audit,indent=2))
                assert audit['median_diff'] < 1e-5 and audit['gap_diff'] < 1e-6 and np.array_equal(pooled[:n,-1,0] >= 1, initial_missing), f'Native inputs differ materially from frozen preparation: {audit}'
            missing[name] = pooled[:n,-1,0] >= 1
            x = pad_pooled(pooled,3000)[None].astype(np.float32)
            outputs[name] = np.mean([np.asarray(jax.nn.softmax(m.model.infer(x),axis=-1))[0,:n] for m in models],axis=0)
            if name == 'native':
                refx = pad_pooled(prepared,3000)[None].astype(np.float32)
                refp = np.mean([np.asarray(jax.nn.softmax(m.model.infer(refx),axis=-1))[0,:n] for m in models],axis=0)
                audit['max_probability_diff'] = float(np.abs(outputs[name]-refp).max())
                audit['changed_calls'] = int(np.sum(hybrid_calls(outputs[name],outputs[name],DIVISOR) != hybrid_calls(refp,refp,DIVISOR)))
                (folder/'native-input-audit.json').write_text(json.dumps(audit,indent=2))
                assert audit['max_probability_diff'] < .001 and audit['changed_calls'] <= 1, audit
                outputs[name] = refp
                pooled = prepared
            ep_power, count = epoch_power(frames,n)
            axis_mean,axis_cov,axis_count=epoch_axis_covariance(signal,n)
            np.savez_compressed(folder/f'{name}_axis_psd.npz', epoch_linear_psd=ep_power.astype(np.float32),
                frame_count=count, frequencies=np.arange(1,61)/10, channels=np.array(CHANNELS),
                gap=missing[name], truth=truth, still=still, model_epoch_log_psd=pooled.astype(np.float32),
                axis_mean=axis_mean.astype(np.float32),axis_covariance=axis_cov.astype(np.float32),axis_sample_count=axis_count)
            power_rows += band_rows(frames,variant=name,name=r['name'],cohort=r['cohort'],participant=r['participant'])
        common_missing = initial_missing | np.logical_or.reduce(list(missing.values()))
        ggir = evaluate_epoch_grid(angles); results=[]
        for name, p in outputs.items():
            calls = hybrid_calls(p,p,DIVISOR); calls[missing[name]] = 4
            filtered = calls.copy(); filtered[(calls>0)&(calls<4)&~ggir.is_sib&~ggir.is_spt]=0
            for mode, call in [('raw',calls),('ggir',filtered)]:
                results.append({k:r[k] for k in ('name','cohort','participant','group')} | dict(variant=name,mode=mode) |
                    metrics(truth,p,call,still,common_missing,True))
        pd.DataFrame(results).to_csv(folder/'metrics.csv',index=False)
        pd.DataFrame(power_rows).to_csv(folder/'power.csv',index=False)
        np.savez_compressed(folder/'probabilities.npz', **outputs, truth=truth, still=still, common_missing=common_missing)
        record=dict(code=protocol['code'],script_sha256=protocol['script_sha256'],seconds=time.monotonic()-start, filter=response,raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            original_covered=int((~initial_missing).sum()),common_covered=int((~common_missing).sum()),
            variants_covered={k:int((~v).sum()) for k,v in missing.items()})
        (folder/'complete.json').write_text(json.dumps(record,indent=2))
        summarize(a.out); print(r['name'],record,flush=True)
    summarize(a.out)
    (a.out/'complete.json').write_text(json.dumps(dict(nights=len(meta),complete=True,smoke=bool(a.limit))))
    print('sampling experiment complete',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('data',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--limit',type=int);p.add_argument('--cohorts',default='SURF,Somerset')
    p.add_argument('--calibration',action='store_true');p.add_argument('--calibration-only',action='store_true')
    main(p.parse_args())
