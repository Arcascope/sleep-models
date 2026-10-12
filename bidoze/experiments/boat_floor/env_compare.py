"""Environmental floors, clock-offset scan and synchronized environment-vs-wrist band power on Somerset nights.

Clocks: env_naive_s (naive ISO timestamps read as UTC) = cleaned_clock + D, where
cleaned_clock = file-name unix + cleaned TIMESTAMP. D = -39600 means the environmental
clock equals the phone (raw accelerometer) unix clock; D = -14400 means it equals the PSG
wall clock (file-name unix rendered at UTC-4).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon

OUT = Path(sys.argv[1])
NIGHTS = Path(sys.argv[2])
SHIFTS = Path(sys.argv[3])
T0 = int(np.datetime64('2025-06-01T00:00:00', 's').astype(np.int64))
FREQ = np.arange(1, 61) / 10
BANDS = {'0.1-0.5': (0.1, 0.5), '2-6': (2.0, 6.0)}
SENSORS = ['1_3-33-0-L', '2_3-33-0-L', '4_3-33-0-L', '5_4-33-2-L', '6_4-33-2-L', '9_icu_ward', '11_icu_ward',
           '12_office', '13_rider_room', 'accel_data-002', 'accel_data-003', 'accel_data-007', 'accel_data-009']
ROOT4 = ['accel_data-002', 'accel_data-003', 'accel_data-007', 'accel_data-009']
D_PHONE, D_WALL = -39600, -14400
rng = np.random.default_rng(20261012)


def band(psd, lo, hi):
    take = (FREQ >= lo - 1e-9) & (FREQ <= hi + 1e-9)
    return psd[:, take, :3].sum(axis=2).mean(axis=1)


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


env = {}
floors = []
for s in SENSORS + ['15_rider_room']:
    d = np.load(OUT / 'spectra' / f'{s}_spectra.npz')
    a = json.loads((OUT / 'spectra' / f'{s}_audit.json').read_text())
    ok = d['epoch_frames'] >= 13
    e = {b: np.where(ok, band(d['epoch_psd'], *r), np.nan) for b, r in BANDS.items()}
    e['first'] = int(d['first_epoch_s']) + T0
    if s != '15_rider_room':
        env[s] = e
    fb = d['frame_band']
    row = dict(sensor=s, start=a['start'], end=a['end'], logged_h=round(a['logged_h'], 1),
               rate_within_logging_hz=round(a['rate_within_logging_hz'], 1),
               value_change_rate_hz=round(a['value_change_rate_within_logging_hz'], 1),
               counts_magnitude_median=round(a['counts_magnitude_median']), valid_frames=int(np.isfinite(fb[:, 2, 0]).sum()))
    for b, i in [('0.1-0.5', 0), ('0.5-2', 1), ('2-6', 2)]:
        v = fb[:, i, 0]; v = v[np.isfinite(v)]
        row[f'trace_{b}_p10_db'] = round(float(db(np.quantile(v, .1))), 2)
        row[f'trace_{b}_median_db'] = round(float(db(np.median(v))), 2)
    floors.append(row)
pd.DataFrame(floors).to_csv(OUT / 'sensor_floors.csv', index=False)

shifts = pd.read_csv(SHIFTS).set_index('name')['source_shift'].to_dict()
nights = {}
for p in sorted(NIGHTS.glob('Somerset_subject_*/native_axis_psd.npz')):
    name = p.parent.name; anchor = int(name.rsplit('_', 1)[1])
    d = np.load(p); psd = d['epoch_linear_psd'].astype(np.float64)
    n = len(psd); k = int(shifts[name]); truth = d['truth']
    corr = np.full(n, -1); corr[k:] = truth[:n - k] if k else truth
    nights[name] = dict(anchor=anchor, n=n, gap=d['gap'], still=d['still'], truth_raw=truth, truth_corr=corr,
                        participant=name.split('_')[2], **{b: band(psd, *r) for b, r in BANDS.items()})


def env_on(sensor, night, D, b):
    """Environmental band power on the wrist epochs of one night for offset D (s); NaN where absent."""
    e = env[sensor]; w = nights[night]
    idx = (w['anchor'] + 30 * np.arange(w['n']) + D - e['first']) / 30
    assert np.allclose(idx, np.round(idx))
    idx = np.round(idx).astype(int); out = np.full(w['n'], np.nan)
    ok = (idx >= 0) & (idx < len(e[b])); out[ok] = e[b][idx[ok]]
    return out


def composite(night, D, b, sensors=ROOT4):
    """Mean dB over the four continuously logging sensors; NaN unless all four are present."""
    v = np.stack([db(env_on(s, night, D, b)) for s in sensors])
    return np.where(np.all(np.isfinite(v), axis=0), v.mean(axis=0), np.nan)


def rho(x, y, min_n=60):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n:
        return np.nan, int(ok.sum())
    return float(spearmanr(x[ok], y[ok])[0]), int(ok.sum())


def masks(w, labels):
    cov = ~w['gap'] & np.isfinite(w['2-6']) & (w['2-6'] > 0)
    return dict(all=cov, still_sleep=cov & w['still'] & (w[labels] > 0), still_wake=cov & w['still'] & (w[labels] == 0))


# 1. Environment-vs-environment: clocks of different sensors agree at lag 0?
pairs = []
for i, a in enumerate(SENSORS):
    for b_ in SENSORS[i + 1:]:
        A, B = env[a], env[b_]
        off = (B['first'] - A['first']) // 30
        row = dict(a=a, b=b_)
        for lag in [-240, -20, -2, -1, 0, 1, 2, 20, 240]:
            ia = np.arange(len(A['0.1-0.5'])); ib = ia - off + lag
            ok = (ib >= 0) & (ib < len(B['0.1-0.5']))
            row[f'rho_lag{lag}'] = rho(db(A['0.1-0.5'][ia[ok]]), db(B['0.1-0.5'][ib[ok]]), 500)[0]
        pairs.append(row)
pd.DataFrame(pairs).to_csv(OUT / 'env_env_lag.csv', index=False)

# 2. Offset scan: environment composite vs wrist, 0.1-0.5 Hz and 2-6 Hz, all covered epochs.
scan = []
for D in list(range(-20 * 3600, 6 * 3600 + 1, 1800)):
    for b in BANDS:
        rs = [rho(composite(nm, D, b), db(w[b]) * np.where(masks(w, 'truth_corr')['all'], 1, np.nan))[0] for nm, w in nights.items()]
        scan.append(dict(D_s=D, band=b, mean_rho=np.nanmean(rs), nights=int(np.isfinite(rs).sum())))
scan = pd.DataFrame(scan); scan.to_csv(OUT / 'offset_scan_coarse.csv', index=False)
best = scan[scan.band == '0.1-0.5'].sort_values('mean_rho').iloc[-1]
fine = []; per_night = []
for center in [D_PHONE, D_WALL, int(best.D_s)]:
    for D in range(center - 1800, center + 1801, 30):
        for b in BANDS:
            rs = [rho(composite(nm, D, b), db(w[b]) * np.where(masks(w, 'truth_corr')['all'], 1, np.nan))[0] for nm, w in nights.items()]
            fine.append(dict(D_s=D, center=center, band=b, mean_rho=np.nanmean(rs), nights=int(np.isfinite(rs).sum())))
            if center == int(best.D_s) and b == '0.1-0.5':
                per_night += [dict(night=nm, D_s=D, rho=r) for nm, r in zip(nights, rs)]
fine = pd.DataFrame(fine); fine.to_csv(OUT / 'offset_scan_fine.csv', index=False)
per_night = pd.DataFrame(per_night)
night_best = per_night.loc[per_night.groupby('night').rho.idxmax()].rename(columns={'D_s': 'best_D_s', 'rho': 'best_rho'})
night_best.to_csv(OUT / 'offset_best_per_night.csv', index=False)
ff = fine[(fine.center == int(best.D_s)) & (fine.band == '0.1-0.5')]
D_SYNC = int(ff.sort_values('mean_rho').iloc[-1].D_s)
summary = dict(D_sync=D_SYNC, coarse_best=best.to_dict(), fine_best_mean_rho=float(ff.mean_rho.max()),
               per_night_best_D=night_best.best_D_s.describe().to_dict())

# 3. Overlap and synchronized vs displaced correlation.
rows = []; overlap = []
for nm, w in nights.items():
    for sensor in SENSORS:
        v = env_on(sensor, nm, D_SYNC, '2-6')
        overlap.append(dict(night=nm, sensor=sensor, epochs=w['n'], env_epochs=int(np.isfinite(v).sum())))
    for b in BANDS:
        sync = composite(nm, D_SYNC, b)
        controls = [D_SYNC + k * 86400 for k in (-3, -2, -1, 1, 2, 3)]
        controls += list(D_SYNC + 30 * rng.integers(2880, 4 * 2880, 20) * rng.choice([-1, 1], 20))
        for labels in ['truth_corr', 'truth_raw']:
            for g, m in masks(w, labels).items():
                if g == 'all' and labels == 'truth_raw':
                    continue
                y = np.where(m, db(w[b]), np.nan)
                r, nn = rho(sync, y, 30)
                cr = [rho(composite(nm, D, b), y, 30)[0] for D in controls]
                cr = np.array([c for c in cr if np.isfinite(c)])
                rows.append(dict(night=nm, participant=w['participant'], band=b, labels=labels, group=g, n=nn,
                                 rho_sync=r, controls=len(cr), rho_control_median=float(np.median(cr)) if len(cr) else np.nan,
                                 rho_control_p95=float(np.quantile(cr, .95)) if len(cr) else np.nan,
                                 control_ge_sync=int(np.sum(cr >= r)) if np.isfinite(r) else -1))
pd.DataFrame(overlap).to_csv(OUT / 'overlap.csv', index=False)
corr = pd.DataFrame(rows); corr.to_csv(OUT / 'sync_vs_displaced.csv', index=False)

# Per-sensor synchronized rho (all covered epochs) for the record.
per_sensor = []
for sensor in SENSORS:
    for b in BANDS:
        for D in [D_SYNC, D_SYNC + 86400, D_SYNC - 86400]:
            rs = []
            for nm, w in nights.items():
                y = np.where(masks(w, 'truth_corr')['all'], db(w[b]), np.nan)
                rs.append(rho(db(env_on(sensor, nm, D, b)), y)[0])
            per_sensor.append(dict(sensor=sensor, band=b, D_s=D, nights=int(np.isfinite(rs).sum()), mean_rho=np.nanmean(rs) if np.isfinite(rs).any() else np.nan))
pd.DataFrame(per_sensor).to_csv(OUT / 'per_sensor_sync.csv', index=False)

# 4. Environment 2-6 Hz on still-wake vs still-sleep wrist epochs; 5. wrist contrast in quiet vs active environment.
contrast = []
for labels in ['truth_corr', 'truth_raw']:
    for nm, w in nights.items():
        ms = masks(w, labels)
        e26 = composite(nm, D_SYNC, '2-6'); e01 = composite(nm, D_SYNC, '0.1-0.5')
        row = dict(labels=labels, night=nm, participant=w['participant'])
        for eb, ev in [('2-6', e26), ('0.1-0.5', e01)]:
            sw, ss = ev[ms['still_wake'] & np.isfinite(ev)], ev[ms['still_sleep'] & np.isfinite(ev)]
            row[f'n_wake'] = len(sw); row[f'n_sleep'] = len(ss)
            row[f'env{eb}_wake_minus_sleep_db'] = float(np.median(sw) - np.median(ss)) if len(sw) >= 5 and len(ss) >= 5 else np.nan
            row[f'env{eb}_auc_wake_gt_sleep'] = float(mannwhitneyu(sw, ss)[0] / (len(sw) * len(ss))) if len(sw) >= 5 and len(ss) >= 5 else np.nan
        still = (ms['still_wake'] | ms['still_sleep']) & np.isfinite(e26)
        if still.sum() >= 20:
            med = np.median(e26[still])
            for half, hm in [('quiet', e26 <= med), ('active', e26 > med)]:
                for wb in BANDS:
                    x = db(w[wb]); sw, ss = x[ms['still_wake'] & hm & still], x[ms['still_sleep'] & hm & still]
                    row[f'{half}_n_wake'] = len(sw); row[f'{half}_n_sleep'] = len(ss)
                    ok = len(sw) >= 5 and len(ss) >= 5
                    row[f'{half}_wrist{wb}_wake_minus_sleep_db'] = float(np.median(sw) - np.median(ss)) if ok else np.nan
                    row[f'{half}_wrist{wb}_auc'] = float(mannwhitneyu(sw, ss)[0] / (len(sw) * len(ss))) if ok else np.nan
        contrast.append(row)
contrast = pd.DataFrame(contrast); contrast.to_csv(OUT / 'wake_sleep_environment.csv', index=False)


def nightly(df, col):
    v = df[col].dropna()
    null = 0.5 if ('auc' in col) else 0.0
    p = float(wilcoxon(v - null)[1]) if len(v) >= 6 else np.nan
    return dict(nights=int(len(v)), mean=float(v.mean()), median=float(v.median()), wilcoxon_p_vs_null=p, null=null)


agg = {}
for labels in ['truth_corr', 'truth_raw']:
    c = contrast[contrast.labels == labels]
    agg[labels] = {col: nightly(c, col) for col in c.columns if col.endswith('_db') or col.endswith('auc') or col.endswith('gt_sleep')}
    for wb in BANDS:
        dd = (c[f'quiet_wrist{wb}_wake_minus_sleep_db'] - c[f'active_wrist{wb}_wake_minus_sleep_db']).dropna()
        agg[labels][f'quiet_minus_active_wrist{wb}_contrast_db'] = dict(nights=int(len(dd)), mean=float(dd.mean()), median=float(dd.median()),
                                                                     wilcoxon_p=float(wilcoxon(dd)[1]) if len(dd) >= 6 else np.nan)
        dd = (c[f'quiet_wrist{wb}_auc'] - c[f'active_wrist{wb}_auc']).dropna()
        agg[labels][f'quiet_minus_active_wrist{wb}_auc'] = dict(nights=int(len(dd)), mean=float(dd.mean()), median=float(dd.median()),
                                                             wilcoxon_p=float(wilcoxon(dd)[1]) if len(dd) >= 6 else np.nan)
cs = corr.groupby(['band', 'labels', 'group']).agg(nights=('rho_sync', lambda v: int(v.notna().sum())),
    epochs=('n', 'sum'), mean_rho_sync=('rho_sync', 'mean'), mean_rho_control=('rho_control_median', 'mean'),
    nights_sync_above_control_p95=('rho_sync', lambda v: 0)).reset_index()
for i, r in cs.iterrows():
    sub = corr[(corr.band == r.band) & (corr.labels == r.labels) & (corr.group == r.group) & corr.rho_sync.notna()]
    cs.loc[i, 'nights_sync_above_control_p95'] = int((sub.rho_sync > sub.rho_control_p95).sum())
    d = (sub.rho_sync - sub.rho_control_median).dropna()
    cs.loc[i, 'wilcoxon_p_sync_vs_control'] = float(wilcoxon(d)[1]) if len(d) >= 6 else np.nan
cs.to_csv(OUT / 'sync_vs_displaced_summary.csv', index=False)
summary['wake_sleep'] = agg

# 6. Levels: wrist minus environment on synchronized still-sleep epochs; slope of wrist dB on environment dB.
from scipy.stats import theilslopes
levels = []
for nm, w in nights.items():
    m = masks(w, 'truth_corr')['still_sleep']
    for b in BANDS:
        ev = composite(nm, D_SYNC, b); y = db(w[b]); ok = m & np.isfinite(ev)
        if ok.sum() < 30:
            continue
        sl = theilslopes(y[ok], ev[ok])
        levels.append(dict(night=nm, band=b, n=int(ok.sum()), wrist_median_db=float(np.median(y[ok])), env_median_db=float(np.median(ev[ok])),
                           wrist_minus_env_median_db=float(np.median(y[ok] - ev[ok])), theil_slope=float(sl[0]),
                           env_iqr_db=float(np.subtract(*np.quantile(ev[ok], [.75, .25]))), wrist_iqr_db=float(np.subtract(*np.quantile(y[ok], [.75, .25])))))
levels = pd.DataFrame(levels); levels.to_csv(OUT / 'levels_still_sleep.csv', index=False)
summary['levels_still_sleep'] = levels.groupby('band')[['n', 'wrist_median_db', 'env_median_db', 'wrist_minus_env_median_db', 'theil_slope', 'env_iqr_db', 'wrist_iqr_db']].median().to_dict('index')
(OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
print(json.dumps(summary, indent=1, default=float))
print(cs.to_string())
