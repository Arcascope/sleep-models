"""Sections 9 and 11 redo: single-group and combined-group ablations of both tree offerings (HR 87 columns, accelerometer-only 62) on
Wearanize+ and SleepAccel. Fitted models of ablation-exp3b are reused unchanged (the arms are unchanged); predictions come from
build_tree_preds.py. Recovery set = epochs BiDoze summed 0.5 calls wake inside PSG sleep (25 h; the 8 h probability-average set too)."""
import json, time
from scipy import stats
from tree_common import *

OUTT = RR / 'tables'; OUTL = RR / 'local'
t0 = time.time()
OOFD = {(m, c): pd.read_parquet(OOF / f'{m}_{c}.parquet') for m in ('25h', '8h') for c in ('Wearanizeplus', 'SleepAccel')}
STEPm = C.EPOCH_S / 60
METRICS = ['kappa_sw', 'kappa4', 'sens', 'spec', 'tst_bias', 'tst_mae']
GROUP_LABEL = {'baseline': 'baseline', 'no_jerk_lfm': 'no jerk/LFM', 'no_angle': 'no angle', 'no_inactivity': 'no inactivity', 'no_activity': 'no activity',
               'no_bout_timing': 'no bout timing', 'no_jerk_derived': 'no jerk-derived', 'no_movement_engineered': 'no engineered movement'}


def cluster_boot(num, den, draws, seed):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(seed).integers(0, len(num), (draws, len(num)))
    return num.sum() / den.sum(), *np.percentile(num[idx].sum(1) / den[idx].sum(1), [2.5, 97.5])


def night_rows(y, pred, subj):
    rows = []
    for s, ix in pd.Series(np.arange(len(y))).groupby(subj).indices.items():
        yy, pp = y[ix], pred[ix]; sleep, wake = yy > 0, yy == 0
        rows.append({'subject': s, 'kappa_sw': cv._kappa(yy > 0, pp > 0), 'kappa4': cv._kappa(yy, pp),
                     'sens': np.mean(pp[sleep] > 0) if sleep.any() else np.nan, 'spec': np.mean(pp[wake] == 0) if wake.any() else np.nan,
                     'tst_bias': ((pp > 0).sum() - sleep.sum()) * STEPm})
    return pd.DataFrame(rows)


per_night, rec_rows, set_rows, ncols = [], [], [], {}
for cohort, oofc in (('wearanize', 'Wearanizeplus'), ('sleepaccel', 'SleepAccel')):
    full = cohort_table(cohort)
    tz = np.load(RR / f'local/tree_{cohort}.npz'); assert (tz['subject'] == full.subject.to_numpy().astype(str)).all() and (tz['epoch'] == full.epoch.to_numpy()).all()
    scope = (full.kept & full.scored & full.ctx).to_numpy() & tz['covered']
    rows = np.flatnonzero(scope)
    tab = full.iloc[rows].reset_index(drop=True)
    y = tab.y4.to_numpy(); subj = tab.subject.to_numpy()
    persons = {s: cv.person_of(cohort, s) for s in set(subj)}
    print(cohort, 'scope nights', len(persons), 'epochs', len(tab), flush=True)
    # BiDoze summed 0.5 false-wake sets (25 h and 8 h) and BiDoze reference metrics on the tree scope
    key = pd.DataFrame({'night': tab.subject, 'epoch': tab.epoch, 'ix': np.arange(len(tab))})
    fwsets = {}
    for m in ('25h', '8h'):
        o = OOFD[(m, oofc)]
        fw = o[o.scored & (o.psg_label > 0) & ~o.sleep_call_summed05][['night', 'epoch']]
        mm = fw.merge(key, on=['night', 'epoch'], how='left')
        fwsets[m] = mm.ix.dropna().astype(int).to_numpy()
        set_rows.append(dict(cohort=cohort, bidoze=m, false_wake_epochs=len(fw), nights=fw.night.nunique(), in_tree_scope=len(fwsets[m]),
                             nights_in_scope=tab.subject.iloc[fwsets[m]].nunique()))
        # reference metrics on the tree scope epochs that this BiDoze model scores
        j = key.merge(o[['night', 'epoch', 'scored', 'sleep_call_summed05', 'stage_call_offsets']], on=['night', 'epoch'], how='inner')
        j = j[j.scored]
        pb = j.stage_call_offsets.to_numpy().astype(int)
        nr = night_rows(y[j.ix.to_numpy()], pb, subj[j.ix.to_numpy()])
        nr['tst_mae'] = nr.tst_bias.abs(); nr['person'] = nr.subject.map(persons)
        for dec in ('offset', 'native'):
            q = nr.copy(); q.insert(0, 'decoding', 'summed05'); q.insert(0, 'cfg', 'bidoze'); q.insert(0, 'arm', f'bidoze_{m}_summed05'); q.insert(0, 'cohort', cohort)
            per_night.append(q); break
    for arm in ARMS:
        for cfg in CFG_LIST:
            for dec in ('offset', 'native'):
                a = tz[f'{"off" if dec == "offset" else "nat"}|{arm}|{cfg}'][:, rows].astype(int)     # (3, n)
                parts = [night_rows(y, a[s], subj).assign(seed=s) for s in range(3)]
                nm = pd.concat(parts).drop(columns='seed').groupby('subject', as_index=False).mean(); nm['tst_mae'] = nm.tst_bias.abs()
                nm['person'] = nm.subject.map(persons)
                nm.insert(0, 'decoding', dec); nm.insert(0, 'cfg', cfg); nm.insert(0, 'arm', arm); nm.insert(0, 'cohort', cohort)
                per_night.append(nm)
                hit = (a > 0).mean(0)
                for m, ix in fwsets.items():
                    sub = pd.DataFrame({'subject': subj[ix], 'called_sleep': hit[ix]})
                    pn = sub.groupby('subject').called_sleep.agg(['sum', 'size']).reset_index(); pn['person'] = pn.subject.map(persons)
                    pn.insert(0, 'decoding', dec); pn.insert(0, 'cfg', cfg); pn.insert(0, 'arm', arm); pn.insert(0, 'set', f'bidoze_{m}_summed05'); pn.insert(0, 'cohort', cohort)
                    rec_rows.append(pn)
    for arm in ARMS:
        for cfg in CFG_LIST:
            ncols[(arm, cfg)] = len(json.load(open(model_path('fold0', arm, cfg, 0, 'json')))['features'])
    print(cohort, 'done', f'{time.time()-t0:.0f}s', flush=True)
pn = pd.concat(per_night, ignore_index=True); pn.to_csv(OUTL / 'ablation_per_night.csv', index=False)
rec = pd.concat(rec_rows, ignore_index=True)
pd.DataFrame(set_rows).to_csv(OUTT / 'ablation_false_wake_sets.csv', index=False)

summary, paired = [], []
for (cohort, cfg, dec), g in pn.groupby(['cohort', 'cfg', 'decoding']):
    base = g[g.arm == 'baseline'].set_index('subject')
    for arm, ga in g.groupby('arm'):
        ga = ga.set_index('subject')
        row = {'cohort': cohort, 'cfg': cfg, 'decoding': dec, 'arm': arm, 'columns': ncols.get((arm, cfg), np.nan), 'n_nights': len(ga), 'n_participants': ga.person.nunique()}
        for m in METRICS:
            s = ga[[m, 'person']].dropna(); ps = s.groupby('person')[m].agg(['sum', 'size'])
            row[m], row[m + '_lo'], row[m + '_hi'] = cluster_boot(ps['sum'], ps['size'], BOOT_DRAWS, BOOT_SEED)
        summary.append(row)
        if arm == 'baseline' or arm.startswith('bidoze') or base.empty: continue
        for m in METRICS:
            both = ga.index.intersection(base.index); diff = (ga.loc[both, m] - base.loc[both, m]).dropna()
            if diff.empty: continue
            ps = pd.DataFrame({'d': diff, 'person': ga.loc[diff.index, 'person']}).groupby('person').d.agg(['sum', 'size'])
            pt, lo, hi = cluster_boot(ps['sum'], ps['size'], BOOT_DRAWS, BOOT_SEED)
            p = stats.wilcoxon(diff).pvalue if len(diff) > 5 and (diff != 0).any() else np.nan
            paired.append({'cohort': cohort, 'cfg': cfg, 'decoding': dec, 'arm': arm, 'columns': ncols.get((arm, cfg), np.nan), 'metric': m, 'n_nights': len(diff), 'baseline': base.loc[diff.index, m].mean(),
                           'arm_mean': ga.loc[diff.index, m].mean(), 'diff': pt, 'ci_lo': lo, 'ci_hi': hi, 'wilcoxon_p': p})
SM = pd.DataFrame(summary); SM.to_csv(OUTT / 'ablation_summary.csv', index=False)
pd.DataFrame(paired).to_csv(OUTT / 'ablation_paired_vs_baseline.csv', index=False)
rec_out = []
for (cohort, setname, cfg, dec), g in rec.groupby(['cohort', 'set', 'cfg', 'decoding']):
    bp = g[g.arm == 'baseline'].groupby('person')[['sum', 'size']].sum()
    for arm, ga in g.groupby('arm'):
        ps = ga.groupby('person')[['sum', 'size']].sum()
        pt, lo, hi = cluster_boot(ps['sum'], ps['size'], BOOT_DRAWS, BOOT_SEED)
        row = {'cohort': cohort, 'set': setname, 'cfg': cfg, 'decoding': dec, 'arm': arm, 'columns': ncols.get((arm, cfg), np.nan), 'n_epochs': int(ps['size'].sum()), 'n_persons': len(ps),
               'frac_called_sleep': pt, 'ci_lo': lo, 'ci_hi': hi}
        if arm != 'baseline':
            both = ps.index.intersection(bp.index); n_, d_, nb_ = ps.loc[both, 'sum'].to_numpy(), ps.loc[both, 'size'].to_numpy(), bp.loc[both, 'sum'].to_numpy()
            idx = np.random.default_rng(BOOT_SEED).integers(0, len(both), (BOOT_DRAWS, len(both)))
            v = (n_[idx].sum(1) - nb_[idx].sum(1)) / d_[idx].sum(1)
            row['diff_vs_baseline'] = (n_.sum() - nb_.sum()) / d_.sum(); row['diff_lo'], row['diff_hi'] = np.percentile(v, [2.5, 97.5])
        rec_out.append(row)
pd.DataFrame(rec_out).to_csv(OUTT / 'ablation_recovery.csv', index=False)

# baseline check against the original run
old = pd.concat([pd.read_csv(ABL / 'summary.csv'), pd.read_csv(ABL / 'summary_combined.csv')]).drop_duplicates(['cohort', 'cfg', 'decoding', 'arm'])
m = SM.merge(old, on=['cohort', 'cfg', 'decoding', 'arm'], suffixes=('', '_old'))
chk = m.groupby(['cohort', 'cfg']).apply(lambda d: pd.Series({c: float(np.abs(d[c] - d[c + '_old']).max()) for c in METRICS}), include_groups=False)
chk.to_csv(OUTT / 'ablation_reproduction_check.csv'); print(chk.to_string())
pd.set_option('display.width', 250, 'display.max_columns', 30)
print(SM[(SM.decoding == 'offset') & (SM.arm == 'baseline')][['cohort', 'cfg', 'n_nights', 'kappa_sw', 'kappa4', 'spec', 'tst_mae']].round(3).to_string())
print('ABLATION_DONE', flush=True)
