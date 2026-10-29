"""Sections 1-3 redo: 26 complete-context Wearanize+ nights (all 92 secondary), same masks as tst-investigation-2026-10-28.
BiDoze: 25 h and 8 h, PSG crop and surrounding-data input, argmax (reference) and summed 0.5; 8 h windows blended by probability average.
Trees: HR (87 columns) and accelerometer-only (62 columns), offsets and native decoding, 3 seeds. Nothing is fitted."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, '/home/eric/Engineering/Work/autopisces/sleep-models-uncropped-loaders/bidoze/results/blend-exp2a-2026-10-28/setup')
from tree_common import *
import blend_exp2a as B
from stats_common import *
from arcasleep.divisor import hybrid_calls

OUTL = RR / 'local'; OUTT = RR / 'tables'
OFFS = pd.read_csv(B.RUN / 'tables/stage_offsets.csv')
t0 = time.time()


def offs(family, k):
    r = OFFS[(OFFS.family == family) & (OFFS.fold == k)].iloc[0]
    return np.array([0.0, r.offset_deep, r.offset_rev])


def sleep_summed(P): return P[:, 1:4].sum(1) >= P[:, 0]


def sum_calls(P, off):
    sl = sleep_summed(P)
    sc = np.log(np.maximum(P[:, 1:4], 1e-12)) + off
    return np.where(sl, 1 + sc.argmax(1), 0).astype(np.int8)


full = cohort_table('wearanize')
tz = np.load(RR / 'local/tree_wearanize.npz')
assert (tz['subject'] == full.subject.to_numpy().astype(str)).all() and (tz['epoch'] == full.epoch.to_numpy()).all()
prior = pd.read_csv(TSTI / 'per_night.csv')
primary = set(prior[(prior.scope == 'primary26') & (prior.model == '8h_context_original')].subject); assert len(primary) == 26
fold_of = {}
for p in (TSTI / 'probabilities').glob('test_f*_wearanize_*_25h_psg.npz'):
    k = int(p.name.split('_')[1][1:]); sid = p.name[len(f'test_f{k}_wearanize_'):-len('_25h_psg.npz')]; fold_of[sid] = k
assert len(fold_of) == 92
rows_ix = full.groupby('subject').indices
MODELS_BZ = ['bz25_crop_orig', 'bz25_ctx_orig', 'bz8_crop_orig', 'bz8_ctx_orig', 'bz25_crop_sum', 'bz25_ctx_sum', 'bz8_crop_sum', 'bz8_ctx_sum']
TREE_M = [f'tree_{c}_{d}' for c in CFG_LIST for d in ('off', 'nat')]
per_night, diag = [], []
div = {k: np.array(json.load(open(TSTI / f'divisor_f{k}.json'))) for k in range(5)}
checks = []
for sid in sorted(fold_of):
    k = fold_of[sid]; g = rows_ix[sid]; N = len(g)
    y = full.y4.to_numpy()[g].astype(np.int64); psg = full.in_psg.to_numpy()[g]; ctx = full.ctx.to_numpy()[g]
    P = {}; gaps = []; calls = {}
    def put(name, p5, off, gap):
        full5 = np.full((N, 5), np.nan); full5[off:off + len(p5)] = p5; P[name] = full5
        gp = np.ones(N, bool); gp[off:off + len(p5)] = gap; gaps.append(gp)
    # 25 h and 8 h stored frozen outputs (logit-mean 8 h): the original argmax rows and the 25 h summed rows
    for arch in ('25h', '8h'):
        for arm in ('psg', 'context'):
            with np.load(TSTI / f'probabilities/test_f{k}_wearanize_{sid}_{arch}_{arm}.npz') as z:
                p = z['probs'].mean(0).astype(np.float64); off = int(z['offset']); assert np.array_equal(y[off:off + len(z['y'])], z['y']); gap = z['gap']
            put(f'{arch}_{arm}_stored', p, off, gap)
    # 8 h probability average, whole recording and PSG crop
    rec = B.blend_rec(B.load_rec(B.RUN / f'windows/test_f{k}_wearanize_{sid}.npz'), grids=('A',)); assert N - rec['n'] in (0, 1)
    pa = np.full((N, 5), np.nan); pa[:rec['n']] = rec['P'][('A', 'prob_average')]; P['8h_context_pa'] = pa
    rc = B.blend_rec(B.load_rec(RR / f'local/windows_8h_crop/test_f{k}_wearanize_{sid}.npz'), grids=('A',))
    off_c = int(np.load(RR / f'local/windows_8h_crop/test_f{k}_wearanize_{sid}.npz')['offset']); assert off_c + rc['n'] <= N and abs(psg.sum() - rc['n']) <= 1, (sid, psg.sum(), rc['n'], off_c)
    pc = np.full((N, 5), np.nan); pc[off_c:off_c + rc['n']] = rc['P'][('A', 'prob_average')]; P['8h_crop_pa'] = pc
    gp = np.zeros(N, bool)
    common = psg & (y >= 0) & ~np.logical_or.reduce(gaps)
    assert common.sum() > 0
    def argmax_calls(p5): return np.where(np.isnan(p5[:, 0]), 4, np.nan_to_num(p5).argmax(1)).astype(np.int8)
    def hyb(p5):
        c = np.full(N, 4, np.int8); v = ~np.isnan(p5[:, 0]); c[v] = hybrid_calls(p5[v], p5[v], div[k]); return c
    def summed(p5, fam):
        c = np.full(N, 4, np.int8); v = ~np.isnan(p5[:, 0]); c[v] = sum_calls(p5[v], offs(fam, k)); return c
    calls['bz25_crop_orig'] = hyb(P['25h_psg_stored']); calls['bz25_ctx_orig'] = hyb(P['25h_context_stored'])
    calls['bz8_crop_orig'] = argmax_calls(P['8h_psg_stored']); calls['bz8_ctx_orig'] = argmax_calls(P['8h_context_stored'])
    calls['bz25_crop_sum'] = summed(P['25h_psg_stored'], '25h'); calls['bz25_ctx_sum'] = summed(P['25h_context_stored'], '25h')
    calls['bz8_crop_sum'] = summed(P['8h_crop_pa'], '8h'); calls['bz8_ctx_sum'] = summed(P['8h_context_pa'], '8h')
    # trees
    tree_calls = {}
    for cfg in CFG_LIST:
        for dec in ('off', 'nat'):
            tree_calls[f'tree_{cfg}_{dec}'] = tz[f'{dec}|baseline|{cfg}'][:, g].astype(np.int8)  # (3, N)
    scopes = [('primary26', common)] if sid in primary else []
    scopes += [('all92', common), ('all92_complete_context', common & ctx)]
    for scope, mask in scopes:
        if mask.sum() == 0: continue
        for m in MODELS_BZ:
            per_night.append(dict(subject=sid, person=sid.split('_s')[0], fold=k, model=m, scope=scope, **measure(y, calls[m], mask)))
        for m in TREE_M:
            rs = pd.DataFrame([measure(y, tree_calls[m][s], mask) for s in range(3)]).mean()
            per_night.append(dict(subject=sid, person=sid.split('_s')[0], fold=k, model=m, scope=scope, **rs.to_dict()))
    # epoch-level diagnostics (cropped BiDoze, as section 3 did) for primary26 and all92
    ok = common
    d = pd.DataFrame({'subject': sid, 'person': sid.split('_s')[0], 'primary26': sid in primary, 'epoch': np.flatnonzero(ok), 'y': y[ok],
                      'jerk': full.jerk.to_numpy()[g][ok], 'sib_b15': full.sib_mean_b15.to_numpy()[g][ok], 'sib_f15': full.sib_mean_f15.to_numpy()[g][ok]})
    for m in ('bz25_crop_orig', 'bz25_crop_sum', 'bz8_crop_sum', 'bz8_ctx_sum'):
        d[m] = calls[m][ok]
    for m in ('tree_hr_off', 'tree_nohr_off'):
        d[m] = (tree_calls[m][:, ok] > 0).mean(0)       # share of the 3 seeds calling sleep
    diag.append(d)
    if len(per_night) % 200 < 25: print(len(per_night), sid, f'{time.time()-t0:.0f}s', flush=True)
PN = pd.DataFrame(per_night); PN.to_csv(OUTL / 'tst123_per_night.csv', index=False)
D = pd.concat(diag, ignore_index=True); D.to_parquet(OUTL / 'tst123_epochs.parquet', index=False)
print('per-night rows', len(PN), 'diag epochs', len(D), f'{time.time()-t0:.0f}s', flush=True)

# ----------------------------------------------------------------------------- summary
LABEL = {'bz25_crop_orig': '25 h, PSG crop, argmax', 'bz25_ctx_orig': '25 h, surrounding data, argmax', 'bz8_crop_orig': '8 h, PSG crop, argmax (logit mean)',
         'bz8_ctx_orig': '8 h, surrounding data, argmax (logit mean)', 'bz25_crop_sum': '25 h, PSG crop, summed 0.5', 'bz25_ctx_sum': '25 h, surrounding data, summed 0.5',
         'bz8_crop_sum': '8 h, PSG crop, summed 0.5 (probability average)', 'bz8_ctx_sum': '8 h, surrounding data, summed 0.5 (probability average)',
         'tree_hr_off': 'HR tree, offsets', 'tree_hr_nat': 'HR tree, no offsets', 'tree_nohr_off': 'Accelerometer-only tree, offsets', 'tree_nohr_nat': 'Accelerometer-only tree, no offsets'}
S = []
for (scope, m), g in PN.groupby(['scope', 'model']):
    r = dict(scope=scope, model=m, label=LABEL[m], nights=len(g), participants=g.person.nunique(), epochs=int(g.n_epochs.sum()))
    for c, nm in (('tst_error', 'tst'), ('waso_error', 'waso')):
        v = g[c].dropna(); r[f'{nm}_bias'] = v.mean(); r[f'{nm}_mae'] = v.abs().mean()
        r[f'{nm}_bias_ci_lo'], r[f'{nm}_bias_ci_hi'] = cluster_ci(g.loc[v.index, 'person'], v)
        r[f'{nm}_mae_ci_lo'], r[f'{nm}_mae_ci_hi'] = cluster_ci(g.loc[v.index, 'person'], v.abs())
    for c in ('sensitivity', 'specificity', 'kappa_sw', 'kappa4', 'false_sleep_min', 'false_wake_min', 'within_false_sleep', 'within_false_wake', 'before_false_wake', 'after_false_wake'):
        r[c] = g[c].mean()
    S.append(r)
S = pd.DataFrame(S); S.to_csv(OUTT / 'tst123_summary.csv', index=False)

# ----------------------------------------------------------------------------- paired differences
Pr = []
def add(scope, name, a, b, metric, absolute=False, extra=''):
    g = PN[PN.scope == scope]
    r = paired(g, a, b, metric, absolute)
    if r: Pr.append(dict(scope=scope, comparison=name, a=a, b=b, metric=metric + ('_abs' if absolute else ''), note=extra, **r))
for scope in ('primary26', 'all92', 'all92_complete_context'):
    for arch, inp in (('bz25', 'crop'), ('bz25', 'ctx'), ('bz8', 'crop'), ('bz8', 'ctx')):
        for metric, ab in (('tst_error', True), ('tst_error', False), ('kappa_sw', False), ('specificity', False), ('sensitivity', False)):
            add(scope, 'summed 0.5 minus argmax', f'{arch}_{inp}_sum', f'{arch}_{inp}_orig', metric, ab)
    for arch, dec in (('bz25', 'orig'), ('bz25', 'sum'), ('bz8', 'orig'), ('bz8', 'sum')):
        add(scope, 'surrounding data minus PSG crop', f'{arch}_ctx_{dec}', f'{arch}_crop_{dec}', 'tst_error', False)
        add(scope, 'surrounding data minus PSG crop', f'{arch}_ctx_{dec}', f'{arch}_crop_{dec}', 'kappa4')
        add(scope, 'surrounding data minus PSG crop', f'{arch}_ctx_{dec}', f'{arch}_crop_{dec}', 'kappa_sw')
    for t in ('tree_hr_off', 'tree_nohr_off'):
        for b in ('bz25_crop_sum', 'bz8_ctx_sum', 'bz8_crop_sum', 'bz25_crop_orig'):
            for metric, ab in (('tst_error', True), ('waso_error', True), ('kappa_sw', False), ('kappa4', False), ('specificity', False), ('sensitivity', False)):
                add(scope, 'tree minus BiDoze', t, b, metric, ab)
Pr = pd.DataFrame(Pr); Pr.to_csv(OUTT / 'tst123_paired.csv', index=False)

# context sensitivity: mean |TST change| from adding surrounding data
C = []
for scope in ('primary26', 'all92'):
    for arch in ('bz25', 'bz8'):
        for dec in ('orig', 'sum'):
            g = PN[PN.scope == scope]
            a = g[g.model == f'{arch}_crop_{dec}'].set_index('subject'); b = g[g.model == f'{arch}_ctx_{dec}'].set_index('subject').loc[a.index]
            dl = b.tst_error - a.tst_error
            lo, hi = cluster_ci(a.person, dl); alo, ahi = cluster_ci(a.person, dl.abs())
            C.append(dict(scope=scope, arch=arch, decision=dec, nights=len(dl), mean_tst_change=dl.mean(), ci_lo=lo, ci_hi=hi, mean_abs_tst_change=dl.abs().mean(), abs_ci_lo=alo, abs_ci_hi=ahi,
                          kappa4_change=(b.kappa4 - a.kappa4).mean(), kappa_sw_change=(b.kappa_sw - a.kappa_sw).mean(),
                          fw_crop=a.false_wake_min.mean(), fw_ctx=b.false_wake_min.mean(), fs_crop=a.false_sleep_min.mean(), fs_ctx=b.false_sleep_min.mean()))
pd.DataFrame(C).to_csv(OUTT / 'tst123_context_changes.csv', index=False)

# ----------------------------------------------------------------------------- section 3 diagnostics
def pooled_ci(person, num, den, seed=29028):
    df = pd.DataFrame({'p': person, 'n': num, 'd': den}).groupby('p').sum().to_numpy()
    ix = np.random.default_rng(seed).integers(0, len(df), (DRAWS, len(df))); s = df[ix].sum(1)
    return np.percentile(s[:, 0] / s[:, 1], [2.5, 97.5])

fwrows, qrows, stagerows, runrows, treerows = [], [], [], [], []
for scope, sub in (('primary26', D[D.primary26]), ('all92', D)):
    for m in ('bz25_crop_orig', 'bz25_crop_sum', 'bz8_crop_sum'):
        run_min = {'<= 2 min': [], '2-10 min': [], '> 10 min': []}
        nsub = sub.subject.nunique(); hi_fw, hi_sl, quiet_fw, quiet_sl, nfw = [], [], [], [], []
        qcount = np.zeros((4, 3)); stg = np.zeros((4, 2))
        for sid, g in sub.groupby('subject'):
            g = g[g[m] < 4]
            sleep = g.y > 0; fw = sleep & (g[m] == 0); fs = (g.y == 0) & (g[m] > 0)
            qs = g.jerk[sleep].quantile([.25, .5, .75]).to_numpy()
            quart = np.digitize(g.jerk.to_numpy(), qs)            # 0..3, quartile of the night's PSG-sleep jerk
            hi = quart == 3; quiet = ((g.sib_b15 + g.sib_f15) / 2 >= .9).to_numpy()
            if fw.any():
                hi_fw.append(hi[fw.to_numpy()].mean()); quiet_fw.append(quiet[fw.to_numpy()].mean())
            hi_sl.append(hi[sleep.to_numpy()].mean()); quiet_sl.append(quiet[sleep.to_numpy()].mean())
            for q in range(4):
                qcount[q] += [np.sum((quart == q) & fw.to_numpy()), np.sum((quart == q) & sleep.to_numpy()), np.sum(quart == q)]
            for st in (1, 2, 3):
                stg[st] += [np.sum((g.y == st) & (g[m] == 0)), np.sum(g.y == st)]
            idx = g.epoch.to_numpy()[fw.to_numpy()]
            lens = []
            if len(idx):
                starts = np.r_[0, np.flatnonzero(np.diff(idx) > 1) + 1]; ends = np.r_[starts[1:], len(idx)]
                lens = (ends - starts) * STEP
            run_min['<= 2 min'].append(np.sum([l for l in lens if l <= 2])); run_min['2-10 min'].append(np.sum([l for l in lens if 2 < l <= 10]))
            run_min['> 10 min'].append(np.sum([l for l in lens if l > 10]))
        fwrows.append(dict(scope=scope, model=m, label=LABEL[m], nights=nsub, nights_with_false_wake=len(hi_fw), fw_in_top_jerk_quartile_pct=100 * np.mean(hi_fw),
                           sleep_in_top_jerk_quartile_pct=100 * np.mean(hi_sl), fw_quiet_surround_pct=100 * np.mean(quiet_fw), sleep_quiet_surround_pct=100 * np.mean(quiet_sl)))
        for q in range(4):
            qrows.append(dict(scope=scope, model=m, jerk_quartile=q + 1, false_wake_epochs=int(qcount[q, 0]), psg_sleep_epochs=int(qcount[q, 1]),
                              share_of_false_wake_pct=100 * qcount[q, 0] / qcount[:, 0].sum(), false_wake_rate_pct=100 * qcount[q, 0] / qcount[q, 1]))
        for st, nm in ((1, 'light'), (2, 'deep'), (3, 'REM')):
            stagerows.append(dict(scope=scope, model=m, psg_stage=nm, epochs=int(stg[st, 1]), false_wake_epochs=int(stg[st, 0]), false_wake_rate_pct=100 * stg[st, 0] / stg[st, 1]))
        for b, v in run_min.items():
            runrows.append(dict(scope=scope, model=m, run_length=b, false_wake_min_per_night=np.sum(v) / nsub))
        # share of this BiDoze model's false wake that each tree calls sleep (seed-mean)
        fwm = (sub.y > 0) & (sub[m] == 0)
        for t in ('tree_hr_off', 'tree_nohr_off'):
            sh = sub.loc[fwm, t]; lo, hi = pooled_ci(sub.loc[fwm, 'person'], sh, np.ones(len(sh)))
            maj = (sh >= 2 / 3 - 1e-9).mean()
            treerows.append(dict(scope=scope, bidoze_model=m, tree=t, false_wake_epochs=int(fwm.sum()), tree_calls_sleep_pct=100 * sh.mean(), ci_lo=100 * lo, ci_hi=100 * hi,
                                 majority_of_seeds_pct=100 * maj))
pd.DataFrame(fwrows).to_csv(OUTT / 'tst3_movement.csv', index=False)
pd.DataFrame(qrows).to_csv(OUTT / 'tst3_jerk_quartiles.csv', index=False)
pd.DataFrame(stagerows).to_csv(OUTT / 'tst3_stage_false_wake.csv', index=False)
pd.DataFrame(runrows).to_csv(OUTT / 'tst3_run_length.csv', index=False)
pd.DataFrame(treerows).to_csv(OUTT / 'tst3_tree_recovery.csv', index=False)
pd.set_option('display.width', 250, 'display.max_columns', 30)
print(S[S.scope == 'primary26'][['label', 'nights', 'tst_bias', 'tst_mae', 'waso_bias', 'waso_mae', 'sensitivity', 'specificity', 'kappa_sw', 'kappa4']].round(3).to_string())
print('TST123_DONE', flush=True)
