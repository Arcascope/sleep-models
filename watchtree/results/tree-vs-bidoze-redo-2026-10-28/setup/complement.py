"""Section 10 redo: tree/BiDoze complementarity against BiDoze summed 0.5 (25 h and 8 h probability average), both trees, both CV cohorts.
Matched epochs: tree scope (kept nights, labelled, complete context) that the BiDoze model also scores. Tree = baseline arm, offsets decoding, seed-mean
probabilities; BiDoze four-stage call = summed 0.5 for sleep/wake and the fold stage offsets within sleep. No threshold, no hybrid."""
import time
from scipy.stats import pearsonr, spearmanr
from tree_common import *

OUTT = RR / 'tables'; OUTL = RR / 'local'
STAGES = {0: 'wake', 1: 'light', 2: 'deep', 3: 'REM'}
CAT = ['both_right', 'only_tree_right', 'only_bidoze_right', 'both_wrong']
cats = lambda tr, bz: np.select([tr & bz, tr & ~bz, ~tr & bz], CAT[:3], 'both_wrong')
BOOT_N = 5000; SEEDB = 29028


def boot(people, stat_fn, arr):
    df = pd.DataFrame(arr).assign(person=np.asarray(people)).groupby('person').sum().to_numpy()
    ix = np.random.default_rng(SEEDB).integers(0, len(df), (BOOT_N, len(df))); s = df[ix].sum(1)
    return np.percentile([stat_fn(x) for x in s], [2.5, 97.5])


tab2, acc, corr, tstc = [], [], [], []
for cohort, oofc in (('wearanize', 'Wearanizeplus'), ('sleepaccel', 'SleepAccel')):
    full = cohort_table(cohort)
    tz = np.load(RR / f'local/tree_{cohort}.npz'); assert (tz['subject'] == full.subject.to_numpy().astype(str)).all()
    scope = (full.kept & full.scored & full.ctx).to_numpy() & tz['covered']
    key = pd.DataFrame({'night': full.subject.to_numpy(), 'epoch': full.epoch.to_numpy(), 'row': np.arange(len(full))})[scope]
    for m in ('25h', '8h'):
        o = pd.read_parquet(OOF / f'{m}_{oofc}.parquet')
        o = o[o.scored].merge(key, on=['night', 'epoch'], how='inner')
        rw = o.row.to_numpy(); y = o.psg_label.to_numpy().astype(int)
        assert (y == full.y4.to_numpy()[rw]).all()
        bz = o.stage_call_offsets.to_numpy().astype(int); p0b = o.p_wake.to_numpy()
        persons = np.array([cv.person_of(cohort, s) for s in o.night])
        print(cohort, m, 'matched epochs', len(o), 'nights', o.night.nunique(), 'participants', len(set(persons)), flush=True)
        for cfg in CFG_LIST:
            pt = tz[f'prob|baseline|{cfg}'][rw]; assert not np.isnan(pt).any()
            tr = pt.argmax(1); p0t = pt[:, 0]
            for task in ('sleep_wake', 'four_stage'):
                if task == 'sleep_wake': ref, ct, cb = y > 0, tr > 0, bz > 0
                else: ref, ct, cb = y, tr, bz
                rt, rb = ct == ref, cb == ref; cat = cats(rt, rb)
                groups = [('all', np.ones(len(y), bool))] + ([('PSG sleep', y > 0), ('PSG wake', y == 0)] if task == 'sleep_wake' else [(f'PSG {n}', y == s) for s, n in STAGES.items()])
                for gname, gm in groups:
                    n = int(gm.sum()); row = dict(cohort=cohort, bidoze=m, tree=cfg, task=task, psg=gname, epochs=n)
                    for c in CAT:
                        row[c] = int(((cat == c) & gm).sum()); row[c + '_pct'] = 100 * row[c] / n
                    row['tree_acc_pct'] = 100 * rt[gm].mean(); row['bidoze_acc_pct'] = 100 * rb[gm].mean(); row['oracle_acc_pct'] = 100 * (rt | rb)[gm].mean()
                    tab2.append(row)
                n = len(y); a_t, a_b, a_o = rt.mean(), rb.mean(), (rt | rb).mean()
                only_t = int((cat == 'only_tree_right').sum()); bz_err = int((~rb).sum()); only_b = int((cat == 'only_bidoze_right').sum()); tr_err = int((~rt).sum())
                arr = np.c_[rt, rb, rt | rb, rt & ~rb, ~rb, rb & ~rt, ~rt, np.ones(n)]
                gl, gh = boot(persons, lambda s: (s[2] - max(s[0], s[1])) / s[7], arr)
                sl, sh = boot(persons, lambda s: s[3] / s[4], arr)
                bl, bh = boot(persons, lambda s: s[5] / s[6], arr)
                acc.append(dict(cohort=cohort, bidoze=m, tree=cfg, task=task, epochs=n, participants=len(set(persons)), nights=o.night.nunique(),
                                acc_tree=100 * a_t, acc_bidoze=100 * a_b, acc_oracle=100 * a_o, oracle_minus_better_pts=100 * (a_o - max(a_t, a_b)), oracle_ci_lo=100 * gl, oracle_ci_hi=100 * gh,
                                oracle_minus_bidoze_pts=100 * (a_o - a_b), oracle_minus_tree_pts=100 * (a_o - a_t), bidoze_errors=bz_err, tree_errors=tr_err,
                                only_tree_right=only_t, only_tree_right_share_of_bidoze_errors_pct=100 * only_t / bz_err, share_ci_lo=100 * sl, share_ci_hi=100 * sh,
                                only_bidoze_right=only_b, only_bidoze_right_share_of_tree_errors_pct=100 * only_b / tr_err, only_bidoze_share_ci_lo=100 * bl, only_bidoze_share_ci_hi=100 * bh,
                                only_tree_right_pct_of_epochs=100 * only_t / n, only_bidoze_right_pct_of_epochs=100 * only_b / n))
            for scope_name, mk in (('all matched epochs', np.ones(len(y), bool)), ('PSG sleep epochs', y > 0), ('PSG wake epochs', y == 0)):
                corr.append(dict(cohort=cohort, bidoze=m, tree=cfg, scope=scope_name, epochs=int(mk.sum()), pearson_p_wake=pearsonr(p0b[mk], p0t[mk])[0], spearman_p_wake=spearmanr(p0b[mk], p0t[mk])[0]))
            d = pd.DataFrame({'night': o.night.to_numpy(), 'ps_t': tr > 0, 'ps_b': bz > 0, 'ref': y > 0})
            pn = d.groupby('night').agg(n=('ref', 'size'), t=('ps_t', 'sum'), b=('ps_b', 'sum'), r=('ref', 'sum'))
            et, eb = (pn.t - pn.r) * .5, (pn.b - pn.r) * .5
            pw = pd.DataFrame({'night': o.night.to_numpy(), 'pt': p0t, 'pb': p0b}).groupby('night').mean()
            tstc.append(dict(cohort=cohort, bidoze=m, tree=cfg, nights=len(pn), tst_err_tree_mean=et.mean(), tst_err_bidoze_mean=eb.mean(), pearson_tst_error=pearsonr(et, eb)[0],
                             spearman_tst_error=spearmanr(et, eb)[0], mae_tree=et.abs().mean(), mae_bidoze=eb.abs().mean(), mean_of_two_errors_mae=((et + eb) / 2).abs().mean()))
pd.DataFrame(tab2).to_csv(OUTT / 'complement_2x2.csv', index=False)
A = pd.DataFrame(acc); A.to_csv(OUTT / 'complement_oracle.csv', index=False)
pd.DataFrame(corr).to_csv(OUTT / 'complement_p_wake_correlation.csv', index=False)
pd.DataFrame(tstc).to_csv(OUTT / 'complement_tst_error_correlation.csv', index=False)
pd.set_option('display.width', 250, 'display.max_columns', 30)
print(A[['cohort', 'bidoze', 'tree', 'task', 'acc_tree', 'acc_bidoze', 'acc_oracle', 'oracle_minus_better_pts', 'oracle_ci_lo', 'oracle_ci_hi', 'only_tree_right_share_of_bidoze_errors_pct']].round(2).to_string(index=False))
print('COMPLEMENT_DONE', flush=True)
