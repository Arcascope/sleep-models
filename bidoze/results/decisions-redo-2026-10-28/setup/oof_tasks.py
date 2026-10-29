"""Task 1: out-of-fold per-epoch predictions under Eric's decisions, and Table A (25 h vs 8 h, both summed 0.5).

infer: 8 h window logits for the SleepAccel test nights (Wearanize+ test windows are reused from blend-exp2a).
score: 25 h (PSG span input) and 8 h (whole recording, probability-average blending) per-epoch parquet files, tables.
"""
import sys, pickle, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/home/eric/Engineering/Work/autopisces/sleep-models-uncropped-loaders/bidoze/results/blend-exp2a-2026-10-28/setup')
import blend_exp2a as B

U = B.U
RUN = U / 'bidoze/results/decisions-redo-2026-10-28'
OOF_PKL = U / 'bidoze/results/decoder-exp1-2026-10-28/inputs/oof.pkl'
WIN_OLD = B.RUN / 'windows'
WIN_NEW = RUN / 'windows'
OFFS = pd.read_csv(B.RUN / 'tables/stage_offsets.csv')


def infer():
    import jax, jax.numpy as jnp
    from arcasleep.data.batching import pad_pooled
    from arcasleep.predict import window_taper
    oof = pickle.load(open(OOF_PKL, 'rb'))
    taper = window_taper(B.L)
    for k in range(5):
        recs = [r for r in oof if r['cohort'] == 'SleepAccel' and r['fold'] == k]
        models = B.load_models(k); module = models[0].module
        f = jax.jit(lambda params, xb: module.apply({'params': params}, xb, jnp.ones(xb.shape[:2], jnp.float32), train=False))
        for r in recs:
            sid = r['night']; dest = WIN_NEW / f'test_f{k}_SleepAccel_{sid}.npz'
            if dest.exists(): continue
            x, y, raw = B.cache(B.ALN / f'SleepAccel_subject_{sid}_features.npz'); n = len(x)
            assert n == len(r['y'])
            a = B.window_starts(n, B.L, B.STEP); starts = np.array(sorted(set(a)))
            logits = np.zeros((3, len(starts), B.L, 5), np.float32)
            for m, model in enumerate(models):
                for i in range(0, len(starts), 2):
                    ss = starts[i:i + 2]
                    xb = np.stack([pad_pooled(x[s:s + B.L], B.L) for s in ss])
                    lg = np.asarray(f(model.params, jnp.asarray(xb))); assert np.isfinite(lg).all()
                    logits[m, i:i + len(ss)] = lg
            vlen = np.minimum(n - starts, B.L)
            np.savez_compressed(dest, logits=logits, starts=starts, valid=np.arange(B.L)[None] < vlen[:, None], taper=taper,
                                idx_a=np.searchsorted(starts, a), idx_b=np.searchsorted(starts, a), gap=x[:, -1, 0] >= 1, y=y, raw_y=raw,
                                subject=sid, cohort='SleepAccel', role='test', fold=k, n=n)
            B.log(f'windows f{k} {sid} n={n} windows={len(starts)}')


def sleep_summed(P):
    """sleep when P(light)+P(deep)+P(REM) >= P(wake); P has the wake, light, deep, REM probabilities in columns 0-3."""
    return P[:, 1:4].sum(1) >= P[:, 0]


def stage_off(P, sleep, off):
    sc = np.log(np.maximum(P[:, 1:4], 1e-12)) + off
    return np.where(sleep, 1 + sc.argmax(1), 0).astype(np.int8)


def offs(family, k):
    r = OFFS[(OFFS.family == family) & (OFFS.fold == k)].iloc[0]
    return np.array([0.0, r.offset_deep, r.offset_rev])


def write(df, name):
    df.to_parquet(RUN / 'oof' / name, index=False)


def score():
    from arcasleep_bidoze.release import ADULT_CV_COHORTS
    oof = pickle.load(open(OOF_PKL, 'rb'))
    prior = pd.read_csv(B.PRIOR / 'per_night.csv')
    primary = set(prior[(prior.scope == 'primary26') & (prior.model == '8h_context_original')].subject); assert len(primary) == 26
    rows = []; frames = {}
    # ---------------- 25 h, PSG span
    for r in oof:
        coh, sid, k = r['cohort'], r['night'], r['fold']
        if coh == 'Wearanize+':
            with np.load(B.PRIOR / f'probabilities/test_f{k}_wearanize_{sid}_25h_psg.npz') as z: off = int(z['offset']); assert (z['y'][:len(r['y'])] == r['y']).all() and 0 <= len(z['y']) - len(r['y']) <= 1
        else: off = 0
        P = r['p'].astype(np.float64); n = len(P); y = r['y'].astype(np.int64)
        sleep = sleep_summed(P)
        c0 = np.where(sleep, 1 + (P[:, 1:] / r['div'][1:]).argmax(1), 0).astype(np.int8)
        co = stage_off(P, sleep, offs('25h', k))
        mask = (y >= 0) & ~r['gap']
        df = pd.DataFrame({'cohort': coh, 'night': sid, 'participant': r['person'], 'fold': k, 'epoch': np.arange(n) + off, 'psg_label': y.astype(np.int8),
                           'p_wake': P[:, 0], 'p_light': P[:, 1], 'p_deep': P[:, 2], 'p_rem': P[:, 3], 'gap_flag': r['gap'], 'scored': mask,
                           'sleep_call_summed05': sleep, 'stage_call_offsets': co, 'stage_call_divisor': c0})
        frames.setdefault(('25h', coh), []).append(df)
        res = B.measure(y, co, mask); res['kappa4_offset'] = res.pop('kappa4'); res['kappa4_divisor'] = B.kappa(y[mask], c0[mask])
        rows.append(dict(model='25h', cohort=coh, night=sid, person=r['person'], fold=k, primary26=sid in primary, **res))
    # ---------------- 8 h, whole recording, probability average
    for k in range(5):
        for f in sorted(list(WIN_OLD.glob(f'test_f{k}_*.npz')) + list(WIN_NEW.glob(f'test_f{k}_*.npz'))):
            rec = B.blend_rec(B.load_rec(f), grids=('A',)); sid = rec['subject']; n = rec['n']; y = rec['y'].astype(np.int64)
            coh = 'Wearanize+' if rec['cohort'] == 'wearanize' else 'SleepAccel'
            span = rec['raw_y'] != -3; mask = (y >= 0) & ~rec['gap'] & ~rec['g4'] & span
            P = rec['P'][('A', 'prob_average')].astype(np.float64)
            sleep = sleep_summed(P)
            co = stage_off(P, sleep, offs('8h', k))
            person = sid.split('_s')[0] if coh == 'Wearanize+' else sid
            df = pd.DataFrame({'cohort': coh, 'night': sid, 'participant': person, 'fold': k, 'epoch': np.arange(n), 'psg_label': y.astype(np.int8),
                               'p_wake': P[:, 0], 'p_light': P[:, 1], 'p_deep': P[:, 2], 'p_rem': P[:, 3], 'p_gap': P[:, 4], 'gap_flag': rec['gap'],
                               'gap_head_call': rec['g4'], 'in_psg_span': span, 'scored': mask, 'sleep_call_summed05': sleep, 'stage_call_offsets': co})
            frames.setdefault(('8h', coh), []).append(df)
            res = B.measure(y, co, mask); res['kappa4_offset'] = res.pop('kappa4')
            rows.append(dict(model='8h', cohort=coh, night=sid, person=person, fold=k, primary26=sid in primary, **res))
        B.log(f'fold {k} scored')
    for (m, coh), L in frames.items():
        d = pd.concat(L, ignore_index=True)
        d['complete_context_26'] = d.night.isin(primary)
        write(d, f'{m}_{coh.replace("+", "plus")}.parquet')
    D = pd.DataFrame(rows); D.to_csv(RUN / 'tables/tableA_per_night.csv', index=False)
    # ---------------- Table A
    sets = [('Wearanize+ 92', lambda d: d[d.cohort == 'Wearanize+']), ('Wearanize+ 26 complete-context', lambda d: d[(d.cohort == 'Wearanize+') & d.primary26]),
            ('SleepAccel 28', lambda d: d[d.cohort == 'SleepAccel'])]
    srows, prow = [], []
    for name, sel in sets:
        d = sel(D)
        for m in ('25h', '8h'):
            x = d[d.model == m]
            srows.append(dict(set=name, model=m, nights=len(x), participants=x.person.nunique(), tst_bias=x.tst_error.mean(), tst_mae=x.tst_error.abs().mean(),
                              waso_bias=x.waso_error.mean(), waso_mae=x.waso_error.abs().mean(), sleep_sensitivity=x.sensitivity.mean(), wake_specificity=x.specificity.mean(),
                              kappa_sw=x.kappa_sw.mean(), kappa4_offsets=x.kappa4_offset.mean(), **({'kappa4_divisor_25h': x.kappa4_divisor.mean()} if m == '25h' else {})))
        a = d[d.model == '8h'].set_index('night'); b = d[d.model == '25h'].set_index('night').loc[a.index]
        for metric, absolute in [('tst_error', False), ('tst_error', True), ('waso_error', False), ('waso_error', True), ('sensitivity', False), ('specificity', False), ('kappa_sw', False), ('kappa4_offset', False)]:
            dd = (a[metric].abs() - b[metric].abs()) if absolute else (a[metric] - b[metric]); dd = dd.dropna()
            ci = B.cluster_ci(a.loc[dd.index, 'person'].to_numpy(), dd.to_numpy())
            prow.append(dict(set=name, metric=metric + ('_abs' if absolute else ''), nights=len(dd), diff_8h_minus_25h=dd.mean(), ci_lo=ci[0], ci_hi=ci[1]))
    S = pd.DataFrame(srows); S.to_csv(RUN / 'tables/tableA_summary.csv', index=False)
    P = pd.DataFrame(prow); P.to_csv(RUN / 'tables/tableA_paired.csv', index=False)
    pd.set_option('display.width', 250)
    print(S.round(3).to_string()); print(P.round(3).to_string())
    B.log('SCORE_DONE')


if __name__ == '__main__':
    {'infer': infer, 'score': score}[sys.argv[1]]()
