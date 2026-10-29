"""Experiment 2c: final 8 h BiDoze ensemble (3 seeds) vs the v0.2.0 adult final 25 h ensemble (3 members) on cohorts outside the final training population.

Stages (run via run_score.sh):
  score    8 h original (whole-recording) and 8 h selected (model-selected second pass) per night; 25 h from Experiment 1's stored probabilities (AAUWSS, EESM19,
           BIDSleep, SleepAccel-Clinical) or the release bundle scored live (Foundry). Decodings: argmax and summed 0.5 (no fitted threshold: the final models have no
           held-out training-side set).
  analyze  summary, paired tables (participant bootstrap, Wilcoxon), block diagnostics.
Scoring/second-pass functions are imported from Experiment 2b (setup/crop_exp2b.py, read only); 25 h inputs from Experiment 1 (inputs/*.pkl, read only).
--smoke N: N nights per cohort, a stand-in model (smoke model seed 0 for all three members), outputs under smoke/.
Masks: epoch scored when PSG/Apple stage >= 0, no gap row, neither ensemble's gap head wins (8 h argmax == 5th class, 25 h gap flag from Experiment 1 / live), same for all models.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

W = Path('/home/eric/Engineering/Work/autopisces/sleep-models-uncropped-loaders')
sys.path.insert(0, str(W / 'bidoze/results/decisions-redo-2026-10-28/setup'))
import crop_redo as C      # noqa: E402  (only imports definitions)

S = C.S; OLD = W / 'bidoze/results/final-8h-exp2c-2026-10-28'; RUN = W / 'bidoze/results/decisions-redo-2026-10-28/final8h'
EXP1 = W / 'bidoze/results/decoder-exp1-2026-10-28'
BUNDLE = S / 'bidoze/results/adult-f2-2026-10-23/release-v0.2.0/adult'
OFFS = C.A2 / 'tables/stage_offsets.json'
DECODINGS = {'argmax': None, 'sum_0.5': 0.5}      # name -> threshold for C.sleep_decision
MODELS = ['25h', '8h_original', '8h_selected']
EXT_COHORTS = ['AAUWSS', 'EESM19', 'BIDSleep', 'SleepAccel-Clinical']
COHORT_ORDER = EXT_COHORTS + ['Foundry']
FILES = {'AAUWSS': (C.ALN, 'aauwss'), 'EESM19': (C.UNC, 'eesm19'), 'BIDSleep': (C.UNC, 'bidsleep')}
T0 = time.time()
log = C.log


def offsets_mean(prefix):
    o = json.loads(OFFS.read_text()); ks = [k for k in o if k.startswith(prefix + '_fold')]
    return np.mean([[0.0, o[k]['deep'], o[k]['rem']] for k in ks], 0)[None]


def calls_25h(P4, sleep, div=None, off=None):
    """25 h stage call inside sleep: divisor rule (native) or log-probability offsets."""
    if off is None: st = 1 + (P4[:, 1:4].astype(np.float64) / np.asarray(div, float)[1:]).argmax(1)
    else: st = 1 + (np.log(np.maximum(P4[:, 1:4], 1e-12)) + off).argmax(1)
    return np.where(sleep, st, 0).astype(np.int8)


def night_row(base, y, c, co, mask):
    res = C.measure(y, c, mask); res['kappa4_native'] = res.pop('kappa4')
    res['kappa4_offset'] = C.kappa(y[mask & (y >= 0)], co[mask & (y >= 0)]) if co is not None else np.nan
    return dict(**base, **res)


def block_row(base, y, n, blocks, core, win, r1, r2, nw1, nw2, sec1, sec2):
    psg = y > 0; nm = int(psg.sum())
    return dict(**base, n=n, blocks=len(blocks), fragmented=len(blocks) > 1, naps=len([b for b in blocks if core is not None and b != core and b[2] >= C.NAP_EPOCHS]),
                has_block=core is not None, block_lo=core[0] if core else np.nan, block_hi=core[1] if core else np.nan,
                rescored_lo=win[0] if win else np.nan, rescored_hi=win[1] if win else np.nan, whole_recording=bool(win == (0, n)),
                psg_sleep_in_block_frac=float(psg[core[0]:core[1]].sum() / nm) if core and nm else np.nan,
                missed_psg_span=bool(core is None or (nm > 0 and psg[core[0]:core[1]].sum() / nm < .5)),
                onset_shift_min=(r2['pred_onset_epoch'] - r1['pred_onset_epoch']) * .5, final_wake_shift_min=(r2['pred_final_epoch'] - r1['pred_final_epoch']) * .5,
                windows_first=nw1, windows_second=nw2 or 0, seconds_first=sec1, seconds_second=sec2 or 0.0)


# ---------------------------------------------------------------- night lists
def external_nights(smoke):
    """25 h records of Experiment 1 (same nights, y, gap) with the cache file of each night."""
    recs = pickle.load(open(EXP1 / 'inputs/external.pkl', 'rb')) + pickle.load(open(EXP1 / 'inputs/clinical.pkl', 'rb'))
    files = {}
    for coh, (d, pre) in FILES.items():
        for f in sorted(d.glob(f'{pre}_subject_*_features.npz')):
            with np.load(f) as z: files[(coh, str(z['subject_ids'][0]))] = f
    out = []
    for coh in EXT_COHORTS:
        rs = [r for r in recs if r['cohort'] == coh]
        if smoke: rs = rs[:smoke]
        for r in rs:
            f = (C.ALN / f'SleepAccel-Clinical_subject_{r["night"]}_features.npz') if coh == 'SleepAccel-Clinical' else files[(coh, r['night'])]
            out.append(dict(cohort=coh, night=r['night'], person=r['person'], path=f, rec=r))
    return out


def load_external(m):
    """(x, y, span lo, hi) of the 8 h input: Exp 1's PSG span for AAUWSS/EESM19/BIDSleep, the whole file for Clinical."""
    x, y, raw = C.cache(m['path'])
    if m['cohort'] == 'SleepAccel-Clinical': a, b = 0, len(y)
    else: v = np.flatnonzero(raw != -3); a, b = int(v[0]), int(v[-1]) + 1
    return x[a:b], y[a:b], raw[a:b]


# ---------------------------------------------------------------- score stage
def score_stage(args):
    out = RUN / ('smoke' if args.smoke else 'results'); ldir = out / 'local'; ldir.mkdir(parents=True, exist_ok=True)
    mdir = Path(args.models_dir)
    if args.smoke: paths = [mdir / 'model_final_seed_0.pisces'] * 3
    else: paths = [mdir / f'model_final_seed_{s}.pisces' for s in range(3)]
    for p in paths: assert p.exists(), p
    from arcasleep_bidoze.model import BiDozeNetModel
    models = [BiDozeNetModel.loadz(p) for p in paths]
    sc = C.Scorer(models); st = C.Store(ldir / 'passes', sc, not args.no_shortcut)
    o8 = offsets_mean('8h'); o25 = offsets_mean('25h'); div25 = np.array([1.0] + [json.loads((BUNDLE / 'config.json').read_text())['stage_divisor'][x] for x in ('light', 'deep', 'rem')])
    log(f'8 h members: {[str(p) for p in paths]}; smoke={bool(args.smoke)}'); rows, blk = [], []

    def one(base, key, load8, y, gap_in, P25, g25, win_eval=None):
        """Score one night with every model and decoding. load8() -> 8 h input x (same span as y)."""
        P1, nw1, sec1, g4 = st.first(key, load8); n = len(y)
        mask = (y >= 0) & ~gap_in & ~g4 & ~g25
        if win_eval is not None: mask &= win_eval
        for d, t in DECODINGS.items():
            s1 = C.sleep_decision(P1, t); blocks = C.find_blocks(s1 & ~g4); core, win = C.select_block(blocks, n)
            P2b = nw2 = sec2 = None
            if win: P2b, nw2, sec2 = st.block(key, *win, load8)
            first, second, _ = C.two_pass_calls(P1, s1, win, P2b, d, t, None)
            fo = C.stage_calls(P1, s1, o8); so = C.two_pass_calls(P1, s1, win, P2b, d, t, o8)[1]
            s25 = C.sleep_decision(P25, t)
            b = dict(cohort=base['cohort'], night=base['night'], person=base['person'], decoding=d)
            rows.append(night_row({**b, 'model': '25h'}, y, calls_25h(P25, s25, div25), calls_25h(P25, s25, off=o25), mask))
            rows.append(night_row({**b, 'model': '8h_original'}, y, first, fo, mask))
            rows.append(night_row({**b, 'model': '8h_selected'}, y, second, so, mask))
            r1 = C.measure(y, first, mask); r2 = C.measure(y, second, mask)
            blk.append(block_row(b, y, n, blocks, core, win, r1, r2, nw1, nw2, sec1, sec2))
        log(f'{key} n={n} mask_epochs={int(mask.sum())}')

    for m in external_nights(args.smoke):
        r = m['rec']; coh = m['cohort']
        x, y, raw = load_external(m)
        assert len(y) == len(r['y']) and np.array_equal(y, r['y']), f'{coh} {m["night"]}: y differs from Experiment 1'
        gap_row = x[:, -1, 0] >= 1
        gap25 = np.asarray(r['gap'], bool)
        # Exp 1 stores gap as (25 h gap-head argmax | gap row) for fold/Clinical, gap row only for external; the gap row is shared, so only the 25 h head part is extra
        one(m, f'{coh}_{m["night"]}', lambda x=x: x, y, gap_row, r['p'], gap25 & ~gap_row)
    incl = pd.read_csv(C.PRIOR / 'external_inclusion.csv'); incl = incl[(incl.cohort == 'foundry') & incl.qualifies]
    if args.smoke: incl = incl.iloc[:args.smoke]
    from arcasleep.predict import Predictor, _stage_and_gap_probabilities
    from arcasleep.data.batching import pad_pooled
    P25m = Predictor(BUNDLE)
    for r in incl.itertuples():
        x, y, raw = C.cache(Path(r.path)); x = x[:r.n]; y = y[:r.n]; n = len(y)
        lg = P25m.model.infer(pad_pooled(x, 3000)[None])[0][:n]; P25, gp = _stage_and_gap_probabilities(lg, 4)
        g25 = (np.exp(lg - lg.max(1, keepdims=True)) / np.exp(lg - lg.max(1, keepdims=True)).sum(1, keepdims=True)).argmax(1) == 4
        we = np.zeros(n, bool); we[r.lo:r.hi] = True
        one(dict(cohort='Foundry', night=r.subject, person=r.subject.split('-')[0]), f'foundry_{r.subject}', lambda x=x: x, y, x[:, -1, 0] >= 1, P25, g25, we)
        pass
    pd.DataFrame(rows).to_csv(out / 'per_night.csv', index=False); pd.DataFrame(blk).to_csv(out / 'block_diagnostics.csv', index=False)
    log('SCORE_DONE')


# ---------------------------------------------------------------- analyze stage
def analyze_stage(args):
    out = RUN / ('smoke' if args.smoke else 'results'); tdir = out / 'tables'; tdir.mkdir(exist_ok=True)
    D = pd.read_csv(out / 'per_night.csv'); B = pd.read_csv(out / 'block_diagnostics.csv')
    for m, a in [('abs_tst', 'tst_error'), ('abs_waso', 'waso_error'), ('abs_onset', 'onset_error'), ('abs_final_wake', 'final_wake_error')]: D[m] = D[a].abs()
    srows, prows, brows = [], [], []
    for coh in COHORT_ORDER:
        for (mod, dec), x in D[D.cohort == coh].groupby(['model', 'decoding']):
            r = dict(cohort=coh, model=mod, decoding=dec, nights=len(x), participants=x.person.nunique())
            for m in ['tst_error', 'waso_error', 'onset_error', 'final_wake_error']:
                v = x[m].dropna(); r.update({m + '_bias': v.mean(), m + '_mae': v.abs().mean(), m + '_loa_lo': v.mean() - 1.96 * v.std(), m + '_loa_hi': v.mean() + 1.96 * v.std()})
            for m in ['sensitivity', 'specificity', 'kappa_sw', 'kappa4_native', 'kappa4_offset']: r[m] = x[m].mean()
            srows.append(r)
        for dec in DECODINGS:
            g = {m: D[(D.cohort == coh) & (D.decoding == dec) & (D.model == m)].set_index('night') for m in MODELS}
            for name, (a, b) in {'8h_original_minus_25h': ('8h_original', '25h'), '8h_selected_minus_25h': ('8h_selected', '25h'), '8h_selected_minus_8h_original': ('8h_selected', '8h_original')}.items():
                for m in ['tst_error', 'abs_tst', 'waso_error', 'abs_waso', 'abs_onset', 'abs_final_wake', 'sensitivity', 'specificity', 'kappa_sw', 'kappa4_native', 'kappa4_offset']:
                    if g[a].empty: continue
                    dd = (g[a][m] - g[b].loc[g[a].index, m]).dropna()
                    if len(dd) < 2: continue
                    ci = C.cluster_ci(g[a].loc[dd.index, 'person'].to_numpy(), dd.to_numpy())
                    prows.append(dict(cohort=coh, decoding=dec, comparison=name, metric=m, nights=len(dd), participants=g[a].loc[dd.index].person.nunique(), mean_difference=dd.mean(),
                                      ci_lo=ci[0], ci_hi=ci[1], wilcoxon_p=wilcoxon(dd).pvalue if (dd != 0).any() else 1.0))
        for dec, x in B[B.cohort == coh].groupby('decoding'):
            brows.append(dict(cohort=coh, decoding=dec, nights=len(x), no_block=int((~x.has_block).sum()), missed_psg_span=int(x.missed_psg_span.sum()), fragmented_gt1_block=int(x.fragmented.sum()),
                              nights_with_nap=int((x.naps > 0).sum()), naps=int(x.naps.sum()), block_spans_whole_recording=int(x.whole_recording.sum()),
                              onset_shift_mean_min=x.onset_shift_min.mean(), onset_shift_abs_mean_min=x.onset_shift_min.abs().mean(), onset_shift_max_abs_min=x.onset_shift_min.abs().max(),
                              final_wake_shift_mean_min=x.final_wake_shift_min.mean(), final_wake_shift_abs_mean_min=x.final_wake_shift_min.abs().mean(), final_wake_shift_max_abs_min=x.final_wake_shift_min.abs().max(),
                              windows_first_total=int(x.windows_first.sum()), windows_second_total=int(x.windows_second.sum()), window_ratio_second_over_first=x.windows_second.sum() / x.windows_first.sum(),
                              seconds_first_total=x.seconds_first.sum(), seconds_second_total=x.seconds_second.sum(), seconds_ratio_second_over_first=x.seconds_second.sum() / x.seconds_first.sum()))
    pd.DataFrame(srows).to_csv(tdir / 'summary.csv', index=False); pd.DataFrame(prows).to_csv(tdir / 'paired.csv', index=False); pd.DataFrame(brows).to_csv(tdir / 'block_summary.csv', index=False)
    counts = D[D.model == '25h'].drop_duplicates(['cohort', 'night', 'decoding']).query("decoding == 'argmax'").groupby('cohort').agg(nights=('night', 'count'), participants=('person', 'nunique'), scored_epochs=('n_epochs', 'sum'))
    counts.to_csv(tdir / 'counts.csv'); log(counts.to_string()); log('ANALYSIS_DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['score', 'analyze']); ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--models-dir', default=str(OLD / 'run/final/model_outputs')); ap.add_argument('--no-shortcut', action='store_true')
    a = ap.parse_args()
    if a.smoke and a.models_dir == str(OLD / 'run/final/model_outputs'): a.models_dir = str(OLD / 'smoke/run/final/model_outputs')
    {'score': score_stage, 'analyze': analyze_stage}[a.stage](a)
