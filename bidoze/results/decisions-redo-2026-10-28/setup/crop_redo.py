"""Experiment 2b: model-selected second pass for the frozen 8 h BiDoze fold models.

No training. Stages (run via setup/run.sh):
  score    Wearanize+ held-out nights: first pass, block selection, second pass, calls and metrics, validation thresholds
  foundry  Foundry windows (three people, Apple labels): same procedure with the 15 fold members, native and summed 0.5 only
  analyze  aggregate tables and paired first-vs-second-pass differences
Scoring functions are copied from Experiment 2a (setup/blend_exp2a.py, Current rule, hourly window grid); that directory is only read.
--smoke N restricts to fold 0, N test nights and 4 validation recordings, and writes under smoke/.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import wilcoxon

ROOT = Path('/home/eric/Engineering/Work/autopisces')
S = ROOT / 'sleep-models'; U = ROOT / 'sleep-models-uncropped-loaders'
RUN = U / 'bidoze/results/decisions-redo-2026-10-28/crop'
A2 = U / 'bidoze/results/blend-exp2a-2026-10-28'
OOF_PKL = U / 'bidoze/results/decoder-exp1-2026-10-28/inputs/oof.pkl'
PRIOR = S / 'bidoze/results/tst-investigation-2026-10-28'
NEW = U / 'bidoze/results/v020-8h-2026-10-28/run'
UNC = S / 'feature_cache/12Hz_log_PSD_uncropped'; ALN = S / 'feature_cache/12Hz_log_PSD_aligned'
LABEL = {0: 0, 1: 1, 2: 1, 3: 2, 5: 3}
L, STEP = 960, 120
JOIN, MARGIN, NAP_EPOCHS = 120, 120, 40          # wake runs < 60 min join blocks; 60 min margins; naps >= 20 min of predicted sleep
DECISIONS = ['native', 'sum_0.5']
GRID_T = np.round(np.arange(0, 1.0001, 0.025), 4)
T0 = time.time()


def log(s): print(f'{time.time() - T0:.1f}s {s}', flush=True)


# ---------------------------------------------------------------- copied from 2a
def cache(path):
    from arcasleep.data.batching import pool_frames_to_epochs
    with np.load(path) as z:
        raw = z['y'][0]; y = np.full(len(raw), -1, dtype=np.int8)
        for a, b in LABEL.items(): y[raw == a] = b
        x = pool_frames_to_epochs(z['X'], np.zeros((1, len(raw))))[0]
    return x, y, raw


def load_models(fold):
    from arcasleep_bidoze.model import BiDozeNetModel
    return [BiDozeNetModel.loadz(NEW / f'f{fold}_s{s}/B/model_outputs/model_fold{fold}_seed_{s}.pisces') for s in range(3)]


def kappa(y, p):
    if len(np.unique(np.r_[y, p])) < 2: return np.nan
    from sklearn.metrics import cohen_kappa_score
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); return cohen_kappa_score(y, p)


def bounds(s):
    edges = np.diff(np.r_[False, s, False].astype(int)); st = np.flatnonzero(edges == 1); en = np.flatnonzero(edges == -1)
    keep = en - st >= 20
    return (int(st[keep][0]), int(en[keep][-1])) if keep.any() else None


def measure(y, p, mask):
    """2a's measure plus onset and final-wake errors (minutes; predicted minus reference)."""
    ok = mask & (y >= 0) & (p >= 0) & (p < 4); t = y[ok]; c = p[ok]; ts = t > 0; ps = c > 0
    fp = np.sum(ps & ~ts) * .5; fn = np.sum(~ps & ts) * .5
    a = bounds((y > 0) & ok); b = bounds((p > 0) & ok)
    r = dict(n_epochs=int(ok.sum()), ref_sleep_min=ts.sum() * .5, false_sleep_min=fp, false_wake_min=fn, tst_error=fp - fn,
             kappa4=kappa(t, c), kappa_sw=kappa(ts, ps), sensitivity=float(ps[ts].mean()) if ts.any() else np.nan,
             specificity=float((~ps[~ts]).mean()) if (~ts).any() else np.nan)
    if a and b:
        r['waso_error'] = (np.sum(ok[b[0]:b[1]] & (p[b[0]:b[1]] == 0)) - np.sum(ok[a[0]:a[1]] & (y[a[0]:a[1]] == 0))) * .5
        r['onset_error'] = (b[0] - a[0]) * .5; r['final_wake_error'] = (b[1] - a[1]) * .5
    else: r['waso_error'] = r['onset_error'] = r['final_wake_error'] = np.nan
    r['pred_onset_epoch'] = b[0] if b else np.nan; r['pred_final_epoch'] = b[1] if b else np.nan
    return r


def wake_ratio(P): return P[:, 0] / np.maximum(P[:, 0] + P[:, 1:4].sum(1), 1e-12)


def sleep_decision(P, t=None):
    """native (t None): argmax over wake/light/deep/REM; else wake when P(wake)/(P(wake)+P(sleep stages)) > t."""
    if t is None: return P[:, :4].argmax(1) > 0
    return ~(wake_ratio(P) > t)


def stage_calls(P, sleep, off=None):
    sc = np.log(np.maximum(P[:, 1:4], 1e-12))
    if off is not None: sc = sc + off
    return np.where(sleep, 1 + sc.argmax(1), 0).astype(np.int8)


def bin_kappa(y, w):
    m = len(y); a = (y[:, None] & w).sum(0); b = (~y[:, None] & w).sum(0); c = (y[:, None] & ~w).sum(0); d = (~y[:, None] & ~w).sum(0)
    po = (a + d) / m; pe = ((a + b) * (a + c) + (c + d) * (b + d)) / m ** 2
    with np.errstate(all='ignore'): return np.where(pe < 1, (po - pe) / (1 - pe), np.nan)


def select_threshold(vals):
    """2a's objective. vals: (cohort, y_sleep, P, inblock, fixed). Epochs outside inblock keep the boolean call `fixed`
    (first-pass calls) in every candidate column; inblock None means every epoch is swept (first pass)."""
    cohorts = sorted({v[0] for v in vals}); G = GRID_T; per = {}
    for cohort, y, P, inb, fixed in vals:
        sl = ~(wake_ratio(P)[:, None] > G[None]); nat = (P[:, :4].argmax(1) > 0)[:, None]
        allsl = np.concatenate([sl, nat], 1)
        if inb is not None: allsl[~inb] = fixed[~inb][:, None]
        kap = bin_kappa(y, allsl)
        ws = ~y; spec = (~allsl[ws]).mean(0) if ws.any() else np.full(allsl.shape[1], np.nan)
        tst = np.abs((allsl.sum(0) - y.sum()) * .5)
        per.setdefault(cohort, []).append(np.stack([kap, spec, tst]))
    mean = {c: np.nanmean(np.stack(v), 0) for c, v in per.items()}
    kmin = np.min([mean[c][0] for c in cohorts], 0); tst = np.mean([np.nanmean(np.stack(v)[:, 2], 0) for v in per.values()], 0)
    feas = np.all([mean[c][1] >= mean[c][1][-1] - 1e-12 for c in cohorts], 0)[:-1]
    nat_k = kmin[-1]; info = dict(native_min_kappa=nat_k, **{f'native_kappa_{c}': mean[c][0][-1] for c in cohorts}, **{f'native_spec_{c}': mean[c][1][-1] for c in cohorts})
    cand = np.flatnonzero(feas & np.isfinite(kmin[:-1]))
    if len(cand) == 0 or kmin[:-1][cand].max() <= nat_k + 1e-12: return None, dict(info, kept_native=True)
    top = kmin[:-1][cand].max(); tied = cand[kmin[:-1][cand] >= top - 1e-12]
    i = sorted(tied, key=lambda i: (round(tst[i], 9), round(abs(G[i] - .5), 9), G[i]))[0]
    return float(G[i]), dict(info, kept_native=False, selected_min_kappa=kmin[i], **{f'selected_kappa_{c}': mean[c][0][i] for c in cohorts}, **{f'selected_spec_{c}': mean[c][1][i] for c in cohorts})


def cluster_ci(person, v, seed=29028, draws=5000):
    d = pd.DataFrame({'p': person, 'v': v}).dropna(); g = d.groupby('p')['v'].agg(['sum', 'count']).to_numpy()
    if len(g) < 2: return [np.nan, np.nan]
    ix = np.random.default_rng(seed).integers(0, len(g), (draws, len(g))); s = g[ix].sum(1)
    return np.percentile(s[:, 0] / s[:, 1], [2.5, 97.5]).tolist()


# ---------------------------------------------------------------- scoring
class Scorer:
    """Whole-input scoring with probability-average blending: softmax per window, taper-weighted mean per member, mean over members.
    Also returns the gap-head call of the Current rule (taper-weighted logit mean), the mask 2a/Task 1 exclude."""

    def __init__(self, models):
        import jax, jax.numpy as jnp
        from arcasleep.predict import window_taper
        self.models = models; self.taper = window_taper(L); module = models[0].module
        self.jit = jax.jit(lambda params, xb: module.apply({'params': params}, xb, jnp.ones(xb.shape[:2], jnp.float32), train=False))
        self.jnp = jnp

    def windows(self, n):
        from arcasleep.predict import window_starts
        return window_starts(n, L, STEP)

    def score(self, x):
        """Returns (P (n,5) float32, number of windows, seconds)."""
        from arcasleep.data.batching import pad_pooled
        t0 = time.time(); n = len(x); starts = self.windows(n); P = np.zeros((n, 5)); Pc = np.zeros((n, 5))
        for model in self.models:
            lg = np.zeros((len(starts), L, 5), np.float32)
            for i in range(0, len(starts), 2):
                ss = starts[i:i + 2]
                lg[i:i + len(ss)] = np.asarray(self.jit(model.params, self.jnp.asarray(np.stack([pad_pooled(x[s:s + L], L) for s in ss]))))
            assert np.isfinite(lg).all()
            tot = np.zeros((n, 5)); ptot = np.zeros((n, 5)); wt = np.zeros(n)
            for i, s in enumerate(starts):
                v = min(n - s, L); w = self.taper[:v]; l = lg[i][:v].astype(np.float64)
                tot[s:s + v] += l * w[:, None]; ptot[s:s + v] += softmax(l, axis=1) * w[:, None]; wt[s:s + v] += w
            assert (wt > 0).all()
            P += ptot / wt[:, None]; Pc += softmax(tot / wt[:, None], axis=1)
        return (P / len(self.models)).astype(np.float32), len(starts), time.time() - t0, (Pc / len(self.models)).argmax(1) == 4


def find_blocks(sleep):
    """Runs of predicted sleep joined across wake runs shorter than 60 min: list of (start, end, sleep_epochs)."""
    idx = np.flatnonzero(sleep)
    if not len(idx): return []
    cut = np.flatnonzero(np.diff(idx) > JOIN) + 1
    return [(int(idx[a]), int(idx[b - 1]) + 1, int(b - a)) for a, b in zip(np.r_[0, cut], np.r_[cut, len(idx)])]


def select_block(blocks, n):
    """Block with the most predicted sleep (earlier wins ties); returns (core block, rescored window with margins) or (None, None)."""
    if not blocks: return None, None
    b = max(blocks, key=lambda z: (z[2], -z[0]))
    return b, (max(b[0] - MARGIN, 0), min(b[1] + MARGIN, n))


def reference_block(y):
    b = find_blocks(y > 0)
    return select_block(b, len(y))[1] if b else None


class Store:
    """Cached first-pass and block probabilities (participant-level, local/)."""

    def __init__(self, ldir, scorer, shortcut=True):
        self.dir = ldir; self.dir.mkdir(parents=True, exist_ok=True); self.sc = scorer; self.shortcut = shortcut; self.first_P = {}

    def first(self, key, load):
        f = self.dir / f'first_{key}.npz'
        if f.exists():
            with np.load(f) as z: out = z['P'], int(z['windows']), float(z['seconds']), z['g4cur']
        else:
            P, nw, sec, g = self.sc.score(load()); np.savez_compressed(f, P=P, windows=nw, seconds=sec, g4cur=g); out = P, nw, sec, g
        self.first_P[key] = out[0]; return out

    def block(self, key, lo, hi, load):
        """Second-pass probabilities for epochs [lo, hi). A window spanning the whole recording is the first-pass input itself;
        its probabilities are reused at no added cost (--no-shortcut rescoring is for the smoke check)."""
        if self.shortcut and lo == 0 and hi == len(self.first_P[key]): return self.first_P[key], 0, 0.0
        f = self.dir / f'second_{key}_{lo}_{hi}.npz'
        if f.exists():
            with np.load(f) as z: return z['P'], int(z['windows']), float(z['seconds'])
        x = load()[lo:hi]; P, nw, sec, _ = self.sc.score(x); np.savez_compressed(f, P=P, windows=nw, seconds=sec); return P, nw, sec


def combine(P1, P2, lo, hi):
    P = P1.copy(); P[lo:hi] = P2; return P


def two_pass_calls(P1, sleep1, blk, P2b, dec, t2, off):
    """First- and second-pass (sleep, stage calls). Outside the rescored window the second pass keeps the first-pass calls."""
    first = stage_calls(P1, sleep1, off)
    if blk is None or P2b is None: return first, first.copy(), sleep1
    lo, hi = blk; s2 = sleep1.copy(); s2[lo:hi] = sleep_decision(P2b, t2)
    return first, stage_calls(combine(P1, P2b, lo, hi), s2, off), s2


# ---------------------------------------------------------------- score stage (Wearanize+ held-out nights)
def test_records(fold):
    """Held-out nights of a fold: Wearanize+ (92 over the folds) and SleepAccel (28), keys as in Task 1."""
    import pickle
    out = [dict(role='test', cohort='wearanize', sid=str(z['subject']), off=int(z['offset']), ny=len(z['y']), file=f)
           for f in sorted(PRIOR.glob(f'probabilities/test_f{fold}_*_8h_context.npz')) for z in [np.load(f)]]
    for r in pickle.load(open(OOF_PKL, 'rb')):
        if r['cohort'] == 'SleepAccel' and r['fold'] == fold: out.append(dict(role='test', cohort='SleepAccel', sid=r['night'], off=0, ny=len(r['y']), file=None))
    return out


def loader(m):
    path = (UNC / f'wearanize_subject_{m["sid"]}_features.npz') if m['cohort'] == 'wearanize' else ALN / f'SleepAccel_subject_{m["sid"]}_features.npz'
    def load():
        x, y, raw = cache(path); assert m['off'] == 0 and len(x) == m['ny']; return x
    return load, path


def score_stage(args):
    out = RUN / ('smoke' if args.smoke else '.'); ldir = out / 'local'; tdir = out / 'tables'; tdir.mkdir(exist_ok=True, parents=True)
    offs = json.loads(Path(args.offsets).read_text())
    prior = pd.read_csv(PRIOR / 'per_night.csv')
    primary = set(prior[(prior.scope == 'primary26') & (prior.model == '8h_context_original')].subject); assert len(primary) == 26
    rows, blk_rows = [], []
    for k in args.folds:
        sc = Scorer(load_models(k)); st = Store(ldir / 'first_second_passes', sc, not args.no_shortcut)
        off = np.array([0.0, offs[f'8h_fold{k}']['deep'], offs[f'8h_fold{k}']['rem']])[None]
        test = test_records(k)
        if args.smoke: test = test[:args.smoke]
        log(f'fold {k}: {len(test)} test recordings')
        for mt in test:
            sid = mt['sid']; load, path = loader(mt); coh = 'Wearanize+' if mt['cohort'] == 'wearanize' else 'SleepAccel'
            key = f'test_f{k}_{mt["cohort"]}_{sid}'; person = sid.split('_s')[0] if coh == 'Wearanize+' else sid
            P1, nw1, sec1, g4 = st.first(key, load); x, y, raw = cache(path); n = len(y); gap = x[:, -1, 0] >= 1
            span = raw != -3; mask = (y >= 0) & ~gap & ~g4 & span
            ref = reference_block(y); Pref = st.block(key, *ref, load)[0] if ref else None
            for d in DECISIONS:
                t = None if d == 'native' else 0.5
                s1 = sleep_decision(P1, t); blocks = find_blocks(s1 & ~g4); core, win = select_block(blocks, n)
                P2b = nw2 = sec2 = None
                if win: P2b, nw2, sec2 = st.block(key, *win, load)
                first, second, s2 = two_pass_calls(P1, s1, win, P2b, d, t, None)
                second_off = two_pass_calls(P1, s1, win, P2b, d, t, off)[1]; first_off = stage_calls(P1, s1, off)
                _, rc, _ = two_pass_calls(P1, s1, ref, Pref, d, t, None); rco = two_pass_calls(P1, s1, ref, Pref, d, t, off)[1]
                for pas, c, co in (('first', first, first_off), ('second', second, second_off), ('reference_crop', rc, rco)):
                    res = measure(y, c, mask); res['kappa4_native'] = res.pop('kappa4'); res['kappa4_offset'] = kappa(y[mask], co[mask])
                    rows.append(dict(subject=sid, person=person, cohort=coh, fold=k, primary26=sid in primary, decision=d, **{'pass': pas}, **res))
                psg = y > 0; nm = int(psg.sum())
                nap = [b for b in blocks if core is not None and b != core and b[2] >= NAP_EPOCHS]
                r1 = measure(y, first, mask); r2 = measure(y, second, mask)
                blk_rows.append(dict(subject=sid, person=person, cohort=coh, fold=k, primary26=sid in primary, decision=d, n=n, blocks=len(blocks), fragmented=len(blocks) > 1, naps=len(nap),
                                     has_block=core is not None, block_lo=core[0] if core else np.nan, block_hi=core[1] if core else np.nan,
                                     rescored_lo=win[0] if win else np.nan, rescored_hi=win[1] if win else np.nan, whole_recording=bool(win == (0, n)),
                                     psg_sleep_in_block_frac=float(psg[core[0]:core[1]].sum() / nm) if core and nm else np.nan,
                                     missed_psg_span=bool(core is None or (nm > 0 and psg[core[0]:core[1]].sum() / nm < .5)),
                                     onset_shift_min=(r2['pred_onset_epoch'] - r1['pred_onset_epoch']) * .5, final_wake_shift_min=(r2['pred_final_epoch'] - r1['pred_final_epoch']) * .5,
                                     sleep_min_changed_in_span=float(((first > 0) != (second > 0))[mask].sum() * .5),
                                     windows_first=nw1, windows_second=nw2 if nw2 else 0, seconds_first=sec1, seconds_second=sec2 if sec2 else 0.0))
            log(f'test {key} n={n}')
        pd.DataFrame(rows).to_csv(ldir / 'per_night.csv', index=False); pd.DataFrame(blk_rows).to_csv(ldir / 'block_diagnostics.csv', index=False)
    log('SCORE_DONE')


# ---------------------------------------------------------------- foundry stage
def foundry_stage(args):
    out = RUN / ('smoke' if args.smoke else '.'); ldir = out / 'local'
    incl = pd.read_csv(PRIOR / 'external_inclusion.csv'); incl = incl[(incl.cohort == 'foundry') & incl.qualifies]
    if args.smoke: incl = incl.iloc[:args.smoke]
    offs = json.loads(Path(args.offsets).read_text()) if Path(args.offsets).exists() else None
    off = np.mean([[0.0, offs[f'8h_fold{k}']['deep'], offs[f'8h_fold{k}']['rem']] for k in range(5) if f'8h_fold{k}' in offs], 0)[None] if offs else None
    models = [m for k in range(5) for m in load_models(k)]; assert len(models) == 15
    sc = Scorer(models); st = Store(ldir / 'foundry_passes', sc, not args.no_shortcut); rows, blk_rows = [], []
    for r in incl.itertuples():
        path = Path(r.path)
        def load(path=path, n=r.n): return cache(path)[0][:n]
        x, y, raw = cache(path); x = x[:r.n]; y = y[:r.n]; n = len(y); key = f'foundry_{r.subject}'
        P1, nw1, sec1, g4 = st.first(key, load); gap = x[:, -1, 0] >= 1
        win_eval = np.zeros(n, bool); win_eval[r.lo:r.hi] = True; mask = win_eval & (y >= 0) & ~gap & ~g4
        ref = (int(r.lo), int(r.hi)); Pref = st.block(key, *ref, load)[0]; person = r.subject.split('-')[0]
        for d in ('native', 'sum_0.5'):
            t = None if d == 'native' else 0.5; s1 = sleep_decision(P1, t); blocks = find_blocks(s1 & ~g4); core, win = select_block(blocks, n)
            P2b = nw2 = sec2 = None
            if win: P2b, nw2, sec2 = st.block(key, *win, load)
            first, second, _ = two_pass_calls(P1, s1, win, P2b, d, t, None)
            fo = stage_calls(P1, s1, off) if off is not None else None; so = two_pass_calls(P1, s1, win, P2b, d, t, off)[1] if off is not None else None
            _, rc, _ = two_pass_calls(P1, s1, ref, Pref, d, t, None); rco = two_pass_calls(P1, s1, ref, Pref, d, t, off)[1] if off is not None else None
            for pas, c, co in (('first', first, fo), ('second', second, so), ('reference_crop', rc, rco)):
                res = measure(y, c, mask); res['kappa4_native'] = res.pop('kappa4'); res['kappa4_offset'] = kappa(y[mask], co[mask]) if co is not None else np.nan
                rows.append(dict(subject=r.subject, person=person, decision=d, **{'pass': pas}, **res))
            psg = y > 0; nm = int(psg.sum()); r1 = measure(y, first, mask); r2 = measure(y, second, mask)
            blk_rows.append(dict(subject=r.subject, person=person, decision=d, n=n, blocks=len(blocks), fragmented=len(blocks) > 1,
                                 naps=len([b for b in blocks if core is not None and b != core and b[2] >= NAP_EPOCHS]), has_block=core is not None,
                                 block_lo=core[0] if core else np.nan, block_hi=core[1] if core else np.nan, rescored_lo=win[0] if win else np.nan, rescored_hi=win[1] if win else np.nan,
                                 whole_recording=bool(win == (0, n)), psg_sleep_in_block_frac=float(psg[core[0]:core[1]].sum() / nm) if core and nm else np.nan,
                                 missed_psg_span=bool(core is None or (nm > 0 and psg[core[0]:core[1]].sum() / nm < .5)),
                                 onset_shift_min=(r2['pred_onset_epoch'] - r1['pred_onset_epoch']) * .5, final_wake_shift_min=(r2['pred_final_epoch'] - r1['pred_final_epoch']) * .5,
                                 windows_first=nw1, windows_second=nw2 if nw2 else 0, seconds_first=sec1, seconds_second=sec2 if sec2 else 0.0))
        log(f'foundry {r.subject} n={n}')
    ldir.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows).to_csv(ldir / 'foundry_per_night.csv', index=False); pd.DataFrame(blk_rows).to_csv(ldir / 'foundry_block_diagnostics.csv', index=False)
    log('FOUNDRY_DONE')


# ---------------------------------------------------------------- analyze stage
def summarize(D, B, tag, tdir, scopes):
    srows, prows, brows = [], [], []
    D = D.copy(); D['abs_tst'] = D.tst_error.abs(); D['abs_waso'] = D.waso_error.abs(); D['abs_onset'] = D.onset_error.abs(); D['abs_final_wake'] = D.final_wake_error.abs()
    for scope, sel in scopes:
        d = D[sel(D)]; b = B[sel(B)]
        for (dec, pas), x in d.groupby(['decision', 'pass']):
            r = dict(scope=scope, decision=dec, **{'pass': pas}, nights=len(x), people=x.person.nunique())
            for m in ['tst_error', 'waso_error', 'onset_error', 'final_wake_error']:
                v = x[m].dropna(); r.update({m + '_bias': v.mean(), m + '_mae': v.abs().mean()})
                if m == 'tst_error': r.update(tst_loa_lo=v.mean() - 1.96 * v.std(), tst_loa_hi=v.mean() + 1.96 * v.std())
            for m in ['sensitivity', 'specificity', 'kappa_sw', 'kappa4_native', 'kappa4_offset']: r[m] = x[m].mean()
            srows.append(r)
        for dec in d.decision.unique():
            f = d[(d.decision == dec) & (d['pass'] == 'first')].set_index('subject'); s = d[(d.decision == dec) & (d['pass'] == 'second')].set_index('subject').loc[f.index]
            for m in ['tst_error', 'abs_tst', 'waso_error', 'abs_waso', 'abs_onset', 'abs_final_wake', 'sensitivity', 'specificity', 'kappa_sw', 'kappa4_native', 'kappa4_offset']:
                dd = (s[m] - f[m]).dropna()
                if len(dd) < 2: continue
                ci = cluster_ci(f.loc[dd.index, 'person'].to_numpy(), dd.to_numpy())
                prows.append(dict(scope=scope, decision=dec, metric=m, nights=len(dd), second_minus_first=dd.mean(), ci_lo=ci[0], ci_hi=ci[1], wilcoxon_p=wilcoxon(dd).pvalue if (dd != 0).any() else 1.0))
        for dec, x in b.groupby('decision'):
            brows.append(dict(scope=scope, decision=dec, nights=len(x), no_block=int((~x.has_block).sum()), missed_psg_span=int(x.missed_psg_span.sum()), fragmented_gt1_block=int(x.fragmented.sum()),
                              nights_with_nap=int((x.naps > 0).sum()), naps=int(x.naps.sum()), block_spans_whole_recording=int(x.whole_recording.sum()),
                              onset_shift_mean_min=x.onset_shift_min.mean(), onset_shift_abs_mean_min=x.onset_shift_min.abs().mean(), onset_shift_max_abs_min=x.onset_shift_min.abs().max(),
                              final_wake_shift_mean_min=x.final_wake_shift_min.mean(), final_wake_shift_abs_mean_min=x.final_wake_shift_min.abs().mean(), final_wake_shift_max_abs_min=x.final_wake_shift_min.abs().max(),
                              windows_first_total=int(x.windows_first.sum()), windows_second_total=int(x.windows_second.sum()), window_ratio_second_over_first=x.windows_second.sum() / x.windows_first.sum(),
                              seconds_first_total=x.seconds_first.sum(), seconds_second_total=x.seconds_second.sum(), seconds_ratio_second_over_first=x.seconds_second.sum() / x.seconds_first.sum()))
    pd.DataFrame(srows).to_csv(tdir / f'{tag}summary.csv', index=False); pd.DataFrame(prows).to_csv(tdir / f'{tag}paired_second_minus_first.csv', index=False)
    pd.DataFrame(brows).to_csv(tdir / f'{tag}block_summary.csv', index=False)


def analyze_stage(args):
    out = RUN / ('smoke' if args.smoke else '.'); ldir = out / 'local'; tdir = out / 'tables'; tdir.mkdir(exist_ok=True, parents=True)
    if (ldir / 'per_night.csv').exists():
        summarize(pd.read_csv(ldir / 'per_night.csv'), pd.read_csv(ldir / 'block_diagnostics.csv'), '', tdir, [('wearanize_complete26', lambda d: (d.cohort == 'Wearanize+') & d.primary26), ('wearanize92', lambda d: d.cohort == 'Wearanize+'), ('sleepaccel28', lambda d: d.cohort == 'SleepAccel')])
    if (ldir / 'foundry_per_night.csv').exists():
        summarize(pd.read_csv(ldir / 'foundry_per_night.csv'), pd.read_csv(ldir / 'foundry_block_diagnostics.csv'), 'foundry_', tdir, [('foundry_complete_window', lambda d: d.person == d.person)])
        f = pd.read_csv(ldir / 'foundry_per_night.csv'); g = f.groupby(['person', 'decision', 'pass'])[['tst_error', 'waso_error', 'onset_error', 'final_wake_error', 'kappa_sw']].agg(['mean', 'count'])
        g.to_csv(tdir / 'foundry_by_person.csv')
    log('ANALYSIS_DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['score', 'foundry', 'analyze']); ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--smoke', type=int, default=0); ap.add_argument('--offsets', default=None, help='2a stage_offsets.json (keys 8h_fold{k}); missing file leaves four-stage offset kappa empty')
    ap.add_argument('--no-shortcut', action='store_true'); a = ap.parse_args()
    if a.offsets is None: a.offsets = str(A2 / ('smoke/tables' if a.smoke else 'tables') / 'stage_offsets.json')
    {'score': score_stage, 'foundry': foundry_stage, 'analyze': analyze_stage}[a.stage](a)
