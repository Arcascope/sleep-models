"""Figures of sections 1-3 (redo) from the tables written by tst123.py. 26 complete-context Wearanize+ nights."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
RR = Path('/home/eric/Engineering/Work/autopisces/sleep-models-instinct-trees/watchtree/results/tree-vs-bidoze-redo-2026-10-28')
T, F, L = RR / 'tables', RR / 'figures', RR / 'local'
S = pd.read_csv(T / 'tst123_summary.csv'); S = S[S.scope == 'primary26'].set_index('model')
C = pd.read_csv(T / 'tst123_context_changes.csv'); C = C[C.scope == 'primary26']
PN = pd.read_csv(L / 'tst123_per_night.csv'); PN = PN[PN.scope == 'primary26']
COL = {'25h': '#386cb0', '8h': '#7b3294', 'hr': '#d28b24', 'nohr': '#2a9d8f', 'ref': '#9aa5b1'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': True})
fmt = lambda t: t.replace('-', '−')
NOTE = 'Same 26 held-out Wearanize+ participants. BiDoze: three-seed ensembles; summed 0.5 = sleep when P(light)+P(deep)+P(REM) ≥ P(wake); 8 h windows blended by probability average.\nTrees: three-seed mean of per-night errors, stage offsets; HR tree 87 columns, accelerometer-only tree 62 columns. Argmax rows are the reference.'

# ---- tst_synthesis
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1, 1.5]})
x = np.arange(2); w = .36
for j, (dec, lab, hatch, alpha) in enumerate((('orig', 'argmax (reference)', '//', .45), ('sum', 'summed 0.5', '', 1))):
    for i, arch in enumerate(('bz25', 'bz8')):
        r = C[(C.arch == arch) & (C.decision == dec)].iloc[0]
        ax.bar(i + (j - .5) * w, r.mean_abs_tst_change, w, color=COL['25h' if arch == 'bz25' else '8h'], alpha=alpha, hatch=hatch, edgecolor='white', label=lab if i == 0 else None)
        ax.errorbar(i + (j - .5) * w, r.mean_abs_tst_change, yerr=[[r.mean_abs_tst_change - r.abs_ci_lo], [r.abs_ci_hi - r.mean_abs_tst_change]], color='#222', capsize=4)
        ax.text(i + (j - .5) * w, r.abs_ci_hi + .5, fmt(f'{r.mean_abs_tst_change:.1f}'), ha='center', fontsize=9)
ax.set_xticks(x, ['25 h model', '8 h model']); ax.set_ylabel('Mean |TST change| after adding surrounding data (min)'); ax.set_ylim(0, 26); ax.legend(frameon=False)
ax.set_title('1. Surrounding data changes 8 h TST less\n95% participant-bootstrap intervals', fontsize=11)
names = ['bz25_crop_orig', 'bz25_crop_sum', 'bz8_ctx_orig', 'bz8_ctx_sum', 'tree_hr_off', 'tree_nohr_off']
labels = ['25 h, PSG crop: argmax', '25 h, PSG crop: summed 0.5', '8 h, whole recording: argmax', '8 h, whole recording: summed 0.5', 'HR tree, offsets', 'Accelerometer-only tree, offsets']
cols = [COL['ref'], COL['25h'], COL['ref'], COL['8h'], COL['hr'], COL['nohr']]
y = np.arange(len(names)); vals = S.loc[names, 'tst_mae'].to_numpy()
bx.barh(y, vals, color=cols, height=.65)
for i, v in enumerate(vals): bx.text(v + .25, i, fmt(f'{v:.1f}'), va='center')
bx.set_yticks(y, labels); bx.invert_yaxis(); bx.set_xlim(0, 24); bx.set_xlabel('TST mean absolute error (min)')
bx.set_title('2. Sleep/wake decoding changes TST error\nSummed 0.5 vs argmax, and both trees', fontsize=11)
fig.suptitle('Wearanize+: context sensitivity and sleep/wake decoding', fontsize=13, y=1.0); fig.text(.5, -.04, NOTE, ha='center', fontsize=8)
fig.tight_layout(w_pad=3); fig.savefig(F / 'tst_synthesis.png', dpi=180, bbox_inches='tight'); plt.close(fig)

# ---- tst_decoding_context
models = ['bz25_crop_orig', 'bz25_crop_sum', 'bz25_ctx_sum', 'bz8_crop_sum', 'bz8_ctx_orig', 'bz8_ctx_sum', 'tree_hr_off', 'tree_nohr_off']
labels = ['25 h, PSG crop: argmax', '25 h, PSG crop: summed 0.5', '25 h, surrounding data: summed 0.5', '8 h, PSG crop: summed 0.5', '8 h, surrounding data: argmax',
          '8 h, surrounding data: summed 0.5', 'HR tree, offsets', 'Accelerometer-only tree, offsets']
cols = [COL['ref'], COL['25h'], '#7ea6d6', COL['8h'], COL['ref'], '#b28abe', COL['hr'], COL['nohr']]
s = S.loc[models]; y = np.arange(len(models))
fig, axs = plt.subplots(1, 3, figsize=(14.5, 5.2))
for ax, col, xl in zip(axs, ('tst_bias', 'tst_mae', 'kappa_sw'), ('TST bias (min)', 'TST mean absolute error (min)', 'Sleep/wake κ')):
    ax.barh(y, s[col], color=cols, height=.65); ax.invert_yaxis(); ax.set_xlabel(xl)
    for i, v in enumerate(s[col]): d = .004 if col == 'kappa_sw' else .4; ax.text(v + (d if v >= 0 else -d), i, fmt(f'{v:.2f}' if col == 'kappa_sw' else f'{v:.1f}'), va='center', ha='left' if v >= 0 else 'right', fontsize=8)
axs[0].set_yticks(y, labels); axs[0].axvline(0, color='.3', lw=.8); axs[1].set_yticks(y, []); axs[2].set_yticks(y, []); axs[2].set_xlim(.5, .75); axs[0].set_xlim(-24, 16)
fig.suptitle('Wearanize+, 26 held-out nights: TST bias, absolute error and sleep/wake κ by decoding', y=1.0, fontsize=13); fig.text(.5, -.05, NOTE, ha='center', fontsize=8)
fig.tight_layout(); fig.savefig(F / 'tst_decoding_context.png', dpi=180, bbox_inches='tight'); plt.close(fig)

# ---- tst_context_sensitivity
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
for arch, key in (('25h', 'bz25'), ('8h', 'bz8')):
    a = PN[PN.model == f'{key}_crop_sum'].set_index('subject'); b = PN[PN.model == f'{key}_ctx_sum'].set_index('subject').loc[a.index]
    axs[0].scatter(a.tst_error, b.tst_error, color=COL[arch], alpha=.8, label=f'{arch[:-1]} h summed 0.5')
axs[0].plot([-55, 45], [-55, 45], color='.5', ls='--'); axs[0].axvline(0, color='.85'); axs[0].axhline(0, color='.85')
axs[0].set_xlabel('TST error with PSG crop (min)'); axs[0].set_ylabel('TST error with surrounding data (min)'); axs[0].legend(frameon=False)
for j, (dec, alpha, hatch) in enumerate((('orig', .45, '//'), ('sum', 1, ''))):
    for i, (arch, key) in enumerate((('25h', 'bz25'), ('8h', 'bz8'))):
        r = C[(C.arch == key) & (C.decision == dec)].iloc[0]; xx = i + (j - .5) * .36
        axs[1].bar(xx, r.mean_abs_tst_change, .36, color=COL[arch], alpha=alpha, hatch=hatch, edgecolor='white', label=('argmax (reference)' if dec == 'orig' else 'summed 0.5') if i == 0 else None)
        axs[1].errorbar(xx, r.mean_abs_tst_change, yerr=[[r.mean_abs_tst_change - r.abs_ci_lo], [r.abs_ci_hi - r.mean_abs_tst_change]], color='#222', capsize=4)
axs[1].set_xticks([0, 1], ['25 h', '8 h']); axs[1].set_ylabel('Mean |TST change| from context (min)'); axs[1].set_title('95% participant-bootstrap interval', fontsize=10); axs[1].legend(frameon=False)
fig.suptitle('Recording context: the same 26 Wearanize+ nights, BiDoze inputs', y=1.0, fontsize=12); fig.tight_layout(); fig.savefig(F / 'tst_context_sensitivity.png', dpi=180, bbox_inches='tight'); plt.close(fig)

# ---- tst_movement_diagnostics
M = pd.read_csv(T / 'tst3_movement.csv'); M = M[M.scope == 'primary26'].set_index('model')
R = pd.read_csv(T / 'tst3_run_length.csv'); R = R[R.scope == 'primary26']
TR = pd.read_csv(T / 'tst3_tree_recovery.csv'); TR = TR[TR.scope == 'primary26']
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={'width_ratios': [1.5, 1]})
cats = ['False wake:\ntop jerk quartile', 'All PSG sleep:\ntop jerk quartile', 'False wake:\nquiet surroundings', 'HR tree calls\nsleep on false wake', 'Accel-only tree calls\nsleep on false wake']
for i, (m, arch) in enumerate((('bz25_crop_sum', '25h'), ('bz8_crop_sum', '8h'))):
    t = TR[TR.bidoze_model == m].set_index('tree')
    v = [M.loc[m, 'fw_in_top_jerk_quartile_pct'], M.loc[m, 'sleep_in_top_jerk_quartile_pct'], M.loc[m, 'fw_quiet_surround_pct'], t.loc['tree_hr_off', 'tree_calls_sleep_pct'], t.loc['tree_nohr_off', 'tree_calls_sleep_pct']]
    axs[0].bar(np.arange(5) + (i - .5) * .36, v, .36, color=COL[arch], label=f'{arch[:-1]} h summed 0.5')
axs[0].set_xticks(np.arange(5), cats, fontsize=8); axs[0].set_ylabel('Share (%)'); axs[0].set_ylim(0, 105); axs[0].legend(frameon=False)
bins = ['<= 2 min', '2-10 min', '> 10 min']
for i, (m, arch) in enumerate((('bz25_crop_sum', '25h'), ('bz8_crop_sum', '8h'))):
    v = R[R.model == m].set_index('run_length').loc[bins, 'false_wake_min_per_night']
    axs[1].bar(np.arange(3) + (i - .5) * .36, v, .36, color=COL[arch], label=f'{arch[:-1]} h summed 0.5')
axs[1].set_xticks(np.arange(3), ['≤ 2 min', '2–10 min', '> 10 min']); axs[1].set_ylabel('False-wake minutes per night'); axs[1].set_xlabel('Consecutive false-wake run length'); axs[1].legend(frameon=False)
fig.suptitle('Wearanize+, 26 held-out nights: false wake inside PSG sleep under summed 0.5 (PSG-crop input)\nJerk quartiles are each night’s PSG-sleep jerk; quiet surroundings = ≥ 90% still 5 s blocks within ±15 min; tree shares pooled over epochs, seed mean', y=1.03, fontsize=11)
fig.tight_layout(); fig.savefig(F / 'tst_movement_diagnostics.png', dpi=180, bbox_inches='tight'); plt.close(fig)
print('FIGS_DONE')
