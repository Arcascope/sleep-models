"""Figure: paired change in 4-stage and sleep/wake kappa (shared epochs, 95% participant-bootstrap CIs) vs B and vs F, plus coverage vs B."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from context_common import *

pr = pd.read_csv(EXP / "paired_vs_current.csv")
cov = pd.read_csv(EXP / "coverage.csv")
bs, fs = [120, 60, 30, 15, 5], [45, 15, 5, 0]
COL = {"wearanize": "#1b6ca8", "sleepaccel": "#d1691f"}
LS = {"hr": ("-", "o"), "nohr": ("--", "s")}
LAB = {"wearanize": "Wearanize+", "sleepaccel": "SleepAccel"}
OFF = {"hr": "HR tree", "nohr": "accelerometer-only tree"}
fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
for r, (metric, title) in enumerate((("kappa4", "4-stage κ"), ("kappa_sw", "sleep/wake κ"))):
    for c, (xs, key, xl) in enumerate(((bs, lambda x: f"b{x}_f45", "backward reach B (min), F = 45"),
                                      (fs, lambda x: f"b120_f{x}", "forward reach F (min), B = 120"))):
        a = ax[r, c]
        for cohort in COL:
            for off in LS:
                d = {x: pr[(pr.cohort == cohort) & (pr.offering == off) & (pr.metric == metric) & (pr.arm == key(x))] for x in xs}
                pts = [(x, 0.0, 0.0, 0.0) if (x == 120 and c == 0) or (x == 45 and c == 1) else
                       (x, *d[x][["diff", "ci_lo", "ci_hi"]].iloc[0]) for x in xs]
                x_ = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
                lo = y - np.array([p[2] for p in pts]); hi = np.array([p[3] for p in pts]) - y
                off_x = x_ * (1 + (0.03 if off == "nohr" else -0.03) * (c == 0)) if c == 0 else x_ + (0.4 if off == "nohr" else -0.4)
                a.errorbar(off_x, y, yerr=[lo, hi], color=COL[cohort], ls=LS[off][0], marker=LS[off][1], ms=5, capsize=3, lw=1.5,
                           label=f"{LAB[cohort]}, {OFF[off]}")
        a.axhline(0, color="#777", lw=0.8)
        a.set_ylabel(f"Δ {title} vs (120, 45)")
        a.set_title(f"Δ {title} vs {'B' if c == 0 else 'F'}", fontsize=11)
        if r == 1:
            a.set_xlabel(xl)
        if c == 0:
            a.set_xscale("log"); a.set_xticks(bs); a.set_xticklabels(bs); a.minorticks_off()
        else:
            a.set_xticks(fs)
    a = ax[r, 2]
    scope = "all PSG" if r == 0 else "first 2 h"
    for cohort in COL:
        d = cov[(cov.cohort == cohort) & (cov.scope == scope)].set_index("arm")
        a.plot(bs, [100 * d.loc[f"b{b}_f45", "share"] for b in bs], color=COL[cohort], marker="o", lw=1.5, label=LAB[cohort])
    a.set_ylim(-3, 103); a.set_ylabel("share of PSG epochs scorable (%)")
    a.set_title(f"Coverage vs B (F = 45), {scope}", fontsize=11)
    a.set_xscale("log"); a.set_xticks(bs); a.set_xticklabels(bs); a.minorticks_off()
    if r == 1:
        a.set_xlabel("backward reach B (min), F = 45")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.06, 1, 1))
(EXP / "figures").mkdir(exist_ok=True)
fig.savefig(EXP / "figures" / "context_ablation.png", dpi=150)
