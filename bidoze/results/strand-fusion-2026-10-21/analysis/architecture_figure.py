"""Draw the BiDoze, F2 and F2 x3 architectures with parameter counts computed from the modules.

Usage: architecture_figure.py OUTPUT.png   (an .svg is written beside it)
Run with JAX_PLATFORMS=cpu PYTHONPATH=core/src:bidoze/src from the repository root.
"""
import json
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "bidoze/experiments/strand_fusion"))
from arcasleep_bidoze.model import BiDozeNetConfig  # noqa: E402
from arcasleep_bidoze.model.definition import BiDozeNetModule  # noqa: E402
from fusion_model import StrandFusionModule, arm_spec, model_config  # noqa: E402

FREQ_ROWS = 60  # floor(fmax * window) - ceil(fmin * window) + 1 with 0.1-6.0 Hz and 10-s windows
ROWS = FREQ_ROWS + 1  # plus the gap row appended by pool_frames_to_epochs
X = jnp.zeros((1, 8, ROWS, 2))


def size(tree):
    return int(sum(leaf.size for leaf in jax.tree.leaves(tree)))


def shapes(module):
    return jax.eval_shape(lambda: module.init(jax.random.PRNGKey(0), X, jnp.ones((1, 8)))["params"])


def bidoze_counts(config_name):
    cfg = BiDozeNetConfig.from_dict(json.loads((ROOT / "bidoze/config" / config_name).read_text()))
    p = shapes(BiDozeNetModule(cfg))
    t = p["bi_causal_transformer"]
    stream = lambda d: sum(size(v) for k, v in t.items() if k.startswith(d) or k == f"in_proj_{d}")
    return cfg, {
        "conv": size(p["conv_bank"]), "fwd": stream("fwd"), "rev": stream("rev"),
        "cat": size(t["out_proj"]) + size(t["out_norm"]), "gate": size(p["gated_fusion"]),
        "hidden": size(p["head_hidden"]), "logits": size(p["head_logits"]), "total": size(p),
    }


def f2_counts():
    spec = arm_spec("final3_fuse")
    p = shapes(StrandFusionModule(model_config(spec), spec))
    pick = lambda pat: sum(size(v) for k, v in p.items() if re.fullmatch(pat, k))
    f = p["fusion"]
    return spec, {
        "conv": pick(r"arm_0_cnn"), "fwd": pick(r"arm_0_fwd_(in|\d)"), "rev": pick(r"arm_0_rev_(in|\d)"),
        "cat": pick(r"arm_0_out(_norm)?"), "gate": pick(r"arm_0_fusion"), "hidden": pick(r"arm_0_head_hidden"),
        "head": pick(r"arm_0_head_logits"),
        "proj": sum(size(v) for k, v in f.items() if k.startswith("project_")) + size(f["strand_embedding"]),
        "mix": size(f["mix"]), "layers": sum(size(v) for k, v in f.items() if k.startswith("layer_")),
        "norm": size(f["norm"]), "out": size(f["head_logits"]), "fusion": size(f),
        "strand": sum(size(v) for k, v in p.items() if k.startswith("arm_0_")), "total": size(p),
    }


def n(v):
    return f"{v:,}"


def box(ax, x, y, w, h, text, fs=6.6, bold=False, dashed=False, fc="white"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="square,pad=0", fc=fc, ec="black",
                                lw=0.8, ls="--" if dashed else "-"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.3)


def arrow(ax, *pts, dashed=False):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=0.8, ls="--" if dashed else "-")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8, ls="--" if dashed else "-",
                                shrinkA=0, shrinkB=0))


def panel(ax, height, title):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.text(0, height - 0.5, title, fontsize=8.6, fontweight="bold", va="top")


def panel_a(ax, cfg, c, cg):
    H = 74
    panel(ax, H, f"(a) BiDoze v0.1.0 / BiDoze-G, one seed: {n(c['total'])} weights (v0.1.0), {n(cg['total'])} (BiDoze-G)")
    ax.set_ylim(-15, H)
    d, s = cfg.d_model, cfg.d_model // 2
    box(ax, 50, 62, 92, 8, f"Input per 30-s epoch: {ROWS} rows × 2 channels\n"
        f"{FREQ_ROWS} log-PSD rows (0.1-6.0 Hz) + 1 gap row (included in both versions); channels: magnitude, jerk\n"
        f"flattened to {ROWS * 2} features per epoch; the whole night is one sequence of epochs")
    box(ax, 50, 50, 70, 8, f"Conv bank (causal): kernels {', '.join(map(str, cfg.conv_kernels))} epochs, "
        f"{len(cfg.conv_kernels)} × {d // len(cfg.conv_kernels)} → {d}\n"
        f"each branch: causal conv, LayerNorm, GELU; concat, Dense {d}, LayerNorm\n{n(c['conv'])} weights")
    stream = lambda name, who, v: (f"{name} stream: sees {who}\nDense {d} → {s}; {cfg.n_transformer_layers} layers, width {s},\n"
                                   f"{cfg.n_heads // 2} heads, feed-forward {cfg.d_ff // 2}, ALiBi\n{n(v)} weights")
    box(ax, 27, 36, 42, 11, stream("Forward", "only earlier epochs\n(and the current one)", c["fwd"]))
    box(ax, 73, 36, 42, 11, stream("Reverse", "only later epochs\n(and the current one)", c["rev"]))
    box(ax, 50, 24, 50, 6.5, f"Concatenate ({s} + {s}) → Dense {d} + LayerNorm\n{n(c['cat'])} weights")
    box(ax, 52, 14.5, 62, 7, f"Gated fusion: gate = sigmoid(W·local + W'·global), width {d}\n"
        f"output = gate × conv + (1 − gate) × transformer; LayerNorm; {n(c['gate'])} weights")
    box(ax, 52, 5.5, 54, 7, f"Readout head: Dense {d} → {d // 2}, GELU, dropout, Dense {d // 2} → outputs\n"
        f"{n(c['hidden'])} weights + final layer {n(c['logits'])} (v0.1.0, 5 outputs)\nor {n(cg['logits'])} (BiDoze-G, 4 outputs)")
    box(ax, 91, 5.5, 16, 7, "5 probabilities\nper epoch:\nwake, light,\ndeep, REM, gap", fs=6)
    arrow(ax, (50, 58), (50, 54))
    arrow(ax, (50, 46), (50, 43), (27, 43), (27, 41.5))
    arrow(ax, (50, 43), (73, 43), (73, 41.5))
    arrow(ax, (27, 30.5), (27, 29), (42, 29), (42, 27.25))
    arrow(ax, (73, 30.5), (73, 29), (58, 29), (58, 27.25))
    arrow(ax, (50, 20.75), (50, 18))
    arrow(ax, (15, 46), (15, 14.5), (21, 14.5), dashed=True)  # conv features (local) bypass the streams
    ax.text(14, 22, "conv features\n(local)", fontsize=6, ha="right", va="center")
    arrow(ax, (52, 11), (52, 9))
    arrow(ax, (79, 5.5), (83, 5.5))
    ax.text(0, -0.8, "Only difference: in v0.1.0 gap is a learned 5th class (the head's 5th output is trained).\n"
            "In BiDoze-G (gap_from_row = true) an epoch is gap when its gap row = 1, and every other epoch has gap probability 0;\n"
            "absent epochs are zeroed at the input and hidden as attention keys in both streams. All other weights have the same shapes.",
            fontsize=6.2, va="top", linespacing=1.4)


def panel_b(ax, spec, f):
    H = 118
    panel(ax, H, f"(b) F2, one model: {n(f['total'])} weights, trained end to end in one fit")
    w, d, h = spec.width, spec.width // 2, spec.width // 2
    cfg = model_config(spec)
    box(ax, 50, 108, 92, 5.5, f"Same input per 30-s epoch: {ROWS} rows × 2 channels ({FREQ_ROWS} log-PSD rows + gap row), "
        f"flattened to {ROWS * 2} features")
    xs = [17, 50, 83]
    for i, x in enumerate(xs):
        arrow(ax, (x, 105.25), (x, 102.5))
        ax.add_patch(plt.Rectangle((x - 15.5, 53), 31, 50, fc="none", ec="black", lw=0.6, ls=":"))
        ax.text(x, 101.7, f"Strand {i + 1}: a narrower BiDoze, {n(f['strand'])} weights", fontsize=6.4, ha="center", va="top", fontweight="bold")
        box(ax, x, 93.5, 28, 7, f"Conv bank (causal): kernels {', '.join(map(str, cfg.conv_kernels))},\n{len(cfg.conv_kernels)} × {w // 4} → {w}\n{n(f['conv'])}", fs=5.6)
        sx = lambda name, who, v: f"{name}: {who}\n{cfg.n_transformer_layers} layers, width {d},\n{cfg.n_heads // 2} heads, ff {spec.strand_ffn}, ALiBi\n{n(v)}"
        box(ax, x - 7.5, 82, 13.6, 11, sx("Forward", "earlier\nepochs only", f["fwd"]), fs=4.9)
        box(ax, x + 7.5, 82, 13.6, 11, sx("Reverse", "later\nepochs only", f["rev"]), fs=4.9)
        box(ax, x, 71.5, 28, 5, f"Concat + Dense {w} + LayerNorm\n{n(f['cat'])}", fs=5.6)
        box(ax, x, 64.5, 28, 5, f"Gated fusion (conv, transformer), width {w}\n{n(f['gate'])}", fs=5.6)
        box(ax, x - 7, 57, 14, 5.5, f"Head hidden:\nDense {w} → {h}, GELU\n{n(f['hidden'])}", fs=5)
        box(ax, x + 9, 57, 11, 5.5, f"Own 5-way head\n(train only)\n{n(f['head'])}", fs=5, dashed=True)
        arrow(ax, (x, 90), (x, 88.5), (x - 7.5, 88.5), (x - 7.5, 87.5))
        arrow(ax, (x, 88.5), (x + 7.5, 88.5), (x + 7.5, 87.5))
        arrow(ax, (x - 7.5, 76.5), (x - 7.5, 75.5), (x, 75.5), (x, 74))
        arrow(ax, (x + 7.5, 76.5), (x + 7.5, 75.5), (x, 75.5), (x, 74))
        arrow(ax, (x, 69), (x, 67))
        arrow(ax, (x, 62), (x, 61.5), (x - 7, 61.5), (x - 7, 59.75))
        arrow(ax, (x - 1, 57), (x + 3.5, 57))
        ax.plot([x - 7, x - 7], [54.25, 49.5], color="black", lw=0.8)
    ax.text(43, 51.2, "head-hidden features (penultimate, width 56 per epoch) of each strand", fontsize=5.6, ha="center", va="center")
    box(ax, 50, 44.5, 70, 7, f"Per-strand projection Dense {h} → {spec.fusion_proj} + strand embedding ({spec.members} × {spec.fusion_proj}), "
        f"concatenate ({spec.members * spec.fusion_proj})\nthen Dense {spec.members * spec.fusion_proj} → {spec.fusion_width} (mix); "
        f"{n(f['proj'])} + {n(f['mix'])} weights")
    box(ax, 50, 32, 70, 9, f"Fusion transformer: {spec.fusion_layers} layers, width {spec.fusion_width}, {spec.fusion_heads} heads, "
        f"feed-forward {spec.fusion_ffn}, ALiBi,\nfull attention (every present epoch sees every present epoch; absent epochs hidden as keys)\n"
        f"{n(f['layers'])} weights")
    box(ax, 50, 21.5, 70, 5.5, f"LayerNorm ({n(f['norm'])}); dropout; Dense {spec.fusion_width} → 5 (fusion head), gap set from the gap row; "
        f"{n(f['out'])}")
    box(ax, 50, 13, 50, 6, "Output: 5 probabilities per epoch\n(wake, light, deep, REM, gap)")
    arrow(ax, (50, 41), (50, 36.5))
    arrow(ax, (50, 27.5), (50, 24.25))
    arrow(ax, (50, 18.75), (50, 16))
    ax.plot([10, 76], [49.5, 49.5], color="black", lw=0.8)
    arrow(ax, (43, 49.5), (43, 48))
    ax.text(0, 5.5, f"Training loss = class-balanced cross-entropy of the fusion head + {spec.strand_loss_weight} × the same loss of each strand's own head "
            f"(strand heads are used only in training: {n(3 * f['head'])} weights).\n"
            f"Strand: {n(f['strand'])} = conv {n(f['conv'])} + forward {n(f['fwd'])} + reverse {n(f['rev'])} + concat {n(f['cat'])} + gate {n(f['gate'])} "
            f"+ head hidden {n(f['hidden'])} + head {n(f['head'])}.\n"
            f"Fusion: {n(f['fusion'])}. Total = 3 × {n(f['strand'])} + {n(f['fusion'])}.", fontsize=5.8, va="center")


def panel_c(ax, f, cg, c0):
    H = 34
    panel(ax, H, "(c) Three-seed ensembles")
    for x, title, per, tot, body in [
        (25, "F2 × 3", f["total"], 3 * f["total"],
         "3 F2 models, different seeds, each trained in its own fit\nmean of the 3 fusion heads' probabilities\nthen the stage divisor"),
        (75, "For comparison: BiDoze-G × 3", cg["total"], 3 * cg["total"],
         "3 BiDoze-G models, different seeds, trained separately\nmean of the 3 probabilities, then the stage divisor"),
    ]:
        box(ax, x, 22, 46, 18, f"{title}\n{body}\n3 × {n(per)} = {n(tot)} weights", fs=6.6)
    box(ax, 50, 5.5, 96, 7, f"Release (v0.1.0 adult): 3 seeds trained separately, mean of probabilities, then the stage divisor; "
        f"3 × {n(c0['total'])} = {n(3 * c0['total'])} weights.\nStage divisor: light, deep and REM probabilities are divided by per-stage weights "
        f"(the wake entry is not used) before the stage call; it adds no weights.", fs=6)


def main(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg, c0 = bidoze_counts("model.json")
    _, cg = bidoze_counts("model-gap-rule.json")
    spec, f = f2_counts()
    fig = plt.figure(figsize=(9.6, 24))
    gs = fig.add_gridspec(3, 1, height_ratios=[89, 118, 34], hspace=0.02, left=0.02, right=0.98, top=0.995, bottom=0.005)
    panel_a(fig.add_subplot(gs[0]), cfg, c0, cg)
    panel_b(fig.add_subplot(gs[1]), spec, f)
    panel_c(fig.add_subplot(gs[2]), f, cg, c0)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".svg"))
    print(json.dumps({"v0.1.0": c0, "G": cg, "F2": f}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
