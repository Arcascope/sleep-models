"""Draw the BiDoze, F2 and F2 x3 architectures with parameter counts computed from the modules.

Usage: architecture_figure.py OUTPUT.svg   (a 2x .png is rendered beside it with headless Chrome)
Run with JAX_PLATFORMS=cpu PYTHONPATH=core/src:bidoze/src from the repository root.
"""
import json
import re
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path

import jax
import jax.numpy as jnp

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



# Stroke and fill per block type; the same type has the same colour in every panel.
INK = "#0f172a"
MUTED = "#475569"
FAINT = "#94a3b8"
COL = {
    "input": ("#475569", "#f1f5f9"),
    "conv": ("#1d4ed8", "#dbeafe"),
    "fwd": ("#6d28d9", "#ede9fe"),
    "rev": ("#be185d", "#fce7f3"),
    "cat": ("#0f766e", "#ccfbf1"),
    "gate": ("#15803d", "#dcfce7"),
    "head": ("#c2410c", "#ffedd5"),
    "proj": ("#a16207", "#fef9c3"),
    "fusion": ("#b45309", "#fde68a"),
}
# Hypnogram colours used by arcasleep.reporting.hypnogram.
STAGES = [("wake", "#f5a623"), ("light", "#5b8def"), ("deep", "#2e3a8c"), ("REM", "#2bc4d6"), ("gap", "#94a3b8")]
W = 1600
FONT = "Lato, 'Liberation Sans', Arial, sans-serif"


class Svg:
    def __init__(self):
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=12, color=INK, weight=400, anchor="start", mono=False, italic=False):
        family = "'DejaVu Sans Mono', monospace" if mono else FONT
        style = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" fill="{color}" '
                 f'font-weight="{weight}" text-anchor="{anchor}"{style}>{escape(s)}</text>')

    def rect(self, x, y, w, h, kind=None, fill=None, stroke=None, r=8, dashed=False, sw=1.5):
        if kind:
            stroke, fill = COL[kind]
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill or "none"}" '
                 f'stroke="{stroke or "none"}" stroke-width="{sw}"{dash}/>')

    def path(self, d, color=MUTED, dashed=False, arrow=True, sw=1.6):
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        end = ' marker-end="url(#arrow)"' if arrow else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{dash}{end}/>')

    def arrow(self, x1, y1, x2, y2, **kw):
        if abs(y1 - y2) < 0.5:
            self.path(f"M{x1:.1f},{y1:.1f} H{x2:.1f}", **kw)
        else:
            mx = (x1 + x2) / 2
            self.path(f"M{x1:.1f},{y1:.1f} C{mx:.1f},{y1:.1f} {mx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}", **kw)

    def render(self, height):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}">'
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>'
                f'<rect width="{W}" height="{height}" fill="#ffffff"/>' + "".join(self.parts) + "</svg>")


def block(svg, x, y, w, h, kind, title, lines=(), weights=None, size=12):
    """A rounded block: bold title, detail lines, and its weight count at the bottom."""
    svg.rect(x, y, w, h, kind)
    stroke = COL[kind][0]
    svg.text(x + w / 2, y + 22, title, size=size + 2, color=stroke, weight=700, anchor="middle")
    for i, line in enumerate(lines):
        svg.text(x + w / 2, y + 41 + i * (size + 4), line, size=size, color=MUTED, anchor="middle")
    if weights is not None:
        svg.text(x + w / 2, y + h - 11, n(weights), size=size, color=stroke, anchor="middle", mono=True)


def attention_dots(svg, cx, cy, mode, color, gap=9, r=3):
    """Seven epochs, the current one ringed; filled dots are the epochs it attends to."""
    for i in range(7):
        x = cx + (i - 3) * gap
        seen = mode == "full" or (mode == "fwd" and i <= 3) or (mode == "rev" and i >= 3)
        if i == 3:
            svg.add(f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="{r + 1.5}" fill="none" stroke="{INK}" stroke-width="1.2"/>')
        svg.add(f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="{r}" fill="{color if seen else "#e2e8f0"}"/>')


def stream(svg, x, y, w, h, kind, title, detail, weights, mode):
    svg.rect(x, y, w, h, kind)
    stroke = COL[kind][0]
    svg.text(x + 14, y + 21, title, size=14, color=stroke, weight=700)
    attention_dots(svg, x + w - 44, y + 16, mode, stroke)
    svg.text(x + 14, y + 40, detail, size=12, color=MUTED)
    svg.text(x + w - 12, y + 40, n(weights), size=12, color=stroke, anchor="end", mono=True)


def outputs(svg, x, y, w, h, stages):
    svg.rect(x, y, w, h, fill="#ffffff", stroke=INK, sw=1.2)
    svg.text(x + w / 2, y + 20, "per epoch", size=12, color=MUTED, anchor="middle")
    for i, (name, colour) in enumerate(stages):
        yy = y + 36 + i * 17
        svg.rect(x + 14, yy - 10, 12, 12, fill=colour, r=2, sw=0)
        svg.text(x + 32, yy, name, size=12)


def weight_bar(svg, x, y, scale, segments, label):
    """Weights drawn to scale: one px is the same number of weights in every panel."""
    svg.text(x, y - 8, label, size=12, color=MUTED)
    cursor = x
    for count, kind, text in segments:
        w = count * scale
        stroke, fill = COL[kind] if kind else (FAINT, "#f1f5f9")
        svg.rect(cursor, y, max(w, 1), 18, fill=fill, stroke=stroke, r=2, sw=0.8)
        if text and w > 8 * len(text):
            svg.text(cursor + w / 2, y + 13.5, text, size=11, color=stroke, anchor="middle")
        cursor += w
    return cursor


def panel_title(svg, y, letter, title, total):
    svg.text(24, y, letter, size=20, weight=900, color=INK)
    svg.text(52, y, title, size=20, weight=700, color=INK)
    svg.text(W - 24, y, total, size=15, color=MUTED, anchor="end")
    svg.add(f'<line x1="24" y1="{y + 10}" x2="{W - 24}" y2="{y + 10}" stroke="#e2e8f0" stroke-width="1.5"/>')


def panel_a(svg, y0, cfg, c, cg, scale):
    panel_title(svg, y0 + 28, "a", "BiDoze, one seed (v0.1.0 and BiDoze-G)",
                f"{n(c['total'])} weights (v0.1.0) · {n(cg['total'])} (BiDoze-G)")
    d, s = cfg.d_model, cfg.d_model // 2
    yc = y0 + 150
    top, h = yc - 48, 96
    block(svg, 24, top, 168, h, "input", "Input", ["61 rows × 2 channels", "per 30-s epoch"], None)
    svg.text(108, top + h - 11, "whole night", size=12, color=MUTED, anchor="middle", italic=True)
    block(svg, 228, top, 200, h, "conv", "Conv bank (causal)",
          [f"kernels {', '.join(map(str, cfg.conv_kernels))} epochs", f"{len(cfg.conv_kernels)} × {d // 4} → {d}"], c["conv"])
    sx, sw_ = 464, 300
    stream(svg, sx, yc - 58, sw_, 52, "fwd", "Forward stream",
           f"{cfg.n_transformer_layers} layers · width {s} · {cfg.n_heads // 2} heads", c["fwd"], "fwd")
    stream(svg, sx, yc + 6, sw_, 52, "rev", "Reverse stream",
           f"{cfg.n_transformer_layers} layers · width {s} · {cfg.n_heads // 2} heads", c["rev"], "rev")
    block(svg, 800, top, 150, h, "cat", "Concat", [f"{s} + {s} → {d}", "+ LayerNorm"], c["cat"])
    block(svg, 986, top, 176, h, "gate", "Gated fusion", ["gate × conv +", "(1 − gate) × transformer"], c["gate"])
    block(svg, 1198, top, 160, h, "head", "Head", [f"Dense {d} → {s} → 5", "v0.1.0: gap learned"], c["hidden"] + c["logits"])
    outputs(svg, 1394, yc - 58, 182, 116, STAGES)

    svg.arrow(192, yc, 228, yc)
    svg.arrow(428, yc, sx, yc - 32)
    svg.arrow(428, yc, sx, yc + 32)
    svg.arrow(sx + sw_, yc - 32, 800, yc)
    svg.arrow(sx + sw_, yc + 32, 800, yc)
    svg.arrow(950, yc, 986, yc)
    svg.arrow(1162, yc, 1198, yc)
    svg.arrow(1358, yc, 1394, yc)
    skip = yc - 78
    svg.path(f"M328,{top} V{skip} H1074 V{top}", color=COL["conv"][0], dashed=True)
    svg.text(700, skip - 7, "conv features skip the streams", size=12, color=COL["conv"][0], anchor="middle", italic=True)

    legend_y = yc + 82
    attention_dots(svg, 46, legend_y - 4, "fwd", FAINT, gap=7, r=2.5)
    svg.text(76, legend_y, "epochs each epoch attends to (ringed: the epoch itself). ALiBi position bias; "
             "absent epochs are hidden as attention keys.", size=12, color=MUTED)

    bar_y = yc + 118
    weight_bar(svg, 24, bar_y, scale, [
        (c["conv"], "conv", "conv 39%"), (c["fwd"], "fwd", "forward 25%"), (c["rev"], "rev", "reverse 25%"),
        (c["cat"], "cat", ""), (c["gate"], "gate", "gate"), (c["hidden"] + c["logits"], "head", ""),
    ], "Weights to scale")
    svg.text(24, bar_y + 46, f"BiDoze-G differs only at the gap output: an epoch is gap when its input gap row is 1, so the head "
             f"has 4 learned outputs ({n(cg['logits'])} final-layer weights vs {n(c['logits'])}). Every other weight has the same shape.",
             size=12, color=MUTED)
    return bar_y + 66


def panel_b(svg, y0, spec, f, scale):
    panel_title(svg, y0 + 28, "b", "F2, one model: three narrow BiDoze-G strands and a fusion transformer, trained in one fit",
                f"{n(f['total'])} weights")
    cfg = model_config(spec)
    w, d = spec.width, spec.width // 2
    cx, cy, cw = 220, y0 + 58, 600
    row_h, row_gap, header = 62, 10, 52
    ch = header + 3 * row_h + 2 * row_gap + 14
    svg.rect(cx, cy, cw, ch, fill="#f8fafc", stroke="#cbd5e1", r=12, dashed=True, sw=1.2)
    svg.text(cx + 16, cy + 22, f"Strand × 3: a BiDoze-G at width {w}", size=14, weight=700)
    svg.text(cx + cw - 16, cy + 22, f"{n(f['strand'])} weights each", size=12, color=MUTED, anchor="end", mono=True)
    svg.text(cx + 16, cy + 41, f"conv {n(f['conv'])} · streams 2 × {n(f['fwd'])} ({cfg.n_transformer_layers} layers, "
             f"width {d}, {cfg.n_heads // 2} heads) · concat {n(f['cat'])} · gate {n(f['gate'])} · hidden {n(f['hidden'])}",
             size=11, color=MUTED)

    yc_rows, hidden_out = [], []
    for i in range(3):
        ry = cy + header + i * (row_h + row_gap)
        rc = ry + row_h / 2 - 4
        yc_rows.append(rc)
        x = cx + 16
        svg.rect(x, rc - 22, 96, 44, "conv")
        svg.text(x + 48, rc - 4, "Conv bank", size=12, weight=700, color=COL["conv"][0], anchor="middle")
        svg.text(x + 48, rc + 12, f"4 × {w // 4} → {w}", size=11, color=MUTED, anchor="middle")
        x2 = x + 96 + 16
        for kind, yy, mode, label in (("fwd", rc - 22, "fwd", "Forward"), ("rev", rc + 1, "rev", "Reverse")):
            svg.rect(x2, yy, 128, 21, kind, r=5)
            svg.text(x2 + 8, yy + 15, label, size=11, weight=700, color=COL[kind][0])
            attention_dots(svg, x2 + 94, yy + 10.5, mode, COL[kind][0], gap=6, r=2)
        svg.arrow(x + 96, rc, x2, rc - 11.5, sw=1.2)
        svg.arrow(x + 96, rc, x2, rc + 11.5, sw=1.2)
        x3 = x2 + 128 + 16
        svg.rect(x3, rc - 22, 64, 44, "cat")
        svg.text(x3 + 32, rc - 4, "Concat", size=12, weight=700, color=COL["cat"][0], anchor="middle")
        svg.text(x3 + 32, rc + 12, f"→ {w}", size=11, color=MUTED, anchor="middle")
        svg.arrow(x2 + 128, rc - 11.5, x3, rc, sw=1.2)
        svg.arrow(x2 + 128, rc + 11.5, x3, rc, sw=1.2)
        x4 = x3 + 64 + 14
        svg.rect(x4, rc - 22, 60, 44, "gate")
        svg.text(x4 + 30, rc - 4, "Gate", size=12, weight=700, color=COL["gate"][0], anchor="middle")
        svg.text(x4 + 30, rc + 12, f"{w}", size=11, color=MUTED, anchor="middle")
        svg.arrow(x3 + 64, rc, x4, rc, sw=1.2)
        x5 = x4 + 60 + 14
        svg.rect(x5, rc - 22, 88, 44, "head")
        svg.text(x5 + 44, rc - 4, "Hidden", size=12, weight=700, color=COL["head"][0], anchor="middle")
        svg.text(x5 + 44, rc + 12, f"{d} per epoch", size=11, color=MUTED, anchor="middle")
        svg.arrow(x4 + 60, rc, x5, rc, sw=1.2)
        x6 = x5 + 88 + 22
        svg.rect(x6, rc + 2, 66, 22, fill="#ffffff", stroke=COL["head"][0], r=11, dashed=True, sw=1.1)
        svg.text(x6 + 33, rc + 17, "own head", size=11, color=COL["head"][0], anchor="middle")
        svg.arrow(x5 + 88, rc + 13, x6, rc + 13, dashed=True, sw=1.1)
        hidden_out.append((x5 + 88, rc - 10))
        svg.text(cx + 8, rc + 4, str(i + 1), size=12, weight=700, color=FAINT, anchor="middle")

    mid = cy + ch / 2
    block(svg, 24, mid - 48, 160, 96, "input", "Input", ["same as (a)", "61 rows × 2 channels"], None)
    svg.text(104, mid + 37, "whole night", size=12, color=MUTED, italic=True, anchor="middle")
    for rc in yc_rows:
        svg.arrow(184, mid, cx + 16, rc)

    px = cx + cw + 30
    block(svg, px, mid - 60, 150, 120, "proj", "Project, mix",
          [f"each strand {d} → {spec.fusion_proj}", "+ strand embedding", f"concat {3 * spec.fusion_proj} → {spec.fusion_width}"],
          f["proj"] + f["mix"])
    for i, (hx0, hy) in enumerate(hidden_out):
        svg.path(f"M{hx0},{hy} H{cx + cw - 4}", color=COL["head"][0], arrow=False, sw=1.2)
        svg.arrow(cx + cw - 4, hy, px, mid - 30 + i * 30, color=COL["head"][0], sw=1.2)

    fx = px + 150 + 28
    fw = 204
    svg.rect(fx, mid - 60, fw, 120, "fusion")
    svg.text(fx + fw / 2, mid - 38, "Fusion transformer", size=14, weight=700, color=COL["fusion"][0], anchor="middle")
    svg.text(fx + fw / 2, mid - 19, f"{spec.fusion_layers} layers · width {spec.fusion_width} · {spec.fusion_heads} heads",
             size=12, color=MUTED, anchor="middle")
    svg.text(fx + fw / 2, mid - 3, "full attention, both ways", size=12, color=MUTED, anchor="middle")
    attention_dots(svg, fx + fw / 2, mid + 18, "full", COL["fusion"][0])
    svg.text(fx + fw / 2, mid + 49, n(f["layers"]), size=12, color=COL["fusion"][0], anchor="middle", mono=True)
    svg.arrow(px + 150, mid, fx, mid)

    hx = fx + fw + 28
    hw = 1394 - 28 - hx
    block(svg, hx, mid - 60, hw, 120, "head", "Head", [f"Dense {spec.fusion_width} → 4", "gap from", "the gap row"],
          f["norm"] + f["out"])
    svg.arrow(fx + fw, mid, hx, mid)
    svg.arrow(hx + hw, mid, 1394, mid)
    outputs(svg, 1394, mid - 58, 182, 116, STAGES)

    bar_y = cy + ch + 40
    strand_segments = []
    for i in range(3):
        strand_segments += [(f["conv"], "conv", f"strand {i + 1}"), (f["fwd"], "fwd", ""), (f["rev"], "rev", ""),
                            (f["cat"] + f["gate"], "gate", ""), (f["hidden"] + f["head"], "head", "")]
    weight_bar(svg, 24, bar_y, scale, strand_segments + [
        (f["proj"] + f["mix"], "proj", ""), (f["layers"], "fusion", "fusion"), (f["norm"] + f["out"], "head", ""),
    ], "Weights to scale (same scale as a)")
    svg.text(24, bar_y + 46, f"Training loss: fusion head cross-entropy + {spec.strand_loss_weight} × each strand's own-head "
             f"cross-entropy, class-balanced. The strands' own heads ({n(3 * f['head'])} weights) are used only in training.",
             size=12, color=MUTED)
    return bar_y + 66


def ensemble(svg, x, y, w, kind_label, member, per, total, colour_kind):
    svg.text(x, y + 16, kind_label, size=16, weight=700)
    svg.text(x + w, y + 16, f"3 × {n(per)} = {n(total)} weights", size=12, color=MUTED, anchor="end", mono=True)
    stroke, fill = COL[colour_kind]
    for i in range(3):
        yy = y + 32 + i * 34
        svg.rect(x, yy, 150, 26, fill=fill, stroke=stroke, r=6)
        svg.text(x + 75, yy + 18, f"{member}, seed {i + 1}", size=12, color=stroke, weight=700, anchor="middle")
        svg.arrow(x + 150, yy + 13, x + 196, y + 81)
    svg.rect(x + 196, y + 59, 128, 44, fill="#ffffff", stroke=INK, r=8, sw=1.2)
    svg.text(x + 260, y + 78, "mean of", size=12, anchor="middle")
    svg.text(x + 260, y + 94, "probabilities", size=12, anchor="middle")
    svg.arrow(x + 324, y + 81, x + 356, y + 81)
    svg.rect(x + 356, y + 59, 120, 44, fill="#ffffff", stroke=INK, r=8, sw=1.2)
    svg.text(x + 416, y + 78, "stage divisor", size=12, anchor="middle")
    svg.text(x + 416, y + 94, "(no weights)", size=12, color=MUTED, anchor="middle")
    svg.arrow(x + 476, y + 81, x + 508, y + 81)
    for i, (name, colour) in enumerate(STAGES):
        svg.rect(x + 516, y + 46 + i * 15 - 1, 10, 10, fill=colour, r=2, sw=0)
        svg.text(x + 532, y + 46 + i * 15 + 8, name, size=11)


def panel_c(svg, y0, f, cg, c0):
    panel_title(svg, y0 + 28, "c", "Three-seed ensembles: each seed trained in its own fit", "")
    ensemble(svg, 24, y0 + 50, 600, "F2 × 3", "F2", f["total"], 3 * f["total"], "fusion")
    ensemble(svg, 840, y0 + 50, 600, "BiDoze-G × 3", "BiDoze-G", cg["total"], 3 * cg["total"], "conv")
    svg.text(24, y0 + 204, f"The adult release v0.1.0 is BiDoze v0.1.0 × 3 combined the same way: {n(3 * c0['total'])} weights. "
             "Stage divisor: the light, deep and REM probabilities are divided by per-stage weights before the stage call.",
             size=12, color=MUTED)
    return y0 + 220


def main(out):
    out = Path(out).with_suffix(".svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg, c0 = bidoze_counts("model.json")
    _, cg = bidoze_counts("model-gap-rule.json")
    spec, f = f2_counts()
    scale = (W - 48) / c0["total"]
    svg = Svg()
    y = panel_a(svg, 4, cfg, c0, cg, scale)
    y = panel_b(svg, y + 8, spec, f, scale)
    y = panel_c(svg, y + 8, f, cg, c0)
    out.write_text(svg.render(int(y)))
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "figure.html"
        page.write_text(f'<html><body style="margin:0">{out.read_text()}</body></html>')
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", f"--window-size={W},{int(y)}",
                        f"--screenshot={out.with_suffix('.png')}", page.as_uri()],
                       check=True, capture_output=True)
    print(json.dumps({"v0.1.0": c0, "G": cg, "F2": f}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
