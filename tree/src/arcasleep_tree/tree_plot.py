"""Makes a picture of trees from a trained Arcasleep LightGBM model.

Being binary trees, if they have even depth then the leaf count is divisible by 4.
Thus, we space these around the 4 edges of the image produced per tree.
"""


import json, sys, os
from xml.sax.saxutils import escape
src, outdir = sys.argv[1], sys.argv[2]
d = json.load(open(src))
FE, TH, LF = (d["trees"][k] for k in ("feature", "threshold", "leaf"))
names, redges, ls = d["features"], d["real_edges"], d["leaf_scale"]
CLS = ["wake", "light", "deep", "REM"]
BW, BH, LW = 150, 36, 70      # box width/height, leaf width
SLOT = 160                    # spacing between leaves across an arm (N/S)
SLOT_EW = 44                  # spacing between leaves across an arm (W/E)
LEVEL_NS, LEVEL_EW = 90, 190  # spacing between levels along an arm
D = 400                       # arm root distance from centre

def build(t, i):
    """Collapse always-left padding; return nested node dict."""
    if i >= 63:
        return {"leaf": LF[t][i - 63] * ls}
    f, th = FE[t][i], TH[t][i]
    miss_left = f >= 128
    f %= 128
    if th >= len(redges[f]):
        return build(t, 2 * i + 1)
    return {"f": names[f], "thr": redges[f][th], "ml": miss_left, "l": build(t, 2 * i + 1), "r": build(t, 2 * i + 2)}

def lay(n, depth, cursor):
    """Assign n['lvl'] and n['pos'] (in leaf slots)."""
    n["lvl"] = depth
    if "leaf" in n:
        n["pos"] = cursor[0]; cursor[0] += 1; return
    lay(n["l"], depth + 1, cursor); lay(n["r"], depth + 1, cursor)
    n["pos"] = (n["l"]["pos"] + n["r"]["pos"]) / 2

def count(n): return 1 if "leaf" in n else count(n["l"]) + count(n["r"])

def draw_tree(t):
    root = build(t, 0)
    out = []
    def box(x, y, lines, leaf=False):
        w = LW if leaf else BW
        fill = "#eef5ee" if leaf else "#ffffff"
        s = f'<rect x="{x-w/2:.1f}" y="{y-BH/2:.1f}" width="{w}" height="{BH}" rx="4" fill="{fill}" stroke="#333"/>'
        for k, ln in enumerate(lines):
            yy = y + (k - (len(lines) - 1) / 2) * 14 + 4
            s += f'<text x="{x:.1f}" y="{yy:.1f}" text-anchor="middle" font-size="11" font-family="monospace" fill="#111">{escape(ln)}</text>'
        out.append(s)
    def edge(p, q, label):
        out.append(f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]:.1f}" y2="{q[1]:.1f}" stroke="#666"/>')
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        out.append(f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#a00" stroke="#fff" stroke-width="4" paint-order="stroke">{label}</text>')
    def text(n):
        return ([f"{n['f']}", f"≤ {n['thr']:.3g} ?"], False) if "leaf" not in n else ([f"{n['leaf']:+.3f}"], True)

    pos = {}
    def place_arm(n, arm):
        c = [0]; lay(n, 0, c)
        width = c[0]
        off = lambda p: (p - (width - 1) / 2)
        def xy(m):
            if arm in "NS":
                s = -1 if arm == "N" else 1
                return (off(m["pos"]) * SLOT, s * (D + m["lvl"] * LEVEL_NS))
            s = -1 if arm == "W" else 1
            return (s * (D + m["lvl"] * LEVEL_EW), off(m["pos"]) * SLOT_EW * 1.0)
        def walk(m):
            m["xy"] = xy(m)
            if "leaf" not in m:
                walk(m["l"]); walk(m["r"])
        walk(n)
    def emit(n, parent=None, label=None):
        x, y = n["xy"]
        if parent: edge(parent["xy"], n["xy"], label)
        lines, leaf = text(n)
        if "leaf" not in n:
            emit(n["l"], n, "≤ or missing" if n["ml"] else "≤"); emit(n["r"], n, ">" if n["ml"] else "> or missing")
        box(x, y, lines, leaf)

    # root and its two children; their four children head the W, N, S, E arms
    r, L, R = root, root["l"], root["r"]
    r["xy"] = (0, 0); L["xy"] = (-190, -120); R["xy"] = (190, 120)
    for n, arm in ((L["l"], "W"), (L["r"], "N"), (R["l"], "S"), (R["r"], "E")):
        place_arm(n, arm)
    edges_out = []
    emit(r.copy() | {}, None)  # placeholder, replaced below
    out.clear()
    # emit arms first (edges under boxes), then top three nodes
    lab = lambda n, left: ("≤ or missing" if left else ">") if n["ml"] else ("≤" if left else "> or missing")
    for parent in (L, R):
        emit(parent["l"], parent, lab(parent, True)); emit(parent["r"], parent, lab(parent, False))
    edge(r["xy"], L["xy"], lab(r, True)); edge(r["xy"], R["xy"], lab(r, False))
    for n in (L, R, r):
        box(*n["xy"], text(n)[0])
    xs = [float(s.split('x="')[1].split('"')[0]) for s in out if s.startswith("<rect")]
    ys = [float(s.split('y="')[1].split('"')[0]) for s in out if s.startswith("<rect")]
    pad = 60
    x0, x1 = min(xs) - pad, max(xs) + LW + pad
    y0, y1 = min(ys) - pad - 40, max(ys) + BH + pad
    title = (f"Round 1, tree for class {t} ({CLS[t]}): each leaf adds its value to the {CLS[t]} score; "
             f"“≤” = value at most the threshold, “>” = above it; “missing” = no value")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.0f} {y0:.0f} {x1-x0:.0f} {y1-y0:.0f}" '
           f'width="{x1-x0:.0f}" height="{y1-y0:.0f}"><rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="#fff"/>'
           f'<text x="{x0+20:.0f}" y="{y0+28:.0f}" font-size="18" font-family="sans-serif" fill="#111">{escape(title)}</text>'
           + "".join(out) + "</svg>")
    return svg

os.makedirs(outdir, exist_ok=True)
for t in range(4):
    p = os.path.join(outdir, f"round1_class{t}_{CLS[t]}.svg")
    open(p, "w").write(draw_tree(t)); print(p, os.path.getsize(p))
