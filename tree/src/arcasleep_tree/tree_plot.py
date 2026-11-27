"""Makes a picture of trees from a trained Arcasleep LightGBM model.

Being binary trees, if they have even depth then the leaf count is divisible by 4.
Thus, we space these around the 4 edges of the image produced per tree.
"""


from pathlib import Path
from xml.sax.saxutils import escape

CLS = ["wake", "light", "deep", "REM"]
BW, BH, LW = 150, 36, 70      # box width/height, leaf width
SLOT = 160                    # spacing between leaves across an arm (N/S)
SLOT_EW = 44                  # spacing between leaves across an arm (W/E)
LEVEL_NS, LEVEL_EW = 90, 190  # spacing between levels along an arm
D = 400                       # arm root distance from centre

ARROW_SVG = """
  <defs>
    <!-- Define the arrowhead marker -->
    <marker 
      id="arrowhead" 
      markerWidth="10" 
      markerHeight="7" 
      refX="10" 
      refY="3.5" 
      orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="black" />
    </marker>
  </defs>
"""

LEAF_COLOR = "#eef5ee"
ROOT_COLOR = "#ffe77a"
MID_COLOR = "#ffffff"

def build(m, t, i):
    """Collapse always-left padding; return nested node dict."""
    if i >= 63:
        return {"leaf": m["trees"]["leaf"][t][i - 63] * m["leaf_scale"]}
    f, th = int(m["trees"]["feature"][t][i]), int(m["trees"]["threshold"][t][i])
    miss_left = f >= 128
    f %= 128
    if th >= len(m["real_edges"][f]):
        return build(m, t, 2 * i + 1)
    return {"f": m["features"][f], "thr": m["real_edges"][f][th], "ml": miss_left,
            "l": build(m, t, 2 * i + 1), "r": build(m, t, 2 * i + 2)}

def lay(n, depth, cursor):
    """Assign n['lvl'] and n['pos'] (in leaf slots)."""
    n["lvl"] = depth
    if "leaf" in n:
        n["pos"] = cursor[0]; cursor[0] += 1; return
    lay(n["l"], depth + 1, cursor); lay(n["r"], depth + 1, cursor)
    n["pos"] = (n["l"]["pos"] + n["r"]["pos"]) / 2

def draw_tree(m, t):
    root = build(m, t, 0)
    out = []
    def box(x, y, lines, leaf=False, root=False):
        w = LW if leaf else BW
        fill = {
            (False, True): ROOT_COLOR,
            (True, True): ROOT_COLOR,
            (True, False): LEAF_COLOR,
            (False, False): MID_COLOR,
        }[(leaf, root)]
        s = f'<rect x="{x-w/2:.1f}" y="{y-BH/2:.1f}" width="{w}" height="{BH}" rx="4" fill="{fill}" stroke="#333"/>'
        for k, ln in enumerate(lines):
            yy = y + (k - (len(lines) - 1) / 2) * 14 + 4
            s += f'<text x="{x:.1f}" y="{yy:.1f}" text-anchor="middle" font-size="11" font-family="monospace" fill="#111">{escape(ln)}</text>'
        out.append(s)
    def edge(p, q, label, hw=BW / 2):
        dx, dy = q[0] - p[0], q[1] - p[1]
        k = 1 - min(hw / abs(dx) if dx else 9, BH / 2 / abs(dy) if dy else 9)  # stop at q's box border
        out.append(f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{p[0] + k * dx:.1f}" y2="{p[1] + k * dy:.1f}" stroke="#666" marker-end="url(#arrowhead)" />')
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        out.append(f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#a00" stroke="#fff" stroke-width="4" paint-order="stroke">{label}</text>')
    def text(n):
        return ([f"{n['f']}", f"≤ {n['thr']:.3g} ?"], False) if "leaf" not in n else ([f"{n['leaf']:+.3f}"], True)

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
        if parent: edge(parent["xy"], n["xy"], label, (LW if "leaf" in n else BW) / 2)
        lines, leaf = text(n)
        if "leaf" not in n:
            emit(n["l"], n, "≤ or missing" if n["ml"] else "≤"); emit(n["r"], n, ">" if n["ml"] else "> or missing")
        box(x, y, lines, leaf)

    # root and its two children; their four children head the W, N, S, E arms
    r, L, R = root, root["l"], root["r"]
    r["xy"] = (0, 0); L["xy"] = (-190, -120); R["xy"] = (190, 120)
    ROOT_KEY = "is_root"
    r[ROOT_KEY] = True; L[ROOT_KEY] = False; R[ROOT_KEY] = False
    for n, arm in ((L["l"], "W"), (L["r"], "N"), (R["l"], "S"), (R["r"], "E")):
        place_arm(n, arm)
    # emit arms first (edges under boxes), then top three nodes
    lab = lambda n, left: ("≤ or missing" if left else ">") if n["ml"] else ("≤" if left else "> or missing")
    for parent in (L, R):
        emit(parent["l"], parent, lab(parent, True)); emit(parent["r"], parent, lab(parent, False))
    edge(r["xy"], L["xy"], lab(r, True)); edge(r["xy"], R["xy"], lab(r, False))
    for n in (L, R, r):
        box(*n["xy"], text(n)[0], root=n[ROOT_KEY])
    xs = [float(s.split('x="')[1].split('"')[0]) for s in out if s.startswith("<rect")]
    ys = [float(s.split('y="')[1].split('"')[0]) for s in out if s.startswith("<rect")]
    pad = 60
    x0, x1 = min(xs) - pad, max(xs) + LW + pad
    y0, y1 = min(ys) - pad - 40, max(ys) + BH + pad
    title = (f"Round {t // 4}, {CLS[t % 4]} tree: each leaf adds its value to the {CLS[t % 4]} score; "
             f"“≤” = value at most the threshold, “>” = above it; “missing” = no value")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.0f} {y0:.0f} {x1-x0:.0f} {y1-y0:.0f}" '
           f'width="{x1-x0:.0f}" height="{y1-y0:.0f}"><rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="#fff"/>'
           f'{ARROW_SVG}'
           f'<text x="{x0+20:.0f}" y="{y0+28:.0f}" font-size="18" font-family="sans-serif" fill="#111">{escape(title)}</text>'
           + "".join(out) + "</svg>")
    return svg

def plot(m, rounds, classes, outdir):
    """Write one SVG per (round, class) to ``outdir``; return the paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in rounds:
        for c in classes:
            paths.append(outdir / f"round{r}_class{c}_{CLS[c]}.svg")
            paths[-1].write_text(draw_tree(m, 4 * r + c))
    return paths
