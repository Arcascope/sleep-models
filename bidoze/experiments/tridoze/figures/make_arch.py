"""Draw the five TriDoze architecture diagrams (release, gap handling, T1, T2, T3)."""
from arch_lib import arrow, box, canvas, label, save

# Placeholder for the BiDoze-XL parameter ratio; change this one string.
XL_PARAM_TEXT = "≈ N× release parameters"


def input_block(ax):
    box(ax, 50, 94, 56, 7, "Input: 3,000 epochs × 122\n2 channels × (60 frequency rows + gap row)", "input", 8.5)
    box(ax, 50, 83, 56, 7, "Gap handling\n(see gap-handling diagram)", "input", 8.5)
    arrow(ax, 50, 90.5, 50, 86.5)


def release():
    fig, ax = canvas()
    box(ax, 50, 93, 56, 8, "Input: 3,000 epochs × 122\n2 channels × (60 frequency rows + gap row)", "input", 8.5)
    xs = [17, 50, 83]
    for i, x in enumerate(xs):
        box(ax, x, 53, 30, 62, "", "plain", dashed=True, fill=False, z=0)
        label(ax, x, 80, f"BiDoze network {i + 1}", 8.5, bold=True, color="#1f2328")
        label(ax, x, 76.5, "trained alone", 7.5)
        for y, t, k in [(68, "Conv bank\nkernels 3, 7, 31, 63", "trunk"),
                        (56, "Bidirectional\ntransformer", "trunk"),
                        (44, "Gated fusion", "trunk"),
                        (33, "Head: 5-way softmax\nincl. learned gap", "head")]:
            box(ax, x, y, 25, 8, t, k, 7.5)
        for y1, y2 in [(64, 60), (52, 48), (40, 37)]:
            arrow(ax, x, y1, x, y2)
        arrow(ax, 50 + (x - 50) * 0.5, 89, x, 84.2)
        arrow(ax, x, 29, 50 + (x - 50) * 0.3, 18.2)
    box(ax, 50, 15, 40, 6.5, "Mean of the 3 softmax outputs", "head", 8.5)
    arrow(ax, 50, 11.7, 50, 8.5)
    box(ax, 50, 5, 70, 6.5, "Adult decode: wake/sleep, stage divisor, GGIR-style filter", "plain", 8)
    save(fig, "tridoze-arch-release.png")


def gap():
    fig, ax = canvas()
    box(ax, 50, 94, 40, 6, "Gap row of each epoch", "input", 9, bold=True)
    box(ax, 25, 60, 40, 44, "", "accent")
    label(ax, 25, 77, "Gap row = 1: epoch is absent", 8.5, color="#1f2328", bold=True)
    for i, t in enumerate(["Input is zeroed", "Hidden from attention (masked as a key)",
                           "Left out of the loss", "Output is set to gap"]):
        box(ax, 25, 69 - i * 8, 34, 6, t, "plain", 8)
    box(ax, 75, 60, 40, 44, "", "accent")
    label(ax, 75, 77, "Gap row < 1: epoch has data", 8.5, color="#1f2328", bold=True)
    box(ax, 75, 66, 34, 8, "Model outputs wake, light,\ndeep or REM", "plain", 8)
    box(ax, 75, 54, 34, 8, "Gap output is fixed at 0", "plain", 8)
    arrow(ax, 40, 91, 25, 82.5)
    arrow(ax, 60, 91, 75, 82.5)
    box(ax, 50, 28, 62, 9, "BiDoze network\nconv bank → bidirectional transformer → fusion → head", "trunk", 8.5)
    arrow(ax, 25, 38, 40, 32.7)
    arrow(ax, 75, 38, 60, 32.7)
    arrow(ax, 50, 23.5, 50, 18.7)
    box(ax, 50, 14.5, 62, 8, "Output: wake, light, deep, REM, gap", "head", 8.5)
    label(ax, 50, 4.5, "No learned gap class: gap comes only from the gap row", 8.5, color="#1f2328", bold=True)
    save(fig, "tridoze-arch-gap.png")


def t1():
    fig, ax = canvas()
    input_block(ax)
    xs = [17, 50, 83]
    box(ax, 50, 56, 96, 36, "", "accent", dashed=True, fill=False, z=0)
    label(ax, 50, 71, "3 networks trained in one fit", 8.5, color="#1f2328", bold=True)
    arrow(ax, 50, 79.5, 50, 74.2)
    for i, x in enumerate(xs):
        box(ax, x, 62, 26, 8, f"BiDoze network {i + 1}", "trunk", 7.5)
        box(ax, x, 48, 26, 9, "4-stage softmax\n(each a full model)", "head", 7.5)
        arrow(ax, x, 58, x, 52.7)
        arrow(ax, x, 43.5, 50 + (x - 50) * 0.3, 33.2)
    box(ax, 50, 30, 40, 6, "Mean of the 3 softmax outputs", "head", 8.5)
    arrow(ax, 50, 27, 50, 22.5)
    box(ax, 50, 17, 92, 11, "Loss\n(1 − β) × CE(mean) + β × mean CE(member),  β = 0.5\n"
        "the member term applies to each network's own softmax", "accent", 8.5)
    arrow(ax, 50, 11.5, 50, 7)
    box(ax, 50, 4, 40, 5, "Adult decode", "plain", 8)
    save(fig, "tridoze-arch-t1.png")


def t2():
    fig, ax = canvas()
    input_block(ax)
    xs = [17, 50, 83]
    info = [("W", "wake vs sleep\n(W alone is a\nsleep/wake model)"), ("D", "deep vs light/REM,\non sleep"), ("R", "REM vs light,\non light/REM")]
    box(ax, 50, 59, 96, 34, "", "accent", dashed=True, fill=False, z=0)
    label(ax, 50, 73, "3 BiDoze trunks, each ending in one logit", 8.5, color="#1f2328", bold=True)
    arrow(ax, 50, 79.5, 50, 76.2)
    for (n, d), x in zip(info, xs):
        box(ax, x, 66, 26, 6, f"BiDoze trunk {n}", "trunk", 7.5)
        box(ax, x, 54, 26, 11, f"logit {n.lower()}\n{n}: {d}", "head", 7.5)
        arrow(ax, x, 63, x, 59.7)
        arrow(ax, x, 48.5, 50 + (x - 50) * 0.35, 39.2)
    box(ax, 50, 34.5, 92, 9, "p(wake) = σ(w)        p(deep) = (1 − σ(w)) σ(d)\n"
        "p(REM) = (1 − σ(w)) (1 − σ(d)) σ(r)        p(light) = the rest", "head", 8)
    box(ax, 50, 20, 92, 8, "Loss: CE(4-stage) + λ × Σ branch binary CE,  λ = 0.5", "accent", 8.5)
    arrow(ax, 50, 30, 50, 24.2)
    arrow(ax, 50, 15.8, 50, 9.2)
    box(ax, 50, 6, 40, 6, "Adult decode", "plain", 8)
    save(fig, "tridoze-arch-t2.png")


def t3():
    fig, ax = canvas()
    input_block(ax)
    box(ax, 50, 45.5, 62, 53, "", "accent", dashed=True, fill=False, z=0)
    label(ax, 50, 69, "BiDoze-XL: one network, wider", 9, color="#1f2328", bold=True)
    arrow(ax, 50, 79.5, 50, 72.2)
    box(ax, 50, 61, 52, 7, "Conv bank\nkernels 3, 7, 31, 63", "trunk", 8)
    box(ax, 50, 47.5, 52, 15, "Bidirectional transformer\n4 layers per direction (as release)\n"
        "d_model 512 (release 256)\nd_ff 1,024 (release 512)", "accent", 8)
    box(ax, 50, 35, 52, 7, "Gated fusion", "trunk", 8)
    box(ax, 50, 25, 52, 6, "Head: 4-stage softmax (+ fixed gap)", "head", 8)
    arrow(ax, 50, 67, 50, 64.7)
    arrow(ax, 50, 57.5, 50, 55.3)
    arrow(ax, 50, 40, 50, 38.7)
    arrow(ax, 50, 31.5, 50, 28.3)
    label(ax, 50, 13.5, XL_PARAM_TEXT, 9.5, color="#1f2328", bold=True)
    arrow(ax, 50, 22, 50, 16.3)
    arrow(ax, 50, 10.7, 50, 8.7)
    box(ax, 50, 5.5, 40, 5, "Adult decode", "plain", 8)
    save(fig, "tridoze-arch-t3.png")


if __name__ == "__main__":
    release(); gap(); t1(); t2(); t3()
