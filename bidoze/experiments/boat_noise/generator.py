"""Random boat-like noise for the raw accelerometer: a parametric family, not a bank of templates.

``sample_spec(rng)`` draws one night's noise: a list of components whose parameters are drawn
from ranges wider than those measured on SURF and Somerset (log §3–4). ``render(spec, accel,
rng)`` adds it to a ``(t, x, y, z)`` recording in g, before any of the model's processing, so the
jerk's folding of energy above 6 Hz (log §4) is reproduced.

Component kinds (zoology types in brackets):

- ``floor`` [the raised floor of §4]: Gaussian noise with PSD ∝ f^−slope over 0.1 Hz–Nyquist,
  independent on each axis (``iso``) or along one direction;
- ``line`` [2, 3, 6, 7, 8, 9]: a sinusoid anywhere from 0.3 Hz to 0.95 × Nyquist; its frequency
  wanders slowly (1–10 min scales) and wobbles (5–60 s period);
- ``comb`` [4, 5, 10]: harmonics of a 0.4–3 Hz fundamental up to Nyquist, amplitude k^−decay,
  sharing the fundamental's wobble;
- ``pulses`` [4]: short Gaussian pulses at 0.3–3 Hz with jittered spacing;
- ``hull`` [1]: band-limited noise centred at 0.1–0.6 Hz.

Every component except ``iso`` floors has one direction, fixed relative to gravity (angle from
the gravity axis and an azimuth in a gravity-derived frame), so it reaches |a| along gravity and
the jerk across it. Every component except the floor is switched on and off through the night
(alternating on and off spells, each lasting 1 min to 3 h, with 5-s ramps) and its amplitude
drifts by up to ±6 dB on 1–10 min scales. The floor is constant through the night, like a
sensor's noise floor: SURF's Galaxy Watch floor is the same on every night and boat.

RMS amplitudes are of the noise vector, in mg while the component is on, drawn log-uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RAMP_S = 5.0
RANGES = {
    "floor_mg": (0.5, 20.0),
    "floor_slope": (-0.5, 1.5),
    "line_mg": (0.3, 10.0),
    "line_count": (0, 6),
    "comb_mg": (0.3, 10.0),
    "comb_f0": (0.4, 3.0),
    "comb_decay": (0.5, 2.0),
    "pulse_mg": (0.3, 10.0),
    "pulse_rate": (0.3, 3.0),
    "pulse_width_s": (0.005, 0.05),
    "hull_mg": (1.0, 30.0),
    "hull_centre": (0.1, 0.6),
    "on_s": (60.0, 3 * 3600.0),
    "envelope_db": (0.0, 6.0),
}
P_PRESENT = {"floor": 0.8, "comb": 0.4, "pulses": 0.3, "hull": 0.6}


@dataclass
class Component:
    kind: str
    rms_mg: float
    params: dict = field(default_factory=dict)
    direction: tuple[float, float] | None = None  # (angle from gravity, azimuth) in radians; None: iso
    seed: int = 0


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def direction(rng: np.random.Generator) -> tuple[float, float]:
    return float(np.arccos(rng.uniform(-1, 1))), float(rng.uniform(0, 2 * np.pi))


def sample_spec(rng: np.random.Generator, nyquist: float = 25.0, ranges: dict = RANGES) -> list[Component]:
    """One night's components."""
    spec = []

    def seed() -> int:
        return int(rng.integers(2**31))

    if rng.random() < P_PRESENT["floor"]:
        iso = rng.random() < 0.5
        spec.append(Component("floor", log_uniform(rng, *ranges["floor_mg"]),
                              {"slope": float(rng.uniform(*ranges["floor_slope"]))},
                              None if iso else direction(rng), seed()))
    for _ in range(int(rng.integers(ranges["line_count"][0], ranges["line_count"][1] + 1))):
        spec.append(Component("line", log_uniform(rng, *ranges["line_mg"]),
                              {"f0": log_uniform(rng, 0.3, 0.95 * nyquist),
                               "wander": float(rng.uniform(0, 0.05)),
                               "wobble_hz": float(rng.uniform(0, 0.2)),
                               "wobble_period_s": log_uniform(rng, 5, 60)},
                              direction(rng), seed()))
    if rng.random() < P_PRESENT["comb"]:
        spec.append(Component("comb", log_uniform(rng, *ranges["comb_mg"]),
                              {"f0": log_uniform(rng, *ranges["comb_f0"]),
                               "decay": float(rng.uniform(*ranges["comb_decay"])),
                               "wobble_hz": float(rng.uniform(0, 0.1)),
                               "wobble_period_s": log_uniform(rng, 5, 60)},
                              direction(rng), seed()))
    if rng.random() < P_PRESENT["pulses"]:
        spec.append(Component("pulses", log_uniform(rng, *ranges["pulse_mg"]),
                              {"rate": log_uniform(rng, *ranges["pulse_rate"]),
                               "width_s": log_uniform(rng, *ranges["pulse_width_s"]),
                               "jitter": float(rng.uniform(0, 0.1))},
                              direction(rng), seed()))
    if rng.random() < P_PRESENT["hull"]:
        centre = float(rng.uniform(*ranges["hull_centre"]))
        spec.append(Component("hull", log_uniform(rng, *ranges["hull_mg"]),
                              {"centre": centre, "width": float(rng.uniform(0.2, 0.8)) * centre},
                              (float(rng.uniform(0, 0.5)), float(rng.uniform(0, 2 * np.pi))), seed()))
    for c in spec:
        c.params["mean_on_s"] = log_uniform(rng, *ranges["on_s"])
        c.params["mean_off_s"] = log_uniform(rng, *ranges["on_s"])
        c.params["envelope_db"] = float(rng.uniform(*ranges["envelope_db"]))
    return spec


def slow_path(t: np.ndarray, rng: np.random.Generator, lo_s: float = 60.0, hi_s: float = 600.0) -> np.ndarray:
    """A smooth path in [−1, 1] varying on ``lo_s``–``hi_s`` scales."""
    periods = rng.uniform(lo_s, hi_s, 8)
    w = sum(np.sin(2 * np.pi * t / p + rng.uniform(0, 2 * np.pi)) for p in periods)
    return w / np.abs(w).max()


def shaped_noise(n: int, fs: float, rng: np.random.Generator, gain) -> np.ndarray:
    """Gaussian noise of ``n`` samples whose amplitude spectrum is ``gain(f)``, unit RMS."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    x = np.fft.irfft(spectrum * gain(f), n)
    return x / np.sqrt(np.mean(x**2))


def gate(t: np.ndarray, mean_on: float, mean_off: float, rng: np.random.Generator) -> np.ndarray:
    """1 while on and 0 while off, alternating exponential spells with linear ramps of ``RAMP_S``."""
    edges, state, now = [t[0]], bool(rng.random() < mean_on / (mean_on + mean_off)), t[0]
    states = [state]
    while now < t[-1]:
        now += rng.exponential(mean_on if state else mean_off)
        state = not state
        edges.append(now)
        states.append(state)
    level = np.asarray(states, float)[np.searchsorted(edges, t, side="right") - 1]
    grid = np.arange(t[0], t[-1] + 1.0, 1.0)
    coarse = np.interp(grid, t, level)
    k = int(RAMP_S)
    coarse = np.convolve(np.pad(coarse, k, mode="edge"), np.ones(k) / k, mode="same")[k:-k]
    return np.interp(t, grid, coarse)


def waveform(c: Component, t: np.ndarray, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS signal of one component, before gating."""
    p = c.params
    if c.kind == "floor":
        return shaped_noise(len(t), fs, rng, lambda f: np.where(f >= 0.1, np.maximum(f, 0.1) ** (-p["slope"] / 2), 0))
    if c.kind == "hull":
        lo, hi = p["centre"] - p["width"] / 2, p["centre"] + p["width"] / 2
        return shaped_noise(len(t), fs, rng, lambda f: ((f >= lo) & (f <= hi)).astype(float))
    if c.kind in ("line", "comb"):
        f = p["f0"] * (1 + p.get("wander", 0.0) * slow_path(t, rng))
        f = f + p["wobble_hz"] * np.sin(2 * np.pi * t / p["wobble_period_s"] + rng.uniform(0, 2 * np.pi))
        phase = 2 * np.pi * np.cumsum(f * np.diff(t, prepend=t[0])) + rng.uniform(0, 2 * np.pi)
        if c.kind == "line":
            x = np.sin(phase)
        else:
            nyquist = fs / 2
            x = sum(np.sin(k * phase + rng.uniform(0, 2 * np.pi)) / k ** p["decay"]
                    for k in range(1, int(0.95 * nyquist / p["f0"]) + 1))
    elif c.kind == "pulses":
        spacing = 1.0 / p["rate"]
        n = int((t[-1] - t[0]) / spacing) + 2
        centres = t[0] + np.cumsum(spacing * (1 + p["jitter"] * rng.standard_normal(n)))
        nearest = centres[np.clip(np.searchsorted(centres, t), 0, n - 1)]
        before = centres[np.clip(np.searchsorted(centres, t) - 1, 0, n - 1)]
        d = np.minimum(np.abs(t - nearest), np.abs(t - before))
        x = np.exp(-0.5 * (d / p["width_s"]) ** 2)
        x = x - x.mean()
    else:
        raise ValueError(c.kind)
    return x / np.sqrt(np.mean(x**2))


def gravity_frame(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample unit vectors: along gravity, and two perpendicular to it."""
    up = a / np.linalg.norm(a, axis=1, keepdims=True)
    e1 = np.cross(up, [1.0, 0.0, 0.0])
    near = np.linalg.norm(e1, axis=1) < 0.3
    e1[near] = np.cross(up[near], [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return up, e1, np.cross(up, e1)


def render(spec: list[Component], accel: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """``accel`` (``(t, x, y, z)`` in g, gravity included) with every component added."""
    t, a = accel[:, 0], accel[:, 1:]
    fs = 1.0 / np.median(np.diff(t))
    up, e1, e2 = gravity_frame(a)
    out = accel.copy()
    for c in spec:
        r = np.random.default_rng(c.seed)
        x = waveform(c, t, fs, r)
        if c.kind == "floor":
            amp = np.full(len(t), c.rms_mg / 1000.0)
        else:
            amp = c.rms_mg / 1000.0 * gate(t, c.params["mean_on_s"], c.params["mean_off_s"], r)
            amp = amp * 10 ** (c.params["envelope_db"] * slow_path(t, r) / 20)
        if c.direction is None:
            noise = np.stack([x] + [waveform(c, t, fs, r) for _ in range(2)], axis=1) / np.sqrt(3)
        else:
            theta, phi = c.direction
            d = np.cos(theta) * up + np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2)
            noise = x[:, None] * d
        out[:, 1:] += amp[:, None] * noise
    return out
