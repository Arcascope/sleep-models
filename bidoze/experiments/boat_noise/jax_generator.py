"""``generator.py``'s noise family in JAX: drawn and rendered on the device inside a training step.

The components, ranges and presence probabilities are ``generator.py``'s. The differences:

- one night's spec is a fixed-shape pytree (``MAX_LINES`` line slots, one comb, one pulse train,
  one hull, one floor), each slot with an ``on`` flag, so draws can be ``vmap``-ed and ``jit``-ed;
- time is the sample index divided by the median sampling rate, as ``generator.py`` already
  assumes for the floor and hull; only the gates use the timestamps;
- phases and pulse positions are accumulated as (integer, fraction) pairs with an associative
  scan, so they stay exact over a night in float32.

``render(key, spec, t, xyz, n, fs)`` adds the noise to the first ``n`` samples of ``xyz``
(``[N, 3]``, in g, gravity included); ``t`` is seconds from the first sample.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from generator import P_PRESENT, RAMP_S, RANGES

MAX_LINES = int(RANGES["line_count"][1])
# Harmonics up to 0.95 × Nyquist of the lowest comb fundamental at 50 Hz sampling; higher
# rates keep this many.
MAX_HARMONICS = int(0.95 * 25.0 / RANGES["comb_f0"][0]) + 1
MAX_SPELLS = 4096  # on/off spells per component and night; the shortest mean spell is 60 s
GATE_GRID_S = 1.0
SLOW_PATH_TERMS = 8


def _log_uniform(key, lo, hi, shape=()):
    return jnp.exp(jax.random.uniform(key, shape, minval=jnp.log(lo), maxval=jnp.log(hi)))


def _direction(key, shape=()):
    k1, k2 = jax.random.split(key)
    return jnp.stack([jnp.arccos(jax.random.uniform(k1, shape, minval=-1.0, maxval=1.0)),
                      jax.random.uniform(k2, shape, maxval=2 * jnp.pi)], axis=-1)


def _gating(key, shape=()):
    k = jax.random.split(key, 3)
    return {"mean_on_s": _log_uniform(k[0], *RANGES["on_s"], shape),
            "mean_off_s": _log_uniform(k[1], *RANGES["on_s"], shape),
            "envelope_db": jax.random.uniform(k[2], shape, minval=RANGES["envelope_db"][0],
                                              maxval=RANGES["envelope_db"][1])}


def sample_spec(key: jax.Array, nyquist: jax.Array) -> dict:
    """One night's components, as ``generator.sample_spec`` draws them."""
    k = iter(jax.random.split(key, 64))
    u = lambda lo, hi, shape=(): jax.random.uniform(next(k), shape, minval=lo, maxval=hi)
    n_lines = jax.random.randint(next(k), (), RANGES["line_count"][0], RANGES["line_count"][1] + 1)
    return {
        "floor": {"on": jax.random.bernoulli(next(k), P_PRESENT["floor"]),
                  "iso": jax.random.bernoulli(next(k), 0.5),
                  "rms_mg": _log_uniform(next(k), *RANGES["floor_mg"]),
                  "slope": u(*RANGES["floor_slope"]),
                  "direction": _direction(next(k))},
        "line": {"on": jnp.arange(MAX_LINES) < n_lines,
                 "rms_mg": _log_uniform(next(k), *RANGES["line_mg"], (MAX_LINES,)),
                 "f0": jnp.exp(u(jnp.log(0.3), jnp.log(0.95 * nyquist), (MAX_LINES,))),
                 "wander": u(0.0, 0.05, (MAX_LINES,)),
                 "wobble_hz": u(0.0, 0.2, (MAX_LINES,)),
                 "wobble_period_s": _log_uniform(next(k), 5.0, 60.0, (MAX_LINES,)),
                 "direction": _direction(next(k), (MAX_LINES,))} | _gating(next(k), (MAX_LINES,)),
        "comb": {"on": jax.random.bernoulli(next(k), P_PRESENT["comb"]),
                 "rms_mg": _log_uniform(next(k), *RANGES["comb_mg"]),
                 "f0": _log_uniform(next(k), *RANGES["comb_f0"]),
                 "decay": u(*RANGES["comb_decay"]),
                 "wobble_hz": u(0.0, 0.1),
                 "wobble_period_s": _log_uniform(next(k), 5.0, 60.0),
                 "direction": _direction(next(k))} | _gating(next(k)),
        "pulses": {"on": jax.random.bernoulli(next(k), P_PRESENT["pulses"]),
                   "rms_mg": _log_uniform(next(k), *RANGES["pulse_mg"]),
                   "rate": _log_uniform(next(k), *RANGES["pulse_rate"]),
                   "width_s": _log_uniform(next(k), *RANGES["pulse_width_s"]),
                   "jitter": u(0.0, 0.1),
                   "direction": _direction(next(k))} | _gating(next(k)),
        "hull": {"on": jax.random.bernoulli(next(k), P_PRESENT["hull"]),
                 "rms_mg": _log_uniform(next(k), *RANGES["hull_mg"]),
                 "centre": (centre := u(*RANGES["hull_centre"])),
                 "width": u(0.2, 0.8) * centre,
                 "direction": jnp.stack([u(0.0, 0.5), u(0.0, 2 * jnp.pi)])} | _gating(next(k)),
    }


def _wrapped_cumsum(steps: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Cumulative sum as (integer part, fraction in [0, 1)), exact to float32 in the fraction."""
    whole = jnp.floor(steps)

    def add(a, b):
        f = a[1] + b[1]
        carry = jnp.floor(f)
        return a[0] + b[0] + carry.astype(jnp.int32), f - carry

    return jax.lax.associative_scan(add, (whole.astype(jnp.int32), steps - whole))


def _unit_rms(x: jax.Array, valid: jax.Array) -> jax.Array:
    v = valid.astype(x.dtype)
    return x / jnp.sqrt(jnp.maximum(jnp.sum(x**2 * v) / jnp.maximum(v.sum(), 1.0), 1e-30))


def _slow_path(key, t, valid, lo_s=60.0, hi_s=600.0):
    k1, k2 = jax.random.split(key)
    periods = jax.random.uniform(k1, (SLOW_PATH_TERMS,), minval=lo_s, maxval=hi_s)
    phases = jax.random.uniform(k2, (SLOW_PATH_TERMS,), maxval=2 * jnp.pi)
    w = jnp.sin(2 * jnp.pi * jnp.fmod(t[:, None] / periods, 1.0) + phases).sum(axis=1)
    return w / jnp.maximum(jnp.max(jnp.where(valid, jnp.abs(w), 0.0)), 1e-6)


def _shaped_noise(key, n_total, fs, valid, gain):
    spectrum = jnp.fft.rfft(jax.random.normal(key, (n_total,)))
    f = jnp.arange(spectrum.shape[0]) * fs / n_total
    return _unit_rms(jnp.fft.irfft(spectrum * gain(f), n_total), valid)


def _cycles(key, f0, wander, wobble_hz, wobble_period_s, s, fs, valid):
    """Phase in cycles (fraction only) of a tone whose frequency wanders and wobbles."""
    k1, k2, k3 = jax.random.split(key, 3)
    f = f0 * (1.0 + wander * _slow_path(k1, s, valid))
    f = f + wobble_hz * jnp.sin(2 * jnp.pi * jnp.fmod(s / wobble_period_s, 1.0)
                                + jax.random.uniform(k2, maxval=2 * jnp.pi))
    steps = jnp.concatenate([jnp.zeros(1), f[1:] / fs])
    return jnp.fmod(_wrapped_cumsum(steps)[1] + jax.random.uniform(k3), 1.0)


def _gate(key, t, valid, mean_on, mean_off):
    """1 while on, 0 while off: alternating exponential spells with ``RAMP_S`` linear ramps."""
    k1, k2 = jax.random.split(key)
    start_on = jax.random.uniform(k1) < mean_on / (mean_on + mean_off)
    parity = (jnp.arange(MAX_SPELLS) % 2 == 0) == start_on  # True: this spell is on
    lengths = jax.random.exponential(k2, (MAX_SPELLS,)) * jnp.where(parity, mean_on, mean_off)
    ends = jnp.cumsum(lengths)
    grid = jnp.arange(int(np.ceil(3000 * 30.0 / GATE_GRID_S)) + 2) * GATE_GRID_S
    level = parity[jnp.minimum(jnp.searchsorted(ends, grid, side="right"), MAX_SPELLS - 1)].astype(t.dtype)
    width = int(RAMP_S / GATE_GRID_S)
    c = jnp.concatenate([jnp.zeros(1), jnp.cumsum(jnp.pad(level, (width // 2, width - 1 - width // 2), mode="edge"))])
    smooth = (c[width:] - c[:-width]) / width
    return jnp.interp(t, grid, smooth)


def _pulse_train(key, rate, width_s, jitter, n_total, fs, valid):
    """Gaussian pulses at ``rate`` per second with jittered spacing, in sample units."""
    spacing = fs / rate
    count = n_total  # an upper bound on the pulses a night can hold at the shortest spacing ≥ 1 sample
    steps = spacing * (1.0 + jitter * jax.random.normal(key, (count,)))
    whole, frac = _wrapped_cumsum(jnp.maximum(steps, 1.0))
    n = jnp.arange(n_total, dtype=jnp.int32)
    after = jnp.clip(jnp.searchsorted(whole, n, side="left"), 0, count - 1)
    before = jnp.clip(after - 1, 0, count - 1)
    d_after = (whole[after] - n).astype(jnp.float32) + frac[after]
    d_before = (n - whole[before]).astype(jnp.float32) - frac[before]
    d = jnp.minimum(jnp.abs(d_after), jnp.abs(d_before)) / fs
    x = jnp.exp(-0.5 * (d / width_s) ** 2)
    v = valid.astype(x.dtype)
    return _unit_rms(x - jnp.sum(x * v) / jnp.maximum(v.sum(), 1.0), valid)


def _gravity_frame(a: jax.Array):
    norm = jnp.linalg.norm(a, axis=1, keepdims=True)
    up = jnp.where(norm > 0, a / jnp.where(norm > 0, norm, 1.0), jnp.array([0.0, 0.0, 1.0]))
    e1 = jnp.cross(up, jnp.array([1.0, 0.0, 0.0]))
    e1 = jnp.where(jnp.linalg.norm(e1, axis=1, keepdims=True) < 0.3, jnp.cross(up, jnp.array([0.0, 1.0, 0.0])), e1)
    e1 = e1 / jnp.linalg.norm(e1, axis=1, keepdims=True)
    return up, e1, jnp.cross(up, e1)


def _along(direction, frame):
    up, e1, e2 = frame
    theta, phi = direction[0], direction[1]
    return jnp.cos(theta) * up + jnp.sin(theta) * (jnp.cos(phi) * e1 + jnp.sin(phi) * e2)


def render(key: jax.Array, spec: dict, t: jax.Array, xyz: jax.Array, n: jax.Array, fs: jax.Array) -> jax.Array:
    """``xyz`` with ``spec``'s noise added to its first ``n`` samples."""
    n_total = xyz.shape[0]
    valid = jnp.arange(n_total) < n
    s = jnp.arange(n_total, dtype=jnp.float32) / fs  # seconds on the sample index
    frame = _gravity_frame(xyz)
    k = iter(jax.random.split(key, 16))
    noise = jnp.zeros_like(xyz)

    def gated(c, kk, x, direction):
        k1, k2 = jax.random.split(kk)
        amp = c["rms_mg"] / 1000.0 * _gate(k1, t, valid, c["mean_on_s"], c["mean_off_s"])
        amp = amp * 10 ** (c["envelope_db"] * _slow_path(k2, t, valid) / 20)
        return jnp.where(c["on"], amp, 0.0)[:, None] * x[:, None] * _along(direction, frame)

    floor = spec["floor"]
    kf = jax.random.split(next(k), 3)
    floor_gain = lambda f: jnp.where(f >= 0.1, jnp.maximum(f, 0.1) ** (-floor["slope"] / 2), 0.0)
    floor_x = jnp.stack([_shaped_noise(kf[i], n_total, fs, valid, floor_gain) for i in range(3)], axis=1)
    floor_noise = jnp.where(floor["iso"], floor_x / jnp.sqrt(3.0), floor_x[:, :1] * _along(floor["direction"], frame))
    noise = noise + jnp.where(floor["on"], floor["rms_mg"] / 1000.0, 0.0) * floor_noise

    line = spec["line"]
    for i in range(MAX_LINES):
        c = jax.tree.map(lambda v: v[i], line)
        kl = jax.random.split(next(k))
        x = jnp.sin(2 * jnp.pi * _cycles(kl[0], c["f0"], c["wander"], c["wobble_hz"], c["wobble_period_s"], s, fs, valid))
        noise = noise + gated(c, kl[1], _unit_rms(x, valid), c["direction"])

    comb = spec["comb"]
    kc = jax.random.split(next(k), 3)
    cycles = _cycles(kc[0], comb["f0"], 0.0, comb["wobble_hz"], comb["wobble_period_s"], s, fs, valid)
    harmonic_phases = jax.random.uniform(kc[1], (MAX_HARMONICS,))
    top = jnp.floor(0.95 * fs / 2 / comb["f0"])

    def harmonic(h, acc):
        kk = (h + 1).astype(jnp.float32)
        term = jnp.sin(2 * jnp.pi * jnp.fmod(kk * cycles + harmonic_phases[h], 1.0)) / kk ** comb["decay"]
        return acc + jnp.where(kk <= top, term, 0.0)

    comb_x = jax.lax.fori_loop(0, MAX_HARMONICS, harmonic, jnp.zeros(n_total))
    noise = noise + gated(comb, kc[2], _unit_rms(comb_x, valid), comb["direction"])

    pulses = spec["pulses"]
    kp = jax.random.split(next(k))
    pulse_x = _pulse_train(kp[0], pulses["rate"], pulses["width_s"], pulses["jitter"], n_total, fs, valid)
    noise = noise + gated(pulses, kp[1], pulse_x, pulses["direction"])

    hull = spec["hull"]
    kh = jax.random.split(next(k))
    lo, hi = hull["centre"] - hull["width"] / 2, hull["centre"] + hull["width"] / 2
    hull_x = _shaped_noise(kh[0], n_total, fs, valid, lambda f: ((f >= lo) & (f <= hi)).astype(jnp.float32))
    noise = noise + gated(hull, kh[1], hull_x, hull["direction"])

    return xyz + jnp.where(valid[:, None], noise, 0.0)
