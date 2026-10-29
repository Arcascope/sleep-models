"""Fixed-point streaming reference of the 87 WatchTree features, exactly as the Monkey C app runs them.

The watch gets 25 Hz x/y/z accelerometer samples in milli-g integers, in batches, and a heart-rate
reading (integer bpm or none) once a second. :class:`Engine` consumes them with integer arithmetic
and fixed ring buffers only; ``Number`` in the comments is a 32-bit Monkey C integer (checked with
:func:`n32` where a product or sum could leave the range), ``Long`` a 64-bit one.

Differences from :mod:`~arcasleep_watchtree.features.table` (the float reference used for training):

* input: integer milli-g and integer bpm, not resampled float g (the cohorts are resampled
  offline; the watch gets 25 Hz natively);
* ``jerk``: the per-sample norm is the exact floor of the root in units of 1/16 mg;
* low-pass: the same two Butterworth sections, coefficients rounded to Q24, state in Q36, the
  section output truncated to Q12 mg (``tdiv``: division toward zero, as Monkey C does);
* angle: ``atan(z / sqrt(x^2 + y^2))`` from a 64-segment interpolated table (error about 0.001 deg),
  from 5-s block means rounded to 1/16 mg;
* ``daz``: from the epoch angle in 1/1024 degree (``angle`` itself is kept in 1/16 degree);
* ``lfm``: deviations in 1/64 mg, norms are exact integer roots;
* windows: running integer sums (mean) and a scan (maximum); a standard deviation is
  ``isqrt(256 * (n * sum2 - sum^2)) / n``; means round half away from zero;
* low baseline: block means rounded to integers;
* heart rate: the epoch is the one the batch's first sample is in; one reading per batch.

Integer units of every feature are in :mod:`~arcasleep_watchtree.watch.spec`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt_zi

from arcasleep_tree.features import constants as C
from arcasleep_tree.features.config import FeatureConfig
from arcasleep_tree.watch.spec import DELAY, NA, SIGNALS, spec

SPE = C.SAMPLES_PER_EPOCH  # 750
Q = 24  # coefficient bits
SOS = butter(4, C.LP1_HZ, fs=C.FS, output="sos")
COEF = [[int(round(v * (1 << Q))) for v in (s[0], s[1], s[2], s[4], s[5])] for s in SOS]  # b0 b1 b2 a1 a2
ZI = [[int(round(v * (1 << Q))) for v in row] for row in sosfilt_zi(SOS)]
BLOCK_LOW = 25  # 5 Hz samples in a 5-s block
BLOCKS_PER_EPOCH = SPE // (C.DECIMATE_1 * BLOCK_LOW)  # 6
STILL_Q10 = int(C.STILL_DEG * 1024)
BOUT_X = 3
BLOCK_EPOCHS, BASE_BLOCKS, BASE_LOWEST, BLOCK_MIN = (
    C.BLOCK_EPOCHS, C.BASELINE_BLOCKS, C.BASELINE_LOWEST, C.BLOCK_MIN_EPOCHS,
)  # fmt: skip
TSB_CAP = C.TSB_CAP_MIN * 2
MIN_VALID_SAMPLES = SPE // 2
MAX_GAP_PER_CALL = 200

#: atan(i / 64) in 1/1024 degree, i = 0..65 (65 repeats 64).
ATAN_LUT = [int(round(math.degrees(math.atan(min(i, 64) / 64.0)) * 1024)) for i in range(66)]


def n32(v: int) -> int:
    assert -(2**31) <= v < 2**31, v
    return v


def tdiv(a: int, b: int) -> int:
    """Division truncating toward zero, as Monkey C's ``/``."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def rdiv(a: int, b: int) -> int:
    """``a / b`` rounded to nearest, halves away from zero; ``b > 0``."""
    return (a + b // 2) // b if a >= 0 else -((-a + b // 2) // b)


def isqrt(v: int) -> int:
    return math.isqrt(v)


def atan_q10(z: int, h: int) -> int:
    """``atan2(z, h)`` in 1/1024 degree for ``h >= 0``; ``z`` and ``h`` below 2^18."""
    az = abs(z)
    if az == 0:
        return 0
    if az <= h:
        r = (az << 12) // h
        base = 0
        sign_flip = False
    else:
        r = (h << 12) // az
        base = 90 * 1024
        sign_flip = True
    i, f = r >> 6, r & 63
    ang = ATAN_LUT[i] + (((ATAN_LUT[i + 1] - ATAN_LUT[i]) * f + 32) >> 6)
    ang = base - ang if sign_flip else ang
    return -ang if z < 0 else ang


class Baseline:
    """Streaming low baseline: mean of the lowest 3 of the last 12 completed 10-min block means."""

    def __init__(self) -> None:
        self.blocks: list[int] = []
        self.bsum = self.bcount = self.psum = self.pcount = self.k = 0

    def push(self, v: int) -> int:
        if v != NA:
            self.bsum += v
            self.bcount += 1
            if not self.blocks:
                self.psum += v
                self.pcount += 1
        if self.blocks:
            low = sorted(self.blocks)[:BASE_LOWEST]
            base = rdiv(sum(low), len(low))
        elif self.pcount:
            base = rdiv(self.psum, self.pcount)
        else:
            base = NA
        self.k += 1
        if self.k % BLOCK_EPOCHS == 0:
            if self.bcount >= BLOCK_MIN:
                self.blocks.append(rdiv(self.bsum, self.bcount))
                self.blocks = self.blocks[-BASE_BLOCKS:]
            self.bsum = self.bcount = 0
        return base


@dataclass
class Model:
    """Trees as the watch holds them: ``edges[f]`` integer edges, per-tree byte arrays."""

    edges: list[list[int]]
    feat: np.ndarray  # (T, 63) feature bytes, bit 7 = missing goes left
    thr: np.ndarray  # (T, 63)
    leaf: np.ndarray  # (T, 64) int8 + 128
    rounds: int

    @staticmethod
    def load(model_json: dict, trees_bin: bytes) -> "Model":
        from arcasleep_tree.watch.export import unpack

        n = model_json["n_trees"]
        feat, thr, leaf = unpack(trees_bin, n)
        return Model(model_json["edges"], feat, thr, leaf + 128, model_json["rounds"])

    def bins(self, F: list[int]) -> list[int]:
        """The 256-entry lookup the trees read: ``x[f]`` (missing 15) and ``x[128 + f]`` (missing 0)."""
        x = [0] * 256
        for f, v in enumerate(F):
            if v == NA:
                x[f], x[128 + f] = 15, 0
            else:
                b = 0
                for e in self.edges[f]:
                    if e < v:
                        b += 1
                x[f] = x[128 + f] = b
        return x

    def scores(self, x: list[int]) -> list[int]:
        s = [0, 0, 0, 0]
        feat, thr, leaf = self.feat.tolist(), self.thr.tolist(), self.leaf.tolist()
        for t in range(len(feat)):
            ft, th = feat[t], thr[t]
            i = 0
            for _ in range(6):
                i = 2 * i + 1 if x[ft[i]] <= th[i] else 2 * i + 2
            s[t & 3] += leaf[t][i - 63]
        return [v - 128 * self.rounds for v in s]


class Engine:
    """The whole on-watch pipeline. Feed samples; read :attr:`out` (one dict per scored epoch)."""

    def __init__(self, config: FeatureConfig, model: Model | None = None) -> None:
        self.model = model
        self.rows = spec(config)
        self.names = [r["name"] for r in self.rows]
        # Rings: windowed signals keep back + forward reach + 2 epochs; others 96.
        reach: dict[str, list[int]] = {}
        for r in self.rows:
            if r["kind"] == "win":
                slot = reach.setdefault(r["signal"], [0, 0])
                i = 0 if r["dir"] == "b" else 1
                slot[i] = max(slot[i], r["n"])
        self.size = {s: DELAY + 2 + (reach[s][0] if s in reach else 0) for s in SIGNALS}
        self.ring = {s: [NA] * self.size[s] for s in SIGNALS}
        self.windows = [r for r in self.rows if r["kind"] == "win"]
        self.w_cnt = [0] * len(self.windows)
        self.w_sum = [0] * len(self.windows)
        self.w_sum2 = [0] * len(self.windows)
        # sample level
        self.n = 0
        self.first = True
        self.prev = (0, 0, 0)
        self.state = [[[0, 0] for _ in range(2)] for _ in range(3)]
        self.lx = [0] * BLOCK_LOW
        self.ly = [0] * BLOCK_LOW
        self.lz = [0] * BLOCK_LOW
        self.low_n = 0
        self.sx = self.sy = self.sz = 0
        self.angle_prev = None
        self.reset_epoch()
        # epoch level
        self.e = 0  # finished epochs
        self.angle_prev_epoch = NA
        self.base_jerk = Baseline()
        self.base_hr = Baseline()
        self.last_bout = -(10**9)
        self.out: list[dict] = []
        self.last_xyz = (0, 0, 0)

    def reset_epoch(self) -> None:
        self.in_epoch = 0
        self.jsum = self.secsum = self.jpeak = self.valid_n = 0
        self.lfm_sum = self.ang_sum = self.step_sum = self.still = self.blocks_n = 0
        self.hr_n = self.hr_s1 = self.hr_s2 = 0

    # ------------------------------------------------------------------ input

    def hr(self, bpm: int | None) -> None:
        """One heart-rate reading (integer bpm) or ``None``; call once per batch before its samples."""
        if bpm is not None and C.HR_MIN_BPM <= bpm <= C.HR_MAX_BPM:
            self.hr_n += 1
            self.hr_s1 += bpm
            self.hr_s2 += bpm * bpm

    def batch(self, xs, ys, zs, gap: int = 0, bpm: int | None = None, valid=None) -> None:
        """A sensor batch preceded by ``gap`` missing samples (held, counted as invalid).

        ``valid`` (offline only) marks individual samples inside the batch as held values.
        """
        self.hr(bpm)
        for _ in range(gap):
            self.sample(*self.last_xyz, False)
        for i in range(len(xs)):
            self.sample(xs[i], ys[i], zs[i], True if valid is None else bool(valid[i]))

    # ------------------------------------------------------------------ per sample

    def sample(self, x: int, y: int, z: int, valid: bool) -> None:
        self.last_xyz = (x, y, z)
        if self.first:
            self.first = False
            self.prev = (x, y, z)
            for a, v in enumerate((x, y, z)):
                for s in range(2):
                    self.state[a][s] = [ZI[s][0] * (v << 12), ZI[s][1] * (v << 12)]
        dx, dy, dz = x - self.prev[0], y - self.prev[1], z - self.prev[2]
        self.prev = (x, y, z)
        j = isqrt(n32(dx * dx + dy * dy + dz * dz) * 256)  # 1/16 mg
        self.jsum += j
        self.secsum += j
        self.in_epoch += 1
        if self.in_epoch % 25 == 0:
            peak = rdiv(self.secsum, 25)
            if peak > self.jpeak:
                self.jpeak = peak
            self.secsum = 0
        if valid:
            self.valid_n += 1
        # stage 1 low-pass, three axes
        out = []
        for a, v in enumerate((x, y, z)):
            xq = v << 12
            for s in range(2):
                b0, b1, b2, a1, a2 = COEF[s]
                st = self.state[a][s]
                yq = tdiv(b0 * xq + st[0], 1 << Q)
                st[0] = b1 * xq - a1 * yq + st[1]
                st[1] = b2 * xq - a2 * yq
                xq = yq
            out.append(n32(xq))
        if self.n % C.DECIMATE_1 == C.DECIMATE_1 - 1:
            i = self.low_n
            self.lx[i], self.ly[i], self.lz[i] = out
            self.sx += out[0]
            self.sy += out[1]
            self.sz += out[2]
            self.low_n += 1
            if self.low_n == BLOCK_LOW:
                self.block_end()
        self.n += 1
        if self.in_epoch == SPE:
            self.epoch_end()

    def block_end(self) -> None:
        mx, my, mz = rdiv(n32(self.sx), 25), rdiv(n32(self.sy), 25), rdiv(n32(self.sz), 25)
        total = 0
        for i in range(BLOCK_LOW):
            d = (tdiv(self.lx[i] - mx, 64), tdiv(self.ly[i] - my, 64), tdiv(self.lz[i] - mz, 64))
            total += isqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])  # Long squares
        self.lfm_sum += total
        x16, y16, z16 = tdiv(mx, 256), tdiv(my, 256), tdiv(mz, 256)
        h = isqrt(x16 * x16 + y16 * y16)
        ang = atan_q10(z16, h)
        step = 0 if self.angle_prev is None else abs(ang - self.angle_prev)
        self.angle_prev = ang
        self.ang_sum += ang
        self.step_sum += step
        if step < STILL_Q10:
            self.still += 1
        self.blocks_n += 1
        self.low_n = 0
        self.sx = self.sy = self.sz = 0

    # ------------------------------------------------------------------ per epoch

    def push(self, signal: str, value: int) -> None:
        self.ring[signal][self.e % self.size[signal]] = value

    def get(self, signal: str, i: int) -> int:
        if i < 0 or i > self.e:
            return NA
        assert self.e - i < self.size[signal], (signal, i, self.e)
        return self.ring[signal][i % self.size[signal]]

    def epoch_end(self) -> None:
        present = self.valid_n >= MIN_VALID_SAMPLES
        e = self.e
        if present:
            jerk = rdiv(n32(self.jsum), SPE)
            peak = self.jpeak
            lfm = rdiv(n32(self.lfm_sum), 25 * BLOCKS_PER_EPOCH)
            angle10 = rdiv(n32(self.ang_sum), BLOCKS_PER_EPOCH)
            angle = rdiv(angle10, 64)
            step = rdiv(n32(self.step_sum), BLOCKS_PER_EPOCH)
            sib = rdiv(self.still * 1024, BLOCKS_PER_EPOCH)
            daz = 0 if e == 0 else (NA if self.angle_prev_epoch == NA else abs(angle10 - self.angle_prev_epoch))
        else:
            jerk = peak = lfm = angle = angle10 = step = sib = daz = NA
        self.angle_prev_epoch = angle10
        # heart rate
        n = self.hr_n
        hr = rdiv(16 * self.hr_s1, n) if n else NA
        hr_std = NA
        if n > 1:
            num = n * self.hr_s2 - self.hr_s1 * self.hr_s1
            hr_std = rdiv(isqrt(256 * max(num, 0)), n)
        base_hr = self.base_hr.push(hr)
        hr_rel = NA if hr == NA or base_hr == NA else hr - base_hr
        hr_valid = (1024 if n > 0 else 0) if present else NA
        base_j = self.base_jerk.push(jerk)
        act = NA
        if present:
            act = 1024 if jerk >= BOUT_X * base_j else 0
        if act == 1024:
            self.last_bout = e
        tsb = min(e - self.last_bout, TSB_CAP)
        for name, v in (
            ("jerk", jerk), ("jerk_peak", peak), ("lfm", lfm), ("angle", angle),
            ("angle_step", step), ("daz", daz), ("sib", sib), ("act", act), ("hr", hr),
            ("hr_std", hr_std), ("hr_n", n), ("hr_base", base_hr), ("hr_rel", hr_rel),
            ("hr_valid", hr_valid), ("tsb", tsb),
        ):  # fmt: skip
            self.push(name, v)
        self.reset_epoch()
        # the newest epoch is ``e``; windows advance to epoch k = e - DELAY
        k = e - DELAY
        self.advance(k)
        if k >= 0:
            self.score(k)
        self.e += 1

    def advance(self, k: int) -> None:
        for w, r in enumerate(self.windows):
            if r["stat"] == "max":
                continue
            s, n = r["signal"], r["n"]
            if r["dir"] == "b":
                add, drop = self.get(s, k), self.get(s, k - n)
            else:
                add, drop = self.get(s, k + n), self.get(s, k)
            if add != NA:
                self.w_cnt[w] += 1
                self.w_sum[w] = n32(self.w_sum[w] + add)
                self.w_sum2[w] = n32(self.w_sum2[w] + (add * add if r["stat"] == "std" else 0))
            if drop != NA:
                self.w_cnt[w] -= 1
                self.w_sum[w] -= drop
                self.w_sum2[w] -= drop * drop if r["stat"] == "std" else 0

    def window_value(self, w: int, r: dict, k: int) -> int:
        s, n = r["signal"], r["n"]
        if r["stat"] == "max":
            lo, hi = (k - n + 1, k) if r["dir"] == "b" else (k + 1, k + n)
            best = NA
            for i in range(lo, hi + 1):
                v = self.get(s, i)
                if v != NA and (best == NA or v > best):
                    best = v
            return best
        cnt, total = self.w_cnt[w], self.w_sum[w]
        if r["stat"] == "mean":
            return rdiv(n32(total), cnt) if cnt > 0 else NA
        if cnt < 2:
            return NA
        num = cnt * self.w_sum2[w] - total * total  # Long
        return rdiv(isqrt(256 * max(num, 0)), cnt)

    def score(self, k: int) -> None:
        F = []
        wi = 0
        for r in self.rows:
            if r["kind"] == "cur":
                name = r["name"]
                if name == "tnb_min":
                    ahead = DELAY
                    for j in range(1, DELAY + 1):
                        if self.get("act", k + j) == 1024:
                            ahead = j
                            break
                    F.append(ahead)
                elif name == "hr_missing":
                    F.append(1 if self.get("hr_n", k) == 0 else 0)
                elif name == "tsb_min":
                    F.append(self.get("tsb", k))
                else:
                    F.append(self.get(r["signal"], k))
            else:
                F.append(self.window_value(wi, r, k))
                wi += 1
        row = {"epoch": k, "features": F}
        if self.model is not None:
            x = self.model.bins(F)
            sc = self.model.scores(x)
            best = 0
            for c in range(1, 4):
                if sc[c] > sc[best]:
                    best = c
            row.update(bins=x[:87], scores=sc, stage=best)
        self.out.append(row)
