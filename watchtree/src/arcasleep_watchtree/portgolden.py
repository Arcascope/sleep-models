"""Golden input streams and expected output of the watch app's test mode.

    python -m arcasleep_watchtree.portgolden --out results/port-2026-10-27

Two runs, each through a fresh :class:`~arcasleep_watchtree.fixed.Engine` with the final model:

* run 0: a stretch of a real Weaver Garmin night (integer milli-g, one heart-rate reading per
  second, natural gaps as ``gap`` counts), stored in the app as chunks of whole batches;
* run 1: a synthetic stream drawn from a 16-bit linear congruential generator, which the app
  generates itself; it is long enough (3 h) for every ring to wrap.

A batch is ``n`` samples (at most 25) preceded by ``gap`` held samples and a heart-rate reading
(0 = none). Chunk layout (bytes): ``u16 batches`` then per batch ``n``, ``gap`` (0..253, or 254
followed by u16), ``bpm``, and for each sample and axis the zigzag of the change from the previous
sample of that axis, one byte below 255, else 255 followed by u16. Chunks hold whole batches, at
most ``CHUNK_BYTES`` each, and are stored as base64.

Output lines of the app (and of ``golden_expected.txt``):

``R s <run> <epoch> <stage> <s0> <s1> <s2> <s3> <crc>`` for every scored epoch;
``R f <run> <epoch> <87 feature integers>`` for the first three and every tenth epoch of a run.
``crc`` is ``c = (c * 31 + (F & 0xFFFF) + ((F >> 16) & 0xFFFF)) % 65521`` over the features.
"""

from __future__ import annotations

import argparse
import base64
import json
import warnings
from pathlib import Path

import numpy as np

from arcasleep_watchtree.fixed import Engine, Model

CHUNK_BYTES = 3000
SYN_BATCHES = 10800
LEVELS = [1, 2, 3, 6, 12, 25, 60, 150, 400]
ORIENT = [(0, 0, 1000), (300, 100, 950), (-500, 200, 800), (100, -700, 700), (0, 900, 400)]
REAL_MINUTES = 66


class Lcg:
    """The synthetic generator's state: ``r = (r * 75 + 74) % 65537``."""

    def __init__(self, seed: int = 1) -> None:
        self.r = seed

    def next(self) -> int:
        self.r = (self.r * 75 + 74) % 65537
        return self.r


def synthetic_batches(n_batches: int = SYN_BATCHES):
    rng = Lcg()
    n = 0
    for b in range(n_batches):
        ep = n // 750
        r = rng.next()
        bpm = 0 if r % 9 == 0 else 50 + (ep % 7) * 3 + rng.next() % 9
        gap = 40 if b % 1500 == 777 else 0
        level = LEVELS[(ep * 7 + ep // 5) % 9]
        ox, oy, oz = ORIENT[(ep // 6) % 5]
        xs, ys, zs = [], [], []
        for i in range(25):
            xs.append(ox + rng.next() % (2 * level + 1) - level)
            ys.append(oy + rng.next() % (2 * level + 1) - level)
            zs.append(oz + rng.next() % (2 * level + 1) - level)
            if ep % 13 == 5 and (n + i) % 750 < 100:
                xs[-1] += rng.next() % 801 - 400
        n += 25 + gap
        yield xs, ys, zs, gap, bpm


# ---------------------------------------------------------------------------------- real stream


def real_batches(acc: np.ndarray, valid: np.ndarray, hr: np.ndarray):
    """Batches of up to 25 valid samples, gaps as counts, one heart-rate reading per second."""
    batches = []
    cur: list[int] = []
    gap = 0
    last_sec = -1
    start = 0
    i, n = 0, acc.shape[0]
    pending_gap = 0
    while i < n:
        if not valid[i]:
            if cur:
                batches.append((cur, pending_gap, start))
                cur, pending_gap = [], 0
            j = i
            while j < n and not valid[j]:
                j += 1
            pending_gap += j - i
            i = j
            continue
        if not cur:
            start = i
        cur.append(i)
        i += 1
        if len(cur) == 25:
            batches.append((cur, pending_gap, start))
            cur, pending_gap = [], 0
    if cur:
        batches.append((cur, pending_gap, start))
    out = []
    for idx, g, s in batches:
        sec = s // 25
        bpm = 0
        if sec != last_sec:
            bpm = int(hr[sec]) if sec < hr.size and hr[sec] >= 0 else 0
            last_sec = sec
        out.append((acc[idx, 0].tolist(), acc[idx, 1].tolist(), acc[idx, 2].tolist(), g, bpm))
    return out


# ---------------------------------------------------------------------------------- encoding


def zigzag(d: int) -> int:
    return 2 * d if d >= 0 else -2 * d - 1


def encode_batch(xs, ys, zs, gap: int, bpm: int, prev: list[int]) -> bytes:
    out = bytearray([len(xs)])
    if gap < 254:
        out.append(gap)
    else:
        out += bytes([254, gap & 255, gap >> 8])
    out.append(bpm)
    for i in range(len(xs)):
        for a, v in enumerate((xs[i], ys[i], zs[i])):
            u = zigzag(v - prev[a])
            prev[a] = v
            if u < 255:
                out.append(u)
            else:
                assert u < 65536
                out += bytes([255, u & 255, u >> 8])
    return bytes(out)


def encode_chunks(batches) -> list[bytes]:
    chunks, body, count = [], bytearray(), 0
    prev = [0, 0, 0]
    for xs, ys, zs, gap, bpm in batches:
        enc = encode_batch(xs, ys, zs, gap, bpm, prev)
        if len(body) + len(enc) + 2 > CHUNK_BYTES:
            chunks.append(count.to_bytes(2, "little") + bytes(body))
            body, count = bytearray(), 0
        body += enc
        count += 1
    chunks.append(count.to_bytes(2, "little") + bytes(body))
    return chunks


def decode_chunks(chunks: list[bytes]):
    """Inverse of :func:`encode_chunks` (the same walk the app does)."""
    prev = [0, 0, 0]
    for chunk in chunks:
        count = int.from_bytes(chunk[:2], "little")
        p = 2
        for _ in range(count):
            n = chunk[p]
            gap = chunk[p + 1]
            p += 2
            if gap == 254:
                gap = chunk[p] | (chunk[p + 1] << 8)
                p += 2
            bpm = chunk[p]
            p += 1
            axes = ([], [], [])
            for _ in range(n):
                for a in range(3):
                    u = chunk[p]
                    p += 1
                    if u == 255:
                        u = chunk[p] | (chunk[p + 1] << 8)
                        p += 2
                    d = (u >> 1) if u % 2 == 0 else -((u + 1) >> 1)
                    prev[a] += d
                    axes[a].append(prev[a])
            yield axes[0], axes[1], axes[2], gap, bpm


# ---------------------------------------------------------------------------------- expected output


def crc(features) -> int:
    c = 0
    for f in features:
        c = (c * 31 + (f & 0xFFFF) + ((f >> 16) & 0xFFFF)) % 65521
    return c


def run_lines(run: int, batches, model: Model) -> list[str]:
    eng = Engine(model)
    lines = []
    for xs, ys, zs, gap, bpm in batches:
        eng.batch(xs, ys, zs, gap, bpm if bpm else None)
    for j, r in enumerate(eng.out):
        k, f, sc = r["epoch"], r["features"], r["scores"]
        lines.append(f"R s {run} {k} {r['stage']} {sc[0]} {sc[1]} {sc[2]} {sc[3]} {crc(f)}")
        if j < 3 or k % 10 == 0:
            lines.append(f"R f {run} {k} " + " ".join(str(v) for v in f))
    lines.append(f"R n {run} {len(eng.out)}")
    return lines


def main() -> None:

    from arcasleep.data.prepare import load_dataset
    from arcasleep_watchtree.cohorts import COHORTS, DATA_ROOT, load_night, subject_ids
    from arcasleep_watchtree.port_run import stream_of

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    out = args.out
    gold = out / "golden"
    gold.mkdir(exist_ok=True)

    model_json = json.loads((out / "model.json").read_text())
    model = Model.load(model_json, (out / "trees.bin").read_bytes())

    cohort = COHORTS["weaver_garmin"]
    ids, reported = subject_ids(cohort)
    data_set = load_dataset(DATA_ROOT / cohort.directory)
    chosen = None
    for sid in ids[1:6]:
        try:
            night = load_night(cohort, sid, reported[sid], data_set)
        except Exception:  # noqa: BLE001
            continue
        s = stream_of(night)
        if s["acc"].shape[0] > 3600 * 25 * 3 and (s["hr"] >= 0).mean() > 0.9 and (~s["valid"]).sum() > 0:
            chosen = s
            break
    assert chosen is not None
    first = 30 * 60 * 25
    n = REAL_MINUTES * 60 * 25
    seg = slice(first, first + n)
    real = real_batches(chosen["acc"][seg], chosen["valid"][seg], chosen["hr"][first // 25 : first // 25 + n // 25])
    chunks = encode_chunks(real)
    decoded = list(decode_chunks(chunks))
    assert len(decoded) == len(real) and all(
        a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3] and a[4] == b[4]
        for a, b in zip(decoded, real)
    )
    syn = list(synthetic_batches())
    lines = run_lines(0, real, model) + run_lines(1, syn, model)
    (gold / "golden_expected.txt").write_text("\n".join(lines) + "\n")
    (gold / "golden_real_chunks.json").write_text(json.dumps([base64.b64encode(c).decode() for c in chunks]))
    info = {
        "real_batches": len(real),
        "real_gap_batches": sum(1 for b in real if b[3]),
        "real_gap_samples": sum(b[3] for b in real),
        "real_bytes": sum(len(c) for c in chunks),
        "real_chunks": len(chunks),
        "real_minutes": REAL_MINUTES,
        "synthetic_batches": len(syn),
        "expected_lines": len(lines),
        "decoded_lines": [l.split()[2] for l in lines if l.startswith("R n")],
    }
    (gold / "golden_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
