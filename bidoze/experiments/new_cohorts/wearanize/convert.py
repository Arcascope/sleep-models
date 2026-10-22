"""Convert the Wearanize+ open-access cohort into the arcasleep input layout.

    python convert.py RAW_ROOT SCRATCH_DIR OUT_DIR [--workers N]

RAW_ROOT is the unzipped ``Wearanize+_OA_raw_v1.1`` folder. It is only read.
SCRATCH_DIR receives the extracted Empatica ``ACC.csv`` files and must be
outside RAW_ROOT. OUT_DIR gets the layout `pisces_lite` loads:

    cleaned_accelerometer/<id>_cleaned_motion.out   time_s x_g y_g z_g
    cleaned_psg/<id>_cleaned_psg.out                time_s stage
    scorer2/<id>_scorer2.out, usleep/<id>_usleep.out  time_s stage (side files)
    data_set.json, conversion_manifest.json

Time zero of every file is the ZMax lights-out moment of the night.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xlsx import read_first_sheet  # noqa: E402

EPOCH_S = 30.0
E4_HZ = 32.0
E4_UNITS_PER_G = 64.0
# Wearanize stage codes: -1 artifact, 0 wake, 1 N1, 2 N2, 3 N3, 4 REM.
# SleepAccel codes: -1 unscored, 0 wake, 1 N1, 2 N2, 3 N3, 5 REM.
STAGE_MAP = {-1: -1, 0: 0, 1: 1, 2: 2, 3: 3, 4: 5}
SYNC_XLSX = "3.Manual_synchronization/Manual_sync_zmax_psg_emp_actpal.xlsx"
DEMOGRAPHICS_XLSX = "4.Demographic_info/Wearanize+_participant_demographic_information.xlsx"
SCORER1, SCORER2, USLEEP = (
    "2.Sleep_scores/1.PSG_manual_scores_scorer1",
    "2.Sleep_scores/2.PSG_manual_scores_scorer2",
    "2.Sleep_scores/3.PSG_autoscores_U-Sleep_v2.0",
)


def subject_id(raw_name: str) -> str:
    """``Sub001s1`` -> ``sub001_s1``."""
    match = re.fullmatch(r"sub(\d+)s(\d+)", raw_name.strip("'").lower())
    if match is None:
        raise ValueError(f"unrecognised subject name {raw_name!r}")
    return f"sub{match.group(1)}_s{match.group(2)}"


def read_stages(path: Path) -> np.ndarray:
    """First column of a score file as raw Wearanize stage codes, header skipped."""
    lines = path.read_text().replace("\r", "").split("\n")[1:]
    return np.array(
        [int(float(line.replace("\t", ",").split(",")[0])) for line in lines if line.strip()]
    )


def map_stages(raw: np.ndarray) -> np.ndarray:
    unknown = set(np.unique(raw).tolist()) - set(STAGE_MAP)
    if unknown:
        raise ValueError(f"unexpected stage codes {sorted(unknown)}")
    return np.array([STAGE_MAP[v] for v in raw.tolist()], dtype=float)


def read_sync(raw_root: Path) -> list[dict]:
    rows = read_first_sheet(raw_root / SYNC_XLSX)
    header = rows[0]
    return [dict(zip(header, r)) for r in rows[1:]]


def window_epochs(n_epochs: int, start_s: float, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Epoch times and indices of a device-time score file inside the window.

    Device epoch ``k`` starts at device time ``30 k``, which is window time
    ``30 k - start_s``. Epochs that begin before the window or end after it are left out.
    """
    index = np.arange(n_epochs)
    times = EPOCH_S * index - start_s
    keep = (times >= 0) & (times + EPOCH_S <= duration_s)
    return times[keep], index[keep]


def extract_e4(raw_root: Path, scratch: Path, raw_name: str) -> Path | None:
    archives = sorted((raw_root / "1.Raw_data" / raw_name / "3.Empatica").glob("*.zip"))
    if not archives:
        return None
    target = scratch / raw_name
    if not (target / "ACC.csv").exists():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archives[0]) as z:
            z.extract("ACC.csv", target)
    return target / "ACC.csv"


def load_e4(path: Path) -> np.ndarray:
    """Empatica ACC.csv as ``(n, 3)`` in 1/64 g, after checking the 32 Hz header."""
    with open(path) as f:
        f.readline()
        rate = float(f.readline().split(",")[0])
    if rate != E4_HZ:
        raise ValueError(f"E4 ACC rate {rate} Hz, expected {E4_HZ}")
    return np.loadtxt(path, delimiter=",", skiprows=2, dtype=np.float32)


def write_table(path: Path, columns: list[np.ndarray]) -> None:
    np.savetxt(path, np.column_stack(columns), fmt="%.6f", delimiter=" ")


def convert_one(args: tuple) -> dict:
    raw_root, scratch, out, row = (Path(args[0]), Path(args[1]), Path(args[2]), args[3])
    raw_name = row["SubjectID"].strip("'")
    sid = subject_id(raw_name)
    info = {"raw_name": raw_name, "id": sid}

    def drop(reason: str) -> dict:
        info["status"] = "dropped"
        info["reason"] = reason
        return info

    for label, key in (("PSG", "PSG_start_sec"), ("Empatica", "Emp_start_sec")):
        value = row[key]
        if row[key.replace("start", "end")] == 0 and value == 0:
            return drop(f"{label} unavailable")
        if value == -999:
            return drop(f"{label} not synchronizable")
    scorer1 = raw_root / SCORER1 / f"{raw_name}.txt"
    if not scorer1.exists():
        return drop("no scorer 1 file")
    acc_path = extract_e4(raw_root, scratch, raw_name)
    if acc_path is None:
        return drop("no Empatica archive")

    duration = float(row["Zmax_dur_sec"])
    psg_start, emp_start = float(row["PSG_start_sec"]), float(row["Emp_start_sec"])
    spans = {
        "PSG": float(row["PSG_end_sec"]) - psg_start,
        "Empatica": float(row["Emp_end_sec"]) - emp_start,
        "ZMax": float(row["Zmax_end_sec"]) - float(row["Zmax_start_sec"]),
    }
    info["span_mismatch_s"] = {k: v - duration for k, v in spans.items() if v != duration}

    acc = load_e4(acc_path)
    # Sample i is recorded at device time 1 + (i + 1) / fs (README: counting starts at 1).
    device_time = 1.0 + (np.arange(acc.shape[0]) + 1.0) / E4_HZ
    t = device_time - emp_start
    inside = (t >= 0) & (t <= duration)
    t, acc = t[inside], acc[inside].astype(np.float64) / E4_UNITS_PER_G
    if t.size < 2:
        return drop("no Empatica samples inside the lights-out window")
    info["e4_coverage_s"] = [float(t[0]), float(t[-1])]

    raw1 = read_stages(scorer1)
    times, index = window_epochs(raw1.size, psg_start, duration)
    stages = map_stages(raw1)[index]
    # An epoch with no E4 samples at all is unscored, like SleepAccel's unscored epochs.
    counts = np.histogram(t, bins=np.append(times, times[-1] + EPOCH_S))[0] if times.size else []
    covered = np.asarray(counts) > 0
    info["epochs"] = int(times.size)
    info["artifact_epochs"] = int(np.sum(stages == -1))
    info["epochs_without_e4"] = int(np.sum(~covered & (stages != -1)))
    stages = np.where(covered, stages, -1.0)
    if times.size == 0:
        return drop("no scored epochs inside the window")

    write_table(out / "cleaned_accelerometer" / f"{sid}_cleaned_motion.out", [t, acc[:, 0], acc[:, 1], acc[:, 2]])
    write_table(out / "cleaned_psg" / f"{sid}_cleaned_psg.out", [times, stages])

    # Scorer 2 scores the window itself: its epoch j starts at window time 30 j.
    p2 = raw_root / SCORER2 / f"{raw_name}.txt"
    if p2.exists():
        s2 = map_stages(read_stages(p2))
        write_table(out / "scorer2" / f"{sid}_scorer2.out", [EPOCH_S * np.arange(s2.size), s2])
        info["scorer2"] = True
    # U-Sleep scores the whole PSG recording on scorer 1's epoch grid.
    pu = raw_root / USLEEP / f"{raw_name}.csv"
    if pu.exists():
        su = map_stages(read_stages(pu))
        ut, ui = window_epochs(su.size, psg_start, duration)
        write_table(out / "usleep" / f"{sid}_usleep.out", [ut, su[ui]])
        info["usleep"] = True
    info["status"] = "converted"
    return info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw_root")
    p.add_argument("scratch")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    raw_root, scratch, out = Path(a.raw_root).resolve(), Path(a.scratch).resolve(), Path(a.out).resolve()
    if raw_root in scratch.parents or raw_root in out.parents or scratch == raw_root or out == raw_root:
        raise SystemExit("scratch and output must be outside the raw tree")
    for sub in ("cleaned_accelerometer", "cleaned_psg", "scorer2", "usleep"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows = read_sync(raw_root)
    jobs = [(str(raw_root), str(scratch), str(out), r) for r in rows]
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(convert_one, jobs))

    (out / "data_set.json").write_text(json.dumps({
        "name": "wearanize",
        "csv": {"delimiter": " "},
        "id_pattern_by_feature": {
            "accelerometer": "<<ID>>_cleaned_motion.out",
            "psg": "<<ID>>_cleaned_psg.out",
        },
        "metadata": {
            "note": "Wearanize+ OA: Empatica E4 wrist ACC (32 Hz, converted from 1/64 g to g) with manual PSG scores "
                    "from scorer 1. Time zero is ZMax lights-out. PSG stages use the 0-5 convention (0=Wake, 5=REM); "
                    "-1 marks artifact epochs and epochs without E4 data. scorer2/ and usleep/ hold the other scorings.",
        },
    }, indent=2) + "\n")
    (out / "conversion_manifest.json").write_text(json.dumps(results, indent=1) + "\n")
    done = [r for r in results if r["status"] == "converted"]
    print(f"converted {len(done)} of {len(results)} nights")
    reasons: dict[str, int] = {}
    for r in results:
        if r["status"] == "dropped":
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    print(reasons)


if __name__ == "__main__":
    main()
