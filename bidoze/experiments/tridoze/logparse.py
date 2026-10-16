"""Parse arcasleep training.log files (one SEED block per fit; a joint or single fit has one block)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

EPOCH = re.compile(
    r"^\s+epoch\s+(\d+)\s+step\s+(\d+)\s+loss\s+(\S+)\s+eval_kappa\s+(\S+)\s+smoothed\s+(\S+)"
    r"\s+best\s+(\S+)@(\d+)\s+lr\s+(\S+)\s+epoch_time=([\d.]+)s"
)
SEED = re.compile(r"^SEED (\d+)/(\d+): (\d+)")
BEST = re.compile(r"^\s+best eval kappa ([\d.\-]+) at epoch (\d+)")
TRAINED = re.compile(r"^\s+trained in ([\d.]+)s")
CAP = re.compile(r"max_train_seconds|time cap|wall-clock", re.I)


@dataclass
class Fit:
    seed: int = 0
    epochs: list[int] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    kappa: list[float] = field(default_factory=list)
    smoothed: list[float] = field(default_factory=list)
    epoch_time: list[float] = field(default_factory=list)
    best_kappa: float | None = None
    best_epoch: int | None = None
    train_seconds: float | None = None
    early_stop: bool = False
    cap_line: bool = False

    @property
    def best_step(self) -> int | None:
        return self.steps[self.epochs.index(self.best_epoch)] if self.best_epoch in self.epochs else None

    @property
    def total_steps(self) -> int:
        return self.steps[-1] if self.steps else 0

    def capped(self, cap_seconds: float | None = None) -> bool:
        """True when the log says the cap ended the fit; without such a line, when the fit did not
        early-stop and ran to within 5% of the cap."""
        if self.cap_line:
            return True
        if cap_seconds and self.train_seconds is not None and not self.early_stop:
            return self.train_seconds >= 0.95 * cap_seconds
        return False


def parse_log(path: Path) -> list[Fit]:
    fits: list[Fit] = []
    cur: Fit | None = None
    for line in Path(path).read_text(errors="replace").splitlines():
        m = SEED.match(line)
        if m:
            cur = Fit(seed=int(m.group(3)))
            fits.append(cur)
            continue
        m = EPOCH.match(line)
        if m:
            if cur is None:
                cur = Fit()
                fits.append(cur)
            cur.epochs.append(int(m.group(1)))
            cur.steps.append(int(m.group(2)))
            cur.kappa.append(float(m.group(4)))
            cur.smoothed.append(float(m.group(5)))
            cur.epoch_time.append(float(m.group(9)))
            continue
        if cur is None:
            continue
        if "early stop" in line:
            cur.early_stop = True
        elif CAP.search(line) and not line.lstrip().startswith(("max epochs", "epoch")):
            cur.cap_line = True
        m = BEST.match(line)
        if m:
            cur.best_kappa, cur.best_epoch = float(m.group(1)), int(m.group(2))
        m = TRAINED.match(line)
        if m:
            cur.train_seconds = float(m.group(1))
    return fits
