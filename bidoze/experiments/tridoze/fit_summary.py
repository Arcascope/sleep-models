"""One summary line for a finished fit: arm, draw, best eval kappa, epoch, train seconds, capped.

Usage: python fit_summary.py <training.log> <arm> <draw> [cap seconds]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logparse import parse_log  # noqa: E402

log, arm, draw = sys.argv[1:4]
cap = float(sys.argv[4]) if len(sys.argv) > 4 else None
for fit in parse_log(Path(log)):
    print(f"fit {arm} draw{draw} seed {fit.seed}: best eval kappa {fit.best_kappa} at epoch {fit.best_epoch}, "
          f"steps {fit.total_steps}, train {fit.train_seconds}s, capped={'yes' if fit.capped(cap) else 'no'}")
