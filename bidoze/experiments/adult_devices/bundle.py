"""Build the adult bundle from an arm's final phase: the model that is released.

A wrapper over ``arcasleep_bidoze.release.build_release``: each seed's checkpoint in
``<run>/<arm>/final/model_outputs/`` becomes a release bundle, and the divisor comes from the
final phase's training nights (its SleepAccel and Wearanize+ sets are the adult stage mix).

The bundle's README.md is written afterwards by ``python -m arcasleep_bidoze.release card``.

Run from ``bidoze/`` with a relative run dir, so provenance records ``results/...``.

Usage: python bundle.py <run dir> <arm> <output dir> <version>
"""
from __future__ import annotations

import sys
from pathlib import Path

from arcasleep_bidoze.release import build_release


def main(run: Path, arm: Path, output: Path, version: str) -> None:
    build_release("adult", run / arm / "final", output, version, train_cache=run / "setup/cache" / arm)
    print(f"{output}: {len(list(output.glob('seed*')))} seeds")


if __name__ == "__main__":
    run, arm, output, version = sys.argv[1:5]
    main(Path(run), Path(arm), Path(output).resolve(), version)
