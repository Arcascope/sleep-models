"""Build the adult bundle from an arm's final phase: the model that is released.

A wrapper over ``arcasleep_bidoze.release.build_release``: each seed's checkpoint in
``<run>/<arm>/final/model_outputs/`` becomes a release bundle, and the divisor comes from the
final phase's training nights and BIDsleep, the adult reference.

The bundle's README.md is ``cards/bidoze-adult.md``. Each seed's README.md is
``cards/bidoze-adult-seed.md``: the generated card would score the seed by itself, without the
divisor, so its figure is dropped and its checksums recomputed.

Run from ``bidoze/`` with a relative run dir, so provenance records ``results/...``.

Usage: python bundle.py <run dir> <arm> <adult cache root> <output dir> <version>
(the adult cache root holds ``12Hz_log_PSD_aligned/`` with BIDsleep).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from arcasleep.artifacts.provenance import sha256_file
from arcasleep.artifacts.publish import ASSETS_DIR, BUNDLE_FILES
from arcasleep_bidoze.release import build_release

BIDOZE = Path(__file__).resolve().parents[2]


def main(run: Path, arm: str, adult_root: Path, output: Path, version: str) -> None:
    build_release("adult", run / arm / "final", output, version,
                  train_cache=run / "setup/cache" / arm, cache_root=adult_root)
    for seed in sorted(output.glob("seed*")):
        shutil.copy(BIDOZE / "cards/bidoze-adult-seed.md", seed / "README.md")
        shutil.rmtree(seed / ASSETS_DIR, ignore_errors=True)
        files = sorted(f for f in BUNDLE_FILES if f != "checksums.sha256")
        (seed / "checksums.sha256").write_text(
            "".join(f"{sha256_file(seed / f)}  {f}\n" for f in files))
    shutil.copy(BIDOZE / "cards/bidoze-adult.md", output / "README.md")
    print(f"{output}: {len(list(output.glob('seed*')))} seeds")


if __name__ == "__main__":
    run, arm, adult_root, output, version = sys.argv[1:6]
    main(Path(run), arm, Path(adult_root).resolve(), Path(output).resolve(), version)
