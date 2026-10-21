"""Write the setup of the TriDoze pilot and full test, reusing the 2026-10-15 draws and feature caches.

Writes, under ``<run root>/setup/``:
- cache                    symlink to the 2026-10-15 run's setup/cache (same draws, same feature caches)
- draws.json, cv/          copies from the 2026-10-15 run
- model-<arm>.json         the release model.json plus ``gap_from_row``; bidoze_xl also ``d_model`` 512, ``d_ff`` 1024
- training-<arm>.json      training-balanced.json plus ``run_seed`` 0 and the arm's keys
- README.md                the source of each file

Arms: tridoze_j (joint_members 3, member_loss_weight 0.5), tridoze_r (model.json architecture "roles"; training role_loss_weight 0.5),
bidoze_xl and bidoze_g (seed_count 1). ``max_train_seconds`` is set per fit by burst.sh.

Usage: python prepare.py <run root> <2026-10-15 run dir>
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

BIDOZE = Path(__file__).resolve().parents[2]

ARMS = {
    "tridoze_j": ({}, {"joint_members": 3, "member_loss_weight": 0.5}),
    "tridoze_r": ({"architecture": "roles"}, {"role_loss_weight": 0.5}),
    "bidoze_xl": ({"d_model": 512, "d_ff": 1024}, {"seed_count": 1}),
    "bidoze_g": ({}, {"seed_count": 1}),
}


def main(run_root: Path, source: Path) -> None:
    setup = run_root / "setup"
    setup.mkdir(parents=True, exist_ok=True)
    cache = setup / "cache"
    if cache.is_symlink() or cache.exists():
        cache.unlink() if cache.is_symlink() else shutil.rmtree(cache)
    cache.symlink_to((source / "setup/cache").resolve())
    shutil.copy2(source / "setup/draws.json", setup / "draws.json")
    shutil.copytree(source / "setup/cv", setup / "cv", dirs_exist_ok=True)

    model = json.loads((BIDOZE / "config/model.json").read_text())
    recipe = json.loads((BIDOZE / "config/training-balanced.json").read_text())
    for arm, (model_keys, training_keys) in ARMS.items():
        (setup / f"model-{arm}.json").write_text(
            json.dumps({**model, "gap_from_row": True, **model_keys}, indent=2) + "\n")
        (setup / f"training-{arm}.json").write_text(
            json.dumps({**recipe, "run_seed": 0, **training_keys}, indent=2) + "\n")

    (setup / "README.md").write_text(
        "Setup of the TriDoze pilot and full test.\n"
        f"- cache: symlink to the setup/cache of {source.name} (same draws, same feature caches).\n"
        f"- draws.json and cv/: copies from {source.name}/setup; no new draws.\n"
        "- model-<arm>.json: bidoze/config/model.json plus gap_from_row; bidoze_xl also d_model 512, d_ff 1024.\n"
        "- training-<arm>.json: bidoze/config/training-balanced.json plus run_seed 0 and the arm's keys.\n"
        "- Arms: tridoze_j, tridoze_r, bidoze_xl, bidoze_g (written by experiments/tridoze/prepare.py).\n"
        "- max_train_seconds is not in these files; burst.sh writes it into <arm>/draw<d>/config-burst/training.json.\n"
        "- Burst command: see the usage text in run.sh.\n")
    write_run_sh(setup)
    print(f"wrote {setup}")


def write_run_sh(setup: Path) -> None:
    sh = setup / "run.sh"
    sh.write_text(
        "#!/usr/bin/env bash\n"
        "# Usage text only: the pilot is run in bursts, one command per burst, launched by the lead:\n"
        "#   timeout 15m flock ~/.cache/arcasleep-gpu.lock bash <worktree>/bidoze/experiments/tridoze/burst.sh \\\n"
        "#       <run root> 840 <arm>:<draw> [<arm>:<draw> ...]\n"
        "# e.g. arms tridoze_j tridoze_r bidoze_xl bidoze_g, draws 0-4. A dry run (no training, no GPU):\n"
        "#   bash burst.sh --dry-run <run root> 840 tridoze_j:0 bidoze_g:0 bidoze_g:1\n"
        "echo 'see the comments in this file'; exit 1\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
