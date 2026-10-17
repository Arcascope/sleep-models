"""Write the setup for the ensemble-size run: extra seeds 3-6 and bags 3-6 of BiDoze-G.

Usage: prepare_ensemble_size.py <source run root> <new run root>
Seeds 0-2 and bags 0-2 are read from the source run; only seeds/bags 3-6 are written here.
"""
import json
import shutil
import sys
from pathlib import Path

src, dst = (Path(a).resolve() for a in sys.argv[1:3])
s, d = src / "setup", dst / "setup"
d.mkdir(parents=True, exist_ok=True)
cache = d / "cache"
if not cache.is_symlink():
    cache.symlink_to((s / "cache").resolve())
shutil.copy(s / "draws.json", d / "draws.json")
shutil.copytree(s / "cv", d / "cv", dirs_exist_ok=True)
base = json.loads((s / "training-bidoze_g.json").read_text())
model = (s / "model-bidoze_g.json").read_text()
for k in range(3, 7):
    seeded = {**base, "run_seed": k}
    bagged = {**seeded, "bootstrap_seed": 100 + k}
    for arm, t in ((f"bidoze_g_s{k}", seeded), (f"bidoze_g_bag{k}", bagged)):
        (d / f"training-{arm}.json").write_text(json.dumps(t, indent=2))
        (d / f"model-{arm}.json").write_text(model)
