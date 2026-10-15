#!/usr/bin/env python3
"""Run a bounded UNet GPU smoke test, collect its outputs, and delete the VM.

    python3 deploy/scripts/run_spot_smoke_gcp.py IMAGE [--zones ZONE ...]

Requires authenticated gcloud. Tries zones sequentially on stockout; only one
VM is created. The VM is deleted on completion and has a one-hour backstop.
The local run directory holds the exact image digest, status, and serial log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--project", default=os.environ.get("ARCASLEEP_PROJECT", "shiftapp-arcascope"))
    parser.add_argument("--zones", nargs="+", default=["us-west1-a", "europe-west1-b", "europe-west1-c"])
    parser.add_argument("--results-bucket", default="arcascope-sleep-models-data")
    parser.add_argument("--machine-type", choices=["g2-standard-4", "g2-standard-8", "a2-highgpu-1g"], default="g2-standard-8")
    parser.add_argument("--provisioning-model", choices=["SPOT", "STANDARD"], default="SPOT")
    args = parser.parse_args()

    def gcloud(*parts: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(["gcloud", *parts, "--project=" + args.project],
                                capture_output=True, text=True)
        if check and result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        return result

    # gcloud accepts a tag or digest reference, but not their combination.
    lookup = args.image
    if "@" in lookup:
        name, digest = lookup.split("@", 1)
        parent, leaf = name.rsplit("/", 1)
        lookup = parent + "/" + leaf.split(":", 1)[0] + "@" + digest
    info = json.loads(gcloud("artifacts", "docker", "images", "describe", lookup, "--format=json").stdout)
    image = info["image_summary"]["fully_qualified_digest"]
    if not re.search(r"@sha256:[0-9a-f]{64}$", image):
        raise RuntimeError("Registry did not return a valid image digest")
    run_id = "unet-gcp-" + args.provisioning_model.lower() + "-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = root / "unet" / "results" / run_id
    output.mkdir(parents=True)
    state = {"run_id": run_id, "image": image, "project": args.project,
             "machine_type": args.machine_type, "provisioning_model": args.provisioning_model,
             "stage": "starting"}

    def save(**updates: object) -> None:
        state.update(updates)
        (output / "gcp-run.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    template = (root / "deploy/gcp/unet-spot-startup.sh").read_text()
    for key, value in {"__RUN_ID__": run_id, "__IMAGE_REF__": image,
                       "__RESULTS_BUCKET__": args.results_bucket}.items():
        template = template.replace(key, shlex.quote(value))
    zone = None
    with tempfile.TemporaryDirectory(prefix="arcasleep-spot-") as temporary:
        startup = Path(temporary) / "startup.sh"
        startup.write_text(template)
        for candidate in args.zones:
            result = gcloud("compute", "instances", "create", run_id,
                "--zone=" + candidate, "--machine-type=" + args.machine_type,
                "--provisioning-model=" + args.provisioning_model, "--maintenance-policy=TERMINATE",
                "--no-restart-on-failure", "--instance-termination-action=DELETE",
                "--max-run-duration=3600s", "--image-project=cos-cloud",
                "--image-family=cos-stable", "--boot-disk-size=100GB",
                "--boot-disk-type=pd-balanced", "--scopes=https://www.googleapis.com/auth/cloud-platform",
                "--metadata-from-file=startup-script=" + str(startup),
                "--labels=model=unet,purpose=training-smoke", "--format=json", check=False)
            if result.returncode == 0:
                zone = candidate
                break
            unavailable_type = "machine type with name" in result.stderr.lower() and "does not exist in zone" in result.stderr.lower()
            if not unavailable_type and "stockout" not in result.stderr and "RESOURCE_POOL_EXHAUSTED" not in result.stderr:
                raise RuntimeError(result.stderr)
            print("No " + args.provisioning_model.lower() + " capacity in " + candidate, flush=True)
    if zone is None:
        save(stage="no-capacity")
        raise RuntimeError("No capacity in the requested zones")
    save(stage="running", zone=zone)
    print(f"{run_id}: {zone}\nimage: {image}\nresults: {output}", flush=True)
    finished = None
    previous = ""
    start = time.monotonic()
    try:
        while time.monotonic() - start < 3900:
            response = gcloud("compute", "instances", "get-serial-port-output", run_id,
                              "--zone=" + zone, "--port=1", "--start=0", check=False)
            if response.returncode:
                if "not found" in response.stderr.lower():
                    break
                print(response.stderr, flush=True)
                time.sleep(30)
                continue
            content = response.stdout
            (output / "serial.log").write_text(content)
            new = content[len(previous):] if content.startswith(previous) else content
            if new:
                print("\n".join(new.splitlines()[-15:]), flush=True)
            previous = content
            finished = re.search(r"ARCASLEEP_FINISHED run=" + re.escape(run_id)
                                 + r" training_exit=(\d+) upload_exit=(\d+)", content)
            if finished:
                break
            time.sleep(30)
        if finished:
            save(training_exit=int(finished[1]), upload_exit=int(finished[2]))
            if finished[2] == "0":
                download = gcloud("storage", "rsync", "--recursive",
                    f"gs://{args.results_bucket}/training_outputs/{run_id}/", str(output))
                print(download.stdout + download.stderr, flush=True)
            save(stage="succeeded" if finished[1] == finished[2] == "0" else "failed")
        else:
            save(stage="interrupted")
    finally:
        deletion = gcloud("compute", "instances", "delete", run_id,
                          "--zone=" + zone, "--quiet", check=False)
        deleted = deletion.returncode == 0 or "not found" in deletion.stderr.lower()
        save(vm_deleted=deleted)
        print(deletion.stdout + deletion.stderr, flush=True)
        if not deleted:
            print("VM cleanup failed; its one-hour automatic deletion remains scheduled.", flush=True)
    return 0 if finished and finished[1] == finished[2] == "0" and state.get("vm_deleted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
