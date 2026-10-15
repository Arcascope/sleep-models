#!/usr/bin/env python3
"""Run training on GCP GPUs: one L4 VM per phase, results collected from GCS.

    python3 deploy/scripts/run_task_gcp.py [options] FAMILY IMAGE [PHASE ...]

The GCP counterpart of deploy/scripts/run_task.sh. FAMILY is a model family
(bidoze, unet); IMAGE is a training image in that family's Artifact Registry
repository. Each PHASE (fold0, fold1, ..., final) runs on its own VM under one
run id; with no PHASE, one VM runs every phase in order. Each VM uploads to
gs://BUCKET/training_outputs/RUN_ID/ (training/PHASE/ as on S3, plus
gcp/PHASE/ with its log and exit code), then the launcher downloads the
results into FAMILY/results/RUN_ID/ and deletes the VM.

The launcher stays in the foreground until every VM has finished, so run it in
the background. A VM that outlives --max-hours by 30 minutes is deleted by GCE.
Requires authenticated gcloud.
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
import threading
import time

FAMILIES = {
    "unet": {"prefix": "UNET", "config": "config/training.json"},
    "bidoze": {"prefix": "BIDOZE", "config": "config/training-balanced.json"},
}
MACHINES = ["g2-standard-4", "g2-standard-8", "a2-highgpu-1g"]
TRAINING_LABEL = "purpose=training"


def label(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "-", value.lower())[:63]


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", choices=sorted(FAMILIES))
    parser.add_argument("image")
    parser.add_argument("phases", nargs="*", metavar="PHASE")
    parser.add_argument("--config", help="recipe inside the image, relative to /opt/FAMILY "
                        "(default: unet config/training.json, bidoze config/training-balanced.json)")
    parser.add_argument("--run-id", help="run id: letters, digits, '.', '_' and '-' "
                        "(default FAMILY-gcp-<UTC launch time>)")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="extra container environment; repeatable")
    parser.add_argument("--max-hours", type=int, default=24,
                        help="stop a phase after N hours (default 24); the VM still uploads")
    parser.add_argument("--queue", action="store_true",
                        help="wait for free VM slots instead of refusing the launch")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; launch nothing")
    parser.add_argument("--project", default=os.environ.get("ARCASLEEP_PROJECT", "shiftapp-arcascope"))
    parser.add_argument("--zones", nargs="+", default=["auto"],
                        help="auto discovers GPU zones; otherwise restrict attempts to the listed zones")
    parser.add_argument("--results-bucket", default="arcascope-sleep-models-data")
    parser.add_argument("--machine-type", choices=MACHINES, default="g2-standard-8",
                        help="g2-standard-8: L4, 32 GiB host RAM (default); g2-standard-4: 16 GiB; "
                        "a2-highgpu-1g: A100 40 GB")
    parser.add_argument("--provisioning-model", choices=["STANDARD", "SPOT"], default="STANDARD",
                        help="SPOT is cheaper but can be preempted mid-training, losing the phase")
    parser.add_argument("--max-vms", type=int,
                        default=int(os.environ.get("ARCASLEEP_GCP_MAX_VMS", "4")),
                        help="cap on training VMs in the project at once, all agents included (default 4)")
    args = parser.parse_args()

    if args.max_hours < 1:
        parser.error("--max-hours takes a whole number >= 1")
    for item in args.env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", item):
            parser.error("--env takes KEY=VALUE")
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("--run-id takes letters, digits, '.', '_' and '-'")
    if "auto" in args.zones and args.zones != ["auto"]:
        parser.error("Use --zones auto alone, or list explicit zones")
    for phase in args.phases:
        if not re.fullmatch(r"[A-Za-z0-9]+([/-][A-Za-z0-9]+)*", phase):
            parser.error("phase names are letters and digits, such as fold0 or final, "
                         "or experiment phases such as random/final")
    if len(set(args.phases)) != len(args.phases):
        parser.error("a phase is named twice")

    family = FAMILIES[args.family]
    config = args.config or family["config"]
    phases = args.phases or ["all"]
    run_id = args.run_id or f"{args.family}-gcp-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = root / args.family / "results" / run_id

    def gcloud(*parts: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(["gcloud", *parts, "--project=" + args.project],
                                capture_output=True, text=True)
        if check and result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        return result

    def active_vms() -> list[str]:
        listing = gcloud("compute", "instances", "list", "--filter=labels." + TRAINING_LABEL,
                         "--format=value(name,zone.basename(),status)").stdout
        return [line for line in listing.splitlines() if line.strip()]

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
    # An image from another family's repository would run that family's
    # harness under this family's environment names.
    if not re.search(r"/arcasleep-train/" + args.family + r"@sha256:", image):
        raise SystemExit(f"{image} is not in the {args.family} repository")

    if args.zones == ["auto"]:
        accelerator = "nvidia-tesla-a100" if args.machine_type.startswith("a2-") else "nvidia-l4"
        available = gcloud("compute", "accelerator-types", "list", "--filter=name=" + accelerator,
                           "--format=value(zone.basename())").stdout
        zones = sorted({z.strip() for z in available.splitlines()
                        if re.fullmatch(r"[a-z]+-[a-z]+\d+-[a-z]", z.strip())},
                       key=lambda z: (0 if z.startswith("us-") else 1 if z.startswith("europe-") else 2, z))
        if not zones:
            raise RuntimeError("No zones returned for " + accelerator)
    else:
        zones = args.zones

    active = active_vms()
    free = args.max_vms - len(active)
    print(f"==> {args.family}  {image}")
    print(f"    recipe   {config}")
    print(f"    phases   {' '.join(args.phases) if args.phases else 'all, in one VM'}")
    print(f"    run id   {run_id}")
    print(f"    machine  {args.machine_type} {args.provisioning_model}")
    print(f"    slots    {free} free of {args.max_vms} ({len(active)} active)")
    print(f"    limit    {args.max_hours} h per phase")
    print(f"    zones    {len(zones)} candidates, first {zones[0]}")
    for line in active:
        print(f"    active   {line}")
    if args.dry_run:
        return 0
    if len(phases) > free and not args.queue:
        print(f"needs {len(phases)} slots and {free} are free; pass --queue to wait for them,\n"
              "or name fewer phases", flush=True)
        return 1

    output.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {
        "run_id": run_id, "family": args.family, "image": image, "project": args.project,
        "machine_type": args.machine_type, "provisioning_model": args.provisioning_model,
        "config": config, "env": args.env, "max_hours": args.max_hours,
        "phases": {p: {"stage": "waiting"} for p in phases}}
    lock = threading.Lock()

    def save(phase: str, **updates: object) -> None:
        with lock:
            state["phases"][phase].update(updates)  # type: ignore[index]
            (output / "gcp-run.json").write_text(json.dumps(state, indent=2) + "\n")

    save(phases[0])
    template = (root / "deploy/gcp/train-startup.sh").read_text()
    extra_env = shlex.join([token for item in args.env for token in ("--env", item)])
    timeout = args.max_hours * 3600
    created: dict[str, str] = {}

    def run_phase(phase: str) -> None:
        vm = label(f"{run_id}-{phase}")
        phase_dir = output / "gcp" / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        values = {"__RUN_ID__": run_id, "__PHASE__": phase, "__IMAGE_REF__": image,
                  "__RESULTS_BUCKET__": args.results_bucket, "__ENV_PREFIX__": family["prefix"],
                  "__TRAINING_CONFIG__": config, "__TIMEOUT_SECONDS__": str(timeout),
                  "__HARNESS_ARGS__": "" if phase == "all" else shlex.quote(phase),
                  "__EXTRA_ENV__": extra_env}
        script = template
        for key, value in values.items():
            script = script.replace(key, value if key in ("__HARNESS_ARGS__", "__EXTRA_ENV__") else shlex.quote(value))
        zone = None
        finished = None
        try:
            with tempfile.TemporaryDirectory(prefix="arcasleep-train-") as temporary:
                startup = Path(temporary) / "startup.sh"
                startup.write_text(script)
                while args.queue and len(active_vms()) >= args.max_vms:
                    print(f"[{phase}] all {args.max_vms} VM slots busy; checking again in 60 s", flush=True)
                    time.sleep(60)
                for candidate in zones:
                    result = gcloud("compute", "instances", "create", vm,
                        "--zone=" + candidate, "--machine-type=" + args.machine_type,
                        "--provisioning-model=" + args.provisioning_model, "--maintenance-policy=TERMINATE",
                        "--no-restart-on-failure", "--instance-termination-action=DELETE",
                        f"--max-run-duration={timeout + 1800}s", "--image-project=cos-cloud",
                        "--image-family=cos-stable", "--boot-disk-size=100GB",
                        "--boot-disk-type=pd-balanced", "--scopes=https://www.googleapis.com/auth/cloud-platform",
                        "--metadata-from-file=startup-script=" + str(startup),
                        f"--labels=model={args.family},{TRAINING_LABEL},run={label(run_id)},phase={label(phase)}",
                        "--format=json", check=False)
                    if result.returncode == 0:
                        zone = candidate
                        break
                    err = result.stderr
                    unavailable_type = "machine type with name" in err.lower() and "does not exist in zone" in err.lower()
                    quota_error = "QUOTA_EXCEEDED" in err or ("Quota '" in err and "exceeded" in err)
                    regional_quota = quota_error and "_ALL_REGIONS" not in err
                    if not unavailable_type and not regional_quota and "stockout" not in err and "RESOURCE_POOL_EXHAUSTED" not in err:
                        raise RuntimeError(err)
                    print(f"[{phase}] unable to allocate {args.provisioning_model.lower()} in {candidate}"
                          + (" (quota)" if regional_quota else ""), flush=True)
            if zone is None:
                save(phase, stage="no-capacity")
                return
            with lock:
                created[vm] = zone
            save(phase, stage="running", vm=vm, zone=zone)
            print(f"[{phase}] {vm} in {zone}", flush=True)
            previous = ""
            start = time.monotonic()
            while time.monotonic() - start < timeout + 1800 + 600:
                response = gcloud("compute", "instances", "get-serial-port-output", vm,
                                  "--zone=" + zone, "--port=1", "--start=0", check=False)
                if response.returncode:
                    if "not found" in response.stderr.lower():
                        break
                    print(f"[{phase}] {response.stderr.strip()}", flush=True)
                    time.sleep(60)
                    continue
                content = response.stdout
                (phase_dir / "serial.log").write_text(content)
                new = content[len(previous):] if content.startswith(previous) else content
                if new:
                    print("\n".join(f"[{phase}] {line}" for line in new.splitlines()[-5:]), flush=True)
                previous = content
                finished = re.search(r"ARCASLEEP_FINISHED run=" + re.escape(run_id) + " phase=" + re.escape(phase)
                                     + r" training_exit=(\d+) upload_exit=(\d+)", content)
                if finished:
                    break
                time.sleep(60)
            if finished:
                save(phase, training_exit=int(finished[1]), upload_exit=int(finished[2]))
                ok = finished[1] == finished[2] == "0"
                save(phase, stage="succeeded" if ok else "timeout" if finished[1] == "124" else "failed")
            else:
                save(phase, stage="interrupted")
        except Exception as error:  # a failed phase must not strand its siblings
            save(phase, stage="error", error=str(error)[-2000:])
            print(f"[{phase}] error: {error}", flush=True)
        finally:
            if zone is not None:
                deletion = gcloud("compute", "instances", "delete", vm, "--zone=" + zone, "--quiet", check=False)
                deleted = deletion.returncode == 0 or "not found" in deletion.stderr.lower()
                save(phase, vm_deleted=deleted)
                if deleted:
                    with lock:
                        created.pop(vm, None)
                else:
                    print(f"[{phase}] VM cleanup failed: {deletion.stderr.strip()}\n"
                          f"[{phase}] GCE deletes it on its own within 30 minutes of the {args.max_hours} h limit.", flush=True)

    slots = threading.Semaphore(max(1, min(len(phases), args.max_vms)))

    def worker(phase: str) -> None:
        with slots:
            run_phase(phase)

    threads = [threading.Thread(target=worker, args=(p,), daemon=True) for p in phases]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            while thread.is_alive():
                thread.join(timeout=5)
    finally:
        # Interrupted launcher: delete what is still running.
        for vm, zone in list(created.items()):
            gcloud("compute", "instances", "delete", vm, "--zone=" + zone, "--quiet", check=False)

    uploaded = [p for p, s in state["phases"].items() if s.get("upload_exit") == 0]  # type: ignore[union-attr]
    if uploaded:
        gcloud("storage", "rsync", "--recursive",
               f"gs://{args.results_bucket}/training_outputs/{run_id}/", str(output))
    summary = state["phases"]  # type: ignore[assignment]
    print(f"\nresults: {output}")
    for phase, entry in summary.items():  # type: ignore[union-attr]
        print(f"  {phase:8} {entry['stage']}" + (f" ({entry['zone']})" if "zone" in entry else ""))
    return 0 if all(e["stage"] == "succeeded" and e.get("vm_deleted") for e in summary.values()) else 1  # type: ignore[union-attr]


if __name__ == "__main__":
    raise SystemExit(main())
