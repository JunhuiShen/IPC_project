#!/usr/bin/env python3
"""Download completed diagnostic timings/GEOs; leave full binary states on the server."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from analyze_example1_full_profile import write_report
from example1_comparison import parse_trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()
    remote = Path(args.remote_root)
    expected_parent = Path("/homes/home01/z/zhhzhao/Desktop/IPC_project/3D_IPC/gpu")
    if remote.parent != expected_parent or not remote.name.startswith(("example1_full_profile_", "profile_all_frames_")):
        parser.error("remote-root must be a designated Example 1 full-profile directory")
    local = args.local_root.resolve()
    local.mkdir(parents=True, exist_ok=True)
    ssh = ["ssh", "-S", args.socket, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    host = "zhhzhao@biot.math.ucdavis.edu"
    read_status = shlex.join(["python3", "-c", "import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())", str(remote / "status.json")])

    def save(status, **fields):
        data = dict(status=status, updated_unix=time.time(), remote_root=str(remote), local_root=str(local), **fields)
        temporary = local / "download_status.tmp"
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(local / "download_status.json")

    deadline = time.monotonic() + 48 * 3600
    previous = None
    while time.monotonic() < deadline:
        try:
            response = subprocess.run(ssh + [host, read_status], check=True, capture_output=True, text=True, timeout=30)
            state = json.loads(response.stdout)
            key = (state["status"], state.get("completed_frames", 0))
            if key != previous:
                print("GPU profile progress:", key, flush=True)
                previous = key
            save("waiting", remote_status=state["status"], completed_frames=state.get("completed_frames", 0))
            if state["status"] not in ("complete", "stopped"):
                time.sleep(30)
                continue
            if shutil.disk_usage(local).free < 2 * 1024**3:
                raise RuntimeError("Less than 2 GiB local disk space; download paused")
            save("downloading", remote_status=state["status"])
            subprocess.run(["rsync", "-av", "--include=*/", "--include=*.geo", "--include=*.csv",
                            "--include=*.json", "--include=*.txt", "--include=*.log", "--exclude=*",
                            "-e", shlex.join(ssh), f"{host}:{remote}/", str(local) + "/"], check=True, timeout=1200)
            if state["status"] == "stopped":
                try:
                    write_report(local, state["frames"], allow_partial=True)
                except (OSError, ValueError, KeyError):
                    pass
                save("partial_download", error=state.get("error"))
                print("Partial files downloaded; no full-simulation completion claim.", flush=True)
                return
            count = state["frames"]
            parsed = parse_trial(local / "run.log", local / "frames", count, "gpu", readme_command=True)
            manifest = json.loads((local / "checkpoint_manifest.json").read_text())
            expected = {f"{prefix}{i:04d}{suffix}" for i in range(count + 1)
                        for prefix, suffix in (("frame_", ".geo"), ("state_", ".bin"))}
            if set(manifest["files"]) != expected or manifest["frames"] != count:
                raise ValueError("Output hash manifest does not contain exactly every GEO/checkpoint")
            for name, item in manifest["files"].items():
                if (not isinstance(item["bytes"], int) or item["bytes"] <= 0 or len(item["sha256"]) != 64
                        or any(c not in "0123456789abcdef" for c in item["sha256"])):
                    raise ValueError("Invalid checkpoint/GEO hash metadata: " + name)
                if name.endswith(".geo"):
                    path = local / "frames" / name
                    if path.stat().st_size != item["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                        raise ValueError("Downloaded GEO does not match server hash: " + name)
            write_report(local, count)
            (local / "local_validation.json").write_text(json.dumps(dict(
                parsed=parsed, validated_geo_frames=count + 1, binary_checkpoints="Retained on server; hashes downloaded, bytes not downloaded",
                binary_checkpoint_manifest=str(local / "checkpoint_manifest.json")), indent=2) + "\n")
            save("complete", frames=count, validated_geo_frames=count + 1, report=str(local / "profile_report.txt"))
            print("Downloaded and validated GPU profile:", local, flush=True)
            return
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError) as error:
            save("retrying", error=str(error))
            print("Retrying profile download:", error, flush=True)
            time.sleep(30)
    save("stopped", error="48-hour download wait limit reached")


if __name__ == "__main__":
    main()
