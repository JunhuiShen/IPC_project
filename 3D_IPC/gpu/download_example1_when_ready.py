#!/usr/bin/env python3
"""Download this task's completed Example 1 comparison and validate local meshes.

Read-only on the server. Authentication failures are retried without prompting
for credentials. A stopped run downloads partial artifacts but is not reported
as a completed comparison. This script never starts/stops simulations.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from example1_comparison import parse_trial, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()
    remote = Path(args.remote_root)
    expected_parent = Path("/homes/home01/z/zhhzhao/Desktop/IPC_project/3D_IPC/gpu")
    if remote.parent != expected_parent or not remote.name.startswith(("example1_cpu_gpu_", "example1_full_")):
        parser.error("remote-root must be an Example 1 comparison directory")
    local = args.local_root.resolve()
    local.mkdir(parents=True, exist_ok=True)
    ssh = ["ssh", "-S", args.socket, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    host = "zhhzhao@biot.math.ucdavis.edu"
    read_status = shlex.join(["python3", "-c",
                             "import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())",
                             str(remote / "status.json")])

    def save(status, **fields):
        data = dict(status=status, updated_unix=time.time(), remote_root=str(remote),
                    local_root=str(local), **fields)
        path = local / "download_status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(path)

    deadline = time.monotonic() + 24 * 3600
    previous = None
    while time.monotonic() < deadline:
        try:
            response = subprocess.run(ssh + [host, read_status], check=True,
                                      capture_output=True, text=True, timeout=30)
            state = json.loads(response.stdout)
            key = (state["status"], state.get("current"), len(state["runs"]))
            if key != previous:
                print("Remote progress:", key, flush=True)
                previous = key
            save("waiting", remote_status=state["status"], completed_runs=len(state["runs"]))
            if state["status"] not in ("complete", "stopped"):
                time.sleep(30)
                continue
            if shutil.disk_usage(local).free < 2 * 1024**3:
                raise RuntimeError("Less than 2 GiB local disk space; download paused")
            save("downloading", remote_status=state["status"])
            subprocess.run(["rsync", "-av", "--include=*/", "--include=*.obj", "--include=*.geo",
                            "--include=*.json", "--include=*.log", "--exclude=*",
                            "-e", shlex.join(ssh), f"{host}:{remote}/", str(local) + "/"],
                           check=True, timeout=600)
            if state["status"] == "stopped":
                save("partial_download", error=state.get("error"))
                print("Run stopped; partial files downloaded. No complete timing result.", flush=True)
                return
            trials = []
            for run in state["runs"]:
                parsed = parse_trial(local / Path(run["log"]).name,
                                     local / Path(run["outdir"]).name,
                                     run["frames"], run["mode"],
                                     readme_command=run.get("readme_command", state.get("readme_command", False)))
                parsed.update(command=run["command"], omp=run["omp"])
                parsed.update({key: run[key] for key in (
                    "max_observed_system_iowait_cores", "observed_system_iowait_cpu_s") if key in run})
                trials.append(parsed)
            report = summarize(trials)
            if state.get("monitoring_policy"):
                report["notes"].append(state["monitoring_policy"])
            report.update(source_status=str(local / "status.json"), runs=trials,
                          expected_binaries=state["expected_binaries"],
                          readme_sha256=state.get("readme_sha256"),
                          monitoring_policy=state.get("monitoring_policy"),
                          max_background_cores=state["max_background_cores"])
            (local / "local_report.json").write_text(json.dumps(report, indent=2) + "\n")
            save("complete", frames=trials[0]["frames"], validated_runs=len(trials),
                 report=str(local / "local_report.json"))
            print("Downloaded and validated:", local, flush=True)
            return
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError) as error:
            save("retrying", error=str(error))
            print("Retrying:", error, flush=True)
            time.sleep(30)
    save("stopped", error="24-hour download wait limit reached")


if __name__ == "__main__":
    main()
