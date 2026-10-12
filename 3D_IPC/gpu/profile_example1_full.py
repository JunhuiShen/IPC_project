#!/usr/bin/env python3
"""Wait for an idle server, then profile only the exact README Example 1 GPU run."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from analyze_example1_full_profile import load_baseline, log_frames, write_report
from example1_comparison import command, environment, parse_trial, verify_readme_command
from wait_example1_comparison import activity, physical_cpus, sample, stop_our_process, MONITORING_NOTE


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def checkpoint_manifest(outdir: Path, frames: int) -> dict:
    result = {}
    for prefix, suffix in (("frame_", ".geo"), ("state_", ".bin")):
        expected = {f"{prefix}{i:04d}{suffix}" for i in range(frames + 1)}
        actual = {path.name for path in outdir.glob(prefix + "*" + suffix)}
        if actual != expected:
            raise ValueError(f"Expected exactly frames 0..{frames} for {prefix}*{suffix}")
        for name in sorted(expected):
            path = outdir / name
            if not path.stat().st_size:
                raise ValueError("Empty result file: " + str(path))
            result[name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    return dict(kind="GPU diagnostic output manifest", frames=frames, files=result,
                note="Compare complete state_*.bin hashes against later GPU optimization runs for byte identity; GEO text alone is insufficient.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-baseline", type=Path, required=True)
    parser.add_argument("--gpu-executable", type=Path)
    parser.add_argument("--gpu-sha256", required=True)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--timeout", type=float, default=43200)
    parser.add_argument("--monitor-interval", type=float, default=10)
    parser.add_argument("--max-background-cores", type=float, default=2)
    args = parser.parse_args()
    if (not 1 <= args.frames <= 240 or not 0 < args.timeout < float("inf")
            or not 0 < args.monitor_interval <= 60 or not 0 <= args.max_background_cores < float("inf")):
        parser.error("Invalid frame count, timeout, monitoring interval, or background CPU ceiling")
    if len(args.gpu_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.gpu_sha256):
        parser.error("gpu-sha256 must be 64 lowercase hexadecimal characters")
    repo = Path(__file__).resolve().parent.parent
    verify_readme_command(repo)
    root = args.output.resolve()
    root.mkdir(exist_ok=True)
    if any(path.name != "supervisor.log" for path in root.iterdir()):
        raise RuntimeError("Use a fresh profiling directory; existing output is never replaced")
    baseline = load_baseline(args.cpu_baseline.resolve())
    (root / "cpu_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    executable = (args.gpu_executable or repo / "gpu/build/3D_sim_gpu_resident").resolve()
    cpus = physical_cpus()
    state = dict(kind="GPU-only instrumented diagnostic, not a clean benchmark", status="waiting_for_idle",
                 pid=os.getpid(), frames=args.frames, directory=str(root), readme_command=True,
                 gpu_sha256=args.gpu_sha256, cpu_affinity=cpus, monitoring_policy=MONITORING_NOTE,
                 max_background_cores=args.max_background_cores, background_observations=[])

    def save(status, **fields):
        state.update(status=status, updated_unix=time.time(), **fields)
        temporary = root / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "status.json")

    def verify():
        verify_readme_command(repo)
        if sha256(executable) != args.gpu_sha256:
            raise RuntimeError("GPU executable changed; refusing mixed-version diagnostics")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    lock = (repo / "gpu/.example1_comparison.lock").open("a+")
    process = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify()
        clean = 0
        while clean < 3:
            before = sample()
            time.sleep(args.monitor_interval)
            after = sample()
            reasons, load = activity(before, after, [], max_background_cores=args.max_background_cores)
            clean = 0 if reasons else clean + 1
            save("waiting_for_idle", reasons=reasons, idle_checks=clean, external_cpu_cores=load)
        verify()
        solver_argv = command(repo, root / "frames", "gpu", args.frames, readme_command=True)
        solver_argv[0] = str(executable)
        argv = ["stdbuf", "-oL", "-eL", "taskset", "--cpu-list", ",".join(map(str, cpus)), *solver_argv]
        env = environment("gpu", readme_command=True)
        env["IPC_GPU_PROFILE_CSV"] = str(root / "phase_frames.csv")
        metadata = dict(command=argv, solver_command=solver_argv,
                        environment={key: value for key, value in env.items() if key.startswith(("OMP_", "IPC_GPU_PROFILE"))},
                        executable_sha256=sha256(executable),
                        readme_sha256=sha256(repo / "README.md"),
                        cpu_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                        cpu_hardware=subprocess.check_output(["lscpu"], text=True),
                        gpu_hardware=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
                        profiling_note="Original README simulation arguments; only executable/output path differ. Timing instrumentation is enabled by environment, no simulation option is substituted.")
        (root / "command.json").write_text(json.dumps(metadata, indent=2) + "\n")
        before = sample()
        start = time.monotonic()
        with (root / "run.log").open("w") as log:
            process = subprocess.Popen(argv, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save("profiling", child_pid=process.pid, command=argv)
            while True:
                time.sleep(args.monitor_interval)
                done = process.poll() is not None
                after = sample(process.pid)
                reasons, load = activity(before, after, [], process.pid, max_background_cores=args.max_background_cores)
                iowait = max(0., after["iowait"] - before["iowait"])
                before = after
                state["background_observations"].append(dict(elapsed_s=time.monotonic() - start,
                    external_cpu_cores=load, reasons=reasons, system_iowait_ticks=iowait))
                if reasons:
                    raise RuntimeError("External workload detected: " + "; ".join(reasons))
                complete_frames = len(log_frames(root / "run.log"))
                save("profiling", completed_frames=complete_frames, elapsed_s=time.monotonic() - start)
                if done:
                    break
                if time.monotonic() - start > args.timeout:
                    raise RuntimeError("Profiling timeout; retained partial output is not a full-run result")
        if process.returncode:
            raise RuntimeError(f"GPU application returned {process.returncode}")
        verify()
        save("validating", child_pid=None)
        parsed = parse_trial(root / "run.log", root / "frames", args.frames, "gpu", readme_command=True)
        parsed.update(instrumented=True, process_elapsed_s=time.monotonic() - start)
        (root / "instrumented_run.json").write_text(json.dumps(parsed, indent=2) + "\n")
        manifest = checkpoint_manifest(root / "frames", args.frames)
        (root / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        write_report(root, args.frames)
        save("complete", completed_frames=args.frames, validated_geo_frames=args.frames + 1,
             validated_binary_checkpoints=args.frames + 1, report=str(root / "profile_report.txt"))
    except BaseException as error:
        save("stopped", error=f"{type(error).__name__}: {error}")
        try:
            write_report(root, args.frames, allow_partial=True)
        except (OSError, ValueError, KeyError):
            pass
        raise
    finally:
        if process is not None:
            stop_our_process(process)
        lock.close()


if __name__ == "__main__":
    main()
