#!/usr/bin/env python3
"""Trace unchanged resident Example 1 kernels; no solver instrumentation edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from example1_comparison import command, environment, parse_trial
from wait_example1_comparison import HZ, activity, physical_cpus, sample, stop_our_process, verify_binaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--gpu-executable", type=Path,
                        help="Explicit alternative GPU executable (requires --gpu-sha256)")
    parser.add_argument("--gpu-sha256", help="Expected digest for the explicit executable")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("frames must be positive")
    if bool(args.gpu_executable) != bool(args.gpu_sha256):
        parser.error("--gpu-executable and --gpu-sha256 must be supplied together")
    repo = Path(__file__).resolve().parent.parent
    root = args.output.resolve()
    root.mkdir(exist_ok=True)
    if any(p.name != "supervisor.log" for p in root.iterdir()):
        raise RuntimeError("Use a fresh profiling directory")
    state = dict(status="waiting_for_idle", pid=os.getpid(), frames=args.frames,
                 kind="diagnostic trace; not a clean timing baseline", background_observations=[])

    def save(status, **fields):
        state.update(status=status, updated_unix=time.time(), **fields)
        temporary = root / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "status.json")

    import signal
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    process = None
    owned = set()

    def verify_inputs():
        if args.gpu_executable:
            if hashlib.sha256(args.gpu_executable.read_bytes()).hexdigest() != args.gpu_sha256:
                raise RuntimeError("Explicit GPU executable changed; refusing mixed-version results")
        else:
            verify_binaries(repo)

    def profile_sample(pid):
        snapshot = sample(pid)
        # Nsight launches its frontend/target in additional process groups.
        # Identify only this trace's unique output path, then its descendants;
        # never exempt another profiler just because it has the same name.
        parents = {}
        for key, item in snapshot["processes"].items():
            try:
                entry = Path(f"/proc/{item['pid']}")
                raw = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                parents[item["pid"]] = int(raw[1])
                argv_bytes = (entry / "cmdline").read_bytes()
                daemon_session = b"--session-name\0profile-" + str(pid).encode() + b"\0"
                if str(root).encode() in argv_bytes or daemon_session in argv_bytes:
                    owned.add(key)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        changed = True
        while changed:
            changed = False
            own_pids = {key[0] for key in owned if key in snapshot["processes"]} | {pid}
            for key, item in snapshot["processes"].items():
                if key not in owned and parents.get(item["pid"]) in own_pids:
                    owned.add(key)
                    changed = True
        for key in owned:
            item = snapshot["processes"].get(key)
            if item and item["pid"] != os.getpid() and item["group"] != pid:
                snapshot["own_cpu"] += item["ticks"] / HZ
                item["group"] = pid
        return snapshot
    try:
        verify_inputs()
        cpus = physical_cpus()
        clean = 0
        while clean < 3:
            before = sample()
            time.sleep(10)
            after = sample()
            reasons, load = activity(before, after, [], max_background_cores=2)
            # GPU kernel attribution is not a clean wall-time benchmark.
            # Record low-level filesystem activity, but do not restart an
            # otherwise exclusive GPU trace on system-wide I/O-wait spikes.
            blockers = [r for r in reasons if r != "I/O wait"]
            clean = 0 if blockers else clean + 1
            save("waiting_for_idle", reasons=blockers, idle_checks=clean)
        solver_argv = command(repo, root / "frames", "gpu", args.frames)
        if args.gpu_executable:
            solver_argv[0] = str(args.gpu_executable.resolve())
        argv = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                "--stats=false", "--export=sqlite", "--output=" + str(root / "trace"),
                "stdbuf", "-oL", "-eL", "taskset", "--cpu-list", str(cpus[0]),
                *solver_argv]
        env = environment("gpu")
        metadata = dict(command=argv, omp={k:v for k,v in env.items() if k.startswith("OMP_")},
                        cpu_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                        executable_sha256=hashlib.sha256(Path(solver_argv[0]).read_bytes()).hexdigest(),
                        gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                                     "--format=csv,noheader"], text=True).strip())
        (root / "command.json").write_text(json.dumps(metadata, indent=2) + "\n")
        before = sample()
        start = time.monotonic()
        with (root / "run.log").open("w") as log:
            process = subprocess.Popen(argv, cwd=repo, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            save("profiling", child_pid=process.pid)
            while True:
                time.sleep(5)
                done = process.poll() is not None
                after = profile_sample(process.pid)
                # A detached profiler target can exit before its parent reaps
                # it, temporarily dropping sampled owned CPU totals. Never
                # interpret that negative delta as somebody else's CPU load.
                before["own_cpu"] = min(before["own_cpu"], after["own_cpu"])
                reasons, load = activity(before, after, [], process.pid, max_background_cores=2)
                before = after
                state["background_observations"].append(dict(elapsed_s=time.monotonic()-start,
                                                            external_cpu_cores=load, reasons=reasons))
                blockers = [r for r in reasons if r != "I/O wait"]
                if blockers:
                    raise RuntimeError("External workload detected: " + "; ".join(blockers))
                if done:
                    break
                if time.monotonic() - start > 1200:
                    raise RuntimeError("20-minute profile cap; partial trace is not a complete run")
                save("profiling")
        if process.returncode:
            raise RuntimeError(f"Profiler/application returned {process.returncode}")
        verify_inputs()
        parsed = parse_trial(root / "run.log", root / "frames", args.frames, "gpu")
        parsed["instrumented"] = True
        (root / "instrumented_run.json").write_text(json.dumps(parsed, indent=2) + "\n")
        with (root / "kernel_stats.txt").open("w") as output:
            subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum",
                            str(root / "trace.sqlite")], stdout=output, stderr=subprocess.STDOUT,
                           check=True, timeout=120)
        save("complete", report=str(root / "trace.nsys-rep"))
    except BaseException as error:
        save("stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process is not None:
            stop_our_process(process)


if __name__ == "__main__":
    main()
