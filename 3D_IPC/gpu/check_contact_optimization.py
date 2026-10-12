#!/usr/bin/env python3
"""Clean GPU contact-optimization trials; preserve every log and output mesh."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
from statistics import median
import subprocess
import time

from example1_comparison import command, environment, parse_trial
from wait_example1_comparison import activity, physical_cpus, sample, stop_our_process


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-frames", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--reference-count", type=int,
                        help="Compare only this many initial frames when extending a validated run")
    args = parser.parse_args()
    if args.trials < 1 or args.frames < 1:
        parser.error("positive trials and frames required")
    reference_count = args.frames if args.reference_count is None else args.reference_count
    if not 1 <= reference_count <= args.frames:
        parser.error("reference-count must be between 1 and frames")
    repo = Path(__file__).resolve().parent.parent
    root = args.output.resolve()
    root.mkdir(exist_ok=False)
    reference = args.reference_frames.resolve()
    expected = {f"frame_{i:04d}.obj": digest(reference / f"frame_{i:04d}.obj")
                for i in range(reference_count + 1)}
    executable = args.executable.resolve()
    env, cpu = environment("gpu"), physical_cpus()[0]
    results = []
    state = dict(status="waiting_for_idle", pid=os.getpid(), trials=results,
                 executable=str(executable), sha256=args.sha256, reference=str(reference),
                 reference_count=reference_count, monitor_interval_s=5,
                 max_background_cores=2, background_observations=[])

    def save(status, **extra):
        state.update(status=status, updated_unix=time.time(), **extra)
        (root / "status.json").write_text(json.dumps(state, indent=2) + "\n")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    process = None
    try:
        for trial in range(args.trials):
            clean = 0
            while clean < 2:
                a = sample()
                time.sleep(5)
                b = sample()
                reasons, load = activity(a, b, [], max_background_cores=2)
                clean = 0 if reasons else clean + 1
                save("waiting_for_idle", reasons=reasons, idle_checks=clean)
            if digest(executable) != args.sha256:
                raise RuntimeError("Executable changed before trial")
            out = root / f"trial_{trial:02d}"
            argv = command(repo, out, "gpu", args.frames)
            argv[0] = str(executable)
            argv = ["taskset", "--cpu-list", str(cpu), *argv]
            (root / f"trial_{trial:02d}_command.json").write_text(json.dumps(
                dict(argv=argv, omp={k:v for k,v in env.items() if k.startswith("OMP_")}), indent=2) + "\n")
            log_path = root / f"trial_{trial:02d}.log"
            start = time.monotonic()
            with log_path.open("w") as log:
                process = subprocess.Popen(argv, cwd=repo, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                save("running", trial=trial, child_pid=process.pid)
                before = sample(process.pid)
                while process.poll() is None:
                    # Use the same five-second load window as the profiling
                    # supervisor. Short startup spikes are not sustained load;
                    # active external simulations/GPU jobs still always block.
                    time.sleep(5)
                    after = sample(process.pid)
                    before["own_cpu"] = min(before["own_cpu"], after["own_cpu"])
                    reasons, load = activity(before, after, [], process.pid, max_background_cores=2)
                    state["background_observations"].append(dict(
                        trial=trial, elapsed_s=time.monotonic()-start,
                        external_cpu_cores=load, reasons=reasons))
                    if reasons:
                        raise RuntimeError("Trial contaminated: " + "; ".join(reasons))
                    if time.monotonic() - start > 1200:
                        raise RuntimeError("20-minute trial limit")
                    before = after
            if process.returncode:
                raise RuntimeError(f"Simulation failed with {process.returncode}")
            if digest(executable) != args.sha256:
                raise RuntimeError("Executable changed during trial")
            parsed = parse_trial(log_path, out, args.frames, "gpu")
            identical = {name: digest(out / name) == value for name, value in expected.items()}
            parsed["byte_identical_to_original_gpu"] = identical
            results.append(parsed)
            save("validated_trial", trial=trial)
            if not all(identical.values()):
                raise RuntimeError("GPU trajectory differs from saved original GPU outputs")
        report = dict(trials=results, all_reference_frames_byte_identical=True,
                      reference_count=reference_count,
                      median_total_solver_s=median(r["total_solver_s"] for r in results),
                      median_average_frame_s=median(r["average_frame_s"] for r in results),
                      note="Same GPU method before/after optimization; not CPU trajectory parity.")
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        save("complete")
    except BaseException as error:
        save("stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process is not None:
            stop_our_process(process)


if __name__ == "__main__":
    main()
