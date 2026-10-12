#!/usr/bin/env python3
"""Check that adding stage timers preserves binary simulation checkpoints.

Run only after checking the server is available. This is a correctness test,
not a performance comparison. No CPU solver or original output is modified.
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from example1_comparison import command, environment, verify_readme_command, _geo_positions
from wait_example1_comparison import physical_cpus


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--instrumented", type=Path, required=True)
    parser.add_argument("--instrumented-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--restart-frame", type=int)
    parser.add_argument("--restart-checkpoint", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.frames < 1 or args.timeout <= 0:
        parser.error("frames and timeout must be positive")
    if (args.restart_frame is None) != (args.restart_checkpoint is None):
        parser.error("restart-frame and restart-checkpoint must be supplied together")
    if args.restart_frame is not None and args.restart_frame < 0:
        parser.error("restart-frame must be nonnegative")
    repo = Path(__file__).resolve().parent.parent
    verify_readme_command(repo)
    root = args.output.resolve()
    root.mkdir(exist_ok=False)
    baseline, instrumented = args.baseline.resolve(strict=True), args.instrumented.resolve(strict=True)
    for path, expected in ((baseline, args.baseline_sha256), (instrumented, args.instrumented_sha256)):
        if digest(path) != expected:
            raise RuntimeError(f"Executable hash mismatch: {path}")
    start = 1 if args.restart_frame is None else args.restart_frame + 1
    end = start + args.frames - 1
    indices = list(range(start, end + 1))
    affinity = ",".join(map(str, physical_cpus()))
    runs = []
    report = {"kind": "profiling instrumentation byte-identity validation",
              "first_frame": start, "last_frame": end, "runs": runs,
              "baseline_sha256": args.baseline_sha256,
              "instrumented_sha256": args.instrumented_sha256, "passed": False,
              "note": "Restart tests reset solver caches identically in each run; not a continuous-run performance baseline."}
    try:
        for name, executable, enabled in (("baseline", baseline, False),
                                           ("profile_off", instrumented, False),
                                           ("profile_on", instrumented, True)):
            out = root / name
            argv = command(repo, out, "gpu", end, readme_command=True)
            argv[0] = str(executable)
            if args.restart_frame is not None:
                out.mkdir()
                shutil.copy2(args.restart_checkpoint, out / f"state_{args.restart_frame:04d}.bin")
                argv.extend(("--restart_frame", str(args.restart_frame)))
            argv = ["taskset", "-c", affinity, *argv]
            env = environment("gpu", readme_command=True)
            env.pop("IPC_GPU_PROFILE_CSV", None)
            if enabled:
                env["IPC_GPU_PROFILE_CSV"] = str(root / "stages.csv")
            log = root / (name + ".log")
            print("Starting", name, "frames", start, "through", end, flush=True)
            with log.open("w") as output:
                subprocess.run(argv, cwd=repo, env=env, stdout=output,
                               stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
            got = [int(x) for x in re.findall(r"^Frame\s+(\d+)\s*\|", log.read_text(), re.M)]
            if got != indices:
                raise RuntimeError(f"{name}: incomplete frame log: {got}")
            check_indices = [0, *indices] if args.restart_frame is None else indices
            hashes = {}
            for frame in check_indices:
                mesh = out / f"frame_{frame:04d}.geo"
                _geo_positions(mesh)
                for path in (mesh, out / f"state_{frame:04d}.bin"):
                    hashes[path.name] = digest(path)
            runs.append({"name": name, "argv": argv, "log": str(log), "hashes": hashes})
            if hashes != runs[0]["hashes"]:
                mismatch = [key for key, value in hashes.items() if runs[0]["hashes"].get(key) != value]
                raise RuntimeError(f"{name}: byte mismatch: {mismatch}")
        with (root / "stages.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        if not rows:
            raise RuntimeError("Profiling produced no CSV data")
        report.update(passed=True, profiling_rows=len(rows))
        print("PASS: baseline/profile-off/profile-on GEO and binary checkpoints match", flush=True)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
