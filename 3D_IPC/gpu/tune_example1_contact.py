#!/usr/bin/env python3
"""Clean sequential contact-scheduling trials; preserve exact GPU output bytes.

Only the launch/scratch worker configuration varies. Every cold restart has an
immutable-baseline restart with the SAME checkpoint and reset solver caches.
Continuous-profile frame hashes are deliberately not used as restart references.
This does not estimate a complete 240-frame speedup. No production files change.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
from statistics import median
import subprocess
import time

from example1_comparison import command, environment, verify_readme_command, _geo_positions, _one_timing
from wait_example1_comparison import activity, physical_cpus, sample, stop_our_process, MONITORING_NOTE, HZ

DEFAULT_CONFIGS = ("1024:8", "512:4", "1024:1", "1024:2", "1024:16", "2048:8", "2048:16", "4096:16")
SCENARIOS = {
    "late": dict(restart_frame=219, first_frame=220, last_frame=220),
    "middle": dict(restart_frame=110, first_frame=111, last_frame=111),
    "early": dict(restart_frame=None, first_frame=1, last_frame=2),
}
SMALL_WORKERS = "IPC_GPU_CONTACT_EXACT_SMALL_WORKERS"
SMALL_THREADS = "IPC_GPU_CONTACT_EXACT_SMALL_THREADS"
BASELINE_SHA256 = "a4314ee76222f71c71479bdea10e2270b4c1bd710d7ef8e190e9981a543e650a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_config(value: str) -> dict:
    try:
        workers, threads = map(int, value.split(":"))
    except (TypeError, ValueError) as error:
        raise ValueError("Configurations must be workers:threads") from error
    if (not 1 <= workers <= 4096 or threads not in (1, 2, 4, 8, 16, 32) or workers % threads):
        raise ValueError("Small workers must be 1..4096, divisible by threads in {1,2,4,8,16,32}")
    return dict(name=f"w{workers}_t{threads}", workers=workers, threads=threads)


def clean_environment(config: dict | None) -> dict:
    # A diagnostic shell must not accidentally enable stage timers, queue
    # readbacks, alternative large-worker choices, or unrelated GPU knobs.
    result = {key: value for key, value in environment("gpu", readme_command=True).items()
              if not key.startswith("IPC_GPU_")}
    if config is not None:
        result.update({SMALL_WORKERS: str(config["workers"]), SMALL_THREADS: str(config["threads"])})
    return result


def segment_command(repo: Path, out: Path, executable: Path, scenario: dict, cpus: list[int]) -> list[str]:
    argv = command(repo, out, "gpu", scenario["last_frame"], readme_command=True)
    argv[0] = str(executable)
    if scenario["restart_frame"] is not None:
        argv.extend(("--restart_frame", str(scenario["restart_frame"])))
    return ["stdbuf", "-oL", "-eL", "taskset", "--cpu-list", ",".join(map(str, cpus)), *argv]


def parse_segment(log: Path, out: Path, scenario: dict) -> dict:
    text = log.read_text()
    indices = list(range(scenario["first_frame"], scenario["last_frame"] + 1))
    if len(re.findall(r"^Simulation finished\.\s*$", text, re.M)) != 1:
        raise ValueError("Missing/duplicate completion marker")
    if re.search(r"^\s*(?:Error:|terminate called|Traceback\b|CUDA error)", text, re.M | re.I):
        raise ValueError("Run contains a failure diagnostic")
    counts = re.findall(r"^(?:Total\s+)?Vertices\s*:\s*(\d+)\s*$", text, re.M | re.I)
    if counts != ["10000"]:
        raise ValueError("Expected exactly one 10,000-vertex Example 1 scene")
    rows = re.findall(r"^Frame\s+(\d+)\s*\|\s*global_iters\s*=\s*(\d+)\s*\|\s*solver_time\s*=\s*([\d.eE+-]+)\s+ms\s*$", text, re.M)
    if [int(row[0]) for row in rows] != indices or any(int(row[1]) != 20 for row in rows):
        raise ValueError("Unexpected frame sequence or iteration count")
    durations = [float(row[2]) for row in rows]
    total = _one_timing(text, "Total solver time", "ms")
    average = _one_timing(text, "Average solver time", "ms/frame")
    simulation = _one_timing(text, "Total simulation time", "ms")
    if any(not math.isfinite(value) or value < 0 for value in [*durations, total, average, simulation]):
        raise ValueError("Invalid solver timing")
    tolerance = .00051 * (len(indices) + 1)
    if abs(sum(durations) - total) > tolerance or abs(average * len(indices) - total) > tolerance:
        raise ValueError("Total/average timing disagrees with frame timings")
    if simulation + .0011 < total:
        raise ValueError("Simulation time is less than solver time")
    output_indices = [0, *indices] if scenario["restart_frame"] is None else indices
    expected_geo = {f"frame_{i:04d}.geo" for i in output_indices}
    expected_bin = {f"state_{i:04d}.bin" for i in output_indices}
    if scenario["restart_frame"] is not None:
        expected_bin.add(f"state_{scenario['restart_frame']:04d}.bin")
    if ({path.name for path in out.glob("frame_*.geo")} != expected_geo
            or {path.name for path in out.glob("state_*.bin")} != expected_bin):
        raise ValueError("Unexpected output frame/checkpoint set")
    hashes = {}
    for frame in output_indices:
        mesh = out / f"frame_{frame:04d}.geo"
        _geo_positions(mesh)
        for path in (mesh, out / f"state_{frame:04d}.bin"):
            if not path.stat().st_size:
                raise ValueError("Empty result file")
            hashes[path.name] = dict(sha256=sha256(path), bytes=path.stat().st_size)
    return dict(first_frame=indices[0], last_frame=indices[-1], simulated_frames=len(indices),
                total_solver_s=total / 1000, average_solver_frame_s=total / 1000 / len(indices),
                total_simulation_s=simulation / 1000, frame_solver_s=dict(zip(map(str, indices), (d / 1000 for d in durations))),
                output_hashes=hashes)


def make_summary(state: dict) -> dict:
    scenarios = {}
    for name in state["scenarios"]:
        accepted = [run for run in state["runs"] if run["scenario"] == name and run.get("status") == "accepted"]
        reference = [run for run in accepted if run["variant"] == "baseline"]
        if not reference:
            continue
        ref_median = median(run["total_solver_s"] for run in reference)
        variants = []
        for variant in ["baseline", *(config["name"] for config in state["configs"])]:
            runs = [run for run in accepted if run["variant"] == variant]
            if not runs:
                continue
            times = [run["total_solver_s"] for run in runs]
            center = median(times)
            variants.append(dict(variant=variant, repetitions=len(runs), total_solver_s=times,
                                 median_total_solver_s=center, min_total_solver_s=min(times), max_total_solver_s=max(times),
                                 median_average_frame_s=median(run["average_solver_frame_s"] for run in runs),
                                 speedup_over_baseline=ref_median / center if center else None,
                                 binary_and_geo_byte_identical=True,
                                 expected_repetitions=state["repetitions"], complete=len(runs) == state["repetitions"]))
        candidates = [value for value in variants if value["variant"] != "baseline"]
        fastest = min(candidates, key=lambda value: value["median_total_solver_s"]) if candidates else None
        scenarios[name] = dict(**SCENARIOS[name], baseline_repetitions=len(reference), variants=variants,
                               fastest_observed_variant=fastest["variant"] if fastest else None)
    return dict(kind="clean cold-restart contact scheduling microbenchmarks", status=state["status"],
                error=state.get("error"), accepted_trials=sum(run.get("status") == "accepted" for run in state["runs"]),
                rejected_trials=[dict(name=run["name"], error=run.get("error"), mismatches=run.get("mismatches"))
                                 for run in state["runs"] if run.get("status") == "rejected"],
                scenarios=scenarios, candidate_executable=state["candidate"], baseline_executable=state["baseline"],
                notes=["Only exact contact scheduling environment knobs change; the documented simulation parameters stay fixed.",
                       "Instrumentation and diagnostic queue readbacks are OFF; all inherited IPC_GPU_* variables are cleared.",
                       "Cold-restart timings include fresh solver-cache setup and are compared only with the matching baseline restart, never continuous-profile frame timing.",
                       "Every accepted candidate must match all generated baseline binary checkpoints AND GEO files byte-for-byte.",
                       "Two repetitions are a small tuning sample. Stage-specific wins do not establish a full-240-frame speedup.",
                       "Candidate order reverses on alternate repetitions; the immutable baseline runs once per stage/repetition.",
                       "No full-run optimization claim is valid until separate continuous-run timing and all-frame checkpoint validation finish.",
                       MONITORING_NOTE])


def render_summary(summary: dict) -> str:
    lines = ["Example 1 exact-contact scheduling trials", "Status: " + summary["status"],
             f"Accepted trials: {summary['accepted_trials']}", ""]
    if summary["error"]:
        lines += ["Stopped reason: " + summary["error"], ""]
    for name, stage in summary["scenarios"].items():
        lines += [f"{name}: frames {stage['first_frame']}..{stage['last_frame']}; restart={stage['restart_frame']}",
                  "variant | n | median solver (s) | min..max (s) | solver/frame (ms) | baseline speedup | bytes"]
        for row in stage["variants"]:
            speedup = "n/a" if row["speedup_over_baseline"] is None else f"{row['speedup_over_baseline']:.3f}x"
            lines.append(f"{row['variant']} | {row['repetitions']} | {row['median_total_solver_s']:.6f} | "
                         f"{row['min_total_solver_s']:.6f}..{row['max_total_solver_s']:.6f} | "
                         f"{row['median_average_frame_s']*1000:.3f} | {speedup} | identical")
        lines += ["Fastest observed candidate: " + str(stage["fastest_observed_variant"]), ""]
    lines += ["Notes:", *("- " + line for line in summary["notes"])]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-sha256", default=BASELINE_SHA256)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--config", action="append", help="workers:threads; repeat to replace the default eight configurations")
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), action="append", help="Repeat; default late,middle,early")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800, help="Per-trial time limit in seconds")
    parser.add_argument("--monitor-interval", type=float, default=2)
    parser.add_argument("--idle-checks", type=int, default=3)
    parser.add_argument("--max-background-cores", type=float, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output is None or args.candidate is None or args.candidate_sha256 is None:
        parser.error("--output, --candidate, and --candidate-sha256 are required")
    if (args.repetitions < 1 or args.idle_checks < 1 or not 0 < args.timeout < float("inf")
            or not 0 < args.monitor_interval <= 60 or not 0 <= args.max_background_cores < float("inf")):
        parser.error("Invalid repetition count, idle checks, timeout, interval, or load ceiling")
    for digest in (args.baseline_sha256, args.candidate_sha256):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            parser.error("Executable hashes must be 64 lowercase hexadecimal characters")
    try:
        configs = [parse_config(value) for value in (args.config or DEFAULT_CONFIGS)]
    except ValueError as error:
        parser.error(str(error))
    if len({config["name"] for config in configs}) != len(configs):
        parser.error("Duplicate configurations")
    scenarios = args.scenario or list(SCENARIOS)
    if len(set(scenarios)) != len(scenarios):
        parser.error("Duplicate scenarios")
    needs_checkpoint = any(SCENARIOS[name]["restart_frame"] is not None for name in scenarios)
    if needs_checkpoint and args.checkpoint_root is None:
        parser.error("--checkpoint-root is required for restart scenarios")
    repo = Path(__file__).resolve().parent.parent
    verify_readme_command(repo)
    baseline = (args.baseline or repo / "gpu/profile_all_frames_20261010_lvvtq2dn/3D_sim_gpu_baseline").resolve(strict=True)
    candidate = args.candidate.resolve(strict=True)
    if baseline == candidate:
        parser.error("Baseline and candidate executable paths must be separate")
    root = args.output.resolve()
    root.mkdir(exist_ok=True)
    if any(path.name != "supervisor.log" for path in root.iterdir()):
        raise RuntimeError("Use a fresh tuning directory; previous output will not be replaced")
    (root / "runs").mkdir()
    cpus = physical_cpus()
    if cpus != list(range(64)):
        raise RuntimeError("Expected the verified physical CPU affinity 0..63 on this server")
    checkpoint_inputs = {}
    if needs_checkpoint:
        checkpoint_root = args.checkpoint_root.resolve(strict=True)
        manifest_path = (args.checkpoint_manifest or checkpoint_root.parent / "checkpoint_manifest.json").resolve(strict=True)
        manifest = json.loads(manifest_path.read_text())
        for name in scenarios:
            frame = SCENARIOS[name]["restart_frame"]
            if frame is None:
                continue
            path = checkpoint_root / f"state_{frame:04d}.bin"
            metadata = manifest["files"][path.name]
            digest = sha256(path)
            if digest != metadata["sha256"] or path.stat().st_size != metadata["bytes"]:
                raise RuntimeError("Checkpoint differs from the completed profile manifest: " + str(path))
            checkpoint_inputs[name] = dict(path=str(path), sha256=digest, bytes=path.stat().st_size,
                                           source_manifest=str(manifest_path), source_manifest_sha256=sha256(manifest_path))
    state = dict(kind="clean GPU contact scheduling trials", status="waiting_for_idle", pid=os.getpid(),
                 baseline=dict(path=str(baseline), sha256=args.baseline_sha256),
                 candidate=dict(path=str(candidate), sha256=args.candidate_sha256),
                 readme_sha256=sha256(repo / "README.md"), scenarios=scenarios, configs=configs,
                 checkpoint_inputs=checkpoint_inputs, repetitions=args.repetitions, cpu_affinity=cpus,
                 instrumentation=False, max_background_cores=args.max_background_cores,
                 monitoring_policy=MONITORING_NOTE, runs=[])

    def save(status, **values):
        state.update(status=status, updated_unix=time.time(), **values)
        for name, content in (("status.json", json.dumps(state, indent=2) + "\n"),
                              ("summary.json", json.dumps(make_summary(state), indent=2) + "\n"),
                              ("summary.txt", render_summary(make_summary(state)))):
            temporary = root / (name + ".tmp")
            temporary.write_text(content)
            temporary.replace(root / name)

    def verify_inputs():
        verify_readme_command(repo)
        if sha256(repo / "README.md") != state["readme_sha256"]:
            raise RuntimeError("README changed during tuning")
        for item in (state["baseline"], state["candidate"], *checkpoint_inputs.values()):
            if sha256(Path(item["path"])) != item["sha256"]:
                raise RuntimeError("Pinned tuning input changed: " + item["path"])
        for item in checkpoint_inputs.values():
            if sha256(Path(item["source_manifest"])) != item["source_manifest_sha256"]:
                raise RuntimeError("Pinned checkpoint manifest changed")

    def wait_idle(label):
        clean = 0
        while clean < args.idle_checks:
            before = sample()
            time.sleep(args.monitor_interval)
            after = sample()
            reasons, load = activity(before, after, [], max_background_cores=args.max_background_cores)
            clean = 0 if reasons else clean + 1
            save("waiting_for_idle", current=label, idle_checks=clean, reasons=reasons, external_cpu_cores=load)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    lock = (repo / "gpu/.example1_comparison.lock").open("a+")
    process = None
    trial = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_inputs()
        hardware = dict(cpu=subprocess.check_output(["lscpu"], text=True),
                        gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
                        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
        (root / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
        references = {}
        reference_names = {}
        for name in scenarios:
            scenario = SCENARIOS[name]
            for repetition in range(1, args.repetitions + 1):
                order = configs if repetition % 2 else list(reversed(configs))
                for config in [None, *order]:
                    variant = "baseline" if config is None else config["name"]
                    label = f"{len(state['runs']):03d}_{name}_r{repetition}_{variant}"
                    wait_idle(label)
                    verify_inputs()
                    trial_root = root / "runs" / label
                    trial_root.mkdir()
                    out = trial_root / "frames"
                    if scenario["restart_frame"] is not None:
                        out.mkdir()
                        source = Path(checkpoint_inputs[name]["path"])
                        copied = out / source.name
                        shutil.copy2(source, copied)
                        if sha256(copied) != checkpoint_inputs[name]["sha256"]:
                            raise RuntimeError("Checkpoint copy verification failed")
                    executable = baseline if config is None else candidate
                    argv = segment_command(repo, out, executable, scenario, cpus)
                    env = clean_environment(config)
                    log = trial_root / "run.log"
                    trial = dict(name=label, scenario=name, variant=variant, repetition=repetition,
                                 status="running", command=argv, log=str(log), outdir=str(out),
                                 environment={key: value for key, value in env.items()
                                              if key.startswith(("OMP_", "IPC_GPU_", "CUDA_", "NVIDIA_"))},
                                 executable_sha256=state["baseline" if config is None else "candidate"]["sha256"],
                                 checkpoint=checkpoint_inputs.get(name), background_observations=[])
                    (trial_root / "command.json").write_text(json.dumps(trial, indent=2) + "\n")
                    state["runs"].append(trial)
                    print("Starting", label, flush=True)
                    before = sample()
                    started = time.monotonic()
                    with log.open("w") as output:
                        process = subprocess.Popen(argv, cwd=repo, env=env, stdout=output,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                        save("running", current=label, child_pid=process.pid)
                        while True:
                            try:
                                process.wait(timeout=args.monitor_interval)
                                done = True
                            except subprocess.TimeoutExpired:
                                done = False
                            elapsed = time.monotonic() - started
                            after = sample(process.pid)
                            reasons, load = activity(before, after, [], process.pid, max_background_cores=args.max_background_cores)
                            trial["background_observations"].append(dict(elapsed_s=elapsed,
                                external_cpu_cores=load, reasons=reasons,
                                system_iowait_cpu_s=max(0., (after["iowait"] - before["iowait"]) / HZ)))
                            before = after
                            if reasons:
                                raise RuntimeError("External workload detected: " + "; ".join(reasons))
                            if done:
                                break
                            if elapsed > args.timeout:
                                raise RuntimeError("Per-trial timeout; partial output is not accepted")
                            save("running", current=label, elapsed_s=elapsed)
                    trial["observed_process_wall_s"] = time.monotonic() - started
                    if process.returncode:
                        raise RuntimeError(f"Application returned {process.returncode}")
                    verify_inputs()
                    parsed = parse_segment(log, out, scenario)
                    trial.update(parsed)
                    if scenario["restart_frame"] is not None:
                        input_copy = out / Path(checkpoint_inputs[name]["path"]).name
                        if sha256(input_copy) != checkpoint_inputs[name]["sha256"]:
                            raise RuntimeError("Restart input was unexpectedly modified")
                    if config is None and name not in references:
                        references[name] = parsed["output_hashes"]
                        reference_names[name] = label
                    reference = references[name]
                    if parsed["output_hashes"] != reference:
                        trial["mismatches"] = [key for key in set(reference) | set(parsed["output_hashes"])
                                               if reference.get(key) != parsed["output_hashes"].get(key)]
                        raise RuntimeError("GPU binary/GEO byte mismatch: " + ", ".join(trial["mismatches"]))
                    trial.update(status="accepted", byte_identical_to_baseline=True, baseline_reference=reference_names[name])
                    (trial_root / "result.json").write_text(json.dumps(trial, indent=2) + "\n")
                    save("between_trials", current=label, child_pid=None)
                    print("Accepted", label, f"solver={trial['total_solver_s']:.6f}s; binary/GEO identical", flush=True)
                    process = None
        save("complete", current=None, child_pid=None)
        print("Completed clean, byte-identical tuning trials:", root, flush=True)
    except BaseException as error:
        if trial is not None and trial["status"] == "running":
            trial.update(status="rejected", error=f"{type(error).__name__}: {error}")
            trial_root = Path(trial["log"]).parent
            (trial_root / "result.json").write_text(json.dumps(trial, indent=2) + "\n")
        save("stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process is not None:
            stop_our_process(process)
        lock.close()


def self_test():
    import tempfile
    from unittest.mock import patch
    assert len(DEFAULT_CONFIGS) == 8 and len({parse_config(c)["name"] for c in DEFAULT_CONFIGS}) == 8
    for bad in ("4097:8", "1024:64", "3:2", "0:1", "1", "eight:1"):
        try:
            parse_config(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Bad launch config accepted: " + bad)
    with patch.dict(os.environ, {"IPC_GPU_PROFILE_CSV": "unsafe.csv", "IPC_GPU_CONTACT_QUEUE_CSV": "queue.csv",
                                  "IPC_GPU_CONTACT_EXACT_LARGE_WORKERS": "999", "OMP_NUM_THREADS": "3"}):
        env = clean_environment(None)
        assert not any(key.startswith("IPC_GPU_") for key in env)
        assert env["OMP_NUM_THREADS"] == "64" and env["OMP_WAIT_POLICY"] == "ACTIVE"
        env = clean_environment(parse_config("512:4"))
        assert {key: value for key, value in env.items() if key.startswith("IPC_GPU_")} == {
            SMALL_WORKERS: "512", SMALL_THREADS: "4"}
    argv = segment_command(Path("/repo"), Path("/out"), Path("/baseline"), SCENARIOS["late"], list(range(64)))
    assert argv[-2:] == ["--restart_frame", "219"] and argv[argv.index("--num_frames") + 1] == "220"
    with tempfile.TemporaryDirectory(prefix="gpu_contact_tuning_test_") as directory:
        root = Path(directory)
        out = root / "frames"
        out.mkdir()
        (out / "state_0219.bin").write_bytes(b"input")
        (out / "state_0220.bin").write_bytes(b"output")
        (out / "frame_0220.geo").write_text("validated by patched parser")
        log = root / "run.log"
        log.write_text("Total Vertices: 10000\nFrame 220 | global_iters = 20 | solver_time = 10.000 ms\n"
                       "Simulation finished.\nTotal simulation time: 20.000 ms\nTotal solver time: 10.000 ms\nAverage solver time: 10.000 ms/frame\n")
        with patch(__name__ + "._geo_positions", return_value=[]):
            parsed = parse_segment(log, out, SCENARIOS["late"])
        assert parsed["simulated_frames"] == 1 and parsed["total_solver_s"] == .01
        assert set(parsed["output_hashes"]) == {"state_0220.bin", "frame_0220.geo"}
        log.write_text(log.read_text().replace("Frame 220", "Frame 219"))
        with patch(__name__ + "._geo_positions", return_value=[]):
            try:
                parse_segment(log, out, SCENARIOS["late"])
            except ValueError:
                pass
            else:
                raise AssertionError("Wrong frame accepted")
    state = dict(scenarios=["late"], configs=[parse_config("512:4")], repetitions=2, status="complete",
                 candidate={}, baseline={}, runs=[])
    for name, times in (("baseline", (2., 4.)), ("w512_t4", (1., 2.))):
        for duration in times:
            state["runs"].append(dict(scenario="late", status="accepted", variant=name,
                                      total_solver_s=duration, average_solver_frame_s=duration))
    report = make_summary(state)
    assert report["scenarios"]["late"]["variants"][1]["speedup_over_baseline"] == 2.
    assert report["scenarios"]["late"]["fastest_observed_variant"] == "w512_t4"
    assert "continuous-profile" in render_summary(report)
    print("Contact tuning harness self-test PASS")


if __name__ == "__main__":
    main()
