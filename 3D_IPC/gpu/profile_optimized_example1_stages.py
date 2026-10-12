#!/usr/bin/env python3
"""Profile the immutable optimized GPU at three stages, checking restart bytes.

Run on the GPU server; no SSH, synchronization, source edits, or builds occur.
The comparison reference is the matching CLEAN COLD RESTART, never a continuous
frame hash: restart histories/caches are intentionally reset in both runs.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
from statistics import mean
import subprocess
import time

from analyze_example1_full_profile import read_phases, TIMINGS
from example1_comparison import verify_readme_command
from tune_example1_contact import SCENARIOS, clean_environment, parse_segment, segment_command, sha256
from wait_example1_comparison import EXPECTED, MONITORING_NOTE, activity, physical_cpus, sample, stop_our_process, HZ

OPTIMIZED_SHA256 = "161ddb14c186c9242be1dcc6de260a5670dedf20999620f63bd23f7f9e4b299d"
SCHEDULE = dict(small_workers=1024, small_threads=1, large_workers=256, large_threads=4)
NOTES = [
    "This is a diagnostic profile, not a clean speed benchmark. Timers and event collection perturb wall time.",
    "CUDA timings are elapsed stream intervals, NOT kernel-active time; they may contain host launch starvation or synchronization gaps.",
    "Host and CUDA stream times overlap and must never be added together.",
    "Stage paths are hierarchical. Inclusive values contain children; sum only exclusive values across all stages.",
    "Profiler bookkeeping is reported, not subtracted to claim a clean timing result. Fresh processes include event-pool warmup costs.",
    "Contact/color counters describe candidate workload, not necessarily geometrically active contacts.",
    "Middle/late cases are cold restarts and are compared byte-for-byte with matching candidate cold-restart trials, not continuous simulation hashes.",
    "Clean reference timing was measured earlier. Profile/reference ratios illustrate observed inflation, not isolated timer overhead or optimization speedup.",
    "Three stage probes do not replace full-simulation profiling or establish a 240-frame speedup.",
    MONITORING_NOTE,
]


def profile_environment(path: Path) -> dict:
    result = clean_environment(None)
    result.update({"IPC_GPU_CONTACT_EXACT_" + key.upper(): str(value) for key, value in SCHEDULE.items()})
    result["IPC_GPU_PROFILE_CSV"] = str(path)
    return result


def load_references(tuning: dict, scenarios: list[str]) -> dict:
    if tuning.get("status") != "complete" or tuning.get("candidate", {}).get("sha256") != OPTIMIZED_SHA256:
        raise ValueError("Reference tuning is not a completed run of the required optimized executable")
    result = {}
    for name in scenarios:
        runs = [run for run in tuning.get("runs", []) if run.get("scenario") == name and run.get("variant") == "w1024_t1"]
        if not runs:
            raise ValueError("No matching clean candidate reference for " + name)
        scenario = SCENARIOS[name]
        output_frames = list(range(scenario["first_frame"], scenario["last_frame"] + 1))
        if scenario["restart_frame"] is None:
            output_frames.insert(0, 0)
        expected = {f"{prefix}{frame:04d}{suffix}" for frame in output_frames
                    for prefix, suffix in (("frame_", ".geo"), ("state_", ".bin"))}
        reference = runs[0].get("output_hashes")
        if not isinstance(reference, dict) or set(reference) != expected:
            raise ValueError("Clean reference does not contain exactly the scenario's output hashes")
        for entry in reference.values():
            if (type(entry.get("bytes")) is not int or entry["bytes"] <= 0
                    or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in entry["sha256"])):
                raise ValueError("Malformed clean reference hash entry")
        checkpoint = tuning.get("checkpoint_inputs", {}).get(name)
        for run in runs:
            if (run.get("status") != "accepted" or not run.get("byte_identical_to_baseline")
                    or run.get("executable_sha256") != OPTIMIZED_SHA256 or run.get("output_hashes") != reference
                    or run.get("checkpoint") != checkpoint):
                raise ValueError("Clean reference runs disagree or lack successful byte validation")
            env = run.get("environment", {})
            if any(key.startswith("IPC_GPU_PROFILE") or key == "IPC_GPU_CONTACT_QUEUE_CSV" for key in env):
                raise ValueError("Reference run enabled diagnostic instrumentation")
            for key, value in SCHEDULE.items():
                actual = env.get("IPC_GPU_CONTACT_EXACT_" + key.upper(), str(value) if key.startswith("large_") else None)
                if actual != str(value):
                    raise ValueError("Reference run has a different exact-contact schedule")
            if not math.isfinite(run.get("total_solver_s", float("nan"))) or run["total_solver_s"] <= 0:
                raise ValueError("Invalid clean reference timing")
        if scenario["restart_frame"] is not None and not checkpoint:
            raise ValueError("Missing matching restart checkpoint metadata")
        result[name] = dict(output_hashes=reference, checkpoint=checkpoint,
                            clean_reference_trials=[run["name"] for run in runs],
                            clean_reference_solver_samples_s=[run["total_solver_s"] for run in runs],
                            clean_reference_mean_solver_s=mean(run["total_solver_s"] for run in runs))
    return result


def phase_summary(path: Path, parsed: dict, scenario: dict) -> dict:
    frames = read_phases(path)
    ids = list(range(scenario["first_frame"], scenario["last_frame"] + 1))
    if set(frames) != set(ids):
        raise ValueError("Profiling CSV does not contain exactly the requested frame IDs")
    phases = {}
    frame_rows = []
    negative_residuals = []
    for frame in ids:
        data = frames[frame]
        if "frame" not in data["stages"] or data["metadata"]["substeps"] != 5 or data["metadata"].get("iterations") != 20:
            raise ValueError("Missing frame root, unexpected substeps, or iteration count")
        for name, values in data["stages"].items():
            total = phases.setdefault(name, dict.fromkeys(("calls", *TIMINGS), 0.))
            for key, value in values.items():
                total[key] += value
            if values["cuda_exclusive_ms"] < 0:
                negative_residuals.append(dict(frame=frame, stage=name, cuda_exclusive_ms=values["cuda_exclusive_ms"]))
        frame_rows.append(dict(frame=frame, solver_wall_ms=parsed["frame_solver_s"][str(frame)] * 1000,
                               **data["metadata"], stages=data["stages"]))
    host_total = sum(row["host_exclusive_ms"] for row in phases.values())
    cuda_total = sum(row["cuda_exclusive_ms"] for row in phases.values())
    stage_rows = [dict(stage=name, **values,
                       host_share=values["host_exclusive_ms"] / host_total if host_total else 0,
                       cuda_stream_share=values["cuda_exclusive_ms"] / cuda_total if cuda_total else 0)
                  for name, values in phases.items()]
    counters = {key: dict(sum=sum(frames[frame]["metadata"][key] for frame in ids),
                          mean_per_frame=mean(frames[frame]["metadata"][key] for frame in ids),
                          max_per_frame=max(frames[frame]["metadata"][key] for frame in ids))
                for key in sorted(frames[ids[0]]["metadata"]) if not key.endswith("_ms")}
    return dict(frame_ids=ids, profiled_solver_s=parsed["total_solver_s"],
                profiled_mean_solver_frame_ms=parsed["average_solver_frame_s"] * 1000,
                profile_scope_wall_s=sum(frames[frame]["metadata"]["frame_wall_ms"] for frame in ids) / 1000,
                profile_cpu_overhead_s=sum(frames[frame]["metadata"]["profile_cpu_overhead_ms"] for frame in ids) / 1000,
                host_exclusive_sum_s=host_total / 1000, cuda_exclusive_stream_sum_s=cuda_total / 1000,
                stages=sorted(stage_rows, key=lambda row: -row["cuda_exclusive_ms"]), frames=frame_rows,
                counters=counters, negative_cuda_exclusive_residuals=negative_residuals)


def write_report(root: Path, state: dict) -> None:
    report = dict(kind="optimized GPU stage diagnostics", status=state["status"], executable_sha256=OPTIMIZED_SHA256,
                  schedule=SCHEDULE, requested_scenarios=state["scenarios"], runs=state["runs"], notes=NOTES,
                  error=state.get("error"))
    lines = ["Optimized GPU stage diagnostics", "Status: " + state["status"], ""]
    for run in state["runs"]:
        if run.get("status") != "complete":
            continue
        profile = run["profile"]
        lines += [f"{run['scenario']}: frames {profile['frame_ids']}; binary and GEO match clean cold-restart reference",
                  f"  Profiled solver: {profile['profiled_solver_s']:.6f} s; mean frame {profile['profiled_mean_solver_frame_ms']:.3f} ms",
                  f"  Previous clean matching solver mean: {run['reference']['clean_reference_mean_solver_s']:.6f} s (NOT a paired overhead benchmark)",
                  f"  Profile scope wall: {profile['profile_scope_wall_s']:.6f} s; recorded bookkeeping: {profile['profile_cpu_overhead_s']:.6f} s",
                  f"  Host exclusive: {profile['host_exclusive_sum_s']:.6f} s; CUDA stream exclusive: {profile['cuda_exclusive_stream_sum_s']:.6f} s (do not add)",
                  "  Top exclusive CUDA-stream intervals:"]
        for row in profile["stages"][:15]:
            lines.append(f"    {row['cuda_exclusive_ms']/1000:.6f} s ({row['cuda_stream_share']*100:.2f}%) | host {row['host_exclusive_ms']/1000:.6f} s | {row['stage']}")
        lines.append("  Top exclusive host intervals:")
        for row in sorted(profile["stages"], key=lambda value: -value["host_exclusive_ms"])[:10]:
            lines.append(f"    {row['host_exclusive_ms']/1000:.6f} s ({row['host_share']*100:.2f}%) | {row['stage']}")
        lines.append("  Counters (sum / mean per frame / maximum per frame):")
        for key, value in profile["counters"].items():
            lines.append(f"    {key}: {value['sum']:g} / {value['mean_per_frame']:g} / {value['max_per_frame']:g}")
        lines.append("")
    lines += ["Notes:", *("- " + note for note in NOTES)]
    if state.get("error"):
        lines += ["", "Stopped: " + state["error"]]
    for name, content in (("stage_profile_report.json", json.dumps(report, indent=2) + "\n"),
                          ("stage_profile_report.txt", "\n".join(lines) + "\n")):
        temporary = root / (name + ".tmp");temporary.write_text(content);temporary.replace(root / name)
    rows = [dict(scenario=run["scenario"], **row) for run in state["runs"] if run.get("status") == "complete" for row in run["profile"]["stages"]]
    if rows:
        temporary = root / "stage_phase_summary.csv.tmp"
        with temporary.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        temporary.replace(root / "stage_phase_summary.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--optimization-directory", type=Path)
    parser.add_argument("--executable", type=Path, help="Optional copy of the same required immutable executable")
    parser.add_argument("--scenario", choices=("early", "middle", "late"), action="append")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--monitor-interval", type=float, default=5)
    parser.add_argument("--max-background-cores", type=float, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test();return
    if args.output is None or args.optimization_directory is None:
        parser.error("--output and --optimization-directory are required")
    if (not 0 < args.timeout < float("inf") or not 0 < args.monitor_interval <= 60
            or not 0 <= args.max_background_cores < float("inf")):
        parser.error("Invalid timeout, monitor interval or background ceiling")
    scenarios = args.scenario or ["early", "middle", "late"]
    if len(scenarios) != len(set(scenarios)):
        parser.error("Duplicate scenarios")
    repo = Path(__file__).resolve().parent.parent
    verify_readme_command(repo)
    optimization = args.optimization_directory.resolve(strict=True)
    tuning_path = optimization / "tuning/status.json"
    references = load_references(json.loads(tuning_path.read_text()), scenarios)
    executable = (args.executable or optimization / "3D_sim_gpu_candidate").resolve(strict=True)
    pinned = {str(executable): OPTIMIZED_SHA256, str(tuning_path): sha256(tuning_path),
              str(repo / "README.md"): sha256(repo / "README.md"),
              str(repo / EXPECTED["cpu"][0]): EXPECTED["cpu"][1]}
    for reference in references.values():
        if reference["checkpoint"]:
            item = reference["checkpoint"]
            pinned[item["path"]] = item["sha256"]
    root = args.output.resolve();root.mkdir(exist_ok=True)
    if any(path.name != "supervisor.log" for path in root.iterdir()):
        raise RuntimeError("Use a fresh stage-profile directory")
    cpus = physical_cpus()
    if cpus != list(range(64)):
        raise RuntimeError("Expected verified physical CPU affinity 0..63")
    state = dict(kind="optimized GPU stage diagnostics", status="preparing", pid=os.getpid(),
                 directory=str(root), optimization_directory=str(optimization), executable=str(executable),
                 executable_sha256=OPTIMIZED_SHA256, schedule=SCHEDULE, scenarios=scenarios, cpu_affinity=cpus,
                 pinned_inputs=pinned, references=references, runs=[], monitoring_policy=MONITORING_NOTE)

    def save(status, **fields):
        state.update(status=status, updated_unix=time.time(), **fields)
        temporary = root / "status.tmp";temporary.write_text(json.dumps(state, indent=2) + "\n");temporary.replace(root / "status.json")

    def verify():
        verify_readme_command(repo)
        for path, digest in pinned.items():
            if sha256(Path(path)) != digest:
                raise RuntimeError("Pinned profile input changed: " + path)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    lock = (repo / "gpu/.example1_comparison.lock").open("a+")
    process = None
    run = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify()
        (root / "hardware.json").write_text(json.dumps(dict(
            cpu=subprocess.check_output(["lscpu"], text=True),
            gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()), indent=2) + "\n")
        for name in scenarios:
            clean = 0
            while clean < 3:
                before = sample();time.sleep(args.monitor_interval);after = sample()
                reasons, load = activity(before, after, [], max_background_cores=args.max_background_cores)
                clean = 0 if reasons else clean + 1
                save("waiting_for_idle", current=name, reasons=reasons, idle_checks=clean, external_cpu_cores=load)
            verify()
            scenario, reference = SCENARIOS[name], references[name]
            directory = root / name;directory.mkdir()
            out, csv_path, log = directory / "frames", directory / "phase_frames.csv", directory / "run.log"
            if reference["checkpoint"]:
                out.mkdir()
                source = Path(reference["checkpoint"]["path"])
                shutil.copy2(source, out / source.name)
                if sha256(out / source.name) != reference["checkpoint"]["sha256"]:
                    raise RuntimeError("Restart checkpoint copy hash mismatch")
            argv = segment_command(repo, out, executable, scenario, cpus)
            env = profile_environment(csv_path)
            run = dict(scenario=name, status="profiling", command=argv, log=str(log), phase_csv=str(csv_path),
                       environment={key: value for key, value in env.items() if key.startswith(("OMP_", "IPC_GPU_"))},
                       reference=reference, background_observations=[])
            state["runs"].append(run)
            (directory / "command.json").write_text(json.dumps(run, indent=2) + "\n")
            print("Profiling", name, "frames", scenario["first_frame"], "through", scenario["last_frame"], flush=True)
            before = sample();started = time.monotonic()
            with log.open("w") as stream:
                process = subprocess.Popen(argv, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                save("profiling", current=name, child_pid=process.pid)
                while True:
                    try:
                        process.wait(timeout=args.monitor_interval);done = True
                    except subprocess.TimeoutExpired:
                        done = False
                    after = sample(process.pid)
                    reasons, load = activity(before, after, [], process.pid, max_background_cores=args.max_background_cores)
                    elapsed = time.monotonic() - started
                    run["background_observations"].append(dict(elapsed_s=elapsed, external_cpu_cores=load, reasons=reasons,
                        system_iowait_cpu_s=max(0., (after["iowait"] - before["iowait"]) / HZ)))
                    before = after
                    if reasons:
                        raise RuntimeError("External workload during profile: " + "; ".join(reasons))
                    save("profiling", current=name, elapsed_s=elapsed)
                    if done:
                        break
                    if elapsed > args.timeout:
                        raise RuntimeError("Stage profile timeout")
            if process.returncode:
                raise RuntimeError(f"GPU application returned {process.returncode}")
            verify()
            parsed = parse_segment(log, out, scenario)
            if parsed["output_hashes"] != reference["output_hashes"]:
                run["mismatches"] = [key for key in set(parsed["output_hashes"]) | set(reference["output_hashes"])
                                     if parsed["output_hashes"].get(key) != reference["output_hashes"].get(key)]
                raise RuntimeError("Profiled GEO/binary state differs from matching clean restart: " + str(run["mismatches"]))
            if reference["checkpoint"]:
                item = reference["checkpoint"]
                if sha256(out / Path(item["path"]).name) != item["sha256"]:
                    raise RuntimeError("Restart input checkpoint changed")
            run.update(parsed=parsed, profile=phase_summary(csv_path, parsed, scenario), status="complete",
                       byte_identical_to_clean_restart=True)
            run["profile"]["profiled_over_previous_clean_ratio"] = parsed["total_solver_s"] / reference["clean_reference_mean_solver_s"]
            (directory / "result.json").write_text(json.dumps(run, indent=2) + "\n")
            process = None
            save("between_stages", current=name, child_pid=None);write_report(root, state)
            print("Validated stage:", name, "binary/GEO identical", flush=True)
        save("complete", current=None, child_pid=None);write_report(root, state)
    except BaseException as error:
        if process is not None:
            stop_our_process(process)
        if run is not None and run.get("status") != "complete":
            run.update(status="stopped", error=f"{type(error).__name__}: {error}")
        save("stopped", child_pid=None, error=f"{type(error).__name__}: {error}");write_report(root, state)
        raise
    finally:
        if process is not None:
            stop_our_process(process)
        lock.close()


def self_test():
    import tempfile
    from unittest.mock import patch
    with patch.dict(os.environ, {"IPC_GPU_CONTACT_QUEUE_CSV": "bad.csv", "IPC_GPU_CONTACT_EXACT_SMALL_THREADS": "32", "OMP_NUM_THREADS": "2"}):
        env = profile_environment(Path("/profile.csv"))
        assert env["OMP_NUM_THREADS"] == "64" and env["OMP_WAIT_POLICY"] == "ACTIVE"
        assert env["IPC_GPU_CONTACT_EXACT_SMALL_THREADS"] == "1" and "IPC_GPU_CONTACT_QUEUE_CSV" not in env
    hashes = {name: dict(bytes=10, sha256="a" * 64) for name in ("frame_0220.geo", "state_0220.bin")}
    checkpoint = dict(path="/state_0219.bin", bytes=10, sha256="b" * 64)
    clean = dict(name="clean_late", scenario="late", variant="w1024_t1", status="accepted",
                 byte_identical_to_baseline=True, executable_sha256=OPTIMIZED_SHA256, output_hashes=hashes,
                 checkpoint=checkpoint, environment={"IPC_GPU_CONTACT_EXACT_SMALL_WORKERS": "1024", "IPC_GPU_CONTACT_EXACT_SMALL_THREADS": "1"}, total_solver_s=1.)
    tuning = dict(status="complete", candidate=dict(sha256=OPTIMIZED_SHA256), checkpoint_inputs=dict(late=checkpoint), runs=[clean])
    assert load_references(tuning, ["late"])["late"]["output_hashes"] == hashes
    with tempfile.TemporaryDirectory(prefix="optimized_stage_profile_test_") as name:
        root = Path(name);path = root / "phases.csv"
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["frame", "stage", "calls", *TIMINGS, "frame_wall_ms", "profile_cpu_overhead_ms", "substeps", "iterations", "solver_candidate_max"])
            writer.writerow([220, "frame", 1, 11, 1, 8, 0, 11, .1, 5, 20, 40])
            writer.writerow([220, "substep", 5, 10, 2, 8, 1, 11, .1, 5, 20, 40])
            writer.writerow([220, "substep/contact", 20, 8, 8, 7, 7, 11, .1, 5, 20, 40])
        parsed = dict(frame_solver_s={"220": .012}, total_solver_s=.012, average_solver_frame_s=.012)
        summary = phase_summary(path, parsed, SCENARIOS["late"])
        assert summary["host_exclusive_sum_s"] == .011 and summary["cuda_exclusive_stream_sum_s"] == .008
        assert summary["stages"][0]["stage"] == "substep/contact"
        assert summary["counters"]["solver_candidate_max"]["sum"] == 40
        try:
            phase_summary(path, parsed, SCENARIOS["middle"])
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong frame IDs accepted")
    clean["environment"]["IPC_GPU_PROFILE_CSV"] = "/bad.csv"
    try:
        load_references(tuning, ["late"])
    except ValueError:
        pass
    else:
        raise AssertionError("Instrumented reference accepted as clean")
    print("Optimized stage profiling self-test PASS")


if __name__ == "__main__":
    main()
