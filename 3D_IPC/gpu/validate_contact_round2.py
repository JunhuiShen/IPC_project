#!/usr/bin/env python3
"""Clean, sequential byte-checked validation of three GPU contact optimizations.

Run locally on the GPU server. No SSH, builds, Git writes, or CPU modifications.
Every short candidate is compared with a FRESH matching old-binary cold restart,
also cross-checked against the previously accepted clean restart. Continuous
reference hashes are used only by the optional uninterrupted 240-frame run.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
from statistics import mean
import subprocess
import tarfile
import time

from example1_comparison import parse_trial, verify_readme_command
from finish_example1_optimization import cpu_source_snapshot, gpu_source_snapshot
from profile_example1_full import checkpoint_manifest
from profile_optimized_example1_stages import OPTIMIZED_SHA256, SCHEDULE, load_references
from tune_example1_contact import SCENARIOS, clean_environment, parse_segment, segment_command, sha256
from validate_optimized_example1_full import (
    build_report, checked_cpu_baseline, load_clean_comparison, read_reference,
    safe_completed_frame, valid_sha, verify_ready_frames, write_report,
)
from wait_example1_comparison import (
    HZ, MONITORING_NOTE, activity, physical_cpus, sample, stop_our_process,
)

FLAGS = ("IPC_GPU_COLOR_PAIR_QUEUES", "IPC_GPU_WARP_CONTACT_REDUCE",
         "IPC_GPU_COOPERATIVE_EXACT_CONTACT")
VARIANTS = {"baseline": (0, 0, 0), "all_off": (0, 0, 0),
            "pairs_only": (1, 0, 0), "pairs_warp": (1, 1, 0), "all_three": (1, 1, 1)}
NOTES = [
    "All timings are clean solver timings: stage timers, queue readbacks, and inherited IPC_GPU_* knobs are disabled.",
    "Each stage/repetition starts with the pinned previous GPU binary, then the all-off new binary, before enabled variants.",
    "Every baseline and candidate GEO/binary output must match the fresh same-checkpoint restart and the earlier accepted clean restart hashes.",
    "Restart cache histories differ from continuous histories: short runs NEVER compare against continuous frame output hashes.",
    "Enabled-variant order reverses on alternate repetitions; two samples are a small timing sample, not a full-simulation speedup claim.",
    "The GPU source archive records candidate-time source provenance, not a claim that it built the old immutable baseline.",
    "CPU source and executable hashes are captured including existing dirty changes, then checked unchanged; CPU files are never edited or built.",
    "Only an explicitly requested full run uses continuous reference hashes; it requires every short variant and both repetitions to pass first.",
    MONITORING_NOTE,
]


def variant_environment(variant: str) -> dict:
    result = clean_environment(None)
    result.update({"IPC_GPU_CONTACT_EXACT_" + key.upper(): str(value) for key, value in SCHEDULE.items()})
    result.update(dict(zip(FLAGS, map(str, VARIANTS[variant]))))
    return result


def trial_plan(scenarios: list[str], repetitions: int) -> list[tuple[str, int, str]]:
    result = []
    for scenario in scenarios:
        for repetition in range(1, repetitions + 1):
            enabled = ["pairs_only", "pairs_warp", "all_three"]
            if repetition % 2 == 0:
                enabled.reverse()
            result.extend((scenario, repetition, variant) for variant in ["baseline", "all_off", *enabled])
    return result


def require_same_hashes(actual: dict, expected: dict, label: str) -> None:
    if actual != expected:
        mismatches = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
        raise ValueError(label + " GEO/binary mismatch: " + ", ".join(mismatches))


def completed_short_trials(state: dict) -> bool:
    expected = trial_plan(state["scenarios"], state["repetitions"])
    actual = [(run["scenario"], run["repetition"], run["variant"]) for run in state["runs"]]
    return (actual == expected and all(run.get("status") == "accepted"
            and run.get("byte_identical_to_fresh_baseline") is True for run in state["runs"]))


def summarize(state: dict) -> dict:
    rows = []
    for scenario in state["scenarios"]:
        runs = [run for run in state["runs"] if run["scenario"] == scenario and run.get("status") == "accepted"]
        old = [run["total_solver_s"] for run in runs if run["variant"] == "baseline"]
        for variant in VARIANTS:
            selected = [run for run in runs if run["variant"] == variant]
            if not selected:
                continue
            values = [run["total_solver_s"] for run in selected]
            center = mean(values)
            paired = [run["fresh_baseline_solver_s"] / run["total_solver_s"] for run in selected]
            rows.append(dict(scenario=scenario, variant=variant, repetitions=len(values), samples_s=values,
                mean_solver_s=center, min_solver_s=min(values), max_solver_s=max(values),
                mean_solver_frame_ms=mean(run["average_solver_frame_s"] for run in selected) * 1000,
                speedup_over_baseline_mean=mean(old) / center, paired_speedups=paired,
                binary_and_geo_byte_identical=True,
                complete=len(values) == state["repetitions"]))
    return dict(kind="clean GPU contact optimization round 2", status=state["status"],
                executable_hashes=state["executable_hashes"], features=list(FLAGS),
                variants=VARIANTS, rows=rows, short_trials_complete=completed_short_trials(state),
                full=state.get("full"), error=state.get("error"), notes=NOTES)


def save_reports(root: Path, state: dict) -> None:
    summary = summarize(state)
    lines = ["GPU contact optimization round 2", "Status: " + state["status"], "",
             "stage | variant | n | mean solver(s) | min..max(s) | solver/frame(ms) | old/new"]
    for row in summary["rows"]:
        lines.append(f"{row['scenario']} | {row['variant']} | {row['repetitions']} | {row['mean_solver_s']:.6f} | "
                     f"{row['min_solver_s']:.6f}..{row['max_solver_s']:.6f} | {row['mean_solver_frame_ms']:.3f} | "
                     f"{row['speedup_over_baseline_mean']:.4f}x")
    if summary.get("error"):
        lines += ["", "Stopped: " + summary["error"]]
    lines += ["", "Notes:", *("- " + note for note in NOTES)]
    for name, content in (("status.json", json.dumps(state, indent=2) + "\n"),
                          ("summary.json", json.dumps(summary, indent=2) + "\n"),
                          ("summary.txt", "\n".join(lines) + "\n")):
        temporary = root / (name + ".tmp")
        temporary.write_text(content);temporary.replace(root / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--optimization-directory", type=Path, help="Previous validated_optimization_v2 directory")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--baseline", type=Path, help="Optional copy of the pinned previous optimized executable")
    parser.add_argument("--baseline-source-archive", type=Path, help="Optional pre-existing old-source archive; copied and hashed, never modified")
    parser.add_argument("--scenario", choices=("early", "middle", "late"), action="append")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--monitor-interval", type=float, default=5)
    parser.add_argument("--idle-checks", type=int, default=3)
    parser.add_argument("--max-background-cores", type=float, default=2)
    parser.add_argument("--full", action="store_true", help="After all 30 short trials pass, validate a clean uninterrupted 240-frame all-three run")
    parser.add_argument("--reference-manifest", type=Path, help="Original complete continuous GPU manifest; needed with --full")
    parser.add_argument("--cpu-baseline", type=Path, help="Preserved 240-frame CPU timings; needed with --full")
    parser.add_argument("--full-timeout", type=float, default=43200)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test();return
    if any(getattr(args, field) is None for field in ("output", "optimization_directory", "candidate", "candidate_sha256")):
        parser.error("--output, --optimization-directory, --candidate and --candidate-sha256 are required")
    if (not valid_sha(args.candidate_sha256) or args.repetitions < 1 or args.idle_checks < 1
            or not 0 < args.monitor_interval <= 60 or not 0 <= args.max_background_cores <= 2
            or any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.full_timeout))):
        parser.error("Invalid hash/count/timeouts/interval; background ceiling must not exceed 2 CPU cores")
    scenarios = args.scenario or ["early", "middle", "late"]
    if len(set(scenarios)) != len(scenarios):
        parser.error("Duplicate scenarios")
    if args.full and (set(scenarios) != set(SCENARIOS) or args.repetitions != 2
                      or args.reference_manifest is None or args.cpu_baseline is None):
        parser.error("--full requires all three stages, exactly two repetitions, --reference-manifest and --cpu-baseline")
    repo = Path(__file__).resolve().parent.parent
    verify_readme_command(repo)
    optimization = args.optimization_directory.resolve(strict=True)
    tuning_path = optimization / "tuning/status.json"
    references = load_references(json.loads(tuning_path.read_text()), scenarios)
    original = (args.baseline or optimization / "3D_sim_gpu_candidate").resolve(strict=True)
    candidate_source = args.candidate.resolve(strict=True)
    if sha256(original) != OPTIMIZED_SHA256 or sha256(candidate_source) != args.candidate_sha256:
        raise ValueError("Baseline/candidate does not have the required pinned SHA256")
    if original == candidate_source or args.candidate_sha256 == OPTIMIZED_SHA256:
        raise ValueError("New candidate and previous baseline must be distinct binaries")
    root = args.output.resolve();root.mkdir(exist_ok=True)
    if any(path.name != "supervisor.log" for path in root.iterdir()):
        raise RuntimeError("Use a fresh directory; previous output is never replaced")
    cpus = physical_cpus()
    if cpus != list(range(64)):
        raise RuntimeError("Expected verified physical CPU affinity 0..63")
    cpu_sources = cpu_source_snapshot(repo)
    gpu_sources = gpu_source_snapshot(repo)
    pinned = {str(tuning_path): sha256(tuning_path), str(repo / "README.md"): sha256(repo / "README.md"),
              str(original): OPTIMIZED_SHA256, str(candidate_source): args.candidate_sha256,
              str(repo / "build/3D_sim"): sha256(repo / "build/3D_sim")}
    for reference in references.values():
        if reference["checkpoint"]:
            item = reference["checkpoint"]
            pinned[item["path"]] = item["sha256"]
            if "source_manifest" in item:
                pinned[item["source_manifest"]] = item["source_manifest_sha256"]
    continuous = cpu_baseline = previous_clean = None
    reference_path = cpu_path = None
    if args.full:
        reference_path = args.reference_manifest.resolve(strict=True)
        cpu_path = args.cpu_baseline.resolve(strict=True)
        continuous = read_reference(reference_path, 240)
        cpu_baseline = checked_cpu_baseline(cpu_path)
        clean_path = optimization / "full_validation/clean_report.json"
        previous_clean = load_clean_comparison(clean_path, 240, pinned[str(repo / "README.md")], sha256(reference_path))
        if previous_clean["command_metadata"]["executable_sha256"] != OPTIMIZED_SHA256:
            raise ValueError("Previous full clean report is not from the required baseline executable")
        pinned.update({str(path): sha256(path) for path in (reference_path, cpu_path, clean_path)})
    executables = dict(baseline=root / "3D_sim_gpu_baseline", candidate=root / "3D_sim_gpu_candidate")
    state = dict(kind="GPU contact optimization round 2", status="preparing", pid=os.getpid(), directory=str(root),
                 scenarios=scenarios, repetitions=args.repetitions, features=list(FLAGS), variants=VARIANTS,
                 executable_hashes=dict(baseline=OPTIMIZED_SHA256, candidate=args.candidate_sha256),
                 pinned_inputs=pinned, references=references, cpu_source_hashes=cpu_sources,
                 gpu_source_hashes=gpu_sources, cpu_affinity=cpus, runs=[], full_requested=args.full,
                 monitoring_policy=MONITORING_NOTE, max_background_cores=args.max_background_cores)

    def save(status, **fields):
        state.update(status=status, updated_unix=time.time(), **fields);save_reports(root, state)

    def verify():
        verify_readme_command(repo)
        for path, digest in pinned.items():
            if sha256(Path(path)) != digest:
                raise RuntimeError("Pinned input changed: " + path)
        if cpu_source_snapshot(repo) != cpu_sources:
            raise RuntimeError("CPU sources changed during validation; refusing mixed provenance")
        if gpu_source_snapshot(repo) != gpu_sources:
            raise RuntimeError("GPU source snapshot changed during validation")

    def wait_idle(label):
        clean = 0
        while clean < args.idle_checks:
            before = sample();time.sleep(args.monitor_interval);after = sample()
            reasons, load = activity(before, after, [], max_background_cores=args.max_background_cores)
            clean = 0 if reasons else clean + 1
            save("waiting_for_idle", current=label, reasons=reasons, idle_checks=clean, external_cpu_cores=load)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    lock = (repo / "gpu/.example1_comparison.lock").open("a+")
    process = None
    trial = None

    def execute(run, executable, scenario, env, directory, timeout, online=None):
        nonlocal process
        wait_idle(run["name"]);verify()
        directory.mkdir();out = directory / "frames";log = directory / "run.log"
        checkpoint = run.get("checkpoint")
        if checkpoint:
            out.mkdir();source = Path(checkpoint["path"]);shutil.copy2(source, out / source.name)
            if sha256(out / source.name) != checkpoint["sha256"]:
                raise RuntimeError("Copied restart checkpoint hash mismatch")
        argv = segment_command(repo, out, executable, scenario, cpus)
        run.update(command=argv, outdir=str(out), log=str(log), status="running", instrumentation=False,
                   environment={k: v for k, v in env.items() if k.startswith(("OMP_", "IPC_GPU_", "CUDA_", "NVIDIA_"))},
                   background_observations=[])
        (directory / "command.json").write_text(json.dumps(run, indent=2) + "\n")
        before = sample();started = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.Popen(argv, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            save("running", current=run["name"], child_pid=process.pid)
            while True:
                try:
                    process.wait(timeout=args.monitor_interval);done = True
                except subprocess.TimeoutExpired:
                    done = False
                elapsed = time.monotonic() - started;after = sample(process.pid)
                reasons, load = activity(before, after, [], process.pid, max_background_cores=args.max_background_cores)
                run["background_observations"].append(dict(elapsed_s=elapsed, external_cpu_cores=load, reasons=reasons,
                    system_iowait_cpu_s=max(0., (after["iowait"] - before["iowait"]) / HZ)))
                before = after
                if reasons:
                    raise RuntimeError("External workload during validation: " + "; ".join(reasons))
                if done and process.returncode:
                    raise RuntimeError(f"GPU process returned {process.returncode}")
                if online is not None:
                    through, count = safe_completed_frame(log, 240, process_complete=done)
                    verify_ready_frames(out, continuous, online, through)
                    run.update(completed_frames=count, verified_through_frame=through, verified_files=len(online))
                if elapsed > timeout:
                    raise RuntimeError("Run exceeded time limit; results are not accepted")
                save("running", current=run["name"], elapsed_s=elapsed)
                if done:
                    break
        run["observed_process_wall_s"] = time.monotonic() - started
        verify()
        if checkpoint and sha256(out / Path(checkpoint["path"]).name) != checkpoint["sha256"]:
            raise RuntimeError("Restart input was unexpectedly modified")
        process = None
        return log, out

    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB);verify()
        (root / "runs").mkdir()
        for name, source in (("baseline", original), ("candidate", candidate_source)):
            shutil.copy2(source, executables[name]);pinned[str(executables[name])] = state["executable_hashes"][name]
        with tarfile.open(root / "candidate_gpu_sources.tgz", "w:gz") as archive:
            for name in gpu_sources:
                archive.add(repo / name, arcname=name, recursive=False)
        state["candidate_source_archive_sha256"] = sha256(root / "candidate_gpu_sources.tgz")
        if args.baseline_source_archive:
            source = args.baseline_source_archive.resolve(strict=True)
            pinned[str(source)] = sha256(source)
            dest = root / "previous_gpu_sources.archive";shutil.copy2(source, dest)
            if sha256(dest) != pinned[str(source)]:
                raise RuntimeError("Previous source archive copy mismatch")
            state["previous_source_archive"] = dict(source=str(source), path=str(dest), sha256=pinned[str(source)])
        state["hardware"] = dict(cpu=subprocess.check_output(["lscpu"], text=True),
            gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True),
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
        verify();fresh = {}
        for name, repetition, variant in trial_plan(scenarios, args.repetitions):
            label = f"{len(state['runs']):03d}_{name}_r{repetition}_{variant}"
            reference = references[name];which = "baseline" if variant == "baseline" else "candidate"
            trial = dict(name=label, scenario=name, repetition=repetition, variant=variant, status="preparing",
                executable_sha256=state["executable_hashes"][which], checkpoint=reference["checkpoint"])
            state["runs"].append(trial);print("Starting", label, flush=True)
            directory = root / "runs" / label
            log, out = execute(trial, executables[which], SCENARIOS[name], variant_environment(variant), directory, args.timeout)
            parsed = parse_segment(log, out, SCENARIOS[name]);trial.update(parsed)
            require_same_hashes(parsed["output_hashes"], reference["output_hashes"], "Previously validated restart")
            key = (name, repetition)
            if variant == "baseline":
                fresh[key] = dict(name=label, hashes=parsed["output_hashes"], solver_s=parsed["total_solver_s"])
            require_same_hashes(parsed["output_hashes"], fresh[key]["hashes"], "Fresh baseline restart")
            trial.update(status="accepted", byte_identical_to_fresh_baseline=True,
                         fresh_baseline=fresh[key]["name"], fresh_baseline_solver_s=fresh[key]["solver_s"])
            (directory / "result.json").write_text(json.dumps(trial, indent=2) + "\n")
            save("between_trials", child_pid=None);print("Accepted", label, f"{parsed['total_solver_s']:.6f}s", flush=True)
        if not completed_short_trials(state):
            raise RuntimeError("Short-run validation gate incomplete")
        if args.full:
            trial = dict(name="full_validation", variant="all_three", status="preparing", checked_hashes={})
            state["full"] = trial;directory = root / "full_validation"
            scenario = dict(restart_frame=None, first_frame=1, last_frame=240)
            log, out = execute(trial, executables["candidate"], scenario, variant_environment("all_three"),
                               directory, args.full_timeout, online=trial["checked_hashes"])
            parsed = parse_trial(log, out, 240, "gpu", readme_command=True)
            parsed.update(instrumented=False, process_elapsed_s=trial["observed_process_wall_s"])
            manifest = checkpoint_manifest(out, 240)
            require_same_hashes(manifest["files"], continuous["files"], "Full continuous reference")
            manifest.update(kind="byte-validated clean GPU output manifest", reference_manifest_sha256=pinned[str(reference_path)])
            (directory / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            metadata = dict(command=trial["command"], environment=trial["environment"],
                executable_sha256=args.candidate_sha256, readme_sha256=pinned[str(repo / "README.md")],
                cpu_affinity=cpus, instrumented=False, features=VARIANTS["all_three"], schedule=SCHEDULE)
            report = build_report(directory, parsed, cpu_baseline, metadata, reference_path,
                                  pinned[str(reference_path)], previous_clean)
            write_report(directory, report)
            trial.update(status="accepted", byte_identical_to_continuous_reference=True,
                         total_solver_s=parsed["total_solver_s"], clean_report=str(directory / "clean_report.json"),
                         clean_speedup=report["clean_gpu_comparison"]["clean_solver_speedup"])
            (directory / "status.json").write_text(json.dumps(trial, indent=2) + "\n")
        save("complete", current=None, child_pid=None)
    except BaseException as error:
        if process is not None:
            stop_our_process(process)
        if trial is not None and trial.get("status") != "accepted":
            trial.update(status="rejected", error=f"{type(error).__name__}: {error}")
        save("stopped", child_pid=None, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process is not None:
            stop_our_process(process)
        lock.close()


def self_test() -> None:
    from unittest.mock import patch
    plan = trial_plan(["early", "middle", "late"], 2)
    assert len(plan) == 30 and len(set(plan)) == 30
    assert [v for s, r, v in plan if s == "late" and r == 1] == list(VARIANTS)
    assert [v for s, r, v in plan if s == "late" and r == 2] == ["baseline", "all_off", "all_three", "pairs_warp", "pairs_only"]
    with patch.dict(os.environ, {"IPC_GPU_PROFILE_CSV": "bad", "IPC_GPU_CONTACT_QUEUE_CSV": "bad", FLAGS[2]: "1"}):
        for variant, values in VARIANTS.items():
            env = variant_environment(variant)
            assert tuple(int(env[key]) for key in FLAGS) == values
            assert "IPC_GPU_PROFILE_CSV" not in env and "IPC_GPU_CONTACT_QUEUE_CSV" not in env
            assert env["OMP_NUM_THREADS"] == "64" and env["OMP_WAIT_POLICY"] == "ACTIVE"
            assert env["IPC_GPU_CONTACT_EXACT_SMALL_THREADS"] == "1"
    hashes = {"state_0220.bin": dict(bytes=1, sha256="a" * 64)}
    require_same_hashes(hashes, hashes, "test")
    try:
        require_same_hashes({}, hashes, "test")
    except ValueError:
        pass
    else:
        raise AssertionError("Missing binary accepted")
    state = dict(scenarios=["early", "middle", "late"], repetitions=2, status="complete", executable_hashes={}, runs=[])
    for stage, repetition, variant in plan:
        value = 2. if variant == "baseline" else 1.
        state["runs"].append(dict(scenario=stage, repetition=repetition, variant=variant, status="accepted",
            byte_identical_to_fresh_baseline=True, total_solver_s=value, average_solver_frame_s=value,
            fresh_baseline_solver_s=2.))
    assert completed_short_trials(state)
    report = summarize(state)
    assert len(report["rows"]) == 15 and report["rows"][1]["speedup_over_baseline_mean"] == 2.
    state["runs"][-1]["status"] = "rejected"
    assert not completed_short_trials(state)
    print("Contact optimization round 2 harness self-test PASS")


if __name__ == "__main__":
    main()
