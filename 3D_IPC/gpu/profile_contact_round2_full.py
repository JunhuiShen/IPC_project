#!/usr/bin/env python3
"""Profile all 240 README Example 1 frames of the selected, immutable GPU solver.

The matching clean run remains the timing baseline. This run enables phase
events, never queue readbacks, and must reproduce every GEO and binary state.
No builds, CPU runs, source edits, transfers, or server connections are made.
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
import subprocess
import time

from analyze_example1_full_profile import analyze, log_frames, read_phases, render, summarize_group, TIMINGS
from example1_comparison import command, parse_trial, verify_readme_command
from finish_example1_optimization import cpu_source_snapshot, gpu_source_snapshot
from profile_example1_full import checkpoint_manifest, sha256
from validate_optimized_example1_full import (
    CheckpointMismatch, FEATURE_ENV, SCHEDULE_ENV, checked_cpu_baseline, clean_environment,
    load_clean_comparison, read_reference, safe_completed_frame, verify_ready_frames,
)
from wait_example1_comparison import EXPECTED, MONITORING_NOTE, activity, physical_cpus, sample, stop_our_process

FRAMES = 240
EXECUTABLE_SHA = "f6fa53ee93e9c000c462959f25e8a8f9df7a43bc5b9e59e678da0b0d8ceec2dd"
CLEAN_SOLVER_S = 1066.265610
FEATURES = dict(color_pair_queues=1, warp_contact_reduce=0, cooperative_exact_contact=1)
SCHEDULE = dict(small_workers=1024, small_threads=1, large_workers=256, large_threads=4)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def normalized_argv(argv: list[str]) -> list[str]:
    result = list(argv)
    if not result or result.count("--outdir") != 1:
        raise ValueError("Missing or repeated output directory in solver command")
    result[0] = "EXECUTABLE"
    result[result.index("--outdir") + 1] = "OUTPUT"
    return result


def check_selected_metadata(report: dict, expected_argv: list[str]) -> None:
    meta = report["command_metadata"]
    expected_env = {key: value for key, value in clean_environment(SCHEDULE, FEATURES).items()
                    if key.startswith(("OMP_", "IPC_GPU_"))}
    if (meta.get("executable_sha256") != EXECUTABLE_SHA or meta.get("environment") != expected_env
            or meta.get("feature_effective") != FEATURES
            or dict(meta.get("schedule_defaults", {}), **meta.get("schedule_overrides", {})) != SCHEDULE
            or normalized_argv(meta.get("solver_command", [])) != normalized_argv(expected_argv)
            or report.get("verified_geo_frames") != 241 or report.get("verified_binary_checkpoints") != 241
            or abs(report["metrics"]["total_solver_s"] - CLEAN_SOLVER_S) > 5e-7):
        raise ValueError("Selected clean run does not match the pinned executable, settings, outputs, or timing baseline")


def load_selected(clean_directory: Path, repo: Path) -> tuple[dict, dict, dict[Path, str]]:
    report_path = clean_directory / "clean_report.json"
    manifest_path = clean_directory / "checkpoint_manifest.json"
    raw = json.loads(report_path.read_text())
    original_path = Path(raw["reference_manifest"])
    original_hash = sha256(original_path)
    reference = read_reference(manifest_path, FRAMES)
    # A clean run's reference_manifest points to the ORIGINAL GPU run, not its
    # own manifest. Their file entries must match; JSON provenance may differ.
    if reference["files"] != read_reference(original_path, FRAMES)["files"]:
        raise ValueError("Selected clean output hashes differ from their validated original reference")
    report = load_clean_comparison(report_path, FRAMES, sha256(repo / "README.md"), original_hash)
    check_selected_metadata(report, command(repo, Path("OUTPUT"), "gpu", FRAMES, readme_command=True))
    pinned = {path: sha256(path) for path in (report_path, manifest_path, original_path)}
    return report, reference, pinned


def add_clean_comparison(report: dict, phases: dict, clean: dict) -> dict:
    solver_times = {row["frame"]: row["profiled_solver_ms"] for row in report["frames"]}
    cpu_times = {int(k): v for k, v in report["baseline"]["frame_solver_ms"].items()}
    for name, first, last in (("early_frames_1_80", 1, 80), ("middle_frames_81_160", 81, 160),
                              ("late_frames_161_240", 161, 240), ("frame_111", 111, 111)):
        ids = sorted(i for i in solver_times if first <= i <= last)
        if ids:
            report["groups"][name] = summarize_group(ids, phases, solver_times, cpu_times)
    clean_times = {i: seconds * 1000 for i, seconds in enumerate(clean["metrics"]["frame_solver_s"], 1)}
    for row in report["frames"]:
        frame = row["frame"]
        row["clean_gpu_solver_ms"] = clean_times[frame]
        row["profiled_over_clean_solver_ratio"] = row["profiled_solver_ms"] / clean_times[frame] if clean_times[frame] else None
    for group in report["groups"].values():
        total = math.fsum(clean_times[i] for i in group["frame_ids"]) / 1000
        group.update(clean_gpu_solver_s=total, clean_gpu_mean_frame_ms=total * 1000 / group["frame_count"],
                     profiled_over_clean_solver_ratio=group["profiled_solver_s"] / total if total else None)
    report.update(instrumented=True, clean_gpu_baseline=clean,
                  feature_effective=FEATURES, schedule_effective=SCHEDULE,
                  comparison_kind="same-GPU-binary clean versus instrumented; NOT optimization speedup")
    report["notes"] += [
        "The selected GPU clean baseline is 1066.265610 solver seconds. Per-frame/group ratios are PROFILED/CLEAN time, not optimization speedups or pure timer overhead estimates.",
        "CPU and GPU use matching input CLI settings, but may use different CCD numerical rules, candidate work, and trajectories. CPU timings are descriptive; byte identity is asserted ONLY between GPU reference and GPU profile outputs.",
        "Exclusive CUDA percentages divide by the sum of exclusive CUDA stream intervals in that frame/group, not by solver wall time, GPU occupancy, or all-worker time.",
        "Host scope time can include CUDA enqueue backpressure and waiting, not just CPU computation. Host and device totals are not additive.",
        "color_visits counts repeated visits across iterations; color_max is a maximum color count, not the number of visits. Candidate maxima are broad-phase proxies, not active exact-contact job counts.",
        "Counter sums of per-frame maxima (for example candidate_max) are sums of maxima, not unique contacts. The report preserves these raw summaries without interpreting them as unique work.",
        "Online hash validation and monitoring add host/I/O activity outside solver scopes. The instrumentation bookkeeping counter excludes some profiling perturbation; no timing correction is subtracted.",
        "Early/middle/late are chronological thirds (1–80, 81–160, 161–240); quartiles independently rank frames by solver candidate maximum. All raw hierarchical stage rows are retained.",
    ]
    return report


def write_reports(root: Path, clean: dict, *, allow_partial=False, validated=False) -> dict:
    phases = read_phases(root / "phase_frames.csv")
    if any(item["metadata"].get("substeps") != 5 or item["metadata"].get("iterations") != 20 for item in phases.values()):
        raise ValueError("Profile counters do not match 5 substeps and 4 iterations per substep")
    report = add_clean_comparison(analyze(root, FRAMES, allow_partial), phases, clean)
    report.update(byte_identical_to_selected_clean=validated,
                  verified_geo_frames=241 if validated else None,
                  verified_binary_checkpoints=241 if validated else None)
    write_json(root / "profile_report.json", report)
    lines = ["Selected GPU configuration: queues=1, warp reduction=0, cooperative exact=1",
             f"Selected clean GPU solver: {CLEAN_SOLVER_S:.6f} s; {CLEAN_SOLVER_S * 1000 / FRAMES:.6f} ms/frame",
             f"Complete GEO/BIN identity validated: {validated}",
             "Profiled/clean time ratios below are NOT optimization speedups:"]
    for name, group in report["groups"].items():
        ratio = group["profiled_over_clean_solver_ratio"]
        lines.append(f"  {name}: clean {group['clean_gpu_solver_s']:.6f} s; profile {group['profiled_solver_s']:.6f} s; ratio {ratio:.6f}" if ratio is not None else f"  {name}: zero clean duration")
    temporary = root / "profile_report.txt.tmp"
    temporary.write_text("\n".join(lines) + "\n\n" + render(report))
    temporary.replace(root / "profile_report.txt")
    with (root / "profile_per_frame.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(report["frames"][0]))
        writer.writeheader();writer.writerows(report["frames"])
    # Raw CSV has all counters; this convenience table adds per-frame exclusive
    # shares without summing hierarchical inclusive values or host+device times.
    with (root / "profile_stages_per_frame.csv").open("w", newline="") as file:
        fields = ["frame", "stage", "calls", *TIMINGS, "cuda_share_of_frame_stream", "host_share_of_frame_wall"]
        writer = csv.DictWriter(file, fieldnames=fields);writer.writeheader()
        for row in report["frames"]:
            frame = row["frame"]
            stream = row["cuda_exclusive_stream_sum_ms"]
            wall = row["frame_wall_ms"]
            for stage, values in phases[frame]["stages"].items():
                writer.writerow(dict(frame=frame, stage=stage, **values,
                    cuda_share_of_frame_stream=values["cuda_exclusive_ms"] / stream if stream else 0,
                    host_share_of_frame_wall=values["host_exclusive_ms"] / wall if wall else 0))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--clean-directory", type=Path)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--cpu-baseline", type=Path)
    parser.add_argument("--timeout", type=float, default=43200)
    parser.add_argument("--monitor-interval", type=float, default=10)
    parser.add_argument("--max-background-cores", type=float, default=2)
    args = parser.parse_args()
    if args.self_test:
        self_test();return
    if not all((args.output, args.clean_directory, args.executable)):
        parser.error("--output, --clean-directory, and --executable are required")
    if (not math.isfinite(args.timeout) or args.timeout <= 0 or not 0 < args.monitor_interval <= 60
            or not math.isfinite(args.max_background_cores) or args.max_background_cores < 0):
        parser.error("Invalid timeout, monitor interval, or background load limit")
    repo = Path(__file__).resolve().parent.parent
    verify_readme_command(repo)
    root = args.output.resolve();root.mkdir(exist_ok=True)
    if any(path.name != "supervisor.log" for path in root.iterdir()):
        raise RuntimeError("Output must be fresh; no existing artifacts are replaced")
    state = dict(kind="selected GPU full-run instrumented profile", status="preparing", pid=os.getpid(),
                 directory=str(root), frames=FRAMES, instrumented=True, accepted=False,
                 feature_effective=FEATURES, schedule_effective=SCHEDULE,
                 monitoring_policy=MONITORING_NOTE, background_observations=[])
    process = None;lock = None;checked = {};clean = None

    def save(status: str, **fields) -> None:
        state.update(status=status, updated_unix=time.time(), **fields)
        write_json(root / "status.json", state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        save("preparing")
        clean_dir, source_exe = args.clean_directory.resolve(), args.executable.resolve()
        clean, reference, pinned = load_selected(clean_dir, repo)
        baseline_path = (args.cpu_baseline or clean_dir / "cpu_baseline.json").resolve()
        baseline = checked_cpu_baseline(baseline_path)
        if baseline != clean["cpu_baseline"]:
            raise ValueError("CPU baseline differs from the selected clean run's preserved baseline")
        pinned.update({baseline_path: sha256(baseline_path), repo / "README.md": sha256(repo / "README.md"),
                       source_exe: EXECUTABLE_SHA, repo / EXPECTED["cpu"][0]: EXPECTED["cpu"][1]})
        for path, digest in pinned.items():
            if sha256(path) != digest:
                raise RuntimeError("Pinned input mismatch: " + str(path))
        frozen = root / "3D_sim_gpu_profiled"
        shutil.copy2(source_exe, frozen);pinned[frozen] = EXECUTABLE_SHA
        write_json(root / "cpu_baseline.json", baseline)
        write_json(root / "selected_clean_report.json", clean)
        write_json(root / "selected_reference_manifest.json", reference)
        for name in ("cpu_baseline.json", "selected_clean_report.json", "selected_reference_manifest.json"):
            pinned[root / name] = sha256(root / name)
        cpu_sources, gpu_sources = cpu_source_snapshot(repo), gpu_source_snapshot(repo)
        write_json(root / "source_snapshot.json", dict(cpu=cpu_sources, gpu=gpu_sources,
            inputs={str(path): digest for path, digest in pinned.items()}))

        def verify() -> None:
            verify_readme_command(repo)
            for path, digest in pinned.items():
                if sha256(path) != digest:
                    raise RuntimeError("Pinned input changed: " + str(path))
            if cpu_source_snapshot(repo) != cpu_sources or gpu_source_snapshot(repo) != gpu_sources:
                raise RuntimeError("CPU/GPU sources changed during profiling; refusing mixed-version results")

        cpus = physical_cpus()
        if cpus != list(range(64)):
            raise RuntimeError("Expected verified physical affinity 0–63")
        lock = (repo / "gpu/.example1_comparison.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        quiet = 0;wait_start = time.monotonic()
        while quiet < 3:
            before = sample();time.sleep(args.monitor_interval);after = sample()
            reasons, load = activity(before, after, [], max_background_cores=args.max_background_cores)
            quiet = 0 if reasons else quiet + 1
            save("waiting_for_idle", reasons=reasons, idle_checks=quiet, external_cpu_cores=load)
            if time.monotonic() - wait_start > args.timeout:
                raise RuntimeError("Timed out waiting for an idle server")
        verify()
        solver_argv = command(repo, root / "frames", "gpu", FRAMES, readme_command=True)
        solver_argv[0] = str(frozen)
        argv = ["stdbuf", "-oL", "-eL", "taskset", "--cpu-list", ",".join(map(str, cpus)), *solver_argv]
        env = clean_environment(SCHEDULE, FEATURES)
        env["IPC_GPU_PROFILE_CSV"] = str(root / "phase_frames.csv")
        metadata = dict(command=argv, solver_command=solver_argv, cpu_affinity=cpus,
            environment={key: value for key, value in env.items() if key.startswith(("OMP_", "IPC_GPU_"))},
            executable_sha256=EXECUTABLE_SHA, cpu_executable_sha256=EXPECTED["cpu"][1],
            readme_sha256=pinned[repo / "README.md"], feature_effective=FEATURES, schedule_effective=SCHEDULE,
            selected_clean_directory=str(clean_dir), selected_clean_total_solver_s=CLEAN_SOLVER_S,
            reference_manifest_sha256=pinned[clean_dir / "checkpoint_manifest.json"],
            cpu_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            cpu_hardware=subprocess.check_output(["lscpu"], text=True),
            gpu_hardware=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True, timeout=10).strip(),
            instrumented=True, note="Exact README physics; only executable/output and timing instrumentation differ from selected clean GPU run. No per-color queue readbacks. CPU/GPU numerical equivalence is not asserted.")
        write_json(root / "command.json", metadata)
        before = sample();start = last_verify = time.monotonic()
        with (root / "run.log").open("w") as log:
            process = subprocess.Popen(argv, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save("profiling", child_pid=process.pid, command=argv)
            while True:
                time.sleep(args.monitor_interval)
                done = process.poll() is not None
                after = sample(process.pid)
                reasons, load = activity(before, after, [], process.pid, max_background_cores=args.max_background_cores)
                state["background_observations"].append(dict(elapsed_s=time.monotonic() - start, external_cpu_cores=load,
                    reasons=reasons, system_iowait_ticks=max(0., after["iowait"] - before["iowait"])))
                before = after
                if reasons:
                    raise RuntimeError("External workload detected: " + "; ".join(reasons))
                through, count = safe_completed_frame(root / "run.log", FRAMES, process_complete=done and process.returncode == 0)
                verify_ready_frames(root / "frames", reference, checked, through)
                save("profiling", completed_frames=count, verified_through_frame=through,
                     verified_files=len(checked), elapsed_s=time.monotonic() - start)
                if time.monotonic() - last_verify >= 60:
                    verify();last_verify = time.monotonic()
                if done:
                    break
                if time.monotonic() - start > args.timeout:
                    raise RuntimeError("Full-run profiling timeout")
        if process.returncode:
            raise RuntimeError(f"GPU application returned {process.returncode}")
        elapsed = time.monotonic() - start
        verify();save("validating", child_pid=None)
        parsed = parse_trial(root / "run.log", root / "frames", FRAMES, "gpu", readme_command=True)
        parsed.update(instrumented=True, process_elapsed_s=elapsed)
        manifest = checkpoint_manifest(root / "frames", FRAMES)
        if manifest["files"] != reference["files"]:
            raise ValueError("Final 482-file manifest differs from selected clean GPU run")
        manifest.update(kind="byte-validated instrumented GPU output manifest", reference_manifest_sha256=metadata["reference_manifest_sha256"])
        write_json(root / "checkpoint_manifest.json", manifest)
        write_json(root / "instrumented_run.json", parsed)
        write_reports(root, clean, validated=True)
        verify()
        save("complete", completed_frames=FRAMES, verified_files=482, verified_through_frame=240,
             validated_geo_frames=241, validated_binary_checkpoints=241, byte_identical_to_reference=True,
             accepted=True, instrumented_solver_s=parsed["total_solver_s"], clean_solver_s=CLEAN_SOLVER_S,
             report=str(root / "profile_report.txt"))
        write_reports(root, clean, validated=True)
    except BaseException as error:
        if process is not None:
            stop_our_process(process)
        write_json(root / "partial_verified_manifest.json", dict(files=checked, note="Partial verified files; not a completed run"))
        save("stopped", child_pid=None, accepted=False, error=f"{type(error).__name__}: {error}",
             mismatch=error.details if isinstance(error, CheckpointMismatch) else None, verified_files=len(checked))
        if clean is not None:
            try:
                write_reports(root, clean, allow_partial=True)
            except (OSError, ValueError, KeyError):
                pass
        raise
    finally:
        if process is not None:
            stop_our_process(process)
        if lock is not None:
            lock.close()


def self_test() -> None:
    import tempfile
    import unittest
    from unittest.mock import patch

    class Tests(unittest.TestCase):
        def test_environment_and_selected_metadata(self):
            with patch.dict(os.environ, {"IPC_GPU_CONTACT_QUEUE_CSV": "bad", "IPC_GPU_PROFILE_CSV": "bad", "IPC_GPU_WARP_CONTACT_REDUCE": "1"}):
                env = clean_environment(SCHEDULE, FEATURES)
            self.assertNotIn("IPC_GPU_CONTACT_QUEUE_CSV", env)
            self.assertNotIn("IPC_GPU_PROFILE_CSV", env)
            self.assertEqual(env[FEATURE_ENV["warp_contact_reduce"]], "0")
            self.assertEqual(env[SCHEDULE_ENV["small_threads"]], "1")
            argv = ["exe", "--example", "1", "--outdir", "out"]
            report = dict(command_metadata=dict(executable_sha256=EXECUTABLE_SHA,
                environment={k: v for k, v in env.items() if k.startswith(("OMP_", "IPC_GPU_"))},
                feature_effective=FEATURES, schedule_defaults=SCHEDULE, solver_command=argv),
                verified_geo_frames=241, verified_binary_checkpoints=241, metrics=dict(total_solver_s=CLEAN_SOLVER_S))
            check_selected_metadata(report, argv)
            report["command_metadata"]["environment"]["IPC_GPU_WARP_CONTACT_REDUCE"] = "1"
            with self.assertRaises(ValueError):
                check_selected_metadata(report, argv)

        def test_full_report_and_incomplete_rejected(self):
            with tempfile.TemporaryDirectory(prefix="gpu_round2_profile_test_") as directory:
                root = Path(directory)
                baseline = dict(kind="accepted_cpu_baseline", frame_solver_ms={str(i): 1. for i in range(1, 241)},
                    cpu_total_solver_s=.240, cpu_average_solver_frame_ms=1., cpu_total_simulation_s=.480,
                    cpu_average_simulation_frame_ms=2.)
                write_json(root / "cpu_baseline.json", baseline)
                (root / "run.log").write_text("".join(f"Frame {i} | solver_time = {i * 2}.0 ms\n" for i in range(1, 241)))
                header = ["frame", "stage", "calls", *TIMINGS, "frame_wall_ms", "profile_cpu_overhead_ms", "substeps", "iterations", "solver_candidate_max"]
                with (root / "phase_frames.csv").open("w", newline="") as file:
                    writer = csv.writer(file);writer.writerow(header)
                    for i in range(1, 241):
                        writer.writerow([i, "frame", 1, 11, 1, 8, 0, 11, .1, 5, 20, i * 10])
                        writer.writerow([i, "substep", 5, 10, 10, 8, 8, 11, .1, 5, 20, i * 10])
                clean = dict(metrics=dict(frame_solver_s=[i / 1000 for i in range(1, 241)]))
                report = write_reports(root, clean, validated=True)
                self.assertEqual(report["groups"]["late_frames_161_240"]["frame_count"], 80)
                self.assertEqual(report["groups"]["candidate_work_quartile_4"]["frame_ids"], list(range(181, 241)))
                self.assertEqual(report["groups"]["all_frames"]["profiled_over_clean_solver_ratio"], 2)
                self.assertEqual(report["frames"][110]["clean_gpu_solver_ms"], 111)
                self.assertEqual(report["verified_binary_checkpoints"], 241)
                self.assertIn("NOT optimization speedups", (root / "profile_report.txt").read_text())
                (root / "run.log").write_text("Frame 1 | solver_time = 2 ms\n")
                with self.assertRaises(ValueError):
                    write_reports(root, clean)
                self.assertEqual(write_reports(root, clean, allow_partial=True)["completed_frames"], 1)

        def test_checkpoint_gate(self):
            with tempfile.TemporaryDirectory(prefix="gpu_round2_hash_test_") as directory:
                root = Path(directory);files = {}
                for name in ("frame_0000.geo", "state_0000.bin"):
                    path = root / name;path.write_bytes(b"reference")
                    files[name] = dict(bytes=9, sha256=sha256(path))
                verify_ready_frames(root, dict(files=files), {}, 0)
                (root / "state_0000.bin").write_bytes(b"different")
                with self.assertRaises(CheckpointMismatch):
                    verify_ready_frames(root, dict(files=files), {}, 0)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
