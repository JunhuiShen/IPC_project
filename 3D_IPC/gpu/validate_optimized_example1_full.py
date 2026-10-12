#!/usr/bin/env python3
"""Monitored clean README Example 1 GPU run with online byte-identity checks.

No diagnostic timers or queue readbacks are enabled. A frame's GEO and binary
checkpoint are checked only after the NEXT frame timing line, or successful
process exit, because its own timing line can precede checkpoint completion.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from analyze_example1_full_profile import load_baseline, log_frames
from example1_comparison import command, environment, parse_trial, verify_readme_command
from profile_example1_full import checkpoint_manifest, sha256
from wait_example1_comparison import (
    EXPECTED, MONITORING_NOTE, activity, physical_cpus, sample, stop_our_process,
)

SCHEDULE_DEFAULTS = dict(small_workers=1024, small_threads=1, large_workers=256, large_threads=4)
SCHEDULE_ENV = {name: "IPC_GPU_CONTACT_EXACT_" + name.upper() for name in SCHEDULE_DEFAULTS}
FEATURE_DEFAULTS = dict(color_pair_queues=1, warp_contact_reduce=1, cooperative_exact_contact=1)
FEATURE_ENV = {name: "IPC_GPU_" + name.upper() for name in FEATURE_DEFAULTS}
COUNTERPARTS = (("frame_", ".geo"), ("state_", ".bin"))


def valid_sha(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_schedule(schedule: dict) -> None:
    if set(schedule) != set(SCHEDULE_DEFAULTS):
        raise ValueError("All four exact-contact scheduling fields are required")
    for name, maximum in (("small_workers", 4096), ("large_workers", 1024)):
        if type(schedule[name]) is not int or not 1 <= schedule[name] <= maximum:
            raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    for kind in ("small", "large"):
        threads = schedule[kind + "_threads"]
        if type(threads) is not int or threads not in (1, 2, 4, 8, 16, 32):
            raise ValueError(kind + "_threads must be one of 1,2,4,8,16,32")
        if schedule[kind + "_workers"] % threads:
            raise ValueError(kind + " workers must be divisible by threads per block")


def effective_features(overrides: dict | None = None) -> dict:
    features = dict(FEATURE_DEFAULTS, **(overrides or {}))
    if set(features) != set(FEATURE_DEFAULTS) or any(type(value) is not int or value not in (0, 1) for value in features.values()):
        raise ValueError("Contact execution features must be known fields with integer values 0 or 1")
    return features


def clean_environment(overrides: dict, feature_overrides: dict | None = None) -> dict:
    effective = dict(SCHEDULE_DEFAULTS, **overrides)
    validate_schedule(effective)
    features = effective_features(feature_overrides)
    # Do not inherit either profiling/queue diagnostics or an old scheduling
    # experiment. Restore explicit scheduling overrides and ALL effective
    # contact feature choices, so the report never relies on inherited flags.
    result = {key: value for key, value in environment("gpu", readme_command=True).items()
              if not key.startswith("IPC_GPU_")}
    result.update({SCHEDULE_ENV[key]: str(value) for key, value in overrides.items()})
    result.update({FEATURE_ENV[key]: str(value) for key, value in features.items()})
    return result


def names_for_frame(frame: int) -> list[str]:
    return [f"{prefix}{frame:04d}{suffix}" for prefix, suffix in COUNTERPARTS]


def read_reference(path: Path, frames: int) -> dict:
    reference = json.loads(path.read_text())
    count = reference.get("frames")
    if type(count) is not int or count < frames or count > 240:
        raise ValueError("Reference manifest must cover the entire requested frame range")
    expected = {name for frame in range(count + 1) for name in names_for_frame(frame)}
    files = reference.get("files", {})
    if set(files) != expected:
        raise ValueError("Reference manifest must contain exactly all GEO and binary frames 0..N")
    for name, item in files.items():
        if (not isinstance(item, dict) or type(item.get("bytes")) is not int or item["bytes"] <= 0
                or not valid_sha(item.get("sha256"))):
            raise ValueError("Invalid reference manifest entry: " + name)
    return reference


class CheckpointMismatch(ValueError):
    def __init__(self, name: str, expected: dict, actual: dict | None):
        self.details = dict(file=name, expected=expected, actual=actual)
        super().__init__("Checkpoint differs from original GPU run: " + name)


def verify_ready_frames(outdir: Path, reference: dict, checked: dict, through: int) -> None:
    for frame in range(through + 1):
        for name in names_for_frame(frame):
            if name in checked:
                continue
            path = outdir / name
            expected = reference["files"][name]
            if not path.is_file():
                raise CheckpointMismatch(name, expected, None)
            actual = dict(bytes=path.stat().st_size, sha256=sha256(path))
            if actual != expected:
                raise CheckpointMismatch(name, expected, actual)
            checked[name] = actual


def safe_completed_frame(log: Path, expected_frames: int, *, process_complete=False) -> tuple[int, int]:
    frames = log_frames(log)
    if list(frames) != list(range(1, len(frames) + 1)) or len(frames) > expected_frames:
        raise ValueError("Run log contains an out-of-order or excessive frame sequence")
    if process_complete:
        if len(frames) != expected_frames:
            raise ValueError("Process finished without the requested frame count")
        return expected_frames, len(frames)
    return len(frames) - 1, len(frames)


def checked_cpu_baseline(path: Path) -> dict:
    baseline = load_baseline(path)
    values = baseline.get("frame_solver_ms", {})
    if set(values) != {str(i) for i in range(1, 241)}:
        raise ValueError("CPU baseline must retain all 240 frame timings")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("Invalid CPU baseline frame timing")
    total = baseline.get("cpu_total_solver_s")
    if not isinstance(total, (int, float)) or total <= 0 or not math.isfinite(total):
        raise ValueError("Invalid CPU baseline solver total")
    if abs(math.fsum(values.values()) / 1000 - total) > .00000051 * 241:
        raise ValueError("CPU baseline total and frame timings disagree")
    return baseline


def load_clean_comparison(path: Path, frames: int, readme_hash: str, reference_hash: str) -> dict:
    report = json.loads(path.read_text())
    if (report.get("kind") != "complete_clean_gpu_validation" or report.get("frames") != frames
            or report.get("byte_identical_to_reference") is not True or report.get("instrumented") is not False
            or report.get("reference_manifest_sha256") != reference_hash
            or report.get("command_metadata", {}).get("readme_sha256") != readme_hash
            or report.get("command_metadata", {}).get("cpu_affinity") != list(range(64))):
        raise ValueError("Clean comparison must be a complete, byte-validated matching README run")
    metrics = report.get("metrics", {})
    if (metrics.get("complete") is not True or metrics.get("frames") != frames
            or metrics.get("instrumented") is not False or metrics.get("readme_command") is not True):
        raise ValueError("Clean comparison is missing complete positive solver timing")
    for field in ("total_solver_s", "total_simulation_s", "average_frame_s"):
        value = metrics.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid clean comparison timing: " + field)
    frame_times = metrics.get("frame_solver_s", [])
    if (len(frame_times) != frames or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in frame_times)
            or abs(math.fsum(frame_times) - metrics["total_solver_s"]) > .00000051 * (frames + 1)):
        raise ValueError("Clean comparison total and per-frame timings disagree")
    if any(key.startswith("IPC_GPU_PROFILE") or key == "IPC_GPU_CONTACT_QUEUE_CSV"
           for key in report["command_metadata"].get("environment", {})):
        raise ValueError("Clean comparison metadata enables diagnostic instrumentation")
    return report


def build_report(root: Path, parsed: dict, baseline: dict, metadata: dict, reference_path: Path,
                 reference_hash: str, baseline_clean: dict | None) -> dict:
    count = parsed["frames"]
    cpu_solver = math.fsum(baseline["frame_solver_ms"][str(i)] for i in range(1, count + 1)) / 1000
    report = dict(kind="complete_clean_gpu_validation", frames=count, instrumented=False,
                  byte_identical_to_reference=True, verified_geo_frames=count + 1,
                  verified_binary_checkpoints=count + 1, reference_manifest=str(reference_path),
                  reference_manifest_sha256=reference_hash, checkpoint_manifest=str(root / "checkpoint_manifest.json"),
                  command_metadata=metadata, metrics=parsed, cpu_baseline=baseline,
                  gpu_over_cpu_solver_ratio=parsed["total_solver_s"] / cpu_solver,
                  compared_cpu_solver_s=cpu_solver,
                  notes=["GPU timing is clean: phase profiling and per-color queue readbacks are disabled.",
                         "The preserved CPU baseline is a separate clean 64-thread run; CPU code/executable are not modified or rerun.",
                         "GEO and complete binary state files are byte-identical to the reference at EVERY frame including frame 0.",
                         "Online verification waits for the next frame log (or process completion), then final validation re-hashes every file.",
                         "Hashing and monitoring add host/I/O activity, not solver timer scopes; this is a monitored clean solver benchmark, not a zero-observer-overhead experiment.",
                         "process_elapsed_s includes process startup/output and supervisor polling latency; use total_simulation_s for the application's measured simulation runtime.",
                         MONITORING_NOTE])
    diagnostic = reference_path.parent / "profile_report.json"
    if diagnostic.is_file():
        profile = json.loads(diagnostic.read_text())
        if (profile.get("kind") == "complete_diagnostic_profile" and profile.get("completed_frames") == count
                and profile.get("command_metadata", {}).get("readme_sha256") == metadata["readme_sha256"]):
            profiled_total = profile["groups"]["all_frames"]["profiled_solver_s"]
            report["instrumented_reference_comparison"] = dict(source=str(diagnostic),
                profiled_solver_s=profiled_total, profiled_over_clean_solver_ratio=profiled_total / parsed["total_solver_s"],
                note="Original GPU reference timing includes profiling. This ratio is NOT a clean optimization speedup; use the optional clean-vs-clean comparison.")
    if baseline_clean is not None:
        old = baseline_clean["metrics"]
        report["clean_gpu_comparison"] = dict(baseline_gpu_sha256=baseline_clean["command_metadata"]["executable_sha256"],
            baseline_total_solver_s=old["total_solver_s"], baseline_average_frame_s=old["average_frame_s"],
            clean_solver_speedup=old["total_solver_s"] / parsed["total_solver_s"],
            baseline_total_simulation_s=old["total_simulation_s"],
            clean_simulation_speedup=old["total_simulation_s"] / parsed["total_simulation_s"])
    report["frames_detail"] = [dict(frame=i, gpu_solver_ms=value * 1000,
        cpu_solver_ms=baseline["frame_solver_ms"][str(i)],
        baseline_clean_gpu_solver_ms=baseline_clean["metrics"]["frame_solver_s"][i - 1] * 1000 if baseline_clean else None)
        for i, value in enumerate(parsed["frame_solver_s"], 1)]
    return report


def write_report(root: Path, report: dict) -> None:
    (root / "clean_report.json").write_text(json.dumps(report, indent=2) + "\n")
    metrics, cpu = report["metrics"], report["cpu_baseline"]
    lines = ["Example 1 clean GPU full-run validation", f"Frames: {report['frames']}",
        f"Byte identity: all {report['verified_geo_frames']} GEO and {report['verified_binary_checkpoints']} binary checkpoint files match.",
        f"GPU total solver: {metrics['total_solver_s']:.6f} s",
        f"GPU average solver/frame: {metrics['average_frame_s']*1000:.6f} ms",
        f"GPU average solver/substep: {metrics['average_substep_s']*1000:.6f} ms",
        f"GPU total simulation: {metrics['total_simulation_s']:.6f} s", "",
        f"Preserved CPU 240-frame solver: {cpu['cpu_total_solver_s']:.6f} s; {cpu['cpu_average_solver_frame_ms']:.6f} ms/frame",
        f"GPU/CPU solver time for matching frames: {report['gpu_over_cpu_solver_ratio']:.6f}x (greater than 1 means GPU slower)"]
    features = report["command_metadata"].get("feature_effective")
    if features is not None:
        lines += ["Contact execution features (explicit 0/1): " + ", ".join(f"{name}={value}" for name, value in features.items())]
    if "clean_gpu_comparison" in report:
        comparison = report["clean_gpu_comparison"]
        lines += ["", f"Clean previous GPU solver: {comparison['baseline_total_solver_s']:.6f} s",
                  f"Clean-vs-clean solver speedup: {comparison['clean_solver_speedup']:.6f}x"]
    if "instrumented_reference_comparison" in report:
        comparison = report["instrumented_reference_comparison"]
        lines += ["", f"Instrumented original GPU solver: {comparison['profiled_solver_s']:.6f} s",
                  f"Instrumented/clean ratio: {comparison['profiled_over_clean_solver_ratio']:.6f}x",
                  comparison["note"]]
    lines += ["", "Notes:", *["- " + note for note in report["notes"]]]
    (root / "clean_report.txt").write_text("\n".join(lines) + "\n")
    with (root / "clean_per_frame.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(report["frames_detail"][0]))
        writer.writeheader();writer.writerows(report["frames_detail"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--cpu-baseline", type=Path)
    parser.add_argument("--baseline-clean-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--timeout", type=float, default=43200)
    parser.add_argument("--monitor-interval", type=float, default=10)
    parser.add_argument("--max-background-cores", type=float, default=2)
    for name in SCHEDULE_DEFAULTS:
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name, default in FEATURE_DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, choices=(0, 1),
                            help=f"Explicit GPU execution feature; effective default {default}")
    args = parser.parse_args()
    if args.self_test:
        self_test();return
    if any(getattr(args, name) is None for name in ("executable", "sha256", "reference_manifest", "cpu_baseline", "output")):
        parser.error("--executable, --sha256, --reference-manifest, --cpu-baseline and --output are required")
    if (not valid_sha(args.sha256) or not 1 <= args.frames <= 240 or not 0 < args.timeout < float("inf")
            or not 0 < args.monitor_interval <= 60 or not 0 <= args.max_background_cores < float("inf")):
        parser.error("Invalid hash, frame count, timeout, monitoring interval, or background CPU ceiling")
    overrides = {name: getattr(args, name) for name in SCHEDULE_DEFAULTS if getattr(args, name) is not None}
    feature_overrides = {name: getattr(args, name) for name in FEATURE_DEFAULTS if getattr(args, name) is not None}
    features = effective_features(feature_overrides)
    env = clean_environment(overrides, feature_overrides)
    repo = Path(__file__).resolve().parent.parent
    executable, reference_path = args.executable.resolve(), args.reference_manifest.resolve()
    reference = read_reference(reference_path, args.frames)
    reference_hash, readme_hash = sha256(reference_path), sha256(repo / "README.md")
    baseline = checked_cpu_baseline(args.cpu_baseline.resolve())
    baseline_hash = sha256(args.cpu_baseline.resolve())
    clean_comparison = load_clean_comparison(args.baseline_clean_report.resolve(), args.frames, readme_hash, reference_hash) if args.baseline_clean_report else None
    comparison_hash = sha256(args.baseline_clean_report.resolve()) if args.baseline_clean_report else None
    root = args.output.resolve();root.mkdir(exist_ok=True)
    if any(path.name != "supervisor.log" for path in root.iterdir()):
        raise RuntimeError("Use a fresh validation directory; existing output is never replaced")
    (root / "cpu_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    cpus = physical_cpus()
    if cpus != list(range(64)):
        raise RuntimeError("Expected verified physical CPU affinity 0-63; refusing a different layout")
    state = dict(kind="clean GPU full-run validation", status="waiting_for_idle", pid=os.getpid(), frames=args.frames,
        directory=str(root), readme_command=True, executable_sha256=args.sha256, reference_manifest_sha256=reference_hash,
        cpu_affinity=cpus, monitoring_policy=MONITORING_NOTE, max_background_cores=args.max_background_cores,
        schedule_overrides=overrides, schedule_defaults=SCHEDULE_DEFAULTS,
        feature_overrides=feature_overrides, feature_defaults=FEATURE_DEFAULTS, feature_effective=features,
        background_observations=[])
    checked = {}

    def save(status: str, **fields) -> None:
        state.update(status=status, updated_unix=time.time(), **fields)
        temporary = root / "status.tmp";temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "status.json")

    def verify() -> None:
        verify_readme_command(repo)
        for path, expected, label in (
            (executable, args.sha256, "GPU executable"),
            (repo / EXPECTED["cpu"][0], EXPECTED["cpu"][1], "production CPU executable"),
            (repo / "README.md", readme_hash, "README"),
            (reference_path, reference_hash, "reference manifest"),
            (args.cpu_baseline.resolve(), baseline_hash, "CPU baseline")):
            if sha256(path) != expected:
                raise RuntimeError(label + " changed; refusing mixed-version validation")
        if args.baseline_clean_report and sha256(args.baseline_clean_report.resolve()) != comparison_hash:
            raise RuntimeError("Clean comparison report changed")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    lock = (repo / "gpu/.example1_comparison.lock").open("a+")
    process = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify();clean = 0
        while clean < 3:
            before = sample();time.sleep(args.monitor_interval);after = sample()
            reasons, load = activity(before, after, [], max_background_cores=args.max_background_cores)
            clean = 0 if reasons else clean + 1
            save("waiting_for_idle", reasons=reasons, idle_checks=clean, external_cpu_cores=load)
        verify()
        solver_argv = command(repo, root / "frames", "gpu", args.frames, readme_command=True)
        solver_argv[0] = str(executable)
        argv = ["stdbuf", "-oL", "-eL", "taskset", "--cpu-list", ",".join(map(str, cpus)), *solver_argv]
        metadata = dict(command=argv, solver_command=solver_argv,
            environment={key: value for key, value in env.items() if key.startswith(("OMP_", "IPC_GPU_"))},
            executable_sha256=args.sha256, cpu_executable_sha256=EXPECTED["cpu"][1], readme_sha256=readme_hash,
            cpu_affinity=cpus, schedule_overrides=overrides, schedule_defaults=SCHEDULE_DEFAULTS,
            feature_overrides=feature_overrides, feature_defaults=FEATURE_DEFAULTS, feature_effective=features,
            cpu_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            cpu_hardware=subprocess.check_output(["lscpu"], text=True),
            gpu_hardware=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
            instrumented=False, frame_count_override=args.frames if args.frames != 240 else None,
            note="README Example 1 argv, except executable and fresh output directory; an explicitly shorter validation uses the recorded frame_count_override. All inherited IPC_GPU_* variables removed; explicit scheduling overrides and all three effective contact execution flags are then set. Contact feature choices affect execution only, not physics parameters.")
        (root / "command.json").write_text(json.dumps(metadata, indent=2) + "\n")
        before = sample();start = time.monotonic()
        with (root / "run.log").open("w") as log:
            process = subprocess.Popen(argv, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save("running", child_pid=process.pid, command=argv)
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
                through, completed = safe_completed_frame(root / "run.log", args.frames,
                    process_complete=done and process.returncode == 0)
                verify_ready_frames(root / "frames", reference, checked, through)
                save("running", completed_frames=completed, verified_through_frame=through,
                     verified_files=len(checked), elapsed_s=time.monotonic() - start)
                if done:
                    break
                if time.monotonic() - start > args.timeout:
                    raise RuntimeError("Clean validation timeout; partial output is not a full-run result")
        if process.returncode:
            raise RuntimeError(f"GPU application returned {process.returncode}")
        observed_elapsed = time.monotonic() - start
        verify();save("validating", child_pid=None)
        parsed = parse_trial(root / "run.log", root / "frames", args.frames, "gpu", readme_command=True)
        parsed.update(instrumented=False, process_elapsed_s=observed_elapsed)
        manifest = checkpoint_manifest(root / "frames", args.frames)
        expected = {name: reference["files"][name] for frame in range(args.frames + 1) for name in names_for_frame(frame)}
        if manifest["files"] != expected:
            for name, entry in manifest["files"].items():
                if entry != expected[name]:
                    raise CheckpointMismatch(name, expected[name], entry)
            raise ValueError("Final checkpoint manifest differs from expected file set")
        manifest.update(kind="byte-validated clean GPU output manifest", reference_manifest_sha256=reference_hash)
        (root / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (root / "clean_run.json").write_text(json.dumps(parsed, indent=2) + "\n")
        report = build_report(root, parsed, baseline, metadata, reference_path, reference_hash, clean_comparison)
        write_report(root, report)
        save("complete", completed_frames=args.frames, verified_through_frame=args.frames,
            verified_files=2 * (args.frames + 1), byte_identical_to_reference=True,
            total_solver_s=parsed["total_solver_s"], average_solver_frame_ms=parsed["average_frame_s"] * 1000,
            total_simulation_s=parsed["total_simulation_s"], report=str(root / "clean_report.txt"))
    except BaseException as error:
        if process is not None:
            stop_our_process(process)
        (root / "partial_verified_manifest.json").write_text(json.dumps(dict(files=checked, note="Only already checked files; NOT a completed run"), indent=2) + "\n")
        save("stopped", child_pid=None, error=f"{type(error).__name__}: {error}",
             mismatch=error.details if isinstance(error, CheckpointMismatch) else None,
             verified_files=len(checked), accepted=False)
        raise
    finally:
        if process is not None:
            stop_our_process(process)
        lock.close()


def self_test() -> None:
    import tempfile
    import unittest
    from unittest.mock import patch

    class Tests(unittest.TestCase):
        def test_clean_environment(self):
            with patch.dict(os.environ, {"IPC_GPU_PROFILE_CSV": "bad.csv", "IPC_GPU_CONTACT_QUEUE_CSV": "bad_queue.csv",
                                        "IPC_GPU_CONTACT_EXACT_SMALL_WORKERS": "64", "IPC_GPU_UNKNOWN": "1", "OMP_NUM_THREADS": "3"}):
                env = clean_environment(dict(small_threads=1, large_threads=1))
            self.assertEqual({k: v for k, v in env.items() if k.startswith("IPC_GPU_")},
                {"IPC_GPU_CONTACT_EXACT_SMALL_THREADS": "1", "IPC_GPU_CONTACT_EXACT_LARGE_THREADS": "1",
                 **{name: "1" for name in FEATURE_ENV.values()}})
            self.assertEqual(env["OMP_NUM_THREADS"], "64")
            self.assertEqual(env["OMP_WAIT_POLICY"], "ACTIVE")
            for changes in (dict(small_workers=0), dict(large_workers=1025), dict(small_threads=3), dict(small_workers=9, small_threads=8)):
                with self.assertRaises(ValueError):
                    clean_environment(changes)

        def test_explicit_contact_features(self):
            with patch.dict(os.environ, {"IPC_GPU_WARP_CONTACT_REDUCE": "1", "IPC_GPU_COLOR_PAIR_QUEUES": "0",
                                        "IPC_GPU_COOPERATIVE_EXACT_CONTACT": "0", "IPC_GPU_PROFILE_CSV": "bad.csv"}):
                env = clean_environment({}, dict(warp_contact_reduce=0))
            self.assertEqual({k: v for k, v in env.items() if k.startswith("IPC_GPU_")},
                             {"IPC_GPU_COLOR_PAIR_QUEUES": "1", "IPC_GPU_WARP_CONTACT_REDUCE": "0",
                              "IPC_GPU_COOPERATIVE_EXACT_CONTACT": "1"})
            self.assertEqual(effective_features(dict(warp_contact_reduce=0)),
                             dict(color_pair_queues=1, warp_contact_reduce=0, cooperative_exact_contact=1))
            self.assertEqual({env[key] for key in FEATURE_ENV.values()}, {"0", "1"})
            for bad in (dict(warp_contact_reduce=2), dict(warp_contact_reduce=True),
                        dict(color_pair_queues="0"), dict(unknown_feature=1)):
                with self.assertRaises(ValueError):
                    clean_environment({}, bad)

        def test_online_waits_and_mismatch(self):
            with tempfile.TemporaryDirectory(prefix="gpu_full_clean_test_") as name:
                root = Path(name);out = root / "frames";out.mkdir()
                files = {}
                for frame in range(3):
                    for filename in names_for_frame(frame):
                        path = out / filename;path.write_bytes((filename + "complete").encode())
                        files[filename] = dict(bytes=path.stat().st_size, sha256=sha256(path))
                reference_path = root / "reference.json"
                reference_path.write_text(json.dumps(dict(frames=2, files=files)))
                reference = read_reference(reference_path, 2);checked = {}
                log = root / "run.log";log.write_text("Frame 1 | solver_time = 1 ms\n")
                (out / "state_0001.bin").write_bytes(b"still writing")
                through, done = safe_completed_frame(log, 2)
                self.assertEqual((through, done), (0, 1))
                verify_ready_frames(out, reference, checked, through)
                self.assertEqual(len(checked), 2)
                log.write_text(log.read_text() + "Frame 2 | solver_time = 2 ms\n")
                through, done = safe_completed_frame(log, 2)
                with self.assertRaises(CheckpointMismatch):
                    verify_ready_frames(out, reference, checked, through)
                (out / "state_0001.bin").write_bytes(b"state_0001.bincomplete")
                through, done = safe_completed_frame(log, 2, process_complete=True)
                verify_ready_frames(out, reference, checked, through)
                self.assertEqual(checked, files)
                log.write_text("Frame 2 | solver_time = 1 ms\n")
                with self.assertRaises(ValueError):
                    safe_completed_frame(log, 2)

        def test_reference_rejects_missing_or_bad_files(self):
            with tempfile.TemporaryDirectory(prefix="gpu_full_manifest_test_") as name:
                path = Path(name) / "reference.json"
                for document in (dict(frames=1, files={}), dict(frames=0, files={}),
                    dict(frames=1, files={key: dict(bytes=1, sha256="bad") for i in range(2) for key in names_for_frame(i)})):
                    path.write_text(json.dumps(document))
                    with self.assertRaises(ValueError):
                        read_reference(path, 1)

        def test_report_labels_and_clean_comparison(self):
            with tempfile.TemporaryDirectory(prefix="gpu_full_report_test_") as name:
                root = Path(name)
                baseline = dict(kind="accepted_cpu_baseline", frame_solver_ms={str(i): 1. for i in range(1, 241)},
                    cpu_total_solver_s=.240, cpu_average_solver_frame_ms=1., cpu_total_simulation_s=.3,
                    cpu_average_simulation_frame_ms=1.25)
                metrics = dict(frames=2, complete=True, instrumented=False, readme_command=True,
                    total_solver_s=.004, average_frame_s=.002, average_substep_s=.0004,
                    total_simulation_s=.005, frame_solver_s=[.002, .002])
                metadata = dict(readme_sha256="1" * 64, executable_sha256="2" * 64,
                    cpu_affinity=list(range(64)), environment={"OMP_NUM_THREADS": "64"})
                reference = root / "checkpoint_manifest.json"
                diagnostic = dict(kind="complete_diagnostic_profile", completed_frames=2,
                    command_metadata=metadata, groups={"all_frames": {"profiled_solver_s": .008}})
                (root / "profile_report.json").write_text(json.dumps(diagnostic))
                report = build_report(root, metrics, baseline, metadata, reference, "3" * 64, None)
                self.assertEqual(report["gpu_over_cpu_solver_ratio"], 2.)
                self.assertNotIn("clean_gpu_comparison", report)
                self.assertIn("NOT a clean optimization speedup", report["instrumented_reference_comparison"]["note"])
                write_report(root, report)
                clean_path = root / "clean_report.json"
                loaded = load_clean_comparison(clean_path, 2, "1" * 64, "3" * 64)
                faster = dict(metrics, total_solver_s=.002, average_frame_s=.001,
                    total_simulation_s=.003, frame_solver_s=[.001, .001])
                compared = build_report(root, faster, baseline, metadata, reference, "3" * 64, loaded)
                self.assertEqual(compared["clean_gpu_comparison"]["clean_solver_speedup"], 2.)
                report["command_metadata"]["environment"]["IPC_GPU_PROFILE_CSV"] = "unexpected.csv"
                clean_path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    load_clean_comparison(clean_path, 2, "1" * 64, "3" * 64)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
