#!/usr/bin/env python3
"""Clean, byte-checked round-3 GPU trials against the current selected solver.

Run on the server only after an isolated build and test chain succeeds. This
driver never builds, edits source, transfers files, or launches a CPU solver.
Cold restarts are compared with fresh same-checkpoint baseline restarts; only
the optional uninterrupted 240-frame run uses continuous reference hashes.
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
import time

from example1_comparison import parse_trial, verify_readme_command
from finish_example1_optimization import cpu_source_snapshot, gpu_source_snapshot
from profile_contact_round2_full import EXECUTABLE_SHA, FEATURES, SCHEDULE, load_selected, write_json
from profile_example1_full import checkpoint_manifest, sha256
from tune_example1_contact import SCENARIOS, parse_segment, segment_command
from validate_optimized_example1_full import (
    build_report, checked_cpu_baseline, clean_environment, safe_completed_frame,
    valid_sha, verify_ready_frames, write_report,
)
from wait_example1_comparison import HZ, MONITORING_NOTE, activity, physical_cpus, process_stat, sample, stop_our_process

CANDIDATE_FLAGS = {"IPC_GPU_COMPACT_DEFORMABLE_CONTACTS": "1", "IPC_GPU_BOX_CONTACT_PRUNING": "1",
                   "IPC_GPU_ELASTIC_COLOR_WORKLISTS": "1"}
NOTES = [
    "Both binaries use identical README physics, 64 host threads and explicit queues=1, warp-reduction=0, cooperative-exact=1; exact workers are 1024:1 and 256:4.",
    "Only the candidate enables compact deformable contacts, whole-box contact pruning, and elastic color worklists. Baseline leaves these new environment switches unset.",
    "All inherited IPC_GPU variables are removed before setting explicit flags. Phase profiling and per-color queue readbacks remain OFF.",
    "Every candidate GEO and complete binary checkpoint is compared with the fresh baseline from the SAME starting checkpoint, with reset caches. Continuous hashes are not cold-restart references.",
    "Two repetitions per stage are a small screening sample. Short-stage speedups are NOT full-simulation speedups.",
    "Only an explicitly requested, completed continuous 240-frame run reports a full clean-vs-clean speedup against the 1066.265610-second selected GPU baseline.",
    "CPU timings are descriptive matching-input-CLI comparisons, not proof of identical CCD rules, candidate work or trajectories. Byte identity is checked between GPU runs only.",
    "Production CPU sources/executable and candidate isolated source hashes are checked unchanged; no production executable is overwritten.",
    MONITORING_NOTE,
]


def execution_environment(candidate: bool) -> dict:
    env = clean_environment(SCHEDULE, FEATURES)
    if candidate:
        env.update(CANDIDATE_FLAGS)
    return env


def validate_marker(path: Path, candidate_sha: str) -> tuple[dict, dict[Path, str]]:
    marker = json.loads(path.read_text())
    if (marker.get("validation_chain_exit_code") != 0 or type(marker.get("validation_chain_exit_code")) is not int
            or marker.get("candidate_sha256") != candidate_sha or not marker.get("evidence")):
        raise ValueError("Validation marker must certify a successful test chain for this exact candidate")
    pinned = {path: sha256(path)}
    for name, digest in marker["evidence"].items():
        log = Path(name)
        if not log.is_absolute() or not valid_sha(digest) or not log.is_file() or not log.stat().st_size or sha256(log) != digest:
            raise ValueError("Missing/changed test evidence: " + name)
        if log.stat().st_mtime > path.stat().st_mtime + .001:
            raise ValueError("Test evidence is newer than its success marker: " + name)
        pinned[log] = digest
    item = marker.get("source_manifest", {})
    source_path = Path(item.get("path", ""))
    if not source_path.is_absolute() or not valid_sha(item.get("sha256")) or sha256(source_path) != item["sha256"]:
        raise ValueError("Missing/changed isolated source manifest")
    if source_path.stat().st_mtime > path.stat().st_mtime + .001:
        raise ValueError("Isolated source manifest is newer than its successful validation")
    pinned[source_path] = item["sha256"]
    source = json.loads(source_path.read_text())
    source_root = Path(source["root"]).resolve(strict=True)
    files = source.get("files", {})
    if not {"CMakeLists.txt", "gpu/CMakeLists.txt"} <= set(files):
        raise ValueError("Isolated source manifest must cover CPU and GPU source roots")
    for name, record in files.items():
        relative = Path(name)
        target = (source_root / relative).resolve(strict=True)
        if (relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(source_root)
                or not target.is_file() or not valid_sha(record.get("sha256"))
                or type(record.get("bytes")) is not int or target.stat().st_size != record["bytes"]
                or sha256(target) != record["sha256"]):
            raise ValueError("Invalid/changed isolated source file: " + name)
        pinned[target] = record["sha256"]
    return marker, pinned


def trial_plan(scenarios: list[str]) -> list[tuple[str, int, str]]:
    return [(stage, repeat, variant) for stage in scenarios for repeat in (1, 2) for variant in ("baseline", "candidate")]


def summarize(state: dict) -> dict:
    rows = []
    for stage in state["scenarios"]:
        selected = [run for run in state["runs"] if run["scenario"] == stage and run.get("status") == "accepted"]
        groups = {name: [r for r in selected if r["variant"] == name] for name in ("baseline", "candidate")}
        if not all(groups.values()):
            continue
        old = mean(r["total_solver_s"] for r in groups["baseline"])
        new = mean(r["total_solver_s"] for r in groups["candidate"])
        rows.append(dict(scenario=stage, baseline_samples_s=[r["total_solver_s"] for r in groups["baseline"]],
            candidate_samples_s=[r["total_solver_s"] for r in groups["candidate"]], baseline_mean_solver_s=old,
            candidate_mean_solver_s=new, mean_solver_speedup=old / new,
            paired_speedups=[r["baseline_solver_s"] / r["total_solver_s"] for r in groups["candidate"]],
            simulated_frames=groups["candidate"][0]["simulated_frames"],
            candidate_mean_solver_frame_ms=mean(r["average_solver_frame_s"] for r in groups["candidate"]) * 1000,
            complete=all(len(runs) == 2 for runs in groups.values())))
    expected = trial_plan(state["scenarios"])
    complete = ([(r["scenario"], r["repetition"], r["variant"]) for r in state["runs"]] == expected
                and all(r.get("status") == "accepted" and r.get("byte_identical_to_fresh_baseline") is True for r in state["runs"]))
    return dict(kind="clean GPU contact round 3", status=state["status"], rows=rows,
        short_trials_complete=complete, executable_hashes=state["executable_hashes"],
        candidate_flags=CANDIDATE_FLAGS, full=state.get("full"), error=state.get("error"), notes=NOTES)


def save_reports(root: Path, state: dict) -> None:
    report = summarize(state)
    write_json(root / "status.json", state);write_json(root / "summary.json", report)
    lines = ["GPU round 3 clean comparison", "Status: " + state["status"],
             "stage | baseline mean(s) | candidate mean(s) | clean short-stage speedup | repetitions complete"]
    for row in report["rows"]:
        lines.append(f"{row['scenario']} | {row['baseline_mean_solver_s']:.6f} | {row['candidate_mean_solver_s']:.6f} | {row['mean_solver_speedup']:.4f}x | {row['complete']}")
    if state.get("full"):
        lines.append("Full-run result: " + json.dumps(state["full"]))
    if state.get("error"):
        lines.append("Stopped: " + state["error"])
    lines += ["", *["- " + note for note in NOTES]]
    temporary = root / "summary.txt.tmp";temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(root / "summary.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--clean-directory", type=Path)
    parser.add_argument("--validation-marker", type=Path)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), action="append")
    parser.add_argument("--wait-pid", type=int, action="append", help="Wait for this PID/start identity to exit, even if temporarily idle")
    parser.add_argument("--full", action="store_true", help="After all 12 accepted short trials, validate all 240 continuous frames")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--full-timeout", type=float, default=43200)
    parser.add_argument("--wait-timeout", type=float, default=172800)
    parser.add_argument("--monitor-interval", type=float, default=10)
    parser.add_argument("--max-background-cores", type=float, default=.5)
    args = parser.parse_args()
    if args.self_test:
        self_test();return
    if any(getattr(args, key) is None for key in ("output", "candidate", "candidate_sha256", "baseline", "clean_directory", "validation_marker")):
        parser.error("--output, --candidate, --candidate-sha256, --baseline, --clean-directory, --validation-marker required")
    stages = args.scenario or list(SCENARIOS)
    if (not valid_sha(args.candidate_sha256) or args.candidate_sha256 == EXECUTABLE_SHA
            or len(stages) != len(set(stages)) or not 0 < args.monitor_interval <= 60
            or not 0 <= args.max_background_cores <= 2
            or any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.full_timeout, args.wait_timeout))):
        parser.error("Invalid candidate/hash, scenarios, duration, or load limit (maximum 2 external CPU cores)")
    if args.full and set(stages) != set(SCENARIOS):
        parser.error("--full requires all three short stages, two repetitions each")
    repo = Path(__file__).resolve().parent.parent
    verify_readme_command(repo)
    root = args.output.resolve();root.mkdir(exist_ok=True)
    if any(path.name != "supervisor.log" for path in root.iterdir()):
        raise ValueError("Use a fresh output directory; existing artifacts are never replaced")
    state = dict(kind="clean GPU contact round 3", status="preparing", pid=os.getpid(), directory=str(root),
        scenarios=stages, repetitions=2, runs=[], full_requested=args.full, instrumented=False,
        executable_hashes=dict(baseline=EXECUTABLE_SHA, candidate=args.candidate_sha256),
        feature_effective=FEATURES, schedule_effective=SCHEDULE, candidate_flags=CANDIDATE_FLAGS,
        monitoring_policy=MONITORING_NOTE, max_background_cores=args.max_background_cores)
    process = None;lock = None;current = None

    def save(status, **fields):
        state.update(status=status, updated_unix=time.time(), **fields);save_reports(root, state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        save("preparing")
        marker, pinned = validate_marker(args.validation_marker.resolve(strict=True), args.candidate_sha256)
        clean_dir = args.clean_directory.resolve(strict=True)
        clean, continuous, clean_pins = load_selected(clean_dir, repo);pinned.update(clean_pins)
        cpu_path = clean_dir / "cpu_baseline.json";cpu_baseline = checked_cpu_baseline(cpu_path)
        if cpu_baseline != clean["cpu_baseline"]:
            raise ValueError("Preserved CPU baseline differs from selected clean run")
        reference_path = Path(clean["reference_manifest"])
        originals = dict(baseline=args.baseline.resolve(strict=True), candidate=args.candidate.resolve(strict=True))
        if originals["baseline"] == originals["candidate"]:
            raise ValueError("Baseline and candidate paths must differ")
        pinned.update({repo / "README.md": sha256(repo / "README.md"), cpu_path: sha256(cpu_path),
                       repo / "build/3D_sim": sha256(repo / "build/3D_sim")})
        for variant, path in originals.items():
            pinned[path] = state["executable_hashes"][variant]
        checkpoints = {}
        for name in stages:
            frame = SCENARIOS[name]["restart_frame"]
            if frame is not None:
                path = clean_dir / "frames" / f"state_{frame:04d}.bin"
                entry = continuous["files"][path.name]
                if dict(bytes=path.stat().st_size, sha256=sha256(path)) != entry:
                    raise ValueError("Restart checkpoint differs from selected clean manifest")
                checkpoints[name] = dict(path=str(path), **entry);pinned[path] = entry["sha256"]
        cpu_sources, gpu_sources = cpu_source_snapshot(repo), gpu_source_snapshot(repo)
        cpus = physical_cpus()
        if cpus != list(range(64)):
            raise ValueError("Expected verified physical CPU affinity0–63")
        watched = []
        for pid in args.wait_pid or []:
            try:
                info = process_stat(pid);watched.append((pid, info["start"]))
            except (FileNotFoundError, ProcessLookupError):
                pass
        state.update(cpu_affinity=cpus, watched_processes=watched, validation_marker=marker,
                     cpu_source_hashes=cpu_sources, gpu_source_hashes=gpu_sources, checkpoints=checkpoints)

        def verify():
            verify_readme_command(repo)
            for path, digest in pinned.items():
                if sha256(path) != digest:
                    raise RuntimeError("Pinned input/source changed: " + str(path))
            if cpu_source_snapshot(repo) != cpu_sources or gpu_source_snapshot(repo) != gpu_sources:
                raise RuntimeError("Production CPU/GPU source snapshot changed")

        def wait_idle(label):
            quiet = 0;started = time.monotonic()
            while quiet < 3:
                before = sample();time.sleep(args.monitor_interval);after = sample()
                reasons, load = activity(before, after, watched, max_background_cores=args.max_background_cores)
                quiet = 0 if reasons else quiet + 1
                save("waiting_for_idle", current=label, idle_checks=quiet, reasons=reasons, external_cpu_cores=load)
                if time.monotonic() - started > args.wait_timeout:
                    raise RuntimeError("Timed out waiting for foreign work to finish")

        def execute(run, executable, scenario, candidate, directory, timeout, online=None):
            nonlocal process
            wait_idle(run["name"]);verify();directory.mkdir()
            out, log = directory / "frames", directory / "run.log"
            if run.get("checkpoint"):
                out.mkdir();item = run["checkpoint"];source = Path(item["path"])
                shutil.copy2(source, out / source.name)
                if sha256(out / source.name) != item["sha256"]:
                    raise ValueError("Restart copy hash mismatch")
            argv = segment_command(repo, out, executable, scenario, cpus)
            env = execution_environment(candidate)
            run.update(command=argv, outdir=str(out), log=str(log), status="running", instrumented=False,
                environment={k: v for k, v in env.items() if k.startswith(("OMP_", "IPC_GPU_"))}, background_observations=[])
            write_json(directory / "command.json", run)
            before = sample();started = last_verify = time.monotonic()
            with log.open("w") as stream:
                process = subprocess.Popen(argv, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                save("running", current=run["name"], child_pid=process.pid)
                while True:
                    try:
                        process.wait(timeout=args.monitor_interval);done = True
                    except subprocess.TimeoutExpired:
                        done = False
                    elapsed = time.monotonic() - started;after = sample(process.pid)
                    reasons, load = activity(before, after, watched, process.pid, max_background_cores=args.max_background_cores)
                    run["background_observations"].append(dict(elapsed_s=elapsed, external_cpu_cores=load, reasons=reasons,
                        system_iowait_cpu_s=max(0., (after["iowait"] - before["iowait"]) / HZ)))
                    before = after
                    if reasons:
                        raise RuntimeError("Foreign workload detected: " + "; ".join(reasons))
                    if done and process.returncode:
                        raise RuntimeError(f"GPU process returned {process.returncode}")
                    if online is not None:
                        through, count = safe_completed_frame(log, 240, process_complete=done)
                        verify_ready_frames(out, continuous, online, through)
                        run.update(completed_frames=count, verified_files=len(online), verified_through_frame=through)
                    if time.monotonic() - last_verify > 60:
                        verify();last_verify = time.monotonic()
                    if elapsed > timeout:
                        raise RuntimeError("Run timed out; partial results not accepted")
                    save("running", current=run["name"], elapsed_s=elapsed)
                    if done:
                        break
            run["observed_process_wall_s"] = time.monotonic() - started
            verify();process = None
            if run.get("checkpoint") and sha256(out / Path(run["checkpoint"]["path"]).name) != run["checkpoint"]["sha256"]:
                raise ValueError("Run modified its restart checkpoint")
            return log, out

        lock = (repo / "gpu/.example1_comparison.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB);verify()
        executables = {name: root / ("3D_sim_gpu_" + name) for name in originals}
        for name, source in originals.items():
            shutil.copy2(source, executables[name]);pinned[executables[name]] = state["executable_hashes"][name]
        state["pinned_inputs"] = {str(path): digest for path, digest in pinned.items()}
        write_json(root / "source_and_input_snapshot.json", dict(cpu=cpu_sources, gpu=gpu_sources, inputs=state["pinned_inputs"]))
        write_json(root / "selected_clean_report.json", clean);write_json(root / "cpu_baseline.json", cpu_baseline)
        state["hardware"] = dict(cpu=subprocess.check_output(["lscpu"], text=True),
            gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True, timeout=10),
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())
        (root / "runs").mkdir();fresh = {};first_reference = {}
        for stage, repeat, variant in trial_plan(stages):
            label = f"{len(state['runs']):03d}_{stage}_r{repeat}_{variant}"
            current = dict(name=label, scenario=stage, repetition=repeat, variant=variant, status="preparing",
                checkpoint=checkpoints.get(stage), executable_sha256=state["executable_hashes"][variant])
            state["runs"].append(current);print("Starting", label, flush=True)
            directory = root / "runs" / label
            log, out = execute(current, executables[variant], SCENARIOS[stage], variant == "candidate", directory, args.timeout)
            parsed = parse_segment(log, out, SCENARIOS[stage]);current.update(parsed)
            key = (stage, repeat)
            if variant == "baseline":
                fresh[key] = dict(hashes=parsed["output_hashes"], solver_s=parsed["total_solver_s"], name=label)
                first_reference.setdefault(stage, parsed["output_hashes"])
            if parsed["output_hashes"] != fresh[key]["hashes"] or parsed["output_hashes"] != first_reference[stage]:
                raise ValueError("Candidate or repeated baseline changed GEO/binary bytes: " + label)
            current.update(status="accepted", byte_identical_to_fresh_baseline=True,
                           baseline_name=fresh[key]["name"], baseline_solver_s=fresh[key]["solver_s"])
            write_json(directory / "result.json", current);save("between_trials", child_pid=None)
            print("Accepted", label, f"solver={current['total_solver_s']:.6f}s", flush=True)
        if not summarize(state)["short_trials_complete"]:
            raise RuntimeError("All expected short trials must pass before completion/full validation")
        if args.full:
            current = dict(name="full_validation", variant="candidate", status="preparing", checked_hashes={})
            state["full"] = current;directory = root / "full_validation"
            log, out = execute(current, executables["candidate"], dict(restart_frame=None, first_frame=1, last_frame=240),
                               True, directory, args.full_timeout, current["checked_hashes"])
            parsed = parse_trial(log, out, 240, "gpu", readme_command=True)
            parsed.update(instrumented=False, process_elapsed_s=current["observed_process_wall_s"])
            manifest = checkpoint_manifest(out, 240)
            if manifest["files"] != continuous["files"]:
                raise ValueError("Final continuous 482-file manifest differs from selected clean GPU run")
            manifest.update(kind="byte-validated clean GPU output manifest", reference_manifest_sha256=pinned[reference_path])
            write_json(directory / "checkpoint_manifest.json", manifest)
            meta = dict(command=current["command"], solver_command=current["command"][6:], environment=current["environment"],
                executable_sha256=args.candidate_sha256, readme_sha256=pinned[repo / "README.md"], cpu_affinity=cpus,
                cpu_executable_sha256=pinned[repo / "build/3D_sim"], instrumented=False,
                feature_effective=FEATURES, feature_overrides=FEATURES, schedule_defaults=SCHEDULE, schedule_overrides=SCHEDULE,
                candidate_flags=CANDIDATE_FLAGS, validation_marker=marker, **state["hardware"])
            report = build_report(directory, parsed, cpu_baseline, meta, reference_path, pinned[reference_path], clean)
            report["notes"].extend(NOTES)
            write_report(directory, report);write_json(directory / "command.json", meta)
            write_json(directory / "cpu_baseline.json", cpu_baseline)
            current.update(status="accepted", byte_identical_to_continuous_reference=True,
                total_solver_s=parsed["total_solver_s"], clean_speedup=report["clean_gpu_comparison"]["clean_solver_speedup"],
                clean_report=str(directory / "clean_report.json"), verified_geo_frames=241, verified_binary_checkpoints=241)
            write_json(directory / "status.json", current)
        verify();save("complete", current=None, child_pid=None)
    except BaseException as error:
        if process is not None:
            stop_our_process(process)
        if current is not None and current.get("status") != "accepted":
            current.update(status="rejected", error=f"{type(error).__name__}: {error}")
        save("stopped", child_pid=None, error=f"{type(error).__name__}: {error}")
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
        def test_environment(self):
            with patch.dict(os.environ, {"IPC_GPU_PROFILE_CSV": "bad", "IPC_GPU_CONTACT_QUEUE_CSV": "bad", **CANDIDATE_FLAGS}):
                baseline, candidate = execution_environment(False), execution_environment(True)
            self.assertTrue(all(name not in baseline for name in CANDIDATE_FLAGS))
            self.assertTrue(all(candidate[name] == value for name, value in CANDIDATE_FLAGS.items()))
            for env in (baseline, candidate):
                self.assertNotIn("IPC_GPU_PROFILE_CSV", env);self.assertNotIn("IPC_GPU_CONTACT_QUEUE_CSV", env)
                self.assertEqual(env["IPC_GPU_WARP_CONTACT_REDUCE"], "0");self.assertEqual(env["OMP_NUM_THREADS"], "64")

        def test_trials_and_summary(self):
            stages = list(SCENARIOS);plan = trial_plan(stages)
            self.assertEqual(len(plan), 12)
            state = dict(status="complete", scenarios=stages, runs=[], executable_hashes={})
            for stage, repeat, variant in plan:
                state["runs"].append(dict(scenario=stage, repetition=repeat, variant=variant,
                    status="accepted", byte_identical_to_fresh_baseline=True, total_solver_s=2. if variant == "baseline" else 1.,
                    average_solver_frame_s=1., baseline_solver_s=2., simulated_frames=1))
            report = summarize(state)
            self.assertTrue(report["short_trials_complete"])
            self.assertEqual([r["mean_solver_speedup"] for r in report["rows"]], [2., 2., 2.])
            state["runs"][-1]["status"] = "rejected"
            self.assertFalse(summarize(state)["short_trials_complete"])

        def test_marker(self):
            with tempfile.TemporaryDirectory(prefix="gpu_round3_test_") as name:
                root = Path(name);source = root / "source";source.mkdir();(source / "gpu").mkdir()
                files = {}
                for relative in ("CMakeLists.txt", "gpu/CMakeLists.txt"):
                    path = source / relative;path.write_text("source")
                    files[relative] = dict(bytes=6, sha256=sha256(path))
                manifest = root / "source.json";write_json(manifest, dict(root=str(source), files=files))
                log = root / "tests.log";log.write_text("successful validated test chain\n")
                marker = root / "validation_exit.json"
                data = dict(validation_chain_exit_code=0, candidate_sha256="b" * 64,
                    evidence={str(log): sha256(log)}, source_manifest=dict(path=str(manifest), sha256=sha256(manifest)))
                write_json(marker, data)
                self.assertEqual(validate_marker(marker, "b" * 64)[0], data)
                with self.assertRaises(ValueError):
                    validate_marker(marker, "c" * 64)
                data["validation_chain_exit_code"] = 1;write_json(marker, data)
                with self.assertRaises(ValueError):
                    validate_marker(marker, "b" * 64)
                data["validation_chain_exit_code"] = 0;write_json(marker, data)
                (source / "gpu/CMakeLists.txt").write_text("edited")
                with self.assertRaises(ValueError):
                    validate_marker(marker, "b" * 64)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
