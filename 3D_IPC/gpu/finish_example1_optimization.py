#!/usr/bin/env python3
"""Finish GPU validation/tuning and then a byte-checked clean 240-frame run.

Run on the server; this script makes no SSH calls. CPU sources/executable are
read-only. It never launches an original-GPU full run or claims that comparing
instrumented and clean full-run timings measures an optimization speedup.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
from statistics import mean
import subprocess
import sys
import time

from example1_comparison import verify_readme_command
from profile_example1_full import sha256
from tune_example1_contact import BASELINE_SHA256, clean_environment
from validate_optimized_example1_full import checked_cpu_baseline, read_reference
from wait_example1_comparison import EXPECTED, activity, process_stat, sample, stop_our_process

TESTS = tuple("gpu_resident_" + name for name in (
    "contact", "contact_rejection", "contact_pipeline", "contact_queue", "contact_schedule",
    "ccd", "exact_pool", "ratio_host", "ratio_cuda", "solver", "linear_pipeline",
    "rigid_ccd", "rigid_pipeline"))
WEIGHTS = dict(early=.03, middle=.15, late=.82)
CONFIGS = ("1024:1", "1024:2", "1024:4", "1024:8")


def cpu_source_snapshot(repo: Path) -> dict:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0")
    selected = [name for name in tracked if name and not name.startswith("gpu/")
                and (Path(name).suffix in (".cpp", ".cc", ".c", ".h", ".hpp", ".cuh", ".cu", ".inl", ".cmake")
                     or Path(name).name == "CMakeLists.txt")]
    if not selected:
        raise RuntimeError("No tracked CPU sources found")
    return {name: sha256(repo / name) if (repo / name).is_file() else None for name in sorted(selected)}


def gpu_source_snapshot(repo: Path) -> dict:
    return {str(path.relative_to(repo)): sha256(path) for path in sorted((repo / "gpu").iterdir())
            if path.is_file() and (path.suffix in (".cpp", ".h", ".cu", ".cuh", ".inc", ".py") or path.name == "CMakeLists.txt")}


def select_schedule(tuning: dict) -> dict:
    if tuning.get("status") != "complete" or tuning.get("repetitions") != 2:
        raise ValueError("Tuning must complete two repetitions before schedule selection")
    if set(tuning.get("scenarios", [])) != set(WEIGHTS):
        raise ValueError("Tuning must include early, middle and late stages")
    names = {"w1024_t" + str(threads) for threads in (1, 2, 4, 8)}
    if {config["name"] for config in tuning.get("configs", [])} != names:
        raise ValueError("Unexpected tuning configurations")
    runs = tuning.get("runs", [])
    if len(runs) != 30 or any(run.get("status") != "accepted" or not run.get("byte_identical_to_baseline") for run in runs):
        raise ValueError("Every one of the 30 baseline/candidate trials must be byte-identical and accepted")
    timings = {}
    for stage in WEIGHTS:
        timings[stage] = {}
        for variant in {"baseline", *names}:
            matching = [run for run in runs if run["scenario"] == stage and run["variant"] == variant]
            if len(matching) != 2 or {run["repetition"] for run in matching} != {1, 2}:
                raise ValueError("Missing, duplicate or incomplete tuning repetition")
            values = [run["total_solver_s"] for run in matching]
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("Tuning solver times must be finite and positive")
            timings[stage][variant] = dict(mean_solver_s=mean(values), min_solver_s=min(values), max_solver_s=max(values), samples_s=values)
    scores = []
    for variant in sorted(names):
        ratios = {stage: timings[stage][variant]["mean_solver_s"] / timings[stage]["baseline"]["mean_solver_s"] for stage in WEIGHTS}
        scores.append(dict(variant=variant, weighted_normalized_time=sum(WEIGHTS[stage] * ratios[stage] for stage in WEIGHTS),
                           relative_times=ratios, late_speedup=1 / ratios["late"]))
    scores.sort(key=lambda value: (value["weighted_normalized_time"], value["variant"]))
    chosen, preferred = scores[0], next(row for row in scores if row["variant"] == "w1024_t1")
    preferred_within_tolerance = preferred["weighted_normalized_time"] <= chosen["weighted_normalized_time"] * 1.02
    if preferred_within_tolerance:
        chosen = preferred
    passed = chosen["late_speedup"] >= 1.02
    return dict(passed=passed, chosen_variant=chosen["variant"], scores=scores, timings=timings,
                selected_config=dict(small_workers=1024, small_threads=int(chosen["variant"].split("_t")[1]),
                                     large_workers=256, large_threads=4),
                chosen_late_speedup=chosen["late_speedup"], required_late_speedup=1.02,
                preferred_one_thread_within_two_percent=preferred_within_tolerance, weights=WEIGHTS,
                note="The score is a declared stage-weighted tuning heuristic, NOT a predicted or measured full-simulation speedup. Mean solver-time ratios use early=.03, middle=.15, late=.82. Prefer 1024:1 when within 2% of the minimum score. Full-run original GPU timing is instrumented and must not be treated as a clean speedup baseline.")


def matching_build(pid: int, repo: Path) -> dict | None:
    directory = Path("/proc") / str(pid)
    try:
        info = process_stat(pid)
        argv = [word.decode() for word in (directory / "cmdline").read_bytes().split(b"\0") if word]
        if info["state"] == "Z":
            return None
        if directory.stat().st_uid != os.getuid() or not argv or Path(argv[0]).name != "cmake" or "--build" not in argv:
            raise ValueError("Wait PID is not this user's cmake --build process")
        index = argv.index("--build")
        if index + 1 >= len(argv):
            raise ValueError("Wait PID has no CMake build directory")
        build = Path(argv[index + 1])
        if not build.is_absolute():
            cwd = (directory / "cwd").resolve()
            # CMake changes its cwd to the build tree after resolving the
            # original --build argument, but /proc retains that original argv.
            expected = (repo / "gpu/build").resolve()
            if cwd == expected and build == Path("gpu/build"):
                build = expected
            else:
                build = cwd / build
        if build.resolve() != (repo / "gpu/build").resolve():
            raise ValueError("Wait PID targets a different build directory")
        return dict(pid=pid, start_tick=info["start"], command=argv)
    except (FileNotFoundError, ProcessLookupError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-directory", type=Path)
    parser.add_argument("--cpu-baseline", type=Path)
    parser.add_argument("--baseline-executable", type=Path)
    parser.add_argument("--skip-build", action="store_true", help="Only when all listed targets were already rebuilt; CTests remain mandatory")
    parser.add_argument("--wait-build-pid", type=int, help="Read-only wait for a verified existing GPU CMake build before the final build")
    parser.add_argument("--build-wait-timeout", type=float, default=3600)
    parser.add_argument("--build-jobs", type=int, default=2)
    parser.add_argument("--build-timeout", type=float, default=14400)
    parser.add_argument("--test-timeout", type=float, default=1800, help="Per-CTest time limit")
    parser.add_argument("--full-timeout", type=float, default=43200)
    parser.add_argument("--max-background-cores", type=float, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test();return
    if args.output is None or args.reference_directory is None:
        parser.error("--output and --reference-directory are required")
    if args.skip_build and args.wait_build_pid is not None:
        parser.error("--wait-build-pid requires the final build; do not combine it with --skip-build")
    if (args.build_jobs < 1 or (args.wait_build_pid is not None and args.wait_build_pid < 1)
            or any(not 0 < value < float("inf") for value in (args.build_wait_timeout, args.build_timeout, args.test_timeout, args.full_timeout))
            or not 0 <= args.max_background_cores < float("inf")):
        parser.error("Invalid job count, PID, timeout or background ceiling")
    repo = Path(__file__).resolve().parent.parent
    root = args.output.resolve();root.mkdir(exist_ok=True)
    if any(path.name != "supervisor.log" for path in root.iterdir()):
        raise RuntimeError("Use a fresh optimization directory; existing results are never replaced")
    reference = args.reference_directory.resolve(strict=True)
    manifest_path = reference / "checkpoint_manifest.json"
    read_reference(manifest_path, 240)
    cpu_baseline_path = (args.cpu_baseline or reference / "cpu_baseline.json").resolve(strict=True)
    baseline = checked_cpu_baseline(cpu_baseline_path)
    baseline_executable = (args.baseline_executable or repo / "gpu/profile_all_frames_20261010_lvvtq2dn/3D_sim_gpu_baseline").resolve(strict=True)
    candidate = root / "3D_sim_gpu_candidate"
    state = dict(kind="GPU optimization validation and full-run orchestration", status="preparing", pid=os.getpid(),
                 phase="preparing", directory=str(root), reference_directory=str(reference),
                 full_directory=str(root / "full_validation"), tuning_directory=str(root / "tuning"),
                 candidate=str(candidate), cpu_total_solver_s=baseline["cpu_total_solver_s"],
                 cpu_average_solver_frame_ms=baseline["cpu_average_solver_frame_ms"],
                 cpu_total_simulation_s=baseline["cpu_total_simulation_s"],
                 baseline_executable=str(baseline_executable), tests=list(TESTS),
                 no_original_gpu_full_run=True, commands=[])

    def save(status: str, **fields):
        state.update(status=status, updated_unix=time.time(), **fields)
        temporary = root / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "status.json")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    owner_lock = (repo / "gpu/.example1_optimization.lock").open("a+")
    comparison_lock = None
    process = None
    process_is_supervisor = False
    sources = None
    inputs = None

    def verify_unchanged():
        verify_readme_command(repo)
        if sources is not None and cpu_source_snapshot(repo) != sources:
            raise RuntimeError("CPU sources changed; refusing to continue optimization validation")
        for path, digest in (inputs or {}).items():
            if sha256(Path(path)) != digest:
                raise RuntimeError("Pinned input changed: " + path)

    def wait_idle(phase):
        clean = 0
        while clean < 3:
            before = sample();time.sleep(10);after = sample()
            reasons, load = activity(before, after, [], max_background_cores=args.max_background_cores)
            clean = 0 if reasons else clean + 1
            save("waiting_for_idle", phase=phase, reasons=reasons, idle_checks=clean, external_cpu_cores=load)

    def stop_owned_child():
        nonlocal process
        if process is None or process.poll() is not None:
            return
        if process_is_supervisor:
            # The child supervisor owns a separate simulation session. Signal
            # only that known child first so its finally block stops its run.
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
                return
            except subprocess.TimeoutExpired:
                pass
        stop_our_process(process)

    def run_child(argv, phase, timeout, *, monitored, child_status=None, env=None):
        nonlocal process, process_is_supervisor
        verify_unchanged()
        log = root / (phase + ".log")
        state["commands"].append(dict(phase=phase, command=argv, log=str(log)))
        print("Starting phase:", phase, flush=True)
        before = sample() if monitored else None
        started = time.monotonic()
        process_is_supervisor = not monitored
        with log.open("w") as stream:
            process = subprocess.Popen(argv, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            save("running", phase=phase, child_pid=process.pid, child_status=None)
            while True:
                try:
                    process.wait(timeout=10)
                    done = True
                except subprocess.TimeoutExpired:
                    done = False
                elapsed = time.monotonic() - started
                if monitored:
                    after = sample(process.pid)
                    reasons, load = activity(before, after, [], process.pid, max_background_cores=args.max_background_cores)
                    before = after
                    if reasons:
                        raise RuntimeError("External workload during " + phase + ": " + "; ".join(reasons))
                progress = None
                if child_status is not None and child_status.is_file():
                    child = json.loads(child_status.read_text())
                    progress = {key: child[key] for key in ("status", "current", "completed_frames", "verified_through_frame", "error") if key in child}
                save("running", phase=phase, elapsed_s=elapsed, child_status=progress)
                if done:
                    break
                if elapsed > timeout:
                    raise RuntimeError("Phase time limit reached: " + phase)
        if process.returncode:
            raise RuntimeError(f"{phase} returned {process.returncode}; see {log}")
        process = None
        verify_unchanged()
        save("between_phases", phase=phase, child_pid=None)
        print("Finished phase:", phase, flush=True)

    try:
        fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.wait_build_pid is not None:
            original = matching_build(args.wait_build_pid, repo)
            state["wait_build"] = original or dict(pid=args.wait_build_pid, already_finished_or_absent=True)
            started = time.monotonic()
            while original is not None:
                try:
                    current = process_stat(args.wait_build_pid)
                except (FileNotFoundError, ProcessLookupError):
                    break
                if current["start"] != original["start_tick"] or current["state"] == "Z":
                    break
                elapsed = time.monotonic() - started
                save("waiting_for_build", phase="waiting_for_build", wait_build_elapsed_s=elapsed)
                if elapsed > args.build_wait_timeout:
                    raise RuntimeError("Read-only wait for the existing build exceeded its limit")
                time.sleep(5)
        verify_readme_command(repo)
        cpu_executable = repo / EXPECTED["cpu"][0]
        if sha256(cpu_executable) != EXPECTED["cpu"][1] or sha256(baseline_executable) != BASELINE_SHA256:
            raise RuntimeError("CPU or immutable original GPU executable hash changed")
        sources = cpu_source_snapshot(repo)
        inputs = {str(cpu_executable): EXPECTED["cpu"][1], str(baseline_executable): BASELINE_SHA256,
                  str(manifest_path): sha256(manifest_path), str(cpu_baseline_path): sha256(cpu_baseline_path),
                  str(repo / "README.md"): sha256(repo / "README.md")}
        (root / "cpu_sources_before.json").write_text(json.dumps(sources, indent=2) + "\n")
        (root / "pinned_inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
        (root / "gpu_sources_for_build.json").write_text(json.dumps(gpu_source_snapshot(repo), indent=2) + "\n")
        comparison_lock = (repo / "gpu/.example1_comparison.lock").open("a+")
        fcntl.flock(comparison_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        test_env = clean_environment(None)
        test_env.update(OMP_NUM_THREADS="4", OMP_WAIT_POLICY="PASSIVE")
        if not args.skip_build:
            wait_idle("build")
            build = ["cmake", "--build", "gpu/build", "--target", "3D_sim_gpu_resident", *(name + "_test" for name in TESTS), "-j", str(args.build_jobs)]
            run_child(build, "build", args.build_timeout, monitored=True, env=test_env)
        else:
            state["build_explicitly_skipped"] = True
        verify_unchanged()
        listing = json.loads(subprocess.check_output(["ctest", "--test-dir", "gpu/build", "--show-only=json-v1"], cwd=repo, text=True))
        registered = {test["name"] for test in listing["tests"]}
        if not set(TESTS) <= registered:
            raise RuntimeError("Required GPU regression tests are not registered: " + str(sorted(set(TESTS) - registered)))
        wait_idle("tests")
        # CTest uses CMake's regular-expression grammar, not Python's; its
        # grouping syntax has no non-capturing (?:...) form.
        pattern = "^(" + "|".join(map(re.escape, TESTS)) + ")$"
        tests = ["ctest", "--test-dir", "gpu/build", "--output-on-failure", "--no-tests=error", "--timeout", str(args.test_timeout),
                 "-R", pattern, "--output-junit", str(root / "tests.xml")]
        run_child(tests, "tests", args.test_timeout * len(TESTS) + 120, monitored=True, env=test_env)
        copied_from = repo / "gpu/build/3D_sim_gpu_resident"
        shutil.copy2(copied_from, candidate)
        candidate_hash = sha256(candidate)
        if candidate_hash != sha256(copied_from):
            raise RuntimeError("Candidate executable copy hash mismatch")
        inputs[str(candidate)] = candidate_hash
        state["candidate_sha256"] = candidate_hash
        (root / "pinned_inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
        comparison_lock.close();comparison_lock = None
        tuning = [sys.executable, "-u", "gpu/tune_example1_contact.py", "--output", str(root / "tuning"),
                  "--candidate", str(candidate), "--candidate-sha256", candidate_hash,
                  "--baseline", str(baseline_executable), "--baseline-sha256", BASELINE_SHA256,
                  "--checkpoint-root", str(reference / "frames"), "--checkpoint-manifest", str(manifest_path),
                  "--repetitions", "2", "--monitor-interval", "10", "--max-background-cores", str(args.max_background_cores)]
        for scenario in ("late", "middle", "early"):
            tuning.extend(("--scenario", scenario))
        for config in CONFIGS:
            tuning.extend(("--config", config))
        run_child(tuning, "tuning", 24 * 3600, monitored=False, child_status=root / "tuning/status.json")
        selection = select_schedule(json.loads((root / "tuning/status.json").read_text()))
        (root / "tuning_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        state["selected_config"] = selection["selected_config"]
        if not selection["passed"]:
            raise RuntimeError(f"Selected schedule's late-stage improvement is only {selection['chosen_late_speedup']:.6f}x; need at least 1.02x before a full run")
        full = [sys.executable, "-u", "gpu/validate_optimized_example1_full.py", "--executable", str(candidate), "--sha256", candidate_hash,
                "--reference-manifest", str(manifest_path), "--cpu-baseline", str(cpu_baseline_path),
                "--output", str(root / "full_validation"), "--frames", "240", "--timeout", str(args.full_timeout),
                "--monitor-interval", "10", "--max-background-cores", str(args.max_background_cores)]
        for key, value in selection["selected_config"].items():
            full.extend(("--" + key.replace("_", "-"), str(value)))
        run_child(full, "full_validation", args.full_timeout + 24 * 3600, monitored=False,
                  child_status=root / "full_validation/status.json")
        final_state = json.loads((root / "full_validation/status.json").read_text())
        final_report = json.loads((root / "full_validation/clean_report.json").read_text())
        if (final_state.get("status") != "complete" or not final_state.get("byte_identical_to_reference")
                or final_state.get("verified_files") != 482 or final_report.get("frames") != 240):
            raise RuntimeError("Full child did not confirm all 241 GEO and 241 binary checkpoint files")
        verify_unchanged()
        (root / "cpu_sources_after.json").write_text(json.dumps(cpu_source_snapshot(repo), indent=2) + "\n")
        outcome = dict(kind="validated GPU optimization", candidate=str(candidate), candidate_sha256=candidate_hash,
                       selected_config=selection["selected_config"], tuning_selection=selection,
                       cpu_sources_unchanged=True, cpu_executable_unchanged=True,
                       full_directory=str(root / "full_validation"), full_report=final_report,
                       note="The optimization is byte-identical to the original GPU reference over all 240 frames. Only the cold-restart trials provide clean old/new GPU speedups; instrumented-original versus clean-optimized full-run timing is explicitly not a clean speedup measurement.")
        (root / "final_report.json").write_text(json.dumps(outcome, indent=2) + "\n")
        metrics = final_report["metrics"]
        lines = ["GPU optimization validation complete", "All 241 GEO and 241 binary states match the original GPU reference.",
                 "CPU sources and executable unchanged.", f"Selected contact schedule: {selection['selected_config']}",
                 f"Late cold-restart clean speedup: {selection['chosen_late_speedup']:.6f}x", "",
                 f"GPU clean total solver: {metrics['total_solver_s']:.6f} s; mean solver/frame: {metrics['average_frame_s']*1000:.6f} ms",
                 f"GPU clean total simulation: {metrics['total_simulation_s']:.6f} s",
                 f"Preserved CPU total solver: {baseline['cpu_total_solver_s']:.6f} s; mean solver/frame: {baseline['cpu_average_solver_frame_ms']:.6f} ms",
                 f"Preserved CPU total simulation: {baseline['cpu_total_simulation_s']:.6f} s", "", outcome["note"]]
        (root / "final_report.txt").write_text("\n".join(lines) + "\n")
        save("complete", phase="complete", child_pid=None, byte_identical_to_reference=True, verified_geo_frames=241,
             verified_binary_checkpoints=241, total_solver_s=metrics["total_solver_s"],
             average_solver_frame_ms=metrics["average_frame_s"] * 1000, final_report=str(root / "final_report.txt"))
        print("Completed GPU optimization validation:", root, flush=True)
    except BaseException as error:
        stop_owned_child()
        save("stopped", child_pid=None, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stop_owned_child()
        if comparison_lock is not None:
            comparison_lock.close()
        owner_lock.close()


def self_test():
    import tempfile
    from unittest.mock import patch
    configurations = [dict(name="w1024_t" + str(threads)) for threads in (1, 2, 4, 8)]
    state = dict(status="complete", repetitions=2, scenarios=list(WEIGHTS), configs=configurations, runs=[])
    for stage in WEIGHTS:
        for repetition in (1, 2):
            for threads in (0, 1, 2, 4, 8):
                state["runs"].append(dict(scenario=stage, variant="baseline" if not threads else f"w1024_t{threads}",
                    repetition=repetition, status="accepted", byte_identical_to_baseline=True,
                    total_solver_s=10. if not threads else {1:5.05, 2:5., 4:6., 8:8.}[threads]))
    selection = select_schedule(state)
    assert selection["passed"] and selection["chosen_variant"] == "w1024_t1"
    assert selection["preferred_one_thread_within_two_percent"]
    for run in state["runs"]:
        if run["variant"] == "w1024_t1":
            run["total_solver_s"] = 5.2
    assert select_schedule(state)["chosen_variant"] == "w1024_t2"
    for run in state["runs"]:
        if run["variant"] != "baseline":
            run["total_solver_s"] = 9.99
    assert not select_schedule(state)["passed"]
    state["runs"][0]["byte_identical_to_baseline"] = False
    try:
        select_schedule(state)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-identical trial was accepted for selection")
    with tempfile.TemporaryDirectory(prefix="gpu_finish_orchestrator_test_") as directory:
        root = Path(directory)
        (root / "cpu.cpp").write_text("original CPU source")
        (root / "gpu").mkdir();(root / "gpu/kernel.cu").write_text("GPU source")
        with patch(__name__ + ".subprocess.check_output", return_value=b"cpu.cpp\0gpu/kernel.cu\0README.md\0"):
            snapshot = cpu_source_snapshot(root)
        assert set(snapshot) == {"cpu.cpp"}
        assert set(gpu_source_snapshot(root)) == {"gpu/kernel.cu"}
    assert "gpu_resident_global_guess" not in TESTS
    assert {"gpu_resident_ratio_cuda", "gpu_resident_contact_schedule", "gpu_resident_exact_pool"} <= set(TESTS)
    print("GPU optimization orchestration self-test PASS")


if __name__ == "__main__":
    main()
