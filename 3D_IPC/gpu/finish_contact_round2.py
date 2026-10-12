#!/usr/bin/env python3
"""Gate finished CUDA regressions, then run clean contact round-2 validation.

This server-local supervisor NEVER builds or runs tests. It requires completed,
hash-associated test evidence, launches the short validation harness, and only
with --full and a passing performance gate starts the clean 240-frame validator.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
from statistics import mean
import subprocess
import sys
import time

from example1_comparison import verify_readme_command
from finish_example1_optimization import cpu_source_snapshot, gpu_source_snapshot
from profile_optimized_example1_stages import OPTIMIZED_SHA256
from tune_example1_contact import clean_environment, sha256
from validate_contact_round2 import completed_short_trials
from validate_optimized_example1_full import checked_cpu_baseline, load_clean_comparison, read_reference, valid_sha
from wait_example1_comparison import stop_our_process

CANDIDATE_SHA256 = "f6fa53ee93e9c000c462959f25e8a8f9df7a43bc5b9e59e678da0b0d8ceec2dd"
GATE_LOGS = {
    "ctest": "ctest_scoped.log",
    "global_guess": "global_guess_quick.log",
    "memcheck": "cooperative_memcheck.log",
    "initcheck": "worklist_initcheck.log",
    "racecheck": "cooperative_racecheck.log",
}
QUICK_SUCCESS = "CUDA global guess: 540 PT/EE fixtures; max TOI error 0; whole-guess checks passed"


def validate_test_evidence(root: Path, candidate_hash: str, not_before: float) -> dict:
    marker_path = root / "validation_exit.json"
    marker = json.loads(marker_path.read_text())
    if (type(marker.get("validation_chain_exit_code")) is not int
            or marker["validation_chain_exit_code"] != 0 or marker.get("candidate_sha256") != candidate_hash):
        raise ValueError("Validation exit marker is unsuccessful or belongs to another candidate")
    texts, evidence = {}, {}
    for name, filename in GATE_LOGS.items():
        path = root / filename
        if not path.is_file() or path.stat().st_mtime < not_before:
            raise ValueError("Missing/stale validation log: " + filename)
        text = path.read_text()
        if not text.strip():
            raise ValueError("Empty validation log: " + filename)
        texts[name] = text
        evidence[filename] = dict(sha256=sha256(path), bytes=path.stat().st_size,
                                 modified_unix=path.stat().st_mtime)
    ctest = texts["ctest"]
    passed_records = re.findall(r"^\s*(\d+)/22\s+Test\s+#\d+:\s+.+?\s+Passed\s+[\d.]+\s+sec\s*$", ctest, re.M)
    if (len(re.findall(r"100% tests passed(?:, 0 tests failed)? out of 22\b", ctest)) != 1
            or len(passed_records) != 22 or {int(value) for value in passed_records} != set(range(1, 23))
            or re.search(r"\*\*\*(?:Failed|Timeout|Exception)|The following tests FAILED|No tests were found", ctest, re.I)):
        raise ValueError("Scoped CTest log does not prove all 22 selected tests passed")
    if texts["global_guess"].count(QUICK_SUCCESS) != 1:
        raise ValueError("Quick global-guess log does not prove 540 exact matching fixtures passed")
    for name in ("memcheck", "initcheck"):
        summaries = re.findall(r"ERROR SUMMARY:\s*(\d+)\s+errors?\b", texts[name])
        if summaries != ["0"]:
            raise ValueError(name + " log is partial, repeated, or reports errors")
    race = re.findall(r"RACECHECK SUMMARY:\s*(\d+)\s+hazards displayed\s*\((\d+) errors,\s*(\d+) warnings\)", texts["racecheck"])
    if race != [("0", "0", "0")]:
        raise ValueError("Racecheck log is partial, repeated, or reports hazards")
    if marker_path.stat().st_mtime < max(row["modified_unix"] for row in evidence.values()):
        raise ValueError("Exit marker predates completed validation logs")
    evidence[marker_path.name] = dict(sha256=sha256(marker_path), marker=marker)
    return dict(passed=True, files=evidence, scoped_ctest_tests=22, global_guess_quick_fixtures=540,
                full_global_guess_stress_test_completed=False,
                note="22 scoped CTests plus the 540-case quick global-guess equivalence test and three sanitizer checks; the interrupted full global-guess stress test is NOT counted as passed.")


def performance_selection(state: dict, candidate_hash: str) -> dict:
    if (state.get("status") != "complete" or state.get("repetitions") != 2
            or state.get("scenarios") != ["late", "middle", "early"]
            or state.get("executable_hashes") != dict(baseline=OPTIMIZED_SHA256, candidate=candidate_hash)
            or not completed_short_trials(state)):
        raise ValueError("All 30 two-repetition clean byte-checked trials must finish before selection")
    rows = {}
    for stage in ("early", "middle", "late"):
        values = {}
        for variant in ("baseline", "all_off", "pairs_only", "pairs_warp", "all_three"):
            matching = [run for run in state["runs"] if run["scenario"] == stage and run["variant"] == variant]
            times = [run.get("total_solver_s") for run in matching]
            if (len(times) != 2 or {run["repetition"] for run in matching} != {1, 2}
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in times)):
                raise ValueError("Missing or invalid solver timing for " + stage + "/" + variant)
            values[variant] = dict(samples_s=times, mean_solver_s=mean(times))
        all_three = values["all_three"]["mean_solver_s"]
        warp = values["pairs_warp"]["mean_solver_s"]
        baseline = values["baseline"]["mean_solver_s"]
        rows[stage] = dict(timings=values, all_three_over_pairs_warp=all_three / warp,
                           baseline_over_all_three=baseline / all_three,
                           within_two_percent_of_pairs_warp=all_three <= warp * 1.02,
                           faster_than_baseline=all_three < baseline)
    passed = all(rows[stage]["within_two_percent_of_pairs_warp"] and rows[stage]["faster_than_baseline"]
                 for stage in ("middle", "late"))
    return dict(passed=passed, selected_variant="all_three" if passed else None, stages=rows,
                rule="For BOTH middle and late: mean all-three time <= 1.02 * mean pairs+warp time AND strictly less than previous-baseline mean time. Early results remain reported, not part of this gate.",
                note="This gate chooses whether to test all three features continuously; short cold-restart timings do not establish a full-simulation speedup. Failure requires user/root selection; no different variant is chosen silently.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--optimization-directory", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-sha", default=CANDIDATE_SHA256)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--cpu-baseline", type=Path)
    parser.add_argument("--full", action="store_true", help="Explicit permission to start full240 after short tests and performance selection pass")
    parser.add_argument("--lock-timeout", type=float, default=14400)
    parser.add_argument("--short-timeout", type=float, default=43200)
    parser.add_argument("--full-timeout", type=float, default=43200)
    parser.add_argument("--monitor-interval", type=float, default=5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test();return
    if any(getattr(args, field) is None for field in ("root", "optimization_directory", "candidate", "reference_manifest", "cpu_baseline")):
        parser.error("--root, --optimization-directory, --candidate, --reference-manifest and --cpu-baseline are required")
    if (not valid_sha(args.candidate_sha) or not 0 < args.monitor_interval <= 60
            or any(not math.isfinite(value) or value <= 0 for value in (args.lock_timeout, args.short_timeout, args.full_timeout))):
        parser.error("Invalid hash, monitor interval or timeout")
    repo = Path(__file__).resolve().parent.parent
    root = args.root.resolve(strict=True)
    if (root / "round2_driver_status.json").exists() or (root / "benchmarks").exists() or (root / "full_validation").exists():
        raise RuntimeError("Driver status or benchmark/full output already exists; use a fresh validation root")
    candidate = args.candidate.resolve(strict=True)
    optimization = args.optimization_directory.resolve(strict=True)
    reference = args.reference_manifest.resolve(strict=True)
    cpu_path = args.cpu_baseline.resolve(strict=True)
    previous_clean_path = optimization / "full_validation/clean_report.json"
    verify_readme_command(repo);read_reference(reference, 240);checked_cpu_baseline(cpu_path)
    previous_clean = load_clean_comparison(previous_clean_path, 240, sha256(repo / "README.md"), sha256(reference))
    if previous_clean["command_metadata"]["executable_sha256"] != OPTIMIZED_SHA256:
        raise ValueError("Previous clean full-run report is not from the pinned previous GPU binary")
    cpu_sources, gpu_sources = cpu_source_snapshot(repo), gpu_source_snapshot(repo)
    pinned = {str(path): sha256(path) for path in (candidate, reference, cpu_path, previous_clean_path,
        optimization / "3D_sim_gpu_candidate", optimization / "tuning/status.json", repo / "README.md", repo / "build/3D_sim")}
    if pinned[str(candidate)] != args.candidate_sha or pinned[str(optimization / "3D_sim_gpu_candidate")] != OPTIMIZED_SHA256:
        raise ValueError("Candidate or previous GPU executable SHA256 does not match")
    candidate_mtime = candidate.stat().st_mtime
    state = dict(kind="contact round2 durable validation supervisor", status="preparing", pid=os.getpid(),
                 directory=str(root), candidate=str(candidate), candidate_sha256=args.candidate_sha,
                 full_authorized=args.full, pinned_inputs=pinned, cpu_source_hashes=cpu_sources,
                 gpu_source_hashes=gpu_sources, commands=[], child_pid=None,
                 notes=["No source edits, builds, CPU runs, or test execution occur in this supervisor.",
                        "Full validator clears inherited feature environment variables; all-three features default ON in the pinned candidate."])

    def save(status, **fields):
        state.update(status=status, updated_unix=time.time(), **fields)
        path = root / "round2_driver_status.json.tmp"
        path.write_text(json.dumps(state, indent=2) + "\n");path.replace(root / "round2_driver_status.json")

    def verify():
        verify_readme_command(repo)
        for path, digest in pinned.items():
            if sha256(Path(path)) != digest:
                raise RuntimeError("Pinned input changed: " + path)
        if cpu_source_snapshot(repo) != cpu_sources or gpu_source_snapshot(repo) != gpu_sources:
            raise RuntimeError("CPU/GPU sources changed during validation")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    owner_lock = (repo / "gpu/.contact_round2_supervisor.lock").open("a+")
    comparison_lock = None
    process = None

    def stop_owned_child():
        nonlocal process
        if process is None or process.poll() is not None:
            return
        # Child is a supervisor with its own independently owned simulation
        # session. Give its finally block time to stop that simulation first.
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            stop_our_process(process)

    def run_child(argv, phase, timeout, status_path):
        nonlocal process
        verify();log = root / ("round2_" + phase + ".log")
        state["commands"].append(dict(phase=phase, command=argv, log=str(log)))
        started = time.monotonic()
        with log.open("x") as stream:
            process = subprocess.Popen(argv, cwd=repo, env=clean_environment(None), stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            save("running", phase=phase, child_pid=process.pid, child_status=None)
            while True:
                try:
                    process.wait(timeout=args.monitor_interval);done = True
                except subprocess.TimeoutExpired:
                    done = False
                elapsed = time.monotonic() - started
                progress = None
                if status_path.is_file():
                    child = json.loads(status_path.read_text())
                    progress = {key: child[key] for key in ("status", "current", "completed_frames",
                        "verified_through_frame", "verified_files", "error", "child_pid") if key in child}
                save("running", phase=phase, elapsed_s=elapsed, child_status=progress)
                if done:
                    break
                if elapsed > timeout:
                    raise RuntimeError(phase + " supervisor timeout")
        if process.returncode:
            raise RuntimeError(f"{phase} returned {process.returncode}; see {log}")
        process = None;verify();save("between_phases", child_pid=None)

    try:
        fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        comparison_lock = (repo / "gpu/.example1_comparison.lock").open("a+")
        started = time.monotonic()
        while True:
            try:
                fcntl.flock(comparison_lock, fcntl.LOCK_EX | fcntl.LOCK_NB);break
            except BlockingIOError:
                elapsed = time.monotonic() - started
                save("waiting_for_validation_lock", elapsed_s=elapsed)
                if elapsed > args.lock_timeout:
                    raise RuntimeError("Validation lock wait timed out; no benchmark launched")
                time.sleep(args.monitor_interval)
        verify()
        state["test_gate"] = validate_test_evidence(root, args.candidate_sha, candidate_mtime)
        for filename, entry in state["test_gate"]["files"].items():
            pinned[str(root / filename)] = entry["sha256"]
        save("tests_verified", phase="test_gate")
        comparison_lock.close();comparison_lock = None
        short = [sys.executable, "-u", "gpu/validate_contact_round2.py", "--output", str(root / "benchmarks"),
                 "--optimization-directory", str(optimization), "--candidate", str(candidate),
                 "--candidate-sha256", args.candidate_sha, "--repetitions", "2",
                 "--monitor-interval", str(args.monitor_interval), "--max-background-cores", "2"]
        for name in ("late", "middle", "early"):
            short.extend(("--scenario", name))
        run_child(short, "benchmarks", args.short_timeout, root / "benchmarks/status.json")
        benchmark_state = json.loads((root / "benchmarks/status.json").read_text())
        selection = performance_selection(benchmark_state, args.candidate_sha)
        state["selection"] = selection
        (root / "round2_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        if not selection["passed"]:
            save("selection_required", phase="selection", child_pid=None,
                 message="All short bytes passed, but all-three did not satisfy the performance gate. No full run or alternate configuration was launched.")
            return
        if not args.full:
            save("complete_short_only", phase="complete", child_pid=None,
                 message="Short validation and performance gate passed; full run was not authorized by --full.")
            return
        # Use the frozen candidate copy produced by the successful short
        # harness, not a mutable build-tree executable for the long run.
        immutable = root / "benchmarks/3D_sim_gpu_candidate"
        if sha256(immutable) != args.candidate_sha:
            raise ValueError("Frozen short-run candidate hash mismatch")
        pinned[str(immutable)] = args.candidate_sha
        full = [sys.executable, "-u", "gpu/validate_optimized_example1_full.py", "--output", str(root / "full_validation"),
            "--executable", str(immutable), "--sha256", args.candidate_sha,
            "--reference-manifest", str(reference), "--cpu-baseline", str(cpu_path),
            "--baseline-clean-report", str(previous_clean_path), "--frames", "240", "--timeout", str(args.full_timeout),
            "--monitor-interval", str(args.monitor_interval), "--max-background-cores", "2",
            "--small-workers", "1024", "--small-threads", "1", "--large-workers", "256", "--large-threads", "4"]
        run_child(full, "full_validation", args.full_timeout + args.lock_timeout, root / "full_validation/status.json")
        final = json.loads((root / "full_validation/status.json").read_text())
        if final.get("status") != "complete" or final.get("byte_identical_to_reference") is not True or final.get("verified_files") != 482:
            raise ValueError("Full validator did not finish all 241 GEO and binary checkpoint comparisons")
        save("complete", phase="complete", full_status=final, full_directory=str(root / "full_validation"), child_pid=None)
    except BaseException as error:
        stop_owned_child()
        save("stopped", child_pid=None, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stop_owned_child()
        if comparison_lock is not None:
            comparison_lock.close()
        owner_lock.close()


def self_test() -> None:
    import tempfile
    from validate_contact_round2 import trial_plan
    with tempfile.TemporaryDirectory(prefix="contact_round2_supervisor_") as name:
        root = Path(name)
        try:
            validate_test_evidence(root, CANDIDATE_SHA256, 0)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing gate evidence accepted")
        passed_lines = "".join(f"{index}/22 Test #{index}: gpu_resident_test_{index} ........ Passed  0.12 sec\n"
                               for index in range(1, 23))
        contents = dict(ctest=passed_lines + "100% tests passed, 0 tests failed out of 22\n",
            global_guess=QUICK_SUCCESS + "\n", memcheck="========= ERROR SUMMARY: 0 errors\n",
            initcheck="========= ERROR SUMMARY: 0 errors\n",
            racecheck="========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n")
        for key, filename in GATE_LOGS.items():
            (root / filename).write_text(contents[key])
        marker = root / "validation_exit.json"
        marker.write_text(json.dumps(dict(validation_chain_exit_code=0, candidate_sha256=CANDIDATE_SHA256)))
        assert validate_test_evidence(root, CANDIDATE_SHA256, 0)["scoped_ctest_tests"] == 22
        (root / GATE_LOGS["ctest"]).write_text(passed_lines + "100% tests passed out of 22\n")
        marker.write_text(marker.read_text())
        assert validate_test_evidence(root, CANDIDATE_SHA256, 0)["scoped_ctest_tests"] == 22
        for key, bad in (("ctest", passed_lines + "100% tests passed, 0 tests failed out of 21"),
                         ("ctest", "100% tests passed out of 22\n"),
                         ("ctest", passed_lines.replace("22/22 Test", "21/22 Test") + "100% tests passed out of 22\n"),
                         ("ctest", passed_lines + "100% tests passed out of 22\n***Failed"),
                         ("global_guess", "CUDA global guess: 540 PT/EE fixtures; max TOI error 1"),
                         ("memcheck", "Starting check"), ("initcheck", "ERROR SUMMARY: 1 errors"),
                         ("racecheck", "RACECHECK SUMMARY: 1 hazards displayed (0 errors, 1 warnings)")):
            path = root / GATE_LOGS[key];path.write_text(bad)
            try:
                validate_test_evidence(root, CANDIDATE_SHA256, 0)
            except ValueError:
                pass
            else:
                raise AssertionError("Partial/failed " + key + " gate accepted")
            path.write_text(contents[key])
        marker.write_text(json.dumps(dict(validation_chain_exit_code=1, candidate_sha256=CANDIDATE_SHA256)))
        try:
            validate_test_evidence(root, CANDIDATE_SHA256, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("Failed chain exit accepted")
    state = dict(status="complete", scenarios=["late", "middle", "early"], repetitions=2,
                 executable_hashes=dict(baseline=OPTIMIZED_SHA256, candidate=CANDIDATE_SHA256), runs=[])
    for stage, repetition, variant in trial_plan(state["scenarios"], 2):
        state["runs"].append(dict(scenario=stage, repetition=repetition, variant=variant, status="accepted",
            byte_identical_to_fresh_baseline=True, total_solver_s=2. if variant in ("baseline", "all_off") else 1.))
    assert performance_selection(state, CANDIDATE_SHA256)["passed"]
    for run in state["runs"]:
        if run["scenario"] == "late" and run["variant"] == "all_three":
            run["total_solver_s"] = 1.021
    assert not performance_selection(state, CANDIDATE_SHA256)["passed"]
    state["runs"].pop()
    try:
        performance_selection(state, CANDIDATE_SHA256)
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete short trials accepted")
    print("Contact round2 finish supervisor self-test PASS")


if __name__ == "__main__":
    main()
