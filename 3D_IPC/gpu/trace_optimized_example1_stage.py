#!/usr/bin/env python3
"""Trace one immutable optimized Example 1 stage with Nsight Systems.

Run locally ON the GPU server. This script performs no SSH, build, source edit,
or transfer. All simulation outputs must match a previously validated clean
cold-restart trial. Nsight timings are diagnostic, never a clean speed result.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time

from example1_comparison import verify_readme_command
from profile_optimized_example1_stages import OPTIMIZED_SHA256, SCHEDULE, load_references
from tune_example1_contact import SCENARIOS, clean_environment, parse_segment, segment_command, sha256
from wait_example1_comparison import EXPECTED, HZ, MONITORING_NOTE, activity, physical_cpus, sample

NOTES = [
    "Nsight CUDA tracing changes runtime. This is not a clean benchmark or an isolated measurement of profiler overhead.",
    "Trace scope is the entire cold-restart application, including initialization and output; kernel totals are NOT solver-only timings.",
    "Kernel-duration sum counts each kernel independently; kernel union-active time merges overlaps across all traced streams and devices.",
    "Kernel span minus union-active time is time without a recorded kernel inside that span, not necessarily GPU idle time (copies, memset and host work also occur).",
    "CUDA API durations are host-side calls; they overlap GPU activity and must not be added to kernel durations.",
    "Nsight kernel-duration percentages and CUDA-event stream-span percentages have different denominators and are not interchangeable.",
    "Restart history/caches reset in both the trace and clean reference. Validation uses matching cold-restart GEO and binary hashes, not continuous-run hashes.",
    "Small trace/previous-clean timing ratios are not full-simulation speedups or paired estimates of instrumentation overhead.",
    MONITORING_NOTE,
]


def trace_environment() -> dict:
    env = clean_environment(None)  # Removes every inherited IPC_GPU_* variable.
    env.update({"IPC_GPU_CONTACT_EXACT_" + key.upper(): str(value) for key, value in SCHEDULE.items()})
    return env


def trace_command(nsys: str, prefix: Path, solver: list[str]) -> list[str]:
    return [nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--stats=false",
            "--export=sqlite", "--cuda-memory-usage=false", "--output=" + str(prefix), *solver]


def owned_keys(processes: dict, parents: dict, arguments: dict, known: set,
               leader: tuple, prefix: Path, frames: Path, supervisor: int) -> set:
    """Identity is (PID,start tick); only this unique trace and descendants count."""
    result = set(known)
    if leader in processes:
        result.add(leader)
    for key in processes:
        if key[0] == supervisor:
            continue
        argv = arguments.get(key[0], [])
        pairs = list(zip(argv, argv[1:]))
        if (os.fsencode("--output=" + str(prefix)) in argv
                or (b"--outdir", os.fsencode(frames)) in pairs
                or (b"--session-name", ("profile-" + str(leader[0])).encode()) in pairs):
            result.add(key)
    changed = True
    while changed:
        changed = False
        alive = {key[0] for key in result if key in processes}
        for key in processes:
            if key[0] != supervisor and key not in result and parents.get(key[0]) in alive:
                result.add(key)
                changed = True
    return result


class TraceOwnership:
    """Nsight may detach children into additional sessions/process groups."""
    def __init__(self, pid: int, prefix: Path, frames: Path):
        self.pid, self.prefix, self.frames = pid, prefix, frames
        self.known = set()
        self.leader = None

    def snapshot(self, adjusted=True):
        data = sample(self.pid)
        parents, arguments = {}, {}
        for key, item in data["processes"].items():
            if item["pid"] == self.pid and self.leader is None:
                self.leader = key
            try:
                directory = Path(f"/proc/{item['pid']}")
                fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
                # Reject a PID recycled between the sample and these reads.
                if int(fields[19]) != key[1]:
                    continue
                parents[item["pid"]] = int(fields[1])
                arguments[item["pid"]] = (directory / "cmdline").read_bytes().split(b"\0")
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        self.known = owned_keys(data["processes"], parents, arguments, self.known,
                               self.leader or (self.pid, -1), self.prefix, self.frames, os.getpid())
        if adjusted:
            for key in self.known:
                item = data["processes"].get(key)
                if item and item["pid"] != os.getpid() and item["group"] != self.pid:
                    data["own_cpu"] += item["ticks"] / HZ
                    item["group"] = self.pid
        return data

    def stop(self):
        """Never kill a named profiler globally or any unverified shared group."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            data = self.snapshot(adjusted=False)
            alive = {key: item for key, item in data["processes"].items()
                     if key in self.known and item["pid"] != os.getpid() and item["state"] != "Z"}
            groups = {}
            for key, item in data["processes"].items():
                groups.setdefault(item["group"], set()).add(key)
            signaled = set()
            for key, item in alive.items():
                group = item["group"]
                try:
                    if (group > 1 and group != os.getpgrp() and group not in signaled
                            and groups[group].issubset(self.known)):
                        os.killpg(group, sig)
                        signaled.add(group)
                    elif group not in signaled:
                        # A detached owned process may share an unrelated group.
                        # Revalidate its identity and signal only this PID.
                        fields = Path(f"/proc/{key[0]}/stat").read_text().rsplit(")", 1)[1].split()
                        if int(fields[19]) == key[1]:
                            os.kill(key[0], sig)
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass
            if sig == signal.SIGTERM and alive:
                time.sleep(2)


def interval_summary(intervals: list[tuple[int, int]]) -> dict:
    if not intervals:
        return dict(count=0, duration_sum_s=0., union_active_s=0., span_s=0., nonactive_within_span_s=0.)
    ordered = sorted(intervals)
    if any(end < start for start, end in ordered):
        raise ValueError("Negative trace duration")
    left, right = ordered[0]
    union = 0
    for start, end in ordered[1:]:
        if start > right:
            union += right - left
            left, right = start, end
        else:
            right = max(right, end)
    union += right - left
    span = max(end for _, end in ordered) - ordered[0][0]
    return dict(count=len(ordered), duration_sum_s=sum(end - start for start, end in ordered) / 1e9,
                union_active_s=union / 1e9, span_s=span / 1e9, nonactive_within_span_s=(span - union) / 1e9)


def sqlite_summary(path: Path) -> dict:
    """Read-only CUPTI export inspection; no dependence on localized stats text."""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        strings = dict(db.execute("SELECT id, value FROM StringIds")) if "StringIds" in tables else {}
        result = dict(scope="entire traced application, not solver-only", tables=sorted(tables))
        kernel_table = "CUPTI_ACTIVITY_KIND_KERNEL"
        if kernel_table not in tables:
            raise ValueError("Nsight export has no CUDA kernel records")
        columns = {row[1] for row in db.execute(f"PRAGMA table_info({kernel_table})")}
        name = next((key for key in ("demangledName", "shortName", "name") if key in columns), None)
        if not name or not {"start", "end"}.issubset(columns):
            raise ValueError("Unrecognized Nsight CUDA kernel schema")
        rows = list(db.execute(f"SELECT start, end, {name} FROM {kernel_table}"))
        if not rows:
            raise ValueError("Empty Nsight CUDA kernel trace")
        result["kernels"] = interval_summary([(start, end) for start, end, _ in rows])
        grouped = {}
        for start, end, identity in rows:
            label = strings.get(identity, str(identity))
            grouped.setdefault(label, []).append((start, end))
        total = result["kernels"]["duration_sum_s"]
        result["kernel_functions"] = sorted(
            [dict(name=label, **interval_summary(events),
                  duration_share=sum(end-start for start, end in events)/1e9/total if total else 0.)
             for label, events in grouped.items()], key=lambda row: -row["duration_sum_s"])
        result["other_cuda_activity"] = {}
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER",
                      "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            if table in tables:
                result["other_cuda_activity"][table] = interval_summary(list(db.execute(f"SELECT start, end FROM {table}")))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization-directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="late")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--monitor-interval", type=float, default=5)
    parser.add_argument("--max-background-cores", type=float, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.optimization_directory is None or args.output is None:
        parser.error("--optimization-directory and --output are required")
    if (not 0 < args.timeout < float("inf") or not 0 < args.monitor_interval <= 60
            or not 0 <= args.max_background_cores < float("inf")):
        parser.error("Invalid monitoring/time limit")
    repo = Path(__file__).resolve().parent.parent
    verify_readme_command(repo)
    optimization = args.optimization_directory.resolve(strict=True)
    optimization_status = optimization / "status.json"
    completed = json.loads(optimization_status.read_text())
    if (completed.get("status") != "complete" or completed.get("candidate_sha256") != OPTIMIZED_SHA256
            or completed.get("selected_config") != SCHEDULE):
        raise RuntimeError("Optimization is not completed with the pinned candidate and schedule")
    tuning_path = optimization / "tuning/status.json"
    reference = load_references(json.loads(tuning_path.read_text()), [args.scenario])[args.scenario]
    executable = (optimization / "3D_sim_gpu_candidate").resolve(strict=True)
    nsys = shutil.which("nsys")
    if not nsys:
        raise RuntimeError("Nsight Systems (nsys) is not installed/on PATH")
    pinned = {str(executable): OPTIMIZED_SHA256, str(tuning_path): sha256(tuning_path),
              str(optimization_status): sha256(optimization_status), str(repo / "README.md"): sha256(repo / "README.md"),
              str(repo / EXPECTED["cpu"][0]): EXPECTED["cpu"][1]}
    if reference["checkpoint"]:
        pinned[reference["checkpoint"]["path"]] = reference["checkpoint"]["sha256"]
    root = args.output.resolve(); root.mkdir(exist_ok=True)
    if any(path.name != "supervisor.log" for path in root.iterdir()):
        raise RuntimeError("Use a fresh trace directory")
    cpus = physical_cpus()
    if cpus != list(range(64)):
        raise RuntimeError("Expected verified physical CPU affinity 0..63")
    scenario, frames, log = SCENARIOS[args.scenario], root / "frames", root / "run.log"
    prefix, trace_db = root / "trace", root / "trace.sqlite"
    env = trace_environment()
    solver_argv = segment_command(repo, frames, executable, scenario, cpus)
    argv = trace_command(nsys, prefix, solver_argv)
    state = dict(kind="optimized GPU Nsight stage diagnostic", status="preparing", pid=os.getpid(),
                 directory=str(root), scenario=args.scenario, schedule=SCHEDULE, executable_sha256=OPTIMIZED_SHA256,
                 cpu_affinity=cpus, command=argv, pinned_inputs=pinned, reference=reference,
                 environment={key: value for key, value in env.items() if key.startswith(("OMP_", "IPC_GPU_"))},
                 notes=NOTES, background_observations=[])

    def save(status, **fields):
        state.update(status=status, updated_unix=time.time(), **fields)
        temporary = root / "status.tmp"; temporary.write_text(json.dumps(state, indent=2)+"\n"); temporary.replace(root / "status.json")

    def verify():
        verify_readme_command(repo)
        for name, digest in pinned.items():
            if sha256(Path(name)) != digest:
                raise RuntimeError("Pinned input changed: " + name)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Termination signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    lock = (repo / "gpu/.example1_comparison.lock").open("a+")
    process, ownership = None, None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify()
        consecutive = 0
        while consecutive < 3:
            before = sample(); time.sleep(args.monitor_interval); after = sample()
            reasons, load = activity(before, after, [], max_background_cores=args.max_background_cores)
            if shutil.disk_usage(root).free < 8*1024**3:
                reasons.append("less than 8 GiB available trace disk space")
            consecutive = 0 if reasons else consecutive + 1
            save("waiting_for_idle", idle_checks=consecutive, reasons=reasons, external_cpu_cores=load)
        verify()
        (root / "hardware.json").write_text(json.dumps(dict(
            cpu=subprocess.check_output(["lscpu"], text=True), nsys=subprocess.check_output([nsys, "--version"], text=True),
            gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True)), indent=2)+"\n")
        if reference["checkpoint"]:
            item = reference["checkpoint"]; frames.mkdir()
            shutil.copy2(item["path"], frames / Path(item["path"]).name)
            if sha256(frames / Path(item["path"]).name) != item["sha256"]:
                raise RuntimeError("Restart copy hash mismatch")
        (root / "command.json").write_text(json.dumps(state, indent=2)+"\n")
        # Monitor both the trace and stats/export subprocesses. Stats is not a
        # simulation but can use substantial CPU and must yield to other users.
        def monitored(command, destination, phase, timeout):
            nonlocal process, ownership
            before = sample(); started = time.monotonic()
            with destination.open("w") as stream:
                process = subprocess.Popen(command, cwd=repo, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                ownership = TraceOwnership(process.pid, prefix, frames)
                ownership.snapshot()  # Capture identities before short children exit.
                save(phase, child_pid=process.pid)
                while True:
                    try:
                        process.wait(timeout=args.monitor_interval); done = True
                    except subprocess.TimeoutExpired:
                        done = False
                    after = ownership.snapshot()
                    before["own_cpu"] = min(before["own_cpu"], after["own_cpu"])
                    reasons, load = activity(before, after, [], process.pid, args.max_background_cores)
                    elapsed = time.monotonic() - started
                    state["background_observations"].append(dict(phase=phase, elapsed_s=elapsed,
                        external_cpu_cores=load, reasons=reasons, system_iowait_cpu_s=max(0., (after["iowait"]-before["iowait"])/HZ)))
                    before = after
                    if reasons:
                        raise RuntimeError("External workload: " + "; ".join(reasons))
                    save(phase, elapsed_s=elapsed)
                    if done:
                        break
                    if elapsed > timeout:
                        raise RuntimeError(phase + " exceeded its time limit")
                if process.returncode:
                    raise RuntimeError(f"{phase} returned {process.returncode}; see {destination}")
            ownership.stop(); process.wait(timeout=5)
            process, ownership = None, None
            return elapsed

        state["trace_process_wall_s"] = monitored(argv, log, "tracing", args.timeout)
        verify()
        parsed = parse_segment(log, frames, scenario)
        if parsed["output_hashes"] != reference["output_hashes"]:
            state["mismatches"] = [name for name in set(parsed["output_hashes"]) | set(reference["output_hashes"])
                                    if parsed["output_hashes"].get(name) != reference["output_hashes"].get(name)]
            raise RuntimeError("Traced GEO/binary state differs from clean cold restart: " + str(state["mismatches"]))
        if reference["checkpoint"]:
            item = reference["checkpoint"]
            if sha256(frames / Path(item["path"]).name) != item["sha256"]:
                raise RuntimeError("Restart input was modified")
        if not trace_db.is_file() or not (root / "trace.nsys-rep").is_file():
            raise RuntimeError("Nsight did not produce both the SQLite export and report")
        save("validated", parsed=parsed, byte_identical_to_clean_restart=True)
        stats_argv = [nsys, "stats", "--report", "cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum", str(trace_db)]
        state["stats_command"] = stats_argv
        monitored(stats_argv, root / "kernel_stats.txt", "collecting_stats", 300)
        summary = sqlite_summary(trace_db)
        verify()
        report = dict(kind=state["kind"], scenario=args.scenario, executable_sha256=OPTIMIZED_SHA256,
                      schedule=SCHEDULE, parsed=parsed, trace=summary, reference=reference,
                      byte_identical_to_clean_restart=True, notes=NOTES,
                      traced_over_previous_clean_solver_ratio=parsed["total_solver_s"]/reference["clean_reference_mean_solver_s"])
        (root / "trace_report.json").write_text(json.dumps(report, indent=2)+"\n")
        kernels = summary["kernels"]
        lines = ["Optimized Example 1 Nsight diagnostic: " + args.scenario,
                 "GEO and binary outputs match the clean cold-restart reference.",
                 f"Traced solver wall: {parsed['total_solver_s']:.6f} s (instrumented)",
                 f"Previous clean matching solver mean: {reference['clean_reference_mean_solver_s']:.6f} s",
                 f"Whole-process kernel duration sum: {kernels['duration_sum_s']:.6f} s",
                 f"Whole-process kernel union-active: {kernels['union_active_s']:.6f} s",
                 f"First-to-last kernel span: {kernels['span_s']:.6f} s",
                 f"Non-kernel time within span: {kernels['nonactive_within_span_s']:.6f} s", "", "Top kernels:"]
        for row in summary["kernel_functions"][:25]:
            lines.append(f"  {row['duration_sum_s']:.6f} s ({100*row['duration_share']:.2f}%), {row['count']} calls: {row['name']}")
        lines += ["", "Notes:", *("- " + note for note in NOTES)]
        (root / "trace_report.txt").write_text("\n".join(lines)+"\n")
        save("complete", child_pid=None, trace_summary=summary, report=str(root / "trace_report.json"))
    except BaseException as error:
        if ownership is not None:
            ownership.stop()
        save("stopped", child_pid=None, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if ownership is not None:
            ownership.stop()
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        lock.close()


def self_test():
    import tempfile
    from unittest.mock import patch
    with patch.dict(os.environ, {"IPC_GPU_PROFILE_CSV": "bad", "IPC_GPU_CONTACT_QUEUE_CSV": "bad",
                                 "IPC_GPU_CONTACT_EXACT_SMALL_THREADS": "8", "OMP_NUM_THREADS": "2"}):
        env = trace_environment()
        assert env["OMP_NUM_THREADS"] == "64" and env["OMP_WAIT_POLICY"] == "ACTIVE"
        assert env["IPC_GPU_CONTACT_EXACT_SMALL_THREADS"] == "1"
        assert not any(key.startswith("IPC_GPU_PROFILE") or key == "IPC_GPU_CONTACT_QUEUE_CSV" for key in env)
    assert trace_command("nsys", Path("/trace/test"), ["solver"])[-1] == "solver"
    assert "--cuda-memory-usage=false" in trace_command("nsys", Path("/trace/test"), ["solver"])
    result = interval_summary([(0, 10), (5, 15), (20, 25)])
    assert result["duration_sum_s"] == 25e-9 and result["union_active_s"] == 20e-9
    assert result["span_s"] == 25e-9 and result["nonactive_within_span_s"] == 5e-9
    processes = {(10, 1): {}, (11, 1): {}, (12, 1): {}, (13, 2): {}, (14, 1): {}, (99, 1): {}}
    owned = owned_keys(processes, {11: 10, 12: 11, 14: 13},
                       {13: [b"nsys", b"--output=/test/trace"], 99: [b"nsys", b"--output=/other/trace"]},
                       {(99, 0)}, (10, 1), Path("/test/trace"), Path("/test/frames"), 500)
    assert {(10, 1), (11, 1), (12, 1), (13, 2), (14, 1)}.issubset(owned)
    assert (99, 1) not in owned  # PID reuse is not ownership; other profiler stays foreign.
    with tempfile.TemporaryDirectory(prefix="trace_optimized_example1_test_") as name:
        path = Path(name) / "trace.sqlite"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE StringIds(id INTEGER, value TEXT)")
            db.execute("INSERT INTO StringIds VALUES(1, 'kernel_A')")
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, demangledName INTEGER)")
            db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?)", [(0, 10, 1), (5, 15, 1), (20, 25, 2)])
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(0, 3)")
        before = sha256(path); summary = sqlite_summary(path)
        assert sha256(path) == before and summary["kernels"] == result
        assert summary["kernel_functions"][0]["name"] == "kernel_A"
        assert summary["kernel_functions"][0]["duration_share"] == .8
    print("Optimized Nsight stage trace self-test PASS")


if __name__ == "__main__":
    main()
