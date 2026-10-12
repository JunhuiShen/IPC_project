#!/usr/bin/env python3
"""Summarize the validated, queued Example 1 CPU/GPU comparison locally.

Reads status.json and report.json without reopening meshes at server paths.
An unfinished/stopped supervisor produces a progress report, never a claimed
full-run comparison. No simulations, SSH operations, or source edits occur.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shlex
from statistics import median

FRAMES = 240
SUBSTEPS = 5
RANGES = ((1, 60), (61, 120), (121, 180), (181, 240))
CAVEATS = [
    "Application comparison, not isolated CPU/GPU hardware throughput or algorithm-identical execution.",
    "Coloring, arithmetic and per-vertex CCD clipping can differ; trajectories need not match.",
    "Printed solver time includes initial guess and required GPU frame download, but excludes mesh writes.",
    "Printed total simulation time includes frame output/checkpoint work, not complete process startup.",
    "Average substep time is solver total / (frames * 5), not an independently instrumented substep timer.",
    "Maximum frame time is the largest printed solver time for one frame, not total application wall time for that frame.",
    "External-load checks are not an exclusive reservation. Contaminated attempts are excluded.",
    "Mesh completeness/finite coordinates were validated by the server harness; this report does not revalidate downloaded meshes or certify collision-free trajectories.",
]


def read_json(path: Path, required: bool = False) -> dict | None:
    if not path.exists() and not required:
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def nonnegative(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def trial_metrics(trial: dict) -> dict:
    if trial.get("complete") is not True or trial.get("contaminated"):
        raise ValueError("Only validated complete, uncontaminated trials can be summarized")
    mode, count = trial.get("mode"), trial.get("frames")
    if mode not in ("cpu", "gpu") or isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Invalid trial mode or frame count")
    values = [nonnegative(x, "frame solver time") for x in trial["frame_solver_s"]]
    if len(values) != count:
        raise ValueError("Frame timing array does not match the validated frame count")
    solver = nonnegative(trial["total_solver_s"], "total solver time")
    simulation = nonnegative(trial["total_simulation_s"], "total simulation time")
    if solver <= 0 or abs(math.fsum(values) - solver) > .00000051 * (count + 1):
        raise ValueError("Invalid total solver time or inconsistent rounded frame timings")
    if simulation + .0000011 < solver:
        raise ValueError("Total simulation time is smaller than solver time")
    result = {
        "mode": mode, "frames": count,
        "total_solver_s": solver, "total_simulation_s": simulation,
        "average_solver_frame_s": solver / count,
        "average_solver_substep_s": solver / (count * SUBSTEPS),
        "average_simulation_frame_s": simulation / count,
        "max_solver_frame_s": max(values),
        "max_solver_frame_ids": [i + 1 for i, x in enumerate(values) if x == max(values)],
        "selected_frame_solver_s": {str(i): values[i - 1] for i in (2, 220) if i <= count},
        "ranges": [],
        "provenance": {key: trial[key] for key in ("attempt", "log", "outdir", "command", "omp", "readme_command", "mesh_format",
                        "max_observed_system_iowait_cores", "observed_system_iowait_cpu_s") if key in trial},
    }
    for first, last in RANGES:
        if last <= count:
            subset = values[first - 1:last]
            result["ranges"].append({"first": first, "last": last,
                                      "total_solver_s": math.fsum(subset),
                                      "average_solver_frame_s": math.fsum(subset) / len(subset),
                                      "max_solver_frame_s": max(subset)})
    return result


def group_metrics(trials: list[dict]) -> dict:
    fields = ("total_solver_s", "total_simulation_s", "average_solver_frame_s",
              "average_solver_substep_s", "average_simulation_frame_s", "max_solver_frame_s")
    result = {"trial_count": len(trials), "frames_per_trial": FRAMES,
              "aggregation": "median across accepted trials; per-trial maxima remain available"}
    result.update({key: median(row[key] for row in trials) for key in fields})
    result["selected_frame_solver_s"] = {
        str(i): median(row["selected_frame_solver_s"][str(i)] for row in trials) for i in (2, 220)}
    result["ranges"] = []
    for index, (first, last) in enumerate(RANGES):
        result["ranges"].append({"first": first, "last": last, **{
            key: median(row["ranges"][index][key] for row in trials)
            for key in ("total_solver_s", "average_solver_frame_s", "max_solver_frame_s")}})
    return result


def local_progress(directory: Path, state: dict) -> list[dict]:
    """Optional progress from copied logs; absence of server-path logs is fine."""
    rows = []
    for attempt in state.get("attempts", []):
        row = {key: attempt[key] for key in ("mode", "frames", "status", "reasons", "log",
               "max_observed_system_iowait_cores", "observed_system_iowait_cpu_s") if key in attempt}
        log_name = Path(attempt.get("log", "missing.log")).name
        path = directory / log_name
        if path.is_file():
            frames = re.findall(r"^Frame\s+(\d+)\s*\|", path.read_text(errors="replace"), re.MULTILINE)
            row["last_logged_frame"] = int(frames[-1]) if frames else 0
            row["progress_source"] = str(path)
        rows.append(row)
    return rows


def build_summary(directory: Path) -> dict:
    state = read_json(directory / "status.json", required=True)
    report = read_json(directory / "report.json")
    runs = [trial_metrics(row) for row in state.get("runs", [])]
    readme_command = state.get("readme_command", False)
    if any(row.get("readme_command", False) != readme_command for row in state.get("runs", [])):
        raise ValueError("Mixed README/legacy command modes in supervisor state")
    if report is not None and report.get("readme_command", False) != readme_command:
        raise ValueError("Report command mode disagrees with supervisor state")
    complete = state.get("status") == "complete"
    if complete:
        if report is None:
            raise ValueError("Supervisor says complete but validated report.json is missing")
        if not runs or any(row["frames"] != FRAMES for row in runs):
            raise ValueError("A full-run report requires exactly 240 frames in every accepted trial")
        if {row["mode"] for row in runs} != {"cpu", "gpu"}:
            raise ValueError("Complete comparison requires accepted CPU and GPU runs")
        for mode in ("cpu", "gpu"):
            group = [row for row in report.get("groups", []) if row.get("mode") == mode and row.get("frames") == FRAMES]
            accepted = [row for row in runs if row["mode"] == mode]
            if len(group) != 1 or group[0].get("trials") != len(accepted):
                raise ValueError("report.json group counts disagree with status.json")
            expected = median(row["total_solver_s"] for row in accepted)
            if not math.isclose(nonnegative(group[0]["total_solver_s_median"], "reported solver median"), expected,
                                rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("report.json total solver median disagrees with status.json")
    summary = {
        "report_kind": "full_comparison" if complete else "progress_only",
        "supervisor_status": state.get("status", "unknown"),
        "target_frames": FRAMES, "substeps": SUBSTEPS,
        "readme_command": readme_command, "mesh_format": "geo" if readme_command else "obj",
        "sources": {"status": str(directory / "status.json"),
                    "report": str(directory / "report.json") if report is not None else None},
        "accepted_trials": runs,
        "attempt_progress": local_progress(directory, state),
        "provenance": {key: state[key] for key in (
            "scope", "directory", "expected_binaries", "cpu_affinity", "max_background_cores",
            "monitor_interval_s", "max_attempts_per_run", "updated_unix", "readme_command",
            "mesh_format", "readme_sha256", "monitoring_policy") if key in state},
        "waiting_reasons": state.get("waiting_reasons", []),
        "error": state.get("error"),
        "notes": list(CAVEATS),
    }
    if report:
        summary["notes"].extend(note for note in report.get("notes", []) if note not in summary["notes"])
    if state.get("monitoring_policy") and state["monitoring_policy"] not in summary["notes"]:
        summary["notes"].append(state["monitoring_policy"])
    summary["notes"].append(
        "README command: standard CCD initial guess and adaptive node-box policy on both implementations; no colored-guess substitution. Both receive the same 64-thread OpenMP environment and 64-physical-core affinity. GPU OpenMP applies to host work, not CUDA worker count. GEO coordinate comparison uses 10-digit exported text, not full-precision checkpoints."
        if readme_command else
        "Legacy command: colored CCD guess on both; CPU adaptive boxes versus GPU fixed maximum boxes. OBJ output.")
    if not complete:
        summary["notes"].insert(0, "Run is not complete: accepted individual trials and observed log progress are not a completed full CPU/GPU comparison. No full-run speed ratio is reported.")
        return summary
    groups = {mode: group_metrics([row for row in runs if row["mode"] == mode]) for mode in ("cpu", "gpu")}
    summary["groups"] = groups
    summary["gpu_over_cpu_solver_ratio"] = groups["gpu"]["total_solver_s"] / groups["cpu"]["total_solver_s"]
    summary["gpu_over_cpu_simulation_ratio"] = groups["gpu"]["total_simulation_s"] / groups["cpu"]["total_simulation_s"]
    summary["ratio_interpretation"] = "GPU / CPU time: above 1 means GPU slower; below 1 means GPU faster."
    summary["trajectory_comparisons"] = [row for row in report.get("comparisons", []) if row.get("frames") == FRAMES]
    return summary


def render_text(summary: dict) -> str:
    complete = summary["report_kind"] == "full_comparison"
    lines = ["EXAMPLE 1: FULL 240-FRAME CPU/GPU COMPARISON" if complete else
             "EXAMPLE 1: PROGRESS ONLY — FULL COMPARISON NOT COMPLETE",
             f"Supervisor status: {summary['supervisor_status']}",
             f"Command mode: {'README Example 1' if summary['readme_command'] else 'legacy colored CCD guess'}; mesh format: {summary['mesh_format'].upper()}", ""]
    if complete:
        lines += ["All timings below are seconds. Repeated trials are summarized by their median.",
                  "Metric                                  CPU (64 threads)         GPU"]
        for label, key in (("Total solver time", "total_solver_s"),
                           ("Total simulation time", "total_simulation_s"),
                           ("Average solver time per frame", "average_solver_frame_s"),
                           ("Average solver time per substep", "average_solver_substep_s"),
                           ("Average simulation time per frame", "average_simulation_frame_s"),
                           ("Maximum solver time per frame", "max_solver_frame_s")):
            lines.append(f"{label:40s} {summary['groups']['cpu'][key]:18.6f} {summary['groups']['gpu'][key]:12.6f}")
        lines += [f"GPU / CPU solver ratio: {summary['gpu_over_cpu_solver_ratio']:.6f}",
                  f"GPU / CPU simulation ratio: {summary['gpu_over_cpu_simulation_ratio']:.6f}",
                  summary["ratio_interpretation"], "", "FRAME RANGES (solver time)",
                  "Range             CPU total      GPU total     GPU/CPU     CPU avg/frame    GPU avg/frame"]
        for cpu, gpu in zip(summary["groups"]["cpu"]["ranges"], summary["groups"]["gpu"]["ranges"]):
            ratio = gpu["total_solver_s"] / cpu["total_solver_s"] if cpu["total_solver_s"] else None
            ratio_text = f"{ratio:.6f}" if ratio is not None else "undefined"
            lines.append(f"{cpu['first']:3d}-{cpu['last']:<3d} {cpu['total_solver_s']:20.6f} {gpu['total_solver_s']:14.6f} {ratio_text:>11s} {cpu['average_solver_frame_s']:17.6f} {gpu['average_solver_frame_s']:16.6f}")
        lines += ["", "SELECTED FRAMES (solver time)"]
        for frame in ("2", "220"):
            cpu = summary["groups"]["cpu"]["selected_frame_solver_s"][frame]
            gpu = summary["groups"]["gpu"]["selected_frame_solver_s"][frame]
            ratio_text = f"{gpu / cpu:.6f}" if cpu else "undefined"
            lines.append(f"Frame {frame}: CPU {cpu:.6f} s; GPU {gpu:.6f} s; GPU/CPU {ratio_text}")
        lines += ["", "TRAJECTORY DRIFT (from validated server report; not a parity assertion)",
                  json.dumps(summary["trajectory_comparisons"], indent=2)]
    else:
        lines += ["No full-run ratio or extrapolated full runtime is reported.",
                  "Accepted individual trials (if any):"]
        for trial in summary["accepted_trials"]:
            lines.append(f"{trial['mode']}: {trial['frames']} complete frames, solver {trial['total_solver_s']:.6f} s; not a completed CPU/GPU comparison.")
        lines += ["Attempt progress:", json.dumps(summary["attempt_progress"], indent=2),
                  f"Waiting reasons: {summary['waiting_reasons']}", f"Error: {summary['error']}"]
    lines += ["", "PROVENANCE", json.dumps(summary["provenance"], indent=2)]
    for index, trial in enumerate(summary["accepted_trials"], 1):
        p = trial["provenance"]
        lines += [f"Trial {index}: {trial['mode']}, {trial['frames']} frames",
                  f"Peak frame IDs: {trial['max_solver_frame_ids']}",
                  "Environment: " + " ".join(f"{key}={shlex.quote(str(value))}" for key, value in sorted(p.get("omp", {}).items())),
                  "Command: " + shlex.join(p.get("command", [])),
                  "Paths: " + json.dumps({key: p[key] for key in ("log", "outdir") if key in p})]
        if "max_observed_system_iowait_cores" in p:
            lines.append(f"Maximum observed system-wide I/O wait: {p['max_observed_system_iowait_cores']:.6f} CPU equivalents; not process-attributed or additive solver wall time.")
    lines += ["", "NOTES AND LIMITATIONS", *[f"- {note}" for note in summary["notes"]], ""]
    return "\n".join(lines)


def write_report(directory: Path) -> dict:
    summary = build_summary(directory)
    (directory / "comparison_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (directory / "comparison_report.txt").write_text(render_text(summary))
    return summary


def self_test() -> None:
    import tempfile
    import unittest

    class ReportTests(unittest.TestCase):
        def test_complete_and_incomplete_without_remote_meshes(self):
            with tempfile.TemporaryDirectory(prefix="ipc_full_report_") as temporary:
                root = Path(temporary)
                trials = []
                for mode, multiplier in (("cpu", 1), ("gpu", 3)):
                    values = [multiplier * frame / 1000 for frame in range(1, 241)]
                    trials.append({"mode": mode, "frames": 240, "complete": True,
                                   "frame_solver_s": values, "total_solver_s": math.fsum(values),
                                   "total_simulation_s": math.fsum(values) + 2,
                                   "command": ["/remote/3D_sim", "--num_frames", "240"],
                                   "omp": {"OMP_NUM_THREADS": "64" if mode == "cpu" else "1"},
                                   "log": f"/remote/{mode}.log", "outdir": f"/remote/{mode}"})
                state = {"status": "complete", "runs": trials,
                         "expected_binaries": {"cpu": ["build/3D_sim", "abc"]}}
                report = {"groups": [{"mode": t["mode"], "frames": 240, "trials": 1,
                                      "total_solver_s_median": t["total_solver_s"]} for t in trials],
                          "comparisons": [{"frames": 240, "final_position_drift": []}]}
                (root / "status.json").write_text(json.dumps(state))
                (root / "report.json").write_text(json.dumps(report))
                result = write_report(root)
                self.assertEqual(result["report_kind"], "full_comparison")
                self.assertAlmostEqual(result["gpu_over_cpu_solver_ratio"], 3)
                self.assertAlmostEqual(result["groups"]["cpu"]["average_solver_substep_s"], .0241)
                self.assertAlmostEqual(result["groups"]["gpu"]["ranges"][3]["total_solver_s"], 37.89)
                self.assertAlmostEqual(result["groups"]["gpu"]["selected_frame_solver_s"]["220"], .66)
                self.assertEqual(result["accepted_trials"][0]["max_solver_frame_ids"], [240])
                self.assertIn("/remote/3D_sim", (root / "comparison_report.txt").read_text())
                state["readme_command"] = True
                state["mesh_format"] = "geo"
                report["readme_command"] = True
                for trial in state["runs"]:
                    trial.update(readme_command=True, mesh_format="geo")
                    trial["omp"]["OMP_NUM_THREADS"] = "64"
                (root / "status.json").write_text(json.dumps(state))
                (root / "report.json").write_text(json.dumps(report))
                readme_result = write_report(root)
                self.assertEqual(readme_result["mesh_format"], "geo")
                self.assertIn("Command mode: README Example 1", (root / "comparison_report.txt").read_text())
                self.assertNotIn("fixed maximum boxes", (root / "comparison_report.txt").read_text())
                self.assertNotIn("short run", (root / "comparison_report.txt").read_text())
                for status in ("running", "stopped", "waiting_for_idle"):
                    state["status"] = status
                    (root / "status.json").write_text(json.dumps(state))
                    partial = write_report(root)
                    self.assertEqual(partial["report_kind"], "progress_only")
                    self.assertNotIn("gpu_over_cpu_solver_ratio", partial)
                    self.assertIn("PROGRESS ONLY", (root / "comparison_report.txt").read_text())
                state["status"] = "complete"
                state["runs"][0]["frame_solver_s"].pop()
                (root / "status.json").write_text(json.dumps(state))
                with self.assertRaisesRegex(ValueError, "array"):
                    build_summary(root)

        def test_progress_and_bad_data(self):
            with tempfile.TemporaryDirectory(prefix="ipc_full_report_") as temporary:
                root = Path(temporary)
                state = {"status": "running", "runs": [], "attempts": [
                    {"status": "running", "mode": "gpu", "frames": 240, "log": "/remote/003_gpu.log"}]}
                (root / "status.json").write_text(json.dumps(state))
                (root / "003_gpu.log").write_text("Frame 1 | global_iters = 20\nFrame 2 | global_iters = 20\n")
                result = write_report(root)
                self.assertEqual(result["attempt_progress"][0]["last_logged_frame"], 2)
                state["status"] = "complete"
                (root / "status.json").write_text(json.dumps(state))
                with self.assertRaisesRegex(ValueError, "missing"):
                    build_summary(root)
                with self.assertRaises(ValueError):
                    nonnegative(float("nan"), "bad")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReportTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, help="Downloaded comparison directory")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.directory is None:
        parser.error("--directory is required unless --self-test is selected")
    result = write_report(args.directory.resolve())
    print(f"Wrote {result['report_kind']}: {args.directory / 'comparison_report.txt'}")


if __name__ == "__main__":
    main()
