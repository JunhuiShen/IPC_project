#!/usr/bin/env python3
"""Analyze every frame of a diagnostic GPU run; never call it a clean benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean

TIMINGS = ("host_inclusive_ms", "host_exclusive_ms", "cuda_inclusive_ms", "cuda_exclusive_ms")
ROW_FIELDS = {"frame", "stage", "calls", *TIMINGS}
FRAME_LINE = re.compile(r"Frame\s+(\d+)\s*\|.*solver_time\s*=\s*([\d.eE+-]+)\s*ms")


def log_frames(path: Path) -> dict[int, float]:
    result = {}
    for match in FRAME_LINE.finditer(path.read_text(errors="replace")):
        frame, duration = int(match[1]), float(match[2])
        if frame in result or not math.isfinite(duration) or duration < 0:
            raise ValueError("Duplicate or invalid solver frame in " + str(path))
        result[frame] = duration
    return result


def load_baseline(path: Path) -> dict:
    """Keep the accepted CPU trial and recover its complete frame timing log."""
    original = json.loads(path.read_text())
    if original.get("kind") == "accepted_cpu_baseline":
        return original
    trials = original.get("accepted_trials", original.get("runs", []))
    candidates = [trial for trial in trials if trial.get("mode") == "cpu" and trial.get("frames") == 240]
    if len(candidates) != 1:
        raise ValueError("CPU baseline must contain exactly one accepted 240-frame CPU trial")
    trial = candidates[0]
    provenance = trial.get("provenance", trial)
    raw_log = provenance.get("log")
    log_path = Path(raw_log) if raw_log else None
    if log_path is not None and not log_path.is_file():
        log_path = path.parent / log_path.name
    frame_times = log_frames(log_path) if log_path is not None and log_path.is_file() else {
        i: value * 1000 for i, value in enumerate(trial.get("frame_solver_s", []), 1)}
    if set(frame_times) != set(range(1, 241)):
        raise ValueError("CPU baseline needs its complete 240-frame log beside the report or at its original path")
    total_solver = trial["total_solver_s"]
    if abs(sum(frame_times.values()) / 1000 - total_solver) > .00000051 * 241:
        raise ValueError("Accepted CPU total and its frame log disagree")
    return dict(kind="accepted_cpu_baseline", source=str(path.resolve()), trial=trial,
                source_metadata={key: original[key] for key in (
                    "expected_binaries", "cpu_affinity", "readme_sha256", "monitoring_policy", "hardware") if key in original},
                frame_solver_ms={str(k): v for k, v in frame_times.items()},
                cpu_total_solver_s=total_solver, cpu_average_solver_frame_ms=total_solver * 1000 / 240,
                cpu_total_simulation_s=trial["total_simulation_s"],
                cpu_average_simulation_frame_ms=trial["total_simulation_s"] * 1000 / 240)


def read_phases(path: Path) -> dict[int, dict]:
    frames = {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if not ROW_FIELDS | {"frame_wall_ms", "profile_cpu_overhead_ms", "substeps"} <= set(reader.fieldnames or []):
            raise ValueError("Missing required frame/stage profiling columns")
        for row in reader:
            try:
                frame = int(row["frame"])
                stage = row["stage"]
                values = {key: float(value) for key, value in row.items() if key not in ("stage", "frame")}
            except (TypeError, ValueError) as error:
                raise ValueError("Malformed profiling CSV row") from error
            if frame < 1 or not stage or any(not math.isfinite(v) or (v < 0 and key != "cuda_exclusive_ms")
                                           for key, v in values.items()):
                raise ValueError("Invalid frame, stage, or profiling duration/counter")
            if values["cuda_exclusive_ms"] < -.25:
                raise ValueError("CUDA exclusive residual below -0.25 ms; investigate interval accounting")
            # Tiny negative event residuals are preserved, not silently clamped.
            metadata = {key: value for key, value in values.items() if key not in ROW_FIELDS}
            entry = frames.setdefault(frame, dict(frame=frame, metadata=metadata, stages={}))
            if metadata != entry["metadata"] or stage in entry["stages"]:
                raise ValueError("Conflicting per-frame counters or repeated stage rows")
            entry["stages"][stage] = {key: values[key] for key in ("calls", *TIMINGS)}
    if not frames:
        raise ValueError("No completed profiling frames")
    for entry in frames.values():
        stages, metadata = entry["stages"], entry["metadata"]
        if "frame" in stages:
            host_sum = sum(value["host_exclusive_ms"] for value in stages.values())
            cuda_sum = sum(value["cuda_exclusive_ms"] for value in stages.values())
            tolerance = max(1e-5, metadata["frame_wall_ms"] * 1e-8)
            if abs(host_sum - metadata["frame_wall_ms"]) > tolerance:
                raise ValueError("Exclusive host intervals do not reconcile to frame wall time")
            if abs(cuda_sum - stages["frame"]["cuda_inclusive_ms"]) > tolerance:
                raise ValueError("Exclusive CUDA intervals do not reconcile to root stream intervals")
    return frames


def summarize_group(ids: list[int], frames: dict, solver_times: dict, cpu_times: dict) -> dict:
    phases = {}
    for frame in ids:
        for name, values in frames[frame]["stages"].items():
            target = phases.setdefault(name, {key: 0. for key in ("calls", *TIMINGS)})
            for key, value in values.items():
                target[key] += value
    host_total = sum(stage["host_exclusive_ms"] for stage in phases.values())
    stream_total = sum(stage["cuda_exclusive_ms"] for stage in phases.values())
    wall_total = sum(frames[i]["metadata"]["frame_wall_ms"] for i in ids)
    rows = []
    for name, values in phases.items():
        rows.append(dict(stage=name, **values,
                         host_share_of_measured_exclusive=values["host_exclusive_ms"] / host_total if host_total else 0,
                         cuda_share_of_measured_exclusive=values["cuda_exclusive_ms"] / stream_total if stream_total else 0,
                         host_share_of_frame_wall=values["host_exclusive_ms"] / wall_total if wall_total else 0))
    metadata_keys = set.intersection(*(set(frames[i]["metadata"]) for i in ids))
    counters = {key: {"sum": sum(frames[i]["metadata"][key] for i in ids),
                      "mean_per_frame": mean(frames[i]["metadata"][key] for i in ids),
                      "max_per_frame": max(frames[i]["metadata"][key] for i in ids)}
                for key in sorted(metadata_keys) if not key.endswith("_ms")}
    return dict(frame_ids=ids, frame_count=len(ids), first_frame=min(ids), last_frame=max(ids),
                profiled_solver_s=sum(solver_times[i] for i in ids) / 1000,
                profiled_solver_mean_frame_ms=mean(solver_times[i] for i in ids),
                cpu_baseline_solver_s=sum(cpu_times[i] for i in ids) / 1000,
                profile_scope_wall_s=wall_total / 1000,
                host_exclusive_sum_s=host_total / 1000, cuda_exclusive_stream_sum_s=stream_total / 1000,
                profile_cpu_overhead_s=sum(frames[i]["metadata"]["profile_cpu_overhead_ms"] for i in ids) / 1000,
                counters=counters, stages=sorted(rows, key=lambda row: -row["cuda_exclusive_ms"]))


def analyze(directory: Path, expected_frames=240, allow_partial=False) -> dict:
    baseline = load_baseline(directory / "cpu_baseline.json")
    frames = read_phases(directory / "phase_frames.csv")
    solver_times = log_frames(directory / "run.log")
    # The phase callback flushes just before the normal Frame timing line. A live
    # report uses only frames present in both, never an incomplete final frame.
    ids = sorted(set(frames) & set(solver_times))
    if not ids or ids != list(range(1, len(ids) + 1)):
        raise ValueError("Frame timing sequence is empty or noncontiguous")
    complete = len(ids) == expected_frames
    if not allow_partial and (not complete or set(frames) != set(ids) or set(solver_times) != set(ids)):
        raise ValueError(f"Expected {expected_frames} complete frames, received {len(ids)}")
    if len(ids) > expected_frames:
        raise ValueError("More frames than requested")
    cpu_times = {int(k): v for k, v in baseline["frame_solver_ms"].items()}
    proxy = next((key for key in ("solver_candidate_max", "candidate_max")
                  if all(key in frames[i]["metadata"] for i in ids)), None)
    if proxy is None:
        raise ValueError("No solver candidate-count field for contact-intensity stratification")
    groups = {"all_frames": summarize_group(ids, frames, solver_times, cpu_times)}
    for first in range(1, expected_frames + 1, 60):
        chosen = [i for i in ids if first <= i <= first + 59]
        if chosen:
            groups[f"frames_{first}_{min(first + 59, expected_frames)}"] = summarize_group(chosen, frames, solver_times, cpu_times)
    ranked = sorted(ids, key=lambda i: (frames[i]["metadata"][proxy], i))
    for quartile in range(4):
        chosen = sorted(ranked[len(ranked) * quartile // 4:len(ranked) * (quartile + 1) // 4])
        if chosen:
            groups[f"candidate_work_quartile_{quartile + 1}"] = summarize_group(chosen, frames, solver_times, cpu_times)
    for frame in (2, 220):
        if frame in ids:
            groups[f"frame_{frame}"] = summarize_group([frame], frames, solver_times, cpu_times)
    notes = [
        "This is an instrumented diagnostic GPU run, NOT a clean performance benchmark or optimized speedup claim.",
        "The unchanged CPU baseline is retained separately: solver timing excludes output; total simulation time includes its original driver work.",
        "Host and CUDA stream timings overlap. Never add them together.",
        "Stage paths are hierarchical. Inclusive values contain children; exclusive values subtract direct children. Sum ONLY exclusive values across stages.",
        "CUDA event durations are elapsed stream intervals, not summed kernel execution durations. They can include host launch starvation and synchronization gaps.",
        "Profile CPU overhead measures instrumentation bookkeeping; it does not capture every timer/event perturbation and must not be subtracted to claim clean speed.",
        "Candidate-count quartiles are broad-phase work proxies, NOT counts of geometrically active or force-bearing contacts. Equal-count ties use frame order.",
        "Fixed 60-frame ranges show chronological evolution; candidate quartiles independently group frames by collision-candidate workload.",
        "CUDA exclusive residuals between -0.25 ms and zero are retained and flagged as event-resolution/subtraction residuals, not clamped. Larger negatives fail validation.",
    ]
    frame_rows = []
    for i in ids:
        stage_values = frames[i]["stages"].values()
        frame_rows.append(dict(frame=i, profiled_solver_ms=solver_times[i], cpu_baseline_solver_ms=cpu_times[i],
                               host_exclusive_sum_ms=sum(v["host_exclusive_ms"] for v in stage_values),
                               cuda_exclusive_stream_sum_ms=sum(v["cuda_exclusive_ms"] for v in stage_values),
                               **frames[i]["metadata"]))
    negatives = [dict(frame=i, stage=name, cuda_exclusive_ms=value["cuda_exclusive_ms"])
                 for i in ids for name, value in frames[i]["stages"].items() if value["cuda_exclusive_ms"] < 0]
    metadata_path = directory / "command.json"
    status_path = directory / "status.json"
    run_status = json.loads(status_path.read_text()) if status_path.is_file() else None
    kind = "complete_diagnostic_profile" if complete else "partial_diagnostic_progress"
    if run_status and run_status.get("status") == "stopped":
        kind = "stopped_diagnostic_profile"
        notes.append("The supervisor stopped this run: results are retained for diagnosis, not accepted as an uncontaminated full-run profile. Reason: " + str(run_status.get("error")))
    return dict(kind=kind, supervisor_status=run_status.get("status") if run_status else None,
                completed_frames=len(ids), expected_frames=expected_frames, baseline=baseline,
                command_metadata=json.loads(metadata_path.read_text()) if metadata_path.is_file() else None,
                output_manifest=str(directory / "checkpoint_manifest.json"),
                contact_intensity_proxy=proxy, groups=groups, frames=frame_rows,
                slowest_frames=sorted(frame_rows, key=lambda r: -r["profiled_solver_ms"])[:10],
                negative_cuda_exclusive_residuals=negatives, notes=notes)


def render(report: dict) -> str:
    baseline = report["baseline"]
    lines = ["Example 1 GPU full-simulation stage profile", report["kind"],
             f"Frames: {report['completed_frames']}/{report['expected_frames']}", "",
             "Unchanged clean CPU baseline (64 threads, 240 frames):",
             f"  Total solver: {baseline['cpu_total_solver_s']:.6f} s; mean solver/frame: {baseline['cpu_average_solver_frame_ms']:.6f} ms",
             f"  Total simulation: {baseline['cpu_total_simulation_s']:.6f} s; mean simulation/frame: {baseline['cpu_average_simulation_frame_ms']:.6f} ms", ""]
    for name, group in report["groups"].items():
        lines += [name + f" ({group['frame_count']} frames):",
                  f"  Profiled solver: {group['profiled_solver_s']:.6f} s, {group['profiled_solver_mean_frame_ms']:.3f} ms/frame",
                  f"  Exclusive host sum: {group['host_exclusive_sum_s']:.6f} s; exclusive stream sum: {group['cuda_exclusive_stream_sum_s']:.6f} s (do not add)",
                  "  Top exclusive CUDA-stream stages:"]
        for stage in group["stages"][:12]:
            lines.append(f"    {stage['cuda_exclusive_ms']/1000:.6f} s ({stage['cuda_share_of_measured_exclusive']*100:.1f}%) | host {stage['host_exclusive_ms']/1000:.6f} s | {stage['stage']}")
        lines.append("  Top exclusive host stages:")
        for stage in sorted(group["stages"], key=lambda row: -row["host_exclusive_ms"])[:8]:
            lines.append(f"    {stage['host_exclusive_ms']/1000:.6f} s ({stage['host_share_of_measured_exclusive']*100:.1f}%) | {stage['stage']}")
        lines.append("")
    lines += ["Notes:", *["- " + note for note in report["notes"]]]
    return "\n".join(lines) + "\n"


def write_report(directory: Path, expected_frames=240, allow_partial=False) -> dict:
    report = analyze(directory, expected_frames, allow_partial)
    for name, content in (("profile_report.json", json.dumps(report, indent=2) + "\n"),
                          ("profile_report.txt", render(report))):
        temporary = directory / (name + ".tmp")
        temporary.write_text(content)
        temporary.replace(directory / name)
    temporary = directory / "profile_per_frame.csv.tmp"
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(report["frames"][0]))
        writer.writeheader()
        writer.writerows(report["frames"])
    temporary.replace(directory / "profile_per_frame.csv")
    return report


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory(prefix="gpu_full_profile_test_") as name:
        root = Path(name)
        baseline = dict(kind="accepted_cpu_baseline", frame_solver_ms={str(i): 1. for i in range(1, 241)},
                        cpu_total_solver_s=.240, cpu_average_solver_frame_ms=1.,
                        cpu_total_simulation_s=.480, cpu_average_simulation_frame_ms=2.)
        (root / "cpu_baseline.json").write_text(json.dumps(baseline))
        header = ["frame", "stage", "calls", *TIMINGS, "frame_wall_ms", "profile_cpu_overhead_ms", "substeps", "solver_candidate_max"]
        with (root / "phase_frames.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for frame in range(1, 9):
                writer.writerow([frame, "frame", 1, 11, 1, 8, 0, 11, .1, 5, frame * 10])
                writer.writerow([frame, "substep", 5, 10, 2, 8, 1, 11, .1, 5, frame * 10])
                writer.writerow([frame, "substep/compute", 20, 8, 8, 7, 7, 11, .1, 5, frame * 10])
        (root / "run.log").write_text("".join(f"Frame {i} | global_iters = 20 | solver_time = 12.0 ms\n" for i in range(1, 9)))
        report = write_report(root, 8)
        assert report["groups"]["all_frames"]["host_exclusive_sum_s"] == .088
        assert report["groups"]["all_frames"]["cuda_exclusive_stream_sum_s"] == .064
        assert report["groups"]["candidate_work_quartile_4"]["frame_ids"] == [7, 8]
        assert report["groups"]["frame_2"]["frame_count"] == 1
        assert "NOT a clean" in render(report)
        try:
            analyze(root, 240)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete profiling accepted as full")
        assert analyze(root, 240, True)["kind"] == "partial_diagnostic_progress"
        (root / "status.json").write_text(json.dumps(dict(status="stopped", error="synthetic contamination")))
        assert analyze(root, 8)["kind"] == "stopped_diagnostic_profile"
    print("Full-frame GPU profile analyzer self-test PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.directory is None or args.frames < 1:
        parser.error("--directory and positive --frames required")
    write_report(args.directory.resolve(), args.frames, args.allow_partial)


if __name__ == "__main__":
    main()
