#!/usr/bin/env python3
"""Pure-data helpers for the queued full-resolution Example 1 comparison.

No simulations are launched here. The opt-in readme_command mode passes exactly
the documented Example 1 arguments (except executable/outdir and an optional
short validation frame count). The legacy mode preserves earlier colored-guess
OBJ runs. Mesh writing is outside the printed solver timer; GPU frame downloads
remain inside that timer. Coordinate drift is reported, never asserted to be
zero. ``python3 gpu/example1_comparison.py --self-test`` tests the helpers
with temporary synthetic logs/meshes; it does not use the server or CUDA.
"""

from __future__ import annotations

import math
import json
import os
from pathlib import Path
import re
from statistics import median

VERTICES = 10_000
TRIANGLES = 19_602
SUBSTEPS = 5
ITERATIONS = 4
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _mode(mode: str) -> str:
    if mode not in ("cpu", "gpu"):
        raise ValueError("mode must be 'cpu' or 'gpu'")
    return mode


def _frames(frames: int) -> int:
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError("frames must be a positive integer")
    return frames


def command(repo: Path, outdir: Path, mode: str, frames: int, *, readme_command: bool = False) -> list[str]:
    """Return argv; caller selects cwd, validates affinity, and launches it."""
    _mode(mode)
    _frames(frames)
    executable = "build/3D_sim" if mode == "cpu" else "gpu/build/3D_sim_gpu_resident"
    if readme_command:
        # Deliberately do not spell out default values: this is the README
        # command, not a near-equivalent selection of flags and initial guesses.
        return [
            str(Path(repo).resolve() / executable), "--example", "1", "--num_frames", str(frames),
            "--E", "115000", "--nu", "0.25", "--kB", "0.009", "--kpin", "1e9", "--twist_rate", "0.5",
            "--d_hat", "0.005", "--k_barrier", "100", "--node_box_min", "0.001", "--node_box_max", "0.01",
            "--fixed_iters", "--substeps", "5", "--max_substep_iters", "4", "--node_box_update_count", "10",
            "--use_basic_experimental", "true", "--use_simd", "true", "--use_parallel", "true",
            "--outdir", str(Path(outdir).resolve()), "--format", "geo",
        ]
    return [
        str(Path(repo).resolve() / executable),
        "--example", "1", "--twist_nx", "99", "--twist_ny", "99",
        "--num_frames", str(frames), "--fps", "30",
        "--E", "115000", "--nu", "0.25", "--kB", "0.009",
        "--kpin", "1e9", "--twist_rate", "0.5",
        "--density", "900", "--thickness", "0.001",
        "--d_hat", "0.005", "--k_barrier", "100",
        "--friction_coefficient", "0", "--use_ccd", "true",
        "--use_ticcd", "false", "--exact_computation_fallback", "true",
        "--node_box_min", "0.001", "--node_box_max", "0.01",
        "--fixed_iters", "--substeps", str(SUBSTEPS),
        "--max_substep_iters", str(ITERATIONS), "--node_box_update_count", "10",
        "--use_basic_experimental", "true", "--use_simd", "true",
        "--use_parallel", "true", "--use_cloth_grid", "false",
        "--use_ogc", "false", "--use_ogc_solver", "false",
        "--use_verlet_guess", "false", "--use_translation_guess", "false",
        "--use_ccd_guess", "false", "--use_colored_ccd_guess", "true",
        "--colored_ccd_guess_iters", "10", "--damping", "1",
        "--write_substeps", "false", "--format", "obj",
        "--outdir", str(Path(outdir).resolve()),
    ]


def environment(mode: str, *, readme_command: bool = False) -> dict[str, str]:
    """Fresh OMP policy; process affinity is the supervisor's responsibility."""
    _mode(mode)
    result = {key: value for key, value in os.environ.items()
              if not key.startswith(("OMP_", "GOMP_", "KMP_"))}
    result.update(OMP_NUM_THREADS="64" if mode == "cpu" or readme_command else "1",
                  OMP_DYNAMIC="FALSE", OMP_PROC_BIND="close",
                  OMP_PLACES="cores", OMP_WAIT_POLICY="ACTIVE" if mode == "cpu" or readme_command else "PASSIVE")
    return result


def verify_readme_command(repo: Path) -> None:
    """Refuse an obsolete copied command when the current README changes."""
    import shlex
    readme = (repo / "README.md").read_text(encoding="utf-8")
    try:
        text = readme.split("# Example 1:", 1)[1].split("# Example 2:", 1)[0]
        documented = shlex.split(text[text.index("./build/3D_sim"):].replace("\\\n", " "))
    except (IndexError, ValueError) as error:
        raise ValueError("Cannot locate the README Example 1 command") from error
    expected = command(repo, repo / "example1_output", "cpu", 240, readme_command=True)
    documented[0] = expected[0]
    output_index = documented.index("--outdir") + 1
    documented[output_index] = str((repo / documented[output_index]).resolve())
    if documented != expected:
        raise ValueError("README Example 1 command changed; update and validate the comparison harness first")


def _obj_positions(path: Path) -> list[tuple[float, float, float]]:
    vertices: list[tuple[float, float, float]] = []
    face_count = 0
    with path.open(encoding="utf-8") as mesh:
        for number, line in enumerate(mesh, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                if len(fields) != 4:
                    raise ValueError(f"{path}:{number}: expected three vertex coordinates")
                value = tuple(float(item) for item in fields[1:])
                if not all(math.isfinite(item) for item in value):
                    raise ValueError(f"{path}:{number}: nonfinite vertex")
                vertices.append(value)
            elif fields[0] == "f":
                if len(fields) != 4:
                    raise ValueError(f"{path}:{number}: expected a triangle")
                indices = [int(item.split("/", 1)[0]) for item in fields[1:]]
                if len(set(indices)) != 3 or not all(1 <= item <= VERTICES for item in indices):
                    raise ValueError(f"{path}:{number}: invalid face indices")
                face_count += 1
    if len(vertices) != VERTICES or face_count != TRIANGLES:
        raise ValueError(f"{path}: incomplete/wrong mesh ({len(vertices)} vertices, {face_count} faces)")
    return vertices


def _geo_fields(value, description: str) -> dict:
    """Parse Houdini's alternating key/value arrays without losing duplicates."""
    if not isinstance(value, list) or len(value) % 2 or not all(isinstance(k, str) for k in value[::2]):
        raise ValueError(f"invalid GEO {description} key/value array")
    result = dict(zip(value[::2], value[1::2]))
    if len(result) != len(value) // 2:
        raise ValueError(f"duplicate GEO {description} key")
    return result


def _geo_positions(path: Path) -> list[tuple[float, float, float]]:
    """Validate export_geo's JSON mesh and read its printed (10-digit) P data."""
    def reject_constant(value):
        raise ValueError(f"{path}: nonfinite JSON constant {value}")
    mesh = _geo_fields(json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant), "root")
    if any(type(mesh.get(key)) is not int or mesh[key] != expected for key, expected in (
            ("pointcount", VERTICES), ("vertexcount", 3 * TRIANGLES), ("primitivecount", TRIANGLES))):
        raise ValueError(f"{path}: incomplete/wrong GEO mesh counts")
    try:
        indices = _geo_fields(_geo_fields(mesh["topology"], "topology")["pointref"], "pointref")["indices"]
        if (not isinstance(indices, list) or len(indices) != 3 * TRIANGLES
                or any(type(index) is not int or not 0 <= index < VERTICES for index in indices)
                or any(len(set(indices[i:i + 3])) != 3 for i in range(0, len(indices), 3))):
            raise ValueError(f"{path}: invalid GEO face indices")
        primitives = mesh["primitives"]
        if not isinstance(primitives, list) or len(primitives) != 1 or len(primitives[0]) != 2:
            raise ValueError(f"{path}: expected one GEO Polygon_run")
        kind = _geo_fields(primitives[0][0], "primitive kind")
        run = _geo_fields(primitives[0][1], "primitive run")
        if (kind.get("type") != "Polygon_run" or run.get("startvertex") != 0
                or run.get("nprimitives") != TRIANGLES or run.get("nvertices_rle") != [3, TRIANGLES]):
            raise ValueError(f"{path}: wrong GEO polygon run")
        attrs = _geo_fields(mesh["attributes"], "attributes")["pointattributes"]
        positions = []
        for attribute in attrs:
            if not isinstance(attribute, list) or len(attribute) != 2:
                raise ValueError(f"{path}: invalid GEO point attribute")
            descriptor = _geo_fields(attribute[0], "point attribute descriptor")
            if descriptor.get("name") != "P":
                continue
            if positions:
                raise ValueError(f"{path}: duplicate GEO P attribute")
            data = _geo_fields(attribute[1], "P data")
            values = _geo_fields(data["values"], "P values")
            tuples = values["tuples"]
            if data.get("size") != 3 or values.get("size") != 3 or len(tuples) != VERTICES:
                raise ValueError(f"{path}: incomplete/wrong GEO P data")
            for point in tuples:
                if (not isinstance(point, list) or len(point) != 3
                        or any(type(x) not in (int, float) or not math.isfinite(x) for x in point)):
                    raise ValueError(f"{path}: invalid/nonfinite GEO position")
                positions.append(tuple(float(x) for x in point))
        if len(positions) != VERTICES:
            raise ValueError(f"{path}: missing GEO P attribute")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"{path}: malformed/incomplete GEO mesh") from error
    return positions


def _mesh_positions(path: Path) -> list[tuple[float, float, float]]:
    if path.suffix == ".obj":
        return _obj_positions(path)
    if path.suffix == ".geo":
        return _geo_positions(path)
    raise ValueError("mesh format must be obj or geo")


def _one_timing(text: str, label: str, unit: str) -> float:
    found = re.findall(rf"^{re.escape(label)}\s*:\s*({_NUMBER})\s+{re.escape(unit)}\s*$",
                       text, re.MULTILINE)
    if len(found) != 1:
        raise ValueError(f"expected one complete '{label}' timing")
    value = float(found[0])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid '{label}' timing")
    return value


def parse_trial(log: Path, outdir: Path, frames: int, mode: str, *, readme_command: bool = False) -> dict:
    """Validate a successful fresh run; process returncode is checked by caller."""
    _frames(frames)
    _mode(mode)
    log, outdir = Path(log).resolve(), Path(outdir).resolve()
    text = log.read_text(encoding="utf-8")
    if len(re.findall(r"^Simulation finished\.\s*$", text, re.MULTILINE)) != 1:
        raise ValueError("missing or duplicate simulation completion marker")
    vertex_counts = re.findall(r"^(?:Total\s+)?Vertices\s*:\s*(\d+)\s*$", text,
                               re.MULTILINE | re.IGNORECASE)
    if vertex_counts != [str(VERTICES)]:
        raise ValueError("log must report exactly one 10000-vertex scene")
    if re.search(r"^\s*(?:Error:|terminate called|Traceback\b|CUDA error)", text,
                 re.MULTILINE | re.IGNORECASE):
        raise ValueError("log contains a failure diagnostic")
    entries = re.findall(rf"^Frame\s+(\d+)\s*\|\s*global_iters\s*=\s*(\d+)"
                         rf"\s*\|\s*solver_time\s*=\s*({_NUMBER})\s+ms\s*$",
                         text, re.MULTILINE)
    if [int(entry[0]) for entry in entries] != list(range(1, frames + 1)):
        raise ValueError("frame timing sequence is incomplete, duplicated, or out of order")
    if any(int(entry[1]) != SUBSTEPS * ITERATIONS for entry in entries):
        raise ValueError("unexpected solver iteration count; expected 5 substeps x 4 iterations")
    frame_ms = [float(entry[2]) for entry in entries]
    if any(not math.isfinite(value) or value < 0 for value in frame_ms):
        raise ValueError("nonfinite or negative frame timing")
    total_ms = _one_timing(text, "Total solver time", "ms")
    average_ms = _one_timing(text, "Average solver time", "ms/frame")
    simulation_ms = _one_timing(text, "Total simulation time", "ms")
    # Every value is printed to .001 ms; permit only the resulting rounding.
    tolerance = .00051 * (frames + 1)
    if abs(math.fsum(frame_ms) - total_ms) > tolerance:
        raise ValueError("total solver time disagrees with per-frame timings")
    if abs(average_ms * frames - total_ms) > tolerance:
        raise ValueError("average solver time disagrees with total")
    if simulation_ms + .0011 < total_ms:
        raise ValueError("simulation wall time is smaller than its solver time")
    mesh_format = "geo" if readme_command else "obj"
    expected = {f"frame_{frame:04d}.{mesh_format}" for frame in range(frames + 1)}
    actual = {path.name for path in outdir.glob(f"frame_*.{mesh_format}")}
    if actual != expected:
        raise ValueError(f"{mesh_format.upper()} frame set must be exactly frames 0 through N")
    final_positions = []
    for frame in range(frames + 1):
        final_positions = _mesh_positions(outdir / f"frame_{frame:04d}.{mesh_format}")
    total_s = total_ms / 1000
    later_s = math.fsum(frame_ms[1:]) / 1000 if frames > 1 else None
    result = {
        "mode": mode, "frames": frames, "complete": True,
        "readme_command": readme_command, "mesh_format": mesh_format,
        "vertices": VERTICES, "triangles": TRIANGLES,
        "log": str(log), "outdir": str(outdir),
        "initial_mesh": str(outdir / f"frame_0000.{mesh_format}"),
        "final_mesh": str(outdir / f"frame_{frames:04d}.{mesh_format}"),
        "total_solver_s": total_s, "average_frame_s": total_s / frames,
        "average_substep_s": total_s / (frames * SUBSTEPS),
        "total_simulation_s": simulation_ms / 1000,
        "frame_solver_s": [value / 1000 for value in frame_ms],
        "frame2_onward_solver_s": later_s,
        "frame2_onward_average_frame_s": later_s / (frames - 1) if later_s is not None else None,
        "final_bounds": {"min": [min(point[k] for point in final_positions) for k in range(3)],
                         "max": [max(point[k] for point in final_positions) for k in range(3)]},
    }
    if mesh_format == "obj":  # Preserve keys consumed by historical helpers.
        result.update(initial_obj=result["initial_mesh"], final_obj=result["final_mesh"])
    return result


def _drift(a: list[tuple[float, float, float]], b: list[tuple[float, float, float]]) -> dict:
    distances = [math.dist(x, y) for x, y in zip(a, b)]
    if len(a) != len(b) or not distances or not all(map(math.isfinite, distances)):
        raise ValueError("invalid coordinate drift inputs/results")
    largest = max(distances)
    return {
        "max_vertex_l2_m": largest,
        "mean_vertex_l2_m": largest * (math.fsum(d / largest for d in distances) / len(distances)) if largest else 0.0,
        "rms_vertex_l2_m": largest * math.sqrt(math.fsum((d / largest) ** 2 for d in distances) / len(distances)) if largest else 0.0,
        "max_coordinate_abs_m": max(abs(x[k] - y[k]) for x, y in zip(a, b) for k in range(3)),
    }


def summarize(trials: list) -> dict:
    """Group complete clean trials; CPU/GPU ratio >1 means GPU is faster."""
    policies = {bool(trial.get("readme_command", False)) for trial in trials}
    if len(policies) > 1:
        raise ValueError("Do not mix README and legacy colored-guess runs in one comparison")
    readme_command = policies == {True}
    groups: dict[tuple[int, str], list[dict]] = {}
    for trial in trials:
        if not trial.get("complete") or trial.get("contaminated"):
            raise ValueError("summarize only complete, uncontaminated parsed trials")
        key = (_frames(trial["frames"]), _mode(trial["mode"]))
        groups.setdefault(key, []).append(trial)
    fields = ("total_solver_s", "average_frame_s", "average_substep_s", "total_simulation_s",
              "frame2_onward_solver_s", "frame2_onward_average_frame_s")
    rows = []
    for (frames, mode), group in sorted(groups.items()):
        row = {"frames": frames, "mode": mode, "trials": len(group)}
        for field in fields:
            values = [trial[field] for trial in group if trial[field] is not None]
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError("invalid summary timing")
            row[field + "_median"] = median(values) if values else None
        rows.append(row)
    comparisons, positions = [], {}
    for frames in sorted({key[0] for key in groups}):
        cpu, gpu = groups.get((frames, "cpu"), []), groups.get((frames, "gpu"), [])
        if not cpu or not gpu:
            continue
        gpu_s = median(trial["total_solver_s"] for trial in gpu)
        if gpu_s <= 0:
            raise ValueError("GPU solver time must be positive for speedup")
        comparison = {"frames": frames,
                      "cpu_over_gpu_solver_speedup": median(trial["total_solver_s"] for trial in cpu) / gpu_s,
                      "final_position_drift": []}
        for c in cpu:
            for g in gpu:
                for trial in (c, g):
                    for stage in ("initial", "final"):
                        path = trial.get(stage + "_mesh", trial.get(stage + "_obj"))
                        if path is None:
                            raise ValueError("trial has no validated mesh path")
                        if path not in positions:
                            positions[path] = _mesh_positions(Path(path))
                comparison["final_position_drift"].append({
                    "cpu_log": c["log"], "gpu_log": g["log"],
                    **{stage + "_positions": _drift(
                        positions[c.get(stage + "_mesh", c.get(stage + "_obj"))],
                        positions[g.get(stage + "_mesh", g.get(stage + "_obj"))])
                       for stage in ("initial", "final")},
                })
        comparisons.append(comparison)
    notes = [
        "Printed solver timings include initial guess and, for GPU, required frame download; mesh writes are outside that timer.",
        "Total simulation time includes frame output/checkpoint work, not complete process startup.",
        "Position drift is reported without asserting numerical parity or certifying a collision-free trajectory.",
    ]
    if readme_command:
        notes += [
            "Both runs pass the README Example 1 arguments without initial-guess or default-value substitutions; GEO output, 5 substeps, 4 iterations.",
            "Both use OMP_NUM_THREADS=64, OMP_DYNAMIC=FALSE, OMP_WAIT_POLICY=ACTIVE and the same 64 physical-CPU affinity. GPU OpenMP controls host setup/output, not CUDA worker count.",
            "Standard CCD initial guess and adaptive node-box policy are used on CPU and GPU; coloring, arithmetic and per-vertex CCD clipping still differ.",
            "GEO position drift uses the exported 10-significant-digit text values, not full-precision checkpoints; Houdini's P metadata specifies fpreal32.",
        ]
    else:
        notes += [
            "Legacy comparison: 10K vertices, 5 substeps, 4 sweeps, colored CCD guess 10, collision on, friction zero, OBJ output.",
            "Legacy CPU adaptive 0.001-0.01 boxes differ from fixed 0.01 GPU boxes; CCD acceptance and coloring also differ.",
        ]
    if any(trial["frames"] < 240 for trial in trials):
        notes.append("Contains short validation runs; those runs do not establish full-240-frame or contact-heavy performance.")
    return {"readme_command": readme_command, "mesh_format": "geo" if readme_command else "obj",
            "groups": rows, "comparisons": comparisons, "notes": notes}


def _self_test() -> None:
    import tempfile
    import unittest
    from unittest.mock import patch

    class Helpers(unittest.TestCase):
        def test_commands_environment(self):
            for mode, threads in (("cpu", "64"), ("gpu", "1")):
                args = command(Path("/repo"), Path("/output"), mode, 2)
                self.assertEqual(args[args.index("--twist_nx") + 1], "99")
                self.assertEqual(args[args.index("--use_ccd_guess") + 1], "false")
                with patch.dict(os.environ, {"OMP_THREAD_LIMIT": "2", "GOMP_CPU_AFFINITY": "1", "KMP_AFFINITY": "bad"}):
                    env = environment(mode)
                    self.assertEqual(env["OMP_NUM_THREADS"], threads)
                    self.assertEqual(env["OMP_WAIT_POLICY"], "ACTIVE" if mode == "cpu" else "PASSIVE")
                    self.assertNotIn("OMP_THREAD_LIMIT", env)
                    self.assertNotIn("GOMP_CPU_AFFINITY", env)
                    self.assertNotIn("KMP_AFFINITY", env)
            with self.assertRaises(ValueError):
                command(Path("/repo"), Path("/output"), "unknown", 2)

        def test_readme_commands(self):
            verify_readme_command(Path(__file__).resolve().parent.parent)
            cpu = command(Path("/repo"), Path("/output"), "cpu", 240, readme_command=True)
            gpu = command(Path("/repo"), Path("/output"), "gpu", 240, readme_command=True)
            self.assertEqual(cpu[1:], gpu[1:])
            self.assertEqual(cpu[cpu.index("--format") + 1], "geo")
            for flag in ("--use_ccd_guess", "--use_colored_ccd_guess", "--fps", "--density",
                         "--damping", "--write_substeps", "--friction_coefficient"):
                self.assertNotIn(flag, cpu)
            self.assertEqual(environment("gpu", readme_command=True)["OMP_WAIT_POLICY"], "ACTIVE")
            self.assertEqual(environment("gpu", readme_command=True)["OMP_NUM_THREADS"], "64")
            self.assertEqual(environment("cpu", readme_command=True), environment("gpu", readme_command=True))
            with tempfile.TemporaryDirectory(prefix="ipc_readme_command_") as temporary:
                root = Path(temporary)
                (root / "README.md").write_text("# Example 1:\n./build/3D_sim --example 1\n# Example 2:\n")
                with self.assertRaises(ValueError):
                    verify_readme_command(root)

        def test_geo_validation_and_drift(self):
            with tempfile.TemporaryDirectory(prefix="ipc_geo_helper_") as temporary:
                base = Path(temporary)
                tuples = [[0., 0., 0.] for _ in range(VERTICES)]
                indices = [0, 1, 2] * TRIANGLES
                run = ["startvertex", 0, "nprimitives", TRIANGLES, "nvertices_rle", [3, TRIANGLES]]
                mesh = ["fileversion", "18.5.408", "hasindex", False,
                        "pointcount", VERTICES, "vertexcount", 3 * TRIANGLES, "primitivecount", TRIANGLES,
                        "topology", ["pointref", ["indices", indices]],
                        "attributes", ["pointattributes", [
                            [["scope", "public", "type", "numeric", "name", "P", "options", {}],
                             ["size", 3, "storage", "fpreal32", "values",
                              ["size", 3, "storage", "fpreal32", "tuples", tuples]]]]],
                        "primitives", [[["type", "Polygon_run"], run]]]
                trials = []
                for mode, timing in (("cpu", 1000), ("gpu", 2000)):
                    outdir = base / mode
                    outdir.mkdir()
                    for frame in (0, 1):
                        tuples[0][0] = .125 if mode == "gpu" and frame == 1 else 0.
                        (outdir / f"frame_{frame:04d}.geo").write_text(json.dumps(mesh))
                    log = base / (mode + ".log")
                    log.write_text(f"Vertices: 10000\nFrame 1 | global_iters = 20 | solver_time = {timing}.000 ms\n"
                                   f"Simulation finished.\nTotal solver time: {timing}.000 ms\n"
                                   f"Average solver time: {timing}.000 ms/frame\nTotal simulation time: {timing + 10}.000 ms\n")
                    trial = parse_trial(log, outdir, 1, mode, readme_command=True)
                    self.assertEqual(trial["mesh_format"], "geo")
                    self.assertNotIn("final_obj", trial)
                    trials.append(trial)
                    with self.assertRaises(ValueError):
                        parse_trial(log, outdir, 1, mode)
                summary = summarize(trials)
                self.assertTrue(summary["readme_command"])
                self.assertEqual(summary["comparisons"][0]["cpu_over_gpu_solver_speedup"], .5)
                self.assertEqual(summary["comparisons"][0]["final_position_drift"][0]["final_positions"]["max_vertex_l2_m"], .125)
                with self.assertRaises(ValueError):
                    summarize([trials[0], dict(trials[1], readme_command=False)])
                path = base / "bad.geo"
                for bad in (float("nan"), float("inf"), True, "0"):
                    tuples[0][0] = bad
                    path.write_text(json.dumps(mesh))
                    with self.assertRaises(ValueError):
                        _geo_positions(path)
                tuples[0][0] = 0.
                indices[0] = VERTICES
                path.write_text(json.dumps(mesh))
                with self.assertRaises(ValueError):
                    _geo_positions(path)
                indices[0] = 0
                run[-1] = [4, TRIANGLES]
                path.write_text(json.dumps(mesh))
                with self.assertRaises(ValueError):
                    _geo_positions(path)
                run[-1] = [3, TRIANGLES]
                path.write_text(json.dumps(mesh + ["pointcount", VERTICES]))
                with self.assertRaises(ValueError):
                    _geo_positions(path)
                path.write_text(json.dumps(mesh)[:-1])
                with self.assertRaises(ValueError):
                    _geo_positions(path)

        def test_parse_summary_and_rejection(self):
            with tempfile.TemporaryDirectory(prefix="ipc_example1_helper_") as temporary:
                base = Path(temporary)
                mesh = "v 0 0 0\n" * VERTICES + "f 1 2 3\n" * TRIANGLES
                trials = []
                for mode, timings in (("cpu", (1000, 2000)), ("gpu", (500, 1000))):
                    output = base / mode
                    output.mkdir()
                    for frame in range(3):
                        (output / f"frame_{frame:04d}.obj").write_text(mesh)
                    if mode == "gpu":
                        (output / "frame_0002.obj").write_text(mesh.replace("v 0 0 0", "v 0.1 0 0", 1))
                    log = base / f"{mode}.log"
                    log.write_text(f"Vertices:  10000\nFrame 1 | global_iters = 20 | solver_time = {timings[0]:.3f} ms\n"
                                   f"Frame 2 | global_iters = 20 | solver_time = {timings[1]:.3f} ms\n"
                                   f"Simulation finished.\nTotal simulation time: {sum(timings) + 10:.3f} ms\n"
                                   f"Total solver time: {sum(timings):.3f} ms\nAverage solver time: {sum(timings)/2:.3f} ms/frame\n")
                    trials.append(parse_trial(log, output, 2, mode))
                self.assertEqual(trials[0]["average_substep_s"], .3)
                self.assertEqual(trials[0]["frame2_onward_solver_s"], 2)
                result = summarize(trials)
                self.assertEqual(result["comparisons"][0]["cpu_over_gpu_solver_speedup"], 2)
                drift = result["comparisons"][0]["final_position_drift"][0]
                self.assertAlmostEqual(drift["final_positions"]["max_vertex_l2_m"], .1)
                self.assertEqual(drift["initial_positions"]["max_vertex_l2_m"], 0)
                with self.assertRaises(ValueError):
                    summarize([dict(trials[0], contaminated=True)])
                final = Path(trials[1]["final_obj"])
                final.write_text(mesh.replace("v 0 0 0", "v nan 0 0", 1))
                with self.assertRaises(ValueError):
                    parse_trial(Path(trials[1]["log"]), Path(trials[1]["outdir"]), 2, "gpu")
                final.unlink()
                with self.assertRaises(ValueError):
                    parse_trial(Path(trials[1]["log"]), Path(trials[1]["outdir"]), 2, "gpu")
                bad_log = Path(trials[0]["log"])
                bad_log.write_text(bad_log.read_text().replace("3000.000 ms", "3001.000 ms"))
                with self.assertRaises(ValueError):
                    parse_trial(bad_log, Path(trials[0]["outdir"]), 2, "cpu")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Helpers)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        parser.print_help()
