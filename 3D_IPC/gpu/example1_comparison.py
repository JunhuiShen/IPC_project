#!/usr/bin/env python3
"""Pure-data helpers for the queued full-resolution Example 1 comparison.

No simulations are launched here. Every comparison binds one explicit, named
protocol, because the documented Example 1 settings changed over time and two
different runs must never be mixed or compared as if they were the same work:

``readme_example1``
    The Example 1 command currently documented in the parent README, plus its
    OpenMP environment. ``verify_readme_command`` parses and binds that block
    from README.md and requires it to match the snapshot below. The command
    selects ``--exact_computation_fallback false``; the GPU device solver now
    runs its original non-exact mode for this deformable cloth protocol.
``historical_example1``
    The former README profile (240 frames, 5 substeps, 4 iterations = 20
    global iterations per frame, exact fallback left at its default ``true``,
    GEO output). The recorded historical totals of GPU 175 s and CPU
    36.916682 s total solver time belong to this protocol only. They are not a
    baseline for current README settings, and this harness does not reproduce
    them.
``legacy_colored_guess``
    The earlier colored-guess OBJ benchmark (10K vertices, 5 substeps,
    4 sweeps), still accepted by both implementations.

Mesh writing is outside the printed solver timer; GPU frame downloads remain
inside that timer. Coordinate drift is reported, never asserted to be zero.
``python3 gpu/example1_comparison.py --self-test`` tests the helpers with
temporary synthetic logs/meshes; it does not use the server or CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import json
import os
from pathlib import Path
import re
import shlex
from statistics import median

VERTICES = 10_000
TRIANGLES = 19_602
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_README_FIRST_LINE = "# Example 1:"
_README_LAST_LINE = "# Example 2:"
_ENVIRONMENT = re.compile(r"\A([A-Z][A-Z0-9_]*)=(.*)\Z")


class UnsupportedProtocol(ValueError):
    """The named protocol cannot run on the requested implementation."""


@dataclass(frozen=True)
class Protocol:
    """One frozen command line, environment and output contract."""

    name: str
    description: str
    frames: int
    substeps: int
    iterations: int
    mesh_format: str
    omp: dict
    argv: tuple
    gpu_rejection: str | None
    recorded_totals_s: dict | None
    provenance: str

    @property
    def global_iters(self) -> int:
        """Fixed solver sweeps per frame; both implementations run --fixed_iters."""
        return self.substeps * self.iterations

    def check_frames(self, frames: int) -> int:
        frames = _frames(frames)
        if frames > self.frames:
            raise ValueError(f"protocol {self.name} documents at most {self.frames} frames")
        return frames

    def check_mode(self, mode: str) -> str:
        mode = _mode(mode)
        if mode == "gpu" and self.gpu_rejection:
            raise UnsupportedProtocol(f"{self.name} cannot run on the GPU: {self.gpu_rejection}")
        return mode


README_PROTOCOL = Protocol(
    name="readme_example1",
    description="current README Example 1 command and environment",
    frames=240,
    substeps=3,
    iterations=3,
    mesh_format="geo",
    omp={"cpu": {"OMP_NUM_THREADS": "64", "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "spread",
                 "OMP_PLACES": "cores", "OMP_WAIT_POLICY": "ACTIVE"},
         # Host execution resources only. The device solver models the CPU
         # dispatch from the logical 64-thread reference team, never from this
         # host team, so one host thread is enough for setup/output.
         "gpu": {"OMP_NUM_THREADS": "1", "OMP_THREAD_LIMIT": "1",
                 "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "spread",
                 "OMP_PLACES": "cores", "OMP_WAIT_POLICY": "ACTIVE"}},
    argv=("{executable}", "--example", "1", "--exact_computation_fallback", "false",
          "--num_frames", "{frames}",
          "--E", "115000", "--nu", "0.25", "--kB", "0.009", "--kpin", "1e9", "--twist_rate", "0.5",
          "--d_hat", "0.005", "--k_barrier", "100",
          "--node_box_min", "0.001", "--node_box_max", "0.01",
          "--fixed_iters", "--substeps", "3", "--max_substep_iters", "3",
          "--node_box_update_count", "3",
          "--use_basic_experimental", "true", "--use_simd", "true", "--use_parallel", "true",
          "--use_ccd_guess", "false", "--use_colored_ccd_guess", "false",
          "--use_colored_ccd_guess_level_based", "true",
          "--colored_ccd_guess_iters", "1", "--colored_ccd_guess_fraction", "0.97",
          "--outdir", "{outdir}", "--format", "geo"),
    gpu_rejection=None,
    recorded_totals_s=None,
    provenance="bound from README.md by verify_readme_command(repo)",
)

HISTORICAL_PROTOCOL = Protocol(
    name="historical_example1",
    description=(
        "former README Example 1 profile: 240 frames, 5 substeps x 4 iterations "
        "(20 global iterations), exact fallback at its documented default true, GEO output"),
    frames=240,
    substeps=5,
    iterations=4,
    mesh_format="geo",
    omp={"cpu": {"OMP_NUM_THREADS": "64", "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "close",
                 "OMP_PLACES": "cores", "OMP_WAIT_POLICY": "ACTIVE"},
         "gpu": {"OMP_NUM_THREADS": "64", "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "close",
                 "OMP_PLACES": "cores", "OMP_WAIT_POLICY": "ACTIVE"}},
    argv=("{executable}", "--example", "1", "--num_frames", "{frames}",
          "--E", "115000", "--nu", "0.25", "--kB", "0.009", "--kpin", "1e9", "--twist_rate", "0.5",
          "--d_hat", "0.005", "--k_barrier", "100",
          "--node_box_min", "0.001", "--node_box_max", "0.01",
          "--fixed_iters", "--substeps", "5", "--max_substep_iters", "4",
          "--node_box_update_count", "10",
          "--use_basic_experimental", "true", "--use_simd", "true", "--use_parallel", "true",
          "--outdir", "{outdir}", "--format", "geo"),
    gpu_rejection=None,
    # Recorded historical evidence for this exact protocol, kept separate from
    # every current measurement. Not re-measured here and not comparable with
    # the current README profile or the legacy short benchmark.
    recorded_totals_s={"gpu": 175.0, "cpu": 36.916682},
    provenance=(
        "frozen former README profile; recorded totals GPU 175 s and CPU 36.916682 s "
        "total solver time belong to this protocol"),
)

LEGACY_PROTOCOL = Protocol(
    name="legacy_colored_guess",
    description="earlier colored-guess OBJ benchmark: 10K vertices, 5 substeps x 4 sweeps",
    frames=240,
    substeps=5,
    iterations=4,
    mesh_format="obj",
    omp={"cpu": {"OMP_NUM_THREADS": "64", "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "close",
                 "OMP_PLACES": "cores", "OMP_WAIT_POLICY": "ACTIVE"},
         "gpu": {"OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "close",
                 "OMP_PLACES": "cores", "OMP_WAIT_POLICY": "PASSIVE"}},
    argv=("{executable}",
          "--example", "1", "--twist_nx", "99", "--twist_ny", "99",
          "--num_frames", "{frames}", "--fps", "30",
          "--E", "115000", "--nu", "0.25", "--kB", "0.009",
          "--kpin", "1e9", "--twist_rate", "0.5",
          "--density", "900", "--thickness", "0.001",
          "--d_hat", "0.005", "--k_barrier", "100",
          "--friction_coefficient", "0", "--use_ccd", "true",
          "--use_ticcd", "false", "--exact_computation_fallback", "true",
          "--node_box_min", "0.001", "--node_box_max", "0.01",
          "--fixed_iters", "--substeps", "5",
          "--max_substep_iters", "4", "--node_box_update_count", "10",
          "--use_basic_experimental", "true", "--use_simd", "true",
          "--use_parallel", "true", "--use_cloth_grid", "false",
          "--use_ogc", "false", "--use_ogc_solver", "false",
          "--use_verlet_guess", "false", "--use_translation_guess", "false",
          "--use_ccd_guess", "false", "--use_colored_ccd_guess", "true",
          "--colored_ccd_guess_iters", "10", "--damping", "1",
          "--write_substeps", "false", "--format", "obj",
          "--outdir", "{outdir}"),
    gpu_rejection=None,
    recorded_totals_s=None,
    provenance="earlier colored-guess benchmark, unchanged",
)

PROTOCOLS = {spec.name: spec for spec in (README_PROTOCOL, HISTORICAL_PROTOCOL, LEGACY_PROTOCOL)}
DEFAULT_PROTOCOL = LEGACY_PROTOCOL.name


def _mode(mode: str) -> str:
    if mode not in ("cpu", "gpu"):
        raise ValueError("mode must be 'cpu' or 'gpu'")
    return mode


def _frames(frames: int) -> int:
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError("frames must be a positive integer")
    return frames


def protocol_spec(name: str = DEFAULT_PROTOCOL) -> Protocol:
    try:
        return PROTOCOLS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown comparison protocol {name!r}; choose one of {sorted(PROTOCOLS)}") from error


def _expand(spec: Protocol, executable: str, outdir: str, frames: int) -> list[str]:
    values = {"executable": executable, "outdir": outdir, "frames": str(frames)}
    return [token.format(**values) for token in spec.argv]


def command(repo: Path, outdir: Path, mode: str, frames: int, *,
            protocol: str = DEFAULT_PROTOCOL) -> list[str]:
    """Return argv for one named protocol; caller selects cwd, affinity and launch."""
    spec = protocol_spec(protocol)
    mode = spec.check_mode(mode)
    frames = spec.check_frames(frames)
    executable = "build/3D_sim" if mode == "cpu" else "gpu/build/3D_sim_gpu"
    return _expand(spec, str(Path(repo).resolve() / executable), str(Path(outdir).resolve()), frames)


def environment(mode: str, *, protocol: str = DEFAULT_PROTOCOL) -> dict[str, str]:
    """Fresh OpenMP policy for the protocol; process affinity is the caller's job."""
    spec = protocol_spec(protocol)
    mode = _mode(mode)  # Launch gating belongs to command(); this reports the policy.
    policy = spec.omp[mode]
    result = {key: value for key, value in os.environ.items()
              if not key.startswith(("OMP_", "GOMP_", "KMP_"))}
    result.update(policy)
    return result


def _readme_binding(repo: Path) -> tuple[dict, list[str]]:
    """Parse the documented Example 1 environment and argv into bindable tokens."""
    readme = (Path(repo) / "README.md").read_text(encoding="utf-8")
    try:
        # Keep the marker so the header comment line is skipped like any other.
        block = _README_FIRST_LINE + readme.split(_README_FIRST_LINE, 1)[1].split(
            _README_LAST_LINE, 1)[0]
    except IndexError as error:
        raise ValueError("Cannot locate the README Example 1 command") from error
    tokens: list[str] = []
    for line in block.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.endswith("\\"):
            text = text[:-1].rstrip()
        tokens.extend(shlex.split(text))
    documented_environment: dict[str, str] = {}
    documented: list[str] = []
    for token in tokens:
        match = _ENVIRONMENT.match(token)
        if match and not documented:
            documented_environment[match[1]] = match[2]
        else:
            documented.append(token)
    if not documented or documented[0] != "./build/3D_sim":
        raise ValueError("Cannot locate the README Example 1 command")
    argv = documented[1:]
    if argv.count("--outdir") != 1:
        raise ValueError("README Example 1 command must name exactly one --outdir")
    outdir = argv[argv.index("--outdir") + 1]
    if Path(outdir).is_absolute() or ".." in Path(outdir).parts:
        raise ValueError("README Example 1 --outdir must stay relative to the repository")
    argv[argv.index("--outdir") + 1] = "{outdir}"
    return documented_environment, ["{executable}", *argv]


def _flags(argv: list[str]) -> dict:
    """Split flag tokens, rejecting duplicates and stray positionals."""
    result: dict = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected positional argument {token!r}")
        if token in result:
            raise ValueError(f"duplicate argument {token}")
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            result[token] = argv[index + 1]
            index += 2
        else:
            result[token] = True
            index += 1
    return result


def verify_readme_command(repo: Path) -> Protocol:
    """Bind the live README Example 1 command and refuse any drift from the snapshot."""
    spec = protocol_spec("readme_example1")
    documented_environment, documented = _readme_binding(repo)
    expected = _expand(spec, "{executable}", "{outdir}", spec.frames)
    if documented_environment != spec.omp["cpu"]:
        raise ValueError(
            "README Example 1 OpenMP environment changed: documented "
            f"{documented_environment} but the bound profile uses {spec.omp['cpu']}")
    if documented != expected:
        for position, (actual, wanted) in enumerate(zip(documented, expected)):
            if actual != wanted:
                raise ValueError(
                    "README Example 1 command changed at argument "
                    f"{position}: documented {actual!r}, bound {wanted!r}; "
                    "re-bind README_PROTOCOL in gpu/example1_comparison.py")
        raise ValueError(
            f"README Example 1 command length changed: documented {len(documented)} "
            f"arguments, bound {len(expected)}")
    flags = _flags(documented[1:])
    for flag, wanted in (("--num_frames", str(spec.frames)), ("--substeps", str(spec.substeps)),
                         ("--max_substep_iters", str(spec.iterations)),
                         ("--format", spec.mesh_format)):
        if flags.get(flag) != wanted:
            raise ValueError(f"README Example 1 {flag} is {flags.get(flag)!r}, bound {wanted!r}")
    return spec


def verify_protocol_contract(name: str) -> Protocol:
    """Check a frozen non-README protocol against its declared work contract."""
    spec = protocol_spec(name)
    if spec.name == README_PROTOCOL.name:
        raise ValueError("readme_example1 is bound from README.md; call verify_readme_command(repo)")
    argv = _expand(spec, "EXECUTABLE", "OUTPUT", spec.frames)
    if argv[0] != "EXECUTABLE":
        raise ValueError(f"{spec.name} argv must start with the executable placeholder")
    flags = _flags(argv[1:])
    for flag, wanted in (("--num_frames", str(spec.frames)), ("--substeps", str(spec.substeps)),
                         ("--max_substep_iters", str(spec.iterations)),
                         ("--format", spec.mesh_format), ("--outdir", "OUTPUT")):
        if flags.get(flag) != wanted:
            raise ValueError(f"{spec.name} argv disagrees with its declared {flag}={wanted!r}")
    return spec


def record_protocol(record: dict) -> str:
    """Resolve a recorded protocol, including older status.json records.

    Older harness records stored a boolean ``readme_command`` flag. ``true``
    meant the README profile *at that time*, which is now the named historical
    protocol, never the current README profile.
    """
    name = record.get("protocol")
    if name is None:
        name = HISTORICAL_PROTOCOL.name if record.get("readme_command") else LEGACY_PROTOCOL.name
    return protocol_spec(name).name


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


def parse_trial(log: Path, outdir: Path, frames: int, mode: str, *,
                protocol: str = DEFAULT_PROTOCOL) -> dict:
    """Validate a successful fresh run of one named protocol.

    The caller checks the process return code. Per-frame solver iteration
    counts must equal this protocol's fixed substeps x iterations budget; a
    value read from the log alone never defines the expected work.
    """
    spec = protocol_spec(protocol)
    frames = spec.check_frames(frames)
    mode = _mode(mode)  # A stored log is validated structurally, not gated per mode.
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
    if any(int(entry[1]) != spec.global_iters for entry in entries):
        raise ValueError(
            f"unexpected solver iteration count; protocol {spec.name} fixes "
            f"{spec.substeps} substeps x {spec.iterations} iterations = "
            f"{spec.global_iters} global iterations per frame")
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
    mesh_format = spec.mesh_format
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
        "mode": mode, "frames": frames, "complete": True, "protocol": spec.name,
        "protocol_description": spec.description,
        "substeps": spec.substeps, "iterations": spec.iterations,
        "global_iters": spec.global_iters, "mesh_format": mesh_format,
        "vertices": VERTICES, "triangles": TRIANGLES,
        "log": str(log), "outdir": str(outdir),
        "initial_mesh": str(outdir / f"frame_0000.{mesh_format}"),
        "final_mesh": str(outdir / f"frame_{frames:04d}.{mesh_format}"),
        "total_solver_s": total_s, "average_frame_s": total_s / frames,
        "average_substep_s": total_s / (frames * spec.substeps),
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
    if not trials:
        raise ValueError("summarize needs at least one parsed trial")
    protocols = {record_protocol(trial) for trial in trials}
    if len(protocols) != 1:
        raise ValueError(
            f"Do not mix comparison protocols in one summary: {sorted(protocols)}")
    protocol = protocols.pop()
    spec = protocol_spec(protocol)
    for trial in trials:
        if trial.get("protocol") not in (None, spec.name):
            raise ValueError("trial protocol field disagrees with the recorded protocol")
        if trial.get("mesh_format", spec.mesh_format) != spec.mesh_format:
            raise ValueError(f"trial mesh format disagrees with protocol {spec.name}")
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
    if spec.name == README_PROTOCOL.name:
        notes += [
            "Both runs pass the current README Example 1 arguments without substitutions: GEO output, "
            f"{spec.substeps} substeps, {spec.iterations} iterations ({spec.global_iters} global iterations per frame).",
            "The CPU run uses the documented OMP_NUM_THREADS=64, OMP_DYNAMIC=FALSE, OMP_WAIT_POLICY=ACTIVE, "
            "OMP_PROC_BIND=spread and OMP_PLACES=cores with 64 physical-CPU affinity. The GPU run pins "
            "OMP_NUM_THREADS=1/OMP_THREAD_LIMIT=1: those are host execution resources for device "
            "setup/output, not the CUDA worker count and not the modelled CPU dispatch, which stays "
            "the logical 64-thread reference team.",
            "The documented level-based colored guess and adaptive node-box policy are used on CPU and GPU; "
            "coloring, arithmetic and per-vertex CCD clipping still differ.",
            "GEO position drift uses the exported 10-significant-digit text values, not full-precision checkpoints; Houdini's P metadata specifies fpreal32.",
            "Not comparable to the historical 240-frame, 5-substep, 4-iteration exact-fallback protocol or its "
            "recorded GPU 175 s / CPU 36.916682 s totals.",
        ]
    elif spec.name == HISTORICAL_PROTOCOL.name:
        notes += [
            "Historical protocol: former README Example 1 arguments (GEO output, "
            f"{spec.substeps} substeps, {spec.iterations} iterations = {spec.global_iters} global iterations "
            "per frame, exact fallback at its default true, node box update every 10 iterations).",
            "Both use OMP_NUM_THREADS=64, OMP_DYNAMIC=FALSE, OMP_WAIT_POLICY=ACTIVE and the same "
            "64 physical-CPU affinity; GPU OpenMP controls host setup/output, not CUDA worker count.",
            "Recorded evidence for this frozen protocol is GPU 175 s and CPU 36.916682 s total solver time. "
            "These measurements are separate from it and do not re-establish that record.",
            "Coloring, arithmetic and per-vertex CCD clipping still differ between implementations.",
        ]
    else:
        notes += [
            "Legacy comparison: 10K vertices, "
            f"{spec.substeps} substeps, {spec.iterations} sweeps, colored CCD guess 10, "
            "collision on, friction zero, OBJ output.",
            "Legacy CPU adaptive 0.001-0.01 boxes differ from fixed 0.01 GPU boxes; CCD acceptance and coloring also differ.",
        ]
    if any(trial["frames"] < spec.frames for trial in trials):
        notes.append(
            f"Contains short validation runs; those runs do not establish full-{spec.frames}-frame "
            "or contact-heavy performance.")
    return {"protocol": spec.name, "protocol_description": spec.description,
            "protocol_provenance": spec.provenance,
            "recorded_protocol_totals_s": (
                dict(spec.recorded_totals_s) if spec.recorded_totals_s is not None else None),
            "substeps": spec.substeps, "iterations": spec.iterations,
            "global_iters": spec.global_iters, "mesh_format": spec.mesh_format,
            "groups": rows, "comparisons": comparisons, "notes": notes}


def _self_test() -> None:
    import tempfile
    import unittest
    from unittest.mock import patch

    class Helpers(unittest.TestCase):
        def test_commands_environment(self):
            # Legacy colored-guess benchmark keeps its earlier pinning policy.
            for mode, threads in (("cpu", "64"), ("gpu", "1")):
                args = command(Path("/repo"), Path("/output"), mode, 2)
                self.assertEqual(args[args.index("--twist_nx") + 1], "99")
                self.assertEqual(args[args.index("--use_ccd_guess") + 1], "false")
                self.assertEqual(args[args.index("--format") + 1], "obj")
                with patch.dict(os.environ, {"OMP_THREAD_LIMIT": "2", "GOMP_CPU_AFFINITY": "1", "KMP_AFFINITY": "bad"}):
                    env = environment(mode)
                    self.assertEqual(env["OMP_NUM_THREADS"], threads)
                    self.assertEqual(env["OMP_WAIT_POLICY"], "ACTIVE" if mode == "cpu" else "PASSIVE")
                    self.assertNotIn("OMP_THREAD_LIMIT", env)
                    self.assertNotIn("GOMP_CPU_AFFINITY", env)
                    self.assertNotIn("KMP_AFFINITY", env)
            with self.assertRaises(ValueError):
                command(Path("/repo"), Path("/output"), "unknown", 2)
            with self.assertRaises(ValueError):
                command(Path("/repo"), Path("/output"), "cpu", 2, protocol="invented")

            # Frozen historical profile keeps 20 fixed global iterations.
            historical = verify_protocol_contract("historical_example1")
            self.assertEqual((historical.substeps, historical.iterations), (5, 4))
            self.assertEqual(historical.global_iters, 20)
            self.assertEqual(historical.recorded_totals_s, {"gpu": 175.0, "cpu": 36.916682})
            arguments = command(Path("/repo"), Path("/output"), "gpu", 3,
                                protocol="historical_example1")
            self.assertEqual(arguments[arguments.index("--substeps") + 1], "5")
            self.assertEqual(arguments[arguments.index("--max_substep_iters") + 1], "4")
            self.assertEqual(arguments[arguments.index("--format") + 1], "geo")
            self.assertNotIn("--exact_computation_fallback", arguments)
            self.assertEqual(environment("gpu", protocol="historical_example1")["OMP_NUM_THREADS"], "64")
            with self.assertRaises(ValueError):
                verify_protocol_contract("readme_example1")
            # Older status.json records stored readme_command=true for the
            # README profile of that time: the historical protocol, not today's.
            self.assertEqual(record_protocol({"readme_command": True}), "historical_example1")
            self.assertEqual(record_protocol({"readme_command": False}), "legacy_colored_guess")
            self.assertEqual(record_protocol({"protocol": "readme_example1"}), "readme_example1")

        def test_readme_commands(self):
            spec = verify_readme_command(Path(__file__).resolve().parent.parent)
            self.assertEqual(spec.name, "readme_example1")
            self.assertEqual((spec.substeps, spec.iterations, spec.global_iters), (3, 3, 9))
            self.assertEqual(spec.mesh_format, "geo")
            cpu = command(Path("/repo"), Path("/output"), "cpu", 240, protocol="readme_example1")
            self.assertEqual(cpu[1:], _expand(spec, cpu[0], str(Path("/output").resolve()), 240)[1:])
            self.assertEqual(cpu[cpu.index("--format") + 1], "geo")
            self.assertEqual(cpu[cpu.index("--substeps") + 1], "3")
            self.assertEqual(cpu[cpu.index("--max_substep_iters") + 1], "3")
            self.assertEqual(cpu[cpu.index("--exact_computation_fallback") + 1], "false")
            self.assertEqual(cpu[cpu.index("--use_colored_ccd_guess_level_based") + 1], "true")
            for flag in ("--fps", "--density", "--damping", "--write_substeps",
                         "--friction_coefficient"):
                self.assertNotIn(flag, cpu)
            # The documented environment binds both modes to the README policy.
            self.assertEqual(environment("gpu", protocol="readme_example1")["OMP_WAIT_POLICY"], "ACTIVE")
            self.assertEqual(environment("gpu", protocol="readme_example1")["OMP_PROC_BIND"], "spread")
            cpu_env = environment("cpu", protocol="readme_example1")
            gpu_env = environment("gpu", protocol="readme_example1")
            # GPU host execution resources are pinned to one thread; the
            # README CPU environment (and the immutable README bytes) keep 64.
            self.assertEqual(cpu_env["OMP_NUM_THREADS"], "64")
            self.assertEqual(gpu_env["OMP_NUM_THREADS"], "1")
            self.assertEqual(gpu_env["OMP_THREAD_LIMIT"], "1")
            self.assertNotIn("OMP_THREAD_LIMIT", cpu_env)
            for key in ("OMP_DYNAMIC", "OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES"):
                self.assertEqual(cpu_env[key], gpu_env[key])
            self.assertEqual({key for key in cpu_env if key != "OMP_THREAD_LIMIT"},
                             {key for key in gpu_env if key != "OMP_THREAD_LIMIT"})
            # The current README profile is wired to the GPU original mode.
            gpu = command(Path("/repo"), Path("/output"), "gpu", 240, protocol="readme_example1")
            self.assertTrue(gpu[0].endswith("gpu/build/3D_sim_gpu"))
            self.assertEqual(gpu[gpu.index("--exact_computation_fallback") + 1], "false")
            self.assertEqual(gpu[gpu.index("--use_colored_ccd_guess_level_based") + 1], "true")
            self.assertEqual(environment("gpu", protocol="readme_example1")["OMP_PROC_BIND"], "spread")
            with self.assertRaises(ValueError):
                command(Path("/repo"), Path("/output"), "cpu", 241, protocol="readme_example1")
            with tempfile.TemporaryDirectory(prefix="ipc_readme_command_") as temporary:
                root = Path(temporary)
                (root / "README.md").write_text("# Example 1:\n./build/3D_sim --example 1\n# Example 2:\n")
                with self.assertRaises(ValueError):
                    verify_readme_command(root)
                # The former README command must not pass as the current profile.
                old = "\n".join([
                    "# Example 1: old",
                    "OMP_NUM_THREADS=64 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_PLACES=cores \\",
                    "./build/3D_sim --example 1 --num_frames 240 \\",
                    "  --fixed_iters --substeps 5 --max_substep_iters 4 --node_box_update_count 10 \\",
                    "  --use_basic_experimental true --use_simd true --use_parallel true \\",
                    "  --outdir example1_output --format geo",
                    "# Example 2:"])
                (root / "README.md").write_text(old)
                with self.assertRaises(ValueError):
                    verify_readme_command(root)
                current = (Path(__file__).resolve().parent.parent / "README.md").read_text()
                head, tail = current.split("# Example 1:", 1)
                (root / "README.md").write_text(head + "# Example 1:" + tail.replace(
                    "--substeps 3", "--substeps 6", 1))
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
                    log.write_text(f"Vertices: 10000\nFrame 1 | global_iters = 9 | solver_time = {timing}.000 ms\n"
                                   f"Simulation finished.\nTotal solver time: {timing}.000 ms\n"
                                   f"Average solver time: {timing}.000 ms/frame\nTotal simulation time: {timing + 10}.000 ms\n")
                    trial = parse_trial(log, outdir, 1, mode, protocol="readme_example1")
                    self.assertEqual(trial["mesh_format"], "geo")
                    self.assertEqual(trial["protocol"], "readme_example1")
                    self.assertEqual(trial["global_iters"], 9)
                    self.assertNotIn("final_obj", trial)
                    trials.append(trial)
                    # The same 9-iteration log cannot pass as the 20-iteration protocol.
                    with self.assertRaises(ValueError):
                        parse_trial(log, outdir, 1, mode, protocol="historical_example1")
                    # Default (legacy) expects OBJ output, so these GEO files are rejected.
                    with self.assertRaises(ValueError):
                        parse_trial(log, outdir, 1, mode)
                summary = summarize(trials)
                self.assertEqual(summary["protocol"], "readme_example1")
                self.assertEqual((summary["substeps"], summary["iterations"]), (3, 3))
                self.assertEqual(summary["mesh_format"], "geo")
                self.assertIsNone(summary["recorded_protocol_totals_s"])
                self.assertTrue(any("Not comparable to the historical" in note
                                    for note in summary["notes"]))
                self.assertEqual(summary["comparisons"][0]["cpu_over_gpu_solver_speedup"], .5)
                self.assertEqual(summary["comparisons"][0]["final_position_drift"][0]["final_positions"]["max_vertex_l2_m"], .125)
                with self.assertRaises(ValueError):
                    summarize([trials[0], dict(trials[1], protocol="historical_example1")])
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
                self.assertEqual(trials[0]["protocol"], "legacy_colored_guess")
                self.assertEqual(trials[0]["average_substep_s"], .3)
                self.assertEqual(trials[0]["frame2_onward_solver_s"], 2)
                result = summarize(trials)
                self.assertEqual(result["protocol"], "legacy_colored_guess")
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
