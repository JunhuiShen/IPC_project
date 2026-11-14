"""Run our equations inside the imported Newton solver's GPU structure."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "gpu"


def _runtime():
    from gpu import _vendor
    root = _vendor
    package = root / "newton"
    loaded = sys.modules.get("newton")
    if loaded is not None and not Path(getattr(loaded, "__file__", "")).resolve().is_relative_to(package):
        raise RuntimeError("An external Newton package is already imported; start a fresh process")
    sys.path.insert(0, str(root))
    try:
        import warp as wp
        import newton
    except ImportError as error:
        raise RuntimeError("Install gpu/newton_dat/requirements-cuda12.txt in a Python3.11 environment") from error
    if not re.fullmatch(r"1\.19\.0(?:\+[^\s]+)?", wp.__version__):
        raise RuntimeError(f"Expected the validated Warp1.19.0 toolchain, found {wp.__version__}")
    if not Path(newton.__file__).resolve().is_relative_to(package):
        raise RuntimeError("Newton did not resolve to the imported project source")
    return newton, wp


def _cuda(value):
    if not re.fullmatch(r"cuda(?::[0-9]+)?", value):
        raise argparse.ArgumentTypeError("Use a CUDA device such as cuda:0; CPU simulation is not a production fallback")
    return value


def _nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("Value must be nonnegative")
    return result


def _positive(value):
    result = _nonnegative(value)
    if result == 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return result


def signature(payload, solver_mode="ipc"):
    immutable = {key: value for key, value in payload.items() if key != "output"}
    immutable["_gpu_solver"] = solver_mode
    return hashlib.sha256(json.dumps(immutable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class FrameGraph:
    """Actual Newton State/control objects and captured inherited VBD timesteps."""
    def __init__(self, model, solver, *, graphs=True, initial_snapshot=None, start_frame=0):
        import warp as wp
        from .motion import BoundaryMotion
        from .output import restore

        self.wp, self.model, self.solver = wp, model, solver
        self.states = [model.state(), model.state()]
        self.control = model.control()
        self.motion = BoundaryMotion(model)
        self.substeps = int(model.ipc.params["substeps"])
        self.dt = float(model.ipc.params["dt"])
        self.frames = start_frame
        self.parity = 0
        self.graphs = {}
        if initial_snapshot is not None:
            for state in self.states:
                restore(model, state, initial_snapshot)
        self.motion.reset(start_frame * self.substeps)
        self.motion.apply(self.dt)
        self.capture_seconds = 0.0
        if graphs and model.device.is_cuda:
            started = time.perf_counter()
            prepare = getattr(solver, "prepare_capture", None)
            if prepare:
                prepare()
            for start in (0, 1) if self.substeps % 2 else (0,):
                with wp.ScopedCapture(device=model.device) as capture:
                    self._simulate(start)
                self.graphs[start] = capture.graph
            self.capture_seconds = time.perf_counter() - started

    def _simulate(self, start):
        current = start
        for _ in range(self.substeps):
            self.motion.step(self.dt)
            self.states[current].clear_forces()
            self.solver.step(self.states[current], self.states[1 - current], self.control, None, self.dt)
            current = 1 - current
        return current

    @property
    def state(self):
        return self.states[self.parity]

    def advance(self):
        if self.graphs:
            self.wp.capture_launch(self.graphs[self.parity])
            self.parity = (self.parity + self.substeps) % 2
        else:
            self.parity = self._simulate(self.parity)
        self.solver.check_status()
        self.frames += 1
        return self.state


def run_simulation(exporter, model, solver, *, graphs=True, benchmark=False, benchmark_snapshot=False,
                   warmup=3, test=False, native_arguments=()):
    from . import output

    payload, p = exporter.scene, model.ipc.params
    solver_mode = getattr(solver, "solver_mode", "ipc")
    settings = payload["output"]
    frames, restart = int(settings["num_frames"]), int(settings["restart_frame"])
    substeps, fps = int(p["substeps"]), float(p["fps"])
    if frames < 0 or restart < -1 or restart > frames or substeps <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid frame configuration")
    if settings["format"] not in output.FORMATS:
        raise ValueError("Unsupported native output format")
    directory = Path(settings["outdir"]).expanduser()
    scene_signature = signature(payload, solver_mode)
    initial = None
    if restart >= 0:
        if not directory.is_dir():
            raise ValueError("Restart directory does not exist")
        meta = directory / "newton_ipc_run.json"
        if meta.is_file():
            stored = json.loads(meta.read_text())
            if stored["scene_signature"] != scene_signature or stored.get("solver", "ipc") != solver_mode:
                raise ValueError("Restart scene/material/solver settings differ from the saved run")
        initial = output.read_checkpoint(directory / f"state_{restart:04d}.bin", particle_count=model.particle_count,
                                         body_count=model.body_count, legacy_defaults=output.snapshot(model, model.state()))
    elif not benchmark:
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError(f"Fresh output directory must be empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
    controller = FrameGraph(model, solver, graphs=graphs, initial_snapshot=initial, start_frame=max(0, restart))
    metadata = {"backend": "newton-ipc" if solver_mode == "ipc" else "newton-dat-own-energies",
        "solver": solver_mode, "solver_configuration": getattr(solver, "configuration", {}),
        "contact_capacities": getattr(solver, "contact_capacities", {}),
        "precision": "float32", "device": str(model.device),
        "graph_capture": bool(controller.graphs), "capture_seconds": controller.capture_seconds,
        "scene_signature": scene_signature, "example": payload["example"], "params": p,
        "native_arguments": list(native_arguments), "completed_frame": max(0, restart)}
    suffix = output.FORMATS[settings["format"]]
    static = payload.get("static_geometry", {})
    static_positions = static.get("positions", [])
    static_triangles = static.get("tris", [])

    def save(frame, state):
        values = output.snapshot(model, state)
        output.write_mesh(directory / f"frame_{frame:04d}{suffix}", values["positions"], payload["tris"], settings["format"])
        output.write_checkpoint(directory / f"state_{frame:04d}.bin", values)

    if not benchmark:
        if restart < 0:
            save(0, controller.state)
            if static_positions:
                output.write_mesh(directory / ("static_colliders" + suffix), static_positions, static_triangles, settings["format"])
        output.write_json(directory / "newton_ipc_run.json", metadata)
    measured = []
    started = time.perf_counter()
    for frame in range(restart + 1 if restart >= 0 else 1, frames + 1):
        before = time.perf_counter()
        if settings["write_substeps"] and not benchmark:
            # Requested substep snapshots require explicit synchronization;
            # ordinary frames retain whole-frame CUDA Graph replay.
            for substep in range(substeps):
                controller.motion.step(controller.dt)
                state = controller.states[controller.parity]
                state.clear_forces()
                solver.step(state, controller.states[1 - controller.parity], controller.control, None, controller.dt)
                controller.parity = 1 - controller.parity
                solver.check_status()
                values = output.snapshot(model, controller.state)
                index = (frame - 1) * substeps + substep + 1
                output.write_mesh(directory / f"frame_{index:04d}{suffix}", values["positions"], payload["tris"], settings["format"])
            controller.frames += 1
        else:
            controller.advance()
        if benchmark and benchmark_snapshot:
            # The former native driver includes its final frame snapshot in
            # the solver timer. Match that boundary without writing output.
            output.snapshot(model, controller.state)
        seconds = time.perf_counter() - before
        if frame > max(0, restart) + warmup:
            measured.append(seconds)
        if not benchmark:
            values = output.snapshot(model, controller.state)
            if not settings["write_substeps"]:
                output.write_mesh(directory / f"frame_{frame:04d}{suffix}", values["positions"], payload["tris"], settings["format"])
            output.write_checkpoint(directory / f"state_{frame:04d}.bin", values)
            if static.get("dynamic"):
                boundary = exporter.boundary(frame / fps)
                if boundary.get("static_positions"):
                    output.write_mesh(directory / f"collider_{frame:04d}{suffix}", boundary["static_positions"], static_triangles, settings["format"])
            metadata["completed_frame"] = frame
            output.write_json(directory / "newton_ipc_run.json", metadata)
        if not benchmark or p.get("verbose", False):
            print(json.dumps({"frame": frame, "substeps": substeps, "iterations": substeps * int(p["max_global_iters"]), "seconds": seconds}), flush=True)
    if test:
        output.snapshot(model, controller.state)
    result = {"completed_frame": frames, "substeps": substeps, "iterations_per_substep": int(p["max_global_iters"]),
        "solver": solver_mode,
        "contact_capacities": getattr(solver, "contact_capacities", {}),
        "total_seconds": time.perf_counter() - started, "timed_frames": len(measured), "graph_capture": bool(controller.graphs),
        "capture_seconds": controller.capture_seconds,
        "snapshot_download_timed": bool(benchmark and benchmark_snapshot)}
    if measured:
        result["milliseconds_per_frame"] = 1000 * sum(measured) / len(measured)
        result["fps"] = len(measured) / sum(measured)
    print(json.dumps(result), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog="Original scene flags are forwarded: --example 1 --fps 30 --substeps 10 --max_substep_iters 10 --fixed_iters --twist_nx 9 --twist_ny 9 --format geo --outdir results/newton_ipc")
    parser.add_argument("--device", type=_cuda, default="cuda:0")
    parser.add_argument("--solver", choices=("ipc", "dat", "dat-reference"), default="ipc",
                        help="our GS/CCD, DAT with our energies, or exact upstream DAT")
    parser.add_argument("--stock-cloth-twist", action="store_true",
                        help="for DAT, run the original 2,500-vertex cloth example with our energy terms")
    parser.add_argument("--vertex-contact-capacity", type=_positive, default=128,
                        help="persistent triangle-contact row capacity per mesh vertex")
    parser.add_argument("--edge-contact-capacity", type=_positive, default=256,
                        help="persistent edge-contact row capacity per mesh edge")
    parser.add_argument("--scene-exporter", type=Path)
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="headless timing, without geometry/checkpoint output")
    parser.add_argument("--benchmark-snapshot", action="store_true",
                        help="with --benchmark, time the final per-frame snapshot download like the native driver")
    parser.add_argument("--benchmark-warmup", type=_nonnegative, default=3)
    parser.add_argument("--test", action="store_true", help="validate final finite state and device errors")
    parser.add_argument("--check", action="store_true")
    options, native = parser.parse_known_args(argv)
    try:
        if options.solver in ("dat", "dat-reference"):
            from gpu import select_newton_runtime
            select_newton_runtime("reference")
        _, wp = _runtime()
        wp.init()
        device = wp.get_device(options.device)
        if not device.is_cuda:
            raise RuntimeError("Production execution requires CUDA")
        backend = {"ipc": "newton-ipc", "dat": "newton-dat-own-energies", "dat-reference": "newton-dat-reference"}[options.solver]
        print(json.dumps({"backend": backend,
                          "solver": options.solver, "device": device.alias, "gpu": device.name,
                          "precision": "float32"}), flush=True)
        if options.check:
            return 0
        if options.stock_cloth_twist and options.solver != "dat":
            raise ValueError("--stock-cloth-twist selects the official scene for --solver dat; reference DAT already uses it")
        if options.solver == "dat-reference" or options.stock_cloth_twist:
            if options.no_graphs or options.benchmark_snapshot or options.scene_exporter is not None:
                raise ValueError("Reference DAT runs the original example CLI; use its original graph/timing/scene options")
            forwarded = ["--device", options.device, *native]
            if options.benchmark:
                forwarded.append("--benchmark")
            if options.test:
                forwarded.append("--test")
            if options.stock_cloth_twist:
                from .stock_twist import run
                return run(forwarded)
            from .dat_twist import run as run_twist
            return run_twist(forwarded)
        scene = importlib.import_module("gpu.scene")
        solver_module = importlib.import_module("gpu.solver" if options.solver == "ipc" else "gpu.solver_dat")
        with scene.SceneExporter(scene.find_exporter(options.scene_exporter), native) as exporter:
            model = scene.create_model(exporter.scene, device)
            solver_class = solver_module.SolverIPC if options.solver == "ipc" else solver_module.SolverDAT
            # Native scenes supply their interaction distance. Use the same
            # margin/gap ratio as the official twist; do not inherit Newton's
            # generic 0.2-unit margin or translate GS movement boxes into DAT.
            dhat = float(model.ipc.params["d_hat"])
            solver_options = {"particle_enable_self_contact": True,
                              "particle_self_contact_margin": dhat,
                              "particle_self_contact_gap": 0.75 * dhat} if options.solver == "dat" else {}
            solver = solver_class(model, iterations=int(model.ipc.params["max_global_iters"]),
                                  vertex_contact_capacity=options.vertex_contact_capacity,
                                  edge_contact_capacity=options.edge_contact_capacity, **solver_options)
            solver.contact_capacities = {"vertex": options.vertex_contact_capacity, "edge": options.edge_contact_capacity}
            solver.configuration = {**getattr(solver, "configuration", {}),
                                    "contact_capacities": solver.contact_capacities}
            return run_simulation(exporter, model, solver, graphs=not options.no_graphs, benchmark=options.benchmark,
                                  benchmark_snapshot=options.benchmark_snapshot,
                                  warmup=options.benchmark_warmup, test=options.test, native_arguments=native)
    except (ImportError, RuntimeError, ValueError, TypeError, OSError, KeyError) as error:
        print(f"Newton GPU: {error}", file=sys.stderr)
        if "collision-row overflow" in str(error):
            print("Increase --vertex-contact-capacity and --edge-contact-capacity for this scene.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
