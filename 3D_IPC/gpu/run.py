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


def _fraction(value):
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("Target fraction must be finite and in [0,1]")
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
    total_seconds = time.perf_counter() - started
    coloring = getattr(solver, "barrier_coloring", None)
    coloring_statistics = coloring.statistics() if coloring is not None else None
    if coloring_statistics is not None and not benchmark:
        metadata["barrier_coloring_statistics"] = coloring_statistics
        output.write_json(directory / "newton_ipc_run.json", metadata)
    result = {"completed_frame": frames, "substeps": substeps, "iterations_per_substep": int(p["max_global_iters"]),
        "solver": solver_mode,
        "solver_configuration": getattr(solver, "configuration", {}),
        "contact_capacities": getattr(solver, "contact_capacities", {}),
        "total_seconds": total_seconds, "timed_frames": len(measured), "graph_capture": bool(controller.graphs),
        "capture_seconds": controller.capture_seconds,
        "barrier_coloring_statistics": coloring_statistics,
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
                        help="run the original 2,500-vertex cloth example with our GS or DAT solver")
    parser.add_argument("--no-planar-dat", action="store_true",
                        help="for our-energy DAT, disable solver contact energies and DAT limits; explicit predictor/color CCD remains enabled; --color-ccd-contact-forces can restore mesh forces")
    parser.add_argument("--level-based-ccd-guess", action="store_true",
                        help="for --solver dat, replace the native predictor with our GPU level-based CCD initial guess")
    parser.add_argument("--ccd-method", choices=("interval", "plane", "direct_rounding"),
                        help="CCD method for GS updates and prediction, or for the DAT predictor")
    parser.add_argument("--colored-ccd-guess-iters", type=_nonnegative,
                        help="with --level-based-ccd-guess, updating sweep cap (default: 1, plus one discarded probe)")
    parser.add_argument("--colored-ccd-guess-fraction", type=_fraction,
                        help="with --level-based-ccd-guess, target displacement fraction (default: 1)")
    parser.add_argument("--build-broad-phase-boxes", action="store_true",
                        help="with --solver dat --no-planar-dat, build node/primitive boxes and BVHs")
    parser.add_argument("--color-barrier-pairs", action="store_true",
                        help="with --solver dat --no-planar-dat --build-broad-phase-boxes, color candidate pairs and use combined colors for VBD sweeps")
    parser.add_argument("--color-ccd", action="store_true",
                        help="with --solver dat --no-planar-dat --build-broad-phase-boxes --color-barrier-pairs, clip actual VBD color updates with independent per-vertex FP32 CCD")
    parser.add_argument("--color-ccd-method", choices=("direct_rounding", "plane"), default="direct_rounding",
                        help="per-color CCD method, independent of --ccd-method (default: direct_rounding)")
    parser.add_argument("--color-ccd-accept-uncertified", action="store_true",
                        help="with --color-ccd-method plane, use the box-clamped proposal when CCD rejects all motion; allows crossings")
    parser.add_argument("--color-ccd-contact-forces", action="store_true",
                        help="with --color-ccd, assemble IPC barrier and friction forces before each VBD color update")
    parser.add_argument("--no-collisions", action="store_true",
                        help="for deformable GS, disable collision processing, CCD and movement-box limits")
    parser.add_argument("--gs-ccd-accept-uncertified", action="store_true",
                        help="for deformable GS, use the box-clamped proposal if update CCD would leave the vertex stationary")
    parser.add_argument("--gs-ccd-guess-accept-uncertified", action="store_true",
                        help="for GS level-based CCD, use the predictor target if it would leave the vertex stationary")
    parser.add_argument("--vertex-contact-capacity", type=_positive,
                        help="triangle-contact/candidate row capacity per vertex (native scenes: 128; stock scene: its original default)")
    parser.add_argument("--edge-contact-capacity", type=_positive,
                        help="edge-contact/candidate row capacity per edge (native scenes: 256; stock scene: its original default)")
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
        if options.level_based_ccd_guess and options.solver != "dat":
            raise ValueError("--level-based-ccd-guess requires --solver dat")
        if options.ccd_method is not None and options.solver != "ipc" and not options.level_based_ccd_guess:
            raise ValueError("--ccd-method for DAT requires --level-based-ccd-guess")
        if not options.level_based_ccd_guess and (options.colored_ccd_guess_iters is not None or
                                                 options.colored_ccd_guess_fraction is not None):
            raise ValueError("--colored-ccd-guess-iters/fraction require --level-based-ccd-guess")
        guess_options = ({"level_based_ccd_guess": True,
                          "colored_ccd_guess_iters": 1 if options.colored_ccd_guess_iters is None else options.colored_ccd_guess_iters,
                          "colored_ccd_guess_fraction": 1.0 if options.colored_ccd_guess_fraction is None else options.colored_ccd_guess_fraction}
                         if options.level_based_ccd_guess else {})
        if options.ccd_method is not None:
            guess_options["ccd_method"] = options.ccd_method
        if options.no_planar_dat and options.solver != "dat":
            raise ValueError("--no-planar-dat requires --solver dat")
        if options.build_broad_phase_boxes and (options.solver != "dat" or not options.no_planar_dat):
            raise ValueError("--build-broad-phase-boxes requires --solver dat --no-planar-dat")
        if options.color_barrier_pairs and (options.solver != "dat" or not options.no_planar_dat or
                                            not options.build_broad_phase_boxes):
            raise ValueError("--color-barrier-pairs requires --solver dat --no-planar-dat --build-broad-phase-boxes")
        if options.color_ccd and (options.solver != "dat" or not options.no_planar_dat or
                                  not options.build_broad_phase_boxes or not options.color_barrier_pairs):
            raise ValueError("--color-ccd requires --solver dat --no-planar-dat --build-broad-phase-boxes --color-barrier-pairs")
        if options.color_ccd_method != "direct_rounding" and not options.color_ccd:
            raise ValueError("--color-ccd-method plane requires --color-ccd")
        if options.color_ccd_accept_uncertified and (not options.color_ccd or options.color_ccd_method != "plane"):
            raise ValueError("--color-ccd-accept-uncertified requires --color-ccd --color-ccd-method plane")
        if options.color_ccd_contact_forces and not options.color_ccd:
            raise ValueError("--color-ccd-contact-forces requires --color-ccd")
        if options.no_collisions and options.solver != "ipc":
            raise ValueError("--no-collisions requires --solver ipc")
        if options.gs_ccd_accept_uncertified and (options.solver != "ipc" or options.no_collisions):
            raise ValueError("--gs-ccd-accept-uncertified requires --solver ipc with collisions enabled")
        if options.gs_ccd_guess_accept_uncertified and (options.solver != "ipc" or options.no_collisions):
            raise ValueError("--gs-ccd-guess-accept-uncertified requires --solver ipc with collisions enabled")
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
        if options.stock_cloth_twist and options.solver == "dat-reference":
            raise ValueError("--solver dat-reference already selects the official scene")
        if options.solver == "dat-reference" or options.stock_cloth_twist:
            if options.no_graphs or options.benchmark_snapshot or options.scene_exporter is not None:
                raise ValueError("Reference DAT runs the original example CLI; use its original graph/timing/scene options")
            forwarded = ["--device", options.device, *native]
            if options.benchmark:
                forwarded.append("--benchmark")
            if options.test:
                forwarded.append("--test")
            if options.stock_cloth_twist:
                if options.solver == "ipc":
                    from .dat_twist import run_gs
                    return run_gs(forwarded, enable_collisions=not options.no_collisions,
                                  **({"ccd_method": options.ccd_method} if options.ccd_method is not None else {}),
                                  **({"accept_uncertified": True} if options.gs_ccd_accept_uncertified else {}),
                                  **({"guess_accept_uncertified": True} if options.gs_ccd_guess_accept_uncertified else {}))
                from .dat_twist import run_own as run
                return run(forwarded, enable_planar_dat=not options.no_planar_dat,
                           enable_broad_phase_boxes=options.build_broad_phase_boxes,
                           enable_barrier_coloring=options.color_barrier_pairs,
                           enable_color_ccd=options.color_ccd,
                           **({"color_ccd_method": options.color_ccd_method} if options.color_ccd_method != "direct_rounding" else {}),
                           **({"color_ccd_accept_uncertified": True} if options.color_ccd_accept_uncertified else {}),
                           vertex_contact_capacity=options.vertex_contact_capacity,
                           edge_contact_capacity=options.edge_contact_capacity,
                           **({"enable_color_ccd_contact_forces": True} if options.color_ccd_contact_forces else {}),
                           **guess_options)
            from .dat_twist import run as run_twist
            return run_twist(forwarded)
        scene = importlib.import_module("gpu.scene")
        solver_module = importlib.import_module("gpu.solver" if options.solver == "ipc" else "gpu.solver_dat")
        with scene.SceneExporter(scene.find_exporter(options.scene_exporter), native) as exporter:
            if options.level_based_ccd_guess:
                exporter.scene["params"].update({"use_ccd_guess": False,"use_colored_ccd_guess": False,
                    "use_colored_ccd_guess_level_based": True,
                    "colored_ccd_guess_iters": guess_options["colored_ccd_guess_iters"],
                    "colored_ccd_guess_fraction": guess_options["colored_ccd_guess_fraction"]})
            if options.ccd_method is not None:
                exporter.scene["params"]["ccd_method"] = options.ccd_method
            if options.no_planar_dat:
                # Include the acceptance ablation in restart signatures while
                # leaving existing default-run checkpoint signatures unchanged.
                exporter.scene["params"]["enable_planar_dat"] = False
                exporter.scene["params"]["collision_processing"] = False
                if exporter.scene["params"].get("use_colored_ccd_guess_level_based",False):
                    exporter.scene["params"]["solver_collision_processing"] = False
                    exporter.scene["params"]["collision_processing"] = exporter.scene["params"].get("colored_ccd_guess_iters",1)>0
            if options.build_broad_phase_boxes:
                exporter.scene["params"]["enable_broad_phase_boxes"] = True
            if options.color_barrier_pairs:
                exporter.scene["params"]["enable_barrier_coloring"] = True
            if options.color_ccd:
                exporter.scene["params"].update({"enable_color_ccd": True,
                    "solver_collision_processing": True, "collision_processing": True})
            if options.color_ccd_method != "direct_rounding":
                exporter.scene["params"]["color_ccd_method"] = options.color_ccd_method
            if options.color_ccd_accept_uncertified:
                exporter.scene["params"]["color_ccd_accept_uncertified"] = True
            if options.color_ccd_contact_forces:
                exporter.scene["params"]["enable_color_ccd_contact_forces"] = True
            if options.no_collisions:
                exporter.scene["params"]["collision_processing"] = False
            if options.gs_ccd_accept_uncertified:
                exporter.scene["params"]["gs_ccd_accept_uncertified"] = True
            if options.gs_ccd_guess_accept_uncertified:
                exporter.scene["params"]["gs_ccd_guess_accept_uncertified"] = True
            model = scene.create_model(exporter.scene, device)
            solver_class = solver_module.SolverIPC if options.solver == "ipc" else solver_module.SolverDAT
            # Native scenes supply their interaction distance. Use the same
            # margin/gap ratio as the official twist; do not inherit Newton's
            # generic 0.2-unit margin or translate GS movement boxes into DAT.
            dhat = float(model.ipc.params["d_hat"])
            solver_options = {"particle_enable_self_contact": not options.no_planar_dat,
                              "particle_self_contact_margin": dhat,
                              "particle_self_contact_gap": 0.75 * dhat,
                              "enable_planar_dat": not options.no_planar_dat,
                              "enable_broad_phase_boxes": options.build_broad_phase_boxes,
                              "enable_barrier_coloring": options.color_barrier_pairs,
                              "enable_color_ccd": options.color_ccd} if options.solver == "dat" else {
                                  "enable_collisions": not options.no_collisions}
            if options.color_ccd_method != "direct_rounding":
                solver_options["color_ccd_method"] = options.color_ccd_method
            if options.color_ccd_accept_uncertified:
                solver_options["color_ccd_accept_uncertified"] = True
            if options.color_ccd_contact_forces:
                solver_options["enable_color_ccd_contact_forces"] = True
            vertex_capacity = options.vertex_contact_capacity or 128
            edge_capacity = options.edge_contact_capacity or 256
            solver = solver_class(model, iterations=int(model.ipc.params["max_global_iters"]),
                                  vertex_contact_capacity=vertex_capacity,
                                  edge_contact_capacity=edge_capacity, **solver_options)
            solver.contact_capacities = {"vertex": vertex_capacity, "edge": edge_capacity}
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
