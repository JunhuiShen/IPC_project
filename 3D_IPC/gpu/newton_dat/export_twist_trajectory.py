r"""Export the actual shared DAT/GS twist trajectory for matched GIF rendering.

Run each solver in a fresh CUDA Python process::

    python gpu/newton_dat/export_twist_trajectory.py \
        --variant reference --output /tmp/reference_twist.npz
    python gpu/newton_dat/export_twist_trajectory.py \
        --variant own --output /tmp/own_twist.npz

All variants use the shared 480-frame, 60 Hz, 2,500-vertex cloth testcase, ten
substeps and four iterations, with opposing endpoint speeds of pi/2 rad/s.
The original graph is stepped 480 times. Host synchronization brackets each
measured step; construction, compilation, capture, downloads, validation and
file output are excluded. Positions are downloaded every two frames outside
the timer, producing 240 samples at 30 FPS for an eight-second, 1x-speed GIF.

NPZ fields: positions[240,2500,3], faces[4802,3], frame_indices[240] (2..480),
times[240] (simulation seconds), step_seconds[480], initial_positions[2500,3],
and metadata_json (a scalar Unicode JSON string). No pickle is required.
The same metadata is also written beside the NPZ with a .json suffix.
"""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time


def _updating_cap(value):
    result=int(value)
    if result<0:
        raise argparse.ArgumentTypeError("Updating cap must be nonnegative")
    return result


def _fraction(value):
    result=float(value)
    if not math.isfinite(result) or not 0.0<=result<=1.0:
        raise argparse.ArgumentTypeError("Target fraction must be finite and in [0,1]")
    return result


def _source_hashes(root):
    result = {}
    for name in (
        "gpu/__init__.py", "gpu/dat_twist.py", "gpu/solver_dat.py",
        "gpu/dat_initial_guess.py", "gpu/initial_guess_schedule.py",
        "gpu/dat_energies.py", "gpu/broad_phase_boxes.py", "gpu/barrier_coloring.py",
        "gpu/dat_color_schedule.py", "gpu/dat_color_ccd.py", "gpu/dat_color_contacts.py",
        "gpu/contact.py", "gpu/contact.cuh", "gpu/contact_exact.cuh",
        "gpu/solver.py",
        "gpu/ccd.py", "gpu/ccd.cuh", "gpu/ccd_fp32.cuh", "gpu/ccd_plane_fp32.cuh", "gpu/ccd_direct_fp32.cuh",
        "gpu/ccd_direct_multi_axis.cuh", "gpu/ccd_clearance.cuh", "gpu/ccd_defaults.py", "gpu/query_groups.py", "gpu/warp_build.py",
        "gpu/exact.cuh", "gpu/original_ccd.cuh", "gpu/original_rigid.cuh", "gpu/direct_bound.cuh",
        "gpu/separation.cuh", "gpu/global_guess.cuh", "gpu/newton_dat/export_twist_trajectory.py",
        "gpu/movement_box_queries.py", "gpu/policies.py",
        "gpu/sdf.py", "gpu/sdf.cuh", "gpu/rigid.py", "gpu/rigid.cuh", "gpu/rigid_types.h",
        "gpu/energy.py", "gpu/cloth_math.cuh", "gpu/solid_math.cuh", "gpu/solve.cuh",
        "gpu/third_party/newton_reference/newton/_src/solvers/vbd/solver_vbd.py",
        "gpu/third_party/newton_reference/newton/_src/solvers/vbd/particle_vbd_kernels.py",
        "gpu/third_party/newton_reference/newton/examples/cloth/example_cloth_twist.py",
        "gpu/third_party/newton/newton/_src/solvers/vbd/solver_vbd.py",
        "gpu/third_party/newton/newton/_src/solvers/vbd/particle_vbd_kernels.py",
        "gpu/third_party/newton/newton/examples/cloth/example_cloth_twist.py",
    ):
        path = root / name
        if path.is_file():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=("reference", "own", "gs"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, choices=(480,), default=480,
                        help="the requested comparison is fixed at 480 simulation frames")
    parser.add_argument("--no-planar-dat", action="store_true",
                        help="for --variant own, disable solver contact energies and DAT limits; explicit predictor/color CCD remains enabled; --color-ccd-contact-forces can restore mesh forces")
    parser.add_argument("--level-based-ccd-guess", action="store_true",
                        help="for --variant own, use our GPU level-based CCD initial guess")
    parser.add_argument("--ccd-method", choices=("interval", "plane", "direct_rounding"), default="interval",
                        help="initial-guess CCD acceptance: interval checks, ten-direction plane clipping, or closest-feature direct clipping with a rounding allowance; independent of --color-ccd-method")
    parser.add_argument("--colored-ccd-guess-iters", type=_updating_cap,
                        help="with --level-based-ccd-guess, updating sweep cap (default: 1, plus one discarded probe)")
    parser.add_argument("--colored-ccd-guess-fraction", type=_fraction,
                        help="with --level-based-ccd-guess, target displacement fraction (default: 1)")
    parser.add_argument("--build-broad-phase-boxes", action="store_true",
                        help="with --variant own --no-planar-dat, build node/primitive boxes and BVHs")
    parser.add_argument("--color-barrier-pairs", action="store_true",
                        help="with --variant own --no-planar-dat --build-broad-phase-boxes, color candidate pairs and use combined colors for VBD sweeps")
    parser.add_argument("--color-ccd", action="store_true",
                        help="with --variant own --no-planar-dat --build-broad-phase-boxes --color-barrier-pairs, clip actual VBD color updates with independent per-vertex FP32 CCD")
    parser.add_argument("--color-ccd-method", choices=("direct_rounding", "direct_rounding_multi_axis", "motion_projection", "plane"), default="direct_rounding",
                        help="per-color CCD acceptance, independent of initial-guess --ccd-method (default: direct_rounding)")
    parser.add_argument("--color-ccd-accept-uncertified", action="store_true",
                        help="with --color-ccd-method plane, use the box-clamped proposal when CCD rejects all motion; allows crossings")
    parser.add_argument("--color-ccd-contact-forces", action="store_true",
                        help="with --color-ccd, assemble IPC barrier and friction forces before each VBD color update")
    parser.add_argument("--no-collisions", action="store_true",
                        help="for --variant gs, disable contact processing, CCD and movement-box limits")
    parser.add_argument("--vertex-contact-capacity", type=int,
                        help="override the per-vertex triangle-contact capacity")
    parser.add_argument("--edge-contact-capacity", type=int,
                        help="override the per-edge contact capacity")
    args = parser.parse_args(argv)
    if args.output.suffix != ".npz":
        parser.error("output must have an .npz suffix")
    if args.level_based_ccd_guess and args.variant!="own":
        parser.error("--level-based-ccd-guess requires --variant own")
    if args.ccd_method != "interval" and not args.level_based_ccd_guess:
        parser.error("--ccd-method plane/direct_rounding requires --level-based-ccd-guess")
    if not args.level_based_ccd_guess and (args.colored_ccd_guess_iters is not None or args.colored_ccd_guess_fraction is not None):
        parser.error("--colored-ccd-guess-iters/fraction require --level-based-ccd-guess")
    guess_cap=1 if args.colored_ccd_guess_iters is None else args.colored_ccd_guess_iters
    guess_fraction=1.0 if args.colored_ccd_guess_fraction is None else args.colored_ccd_guess_fraction
    predictor_ccd_enabled=args.level_based_ccd_guess and guess_cap>0
    if args.no_planar_dat and args.variant != "own":
        parser.error("--no-planar-dat requires --variant own")
    if args.build_broad_phase_boxes and (args.variant != "own" or not args.no_planar_dat):
        parser.error("--build-broad-phase-boxes requires --variant own --no-planar-dat")
    if args.color_barrier_pairs and (args.variant != "own" or not args.no_planar_dat or
                                     not args.build_broad_phase_boxes):
        parser.error("--color-barrier-pairs requires --variant own --no-planar-dat --build-broad-phase-boxes")
    if args.color_ccd and (args.variant != "own" or not args.no_planar_dat or
                           not args.build_broad_phase_boxes or not args.color_barrier_pairs):
        parser.error("--color-ccd requires --variant own --no-planar-dat --build-broad-phase-boxes --color-barrier-pairs")
    if args.color_ccd_method != "direct_rounding" and not args.color_ccd:
        parser.error(f"--color-ccd-method {args.color_ccd_method} requires --color-ccd")
    if args.color_ccd_accept_uncertified and (not args.color_ccd or args.color_ccd_method != "plane"):
        parser.error("--color-ccd-accept-uncertified requires --color-ccd --color-ccd-method plane")
    if args.color_ccd_contact_forces and not args.color_ccd:
        parser.error("--color-ccd-contact-forces requires --color-ccd")
    if args.no_collisions and args.variant != "gs":
        parser.error("--no-collisions requires --variant gs")
    if any(value is not None and value <= 0 for value in
           (args.vertex_contact_capacity, args.edge_contact_capacity)):
        parser.error("contact capacities must be positive integers")
    collision_processing_enabled = not (args.no_planar_dat or args.no_collisions)
    solver_collision_processing_enabled = collision_processing_enabled or args.color_ccd
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import gpu

    gpu.select_newton_runtime("project" if args.variant == "gs" else "reference")
    import numpy as np
    import warp as wp
    import newton.examples
    import newton.solvers
    from gpu.dat_twist import ANGULAR_VELOCITY, DEFAULT_FRAMES, FPS, ITERATIONS, SUBSTEPS, make_example_type

    if (args.frames, DEFAULT_FRAMES, FPS, SUBSTEPS, ITERATIONS) != (480, 480, 60, 10, 4):
        raise RuntimeError("The requested trajectory protocol changed")
    wp.init()
    wp.set_device(args.device)
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("Trajectory export requires a CUDA device")
    source_hashes = _source_hashes(root)
    native_solver = newton.solvers.SolverVBD
    own_module = (importlib.import_module("gpu.solver") if args.variant == "gs" else
                  importlib.import_module("gpu.solver_dat") if args.variant == "own" else None)
    bridge_receipts = []

    def solver_factory(model, **stock_options):
        if args.vertex_contact_capacity is not None:
            stock_options["particle_vertex_contact_buffer_size"] = args.vertex_contact_capacity
        if args.edge_contact_capacity is not None:
            stock_options["particle_edge_contact_buffer_size"] = args.edge_contact_capacity
        if own_module is None:
            return native_solver(model, **stock_options)
        from gpu.dat_twist import attach_stock_ipc

        if args.level_based_ccd_guess:
            receipt=attach_stock_ipc(model,gs_predictor="level",colored_ccd_guess_iters=guess_cap,
                                     colored_ccd_guess_fraction=guess_fraction)
        else:
            receipt=attach_stock_ipc(model, gs_predictor="colored" if args.variant == "gs" else "none")
        model.ipc.params['ccd_method']=args.ccd_method
        bridge_receipts.append(receipt)
        if args.variant == "gs":
            return own_module.SolverIPC(model, iterations=ITERATIONS,
                vertex_contact_capacity=args.vertex_contact_capacity or 512,
                edge_contact_capacity=args.edge_contact_capacity or 1024,
                enable_collisions=collision_processing_enabled)
        return own_module.SolverDAT(model, enable_planar_dat=not args.no_planar_dat,
                                   enable_broad_phase_boxes=args.build_broad_phase_boxes,
                                   enable_barrier_coloring=args.color_barrier_pairs,
                                   enable_color_ccd=args.color_ccd,
                                   **({"color_ccd_method": args.color_ccd_method} if args.color_ccd_method != "direct_rounding" else {}),
                                   **({"color_ccd_accept_uncertified": True} if args.color_ccd_accept_uncertified else {}),
                                   **({"enable_color_ccd_contact_forces": True} if args.color_ccd_contact_forces else {}),
                                   **stock_options)

    old_argv, old_solver = sys.argv, newton.solvers.SolverVBD
    try:
        sys.argv = ["export-dat-twist", "--device", args.device, "--viewer", "null",
                    "--num-frames", str(args.frames)]
        viewer, example_args = newton.examples.init(newton.examples.create_parser())
        module = importlib.import_module("newton.examples.cloth.example_cloth_twist")
        newton.solvers.SolverVBD = solver_factory
        example = make_example_type(module.Example,
            enable_collision_pipeline=collision_processing_enabled)(viewer, example_args)
        wp.synchronize_device(device)
        faces = np.asarray(example.faces, np.int32).reshape(-1, 3)
        initial_positions = example.model.particle_q.numpy().astype(np.float32, copy=False)
        if faces.shape != (4802, 3) or initial_positions.shape != (2500, 3):
            raise RuntimeError("The requested cloth topology changed")
        if (example.fps, example.sim_substeps, example.iterations, example.rot_angular_velocity) != (60, 10, 4, ANGULAR_VELOCITY):
            raise RuntimeError("The requested twist integration settings changed")
        positions = np.empty((240, 2500, 3), np.float32)
        step_seconds = np.empty(480, np.float64)
        frame_indices = np.arange(2, 481, 2, dtype=np.int32)
        mesh_contact_processing = collision_processing_enabled or args.color_ccd_contact_forces
        params = getattr(getattr(example.model, "ipc", None), "params", {})
        barrier_stiffness = float(getattr(example.solver, "_contact_k_barrier", params.get("k_barrier", example.model.soft_contact_ke)))
        friction_coefficient = float(getattr(example.solver, "_energy_friction", params.get("friction_coefficient", example.model.soft_contact_mu)))
        barrier_enabled = mesh_contact_processing and barrier_stiffness > 0.0
        friction_enabled = barrier_enabled and friction_coefficient > 0.0
        contact_precision = (None if not mesh_contact_processing else
            getattr(example.solver.contact_backend, "grouped_precision", "float64") if args.variant == "gs" else
            (example.solver.color_contact_forces.configuration['geometry_and_derivative_precision'] if args.color_ccd_contact_forces else
             getattr(getattr(example.solver, "_energy_contacts", None), "precision", "upstream float32")))
        print(json.dumps({"variant": args.variant, "frames": 480, "samples": 240,
                          "gpu": device.name, "angular_velocity": example.rot_angular_velocity,
                          "planar_dat_enabled": None if args.variant == "gs" else not args.no_planar_dat,
                          "broad_phase_boxes_enabled": args.build_broad_phase_boxes,
                          "barrier_coloring_enabled": args.color_barrier_pairs,
                          "candidate_pair_generation_enabled": args.color_barrier_pairs,
                          "combined_coloring_enabled": args.color_barrier_pairs,
                          "use_colors_for_sweeps": args.color_barrier_pairs,
                          "color_ccd_enabled": args.color_ccd,
                          "color_ccd_contact_forces_enabled": args.color_ccd_contact_forces,
                          "mesh_collision_energies_enabled": barrier_enabled,
                          "friction_enabled": friction_enabled,
                          "color_ccd_method": args.color_ccd_method if args.color_ccd else None,
                          "color_ccd_accept_uncertified": args.color_ccd_accept_uncertified,
                          "collision_response_enabled": solver_collision_processing_enabled,
                          "solver_collision_processing_enabled": solver_collision_processing_enabled,
                          "collision_processing_enabled": solver_collision_processing_enabled or predictor_ccd_enabled,
                          "predictor_ccd_enabled": predictor_ccd_enabled if args.variant=="own" else args.variant=="gs" and collision_processing_enabled,
                          "contact_precision": contact_precision}), flush=True)
        if args.color_barrier_pairs:
            example.solver.barrier_coloring.reset_statistics()
        sample = 0
        for index in range(480):
            wp.synchronize_device(device)
            started = time.perf_counter()
            example.step()
            wp.synchronize_device(device)
            step_seconds[index] = time.perf_counter() - started
            frame = index + 1
            if frame % 2 == 0:
                snapshot = example.state_0.particle_q.numpy()
                if snapshot.shape != (2500, 3) or not np.isfinite(snapshot).all():
                    raise RuntimeError(f"Invalid particle positions at frame {frame}")
                positions[sample] = snapshot
                sample += 1
            if frame % 30 == 0:
                print(json.dumps({"variant": args.variant, "frame": frame,
                                  "measured_step_seconds_sum": float(step_seconds[:frame].sum())}), flush=True)
        # The original physics assertions and all downloads happen after timing.
        example.test_final()
        if own_module is not None:
            example.solver.check_status()
        # Device aggregates cover every coloring checkpoint; download once
        # after timing rather than synchronizing for host statistics per frame.
        barrier_coloring_statistics = (example.solver.barrier_coloring.statistics()
                                       if args.color_barrier_pairs else None)
        velocity = example.state_0.particle_qd.numpy()
        if not np.isfinite(velocity).all() or not np.isfinite(step_seconds).all():
            raise RuntimeError("Nonfinite final velocities or timing results")
        collision_info = getattr(example.solver, "trimesh_collision_info", None)
        final_rows = {"enabled": collision_info is not None,
                      "scope": "final stored collision rows only; earlier frame peaks are not measured"}
        collision_rows = ()
        if collision_info is not None:
            collision_rows = (
                ("point_triangle", collision_info.vertex_colliding_triangles_count,
                 collision_info.vertex_colliding_triangles_buffer_sizes),
                ("directed_edge_edge", collision_info.edge_colliding_edges_count,
                 collision_info.edge_colliding_edges_buffer_sizes),
            )
        for name, counts, capacities in collision_rows:
            raw, capacity = counts.numpy(), capacities.numpy()
            final_rows[name] = {"rows": int(raw.size), "max_raw_count": int(raw.max()) if raw.size else 0,
                                "min_capacity": int(capacity.min()) if capacity.size else 0,
                                "rows_over_capacity": int(np.count_nonzero(raw > capacity))}
        if source_hashes != _source_hashes(root):
            raise RuntimeError("Source files changed during export; trajectory provenance is no longer stable")
        material = example.model.tri_materials.numpy()
        bending = example.model.edge_bending_properties.numpy()
        elapsed = float(step_seconds.sum())
        metadata = {
            "schema": 1, "variant": args.variant, "device": device.alias, "gpu": device.name,
            "planar_dat_enabled": None if args.variant == "gs" else not args.no_planar_dat,
            "broad_phase_boxes_enabled": args.build_broad_phase_boxes,
            "barrier_coloring_enabled": args.color_barrier_pairs,
            "candidate_pair_generation_enabled": args.color_barrier_pairs,
            "combined_coloring_enabled": args.color_barrier_pairs,
            "use_colors_for_sweeps": args.color_barrier_pairs,
            "color_ccd_enabled": args.color_ccd,
            "color_ccd_contact_forces_enabled": args.color_ccd_contact_forces,
            "color_ccd_method": args.color_ccd_method if args.color_ccd else None,
            "color_ccd_accept_uncertified": args.color_ccd_accept_uncertified,
            "collision_queries_enabled": collision_processing_enabled or args.color_barrier_pairs or predictor_ccd_enabled,
            "collision_response_enabled": solver_collision_processing_enabled,
            "solver_collision_processing_enabled": solver_collision_processing_enabled,
            "collision_processing_enabled": solver_collision_processing_enabled or predictor_ccd_enabled,
            "predictor_ccd_enabled": predictor_ccd_enabled if args.variant=="own" else args.variant=="gs" and collision_processing_enabled,
            "level_based_ccd_initial_guess": {"selected":args.level_based_ccd_guess,
                "updating_cap":guess_cap if args.level_based_ccd_guess else None,
                "maximum_discarded_probes_per_guess":1 if predictor_ccd_enabled else 0,
                "target_fraction":guess_fraction if args.level_based_ccd_guess else None,
                "method":args.ccd_method if args.level_based_ccd_guess else None,
                "early_exit_enabled":args.level_based_ccd_guess},
            "mesh_collision_energies_enabled": barrier_enabled,
            "friction_enabled": friction_enabled,
            "barrier_stiffness": barrier_stiffness,
            "friction_coefficient": friction_coefficient,
            "sdf_collision_energies_enabled": False,
            "barrier_coloring_statistics": barrier_coloring_statistics,
            "requested_contact_capacities": {"vertex": args.vertex_contact_capacity,
                                             "edge": args.edge_contact_capacity},
            "warp_version": wp.__version__, "frames": 480, "simulation_fps": 60,
            "simulation_seconds": 8.0, "substeps": 10, "iterations_per_substep": 4,
            "particles": 2500, "triangles": 4802, "edges": example.model.edge_count,
            "samples": 240, "sample_fps": 30, "sample_stride": 2,
            "sample_frame_indices": "2,4,...,480", "playback_speed": 1.0,
            "angular_velocity_per_endpoint_radians_per_second": example.rot_angular_velocity,
            "opposing_endpoint_axes": np.unique(example.rot_axes.numpy(), axis=0).tolist(),
            "rotation_end_time_seconds": example.rot_end_time,
            "cuda_graph": example.graph is not None, "official_test_final_passed": True,
            "energy_status_checked": own_module is not None, "finite_samples_and_final_velocity": True,
            "measured_step_seconds_sum": elapsed,
            "measured_step_ms_mean": 1000.0 * elapsed / 480,
            "measured_step_fps": 480.0 / elapsed,
            "timer": "host perf_counter immediately before example.step through following CUDA synchronization; preceding CUDA synchronization is outside the timer",
            "timer_excludes": ["setup", "compilation", "graph capture", "all position downloads", "coloring statistics downloads", "rendering", "validation", "file output"],
            "solver_configuration": getattr(example.solver, "configuration", None),
            "gs_initial_guess": ("colored CCD" if collision_processing_enabled else
                                 "unrestricted velocity predictor; no CCD") if args.variant == "gs" else None,
            "gs_parameters": dict(example.model.ipc.params) if args.variant == "gs" else None,
            "contact_precision": contact_precision,
            "energy_formulas": (
                f"project corotated analytic self-Hessian, project hinge Gauss-Newton self-Hessian and pins; BVH candidate-pair generation and combined material/pair coloring drive VBD sweeps; actual color updates use gathered FP32 {args.color_ccd_method} CCD with per-vertex atomicMin fractions; full project IPC and frozen-friction gradient/Hessian evaluated in binary64 and assembled in FP32 per color; SDF and DAT motion limits disabled"
                if args.color_ccd_contact_forces else
                f"project corotated/hinge and pins; BVH candidate-pair generation and combined material/pair coloring drive VBD sweeps; actual color updates use gathered FP32 {args.color_ccd_method} CCD with per-vertex atomicMin fractions; contact energies, friction, SDF and DAT motion limits disabled"
                if args.color_ccd else
                "project corotated/hinge and pins; BVH candidate-pair generation and combined material/pair coloring drive VBD sweeps; contact energies, CCD and DAT motion limits disabled"
                if args.color_barrier_pairs else
                "project corotated/hinge and pins; broad-phase boxes/BVHs only; collision queries, energies and DAT motion limits disabled"
                if args.build_broad_phase_boxes else
                "project corotated/hinge and pins; all collision processing and energies disabled"
                if not collision_processing_enabled else
                "project corotated/hinge and IPC barrier; native PSD material Hessian, rank-one contact Hessian and native friction"
                if args.variant == "gs" else
                "project corotated gradient and analytic self-Hessian, project hinge Gauss-Newton self-Hessian and IPC barrier; native rank-one contact Hessian and native friction in Planar DAT mode"
                if args.variant == "own" else
                "pinned upstream DAT energy laws and native contact Hessian"),
            "native_triangle_material_values": np.unique(material, axis=0).tolist(),
            "native_bending_material_values": np.unique(bending, axis=0).tolist(),
            "native_soft_contact_stiffness": float(example.model.soft_contact_ke),
            "native_soft_contact_damping": float(example.model.soft_contact_kd),
            "native_soft_contact_friction": float(example.model.soft_contact_mu),
            "final_row_capacity": final_rows, "source_sha256": source_hashes,
            "bridge_receipt": bridge_receipts[0] if bridge_receipts else None,
            "positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
            "faces_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
            "initial_positions_sha256": hashlib.sha256(initial_positions.tobytes()).hexdigest(),
            "final_velocity_max_abs": float(np.abs(velocity).max()),
        }
        if args.level_based_ccd_guess:
            metadata["energy_formulas"]=metadata["energy_formulas"].replace("contact energies, CCD and DAT motion limits disabled",
                "solver contact energies and DAT motion limits disabled").replace(
                "collision queries, energies and DAT motion limits disabled","solver collision queries, energies and DAT motion limits disabled").replace(
                "all collision processing and energies disabled","solver collision processing and energies disabled")
            metadata["energy_formulas"] += "; level-based initial-guess CCD " + ("enabled" if predictor_ccd_enabled else "disabled by zero updating cap")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, positions=positions, faces=faces, frame_indices=frame_indices,
                            times=frame_indices.astype(np.float64) / 60.0, step_seconds=step_seconds,
                            initial_positions=initial_positions, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        sidecar = args.output.with_suffix(".json")
        sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        viewer.close()
        print(json.dumps({"output": str(args.output.resolve()), "metadata": str(sidecar.resolve()),
                          "measured_step_seconds_sum": elapsed, "samples": 240,
                          "official_test_final_passed": True, "final_row_capacity": final_rows}), flush=True)
        return 0
    finally:
        newton.solvers.SolverVBD = old_solver
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
