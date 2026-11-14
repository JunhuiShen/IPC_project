r"""Export the actual shared DAT twist trajectory for matched GIF rendering.

Run each solver in a fresh CUDA Python process::

    python gpu/newton_dat/export_twist_trajectory.py \
        --variant reference --output /tmp/reference_twist.npz
    python gpu/newton_dat/export_twist_trajectory.py \
        --variant own --output /tmp/own_twist.npz

Both runs use the shared 480-frame, 60 Hz, 2,500-vertex cloth testcase, ten
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
from pathlib import Path
import sys
import time


def _source_hashes(root):
    result = {}
    for name in (
        "gpu/dat_twist.py", "gpu/stock_twist.py", "gpu/solver_dat.py",
        "gpu/dat_contacts.py", "gpu/dat_contact_fast.py", "gpu/dat_contact_fast.cuh",
        "gpu/dat_material_kernels.py", "gpu/dat_materials.py", "gpu/dat_stock_scene.py",
        "gpu/contact.py", "gpu/contact.cuh", "gpu/contact_exact.py", "gpu/contact_exact.cuh",
        "gpu/third_party/newton_reference/newton/_src/solvers/vbd/solver_vbd.py",
        "gpu/third_party/newton_reference/newton/_src/solvers/vbd/particle_vbd_kernels.py",
        "gpu/third_party/newton_reference/newton/examples/cloth/example_cloth_twist.py",
    ):
        path = root / name
        if path.is_file():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=("reference", "own"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, choices=(480,), default=480,
                        help="the requested comparison is fixed at 480 simulation frames")
    args = parser.parse_args(argv)
    if args.output.suffix != ".npz":
        parser.error("output must have an .npz suffix")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import gpu

    gpu.select_newton_runtime("reference")
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
    own_module = importlib.import_module("gpu.solver_dat") if args.variant == "own" else None
    bridge_receipts = []

    def solver_factory(model, **stock_options):
        if own_module is None:
            return native_solver(model, **stock_options)
        from gpu.dat_stock_scene import attach_stock_ipc

        bridge_receipts.append(attach_stock_ipc(model, gs_predictor="none"))
        return own_module.SolverDAT(model, **stock_options)

    old_argv, old_solver = sys.argv, newton.solvers.SolverVBD
    try:
        sys.argv = ["export-dat-twist", "--device", args.device, "--viewer", "null",
                    "--num-frames", str(args.frames)]
        viewer, example_args = newton.examples.init(newton.examples.create_parser())
        module = importlib.import_module("newton.examples.cloth.example_cloth_twist")
        newton.solvers.SolverVBD = solver_factory
        example = make_example_type(module.Example)(viewer, example_args)
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
        print(json.dumps({"variant": args.variant, "frames": 480, "samples": 240,
                          "gpu": device.name, "angular_velocity": example.rot_angular_velocity,
                          "contact_precision": getattr(getattr(example.solver, "_energy_contacts", None), "precision", "upstream float32")}), flush=True)
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
        velocity = example.state_0.particle_qd.numpy()
        if not np.isfinite(velocity).all() or not np.isfinite(step_seconds).all():
            raise RuntimeError("Nonfinite final velocities or timing results")
        collision_info = example.solver.trimesh_collision_info
        final_rows = {"scope": "final stored collision rows only; earlier frame peaks are not measured"}
        for name, counts, capacities in (
            ("point_triangle", collision_info.vertex_colliding_triangles_count,
             collision_info.vertex_colliding_triangles_buffer_sizes),
            ("directed_edge_edge", collision_info.edge_colliding_edges_count,
             collision_info.edge_colliding_edges_buffer_sizes),
        ):
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
            "timer_excludes": ["setup", "compilation", "graph capture", "all position downloads", "rendering", "validation", "file output"],
            "solver_configuration": getattr(example.solver, "configuration", None),
            "contact_precision": getattr(getattr(example.solver, "_energy_contacts", None), "precision", "upstream float32"),
            "energy_formulas": ("project corotated/hinge and IPC barrier/friction, full geometric diagonal Hessian"
                                if own_module else "pinned upstream DAT energy laws and native contact Hessian"),
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
