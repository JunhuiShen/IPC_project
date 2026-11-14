"""Historical adapter regressions, preserved for the retired implementation.

The production energy-only DAT acceptance runs separately in gpu/dat_tests
because it uses the pristine Newton runtime rather than the GS hook fork.
"""

from copy import deepcopy
import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.solvers import SolverVBD

from gpu.run import FrameGraph, run_simulation, signature
from gpu.scene import create_model
from gpu.solver import SolverIPC
from gpu.reference.legacy_dat_adapter import SolverDAT
from gpu.output import read_checkpoint
from .test_body_solver import body_payload
from .test_solver import point_payload
from .test_model import cloth_payload


def point_over_triangle(*, height=0.02, velocity=(0, 0, -20), iterations=0, friction=0.0):
    payload = point_payload()
    payload.update(positions=[[-1, -1, 0], [1, -1, 0], [0, 1, 0], [0, 0, height]],
        velocities=[[0, 0, 0]] * 3 + [list(velocity)], mass=[0, 0, 0, 1], node_to_rb=[-1] * 4,
        tris=[[0, 1, 2]], triangle_dm_inverse=[[1, 0, 0, 1]], triangle_areas=[2.0],
        hinges=[], surface_nodes=[0, 1, 2, 3], pins=[], pin_targets=[[0, 0, 0]] * 4)
    payload["params"].update(mu=0.0, **{"lambda": 0.0}, gravity=[0, 0, 0], max_global_iters=iterations,
        node_box_max=0.1, d_hat=0.01, friction_coefficient=friction,
        dat_gamma=0.9, dat_node_box_padding=1.0, use_ccd_guess=False,
        use_colored_ccd_guess=False, use_colored_ccd_guess_level_based=False)
    return payload


class SolverDATTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_own_finite_pin_potential_and_original_step_state_with_guess(self):
        self.assertIs(SolverDAT.step, SolverVBD.step)
        self.assertIs(SolverDAT._solve_particle_iteration, SolverVBD._solve_particle_iteration)
        self.assertNotIn(SolverIPC, SolverDAT.__mro__)
        for device in self.devices:
            for guess in (False, True):
                payload = point_payload()
                payload["params"]["use_ccd_guess"] = guess
                model = create_model(payload, device)
                solver = SolverDAT(model, allow_cpu=True)
                before, after = model.state(), model.state()
                original = before.particle_q.numpy().copy()
                dt = payload["params"]["dt"]
                inertia = original + dt * np.asarray(payload["velocities"]) + dt * dt * np.asarray([0, -2, 0])
                mass, stiffness = 2.0, dt * dt * payload["params"]["kpin"]
                expected = (mass * inertia + stiffness * np.asarray(payload["pin_targets"])) / (mass + stiffness)
                with patch.object(SolverIPC, "_finish_particle_color", side_effect=AssertionError("IPC commit invoked")):
                    solver.step(before, after, model.control(), None, dt)
                solver.check_status()
                np.testing.assert_array_equal(solver.particle_q_prev.numpy(), original)
                np.testing.assert_allclose(solver.inertia.numpy(), inertia, atol=3e-8)
                np.testing.assert_allclose(after.particle_q.numpy(), expected, rtol=2e-6, atol=3e-8)
                np.testing.assert_array_equal(solver.particle_displacements.numpy(),
                    after.particle_q.numpy() - solver.pos_prev_collision_detection.numpy())

    def test_native_dat_alone_truncates_a_crossing_predictor(self):
        for device in self.devices:
            payload = point_over_triangle()
            model = create_model(payload, device)
            solver = SolverDAT(model, allow_cpu=True)
            before, after = model.state(), model.state()
            solver.step(before, after, model.control(), None, payload["params"]["dt"])
            solver.check_status()
            self.assertIsNone(solver.ccd_backend)
            self.assertLess(float(solver.inertia.numpy()[3, 2]), 0.0, "raw predictor must cross the face")
            accepted = after.particle_q.numpy()
            self.assertGreater(accepted[3, 2], 0.0, "native DAT must stop on the original side")
            self.assertLess(accepted[3, 2], payload["positions"][3][2])
            np.testing.assert_array_equal(accepted[:3], np.asarray(payload["positions"][:3], dtype=np.float32))

    def test_our_ipc_barrier_and_friction_reduce_tangential_slip(self):
        for device in self.devices:
            results = []
            for friction in (0.0, 0.4):
                payload = point_over_triangle(height=0.003, velocity=(0.1, 0, 0), iterations=2, friction=friction)
                model = create_model(payload, device)
                solver = SolverDAT(model, allow_cpu=True)
                before, after = model.state(), model.state()
                with patch.object(SolverVBD, "_accumulate_particle_contact_terms", side_effect=AssertionError("stock contact law invoked")):
                    solver.step(before, after, model.control(), None, payload["params"]["dt"])
                solver.check_status()
                self.assertGreater(after.particle_q.numpy()[3, 2], 0.003)
                results.append(after.particle_qd.numpy()[3, 0])
            self.assertLess(abs(results[1]), abs(results[0]))

    def test_absolute_proposals_replace_corrupted_accumulated_displacement(self):
        class CorruptedScratchDAT(SolverDAT):
            def _solve_particle_elasticity_color(self, *args):
                super()._solve_particle_elasticity_color(*args)
                self.particle_displacements.fill_(wp.vec3(17.0))

        for device in self.devices:
            results = []
            for solver_class in (SolverDAT, CorruptedScratchDAT):
                payload = point_payload()
                model = create_model(payload, device)
                solver = solver_class(model, allow_cpu=True)
                before, after = model.state(), model.state()
                solver.step(before, after, model.control(), None, payload["params"]["dt"])
                solver.check_status()
                results.append(after.particle_q.numpy())
            np.testing.assert_array_equal(results[0], results[1])

    def test_disabled_predictor_and_explicit_conflicting_guess_rejection(self):
        payload = point_payload()
        payload["params"].update(dat_inertial_predictor=False, max_global_iters=0)
        for device in self.devices:
            model = create_model(payload, device)
            solver = SolverDAT(model, allow_cpu=True)
            before, after = model.state(), model.state()
            original = before.particle_q.numpy().copy()
            solver.step(before, after, model.control(), None, payload["params"]["dt"])
            solver.check_status()
            np.testing.assert_array_equal(after.particle_q.numpy(), original)
            with_guess = deepcopy(payload)
            with_guess["params"]["use_ccd_guess"] = True
            with self.assertRaisesRegex(ValueError, "all CCD initial-guess flags"):
                SolverDAT(create_model(with_guess, device), allow_cpu=True)
            with self.assertRaisesRegex(ValueError, "rigid bodies"):
                SolverDAT(create_model(body_payload(), device), allow_cpu=True)

    def test_our_sdf_energy_joins_dat_local_solve(self):
        for device in self.devices:
            payload = point_payload()
            payload.update(positions=[[0.1, 0.005, 0.3]], velocities=[[0, 0, 0]], pins=[],
                pin_targets=[[0, 0, 0]], colliders=[{"kind": "plane", "point": [0, 0, 0], "axis": [0, 1, 0], "radius": 0,
                    "previous_pose": {"rotation": np.eye(3).reshape(-1).tolist(), "translation": [0, 0, 0]},
                    "current_pose": {"rotation": np.eye(3).reshape(-1).tolist(), "translation": [0, 0, 0]}}])
            payload["params"].update(gravity=[0, 0, 0], k_sdf=100.0, eps_sdf=0.05)
            model = create_model(payload, device)
            solver = SolverDAT(model, allow_cpu=True)
            before, after = model.state(), model.state()
            solver.step(before, after, model.control(), None, payload["params"]["dt"])
            solver.check_status()
            # Native soft SDF energy is cubic in the epsilon band:
            # E = k*(epsilon-y)^3/(2*epsilon), not a quadratic spring.
            dt = payload["params"]["dt"]
            a, b, c = 1.5 * 100.0 / 0.05, 2.0 / (dt * dt), -(2.0 / (dt * dt)) * (0.05 - 0.005)
            d = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
            expected = 0.05 - d
            self.assertAlmostEqual(float(after.particle_q.numpy()[0, 1]), expected, delta=2e-7)

    def test_native_checkpoint_intervals_and_cross_solver_restart_guard(self):
        for interval in (1, 2, 10):
            payload = point_payload()
            payload["params"].update(dat_collision_detection_interval=interval, max_global_iters=3)
            model = create_model(payload, "cpu")
            solver = SolverDAT(model, allow_cpu=True)
            before, after = model.state(), model.state()
            solver.step(before, after, model.control(), None, payload["params"]["dt"])
            solver.check_status()
            # Inherited VBD schedule checks (iter_num+1) % interval.
            self.assertEqual([solver._self_contact_is_due(i) for i in range(3)],
                             [(i + 1) % interval == 0 for i in range(3)])
            np.testing.assert_array_equal(solver.particle_displacements.numpy(),
                after.particle_q.numpy() - solver.pos_prev_collision_detection.numpy())
        self.assertNotEqual(signature(payload, "ipc"), signature(payload, "dat"))
        with tempfile.TemporaryDirectory() as directory:
            payload = point_payload()
            payload["output"] = {"outdir": directory, "format": "ply", "num_frames": 1,
                                  "restart_frame": -1, "write_substeps": False}
            model = create_model(payload, "cpu")
            exporter = type("Export", (), {"scene": payload})()
            run_simulation(exporter, model, SolverDAT(model, allow_cpu=True), graphs=False)
            payload["output"].update(restart_frame=1, num_frames=2)
            other = create_model(payload, "cpu")
            with self.assertRaisesRegex(ValueError, "solver settings differ"):
                run_simulation(exporter, other, SolverIPC(other, allow_cpu=True), graphs=False)

    def test_shared_cloth_and_solid_energies_run_in_dat_adapter(self):
        for device in self.devices:
            for solid in (False, True):
                payload = cloth_payload()
                payload["params"] = {**point_payload()["params"], **payload["params"]}
                payload["params"].update(node_box_max=0.1, use_ccd_guess=False)
                payload.update(pins=[], pin_targets=[[0, 0, 0]] * 4,
                               velocities=[[0.05, 0, 0]] * 4)
                if solid:
                    payload.update(positions=[[0, 0, 0], [1.05, 0, 0], [0, 1, 0], [0, 0, 1]],
                        tris=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], tets=[[0, 1, 2, 3]],
                        triangle_dm_inverse=[], triangle_areas=[], hinges=[], tet_nodes=[0, 1, 2, 3],
                        surface_nodes=[0, 1, 2, 3], uses_general_solver=True,
                        tet_rest_data=[{"dm_inverse": np.eye(3).reshape(-1).tolist(), "measure": 1 / 6,
                            "grad_N": [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]}])
                    payload["velocities"] = [[0, 0, 0]] * 4
                model = create_model(payload, device)
                solver = SolverDAT(model, allow_cpu=True)
                before, after = model.state(), model.state()
                original = before.particle_q.numpy().copy()
                solver.step(before, after, model.control(), None, payload["params"]["dt"])
                solver.check_status()
                self.assertTrue(np.isfinite(after.particle_q.numpy()).all())
                self.assertGreater(float(np.linalg.norm(after.particle_q.numpy() - original)), 0.0)
                if solid:
                    self.assertLess(float(after.particle_q.numpy()[1, 0]), float(original[1, 0]))

    def test_dat_restart_matches_continuous_odd_substep_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            def execute(payload):
                model = create_model(payload, "cpu")
                exporter = type("Export", (), {"scene": payload})()
                run_simulation(exporter, model, SolverDAT(model, allow_cpu=True), graphs=False)

            split = point_payload(3)
            split["output"] = {"outdir": directory + "/split", "format": "ply", "num_frames": 2,
                               "restart_frame": -1, "write_substeps": False}
            execute(split)
            resumed = deepcopy(split)
            resumed["output"].update(restart_frame=2, num_frames=3)
            execute(resumed)
            continuous = deepcopy(split)
            continuous["output"].update(outdir=directory + "/continuous", num_frames=3)
            execute(continuous)
            from pathlib import Path
            a = read_checkpoint(Path(directory) / "split/state_0003.bin", particle_count=1, body_count=0)
            b = read_checkpoint(Path(directory) / "continuous/state_0003.bin", particle_count=1, body_count=0)
            np.testing.assert_array_equal(a["positions"], b["positions"])
            np.testing.assert_array_equal(a["velocities"], b["velocities"])

    def test_captured_and_eager_dat_frames_match_with_odd_even_substeps(self):
        for device in self.devices:
            for substeps in (1, 3, 10):
                results = []
                for capture in (False, True):
                    payload = point_payload(substeps)
                    model = create_model(payload, device)
                    solver = SolverDAT(model, allow_cpu=True)
                    frames = FrameGraph(model, solver, graphs=capture)
                    for _ in range(3):
                        frames.advance()
                    self.assertEqual(int(frames.motion.counter.numpy()[0]), 3 * substeps)
                    results.append((frames.state.particle_q.numpy(), frames.state.particle_qd.numpy()))
                for eager, captured in zip(results[0], results[1]):
                    np.testing.assert_array_equal(captured, eager)


if __name__ == "__main__":
    unittest.main()
