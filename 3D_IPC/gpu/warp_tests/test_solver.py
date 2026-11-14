"""End-to-end checks of our physics in the inherited upstream timestep."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.solvers import SolverVBD

from gpu.run import FrameGraph
from gpu.scene import create_model
from gpu.solver import SolverIPC
from .test_body_solver import body_payload


def point_payload(substeps=1):
    payload = body_payload()
    payload.update(positions=[[0.1, 0.2, 0.3]], velocities=[[0.1, 0.2, 0.3]],
        mass=[2.0], node_to_rb=[-1], tris=[], bodies=[], surface_nodes=[0],
        pins=[{"node": 0, "target": [0.15, 0.25, 0.35], "prescribed": False}],
        pin_targets=[[0.15, 0.25, 0.35]])
    payload["params"].update(substeps=substeps, dt=0.05/substeps)
    return payload


class SolverIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_inherited_timestep_solves_our_finite_pin_potential(self):
        self.assertIs(SolverIPC.step, SolverVBD.step)
        self.assertIs(SolverIPC._solve_particle_iteration, SolverVBD._solve_particle_iteration)
        for device in self.devices:
            with self.subTest(device=str(device)):
                payload = point_payload()
                model = create_model(payload, device)
                solver = SolverIPC(model, allow_cpu=True)
                before, after = model.state(), model.state()
                dt = payload["params"]["dt"]
                x = np.asarray(payload["positions"])
                velocity = np.asarray(payload["velocities"])
                target = np.asarray(payload["pin_targets"])
                mass, stiffness = 2.0, dt*dt*payload["params"]["kpin"]
                expected = (mass*(x+dt*velocity+dt*dt*np.array([0, -2, 0]))+stiffness*target)/(mass+stiffness)
                # The parent default finish hook invokes DAT. Our physical
                # update must run entirely through the substituted CCD hook.
                with patch.object(SolverVBD, "_finish_particle_color", side_effect=AssertionError("DAT invoked")):
                    solver.step(before, after, model.control(), None, dt)
                solver.check_status()
                np.testing.assert_allclose(after.particle_q.numpy(), expected, rtol=2e-6, atol=2e-7)
                np.testing.assert_allclose(after.particle_qd.numpy(), (expected-x)/dt, rtol=2e-5, atol=2e-6)

    def test_zero_mass_particle_stays_fixed_with_each_initial_guess(self):
        for device in self.devices:
            for mode in ("global", "colored", "level"):
                with self.subTest(device=str(device), mode=mode):
                    payload = point_payload()
                    payload.update(positions=[[.1, .2, .3], [2, 2, 2]],
                        velocities=[[1, 0, 0], [0, 0, 0]], mass=[0, 2], node_to_rb=[-1, -1],
                        pins=[], pin_targets=[[0, 0, 0], [0, 0, 0]], surface_nodes=[0, 1])
                    payload["params"].update(use_ccd_guess=mode == "global",
                        use_colored_ccd_guess=mode == "colored", use_colored_ccd_guess_level_based=mode == "level",
                        colored_ccd_guess_iters=1, colored_ccd_guess_fraction=1.0)
                    model = create_model(payload, device)
                    self.assertEqual(int(model.particle_flags.numpy()[0]) & 1, 0)
                    self.assertEqual(int(model.ipc.surface.numpy()[0]), 1)
                    solver = SolverIPC(model, allow_cpu=True)
                    before, after = model.state(), model.state()
                    initial = before.particle_q.numpy().copy()
                    solver.step(before, after, model.control(), None, payload["params"]["dt"])
                    solver.check_status()
                    np.testing.assert_array_equal(after.particle_q.numpy()[0], initial[0])
                    np.testing.assert_array_equal(solver.ccd_backend._initial_guess.targets.numpy()[0], initial[0])

    def test_empty_triangle_detector_has_no_bvh_and_clears_results(self):
        for device in self.devices:
            with self.subTest(device=str(device)):
                model = create_model(point_payload(), device)
                solver = SolverIPC(model, allow_cpu=True)
                detector = solver.trimesh_collision_detector
                self.assertIsNone(detector.bvh_tris)
                self.assertIsNone(detector.bvh_edges)
                detector.rebuild(model.particle_q)
                detector.refit(model.particle_q)
                detector.vertex_triangle_collision_detection(0.01)
                detector.triangle_triangle_intersection_detection()
                wp.synchronize_device(device)
                np.testing.assert_array_equal(detector.bvh_tris_group_roots.numpy(), -1)
                np.testing.assert_array_equal(detector.vertex_colliding_triangles_count.numpy(), 0)
                np.testing.assert_allclose(detector.vertex_colliding_triangles_min_dist.numpy(), 0.01)

    def test_captured_particle_frames_equal_eager_for_even_and_odd_substeps(self):
        for device in self.devices:
            for substeps in (1, 3, 10):
                with self.subTest(device=str(device), substeps=substeps):
                    results = []
                    for graphs in (False, True):
                        model = create_model(deepcopy(point_payload(substeps)), device)
                        solver = SolverIPC(model, allow_cpu=True)
                        frames = FrameGraph(model, solver, graphs=graphs)
                        for _ in range(3):
                            frames.advance()
                        results.append((frames.state.particle_q.numpy(), frames.state.particle_qd.numpy()))
                        self.assertEqual(frames.parity, (3*substeps) % 2)
                        self.assertEqual(int(frames.motion.counter.numpy()[0]), 3*substeps)
                    for eager, captured in zip(*results):
                        np.testing.assert_array_equal(captured, eager)

    def test_zero_iterations_publishes_current_state_instead_of_stale_output(self):
        for device in self.devices:
            payload = point_payload()
            model = create_model(payload, device)
            solver = SolverIPC(model, iterations=0, allow_cpu=True)
            before, after = model.state(), model.state()
            before.particle_q.assign([[0.4, 0.5, 0.6]])
            solver.step(before, after, model.control(), None, payload["params"]["dt"])
            solver.check_status()
            np.testing.assert_array_equal(after.particle_q.numpy(), before.particle_q.numpy())
            np.testing.assert_array_equal(after.particle_qd.numpy(), 0.0)


if __name__ == "__main__":
    unittest.main()
