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


def checkpoint_payload(prune):
    """A warm movement history brings a point within one box of a fixed face."""
    payload = body_payload()
    points = [[.25, .25, .108], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
    payload.update(positions=points, velocities=[[0, 0, 0]] * 4, mass=[1, 0, 0, 0],
        node_to_rb=[-1] * 4, tris=[[1, 2, 3]], bodies=[], surface_nodes=list(range(4)),
        pins=[{"node": 0, "target": [.25, .25, -.5], "prescribed": False}],
        pin_targets=[[.25, .25, -.5], *points[1:]])
    payload["params"].update(gravity=[0, 0, 0], kpin=1e6, k_barrier=0,
        node_box_max=.1, node_box_min=.001, node_box_update_count=1,
        max_global_iters=5, gs_prune_box_contacts=prune,
        use_ccd_guess=False, use_colored_ccd_guess=False, use_colored_ccd_guess_level_based=False)
    return payload


class SolverIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_inherited_timestep_solves_our_finite_pin_potential(self):
        self.assertIs(SolverIPC.step, SolverVBD.step)
        for device in self.devices:
            with self.subTest(device=str(device)):
                payload = point_payload()
                model = create_model(payload, device)
                solver = SolverIPC(model, allow_cpu=True)
                self.assertIsNotNone(solver._gs_schedule)
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

    def test_upstream_mid_checkpoint_uses_new_centers_and_finds_reachable_contact(self):
        from .test_movement_box_queries import pairs

        self.assertIs(SolverIPC._refresh_collision_sets, SolverVBD._refresh_collision_sets)
        for device in self.devices:
            for prune in (False, True):
                with self.subTest(device=str(device), prune=prune):
                    payload = checkpoint_payload(prune)
                    model = create_model(payload, device)
                    solver = SolverIPC(model, allow_cpu=True)
                    # Mimic a later substep: only the moving point needs the
                    # large budget; the stationary triangle has minimum boxes.
                    solver.movement_boxes.previous_displacement.assign([.1, 0, 0, 0])
                    before, after = model.state(), model.state()
                    dt = payload["params"]["dt"]
                    solver._initialize_particles(before, after, None, dt)
                    solver.check_status()
                    self.assertEqual(pairs(solver.contact_backend)[0], set())
                    old_reference = solver.pos_prev_collision_detection.numpy().copy()
                    proposal = before.particle_q.numpy().copy()
                    proposal[0, 2] = .0085
                    gs = solver._gs_schedule
                    solver.ccd_backend.accept(before.particle_q,
                        wp.array(proposal, wp.vec3, device=device), -1, dt,
                        center=solver.pos_prev_collision_detection, radii=solver.movement_boxes.radii,
                        colors=gs.colors, current_color=gs.current_color, query_view=gs.query_view,
                        independent=True)
                    solver.check_status()
                    checkpoint = before.particle_q.numpy().copy()
                    self.assertGreater(float(old_reference[0, 2] - checkpoint[0, 2]), .09)

                    # This inherited method invokes our detection hook with
                    # reset_reference=False, then updates the CCD reference.
                    solver._refresh_collision_sets(before, dt,
                        run_rigid_collision=False, run_soft_self_collision=True)
                    solver.check_status()
                    np.testing.assert_array_equal(solver.pos_prev_collision_detection.numpy(), checkpoint)
                    radii = solver.movement_boxes.radii.numpy()[:, None]
                    np.testing.assert_allclose(solver.contact_backend.swept_lower.numpy(),
                        checkpoint.astype(np.float64) - radii, rtol=0, atol=8e-8)
                    np.testing.assert_allclose(solver.contact_backend.swept_upper.numpy(),
                        checkpoint.astype(np.float64) + radii, rtol=0, atol=8e-8)
                    # The old boxes excluded this pair. The new point box
                    # crosses the plane, so it must be in the next CCD bank.
                    self.assertEqual(pairs(solver.contact_backend)[0], {(0, 0)})
                    proposal = checkpoint.copy(); proposal[0, 2] = -.05
                    solver.ccd_backend.accept(before.particle_q,
                        wp.array(proposal, wp.vec3, device=device), -1, dt,
                        center=solver.pos_prev_collision_detection, radii=solver.movement_boxes.radii,
                        colors=gs.colors, current_color=gs.current_color, query_view=gs.query_view,
                        independent=True)
                    solver.check_status()
                    accepted = before.particle_q.numpy()
                    self.assertGreater(float(accepted[0, 2]), 0)
                    self.assertLess(float(accepted[0, 2]), float(checkpoint[0, 2]))
                    np.testing.assert_array_equal(accepted[1:], checkpoint[1:])

    def test_multiple_box_checkpoints_stay_safe_in_eager_and_captured_frames(self):
        for device in self.devices:
            for prune in (False, True):
                results = []
                for graphs in (False, True):
                    with self.subTest(device=str(device), prune=prune, graphs=graphs):
                        payload = checkpoint_payload(prune)
                        self.assertGreater(payload["params"]["max_global_iters"], payload["params"]["node_box_update_count"])
                        model = create_model(payload, device)
                        solver = SolverIPC(model, allow_cpu=True)
                        solver.movement_boxes.previous_displacement.assign([.1, 0, 0, 0])
                        frames = FrameGraph(model, solver, graphs=graphs)
                        for _ in range(2):
                            frames.advance()
                            q = frames.state.particle_q.numpy()
                            self.assertGreater(float(q[0, 2]), 0)
                            np.testing.assert_array_equal(q[1:], np.asarray(payload["positions"], np.float32)[1:])
                            reference = solver.pos_prev_collision_detection.numpy().astype(np.float64)
                            radius = solver.movement_boxes.radii.numpy()[:, None]
                            np.testing.assert_allclose(solver.contact_backend.swept_lower.numpy(),
                                reference - radius, rtol=0, atol=8e-8)
                            np.testing.assert_allclose(solver.contact_backend.swept_upper.numpy(),
                                reference + radius, rtol=0, atol=8e-8)
                        results.append((frames.state.particle_q.numpy(), frames.state.particle_qd.numpy()))
                for eager, captured in zip(*results):
                    np.testing.assert_array_equal(captured, eager)


if __name__ == "__main__":
    unittest.main()
