"""Rigid state integration, contact coloring and curved-path safety checks."""

import unittest

import numpy as np
import warp as wp

from gpu import body_solver
from gpu.scene import SCHEMA, create_model
from gpu.solver import SolverIPC


def body_payload(centers=((0.0, 2.0, 0.0),), *, rotate=False):
    reference = np.asarray([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0, 0.2, 0]])
    positions, owners, bodies, tris = [], [], [], []
    for b, center in enumerate(centers):
        nodes = list(range(len(positions), len(positions) + 3))
        positions.extend((reference + center).tolist()); owners.extend([b] * 3); tris.append(nodes)
        bodies.append({"nodes": nodes, "reference_positions": reference.tolist(), "mass": 3.0,
            "I_hat": np.diag([0.02, 0.06, 0.01]).reshape(-1).tolist(), "mode": 0,
            "translate": True, "rotate": rotate, "com": list(center), "velocity": [0.1, 0.2, 0.3],
            "orientation": [1, 0, 0, 0], "omega": [0, 0, 0]})
    n = len(positions)
    return {"schema": SCHEMA, "kind": "scene", "example": 5,
        "params": {"mu": 0.0, "lambda": 0.0, "solid_mu": 0.0, "solid_lambda": 0.0,
            "kB": 0.0, "kpin": 40.0, "gravity": [0.0, -2.0, 0.0], "rigid_pin_time": 0.0,
            "dt": 0.05, "fps": 20.0, "substeps": 1, "fixed_iters": True,
            "max_global_iters": 2, "node_box_max": 0.02, "node_box_min": 0.001,
            "theta_box_max": 0.1, "theta_box_min": 0.01, "node_box_update_count": 10,
            "d_hat": 0.005, "k_barrier": 100.0, "k_sdf": 0.0, "eps_sdf": 0.002,
            "friction_coefficient": 0.0, "friction_velocity_epsilon": 0.01, "damping": 1.0,
            "use_ccd": True, "use_ccd_guess": False, "exact_computation_fallback": True},
        "positions": positions, "velocities": [[0.1, 0.2, 0.3]] * n, "mass": [1.0] * n,
        "node_to_rb": owners, "tris": tris, "tets": [], "triangle_dm_inverse": [],
        "triangle_areas": [], "hinges": [], "tet_rest_data": [], "pins": [],
        "pin_targets": [[0, 0, 0]] * n, "bodies": bodies, "rigid_pins": [], "rigid_pin_targets": [],
        "tet_nodes": [], "surface_nodes": list(range(n)), "deformable_nodes": [], "colliders": []}


@wp.kernel
def _pair(q: wp.array[wp.vec3], reference: wp.array[wp.vec3], owners: wp.array[int],
           target_center: wp.vec3, target_rotation: wp.quat, rotational: int,
           exact_geometry: int, words: wp.array[wp.uint32], slots: wp.array[int], result: wp.array[wp.vec2d]):
    result[0] = body_solver.certify_pair(q, reference, owners, wp.vec4i(0, 1, 2, 3), 0, 0,
        wp.vec3(0.0), wp.quat_identity(), target_center, target_rotation, rotational, exact_geometry, words, slots, 0)


body_solver.contact.configure_module(__name__, include_filter=False)
body_solver.rigid.configure_module(__name__)
body_solver.sdf.configure_module(__name__)
body_solver.configure_module(__name__, ("rigid_ccd.cuh", "original_rigid.cuh"))


class BodySolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_free_translation_and_finite_pin_closed_solution(self):
        for pinned in (False, True):
            for device in self.devices:
                with self.subTest(pinned=pinned, device=str(device)):
                    payload = body_payload()
                    if pinned:
                        payload["rigid_pins"] = [{"body": 0, "local_position": [0, 0, 0], "initial_target": [0.01, 2.01, 0.01]}]
                        payload["rigid_pin_targets"] = [[0.01, 2.01, 0.01]]
                    model = create_model(payload, device)
                    solver = SolverIPC(model, allow_cpu=True)
                    before, after = model.state(), model.state()
                    previous_q = before.particle_q.numpy().copy()
                    dt = payload["params"]["dt"]
                    solver.step(before, after, model.control(), None, dt)
                    solver.check_status()
                    center = np.asarray(payload["bodies"][0]["com"])
                    velocity = np.asarray(payload["bodies"][0]["velocity"])
                    mass = payload["bodies"][0]["mass"]
                    rhs = mass * (center + dt * velocity + dt * dt * np.asarray([0, -2, 0]))
                    denominator = mass
                    if pinned:
                        stiffness = dt * dt * payload["params"]["kpin"]
                        rhs += stiffness * np.asarray(payload["rigid_pin_targets"][0])
                        denominator += stiffness
                    expected = rhs / denominator
                    np.testing.assert_allclose(after.body_q.numpy()[0, :3], expected, rtol=2e-6, atol=2e-7)
                    np.testing.assert_allclose(after.body_qd.numpy()[0, :3], (expected - center) / dt, rtol=5e-5, atol=5e-6)
                    np.testing.assert_array_equal(after.body_qd.numpy()[0, 3:], 0.0)
                    refs = np.asarray(payload["bodies"][0]["reference_positions"])
                    np.testing.assert_allclose(after.particle_q.numpy(), refs + expected, rtol=1e-6, atol=3e-7)
                    np.testing.assert_allclose(after.particle_qd.numpy(), (after.particle_q.numpy() - previous_q) / dt,
                        rtol=2e-5, atol=3e-6)

    def test_free_body_honors_all_gravity_components(self):
        for device in self.devices:
            with self.subTest(device=str(device)):
                payload = body_payload()
                payload["params"].update(gravity=[1.0, -2.0, 3.0], node_box_max=1.0)
                payload["bodies"][0]["omega"] = [0.4, -0.3, 0.2]
                model = create_model(payload, device)
                solver = SolverIPC(model, allow_cpu=True)
                before, after = model.state(), model.state()
                dt = payload["params"]["dt"]
                center = np.asarray(payload["bodies"][0]["com"])
                velocity = np.asarray(payload["bodies"][0]["velocity"])
                gravity = np.asarray(payload["params"]["gravity"])
                predicted_velocity = velocity + dt * gravity
                expected = center + dt * predicted_velocity
                solver.step(before, after, model.control(), None, dt)
                solver.check_status()
                np.testing.assert_allclose(after.body_q.numpy()[0, :3], expected, rtol=2e-6, atol=3e-7)
                np.testing.assert_allclose(after.body_qd.numpy()[0, :3], predicted_velocity,
                                           rtol=5e-5, atol=5e-6)
                bank = solver._rigid.previous_velocity.numpy()[0]
                np.testing.assert_allclose(bank[:3], predicted_velocity, rtol=2e-6, atol=1e-7)
                np.testing.assert_array_equal(bank[3:], np.asarray(payload["bodies"][0]["omega"], dtype=np.float32))

    def test_body_gravity_reads_mutable_newton_world_array(self):
        for device in self.devices:
            payload = body_payload()
            payload["params"]["node_box_max"] = 1.0
            model = create_model(payload, device)
            solver = SolverIPC(model, allow_cpu=True)
            gravity = np.asarray([1.0, -2.0, 3.0], dtype=np.float32)
            model.gravity.assign(np.tile(gravity, (model.gravity.shape[0], 1)))
            before, after = model.state(), model.state()
            dt = payload["params"]["dt"]
            center = np.asarray(payload["bodies"][0]["com"])
            velocity = np.asarray(payload["bodies"][0]["velocity"])
            expected = center + dt * velocity + dt * dt * gravity
            solver.step(before, after, model.control(), None, dt)
            solver.check_status()
            np.testing.assert_allclose(after.body_q.numpy()[0, :3], expected, rtol=2e-6, atol=3e-7)

    def test_gpu_conflict_colors_separate_contacting_bodies(self):
        for device in self.devices:
            payload = body_payload(((0, 2, 0), (0, 2, 0.012), (10, 2, 0)))
            model = create_model(payload, device)
            solver = SolverIPC(model, allow_cpu=True)
            state = model.state()
            solver._collision_detection_penetration_free(state)
            solver.check_status()
            body = solver._rigid
            colors = body.colors.numpy()
            self.assertNotEqual(colors[0], colors[1])
            self.assertEqual(colors[0], colors[2])
            offsets = body.offsets.numpy()
            self.assertGreater(offsets[1], 0)
            self.assertGreater(offsets[2] - offsets[1], 0)
            self.assertEqual(offsets[3] - offsets[2], 0)
            # Every candidate involving two actual bodies adds both incidence
            # rows and imposes a conflict, regardless of barrier activation.
            self.assertEqual(int(body.color_count.numpy()[0]), 2)

    def test_large_com_proposal_stays_inside_exact_checkpoint_radius(self):
        for device in self.devices:
            payload = body_payload(((1000.0, 2.0, 0.0),))
            payload["bodies"][0]["velocity"] = [100.0, 0.0, 0.0]
            payload["velocities"] = [[100.0, 0.0, 0.0]] * len(payload["positions"])
            payload["params"].update(gravity=[0.0, 0.0, 0.0], max_global_iters=1)
            model = create_model(payload, device)
            solver = SolverIPC(model, allow_cpu=True)
            before, after = model.state(), model.state()
            anchor = before.body_q.numpy()[0, :3].astype(np.float64).copy()
            solver.step(before, after, model.control(), None, payload["params"]["dt"])
            solver.check_status()
            displacement = after.body_q.numpy()[0, :3].astype(np.float64) - anchor
            radius = payload["params"]["node_box_max"]
            self.assertGreater(displacement[0], 0.0)
            self.assertTrue(np.all(np.abs(displacement) <= radius - 1.0e-10))

    def test_full_ipc_packet_pushes_two_rigid_surfaces_apart(self):
        for device in self.devices:
            payload = body_payload(((0, 2, 0), (0, 2, 0.003)))
            payload["params"].update(dt=0.005, gravity=[0, 0, 0], max_global_iters=1)
            for body in payload["bodies"]:
                body["velocity"] = [0, 0, 0]
            payload["velocities"] = [[0, 0, 0]] * len(payload["positions"])
            model = create_model(payload, device)
            solver = SolverIPC(model, allow_cpu=True)
            before, after = model.state(), model.state()
            solver.step(before, after, model.control(), None, 0.005)
            solver.check_status()
            centers = after.body_q.numpy()[:, :3]
            self.assertLess(centers[0, 2], 0.0)
            self.assertGreater(centers[1, 2], 0.003)
            self.assertGreater(centers[1, 2] - centers[0, 2], 0.003)
            self.assertTrue(np.isfinite(after.body_qd.numpy()).all())

    def test_captured_frames_match_eager_rigid_rotation_and_odd_buffers(self):
        from gpu.run import FrameGraph

        for device in self.devices:
            for substeps in (1, 3, 10):
                with self.subTest(device=str(device), substeps=substeps):
                    payload = body_payload(rotate=True)
                    payload["bodies"][0]["omega"] = [0.1, 0.2, 0.3]
                    payload["params"].update(substeps=substeps, dt=1.0 / (20.0 * substeps))
                    results = []
                    for graphs in (False, True):
                        model = create_model(payload, device)
                        solver = SolverIPC(model, allow_cpu=True)
                        controller = FrameGraph(model, solver, graphs=graphs)
                        for _ in range(3):
                            controller.advance()
                        self.assertEqual(int(controller.motion.counter.numpy()[0]), 3 * substeps)
                        self.assertEqual(controller.parity, (3 * substeps) % 2)
                        state = controller.state
                        results.append((state.body_q.numpy(), state.body_qd.numpy(), state.particle_q.numpy()))
                    for eager, captured in zip(results[0], results[1]):
                        np.testing.assert_array_equal(captured, eager)

    def test_analytic_arc_hits_when_endpoint_chord_misses(self):
        q = np.asarray([[1, 0, 0], [0, 0.9, -0.2], [0, 1.1, -0.2], [0, 1, 0.2]], dtype=np.float32)
        # The chord has y=0 throughout; the half-circle crosses (0,1,0).
        self.assertTrue(np.all(q[1:, 1] > 0.0))
        for device in self.devices:
            positions = wp.array(q, dtype=wp.vec3, device=device)
            references = wp.array(q, dtype=wp.vec3, device=device)
            owners = wp.array([0, -1, -1, -1], dtype=int, device=device)
            words = wp.empty(2048 * 512 if device.is_cuda else 0, wp.uint32, device=device)
            slots = wp.empty(512 if device.is_cuda else 0, int, device=device)
            result = wp.empty(1, wp.vec2d, device=device)
            wp.launch(_pair, 1, [positions, references, owners, wp.vec3(0), wp.quat(0, 0, 1, 0),
                1, 1, words, slots, result], block_dim=1, device=device)
            fraction, status = result.numpy()[0]
            self.assertEqual(status, 0)
            self.assertGreater(fraction, 0.0)
            self.assertLess(fraction, 0.5)

    def test_translation_crossing_and_initial_intersection_fail_closed(self):
        triangle = [[0, -1, -1], [0, 1, -1], [0, 0, 1]]
        for start in ([1, 0, 0], [0, 0, 0]):
            for device in self.devices:
                q = np.asarray([start, *triangle], dtype=np.float32)
                arrays = [wp.array(q, dtype=wp.vec3, device=device), wp.array(q, dtype=wp.vec3, device=device),
                    wp.array([0, -1, -1, -1], dtype=int, device=device)]
                words = wp.empty(2048 * 512 if device.is_cuda else 0, wp.uint32, device=device)
                slots = wp.empty(512 if device.is_cuda else 0, int, device=device)
                result = wp.empty(1, wp.vec2d, device=device)
                for exact_geometry in (0, 1):
                    wp.launch(_pair, 1, [*arrays, wp.vec3(-2, 0, 0), wp.quat_identity(), 0, exact_geometry,
                        words, slots, result], block_dim=1, device=device)
                    fraction, status = result.numpy()[0]
                    if start[0] == 0:
                        self.assertEqual(status, 1)
                    else:
                        self.assertEqual(status, 0)
                        self.assertLess(fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
