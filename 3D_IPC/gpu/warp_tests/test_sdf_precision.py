"""Independent full-curvature SDF checks for the compact physical evaluator.

The unchanged 42-double diagnostic evaluator is an oracle. Friction-free
forces and Hessians are also checked against finite differences of a separate
NumPy penalty energy, including the tangential curvature of curved obstacles.
"""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import warp as wp

import gpu

_gpu_path = str(Path(__file__).resolve().parents[1])
if _gpu_path not in gpu.__path__:
    gpu.__path__.append(_gpu_path)

from gpu import sdf, sdf_fast as fast


@wp.kernel(module="unique", enable_backward=False)
def _oracle(q: wp.array[wp.vec3], previous: wp.array[wp.vec3],
            colliders: wp.array[sdf.Collider], params: sdf.Parameters,
            packets: wp.array[sdf.Packet], forces: wp.array[wp.vec3],
            hessians: wp.array[wp.mat33]):
    i = wp.tid()
    packet = sdf.evaluate(q[i], previous[i], colliders, params)
    packets[i] = packet
    if packet[41] == wp.float64(0.0):
        f, h = sdf.vertex_terms(packet, wp.float32(params.dt))
        forces[i] = f
        hessians[i] = h
    else:
        forces[i] = wp.vec3(0.0)
        hessians[i] = wp.mat33(0.0)


@wp.kernel(module="unique", enable_backward=False)
def _compact(q: wp.array[wp.vec3], previous: wp.array[wp.vec3],
             colliders: wp.array[sdf.Collider], params: sdf.Parameters,
             result: wp.array[fast.Result]):
    i = wp.tid()
    result[i] = fast.evaluate(q[i], previous[i], colliders, params)


sdf.configure_module(_oracle)
fast.configure_module(_compact)


def _record(kind, *, point=(0, 0, 0), axis=(0, 1, 0), radius=1.0,
            previous_pose=None, current_pose=None):
    result = {"kind": kind, "point": list(point), "axis": list(axis), "radius": radius}
    if previous_pose is not None:
        result["previous_pose"] = previous_pose
    if current_pose is not None:
        result["current_pose"] = current_pose
    return result


class SdfPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def fixture(self, device, points, records, *, previous=None, friction=0.0,
                epsilon=.05, stiffness=370.0, dt=.031, eps_v=.1,
                colors=None, flags=None, solid=None, surface=None):
        points = np.asarray(points, np.float32).reshape(-1, 3)
        n = len(points)
        if previous is None:
            previous = points.copy()
        data = sdf.pack_colliders(records)
        params = {"k_sdf": stiffness, "eps_sdf": epsilon, "dt": dt,
                  "friction_coefficient": friction, "friction_velocity_epsilon": eps_v}
        model = SimpleNamespace(
            device=wp.get_device(device), particle_count=n,
            particle_flags=wp.array(np.ones(n, np.int32) if flags is None else flags,
                                    int, device=device),
            particle_colors=wp.array(np.zeros(n, np.int32) if colors is None else colors,
                                     int, device=device),
            ipc=SimpleNamespace(
                params=params,
                collider_data=wp.array(data, float, device=device),
                solid=wp.array(np.zeros(n, np.int32) if solid is None else solid,
                               int, device=device),
                surface=wp.array(np.ones(n, np.int32) if surface is None else surface,
                                 int, device=device)),
        )
        error = wp.zeros(1, int, device=device)
        backend = sdf.SdfBackend(model, error, include_curvature=True)
        q = wp.array(points, wp.vec3, device=device)
        old = wp.array(np.asarray(previous, np.float32), wp.vec3, device=device)
        backend.prepare(dt)
        return SimpleNamespace(q=q, previous=old, model=model, backend=backend,
                               error=error, records=records)

    def evaluate(self, data):
        n = data.model.particle_count
        packets = wp.empty(n, sdf.Packet, device=data.q.device)
        forces = wp.empty(n, wp.vec3, device=data.q.device)
        hessians = wp.empty(n, wp.mat33, device=data.q.device)
        compact = wp.empty(n, fast.Result, device=data.q.device)
        wp.launch(_oracle, n, [data.q, data.previous, data.backend.colliders,
                              data.backend.params, packets, forces, hessians], device=data.q.device)
        wp.launch(_compact, n, [data.q, data.previous, data.backend.colliders,
                               data.backend.params, compact], device=data.q.device)
        return packets.numpy(), forces.numpy(), hessians.numpy(), compact.numpy()

    def accumulate(self, data, color=0, *, check=True):
        n = data.model.particle_count
        f = wp.zeros(n, wp.vec3, device=data.q.device)
        h = wp.zeros(n, wp.mat33, device=data.q.device)
        data.backend.accumulate(data.q, data.previous, color, f, h, float(data.backend.params.dt))
        if check:
            self.assertEqual(int(data.error.numpy()[0]), 0)
        return f.numpy(), h.numpy()

    def assert_norm(self, actual, expected, *, rtol=5e-4, atol=2e-6):
        actual = np.asarray(actual, np.float64)
        expected = np.asarray(expected, np.float64)
        self.assertTrue(np.isfinite(actual).all())
        error = float(np.linalg.norm(actual - expected))
        norm = float(np.linalg.norm(expected))
        self.assertLessEqual(error, atol + rtol * norm,
                             f"relative error {error / max(norm, atol, np.finfo(np.float64).tiny):.8g}")

    def assert_oracle(self, data, *, expected_selected=None):
        packets, expected_f, expected_h, compact = self.evaluate(data)
        self.assertTrue(np.all(packets[:, 41] == 0))
        actual_f, actual_h = self.accumulate(data)
        self.assert_norm(actual_f, expected_f)
        self.assert_norm(actual_h, expected_h)
        for i, result in enumerate(compact):
            self.assertIn(int(result[12]), (0, 6))
            if int(result[12]) == 0:
                self.assert_norm(result[:3], expected_f[i])
                self.assert_norm(result[3:12].reshape(3, 3), expected_h[i])
                self.assertEqual(int(result[13]), int(packets[i, 40]))
        if expected_selected is not None:
            np.testing.assert_array_equal(packets[:, 40], expected_selected)
        return actual_f, actual_h, compact

    @staticmethod
    def penalty_energy(q, record, stiffness, epsilon):
        delta = np.asarray(q, np.float64) - np.asarray(record["point"], np.float64)
        axis = np.asarray(record.get("axis", (0, 1, 0)), np.float64)
        if record["kind"] == "plane":
            phi = float(np.dot(delta, axis))
        else:
            if record["kind"] == "cylinder":
                delta = delta - axis * np.dot(delta, axis)
            phi = float(np.linalg.norm(delta) - record["radius"])
        if epsilon <= 0:
            return .5 * stiffness * min(phi, 0.0) ** 2
        weight = 1.0 if phi < 0 else max(0.0, (epsilon - phi) / epsilon)
        return .5 * stiffness * weight * (epsilon - phi) ** 2

    def test_penalty_force_and_full_curvature_match_independent_energy_differences(self):
        fixtures = [(_record("plane"), [(.2, .02, -.1)]),
                    (_record("sphere"), [(1.02, .1, -.05)]),
                    (_record("cylinder"), [(1.02, .2, .05)])]
        for device in self.devices:
            for record, point in fixtures:
                for epsilon, scale in ((.05, 1.0), (0.0, .93)):
                    q = np.asarray(point, np.float32)
                    if epsilon == 0:
                        q[0] *= scale
                        if record["kind"] == "plane":
                            q[0, 1] = -.03
                    data = self.fixture(device, q, [record], epsilon=epsilon)
                    self.assertEqual(int(data.backend.params.include_curvature), 1)
                    f, h, _ = self.assert_oracle(data)
                    x = data.q.numpy()[0].astype(np.float64)
                    packed_record = dict(record)
                    packed_record["point"] = np.asarray(record["point"], np.float32).astype(np.float64)
                    packed_record["radius"] = float(np.float32(record["radius"]))
                    energy = lambda value: self.penalty_energy(value, packed_record, 370.0, epsilon)
                    step = 2e-5
                    base = energy(x)
                    gradient, hessian = np.zeros(3), np.zeros((3, 3))
                    for k in range(3):
                        delta = np.eye(3)[k] * step
                        plus, minus = energy(x + delta), energy(x - delta)
                        gradient[k] = (plus - minus) / (2 * step)
                        hessian[k, k] = (plus - 2 * base + minus) / step**2
                        for l in range(k):
                            other = np.eye(3)[l] * step
                            value = (energy(x + delta + other) - energy(x + delta - other)
                                     - energy(x - delta + other) + energy(x - delta - other)) / (4 * step**2)
                            hessian[k, l] = hessian[l, k] = value
                    self.assert_norm(f[0], -gradient)
                    self.assert_norm(h[0], hessian)
                    if record["kind"] != "plane":
                        normal = x.copy()
                        if record["kind"] == "cylinder":
                            normal[1] = 0
                        normal /= np.linalg.norm(normal)
                        tangent = np.cross(normal, [0., 1., 0.])
                        tangent /= np.linalg.norm(tangent)
                        self.assertGreater(abs(tangent @ hessian @ tangent), 1.0)
                        np.testing.assert_allclose(tangent @ h[0] @ tangent,
                                                   tangent @ hessian @ tangent, rtol=5e-4, atol=2e-4)

    def test_active_penetrated_transition_and_inactive_friction_regimes(self):
        for device in self.devices:
            for kind in ("plane", "sphere", "cylinder"):
                points = ([[.2, phi, .1] for phi in (-.04, .015, .09)] if kind == "plane" else
                          [[1 + phi, .0 if kind == "sphere" else .2, 0] for phi in (-.04, .015, .09)])
                for friction, slip in ((0.0, 0.0), (.2, 2e-5), (.2, .02)):
                    previous = np.asarray(points, np.float32).copy()
                    previous[:, 2] -= slip
                    data = self.fixture(device, points, [_record(kind)], previous=previous,
                                        friction=friction)
                    f, h, _ = self.assert_oracle(data)
                    self.assertFalse(np.any(f[-1]))
                    self.assertFalse(np.any(h[-1]))
                    if friction and slip:
                        self.assertLess(float(f[0, 2]), 0.0)

    def test_union_minimum_and_first_tie_preserve_material_identity(self):
        stationary = _record("plane")
        moving = _record("plane", current_pose={"translation": [.003, 0, 0]})
        for device in self.devices:
            first = self.fixture(device, [[.2, .02, .1]], [stationary, moving], friction=.2)
            second = self.fixture(device, [[.2, .02, .1]], [moving, stationary], friction=.2)
            a, _, _ = self.assert_oracle(first, expected_selected=[0])
            b, _, _ = self.assert_oracle(second, expected_selected=[0])
            self.assertEqual(float(a[0, 0]), 0.0)
            self.assertGreater(float(b[0, 0]), 0.0)
            deeper = self.fixture(device, [[.2, .02, .1]],
                                  [stationary, _record("plane", point=(0, .01, 0))])
            self.assert_oracle(deeper, expected_selected=[1])

    def test_moving_material_translation_and_rotation_match_legacy_friction(self):
        angle = .17
        rotation = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]]
        for device in self.devices:
            for kind, point in (("plane", [.3, .02, -.1]),
                                ("sphere", [1.02, 0, 0]),
                                ("cylinder", [1.02, .2, 0])):
                for pose in ({"translation": [.003, 0, -.002]},
                             {"rotation": rotation, "translation": [.003, 0, -.002]}):
                    data = self.fixture(device, [point], [_record(kind, current_pose=pose)], friction=.2)
                    self.assert_oracle(data)

    def test_color_flags_and_surface_selection_only_commit_owned_vertices(self):
        for device in self.devices:
            points = [[.2, .02, .1]] * 6
            data = self.fixture(device, points, [_record("plane")],
                                colors=[0, 1, 0, 1, 0, 0], flags=[1, 1, 0, 1, 1, 1],
                                solid=[0, 0, 0, 0, 1, 1], surface=[1, 1, 1, 1, 0, 1])
            packets, expected_f, expected_h, _ = self.evaluate(data)
            self.assertTrue(np.all(packets[:, 41] == 0))
            for color, active in ((0, [0, 5]), (1, [1, 3])):
                f, h = self.accumulate(data, color)
                mask = np.zeros(6, bool)
                mask[active] = True
                expected = expected_f.copy()
                expected[~mask] = 0
                self.assert_norm(f, expected)
                expected = expected_h.copy()
                expected[~mask] = 0
                self.assert_norm(h, expected)

    def test_device_color_selector_preserves_physics_and_capture_replay(self):
        groups = [0, 1, 0, 1, 0, 0]
        for device in self.devices:
            for precision in ("float32", "float64"):
                for friction in (0.0, .2):
                    with self.subTest(device=str(device), precision=precision, friction=friction):
                        data = self.fixture(device, [[.2, .02, .1]] * 6, [_record("plane")],
                            previous=[[.19, .02, .1]] * 6, friction=friction,
                            colors=[7] * 6, flags=[1, 1, 0, 1, 1, 1],
                            solid=[0, 0, 0, 0, 1, 1], surface=[1, 1, 1, 1, 0, 1])
                        data.backend = sdf.SdfBackend(data.model, data.error,
                            include_curvature=True, precision=precision)
                        data.backend.prepare(float(data.backend.params.dt))
                        _, expected_f, expected_h, _ = self.evaluate(data)
                        colors = wp.array(groups, int, device=device)
                        current = wp.zeros(1, int, device=device)
                        f = wp.zeros(6, wp.vec3, device=device)
                        h = wp.zeros(6, wp.mat33, device=device)

                        def launch():
                            data.backend.accumulate(data.q, data.previous, -1, f, h,
                                float(data.backend.params.dt), colors=colors, current_color=current)

                        graph = None
                        if device.is_cuda:
                            launch()  # Compile before capture.
                            with wp.ScopedCapture(device=device) as capture:
                                launch()
                            graph = capture.graph
                        for color, active in ((0, [0, 5]), (1, [1, 3])):
                            f.zero_()
                            h.zero_()
                            current.fill_(color)
                            if graph is None:
                                launch()
                            else:
                                wp.capture_launch(graph)
                            mask = np.zeros(6, bool)
                            mask[active] = True
                            self.assertEqual(int(data.error.numpy()[0]), 0)
                            self.assert_norm(f.numpy(), np.where(mask[:, None], expected_f, 0))
                            self.assert_norm(h.numpy(), np.where(mask[:, None, None], expected_h, 0))

    def test_device_color_checked_path_still_validates_unprepared_material_pose(self):
        invalid_pose = {"rotation": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]}
        for device in self.devices:
            data = self.fixture(device, [[.2, .02, .1]],
                [_record("plane", current_pose=invalid_pose)], friction=0.0)
            data.backend.params.friction = .2
            current = wp.zeros(1, int, device=device)
            f = wp.zeros(1, wp.vec3, device=device)
            h = wp.zeros(1, wp.mat33, device=device)
            data.backend.accumulate(data.q, data.previous, -1, f, h,
                float(data.backend.params.dt), current_color=current)
            self.assertNotEqual(int(data.error.numpy()[0]), 0)
            self.assertFalse(np.any(f.numpy()))
            self.assertFalse(np.any(h.numpy()))

    def test_large_translation_tiny_geometry_and_cutoff_neighbors_match_oracle(self):
        fixtures = [
            ([[4097.02, 8192, -4096]], [_record("sphere", point=(4096, 8192, -4096))], .05),
            ([[1.02e-9, 0, 0]], [_record("sphere", radius=1e-9)], 5e-11),
            ([[0, 0, 0]], [_record("sphere")], .05),
            ([[0, .3, 0]], [_record("cylinder")], .05),
        ]
        epsilon = np.float32(.05)
        gaps = [np.nextafter(epsilon, np.float32(-np.inf)), epsilon,
                np.nextafter(epsilon, np.float32(np.inf))]
        fixtures.append(([[.2, gap, .1] for gap in gaps], [_record("plane")], .05))
        for device in self.devices:
            for points, records, smoothing in fixtures:
                data = self.fixture(device, points, records, epsilon=smoothing, friction=.2)
                self.assert_oracle(data)

    def test_well_conditioned_obstacles_exercise_float32_path(self):
        rng = np.random.default_rng(93127)
        for device in self.devices:
            accepted = total = 0
            for kind in ("plane", "sphere", "cylinder"):
                if kind == "plane":
                    points = rng.uniform(-.2, .2, (64, 3))
                    points[:, 1] = rng.uniform(.01, .035, 64)
                elif kind == "sphere":
                    direction = rng.normal(size=(64, 3))
                    direction /= np.linalg.norm(direction, axis=1)[:, None]
                    points = direction * rng.uniform(1.01, 1.035, (64, 1))
                else:
                    theta = rng.uniform(-np.pi, np.pi, 64)
                    radius = rng.uniform(1.01, 1.035, 64)
                    points = np.column_stack((radius * np.cos(theta), rng.uniform(-.3, .3, 64),
                                              radius * np.sin(theta)))
                previous = points.astype(np.float32)
                previous[:, 2] -= 1e-4
                data = self.fixture(device, points, [_record(kind)], previous=previous, friction=.2)
                _, _, compact = self.assert_oracle(data)
                accepted += int(np.count_nonzero(compact[:, 12] == 0))
                total += len(compact)
            self.assertGreaterEqual(accepted, int(.8 * total))

    def test_invalid_geometry_history_and_moving_pose_fail_closed(self):
        for device in self.devices:
            for records in ([_record("sphere", radius=-1)],
                            [_record("plane", axis=(0, np.nan, 0))],
                            [_record("plane", current_pose={"rotation": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]})]):
                data = self.fixture(device, [[.2, .02, .1]], records, friction=.2)
                f, h = self.accumulate(data, check=False)
                self.assertNotEqual(int(data.error.numpy()[0]), 0)
                self.assertFalse(np.any(f))
                self.assertFalse(np.any(h))
                # The public compact evaluator accepts an arbitrary collider
                # bank and must validate it independently of prepare's error
                # latch. Only the backend's private prepared-bank entry may
                # rely on preparation to have validated material poses.
                packets, _, _, compact = self.evaluate(data)
                self.assertNotEqual(int(packets[0, 41]), 0)
                self.assertNotEqual(int(compact[0, 12]), 0)
                self.assertFalse(np.any(compact[0, :12]))
            data = self.fixture(device, [[.2, .02, .1]], [_record("plane")],
                                previous=[[np.nan, .02, .1]], friction=.2)
            f, h = self.accumulate(data, check=False)
            self.assertNotEqual(int(data.error.numpy()[0]), 0)
            self.assertFalse(np.any(f))
            self.assertFalse(np.any(h))

    def test_empty_obstacles_and_ignored_material_pose_shortcuts_match_legacy(self):
        invalid_pose = {"rotation": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]}
        for device in self.devices:
            data = self.fixture(device, [[.2, .02, .1]], [], previous=[[np.nan, 0, 0]], friction=.2)
            self.assert_oracle(data, expected_selected=[-1])
            data = self.fixture(device, [[.2, .02, .1]],
                                [_record("plane", current_pose=invalid_pose)], friction=0)
            self.assert_oracle(data)
            data = self.fixture(device, [[.2, .02, .1]],
                                [_record("plane", previous_pose=invalid_pose, current_pose=invalid_pose)], friction=.2)
            self.assert_oracle(data)

    def test_nonrepresentable_physical_hessian_does_not_commit_infinity(self):
        for device in self.devices:
            data = self.fixture(device, [[0, -1e-20, 0]], [_record("plane")],
                                stiffness=1e40, epsilon=0)
            f, h = self.accumulate(data, check=False)
            self.assertNotEqual(int(data.error.numpy()[0]), 0)
            self.assertFalse(np.any(f))
            self.assertFalse(np.any(h))

    def test_invalid_scalar_parameters_are_rejected_before_force_commit(self):
        for arguments in ({"stiffness": -1}, {"epsilon": np.nan}, {"friction": -.1},
                          {"dt": 0}, {"eps_v": 0}):
            with self.assertRaises(ValueError):
                sdf.parameters(**arguments)
        for device in self.devices:
            for field, value in (("stiffness", np.nan), ("friction", -.1), ("eps_v", 0.0)):
                data = self.fixture(device, [[.2, .02, .1]], [_record("plane")], friction=.2)
                setattr(data.backend.params, field, value)
                f, h = self.accumulate(data, check=False)
                self.assertNotEqual(int(data.error.numpy()[0]), 0)
                self.assertFalse(np.any(f))
                self.assertFalse(np.any(h))

    def test_tiny_penalty_gradient_does_not_erase_representable_curvature(self):
        # At phi==0 the gradient is -k*epsilon, which is too small for
        # float32 here. Multiplication by the sphere curvature 1/r produces
        # a representable tangential Hessian of about -1e-20. Casting the
        # gradient to zero before that multiplication loses physical terms.
        for device in self.devices:
            radius = float(np.float32(1e-30))
            data = self.fixture(device, [[radius, 0, 0]],
                                [_record("sphere", radius=radius)],
                                stiffness=1e-20, epsilon=1e-30)
            packets, expected_f, expected_h, compact = self.evaluate(data)
            self.assertEqual(int(packets[0, 41]), 0)
            self.assertLess(float(expected_h[0, 1, 1]), -1e-21)
            self.assertLess(float(expected_h[0, 2, 2]), -1e-21)
            f, h = self.accumulate(data)
            self.assert_norm(f, expected_f, atol=0.0)
            self.assert_norm(h, expected_h, atol=0.0)
            if int(compact[0, 12]) == 0:
                self.assert_norm(compact[0, 3:12].reshape(3, 3), expected_h[0], atol=0.0)
            else:
                self.assertEqual(int(compact[0, 12]), 6)

    def test_friction_normal_threshold_keeps_representable_tangential_stiffness(self):
        # The legacy friction cutoff is ||grad(phi)|| > 1e-12. Squaring and
        # rounding the tiny normal in float32 can cross the branch despite a
        # representable friction Hessian. Preserve the binary64 decision.
        axis = (4.225212302429171e-13, 9.063530633478256e-13, 0.0)
        for device in self.devices:
            data = self.fixture(device, [[0, -.5, 0]], [_record("plane", axis=axis)],
                                previous=[[0, -.5, -.001]], stiffness=100,
                                epsilon=.02, dt=.01, friction=.2, eps_v=.1)
            packets, expected_f, expected_h, compact = self.evaluate(data)
            self.assertEqual(int(packets[0, 41]), 0)
            self.assertGreater(float(expected_h[0, 2, 2]), 1e-11)
            f, h = self.accumulate(data)
            self.assert_norm(f, expected_f, atol=0.0)
            self.assert_norm(h, expected_h, atol=0.0)
            if int(compact[0, 12]) == 0:
                self.assert_norm(compact[0, :3], expected_f[0], atol=0.0)
                self.assert_norm(compact[0, 3:12].reshape(3, 3), expected_h[0], atol=0.0)
            else:
                self.assertEqual(int(compact[0, 12]), 6)

    def test_enabling_friction_after_unchecked_prepare_still_validates_material_pose(self):
        # Friction-free preparation is allowed to leave material poses
        # unchecked. Enabling friction before the next prepare must therefore
        # use a checked evaluator, rather than treating that bank as trusted.
        invalid_pose = {"rotation": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]}
        for device in self.devices:
            data = self.fixture(device, [[.2, .02, .1]],
                                [_record("plane", current_pose=invalid_pose)], friction=0.0)
            self.assertEqual(int(data.error.numpy()[0]), 0)
            data.backend.params.friction = .2
            packets, _, _, compact = self.evaluate(data)
            self.assertNotEqual(int(packets[0, 41]), 0)
            self.assertNotEqual(int(compact[0, 12]), 0)
            f, h = self.accumulate(data, check=False)
            self.assertNotEqual(int(data.error.numpy()[0]), 0)
            self.assertFalse(np.any(f))
            self.assertFalse(np.any(h))


if __name__ == "__main__":
    unittest.main()
