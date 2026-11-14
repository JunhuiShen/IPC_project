"""Native function reuse and the retained own friction derivative limit."""

import inspect
import unittest

import numpy as np
import warp as wp
from newton._src.solvers.vbd import particle_vbd_kernels as stock

from gpu import contact, dat_friction


@wp.kernel
def friction_probe(
    normal: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
    normal_force: wp.array[float],
    coefficient: wp.array[float],
    epsilon: wp.array[float],
    force: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
    native_force: wp.array[wp.vec3],
    native_hessian: wp.array[wp.mat33],
    replay: wp.array[int],
):
    i = wp.tid()
    f, h, retry = dat_friction.evaluate_reference_friction_checked(
        normal[i], displacement[i], normal_force[i], coefficient[i], epsilon[i]
    )
    force[i] = f
    hessian[i] = h
    replay[i] = retry
    first, second = stock.orthonormal_basis(normal[i])
    tangent = stock.mat32(first[0], second[0], first[1], second[1], first[2], second[2])
    slip = wp.transpose(tangent) * displacement[i]
    f_native, h_native = stock.compute_friction(
        coefficient[i], normal_force[i], tangent, slip, epsilon[i]
    )
    native_force[i] = f_native
    native_hessian[i] = h_native


@wp.kernel
def contact_packet_probe(
    positions: wp.array[wp.vec3], previous: wp.array[wp.vec3],
    without_friction: contact.Parameters, with_friction: contact.Parameters,
    normal: wp.array[contact.Result], combined: wp.array[contact.Result],
):
    nodes = wp.vec4i(0, 1, 2, 3)
    normal[0] = contact.evaluate_self(positions, previous, nodes, 0, without_friction)
    combined[0] = contact.evaluate_self(positions, previous, nodes, 0, with_friction)


contact.configure_module(__name__)


class ReferenceFrictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device('cpu'), *wp.get_cuda_devices()]

    def evaluate(self, device, normals, displacements, *, normal_force=3.5, coefficient=.2, epsilon=.01, with_status=False):
        normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        displacements = np.asarray(displacements, dtype=np.float32).reshape(-1, 3)
        n = len(normals)
        self.assertEqual(displacements.shape, normals.shape)
        def scalar(value):
            return wp.array(np.broadcast_to(value, (n,)).astype(np.float32).copy(), float, device=device)
        outputs = [wp.empty(n, dtype, device=device) for dtype in (wp.vec3, wp.mat33, wp.vec3, wp.mat33, int)]
        wp.launch(friction_probe, n,
            [wp.array(normals, wp.vec3, device=device), wp.array(displacements, wp.vec3, device=device),
             scalar(normal_force), scalar(coefficient), scalar(epsilon), *outputs], device=device)
        values = [value.numpy() for value in outputs]
        return values if with_status else values[:4]

    def test_actual_native_function_objects_are_reused(self):
        self.assertIs(dat_friction.reference_compute_friction, stock.compute_friction)
        self.assertIs(dat_friction.reference_orthonormal_basis, stock.orthonormal_basis)
        self.assertIs(dat_friction.mat32, stock.mat32)
        source = inspect.getsource(dat_friction.evaluate_reference_friction_checked.func)
        self.assertIn('reference_compute_friction(', source)

    def test_nonzero_slip_matches_native_both_smoothing_regions_and_orientations(self):
        normals = np.asarray([[0, 0, 1], [1, 0, 0], [0, -1, 0], [.3, .4, np.sqrt(.75)]], np.float64)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        for device in self.devices:
            for length in (.002, .01, .03):
                with self.subTest(device=str(device), slip=length):
                    displacements = np.tile([length, -.7 * length, .4 * length], (4, 1))
                    f, h, native_f, native_h, replay = self.evaluate(device, normals, displacements, with_status=True)
                    np.testing.assert_array_equal(f, native_f)
                    np.testing.assert_array_equal(h, native_h)
                    np.testing.assert_array_equal(replay, 0)

    def test_zero_slip_keeps_own_positive_tangential_hessian(self):
        normals = np.asarray([[0, 0, 1], [1, 0, 0], [0, -1, 0], [.3, .4, np.sqrt(.75)]], np.float64)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        for device in self.devices:
            f, h, native_f, native_h = self.evaluate(device, normals, np.zeros((4, 3)))
            np.testing.assert_array_equal(f, np.zeros((4, 3), np.float32))
            np.testing.assert_array_equal(native_f, f)
            np.testing.assert_array_equal(native_h, np.zeros((4, 3, 3), np.float32))
            expected = np.asarray([.2 * 3.5 * (2 / .01) * (np.eye(3) - np.outer(n, n)) for n in normals])
            np.testing.assert_allclose(h, expected, rtol=2e-6, atol=2e-5)
            for matrix in h:
                self.assertGreater(np.linalg.eigvalsh(matrix).max(), 100.)

    def test_force_and_hessian_match_own_analytic_regularization(self):
        # The axis-aligned normal gives an independent tangent projection and
        # checks the constitutive force/Hessian rather than only native parity.
        motion = np.asarray([[.002, .001, .3], [.01, 0, -.1], [.03, .04, 0]], np.float64)
        for device in self.devices:
            f, h, _, _ = self.evaluate(device, [[0, 0, 1]] * 3, motion)
            tangent_motion = motion.copy(); tangent_motion[:, 2] = 0
            distance = np.linalg.norm(tangent_motion, axis=1)
            mollifier = np.where(distance < .01, 2 / .01 - distance / .01**2, 1 / distance)
            expected_force = -.2 * 3.5 * mollifier[:, None] * tangent_motion
            expected_hessian = .2 * 3.5 * mollifier[:, None, None] * np.diag([1, 1, 0])[None, :, :]
            np.testing.assert_allclose(f, expected_force, rtol=2e-6, atol=2e-7)
            np.testing.assert_allclose(h, expected_hessian, rtol=2e-6, atol=2e-5)

    def test_mu_zero_or_normal_force_zero_has_no_friction(self):
        for device in self.devices:
            for coefficient, force in ((0, 3.5), (.2, 0)):
                f, h, _, _ = self.evaluate(device, [[0, 0, 1]] * 2,
                    [[0, 0, 0], [.1, -.2, .7]], coefficient=coefficient, normal_force=force)
                np.testing.assert_array_equal(f, np.zeros((2, 3), np.float32))
                np.testing.assert_array_equal(h, np.zeros((2, 3, 3), np.float32))

    def test_normal_motion_does_not_create_macroscopic_tangential_force(self):
        normals = np.asarray([[0, 0, 1], [1, 0, 0], [.3, .4, np.sqrt(.75)]], np.float64)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        for device in self.devices:
            f, h, _, _ = self.evaluate(device, normals, 3 * normals)
            np.testing.assert_allclose(f, 0, rtol=0, atol=5e-5)
            expected = np.asarray([140 * (np.eye(3) - np.outer(n, n)) for n in normals])
            np.testing.assert_allclose(h, expected, rtol=4e-5, atol=3e-5)

    def test_nonzero_slip_matches_ordinary_float64_own_contact_packet(self):
        q = np.asarray([[.25, .25, .001], [0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        radius = float(np.float32(.002))
        dt = float(np.float32(1 / 600))
        eps_v = float(np.float32(.01))
        coefficient = float(np.float32(.2))
        eps_u = np.float32(np.float32(dt) * np.float32(eps_v))
        distance = float(q[0, 2])
        gap = distance - radius
        derivative = -2 * gap * np.log(distance / radius) - gap * gap / distance
        normal_force = -100 * derivative
        for device in self.devices:
            positions = wp.array(q, wp.vec3, device=device)
            for slip in (.5 * eps_u, eps_u, 2 * eps_u):
                old = q.copy(); old[0, 0] -= slip
                previous = wp.array(old, wp.vec3, device=device)
                normal = wp.empty(1, contact.Result, device=device)
                combined = wp.empty_like(normal)
                common = dict(d_hat=radius, k_barrier=100., dt=dt, eps_v=eps_v, exact_geometry=True)
                wp.launch(contact_packet_probe, 1,
                    [positions, previous, contact.parameters(friction=0., **common),
                     contact.parameters(friction=coefficient, **common), normal, combined], device=device)
                base, full = normal.numpy()[0], combined.numpy()[0]
                self.assertEqual(int(base[48]), 0)
                self.assertEqual(int(full[48]), 0)
                self.assertEqual(float(full[49]), 1.)
                f, h, _, _ = self.evaluate(device, [[0, 0, 1]], [q[0] - old[0]],
                    normal_force=normal_force, coefficient=coefficient, epsilon=eps_u)
                np.testing.assert_allclose(f[0], full[:3] - base[:3], rtol=3e-6, atol=2e-7)
                np.testing.assert_allclose(h[0], (full[3:12] - base[3:12]).reshape(3, 3), rtol=3e-6, atol=2e-4)

    def test_squared_norm_underflow_preserves_limiting_force(self):
        for device in self.devices:
            f, h, native_f, native_h, replay = self.evaluate(device, [[0, 0, 1]], [[1e-25, 0, 0]],
                normal_force=1e10, coefficient=.2, epsilon=.01, with_status=True)
            self.assertNotEqual(float(f[0, 0]), 0.)
            np.testing.assert_allclose(f[0], [-4e-14, 0, 0], rtol=2e-6, atol=0)
            np.testing.assert_allclose(h[0], 4e11 * np.diag([1, 1, 0]), rtol=2e-6, atol=0)
            np.testing.assert_array_equal(native_f, np.zeros((1, 3), np.float32))
            np.testing.assert_array_equal(native_h, np.zeros((1, 3, 3), np.float32))
            self.assertEqual(int(replay[0]), 1)

    def test_zero_norm_large_strength_combines_scalars_before_tiny_motion(self):
        # m*u underflows, but (mu*N*m)*u is a normal float32 force.
        mu = np.float32(1e31); normal_force = np.float32(.135); epsilon = np.float32(1e30)
        motion = np.float32(1e-25)
        scale = float(mu) * float(normal_force) * (2 / float(epsilon))
        for device in self.devices:
            f, h, native_f, _, replay = self.evaluate(device, [[0, 0, 1]], [[motion, 0, 0]],
                normal_force=normal_force, coefficient=mu, epsilon=epsilon, with_status=True)
            self.assertEqual(int(replay[0]), 1)
            self.assertNotEqual(float(f[0, 0]), 0.)
            np.testing.assert_allclose(f[0], [-scale * float(motion), 0, 0], rtol=3e-6, atol=0)
            np.testing.assert_allclose(h[0], scale * np.diag([1, 1, 0]), rtol=3e-6, atol=0)
            np.testing.assert_array_equal(native_f, 0)

    def test_nonzero_norm_mollifier_motion_underflow_requests_replay(self):
        # The source native function loses m*u before the large normal force
        # can rescale it. Even finite output zeros must be rejected.
        for device in self.devices:
            for motion in (1e-15, 1e-10):
                _, _, native_f, _, replay = self.evaluate(device, [[0, 0, 1]], [[motion, 0, 0]],
                    normal_force=.135, coefficient=1e31, epsilon=1e31, with_status=True)
                self.assertEqual(int(replay[0]), 1)
                if motion == 1e-15:
                    np.testing.assert_array_equal(native_f, 0)

    def test_native_strength_tangent_subnormal_requests_replay(self):
        for device in self.devices:
            _, _, _, _, replay = self.evaluate(device, [[1e-10, 0, 1]], [[1e-11, 0, 0]],
                normal_force=1e-35, coefficient=1., epsilon=1e-10, with_status=True)
            self.assertEqual(int(replay[0]), 1)

    def test_nonfinite_norm_and_subnormal_mollifier_request_replay(self):
        for device in self.devices:
            for motion, epsilon in ((1e30, .01), (1e-15, 3e38)):
                _, _, _, _, replay = self.evaluate(device, [[0, 0, 1]], [[motion, 0, 0]],
                    normal_force=.135, coefficient=1e31, epsilon=epsilon, with_status=True)
                self.assertEqual(int(replay[0]), 1)


if __name__ == '__main__':
    unittest.main()
