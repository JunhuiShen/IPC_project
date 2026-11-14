"""Check reference-routine reuse against the previous energy implementations."""

import unittest
import numpy as np
import warp as wp

from gpu import contact, dat_contact_fast, dat_contact_reference, dat_materials, energy
from gpu.dat_contacts import EnergyWorkspace
from newton._src.geometry.tri_mesh_collision import build_tri_mesh_collision_info


@wp.kernel(module="unique", enable_backward=False)
def bending(q: wp.array[wp.vec3], previous: wp.array[wp.vec3], edges: wp.array2d[int],
            rest: wp.array[float], lengths: wp.array[float], coefficients: wp.array[float],
            stiffness: wp.array[float], old: wp.array[energy.Pack],
            actual_f: wp.array[wp.vec3], actual_h: wp.array[wp.mat33]):
    role = wp.tid()
    if edges[0, 0] >= 0 and edges[0, 1] >= 0:
        own_role = role+2
        if role >= 2:
            own_role = role-2
        old[role] = energy.hinge(q[edges[0,2]], q[edges[0,3]], q[edges[0,0]], q[edges[0,1]],
                                 coefficients[0], rest[0], stiffness[0], own_role)
    f, h = dat_materials.evaluate_own_bending_force_hessian(
        0, role, q, previous, edges, rest, lengths, 99.0, 12.0, .01, coefficients, stiffness)
    actual_f[role] = f
    actual_h[role] = h


@wp.kernel(module="unique", enable_backward=False)
def contacts(q: wp.array[wp.vec3], previous: wp.array[wp.vec3], kind: int,
             p: contact.Parameters, old: wp.array[contact.Result],
             actual: wp.array[contact.Result], oracle: wp.array[contact.Result]):
    nodes = wp.vec4i(0, 1, 2, 3)
    old[0] = dat_contact_fast.evaluate_self(q, previous, nodes, kind, p)
    actual[0] = dat_contact_reference.evaluate_self(q, previous, nodes, kind, p)
    oracle[0] = contact.evaluate_self(q, previous, nodes, kind, p)


energy.configure_module(bending)
contact.configure_module(contacts)
dat_contact_fast.configure_module(contacts)
dat_contact_reference.configure_module(contacts)


class ReferenceReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def bend_case(self, points, *, length=1.1, coefficient=2.75, stiffness=1.75,
                  edge=(2, 3, 0, 1), compatibility=False):
        for device in self.devices:
            with self.subTest(device=str(device)):
                q = wp.array(np.asarray(points, np.float32), wp.vec3, device=device)
                previous = wp.array(np.asarray(points, np.float32)+.125, wp.vec3, device=device)
                scalar = lambda v: wp.array([v], float, device=device)
                old = wp.zeros(4, energy.Pack, device=device)
                f = wp.empty(4, wp.vec3, device=device)
                h = wp.empty(4, wp.mat33, device=device)
                wp.launch(bending, 4, [q, previous, wp.array([edge], int, device=device), scalar(-.17),
                    scalar(length), scalar(coefficient), scalar(stiffness), old, f, h], device=device)
                expected = old.numpy()
                np.testing.assert_allclose(f.numpy(), expected[:, :3], rtol=6e-4, atol=5e-5)
                np.testing.assert_allclose(h.numpy(), expected[:, 3:].reshape(4, 3, 3), rtol=8e-4, atol=5e-5)
                self.assertTrue(np.isfinite(f.numpy()).all() and np.isfinite(h.numpy()).all())
                if compatibility:
                    np.testing.assert_array_equal(f.numpy(), expected[:, :3])
                    np.testing.assert_array_equal(h.numpy(), expected[:, 3:].reshape(4, 3, 3))

    def test_bending_all_roles_keeps_own_energy_and_zero_damping(self):
        self.bend_case([[0,0,0], [1.1,.1,.1], [.2,.8,.25], [.3,-.9,-.2]])

    def test_bending_native_guard_band_keeps_existing_nonzero_response(self):
        self.bend_case([[0,0,0], [1,0,0], [.3,.005,.001], [.7,-.006,.002]], compatibility=True)
        self.bend_case([[0,0,0], [5e-7,0,0], [0,1,.2], [0,-1,.1]], compatibility=True)

    def test_bending_unrepresentable_stiffness_mapping_keeps_existing_response(self):
        points = [[0,0,0], [1.1,.1,.1], [.2,.8,.25], [.3,-.9,-.2]]
        self.bend_case(points, length=0., compatibility=True)
        self.bend_case(points, length=1e-40, compatibility=True)
        self.bend_case(points, length=1e5, coefficient=.5, stiffness=1e-40, compatibility=True)

    def test_boundary_and_collapsed_bending_stays_zero(self):
        points = [[0,0,0], [1,0,0], [0,1,0], [0,-1,0]]
        self.bend_case(points, edge=(-1,3,0,1))
        self.bend_case([[0,0,0], [0,0,0], [0,1,0], [0,-1,0]], compatibility=True)

    def test_native_friction_integration_matches_previous_packets_and_float64_law(self):
        for device in self.devices:
            for kind in (0, 1):
                q = np.array([[.24,.28,.0012],[0,0,0],[1,0,0],[0,1,0]] if kind == 0 else
                             [[0,0,0],[1,0,0],[.4,-.4,.0012],[.4,.6,.0012]], np.float32)
                for friction in (0., .2):
                    for slip in (0., 1e-7, 1e-5, 1e-4, .001):
                        with self.subTest(device=str(device), kind=kind, friction=friction, slip=slip):
                            previous = q.copy()
                            previous[0, 0] -= slip
                            p = contact.parameters(d_hat=.002, k_barrier=100., dt=.01,
                                friction=friction, eps_v=.01, exact_geometry=True)
                            output = [wp.zeros(1, contact.Result, device=device) for _ in range(3)]
                            wp.launch(contacts, 1, [wp.array(q, wp.vec3, device=device),
                                wp.array(previous, wp.vec3, device=device), kind, p, *output], device=device)
                            old, actual, oracle = [x.numpy()[0] for x in output]
                            self.assertEqual(int(old[48]), 0)
                            self.assertEqual(int(actual[48]), 0)
                            self.assertEqual(int(actual[49]), int(old[49]))
                            if friction == 0:
                                np.testing.assert_array_equal(actual, old)
                            else:
                                np.testing.assert_allclose(actual[:48], old[:48], rtol=4e-4, atol=4e-5)
                            np.testing.assert_allclose(actual[:48], oracle[:48], rtol=2e-3, atol=2e-4)

    def test_subnormal_native_friction_intermediate_requests_precision_replay(self):
        q = np.array([[.24,.28,.0012],[0,0,0],[1,0,0],[0,1,0]], np.float32)
        previous = q.copy(); previous[0, 0] -= 1e-5
        for device in self.devices:
            p = contact.parameters(d_hat=.002, k_barrier=.001, dt=.01,
                                   friction=1e-34, eps_v=.01, exact_geometry=True)
            output = [wp.zeros(1, contact.Result, device=device) for _ in range(3)]
            wp.launch(contacts, 1, [wp.array(q, wp.vec3, device=device),
                wp.array(previous, wp.vec3, device=device), 0, p, *output], device=device)
            old, actual, oracle = [x.numpy()[0] for x in output]
            self.assertEqual(int(old[48]), 0)
            self.assertEqual(int(oracle[48]), 0)
            self.assertEqual(int(actual[48]), 6)
            self.assertEqual(float(actual[49]), 0.)
            np.testing.assert_array_equal(actual[:48], np.zeros(48, np.float32))

    def test_large_smoothing_and_tiny_slip_do_not_silently_drop_friction(self):
        q = np.array([[0,0,.0012],[-1,-1,0],[1,-1,0],[0,1,0]], np.float32)
        for device in self.devices:
            for slip, epsilon in ((1e-25, 1e32), (1e-15, 1e33)):
                with self.subTest(device=str(device), slip=slip):
                    previous = q.copy(); previous[0, 0] = -slip
                    p = contact.parameters(d_hat=.002, k_barrier=100., dt=.01,
                        friction=1e31, eps_v=epsilon, exact_geometry=True)
                    output = [wp.zeros(1, contact.Result, device=device) for _ in range(3)]
                    wp.launch(contacts, 1, [wp.array(q, wp.vec3, device=device),
                        wp.array(previous, wp.vec3, device=device), 0, p, *output], device=device)
                    old, actual, oracle = [x.numpy()[0] for x in output]
                    self.assertEqual(int(old[48]), 0)
                    self.assertEqual(int(oracle[48]), 0)
                    self.assertNotEqual(float(old[0]), 0.)
                    self.assertNotEqual(float(oracle[0]), 0.)
                    self.assertEqual(int(actual[48]), 6)
                    self.assertEqual(float(actual[49]), 0.)
                    np.testing.assert_array_equal(actual[:48], np.zeros(48, np.float32))

    def test_precision_requests_replay_in_actual_rows_and_captured_dispatch(self):
        q = np.array([[0,0,.0012],[-1,-1,0],[1,-1,0],[0,1,0]], np.float32)
        old = q.copy(); old[0, 0] = -1e-15
        for device in self.devices:
            positions = wp.array(q, wp.vec3, device=device)
            previous = wp.array(old, wp.vec3, device=device)
            triangles = wp.array([[1,2,3]], int, device=device)
            edges = wp.empty((0,4), int, device=device)
            colors = wp.zeros(4, int, device=device)
            info = build_tri_mesh_collision_info(4, 1, 0,
                vertex_collision_buffer_pre_alloc=1, edge_collision_buffer_pre_alloc=1, device=device)
            info.vertex_colliding_triangles.fill_(-1)
            pairs = info.vertex_colliding_triangles.numpy(); pairs[:2] = [0,0]
            info.vertex_colliding_triangles.assign(pairs)
            info.vertex_colliding_triangles_count.assign([1,0,0,0])
            workspace = EnergyWorkspace(info, device=device, precision="float32", exact_geometry=False)
            force = wp.zeros(4, wp.vec3, device=device)
            hessian = wp.zeros(4, wp.mat33, device=device)

            def assemble():
                force.zero_(); hessian.zero_()
                context = workspace.begin(.01, .002, 1e31, 1e33)
                wp.launch(workspace.kernel, 16,
                    [.01, 0, previous, positions, colors, triangles, edges, info,
                     .002, 1000., 99., 1e31, 1e33, True, 1e-5, context, force, hessian],
                    device=device, max_blocks=device.sm_count)
                workspace.resolve(positions, previous, colors, 0, force, hessian)

            assemble(); workspace.check()
            output = [wp.zeros(1, contact.Result, device=device) for _ in range(3)]
            wp.launch(contacts, 1, [positions, previous, 0, workspace.context.parameters, *output], device=device)
            expected = output[2].numpy()[0][:48].reshape(4,12)
            self.assertNotEqual(float(expected[0,0]), 0.)
            actual = (force.numpy(), hessian.numpy())
            np.testing.assert_allclose(actual[0], expected[:,:3], rtol=2e-6, atol=0.)
            np.testing.assert_allclose(actual[1], expected[:,3:].reshape(4,3,3), rtol=2e-6, atol=0.)
            self.assertEqual(int(workspace.context.pending_count.numpy()[0]), 1)
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    assemble()
                wp.capture_launch(capture.graph); workspace.check()
                np.testing.assert_array_equal(force.numpy(), actual[0])
                np.testing.assert_array_equal(hessian.numpy(), actual[1])


if __name__ == "__main__":
    unittest.main()
