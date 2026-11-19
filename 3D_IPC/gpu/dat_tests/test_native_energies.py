"""Native DAT energy laws, solver integration and CPU/CUDA numerical guards."""

import ast
import inspect
import unittest
from unittest import mock

import numpy as np
import warp as wp
from newton._src.geometry.tri_mesh_collision import build_tri_mesh_collision_info
from newton._src.solvers.vbd import particle_vbd_kernels as stock

from gpu import dat_energies as energies, energy
from gpu.dat_energies import EnergyWorkspace


# Contact geometry, scalar barrier and native assembly.


@wp.struct
class NativeGeometry:
    distance: float
    normal: wp.vec3
    weights: wp.vec4
    friction_force: wp.vec3
    friction_hessian: wp.mat33


@wp.kernel(module="unique", enable_backward=False)
def reference_geometry(
    q: wp.array[wp.vec3], previous: wp.array[wp.vec3], nodes: wp.vec4i,
    kind: int, parallel_epsilon: float, radius: float, stiffness: float,
    friction: float, eps_v: float, dt: float, output: wp.array[NativeGeometry],
):
    reference = NativeGeometry()
    difference = wp.vec3(0.0)
    motion = wp.vec3(0.0)
    if kind == 0:
        p = q[nodes[0]]
        a = q[nodes[1]]
        b = q[nodes[2]]
        c = q[nodes[3]]
        closest, bary, feature = stock.triangle_closest_point(a, b, c, p)
        difference = p - closest
        reference.distance = wp.length(difference)
        reference.weights = wp.vec4(1.0, -bary[0], -bary[1], -bary[2])
        dx_v = p - previous[nodes[0]]
        closest_previous = (
            bary[0] * previous[nodes[1]]
            + bary[1] * previous[nodes[2]]
            + bary[2] * previous[nodes[3]]
        )
        motion = dx_v - (closest - closest_previous)
    else:
        a = q[nodes[0]]
        b = q[nodes[1]]
        c = q[nodes[2]]
        d = q[nodes[3]]
        st = wp.closest_point_edge_edge(a, b, c, d, parallel_epsilon)
        s = st[0]
        t = st[1]
        e1_vec = b - a
        e2_vec = d - c
        c1 = a + e1_vec * s
        c2 = c + e2_vec * t
        reference.distance = st[2]
        difference = c1 - c2
        reference.weights = wp.vec4(1.0 - s, s, -1.0 + t, -t)
        c1_previous = previous[nodes[0]] + (previous[nodes[1]] - previous[nodes[0]]) * s
        c2_previous = previous[nodes[2]] + (previous[nodes[3]] - previous[nodes[2]]) * t
        motion = (c1 - c1_previous) - (c2 - c2_previous)
    if reference.distance > 0.0 and reference.distance < radius:
        reference.normal = difference / reference.distance
        # Independently differentiate -k*(d-r)^2*log(d/r) in binary64.
        distance = wp.float64(reference.distance)
        gap = distance - wp.float64(radius)
        derivative = (-wp.float64(2.0) * gap * wp.log(distance / wp.float64(radius))
                      - gap * gap / distance)
        normal_force = float(-wp.float64(stiffness) * derivative)
        first, second = stock.orthonormal_basis(reference.normal)
        tangent = stock.mat32(first[0], second[0], first[1], second[1], first[2], second[2])
        slip = wp.transpose(tangent) * motion
        f, h = stock.compute_friction(friction, normal_force, tangent, slip, eps_v * dt)
        reference.friction_force = f
        reference.friction_hessian = h
    output[0] = reference


_native_options = wp.get_module_options(stock.accumulate_self_contact_force_and_hessian.module)


wp.set_module_options({key: _native_options[key] for key in ("fast_math", "fuse_fp") if key in _native_options},
                      module=reference_geometry.module)


class NativeContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_guarded_clones_preserve_native_geometry_ast_and_friction_functions(self):
        geometry_names = {"a", "b", "c", "p", "closest_p", "bary", "_feature_type",
                          "e1_v1", "e1_v2", "e2_v1", "e2_v2", "e1_v1_pos", "e1_v2_pos",
                          "e2_v1_pos", "e2_v2_pos", "st", "s", "t", "e1_vec", "e2_vec",
                          "c1", "c2", "diff", "dis", "collision_normal", "bs", "dx_v",
                          "closest_p_prev", "c1_prev", "c2_prev", "dx", "e0", "e1",
                          "axis_1", "axis_2", "T", "u", "eps_U"}

        def geometry_ast(source):
            statements = []
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Assign):
                    targets = {item.id for target in node.targets for item in ast.walk(target)
                               if isinstance(item, ast.Name)}
                    if targets & geometry_names:
                        statements.append(ast.dump(node, include_attributes=False))
            return statements

        native = {"ee": stock.evaluate_edge_edge_contact_2_vertices,
                  "vt": stock.evaluate_vertex_triangle_collision_force_hessian_4_vertices}
        for precision in ("float32", "float64"):
            evaluators = energies.make_evaluators(precision)
            for kind, evaluator in zip(("ee", "vt"), evaluators):
                with self.subTest(kind=kind, precision=precision):
                    source, proof = energies.native_contact_source(kind, precision)
                    original = inspect.getsource(native[kind].func)
                    restored = source
                    for before, after in reversed(proof["substitutions"]):
                        self.assertEqual(restored.count(after), 1)
                        restored = restored.replace(after, before, 1)
                    self.assertEqual(restored, original)
                    self.assertTrue(proof["restored_byte_identical"])
                    self.assertTrue(proof["geometry_unchanged"])
                    self.assertEqual(geometry_ast(source), geometry_ast(original))
                    self.assertIs(evaluator.func.__globals__["compute_friction"], stock.compute_friction)
                    if kind == "vt":
                        self.assertIs(evaluator.func.__globals__["triangle_closest_point"], stock.triangle_closest_point)
        source, proof = energies.cloned_source()
        self.assertTrue(proof["restored_byte_identical"])

    def fixture(self, device, kind, points, *, precision="float32", colors=(0, 0, 0, 0), slip=0.):
        points = np.asarray(points, np.float32)
        previous = points.copy()
        previous[0] -= np.asarray([slip, .3 * slip, 0.], np.float32)
        q = wp.array(points, wp.vec3, device=device)
        old = wp.array(previous, wp.vec3, device=device)
        triangles = (wp.array([[1, 2, 3]], int, device=device) if kind == 0 else
                     wp.empty((0, 3), int, device=device))
        edges = (wp.empty((0, 4), int, device=device) if kind == 0 else
                 wp.array([[-1, -1, 0, 1], [-1, -1, 2, 3]], int, device=device))
        info = build_tri_mesh_collision_info(4, 1 if kind == 0 else 0, 0 if kind == 0 else 2,
            vertex_collision_buffer_pre_alloc=1, edge_collision_buffer_pre_alloc=1, device=device)
        info.vertex_colliding_triangles.fill_(-1)
        info.vertex_colliding_triangles_count.zero_()
        info.edge_colliding_edges.fill_(-1)
        info.edge_colliding_edges_count.zero_()
        if kind == 0:
            pairs = info.vertex_colliding_triangles.numpy()
            pairs[:2] = [0, 0]
            info.vertex_colliding_triangles.assign(pairs)
            info.vertex_colliding_triangles_count.assign([1, 0, 0, 0])
        else:
            info.edge_colliding_edges.assign([0, 1, 1, 0])
            info.edge_colliding_edges_count.assign([1, 1])
        workspace = energies.EnergyWorkspace(info, device=device, precision=precision)
        return {"q": q, "previous": old, "triangles": triangles, "edges": edges, "info": info,
                "colors": wp.array(colors, int, device=device), "workspace": workspace,
                "force": wp.zeros(4, wp.vec3, device=device), "hessian": wp.zeros(4, wp.mat33, device=device)}

    def assemble(self, data, *, epsilon=1e-5, friction=0., eps_v=.01, radius=.002, ke=1000., kd=99.):
        workspace = data["workspace"]
        context = workspace.begin(.01, radius, friction, eps_v)
        data["force"].zero_()
        data["hessian"].zero_()
        wp.launch(workspace.kernel, 16,
            [.01, 0, data["previous"], data["q"], data["colors"], data["triangles"], data["edges"],
             data["info"], radius, ke, kd, friction, eps_v, True, epsilon, context,
             data["force"], data["hessian"]], device=data["q"].device,
            max_blocks=data["q"].device.sm_count)
        workspace.check()
        return data["force"].numpy(), data["hessian"].numpy()

    def expected(self, data, kind, *, epsilon=1e-5, friction=0., eps_v=.01, radius=.002):
        radius = float(np.float32(radius))
        forces = np.zeros((4, 3), np.float64)
        hessians = np.zeros((4, 3, 3), np.float64)
        nodes_per_row = [(0, 1, 2, 3)] if kind == 0 else [(0, 1, 2, 3), (2, 3, 0, 1)]
        for nodes in nodes_per_row:
            output = wp.empty(1, NativeGeometry, device=data["q"].device)
            wp.launch(reference_geometry, 1,
                [data["q"], data["previous"], wp.vec4i(*nodes), kind, epsilon, radius, 100.,
                 friction, eps_v, .01, output], device=data["q"].device)
            native = output.numpy()[0]
            d = float(native["distance"])
            if not 0. < d < radius:
                continue
            gap = d - radius
            first = -2. * gap * np.log(d / radius) - gap * gap / d
            second = -2. * np.log(d / radius) - 4. * gap / d + gap * gap / (d * d)
            normal = native["normal"].astype(np.float64)
            normal_force = -100. * first * normal
            normal_hessian = 100. * second * np.outer(normal, normal)
            for role in (range(4) if kind == 0 else range(2)):
                weight = float(native["weights"][role])
                forces[nodes[role]] += weight * (normal_force + native["friction_force"].astype(np.float64))
                hessians[nodes[role]] += weight * weight * (
                    normal_hessian + native["friction_hessian"].astype(np.float64))
        colors = data["colors"].numpy()
        forces[colors != 0] = 0.
        hessians[colors != 0] = 0.
        return forces, hessians

    def test_native_distance_ipc_force_and_rank_one_hessian_in_production_rows(self):
        fixtures = [
            (0, [[.24, .28, .0012], [0, 0, 0], [1, 0, 0], [0, 1, 0]], 1e-5),
            (0, [[.4, -.0006, .0008], [0, 0, 0], [1, 0, 0], [0, 1, 0]], 1e-5),
            (0, [[-.0006, -.0006, .0006], [0, 0, 0], [1, 0, 0], [0, 1, 0]], 1e-5),
            (1, [[0, 0, 0], [1, 0, 0], [.4, -.4, .001], [.4, .6, .001]], 1e-5),
            (1, [[0, 0, 0], [1, 0, 0], [.1, .001, 0], [.9, .001, 0]], 1e-5),
            (1, [[0, 0, 0], [1, 0, 0], [.2, -.0005, .001], [.8, .0005, .001]], 1e-9),
            (1, [[0, 0, 0], [1, 0, 0], [.2, -.0005, .001], [.8, .0005, .001]], .01),
        ]
        for device in self.devices:
            for precision in ("float32", "float64"):
                for kind, points, epsilon in fixtures:
                    for colors in ((0, 0, 0, 0), (0, 1, 0, 1)):
                        with self.subTest(device=str(device), precision=precision, kind=kind,
                                          epsilon=epsilon, colors=colors):
                            data = self.fixture(device, kind, points, precision=precision, colors=colors)
                            expected = self.expected(data, kind, epsilon=epsilon)
                            actual = self.assemble(data, epsilon=epsilon)
                            for value, oracle in zip(actual, expected):
                                np.testing.assert_allclose(value, oracle, rtol=5e-5, atol=2e-7)

    def test_native_friction_and_zero_slip_hessian_are_retained(self):
        for device in self.devices:
            for precision in ("float32", "float64"):
                for kind in (0, 1):
                    points = ([[.25, .25, .0012], [0, 0, 0], [1, 0, 0], [0, 1, 0]] if kind == 0 else
                              [[0, 0, 0], [1, 0, 0], [.4, -.4, .0012], [.4, .6, .0012]])
                    for slip in (0., 1e-5, .001):
                        with self.subTest(device=str(device), precision=precision, kind=kind, slip=slip):
                            data = self.fixture(device, kind, points, precision=precision, slip=slip)
                            actual = self.assemble(data, friction=.2)
                            expected = self.expected(data, kind, friction=.2)
                            for value, oracle in zip(actual, expected):
                                np.testing.assert_allclose(value, oracle, rtol=5e-5, atol=2e-7)
                            if slip == 0.:
                                without_friction = self.assemble(data, friction=0.)
                                for first, second in zip(actual, without_friction):
                                    np.testing.assert_array_equal(first, second)

    def test_own_barrier_coefficient_and_absent_normal_damping(self):
        points = [[.25, .25, .0012], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
        for device in self.devices:
            for precision in ("float32", "float64"):
                data = self.fixture(device, 0, points, precision=precision, slip=.0003)
                original = self.assemble(data, ke=1., kd=0.)
                changed_stock_parameters = self.assemble(data, ke=12345., kd=100.)
                for first, second in zip(original, changed_stock_parameters):
                    np.testing.assert_array_equal(first, second)

    def test_native_activation_boundary_and_zero_distance_skip(self):
        radius = np.float32(.002)
        for device in self.devices:
            for precision in ("float32", "float64"):
                for gap in (0., np.nextafter(radius, np.float32(-np.inf)), radius,
                            np.nextafter(radius, np.float32(np.inf))):
                    data = self.fixture(device, 0, [[.25, .25, gap], [0, 0, 0], [1, 0, 0], [0, 1, 0]],
                                        precision=precision)
                    with self.subTest(device=str(device), precision=precision, gap=gap):
                        actual = self.assemble(data, radius=radius)
                        if gap == 0. or gap >= radius:
                            for value in actual:
                                np.testing.assert_array_equal(value, np.zeros_like(value))
                        else:
                            self.assertGreater(float(actual[0][0, 2]), 0.)


# Project corotated membrane forces and analytic self-Hessians.


@wp.kernel(module="unique", enable_backward=False)
def membrane_probe(
    positions: wp.array[wp.vec3], triangles: wp.array2d[int],
    inverses: wp.array[wp.mat22], areas: wp.array[float], materials: wp.array2d[float],
    force: wp.array[wp.vec3], hessian: wp.array[wp.mat33],
    old_force: wp.array[wp.vec3], old_hessian: wp.array[wp.mat33],
    native_hessian: wp.array[wp.mat33],
):
    i = wp.tid()
    face = i // 3
    role = i % 3
    f, h = energies.evaluate_own_membrane_force_hessian(
        face, role, positions, positions, triangles, inverses[face], areas[face],
        999.0, 999.0, 123.0, .1, materials)
    force[i] = f
    hessian[i] = h
    packet = energy.triangle_reference(
        positions[triangles[face, 0]], positions[triangles[face, 1]],
        positions[triangles[face, 2]], inverses[face], areas[face],
        materials[face, 0], materials[face, 1], role)
    f_old, h_old = energy.unpack(packet)
    old_force[i] = f_old
    old_hessian[i] = h_old
    native_f, native_h = stock.evaluate_neo_hookean_membrane_force_hessian(
        face, role, positions, positions, triangles, inverses[face], areas[face],
        materials[face, 0], materials[face, 1], 0.0, .1)
    native_hessian[i] = native_h


energy.configure_module(membrane_probe)


def scalar_energy(x, inverse, area, mu, lam):
    """Independent binary64 SVD spelling of the original corotated potential."""
    f = np.column_stack((x[1] - x[0], x[2] - x[0])) @ inverse
    u, singular, vt = np.linalg.svd(f, full_matrices=False)
    rotation = u @ vt
    return area * (mu * np.sum((f - rotation)**2) + .5 * lam * (np.prod(singular) - 1)**2)


def finite_difference_derivatives(x, inverse, area, mu, lam):
    x = np.asarray(x, dtype=np.float64)
    step = 1.e-4
    value = scalar_energy(x, inverse, area, mu, lam)
    gradient = np.zeros_like(x)
    hessian = np.zeros((3, 3, 3), dtype=np.float64)
    for role in range(3):
        for axis in range(3):
            plus, minus = x.copy(), x.copy()
            plus[role, axis] += step
            minus[role, axis] -= step
            ep = scalar_energy(plus, inverse, area, mu, lam)
            em = scalar_energy(minus, inverse, area, mu, lam)
            gradient[role, axis] = (ep - em) / (2 * step)
            hessian[role, axis, axis] = (ep - 2 * value + em) / step**2
            for other in range(axis):
                mixed = 0.0
                for first in (-1, 1):
                    for second in (-1, 1):
                        shifted = x.copy()
                        shifted[role, axis] += first * step
                        shifted[role, other] += second * step
                        mixed += first * second * scalar_energy(shifted, inverse, area, mu, lam)
                hessian[role, axis, other] = mixed / (4 * step**2)
                hessian[role, other, axis] = hessian[role, axis, other]
    return -gradient, hessian


class NativeMembraneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def evaluate(self, device, positions, inverses, areas, materials):
        positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3, 3)
        count = len(positions)
        outputs = [wp.empty(3 * count, dtype, device=device)
                   for dtype in (wp.vec3, wp.mat33, wp.vec3, wp.mat33, wp.mat33)]
        wp.launch(membrane_probe, 3 * count,
            [wp.array(positions.reshape(-1, 3), wp.vec3, device=device),
             wp.array(np.arange(3 * count).reshape(count, 3), int, device=device),
             wp.array(np.asarray(inverses, np.float32), wp.mat22, device=device),
             wp.array(np.asarray(areas, np.float32), float, device=device),
             wp.array(np.asarray(materials, np.float32), float, device=device), *outputs],
            device=device)
        return [item.numpy().reshape((count, 3) + item.numpy().shape[1:]) for item in outputs]

    def test_legacy_clone_changes_only_the_two_constitutive_stresses(self):
        source, proof = energies.membrane_source()
        self.assertTrue(proof['restored_byte_identical'])
        self.assertTrue(proof['geometry_unchanged'])
        self.assertTrue(proof['hessian_unchanged'])
        self.assertEqual(len(proof['substitutions']), 3)  # Rename and two stresses.
        self.assertNotIn('energy.triangle', source)
        original = energies.inspect.getsource
        with mock.patch.object(energies.inspect, 'getsource',
                side_effect=lambda target: original(target).replace('hessian = I_coeff', 'hessian = 2.0 * I_coeff')):
            with self.assertRaisesRegex(RuntimeError, 'Pinned native DAT membrane'):
                energies.membrane_source()

    def test_regular_derivatives_match_project_binary64_reference(self):
        rng = np.random.default_rng(9281)
        positions, inverses = [], []
        for _ in range(64):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            frame = rotation[:, :2] @ np.diag(rng.uniform(.35, 2.5, size=2))
            origin = rng.uniform(-.5, .5, size=3)
            positions.append(np.stack((origin, origin + frame[:, 0], origin + frame[:, 1])))
            inverses.append([[1.1, -.2], [.05, .9]])
        for device in self.devices:
            with self.subTest(device=str(device)):
                f, h, reference_f, reference_h, native_h = self.evaluate(
                    device, positions, inverses, [.48] * 64, [[17.3, 23.1]] * 64)
                self.assertTrue(np.isfinite(f).all() and np.isfinite(h).all())
                np.testing.assert_allclose(f, reference_f, rtol=5.e-4, atol=3.e-4)
                np.testing.assert_allclose(h, reference_h, rtol=5.e-4, atol=5.e-4)
                np.testing.assert_allclose(h, h.transpose(0, 1, 3, 2), rtol=0, atol=5.e-5)
                # Compression can give negative curvature; retaining it is
                # intentional, matching the project's unprojected Hessian.
                self.assertLess(float(np.linalg.eigvalsh(h).min()), -.1)
                self.assertGreater(float(np.max(np.abs(h - native_h))), .5)

    def test_force_and_hessian_independently_differentiate_own_scalar_energy(self):
        positions = np.asarray([
            [[.1, -.2, .3], [1.2, .1, .25], [.2, 1.1, .55]],
            [[0, 0, 0], [.45, .1, -.15], [-.1, .6, .2]],
            [[0, 0, 0], [2.1, -.3, .4], [.3, 1.7, -.2]],
        ], dtype=np.float32)
        inverse = np.asarray([[1.1, -.2], [.05, .9]], dtype=np.float32)
        expected = [finite_difference_derivatives(x, inverse.astype(np.float64), .48, 17.3, 23.1)
                    for x in positions]
        expected_f = np.asarray([item[0] for item in expected])
        expected_h = np.asarray([item[1] for item in expected])
        for device in self.devices:
            with self.subTest(device=str(device)):
                f, h, _, _, native_h = self.evaluate(device, positions, [inverse] * 3, [.48] * 3, [[17.3, 23.1]] * 3)
                np.testing.assert_allclose(f, expected_f, rtol=5.e-4, atol=3.e-4)
                np.testing.assert_allclose(h, expected_h, rtol=5.e-4, atol=5.e-4)
                self.assertGreater(float(np.max(np.abs(h - native_h))), .5)

    def test_rank_loss_keeps_project_regularization_and_nonfinite_behavior(self):
        positions = [
            [[0, 0, 0], [1.e-7, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
        ]
        for device in self.devices:
            with self.subTest(device=str(device)):
                f, h, reference_f, reference_h, _ = self.evaluate(
                    device, positions, [np.eye(2)] * 3, [.5] * 3, [[3., 5.]] * 3)
                # The polar eigenvalue floor does not regularize C inverse.
                # Exactly rank-deficient triangles keep the project's NaNs;
                # substituting the native finite floor would change the model.
                np.testing.assert_array_equal(np.isfinite(f), np.isfinite(reference_f))
                np.testing.assert_array_equal(np.isfinite(h), np.isfinite(reference_h))
                np.testing.assert_allclose(f[0], reference_f[0], rtol=5.e-4, atol=3.e-4)
                np.testing.assert_allclose(h[0], reference_h[0], rtol=5.e-4, atol=5.e-4)
                self.assertFalse(np.isfinite(f[1:]).all())
                self.assertFalse(np.isfinite(h[1:]).all())

    def test_skinny_membrane_derivatives_remain_scale_invariant(self):
        # Keep the deformation gradient and dimensionless conditioning fixed
        # while changing world length units. The rest area changes quadratically.
        skinny = np.array([[0, 0, 0], [1.e-7, 0, 0], [0, 1, 0]], np.float32)
        scales = (1.e-2, 1., 1.e2)
        positions = [skinny * scale for scale in scales]
        inverses = [np.eye(2) / scale for scale in scales]
        areas = [.5 * scale**2 for scale in scales]
        for device in self.devices:
            with self.subTest(device=str(device)):
                f, h, reference_f, reference_h, _ = self.evaluate(
                    device, positions, inverses, areas, [[3., 5.]] * len(scales))
                self.assertTrue(np.isfinite(f).all() and np.isfinite(h).all())
                np.testing.assert_allclose(f, reference_f, rtol=5.e-4, atol=3.e-4)
                np.testing.assert_allclose(h, reference_h, rtol=5.e-4, atol=5.e-4)
                np.testing.assert_allclose(h, h.transpose(0, 1, 3, 2), rtol=5.e-6, atol=5.e-4)

    def test_zero_materials_have_zero_force_and_matrix_including_rank_loss(self):
        positions = [
            [[0, 0, 0], [1, .2, -.1], [-.1, .9, .2]],
            [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
        ]
        for device in self.devices:
            with self.subTest(device=str(device)):
                f, h, _, _, _ = self.evaluate(device, positions, [np.eye(2)] * 2, [.5] * 2, [[0., 0.]] * 2)
                np.testing.assert_array_equal(f, np.zeros((2, 3, 3), dtype=np.float32))
                np.testing.assert_array_equal(h, np.zeros((2, 3, 3, 3), dtype=np.float32))


# Bending energy and captured contact dispatch.


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
        old[role] = energy.hinge_reference(q[edges[0,2]], q[edges[0,3]], q[edges[0,0]], q[edges[0,1]],
                                 coefficients[0], rest[0], stiffness[0], own_role)
    f, h = energies.evaluate_own_bending_force_hessian(
        0, role, q, previous, edges, rest, lengths, 99.0, 12.0, .01, coefficients, stiffness)
    actual_f[role] = f
    actual_h[role] = h


@wp.kernel(module="unique", enable_backward=False)
def native_bending(q: wp.array[wp.vec3], previous: wp.array[wp.vec3], edges: wp.array2d[int],
                   rest: wp.array[float], lengths: wp.array[float], coefficients: wp.array[float],
                   stiffness: wp.array[float], force: wp.array[wp.vec3], hessian: wp.array[wp.mat33]):
    role = wp.tid()
    f = wp.vec3(0.0)
    h = wp.mat33(0.0)
    if (edges[0,0] >= 0 and edges[0,1] >= 0 and coefficients[0] > 0.0 and stiffness[0] > 0.0
        and wp.isfinite(lengths[0]) and lengths[0] > 0.0):
        mapped_ke = (2.0 * stiffness[0] * coefficients[0]) / lengths[0]
        if wp.isfinite(mapped_ke) and mapped_ke > 0.0:
            f, h = stock.evaluate_dihedral_angle_based_bending_force_hessian(
                0, role, q, previous, edges, rest, lengths, mapped_ke, 0.0, .01)
    force[role] = f
    hessian[role] = h


_native_triangle = energies.make_evaluators("float32")[1]


@wp.kernel(module="unique", enable_backward=False)
def native_triangle(q: wp.array[wp.vec3], previous: wp.array[wp.vec3], triangles: wp.array2d[int],
                    error: wp.array[int], force: wp.array[wp.vec3], hessian: wp.array[wp.mat33]):
    active, fa, fb, fc, fv, ha, hb, hc, hv = _native_triangle(
        0, 0, q, previous, triangles, .002, 100., 0., .2, .01, .01, True, error)
    if active:
        force[0] = fv; force[1] = fa; force[2] = fb; force[3] = fc
        hessian[0] = hv; hessian[1] = ha; hessian[2] = hb; hessian[3] = hc


energy.configure_module(bending)


energy.configure_module(native_bending)


energies.configure_module(native_triangle)


class NativeReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def bend_case(self, points, *, length=1.1, coefficient=2.75, stiffness=1.75,
                  edge=(2, 3, 0, 1), compare_native=False):
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
                expected_f, expected_h = expected[:, :3], expected[:, 3:].reshape(4, 3, 3)
                if compare_native:
                    # Regular geometry also matches the native angle energy
                    # after its stiffness conversion; degeneracy guards differ.
                    nf = wp.zeros(4, wp.vec3, device=device)
                    nh = wp.zeros(4, wp.mat33, device=device)
                    wp.launch(native_bending, 4,
                        [q, previous, wp.array([edge], int, device=device), scalar(-.17),
                         scalar(length), scalar(coefficient), scalar(stiffness), nf, nh], device=device)
                    np.testing.assert_allclose(expected_f, nf.numpy(), rtol=6e-4, atol=5e-5)
                    np.testing.assert_allclose(expected_h, nh.numpy(), rtol=8e-4, atol=5e-5)
                np.testing.assert_allclose(f.numpy(), expected_f, rtol=6e-4, atol=5e-5)
                np.testing.assert_allclose(h.numpy(), expected_h, rtol=8e-4, atol=5e-5)
                self.assertTrue(np.isfinite(f.numpy()).all() and np.isfinite(h.numpy()).all())

    def test_bending_all_roles_keeps_own_energy_and_zero_damping(self):
        self.bend_case([[0,0,0], [1.1,.1,.1], [.2,.8,.25], [.3,-.9,-.2]])

    def test_regular_bending_equals_actual_native_callback(self):
        self.bend_case([[0,0,0], [1.1,.1,.1], [.2,.8,.25], [.3,-.9,-.2]], compare_native=True)

    def test_thin_bending_keeps_project_geometry_instead_of_native_skip(self):
        self.bend_case([[0,0,0], [1,0,0], [.3,.005,.001], [.7,-.006,.002]])
        self.bend_case([[0,0,0], [5e-7,0,0], [0,1,.2], [0,-1,.1]])

    def test_short_hinge_conditioning_is_relative_to_geometry_scale(self):
        skinny = np.array([[0,0,0], [5e-7,0,0], [0,1,.2], [0,-1,.1]], np.float32)
        for scale in (1.e-2, 1., 1.e2):
            with self.subTest(world_scale=scale):
                self.bend_case(skinny * scale, length=5.e-7 * scale)

    def test_bending_uses_project_coefficient_without_native_rest_length_mapping(self):
        points = [[0,0,0], [1.1,.1,.1], [.2,.8,.25], [.3,-.9,-.2]]
        for length in (0., -1., float("inf"), float("nan"), 1e-40):
            self.bend_case(points, length=length)
        self.bend_case(points, length=1e5, coefficient=.5, stiffness=1e-6)

    def test_boundary_and_collapsed_bending_stays_zero(self):
        points = [[0,0,0], [1,0,0], [0,1,0], [0,-1,0]]
        self.bend_case(points, edge=(-1,3,0,1))
        self.bend_case([[0,0,0], [0,0,0], [0,1,0], [0,-1,0]])


    def test_native_contact_in_actual_rows_and_captured_dispatch(self):
        q = np.array([[.25,.25,.0012],[0,0,0],[1,0,0],[0,1,0]], np.float32)
        old = q.copy(); old[0, 0] -= 1e-5
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
            workspace = EnergyWorkspace(info, device=device, precision="float32")
            force = wp.zeros(4, wp.vec3, device=device)
            hessian = wp.zeros(4, wp.mat33, device=device)

            def assemble():
                force.zero_(); hessian.zero_()
                context = workspace.begin(.01, .002, .2, .01)
                wp.launch(workspace.kernel, 16,
                    [.01, 0, previous, positions, colors, triangles, edges, info,
                     .002, 1000., 99., .2, .01, True, 1e-5, context, force, hessian],
                    device=device, max_blocks=device.sm_count)

            assemble(); workspace.check()
            expected_force = wp.zeros(4, wp.vec3, device=device)
            expected_hessian = wp.zeros(4, wp.mat33, device=device)
            error = wp.zeros(1, int, device=device)
            wp.launch(native_triangle, 1,
                [positions, previous, triangles, error, expected_force, expected_hessian], device=device)
            self.assertEqual(int(error.numpy()[0]), 0)
            self.assertNotEqual(float(expected_force.numpy()[0,0]), 0.)
            actual = (force.numpy(), hessian.numpy())
            np.testing.assert_allclose(actual[0], expected_force.numpy(), rtol=2e-6, atol=0.)
            np.testing.assert_allclose(actual[1], expected_hessian.numpy(), rtol=2e-6, atol=0.)
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    assemble()
                wp.capture_launch(capture.graph); workspace.check()
                np.testing.assert_array_equal(force.numpy(), actual[0])
                np.testing.assert_array_equal(hessian.numpy(), actual[1])


# Finite output guards.


@wp.kernel(module="unique", enable_backward=False)
def finite_probe(
    force: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
    result: wp.array[int],
):
    index = wp.tid()
    result[index] = int(energies._finite_block(force[index], hessian[index]))


wp.set_module_options({"enable_backward": False, "fast_math": True, "fuse_fp": True},
                      module=finite_probe.module)


def fields_to_inputs(fields, device):
    force = wp.array(np.ascontiguousarray(fields[:, :3]), wp.vec3, device=device)
    hessian = wp.array(np.ascontiguousarray(fields[:, 3:].reshape(-1, 3, 3)),
                       wp.mat33, device=device)
    result = wp.empty(len(fields), int, device=device)
    wp.launch(finite_probe, len(fields), [force, hessian, result], device=device)
    return result.numpy()


class FiniteBlockAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def assert_classification(self, fields):
        fields = np.asarray(fields, dtype=np.float32)
        expected = np.isfinite(fields).all(axis=1).astype(np.int32)
        for device in self.devices:
            with self.subTest(device=str(device)):
                np.testing.assert_array_equal(fields_to_inputs(fields, device), expected)

    def test_each_force_and_hessian_component_rejects_nan_and_both_infinities(self):
        rows = []
        for component in range(12):
            for invalid in (np.nan, np.inf, -np.inf):
                row = np.arange(12, dtype=np.float32) - np.float32(5.5)
                row[component] = invalid
                rows.append(row)
        self.assertEqual(len(rows), 36)
        self.assert_classification(rows)

    def test_normal_subnormal_signed_zero_and_extreme_finite_values_are_accepted(self):
        info = np.finfo(np.float32)
        subnormal = np.nextafter(np.float32(0), np.float32(1))
        values = (np.float32(0), np.float32(-0.0), np.float32(1), np.float32(-2.5),
                  info.tiny, -info.tiny, subnormal, -subnormal, info.max, -info.max)
        rows = []
        for component in range(12):
            for finite in values:
                row = np.arange(12, dtype=np.float32) - np.float32(5.5)
                row[component] = finite
                rows.append(row)
        self.assertTrue(np.isfinite(np.asarray(rows, np.float32)).all())
        self.assert_classification(rows)


if __name__ == "__main__":
    unittest.main()
