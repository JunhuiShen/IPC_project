"""Accuracy checks for compact DAT contact against the unchanged binary64 law.

The oracle is the existing shared contact evaluator, not a second invocation
of the optimized evaluator. Barrier-only finite differences additionally check
the energy formula without either device implementation.
"""
from pathlib import Path
import unittest

import numpy as np
import warp as wp

import gpu

_isolated_gpu = Path(__file__).resolve().parents[1]
if str(_isolated_gpu) not in gpu.__path__:
    gpu.__path__.append(str(_isolated_gpu))

from gpu import contact, dat_contact_fast as fast, dat_contacts
from gpu.contact_exact import evaluate_self_exact
from newton._src.geometry.tri_mesh_collision import build_tri_mesh_collision_info


@wp.kernel(module="unique", enable_backward=False)
def _packets(q: wp.array[wp.vec3], previous: wp.array[wp.vec3], kind: int,
             parameters: contact.Parameters, ordinary: wp.array[contact.Result],
             optimized: wp.array[contact.Result]):
    nodes = wp.vec4i(0, 1, 2, 3)
    ordinary[0] = contact.evaluate_self(q, previous, nodes, kind, parameters)
    optimized[0] = fast.evaluate_self(q, previous, nodes, kind, parameters)


@wp.kernel(module="unique", enable_backward=False)
def _exact_packet(q: wp.array[wp.vec3], previous: wp.array[wp.vec3], kind: int,
                  context: dat_contacts.EnergyContext, output: wp.array[contact.Result]):
    output[0] = evaluate_self_exact(q, previous, wp.vec4i(0, 1, 2, 3), kind,
                                    context.parameters, context.words, context.slots, 0)


contact.configure_module(_packets)
fast.configure_module(_packets)
contact.configure_module(_exact_packet, include_filter=False)


class DatContactPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def fixture(self, device, kind, points, *, previous=None, friction=0.0,
                exact=True, colors=(0, 0, 0, 0), radius=.002, stiffness=100.0):
        points = np.asarray(points, np.float32)
        if previous is None:
            previous = points.copy()
        q = wp.array(points, wp.vec3, device=device)
        previous = wp.array(np.asarray(previous, np.float32), wp.vec3, device=device)
        triangles = (wp.array([[1, 2, 3]], int, device=device) if kind == 0
                     else wp.empty((0, 3), int, device=device))
        edges = (wp.empty((0, 4), int, device=device) if kind == 0 else
                 wp.array([[-1, -1, 0, 1], [-1, -1, 2, 3]], int, device=device))
        info = build_tri_mesh_collision_info(
            4, 1 if kind == 0 else 0, 0 if kind == 0 else 2,
            vertex_collision_buffer_pre_alloc=1, edge_collision_buffer_pre_alloc=1,
            device=device)
        info.vertex_colliding_triangles.fill_(-1)
        info.vertex_colliding_triangles_count.zero_()
        info.edge_colliding_edges.fill_(-1)
        info.edge_colliding_edges_count.zero_()
        if kind == 0:
            rows = np.full(info.vertex_colliding_triangles.shape[0], -1, np.int32)
            rows[:2] = [0, 0]
            info.vertex_colliding_triangles.assign(rows)
            info.vertex_colliding_triangles_count.assign([1, 0, 0, 0])
        else:
            # The stock EE stream is directed. Each direction owns two roles.
            info.edge_colliding_edges.assign([0, 1, 1, 0])
            info.edge_colliding_edges_count.assign([1, 1])
        groups = wp.array(colors, int, device=device)
        workspace = dat_contacts.EnergyWorkspace(
            info, device=device, k_barrier=stiffness,
            exact_geometry=exact, precision="float32")
        workspace.begin(.01, radius, friction, .01)
        return q, previous, triangles, edges, info, groups, workspace

    def packets(self, data, kind):
        q, previous, *_, workspace = data
        ordinary = wp.empty(1, contact.Result, device=q.device)
        optimized = wp.empty(1, contact.Result, device=q.device)
        wp.launch(_packets, 1, [q, previous, kind, workspace.context.parameters,
                               ordinary, optimized], device=q.device)
        oracle, candidate = ordinary.numpy()[0], optimized.numpy()[0]
        if int(oracle[48]) == 2 and workspace.exact_geometry:
            wp.launch(_exact_packet, 1, [q, previous, kind, workspace.context, ordinary],
                      device=q.device, block_dim=1)
            oracle = ordinary.numpy()[0]
        return oracle, candidate

    def scatter(self, data, color=0, *, check=True):
        q, previous, triangles, edges, info, groups, workspace = data
        forces = wp.zeros(q.shape[0], wp.vec3, device=q.device)
        hessians = wp.zeros(q.shape[0], wp.mat33, device=q.device)
        p = workspace.context.parameters
        workspace.begin(float(p.dt), float(p.d_hat), float(p.friction), float(p.eps_v))
        dim = max(4 * info.vertex_colliding_triangles_count.shape[0],
                  4 * info.edge_colliding_edges_count.shape[0])
        wp.launch(workspace.kernel, dim,
                  [float(p.dt), color, previous, q, groups, triangles, edges, info,
                   float(p.d_hat), 1000.0, .1, float(p.friction), float(p.eps_v),
                   True, 1e-5, workspace.context, forces, hessians],
                  device=q.device, max_blocks=q.device.sm_count)
        workspace.resolve(q, previous, groups, color, forces, hessians)
        if check:
            workspace.check()
        return forces.numpy(), hessians.numpy()

    def assert_relative_norm(self, actual, expected, *, rtol=5e-4, atol=1e-10):
        actual = np.asarray(actual, np.float64)
        expected = np.asarray(expected, np.float64)
        self.assertTrue(np.isfinite(actual).all())
        error = np.linalg.norm(actual - expected)
        limit = atol + rtol * np.linalg.norm(expected)
        self.assertLessEqual(error, limit,
                             f"relative error {error / max(np.linalg.norm(expected), atol):.6g}")

    def assert_scatter_matches(self, data, packet, color=0):
        self.assertEqual(int(packet[48]), 0)
        expected = packet[:48].reshape(4, 12).copy()
        expected[data[5].numpy() != color] = 0
        forces, hessians = self.scatter(data, color)
        self.assert_relative_norm(forces, expected[:, :3])
        self.assert_relative_norm(hessians, expected[:, 3:].reshape(4, 3, 3))

    def test_feature_families_friction_regimes_and_directed_color_ownership(self):
        fixtures = [
            (0, [[.2, .2, .001], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            (0, [[.4, -.0006, .0008], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            (0, [[-.0006, -.0006, .0006], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            (1, [[0, 0, 0], [1, 0, 0], [.5, -.5, .001], [.5, .5, .001]]),
            (1, [[0, 0, 0], [1, 0, 0], [-.0008, -.5, .0006], [-.0008, .5, .0006]]),
            (1, [[0, 0, 0], [1, 0, 0], [-.0006, .0008, 0], [-.0006, 1, 0]]),
        ]
        for device in self.devices:
            for kind, points in fixtures:
                for friction, slip in ((0.0, 0.0), (.2, 2e-5), (.2, .003)):
                    previous = np.asarray(points, np.float32)
                    previous[0] -= np.asarray([slip, .3 * slip, 0], np.float32)
                    with self.subTest(device=str(device), kind=kind, points=points,
                                      friction=friction, slip=slip):
                        data = self.fixture(device, kind, points, previous=previous,
                                            friction=friction, colors=(0, 1, 0, 1))
                        oracle, _ = self.packets(data, kind)
                        for color in (0, 1):
                            self.assert_scatter_matches(data, oracle, color)

    def test_well_conditioned_random_contacts_use_fast_path_and_match_binary64(self):
        rng = np.random.default_rng(28719)
        fixtures = []
        for _ in range(12):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            gap = rng.uniform(.025, .075)
            pt = np.array([[rng.uniform(.15, .3), rng.uniform(.15, .3), gap],
                           [0, 0, 0], [1, 0, 0], [0, 1, 0]])
            ee = np.array([[-.5, 0, 0], [.5, 0, 0],
                           [0, -.5, gap], [0, .5, gap]])
            fixtures.extend([(0, pt @ rotation), (1, ee @ rotation)])
        for device in self.devices:
            accepted = 0
            for kind, points in fixtures:
                previous = np.asarray(points, np.float32).copy()
                previous[0] -= np.asarray([.0002, -.0001, .00005], np.float32)
                data = self.fixture(device, kind, points, previous=previous,
                                    friction=.3, radius=.1)
                oracle, candidate = self.packets(data, kind)
                self.assertEqual(int(oracle[48]), 0)
                self.assertEqual(int(oracle[49]), 1)
                if int(candidate[48]) == 0:
                    accepted += 1
                    self.assertEqual(int(candidate[49]), 1)
                    self.assert_relative_norm(candidate[:48].reshape(4, 12)[:, :3],
                                              oracle[:48].reshape(4, 12)[:, :3])
                    self.assert_relative_norm(candidate[:48].reshape(4, 12)[:, 3:],
                                              oracle[:48].reshape(4, 12)[:, 3:])
                else:
                    self.assertEqual(int(candidate[48]), 6)
                self.assert_scatter_matches(data, oracle)
            # A numerically safe implementation must still exercise the new
            # evaluator; unconditional binary64 replay is not this feature.
            self.assertGreaterEqual(accepted, int(.8 * len(fixtures)))

    def test_cutoff_neighbors_preserve_active_classification_after_replay(self):
        radius = np.float32(.002)
        gaps = [np.nextafter(radius, np.float32(-np.inf)), radius,
                np.nextafter(radius, np.float32(np.inf))]
        for device in self.devices:
            for gap in gaps:
                data = self.fixture(device, 0,
                                    [[.25, .25, gap], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
                oracle, candidate = self.packets(data, 0)
                self.assertIn(int(candidate[48]), (0, 6))
                if int(candidate[48]) == 0:
                    self.assertEqual(int(candidate[49]), int(oracle[49]))
                self.assertEqual(int(oracle[49]), int(gap < radius))
                self.assert_scatter_matches(data, oracle)

    def test_ill_conditioned_and_translated_geometry_preserves_robust_replay(self):
        fixtures = [
            (0, [[4096.25, 4096.25, .001], [4096, 4096, 0],
                 [4097, 4096, 0], [4096, 4097, 0]]),
            (0, [[.2, .00001, .001], [0, 0, 0], [1, 0, 0], [1, .0001, 0]]),
            (0, [[.2, .0002, .001], [-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
            (1, [[-1, 0, 0], [1, 0, 0], [-.5, .001, .0002], [.5, .001, .0002]]),
            (1, [[0, 0, 0], [0, 0, 0], [0, .001, 0], [1, .001, 0]]),
        ]
        for device in self.devices:
            for kind, points in fixtures:
                with self.subTest(device=str(device), kind=kind, points=points):
                    data = self.fixture(device, kind, points, friction=.2)
                    oracle, candidate = self.packets(data, kind)
                    self.assertIn(int(candidate[48]), (0, 6))
                    # A translation can be harmless (Sterbenz subtraction).
                    # The other cases deliberately require precision replay.
                    if points[0][0] < 4096:
                        self.assertEqual(int(candidate[48]), 6)
                    self.assert_scatter_matches(data, oracle)

    def test_original_geometry_mode_also_replays_binary64_when_float32_is_unsafe(self):
        for device in self.devices:
            data = self.fixture(device, 0,
                                [[.2, .00001, .001], [0, 0, 0],
                                 [1, 0, 0], [1, .0001, 0]], exact=False)
            oracle, candidate = self.packets(data, 0)
            self.assertEqual(int(candidate[48]), 6)
            self.assert_scatter_matches(data, oracle)

    def test_overflow_and_nonfinite_history_fail_without_partial_scatter(self):
        for device in self.devices:
            data = self.fixture(device, 0,
                                [[0, 0, 1e-25], [0, 0, 0], [1, 0, 0], [0, 1, 0]],
                                colors=(1, 1, 0, 0), exact=False)
            oracle, _ = self.packets(data, 0)
            self.assertEqual(int(oracle[48]), 4)
            forces, hessians = self.scatter(data, check=False)
            self.assertEqual(int(data[-1].context.error.numpy()[0]), 104)
            self.assertFalse(np.any(forces))
            self.assertFalse(np.any(hessians))
            for role in range(4):
                points = np.array([[.2, .2, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
                previous = points.copy()
                previous[role, 0] = np.nan
                data = self.fixture(device, 0, points, previous=previous)
                forces, hessians = self.scatter(data, check=False)
                self.assertEqual(int(data[-1].context.error.numpy()[0]), 101)
                self.assertFalse(np.any(forces))
                self.assertFalse(np.any(hessians))

    def test_scalar_cast_underflow_does_not_erase_representable_derivatives(self):
        for device in self.devices:
            for stiffness in (1e-50, 1e-45):
                # The first casts to zero; the second rounds to a nonzero
                # subnormal with a large relative error. Both can multiply
                # large barrier derivatives into ordinary float32 results.
                data = self.fixture(device, 0,
                                    [[.2, .2, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]],
                                    radius=1e18, stiffness=stiffness)
                oracle, candidate = self.packets(data, 0)
                self.assertEqual(int(oracle[48]), 0)
                self.assertEqual(int(oracle[49]), 1)
                self.assertGreater(float(np.max(np.abs(oracle[:48]))), 1e-16)
                self.assertEqual(int(candidate[48]), 6)
                forces, hessians = self.scatter(data)
                expected = oracle[:48].reshape(4, 12)
                np.testing.assert_array_equal(forces, expected[:, :3])
                np.testing.assert_array_equal(hessians, expected[:, 3:].reshape(4, 3, 3))

    def test_large_finite_friction_history_preserves_friction_through_replay(self):
        for device in self.devices:
            points = np.array([[.2, .2, .001], [0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
            previous = points.copy()
            previous[0, 0] = -1e30
            data = self.fixture(device, 0,
                                points, previous=previous, friction=.2)
            oracle, candidate = self.packets(data, 0)
            self.assertEqual(int(oracle[48]), 0)
            self.assertEqual(int(oracle[49]), 1)
            self.assertGreater(abs(float(oracle[0])), .01)
            self.assertEqual(int(candidate[48]), 6)
            forces, hessians = self.scatter(data)
            expected = oracle[:48].reshape(4, 12)
            np.testing.assert_array_equal(forces, expected[:, :3])
            np.testing.assert_array_equal(hessians, expected[:, 3:].reshape(4, 3, 3))

    def test_large_friction_smoothing_scale_keeps_mollifier_correction(self):
        for device in self.devices:
            points = np.array([[.2, .2, .001], [0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
            previous = points.copy()
            previous[0, 0] -= 1.5e19
            data = self.fixture(device, 0, points, previous=previous, friction=.2)
            # eps_u is finite, but squaring it in float32 overflows. The
            # -slip/eps_u**2 term is still a material part of this force.
            data[-1].begin(1e10, .002, .2, 2e9)
            oracle, candidate = self.packets(data, 0)
            self.assertEqual(int(oracle[48]), 0)
            self.assertEqual(int(oracle[49]), 1)
            self.assertGreater(abs(float(oracle[0])), .01)
            self.assertIn(int(candidate[48]), (0, 6))
            if int(candidate[48]) == 0:
                self.assertAlmostEqual(float(candidate[0]) / float(oracle[0]), 1.0, delta=5e-4)
            self.assert_scatter_matches(data, oracle)

    def test_second_exact_queue_resets_and_capacity_overflow_fails_closed(self):
        points = [[.2, .0002, .001], [-1, 0, 0], [0, 0, 0], [1, 0, 0]]
        for device in self.devices:
            data = self.fixture(device, 0, points, friction=.2, exact=True)
            oracle, candidate = self.packets(data, 0)
            self.assertEqual(int(candidate[48]), 6)
            self.assert_scatter_matches(data, oracle)
            workspace = data[-1]
            self.assertEqual(int(workspace.context.pending_count.numpy()[0]), 1)
            self.assertEqual(int(workspace.context.exact_count.numpy()[0]), 1)
            p = workspace.context.parameters
            workspace.begin(float(p.dt), float(p.d_hat), float(p.friction), float(p.eps_v))
            self.assertEqual(int(workspace.context.pending_count.numpy()[0]), 0)
            self.assertEqual(int(workspace.context.exact_count.numpy()[0]), 0)
            workspace.context.exact_indices = wp.empty(0, int, device=device)
            forces, hessians = self.scatter(data, check=False)
            self.assertEqual(int(workspace.context.error.numpy()[0]), 12)
            self.assertEqual(int(workspace.context.exact_count.numpy()[0]), 1)
            self.assertFalse(np.any(forces))
            self.assertFalse(np.any(hessians))

    def test_parallel_ordinary_replay_handles_513_disjoint_rows_and_color_ownership(self):
        count = 513
        points = np.array([[.2, .00001, .001], [0, 0, 0],
                           [1, 0, 0], [1, .0001, 0]], np.float32)
        previous = points.copy()
        previous[0, 0] -= .0001
        colors = np.tile(np.array([0, 1, 0, 1], np.int32), count)
        for device in self.devices:
            reference = self.fixture(device, 0, points, previous=previous,
                                     friction=.2, exact=False)
            oracle, candidate = self.packets(reference, 0)
            self.assertEqual(int(oracle[48]), 0)
            self.assertEqual(int(oracle[49]), 1)
            self.assertEqual(int(candidate[48]), 6)
            q = wp.array(np.tile(points, (count, 1)), wp.vec3, device=device)
            history = wp.array(np.tile(previous, (count, 1)), wp.vec3, device=device)
            triangles = wp.array(np.arange(4 * count, dtype=np.int32).reshape(count, 4)[:, 1:],
                                 int, device=device)
            edges = wp.empty((0, 4), int, device=device)
            info = build_tri_mesh_collision_info(
                4 * count, count, 0, vertex_collision_buffer_pre_alloc=1,
                edge_collision_buffer_pre_alloc=1, device=device)
            rows = np.full(info.vertex_colliding_triangles.shape[0], -1, np.int32)
            offsets = info.vertex_colliding_triangles_offsets.numpy()
            counts = np.zeros(4 * count, np.int32)
            for row in range(count):
                vertex = 4 * row
                rows[2 * offsets[vertex]:2 * offsets[vertex] + 2] = [vertex, row]
                counts[vertex] = 1
            info.vertex_colliding_triangles.assign(rows)
            info.vertex_colliding_triangles_count.assign(counts)
            info.edge_colliding_edges.fill_(-1)
            info.edge_colliding_edges_count.zero_()
            groups = wp.array(colors, int, device=device)
            workspace = dat_contacts.EnergyWorkspace(
                info, device=device, exact_geometry=False, precision="float32")
            workspace.begin(.01, .002, .2, .01)
            if device.is_cuda:
                # Three 128-thread blocks process the ordinary queue. The
                # remaining 129 entries exercise the second grid-stride pass.
                workspace.context.ordinary_workers = 384
            data = q, history, triangles, edges, info, groups, workspace
            expected = np.tile(oracle[:48].reshape(4, 12), (count, 1))
            for color in (0, 1):
                actual_f, actual_h = self.scatter(data, color)
                selected = expected.copy()
                selected[colors != color] = 0
                np.testing.assert_array_equal(actual_f, selected[:, :3])
                np.testing.assert_array_equal(actual_h, selected[:, 3:].reshape(4 * count, 3, 3))
                self.assertEqual(int(workspace.context.pending_count.numpy()[0]), count)
                self.assertEqual(int(workspace.context.exact_count.numpy()[0]), 0)

    @staticmethod
    def independent_barrier(q, kind, radius, stiffness):
        """Smooth interior-feature energy from a separate NumPy projection."""
        if kind == 0:
            basis = np.column_stack((q[2] - q[1], q[3] - q[1]))
            uv = np.linalg.lstsq(basis, q[0] - q[1], rcond=None)[0]
            if not (uv[0] > .05 and uv[1] > .05 and uv.sum() < .95):
                raise AssertionError("Finite-difference fixture crossed a triangle feature")
            residual = q[0] - q[1] - basis @ uv
        else:
            basis = np.column_stack((q[1] - q[0], -(q[3] - q[2])))
            uv = np.linalg.lstsq(basis, q[2] - q[0], rcond=None)[0]
            if not np.all((uv > .05) & (uv < .95)):
                raise AssertionError("Finite-difference fixture crossed an edge feature")
            residual = q[0] - q[2] + basis @ uv
        distance = np.linalg.norm(residual)
        return -stiffness * (distance - radius) ** 2 * np.log(distance / radius)

    def test_barrier_force_and_diagonal_curvature_match_independent_energy_differences(self):
        fixtures = [
            (0, [[.25, .2, .055], [0, 0, 0], [1, .1, .02], [-.1, 1, .05]]),
            (1, [[-.5, -.05, 0], [.5, .05, .02],
                 [.02, -.5, .055], [-.02, .5, .075]]),
        ]
        for device in self.devices:
            for kind, points in fixtures:
                data = self.fixture(device, kind, points, radius=.1)
                forces, hessians = self.scatter(data)
                q = data[0].numpy().astype(np.float64)
                p = data[-1].context.parameters
                energy = lambda x: self.independent_barrier(x, kind, float(p.d_hat), float(p.k_barrier))
                base = energy(q)
                step = 2e-5
                expected_f = np.zeros((4, 3))
                expected_h = np.zeros((4, 3, 3))
                for role in range(4):
                    for k in range(3):
                        delta = np.zeros_like(q)
                        delta[role, k] = step
                        plus, minus = energy(q + delta), energy(q - delta)
                        expected_f[role, k] = -(plus - minus) / (2 * step)
                        expected_h[role, k, k] = (plus - 2 * base + minus) / step**2
                        for l in range(k):
                            other = np.zeros_like(q)
                            other[role, l] = step
                            value = (energy(q + delta + other) - energy(q + delta - other)
                                     - energy(q - delta + other) + energy(q - delta - other)) / (4 * step**2)
                            expected_h[role, k, l] = expected_h[role, l, k] = value
                self.assert_relative_norm(forces, expected_f)
                self.assert_relative_norm(hessians, expected_h)


if __name__ == "__main__":
    unittest.main()
