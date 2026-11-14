"""GS compact contact packets, replay routing and full barrier curvature."""

import unittest

import numpy as np
import warp as wp

from gpu import contact
from gpu.contact_original import ContactBackendIPC
from gpu.gs_contact_fast import make_ordinary
from gpu.query_groups import QueryGroups
from newton._src.geometry.tri_mesh_collision import build_tri_mesh_collision_info

from .test_movement_box_queries import fixture
from . import test_dat_contact_precision as dat_checks


def make_case(device, kind, points, *, previous=None, friction=0.0, exact=True,
              colors=(0, 1, 2, 3), radius=.002, stiffness=100.0, precision="float32"):
    triangles = [[1, 2, 3]] if kind == 0 else []
    edges = [] if kind == 0 else [[-1, -1, 0, 1], [-1, -1, 2, 3]]
    model, original, unused = fixture(device, points, triangles, edges, capacity=1)
    model.particle_colors.assign(colors)
    info = original.info
    if kind == 0:
        info.vertex_colliding_triangles.assign([0, 0, 0, 0, 0, 0, 0, 0])
        info.vertex_colliding_triangles_count.assign([1, 0, 0, 0])
    else:
        info.edge_colliding_edges.assign([0, 1, 1, 0])
        info.edge_colliding_edges_count.assign([1, 1])
    backend = ContactBackendIPC(model, info, d_hat=radius, k_barrier=stiffness, friction=friction,
        eps_v=.01, exact_geometry=exact, grouped_precision=precision, track_query_epoch=True)
    backend.refresh()
    history = wp.array(np.asarray(points if previous is None else previous, np.float32), wp.vec3, device=device)
    groups = QueryGroups(backend)
    groups.build(model.particle_colors)
    current = wp.zeros(1, int, device=device)
    return model, backend, history, current, groups.view(current)


def scatter(case, color=0, *, check=True):
    model, backend, previous, current, view = case
    current.fill_(color)
    forces = wp.zeros(4, wp.vec3, device=model.device)
    hessians = wp.zeros(4, wp.mat33, device=model.device)
    backend.accumulate(model.particle_q, previous, -1, forces, hessians, .01,
        colors=model.particle_colors, current_color=current, query_view=view)
    if check:
        backend.check()
    return forces.numpy(), hessians.numpy()


class GsContactPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def assert_close(self, actual, expected):
        dat_checks.DatContactPrecisionTests.assert_relative_norm(self, actual, expected)

    def test_all_roles_feature_families_and_both_friction_regimes_match_binary64(self):
        fixtures = [
            (0, [[.2, .2, .001], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            (0, [[.4, -.0006, .0008], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            (0, [[-.0006, -.0006, .0006], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            (1, [[0, 0, 0], [1, 0, 0], [.5, -.5, .001], [.5, .5, .001]]),
            (1, [[0, 0, 0], [1, 0, 0], [-.0008, -.5, .0006], [-.0008, .5, .0006]]),
            (1, [[0, 0, 0], [1, 0, 0], [-.0006, .0008, 0], [-.0006, 1, 0]]),
        ]
        accepted = 0
        for device in self.devices:
            for kind, points in fixtures:
                for friction, slip in ((0.0, 0.0), (.2, 2e-5), (.2, .003)):
                    previous = np.asarray(points, np.float32).copy()
                    previous[0] -= np.asarray([slip, .3 * slip, 0], np.float32)
                    cases = [make_case(device, kind, points, previous=previous, friction=friction,
                                      precision=p) for p in ("float32", "float64")]
                    for color in range(4):
                        result, oracle = [scatter(case, color) for case in cases]
                        self.assert_close(result[0], oracle[0])
                        self.assert_close(result[1], oracle[1])
                        self.assertTrue(np.all(result[0][np.arange(4) != color] == 0))
                        self.assertTrue(np.all(result[1][np.arange(4) != color] == 0))
                        if int(cases[0][1]._grouped_fast.ordinary_count.numpy()[0]) == 0:
                            accepted += 1
        self.assertGreater(accepted, 0, "Regular contacts must execute the compact path")

    def test_uncertain_float32_uses_ordinary_and_exact_only_when_required(self):
        fixtures = [
            (0, [[.2, .00001, .001], [0, 0, 0], [1, 0, 0], [1, .0001, 0]]),
            (0, [[.2, .0002, .001], [-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
            (1, [[-1, 0, 0], [1, 0, 0], [-.5, .001, .0002], [.5, .001, .0002]]),
        ]
        for device in self.devices:
            for exact in (False, True):
                for kind, points in fixtures:
                    compact = make_case(device, kind, points, exact=exact, friction=.2)
                    ordinary = make_case(device, kind, points, exact=exact, friction=.2, precision="float64")
                    actual, expected = scatter(compact), scatter(ordinary)
                    self.assertEqual(int(compact[1]._grouped_fast.ordinary_count.numpy()[0]), 1)
                    for a, b in zip(actual, expected):
                        np.testing.assert_array_equal(a, b)
                    if not exact:
                        self.assertEqual(int(compact[1].pending_count.numpy()[0]), 0)
            points = [[.2, .2, 1e-12], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
            compact = make_case(device, 0, points)
            ordinary = make_case(device, 0, points, precision="float64")
            actual, expected = scatter(compact), scatter(ordinary)
            self.assertEqual(int(compact[1]._grouped_fast.ordinary_count.numpy()[0]), 1)
            self.assertEqual(int(compact[1].pending_count.numpy()[0]), 1)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    def test_cutoff_neighbors_and_scalar_underflow_preserve_replay_results(self):
        radius = np.float32(.002)
        for device in self.devices:
            for gap in (np.nextafter(radius, np.float32(-np.inf)), radius,
                        np.nextafter(radius, np.float32(np.inf))):
                points = [[.25, .25, gap], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
                compact = make_case(device, 0, points, radius=float(radius))
                ordinary = make_case(device, 0, points, radius=float(radius), precision="float64")
                for a, b in zip(scatter(compact), scatter(ordinary)):
                    np.testing.assert_array_equal(a, b)
            points = [[.2, .2, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
            compact = make_case(device, 0, points, radius=1e18, stiffness=1e-50)
            ordinary = make_case(device, 0, points, radius=1e18, stiffness=1e-50, precision="float64")
            actual, expected = scatter(compact), scatter(ordinary)
            self.assertEqual(int(compact[1]._grouped_fast.ordinary_count.numpy()[0]), 1)
            self.assertGreater(np.linalg.norm(actual[0]), 0)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    def test_barrier_hessian_matches_independent_energy_differences(self):
        fixtures = [
            (0, [[.25, .2, .055], [0, 0, 0], [1, .1, .02], [-.1, 1, .05]]),
            (1, [[-.5, -.05, 0], [.5, .05, .02], [.02, -.5, .055], [-.02, .5, .075]]),
        ]
        for device in self.devices:
            for kind, points in fixtures:
                case = make_case(device, kind, points, colors=(0, 0, 0, 0), radius=.1)
                force, hessian = scatter(case)
                self.assertEqual(int(case[1]._grouped_fast.ordinary_count.numpy()[0]), 0)
                q = case[0].particle_q.numpy().astype(np.float64)
                energy = lambda x: dat_checks.DatContactPrecisionTests.independent_barrier(x, kind, .1, 100.0)
                base, step = energy(q), 2e-5
                expected_f, expected_h = np.zeros((4, 3)), np.zeros((4, 3, 3))
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
                self.assert_close(force, expected_f)
                self.assert_close(hessian, expected_h)

    def test_nonfinite_history_inactive_roles_and_queue_overflow_fail_closed(self):
        points = np.asarray([[.2, .2, .001], [0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        for device in self.devices:
            case = make_case(device, 0, points, colors=(0, 0, 0, 0))
            case[0].particle_flags.assign([1, 0, 1, 1])
            case[0].particle_mass.assign([1, 1, 0, 1])
            force, hessian = scatter(case)
            np.testing.assert_array_equal(force[1:3], 0)
            np.testing.assert_array_equal(hessian[1:3], 0)
            for role in range(4):
                previous = points.copy()
                previous[role, 0] = np.nan
                invalid = make_case(device, 0, points, previous=previous)
                force, hessian = scatter(invalid, check=False)
                self.assertEqual(int(invalid[1].error.numpy()[0]), 101)
                self.assertFalse(np.any(force))
                self.assertFalse(np.any(hessian))
            uncertain = [[.2, .2, 1e-12], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
            for which in ("ordinary", "exact"):
                case = make_case(device, 0, uncertain)
                if which == "ordinary":
                    case[1]._grouped_fast.ordinary_pending = wp.empty(0, int, device=device)
                else:
                    case[1].pending = wp.empty(0, int, device=device)
                force, hessian = scatter(case, check=False)
                self.assertEqual(int(case[1].error.numpy()[0]), 12)
                self.assertFalse(np.any(force))
                self.assertFalse(np.any(hessian))

    def test_cuda_capture_replays_fast_and_uncertain_geometry_with_device_colors(self):
        for device in wp.get_cuda_devices():
            regular = [[.2, .2, .001], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
            case = make_case(device, 0, regular)
            model, backend, history, current, view = case
            force = wp.zeros(4, wp.vec3, device=device)
            hessian = wp.zeros(4, wp.mat33, device=device)

            def run():
                force.zero_()
                hessian.zero_()
                backend.accumulate(model.particle_q, history, -1, force, hessian, .01,
                    colors=model.particle_colors, current_color=current, query_view=view)

            run()
            with wp.ScopedCapture(device=device) as capture:
                run()
            for selected, gap in ((0, .001), (2, 1e-12), (3, .001)):
                points = np.asarray(regular, np.float32)
                points[0, 2] = gap
                model.particle_q.assign(points)
                history.assign(points)
                current.fill_(selected)
                wp.capture_launch(capture.graph)
                backend.check()
                oracle = make_case(device, 0, points, precision="float64")
                expected = scatter(oracle, selected)
                self.assert_close(force.numpy(), expected[0])
                self.assert_close(hessian.numpy(), expected[1])

    def test_distributed_ordinary_pairs_are_bitwise_equal_and_queue_complete(self):
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA device unavailable")
        for device in devices:
            for pair_count in (1, 65, 401):
                # Disjoint pairs eliminate atomic-sum ordering: every local
                # force/Hessian component is exactly one ordinary packet's
                # component. 401 pairs exceed three 128-worker strides.
                points = np.tile(np.asarray([[.2, .00001, .001], [0, 0, 0],
                                             [1, 0, 0], [1, .0001, 0]], np.float32), (pair_count, 1))
                points[:, 0] += np.repeat(np.arange(pair_count, dtype=np.float32) * 3, 4)
                previous = points.copy()
                previous[::4, 0] -= np.float32(2e-5)
                triangles = np.arange(4 * pair_count, dtype=np.int32).reshape(-1, 4)[:, 1:]
                info = build_tri_mesh_collision_info(4 * pair_count, pair_count, 0,
                    vertex_collision_buffer_pre_alloc=1, device=device)
                pairs = np.zeros(info.vertex_colliding_triangles.shape[0], np.int32)
                pairs[:2 * pair_count:2] = np.arange(pair_count, dtype=np.int32) * 4
                pairs[1:2 * pair_count:2] = np.arange(pair_count, dtype=np.int32)
                info.vertex_colliding_triangles.assign(pairs)
                q = wp.array(points, wp.vec3, device=device)
                history = wp.array(previous, wp.vec3, device=device)
                tri = wp.array(triangles, int, device=device)
                edges = wp.empty((0, 4), int, device=device)
                ids = wp.array(np.arange(pair_count, dtype=np.int32), int, device=device)
                pending = wp.array(np.random.default_rng(9271).permutation(pair_count).astype(np.int32), int, device=device)
                count = wp.array([pair_count], int, device=device)
                colors = wp.array(np.tile(np.arange(4, dtype=np.int32), pair_count), int, device=device)
                flags = wp.ones(4 * pair_count, int, device=device)
                masses = wp.ones(4 * pair_count, float, device=device)
                current = wp.zeros(1, int, device=device)
                parameters = contact.parameters(d_hat=.002, k_barrier=100, dt=.01,
                    friction=.2, eps_v=.01, exact_geometry=False)
                outputs = []
                for distributed in (False, True):
                    force = wp.zeros(4 * pair_count, wp.vec3, device=device)
                    hessian = wp.zeros(4 * pair_count, wp.mat33, device=device)
                    exact_pending = wp.empty(pair_count, int, device=device)
                    exact_count = wp.zeros(1, int, device=device)
                    error = wp.zeros(1, int, device=device)
                    for selected in range(4):
                        current.fill_(selected)
                        wp.launch(make_ordinary(distributed), 128,
                            [q, history, tri, edges, info, ids, colors, flags, masses, current,
                             parameters, 128, pending, count, exact_pending, exact_count,
                             force, hessian, error], device=device, block_dim=64)
                    self.assertEqual(int(error.numpy()[0]), 0)
                    self.assertEqual(int(exact_count.numpy()[0]), 0)
                    outputs.append((force.numpy(), hessian.numpy()))
                    self.assertTrue(np.all(np.linalg.norm(outputs[-1][0][::4], axis=1) > 0),
                                    "Every queued pair must contribute its nonzero point force")
                for actual, expected in zip(outputs[1], outputs[0]):
                    self.assertEqual(actual.tobytes(), expected.tobytes(),
                                     "Disjoint pair packets must preserve every force/Hessian bit")

    def test_grouped_ordinary_distribution_preserves_replay_and_exact_routing(self):
        fixtures = [
            (0, [[.2, .00001, .001], [0, 0, 0], [1, 0, 0], [1, .0001, 0]]),
            (0, [[.2, .2, 1e-12], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            (1, [[-1, 0, 0], [1, 0, 0], [-.5, .001, .0002], [.5, .001, .0002]]),
        ]
        for device in self.devices:
            for kind, points in fixtures:
                for exact in (False, True):
                    actual = make_case(device, kind, points, friction=.2, exact=exact)
                    oracle = make_case(device, kind, points, friction=.2, exact=exact)
                    context = actual[1]._grouped_fast
                    self.assertEqual(context.ordinary_workers, device.sm_count * 64 if device.is_cuda else 1)
                    self.assertEqual(context.ordinary_block_dim, 64 if device.is_cuda else 128)
                    linear = oracle[1]._grouped_fast
                    linear._ordinary = make_ordinary(False)
                    linear.ordinary_workers = device.sm_count * 128 if device.is_cuda else 1
                    linear.ordinary_block_dim = 128
                    for selected in range(4):
                        result, reference = scatter(actual, selected), scatter(oracle, selected)
                        self.assertEqual(int(context.ordinary_count.numpy()[0]), 1)
                        self.assertEqual(int(context.ordinary_count.numpy()[0]),
                                         int(linear.ordinary_count.numpy()[0]))
                        self.assertEqual(int(actual[1].pending_count.numpy()[0]),
                                         int(oracle[1].pending_count.numpy()[0]))
                        for value, expected in zip(result, reference):
                            self.assertEqual(value.tobytes(), expected.tobytes())


if __name__ == "__main__":
    unittest.main()
