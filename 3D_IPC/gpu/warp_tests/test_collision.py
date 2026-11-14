"""Independent IPC/CCD checks on the actual imported float32 data layout."""

from pathlib import Path
import math
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/newton"))

import numpy as np
import warp as wp

from newton._src.geometry.tri_mesh_collision import build_tri_mesh_collision_info
from gpu.contact import ContactBackend, parameters
from gpu.ccd import CcdBackend
from gpu.policies import InitialGuess


class CollisionFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def fixture(self, device, *, z=.002, k_barrier=100, d_hat=.005, friction=0):
        q = wp.array([[0, 0, z], [-1, -1, 0], [1, -1, 0], [0, 1, 0]], wp.vec3, device=device)
        model = SimpleNamespace(device=device, particle_count=4, tri_count=1, edge_count=0, body_count=0, tet_count=0, particle_q=q,
            tri_indices=wp.array([[1, 2, 3]], int, device=device), edge_indices=wp.empty((0, 4), int, device=device),
            particle_colors=wp.array([0, 1, 2, 3], int, device=device), particle_flags=wp.ones(4, int, device=device),
            particle_mass=wp.ones(4, float, device=device), particle_world=wp.zeros(4, int, device=device))
        info = build_tri_mesh_collision_info(4, 1, 0, vertex_collision_buffer_pre_alloc=2, device=device)
        info.vertex_colliding_triangles.assign(np.array([0, 0, 0, 0] + [0] * 12, dtype=np.int32))
        info.vertex_colliding_triangles_count.assign([1, 0, 0, 0])
        backend = ContactBackend(model, info, d_hat=d_hat, k_barrier=k_barrier, friction=friction)
        backend.refresh()
        return q, model, info, backend



class ContactTests(CollisionFixture):
    def test_inactive_geometry_filter_preserves_invalid_friction_history_error(self):
        for device in self.devices:
            q, model, info, backend = self.fixture(device, z=0.1)
            previous_values = q.numpy().copy()
            previous_values[0, 0] = np.nan
            previous = wp.array(previous_values, wp.vec3, device=device)
            force = wp.zeros(4, wp.vec3, device=device)
            hessian = wp.zeros(4, wp.mat33, device=device)
            backend.accumulate(q, previous, 0, force, hessian, 0.01)
            with self.assertRaisesRegex(RuntimeError, "invalid contact input"):
                backend.check()

    def test_exact_contact_matches_cpu_exact_geometry_fixtures(self):
        fixtures = (
            (0, [[.1, 0, .002], [-1, -1, 0], [1, -1, 0], [0, 1, 0]]),
            (0, [[.2, .002, .001], [-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
            (1, [[-1, 0, 0], [1, 0, 0], [-.5, .002, .001], [.5, .002, .001]]),
        )
        for kind, points in fixtures:
            results = []
            for device in ("cpu", *wp.get_cuda_devices()):
                with self.subTest(kind=kind, device=device):
                    q, model, info, unused = self.fixture(device)
                    q.assign(points)
                    model.particle_colors.zero_()
                    if kind == 1:
                        model.tri_count, model.edge_count = 0, 2
                        model.tri_indices = wp.empty((0, 3), int, device=device)
                        model.edge_indices = wp.array([[-1, -1, 0, 1], [-1, -1, 2, 3]], int, device=device)
                        info = build_tri_mesh_collision_info(4, 0, 2, edge_collision_buffer_pre_alloc=2, device=device)
                        info.edge_colliding_edges.assign([0, 1, 0, 0, 1, 0, 0, 0])
                        info.edge_colliding_edges_count.assign([1, 1])
                    backend = ContactBackend(model, info, d_hat=.01, k_barrier=100, friction=.25)
                    backend.refresh()
                    previous_values = np.asarray(points, dtype=np.float32)
                    previous_values[0, 0] -= .001
                    previous = wp.array(previous_values, wp.vec3, device=device)
                    forces = wp.zeros(4, wp.vec3, device=device)
                    hessians = wp.zeros(4, wp.mat33, device=device)
                    # Force the native cold queue, including on well-conditioned
                    # fixtures that the fast path could otherwise certify.
                    backend.pending.zero_()
                    backend.pending_count.fill_(1)
                    wp.launch(backend._resolve, backend.workers,
                        [q, previous, model.tri_indices, model.edge_indices, info, backend.candidate_ids,
                         model.particle_colors, model.particle_flags, model.particle_mass, 0,
                         parameters(dt=.01, **backend.options), backend.words, backend.slots, backend.workers,
                         backend.pending, backend.pending_count, forces, hessians, backend.error],
                        device=device, block_dim=1)
                    backend.check()
                    results.append((forces.numpy(), hessians.numpy()))
            for actual in results[1:]:
                np.testing.assert_allclose(actual[0], results[0][0], rtol=3e-6, atol=1e-7)
                np.testing.assert_allclose(actual[1], results[0][1], rtol=3e-6, atol=1e-7)

    def test_barrier_force_and_hessian_match_independent_plane_energy(self):
        for device in self.devices:
            with self.subTest(device=str(device)):
                q, model, info, backend = self.fixture(device)
                forces = wp.zeros(4, wp.vec3, device=device)
                hessians = wp.zeros(4, wp.mat33, device=device)
                backend.accumulate(q, q, 0, forces, hessians, .01)
                backend.check()
                distance = float(q.numpy()[0, 2])
                dh = .005
                bp = -2 * (distance - dh) * math.log(distance / dh) - (distance - dh)**2 / distance
                bpp = (dh / distance)**2 + 2 * dh / distance - 3 - 2 * math.log(distance / dh)
                np.testing.assert_allclose(forces.numpy()[0], [0, 0, -100 * bp], rtol=2e-6, atol=1e-7)
                self.assertAlmostEqual(float(hessians.numpy()[0, 2, 2]), 100 * bpp, delta=abs(100 * bpp) * 2e-6)
                np.testing.assert_array_equal(forces.numpy()[1:], 0)
                self.assertEqual(int(backend.candidate_count.numpy()[0]), 1)

    def test_friction_uses_fixed_step_reference_and_physical_scaling(self):
        for device in self.devices:
            q, model, info, backend = self.fixture(device, friction=.3)
            previous = wp.clone(q)
            values = previous.numpy()
            values[0, 0] -= .001
            previous.assign(values)
            force = wp.zeros(4, wp.vec3, device=device)
            hessian = wp.zeros(4, wp.mat33, device=device)
            backend.accumulate(q, previous, 0, force, hessian, .01)
            backend.check()
            actual = force.numpy()[0]
            self.assertGreater(actual[2], 0)
            self.assertLess(actual[0], 0)
            self.assertAlmostEqual(float(-actual[0] / actual[2]), .3, places=5)

    def test_incomplete_rows_latch_error_across_later_refresh(self):
        q, model, info, backend = self.fixture("cpu")
        info.vertex_colliding_triangles_count.assign([3, 0, 0, 0])
        backend.refresh()
        info.vertex_colliding_triangles_count.assign([1, 0, 0, 0])
        backend.refresh()
        with self.assertRaisesRegex(RuntimeError, "overflow"):
            backend.check()
        backend.reset_status()
        backend.refresh()
        backend.check()

    def test_disabled_barrier_produces_no_contact_derivatives(self):
        for disabled in (dict(k_barrier=0), dict(d_hat=0)):
            q, model, info, backend = self.fixture("cpu", **disabled)
            force = wp.zeros(4, wp.vec3, device="cpu")
            hessian = wp.zeros(4, wp.mat33, device="cpu")
            backend.accumulate(q, q, 0, force, hessian, .01)
            backend.check()
            np.testing.assert_array_equal(force.numpy(), 0)
            np.testing.assert_array_equal(hessian.numpy(), 0)

    def test_conservative_queries_keep_aabb_hits_outside_closest_distance(self):
        for device in self.devices:
            q, model, info, backend = self.fixture(device)
            values = q.numpy()
            values[0] = [0.9, 0.9, .01]
            values[1:] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            q.assign(values)
            lower = wp.array([[0, 0, 0]], wp.vec3, device=device)
            upper = wp.array([[1, 1, 0]], wp.vec3, device=device)
            bvh = wp.Bvh(lower, upper)
            detector = SimpleNamespace(bvh_tris=bvh, bvh_edges=None)
            backend.detect(detector, q, .02, refit=False)
            backend.check()
            self.assertEqual(int(backend.candidate_count.numpy()[0]), 1)
            # The actual distance to the triangle is much greater than .02;
            # rejecting by approximate distance would lose a swept CCD candidate.
            self.assertGreater(np.linalg.norm(values[0] - [.5, .5, 0]), .02)

    def test_detection_skips_tet_interior_but_queries_inactive_surface_proxies(self):
        for device in self.devices:
            q, model, info, backend = self.fixture(device)
            lower = wp.array([[-1, -1, 0]], wp.vec3, device=device)
            upper = wp.array([[1, 1, 0]], wp.vec3, device=device)
            detector = SimpleNamespace(bvh_tris=wp.Bvh(lower, upper), bvh_edges=None)
            backend.surface.assign([0, 1, 1, 1])
            backend.detect(detector, q, .01, refit=False)
            backend.check()
            self.assertEqual(int(backend.candidate_count.numpy()[0]), 0)
            backend.surface.assign([1, 1, 1, 1])
            model.particle_flags.assign([0, 1, 1, 1])
            backend.detect(detector, q, .01, refit=False)
            backend.check()
            self.assertEqual(int(backend.candidate_count.numpy()[0]), 1)

    def test_large_coordinate_rigid_rounding_couples_cross_axis_arcs(self):
        from gpu import body_graph
        angle = .1
        reference = np.asarray([[0, -512, -1], [0, 512, -1],
                                [0, 512, 1], [0, -512, 1]], dtype=np.float32)
        center_y = np.float32(1e10)
        points = np.concatenate([np.float32(reference.astype(np.float64) + [x, float(center_y), 0])
                                 for x in (0, 150)])
        # A native checkpoint conversion can double this represented radial
        # offset. Opposite .09-radian proxy arcs intersect although their
        # checkpoint X gap exceeds the old radius and all per-axis X padding.
        np.testing.assert_array_equal(points[:4, 1].astype(np.float64) - float(center_y),
                                      [-1024, 1024, 1024, -1024])
        self.assertLess(75 / math.tan(.09), 1024 * math.cos(.09))
        rho = float(np.linalg.norm(reference.astype(np.float64), axis=1).max())
        query_radius = .005 + 2 * (.01 + 2 * rho * math.sin(angle / 2) +
                                   64 * np.finfo(np.float32).eps * max(1, rho))
        self.assertLess(query_radius, 150)
        triangles = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int32)
        edge_pairs = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3),
                      (4, 5), (5, 6), (4, 6), (6, 7), (4, 7)]
        edges = np.asarray([[-1, -1, a, b] for a, b in edge_pairs], dtype=np.int32)
        for device in self.devices:
            q = wp.array(points, wp.vec3, device=device)
            owners = wp.array([0] * 4 + [1] * 4, int, device=device)
            model = SimpleNamespace(device=device, particle_count=8, tri_count=4, edge_count=10,
                body_count=2, particle_q=q, tri_indices=wp.array(triangles, int, device=device),
                edge_indices=wp.array(edges, int, device=device), particle_colors=wp.zeros(8, int, device=device),
                particle_flags=wp.zeros(8, int, device=device), particle_mass=wp.ones(8, float, device=device),
                particle_world=wp.zeros(8, int, device=device))
            info = build_tri_mesh_collision_info(8, 4, 10, vertex_collision_buffer_pre_alloc=64,
                                                 edge_collision_buffer_pre_alloc=64, device=device)
            backend = ContactBackend(model, info, d_hat=.005, k_barrier=100, node_body=owners)
            lower = wp.array(points[triangles].min(axis=1), wp.vec3, device=device)
            upper = wp.array(points[triangles].max(axis=1), wp.vec3, device=device)
            edge_lower = wp.array(points[edges[:, 2:]].min(axis=1), wp.vec3, device=device)
            edge_upper = wp.array(points[edges[:, 2:]].max(axis=1), wp.vec3, device=device)
            detector = SimpleNamespace(bvh_tris=wp.Bvh(lower, upper), bvh_edges=wp.Bvh(edge_lower, edge_upper))
            backend.detect(detector, q, query_radius, refit=False)
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    backend.detect(detector, q, query_radius, refit=False)
                wp.capture_launch(capture.graph)
            backend.check()
            count = int(backend.candidate_count.numpy()[0])
            ids = backend.candidate_ids.numpy()[:count]
            self.assertTrue(np.any(ids >= 0))
            self.assertTrue(np.any(ids < 0))
            counts = wp.zeros(3, int, device=device)
            conflicts = wp.zeros((2, 1), wp.uint32, device=device)
            colors = wp.zeros(2, int, device=device)
            marks = wp.zeros(2, int, device=device)
            color_count = wp.zeros(1, int, device=device)
            wp.launch(body_graph.count_incidence, backend.capacity,
                [backend.candidate_ids, backend.candidate_count, model.tri_indices, model.edge_indices,
                 info, owners, counts, conflicts, backend.error], device=device)
            wp.launch(body_graph.color_bodies, 1,
                [conflicts, colors, marks, color_count, backend.error], device=device)
            backend.check()
            self.assertNotEqual(*colors.numpy())

    def test_actual_cuda_graph_contact_pipeline(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA device")
        q, model, info, backend = self.fixture("cuda:0")
        force = wp.zeros(4, wp.vec3, device=q.device)
        hessian = wp.zeros(4, wp.mat33, device=q.device)
        # One uncaptured invocation compiles and reserves every owned buffer.
        backend.accumulate(q, q, 0, force, hessian, .01)
        force.zero_(); hessian.zero_()
        with wp.ScopedCapture(device=q.device) as capture:
            backend.refresh()
            backend.accumulate(q, q, 0, force, hessian, .01)
        wp.capture_launch(capture.graph)
        backend.check()
        self.assertGreater(float(force.numpy()[0, 2]), 0)


class CcdTests(CollisionFixture):
    def test_zero_mass_surface_is_queried_and_ignores_active_proposals(self):
        for device in self.devices:
            q, model, info, contacts = self.fixture(device, z=.1)
            model.particle_mass.assign([0, 1, 1, 1])
            before = q.numpy().copy()
            lower = wp.array([[-1, -1, 0]], wp.vec3, device=device)
            upper = wp.array([[1, 1, 0]], wp.vec3, device=device)
            contacts.detect(SimpleNamespace(bvh_tris=wp.Bvh(lower, upper), bvh_edges=None),
                            q, .2, refit=False)
            self.assertEqual(int(contacts.candidate_count.numpy()[0]), 1)
            proposal = before.copy()
            proposal[0] = np.nan
            ccd = CcdBackend(contacts, node_box_radius=1)
            ccd.accept(q, wp.array(proposal, wp.vec3, device=device), 0, .01)
            ccd.check()
            np.testing.assert_array_equal(q.numpy(), before)

    def test_exact_step_bound_uses_a_verified_inward_float32_proposal(self):
        for device in self.devices:
            endpoints = []
            for bound in (False, True):
                q, model, info, contacts = self.fixture(device, z=.1)
                values = q.numpy().copy()
                values[0, 2] = -.1
                ccd = CcdBackend(contacts, node_box_radius=1, exact_step_bound=bound)
                ccd.accept(q, wp.array(values, wp.vec3, device=device), 0, .01)
                ccd.check()
                endpoints.append(float(q.numpy()[0, 2]))
            self.assertGreaterEqual(endpoints[1], 1e-8)
            self.assertLess(endpoints[1], 1e-6)
            self.assertLess(endpoints[1], endpoints[0])

    def test_exact_step_bound_requires_robust_geometry(self):
        q, model, info, contacts = self.fixture("cpu")
        with self.assertRaisesRegex(ValueError, "require robust CCD"):
            CcdBackend(contacts, exact_geometry=False, exact_step_bound=True)

    def test_actual_cuda_graph_crossing_and_rounded_endpoint_reproof(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA device")
        q, model, info, contacts = self.fixture("cuda:0", z=.1)
        start = wp.clone(q)
        proposal_values = q.numpy()
        proposal_values[0, 2] = -.1
        proposal = wp.array(proposal_values, wp.vec3, device=q.device)
        ccd = CcdBackend(contacts, node_box_radius=1)
        ccd.accept(q, proposal, 0, .01)
        ccd.check()
        wp.copy(q, start)
        with wp.ScopedCapture(device=q.device) as capture:
            wp.copy(q, start)
            ccd.accept(q, proposal, 0, .01)
        for _ in range(2):
            wp.capture_launch(capture.graph)
            ccd.check()
            self.assertGreater(float(q.numpy()[0, 2]), 0)
            self.assertLess(float(q.numpy()[0, 2]), .1)

    def test_single_vertex_crossing_is_clipped_in_both_policies(self):
        for device in self.devices:
            for exact in (True, False):
                with self.subTest(device=str(device), exact=exact):
                    q, model, info, contacts = self.fixture(device, z=.1)
                    before = q.numpy()
                    values = before.copy()
                    values[0, 2] = -.1
                    proposal = wp.array(values, wp.vec3, device=device)
                    ccd = CcdBackend(contacts, node_box_radius=1, exact_geometry=exact)
                    ccd.accept(q, proposal, 0, .01)
                    ccd.check()
                    actual = q.numpy()
                    self.assertGreater(float(actual[0, 2]), 0)
                    self.assertLess(float(actual[0, 2]), .1)
                    np.testing.assert_array_equal(actual[1:], before[1:])

    def test_coupled_same_color_edges_use_a_common_certified_fraction(self):
        for device in self.devices:
            points = np.array([[-1, 0, .1], [1, 0, .1], [0, -1, -.1], [0, 1, -.1]], dtype=np.float32)
            q = wp.array(points, wp.vec3, device=device)
            model = SimpleNamespace(device=device, particle_count=4, tri_count=0, edge_count=2, particle_q=q,
                tri_indices=wp.empty((0, 3), int, device=device), edge_indices=wp.array([[-1,-1,0,1],[-1,-1,2,3]], int, device=device),
                particle_colors=wp.zeros(4, int, device=device), particle_flags=wp.ones(4, int, device=device),
                particle_mass=wp.ones(4, float, device=device), particle_world=wp.zeros(4, int, device=device))
            info = build_tri_mesh_collision_info(4, 0, 2, vertex_collision_buffer_pre_alloc=2,
                                                  edge_collision_buffer_pre_alloc=2, device=device)
            info.edge_colliding_edges.assign([0,1,0,0,1,0,0,0])
            info.edge_colliding_edges_count.assign([1,1])
            contacts = ContactBackend(model, info, d_hat=.005, k_barrier=100)
            contacts.refresh()
            proposal = points.copy()
            proposal[:, 2] *= -1
            ccd = CcdBackend(contacts, node_box_radius=1)
            ccd.accept(q, wp.array(proposal, wp.vec3, device=device), 0, .01)
            ccd.check()
            actual = q.numpy()
            self.assertGreater(float(actual[:2, 2].min()), float(actual[2:, 2].max()))
            fractions = (actual[:, 2] - points[:, 2]) / (proposal[:, 2] - points[:, 2])
            np.testing.assert_allclose(fractions, fractions[0], rtol=0, atol=1e-6)
            self.assertGreater(float(fractions[0]), 0)
            self.assertLess(float(fractions[0]), .5)
            q.assign(points)
            direct = CcdBackend(contacts, node_box_radius=1, exact_step_bound=True)
            direct.accept(q, wp.array(proposal, wp.vec3, device=device), 0, .01)
            direct.check()
            np.testing.assert_array_equal(q.numpy(), actual)

    def test_rounded_float32_endpoint_is_recertified(self):
        q, model, info, contacts = self.fixture("cpu", z=.1)
        values = q.numpy()
        values[:, 2] = 100000
        values[0, 2] = np.nextafter(np.float32(100000), np.float32(np.inf))
        q.assign(values)
        proposal = values.copy()
        proposal[0, 2] = np.nextafter(np.float32(100000), np.float32(-np.inf))
        ccd = CcdBackend(contacts, node_box_radius=1)
        ccd.set_reference(q)
        ccd.accept(q, wp.array(proposal, wp.vec3, device="cpu"), 0, .01)
        ccd.check()
        self.assertGreater(float(q.numpy()[0, 2]), 100000)

    def test_true_initial_intersection_is_not_a_successful_move(self):
        q, model, info, contacts = self.fixture("cpu", z=0)
        before = q.numpy().copy()
        proposal = before.copy(); proposal[0, 2] = .1
        ccd = CcdBackend(contacts, node_box_radius=1)
        ccd.accept(q, wp.array(proposal, wp.vec3, device="cpu"), 0, .01)
        with self.assertRaisesRegex(RuntimeError, "initial intersection"):
            ccd.check()
        np.testing.assert_array_equal(q.numpy(), before)

    def test_stale_proposals_from_other_colors_are_not_consumed(self):
        q, model, info, contacts = self.fixture("cpu", z=.1)
        before = q.numpy().copy()
        proposal = before.copy()
        proposal[0, 2] = .09
        proposal[1:] = np.nan
        ccd = CcdBackend(contacts, node_box_radius=1)
        ccd.accept(q, wp.array(proposal, wp.vec3, device="cpu"), 0, .01)
        ccd.check()
        np.testing.assert_array_equal(q.numpy()[1:], before[1:])


class PredictorTests(CollisionFixture):
    def scene(self, device):
        q, model, info, contacts = self.fixture(device, z=.1)
        model.particle_flags.assign([1, 0, 0, 0])
        lower = wp.array([[-1, -1, 0]], wp.vec3, device=device)
        upper = wp.array([[1, 1, 0]], wp.vec3, device=device)
        detector = SimpleNamespace(bvh_tris=wp.Bvh(lower, upper), bvh_edges=None,
            lower_bounds_tris=lower, upper_bounds_tris=upper)
        ccd = CcdBackend(contacts, node_box_radius=1)
        velocity = wp.array([[0, 0, -20], [0,0,0], [0,0,0], [0,0,0]], wp.vec3, device=device)
        return q, model, contacts, detector, ccd, velocity

    def test_global_initial_guess_uses_stock_own_linear_toi(self):
        for device in self.devices:
            q, model, contacts, detector, ccd, velocity = self.scene(device)
            before = q.numpy().copy()
            policy = InitialGuess(ccd, detector, params={"use_ccd_guess": True})
            policy.apply(q, wp.clone(q), velocity, wp.empty_like(q), .01, {"use_ccd_guess": True})
            ccd.check()
            actual = q.numpy()
            self.assertGreater(float(actual[0, 2]), 0)
            self.assertAlmostEqual(float(actual[0, 2]), .01, delta=2e-5)
            np.testing.assert_array_equal(actual[1:], before[1:])

    def test_zero_mass_active_flag_does_not_generate_a_prediction(self):
        for device in self.devices:
            q, model, contacts, detector, ccd, velocity = self.scene(device)
            model.particle_mass.assign([0, 1, 1, 1])
            before = q.numpy().copy()
            velocity.assign(np.full((4, 3), np.nan, dtype=np.float32))
            policy = InitialGuess(ccd, detector, params={"use_ccd_guess": True})
            policy.apply(q, wp.clone(q), velocity, wp.empty_like(q), .01, {"use_ccd_guess": True})
            ccd.check()
            np.testing.assert_array_equal(q.numpy(), before)
            np.testing.assert_array_equal(policy.targets.numpy(), before)

    def test_actual_cuda_graph_global_colored_and_level_predictors(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA device")
        modes = ({"use_ccd_guess": True}, {"use_colored_ccd_guess": True},
                 {"use_colored_ccd_guess_level_based": True})
        for mode in modes:
            with self.subTest(mode=mode):
                q, model, contacts, detector, ccd, velocity = self.scene("cuda:0")
                start = wp.clone(q)
                inertia = wp.empty_like(q)
                params = {**mode, "colored_ccd_guess_iters": 2, "colored_ccd_guess_fraction": 1.0}
                policy = InitialGuess(ccd, detector, params=params)
                policy.apply(q, start, velocity, inertia, .01, params)
                ccd.check()
                expected = q.numpy()
                wp.copy(q, start)
                with wp.ScopedCapture(device=q.device) as capture:
                    wp.copy(q, start)
                    policy.apply(q, start, velocity, inertia, .01, params)
                for _ in range(2):
                    wp.capture_launch(capture.graph)
                    ccd.check()
                    np.testing.assert_array_equal(q.numpy(), expected)

    def test_colored_and_level_predictors_keep_targets_and_discard_trial(self):
        results = []
        for level in (False, True):
            q, model, contacts, detector, ccd, velocity = self.scene("cpu")
            before = wp.clone(q)
            params = {"use_colored_ccd_guess": not level, "use_colored_ccd_guess_level_based": level,
                      "colored_ccd_guess_iters": 2, "colored_ccd_guess_fraction": 1.0}
            policy = InitialGuess(ccd, detector, params=params)
            policy.apply(q, before, velocity, wp.empty_like(q), .01, params)
            ccd.check()
            self.assertAlmostEqual(float(policy.targets.numpy()[0, 2]), -.1, places=6)
            self.assertGreater(float(q.numpy()[0, 2]), 0)
            self.assertLess(float(q.numpy()[0, 2]), .1)
            self.assertEqual(int(policy.sweep.numpy()[0]), 2)
            results.append(q.numpy())
        np.testing.assert_array_equal(results[0], results[1])

    def test_zero_budget_validates_before_detection_or_trial(self):
        q, model, contacts, detector, ccd, velocity = self.scene("cpu")
        before = q.numpy().copy()
        policy = InitialGuess(ccd, None, params={"use_colored_ccd_guess": True})
        params = {"use_colored_ccd_guess": True, "colored_ccd_guess_iters": 0}
        policy.apply(q, wp.clone(q), velocity, wp.empty_like(q), .01, params)
        np.testing.assert_array_equal(q.numpy(), before)
        self.assertEqual(int(policy.graph.metadata.numpy()[0]), 0)
        q.assign(np.full_like(before, np.nan))
        policy.apply(q, wp.clone(q), velocity, wp.empty_like(q), .01, params)
        with self.assertRaises(RuntimeError):
            ccd.check()


if __name__ == "__main__":
    unittest.main()
