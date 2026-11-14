"""Independent group-span completeness and unchanged CCD/predictor checks."""

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
import warp as wp

from newton._src.geometry.tri_mesh_collision import build_tri_mesh_collision_info

from gpu.contact import ContactBackend
from gpu.ccd import CcdBackend
from gpu.policies import InitialGuess, PredictorGraph
from gpu.query_groups import QueryGroups
from gpu.run import FrameGraph
from gpu.scene import create_model
from gpu.solver import SolverIPC
from gpu.reference.legacy_dat_adapter import SolverDAT
from . import test_collision as collision_tests
from .test_solver_dat import point_over_triangle


def fixture(device):
    points = np.asarray([[0, 0, .1], [-1, -1, 0], [1, -1, 0], [0, 1, 0],
                         [-2, -1, 10], [2, -1, 10], [0, 1, 10]], dtype=np.float32)
    q = wp.array(points, wp.vec3, device=device)
    model = SimpleNamespace(device=device, particle_count=7, tri_count=2, edge_count=0,
        body_count=0, tet_count=0, particle_q=q, tri_indices=wp.array([[1, 2, 3], [4, 5, 6]], int, device=device),
        edge_indices=wp.empty((0, 4), int, device=device), particle_colors=wp.zeros(7, int, device=device),
        particle_flags=wp.array([1, 1, 1, 1, 0, 1, 1], int, device=device),
        particle_mass=wp.array([1, 1, 1, 1, 1, 0, 1], float, device=device),
        particle_world=wp.zeros(7, int, device=device))
    info = build_tri_mesh_collision_info(7, 2, 0, vertex_collision_buffer_pre_alloc=2, device=device)
    storage = np.full(info.vertex_colliding_triangles.shape[0], -1, dtype=np.int32)
    storage[:4] = [0, 0, 0, 1]
    info.vertex_colliding_triangles.assign(storage)
    info.vertex_colliding_triangles_count.assign([2, 0, 0, 0, 0, 0, 0])
    contacts = ContactBackend(model, info, d_hat=.005, k_barrier=100, track_query_epoch=True)
    contacts.refresh()
    return q, model, contacts


def independent_predictor_fixture(device):
    points = np.asarray([[0, 0, .1], [-1, -1, 0], [1, -1, 0], [0, 1, 0], [3, 2, .2]], np.float32)
    q = wp.array(points, wp.vec3, device=device)
    model = SimpleNamespace(device=device, particle_count=5, tri_count=1, edge_count=0,
        body_count=0, tet_count=0, particle_q=q, tri_indices=wp.array([[1, 2, 3]], int, device=device),
        edge_indices=wp.empty((0, 4), int, device=device), particle_colors=wp.zeros(5, int, device=device),
        particle_flags=wp.array([1, 0, 0, 0, 1], int, device=device),
        particle_mass=wp.array([1, 0, 0, 0, 1], float, device=device),
        particle_world=wp.zeros(5, int, device=device))
    info = build_tri_mesh_collision_info(5, 1, 0, vertex_collision_buffer_pre_alloc=4, device=device)
    contacts = ContactBackend(model, info, d_hat=.005, k_barrier=100, track_query_epoch=True)
    lower = wp.array([[-1, -1, 0]], wp.vec3, device=device)
    upper = wp.array([[1, 1, 0]], wp.vec3, device=device)
    detector = SimpleNamespace(bvh_tris=wp.Bvh(lower, upper), bvh_edges=None,
        lower_bounds_tris=lower, upper_bounds_tris=upper,
        lower_bounds_edges=wp.empty(0, wp.vec3, device=device),
        upper_bounds_edges=wp.empty(0, wp.vec3, device=device))
    velocity = wp.array([[0, 0, -20], [0, 0, 0], [0, 0, 0], [0, 0, 0], [30, -10, 20]], wp.vec3, device=device)
    return q, model, contacts, detector, velocity


@wp.kernel
def _advance_device_epoch(epoch: wp.array[wp.uint64]):
    epoch[0] += wp.uint64(1)


class GroupQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_spans_include_each_candidate_once_per_unique_nonnegative_group(self):
        for device in self.devices:
            q, model, contacts = fixture(device)
            bank = QueryGroups(contacts)
            maps = ([2, 1, 1, -1, 0, 0, 0], [1] * 7, [-1] * 7, [6, 5, 4, 3, 2, 1, 0])
            for values in maps:
                groups = wp.array(values, int, device=device)
                bank.build(groups)
                contacts.check()
                offsets = bank.offsets.numpy()
                ids = bank.ids.numpy()
                expected = {group: [] for group in range(7)}
                for encoded, nodes in ((0, [0, 1, 2, 3]), (1, [0, 4, 5, 6])):
                    for group in {values[node] for node in nodes if values[node] >= 0}:
                        expected[group].append(encoded)
                for group in range(7):
                    self.assertEqual(Counter(ids[offsets[group]:offsets[group + 1]]), Counter(expected[group]))
                    self.assertLessEqual(offsets[group + 1] - offsets[group], 2)
                self.assertLessEqual(offsets[-1], 4 * int(contacts.candidate_count.numpy()[0]))
                groups.fill_(-1)
                np.testing.assert_array_equal(bank.node_groups.numpy(), values)

    def test_activation_after_build_cannot_omit_a_now_moving_corner(self):
        for device in self.devices:
            q, model, contacts = fixture(device)
            bank = QueryGroups(contacts)
            bank.build(wp.array([-1, -1, -1, -1, -1, 4, -1], int, device=device))
            self.assertEqual(int(bank.offsets.numpy()[5] - bank.offsets.numpy()[4]), 1)
            model.particle_mass.assign([1] * 7)
            proposal = q.numpy().copy()
            proposal[5, 2] = -40
            proposal = wp.array(proposal, wp.vec3, device=device)
            oracle, grouped = wp.clone(q), wp.clone(q)
            CcdBackend(contacts, node_box_radius=1, exact_geometry=False).accept(oracle, proposal, -1, .01, bounded=False)
            backend = CcdBackend(contacts, node_box_radius=1, exact_geometry=False)
            backend.accept(grouped, proposal, -1, .01, bounded=False,
                           query_view=bank.view(wp.array([4], int, device=device)))
            backend.check()
            np.testing.assert_array_equal(grouped.numpy(), oracle.numpy())
            self.assertGreater(float(grouped.numpy()[5, 2]), -40)

    def test_nonzero_span_indices_reach_unchanged_cold_resolver(self):
        for device in self.devices:
            for exact in (False, True):
                q, model, contacts = fixture(device)
                bank = QueryGroups(contacts)
                groups = wp.array([2, 1, 1, -1, 0, 0, 0], int, device=device)
                bank.build(groups)
                self.assertEqual(int(bank.offsets.numpy()[2]), 2)
                self.assertEqual(int(bank.ids.numpy()[0]), 1, "wrong local pending indices would test the far face")
                current = wp.array([2], int, device=device)
                proposal = q.numpy().copy()
                proposal[0, 2] = -.1
                proposal = wp.array(proposal, wp.vec3, device=device)
                oracle = wp.clone(q)
                CcdBackend(contacts, node_box_radius=1, exact_geometry=exact).accept(oracle, proposal, -1, .01, bounded=False)
                for width in (32, 64, 128, 256):
                    grouped = wp.clone(q)
                    backend = CcdBackend(contacts, node_box_radius=1, exact_geometry=exact,
                                         group_query_block_dim=width)
                    backend.accept(grouped, proposal, -1, .01, bounded=False, query_view=bank.view(current))
                    backend.check()
                    np.testing.assert_array_equal(grouped.numpy(), oracle.numpy())
                    self.assertGreater(float(grouped.numpy()[0, 2]), 0)

    def test_coupled_roles_deduplicate_and_keep_the_same_common_fraction(self):
        for device in self.devices:
            q, model, contacts = fixture(device)
            bank = QueryGroups(contacts)
            bank.build(wp.array([2, 2, 2, 2, -1, -1, 0], int, device=device))
            offsets = bank.offsets.numpy()
            self.assertEqual(int(offsets[3] - offsets[2]), 2)
            proposal = q.numpy().copy()
            proposal[0, 2] = -.1
            proposal[1:4, 2] = .05
            proposal = wp.array(proposal, wp.vec3, device=device)
            oracle, grouped = wp.clone(q), wp.clone(q)
            CcdBackend(contacts, node_box_radius=1).accept(oracle, proposal, -1, .01, bounded=False)
            backend = CcdBackend(contacts, node_box_radius=1)
            backend.accept(grouped, proposal, -1, .01, bounded=False,
                           query_view=bank.view(wp.array([2], int, device=device)))
            backend.check()
            np.testing.assert_array_equal(grouped.numpy(), oracle.numpy())

    def test_encoded_edge_candidates_and_two_moving_roles_match_oracle(self):
        for device in self.devices:
            points = np.asarray([[-1, 0, .1], [1, 0, .1], [0, -1, -.1], [0, 1, -.1]], dtype=np.float32)
            q = wp.array(points, wp.vec3, device=device)
            model = SimpleNamespace(device=device, particle_count=4, tri_count=0, edge_count=2,
                body_count=0, particle_q=q, tri_indices=wp.empty((0, 3), int, device=device),
                edge_indices=wp.array([[-1, -1, 0, 1], [-1, -1, 2, 3]], int, device=device),
                particle_colors=wp.zeros(4, int, device=device), particle_flags=wp.ones(4, int, device=device),
                particle_mass=wp.ones(4, float, device=device), particle_world=wp.zeros(4, int, device=device))
            info = build_tri_mesh_collision_info(4, 0, 2, edge_collision_buffer_pre_alloc=2, device=device)
            info.edge_colliding_edges.assign([0, 1, 0, 0, 1, 0, 0, 0])
            info.edge_colliding_edges_count.assign([1, 1])
            contacts = ContactBackend(model, info, d_hat=.005, k_barrier=100, track_query_epoch=True)
            contacts.refresh()
            bank = QueryGroups(contacts)
            bank.build(wp.array([0, 1, 0, 1], int, device=device))
            np.testing.assert_array_equal(bank.ids.numpy()[:2], [-1, -1])
            proposal = points.copy()
            proposal[[0, 2], 2] *= -1
            proposal = wp.array(proposal, wp.vec3, device=device)
            oracle, grouped = wp.clone(q), wp.clone(q)
            CcdBackend(contacts, node_box_radius=1).accept(oracle, proposal, -1, .01, bounded=False)
            backend = CcdBackend(contacts, node_box_radius=1)
            backend.accept(grouped, proposal, -1, .01, bounded=False,
                           query_view=bank.view(wp.zeros(1, int, device=device)))
            backend.check()
            np.testing.assert_array_equal(grouped.numpy(), oracle.numpy())

    def test_moving_outside_group_and_stale_host_view_cannot_commit(self):
        q, model, contacts = fixture("cpu")
        bank = QueryGroups(contacts)
        groups = wp.array([2, 1, 1, -1, 0, 0, 0], int, device="cpu")
        bank.build(groups)
        view = bank.view(wp.array([2], int, device="cpu"))
        before = q.numpy().copy()
        proposal = before.copy()
        proposal[1, 2] = .01
        backend = CcdBackend(contacts, node_box_radius=1)
        backend.accept(q, wp.array(proposal, wp.vec3, device="cpu"), -1, .01,
                       bounded=False, query_view=view)
        with self.assertRaises(RuntimeError):
            backend.check()
        np.testing.assert_array_equal(q.numpy(), before)
        contacts.reset_status()
        bank.build(groups)
        with self.assertRaisesRegex(ValueError, "epoch"):
            backend.accept(q, wp.array(before, wp.vec3, device="cpu"), -1, .01, query_view=view)

    def test_too_large_group_bank_falls_back_to_all_pairs(self):
        fake = SimpleNamespace(device=wp.get_device("cpu"), capacity=2**29, track_query_epoch=True,
                               model=SimpleNamespace(particle_count=4))
        bank = QueryGroups(fake)
        self.assertFalse(bank.enabled)
        bank.build(wp.zeros(4, int, device="cpu"))
        self.assertIsNone(bank.view(wp.zeros(1, int, device="cpu")))
        self.assertEqual(bank.ids.shape[0], 0)

    def test_grouped_view_requires_the_complete_model_arrays(self):
        q, model, contacts = fixture("cpu")
        bank = QueryGroups(contacts)
        bank.build(wp.zeros(7, int, device="cpu"))
        backend = CcdBackend(contacts, node_box_radius=1)
        short = wp.array(q.numpy()[:6].copy(), wp.vec3, device="cpu")
        with self.assertRaisesRegex(TypeError, "complete model"):
            backend.accept(short, short, -1, .01, query_view=bank.view(wp.zeros(1, int, device="cpu")))

    def test_legacy_dat_keeps_original_default_dispatch_allocation_and_no_epoch(self):
        from gpu.ccd import resolve_original as original_resolve
        payload = point_over_triangle(height=.1, velocity=(.1, 0, 0), iterations=0)
        payload["params"].update(use_colored_ccd_guess_level_based=True,
                                  colored_ccd_guess_iters=1, colored_ccd_guess_fraction=.97)
        for device in self.devices:
            model = create_model(deepcopy(payload), device)
            solver = SolverDAT(model, allow_cpu=True)
            self.assertIs(solver.ccd_backend._resolve, original_resolve)
            contacts = solver.contact_backend
            policy = solver.ccd_backend._initial_guess
            self.assertFalse(contacts.track_query_epoch)
            self.assertIsNone(contacts.query_epoch)
            self.assertIsNone(policy.query_groups)
            self.assertFalse(policy.independent_colors)
            self.assertEqual(policy.graph.adjacency.shape[0], 12 * contacts.capacity)
            self.assertIsNone(policy.graph._parallel_levels)
            self.assertIsNone(policy.graph._parallel_colors)
            before, after = model.state(), model.state()
            solver.step(before, after, model.control(), None, payload["params"]["dt"])
            solver.check_status()

    def test_device_epoch_rejects_stale_view_even_without_host_generation_change(self):
        for device in self.devices:
            q, model, contacts = fixture(device)
            bank = QueryGroups(contacts)
            bank.build(wp.array([2, 1, 1, -1, 0, 0, 0], int, device=device))
            view = bank.view(wp.array([2], int, device=device))
            wp.launch(_advance_device_epoch, 1, [contacts.query_epoch], device=device)
            before = q.numpy().copy()
            proposed = before.copy()
            proposed[0, 2] = .09
            backend = CcdBackend(contacts, node_box_radius=1)
            backend.accept(q, wp.array(proposed, wp.vec3, device=device), -1, .01, query_view=view)
            with self.assertRaises(RuntimeError):
                backend.check()
            np.testing.assert_array_equal(q.numpy(), before)

    def test_predictor_canonical_arrays_and_state_match_all_pair_oracle(self):
        for device in self.devices:
            for level in (False, True):
                results = []
                for accelerated in (False, True):
                    q, model, contacts, detector, ccd, velocity = collision_tests.PredictorTests().scene(device)
                    if accelerated:
                        contacts = ContactBackend(model, contacts.info, **contacts.options, track_query_epoch=True)
                        contacts.refresh()
                        ccd = CcdBackend(contacts, node_box_radius=1)
                    params = {"use_colored_ccd_guess": not level, "use_colored_ccd_guess_level_based": level,
                              "colored_ccd_guess_iters": 1, "colored_ccd_guess_fraction": .97}
                    policy = InitialGuess(ccd, detector, params=params, use_group_queries=accelerated)
                    policy.apply(q, wp.clone(q), velocity, wp.empty_like(q), .01, params)
                    ccd.check()
                    graph = policy.graph
                    metadata = graph.metadata.numpy()
                    nnz = int(graph.rows.numpy()[-1])
                    results.append(dict(q=q.numpy(), target=policy.targets.numpy(), metadata=metadata,
                        colors=graph.colors.numpy(), trial_order=graph.trial_order.numpy(), rows=graph.rows.numpy(),
                        adjacency=graph.adjacency.numpy()[:nnz], levels=graph.levels.numpy() if level else np.empty(0),
                        groups=graph.groups.numpy() if level else np.empty(0),
                        order=graph.order.numpy()[:int(metadata[2])] if level else np.empty(0),
                        offsets=graph.offsets.numpy()[:int(metadata[1]) + 1] if level else np.empty(0),
                        movement=graph.movement.numpy() if level else np.empty(0)))
                for key in results[0]:
                    np.testing.assert_array_equal(results[1][key], results[0][key], err_msg=key)

    def test_actual_frame_states_match_all_pair_predictor(self):
        for device in self.devices:
            results = []
            for accelerated in (False, True):
                payload = point_over_triangle(height=.02, velocity=(.1, 0, -.5), iterations=3)
                payload["params"].update(substeps=3, dt=.05/3, exact_computation_fallback=False,
                    use_colored_ccd_guess_level_based=True, colored_ccd_guess_iters=1,
                    colored_ccd_guess_fraction=.97)
                model = create_model(deepcopy(payload), device)
                solver = SolverIPC(model, allow_cpu=True)
                if not accelerated:
                    solver.ccd_backend._initial_guess = InitialGuess(solver.ccd_backend,
                        solver.trimesh_collision_detector, params=solver.params, use_group_queries=False)
                frames = FrameGraph(model, solver, graphs=True)
                for _ in range(3):
                    frames.advance()
                results.append((frames.state.particle_q.numpy(), frames.state.particle_qd.numpy()))
            for actual, expected in zip(results[1], results[0]):
                np.testing.assert_array_equal(actual, expected)

    def test_device_epoch_rejects_replayed_stale_view(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA device")
        q, model, contacts = fixture(wp.get_device("cuda:0"))
        bank = QueryGroups(contacts)
        bank.build(wp.array([2, 1, 1, -1, 0, 0, 0], int, device=q.device))
        current = wp.array([2], int, device=q.device)
        view = bank.view(current)
        start = wp.clone(q)
        proposal = q.numpy().copy()
        proposal[0, 2] = .09
        proposal = wp.array(proposal, wp.vec3, device=q.device)
        backend = CcdBackend(contacts, node_box_radius=1)
        backend.accept(q, proposal, -1, .01, query_view=view)
        with wp.ScopedCapture(device=q.device) as capture:
            wp.copy(q, start)
            backend.accept(q, proposal, -1, .01, query_view=view)
        contacts.refresh()
        wp.capture_launch(capture.graph)
        with self.assertRaises(RuntimeError):
            backend.check()
        np.testing.assert_array_equal(q.numpy(), start.numpy())

    def test_capture_rebuilds_each_group_map_and_epoch_on_every_replay(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA device")
        q, model, contacts = fixture(wp.get_device("cuda:0"))
        bank = QueryGroups(contacts)
        groups = [wp.array(values, int, device=q.device) for values in
                  ([2, 1, 1, -1, 0, 0, 0], [1, 0, 0, -1, 2, 2, 2])]
        counters = [wp.array([value], int, device=q.device) for value in (2, 1)]
        start = wp.clone(q)
        proposals = []
        for value in (.09, .08):
            target = q.numpy().copy()
            target[0, 2] = value
            proposals.append(wp.array(target, wp.vec3, device=q.device))
        backend = CcdBackend(contacts, node_box_radius=1)
        for mapping, current, proposal in zip(groups, counters, proposals):
            bank.build(mapping)
            backend.accept(q, proposal, -1, .01, query_view=bank.view(current))
        expected = q.numpy().copy()
        with wp.ScopedCapture(device=q.device) as capture:
            contacts.refresh()
            wp.copy(q, start)
            for mapping, current, proposal in zip(groups, counters, proposals):
                bank.build(mapping)
                backend.accept(q, proposal, -1, .01, query_view=bank.view(current))
        for _ in range(2):
            wp.capture_launch(capture.graph)
            backend.check()
            np.testing.assert_array_equal(q.numpy(), expected)

    def test_independent_colored_predictor_keeps_unrelated_vertices_full_target(self):
        params = {"use_colored_ccd_guess": True, "colored_ccd_guess_iters": 1,
                  "colored_ccd_guess_fraction": 1.0}
        for device in self.devices:
            results = []
            for independent in (None, False, True):
                q, model, contacts, detector, velocity = independent_predictor_fixture(device)
                ccd = CcdBackend(contacts, node_box_radius=1)
                options = {} if independent is None else {"independent_colors": independent}
                policy = InitialGuess(ccd, detector, params=params, use_group_queries=True, **options)
                self.assertEqual(policy.independent_colors, independent is True)
                initial = q.numpy().copy()
                policy.apply(q, wp.clone(q), velocity, wp.empty_like(q), .01, params)
                ccd.check()
                colors = policy.graph.colors.numpy()
                self.assertEqual(colors[0], colors[4], "Fixture must share a predictor color")
                actual, target = q.numpy(), policy.targets.numpy()
                self.assertGreater(float(actual[0, 2]), 0)
                self.assertLess(float(actual[0, 2]), float(initial[0, 2]))
                np.testing.assert_array_equal(actual[1:4], initial[1:4])
                if independent:
                    # x^n + dt*v is the exact represented target, unaffected
                    # by the unrelated point's collision with the fixed face.
                    np.testing.assert_array_equal(actual[4], target[4])
                    np.testing.assert_allclose(actual[4], initial[4] + np.float32(.01) * velocity.numpy()[4], rtol=0, atol=0)
                else:
                    self.assertFalse(np.array_equal(actual[4], target[4]))
                results.append(actual)
            np.testing.assert_array_equal(results[0], results[1], "Generic default dispatch must stay unchanged")

    def test_independent_predictor_requires_grouped_original_mode_and_solver_opt_in(self):
        params = {"use_colored_ccd_guess": True, "colored_ccd_guess_iters": 1,
                  "colored_ccd_guess_fraction": 1.0}
        for device in self.devices:
            q, model, contacts, detector, velocity = independent_predictor_fixture(device)
            ccd = CcdBackend(contacts, node_box_radius=1)
            with self.assertRaisesRegex(ValueError, "complete grouped"):
                InitialGuess(ccd, detector, params=params, independent_colors=True)
            policy = InitialGuess(ccd, detector, params=params, use_group_queries=True, independent_colors=True)
            initial = q.numpy().copy()
            for mode in ({"use_ccd_guess": True}, {"use_colored_ccd_guess_level_based": True}):
                wrong = {**mode, "colored_ccd_guess_iters": 1, "colored_ccd_guess_fraction": 1.0}
                with self.assertRaisesRegex(ValueError, "original colored CCD mode"):
                    policy.apply(q, wp.clone(q), velocity, wp.empty_like(q), .01, wrong)
                np.testing.assert_array_equal(q.numpy(), initial)
            for mode in ("colored", "level", "global"):
                payload = point_over_triangle(height=.1, velocity=(.1, 0, 0), iterations=0)
                payload["params"].update(use_ccd_guess=mode == "global", use_colored_ccd_guess=mode == "colored",
                    use_colored_ccd_guess_level_based=mode == "level", colored_ccd_guess_iters=1,
                    colored_ccd_guess_fraction=1.0)
                solver = SolverIPC(create_model(payload, device), allow_cpu=True)
                self.assertEqual(solver.ccd_backend._initial_guess.independent_colors, mode == "colored")

    def test_independent_predictor_capture_replay_matches_eager_and_full_free_target(self):
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("No CUDA device")
        params = {"use_colored_ccd_guess": True, "colored_ccd_guess_iters": 1,
                  "colored_ccd_guess_fraction": 1.0}
        for device in devices:
            q, model, contacts, detector, velocity = independent_predictor_fixture(device)
            ccd = CcdBackend(contacts, node_box_radius=1)
            policy = InitialGuess(ccd, detector, params=params, use_group_queries=True, independent_colors=True)
            start, previous, inertia = wp.clone(q), wp.clone(q), wp.empty_like(q)

            def run():
                wp.copy(q, start)
                policy.apply(q, previous, velocity, inertia, .01, params)

            run()
            with wp.ScopedCapture(device=device) as capture:
                run()
            for free_velocity in ([30, -10, 20], [-40, 15, -5]):
                values = velocity.numpy(); values[4] = free_velocity; velocity.assign(values)
                run()
                ccd.check()
                expected = q.numpy().copy()
                wp.capture_launch(capture.graph)
                ccd.check()
                actual = q.numpy()
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(actual[4], policy.targets.numpy()[4])
                self.assertGreater(float(actual[0, 2]), 0)


if __name__ == "__main__":
    unittest.main()
