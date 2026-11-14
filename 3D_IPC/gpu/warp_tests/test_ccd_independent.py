"""Acceptance checks for per-node CCD in contact-independent GS batches.

The geometric assertions use a point crossing a stationary plane or a single
edge endpoint crossing another edge. The unrelated point has no candidate,
so it must retain its entire represented proposal when a collision is clipped.
"""

from types import SimpleNamespace
import unittest

import numpy as np
import warp as wp

from newton._src.geometry.tri_mesh_collision import build_tri_mesh_collision_info

from gpu.ccd import CcdBackend
from gpu.contact import ContactBackend
from gpu.query_groups import QueryGroups


@wp.kernel
def _advance_epoch(epoch: wp.array[wp.uint64]):
    epoch[0] += wp.uint64(1)


def fixture(device, *, edge=False, plane=0.0, height=.1, groups=None):
    device = wp.get_device(device)
    if edge:
        points = [[-1, 0, .1], [1, 0, .1], [0, -1, -.1], [0, 1, -.1], [3, 2, .2]]
        triangles, edges = [], [[-1, -1, 0, 1], [-1, -1, 2, 3]]
    else:
        points = [[0, 0, plane + height], [-1, -1, plane], [1, -1, plane],
                  [0, 1, plane], [3, 2, plane + .2]]
        triangles, edges = [[1, 2, 3]], []
    q = wp.array(np.asarray(points, np.float32), wp.vec3, device=device)
    groups = [0, 1, 2, 3, 0] if groups is None else groups
    colors = wp.array(groups, int, device=device)
    model = SimpleNamespace(device=device, particle_count=5, tri_count=len(triangles),
        edge_count=len(edges), body_count=0, tet_count=0, particle_q=q,
        tri_indices=wp.array(np.asarray(triangles, np.int32).reshape(-1, 3), int, device=device),
        edge_indices=wp.array(np.asarray(edges, np.int32).reshape(-1, 4), int, device=device),
        particle_colors=colors, particle_flags=wp.ones(5, int, device=device),
        particle_mass=wp.ones(5, float, device=device), particle_world=wp.zeros(5, int, device=device))
    info = build_tri_mesh_collision_info(5, len(triangles), len(edges),
        vertex_collision_buffer_pre_alloc=2, edge_collision_buffer_pre_alloc=2, device=device)
    if edge:
        info.edge_colliding_edges.assign([0, 1, 0, 0, 1, 0, 0, 0])
        info.edge_colliding_edges_count.assign([1, 1])
    else:
        values = np.full(info.vertex_colliding_triangles.shape[0], -1, np.int32)
        values[:2] = [0, 0]
        info.vertex_colliding_triangles.assign(values)
        info.vertex_colliding_triangles_count.assign([1, 0, 0, 0, 0])
    contacts = ContactBackend(model, info, d_hat=.005, k_barrier=100, track_query_epoch=True)
    contacts.refresh()
    bank = QueryGroups(contacts)
    bank.build(colors)
    current = wp.zeros(1, int, device=device)
    return SimpleNamespace(q=q, model=model, contacts=contacts, bank=bank, colors=colors,
        current=current, view=bank.view(current), start=q.numpy().copy())


def accept(data, backend, proposal, **options):
    return backend.accept(data.q, proposal, -1, .01,
        colors=data.colors, current_color=data.current, query_view=data.view,
        independent=True, **options)


class IndependentCcdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_colliding_corner_is_clipped_without_throttling_unrelated_vertex(self):
        for device in self.devices:
            for exact, bound in ((False, False), (True, False), (True, True)):
                with self.subTest(device=str(device), exact=exact, bound=bound):
                    data = fixture(device)
                    proposed = data.start.copy()
                    proposed[0, 2] = -.1
                    proposed[4] += np.array([.3, -.1, .2], np.float32)
                    backend = CcdBackend(data.contacts, node_box_radius=1,
                        exact_geometry=exact, exact_step_bound=bound)
                    accept(data, backend, wp.array(proposed, wp.vec3, device=device), bounded=False)
                    backend.check()
                    actual = data.q.numpy()
                    self.assertGreater(float(actual[0, 2]), 0)
                    self.assertLess(float(actual[0, 2]), float(data.start[0, 2]))
                    np.testing.assert_array_equal(actual[4], proposed[4])
                    np.testing.assert_array_equal(actual[1:4], data.start[1:4])
                    self.assertGreaterEqual(int(backend.attempts.numpy()[0]), 2)
                    self.assertEqual(int(backend.verified.numpy()[0]), 1)

    def test_edge_endpoint_fraction_is_reduced_only_onto_that_endpoint(self):
        for device in self.devices:
            data = fixture(device, edge=True)
            proposed = data.start.copy()
            proposed[0, 2] = -1
            proposed[4, 0] += .4
            backend = CcdBackend(data.contacts, node_box_radius=2, exact_geometry=False)
            accept(data, backend, wp.array(proposed, wp.vec3, device=device), bounded=False)
            backend.check()
            actual = data.q.numpy()
            # At x=0 the moving first edge has the mean of its endpoint Zs.
            # It starts above the second edge and must remain above it.
            self.assertGreater(float(.5 * (actual[0, 2] + actual[1, 2])), float(actual[2, 2]))
            self.assertLess(float(actual[0, 2]), float(data.start[0, 2]))
            np.testing.assert_array_equal(actual[1:4], data.start[1:4])
            np.testing.assert_array_equal(actual[4], proposed[4])

    def test_two_moving_corners_in_one_candidate_fail_closed(self):
        for device in self.devices:
            for edge in (False, True):
                with self.subTest(device=str(device), edge=edge):
                    data = fixture(device, edge=edge, groups=[0, 0, 2, 3, 0])
                    proposed = data.start.copy()
                    proposed[0, 2] -= .2
                    proposed[1, 0] += .1
                    proposed[4, 0] += .4
                    backend = CcdBackend(data.contacts, node_box_radius=2)
                    accept(data, backend, wp.array(proposed, wp.vec3, device=device), bounded=False)
                    self.assertEqual(int(data.contacts.error.numpy()[0]), 205)
                    with self.assertRaises(RuntimeError):
                        backend.check()
                    np.testing.assert_array_equal(data.q.numpy(), data.start)

    def test_heterogeneous_radii_round_inward_and_ignore_other_colors_stale_proposals(self):
        for device in self.devices:
            results = []
            for grouped in (False, True):
                with self.subTest(device=str(device), grouped=grouped):
                    data = fixture(device)
                    budgets = np.array([.1, .001, .002, .003, .03141592653], np.float64)
                    radii = wp.array(budgets, wp.float64, device=device)
                    proposed = np.full_like(data.start, np.nan)
                    proposed[[0, 4]] = [[100, -100, 100], [-100, 100, 100]]
                    backend = CcdBackend(data.contacts, node_box_radius=1)
                    proposal = wp.array(proposed, wp.vec3, device=device)
                    if grouped:
                        accept(data, backend, proposal, radii=radii)
                    else:
                        backend.accept(data.q, proposal, 0, .01, radii=radii)
                    backend.check()
                    actual = data.q.numpy()
                    changed = [0, 4]
                    lower = data.start[changed].astype(np.float64) - budgets[changed, None] + 1e-10
                    upper = data.start[changed].astype(np.float64) + budgets[changed, None] - 1e-10
                    self.assertTrue(np.all(actual[changed].astype(np.float64) >= lower))
                    self.assertTrue(np.all(actual[changed].astype(np.float64) <= upper))
                    self.assertGreater(float(np.linalg.norm(actual[0] - data.start[0])), .1)
                    self.assertLess(float(np.linalg.norm(actual[4] - data.start[4])), .06)
                    np.testing.assert_array_equal(actual[1:4], data.start[1:4])
                    results.append(actual)
            np.testing.assert_array_equal(results[1], results[0])

    def test_nonfinite_per_node_radius_cannot_disable_movement_bound(self):
        for device in self.devices:
            data = fixture(device)
            proposed = data.start.copy()
            proposed[4, 0] += .1
            radii = wp.array([.1, .1, .1, .1, np.inf], wp.float64, device=device)
            backend = CcdBackend(data.contacts, node_box_radius=1)
            accept(data, backend, wp.array(proposed, wp.vec3, device=device), radii=radii)
            self.assertEqual(int(data.contacts.error.numpy()[0]), 202)
            with self.assertRaises(RuntimeError):
                backend.check()
            np.testing.assert_array_equal(data.q.numpy(), data.start)

    def test_rounding_near_a_large_plane_is_reproved_before_commit(self):
        for device in self.devices:
            plane = np.float32(100000)
            above = np.nextafter(plane, np.float32(np.inf))
            data = fixture(device, plane=float(plane), height=float(above - plane))
            proposed = data.start.copy()
            proposed[0, 2] = np.nextafter(plane, np.float32(-np.inf))
            proposed[4, 0] += .25
            backend = CcdBackend(data.contacts, node_box_radius=1)
            accept(data, backend, wp.array(proposed, wp.vec3, device=device), bounded=False)
            backend.check()
            self.assertGreater(float(data.q.numpy()[0, 2]), float(plane))
            np.testing.assert_array_equal(data.q.numpy()[4], proposed[4])
            self.assertGreaterEqual(int(backend.attempts.numpy()[0]), 2)
            self.assertEqual(int(backend.verified.numpy()[0]), 1)

    def test_host_and_device_stale_epochs_cannot_commit_independent_updates(self):
        for device in self.devices:
            for host_stale in (False, True):
                with self.subTest(device=str(device), host_stale=host_stale):
                    data = fixture(device)
                    proposed = data.start.copy()
                    proposed[0, 2] = .09
                    backend = CcdBackend(data.contacts, node_box_radius=1)
                    if host_stale:
                        data.bank.build(data.colors)
                        with self.assertRaisesRegex(ValueError, "epoch"):
                            accept(data, backend, wp.array(proposed, wp.vec3, device=device))
                    else:
                        wp.launch(_advance_epoch, 1, [data.contacts.query_epoch], device=device)
                        accept(data, backend, wp.array(proposed, wp.vec3, device=device))
                        with self.assertRaises(RuntimeError):
                            backend.check()
                    np.testing.assert_array_equal(data.q.numpy(), data.start)

    def test_cuda_capture_rebuilds_epoch_and_uses_new_proposals_on_replay(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA device")
        for device in wp.get_cuda_devices():
            data = fixture(device)
            start = wp.clone(data.q)
            proposed = data.start.copy()
            proposed[0, 2] = -.1
            proposed[4, 0] += .25
            proposal = wp.array(proposed, wp.vec3, device=device)
            backend = CcdBackend(data.contacts, node_box_radius=1)
            accept(data, backend, proposal, bounded=False)  # Compile before capture.
            backend.check()
            with wp.ScopedCapture(device=device) as capture:
                wp.copy(data.q, start)
                data.contacts.refresh()
                data.bank.build(data.colors)
                data.view = data.bank.view(data.current)
                accept(data, backend, proposal, bounded=False)
            previous_epoch = int(data.contacts.query_epoch.numpy()[0])
            for displacement in (.3, .4):
                proposed[0, 2] = -displacement
                proposed[4, 0] = data.start[4, 0] + displacement
                proposal.assign(proposed)
                wp.capture_launch(capture.graph)
                backend.check()
                actual = data.q.numpy()
                self.assertGreater(float(actual[0, 2]), 0)
                self.assertLess(float(actual[0, 2]), float(data.start[0, 2]))
                np.testing.assert_array_equal(actual[4], proposed[4])
                epoch = int(data.contacts.query_epoch.numpy()[0])
                self.assertGreater(epoch, previous_epoch)
                previous_epoch = epoch

    def test_cuda_replayed_stale_epoch_fails_before_commit(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA device")
        for device in wp.get_cuda_devices():
            data = fixture(device)
            start = wp.clone(data.q)
            proposed = data.start.copy()
            proposed[0, 2] = .09
            proposal = wp.array(proposed, wp.vec3, device=device)
            backend = CcdBackend(data.contacts, node_box_radius=1)
            accept(data, backend, proposal)
            backend.check()
            with wp.ScopedCapture(device=device) as capture:
                wp.copy(data.q, start)
                accept(data, backend, proposal)
            wp.launch(_advance_epoch, 1, [data.contacts.query_epoch], device=device)
            wp.capture_launch(capture.graph)
            with self.assertRaises(RuntimeError):
                backend.check()
            np.testing.assert_array_equal(data.q.numpy(), data.start)


if __name__ == "__main__":
    unittest.main()
