"""Conservative adaptive-box queries and dynamic contact assembly checks."""

from fractions import Fraction
from types import SimpleNamespace
import unittest

import numpy as np
import warp as wp

from gpu.contact import ContactBackend
from gpu.contact_original import ContactBackendIPC
from gpu.movement_box_queries import build_nodes
from gpu.query_groups import QueryGroups
from newton._src.geometry.tri_mesh_collision import build_tri_mesh_collision_info


def fixture(device, points, triangles, edges, *, capacity=64, worlds=None, owners=None, surface=None):
    points = np.asarray(points, np.float32).reshape(-1, 3)
    triangles = np.asarray(triangles, np.int32).reshape(-1, 3)
    edges = np.asarray(edges, np.int32).reshape(-1, 4)
    n, nt, ne = len(points), len(triangles), len(edges)
    q = wp.array(points, wp.vec3, device=device)
    model = SimpleNamespace(device=device, particle_count=n, tri_count=nt, edge_count=ne,
        body_count=0, particle_q=q, tri_indices=wp.array(triangles, int, device=device),
        edge_indices=wp.array(edges, int, device=device), particle_world=wp.array(
            np.zeros(n, np.int32) if worlds is None else worlds, int, device=device),
        particle_colors=wp.array(np.arange(n, dtype=np.int32), int, device=device), particle_flags=wp.ones(n, int, device=device),
        particle_mass=wp.ones(n, float, device=device))
    info = build_tri_mesh_collision_info(n, nt, ne, vertex_collision_buffer_pre_alloc=capacity,
                                        edge_collision_buffer_pre_alloc=capacity, device=device)
    backend = ContactBackend(model, info, d_hat=.01, k_barrier=100, track_query_epoch=True,
        node_body=None if owners is None else wp.array(owners, int, device=device))
    if surface is not None:
        backend.surface.assign(surface)
    lo_t = wp.array(points[triangles].min(axis=1), wp.vec3, device=device) if nt else wp.empty(0, wp.vec3, device=device)
    hi_t = wp.array(points[triangles].max(axis=1), wp.vec3, device=device) if nt else wp.empty(0, wp.vec3, device=device)
    lo_e = wp.array(points[edges[:, 2:]].min(axis=1), wp.vec3, device=device) if ne else wp.empty(0, wp.vec3, device=device)
    hi_e = wp.array(points[edges[:, 2:]].max(axis=1), wp.vec3, device=device) if ne else wp.empty(0, wp.vec3, device=device)
    detector = SimpleNamespace(lower_bounds_tris=lo_t, upper_bounds_tris=hi_t,
        lower_bounds_edges=lo_e, upper_bounds_edges=hi_e,
        bvh_tris=wp.Bvh(lo_t, hi_t) if nt else None, bvh_edges=wp.Bvh(lo_e, hi_e) if ne else None)
    return model, backend, detector


def pairs(backend):
    info, model = backend.info, backend.model
    vt, ee = set(), set()
    ids = backend.candidate_ids.numpy()[:int(backend.candidate_count.numpy()[0])]
    vertex_rows, edge_rows = info.vertex_colliding_triangles.numpy(), info.edge_colliding_edges.numpy()
    for encoded in ids:
        if encoded >= 0:
            vt.add(tuple(int(x) for x in vertex_rows[2 * encoded:2 * encoded + 2]))
        else:
            slot = -int(encoded) - 1
            ee.add(tuple(int(x) for x in edge_rows[2 * slot:2 * slot + 2]))
    return vt, ee


class MovementBoxQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_bounds_enclose_exact_binary_inputs_and_full_reach(self):
        rng = np.random.default_rng(3812)
        q = np.float32(rng.normal(size=(64, 3)) * 10.0 ** rng.uniform(-8, 8, (64, 1)))
        radii = 10.0 ** rng.uniform(-9, 4, 64)
        # Include cancellation, F32 halfway radii, signed zero, and tiny reach.
        q[:3] = [[1, -1, 0], [2**20, -(2**20), -0.0], [2**-20, -(2**-20), 0]]
        radii[:3] = [1.0, 2**-4 + 2**-40, np.nextafter(1e-10, np.inf)]
        reach = np.nextafter(.01, 0.0)
        for device in self.devices:
            lower, upper, qlo, qhi = [wp.empty(len(q), wp.vec3, device=device) for _ in range(4)]
            error = wp.zeros(1, int, device=device)
            wp.launch(build_nodes, len(q), [wp.array(q, wp.vec3, device=device),
                wp.array(radii, wp.float64, device=device), reach, lower, upper, qlo, qhi, error], device=device)
            self.assertEqual(int(error.numpy()[0]), 0)
            lo, hi, expanded_lo, expanded_hi = [x.numpy() for x in (lower, upper, qlo, qhi)]
            for i in range(len(q)):
                r, h = Fraction(float(radii[i])), Fraction(reach)
                for k in range(3):
                    center = Fraction(float(q[i, k]))
                    self.assertLessEqual(Fraction(float(lo[i, k])), center - r)
                    self.assertGreaterEqual(Fraction(float(hi[i, k])), center + r)
                    self.assertLessEqual(Fraction(float(expanded_lo[i, k])), Fraction(float(lo[i, k])) - h)
                    self.assertGreaterEqual(Fraction(float(expanded_hi[i, k])), Fraction(float(hi[i, k])) + h)

    def test_complete_box_pairs_world_body_surface_and_incidence_filters(self):
        rng = np.random.default_rng(9911)
        q = np.float32(rng.uniform(-1, 1, (24, 3)))
        triangles = np.asarray([rng.choice(len(q), 3, replace=False) for _ in range(12)], np.int32)
        edges = np.asarray([[-1, -1, *rng.choice(len(q), 2, replace=False)] for _ in range(14)], np.int32)
        radii = rng.uniform(.02, .65, len(q))
        worlds = rng.integers(-1, 2, len(q), dtype=np.int32)
        owners = np.asarray([0] * 8 + [-1] * 8 + [1] * 8, np.int32)
        surface = np.ones(len(q), np.int32)
        surface[::4] = 0
        reach = .025
        lo, hi = q.astype(np.float64) - radii[:, None], q.astype(np.float64) + radii[:, None]
        overlap = lambda a, b, c, d: bool(np.all(a <= d) and np.all(c <= b))
        same_world = lambda a, b: worlds[a] < 0 or worlds[b] < 0 or worlds[a] == worlds[b]
        same_body = lambda nodes: owners[nodes[0]] >= 0 and np.all(owners[nodes] == owners[nodes[0]])
        expected_vt, expected_ee = set(), set()
        for vertex in range(len(q)):
            for triangle, nodes in enumerate(triangles):
                if (surface[vertex] and vertex not in nodes and same_world(vertex, nodes[0])
                        and not same_body(np.r_[vertex, nodes])
                        and overlap(lo[vertex] - reach, hi[vertex] + reach, lo[nodes].min(axis=0), hi[nodes].max(axis=0))):
                    expected_vt.add((vertex, triangle))
        for edge, first in enumerate(edges[:, 2:]):
            for other in range(edge + 1, len(edges)):
                second = edges[other, 2:]
                if (not set(first).intersection(second) and same_world(first[0], second[0])
                        and not same_body(np.r_[first, second])
                        and overlap(lo[first].min(axis=0) - reach, hi[first].max(axis=0) + reach,
                                    lo[second].min(axis=0), hi[second].max(axis=0))):
                    expected_ee.add((edge, other))
        for device in self.devices:
            model, backend, detector = fixture(device, q, triangles, edges, worlds=worlds, owners=owners, surface=surface)
            backend.detect_boxes(detector, model.particle_q, wp.array(radii, wp.float64, device=device), reach)
            backend.check()
            actual_vt, actual_ee = pairs(backend)
            self.assertEqual(actual_vt, expected_vt)
            self.assertEqual(actual_ee, expected_ee)

    def test_far_current_primitives_can_meet_inside_asymmetric_boxes(self):
        q = [[0, 0, 1], [-1, -1, 0], [1, -1, 0], [0, 1, 0]]
        for device in self.devices:
            model, backend, detector = fixture(device, q, [[1, 2, 3]], [])
            radii = wp.array([.01, 1.1, 1.1, 1.1], wp.float64, device=device)
            backend.detect_boxes(detector, model.particle_q, radii, 0.0)
            backend.check()
            self.assertEqual(pairs(backend)[0], {(0, 0)})
            radii.assign([.01] * 4)
            backend.detect_boxes(detector, model.particle_q, radii, 0.0)
            backend.check()
            self.assertEqual(pairs(backend)[0], set())

    def test_invalid_data_and_candidate_overflow_fail_closed(self):
        for device in self.devices:
            for invalid in (np.nan, np.inf, -1.0, 0.0, 1.0e-10, 1.0e300):
                model, backend, detector = fixture(device, [[0, 0, .001], [-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[1, 2, 3]], [])
                radii = wp.array([invalid, .01, .01, .01], wp.float64, device=device)
                backend.detect_boxes(detector, model.particle_q, radii, .01)
                with self.assertRaisesRegex(RuntimeError, "invalid represented CCD motion"):
                    backend.check()
                self.assertTrue(np.all(np.isfinite(detector.lower_bounds_tris.numpy())))
                self.assertEqual(int(backend.candidate_count.numpy()[0]), 0)
            q = [[0, 0, .001], [-1, -1, 0], [1, -1, 0], [0, 1, 0], [-1, -1, -.001], [1, -1, -.001], [0, 1, -.001]]
            model, backend, detector = fixture(device, q, [[1, 2, 3], [4, 5, 6]], [], capacity=1)
            backend.detect_boxes(detector, model.particle_q, wp.full(len(q), .01, wp.float64, device=device), .01)
            with self.assertRaisesRegex(RuntimeError, "overflow"):
                backend.check()

    def test_cuda_capture_replays_changed_box_radii_without_reallocation(self):
        for device in wp.get_cuda_devices():
            model, backend, detector = fixture(device, [[0, 0, 1], [-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[1, 2, 3]], [])
            radii = wp.full(4, .01, wp.float64, device=device)
            backend.detect_boxes(detector, model.particle_q, radii, .01)
            with wp.ScopedCapture(device=device) as capture:
                backend.detect_boxes(detector, model.particle_q, radii, .01)
            wp.capture_launch(capture.graph)
            backend.check()
            self.assertEqual(pairs(backend)[0], set())
            radii.assign([.01, 1.1, 1.1, 1.1])
            wp.capture_launch(capture.graph)
            backend.check()
            self.assertEqual(pairs(backend)[0], {(0, 0)})

    def test_grouped_assembly_preserves_each_selected_role_and_friction(self):
        from .test_collision import CollisionFixture
        for device in self.devices:
            for exact, friction in ((True, .2), (True, 0.0), (False, 0.0)):
                q, model, info, unused = CollisionFixture().fixture(device, friction=friction)
                backend = ContactBackendIPC(model, info, d_hat=.005, k_barrier=100, friction=friction,
                    exact_geometry=exact, track_query_epoch=True)
                backend.refresh()
                previous = wp.clone(q)
                history = previous.numpy()
                history[0, 0] -= .001
                previous.assign(history)
                groups = QueryGroups(backend)
                groups.build(model.particle_colors)
                current = wp.zeros(1, int, device=device)
                view = groups.view(current)
                for selected in range(4):
                    current.fill_(selected)
                    outputs = []
                    for grouped in (False, True):
                        force = wp.zeros(4, wp.vec3, device=device)
                        hessian = wp.zeros(4, wp.mat33, device=device)
                        kwargs = dict(colors=model.particle_colors, current_color=current, query_view=view) if grouped else {}
                        backend.accumulate(q, previous, selected, force, hessian, .01, **kwargs)
                        backend.check()
                        outputs.append((force.numpy(), hessian.numpy()))
                    for ordinary, grouped in zip(*outputs):
                        self.assertEqual(ordinary.tobytes(), grouped.tobytes())
                backend.refresh()
                with self.assertRaisesRegex(ValueError, "Rebuild grouped"):
                    backend.accumulate(q, previous, 0, force, hessian, .01,
                        colors=model.particle_colors, current_color=current, query_view=view)


if __name__ == "__main__":
    unittest.main()
