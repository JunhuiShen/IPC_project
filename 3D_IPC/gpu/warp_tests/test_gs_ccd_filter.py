"""Production FP32 full-path certificates and resolver propagation.

Analytic crossings are rejected before clipping; certified full endpoints must
remain full steps when routed through the independent production resolver.
"""

from types import SimpleNamespace
import unittest

import numpy as np
import warp as wp

from newton._src.geometry.tri_mesh_collision import build_tri_mesh_collision_info

from gpu.ccd import CcdBackend, get_independent_resolver
from gpu.contact import ContactBackend
from gpu import ccd


@wp.func_native(r"""
    ipc_gpu::ccd32::LinearQuery q;
    q.vertex_face=face!=0;
    q.gap_floor=gap;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        q.start[i][k]=start.data[12*at+3*i+k];
        q.end[i][k]=end.data[12*at+3*i+k];
    }
    return ipc_gpu::ccd32::validate_linear(q)==ipc_gpu::ccd32::Status::Safe?1:0;
""")
def _filter(start: wp.array3d[wp.float32], end: wp.array3d[wp.float32], at: int,
            face: int, gap: float) -> int: ...

@wp.kernel(module="unique", enable_backward=False)
def _evaluate(start: wp.array3d[wp.float32], end: wp.array3d[wp.float32],
              face: wp.array[int], gap: wp.array[wp.float32], result: wp.array[int]):
    at = wp.tid()
    result[at] = _filter(start, end, at, face[at], gap[at])


ccd.configure_module(_evaluate, mode="single_vertex")


@wp.kernel(module="unique", enable_backward=False)
def _production(start: wp.array[wp.vec3], end: wp.array[wp.vec3],
                nodes: wp.array[wp.vec4i], kind: wp.array[int], result: wp.array[int]):
    at = wp.tid()
    result[at] = int(ccd.single_vertex_safe(start, end, nodes[at], kind[at]))


ccd.configure_module(_production, mode="single_vertex")


def production_floors(start):
    return np.full(len(start), 1e-8, np.float32)


class GsCcdFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def evaluate(self, device, start, end, face, gap=None):
        start, end = np.asarray(start, np.float32), np.asarray(end, np.float32)
        if gap is None:
            gap = production_floors(start)
        result = wp.empty(len(start), int, device=device)
        wp.launch(_evaluate, len(start), [wp.array(start, wp.float32, device=device),
            wp.array(end, wp.float32, device=device), wp.array(face, int, device=device),
            wp.array(gap, wp.float32, device=device), result], device=device)
        return result.numpy().astype(bool)

    def assert_resolver_full_steps(self, device, start, end, face):
        """Send certified full candidates through the independent FP32 resolver."""
        count = len(start)
        if not count:
            return
        n = 4 * count
        q = wp.array(np.asarray(start, np.float32).reshape(n, 3), wp.vec3, device=device)
        triangles = np.array([[4*i+1, 4*i+2, 4*i+3] for i in range(count)], np.int32) if face else np.empty((0, 3), np.int32)
        edges = np.empty((0, 4), np.int32) if face else np.array(
            [[-1, -1, 4*i+j, 4*i+j+1] for i in range(count) for j in (0, 2)], np.int32)
        model = SimpleNamespace(device=device, particle_count=n, tri_count=len(triangles),
            edge_count=len(edges), body_count=0, tet_count=0, particle_q=q,
            tri_indices=wp.array(triangles, int, device=device), edge_indices=wp.array(edges, int, device=device),
            particle_colors=wp.zeros(n, int, device=device), particle_flags=wp.ones(n, int, device=device),
            particle_mass=wp.ones(n, float, device=device), particle_world=wp.zeros(n, int, device=device))
        info = build_tri_mesh_collision_info(n, len(triangles), len(edges),
            vertex_collision_buffer_pre_alloc=1, edge_collision_buffer_pre_alloc=1, device=device)
        if face:
            values = np.full(info.vertex_colliding_triangles.shape[0], -1, np.int32)
            counts = np.zeros(n, np.int32)
            offsets = info.vertex_colliding_triangles_offsets.numpy()
            for i in range(count):
                values[2*offsets[4*i]:2*offsets[4*i]+2] = [4*i, i]
                counts[4*i] = 1
            info.vertex_colliding_triangles.assign(values)
            info.vertex_colliding_triangles_count.assign(counts)
        else:
            values = np.full(info.edge_colliding_edges.shape[0], -1, np.int32)
            offsets = info.edge_colliding_edges_offsets.numpy()
            for i in range(count):
                for first, second in ((2*i, 2*i+1), (2*i+1, 2*i)):
                    values[2*offsets[first]:2*offsets[first]+2] = [first, second]
            info.edge_colliding_edges.assign(values)
            info.edge_colliding_edges_count.fill_(1)
        contacts = ContactBackend(model, info, d_hat=.005, k_barrier=100)
        contacts.refresh()
        self.assertEqual(int(contacts.candidate_count.numpy()[0]), count)
        backend = CcdBackend(contacts, node_box_radius=1, exact_geometry=True)
        backend.candidate.assign(np.asarray(end, np.float32).reshape(n, 3))
        pending = np.zeros(backend.pending.shape[0], np.int32)
        pending[:count] = np.arange(count)
        backend.pending.assign(pending)
        backend.pending_count.fill_(count)
        wp.launch(get_independent_resolver(False), contacts.workers,
            [q, backend.candidate, contacts.candidate_ids, model.tri_indices, model.edge_indices,
             info, 1, 0, contacts.words, contacts.slots, contacts.workers,
             backend.pending, backend.pending_count, backend.node_fractions, backend.fraction, contacts.error],
            device=device, block_dim=1)
        contacts.check()
        np.testing.assert_array_equal(backend.node_fractions.numpy(), 1.0)
        self.assertEqual(float(backend.fraction.numpy()[0]), 1.0)

    def test_ordinary_pt_and_ee_proofs_retain_full_step(self):
        bases = (np.array([[.1, .1, .2], [-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32),
                 np.array([[-1, 0, .2], [1, 0, .2], [0, -1, 0], [0, 1, 0]], np.float32))
        for device in self.devices:
            for face, base in zip((True, False), bases):
                end = base.copy()
                end[0] += [.13, -.07, .1]
                accepted = self.evaluate(device, [base], [end], [int(face)])
                self.assertTrue(accepted[0])
                self.assert_resolver_full_steps(device, [base], [end], face)

    def test_production_wrapper_uses_represented_arrays_and_rejects_invalid_indices(self):
        points = np.array([[.1, .1, .2], [-1, -1, 0], [1, -1, 0], [0, 1, 0],
            [-1, 0, .2], [1, 0, .2], [0, -1, 0], [0, 1, 0]], np.float32)
        proposed = points.copy()
        proposed[[0, 4], 2] += .1
        corners = [[0, 1, 2, 3], [4, 5, 6, 7], [-1, 1, 2, 3], [0, 1, 2, 99], [0, 1, 2, 3]]
        for device in self.devices:
            result = wp.empty(5, int, device=device)
            wp.launch(_production, 5, [wp.array(points, wp.vec3, device=device),
                wp.array(proposed, wp.vec3, device=device), wp.array(corners, wp.vec4i, device=device),
                wp.array([0, 1, 0, 0, 2], int, device=device), result], device=device)
            np.testing.assert_array_equal(result.numpy(), [1, 1, 0, 0, 0])

    def test_random_rotated_scaled_all_corner_roles_preserve_certified_full_steps(self):
        rng = np.random.default_rng(908127)
        data = []
        for face in (True, False):
            starts, ends = [], []
            for i in range(64):
                rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                base = np.array([[.1, .1, .25], [-1, -1, 0], [1, -1, 0], [0, 1, 0]]) if face else np.array(
                    [[-1, 0, .25], [1, 0, .25], [0, -1, 0], [0, 1, 0]])
                scale = 2.0 ** (-30, -10, 0, 10, 30)[i % 5]
                translation = rng.uniform(-2, 2, 3) * scale
                start = ((base @ rotation.T) * scale + translation).astype(np.float32)
                end = start.copy()
                end[i % 4] += (rng.uniform(-.4, .4, 3) * scale).astype(np.float32)
                starts.append(start)
                ends.append(end)
            data.append((face, np.asarray(starts), np.asarray(ends)))
        for device in self.devices:
            for face, start, end in data:
                accepted = self.evaluate(device, start, end, [int(face)] * len(start))
                self.assertGreater(int(np.count_nonzero(accepted)), 12)
                self.assert_resolver_full_steps(device, start[accepted], end[accepted], face)

    def test_large_translation_and_subnormal_geometry_retain_conservative_policy(self):
        base = np.array([[.1, .1, .2], [-1, -1, 0], [1, -1, 0], [0, 1, 0]])
        translated = (4096 * base + [2.0**30, -2.0**29, 2.0**28]).astype(np.float32)
        end = translated.copy()
        end[0] += [512, -256, 512]
        tiny = (base * 2.0**-140).astype(np.float32)
        tiny_end = tiny.copy()
        tiny_end[0, 0] += np.float32(2.0**-145)
        for device in self.devices:
            accepted = self.evaluate(device, [translated, tiny], [end, tiny_end], [1, 1])
            np.testing.assert_array_equal(accepted, [True, False])
            self.assert_resolver_full_steps(device, [translated], [end], True)

    def test_collision_gap_floor_coplanarity_and_invalid_inputs_are_uncertain(self):
        base = np.array([[.1, .1, .2], [-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32).astype(np.float32)
        starts, ends, gaps = [], [], []

        def case(start, end, gap=1e-8):
            starts.append(start.copy())
            ends.append(end.copy())
            gaps.append(gap)

        end = base.copy(); end[0, 2] = -.2
        case(base, end)  # A plane crossing.
        end = base.copy(); end[0, 2] = 5e-9
        case(base, end)  # Final gap below the required floor.
        for z in (0.0,):
            start = base.copy(); start[0, 2] = np.float32(z)
            end = start.copy(); end[0, 2] = .3
            case(start, end, 1e-12 if z == 5e-11 else 1e-8)
        start = base.copy(); start[1:4, 2] = start[0, 2]
        end = start.copy(); end[0, 0] += .1
        case(start, end)  # Coplanar determinant is ambiguous.
        end = base.copy(); end[0, 2] = np.inf
        case(base, end)
        end = base.copy(); end[0, 2] = np.nan
        case(base, end)
        end = base.copy(); end[0, 2] = .3
        for gap in (0.0, -1.0, np.nan, np.inf):
            case(base, end, gap)
        for device in self.devices:
            accepted = self.evaluate(device, starts, ends, [1] * len(starts), gaps)
            np.testing.assert_array_equal(accepted, False)


if __name__ == "__main__":
    unittest.main()
