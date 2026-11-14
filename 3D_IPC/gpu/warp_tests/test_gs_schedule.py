"""Independent CPU greedy oracle and device epoch checks for GS scheduling."""

from collections import Counter
from types import SimpleNamespace
import unittest

import numpy as np
import warp as wp
import newton

from newton._src.geometry.tri_mesh_collision import build_tri_mesh_collision_info
from gpu import kernels
from gpu.gs import GsSchedule, make_dynamic_elasticity


TRIANGLES = [[1, 2, 3], [4, 5, 6]]
EDGES = [[1, 4, 2, 3], [-1, -1, 7, 8], [-1, -1, 9, 10]]
TETS = [[10, 11, 12, 13]]


def oracle(vertex_count, vt_vertex=0, include_edge=True):
    cliques = TRIANGLES + [EDGES[0]] + TETS + [[vt_vertex, *TRIANGLES[0]]]
    if include_edge:
        cliques += [[7, 8, 9, 10]]
    adjacency = [set() for _ in range(vertex_count)]
    for nodes in cliques:
        for a in nodes:
            adjacency[a].update(b for b in nodes if b != a)
    colors = []
    for vertex in range(vertex_count):
        used = {colors[neighbor] for neighbor in adjacency[vertex] if neighbor < vertex}
        color = 0
        while color in used:
            color += 1
        colors.append(color)
    return np.asarray(colors), adjacency


def fixture(device):
    n = 14
    model = SimpleNamespace(device=device, particle_count=n, tri_count=2, edge_count=3, tet_count=1,
        tri_indices=wp.array(TRIANGLES, int, device=device), edge_indices=wp.array(EDGES, int, device=device),
        tet_indices=wp.array(TETS, int, device=device))
    info = build_tri_mesh_collision_info(n, 2, 3, vertex_collision_buffer_pre_alloc=1,
                                         edge_collision_buffer_pre_alloc=1, device=device)
    vt = np.full(info.vertex_colliding_triangles.shape[0], -1, np.int32)
    vt[:2] = [0, 0]
    info.vertex_colliding_triangles.assign(vt)
    ee = np.full(info.edge_colliding_edges.shape[0], -1, np.int32)
    ee[:2] = [1, 2]
    info.edge_colliding_edges.assign(ee)
    capacity = info.vertex_colliding_triangles.shape[0] // 2 + info.edge_colliding_edges.shape[0] // 2
    contacts = SimpleNamespace(model=model, info=info, device=device, capacity=capacity,
        candidate_ids=wp.array([0, -1], int, device=device), candidate_count=wp.array([2], int, device=device),
        track_query_epoch=True, query_epoch=wp.array([1], wp.uint64, device=device), query_generation=1,
        fast_workers=128, error=wp.zeros(1, int, device=device))
    return contacts


def dynamic_arguments(original, order, offsets, current, budget=0):
    return [original[0], order, offsets, current, *original[2:-3], budget, *original[-3:]]


def point_material_fixture(device, n=257):
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    positions = np.column_stack((np.arange(n) * .01, np.arange(n) % 7 * .02,
                                 np.arange(n) % 11 * -.015)).astype(np.float32)
    for vertex, point in enumerate(positions):
        builder.add_particle(wp.vec3(point), wp.vec3(0), mass=1.0 + (vertex % 5) * .25)
    builder.color()
    model = builder.finalize(device=device)
    flags = model.particle_flags.numpy()
    flags[::13] = 0
    model.particle_flags.assign(flags)
    masses = model.particle_mass.numpy()
    masses[7::17] = 0
    model.particle_mass.assign(masses)
    targets = positions + np.asarray([.004, -.006, .003], np.float32)
    pins = np.zeros(n, np.int32)
    pins[::11] = 1
    force = np.column_stack((np.arange(n) % 3 * .007, np.arange(n) % 5 * -.003,
                             np.arange(n) % 7 * .002)).astype(np.float32)
    hessian = np.repeat(np.asarray([[[.4, .03, -.02], [.03, .3, .01], [-.02, .01, .2]]], np.float32), n, axis=0)
    args = [.01, wp.array(np.arange(n), int, device=device), wp.array(positions, wp.vec3, device=device),
        model.particle_q, model.particle_mass,
        wp.array(positions + np.asarray([.007, -.003, .005], np.float32), wp.vec3, device=device),
        model.particle_flags, model.tri_indices, model.tri_poses, model.tri_materials, model.tri_areas,
        model.edge_indices, model.edge_rest_angle, model.edge_rest_length, model.edge_bending_properties,
        model.tet_indices, model.tet_poses, model.tet_materials, model.soft_mesh_adjacency_device,
        wp.array(force, wp.vec3, device=device), wp.array(hessian, wp.mat33, device=device),
        wp.array(targets, wp.vec3, device=device), wp.array(pins, int, device=device), 7.0,
        wp.zeros(model.edge_count, float, device=device), wp.empty((0, 4), wp.vec3, device=device),
        wp.empty(0, float, device=device), .6, wp.zeros(n, wp.vec3, device=device),
        wp.full(n, wp.vec3(-123), wp.vec3, device=device), wp.zeros(1, int, device=device)]
    return model, args


@wp.kernel
def stamp(colors: wp.array[int], current: wp.array[int], seen: wp.array[int], visits: wp.array[int]):
    vertex = wp.tid()
    if colors[vertex] == current[0]:
        seen[vertex] = current[0]
        visits[vertex] += 1


class GsScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def assert_graph(self, schedule, expected_colors, expected_adjacency):
        schedule.validate_independence()
        self.assertEqual(int(schedule.contacts.error.numpy()[0]), 0)
        colors = schedule.colors.numpy()
        np.testing.assert_array_equal(colors, expected_colors)
        rows, adjacency = schedule.rows.numpy(), schedule.adjacency.numpy()
        order, offsets = schedule.order.numpy(), schedule.offsets.numpy()
        self.assertEqual(sorted(order.tolist()), list(range(schedule.n)))
        for vertex in range(schedule.n):
            np.testing.assert_array_equal(adjacency[int(rows[vertex]):int(rows[vertex + 1])],
                sorted(other for other in expected_adjacency[vertex] if other < vertex))
        for color in range(int(schedule.metadata.numpy()[0])):
            self.assertEqual(set(order[offsets[color]:offsets[color + 1]].tolist()),
                             set(np.flatnonzero(colors == color).tolist()))

    def test_union_includes_hinge_apex_tets_and_both_contact_kinds(self):
        expected, adjacency = oracle(14)
        for device in self.devices:
            with self.subTest(device=str(device)):
                contacts = fixture(device)
                schedule = GsSchedule(contacts)
                schedule.rebuild()
                self.assert_graph(schedule, expected, adjacency)
                self.assertNotEqual(int(expected[1]), int(expected[4]), "hinge apex-only coupling must be colored")
                ids, offsets = schedule.query_view.ids.numpy(), schedule.query_view.offsets.numpy()
                for color in range(14):
                    candidates = []
                    if color in expected[[0, 1, 2, 3]]:
                        candidates.append(0)
                    if color in expected[[7, 8, 9, 10]]:
                        candidates.append(-1)
                    self.assertEqual(Counter(ids[offsets[color]:offsets[color + 1]]), Counter(candidates))

    def test_new_box_epoch_replaces_old_contact_conflicts(self):
        for device in self.devices:
            contacts = fixture(device)
            schedule = GsSchedule(contacts)
            schedule.rebuild()
            old_view = schedule.query_view
            vt = contacts.info.vertex_colliding_triangles.numpy()
            vt[:2] = [13, 0]
            contacts.info.vertex_colliding_triangles.assign(vt)
            contacts.candidate_count.assign([1])
            contacts.query_generation += 1
            contacts.query_epoch.assign([2])
            schedule.rebuild()
            self.assert_graph(schedule, *oracle(14, vt_vertex=13, include_edge=False))
            with self.assertRaises(ValueError):
                old_view.validate(contacts)
            np.testing.assert_array_equal(schedule.query_view.epoch.numpy(), [2])

    def test_device_color_sweep_visits_each_vertex_once(self):
        for device in self.devices:
            contacts = fixture(device)
            schedule = GsSchedule(contacts)
            schedule.rebuild()
            seen = wp.full(14, -1, int, device=device)
            visits = wp.zeros(14, int, device=device)
            def body():
                wp.launch(stamp, 14, [schedule.colors, schedule.current_color, seen, visits], device=device)
            schedule.sweep(body)
            np.testing.assert_array_equal(seen.numpy(), oracle(14)[0])
            np.testing.assert_array_equal(visits.numpy(), np.ones(14, np.int32))

    def test_cuda_graph_rebuild_and_sweep_replay_device_data(self):
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA device unavailable")
        for device in devices:
            contacts = fixture(device)
            schedule = GsSchedule(contacts)
            seen = wp.full(14, -1, int, device=device)
            visits = wp.zeros(14, int, device=device)
            def body():
                wp.launch(stamp, 14, [schedule.colors, schedule.current_color, seen, visits], device=device)
            schedule.rebuild()
            schedule.sweep(body)
            wp.synchronize_device(device)
            with wp.ScopedCapture(device=device) as capture:
                schedule.rebuild()
                schedule.validate_independence()
                schedule.sweep(body)
            for vertex, include_edge in ((0, True), (13, False)):
                vt = contacts.info.vertex_colliding_triangles.numpy()
                vt[:2] = [vertex, 0]
                contacts.info.vertex_colliding_triangles.assign(vt)
                contacts.candidate_count.assign([2 if include_edge else 1])
                visits.zero_()
                wp.capture_launch(capture.graph)
                expected = oracle(14, vt_vertex=vertex, include_edge=include_edge)
                self.assert_graph(schedule, *expected)
                np.testing.assert_array_equal(seen.numpy(), expected[0])
                np.testing.assert_array_equal(visits.numpy(), np.ones(14, np.int32))

    def test_dynamic_material_selection_matches_existing_scalar_newton_update(self):
        # Existing independent material fixture includes finite pins, contact
        # force/Hessian inputs, damping, a three-vertex color and an inactive
        # middle slot. This catches indexing and half-warp packing mistakes.
        from .test_energies import Float32MaterialTests
        fixture_owner = Float32MaterialTests()
        for device in self.devices:
            _, reference, _ = fixture_owner._scalar_scene(device, inactive_middle=True)
            wp.launch(kernels.solve_elasticity_scalar, 3, reference, device=device)
            _, actual, _ = fixture_owner._scalar_scene(device, inactive_middle=True)
            dynamic_args = dynamic_arguments(actual, actual[1], wp.array([0, 3], int, device=device),
                                             wp.zeros(1, int, device=device))
            dynamic = make_dynamic_elasticity(True, False, device.is_cuda)
            wp.launch(dynamic, 64 if device.is_cuda else 3, dynamic_args,
                      device=device, block_dim=32 if device.is_cuda else 64)
            self.assertEqual(int(actual[-1].numpy()[0]), 0)
            np.testing.assert_allclose(actual[-2].numpy(), reference[-2].numpy(), rtol=3e-6, atol=3e-6)

    def test_dynamic_solid_and_mixed_materials_match_scalar_for_every_tet_role(self):
        # Exercise every rest-gradient role, signed/inverted tets, an inactive
        # solid node, a nonzero packed-color offset and an odd-size color. A
        # separate triangle in the mixed case exercises both material branches.
        from .test_energies import Float32MaterialTests
        fixture_owner = Float32MaterialTests()
        for device in self.devices:
            for mixed in (False, True):
                for inverted in (False, True):
                    for role in range(4):
                        with self.subTest(device=str(device), mixed=mixed, inverted=inverted, role=role):
                            ids = [role, 4 + role, 8 + role % 3]
                            _, reference = fixture_owner._tet_scene(device, mixed, True)
                            _, actual = fixture_owner._tet_scene(device, mixed, True)
                            reference[1] = wp.array(ids, int, device=device)
                            for arguments in (reference, actual):
                                if inverted:
                                    position = arguments[3].numpy()
                                    position[[3, 7], 2] *= -1
                                    arguments[3].assign(position)
                                    arguments[5].assign(position + np.asarray([.01, -.02, .005], np.float32))
                            wp.launch(kernels.solve_elasticity_scalar, 3, reference, device=device)
                            order = wp.array([10, 2, *ids, 7], int, device=device)
                            offsets = wp.array([0, 2, 5, 6], int, device=device)
                            current = wp.array([1], int, device=device)
                            dynamic_args = dynamic_arguments(actual, order, offsets, current)
                            dynamic = make_dynamic_elasticity(mixed, True, device.is_cuda)
                            wp.launch(dynamic, 64 if device.is_cuda else 3, dynamic_args,
                                      device=device, block_dim=32 if device.is_cuda else 64)
                            self.assertEqual(int(reference[-1].numpy()[0]), 0)
                            self.assertEqual(int(actual[-1].numpy()[0]), 0)
                            np.testing.assert_allclose(actual[-2].numpy(), reference[-2].numpy(),
                                                       rtol=3e-6, atol=3e-6)

    def test_persistent_pool_covers_empty_tiny_odd_and_large_groups(self):
        n = 257
        permutation = np.random.default_rng(18735).permutation(n).astype(np.int32)
        packed = np.concatenate(([-1, -1], permutation, [-1, -1])).astype(np.int32)
        initial_displacement = np.repeat(np.asarray([[.125, -.25, .5]], np.float32), n, axis=0)
        for device in self.devices:
            _, reference = point_material_fixture(device, n)
            _, actual = point_material_fixture(device, n)
            order = wp.array(packed, int, device=device)
            offsets = wp.array([0, 2, 2, len(packed)], int, device=device)
            current = wp.array([1], int, device=device)
            kernel = make_dynamic_elasticity(False, False, device.is_cuda)
            immutable = (actual[3].numpy().copy(), actual[19].numpy().copy(), actual[20].numpy().copy())
            for budget in (2, 8, 32):
                for count in (0, 1, 2, 3, 7, 8, 9, 31, 64, 128, n):
                    with self.subTest(device=str(device), budget=budget, count=count):
                        reference[1] = wp.array(permutation[:count], int, device=device)
                        offsets.assign([0, 2, 2 + count, len(packed)])
                        for arguments in (reference, actual):
                            arguments[-3].assign(initial_displacement)
                            arguments[-2].fill_(wp.vec3(-123))
                            arguments[-1].zero_()
                        if count:
                            wp.launch(kernels.solve_elasticity_scalar, count, reference, device=device)
                        wp.launch(kernel, budget * 16 if device.is_cuda else budget,
                            dynamic_arguments(actual, order, offsets, current, budget),
                            device=device, block_dim=32 if device.is_cuda else 64)
                        self.assertEqual(int(actual[-1].numpy()[0]), 0)
                        for candidate, expected in zip(actual[-3:-1], reference[-3:-1]):
                            np.testing.assert_allclose(candidate.numpy(), expected.numpy(), rtol=3e-6, atol=3e-6)
            for candidate, expected in zip((actual[3], actual[19], actual[20]), immutable):
                np.testing.assert_array_equal(candidate.numpy(), expected)

    def test_persistent_material_rounds_preserve_cloth_solid_and_mixed_updates(self):
        from .test_energies import Float32MaterialTests
        fixture_owner = Float32MaterialTests()
        for device in self.devices:
            for triangles, tets in ((True, False), (False, True), (True, True)):
                with self.subTest(device=str(device), triangles=triangles, tets=tets):
                    if tets:
                        _, reference = fixture_owner._tet_scene(device, triangles, True)
                        _, actual = fixture_owner._tet_scene(device, triangles, True)
                        for arguments in (reference, actual):
                            position = arguments[3].numpy()
                            position[[3, 7], 2] *= -1
                            arguments[3].assign(position)
                    else:
                        _, reference, _ = fixture_owner._scalar_scene(device, inactive_middle=True)
                        _, actual, _ = fixture_owner._scalar_scene(device, inactive_middle=True)
                    n = actual[3].shape[0]
                    seed = np.repeat(np.asarray([[.125, -.25, .5]], np.float32), n, axis=0)
                    for arguments in (reference, actual):
                        arguments[-3].assign(seed)
                    wp.launch(kernels.solve_elasticity_scalar, 3, reference, device=device)
                    ids = actual[1].numpy()
                    order = wp.array([-1, -1, *ids, -1], int, device=device)
                    offsets = wp.array([0, 2, 5, 6], int, device=device)
                    current = wp.array([1], int, device=device)
                    kernel = make_dynamic_elasticity(triangles, tets, device.is_cuda)
                    wp.launch(kernel, 32 if device.is_cuda else 2,
                        dynamic_arguments(actual, order, offsets, current, 2),
                        device=device, block_dim=32 if device.is_cuda else 64)
                    self.assertEqual(int(actual[-1].numpy()[0]), 0)
                    for candidate, expected in zip(actual[-3:-1], reference[-3:-1]):
                        np.testing.assert_allclose(candidate.numpy(), expected.numpy(), rtol=3e-6, atol=3e-6)

    def test_material_pool_capture_replay_reads_changed_counts_and_current_group(self):
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA device unavailable")
        n = 257
        permutation = np.random.default_rng(773).permutation(n).astype(np.int32)
        seed = np.repeat(np.asarray([[.125, -.25, .5]], np.float32), n, axis=0)
        for device in devices:
            _, reference = point_material_fixture(device, n)
            _, actual = point_material_fixture(device, n)
            order = wp.array(np.concatenate(([-1, -1], permutation, [-1])), int, device=device)
            offsets = wp.array([2, 2, 2 + n], int, device=device)
            current = wp.zeros(1, int, device=device)
            kernel = make_dynamic_elasticity(False, False, True)
            arguments = dynamic_arguments(actual, order, offsets, current, 8)
            wp.launch(kernel, 128, arguments, device=device, block_dim=32)
            wp.synchronize_device(device)
            with wp.ScopedCapture(device=device) as capture:
                wp.launch(kernel, 128, arguments, device=device, block_dim=32)
            for selected, split in ((0, 0), (0, 1), (1, 1), (0, 3), (1, n), (0, n), (1, 2), (0, 128)):
                current.fill_(selected)
                offsets.assign([2, 2 + split, 2 + n])
                ids = permutation[:split] if selected == 0 else permutation[split:]
                reference[1] = wp.array(ids, int, device=device)
                for candidate in (reference, actual):
                    candidate[-3].assign(seed)
                    candidate[-2].fill_(wp.vec3(-123))
                    candidate[-1].zero_()
                if len(ids):
                    wp.launch(kernels.solve_elasticity_scalar, len(ids), reference, device=device)
                wp.capture_launch(capture.graph)
                self.assertEqual(int(actual[-1].numpy()[0]), 0)
                for candidate, expected in zip(actual[-3:-1], reference[-3:-1]):
                    np.testing.assert_allclose(candidate.numpy(), expected.numpy(), rtol=3e-6, atol=3e-6)

    def test_invalid_material_groups_and_negative_padding_fail_closed(self):
        for device in self.devices:
            _, arguments = point_material_fixture(device, 9)
            order = wp.array([0], int, device=device)
            offsets = wp.array([0, 1], int, device=device)
            current = wp.zeros(1, int, device=device)
            kernel = make_dynamic_elasticity(False, False, device.is_cuda)
            budget = 2
            for group, node, bounds, slots in ((-1, 0, [0, 1], budget),
                    (1, 0, [0, 1], budget), (0, -1, [0, 1], budget),
                    (0, 9, [0, 1], budget), (0, 0, [-1, 1], budget),
                    (0, 0, [0, 2], budget), (0, 0, [0, 1], -1)):
                current.fill_(group)
                order.assign([node])
                offsets.assign(bounds)
                arguments[-3].zero_()
                arguments[-2].fill_(wp.vec3(-123))
                arguments[-1].zero_()
                wp.launch(kernel, 32 if device.is_cuda else 2,
                    dynamic_arguments(arguments, order, offsets, current, slots),
                    device=device, block_dim=32 if device.is_cuda else 64)
                self.assertEqual(int(arguments[-1].numpy()[0]), 385)
                np.testing.assert_array_equal(arguments[-3].numpy(), 0)
                np.testing.assert_array_equal(arguments[-2].numpy(), -123)


if __name__ == "__main__":
    unittest.main()
