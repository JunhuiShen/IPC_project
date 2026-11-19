"""Adaptive BVH maintenance must remain observational in collision-free DAT."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverVBD

from gpu import solver_dat as candidate
from gpu import sdf
from gpu.broad_phase_boxes import BroadPhaseBoxes
from gpu.dat_tests.test_planar_toggle import (
    ForbiddenCollisionObject,
    attach,
    fixture,
    solver,
)


@wp.kernel(module="unique", enable_backward=False)
def _query_hits(bvh: wp.uint64, lower: wp.array[wp.vec3], upper: wp.array[wp.vec3],
                hits: wp.array2d[int]):
    row = wp.tid()
    query = wp.bvh_query_aabb(bvh, lower[row], upper[row])
    item = int(0)
    while wp.bvh_query_next(query, item):
        hits[row, item] = 1


def box_model(device, points, triangles=(), edges=(), *, worlds=None, world_count=1):
    """Only the unchanged Newton model fields needed by the box workspace."""
    points = np.asarray(points, np.float32).reshape(-1, 3)
    triangles = np.asarray(triangles, np.int32).reshape(-1, 3)
    edges = np.asarray(edges, np.int32).reshape(-1, 4)
    return SimpleNamespace(
        device=wp.get_device(device), particle_count=len(points),
        tri_count=len(triangles), edge_count=len(edges), world_count=world_count,
        particle_q=wp.array(points, wp.vec3, device=device),
        particle_qd=wp.zeros(len(points), wp.vec3, device=device),
        tri_indices=wp.array(triangles, int, device=device),
        edge_indices=wp.array(edges, int, device=device),
        particle_world=wp.array(np.zeros(len(points), np.int32) if worlds is None else worlds,
                                int, device=device),
    )


def workspace(model, **options):
    error = wp.zeros(1, int, device=model.device)
    return BroadPhaseBoxes(model, error, **options), error


def with_box_parameters(model, *, interval=2):
    model.ipc.params.update(node_box_min=.001, node_box_max=.01,
                            node_box_update_count=interval)
    return model


def material_model(device):
    builder = newton.ModelBuilder(gravity=(0., -1.5, 0.))
    points = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.]]
    velocities = [[0., 0., 0.], [.1, 0., .2], [0., -.2, 0.], [.3, -.1, -.4]]
    builder.add_particles(points, velocities, [1.] * 4, radius=[0.] * 4,
                          flags=[0, 1, 1, 1])
    builder.add_triangles([0, 1], [1, 0], [2, 3], tri_ke=[9.] * 2,
                          tri_ka=[13.] * 2, tri_kd=[.02] * 2)
    builder.add_edges([2], [3], [0], [1], rest=[0.], edge_ke=[.7], edge_kd=[.01])
    builder.color()
    return with_box_parameters(attach(builder.finalize(device=device),
                                     pin=8., with_sdf=True, coefficients=[2.75]))


class BroadPhaseBoxesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def assert_outward_close(self, actual_lower, actual_upper, lower, upper):
        """Enclose the binary64 oracle without an arbitrary large epsilon."""
        self.assertTrue(np.all(actual_lower.astype(np.float64) <= lower))
        self.assertTrue(np.all(actual_upper.astype(np.float64) >= upper))
        for actual, expected in ((actual_lower, lower), (actual_upper, upper)):
            ulp = np.abs(np.spacing(expected.astype(np.float32))).astype(np.float64)
            self.assertTrue(np.all(np.abs(actual.astype(np.float64) - expected) <= 2. * ulp))

    def assert_bvh_matches(self, bvh, lower, upper, query_lower, query_upper, device):
        hits = wp.zeros((len(query_lower), len(lower)), int, device=device)
        wp.launch(_query_hits, len(query_lower),
                  [bvh.id, wp.array(query_lower, wp.vec3, device=device),
                   wp.array(query_upper, wp.vec3, device=device), hits], device=device)
        expected = np.all(query_lower[:, None, :] <= upper[None, :, :], axis=2)
        expected &= np.all(lower[None, :, :] <= query_upper[:, None, :], axis=2)
        np.testing.assert_array_equal(hits.numpy(), expected.astype(np.int32))

    def test_outward_bounds_union_padding_and_edge_endpoint_slots(self):
        rng = np.random.default_rng(19481)
        points = np.float32(rng.normal(size=(48, 3)) * 10. ** rng.uniform(-8, 8, (48, 1)))
        points[:3] = [[1., -1., 0.], [2. ** 20, -2. ** 20, -0.], [2. ** -20, 0., -2. ** -20]]
        radii = 10. ** rng.uniform(-9, 3, len(points))
        radii[:3] = [1., 2. ** -4 + 2. ** -40, np.nextafter(1.e-10, np.inf)]
        triangles = np.asarray([[0, 1, 2], [4, 5, 6], [17, 23, 47]], np.int32)
        # Opposite vertices are deliberately unrelated to the actual endpoints;
        # boundary edges also use the Newton -1 opposite-vertex sentinel.
        edges = np.asarray([[1, 2, 4, 5], [-1, -1, 17, 23], [0, 47, 6, 7]], np.int32)
        reach = np.nextafter(.002, np.inf)
        for device in self.devices:
            with self.subTest(device=str(device)):
                model = box_model(device, points, triangles, edges)
                boxes, error = workspace(model, d_hat=reach)
                self.assertEqual(boxes.radii.dtype, wp.float64)
                boxes.radii.assign(radii)
                boxes.update(model.particle_q)
                self.assertEqual(int(error.numpy()[0]), 0)
                lower, upper = boxes.node_lower.numpy(), boxes.node_upper.numpy()
                self.assert_outward_close(lower, upper, points.astype(np.float64) - radii[:, None],
                                          points.astype(np.float64) + radii[:, None])
                tri_lower = lower[triangles].min(axis=1).astype(np.float64)
                tri_upper = upper[triangles].max(axis=1).astype(np.float64)
                self.assert_outward_close(boxes.triangle_lower.numpy(), boxes.triangle_upper.numpy(),
                                          tri_lower - reach, tri_upper + reach)
                edge_lower = lower[edges[:, 2:]].min(axis=1)
                edge_upper = upper[edges[:, 2:]].max(axis=1)
                np.testing.assert_array_equal(boxes.edge_lower.numpy(), edge_lower)
                np.testing.assert_array_equal(boxes.edge_upper.numpy(), edge_upper)
                self.assert_outward_close(boxes.edge_query_lower.numpy(), boxes.edge_query_upper.numpy(),
                                          edge_lower.astype(np.float64) - reach,
                                          edge_upper.astype(np.float64) + reach)

    def test_bvh_queries_match_brute_force_after_large_refits_and_rebuild(self):
        rng = np.random.default_rng(391)
        points = rng.uniform(-2., 2., (24, 3)).astype(np.float32)
        triangles = np.asarray([rng.choice(len(points), 3, replace=False) for _ in range(12)], np.int32)
        edges = np.asarray([[-1, -1, *rng.choice(len(points), 2, replace=False)] for _ in range(14)], np.int32)
        for device in self.devices:
            model = box_model(device, points, triangles, edges)
            boxes, error = workspace(model)
            boxes.radii.assign(rng.uniform(.001, .01, len(points)))
            bvhs = (boxes.node_bvh, boxes.triangle_bvh, boxes.edge_bvh)
            for phase, rebuild in enumerate((False, False, True)):
                with self.subTest(device=str(device), phase=phase):
                    moved = points.copy() if phase == 0 else points[::-1].copy() + np.float32(phase * 30.)
                    model.particle_q.assign(moved)
                    boxes.update(model.particle_q, rebuild=rebuild)
                    self.assertEqual(int(error.numpy()[0]), 0)
                    if not rebuild:
                        self.assertEqual((boxes.node_bvh, boxes.triangle_bvh, boxes.edge_bvh), bvhs)
                    for name in ("node", "triangle", "edge"):
                        lower = getattr(boxes, name + "_lower").numpy()
                        upper = getattr(boxes, name + "_upper").numpy()
                        # Include zero-volume boundary queries to exercise the
                        # inclusive AABB-overlap convention as well as misses.
                        query_lower = np.vstack((lower, upper, moved - .05, [[-100., -100., -100.]]))
                        query_upper = np.vstack((lower, upper, moved + .05, [[-99., -99., -99.]]))
                        self.assert_bvh_matches(getattr(boxes, name + "_bvh"), lower, upper,
                                                query_lower, query_upper, device)

    def test_empty_primitive_sets_and_empty_model(self):
        for device in self.devices:
            for points, triangles, edges in (
                ([], [], []),
                ([[0., 0., 0.]], [], []),
                ([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], [[0, 1, 2]], []),
                ([[0., 0., 0.], [1., 0., 0.]], [], [[-1, -1, 0, 1]]),
            ):
                with self.subTest(device=str(device), nodes=len(points), tris=len(triangles), edges=len(edges)):
                    model = box_model(device, points, triangles, edges)
                    boxes, error = workspace(model)
                    boxes.begin(model.particle_qd, .01)
                    boxes.anchor(model.particle_q)
                    boxes.update(model.particle_q)
                    boxes.update(model.particle_q, rebuild=True)
                    boxes.finish(model.particle_q)
                    boxes.reset(model.particle_q)
                    self.assertEqual(int(error.numpy()[0]), 0)
                    for name, count in (("node", len(points)), ("triangle", len(triangles)), ("edge", len(edges))):
                        self.assertEqual(len(getattr(boxes, name + "_lower")), count)
                        self.assertEqual(len(getattr(boxes, name + "_upper")), count)
                        self.assertEqual(getattr(boxes, name + "_bvh") is None, count == 0)

    def test_radius_history_is_frozen_until_next_begin(self):
        points = np.array([[0., 0., 0.], [1., 2., 3.], [-1., -2., -3.]], np.float32)
        velocity = np.array([[0., 0., 0.], [.03, .04, 0.], [1., 0., 0.]], np.float32)
        for device in self.devices:
            model = box_model(device, points)
            boxes, error = workspace(model, minimum=.001, maximum=.01, padding=1.2)
            np.testing.assert_array_equal(boxes.previous_displacement.numpy(), .01)
            boxes.previous_displacement.assign(np.array([0., .003, .02]))
            boxes.begin(wp.array(velocity, wp.vec3, device=device), .1)
            expected = np.clip(1.2 * np.maximum([0., .003, .02],
                               .1 * np.linalg.norm(velocity.astype(np.float64), axis=1)), .001, .01)
            np.testing.assert_allclose(boxes.radii.numpy(), expected, rtol=2.e-15, atol=0.)
            frozen = boxes.radii.numpy().copy()
            predictor = points + np.float32(.125)
            boxes.anchor(wp.array(predictor, wp.vec3, device=device))
            final = predictor + np.array([[.001, 0., 0.], [0., .002, -.003], [.02, 0., 0.]], np.float32)
            position = wp.array(final, wp.vec3, device=device)
            boxes.update(position)
            boxes.update(position)
            np.testing.assert_array_equal(boxes.start.numpy(), predictor)
            np.testing.assert_array_equal(boxes.radii.numpy(), frozen)
            boxes.finish(position)
            history = np.linalg.norm(final.astype(np.float64) - predictor.astype(np.float64), axis=1)
            np.testing.assert_allclose(boxes.previous_displacement.numpy(), history, rtol=2.e-15, atol=0.)
            boxes.begin(model.particle_qd, .1)
            np.testing.assert_allclose(boxes.radii.numpy(), np.clip(1.2 * history, .001, .01),
                                       rtol=2.e-15, atol=0.)
            self.assertEqual(int(error.numpy()[0]), 0)

    def test_reset_discards_only_selected_world_history(self):
        for device in self.devices:
            model = box_model(device, np.zeros((4, 3)), worlds=[0, 1, -1, 0], world_count=2)
            boxes, error = workspace(model)
            previous = np.array([.001, .002, .003, .004])
            radii = np.array([.005, .006, .007, .008])
            boxes.previous_displacement.assign(previous)
            boxes.radii.assign(radii)
            position = wp.array(np.arange(12, dtype=np.float32).reshape(4, 3), wp.vec3, device=device)
            mask = wp.array([True, False, False], wp.bool, device=device)
            boxes.reset(position, world_mask=mask)
            previous[[0, 3]] = .01
            radii[[0, 3]] = .01
            np.testing.assert_array_equal(boxes.previous_displacement.numpy(), previous)
            np.testing.assert_array_equal(boxes.radii.numpy(), radii)
            boxes.reset(position)
            np.testing.assert_array_equal(boxes.previous_displacement.numpy(), .01)
            np.testing.assert_array_equal(boxes.radii.numpy(), .01)
            np.testing.assert_array_equal(boxes.start.numpy(), position.numpy())
            self.assertEqual(int(error.numpy()[0]), 0)

    def test_invalid_configuration_and_nonfinite_device_inputs(self):
        model = box_model("cpu", [[0., 0., 0.]])
        for options in ({"minimum": 0.}, {"minimum": np.nan}, {"maximum": np.inf},
                        {"minimum": .02, "maximum": .01}, {"padding": 0.},
                        {"padding": np.nan}, {"d_hat": -.1}, {"d_hat": np.inf},
                        {"update_interval": 0}, {"update_interval": 1.5}):
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                workspace(model, **options)
        for device in self.devices:
            for field in ("position", "velocity", "history", "radius"):
                with self.subTest(device=str(device), field=field):
                    model = box_model(device, [[0., 0., 0.]])
                    boxes, error = workspace(model)
                    if field == "position":
                        model.particle_q.assign([[np.nan, 0., 0.]])
                    elif field == "velocity":
                        model.particle_qd.assign([[np.inf, 0., 0.]])
                        boxes.begin(model.particle_qd, .1)
                    elif field == "history":
                        boxes.previous_displacement.assign([np.nan])
                        boxes.begin(model.particle_qd, .1)
                    else:
                        boxes.radii.assign([np.inf])
                    boxes.update(model.particle_q)
                    self.assertNotEqual(int(error.numpy()[0]), 0)
                    self.assertTrue(np.isfinite(boxes.node_lower.numpy()).all())
                    self.assertTrue(np.isfinite(boxes.node_upper.numpy()).all())

    def test_toggle_validation_and_collision_workspaces_are_never_constructed(self):
        native = __import__(SolverVBD.__module__, fromlist=["SolverVBD"])
        for device in self.devices:
            model = with_box_parameters(fixture(device, with_sdf=True))
            with self.assertRaisesRegex(ValueError, "enable_planar_dat=False"):
                solver(model, enabled=True, enable_broad_phase_boxes=True)
            for value in (None, 0, 1, "false", np.bool_(False)):
                with self.subTest(value=value), self.assertRaisesRegex(TypeError, "enable_broad_phase_boxes"):
                    solver(model, enabled=False, enable_broad_phase_boxes=value)
            with patch.object(native, "TriMeshCollisionDetector", side_effect=AssertionError("Detector created")), \
                 patch.object(candidate.energies, "EnergyWorkspace", side_effect=AssertionError("Contact workspace created")), \
                 patch.object(sdf, "SdfBackend", side_effect=AssertionError("SDF workspace created")):
                own = solver(model, enabled=False, enable_broad_phase_boxes=True,
                             collision_pipeline=ForbiddenCollisionObject(), rigid_soft_enable_dat=True)
                state, out = model.state(), model.state()
                own.collision_pipeline = ForbiddenCollisionObject()
                own._pipeline_contacts = ForbiddenCollisionObject()
                with patch.object(own, "_collision_detection_penetration_free",
                                  side_effect=AssertionError("Collision query ran")), \
                     patch.object(wp, "launch", wraps=wp.launch) as launches:
                    own.step(state, out, model.control(), ForbiddenCollisionObject(), .1)
                own.check_status()
            self.assertIsNotNone(own.broad_phase_boxes)
            self.assertIsNone(own._energy_contacts)
            self.assertIsNone(own.sdf_backend)
            self.assertIsNone(getattr(own, "trimesh_collision_detector", None))
            self.assertIsNone(own.collision_pipeline)
            launched = [call.kwargs.get("kernel", call.args[0] if call.args else None)
                        for call in launches.call_args_list]
            for kernel in (native.apply_truncation_ts, native.apply_planar_truncation_parallel_by_collision,
                           native.accumulate_self_contact_force_and_hessian):
                self.assertNotIn(kernel, launched)
            self.assertTrue(all("contact" not in getattr(kernel, "key", "").lower() for kernel in launched))
            np.testing.assert_array_equal(own.truncation_ts.numpy(), 0.)

    def test_solver_hooks_freeze_before_predictor_and_anchor_after_predictor(self):
        dt = .1
        for device in self.devices:
            model = with_box_parameters(fixture(device), interval=2)
            own = solver(model, enabled=False, enable_broad_phase_boxes=True, iterations=5)
            boxes = own.broad_phase_boxes
            state, out = model.state(), model.state()
            initial = state.particle_q.numpy().copy()
            expected_anchor = initial + dt * state.particle_qd.numpy()
            events = []
            def observe(name):
                original = getattr(boxes, name)
                def call(*args, **kwargs):
                    events.append(name)
                    if name == "begin":
                        np.testing.assert_array_equal(state.particle_q.numpy(), initial)
                    if name == "anchor":
                        np.testing.assert_allclose(args[0].numpy(), expected_anchor, rtol=3.e-6, atol=8.e-8)
                    return original(*args, **kwargs)
                return call
            with patch.object(boxes, "begin", side_effect=observe("begin")), \
                 patch.object(boxes, "anchor", side_effect=observe("anchor")), \
                 patch.object(boxes, "update", side_effect=observe("update")), \
                 patch.object(boxes, "finish", side_effect=observe("finish")):
                own.step(state, out, model.control(), None, dt)
            own.check_status()
            self.assertEqual(events, ["begin", "anchor", "update", "update", "update", "finish"])
            np.testing.assert_allclose(boxes.start.numpy(), expected_anchor, rtol=3.e-6, atol=8.e-8)
            with patch.object(boxes, "update", wraps=boxes.update) as update:
                own.rebuild_bvh(out)
            update.assert_called_once_with(out.particle_q, rebuild=True)
            own.reset(out, flags=0)
            np.testing.assert_array_equal(boxes.previous_displacement.numpy(), .01)
            np.testing.assert_array_equal(boxes.radii.numpy(), .01)

    def test_material_trajectory_is_byte_identical_with_boxes_eager_and_captured(self):
        for device in self.devices:
            for captured in ((False, True) if device.is_cuda else (False,)):
                with self.subTest(device=str(device), captured=captured):
                    results = []
                    for enabled in (False, True):
                        model = material_model(device)
                        own = solver(model, enabled=False, enable_broad_phase_boxes=enabled, iterations=5)
                        a, b = model.state(), model.state()
                        initial = a.particle_q.numpy().copy()
                        initial[3, 2] = .25
                        a.particle_q.assign(initial)
                        control = model.control()
                        def run_pair():
                            own.step(a, b, control, None, .01)
                            own.step(b, a, control, None, .01)
                        if captured:
                            with wp.ScopedCapture(device=device) as capture:
                                run_pair()
                            for _ in range(3):
                                wp.capture_launch(capture.graph)
                        else:
                            for _ in range(3):
                                run_pair()
                        own.check_status()
                        result = (a.particle_q.numpy().copy(), a.particle_qd.numpy().copy())
                        self.assertGreater(np.linalg.norm(result[0] - initial), 1.e-4)
                        results.append(result)
                    for plain, maintained in zip(*results):
                        self.assertEqual(plain.dtype, maintained.dtype)
                        self.assertEqual(plain.shape, maintained.shape)
                        self.assertEqual(plain.tobytes(), maintained.tobytes())

    def test_zero_iteration_predictor_is_byte_identical_with_boxes(self):
        for device in self.devices:
            results = []
            for enabled in (False, True):
                model = with_box_parameters(fixture(device))
                own = solver(model, enabled=False, enable_broad_phase_boxes=enabled, iterations=0)
                state, out = model.state(), model.state()
                initial = state.particle_q.numpy().copy()
                velocity = state.particle_qd.numpy().copy()
                for _ in range(3):
                    own.step(state, out, model.control(), None, .1)
                    state, out = out, state
                own.check_status()
                result = (state.particle_q.numpy().copy(), state.particle_qd.numpy().copy())
                np.testing.assert_allclose(result[0], initial + .3 * velocity, rtol=3.e-6, atol=8.e-8)
                np.testing.assert_allclose(result[1], velocity, rtol=3.e-6, atol=8.e-7)
                results.append(result)
            for plain, maintained in zip(*results):
                self.assertEqual(plain.tobytes(), maintained.tobytes())


if __name__ == "__main__":
    unittest.main()
