"""Per-color CCD acceptance, independent of VBD contact/barrier energies.

Run with the pristine Newton runtime, separately from gpu.warp_tests.
"""

from itertools import product
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from gpu.dat_color_ccd import ColorCcd
from gpu.dat_tests.test_barrier_coloring import graph_model, make_graph
from gpu.dat_tests.test_level_ccd_guess import fixture, advance
from gpu.dat_tests.test_planar_toggle import ForbiddenCollisionObject
from gpu.solver_dat import SolverDAT


def workspace(device, points, triangles=(), edges=(), *, radius=3., flags=None, masses=None,
              method="direct_rounding"):
    model = graph_model(device, points, triangles, edges)
    if flags is not None:
        model.particle_flags.assign(flags)
    if masses is not None:
        model.particle_mass.assign(masses)
    boxes, graph, error = make_graph(model)
    ccd = ColorCcd(model, boxes, graph, error, method=method)
    boxes.radii.fill_(radius)
    boxes.update(model.particle_q)
    graph.update()
    ccd.refresh(model.particle_q)
    reference = wp.clone(model.particle_q)
    displacement = wp.zeros_like(model.particle_q)
    return model, boxes, graph, ccd, error, reference, displacement


def solver(model, *, enabled=True, iterations=2, method="direct_rounding"):
    return SolverDAT(model, allow_cpu=True, iterations=iterations,
        enable_planar_dat=False, enable_broad_phase_boxes=True,
        enable_barrier_coloring=True, enable_color_ccd=enabled, color_ccd_method=method,
        particle_enable_self_contact=True, particle_self_contact_margin=.002,
        particle_self_contact_gap=.0015, vertex_contact_capacity=16,
        edge_contact_capacity=16, enable_fast_math=False)


class ColorCcdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_point_triangle_and_edge_edge_acceptance_for_every_moving_corner(self):
        point_triangle = np.float32([[.25, .25, .25], [0., 0., 0.],
                                     [1., 0., 0.], [0., 1., 0.]])
        edge_edge = np.float32([[-1., 0., .25], [1., 0., .25],
                                [0., -1., 0.], [0., 1., 0.]])
        for device, method in product(self.devices, ("direct_rounding", "plane")):
            for kind, points, triangles, edges in (
                ("PT", point_triangle, [[1, 2, 3]], []),
                ("EE", edge_edge, [], [[-1, -1, 0, 1], [-1, -1, 2, 3]]),
            ):
                for moving in range(4):
                    with self.subTest(device=str(device), method=method, kind=kind, moving=moving):
                        model, _, graph, ccd, error, reference, displacement = workspace(
                            device, points, triangles, edges, method=method)
                        self.assertEqual(int(graph.candidate_count.numpy()[0]), 1)
                        colors = graph.colors.numpy()
                        self.assertEqual(len(set(colors)), 4)
                        graph.current_color.fill_(int(colors[moving]))
                        # Every vertex has a pending correction: only the
                        # selected color may contribute to this CCD motion.
                        proposal = np.full_like(points, .125)
                        proposal[moving] = [0., 0., -2. if moving < (1 if kind == "PT" else 2) else 2.]
                        displacement.assign(proposal)
                        ccd.apply(model.particle_q, reference, displacement)
                        accepted = model.particle_q.numpy()
                        fraction = float(ccd.fractions.numpy()[moving])
                        self.assertGreater(fraction, 0.)
                        self.assertLess(fraction, 1.)
                        np.testing.assert_allclose(accepted[moving], points[moving] + fraction * proposal[moving],
                                                   rtol=2.e-6, atol=2.e-7)
                        fixed = np.arange(4) != moving
                        np.testing.assert_array_equal(accepted[fixed], points[fixed])
                        np.testing.assert_array_equal(reference.numpy()[moving], accepted[moving])
                        np.testing.assert_array_equal(displacement.numpy()[moving], np.zeros(3))
                        np.testing.assert_array_equal(displacement.numpy()[fixed], proposal[fixed])
                        # A signed volume changes sign at the intended
                        # crossing, providing an independent geometric check.
                        def volume(q):
                            a, b, c, d = q.astype(np.float64)
                            return np.dot(a - d, np.cross(b - d, c - d))
                        self.assertGreater(volume(points) * volume(accepted), 0.)
                        self.assertEqual(int(error.numpy()[0]), 0)

    def test_plane_accepts_separated_near_parallel_edges_without_changing_direct_default(self):
        # Parallel edges separated by 15.625 mm. Moving the second edge's
        # endpoint farther away is safe for every point along the trajectory.
        points = np.float32([[0., 0., 0.], [.0125, 0., 0.],
                             [0., .015625, 0.], [.012515625, .015625, 0.]])
        for device, method in product(self.devices, ("direct_rounding", "plane")):
            model, _, graph, ccd, error, reference, displacement = workspace(
                device, points, edges=[[-1, -1, 0, 1], [-1, -1, 2, 3]], method=method)
            graph.current_color.fill_(int(graph.colors.numpy()[2]))
            motion = np.zeros_like(points)
            motion[2, 1] = .001
            displacement.assign(motion)
            ccd.apply(model.particle_q, reference, displacement)
            self.assertEqual(ccd.method, method)
            self.assertEqual(int(error.numpy()[0]), 0)
            if method == "plane":
                np.testing.assert_array_equal(model.particle_q.numpy(), points + motion)
                self.assertEqual(float(ccd.fractions.numpy()[2]), 1.)
            else:
                # Preserve the previously measured conservative direct path.
                np.testing.assert_array_equal(model.particle_q.numpy(), points)
                self.assertEqual(float(ccd.fractions.numpy()[2]), 0.)

    def test_plane_rejects_a_rounded_shortened_endpoint_that_lands_on_another_face(self):
        plane = np.float32(100000.)
        ulp = np.nextafter(plane, np.float32(np.inf)) - plane
        points = [[0., 0., plane + 2 * ulp]]
        for z in (plane, plane + ulp):
            points.extend([[-1., -1., z], [1., -1., z], [0., 1., z]])
        points = np.asarray(points, np.float32)
        for device in self.devices:
            model, _, graph, ccd, error, reference, displacement = workspace(
                device, points, [[1, 2, 3], [4, 5, 6]], method="plane")
            graph.current_color.fill_(int(graph.colors.numpy()[0]))
            motion = np.zeros_like(points)
            motion[0, 2] = -4 * ulp
            displacement.assign(motion)
            ccd.apply(model.particle_q, reference, displacement)
            self.assertEqual(int(error.numpy()[0]), 0)
            np.testing.assert_array_equal(model.particle_q.numpy(), points)
            self.assertEqual(int(ccd.rejected.numpy()[0]), 1)

    def test_color_method_is_separate_from_predictor_and_validated(self):
        model = fixture("cpu", predictor=False)
        model.ipc.params["ccd_method"] = "direct_rounding"
        own = solver(model, method="plane")
        self.assertEqual(own.color_ccd.method, "plane")
        self.assertEqual(own.configuration["color_ccd"]["method"], "plane")
        self.assertEqual(model.ipc.params["ccd_method"], "direct_rounding")
        for method in ("interval", "bypass", None):
            with self.subTest(method=method), self.assertRaises(ValueError):
                solver(fixture("cpu", predictor=False), method=method)
        with self.assertRaises(ValueError):
            solver(fixture("cpu", predictor=False), enabled=False, method="plane")

    def test_blocked_vertex_does_not_throttle_unrelated_vertex_or_reapply_motion(self):
        points = np.float32([[.25, .25, .25], [0., 0., 0.], [1., 0., 0.],
                             [0., 1., 0.], [100., 100., 100.]])
        for device in self.devices:
            model, _, graph, ccd, error, reference, displacement = workspace(
                device, points, [[1, 2, 3]])
            colors = graph.colors.numpy()
            self.assertEqual(colors[0], colors[4])
            graph.current_color.fill_(int(colors[0]))
            motion = np.zeros_like(points)
            motion[0] = [0., 0., -1.]
            motion[4] = [.5, -.25, .75]
            displacement.assign(motion)
            ccd.apply(model.particle_q, reference, displacement)
            accepted = model.particle_q.numpy().copy()
            self.assertGreater(float(accepted[0, 2]), 0.)
            self.assertLess(float(accepted[0, 2]), float(points[0, 2]))
            np.testing.assert_array_equal(accepted[4], points[4] + motion[4])
            self.assertEqual(float(ccd.fractions.numpy()[4]), 1.)
            # Subsequent colors and a second visit with no new correction
            # must not recover the rejected part of the earlier proposal.
            for color in (int(colors[1]), int(colors[0])):
                graph.current_color.fill_(color)
                ccd.apply(model.particle_q, reference, displacement)
                np.testing.assert_array_equal(model.particle_q.numpy(), accepted)
            np.testing.assert_array_equal(reference.numpy(), accepted)
            np.testing.assert_array_equal(displacement.numpy(), np.zeros_like(points))
            departing = np.zeros_like(points)
            departing[0, 2] = .125
            displacement.assign(departing)
            ccd.apply(model.particle_q, reference, displacement)
            np.testing.assert_array_equal(model.particle_q.numpy(), accepted + departing)
            self.assertEqual(float(ccd.fractions.numpy()[0]), 1.)
            self.assertEqual(int(ccd.rejected.numpy()[0]), 0)
            self.assertEqual(int(error.numpy()[0]), 0)

    def test_multiple_pairs_use_the_smallest_fraction(self):
        points = np.float32([[.25, .25, .25], [0., 0., 0.], [1., 0., 0.],
                             [0., 1., 0.], [0., 0., -.25], [1., 0., -.25], [0., 1., -.25]])
        for device in self.devices:
            values = []
            for triangles in ([[1, 2, 3]], [[4, 5, 6]], [[4, 5, 6], [1, 2, 3]]):
                model, _, graph, ccd, error, reference, displacement = workspace(
                    device, points, triangles, flags=[1, 0, 0, 0, 0, 0, 0])
                self.assertEqual(int(graph.candidate_count.numpy()[0]), len(triangles))
                graph.current_color.fill_(int(graph.colors.numpy()[0]))
                motion = np.zeros_like(points)
                motion[0, 2] = -1.
                displacement.assign(motion)
                ccd.apply(model.particle_q, reference, displacement)
                values.append((float(ccd.fractions.numpy()[0]), model.particle_q.numpy()[0].copy()))
                self.assertEqual(int(error.numpy()[0]), 0)
            self.assertLess(values[0][0], values[1][0])
            self.assertEqual(values[2][0], min(values[0][0], values[1][0]))
            np.testing.assert_array_equal(values[2][1], values[0][1])

    def test_fixed_and_zero_mass_vertices_stay_fixed_and_box_budget_is_not_reused(self):
        points = np.float32([[0., 0., 0.], [2., 0., 0.], [4., 0., 0.]])
        for device in self.devices:
            model, boxes, graph, ccd, error, reference, displacement = workspace(
                device, points, radius=.01, flags=[1, 0, 1], masses=[1., 1., 0.])
            graph.current_color.fill_(0)
            for _ in range(3):
                displacement.assign(np.tile([.03, -.03, .02], (3, 1)))
                ccd.apply(model.particle_q, reference, displacement)
                accepted = model.particle_q.numpy()
                np.testing.assert_array_equal(accepted[1:], points[1:])
                self.assertTrue(np.all(np.abs(accepted[0].astype(np.float64) - points[0]) <= .01))
                self.assertGreater(float(accepted[0, 0]), .009)
                np.testing.assert_array_equal(reference.numpy(), accepted)
                np.testing.assert_array_equal(displacement.numpy(), np.zeros_like(points))
            old = accepted.copy()
            boxes.update(model.particle_q)
            graph.update()
            ccd.refresh(model.particle_q)
            displacement.assign(np.tile([.03, 0., 0.], (3, 1)))
            ccd.apply(model.particle_q, reference, displacement)
            self.assertGreater(float(model.particle_q.numpy()[0, 0]), float(old[0, 0]) + .009)
            self.assertEqual(int(error.numpy()[0]), 0)

    def test_full_vbd_crossing_is_clipped_with_and_without_level_predictor(self):
        for device in self.devices:
            for predictor in (False, True):
                for iterations in (0, 3):
                    with self.subTest(device=str(device), predictor=predictor, iterations=iterations):
                        model = fixture(device, predictor=predictor, height=.02, velocity=(0., 0., -.4))
                        model.ipc.params.update(node_box_min=.1, node_box_max=.1, node_box_update_count=2)
                        own = solver(model, iterations=iterations)
                        initial = model.particle_q.numpy().copy()
                        _, after = advance(own, model)
                        result = after.particle_q.numpy()
                        self.assertLess(float(own.inertia.numpy()[0, 2]), 0.)
                        self.assertGreater(float(result[0, 2]), 0.)
                        self.assertLessEqual(float(result[0, 2]), float(initial[0, 2]))
                        if not iterations and not predictor:
                            np.testing.assert_array_equal(result, initial)
                        else:
                            self.assertLess(float(result[0, 2]), float(initial[0, 2]))
                        np.testing.assert_array_equal(result[1:], initial[1:])
                        np.testing.assert_array_equal(own.pos_prev_collision_detection.numpy(), result)
                        np.testing.assert_array_equal(own.particle_displacements.numpy(), np.zeros_like(result))
                        self.assertGreater(int(own.barrier_coloring.candidate_count.numpy()[0]), 0)
                        self.assertIsNone(own._energy_contacts)
                        self.assertIsNone(own.sdf_backend)
                        self.assertFalse(own.particle_enable_self_contact)

    def test_acceptance_uses_the_bounds_checkpoint_despite_center_or_radius_changes(self):
        for device in self.devices:
            for mutation in ("refresh_with_moved_q", "begin", "reset_without_positions"):
                with self.subTest(device=str(device), mutation=mutation):
                    model, boxes, graph, ccd, error, reference, displacement = workspace(
                        device, [[0., 0., 0.]], radius=.001)
                    if mutation == "refresh_with_moved_q":
                        model.particle_q.assign([[.0005, 0., 0.]])
                        reference.assign([[.0005, 0., 0.]])
                        # Refreshing incidence alone cannot enlarge or move
                        # the bounds that supplied the candidate graph.
                        ccd.refresh(model.particle_q)
                    elif mutation == "begin":
                        model.particle_qd.assign([[3., 0., 0.]])
                        boxes.begin(model.particle_qd, .1)
                        self.assertGreater(float(boxes.radii.numpy()[0]), .001)
                    else:
                        boxes.reset(None)
                        self.assertGreater(float(boxes.radii.numpy()[0]), .001)
                    graph.current_color.fill_(0)
                    displacement.assign([[.02, 0., 0.]])
                    ccd.apply(model.particle_q, reference, displacement)
                    accepted = model.particle_q.numpy().astype(np.float64)
                    self.assertGreater(float(accepted[0, 0]), .0009)
                    self.assertTrue(np.all(np.abs(accepted) <= .001))
                    np.testing.assert_array_equal(ccd.center.numpy(), np.zeros((1, 3)))
                    self.assertEqual(int(error.numpy()[0]), 0)

    def test_stale_box_or_graph_checkpoint_prevents_publication(self):
        for device in self.devices:
            for mutation in ("boxes_without_graph", "graph_without_refresh"):
                with self.subTest(device=str(device), mutation=mutation):
                    model, boxes, graph, ccd, error, reference, displacement = workspace(
                        device, [[0., 0., 0.]], radius=.01)
                    if mutation == "boxes_without_graph":
                        model.particle_q.assign([[.1, 0., 0.]])
                        reference.assign([[.1, 0., 0.]])
                        boxes.update(model.particle_q)
                        ccd.refresh(model.particle_q)
                    else:
                        graph.update()
                    original = model.particle_q.numpy().copy()
                    graph.current_color.fill_(0)
                    displacement.assign([[.005, 0., 0.]])
                    ccd.apply(model.particle_q, reference, displacement)
                    self.assertNotEqual(int(error.numpy()[0]), 0)
                    np.testing.assert_array_equal(model.particle_q.numpy(), original)

    def test_checkpoints_rebuild_and_reset_refresh_the_box_center(self):
        for device in self.devices:
            model = fixture(device, predictor=False, height=2., velocity=(3., -2., 1.))
            model.ipc.params.update(node_box_min=.01, node_box_max=.01, node_box_update_count=2)
            own = solver(model, iterations=5)
            original_apply = own.color_ccd.apply
            original_refresh = own.color_ccd.refresh
            checkpoints, accepted_points = [], []

            def record_refresh(q):
                original_refresh(q)
                checkpoints.append(q.numpy().copy())
                np.testing.assert_array_equal(own.color_ccd.center.numpy(), q.numpy())

            def check_apply(q, reference, displacement):
                original_apply(q, reference, displacement)
                positions = q.numpy().astype(np.float64)
                center = own.color_ccd.center.numpy().astype(np.float64)
                radii = own.broad_phase_boxes.radii.numpy()
                self.assertTrue(np.all(np.abs(positions - center) <= radii[:, None]))
                accepted_points.append(positions.copy())

            with patch.object(own.color_ccd, "refresh", side_effect=record_refresh), \
                    patch.object(own.color_ccd, "apply", side_effect=check_apply):
                _, after = advance(own, model)
            self.assertEqual(len(checkpoints), 3)
            self.assertGreater(len(accepted_points), len(checkpoints))
            self.assertGreater(float(after.particle_q.numpy()[0, 0]), float(checkpoints[0][0, 0]) + .02)
            translated = after.particle_q.numpy() + np.float32([4., 5., 6.])
            after.particle_q.assign(translated)
            own.rebuild_bvh(after)
            np.testing.assert_array_equal(own.color_ccd.center.numpy(), translated)
            own.barrier_coloring.validate_independence()
            translated += np.float32([2., 3., 4.])
            after.particle_q.assign(translated)
            own.reset(after, flags=0)
            np.testing.assert_array_equal(own.color_ccd.center.numpy(), translated)
            own.reset(after)
            np.testing.assert_array_equal(own.color_ccd.center.numpy(), model.particle_q.numpy())
            np.testing.assert_array_equal(own.color_ccd.center.numpy(), after.particle_q.numpy())
            own.barrier_coloring.validate_independence()
            own.check_status()

    def test_flag_off_keeps_unrestricted_native_motion(self):
        for device in self.devices:
            model = fixture(device, predictor=False, height=.02, velocity=(0., 0., -.4))
            own = solver(model, enabled=False)
            before, after = model.state(), model.state()
            own.step(before, after, model.control(), ForbiddenCollisionObject(), .1)
            own.check_status()
            self.assertIsNone(own.color_ccd)
            self.assertLess(float(after.particle_q.numpy()[0, 2]), 0.)
            np.testing.assert_allclose(after.particle_q.numpy()[0], own.inertia.numpy()[0],
                                       rtol=2.e-6, atol=2.e-7)

    def test_option_requires_bool_boxes_coloring_and_no_planar_dat(self):
        model = fixture("cpu", predictor=False)
        for value in (None, 0, 1, "true"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                SolverDAT(model, allow_cpu=True, enable_planar_dat=False,
                          enable_broad_phase_boxes=True, enable_barrier_coloring=True,
                          enable_color_ccd=value)
        for options in ({}, {"enable_planar_dat": False},
                        {"enable_planar_dat": False, "enable_broad_phase_boxes": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SolverDAT(model, allow_cpu=True, enable_color_ccd=True, **options)

    def test_cuda_capture_matches_eager_over_repeated_crossing_steps(self):
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("No CUDA device")
        from gpu.run import FrameGraph
        for device, method in product(devices, ("direct_rounding", "plane")):
            results = []
            for captured in (False, True):
                model = fixture(device, predictor=False, height=.02, velocity=(0., 0., -.8))
                model.ipc.params.update(substeps=2, dt=.05, dt2=.0025,
                                        node_box_min=.1, node_box_max=.1, node_box_update_count=2)
                own = solver(model, iterations=3, method=method)
                frames = FrameGraph(model, own, graphs=captured)
                for _ in range(3):
                    frames.advance()
                    own.check_status()
                    self.assertGreater(float(frames.state.particle_q.numpy()[0, 2]), 0.)
                results.append((frames.state.particle_q.numpy(), frames.state.particle_qd.numpy()))
            for eager, captured in zip(*results):
                np.testing.assert_array_equal(eager, captured)


if __name__ == "__main__":
    unittest.main()
