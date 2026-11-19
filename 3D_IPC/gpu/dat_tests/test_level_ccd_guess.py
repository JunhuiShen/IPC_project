"""Level-based CCD prediction on the pristine VBD/DAT runtime.

Run in a fresh process, separately from gpu.warp_tests (the project Newton
runtime).  The integration deliberately reuses the existing CCD policy;
these checks exercise its effect on states rather than replacing its math.
"""

from unittest.mock import patch
from types import SimpleNamespace
import unittest

import numpy as np
import warp as wp
import gpu

gpu.select_newton_runtime("reference")

import newton
from newton.solvers import SolverVBD
from gpu.dat_twist import attach_stock_ipc
from gpu.solver_dat import SolverDAT


def fixture(device, *, predictor=True, fraction=1., cap=1, height=2.,
            velocity=(.3, -.2, .1), zero_mass=False, exact=False, direct=False, free_point=False,
            ccd_method="interval"):
    builder = newton.ModelBuilder(gravity=(0., 0., 0.))
    points = [[.25, .25, height], [0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
    velocities = [list(velocity), [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
    masses, flags = [2., 1., 1., 1.], [1, 0, 0, 0]
    if free_point:
        points.append([3., 2., .2])
        velocities.append([3., -1., 2.])
        masses.append(2.)
        flags.append(1)
    if zero_mass:
        points.append([2., 2., 2.])
        velocities.append([3., 4., 5.])
        masses.append(0.)
        flags.append(1)
    builder.add_particles(points, velocities, masses, radius=[0.] * len(points), flags=flags)
    builder.add_triangles([1], [2], [3], tri_ke=[0.], tri_ka=[0.], tri_kd=[0.])
    builder.add_edges([-1] * 3, [-1] * 3, [1, 2, 3], [2, 3, 1],
                      rest=[0.] * 3, edge_ke=[0.] * 3, edge_kd=[0.] * 3)
    builder.color()
    model = builder.finalize(device=device)
    attach_stock_ipc(model, fps=10, substeps=1, iterations=1, exact_geometry=exact,
                     friction=0., gs_predictor="level" if predictor else "none")
    model.ipc.params.update(colored_ccd_guess_iters=cap, colored_ccd_guess_fraction=fraction,
                            ccd_exact_step_bound=direct, ccd_method=ccd_method)
    return model


def solver(model, *, dat=False, iterations=0, boxes=False, colors=False):
    return SolverDAT(model, allow_cpu=True, iterations=iterations,
        enable_planar_dat=dat, enable_broad_phase_boxes=boxes, enable_barrier_coloring=colors,
        particle_enable_self_contact=True, particle_self_contact_margin=.002,
        particle_self_contact_gap=.0015, vertex_contact_capacity=8, edge_contact_capacity=8,
        enable_fast_math=False)


def advance(own, model, before=None, dt=.1):
    before = model.state() if before is None else before
    after = model.state()
    own.step(before, after, model.control(), None, dt)
    own.check_status()
    return before, after


class LevelCcdGuessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_default_native_predictor_and_lifecycle_are_unchanged(self):
        # The opt-in must not replace the default class lifecycle or silently
        # create CCD workspaces for the existing collision-off ablation.
        self.assertIs(SolverDAT._initialize_particles, SolverVBD._initialize_particles)
        for device in self.devices:
            model = fixture(device, predictor=False)
            own = solver(model)
            self.assertFalse(own._guess_requested)
            self.assertIsNone(own.ccd_backend)
            before = model.state()
            initial = before.particle_q.numpy().copy()
            velocity = before.particle_qd.numpy().copy()
            model.gravity.fill_(wp.vec3(0., -2., 0.))
            _, out = advance(own, model, before)
            expected = initial.copy()
            expected[0] += .1 * velocity[0] + np.asarray([0., -.02, 0.])
            np.testing.assert_allclose(out.particle_q.numpy(), expected, rtol=2.e-6, atol=2.e-7)

    def test_velocity_target_fraction_fixed_vertices_and_zero_cap(self):
        for device in self.devices:
            for cap, fraction in ((1, 1.), (5, .25), (0, 1.), (3, 0.)):
                with self.subTest(device=str(device), cap=cap, fraction=fraction):
                    model = fixture(device, cap=cap, fraction=fraction, zero_mass=True)
                    own = solver(model)
                    before = model.state()
                    initial = before.particle_q.numpy().copy()
                    velocity = before.particle_qd.numpy().copy()
                    model.gravity.fill_(wp.vec3(0., -2., 0.))
                    forces = np.zeros_like(initial)
                    forces[0] = [2., 0., 0.]
                    before.particle_f.assign(forces)
                    _, out = advance(own, model, before)
                    expected = initial.copy()
                    if cap:
                        expected[0] += fraction * .1 * velocity[0]
                    np.testing.assert_allclose(out.particle_q.numpy(), expected, rtol=2.e-6, atol=2.e-7)
                    np.testing.assert_array_equal(out.particle_q.numpy()[1:], initial[1:])
                    # Our existing predictor targets velocity motion. The VBD
                    # objective still retains gravity and external forces.
                    expected_inertia = initial[0] + .1 * velocity[0] + np.asarray([.01, -.02, 0.])
                    np.testing.assert_allclose(own.inertia.numpy()[0], expected_inertia, rtol=2.e-6, atol=2.e-7)
                    np.testing.assert_allclose(out.particle_qd.numpy()[0],
                        (expected[0] - initial[0]) / .1, rtol=2.e-6, atol=2.e-6)

    def test_probe_is_discarded_and_free_motion_exits_before_cap(self):
        from gpu.policies import InitialGuess
        for device in self.devices:
            model = fixture(device, cap=6, fraction=.5)
            own = solver(model)
            policy = own.ccd_backend._initial_guess
            self.assertIsInstance(policy, InitialGuess)
            self.assertTrue(policy.independent_colors)
            self.assertIsNotNone(policy.graph._parallel_colors)
            self.assertIsNotNone(policy.graph._parallel_levels)
            self.assertIsNotNone(policy.query_groups)
            original_sweep = policy._sweep
            beginnings, endings = [], []

            def record(q, *args, **kwargs):
                beginnings.append(q.numpy().copy())
                result = original_sweep(q, *args, **kwargs)
                endings.append(q.numpy().copy())
                return result

            before = model.state()
            initial = before.particle_q.numpy().copy()
            with patch.object(policy, "_sweep", side_effect=record):
                _, out = advance(own, model, before)
            self.assertEqual(len(beginnings), 2, "one discarded probe plus one accepted sweep")
            np.testing.assert_array_equal(beginnings[0], initial)
            np.testing.assert_array_equal(beginnings[1], initial)
            np.testing.assert_array_equal(endings[0], endings[1])
            np.testing.assert_array_equal(out.particle_q.numpy(), endings[1])
            self.assertEqual(int(policy.sweep.numpy()[0]), 1)
            self.assertEqual(int(policy.sweep_condition.numpy()[0]), 0)

    def test_crossing_predictor_is_clipped_even_when_vbd_collisions_are_off(self):
        for device in self.devices:
            results = {}
            for predictor in (False, True):
                model = fixture(device, predictor=predictor, height=.02, velocity=(0., 0., -.4))
                own = solver(model)
                initial = model.particle_q.numpy().copy()
                _, out = advance(own, model)
                result = out.particle_q.numpy()
                results[predictor] = result
                np.testing.assert_array_equal(result[1:], initial[1:])
                self.assertFalse(own.particle_enable_self_contact)
                self.assertIsNone(own._energy_contacts)
                self.assertIsNone(own.sdf_backend)
                if predictor:
                    self.assertGreater(int(own.ccd_backend.contacts.candidate_count.numpy()[0]), 0)
                    self.assertGreater(float(result[0, 2]), 0.)
                    self.assertLess(float(result[0, 2]), .02)
                    np.testing.assert_array_equal(own.pos_prev_collision_detection.numpy(), result)
                    np.testing.assert_array_equal(own.particle_displacements.numpy(), np.zeros_like(result))
            self.assertLess(float(results[False][0, 2]), 0., "unrestricted native prediction must cross")

    def test_probe_and_update_do_not_share_blocked_vertex_fraction(self):
        for device in self.devices:
            for method, exact, direct in (("interval", False, False), ("interval", True, False),
                                          ("interval", True, True), ("plane", False, False),
                                          ("direct_rounding", False, False)):
                with self.subTest(device=str(device), method=method, exact=exact, direct=direct):
                    model = fixture(device, height=.02, velocity=(0., 0., -.4),
                                    exact=exact, direct=direct, free_point=True, ccd_method=method)
                    own = solver(model)
                    self.assertEqual(own.ccd_backend.method, method)
                    policy = own.ccd_backend._initial_guess
                    self.assertTrue(policy.independent_colors)
                    original_sweep = policy._sweep
                    starts, ends, maps, independent_flags = [], [], [], []

                    def record(q, colors, dt, **kwargs):
                        starts.append(q.numpy().copy())
                        maps.append(colors.numpy().copy())
                        independent_flags.append(kwargs.get("independent", False))
                        original_sweep(q, colors, dt, **kwargs)
                        ends.append(q.numpy().copy())

                    before = model.state()
                    initial = before.particle_q.numpy().copy()
                    with patch.object(policy, "_sweep", side_effect=record):
                        _, out = advance(own, model, before)
                    self.assertEqual(independent_flags, [True, True])
                    self.assertEqual(len(ends), 2)
                    target = policy.targets.numpy()
                    for mapping, start, end in zip(maps, starts, ends):
                        self.assertEqual(mapping[0], mapping[4],
                                         "Test both vertices in the same color or level/color group")
                        np.testing.assert_array_equal(start, initial, "Probe must be discarded")
                        np.testing.assert_array_equal(end[4], target[4],
                                                      "Unrelated vertex must reach its own full target")
                        np.testing.assert_array_equal(end[1:4], initial[1:4])
                        self.assertGreater(float(end[0, 2]), 0.)
                        self.assertLess(float(end[0, 2]), .02)
                    np.testing.assert_array_equal(out.particle_q.numpy(), ends[1])
                    np.testing.assert_array_equal(own.pos_prev_collision_detection.numpy(), ends[1])
                    self.assertFalse(own.particle_enable_self_contact)
                    self.assertIsNone(own._energy_contacts)

    def test_native_vbd_solve_remains_unrestricted_after_ccd_predictor(self):
        # The option only changes initialization. With DAT disabled the next
        # energy solve must still reach the unrestricted inertial minimizer.
        for device in self.devices:
            for method in ("interval", "plane", "direct_rounding"):
                with self.subTest(device=str(device), method=method):
                    model = fixture(device, height=.02, velocity=(0., 0., -.4), ccd_method=method)
                    own = solver(model, iterations=1)
                    _, out = advance(own, model)
                    np.testing.assert_allclose(out.particle_q.numpy()[0], own.inertia.numpy()[0],
                                               rtol=2.e-6, atol=2.e-7)
                    self.assertLess(float(out.particle_q.numpy()[0, 2]), 0.)

    def test_original_robust_and_direct_step_modes_clip_crossing_predictor(self):
        for device in self.devices:
            for exact, direct in ((False, False), (True, False), (True, True)):
                with self.subTest(device=str(device), exact=exact, direct=direct):
                    model = fixture(device, height=.02, velocity=(0., 0., -.4),
                                    exact=exact, direct=direct)
                    own = solver(model)
                    initial = model.particle_q.numpy().copy()
                    _, out = advance(own, model)
                    self.assertEqual(own.ccd_backend.exact_geometry, exact)
                    self.assertEqual(own.ccd_backend.exact_step_bound, direct)
                    self.assertGreater(int(own.ccd_backend.contacts.candidate_count.numpy()[0]), 0)
                    self.assertLess(float(own.inertia.numpy()[0, 2]), 0.)
                    self.assertGreater(float(out.particle_q.numpy()[0, 2]), 0.)
                    self.assertLess(float(out.particle_q.numpy()[0, 2]), .02)
                    np.testing.assert_array_equal(out.particle_q.numpy()[1:], initial[1:])

    def test_dat_and_combined_coloring_start_from_the_accepted_guess(self):
        for device in self.devices:
            for dat, boxes, colors in ((True, False, False), (False, True, False), (False, True, True)):
                with self.subTest(device=str(device), dat=dat, boxes=boxes, colors=colors):
                    model = fixture(device, height=.02, velocity=(0., 0., -.4))
                    own = solver(model, dat=dat, boxes=boxes, colors=colors)
                    _, out = advance(own, model)
                    accepted = out.particle_q.numpy()
                    self.assertGreater(float(accepted[0, 2]), 0.)
                    self.assertLess(float(accepted[0, 2]), .02)
                    np.testing.assert_array_equal(own.pos_prev_collision_detection.numpy(), accepted)
                    np.testing.assert_array_equal(own.particle_displacements.numpy(), np.zeros_like(accepted))
                    if dat:
                        # The predictor must not overwrite native contact
                        # storage with its broader swept candidate list.
                        self.assertIsNot(own.ccd_backend.contacts.info, own.trimesh_collision_info)
                        self.assertTrue(own.particle_enable_self_contact)
                        self.assertIsNotNone(own._energy_contacts)
                    if colors:
                        own.barrier_coloring.validate_independence()
                        self.assertEqual(int(own.error.numpy()[0]), 0)

    def test_invalid_predictor_settings_are_rejected_at_construction(self):
        for key, values in (("colored_ccd_guess_iters", (-1, 1.5, True)),
                            ("colored_ccd_guess_fraction", (-.1, 1.1, float("nan"), True))):
            for value in values:
                with self.subTest(key=key, value=value):
                    model = fixture("cpu")
                    model.ipc.params[key] = value
                    with self.assertRaises((ValueError, TypeError)):
                        solver(model)
        for conflicting in ("use_ccd_guess", "use_colored_ccd_guess"):
            model = fixture("cpu")
            model.ipc.params[conflicting] = True
            with self.assertRaises(ValueError):
                solver(model)

    def test_triangleless_particles_reject_level_predictor_before_detector_setup(self):
        for device in self.devices:
            with self.subTest(device=str(device)):
                builder = newton.ModelBuilder(gravity=(0., 0., 0.))
                builder.add_particles([[0., 0., 0.]], [[0., 0., 0.]], [1.],
                                      radius=[0.], flags=[1])
                builder.color()
                model = builder.finalize(device=device)
                # Only the requested predictor is needed: rejection must
                # precede energy preparation and detector allocation.
                model.ipc = SimpleNamespace(params={"use_colored_ccd_guess_level_based": True})
                self.assertEqual(model.particle_count, 1)
                self.assertEqual(model.tri_count, 0)
                with self.assertRaisesRegex(ValueError, "requires surface triangles"):
                    solver(model)

    def test_cuda_graph_matches_eager_over_repeated_substeps(self):
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("No CUDA device")
        from gpu.run import FrameGraph
        for device in devices:
            for dat, boxes, colors in ((True, False, False), (False, True, True)):
                for crossing in (False, True):
                    with self.subTest(device=str(device), dat=dat, crossing=crossing):
                        results = []
                        for capture in (False, True):
                            options = dict(height=.02, velocity=(0., 0., -.8), fraction=1.) if crossing else dict(fraction=.75)
                            model = fixture(device, cap=3, **options)
                            model.ipc.params.update(substeps=2, dt=.05, dt2=.0025)
                            # No material corrections in the crossing case:
                            # collision-off VBD would otherwise deliberately
                            # undo the predictor's protection on its first solve.
                            own = solver(model, dat=dat, boxes=boxes, colors=colors,
                                         iterations=0 if crossing else 1)
                            frames = FrameGraph(model, own, graphs=capture)
                            for _ in range(3):
                                frames.advance()
                                if crossing:
                                    self.assertGreater(float(frames.state.particle_q.numpy()[0, 2]), 0.)
                                    self.assertGreater(int(own.ccd_backend.contacts.candidate_count.numpy()[0]), 0)
                            results.append((frames.state.particle_q.numpy(), frames.state.particle_qd.numpy()))
                        for eager, captured in zip(*results):
                            np.testing.assert_array_equal(eager, captured)


if __name__ == "__main__":
    unittest.main()
