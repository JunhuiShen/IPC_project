"""Collision-disabled project GS retains materials, masks and GPU scheduling."""
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import gpu
gpu.select_newton_runtime("project")
import warp as wp
import newton
from newton.solvers import SolverVBD
from newton._src.solvers.solver import SolverBase
from gpu import solver as candidate
from gpu import contact, sdf
from gpu.dat_twist import attach_stock_ipc
from gpu.solver import SolverIPC

def fixture(device, *, pin=0., predictor="colored", fraction=1., with_sdf=False,
            velocity=(3., -2., 1.), height=.001, contact_aware=True):
    builder = newton.ModelBuilder(gravity=(0., 0., 0.))
    points = [[.25, .25, height], [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [2., 2., 2.]]
    velocities = [list(velocity), [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [9., 8., 7.]]
    builder.add_particles(points, velocities, [2., 1., 1., 1., 0.], radius=[0.] * 5, flags=[1, 0, 0, 0, 1])
    builder.add_triangles([1], [2], [3], tri_ke=[0.], tri_ka=[0.], tri_kd=[0.])
    # Stock cloth has edge banks; include harmless boundary rows so the shared
    # metadata bridge receives the same array-bearing model structure.
    builder.add_edges([-1] * 3, [-1] * 3, [1, 2, 3], [2, 3, 1],
                      rest=[0.] * 3, edge_ke=[0.] * 3, edge_kd=[0.] * 3)
    builder.color()
    model = builder.finalize(device=device)
    attach_stock_ipc(model, fps=10, substeps=1, iterations=1, gs_predictor=predictor)
    model.ipc.params.update(kpin=pin, colored_ccd_guess_fraction=fraction, contact_aware_gs=contact_aware)
    if pin:
        targets = model.particle_q.numpy(); targets[0] += [2.75, -2.25, 1.]
        model.ipc.pin_targets.assign(targets)
        model.ipc.pinned.assign([1, 0, 0, 0, 0])
    if with_sdf:
        records = [{"kind": "plane", "point": [0., 0., 0.], "axis": [0., 0., 1.]}]
        model.ipc.collider_data = wp.array(sdf.pack_colliders(records), float, device=model.device)
        model.ipc.collider_records = records
        model.ipc.params.update(k_sdf=1.e6, eps_sdf=.5)
    return model

def solver(model, **options):
    return SolverIPC(model, allow_cpu=True, enable_collisions=False, **options)

class ForbiddenCollisionObject:
    def __getattr__(self, name):
        raise AssertionError("Collision object accessed: " + name)

class GsCollisionToggleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_no_collision_workspaces_or_kernels_and_material_schedule_is_reused(self):
        native = __import__(SolverVBD.__module__, fromlist=["SolverVBD"])
        for device in self.devices:
            model = fixture(device, pin=80., with_sdf=True)
            with ExitStack() as guards:
                for owner, name in ((native, "TriMeshCollisionDetector"), (contact, "ContactBackendIPC"),
                                    (candidate, "CcdBackend"), (candidate, "SdfBackend"),
                                    (candidate, "MovementBoxes"), (candidate, "QueryGroups")):
                    guards.enter_context(patch.object(owner, name, side_effect=AssertionError(name + " created")))
                own = solver(model)
            self.assertIs(SolverIPC.step, SolverVBD.step)
            for name in ("contact_backend", "ccd_backend", "sdf_backend", "movement_boxes", "_box_filter",
                         "trimesh_collision_detector", "trimesh_collision_info"):
                self.assertIsNone(getattr(own, name, None), name)
            self.assertIsNotNone(own._gs_schedule)
            self.assertIsNone(own._gs_schedule.contacts)
            self.assertIsNone(own._gs_schedule.query_groups)
            self.assertFalse(own._guess_requested)
            self.assertTrue(own._unrestricted_guess)
            for name in ("enable_collisions", "collision_processing", "mesh_collision_detection", "ccd_enabled",
                         "predictor_ccd_enabled", "movement_bounds_enabled", "planar_dat_enabled"):
                self.assertIs(own.configuration[name], False)
            self.assertFalse(own.configuration["energy_contact"]["enabled"])
            self.assertFalse(own.configuration["energy_sdf"]["enabled"])
            for slot in SolverBase.CollisionSlot:
                self.assertEqual(own.collision_frequency_type[slot], SolverBase.CollisionFrequencyType.NONE)
            state, out = model.state(), model.state()
            own.collision_pipeline = own._pipeline_contacts = ForbiddenCollisionObject()
            own.particle_enable_self_contact = own.rigid_soft_enable_dat = True
            with ExitStack() as guards:
                for owner, name in ((own, "_collision_detection_penetration_free"),
                                    (own._gs_schedule, "rebuild"), (SolverVBD, "_penetration_free_truncation"),
                                    (SolverVBD, "_finish_particle_color")):
                    guards.enter_context(patch.object(owner, name, side_effect=AssertionError(name + " invoked")))
                launches = guards.enter_context(patch.object(wp, "launch", wraps=wp.launch))
                for _ in range(2):
                    own.step(state, out, model.control(), ForbiddenCollisionObject(), .1)
                    state, out = out, state
                own.rebuild_bvh(state)
            own.check_status()
            names = [getattr(call.kwargs.get("kernel", call.args[0] if call.args else None), "key", "")
                     for call in launches.call_args_list]
            self.assertTrue(all(not any(word in name.lower() for word in ("contact", "ccd", "truncation", "box_"))
                                for name in names), names)
            self.assertTrue(any("publish_unrestricted" in name for name in names))
            self.assertTrue(any("solve" in name for name in names))

    def test_free_and_finite_pin_closed_forms_exceed_old_box_cap(self):
        for device in self.devices:
            for pin in (0., 80.):
                for contact_aware in (False, True):
                    with self.subTest(device=str(device), pin=pin, contact_aware=contact_aware):
                        model = fixture(device, pin=pin, contact_aware=contact_aware)
                        own = solver(model)
                        state, out = model.state(), model.state()
                        initial = state.particle_q.numpy().copy()
                        inertia = initial[0] + .1 * state.particle_qd.numpy()[0]
                        expected = (2. * inertia + .01 * pin * model.ipc.pin_targets.numpy()[0]) / (2. + .01 * pin)
                        own.step(state, out, model.control(), None, .1)
                        own.check_status()
                        np.testing.assert_allclose(out.particle_q.numpy()[0], expected, rtol=3.e-6, atol=2.e-7)
                        np.testing.assert_allclose(out.particle_qd.numpy()[0], (expected - initial[0]) / .1,
                                                   rtol=3.e-6, atol=2.e-6)
                        self.assertGreater(np.linalg.norm(expected - initial[0]), 100. * model.ipc.params["node_box_max"])
                        np.testing.assert_array_equal(out.particle_q.numpy()[1:], initial[1:])

    def test_unrestricted_velocity_predictor_fraction_and_zero_iteration_publication(self):
        for device in self.devices:
            for mode, fraction, sweeps in (("none", 1., 1), ("global", 1., 1), ("colored", 1., 1),
                                           ("colored", .25, 1), ("level", .5, 1), ("colored", 1., 0)):
                with self.subTest(device=str(device), mode=mode, fraction=fraction, sweeps=sweeps):
                    model = fixture(device, predictor=mode, fraction=fraction)
                    model.ipc.params["colored_ccd_guess_iters"] = sweeps
                    own = solver(model, iterations=0)
                    state, out = model.state(), model.state()
                    initial = state.particle_q.numpy().copy()
                    expected = initial.copy()
                    if mode != "none" and sweeps:
                        expected[0] += fraction * (.1 * state.particle_qd.numpy()[0])
                    own.step(state, out, model.control(), None, .1)
                    own.check_status()
                    np.testing.assert_allclose(out.particle_q.numpy(), expected, rtol=3.e-6, atol=2.e-7)
                    np.testing.assert_allclose(out.particle_qd.numpy(), (expected - initial) / .1,
                                               rtol=3.e-6, atol=2.e-6)
                    if mode == "colored" and fraction == 1. and sweeps:
                        np.testing.assert_array_equal(out.particle_q.numpy(), own.inertia.numpy())

    def test_mesh_friction_sdf_and_crossing_restrictions_are_absent(self):
        for device in self.devices:
            for velocity in ((0., 0., 0.), (3., 0., -2.)):
                model = fixture(device, with_sdf=True, velocity=velocity)
                model.ipc.params.update(k_barrier=1.e8, friction_coefficient=1.e4)
                own = solver(model)
                state, out = model.state(), model.state()
                initial = state.particle_q.numpy().copy()
                expected = initial.copy(); expected[0] += .1 * np.asarray(velocity)
                own.step(state, out, model.control(), None, .1)
                own.check_status()
                np.testing.assert_allclose(out.particle_q.numpy(), expected, rtol=3.e-6, atol=2.e-7)

    def test_material_terms_remain_active_with_dynamic_and_static_color_schedules(self):
        for device in self.devices:
            for membrane in (True, False):
                results = []
                for contact_aware in (False, True):
                    builder = newton.ModelBuilder(gravity=(0., 0., 0.))
                    points = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.]]
                    builder.add_particles(points, [[0., 0., 0.]] * 4, [1.] * 4,
                                          radius=[0.] * 4, flags=[0, 0, 0, 1])
                    builder.add_triangles([0, 1], [1, 0], [2, 3],
                        tri_ke=[9. if membrane else 0.] * 2, tri_ka=[13. if membrane else 0.] * 2, tri_kd=[0.] * 2)
                    builder.add_edges([2], [3], [0], [1], rest=[0.], edge_ke=[0. if membrane else .7], edge_kd=[0.])
                    builder.color()
                    model = builder.finalize(device=device)
                    receipt = attach_stock_ipc(model, fps=5, substeps=1, iterations=1, gs_predictor="none")
                    self.assertTrue(receipt["existing_arrays_unchanged"])
                    model.ipc.params["contact_aware_gs"] = contact_aware
                    own = solver(model)
                    state, out = model.state(), model.state()
                    initial = state.particle_q.numpy().copy(); initial[3, 2] = .25
                    state.particle_q.assign(initial)
                    own.step(state, out, model.control(), None, .2)
                    own.check_status()
                    result = out.particle_q.numpy()
                    self.assertGreater(np.linalg.norm(result[3] - initial[3]), 1.e-4)
                    self.assertLess(abs(result[3, 2]), abs(initial[3, 2]))
                    np.testing.assert_array_equal(result[:3], initial[:3])
                    results.append(result)
                np.testing.assert_allclose(*results, rtol=5.e-5, atol=2.e-6)

    def test_invalid_modes_and_nonfinite_state_fail_explicitly(self):
        model = fixture("cpu")
        for value in (None, 0, 1, "false", np.bool_(False)):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "enable_collisions"):
                SolverIPC(model, allow_cpu=True, enable_collisions=value)
        from types import SimpleNamespace
        with self.assertRaisesRegex(ValueError, "deformables only"):
            solver(SimpleNamespace(body_count=1))
        for device in self.devices:
            model = fixture(device, predictor="none")
            own = solver(model, iterations=0)
            state, out = model.state(), model.state()
            positions = state.particle_q.numpy(); positions[0, 0] = np.nan
            state.particle_q.assign(positions)
            own.step(state, out, model.control(), None, .1)
            with self.assertRaisesRegex(RuntimeError, "status"):
                own.check_status()
            with self.assertRaisesRegex(RuntimeError, "invalidated"):
                own.check_status()

    def test_cuda_capture_replays_the_same_material_schedule_and_motion(self):
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA device")
        from gpu.run import FrameGraph
        for device in wp.get_cuda_devices():
            for substeps in (1, 2, 3):
                results = []
                for graphs in (False, True):
                    model = fixture(device, pin=80.)
                    model.ipc.params.update(substeps=substeps, dt=.1 / substeps)
                    own = solver(model)
                    controller = FrameGraph(model, own, graphs=graphs)
                    with patch.object(own._gs_schedule, "rebuild", side_effect=AssertionError("Material graph rebuilt")):
                        for _ in range(3):
                            controller.advance()
                    results.append((controller.state.particle_q.numpy(), controller.state.particle_qd.numpy()))
                for eager, captured in zip(*results):
                    np.testing.assert_array_equal(captured, eager)


if __name__ == "__main__":
    unittest.main()
