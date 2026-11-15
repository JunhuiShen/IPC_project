"""Production GS material proposals use DAT's energy/PSD-block policy."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from gpu import dat_energies as dat
from gpu import solver as gs
from gpu.scene import create_model
from .test_model import cloth_payload
from .test_solver import point_payload
from .test_gs_schedule import dynamic_arguments


@wp.kernel(module="unique", enable_backward=False)
def dat_material_terms(
    dt: float, previous: wp.array[wp.vec3], q: wp.array[wp.vec3],
    triangles: wp.array2d[int], poses: wp.array[wp.mat22], materials: wp.array2d[float], areas: wp.array[float],
    edges: wp.array2d[int], angles: wp.array[float], lengths: wp.array[float], bending: wp.array2d[float],
    tets: wp.array2d[int], tet_poses: wp.array[wp.mat33], tet_materials: wp.array2d[float],
    adjacency: dat.MeshAdjacencyData, own_materials: wp.array2d[float], coefficients: wp.array[float],
    stiffness: wp.array[float], gradients: wp.array2d[wp.vec3], measures: wp.array[float],
    forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33],
):
    vertex = wp.tid()
    force, hessian = dat.accumulate_elasticity_force_and_hessian(
        vertex, dt, previous, q, triangles, poses, materials, areas, edges, angles, lengths, bending,
        tets, tet_poses, tet_materials, adjacency, wp.vec3(0.0), wp.mat33(0.0),
        own_materials, coefficients, stiffness, gradients, measures)
    forces[vertex] = force
    hessians[vertex] = hessian


class GsMaterialParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def fixture(self, device, *, inactive_original=False, contact_aware=True):
        payload = cloth_payload()
        parameters = point_payload()["params"]
        parameters.update(payload["params"])
        parameters.update(gravity=[0, 0, 0], damping=.73, kpin=7.,
                          use_ccd_guess=False, use_colored_ccd_guess=False,
                          use_colored_ccd_guess_level_based=False, contact_aware_gs=contact_aware)
        payload["params"] = parameters
        model = create_model(payload, device)
        # Separate DAT energy banks must win over the original model slots.
        model.ipc.dat_triangle_materials = wp.array([[17.3, 23.1], [9.2, 13.1]], float, device=device)
        coefficient = model.ipc.edge_coefficient.numpy()
        stiffness = np.where(coefficient > 0, .23, 0).astype(np.float32)
        model.ipc.dat_bending_stiffness = wp.array(stiffness, float, device=device)
        if inactive_original:
            original = model.tri_materials.numpy()
            original[0, :2] = 0.
            model.tri_materials.assign(original)
            original_bending = model.edge_bending_properties.numpy()
            original_bending[:, 0] = 0.
            model.edge_bending_properties.assign(original_bending)
        solver = gs.SolverIPC(model, iterations=1, allow_cpu=True)
        state = model.state()
        q = state.particle_q.numpy()
        q[1] += [.12, .07, -.03]
        q[2] += [-.05, .13, .08]
        q[3] += [.09, -.11, .04]
        state.particle_q.assign(q)
        solver.particle_q_prev.assign(q + np.array([.2, -.1, .3], np.float32))
        solver.inertia.assign(q + np.array([.03, -.02, .01], np.float32))
        solver.particle_forces.assign(np.repeat([[.07, -.05, .02]], 4, axis=0).astype(np.float32))
        solver.particle_hessians.assign(np.repeat((.4 * np.eye(3))[None], 4, axis=0).astype(np.float32))
        return model, solver, state

    def expected(self, model, solver, state, dt):
        data = solver._energy_materials
        force = wp.empty(4, wp.vec3, device=model.device)
        hessian = wp.empty(4, wp.mat33, device=model.device)
        wp.launch(dat_material_terms, 4,
            [dt, solver.particle_q_prev, state.particle_q,
             model.tri_indices, model.tri_poses, model.tri_materials, model.tri_areas,
             model.edge_indices, model.edge_rest_angle, model.edge_rest_length, model.edge_bending_properties,
             model.tet_indices, model.tet_poses, model.tet_materials, solver.particle_adjacency,
             data.triangle_materials, data.edge_coefficients, data.bending_stiffness,
             data.tet_gradients, data.tet_measures, force, hessian], device=model.device)
        q = state.particle_q.numpy().astype(np.float64)
        f = force.numpy().astype(np.float64) + solver.particle_forces.numpy()
        h = hessian.numpy().astype(np.float64) + solver.particle_hessians.numpy()
        mass = model.particle_mass.numpy()
        targets, pins = data.pin_targets.numpy(), data.pin_mask.numpy()
        expected = q.copy()
        for vertex in range(4):
            f[vertex] += mass[vertex] / dt**2 * (solver.inertia.numpy()[vertex] - q[vertex])
            h[vertex] += mass[vertex] / dt**2 * np.eye(3)
            if pins[vertex]:
                f[vertex] -= data.pin_stiffness * (q[vertex] - targets[vertex])
                h[vertex] += data.pin_stiffness * np.eye(3)
            expected[vertex] += solver.params["damping"] * np.linalg.solve(h[vertex], f[vertex])
        return expected

    def test_production_static_color_proposals_match_DAT_material_banks_and_psd_blocks(self):
        for device in self.devices:
            with self.subTest(device=str(device)):
                model, solver, state = self.fixture(device)
                dt = .075
                expected = self.expected(model, solver, state, dt)
                self.assertIs(solver._energy_materials.triangle_materials, model.ipc.dat_triangle_materials)
                self.assertIs(solver._energy_materials.bending_stiffness, model.ipc.dat_bending_stiffness)
                self.assertIs(solver._energy_materials.pin_targets, model.ipc.pin_targets)
                for color in range(len(model.particle_color_groups)):
                    solver._solve_particle_elasticity_color(state, model.state(), None, dt, color)
                np.testing.assert_allclose(solver.proposals.numpy(), expected, rtol=3e-6, atol=3e-6)
                self.assertEqual(int(solver.error.numpy()[0]), 0)

    def test_dynamic_and_tiled_kernels_match_DAT_on_the_same_frozen_snapshot(self):
        for device in self.devices:
            model, solver, state = self.fixture(device)
            data = solver._energy_materials
            dt = .075
            expected = self.expected(model, solver, state, dt)
            order = wp.array([3, 0, 2, 1], int, device=device)
            arguments = [dt, order, solver.particle_q_prev, state.particle_q, model.particle_mass,
                solver.inertia, model.particle_flags, model.tri_indices, model.tri_poses,
                solver._energy_triangle_materials, model.tri_areas, model.edge_indices, model.edge_rest_angle,
                model.edge_rest_length, solver._energy_bending, model.tet_indices, model.tet_poses,
                model.tet_materials, solver.particle_adjacency, solver.particle_forces,
                solver.particle_hessians, data.pin_targets, data.pin_mask, data.pin_stiffness,
                data.edge_coefficients, data.tet_gradients, data.tet_measures, solver.params["damping"],
                solver.particle_displacements, solver.proposals, solver.error]
            offsets = wp.array([0, 4], int, device=device)
            current = wp.zeros(1, int, device=device)
            dynamic = gs.make_dynamic_elasticity(True, False, device.is_cuda)
            wp.launch(dynamic, 64 if device.is_cuda else 4,
                dynamic_arguments(arguments, order, offsets, current),
                block_dim=32 if device.is_cuda else 64, device=device)
            np.testing.assert_allclose(solver.proposals.numpy(), expected, rtol=3e-6, atol=3e-6)
            self.assertEqual(int(solver.error.numpy()[0]), 0)
            if device.is_cuda:
                solver.proposals.fill_(wp.vec3(-123.))
                tiled = gs.make_solve_elasticity_tile(True, False, True)
                wp.launch(tiled, 64, arguments, block_dim=32, device=device)
                np.testing.assert_allclose(solver.proposals.numpy(), expected, rtol=3e-6, atol=3e-6)

    def test_inactive_original_elements_stay_inactive_after_gpu_energy_bank_updates(self):
        for device in self.devices:
            for contact_aware in (False, True):
                with self.subTest(device=str(device), contact_aware=contact_aware):
                    model, solver, state = self.fixture(device, inactive_original=True, contact_aware=contact_aware)
                    original_materials = model.tri_materials.numpy().copy()
                    original_bending = model.edge_bending_properties.numpy().copy()
                    own_materials = model.ipc.dat_triangle_materials
                    own_stiffness = model.ipc.dat_bending_stiffness
                    self.assertIs(solver._energy_materials.triangle_materials, own_materials)
                    self.assertIs(solver._energy_materials.bending_stiffness, own_stiffness)
                    self.assertTrue(solver._mask_triangle_materials)
                    self.assertTrue(solver._mask_bending_materials)
                    if contact_aware:
                        solver._gs_schedule.rebuild()
                    for factor in (1., 2.3):
                        own_materials.assign(np.array([[17.3, 23.1], [9.2, 13.1]], np.float32) * factor)
                        own_stiffness.fill_(.23 * factor)
                        # Isolate assembly/proposal from acceptance. The real
                        # iteration must refresh views after GPU bank writes.
                        with patch.object(solver.ccd_backend, "accept"):
                            solver._solve_particle_iteration(state, model.state(), None, .075)
                        expected = self.expected(model, solver, state, .075)
                        np.testing.assert_allclose(solver.proposals.numpy(), expected, rtol=3e-6, atol=3e-6)
                        np.testing.assert_array_equal(solver._energy_triangle_materials.numpy()[0], [0., 0.])
                        np.testing.assert_array_equal(solver._energy_bending_stiffness.numpy(), 0.)
                        np.testing.assert_array_equal(model.tri_materials.numpy(), original_materials)
                        np.testing.assert_array_equal(model.edge_bending_properties.numpy(), original_bending)
                        self.assertEqual(int(solver.error.numpy()[0]), 0)

    def test_retained_legacy_adapter_initializes_and_refreshes_shared_material_banks(self):
        from newton.solvers import SolverVBD
        from gpu.reference.legacy_dat_adapter import SolverDAT

        self.assertIs(SolverDAT._solve_particle_iteration, SolverVBD._solve_particle_iteration)
        for device in self.devices:
            model, original_solver, state = self.fixture(device, inactive_original=True)
            solver = SolverDAT(model, iterations=1, allow_cpu=True)
            solver.inertia.assign(original_solver.inertia)
            solver.particle_q_prev.assign(original_solver.particle_q_prev)
            own_materials = model.ipc.dat_triangle_materials
            own_stiffness = model.ipc.dat_bending_stiffness
            self.assertIs(solver._energy_materials.triangle_materials, own_materials)
            self.assertIs(solver._energy_materials.bending_stiffness, own_stiffness)
            for factor in (1., 2.3):
                own_materials.assign(np.array([[17.3, 23.1], [9.2, 13.1]], np.float32) * factor)
                own_stiffness.fill_(.23 * factor)
                # The unchanged legacy iteration must prepare the shared banks
                # before its local proposals, independently of DAT acceptance.
                with patch.object(solver, "_finish_particle_color"):
                    solver._solve_particle_iteration(state, model.state(), None, .075)
                expected = self.expected(model, solver, state, .075)
                np.testing.assert_allclose(solver.proposals.numpy(), expected, rtol=3e-6, atol=3e-6)
                np.testing.assert_array_equal(solver._energy_triangle_materials.numpy()[0], [0., 0.])
                np.testing.assert_array_equal(solver._energy_bending_stiffness.numpy(), 0.)
                self.assertEqual(int(solver.error.numpy()[0]), 0)


gs.energy.configure_module(dat_material_terms)
gs.energy.configure_module(__name__)

if __name__ == "__main__":
    unittest.main()
