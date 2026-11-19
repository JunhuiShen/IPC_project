"""Live IPC/friction assembly on the same pair/color schedule as color CCD.

Run in a fresh process with the reference Newton runtime.  The pair evaluators
are checked separately from their new indexing, ownership and color reduction.
"""
import unittest

import numpy as np
import warp as wp
import gpu

gpu.select_newton_runtime("reference")

from gpu import contact, dat_energies as energies
from gpu.dat_color_contacts import ColorContactForces
from gpu.dat_tests.test_color_ccd import workspace
from gpu.dat_tests.test_level_ccd_guess import fixture, advance
from gpu.solver_dat import SolverDAT


@wp.kernel(module="unique", enable_backward=False)
def _reference_triangle(q: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                        triangles: wp.array2d[int], context: energies.EnergyContext,
                        forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    nodes = wp.vec4i(0, triangles[0, 0], triangles[0, 1], triangles[0, 2])
    value = contact.evaluate_self(q, previous, nodes, 0, context.parameters)
    if int(value[48]) != 0:
        wp.atomic_max(context.error, 0, 100 + int(value[48]))
    elif value[49] != 0.0:
        for role in range(4):
            forces[nodes[role]] = wp.vec3(value[12 * role], value[12 * role + 1], value[12 * role + 2])
            h = wp.mat33(0.0)
            for row in range(3):
                for column in range(3):
                    h[row, column] = value[12 * role + 3 + 3 * row + column]
            hessians[nodes[role]] = h


@wp.kernel(module="unique", enable_backward=False)
def _reference_edges(q: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                     edges: wp.array2d[int], context: energies.EnergyContext,
                     forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    nodes = wp.vec4i(edges[0, 2], edges[0, 3], edges[1, 2], edges[1, 3])
    value = contact.evaluate_self(q, previous, nodes, 1, context.parameters)
    if int(value[48]) != 0:
        wp.atomic_max(context.error, 0, 100 + int(value[48]))
    elif value[49] != 0.0:
        for role in range(4):
            forces[nodes[role]] = wp.vec3(value[12 * role], value[12 * role + 1], value[12 * role + 2])
            h = wp.mat33(0.0)
            for row in range(3):
                for column in range(3):
                    h[row, column] = value[12 * role + 3 + 3 * row + column]
            hessians[nodes[role]] = h


contact.configure_module(_reference_triangle)
contact.configure_module(_reference_edges)


def _scene(device, kind="PT", gap=.001):
    if kind == "PT":
        points = [[.25, .25, gap], [0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
        triangles, edges = [[1, 2, 3]], []
    else:
        points = [[-1., 0., gap], [1., 0., gap], [0., -1., 0.], [0., 1., 0.]]
        triangles, edges = [], [[-1, -1, 0, 1], [-1, -1, 2, 3]]
    model, boxes, graph, ccd, error, _, _ = workspace(device, points, triangles, edges)
    assembly = ColorContactForces(model, graph, ccd, error, k_barrier=100.)
    assembly.refresh()
    previous = wp.clone(model.particle_q)
    forces = wp.zeros(model.particle_count, wp.vec3, device=device)
    hessians = wp.zeros(model.particle_count, wp.mat33, device=device)
    return model, boxes, graph, ccd, error, assembly, previous, forces, hessians


def _assemble(parts, *, mu=.2, radius=.002):
    model, _, _, _, _, assembly, previous, forces, hessians = parts
    assembly.assemble(model.particle_q, previous, .01, radius, mu, .01, forces, hessians)
    return forces.numpy().copy(), hessians.numpy().copy()


def _oracle(parts, kind, *, mu=.2, radius=.002):
    model, _, _, _, error, _, previous, _, _ = parts
    reference = energies.EnergyWorkspace(None, device=model.device, k_barrier=100.,
                                        precision="float32", error=error)
    forces = wp.zeros(model.particle_count, wp.vec3, device=model.device)
    hessians = wp.zeros(model.particle_count, wp.mat33, device=model.device)
    context = reference.begin(.01, radius, mu, .01)
    wp.launch(_reference_triangle if kind == "PT" else _reference_edges,
              dim=1,
              inputs=[model.particle_q, previous,
                      model.tri_indices if kind == "PT" else model.edge_indices,
                      context], outputs=[forces, hessians], device=model.device)
    return forces.numpy().copy(), hessians.numpy().copy()


def _solver(model, forces, iterations=3):
    return SolverDAT(model, allow_cpu=True, iterations=iterations,
        enable_planar_dat=False, enable_broad_phase_boxes=True,
        enable_barrier_coloring=True, enable_color_ccd=True,
        enable_color_ccd_contact_forces=forces,
        vertex_contact_capacity=16, edge_contact_capacity=16,
        enable_fast_math=False)


class ColorContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_each_pt_and_ee_corner_matches_existing_force_and_hessian(self):
        for device in self.devices:
            for kind in ("PT", "EE"):
                with self.subTest(device=str(device), pair=kind):
                    parts = _scene(device, kind)
                    model, _, graph, _, error, _, previous, _, _ = parts
                    self.assertEqual(int(graph.candidate_count.numpy()[0]), 1)
                    anchor = model.particle_q.numpy().copy()
                    anchor[:1 if kind == "PT" else 2, 0] -= .001
                    previous.assign(anchor)
                    expected_f, expected_h = _oracle(parts, kind)
                    self.assertGreater(float(np.linalg.norm(expected_f)), 0.)
                    total = np.zeros_like(expected_f)
                    for vertex, color in enumerate(graph.colors.numpy()):
                        graph.current_color.fill_(int(color))
                        f, h = _assemble(parts)
                        np.testing.assert_allclose(f[vertex], expected_f[vertex], rtol=3.e-6, atol=1.e-6)
                        np.testing.assert_allclose(h[vertex], expected_h[vertex], rtol=3.e-6, atol=1.e-5)
                        other = np.arange(4) != vertex
                        np.testing.assert_array_equal(f[other], np.zeros((3, 3)))
                        np.testing.assert_array_equal(h[other], np.zeros((3, 3, 3)))
                        total += f
                    np.testing.assert_allclose(total.sum(axis=0), np.zeros(3), atol=2.e-6)
                    self.assertEqual(int(error.numpy()[0]), 0)

    def test_friction_opposes_sliding_and_uses_previous_substep_positions(self):
        for device in self.devices:
            for kind in ("PT", "EE"):
                with self.subTest(device=str(device), pair=kind):
                    parts = _scene(device, kind)
                    model, _, graph, _, error, _, previous, _, _ = parts
                    graph.current_color.fill_(int(graph.colors.numpy()[0]))
                    baseline, _ = _assemble(parts, mu=0.)
                    rest_friction, _ = _assemble(parts, mu=.2)
                    np.testing.assert_allclose(rest_friction, baseline, atol=1.e-7)
                    anchor = model.particle_q.numpy().copy()
                    anchor[:1 if kind == "PT" else 2, 0] -= .001
                    previous.assign(anchor)
                    sliding, _ = _assemble(parts, mu=.2)
                    self.assertLess(float(sliding[0, 0] - baseline[0, 0]), 0.)
                    np.testing.assert_allclose(sliding[0, 2], baseline[0, 2], rtol=2.e-6)
                    anchor[:1 if kind == "PT" else 2, 0] += .002
                    previous.assign(anchor)
                    reversed_sliding, _ = _assemble(parts, mu=.2)
                    self.assertGreater(float(reversed_sliding[0, 0] - baseline[0, 0]), 0.)
                    self.assertEqual(int(error.numpy()[0]), 0)

    def test_live_geometry_recomputed_and_inactive_pairs_clear_stale_forces(self):
        for device in self.devices:
            for kind in ("PT", "EE"):
                parts = _scene(device, kind)
                model, _, graph, _, error, _, _, forces, hessians = parts
                graph.current_color.fill_(int(graph.colors.numpy()[0]))
                original, _ = _assemble(parts, mu=0.)
                self.assertGreater(float(np.linalg.norm(original)), 0.)
                changed = model.particle_q.numpy().copy()
                changed[:1 if kind == "PT" else 2, 2] = .0015
                model.particle_q.assign(changed)
                live, live_h = _assemble(parts, mu=0.)
                expected_f, expected_h = _oracle(parts, kind, mu=0.)
                np.testing.assert_allclose(live[0], expected_f[0], rtol=3.e-6, atol=1.e-7)
                np.testing.assert_allclose(live_h[0], expected_h[0], rtol=3.e-6, atol=1.e-6)
                self.assertLess(float(np.linalg.norm(live)), float(np.linalg.norm(original)))
                changed[:1 if kind == "PT" else 2, 2] = .05
                model.particle_q.assign(changed)
                forces.fill_(wp.vec3(123.))
                hessians.fill_(wp.mat33(123.))
                inactive, inactive_h = _assemble(parts)
                np.testing.assert_array_equal(inactive, np.zeros((4, 3)))
                np.testing.assert_array_equal(inactive_h, np.zeros((4, 3, 3)))
                self.assertEqual(int(error.numpy()[0]), 0)

    def test_zero_gap_does_not_publish_nonfinite_contact_terms(self):
        for device in self.devices:
            for kind in ("PT", "EE"):
                parts = _scene(device, kind, gap=0.)
                graph, error = parts[2], parts[4]
                graph.current_color.fill_(int(graph.colors.numpy()[0]))
                f, h = _assemble(parts)
                self.assertTrue(np.isfinite(f).all() and np.isfinite(h).all())
                # An explicit evaluator error is permitted; silently
                # spreading NaNs to the local solve is not.
                if int(error.numpy()[0]) != 0:
                    np.testing.assert_array_equal(f, np.zeros_like(f))
                    np.testing.assert_array_equal(h, np.zeros_like(h))

    def test_multiple_pairs_accumulate_once_each_for_one_vertex(self):
        points = [[.25, .25, .001], [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                  [0., 0., -.0004], [1., 0., -.0004], [0., 1., -.0004]]
        for device in self.devices:
            results = []
            for triangles in ([[1, 2, 3]], [[4, 5, 6]], [[1, 2, 3], [4, 5, 6]]):
                model, boxes, graph, ccd, error, _, _ = workspace(
                    device, points, triangles, flags=[1, 0, 0, 0, 0, 0, 0])
                self.assertEqual(int(graph.candidate_count.numpy()[0]), len(triangles))
                assembly = ColorContactForces(model, graph, ccd, error, k_barrier=100.)
                assembly.refresh()
                previous = wp.clone(model.particle_q)
                forces = wp.zeros(7, wp.vec3, device=device)
                hessians = wp.zeros(7, wp.mat33, device=device)
                graph.current_color.fill_(int(graph.colors.numpy()[0]))
                f, h = _assemble((model, boxes, graph, ccd, error, assembly,
                                  previous, forces, hessians), mu=0.)
                results.append((f[0].copy(), h[0].copy()))
                self.assertEqual(int(error.numpy()[0]), 0)
            for term in (0, 1):
                np.testing.assert_allclose(results[2][term], results[0][term] + results[1][term],
                                           rtol=3.e-6, atol=1.e-6)

    def test_stale_graph_or_box_epoch_rejected_before_assembly(self):
        for device in self.devices:
            for mutation in ("graph", "boxes"):
                with self.subTest(device=str(device), mutation=mutation):
                    parts = _scene(device)
                    model, boxes, graph, _, error, _, _, _, _ = parts
                    if mutation == "graph":
                        graph.update()
                    else:
                        boxes.update(model.particle_q)
                    graph.current_color.fill_(int(graph.colors.numpy()[0]))
                    f, h = _assemble(parts)
                    self.assertNotEqual(int(error.numpy()[0]), 0)
                    np.testing.assert_array_equal(f, np.zeros_like(f))
                    np.testing.assert_array_equal(h, np.zeros_like(h))

    def test_solver_force_flag_requires_ccd_and_changes_near_contact_update(self):
        for device in self.devices:
            model = fixture(device, predictor=False, height=.001, velocity=(.05, 0., 0.))
            model.ipc.params.update(node_box_min=.01, node_box_max=.01,
                                    node_box_update_count=2, friction_coefficient=.2)
            with self.assertRaises(ValueError):
                SolverDAT(model, allow_cpu=True, enable_color_ccd_contact_forces=True)
            results = []
            for enabled in (False, True):
                own = _solver(model, enabled)
                _, after = advance(own, model, dt=.01)
                self.assertFalse(own.enable_planar_dat)
                results.append(after.particle_q.numpy().copy())
                self.assertGreater(float(results[-1][0, 2]), 0.)
            self.assertGreater(float(results[1][0, 2]), float(results[0][0, 2]))
            self.assertLess(float(results[1][0, 0]), float(results[0][0, 0]))

    def test_cuda_graph_matches_eager_for_repeated_force_enabled_steps(self):
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("No CUDA device")
        from gpu.run import FrameGraph
        for device in devices:
            outputs = []
            for captured in (False, True):
                model = fixture(device, predictor=True, height=.001,
                                velocity=(.05, 0., 0.), ccd_method="direct_rounding")
                model.ipc.params.update(fps=100, substeps=1, dt=.01, dt2=.0001,
                                        node_box_min=.01, node_box_max=.01,
                                        node_box_update_count=2, friction_coefficient=.2)
                own = _solver(model, True)
                frames = FrameGraph(model, own, graphs=captured)
                for _ in range(3):
                    frames.advance()
                    own.check_status()
                outputs.append((frames.state.particle_q.numpy(), frames.state.particle_qd.numpy()))
            for eager, captured in zip(*outputs):
                np.testing.assert_array_equal(eager, captured)


if __name__ == "__main__":
    unittest.main()
