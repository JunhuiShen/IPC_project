"""GS contact assembly must retain native DAT forces and PSD approximations."""

from functools import cache
import unittest

import numpy as np
import warp as wp

from gpu import contact, dat_energies
from gpu.gs_contact_energies import make_full_evaluator
from gpu.query_groups import QueryGroups

from .test_movement_box_queries import fixture


def make_case(device, kind, points, *, precision="float32", friction=.2, slip=0.0,
              colors=(0, 1, 2, 3)):
    triangles = [[1, 2, 3]] if kind == 0 else []
    edges = [] if kind == 0 else [[-1, -1, 0, 1], [-1, -1, 2, 3]]
    model, original, detector = fixture(device, points, triangles, edges, capacity=1)
    model.particle_colors.assign(colors)
    info = original.info
    info.vertex_colliding_triangles_count.zero_()
    info.edge_colliding_edges_count.zero_()
    if kind == 0:
        rows = np.full(info.vertex_colliding_triangles.shape[0], -1, np.int32)
        rows[:2] = [0, 0]
        info.vertex_colliding_triangles.assign(rows)
        info.vertex_colliding_triangles_count.assign([1, 0, 0, 0])
    else:
        info.edge_colliding_edges.assign([0, 1, 1, 0])
        info.edge_colliding_edges_count.assign([1, 1])
    backend = contact.ContactBackendIPC(model, info, d_hat=.002, k_barrier=100.,
        friction=friction, eps_v=.01, grouped_precision=precision, energy_model="dat",
        track_query_epoch=True)
    backend.refresh()
    previous = np.asarray(points, np.float32).copy()
    previous[0] -= np.asarray([slip, .3 * slip, 0], np.float32)
    history = wp.array(previous, wp.vec3, device=device)
    current = wp.zeros(1, int, device=device)
    groups = QueryGroups(backend)
    groups.build(model.particle_colors)
    view = groups.view(current)
    return model, backend, history, current, groups, view


def assemble(case, color, *, grouped=True, use_view=True):
    model, backend, previous, current, groups, view = case
    current.fill_(color)
    forces = wp.zeros(4, wp.vec3, device=model.device)
    hessians = wp.zeros(4, wp.mat33, device=model.device)
    if grouped:
        backend.accumulate(model.particle_q, previous, -1, forces, hessians, .01,
            colors=model.particle_colors, current_color=current, query_view=view if use_view else None)
    else:
        backend.accumulate(model.particle_q, previous, color, forces, hessians, .01)
    return forces, hessians


def native_oracle(case, color):
    model, backend, previous, current, groups, view = case
    options = backend.options
    workspace = dat_energies.EnergyWorkspace(backend.info, device=model.device,
        precision=backend.grouped_precision, k_barrier=options["k_barrier"])
    ctx = workspace.begin(.01, options["d_hat"], options["friction"], options["eps_v"])
    forces = wp.zeros(4, wp.vec3, device=model.device)
    hessians = wp.zeros(4, wp.mat33, device=model.device)
    wp.launch(workspace.kernel, 128,
        [.01, color, previous, model.particle_q, model.particle_colors, model.tri_indices,
         model.edge_indices, backend.info, options["d_hat"], 999., 0., options["friction"],
         options["eps_v"], True, 1e-5, ctx, forces, hessians], device=model.device)
    workspace.check()
    return forces.numpy(), hessians.numpy()


@cache
def full_probe(precision):
    evaluator = make_full_evaluator(precision)

    @wp.kernel(module="unique", enable_backward=False)
    def probe(q: wp.array[wp.vec3], previous: wp.array[wp.vec3], kind: int,
              params: contact.Parameters, result: wp.array[contact.FullPacket]):
        result[0] = evaluator(q, previous, wp.vec4i(0, 1, 2, 3), kind, params)

    dat_energies.configure_module(probe)
    return probe


class GsDatContactParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_static_grouped_and_query_spans_match_native_dat_all_roles_and_friction(self):
        fixtures = [
            (0, [[.2, .2, .001], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            (0, [[.4, -.0006, .0008], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            (0, [[-.0006, -.0006, .0006], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            (1, [[0, 0, 0], [1, 0, 0], [.5, -.5, .001], [.5, .5, .001]]),
            (1, [[0, 0, 0], [1, 0, 0], [.1, .001, 0], [.9, .001, 0]]),
            (1, [[0, 0, 0], [1, 0, 0], [.8, .001, 0], [.2, .001, 0]]),
        ]
        for device in self.devices:
            for precision in ("float32", "float64"):
                for kind, points in fixtures:
                    for friction, slip in ((0., 0.), (.2, 0.), (.2, 2e-5), (.2, .003)):
                        case = make_case(device, kind, points, precision=precision, friction=friction, slip=slip)
                        for color in range(4):
                            expected = native_oracle(case, color)
                            for grouped, view in ((False, False), (True, False), (True, True)):
                                with self.subTest(device=str(device), kind=kind, color=color,
                                                  precision=precision, friction=friction, slip=slip,
                                                  grouped=grouped, view=view):
                                    actual = assemble(case, color, grouped=grouped, use_view=view)
                                    case[1].check()
                                    for array, oracle in zip(actual, expected):
                                        np.testing.assert_allclose(array.numpy(), oracle, rtol=2e-6, atol=2e-7)
                                    hessian = actual[1].numpy()[color].astype(np.float64)
                                    self.assertGreaterEqual(np.linalg.eigvalsh(hessian).min(), -2e-5 * max(1., np.linalg.norm(hessian)))

    def test_full_world_packet_diagonals_match_native_dat_assembly(self):
        for device in self.devices:
            for precision in ("float32", "float64"):
                for kind, points in (
                    (0, [[.2, .2, .001], [0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                    (1, [[0, 0, 0], [1, 0, 0], [.5, -.5, .001], [.5, .5, .001]]),
                ):
                    for slip in (0., 2e-5, .003):
                        case = make_case(device, kind, points, precision=precision, slip=slip, colors=(0, 0, 0, 0))
                        expected_f, expected_h = native_oracle(case, 0)
                        result = wp.empty(1, contact.FullPacket, device=device)
                        params = contact.parameters(dt=.01, **case[1].options)
                        wp.launch(full_probe(precision), 1, [case[0].particle_q, case[2], kind, params, result], device=device)
                        packet = result.numpy()[0]
                        self.assertEqual(packet[176], 0.)
                        self.assertEqual(packet[177], 1.)
                        force = np.zeros((4, 3))
                        hessian = np.zeros((4, 3, 3))
                        for role in range(4):
                            weight = packet[1 + role]
                            force[role] = -100. * packet[8 + 3 * role:11 + 3 * role] - weight * packet[164:167] / .01**2
                            start = 20 + 45 * role
                            hessian[role] = 100. * packet[start:start + 9].reshape(3, 3) + weight**2 * packet[167:176].reshape(3, 3) / .01**2
                        np.testing.assert_allclose(force, expected_f, rtol=3e-5, atol=2e-7)
                        np.testing.assert_allclose(hessian, expected_h, rtol=3e-5, atol=2e-6)

    def test_parallel_ee_full_packet_matches_each_source_edge_and_is_psd(self):
        # A reverse native closest-point query can select a different point
        # from this continuum of minimizers. Rigid assembly orients its own
        # edge first; assert the source side in BOTH orientations instead of
        # requiring one frame to equal both distinct directed native frames.
        points = np.asarray([[0, 0, 0], [1, 0, 0], [.8, .001, 0], [.2, .001, 0]], np.float32)
        for device in self.devices:
            for precision in ("float32", "float64"):
                for slip in (0., 2e-5, .003):
                    previous = points.copy()
                    previous[0] -= np.asarray([slip, .3 * slip, 0.], np.float32)
                    for permutation in ([0, 1, 2, 3], [2, 3, 0, 1]):
                        with self.subTest(device=str(device), precision=precision,
                                          slip=slip, permutation=permutation):
                            case = make_case(device, 1, points[permutation], precision=precision,
                                             colors=(0, 0, 0, 0))
                            case[2].assign(previous[permutation])
                            expected_f, expected_h = native_oracle(case, 0)
                            result = wp.empty(1, contact.FullPacket, device=device)
                            params = contact.parameters(dt=.01, **case[1].options)
                            wp.launch(full_probe(precision), 1,
                                [case[0].particle_q, case[2], 1, params, result], device=device)
                            packet = result.numpy()[0]
                            self.assertEqual(packet[176], 0.)
                            self.assertEqual(packet[177], 1.)
                            weights = packet[1:5]
                            gradients = packet[8:20].reshape(4, 3)
                            forces = -100. * gradients - weights[:, None] * packet[164:167] / .01**2
                            normal_blocks = packet[20:164].reshape(4, 4, 3, 3)
                            friction = packet[167:176].reshape(3, 3) / .01**2
                            blocks = (100. * normal_blocks
                                + weights[:, None, None, None] * weights[None, :, None, None] * friction)
                            diagonal = np.stack([blocks[role, role] for role in range(4)])
                            np.testing.assert_allclose(forces[:2], expected_f[:2], rtol=3e-5, atol=2e-7)
                            np.testing.assert_allclose(diagonal[:2], expected_h[:2], rtol=3e-5, atol=2e-6)
                            np.testing.assert_allclose(forces.sum(axis=0), 0., atol=2e-7)
                            matrix = blocks.transpose(0, 2, 1, 3).reshape(12, 12)
                            np.testing.assert_allclose(matrix, matrix.T, rtol=2e-6, atol=2e-7)
                            tolerance = 2e-6 * max(1., np.linalg.norm(matrix))
                            self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -tolerance)

    def test_zero_gap_nonfinite_history_inactive_roles_and_stale_epochs(self):
        points = np.asarray([[.2, .2, .001], [0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        for device in self.devices:
            case = make_case(device, 0, points, colors=(0, 0, 0, 0))
            case[0].particle_flags.assign([1, 0, 1, 1])
            case[0].particle_mass.assign([1, 1, 0, 1])
            forces, hessians = assemble(case, 0)
            case[1].check()
            np.testing.assert_array_equal(forces.numpy()[1:3], 0.)
            np.testing.assert_array_equal(hessians.numpy()[1:3], 0.)
            case[1].refresh()
            with self.assertRaises((ValueError, RuntimeError)):
                assemble(case, 0)
                case[1].check()
            for which in ("gap", "history"):
                case = make_case(device, 0, points)
                if which == "gap":
                    changed = points.copy()
                    # Exactly representable barycentrics avoid a tiny native
                    # float32 closest-point residual for decimal .2,.2.
                    changed[0] = [.25, .25, 0.]
                    case[0].particle_q.assign(changed)
                else:
                    changed = points.copy()
                    changed[3, 0] = np.nan
                    case[2].assign(changed)
                assemble(case, 0)
                with self.assertRaisesRegex(RuntimeError, "zero contact distance|invalid contact input"):
                    case[1].check()

    def test_cuda_capture_replay_uses_current_color_and_history(self):
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA device unavailable")
        points = np.asarray([[.2, .2, .001], [0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        for device in devices:
            case = make_case(device, 0, points, slip=.003)
            forces = wp.zeros(4, wp.vec3, device=device)
            hessians = wp.zeros(4, wp.mat33, device=device)
            model, backend, previous, current, groups, view = case
            def run():
                forces.zero_()
                hessians.zero_()
                backend.accumulate(model.particle_q, previous, -1, forces, hessians, .01,
                    colors=model.particle_colors, current_color=current, query_view=view)
            run()
            wp.synchronize_device(device)
            with wp.ScopedCapture(device=device) as capture:
                run()
            for color in (3, 0, 2, 1):
                current.fill_(color)
                previous.assign(points if color % 2 else points - np.asarray([.003, .0009, 0.], np.float32))
                expected = native_oracle(case, color)
                wp.capture_launch(capture.graph)
                backend.check()
                for actual, oracle in zip((forces, hessians), expected):
                    np.testing.assert_allclose(actual.numpy(), oracle, rtol=2e-6, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
