"""The scene CCD flag must leave project contact and predictor ordering alone."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from gpu.contact import ContactBackendIPC
from gpu.ccd import get_plane_kernels
from gpu.scene import create_model
from gpu.solver import SolverIPC

from .test_collision import CollisionFixture
from .test_solver_dat import point_over_triangle


class ContactPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_scene_ccd_flag_preserves_project_gs_and_predictor(self):
        for device in self.devices:
            results = []
            for exact in (False, True):
                with self.subTest(device=str(device), ccd_exact_fallback=exact):
                    payload = point_over_triangle(height=.002, velocity=(.02, 0, -.15),
                                                  iterations=2, friction=.2)
                    params = payload["params"]
                    params.update(exact_computation_fallback=exact, ccd_method="direct_rounding",
                                  gs_ccd_accept_uncertified=True, gs_ccd_guess_accept_uncertified=True,
                                  use_ccd_guess=False, use_colored_ccd_guess=False,
                                  use_colored_ccd_guess_level_based=True,
                                  colored_ccd_guess_iters=1, colored_ccd_guess_fraction=1.)
                    model = create_model(payload, device)
                    solver = SolverIPC(model, allow_cpu=True)
                    self.assertEqual(solver.configuration["energy_contact"]["energy_model"], "project")
                    self.assertEqual(solver.configuration["energy_contact"]["scalar_precision"], "float64")
                    self.assertEqual(solver.ccd_backend.method, "direct_rounding")
                    self.assertEqual(solver.ccd_backend.exact_geometry, exact)
                    self.assertFalse(solver.configuration["gs_ccd_accept_uncertified"])
                    self.assertFalse(solver.configuration["gs_ccd_guess_accept_uncertified"])
                    before, after = model.state(), model.state()
                    _, verify, _ = get_plane_kernels("direct_rounding")
                    with patch.object(wp, "launch", wraps=wp.launch) as launch:
                        solver.step(before, after, model.control(), None, params["dt"])
                    solver.check_status()
                    self.assertEqual(any(call.args and call.args[0] is verify
                                         for call in launch.call_args_list), exact)
                    graph = solver.ccd_backend._initial_guess.graph
                    self.assertGreater(int(graph.incidence.numpy().max()), 0)
                    results.append((after.particle_q.numpy(), after.particle_qd.numpy(),
                                    graph.keys.numpy()[:model.particle_count]))
            for off, on in zip(*results):
                np.testing.assert_array_equal(off, on)

    def test_exact_contact_replay_is_always_available(self):
        for device in self.devices:
            outputs = []
            for previous_contact_flag in (False, True):
                q, model, info, _ = CollisionFixture().fixture(device, z=1.e-12, friction=.2)
                backend = ContactBackendIPC(model, info, d_hat=.005, k_barrier=100,
                                            friction=.2, exact_geometry=previous_contact_flag)
                backend.refresh()
                f = wp.zeros(4, wp.vec3, device=device)
                h = wp.zeros(4, wp.mat33, device=device)
                backend.accumulate(q, q, 0, f, h, .01)
                backend.check()
                self.assertEqual(int(backend.pending_count.numpy()[0]), 1)
                self.assertGreater(float(f.numpy()[0, 2]), 0.)
                self.assertGreater(float(h.numpy()[0, 2, 2]), 0.)
                outputs.append((f.numpy(), h.numpy()))
                # A new call rebuilds the queue from the current positions.
                q[0:1].assign(wp.array([[0, 0, 1]], wp.vec3, device=device))
                f.zero_()
                h.zero_()
                backend.accumulate(q, q, 0, f, h, .01)
                backend.check()
                self.assertEqual(int(backend.pending_count.numpy()[0]), 0)
                np.testing.assert_array_equal(f.numpy(), 0.)
                np.testing.assert_array_equal(h.numpy(), 0.)
            for off, on in zip(*outputs):
                np.testing.assert_array_equal(off, on)
