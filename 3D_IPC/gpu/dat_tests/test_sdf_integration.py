"""Full-curvature SDF integration through the unchanged DAT controller."""
import unittest

import numpy as np
import warp as wp
import newton

from gpu.dat_stock_scene import attach_stock_ipc
from gpu.sdf import pack_colliders
from gpu.solver_dat import SolverDAT


def fixture(device):
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    builder.add_cloth_mesh(pos=wp.vec3(0), rot=wp.quat_identity(), scale=1.,
        vertices=[wp.vec3(1.015, 0, 0), wp.vec3(1.015, .05, 0), wp.vec3(1.015, 0, .05)],
        indices=[0, 1, 2], vel=wp.vec3(0, .01, 0), density=2400.,
        tri_ke=0., tri_ka=0., tri_kd=0., edge_ke=0., edge_kd=0.)
    builder.color()
    model = builder.finalize(device=device)
    attach_stock_ipc(model, gs_predictor='none')
    model.ipc.params.update(k_sdf=370., eps_sdf=.05, dt=1. / 600.,
                            friction_coefficient=.2, friction_velocity_epsilon=.01)
    model.ipc.collider_data = wp.array(pack_colliders([
        {'kind': 'sphere', 'point': [0, 0, 0], 'radius': 1., 'axis': [0, 1, 0]}
    ]), float, device=device)
    solver = SolverDAT(model, iterations=2, allow_cpu=True,
                       particle_enable_self_contact=False)
    return model, solver


class DatSdfIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_dat_defaults_to_full_curvature_and_resolves_sphere_contact(self):
        for device in [wp.get_device('cpu'), *wp.get_cuda_devices()]:
            model, solver = fixture(device)
            self.assertEqual(solver.sdf_backend.params.include_curvature, 1)
            self.assertEqual(solver.sdf_backend.precision, 'float32')
            before, after = model.state(), model.state()
            initial = before.particle_q.numpy().copy()
            for _ in range(10):
                solver.step(before, after, None, None, 1. / 600.)
                before, after = after, before
            solver.check_status()
            result = before.particle_q.numpy()
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(np.all(result[:, 0] > initial[:, 0]))

    def test_captured_curved_contact_matches_eager_steps(self):
        for device in wp.get_cuda_devices():
            expected = None
            for captured in (False, True):
                model, solver = fixture(device)
                a, b = model.state(), model.state()

                def pair():
                    solver.step(a, b, None, None, 1. / 600.)
                    solver.step(b, a, None, None, 1. / 600.)

                if captured:
                    with wp.ScopedCapture(device=device) as graph:
                        pair()
                    for _ in range(5):
                        wp.capture_launch(graph.graph)
                else:
                    for _ in range(5):
                        pair()
                solver.check_status()
                result = a.particle_q.numpy()
                if expected is None:
                    expected = result
                else:
                    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
