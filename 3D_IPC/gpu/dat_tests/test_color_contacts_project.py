"""Independent curvature checks for project per-color contact derivatives."""

import unittest

import numpy as np
import warp as wp
import gpu

gpu.select_newton_runtime("reference")

from gpu.dat_color_contacts import ColorContactForces
from gpu.dat_tests.test_color_ccd import workspace


class ProjectColorContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_vertex_feature_includes_distance_curvature(self):
        """b''nn^T alone would have zero transverse eigenvalues and fail."""
        points = np.float32([[-.0006, -.0007, .0003], [0., 0., 0.],
                             [1., 0., 0.], [0., 1., 0.]])
        radius, stiffness = float(np.float32(.002)), 100.
        displacement = points[0].astype(np.float64) - points[1].astype(np.float64)
        distance = np.linalg.norm(displacement)
        normal = displacement / distance
        t = distance - radius
        log = np.log(distance / radius)
        first = stiffness * (-2. * t * log - t * t / distance)
        second = stiffness * (-2. * log - 4. * t / distance + t * t / distance**2)
        expected_force = -first * normal
        expected_hessian = (second * np.outer(normal, normal)
                            + first / distance * (np.eye(3) - np.outer(normal, normal)))
        self.assertLess(np.linalg.eigvalsh(expected_hessian)[0], 0.)
        for device in self.devices:
            with self.subTest(device=str(device)):
                model, _, graph, ccd, error, _, _ = workspace(device, points, [[1, 2, 3]])
                assembly = ColorContactForces(model, graph, ccd, error, k_barrier=stiffness)
                assembly.refresh()
                self.assertEqual(assembly.configuration["geometry_and_derivative_precision"], "float64")
                self.assertEqual(assembly.configuration["assembled_storage_precision"], "float32")
                previous = wp.clone(model.particle_q)
                force = wp.zeros(4, wp.vec3, device=device)
                hessian = wp.zeros(4, wp.mat33, device=device)
                graph.current_color.fill_(int(graph.colors.numpy()[0]))
                assembly.assemble(model.particle_q, previous, .01, radius, 0., .01, force, hessian)
                self.assertEqual(int(error.numpy()[0]), 0)
                np.testing.assert_allclose(force.numpy()[0], expected_force, rtol=2.e-6, atol=1.e-7)
                np.testing.assert_allclose(hessian.numpy()[0], expected_hessian, rtol=2.e-6, atol=1.e-4)
                np.testing.assert_array_equal(force.numpy()[1:], np.zeros((3, 3)))
                np.testing.assert_array_equal(hessian.numpy()[1:], np.zeros((3, 3, 3)))

    def test_triangle_vertex_hessian_matches_barrier_energy_finite_difference(self):
        """A moving triangle corner changes the normal; retain its curvature."""
        points = np.float32([[.25, .25, .001], [0., 0., 0.],
                             [1., 0., 0.], [0., 1., 0.]])
        radius, stiffness = float(np.float32(.002)), 100.

        def energy(q):
            normal = np.cross(q[2] - q[1], q[3] - q[1])
            distance = abs(np.dot(q[0] - q[1], normal)) / np.linalg.norm(normal)
            return -stiffness * (distance - radius)**2 * np.log(distance / radius)

        role = 1
        q = points.astype(np.float64)
        h = 2.e-6
        expected_force, expected_hessian = np.zeros(3), np.zeros((3, 3))
        for a in range(3):
            da = np.zeros_like(q); da[role, a] = h
            expected_force[a] = -(energy(q + da) - energy(q - da)) / (2. * h)
            for b in range(3):
                db = np.zeros_like(q); db[role, b] = h
                expected_hessian[a, b] = (energy(q + da + db) - energy(q + da - db)
                                          - energy(q - da + db) + energy(q - da - db)) / (4. * h * h)
        for device in self.devices:
            with self.subTest(device=str(device)):
                model, _, graph, ccd, error, _, _ = workspace(device, points, [[1, 2, 3]])
                assembly = ColorContactForces(model, graph, ccd, error, k_barrier=stiffness)
                assembly.refresh()
                graph.current_color.fill_(int(graph.colors.numpy()[role]))
                force = wp.zeros(4, wp.vec3, device=device)
                hessian = wp.zeros(4, wp.mat33, device=device)
                assembly.assemble(model.particle_q, wp.clone(model.particle_q), .01,
                                  radius, 0., .01, force, hessian)
                self.assertEqual(int(error.numpy()[0]), 0)
                np.testing.assert_allclose(force.numpy()[role], expected_force, rtol=2.e-5, atol=1.e-6)
                np.testing.assert_allclose(hessian.numpy()[role], expected_hessian, rtol=5.e-5, atol=5.e-5)


if __name__ == "__main__":
    unittest.main()
