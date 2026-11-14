"""CPU movement-budget rule and persistent history on the GPU state layout."""

from types import SimpleNamespace
import unittest

import numpy as np
import warp as wp

from gpu.gs import MovementBoxes


class MovementBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def fixture(self, device):
        q = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], wp.vec3, device=device)
        model = SimpleNamespace(device=device, particle_count=4, particle_q=q,
            particle_world=wp.array([0, 0, 1, -1], int, device=device), world_count=2)
        return q, MovementBoxes(model, .001, .02)

    def test_cpu_radius_rule_uses_previous_solve_and_input_velocity(self):
        for device in self.devices:
            with self.subTest(device=str(device)):
                q, boxes = self.fixture(device)
                velocity = wp.array([[0, 0, 0], [0, 3, 4], [0, 0, 0], [2, 0, 0]], wp.vec3, device=device)
                boxes.begin(velocity, .002)
                np.testing.assert_array_equal(boxes.radii.numpy(), .02)
                self.assertEqual(boxes.radii.dtype, wp.float64)
                self.assertEqual(boxes.previous_displacement.dtype, wp.float64)

                # These are displacements after the predictor, which must not
                # be confused with the larger predictor motion from step start.
                q.assign(q.numpy() + np.array([.5, -.25, .125], np.float32))
                boxes.anchor(q)
                anchored = q.numpy().copy()
                q.assign(anchored + np.array([[0, 0, 0], [.002, 0, 0], [0, .01, 0], [0, 0, .03]], np.float32))
                expected_history = np.linalg.norm(q.numpy().astype(np.float64) - anchored.astype(np.float64), axis=1)
                boxes.finish(q)
                np.testing.assert_allclose(boxes.previous_displacement.numpy(), expected_history, rtol=2e-15, atol=0)
                # Finalizing history does not alter the current box epoch.
                np.testing.assert_array_equal(boxes.radii.numpy(), .02)

                expected = np.clip(1.2 * np.maximum(expected_history,
                    .002 * np.linalg.norm(velocity.numpy().astype(np.float64), axis=1)), .001, .02)
                boxes.begin(velocity, .002)
                np.testing.assert_allclose(boxes.radii.numpy(), expected, rtol=2e-15, atol=0)
                self.assertEqual(boxes.radii.numpy()[0], .001)
                self.assertEqual(boxes.radii.numpy()[3], .02)

    def test_reset_discards_only_selected_worlds_history(self):
        for device in self.devices:
            with self.subTest(device=str(device)):
                _, boxes = self.fixture(device)
                boxes.previous_displacement.assign([.002, .003, .004, .005])
                boxes.radii.assign([.006, .007, .008, .009])
                boxes.reset(wp.array([False, True, False], wp.bool, device=device))
                np.testing.assert_array_equal(boxes.previous_displacement.numpy(), [.002, .003, .02, .005])
                np.testing.assert_array_equal(boxes.radii.numpy(), [.006, .007, .02, .009])
                boxes.reset()
                np.testing.assert_array_equal(boxes.previous_displacement.numpy(), .02)
                np.testing.assert_array_equal(boxes.radii.numpy(), .02)


if __name__ == "__main__":
    unittest.main()
