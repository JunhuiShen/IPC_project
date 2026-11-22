"""Regression for the driven-edge crossing that preceded the level-based guess."""
import unittest

import numpy as np
import warp as wp

from gpu.ccd_defaults import LINEAR_CCD_GAP_FLOOR
from gpu.twist_boundary import _certify, _correct_obstruction, _extend, _first_obstruction, _publish
from gpu.warp_build import configure_module


@wp.kernel(module="unique", enable_backward=False)
def _edge_check(q: wp.array[wp.vec3], target: wp.array[wp.vec3], accepted: wp.array[int]):
    i = wp.tid()
    result = _certify(q, target, wp.vec4i(0, 1, 2, 3), 1)
    accepted[i] = int(result[2])


configure_module(_edge_check, ("ccd_direct_fp32.cuh", "twist_ccd.cuh"))


@wp.func_native(r"""
    float a[4][3], b[4][3];
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        a[i][k] = start.data[i][k];
        b[i][k] = end.data[i][k];
    }
    return ipc_gpu::twist_ccd::swept_certificate(a, b, true, gap);
""")
def _face_certificate(start: wp.array[wp.vec3], end: wp.array[wp.vec3], gap: wp.float64) -> bool: ...


@wp.kernel(module="unique", enable_backward=False)
def _near_floor(q: wp.array[wp.vec3], target: wp.array[wp.vec3], result: wp.array[int]):
    i = wp.tid()
    floor = wp.float64(1.0) / wp.float64(121500.0)
    if i == 1:
        floor *= wp.float64(2.0)
    result[i] = int(_face_certificate(q, target, floor))


configure_module(_near_floor, ("twist_ccd.cuh",))


class TwistBoundaryTests(unittest.TestCase):
    def test_coordinated_rotation_is_continuous_at_midheight(self):
        start = np.float32([[.4, -.75, .1], [.4, .75, .1], [.2, -.00001, .3], [.2, .00001, .3]])
        target = start.copy()
        for i, a in enumerate((-.01, .01)):
            x, _, z = start[i]
            target[i, 0] = np.cos(a)*x + np.sin(a)*z
            target[i, 2] = np.cos(a)*z - np.sin(a)*x
        for device in (wp.get_device("cpu"), *wp.get_cuda_devices()):
            q = wp.array(start, wp.vec3, device=device)
            extended = wp.empty_like(q)
            error = wp.zeros(1, int, device=device)
            wp.launch(_extend, 4, [q, wp.array(target, wp.vec3, device=device),
                      wp.array([0, 1], int, device=device), wp.array([0, 0, 1, 1], int, device=device),
                      extended, error], device=device)
            self.assertEqual(int(error.numpy()[0]), 0)
            values = extended.numpy()
            np.testing.assert_array_equal(values[:2], target[:2])
            np.testing.assert_allclose(values[2:, [0, 2]], start[2:, [0, 2]], rtol=0, atol=2e-7)
            np.testing.assert_array_equal(values[2:, 1], start[2:, 1])

    def test_only_free_corners_can_repair_a_blocked_trial(self):
        from .test_query_groups import independent_predictor_fixture

        for device in (wp.get_device("cpu"), *wp.get_cuda_devices()):
            for free in (True, False):
                q, model, contacts, detector, _ = independent_predictor_fixture(device)
                start = q.numpy().copy()
                proposed = start.copy()
                proposed[0, 2] = -.002
                target = wp.array(proposed, wp.vec3, device=device)
                center = wp.clone(target)
                model.particle_flags.assign([int(free), 0, 0, 0, 0])
                index = wp.empty(1, int, device=device)
                attempts, condition = wp.zeros_like(index), wp.zeros_like(index)

                def find_obstruction():
                    contacts.detect_swept(detector, q, target)
                    index.fill_(2147483647)
                    wp.launch(_first_obstruction, contacts.fast_workers, [q, target,
                              contacts.candidate_ids, contacts.candidate_count, model.tri_indices,
                              model.edge_indices, contacts.info, contacts.fast_workers,
                              index, contacts.error], device=device)

                find_obstruction()
                self.assertNotEqual(int(index.numpy()[0]), 2147483647)
                wp.launch(_correct_obstruction, 1, [q, target, center, model.particle_flags,
                          LINEAR_CCD_GAP_FLOOR, .01, contacts.candidate_ids, model.tri_indices,
                          model.edge_indices, contacts.info, index, attempts, condition,
                          contacts.error], device=device)
                np.testing.assert_array_equal(q.numpy(), start)
                np.testing.assert_array_equal(target.numpy()[1:], proposed[1:])
                if free:
                    contacts.check()
                    self.assertEqual(int(condition.numpy()[0]), 1)
                    self.assertGreater(float(target.numpy()[0, 2]), LINEAR_CCD_GAP_FLOOR)
                    find_obstruction()
                    self.assertEqual(int(index.numpy()[0]), 2147483647)
                else:
                    self.assertEqual(int(contacts.error.numpy()[0]), 806)
                    out = wp.clone(q)
                    wp.launch(_publish, q.shape[0], [q, out, target, model.particle_flags,
                              contacts.error], device=device)
                    np.testing.assert_array_equal(q.numpy(), start)
                    np.testing.assert_array_equal(out.numpy(), start)

    def test_near_floor_point_and_oblique_face_rotate_together(self):
        # All four corners move. The gap stays close to 8.97e-6.
        start = [[-.1332721412, .7063695788, .3091473579], [-.1670103818, .6931118369, .2573735416],
                 [-.1923698783, .6862351894, .2397041321], [-.1186785251, .7113848925, .3265143633]]
        end = [[-.1305694580, .7063695788, .3102985024], [-.1647581905, .6931118369, .2588210702],
               [-.1902709007, .6862351894, .2413736135], [-.1158248559, .7113848925, .3275375068]]
        for device in (wp.get_device("cpu"), *wp.get_cuda_devices()):
            q = wp.array(start, wp.vec3, device=device)
            target = wp.array(end, wp.vec3, device=device)
            result = wp.zeros(2, int, device=device)
            wp.launch(_near_floor, 2, [q, target, result], device=device)
            np.testing.assert_array_equal(result.numpy(), [1, 0])

    def test_registered_edges_cross_when_only_pins_move(self):
        # Frame 80, substep 3. The two segments meet in their interiors
        # at t = 0.57044754 even though both endpoints of the step are clean.
        start = np.float32([[.401611238718, .749999940395, -.222616165876],
                            [.405101448298, .712714254856, -.238696321845],
                            [.398511350155, .725380778313, -.231092989445],
                            [.437310069799, .749999940395, -.242404296994]])
        pins_only = start.copy()
        pins_only[[0, 3], 0] = [.399653285742, .435178071260]
        pins_only[[0, 3], 2] = [-.226112350821, -.246211245656]
        p, t = start[0].astype(float), pins_only[0].astype(float)
        angle = np.arctan2(t[0]*p[2] - t[2]*p[0], p[0]*t[0] + p[2]*t[2])
        extended = pins_only.copy()
        x, z = start[1:3, 0], start[1:3, 2]
        extended[1:3, 0] = np.cos(angle)*x + np.sin(angle)*z
        extended[1:3, 2] = np.cos(angle)*z - np.sin(angle)*x
        for device in (wp.get_device("cpu"), *wp.get_cuda_devices()):
            q = wp.array(start, wp.vec3, device=device)
            target = wp.array(pins_only, wp.vec3, device=device)
            accepted = wp.zeros(1, int, device=device)
            wp.launch(_edge_check, 1, [q, target, accepted], device=device)
            self.assertEqual(int(accepted.numpy()[0]), 0)
            # A simultaneous boundary and free-corner move can be certified.
            target.assign(extended)
            wp.launch(_edge_check, 1, [q, target, accepted], device=device)
            self.assertEqual(int(accepted.numpy()[0]), 1)
