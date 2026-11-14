"""Exact full-box witnesses and checkpoint compaction/validation checks."""

from fractions import Fraction
import unittest

import numpy as np
import warp as wp

from gpu import contact
from gpu.gs import GsSchedule
from gpu.movement_box_queries import BoxProof, GsBoxEpochFilter, _box_certificate, build_nodes
from gpu.warp_build import configure_module

from .test_movement_box_queries import fixture, pairs


@wp.kernel
def _proofs(centers: wp.array[wp.vec3], previous: wp.array[wp.vec3],
             lower: wp.array[wp.vec3], upper: wp.array[wp.vec3], kinds: wp.array[int],
             radii: wp.array[wp.float64], output: wp.array[BoxProof]):
    i = wp.tid()
    output[i] = _box_certificate(centers, previous, lower, upper,
        wp.vec4i(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3), kinds[i], radii[i], 1)


configure_module(_proofs, ("ccd.cuh",))


def bounds(points, radii):
    points = np.asarray(points, np.float32)
    radii = np.broadcast_to(np.asarray(radii, np.float64), points.shape[:-1])
    lo = np.float32(points.astype(np.float64) - radii[..., None])
    hi = np.float32(points.astype(np.float64) + radii[..., None])
    return np.nextafter(lo, np.float32(-np.inf)), np.nextafter(hi, np.float32(np.inf))


def ceil_f32(value):
    rounded = np.float32(value)
    return float(np.nextafter(rounded, np.float32(np.inf))) if float(rounded) < value else float(rounded)


class GsBoxFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def evaluate(self, device, centers, lower, upper, kinds, radius=.002, previous=None):
        centers = np.asarray(centers, np.float32).reshape(-1, 4, 3)
        lower = np.asarray(lower, np.float32).reshape(-1, 4, 3)
        upper = np.asarray(upper, np.float32).reshape(-1, 4, 3)
        history = centers if previous is None else np.asarray(previous, np.float32).reshape(-1, 4, 3)
        output = wp.empty(len(centers), BoxProof, device=device)
        wp.launch(_proofs, len(centers), [
            *[wp.array(x.reshape(-1, 3), wp.vec3, device=device) for x in (centers, history, lower, upper)],
            wp.array(kinds, int, device=device),
            wp.array(np.broadcast_to(np.asarray(radius, np.float64), (len(centers),)), wp.float64, device=device),
            output], device=device)
        return output.numpy()

    def assert_exact_witness(self, lower, upper, kind, radius, packet):
        self.assertEqual(packet[0], 1)
        axis = [Fraction(float(x)) for x in packet[1:4]]
        required = Fraction(float(packet[4]))
        scale = max(Fraction(abs(float(x))) for x in np.r_[lower.ravel(), upper.ravel()])
        floor = max(Fraction(1e-8), Fraction(1e-10), Fraction(128 * np.finfo(np.float64).eps) * scale)
        self.assertGreaterEqual(required, max(Fraction(radius), Fraction(ceil_f32(radius))) + floor)
        projected = []
        for lo, hi in zip(lower, upper):
            a, b = Fraction(0), Fraction(0)
            for k, value in enumerate(axis):
                first, second = Fraction(float(lo[k])), Fraction(float(hi[k]))
                a += value * (first if value >= 0 else second)
                b += value * (second if value >= 0 else first)
            projected.append((a, b))
        split = 1 if kind == 0 else 2
        alo = min(x[0] for x in projected[:split])
        ahi = max(x[1] for x in projected[:split])
        blo = min(x[0] for x in projected[split:])
        bhi = max(x[1] for x in projected[split:])
        gap = max(alo - bhi, blo - ahi)
        self.assertGreater(gap, 0)
        self.assertGreater(gap * gap, required * required * sum(x * x for x in axis))

    def test_random_rotated_scaled_boxes_have_exact_rational_separation_witnesses(self):
        rng = np.random.default_rng(78311)
        centers, kinds, widths = [], [], []
        for index in range(96):
            kind = index % 2
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            scale = 2.0 ** (-80, -20, 0, 20, 70, 120)[index % 6]
            points = np.asarray([[.2, .2, .2], [0, 0, 0], [1, 0, 0], [0, 1, 0]]) if kind == 0 else np.asarray(
                [[-1, 0, 0], [1, 0, 0], [0, -1, .2], [0, 1, .2]])
            translation = rng.uniform(-2, 2, 3) * scale
            centers.append(np.float32(points @ rotation * scale + translation))
            kinds.append(kind)
            widths.append(scale * rng.uniform(.0001, .005, 4))
        centers = np.asarray(centers)
        lo, hi = bounds(centers, widths)
        for device in self.devices:
            result = self.evaluate(device, centers, lo, hi, kinds)
            accepted = 0
            for index, packet in enumerate(result):
                self.assertIn(int(packet[0]), (0, 1))
                if packet[0] == 1:
                    accepted += 1
                    self.assert_exact_witness(lo[index], hi[index], kinds[index], .002, packet)
            self.assertGreater(accepted, 48)

    def test_active_future_crossings_overlap_and_near_threshold_are_retained(self):
        q = np.asarray([[[.25, .25, .001], [0, 0, 0], [1, 0, 0], [0, 1, 0]],
                        [[.25, .25, .05], [0, 0, 0], [1, 0, 0], [0, 1, 0]],
                        [[0, 0, 0], [1, 1, 0], [0, .001, 0], [1, 1.001, 0]]], np.float32)
        lo, hi = bounds(q, np.asarray([.0001, .05, .001])[:, None])
        threshold = ceil_f32(.002) + 1e-8
        low = np.float32(threshold)
        if float(low) > threshold:
            low = np.nextafter(low, np.float32(-np.inf))
        high = np.nextafter(low, np.float32(np.inf))
        neighbors = np.asarray([[[.25, .25, gap], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
                                for gap in (low, high)], np.float32)
        for device in self.devices:
            np.testing.assert_array_equal(self.evaluate(device, q, lo, hi, [0, 0, 1])[:, 0], 0)
            result = self.evaluate(device, neighbors, neighbors, neighbors, [0, 0])
            np.testing.assert_array_equal(result[:, 0], [0, 1])
            self.assert_exact_witness(neighbors[1], neighbors[1], 0, .002, result[1])

    def make_case(self, device, kind=0, *, active=False):
        points = ([[.25, .25, .001], [0, 0, 0], [1, 0, 0], [0, 1, 0]] if active else
                  [[.9, .9, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]]) if kind == 0 else [
                      [0, 0, 0], [1, 1, 0], [0, .5, 0], [.5, 1, 0]]
        triangles = [[1, 2, 3]] if kind == 0 else []
        edges = [] if kind == 0 else [[-1, -1, 0, 1], [-1, -1, 2, 3]]
        model, backend, detector = fixture(device, points, triangles, edges, capacity=4)
        model.tet_count = 0
        model.tet_indices = wp.empty((0, 4), int, device=device)
        radii = wp.full(4, .001, wp.float64, device=device)
        backend.detect_boxes(detector, model.particle_q, radii, .01)
        backend.check()
        self.assertEqual(int(backend.candidate_count.numpy()[0]), 1)
        return model, backend, detector, radii, wp.clone(model.particle_q)

    def apply(self, filter, case, *, params=None):
        model, backend, _, _, previous = case
        params = contact.parameters(dt=.01, **backend.options) if params is None else params
        filter.apply(model.particle_q, previous, backend.swept_lower, backend.swept_upper, params)

    def test_false_positive_pairs_are_compacted_before_greedy_graph_rebuild(self):
        for device in self.devices:
            for kind in (0, 1):
                case = self.make_case(device, kind)
                model, backend, *_ = case
                graph = GsSchedule(backend)
                graph.rebuild()
                graph.validate_independence()
                backend.check()
                self.assertEqual(int(graph.metadata.numpy()[0]), 4)
                epoch = backend.query_epoch.numpy().copy()
                generation = backend.query_generation
                filter = GsBoxEpochFilter(backend)
                self.apply(filter, case)
                backend.check()
                self.assertEqual(int(filter.raw_count.numpy()[0]), 1)
                self.assertEqual(int(filter.retained_count.numpy()[0]), 0)
                self.assertEqual(pairs(backend), (set(), set()))
                np.testing.assert_array_equal(backend.query_epoch.numpy(), epoch)
                self.assertEqual(backend.query_generation, generation)
                graph.rebuild()
                graph.validate_independence()
                backend.check()
                self.assertLess(int(graph.metadata.numpy()[0]), 4)
                with self.assertRaisesRegex(ValueError, "Refresh raw candidates"):
                    self.apply(filter, case)

    def test_raw_overflow_bad_history_boxes_and_ids_are_never_hidden(self):
        for device in self.devices:
            for problem, expected in (("history", 101), ("center", 202), ("bounds", 202), ("id", 11), ("overflow", 10)):
                case = self.make_case(device)
                model, backend, _, _, previous = case
                filter = GsBoxEpochFilter(backend)
                if problem == "history":
                    values = previous.numpy(); values[3, 0] = np.nan; previous.assign(values)
                elif problem == "center":
                    values = model.particle_q.numpy(); values[0, 0] += 1; model.particle_q.assign(values)
                elif problem == "bounds":
                    values = backend.swept_upper.numpy(); values[0, 0] = np.inf; backend.swept_upper.assign(values)
                elif problem == "id":
                    values = backend.candidate_ids.numpy(); values[0] = 10000; backend.candidate_ids.assign(values)
                else:
                    backend.info.vertex_colliding_triangles_count.assign([5, 0, 0, 0])
                    backend.refresh()
                self.apply(filter, case)
                self.assertEqual(int(backend.error.numpy()[0]), expected)
            for disabled in ("barrier", "static"):
                case = self.make_case(device)
                model, backend, _, _, previous = case
                values = previous.numpy(); values[0, 0] = np.nan; previous.assign(values)
                if disabled == "barrier":
                    backend.options["k_barrier"] = 0.0
                else:
                    model.particle_flags.zero_()
                self.apply(GsBoxEpochFilter(backend), case)
                backend.check()
                self.assertEqual(int(backend.candidate_count.numpy()[0]), 0)

    def test_invalid_parameters_and_cuda_replay_do_not_hide_epoch_inputs(self):
        for device in self.devices:
            for field, invalid in (("dt", 0), ("friction", -1), ("eps_v", np.nan), ("eps", 0)):
                case = self.make_case(device)
                p = contact.parameters(dt=.01, **case[1].options)
                setattr(p, field, invalid)
                with self.assertRaises((ValueError, TypeError)):
                    self.apply(GsBoxEpochFilter(case[1]), case, params=p)
                self.assertEqual(int(case[1].candidate_count.numpy()[0]), 1)
        for device in wp.get_cuda_devices():
            case = self.make_case(device)
            model, backend, _, radii, previous = case
            filter = GsBoxEpochFilter(backend)
            params = contact.parameters(dt=.01, **backend.options)

            def run():
                wp.launch(build_nodes, 4,
                    [model.particle_q, radii, .01, backend.swept_lower, backend.swept_upper,
                     backend.box_query_lower, backend.box_query_upper, backend.error], device=device)
                backend.refresh()
                self.apply(filter, case, params=params)

            run()
            with wp.ScopedCapture(device=device) as capture:
                run()
            for point, retained in (([.25, .25, .001], 1), ([.9, .9, 0], 0)):
                values = model.particle_q.numpy(); values[0] = point
                model.particle_q.assign(values); previous.assign(values)
                wp.capture_launch(capture.graph)
                backend.check()
                self.assertEqual(int(filter.raw_count.numpy()[0]), 1)
                self.assertEqual(int(backend.candidate_count.numpy()[0]), retained)


if __name__ == "__main__":
    unittest.main()
