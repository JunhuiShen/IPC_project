"""Warm coloring must equal fresh canonical greedy after every graph change."""

from types import SimpleNamespace
import unittest

import numpy as np
import warp as wp

from gpu.gs import WarmColors
from gpu.parallel_colors import ParallelColors


def oracle(n, edges):
    rows = [set() for _ in range(n)]
    for a, b in edges:
        if a != b:
            rows[max(a, b)].add(min(a, b))
    colors = []
    for row in rows:
        used = {colors[neighbor] for neighbor in row}
        color = 0
        while color in used:
            color += 1
        colors.append(color)
    return np.asarray(colors, np.int32), rows


def set_edges(graph, edges):
    expected, neighbors = oracle(graph.n, edges)
    rows = [0]
    adjacency = []
    for row in neighbors:
        adjacency.extend(sorted(row))
        rows.append(len(adjacency))
    storage = np.zeros(graph.adjacency.shape[0], np.int32)
    storage[:len(adjacency)] = adjacency
    graph.rows.assign(np.asarray(rows, np.uint64))
    graph.adjacency.assign(storage)
    return expected


def graph_fixture(device, n):
    return SimpleNamespace(n=n, device=device, rows=wp.zeros(n + 1, wp.uint64, device=device),
        adjacency=wp.zeros(n * (n - 1) // 2, int, device=device),
        colors=wp.full(n, -1, int, device=device), trial_order=wp.empty(n, int, device=device),
        metadata=wp.zeros(3, int, device=device))


class WarmColorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_single_vertex_first_epoch_and_fixed_point(self):
        for device in self.devices:
            graph = graph_fixture(device, 1)
            error = wp.zeros(1, int, device=device)
            workspace = WarmColors(graph)
            for _ in range(3):
                workspace.apply(SimpleNamespace(error=error))
                self.assertEqual(int(error.numpy()[0]), 0)
                np.testing.assert_array_equal(graph.colors.numpy(), [0])
                self.assertEqual(int(workspace.iterations.numpy()[0]), 1)

    def test_arbitrary_seeds_long_chain_and_more_than_128_colors(self):
        cases = [(257, [(v, v - 1) for v in range(1, 257)]),
                 (129, [(v, u) for v in range(129) for u in range(v)])]
        for device in self.devices:
            for n, edges in cases:
                graph = graph_fixture(device, n)
                expected = set_edges(graph, edges)
                workspace = WarmColors(graph)
                error = wp.zeros(1, int, device=device)
                for seed in (np.full(n, -1), np.zeros(n), np.arange(n)[::-1]):
                    graph.colors.assign(np.asarray(seed, np.int32))
                    workspace.apply(SimpleNamespace(error=error))
                    self.assertEqual(int(error.numpy()[0]), 0)
                    np.testing.assert_array_equal(graph.colors.numpy(), expected)
                    self.assertLessEqual(int(workspace.iterations.numpy()[0]), n + 1)

    def test_added_and_removed_edges_match_python_and_reset_oracles(self):
        rng = np.random.default_rng(729541)
        n = 65
        for device in self.devices:
            warm, cold = graph_fixture(device, n), graph_fixture(device, n)
            repair, reset = WarmColors(warm), ParallelColors(cold)
            error = wp.zeros(1, int, device=device)
            contacts = SimpleNamespace(error=error)
            for epoch in range(20):
                probability = (epoch % 5) / 5
                edges = [(v, u) for v in range(n) for u in range(v) if rng.random() < probability]
                expected = set_edges(warm, edges)
                set_edges(cold, edges)
                repair.apply(contacts)
                reset.apply(contacts)
                self.assertEqual(int(error.numpy()[0]), 0)
                np.testing.assert_array_equal(warm.colors.numpy(), expected)
                np.testing.assert_array_equal(warm.colors.numpy(), cold.colors.numpy())
                np.testing.assert_array_equal(warm.metadata.numpy(), cold.metadata.numpy())
                # An unchanged epoch must terminate after its first double pass.
                repair.apply(contacts)
                self.assertEqual(int(repair.iterations.numpy()[0]), 1)

    def test_invalid_seed_fails_closed(self):
        for device in self.devices:
            for seed in (-2, 2):
                graph = graph_fixture(device, 2)
                graph.colors.assign([seed, 0])
                error = wp.zeros(1, int, device=device)
                WarmColors(graph).apply(SimpleNamespace(error=error))
                self.assertEqual(int(error.numpy()[0]), 391)

    def test_cuda_capture_replay_repairs_device_graph_changes(self):
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA device unavailable")
        for device in devices:
            graph = graph_fixture(device, 65)
            error = wp.zeros(1, int, device=device)
            contacts = SimpleNamespace(error=error)
            workspace = WarmColors(graph)
            set_edges(graph, [])
            workspace.apply(contacts)
            wp.synchronize_device(device)
            with wp.ScopedCapture(device=device) as capture:
                workspace.apply(contacts)
            for edges in ([(v, v - 1) for v in range(1, 65)],
                          [(v, u) for v in range(65) for u in range(v)], []):
                expected = set_edges(graph, edges)
                wp.capture_launch(capture.graph)
                self.assertEqual(int(error.numpy()[0]), 0)
                np.testing.assert_array_equal(graph.colors.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
