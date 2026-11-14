"""Native checkpoint compatibility and real Newton State frame orchestration."""

from copy import deepcopy
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import warp as wp

from gpu import output
from gpu.run import FrameGraph, run_simulation
from gpu.scene import create_model
from .test_model import cloth_payload


@wp.kernel
def _linear_step(x: wp.array[wp.vec3], velocity: wp.array[wp.vec3], dt: float,
                 result: wp.array[wp.vec3], result_velocity: wp.array[wp.vec3]):
    i = wp.tid()
    result[i] = x[i] + dt * velocity[i]
    result_velocity[i] = velocity[i]


@wp.kernel
def _count_steps(counter: wp.array[int]):
    counter[0] += 1


class StepFixture:
    """Predictable device update for testing graph buffers, not physics equations."""
    def __init__(self, model):
        self.model = model
        self.count = wp.zeros(1, int, device=model.device)

    def step(self, state0, state1, control, contacts, dt):
        wp.launch(_linear_step, self.model.particle_count,
            [state0.particle_q, state0.particle_qd, dt, state1.particle_q, state1.particle_qd], device=self.model.device)
        wp.launch(_count_steps, 1, [self.count], device=self.model.device)

    def check_status(self):
        wp.synchronize_device(self.model.device)


class OutputTests(unittest.TestCase):
    def state(self):
        return {"positions": np.array([[0.1, 0.2, 0.3], [-1, 2, 3], [4, 5, 6]]),
                "velocities": np.array([[0, 0, 0], [1, 2, 3], [3, 2, 1]], dtype=float),
                "com": np.array([[0, 0, 1]], dtype=float), "com_velocity": np.array([[1, 0, 0]], dtype=float),
                "orientation": np.array([[1, 0, 0, 0]], dtype=float), "omega": np.array([[0, 0, 0]], dtype=float)}

    def test_counted_binary_layout_matches_native_format(self):
        values = self.state()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.bin"
            output.write_checkpoint(path, values)
            expected = b""
            for i, (name, width) in enumerate(output._FIELDS):
                if i == 2:
                    expected += b"RBSTATE1"
                data = np.asarray(values[name], dtype="=f8")
                expected += struct.pack("=Q", len(data)) + data.tobytes()
            self.assertEqual(path.read_bytes(), expected)
            actual = output.read_checkpoint(path, particle_count=3, body_count=1)
            for name in values:
                np.testing.assert_array_equal(actual[name], values[name])
            for bad in (expected[:13], expected[:-1], struct.pack("=Q", 2**63) + expected[8:], expected + b"bad"):
                path.write_bytes(bad)
                with self.assertRaises(ValueError):
                    output.read_checkpoint(path, particle_count=3, body_count=1)

    def test_invalid_output_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.bin"
            path.write_bytes(b"existing")
            bad = self.state()
            bad["positions"][0, 0] = np.nan
            with self.assertRaises(ValueError):
                output.write_checkpoint(path, bad)
            self.assertEqual(path.read_bytes(), b"existing")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_mesh_formats_retain_topology(self):
        values = self.state()
        with tempfile.TemporaryDirectory() as directory:
            for format, suffix in output.FORMATS.items():
                path = Path(directory) / ("mesh" + suffix)
                output.write_mesh(path, values["positions"], [[0, 1, 2]], format)
                self.assertTrue(path.stat().st_size)
                if format == "geo":
                    document = json.loads(path.read_text())
                    record = dict(zip(document[::2], document[1::2]))
                    self.assertEqual(record["pointcount"], 3)
                    self.assertEqual(record["primitivecount"], 1)
                    self.assertEqual(record["primitives"][0][0], ["type", "Polygon_run"])
                    self.assertEqual(dict(zip(record["primitives"][0][1][::2], record["primitives"][0][1][1::2]))["nvertices_rle"], [3, 1])
                elif format == "obj":
                    self.assertIn("f 1 2 3", path.read_text())
                elif format == "usd":
                    self.assertIn("point3f[] points", path.read_text())
                else:
                    header, data = path.read_bytes().split(b"end_header\n", 1)
                    self.assertIn(b"property double x", header)
                    self.assertEqual(len(data), 3 * 3 * 8 + 13)

    def test_cli_help_and_cpu_rejection_need_no_site_packages(self):
        root = Path(__file__).resolve().parents[2]
        for arguments, code in ((("--help",), 0), (("--device", "cpu"), 2)):
            result = subprocess.run([sys.executable, "-S", "-m", "gpu.run", *arguments], cwd=root,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, code, result.stdout + result.stderr)
            self.assertNotIn("Traceback", result.stdout + result.stderr)


class FrameControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def payload(self, substeps, directory):
        result = cloth_payload()
        result["params"].update(fps=30, substeps=substeps, dt=1 / (30 * substeps), max_global_iters=10,
                                fixed_iters=True, write_substeps=False)
        result["velocities"] = [[0.1, 0.2, 0.3]] * 4
        result["output"] = {"outdir": str(directory), "format": "geo", "num_frames": 2,
                            "restart_frame": -1, "write_substeps": False}
        return result

    def test_actual_newton_states_and_even_odd_frame_buffer_order(self):
        for device in self.devices:
            for substeps in (1, 3, 10):
                with self.subTest(device=str(device), substeps=substeps):
                    payload = self.payload(substeps, "unused")
                    model = create_model(payload, device)
                    step = StepFixture(model)
                    controller = FrameGraph(model, step, graphs=True)
                    for _ in range(3):
                        controller.advance()
                    self.assertEqual(int(step.count.numpy()[0]), 3 * substeps)
                    self.assertEqual(int(controller.motion.counter.numpy()[0]), 3 * substeps)
                    expected = np.asarray(payload["positions"]) + np.array([0.1, 0.2, 0.3]) / 10
                    np.testing.assert_allclose(controller.state.particle_q.numpy(), expected, rtol=2e-6, atol=2e-6)
                    self.assertEqual(controller.parity, 3 * substeps % 2)

    def test_frame_output_and_restart_use_the_same_original_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "output"
            payload = self.payload(10, directory)
            model = create_model(payload, "cpu")
            step = StepFixture(model)
            exporter = type("Export", (), {"scene": payload})()
            self.assertEqual(run_simulation(exporter, model, step, graphs=False, test=True), 0)
            self.assertEqual(int(step.count.numpy()[0]), 20)
            saved = output.read_checkpoint(directory / "state_0002.bin", particle_count=4, body_count=0)
            expected = np.asarray(payload["positions"]) + np.array([0.1, 0.2, 0.3]) * (2 / 30)
            np.testing.assert_allclose(saved["positions"], expected, rtol=2e-6, atol=2e-6)
            second = deepcopy(payload)
            second["output"]["restart_frame"] = 2
            second["output"]["num_frames"] = 3
            model2 = create_model(second, "cpu")
            step2 = StepFixture(model2)
            exporter2 = type("Export", (), {"scene": second})()
            self.assertEqual(run_simulation(exporter2, model2, step2, graphs=False), 0)
            self.assertEqual(int(step2.count.numpy()[0]), 10)
            final = output.read_checkpoint(directory / "state_0003.bin", particle_count=4, body_count=0)
            np.testing.assert_allclose(final["positions"], np.asarray(payload["positions"]) + np.array([0.1, 0.2, 0.3]) / 10, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
