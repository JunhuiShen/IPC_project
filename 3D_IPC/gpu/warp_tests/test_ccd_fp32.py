"""Behavioral FP32 CCD regressions, with a compiler-only host entry point.

The host checks run without Warp. When Warp is installed the same certificate
is also compiled through the production header builder on CPU and CUDA.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

try:
    import numpy as np
    import warp as wp
except ImportError:
    np = wp = None


class Float32CertificateHostTests(unittest.TestCase):
    def test_analytic_crossings_rounding_degeneracy_and_invalid_inputs(self):
        compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("A C++17 compiler is required")
        test_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="ipc-ccd-fp32-") as temporary:
            executable = Path(temporary) / "checks"
            command = [compiler, "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
                       "-Wdouble-promotion", "-Werror=double-promotion",
                       "-I" + str(test_root.parent), str(test_root / "ccd_fp32_checks.cpp"),
                       "-o", str(executable)]
            build = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(executable)], text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads(run.stdout)
        self.assertTrue(result["all_passed"])
        self.assertGreater(result["crossings"], 200)
        self.assertGreater(result["safe"], 200)
        self.assertGreaterEqual(result["degeneracies"], 4)
        self.assertGreaterEqual(result["invalid"], 13)


if wp is not None:
    from gpu.warp_build import configure_module

    @wp.func_native(r"""
        ipc_gpu::ccd32::LinearQuery q;
        q.vertex_face=kind==0;
        q.gap_floor=gap;
        for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
            q.start[i][k]=start.data[4*index+i][k];
            q.end[i][k]=end.data[4*index+i][k];
        }
        return static_cast<int>(ipc_gpu::ccd32::validate_linear(q));
    """)
    def _native_certificate(start: wp.array[wp.vec3], end: wp.array[wp.vec3],
                            index: int, kind: int, gap: float) -> int: ...

    @wp.kernel
    def _certificates(start: wp.array[wp.vec3], end: wp.array[wp.vec3],
                       kinds: wp.array[int], gaps: wp.array[float], result: wp.array[int]):
        i = wp.tid()
        result[i] = _native_certificate(start, end, i, kinds[i], gaps[i])

    configure_module(__name__, ("ccd_fp32.cuh",))


@unittest.skipIf(wp is None, "Warp is not installed; compiler-only checks remain available")
class Float32CertificateWarpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_actual_cpu_cuda_certificates_match_analytic_cases(self):
        starts, ends, kinds, gaps, expectations = [], [], [], [], []
        for kind in (0, 1):
            start = np.array([[.25, .25, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
                             if kind == 0 else
                             [[-1, 0, 1], [1, 0, 1], [0, -1, 0], [0, 1, 0]], np.float32)
            split = 1 if kind == 0 else 2
            for end_height, expectation in ((2, "safe"), (-1, "closed"), (0, "closed")):
                end = start.copy(); end[:split, 2] = end_height
                starts.append(start.copy()); ends.append(end)
                kinds.append(kind); gaps.append(1e-8); expectations.append(expectation)
            end = start.copy(); end[0, 0] = np.nan
            starts.append(start.copy()); ends.append(end)
            kinds.append(kind); gaps.append(1e-8); expectations.append("invalid")
        previous = None
        for device in self.devices:
            with self.subTest(device=str(device)):
                result = wp.empty(len(kinds), int, device=device)
                args = [wp.array(np.concatenate(starts), wp.vec3, device=device),
                        wp.array(np.concatenate(ends), wp.vec3, device=device),
                        wp.array(kinds, int, device=device),
                        wp.array(gaps, float, device=device), result]
                wp.launch(_certificates, len(kinds), args, device=device)
                actual = result.numpy()
                for expected, status in zip(expectations, actual):
                    if expected == "safe": self.assertEqual(status, 0)
                    elif expected == "invalid": self.assertEqual(status, 2)
                    else: self.assertNotEqual(status, 0)
                if previous is not None:
                    np.testing.assert_array_equal(actual, previous)
                previous = actual

    def test_backend_recertifies_rounded_endpoints_and_preserves_unrelated_motion(self):
        from gpu.ccd import CcdBackend
        from .test_ccd_independent import accept, fixture
        for device in self.devices:
            for plane in (np.float32(0), np.float32(100000)):
                with self.subTest(device=str(device), plane=float(plane)):
                    above = np.nextafter(plane, np.float32(np.inf)) if plane else np.float32(.1)
                    below = np.nextafter(plane, np.float32(-np.inf)) if plane else np.float32(-.1)
                    data = fixture(device, plane=float(plane), height=float(above - plane))
                    proposal = data.start.copy()
                    proposal[0, 2] = below; proposal[4, 0] += np.float32(.25)
                    backend = CcdBackend(data.contacts, node_box_radius=1)
                    accept(data, backend, wp.array(proposal, wp.vec3, device=device), bounded=False)
                    backend.check()
                    actual = data.q.numpy()
                    self.assertGreater(actual[0, 2], plane)
                    np.testing.assert_array_equal(actual[4], proposal[4])
                    np.testing.assert_array_equal(actual[1:4], data.start[1:4])
                    self.assertEqual(int(backend.verified.numpy()[0]), 1)
                    self.assertEqual(backend.fraction.dtype, wp.float32)


if __name__ == "__main__":
    unittest.main()
