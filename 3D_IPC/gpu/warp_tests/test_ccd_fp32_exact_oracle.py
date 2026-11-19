"""Host-only exact oracle for accepted production FP32 CCD trajectories."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class Float32ExactOracleTests(unittest.TestCase):
    def test_rotated_scaled_and_degenerate_paths_agree_with_exact_dyadic_geometry(self):
        compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("A C++17 compiler is required")
        source = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="ipc-ccd-fp32-exact-") as temporary:
            executable = Path(temporary) / "oracle"
            build = subprocess.run([compiler, "-std=c++17", "-O2", "-fno-fast-math",
                                    "-ffp-contract=off", "-I" + str(source.parent),
                                    str(source / "ccd_fp32_exact_oracle.cpp"), "-o", str(executable)],
                                   capture_output=True, text=True, timeout=30)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads(run.stdout)
        self.assertTrue(result["all_passed"])
        self.assertGreaterEqual(result["queries"], 128)
        self.assertGreaterEqual(result["accepted"], 80)
        self.assertGreaterEqual(result["exact_unsafe"], 20)
        self.assertGreaterEqual(result["clipped"], 20)


if __name__ == "__main__":
    unittest.main()
