"""Source integrity and process isolation for the three public solvers."""

from pathlib import Path
import subprocess
import sys
import unittest

import gpu
import newton
from newton.solvers import SolverVBD

from gpu.newton_dat.verify_import import verify
from gpu.solver_dat_reference import SolverDATReference


class DatRuntimeTests(unittest.TestCase):
    def test_reference_is_the_exact_upstream_class_and_package(self):
        root = Path(gpu.__file__).resolve().parent / "third_party/newton_reference"
        self.assertEqual(verify(root, pristine=True), 1115)
        self.assertIs(SolverDATReference, SolverVBD)
        self.assertTrue(Path(newton.__file__).resolve().is_relative_to(root))

    def test_public_solver_imports_select_their_own_runtime(self):
        for module, name, folder in (("gpu.solver", "SolverIPC", "newton"),
                                     ("gpu.solver_dat", "SolverDAT", "newton_reference"),
                                     ("gpu.solver_dat_reference", "SolverDATReference", "newton_reference")):
            with self.subTest(solver=name):
                code = (f"from {module} import {name}; import newton; from pathlib import Path; "
                        f"assert Path(newton.__file__).resolve().parents[1].name == {folder!r}")
                completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_mixing_solver_runtimes_is_rejected_before_construction(self):
        for first, second in (("gpu.solver", "gpu.solver_dat"), ("gpu.solver_dat", "gpu.solver")):
            with self.subTest(first=first):
                code = (f"import {first}\ntry:\n import {second}\n"
                        "except ImportError as error:\n assert 'fresh process' in str(error)\n"
                        "else:\n raise AssertionError('Mixed solver runtimes were accepted')\n")
                completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
