"""Independent numerical checks of the FP32 project local solve.

The oracle is NumPy's binary64 SVD pseudoinverse of the *represented FP32*
inputs. These tests do not reconstruct the helper's inverse/QR algorithm.
Host tests compile the production arithmetic header without requiring Warp;
when Warp is installed the same cases also exercise its CPU/CUDA wrappers.
"""

import ctypes
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


GPU_ROOT = Path(__file__).resolve().parents[1]
def reference_solution(h, force, damping=1.):
    """Minimum-norm least-squares solution, with no positive-definite clamp."""
    represented_h = np.asarray(h, np.float32).astype(np.float64)
    represented_force = np.asarray(force, np.float32).astype(np.float64)
    # Cases are well away from the production FP32 numerical-rank threshold.
    return (float(np.float32(damping))
            * (np.linalg.pinv(represented_h, rcond=1.e-12) @ represented_force))


class NumericalSolveCases:
    def check_reference(self, matrices, forces, dampings=None, *, rtol=2.e-5):
        matrices = np.ascontiguousarray(matrices, dtype=np.float32)
        forces = np.ascontiguousarray(forces, dtype=np.float32)
        if dampings is None:
            dampings = np.ones(len(matrices), dtype=np.float32)
        dampings = np.ascontiguousarray(dampings, dtype=np.float32)
        expected = np.asarray([reference_solution(h, f, d)
                               for h, f, d in zip(matrices, forces, dampings)])
        for backend in self.backends:
            with self.subTest(backend=str(backend)):
                actual, status = self.evaluate(matrices, forces, dampings, backend)
                self.assertEqual(actual.dtype, np.dtype(np.float32))
                self.assertTrue(np.all(np.isfinite(actual)))
                self.assertTrue(np.all(status >= 0), status)
                # A vector-norm bound is meaningful even when a true component
                # is zero; componentwise relative errors would not be.
                errors = np.linalg.norm(actual.astype(np.float64) - expected, axis=1)
                scales = np.maximum(np.linalg.norm(expected, axis=1), 1.e-30)
                self.assertTrue(np.all(errors <= rtol * scales),
                                (errors / scales, actual, expected, status))
        return expected

    def test_well_conditioned_spd(self):
        rng = np.random.default_rng(20261118)
        matrices, forces = [], []
        for _ in range(24):
            a = rng.normal(size=(3, 3))
            matrices.append(a.T @ a + 2. * np.eye(3))
            forces.append(rng.normal(size=3))
        self.check_reference(matrices, forces)

    def test_indefinite_hessian_is_solved_without_psd_projection(self):
        rng = np.random.default_rng(909)
        matrices, forces = [], []
        for _ in range(24):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            matrices.append(q @ np.diag([-3., .5, 2.]) @ q.T)
            forces.append(rng.normal(size=3))
        self.check_reference(matrices, forces)

    def test_rank_one_consistent_and_inconsistent_minimum_norm(self):
        direction = np.array([1., 2., -2.])
        h = np.outer(direction, direction)
        self.check_reference([h, h], [h @ np.array([.25, -2., 1.]),
                                      np.array([3., -1., 2.])])

    def test_rank_two_consistent_and_inconsistent_minimum_norm(self):
        # Integer entries keep the singularity exact in the input precision.
        basis = np.array([[1., 2.], [2., 1.], [-2., 2.]])
        matrices = [np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., -2.]]),
                    basis @ np.diag([1., -1.]) @ basis.T]
        self.check_reference([h for h in matrices for _ in range(2)],
                             [f for h in matrices
                              for f in (h @ np.array([.25, -2., 1.]),
                                        np.array([3., -1., 2.]))])

    def test_minimum_norm_solution_has_no_nullspace_component(self):
        h = np.array([[1., 2., -2.], [2., 4., -4.], [-2., -4., 4.]], np.float32)
        f = np.array([3., -1., 2.], np.float32)
        _, _, vt = np.linalg.svd(h.astype(np.float64))
        for backend in self.backends:
            with self.subTest(backend=str(backend)):
                actual, status = self.evaluate(h[None], f[None], np.ones(1, np.float32), backend)
                self.assertGreaterEqual(status[0], 0)
                x = actual[0].astype(np.float64)
                np.testing.assert_allclose(vt[1:] @ x, 0., atol=2.e-6)
                # Least-squares normal equations hold for inconsistent forces.
                np.testing.assert_allclose(h.T @ (h @ x - f), 0., atol=2.e-5)

    def test_small_nonsingular_determinant_does_not_freeze(self):
        matrices = [np.diag([1., 1., 1.e-6]), np.diag([1., -1., 1.e-6])]
        self.check_reference(matrices, [[1., 2., 3.e-6], [1., 2., 3.e-6]], rtol=1.e-5)
        for backend in self.backends:
            with self.subTest(backend=str(backend)):
                _, status = self.evaluate(np.asarray(matrices, np.float32),
                                          np.asarray([[1., 2., 3.e-6]] * 2, np.float32),
                                          np.ones(2, np.float32), backend)
                # Exercise the public fallback status, not its internals.
                np.testing.assert_array_equal(status, [1, 1])

    def test_common_scale_does_not_change_displacement(self):
        h = np.array([[4., 1., -.5], [1., 3., .25], [-.5, .25, 2.]])
        f = np.array([1., -2., 3.])
        scales = [1.e-30, 1.e-18, 1.e-9, 1., 1.e9, 1.e18, 1.e30]
        self.check_reference([s * h for s in scales], [s * f for s in scales])
        for backend in self.backends:
            with self.subTest(backend=str(backend)):
                actual, _ = self.evaluate(np.asarray([s * h for s in scales], np.float32),
                                          np.asarray([s * f for s in scales], np.float32),
                                          np.ones(len(scales), np.float32), backend)
                np.testing.assert_allclose(actual, np.broadcast_to(actual[3], actual.shape),
                                           rtol=2.e-5, atol=1.e-6)

    def test_damping_scales_force_solution_in_fp32(self):
        h = np.array([[3., .5, -.25], [.5, 2., .125], [-.25, .125, 1.]])
        f = np.array([1., -3., .5])
        damping = [0., .125, .3, 1., 1.5]
        self.check_reference([h] * len(damping), [f] * len(damping), damping)

    def test_zero_matrix_rhs_and_damping(self):
        matrices = np.asarray([np.zeros((3, 3)), np.eye(3), np.eye(3)], np.float32)
        forces = np.asarray([[1., -2., 3.], [0., 0., 0.], [1., -2., 3.]], np.float32)
        for backend in self.backends:
            with self.subTest(backend=str(backend)):
                actual, status = self.evaluate(matrices, forces, np.array([1., 1., 0.], np.float32), backend)
                np.testing.assert_array_equal(actual, np.zeros((3, 3), np.float32))
                np.testing.assert_array_equal(status, [2, 2, 2])

    def test_nonfinite_inputs_and_negative_damping_are_rejected(self):
        matrices, forces, dampings = [], [], []
        for bad in (np.nan, np.inf, -np.inf):
            h = np.eye(3, dtype=np.float32); h[1, 2] = bad
            matrices.append(h); forces.append([1., 2., 3.]); dampings.append(1.)
            matrices.append(np.eye(3)); forces.append([1., bad, 3.]); dampings.append(1.)
            matrices.append(np.eye(3)); forces.append([1., 2., 3.]); dampings.append(bad)
        matrices.append(np.eye(3)); forces.append([1., 2., 3.]); dampings.append(-.1)
        for backend in self.backends:
            with self.subTest(backend=str(backend)):
                actual, status = self.evaluate(np.asarray(matrices, np.float32),
                                              np.asarray(forces, np.float32),
                                              np.asarray(dampings, np.float32), backend)
                np.testing.assert_array_equal(actual, np.zeros((len(matrices), 3), np.float32))
                np.testing.assert_array_equal(status, -np.ones(len(matrices)))

    def test_unrepresentable_displacement_is_not_published(self):
        for backend in self.backends:
            with self.subTest(backend=str(backend)):
                actual, status = self.evaluate((np.eye(3, dtype=np.float32) * 1.e-30)[None],
                                              np.array([[1.e30, -1.e30, 1.e30]], np.float32),
                                              np.ones(1, np.float32), backend)
                np.testing.assert_array_equal(actual, np.zeros((1, 3), np.float32))
                np.testing.assert_array_equal(status, [-2])


class HostLocalSolveTests(NumericalSolveCases, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            raise unittest.SkipTest("A C++ compiler is needed for the host arithmetic tests")
        cls.temporary = tempfile.TemporaryDirectory(prefix="ipc-local-solve-fp32-")
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        source = root / "wrapper.cpp"
        source.write_text('#include "local_solve_fp32.cuh"\n'
                          'extern "C" int evaluate(const float* h, const float* f, '
                          'float damping, float* out) {\n'
                          'return ipc_gpu::local_solve_fp32::solve3(h, f, damping, out);\n}\n')
        library = root / ("local_solve.dylib" if sys.platform == "darwin" else "local_solve.so")
        command = [compiler, "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
                   "-shared", "-fPIC", "-I", str(GPU_ROOT), str(source), "-o", str(library)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        cls.library = ctypes.CDLL(str(library))
        cls.native = cls.library.evaluate
        fp32_pointer = ctypes.POINTER(ctypes.c_float)
        cls.native.argtypes = [fp32_pointer, fp32_pointer, ctypes.c_float, fp32_pointer]
        cls.native.restype = ctypes.c_int
        cls.backends = ("host C++",)

    def evaluate(self, matrices, forces, dampings, backend):
        del backend
        matrices, forces = np.ascontiguousarray(matrices, np.float32), np.ascontiguousarray(forces, np.float32)
        output = np.zeros((len(matrices), 3), np.float32)
        status = np.empty(len(matrices), np.int32)
        pointer = ctypes.POINTER(ctypes.c_float)
        for i, (h, f, damping) in enumerate(zip(matrices, forces, dampings)):
            status[i] = self.native(h.ctypes.data_as(pointer), f.ctypes.data_as(pointer),
                                    float(damping), output[i].ctypes.data_as(pointer))
        return output, status


WARP_AVAILABLE = importlib.util.find_spec("warp") is not None
if WARP_AVAILABLE:
    import warp as wp
    from gpu.dat_local_solve import configure_module, solve, solve_with_status

    @wp.kernel(module="unique", enable_backward=False)
    def evaluate_warp(h: wp.array[wp.mat33], f: wp.array[wp.vec3], damping: wp.array[float],
                      output: wp.array[wp.vec3], diagnostic: wp.array[wp.vec4]):
        i = wp.tid()
        output[i] = solve(h[i], f[i], damping[i])
        diagnostic[i] = solve_with_status(h[i], f[i], damping[i])

    configure_module(evaluate_warp.module)


@unittest.skipUnless(WARP_AVAILABLE, "Warp is not installed in this Python environment")
class WarpLocalSolveTests(NumericalSolveCases, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.backends = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def evaluate(self, matrices, forces, dampings, backend):
        output = wp.empty(len(matrices), wp.vec3, device=backend)
        diagnostic = wp.empty(len(matrices), wp.vec4, device=backend)
        wp.launch(evaluate_warp, len(matrices), inputs=[
            wp.array(matrices, wp.mat33, device=backend),
            wp.array(forces, wp.vec3, device=backend),
            wp.array(dampings, float, device=backend), output, diagnostic], device=backend)
        output_np, diagnostic_np = output.numpy(), diagnostic.numpy()
        # Both public entry points must agree, including failure handling.
        np.testing.assert_array_equal(output_np, diagnostic_np[:, :3])
        return output_np, diagnostic_np[:, 3].astype(np.int32)


@unittest.skipUnless(WARP_AVAILABLE, "Warp is not installed in this Python environment")
class SolverLocalSolveIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gpu
        gpu.select_newton_runtime("reference")
        from gpu.dat_tests.test_dat_structure import model_fixture
        from gpu.solver_dat import SolverDAT
        wp.init()
        cls.fixture = staticmethod(model_fixture)
        cls.solver_type = SolverDAT

    def test_pin_inertia_closed_form_and_damping(self):
        for damping in (.25, .3, 1.):
            with self.subTest(damping=damping):
                model = self.fixture(pin=80.)
                solver = self.solver_type(
                    model, iterations=1, allow_cpu=True, enable_planar_dat=False,
                    particle_enable_tile_solve=False, enable_fast_math=False,
                    local_solve="cpu_fp32", local_solve_damping=damping)
                state, output = model.state(), model.state()
                initial = state.particle_q.numpy().copy()
                dt = .05
                # Independent minimizer of 1/(2dt²)||x-x0||² + k/2||x-target||².
                expected = initial.astype(np.float64)
                expected[0] += (float(np.float32(damping)) * 80. / (1. / dt**2 + 80.)
                                * (model.ipc.pin_targets.numpy()[0] - initial[0]))
                solver.step(state, output, model.control(), None, dt)
                solver.check_status()
                np.testing.assert_allclose(output.particle_q.numpy(), expected,
                                           rtol=4.e-6, atol=3.e-8)
                np.testing.assert_array_equal(output.particle_q.numpy()[1:], initial[1:])

    def test_small_mass_and_stiffness_system_still_moves(self):
        # Uniformly scaling a quadratic objective must preserve its minimizer.
        # det(2e-6 I) is below the old absolute 1e-8 inverse gate.
        for objective_scale in (1., 1.e-6):
            with self.subTest(objective_scale=objective_scale):
                model = self.fixture(pin=objective_scale)
                model.particle_mass.assign(np.full(3, objective_scale, np.float32))
                model.particle_inv_mass.assign(np.full(3, 1. / objective_scale, np.float32))
                solver = self.solver_type(
                    model, iterations=1, allow_cpu=True, enable_planar_dat=False,
                    particle_enable_tile_solve=False, enable_fast_math=False,
                    local_solve="cpu_fp32")
                state, output = model.state(), model.state()
                expected = state.particle_q.numpy().copy()
                expected[0] = .5 * (expected[0] + model.ipc.pin_targets.numpy()[0])
                solver.step(state, output, model.control(), None, 1.)
                solver.check_status()
                np.testing.assert_allclose(output.particle_q.numpy(), expected,
                                           rtol=4.e-6, atol=3.e-8)

    def test_cuda_tiles_and_scalar_agree_for_fixed_and_dynamic_colors(self):
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA is required to exercise material tiles")
        from gpu.dat_tests.test_broad_phase_boxes import material_model, with_box_parameters
        from gpu.dat_tests.test_planar_toggle import fixture as pin_fixture

        for device in devices:
            for fixture_kind in ("material", "pin"):
                for combined_colors in (False, True):
                    with self.subTest(device=str(device), fixture=fixture_kind,
                                      combined_colors=combined_colors):
                        outcomes = []
                        for tiled in (False, True):
                            if fixture_kind == "material":
                                model = material_model(device)
                                model.gravity.zero_()
                                model.particle_qd.zero_()
                                q = model.particle_q.numpy().copy()
                                q[3, 2] = .2
                                model.particle_q.assign(q)
                                # Include the entire bending stencil in the
                                # fixed schedule too. Otherwise concurrent
                                # coupled updates would confound solve parity.
                                colors = np.arange(model.particle_count, dtype=np.int32)
                                model.particle_colors = wp.array(colors, int, device=device)
                                model.particle_color_groups = [wp.array([i], int, device=device)
                                                               for i in colors]
                                dt, iterations = .01, 2
                            else:
                                model = with_box_parameters(pin_fixture(
                                    device, velocity=(0., 0., 0.), pin=80., with_sdf=False))
                                dt, iterations = .05, 1
                            initial = model.particle_q.numpy().copy()
                            fixed = (model.particle_flags.numpy() & 1) == 0
                            solver = self.solver_type(
                                model, iterations=iterations, allow_cpu=True,
                                enable_planar_dat=False,
                                enable_broad_phase_boxes=combined_colors,
                                enable_barrier_coloring=combined_colors,
                                particle_enable_tile_solve=tiled, enable_fast_math=False,
                                local_solve="cpu_fp32", local_solve_damping=.3)
                            self.assertEqual(solver.use_particle_tile_solve, tiled)
                            state, output = model.state(), model.state()
                            solver.step(state, output, model.control(), None, dt)
                            solver.check_status()
                            position, velocity = output.particle_q.numpy(), output.particle_qd.numpy()
                            self.assertTrue(np.all(np.isfinite(position)))
                            self.assertTrue(np.all(np.isfinite(velocity)))
                            np.testing.assert_array_equal(position[fixed], initial[fixed])
                            self.assertGreater(np.linalg.norm(position[~fixed] - initial[~fixed]), 1.e-7)
                            if combined_colors:
                                solver.barrier_coloring.validate_independence()
                                solver.check_status()
                            if fixture_kind == "pin":
                                # The active node has mass two, zero velocity
                                # and no incident elements: a closed-form
                                # pin/inertia minimizer is independent of tiles.
                                expected = initial.astype(np.float64)
                                expected[0] += (float(np.float32(.3)) * 80. / (2. / dt**2 + 80.)
                                                * (model.ipc.pin_targets.numpy()[0] - initial[0]))
                                np.testing.assert_allclose(position, expected, rtol=4.e-6, atol=3.e-8)
                            outcomes.append((position.copy(), velocity.copy()))
                        # Element accumulation order differs between scalar
                        # and tiled assembly, so require FP32 agreement rather
                        # than byte equality after multiple material updates.
                        np.testing.assert_allclose(outcomes[1][0], outcomes[0][0],
                                                   rtol=3.e-5, atol=2.e-6)
                        np.testing.assert_allclose(outcomes[1][1], outcomes[0][1],
                                                   rtol=1.e-4, atol=2.e-4)


if __name__ == "__main__":
    unittest.main()
