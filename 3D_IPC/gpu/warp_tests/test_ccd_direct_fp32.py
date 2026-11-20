"""Safety regressions for closest-feature FP32 direct clipping.

The compiler-only suite uses independent exact-dyadic collision and endpoint-gap
oracles, plus analytic whole-path gap fixtures. Actual downloaded Warp
endpoints are checked by the same independent host oracle.
Optional Warp tests exercise the production header builder on CPU and CUDA.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

try:
    import numpy as np
    import warp as wp
except ImportError:
    np = wp = None


class Float32DirectHostTests(unittest.TestCase):
    def test_direct_clips_match_exact_geometry_for_every_moving_vertex_role(self):
        compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("A C++17 compiler is required")
        source = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="ipc-ccd-direct-fp32-") as temporary:
            executable = Path(temporary) / "checks"
            build = subprocess.run(
                [compiler, "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
                 "-I" + str(source.parent), str(source / "ccd_direct_fp32_checks.cpp"),
                 "-o", str(executable)], capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads(run.stdout)
        self.assertTrue(result["all_passed"])
        self.assertGreaterEqual(result["queries"], 280)
        self.assertGreaterEqual(result["crossings"], 40)
        self.assertGreaterEqual(result["accepted"], 100)
        self.assertGreaterEqual(result["clipped"], 40)
        self.assertGreaterEqual(result["exact_unsafe"], 40)
        self.assertGreaterEqual(result["rounding"], 7)
        self.assertGreaterEqual(result["degeneracies"], 12)
        self.assertGreaterEqual(result["invalid"], 17)
        self.assertGreaterEqual(result["gap_checks"], 100)
        self.assertGreaterEqual(result["analytic_gap_checks"], 1)
        self.assertEqual(result["subfloor"], 8)
        self.assertEqual(result["thin_features"], 32)

    def test_header_arithmetic_does_not_promote_to_double(self):
        compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("A C++17 compiler is required")
        gpu = Path(__file__).resolve().parent.parent
        source = '#include "ccd_direct_fp32.cuh"\nint main(){ipc_gpu::ccd32::LinearQuery q; return int(ipc_gpu::ccd32direct::propose_fraction(q));}\n'
        result = subprocess.run([compiler, "-std=c++17", "-fsyntax-only", "-x", "c++",
            "-Wdouble-promotion", "-Werror=double-promotion", "-I" + str(gpu), "-"],
            input=source, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if wp is not None:
    from gpu.warp_build import configure_module

    @wp.func_native(r"""
        ipc_gpu::ccd32::LinearQuery q;
        q.vertex_face = kind == 0;
        q.gap_floor = gap;
        for (int i=0; i<4; ++i) for (int k=0; k<3; ++k) {
            q.start[i][k] = start.data[4*index+i][k];
            q.end[i][k] = end.data[4*index+i][k];
        }
        const auto result = ipc_gpu::ccd32direct::clip_linear(q);
        fractions.data[index] = result.fraction;
        for (int i=0; i<4; ++i) for (int k=0; k<3; ++k)
            endpoints.data[4*index+i][k] = result.endpoint[i][k];
        return static_cast<int>(result.status);
    """)
    def _native_clip(start: wp.array[wp.vec3], end: wp.array[wp.vec3], index: int,
                     kind: int, gap: float, fractions: wp.array[float],
                     endpoints: wp.array[wp.vec3]) -> int: ...

    @wp.func_native(r"""
        ipc_gpu::ccd32::LinearQuery q;
        q.vertex_face = kind == 0;
        q.gap_floor = gap;
        for (int i=0; i<4; ++i) for (int k=0; k<3; ++k) {
            q.start[i][k] = start.data[4*index+i][k];
            q.end[i][k] = end.data[4*index+i][k];
        }
        return static_cast<int>(ipc_gpu::ccd32direct::validate_linear(q));
    """)
    def _native_validate(start: wp.array[wp.vec3], end: wp.array[wp.vec3], index: int,
                         kind: int, gap: float) -> int: ...

    @wp.kernel
    def _clip(start: wp.array[wp.vec3], end: wp.array[wp.vec3], kinds: wp.array[int],
              gaps: wp.array[float], fractions: wp.array[float], endpoints: wp.array[wp.vec3],
              status: wp.array[int], initial_status: wp.array[int]):
        i = wp.tid()
        initial_status[i] = _native_validate(start, end, i, kinds[i], gaps[i])
        status[i] = _native_clip(start, end, i, kinds[i], gaps[i], fractions, endpoints)

    @wp.kernel
    def _validate(start: wp.array[wp.vec3], end: wp.array[wp.vec3], kinds: wp.array[int],
                  gaps: wp.array[float], status: wp.array[int]):
        i = wp.tid()
        status[i] = _native_validate(start, end, i, kinds[i], gaps[i])

    configure_module(__name__, ("ccd_direct_fp32.cuh",))


@unittest.skipIf(wp is None, "Warp is not installed; compiler-only checks remain available")
class Float32DirectWarpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]
        compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            raise unittest.SkipTest("A host C++17 compiler is needed for independent exact replay")
        cls.temporary = tempfile.TemporaryDirectory(prefix="ipc-ccd-direct-oracle-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.oracle = Path(cls.temporary.name) / "checks"
        source = Path(__file__).resolve().parent
        build = subprocess.run([compiler, "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
            "-I" + str(source.parent), str(source / "ccd_direct_fp32_checks.cpp"), "-o", str(cls.oracle)],
            capture_output=True, text=True, timeout=60)
        if build.returncode:
            raise RuntimeError(build.stdout + build.stderr)

    def assert_exact_committed(self, cases):
        """Check the downloaded stored paths, without calling direct CCD in the oracle."""
        lines = [str(len(cases))]
        for start, end, kind, gap in cases:
            values = np.concatenate([np.asarray(start, np.float32).ravel(),
                                     np.asarray(end, np.float32).ravel()])
            lines.append(f"{kind} {float(np.float32(gap)):.9g} " +
                         " ".join(format(float(value), ".9g") for value in values))
        result = subprocess.run([str(self.oracle), "--oracle"], input="\n".join(lines) + "\n",
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout)["all_passed"])

    def test_each_moving_role_clips_and_revalidates_actual_endpoints_on_cpu_cuda(self):
        starts, ends, kinds, labels, movers = [], [], [], [], []
        for kind in (0, 1):
            for mover in range(4):
                start = np.array([[.25, .25, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
                                 if kind == 0 else
                                 [[-1, 0, 1], [1, 0, 1], [0, -1, 0], [0, 1, 0]], np.float32)
                end = start.copy()
                end[mover, 2] = (-1 if mover == 0 else (4 if mover == 1 else 8)) \
                    if kind == 0 else (-3 if mover < 2 else 4)
                for label in ("crossing", "departing", "unchanged", "invalid"):
                    proposal = end.copy()
                    if label == "departing":
                        proposal[mover] = start[mover] - np.float32(.125) * (end[mover] - start[mover])
                    elif label == "unchanged":
                        proposal = start.copy()
                    elif label == "invalid":
                        proposal[(mover + 1) % 4, 0] += np.float32(.125)
                    starts.append(start.copy()); ends.append(proposal)
                    kinds.append(kind); labels.append(label); movers.append(mover)
        count = len(kinds)
        for device in self.devices:
            with self.subTest(device=str(device)):
                a = wp.array(np.concatenate(starts), wp.vec3, device=device)
                b = wp.array(np.concatenate(ends), wp.vec3, device=device)
                k = wp.array(kinds, int, device=device)
                gaps = wp.full(count, 1e-8, float, device=device)
                fractions = wp.empty(count, float, device=device)
                endpoints = wp.empty(4*count, wp.vec3, device=device)
                status = wp.empty(count, int, device=device)
                original = wp.empty(count, int, device=device)
                verified = wp.empty(count, int, device=device)
                wp.launch(_clip, count, [a, b, k, gaps, fractions, endpoints, status, original], device=device)
                wp.launch(_validate, count, [a, endpoints, k, gaps, verified], device=device)
                actual_fraction = fractions.numpy()
                actual_end = endpoints.numpy().reshape(count, 4, 3)
                actual_status, original_status, replay_status = status.numpy(), original.numpy(), verified.numpy()
                for i, (label, mover) in enumerate(zip(labels, movers)):
                    with self.subTest(case=i, label=label):
                        if label == "invalid":
                            self.assertEqual(actual_status[i], 2)
                            self.assertEqual(original_status[i], 2)
                            continue
                        self.assertEqual(actual_status[i], 0)
                        self.assertEqual(replay_status[i], 0)
                        for vertex in range(4):
                            if vertex != mover:
                                np.testing.assert_array_equal(actual_end[i, vertex], starts[i][vertex])
                        if label == "crossing":
                            self.assertNotEqual(original_status[i], 0)
                            self.assertGreater(actual_fraction[i], 0)
                            self.assertLess(actual_fraction[i], .5)
                        else:
                            self.assertEqual(actual_fraction[i], 1)
                            np.testing.assert_array_equal(actual_end[i], ends[i])
                self.assert_exact_committed([(starts[i], actual_end[i], kinds[i], 1e-8)
                    for i in range(count) if labels[i] != "invalid"])

    def test_adversarial_batch_stored_endpoints_pass_independent_exact_oracles(self):
        cases = []
        def add(kind, height=1.0, mover=0, width=None, plane=0.0, floor=1e-8):
            start = np.array([[.25, .25, plane+height], [0, 0, plane],
                              [1, 0, plane], [0, 1, plane]] if kind == 0 else
                             [[-1, 0, plane+height], [1, 0, plane+height],
                              [0, -1, plane], [0, 1, plane]], np.float32)
            end = start.copy()
            end[mover, 2] = plane + ((-height if mover == 0 else (4 if mover == 1 else 8)*height)
                if kind == 0 else (-3*height if mover < 2 else 4*height))
            if width is not None:
                if kind == 0:
                    start[0, 1] = end[0, 1] = np.float32(width)*np.float32(.25)
                    start[3, 1] = end[3, 1] = np.float32(width)
                else:
                    start[:, :2] = end[:, :2] = [[-1, 0], [1, width], [-1, width], [1, 0]]
            cases.append((start, end, kind, floor))
        for kind in (0, 1):
            for height in (np.nextafter(np.float32(0), np.float32(1)), np.finfo(np.float32).tiny,
                           1e-30, 1e-10, 1e-8, .125, 1, 1e10):
                add(kind, height=float(height))
            for width in (2.0**-4, 2.0**-16, 2.0**-28, 2.0**-60):
                for mover in range(4):
                    add(kind, mover=mover, width=width)
            for plane in (100000.0, 2.0**30):
                above = np.nextafter(np.float32(plane), np.float32(np.inf))
                add(kind, height=float(above-np.float32(plane)), plane=plane)
                # A reduced fraction can store the exact contact coordinate.
                start, end, _, floor = cases[-1]
                rounded = end.copy(); rounded[0, 2] = np.float32(plane)
                cases.append((start.copy(), rounded, kind, floor))
            add(kind)
            start, end, _, floor = cases[-1]
            start[0, 2] = np.finfo(np.float32).max
            end[0, 2] = -np.finfo(np.float32).max
            add(kind)
            # Dhat=RN(-2**24-1) loses one unit before FMA reconstruction.
            cases[-1][1][0, 2] = np.float32(-16777216)
            # Feature change with a valid fixed z separator.
            add(kind)
            start, end, _, floor = cases[-1]
            end[0] = [4, -2, start[0, 2]]
            # Fully collapsed counterpart: exact collision remains meaningful.
            start = np.zeros((4, 3), np.float32)
            start[0, 0] = -1
            if kind == 1:
                start[1, 0] = -1
            end = start.copy(); end[0, 0] = 1
            cases.append((start, end, kind, 1e-8))
            for height in (np.nextafter(np.float32(.01), np.float32(0)), np.float32(.01)):
                add(kind, height=float(height), floor=.01)
        # More than one 64-thread block exercises tails without a large corpus.
        self.assertGreater(len(cases), 64)
        for device in self.devices:
            with self.subTest(device=str(device)):
                count = len(cases)
                a = wp.array(np.concatenate([q[0] for q in cases]), wp.vec3, device=device)
                b = wp.array(np.concatenate([q[1] for q in cases]), wp.vec3, device=device)
                kinds = wp.array([q[2] for q in cases], int, device=device)
                gaps = wp.array([q[3] for q in cases], float, device=device)
                fractions = wp.empty(count, float, device=device)
                endpoints = wp.empty(4*count, wp.vec3, device=device)
                status = wp.empty(count, int, device=device)
                original = wp.empty(count, int, device=device)
                verified = wp.empty(count, int, device=device)
                wp.launch(_clip, count, [a, b, kinds, gaps, fractions, endpoints, status, original],
                          device=device, block_dim=64)
                wp.launch(_validate, count, [a, endpoints, kinds, gaps, verified],
                          device=device, block_dim=64)
                np.testing.assert_array_equal(status.numpy(), np.zeros(count, np.int32))
                np.testing.assert_array_equal(verified.numpy(), np.zeros(count, np.int32))
                actual = endpoints.numpy().reshape(count, 4, 3)
                alpha = fractions.numpy()
                self.assertTrue(np.all(np.isfinite(actual)))
                self.assertTrue(np.all(np.isfinite(alpha) & (alpha >= 0) & (alpha <= 1)))
                self.assert_exact_committed([(start, actual[i], kind, floor)
                    for i, (start, _, kind, floor) in enumerate(cases)])

    def test_opt_in_backend_all_moving_roles_preserve_unrelated_motion_and_default(self):
        from gpu.ccd import CcdBackend
        from .test_ccd_independent import accept, fixture
        for device in self.devices:
            cases = []
            for edge in (False, True):
                for mover in range(4):
                    with self.subTest(device=str(device), edge=edge, mover=mover):
                        groups = [1, 2, 3, 4, 0]
                        groups[mover] = 0
                        data = fixture(device, edge=edge, groups=groups)
                        self.assertEqual(CcdBackend(data.contacts, node_box_radius=2).method, "interval")
                        data.model.ipc = SimpleNamespace(params={"ccd_method": "direct_rounding"})
                        backend = CcdBackend(data.contacts, node_box_radius=2)
                        self.assertEqual(backend.method, "direct_rounding")
                        proposed = data.start.copy()
                        proposed[mover, 2] = (-.7 if mover < 2 else .7) if edge else (
                            -.1 if mover == 0 else (.4 if mover == 3 else .8))
                        proposed[4, 0] += np.float32(.25)
                        accept(data, backend, wp.array(proposed, wp.vec3, device=device), bounded=False)
                        backend.check()
                        actual = data.q.numpy()
                        self.assertFalse(np.array_equal(actual[mover], data.start[mover]))
                        self.assertLess(float(backend.node_fractions.numpy()[mover]), .5)
                        stationary = [i for i in range(4) if i != mover]
                        np.testing.assert_array_equal(actual[stationary], data.start[stationary])
                        np.testing.assert_array_equal(actual[4], proposed[4])
                        self.assertEqual(int(backend.attempts.numpy()[0]), 1)
                        self.assertEqual(int(backend.verified.numpy()[0]), 1)
                        cases.append((data.start[:4], actual[:4], int(edge), 1e-7))
            self.assert_exact_committed(cases)

    def test_opt_in_backend_rejects_two_movers_before_any_commit(self):
        from gpu.ccd import CcdBackend
        from .test_ccd_independent import accept, fixture
        for device in self.devices:
            for edge in (False, True):
                with self.subTest(device=str(device), edge=edge):
                    data = fixture(device, edge=edge, groups=[0, 0, 2, 3, 0])
                    data.model.ipc = SimpleNamespace(params={"ccd_method": "direct_rounding"})
                    backend = CcdBackend(data.contacts, node_box_radius=2)
                    proposed = data.start.copy()
                    proposed[0, 2] -= np.float32(.2)
                    proposed[1, 0] += np.float32(.1)
                    proposed[4, 0] += np.float32(.4)
                    accept(data, backend, wp.array(proposed, wp.vec3, device=device), bounded=False)
                    self.assertEqual(int(data.contacts.error.numpy()[0]), 205)
                    with self.assertRaises(RuntimeError):
                        backend.check()
                    np.testing.assert_array_equal(data.q.numpy(), data.start)

    def test_cuda_graph_replay_resets_fractions_and_uses_fresh_geometry(self):
        from gpu.ccd import CcdBackend
        from .test_ccd_independent import accept, fixture
        if not wp.get_cuda_devices():
            self.skipTest("No CUDA device")
        for device in wp.get_cuda_devices():
            data = fixture(device)
            data.model.ipc = SimpleNamespace(params={"ccd_method": "direct_rounding"})
            saved = wp.clone(data.q)
            proposed = data.start.copy()
            proposed[0, 2] = -.2; proposed[4, 0] += np.float32(.25)
            proposal = wp.array(proposed, wp.vec3, device=device)
            backend = CcdBackend(data.contacts, node_box_radius=1)
            accept(data, backend, proposal, bounded=False)
            backend.check()
            with wp.ScopedCapture(device=device) as capture:
                wp.copy(data.q, saved)
                data.contacts.refresh()
                data.bank.build(data.colors)
                data.view = data.bank.view(data.current)
                accept(data, backend, proposal, bounded=False)
            previous_epoch = int(data.contacts.query_epoch.numpy()[0])
            cases = []
            # A full departing step between two different crossing steps must
            # not inherit a stale reduced fraction or rejection from replay.
            for target in (-.3, .4, -.6):
                proposed[0, 2] = target
                proposed[4, 0] = data.start[4, 0] + np.float32(abs(target))
                proposal.assign(proposed)
                wp.capture_launch(capture.graph)
                backend.check()
                actual = data.q.numpy()
                if target > 0:
                    np.testing.assert_array_equal(actual[0], proposed[0])
                    self.assertEqual(float(backend.node_fractions.numpy()[0]), 1)
                else:
                    self.assertGreater(actual[0, 2], np.float32(1e-7))
                    self.assertLess(actual[0, 2], data.start[0, 2])
                np.testing.assert_array_equal(actual[4], proposed[4])
                np.testing.assert_array_equal(actual[1:4], data.start[1:4])
                epoch = int(data.contacts.query_epoch.numpy()[0])
                self.assertGreater(epoch, previous_epoch)
                previous_epoch = epoch
                cases.append((data.start[:4], actual[:4], 0, 1e-7))
            self.assert_exact_committed(cases)

    def test_shared_vertex_reduction_is_order_independent_and_recertifies_stored_endpoints(self):
        from gpu.ccd import CcdBackend
        from gpu.contact import ContactBackend
        from gpu.query_groups import QueryGroups
        from newton._src.geometry.tri_mesh_collision import build_tri_mesh_collision_info
        from .test_ccd_independent import accept
        for device in self.devices:
            cases = []
            for large in (False, True):
                outputs = []
                for reverse in (False, True):
                    with self.subTest(device=str(device), large=large, reverse=reverse):
                        plane = np.float32(100000 if large else 0)
                        ulp = np.nextafter(plane, np.float32(np.inf)) - plane if large else np.float32(.05)
                        planes = [plane, plane + ulp] if large else [plane, plane - ulp]
                        height = plane + np.float32(2) * ulp
                        target = plane - np.float32(2 if large else 4) * ulp
                        points = [[0, 0, height]]
                        for z in planes:
                            points.extend([[-1, -1, z], [1, -1, z], [0, 1, z]])
                        points.append([3, 2, height])
                        start = np.asarray(points, np.float32)
                        q = wp.array(start, wp.vec3, device=device)
                        colors = wp.array([0, 1, 2, 3, 4, 5, 6, 0], int, device=device)
                        model = SimpleNamespace(device=device, particle_count=8, tri_count=2,
                            edge_count=0, body_count=0, tet_count=0, particle_q=q,
                            tri_indices=wp.array([[1, 2, 3], [4, 5, 6]], int, device=device),
                            edge_indices=wp.array(np.empty((0, 4), np.int32), int, device=device),
                            particle_colors=colors, particle_flags=wp.ones(8, int, device=device),
                            particle_mass=wp.ones(8, float, device=device), particle_world=wp.zeros(8, int, device=device),
                            ipc=SimpleNamespace(params={"ccd_method": "direct_rounding"}))
                        info = build_tri_mesh_collision_info(8, 2, 0,
                            vertex_collision_buffer_pre_alloc=2, edge_collision_buffer_pre_alloc=2, device=device)
                        values = np.full(info.vertex_colliding_triangles.shape[0], -1, np.int32)
                        values[:4] = [0, 1, 0, 0] if reverse else [0, 0, 0, 1]
                        info.vertex_colliding_triangles.assign(values)
                        info.vertex_colliding_triangles_count.assign([2, 0, 0, 0, 0, 0, 0, 0])
                        contacts = ContactBackend(model, info, d_hat=.005, k_barrier=100, track_query_epoch=True)
                        contacts.refresh()
                        bank = QueryGroups(contacts); bank.build(colors)
                        current = wp.zeros(1, int, device=device)
                        data = SimpleNamespace(q=q, colors=colors, current=current, view=bank.view(current))
                        proposed = start.copy(); proposed[0, 2] = target; proposed[7, 0] += np.float32(.25)
                        backend = CcdBackend(contacts, node_box_radius=2)
                        accept(data, backend, wp.array(proposed, wp.vec3, device=device), bounded=False)
                        backend.check()
                        actual = q.numpy()
                        np.testing.assert_array_equal(actual[1:7], start[1:7])
                        np.testing.assert_array_equal(actual[7], proposed[7])
                        self.assertGreater(float(actual[0, 2]), float(max(planes)))
                        self.assertLess(float(backend.node_fractions.numpy()[0]), .5)
                        self.assertLessEqual(actual[0, 2], start[0, 2])
                        if not large:
                            self.assertLess(actual[0, 2], start[0, 2])
                            self.assertEqual(int(backend.plane_rejected.numpy()[0]), 0)
                        # The rounding-aware proposal may preemptively block a
                        # few-ULP move, or endpoint verification may restore it.
                        # Safety and independent motion are the contract.
                        for nodes in ([0, 1, 2, 3], [0, 4, 5, 6]):
                            cases.append((start[nodes], actual[nodes], 0, 1e-7))
                        outputs.append(actual.copy())
                np.testing.assert_array_equal(outputs[0], outputs[1])
            self.assert_exact_committed(cases)


if __name__ == "__main__":
    unittest.main()
