"""DAT collision-processing options must reach the solver and distinguish restarts."""

import copy
import io
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from gpu import run, scene, solver_dat, dat_twist
from gpu.newton_dat import export_twist_trajectory


class PlanarCliTests(unittest.TestCase):
    def runtime(self):
        device = SimpleNamespace(is_cuda=True, alias="cuda:0", name="test CUDA")
        return None, SimpleNamespace(init=lambda: None, get_device=lambda _: device)

    def test_option_is_rejected_for_other_solvers_before_runtime_initialization(self):
        for solver in ("ipc", "dat-reference"):
            with patch.object(run, "_runtime") as runtime, redirect_stderr(io.StringIO()):
                self.assertEqual(run.main(["--solver", solver, "--no-planar-dat"]), 1)
                runtime.assert_not_called()
        for variant in ("reference", "gs"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                export_twist_trajectory.main(["--variant", variant, "--no-planar-dat", "--output", "/tmp/unused.npz"])
            self.assertEqual(result.exception.code, 2)

    def test_stock_scene_consumes_option_and_forwards_solver_keyword(self):
        for enabled in (True, False):
            arguments = ["--solver", "dat", "--stock-cloth-twist", "--num-frames", "480"]
            if not enabled:
                arguments.append("--no-planar-dat")
            with patch.object(run, "_runtime", side_effect=self.runtime), \
                 patch.object(dat_twist, "run_own", return_value=0) as own, redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(arguments), 0)
            forwarded = own.call_args.args[0]
            self.assertNotIn("--no-planar-dat", forwarded)
            self.assertEqual(own.call_args.kwargs, {"enable_planar_dat": enabled})
            self.assertIn("480", forwarded)

    def test_exporter_rejects_nonpositive_contact_capacities_before_runtime(self):
        for flag in ("--vertex-contact-capacity", "--edge-contact-capacity"):
            for value in ("0", "-1"):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                    export_twist_trajectory.main(["--variant", "own", flag, value, "--output", "/tmp/unused.npz"])
                self.assertEqual(result.exception.code, 2)

    def test_native_scene_option_changes_restart_signature_and_constructor_only_when_requested(self):
        baseline = {"params": {"d_hat": .002, "max_global_iters": 4}}
        signatures = []
        for enabled in (True, False):
            payload = copy.deepcopy(baseline)
            exporter = SimpleNamespace(scene=payload)
            manager = unittest.mock.MagicMock()
            manager.__enter__.return_value = exporter
            def create_model(data, device):
                return SimpleNamespace(ipc=SimpleNamespace(params=dict(data["params"])))
            arguments = ["--solver", "dat"] + ([] if enabled else ["--no-planar-dat"])
            with patch.object(run, "_runtime", side_effect=self.runtime), \
                 patch.object(scene, "find_exporter", return_value="unused"), \
                 patch.object(scene, "SceneExporter", return_value=manager), \
                 patch.object(scene, "create_model", side_effect=create_model), \
                 patch.object(solver_dat, "SolverDAT") as solver, \
                 patch.object(run, "run_simulation", return_value=0), redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(arguments), 0)
            self.assertEqual(solver.call_args.kwargs["enable_planar_dat"], enabled)
            self.assertIs(solver.call_args.kwargs["particle_enable_self_contact"], enabled)
            if enabled:
                self.assertEqual(payload, baseline)
            else:
                self.assertIs(payload["params"]["enable_planar_dat"], False)
                self.assertIs(payload["params"]["collision_processing"], False)
                previous_planar_only = copy.deepcopy(baseline)
                previous_planar_only["params"]["enable_planar_dat"] = False
                self.assertNotEqual(run.signature(payload, "dat"), run.signature(previous_planar_only, "dat"))
            signatures.append(run.signature(payload, "dat"))
        self.assertNotEqual(*signatures)

    def test_stock_graph_bypasses_external_collision_pipeline_only_when_disabled(self):
        class Original:
            def capture(self):
                self.collision_pipeline.collide(None, self.contacts)
                return "captured"
        for enabled in (True, False):
            example = dat_twist.make_example_type(Original, enable_collision_pipeline=enabled)()
            example.model = SimpleNamespace(particle_count=2500, tri_count=4802, shape_count=0)
            example.fps, example.sim_substeps, example.iterations = 60, 10, 4
            pipeline = MagicMock()
            contacts = object()
            example.collision_pipeline, example.contacts = pipeline, contacts
            self.assertEqual(example.capture(), "captured")
            if enabled:
                pipeline.collide.assert_called_once_with(None, contacts)
                self.assertIs(example.contacts, contacts)
            else:
                pipeline.collide.assert_not_called()
                self.assertIsNone(example.contacts)
                self.assertIsNone(example.collision_pipeline.contacts())


if __name__ == "__main__":
    unittest.main()
