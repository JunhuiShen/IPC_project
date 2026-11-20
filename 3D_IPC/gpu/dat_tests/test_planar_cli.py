"""DAT collision-processing options must reach the solver and distinguish restarts."""

import copy
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
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
        for enabled, boxes, coloring, color_ccd in ((True, False, False, False),
                (False, False, False, False), (False, True, False, False),
                (False, True, True, False), (False, True, True, True)):
            arguments = ["--solver", "dat", "--stock-cloth-twist", "--num-frames", "480"]
            if not enabled:
                arguments.append("--no-planar-dat")
            if boxes:
                arguments.append("--build-broad-phase-boxes")
            if coloring:
                arguments.append("--color-barrier-pairs")
            if color_ccd:
                arguments.append("--color-ccd")
            with patch.object(run, "_runtime", side_effect=self.runtime), \
                 patch.object(dat_twist, "run_own", return_value=0) as own, redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(arguments), 0)
            forwarded = own.call_args.args[0]
            self.assertNotIn("--no-planar-dat", forwarded)
            self.assertNotIn("--build-broad-phase-boxes", forwarded)
            self.assertNotIn("--color-barrier-pairs", forwarded)
            self.assertNotIn("--color-ccd", forwarded)
            self.assertEqual(own.call_args.kwargs,
                             {"enable_planar_dat": enabled, "enable_broad_phase_boxes": boxes,
                              "enable_barrier_coloring": coloring, "enable_color_ccd": color_ccd,
                              "vertex_contact_capacity": None,
                              "edge_contact_capacity": None})
            self.assertIn("480", forwarded)

    def test_boxes_option_requires_collision_off_own_dat_before_runtime(self):
        for solver, flags in (("dat", []), ("ipc", []), ("dat-reference", []),
                              ("ipc", ["--no-planar-dat"]), ("dat-reference", ["--no-planar-dat"])):
            with patch.object(run, "_runtime") as runtime, redirect_stderr(io.StringIO()):
                self.assertEqual(run.main(["--solver", solver, "--build-broad-phase-boxes", *flags]), 1)
                runtime.assert_not_called()
        for variant, flags in (("own", []), ("gs", []), ("reference", []),
                               ("gs", ["--no-planar-dat"]), ("reference", ["--no-planar-dat"])):
            with patch("gpu.select_newton_runtime") as runtime, redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as result:
                export_twist_trajectory.main(["--variant", variant, "--build-broad-phase-boxes",
                                              *flags, "--output", "/tmp/unused.npz"])
            self.assertEqual(result.exception.code, 2)
            runtime.assert_not_called()

    def test_coloring_requires_own_dat_with_boxes_and_disabled_planar_dat_before_runtime(self):
        for mode in ("ipc", "dat", "dat-reference"):
            for flags in ([], ["--no-planar-dat"], ["--build-broad-phase-boxes"],
                          ["--no-planar-dat", "--build-broad-phase-boxes"]):
                if mode == "dat" and len(flags) == 2:
                    continue
                with patch.object(run, "_runtime") as runtime, redirect_stderr(io.StringIO()):
                    self.assertEqual(run.main(["--solver", mode, "--color-barrier-pairs", *flags]), 1)
                    runtime.assert_not_called()
        for variant in ("reference", "own", "gs"):
            for flags in ([], ["--no-planar-dat"], ["--build-broad-phase-boxes"],
                          ["--no-planar-dat", "--build-broad-phase-boxes"]):
                if variant == "own" and len(flags) == 2:
                    continue
                with patch("gpu.select_newton_runtime") as runtime, redirect_stderr(io.StringIO()), \
                     self.assertRaises(SystemExit) as result:
                    export_twist_trajectory.main(["--variant", variant, "--color-barrier-pairs",
                                                  *flags, "--output", "/tmp/unused.npz"])
                self.assertEqual(result.exception.code, 2)
                runtime.assert_not_called()

    def test_color_ccd_requires_all_stage_options_before_runtime(self):
        prerequisites = ("--no-planar-dat", "--build-broad-phase-boxes", "--color-barrier-pairs")
        for mask in range(8):
            flags = [flag for i, flag in enumerate(prerequisites) if mask & (1 << i)]
            for mode in ("ipc", "dat", "dat-reference"):
                if mode == "dat" and mask == 7:
                    continue
                with self.subTest(mode=mode, flags=flags), \
                     patch.object(run, "_runtime") as runtime, redirect_stderr(io.StringIO()):
                    self.assertEqual(run.main(["--solver", mode, "--color-ccd", *flags]), 1)
                    runtime.assert_not_called()
            for variant in ("reference", "own", "gs"):
                if variant == "own" and mask == 7:
                    continue
                with self.subTest(variant=variant, flags=flags), \
                     patch("gpu.select_newton_runtime") as runtime, redirect_stderr(io.StringIO()), \
                     self.assertRaises(SystemExit) as result:
                    export_twist_trajectory.main(["--variant", variant, "--color-ccd", *flags,
                                                  "--output", "/tmp/unused.npz"])
                self.assertEqual(result.exception.code, 2)
                runtime.assert_not_called()

    def test_stock_scene_forwards_explicit_candidate_capacities(self):
        with patch.object(run, "_runtime", side_effect=self.runtime), \
             patch.object(dat_twist, "run_own", return_value=0) as own, redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(["--solver", "dat", "--stock-cloth-twist", "--no-planar-dat",
                                       "--build-broad-phase-boxes", "--color-barrier-pairs",
                                       "--vertex-contact-capacity", "48", "--edge-contact-capacity", "96"]), 0)
        self.assertEqual(own.call_args.kwargs["vertex_contact_capacity"], 48)
        self.assertEqual(own.call_args.kwargs["edge_contact_capacity"], 96)
        self.assertNotIn("--vertex-contact-capacity", own.call_args.args[0])
        self.assertNotIn("--edge-contact-capacity", own.call_args.args[0])

    def test_level_guess_controls_are_consumed_for_stock_dat(self):
        for planar in (True,False):
            arguments=["--solver","dat","--stock-cloth-twist","--level-based-ccd-guess",
                       "--colored-ccd-guess-iters","3","--colored-ccd-guess-fraction","0.9"]
            if not planar:arguments.append("--no-planar-dat")
            with patch.object(run,"_runtime",side_effect=self.runtime), \
                 patch.object(dat_twist,"run_own",return_value=0) as own,redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(arguments),0)
            self.assertIs(own.call_args.kwargs["level_based_ccd_guess"],True)
            self.assertEqual(own.call_args.kwargs["colored_ccd_guess_iters"],3)
            self.assertEqual(own.call_args.kwargs["colored_ccd_guess_fraction"],.9)
            self.assertIs(own.call_args.kwargs["enable_planar_dat"],planar)
            self.assertNotIn("--level-based-ccd-guess",own.call_args.args[0])
            self.assertNotIn("--colored-ccd-guess-iters",own.call_args.args[0])

    def test_level_guess_rejects_wrong_modes_and_unselected_controls_before_runtime(self):
        for mode in ("ipc","dat-reference"):
            with patch.object(run,"_runtime") as runtime,redirect_stderr(io.StringIO()):
                self.assertEqual(run.main(["--solver",mode,"--level-based-ccd-guess"]),1)
                runtime.assert_not_called()
        for flag,value in (("--colored-ccd-guess-iters","2"),("--colored-ccd-guess-fraction",".9")):
            with patch.object(run,"_runtime") as runtime,redirect_stderr(io.StringIO()):
                self.assertEqual(run.main(["--solver","dat",flag,value]),1)
                runtime.assert_not_called()
        for variant,flags in (("reference",["--level-based-ccd-guess"]),
                              ("gs",["--level-based-ccd-guess"]),
                              ("own",["--colored-ccd-guess-iters","3"]),
                              ("own",["--colored-ccd-guess-fraction",".8"])):
            with patch("gpu.select_newton_runtime") as runtime,redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                export_twist_trajectory.main(["--variant",variant,*flags,"--output","/tmp/unused.npz"])
            runtime.assert_not_called()

    def test_level_guess_rejects_invalid_controls_before_runtime(self):
        for flag,value in (("--colored-ccd-guess-iters","-1"),("--colored-ccd-guess-iters","1.5"),
                           ("--colored-ccd-guess-fraction","nan"),("--colored-ccd-guess-fraction","inf"),
                           ("--colored-ccd-guess-fraction","-0.1"),("--colored-ccd-guess-fraction","1.1")):
            with patch.object(run,"_runtime") as runtime,redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                run.main(["--solver","dat","--level-based-ccd-guess",flag,value])
            runtime.assert_not_called()
            with patch("gpu.select_newton_runtime") as runtime,redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                export_twist_trajectory.main(["--variant","own","--level-based-ccd-guess",flag,value,"--output","/tmp/unused.npz"])
            runtime.assert_not_called()

    def test_gs_collision_option_is_explicit_and_not_forwarded_to_scene(self):
        for enabled in (True, False):
            arguments = ["--solver", "ipc", "--stock-cloth-twist", "--num-frames", "480"]
            if not enabled:
                arguments.append("--no-collisions")
            with patch.object(run, "_runtime", side_effect=self.runtime), \
                 patch.object(dat_twist, "run_gs", return_value=0) as gs, redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(arguments), 0)
            self.assertNotIn("--no-collisions", gs.call_args.args[0])
            self.assertEqual(gs.call_args.kwargs, {"enable_collisions": enabled})
        for mode in ("dat", "dat-reference"):
            with patch.object(run, "_runtime") as runtime, redirect_stderr(io.StringIO()):
                self.assertEqual(run.main(["--solver", mode, "--no-collisions"]), 1)
                runtime.assert_not_called()
        for variant in ("reference", "own"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                export_twist_trajectory.main(["--variant", variant, "--no-collisions", "--output", "/tmp/unused.npz"])
            self.assertEqual(result.exception.code, 2)

    def test_exporter_rejects_nonpositive_contact_capacities_before_runtime(self):
        for flag in ("--vertex-contact-capacity", "--edge-contact-capacity"):
            for value in ("0", "-1"):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                    export_twist_trajectory.main(["--variant", "own", flag, value, "--output", "/tmp/unused.npz"])
                self.assertEqual(result.exception.code, 2)

    def test_native_scene_option_changes_restart_signature_and_constructor_only_when_requested(self):
        baseline = {"params": {"d_hat": .002, "max_global_iters": 4}}
        signatures = []
        for enabled, boxes, coloring, color_ccd in ((True, False, False, False),
                (False, False, False, False), (False, True, False, False),
                (False, True, True, False), (False, True, True, True)):
            payload = copy.deepcopy(baseline)
            exporter = SimpleNamespace(scene=payload)
            manager = unittest.mock.MagicMock()
            manager.__enter__.return_value = exporter
            def create_model(data, device):
                return SimpleNamespace(ipc=SimpleNamespace(params=dict(data["params"])))
            arguments = ["--solver", "dat"] + ([] if enabled else ["--no-planar-dat"])
            if boxes:
                arguments.append("--build-broad-phase-boxes")
            if coloring:
                arguments.append("--color-barrier-pairs")
            if color_ccd:
                arguments.append("--color-ccd")
            with patch.object(run, "_runtime", side_effect=self.runtime), \
                 patch.object(scene, "find_exporter", return_value="unused"), \
                 patch.object(scene, "SceneExporter", return_value=manager), \
                 patch.object(scene, "create_model", side_effect=create_model), \
                 patch.object(solver_dat, "SolverDAT") as solver, \
                 patch.object(run, "run_simulation", return_value=0), redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(arguments), 0)
            self.assertEqual(solver.call_args.kwargs["enable_planar_dat"], enabled)
            self.assertEqual(solver.call_args.kwargs["enable_broad_phase_boxes"], boxes)
            self.assertEqual(solver.call_args.kwargs["enable_barrier_coloring"], coloring)
            self.assertEqual(solver.call_args.kwargs["enable_color_ccd"], color_ccd)
            self.assertIs(solver.call_args.kwargs["particle_enable_self_contact"], enabled)
            self.assertEqual(solver.call_args.kwargs["vertex_contact_capacity"], 128)
            self.assertEqual(solver.call_args.kwargs["edge_contact_capacity"], 256)
            if enabled:
                self.assertEqual(payload, baseline)
            else:
                self.assertIs(payload["params"]["enable_planar_dat"], False)
                self.assertIs(payload["params"]["collision_processing"], color_ccd)
                previous_planar_only = copy.deepcopy(baseline)
                previous_planar_only["params"]["enable_planar_dat"] = False
                self.assertNotEqual(run.signature(payload, "dat"), run.signature(previous_planar_only, "dat"))
            if boxes:
                self.assertIs(payload["params"]["enable_broad_phase_boxes"], True)
            else:
                self.assertNotIn("enable_broad_phase_boxes", payload["params"])
            if coloring:
                self.assertIs(payload["params"]["enable_barrier_coloring"], True)
            else:
                self.assertNotIn("enable_barrier_coloring", payload["params"])
            if color_ccd:
                self.assertIs(payload["params"]["enable_color_ccd"], True)
                self.assertIs(payload["params"]["solver_collision_processing"], True)
            else:
                self.assertNotIn("enable_color_ccd", payload["params"])
            signatures.append(run.signature(payload, "dat"))
        self.assertEqual(len(set(signatures)), 5)

    def test_stock_solver_factory_forwards_boxes_without_reenabling_collision_pipeline(self):
        import newton.solvers

        model = object()
        solver = MagicMock()
        original = newton.solvers.SolverVBD

        def run_example(arguments, *, enable_collision_pipeline):
            self.assertFalse(enable_collision_pipeline)
            self.assertIs(newton.solvers.SolverVBD(model, iterations=4), solver)
            return 0

        with patch.object(dat_twist, "attach_stock_ipc") as attach, \
             patch.object(solver_dat, "SolverDAT", return_value=solver) as construct, \
             patch.object(dat_twist, "run", side_effect=run_example):
            self.assertEqual(dat_twist.run_own([], enable_planar_dat=False, enable_broad_phase_boxes=True), 0)
        attach.assert_called_once_with(model, gs_predictor="none")
        construct.assert_called_once_with(model, enable_planar_dat=False, enable_broad_phase_boxes=True,
                                          enable_barrier_coloring=False, enable_color_ccd=False, iterations=4)
        solver.check_status.assert_called_once_with()
        self.assertIs(newton.solvers.SolverVBD, original)

    def test_native_level_guess_override_is_in_checkpoint_signature(self):
        baseline={"params":{"d_hat":.002,"max_global_iters":4,"use_ccd_guess":True,
                            "use_colored_ccd_guess":True,"use_colored_ccd_guess_level_based":False}}
        signatures=[]
        for cap,fraction,color_ccd in ((1,1.,False),(2,1.,False),(2,.9,False),(0,.9,False),
                                       (0,.9,True),(2,.9,True)):
            payload=copy.deepcopy(baseline)
            manager=MagicMock()
            manager.__enter__.return_value=SimpleNamespace(scene=payload)
            def create_model(data,device):
                return SimpleNamespace(ipc=SimpleNamespace(params=dict(data["params"])))
            extra = (["--build-broad-phase-boxes", "--color-barrier-pairs", "--color-ccd"]
                     if color_ccd else [])
            with patch.object(run,"_runtime",side_effect=self.runtime), \
                 patch.object(scene,"find_exporter",return_value="unused"), \
                 patch.object(scene,"SceneExporter",return_value=manager), \
                 patch.object(scene,"create_model",side_effect=create_model), \
                 patch.object(solver_dat,"SolverDAT") as solver, \
                 patch.object(run,"run_simulation",return_value=0),redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(["--solver","dat","--no-planar-dat","--level-based-ccd-guess",
                    "--colored-ccd-guess-iters",str(cap),"--colored-ccd-guess-fraction",str(fraction),*extra]),0)
            params=solver.call_args.args[0].ipc.params
            self.assertIs(params["use_colored_ccd_guess_level_based"],True)
            self.assertIs(params["use_ccd_guess"],False)
            self.assertIs(params["use_colored_ccd_guess"],False)
            self.assertEqual(params["colored_ccd_guess_iters"],cap)
            self.assertEqual(params["colored_ccd_guess_fraction"],fraction)
            self.assertIs(params["solver_collision_processing"],color_ccd)
            self.assertIs(params["collision_processing"],color_ccd or cap>0)
            self.assertIs(solver.call_args.kwargs["enable_color_ccd"],color_ccd)
            signatures.append(run.signature(payload,"dat"))
        self.assertEqual(len(set(signatures)),6)

    def test_stock_solver_factory_selects_level_guess_without_reenabling_dat(self):
        import newton.solvers
        model=object();solver=MagicMock()
        def run_example(arguments,*,enable_collision_pipeline):
            self.assertFalse(enable_collision_pipeline)
            self.assertIs(newton.solvers.SolverVBD(model,iterations=4),solver)
            return 0
        with patch.object(dat_twist,"attach_stock_ipc") as attach, \
             patch.object(solver_dat,"SolverDAT",return_value=solver) as construct, \
             patch.object(dat_twist,"run",side_effect=run_example):
            self.assertEqual(dat_twist.run_own([],enable_planar_dat=False,level_based_ccd_guess=True,
                colored_ccd_guess_iters=2,colored_ccd_guess_fraction=.97),0)
        attach.assert_called_once_with(model,gs_predictor="level",colored_ccd_guess_iters=2,colored_ccd_guess_fraction=.97)
        self.assertIs(construct.call_args.kwargs["enable_planar_dat"],False)

    def test_stock_solver_factory_forwards_coloring_and_overrides_candidate_capacity(self):
        import newton.solvers

        model = object()
        statistics = {"checkpoints": 12, "pair_count": {"min": 3, "max": 8, "mean": 5.5}}
        solver = MagicMock()
        solver.barrier_coloring.statistics.return_value = statistics
        original = newton.solvers.SolverVBD

        def run_example(arguments, *, enable_collision_pipeline):
            self.assertFalse(enable_collision_pipeline)
            self.assertIs(newton.solvers.SolverVBD(model, iterations=4,
                particle_vertex_contact_buffer_size=64, particle_edge_contact_buffer_size=128), solver)
            solver.barrier_coloring.statistics.assert_not_called()
            return 0

        output = io.StringIO()
        with patch.object(dat_twist, "attach_stock_ipc"), \
             patch.object(solver_dat, "SolverDAT", return_value=solver) as construct, \
             patch.object(dat_twist, "run", side_effect=run_example), redirect_stdout(output):
            self.assertEqual(dat_twist.run_own([], enable_planar_dat=False, enable_broad_phase_boxes=True,
                enable_barrier_coloring=True, enable_color_ccd=True,
                vertex_contact_capacity=48, edge_contact_capacity=96), 0)
        construct.assert_called_once_with(model, enable_planar_dat=False, enable_broad_phase_boxes=True,
            enable_barrier_coloring=True, enable_color_ccd=True,
            iterations=4, particle_vertex_contact_buffer_size=48,
            particle_edge_contact_buffer_size=96)
        solver.check_status.assert_called_once_with()
        solver.barrier_coloring.statistics.assert_called_once_with()
        self.assertEqual(json.loads(output.getvalue())["barrier_coloring_statistics"], statistics)
        self.assertIs(newton.solvers.SolverVBD, original)

    def _export_mock_trajectory(self, coloring,guess=False,cap=2,fraction=.9,method="interval",color_ccd=False,
                                contact_forces=False,k_barrier=100.,friction=.2,color_method="direct_rounding",
                                accept_uncertified=False):
        import numpy as np
        import newton.examples
        import newton.solvers
        import warp as wp

        def array(value):
            return SimpleNamespace(numpy=lambda: np.asarray(value))

        q = array(np.zeros((2500, 3), dtype=np.float32))
        model = SimpleNamespace(particle_q=q, edge_count=4, ipc=SimpleNamespace(
            params={"k_barrier":k_barrier,"friction_coefficient":friction}),
            tri_materials=array([[1., 1.]]), edge_bending_properties=array([[1., 0.]]),
            soft_contact_ke=100., soft_contact_kd=0., soft_contact_mu=.2)
        solver = SimpleNamespace(configuration={"enable_broad_phase_boxes": True}, check_status=MagicMock())
        solver.color_contact_forces = (SimpleNamespace(configuration={
            "geometry_and_derivative_precision": "float64"}) if contact_forces else None)
        statistics = {"checkpoints": 4800, "pair_count": {"min": 2, "max": 8, "mean": 5.0},
                      "color_count": {"min": 4, "max": 9, "mean": 6.5}}
        solver.barrier_coloring = SimpleNamespace(statistics=MagicMock(return_value=statistics),
                                                  reset_statistics=MagicMock())
        example = SimpleNamespace(model=model, solver=solver, faces=np.zeros((4802, 3), dtype=np.int32),
            fps=60, sim_substeps=10, iterations=4, rot_angular_velocity=dat_twist.ANGULAR_VELOCITY,
            rot_axes=array([[1., 0., 0.], [-1., 0., 0.]]), rot_end_time=8., graph=None,
            state_0=SimpleNamespace(particle_q=q, particle_qd=q), step=MagicMock(), test_final=MagicMock())
        viewer = MagicMock()
        device = SimpleNamespace(is_cuda=True, alias="cuda:0", name="test CUDA")

        def example_type(original, *, enable_collision_pipeline):
            self.assertFalse(enable_collision_pipeline)
            def construct(viewer, args):
                self.assertIs(newton.solvers.SolverVBD(model, iterations=4), solver)
                return example
            return construct

        def read_statistics():
            self.assertEqual(example.step.call_count, 480)
            self.assertEqual(timer.call_count, 960)
            solver.check_status.assert_called_once_with()
            return statistics
        solver.barrier_coloring.statistics.side_effect = read_statistics

        def step():
            if coloring:
                solver.barrier_coloring.reset_statistics.assert_called_once_with()
            solver.barrier_coloring.statistics.assert_not_called()
        example.step.side_effect = step

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "boxes.npz"
            with patch.object(wp, "init"), patch.object(wp, "set_device"), \
                 patch.object(wp, "get_device", return_value=device), patch.object(wp, "synchronize_device"), \
                 patch.object(newton.examples, "init", return_value=(viewer, object())), \
                 patch.object(dat_twist, "make_example_type", side_effect=example_type), \
                 patch.object(dat_twist, "attach_stock_ipc", return_value={"existing_arrays_unchanged": True,
                              "params": model.ipc.params}) as attach, \
                 patch.object(solver_dat, "SolverDAT", return_value=solver) as construct, \
                 patch.object(export_twist_trajectory, "_source_hashes", return_value={}), \
                 patch.object(export_twist_trajectory.time, "perf_counter", side_effect=range(960)) as timer, \
                 redirect_stdout(io.StringIO()):
                extra = (["--color-barrier-pairs", "--vertex-contact-capacity", "48",
                          "--edge-contact-capacity", "96"] if coloring else [])
                if guess:extra += ["--level-based-ccd-guess","--colored-ccd-guess-iters",str(cap),
                                   "--colored-ccd-guess-fraction",str(fraction)]
                if method != "interval":extra += ["--ccd-method",method]
                if color_ccd:extra.append("--color-ccd")
                if color_method != "direct_rounding":extra += ["--color-ccd-method",color_method]
                if accept_uncertified:extra.append("--color-ccd-accept-uncertified")
                if contact_forces:extra.append("--color-ccd-contact-forces")
                self.assertEqual(export_twist_trajectory.main([
                    "--variant", "own", "--no-planar-dat", "--build-broad-phase-boxes",
                    *extra, "--output", str(output)]), 0)
            capacity_options = ({"particle_vertex_contact_buffer_size": 48,
                                 "particle_edge_contact_buffer_size": 96} if coloring else {})
            construct.assert_called_once_with(model, enable_planar_dat=False,
                                               enable_broad_phase_boxes=True, enable_barrier_coloring=coloring,
                                               enable_color_ccd=color_ccd,
                                               **({"color_ccd_method":color_method} if color_method != "direct_rounding" else {}),
                                               **({"color_ccd_accept_uncertified":True} if accept_uncertified else {}),
                                               iterations=4,
                                               **({"enable_color_ccd_contact_forces":True} if contact_forces else {}),
                                               **capacity_options)
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(model.ipc.params["ccd_method"], method)
            self.assertEqual(metadata["bridge_receipt"]["params"]["ccd_method"], method)
            self.assertIs(metadata["broad_phase_boxes_enabled"], True)
            for key in ("planar_dat_enabled", "sdf_collision_energies_enabled"):
                self.assertIs(metadata[key], False)
            self.assertIs(metadata["color_ccd_contact_forces_enabled"],contact_forces)
            self.assertEqual(metadata["contact_precision"], "float64" if contact_forces else None)
            self.assertIs(metadata["mesh_collision_energies_enabled"],contact_forces and k_barrier>0.)
            self.assertIs(metadata["friction_enabled"],contact_forces and k_barrier>0. and friction>0.)
            if contact_forces:
                formulas = metadata["energy_formulas"]
                self.assertIn("project corotated analytic self-Hessian", formulas)
                self.assertIn("project hinge Gauss-Newton self-Hessian", formulas)
                self.assertIn("full project IPC and frozen-friction gradient/Hessian", formulas)
                self.assertIn("evaluated in binary64 and assembled in FP32 per color", formulas)
                self.assertNotIn("native friction", formulas)
                self.assertNotIn("rank-one contact Hessian", formulas)
                self.assertNotIn("contact energies, friction, SDF and DAT motion limits disabled", formulas)
            for key in ("solver_collision_processing_enabled", "collision_response_enabled", "color_ccd_enabled"):
                self.assertIs(metadata[key], color_ccd)
            self.assertEqual(metadata["color_ccd_method"], color_method if color_ccd else None)
            self.assertIs(metadata["color_ccd_accept_uncertified"], accept_uncertified)
            self.assertIs(metadata["collision_processing_enabled"],color_ccd or (guess and cap>0))
            self.assertIs(metadata["predictor_ccd_enabled"],guess and cap>0)
            for key in ("candidate_pair_generation_enabled", "combined_coloring_enabled",
                        "use_colors_for_sweeps", "barrier_coloring_enabled"):
                self.assertIs(metadata[key], coloring)
            self.assertIs(metadata["collision_queries_enabled"],coloring or (guess and cap>0))
            if guess:
                attach.assert_called_once_with(model,gs_predictor="level",colored_ccd_guess_iters=cap,
                                               colored_ccd_guess_fraction=fraction)
                self.assertEqual(metadata["level_based_ccd_initial_guess"]["updating_cap"],cap)
                self.assertEqual(metadata["level_based_ccd_initial_guess"]["target_fraction"],fraction)
                self.assertEqual(metadata["level_based_ccd_initial_guess"]["method"],method)
                self.assertNotIn("contact energies, CCD and DAT motion limits disabled",metadata["energy_formulas"])
                self.assertIn("level-based initial-guess CCD",metadata["energy_formulas"])
            if color_ccd:
                self.assertIn(f"actual color updates use gathered FP32 {color_method} CCD", metadata["energy_formulas"])
                self.assertIn("per-vertex atomicMin", metadata["energy_formulas"])
            self.assertEqual(metadata["barrier_coloring_statistics"], statistics if coloring else None)
            self.assertEqual(metadata["measured_step_seconds_sum"], 480.)
            self.assertIn("coloring statistics downloads", metadata["timer_excludes"])
            self.assertIs(metadata["final_row_capacity"]["enabled"], False)
            with np.load(output) as trajectory:
                self.assertEqual(json.loads(str(trajectory["metadata_json"])), metadata)
        solver.check_status.assert_called_once_with()
        if coloring:
            solver.barrier_coloring.statistics.assert_called_once_with()
            solver.barrier_coloring.reset_statistics.assert_called_once_with()
            self.assertIn("combined material/pair coloring drive VBD sweeps", metadata["energy_formulas"])
            self.assertEqual(metadata["requested_contact_capacities"], {"vertex": 48, "edge": 96})
        else:
            solver.barrier_coloring.statistics.assert_not_called()
            solver.barrier_coloring.reset_statistics.assert_not_called()

    def test_exporter_forwards_boxes_and_records_disabled_collision_stages(self):
        self._export_mock_trajectory(False)

    def test_exporter_forwards_coloring_and_records_untimed_statistics_and_stage_metadata(self):
        self._export_mock_trajectory(True)

    def test_exporter_distinguishes_predictor_ccd_from_collision_disabled_solve(self):
        self._export_mock_trajectory(True,guess=True)

    def test_exporter_records_zero_cap_without_claiming_predictor_ccd_work(self):
        self._export_mock_trajectory(False,guess=True,cap=0)

    def test_exporter_direct_methods_reach_runtime_and_record_selected_method(self):
        for method in ("plane", "direct_rounding"):
            with self.subTest(method=method):
                self._export_mock_trajectory(True,guess=True,method=method)

    def test_exporter_color_ccd_is_independent_of_initial_guess_method(self):
        self._export_mock_trajectory(True, color_ccd=True)
        for method in ("interval", "plane", "direct_rounding"):
            with self.subTest(method=method):
                self._export_mock_trajectory(True, guess=True, method=method, color_ccd=True)
        self._export_mock_trajectory(True, guess=True, cap=0, color_ccd=True)

    def test_exporter_plane_color_method_is_independent_and_records_metadata(self):
        self._export_mock_trajectory(True, color_ccd=True, color_method="plane")
        self._export_mock_trajectory(True, guess=True, method="direct_rounding",
                                     color_ccd=True, color_method="plane", contact_forces=True)

    def test_uncertified_color_ccd_option_is_forwarded_and_recorded(self):
        self._export_mock_trajectory(True, color_ccd=True, color_method="plane", accept_uncertified=True)
        with patch.object(run, "_runtime", side_effect=self.runtime), \
             patch.object(dat_twist, "run_own", return_value=0) as own, redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(["--solver", "dat", "--stock-cloth-twist", "--no-planar-dat",
                "--build-broad-phase-boxes", "--color-barrier-pairs", "--color-ccd",
                "--color-ccd-method", "plane", "--color-ccd-accept-uncertified"]), 0)
        self.assertIs(own.call_args.kwargs["color_ccd_accept_uncertified"], True)

    def test_uncertified_color_ccd_option_requires_plane_before_runtime(self):
        flags = ["--no-planar-dat", "--build-broad-phase-boxes", "--color-barrier-pairs",
                 "--color-ccd", "--color-ccd-accept-uncertified"]
        with patch.object(run, "_runtime") as runtime, redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(run.main(["--solver", "dat", *flags]), 1)
            runtime.assert_not_called()
        self.assertIn("requires --color-ccd --color-ccd-method plane", errors.getvalue())
        with patch("gpu.select_newton_runtime") as runtime, redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as result:
                export_twist_trajectory.main(["--variant", "own", "--output", "/tmp/unused.npz", *flags])
            self.assertEqual(result.exception.code, 2)
            runtime.assert_not_called()
        self.assertIn("requires --color-ccd --color-ccd-method plane", errors.getvalue())

    def test_plane_color_method_requires_color_ccd_before_runtime(self):
        with patch.object(run, "_runtime") as runtime, redirect_stderr(io.StringIO()):
            self.assertEqual(run.main(["--solver", "dat", "--color-ccd-method", "plane"]), 1)
            runtime.assert_not_called()
        with patch("gpu.select_newton_runtime") as runtime, redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as result:
            export_twist_trajectory.main(["--variant", "own", "--color-ccd-method", "plane",
                                          "--output", "/tmp/unused.npz"])
        self.assertEqual(result.exception.code, 2)
        runtime.assert_not_called()

    def test_stock_methods_are_consumed_and_forwarded_independently(self):
        with patch.object(run, "_runtime", side_effect=self.runtime), \
             patch.object(dat_twist, "run_own", return_value=0) as own, redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(["--solver", "dat", "--stock-cloth-twist", "--no-planar-dat",
                "--build-broad-phase-boxes", "--color-barrier-pairs", "--color-ccd",
                "--color-ccd-method", "plane", "--level-based-ccd-guess", "--ccd-method", "direct_rounding"]), 0)
        self.assertEqual(own.call_args.kwargs["color_ccd_method"], "plane")
        self.assertEqual(own.call_args.kwargs["ccd_method"], "direct_rounding")
        self.assertNotIn("--color-ccd-method", own.call_args.args[0])
        self.assertNotIn("--ccd-method", own.call_args.args[0])

    def test_plane_color_selection_changes_native_restart_signature(self):
        signatures = []
        for color_method in ("direct_rounding", "plane"):
            payload = {"params": {"d_hat": .002, "max_global_iters": 4}}
            manager = MagicMock()
            manager.__enter__.return_value = SimpleNamespace(scene=payload)
            def create_model(data, device):
                return SimpleNamespace(ipc=SimpleNamespace(params=dict(data["params"])))
            with patch.object(run, "_runtime", side_effect=self.runtime), \
                 patch.object(scene, "find_exporter", return_value="unused"), \
                 patch.object(scene, "SceneExporter", return_value=manager), \
                 patch.object(scene, "create_model", side_effect=create_model), \
                 patch.object(solver_dat, "SolverDAT") as solver, \
                 patch.object(run, "run_simulation", return_value=0), redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(["--solver", "dat", "--no-planar-dat", "--build-broad-phase-boxes",
                    "--color-barrier-pairs", "--color-ccd", "--color-ccd-method", color_method,
                    "--level-based-ccd-guess", "--ccd-method", "direct_rounding"]), 0)
            self.assertEqual(payload["params"]["ccd_method"], "direct_rounding")
            self.assertEqual(payload["params"].get("color_ccd_method", "direct_rounding"), color_method)
            self.assertEqual(solver.call_args.kwargs.get("color_ccd_method", "direct_rounding"), color_method)
            signatures.append(run.signature(payload, "dat"))
        self.assertNotEqual(*signatures)

    def test_contact_forces_require_color_ccd_before_runtime(self):
        flags = ["--no-planar-dat", "--build-broad-phase-boxes", "--color-barrier-pairs"]
        for mode in ("ipc", "dat", "dat-reference"):
            with patch.object(run,"_runtime") as runtime, redirect_stderr(io.StringIO()):
                self.assertEqual(run.main(["--solver",mode,"--color-ccd-contact-forces",*flags]),1)
                runtime.assert_not_called()
        for variant in ("own","reference","gs"):
            with patch("gpu.select_newton_runtime") as runtime, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                export_twist_trajectory.main(["--variant",variant,"--color-ccd-contact-forces",*flags,
                                              "--output","/tmp/unused.npz"])
            runtime.assert_not_called()

    def test_stock_contact_forces_flag_consumed_and_forwarded(self):
        flags=["--no-planar-dat","--build-broad-phase-boxes","--color-barrier-pairs",
               "--color-ccd","--color-ccd-contact-forces"]
        with patch.object(run,"_runtime",side_effect=self.runtime), \
             patch.object(dat_twist,"run_own",return_value=0) as own,redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(["--solver","dat","--stock-cloth-twist",*flags]),0)
        self.assertIs(own.call_args.kwargs["enable_color_ccd_contact_forces"],True)
        self.assertIs(own.call_args.kwargs["enable_planar_dat"],False)
        self.assertNotIn("--color-ccd-contact-forces",own.call_args.args[0])

    def test_native_contact_forces_changes_checkpoint_signature(self):
        signatures=[]
        for enabled in (False,True):
            payload={"params":{"d_hat":.002,"max_global_iters":4}}
            manager=MagicMock();manager.__enter__.return_value=SimpleNamespace(scene=payload)
            flags=["--no-planar-dat","--build-broad-phase-boxes","--color-barrier-pairs","--color-ccd"]
            if enabled:flags.append("--color-ccd-contact-forces")
            with patch.object(run,"_runtime",side_effect=self.runtime), \
                 patch.object(scene,"find_exporter",return_value="unused"), \
                 patch.object(scene,"SceneExporter",return_value=manager), \
                 patch.object(scene,"create_model",side_effect=lambda data,device: SimpleNamespace(ipc=SimpleNamespace(params=data["params"]))), \
                 patch.object(solver_dat,"SolverDAT") as construct, \
                 patch.object(run,"run_simulation",return_value=0),redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(["--solver","dat",*flags]),0)
            self.assertIs(construct.call_args.kwargs.get("enable_color_ccd_contact_forces",False),enabled)
            self.assertIs(payload["params"].get("enable_color_ccd_contact_forces",False),enabled)
            self.assertIs(construct.call_args.kwargs["particle_enable_self_contact"],False)
            signatures.append(run.signature(payload,"dat"))
        self.assertNotEqual(*signatures)

    def test_stock_contact_forces_factory_keeps_native_dat_off(self):
        import newton.solvers
        model=object();solver=MagicMock()
        solver.barrier_coloring.statistics.return_value={"error":0}
        def run_example(arguments,*,enable_collision_pipeline):
            self.assertFalse(enable_collision_pipeline)
            self.assertIs(newton.solvers.SolverVBD(model,iterations=4),solver)
            return 0
        with patch.object(dat_twist,"attach_stock_ipc"), \
             patch.object(solver_dat,"SolverDAT",return_value=solver) as construct, \
             patch.object(dat_twist,"run",side_effect=run_example),redirect_stdout(io.StringIO()):
            self.assertEqual(dat_twist.run_own([],enable_planar_dat=False,enable_broad_phase_boxes=True,
                enable_barrier_coloring=True,enable_color_ccd=True,enable_color_ccd_contact_forces=True),0)
        self.assertIs(construct.call_args.kwargs["enable_color_ccd_contact_forces"],True)
        self.assertIs(construct.call_args.kwargs["enable_planar_dat"],False)

    def test_exporter_records_contact_forces_and_zero_coefficients(self):
        for stiffness,mu in ((100.,.2),(100.,0.),(0.,.2)):
            with self.subTest(stiffness=stiffness,mu=mu):
                self._export_mock_trajectory(True,guess=True,method="direct_rounding",color_ccd=True,
                    contact_forces=True,k_barrier=stiffness,friction=mu)

    def test_exporter_provenance_includes_actual_color_ccd_sources(self):
        sources = export_twist_trajectory._source_hashes(Path(__file__).resolve().parents[2])
        for name in ("gpu/dat_color_ccd.py", "gpu/ccd_direct_fp32.cuh", "gpu/dat_color_contacts.py"):
            self.assertIn(name, sources)
            self.assertEqual(len(sources[name]), 64)

    def test_exporter_direct_methods_require_level_guess_before_runtime(self):
        for method in ("plane", "direct_rounding"):
            with patch("gpu.select_newton_runtime") as runtime, redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit):
                export_twist_trajectory.main(["--variant", "own", "--ccd-method", method,
                                              "--output", "/tmp/unused.npz"])
            runtime.assert_not_called()

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
