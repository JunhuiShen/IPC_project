"""Actual Newton Model/State mapping and native prescribed-motion oracles."""

from copy import deepcopy
import os
from pathlib import Path
import unittest

import numpy as np
import warp as wp

from gpu.scene import SCHEMA, SceneExporter, create_model, update_boundary
from gpu.motion import BoundaryMotion


def cloth_payload():
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], dtype=float)
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    triangles = [[0, 1, 2], [1, 3, 2]]
    inverse = [np.linalg.inv(np.column_stack((uv[t[1]] - uv[t[0]], uv[t[2]] - uv[t[0]]))).reshape(-1).tolist() for t in triangles]
    return {"schema": SCHEMA, "kind": "scene", "example": 1,
        "params": {"mu": 10.0, "lambda": 12.0, "solid_mu": 20.0, "solid_lambda": 30.0,
                   "kB": 0.1, "kpin": 20.0, "gravity": [0, -9.81, 0], "rigid_pin_time": 0.0},
        "positions": x.tolist(), "velocities": [[0, 0, 0]] * 4, "mass": [1, 2, 3, 4],
        "node_to_rb": [-1] * 4, "tris": triangles, "tets": [], "triangle_dm_inverse": inverse,
        "triangle_areas": [0.5, 0.5], "hinges": [{"nodes": [1, 2, 0, 3], "coefficient": 2.0, "rest_angle": 0.0}],
        "pins": [{"node": 0, "target": [0, 0, 0], "prescribed": True}], "pin_targets": [[0, 0, 0]] * 4,
        "bodies": [], "tet_nodes": [], "surface_nodes": [], "rigid_pins": [], "rigid_pin_targets": [], "colliders": []}


class ModelMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.devices = [wp.get_device("cpu"), *wp.get_cuda_devices()]

    def test_actual_float32_model_preserves_materials_finite_pins_and_coloring(self):
        import newton
        for device in self.devices:
            with self.subTest(device=str(device)):
                payload = cloth_payload()
                model = create_model(payload, device)
                self.assertIsInstance(model, newton.Model)
                self.assertEqual(model.particle_q.dtype, wp.vec3)
                self.assertEqual(model.tri_poses.dtype, wp.mat22)
                self.assertEqual(model.state().particle_q.dtype, wp.vec3)
                np.testing.assert_array_equal(model.particle_q.numpy(), payload["positions"])
                np.testing.assert_array_equal(model.particle_mass.numpy(), payload["mass"])
                self.assertTrue(np.all(model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)))
                self.assertEqual(model.ipc.pinned.numpy()[0], 1)
                np.testing.assert_allclose(model.tri_materials.numpy()[:, :2], [[10, 12], [10, 12]])
                np.testing.assert_allclose(model.tri_poses.numpy().reshape(-1, 4), payload["triangle_dm_inverse"])
                self.assertEqual(model.edge_count, 5)
                edge = np.flatnonzero(model.ipc.edge_active.numpy())
                self.assertEqual(len(edge), 1)
                np.testing.assert_array_equal(model.edge_indices.numpy()[edge[0]], [0, 3, 1, 2])
                np.testing.assert_array_equal(model.ipc.edge_native_order.numpy()[edge[0]], [1, 2, 0, 3])
                self.assertAlmostEqual(float(model.ipc.edge_coefficient.numpy()[edge[0]]), 2.0)
                self.assertAlmostEqual(float(model.edge_bending_properties.numpy()[edge[0], 0]),
                                       payload["params"]["kB"])
                inactive = np.flatnonzero(model.ipc.edge_active.numpy() == 0)
                np.testing.assert_array_equal(model.edge_bending_properties.numpy()[inactive, 0], 0)
                colors = np.empty(4, dtype=int)
                for color, group in enumerate(model.particle_color_groups):
                    colors[group.numpy()] = color
                self.assertEqual(len(set(colors)), 4, "Newton bending coloring must separate the whole hinge")

    def test_rigid_proxy_particles_and_real_newton_bodies_retain_global_indices(self):
        payload = cloth_payload()
        reference = np.array([[-0.1, 0, 0], [0.1, 0, 0], [0, 0.1, 0]])
        com = np.array([0, 1, 0])
        payload["positions"] += (reference + com).tolist()
        payload["velocities"] += [[1, 2, 3]] * 3
        payload["mass"] += [0.2] * 3
        payload["node_to_rb"] += [0] * 3
        payload["pin_targets"] += [[0, 0, 0]] * 3
        payload["tris"] += [[4, 5, 6]]
        payload["bodies"] = [{"nodes": [4, 5, 6], "reference_positions": reference.tolist(), "mass": 0.6,
            "I_hat": np.diag([0.1, 0.2, 0.3]).reshape(-1).tolist(), "mode": 0, "translate": True, "rotate": True,
            "com": com.tolist(), "velocity": [1, 2, 3], "orientation": [1, 0, 0, 0], "omega": [0.1, 0.2, 0.3]}]
        for device in self.devices:
            model = create_model(payload, device)
            self.assertEqual((model.particle_count, model.body_count, model.joint_count), (7, 1, 0))
            np.testing.assert_array_equal(model.particle_mass.numpy()[4:], 0)
            np.testing.assert_array_equal(model.particle_flags.numpy()[4:], 0)
            np.testing.assert_allclose(model.body_q.numpy()[0], [0, 1, 0, 0, 0, 0, 1])
            np.testing.assert_allclose(model.body_qd.numpy()[0], [1, 2, 3, 0.1, 0.2, 0.3])
            np.testing.assert_allclose(model.body_inertia.numpy()[0], np.diag([0.5, 0.4, 0.3]))
            np.testing.assert_array_equal(model.ipc.tri_active.numpy(), [1, 1, 0])
            np.testing.assert_array_equal(model.ipc.body_references.numpy()[4:], reference.astype(np.float32))

    def test_solid_rest_data_is_native_and_surface_triangles_have_no_shell_energy(self):
        payload = cloth_payload()
        payload.update(positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            tris=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], tets=[[0, 1, 2, 3]],
            triangle_dm_inverse=[], triangle_areas=[], hinges=[], tet_nodes=[0, 1, 2, 3], surface_nodes=[0, 1, 2, 3],
            tet_rest_data=[{"dm_inverse": np.eye(3).reshape(-1).tolist(), "measure": 1 / 6,
                            "grad_N": [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]}])
        model = create_model(payload, "cpu")
        self.assertEqual(model.tet_count, 1)
        np.testing.assert_allclose(model.tet_poses.numpy()[0], np.eye(3))
        np.testing.assert_allclose(model.ipc.tet_grad_N.numpy()[0], payload["tet_rest_data"][0]["grad_N"])
        np.testing.assert_array_equal(model.tri_materials.numpy()[:, :2], 0)
        np.testing.assert_allclose(model.tet_materials.numpy()[0], [20, 30, 0])

    def test_boundary_upload_keeps_state_separate_and_rejects_topology_changes(self):
        payload = cloth_payload()
        model = create_model(payload, "cpu")
        state = model.state()
        before = state.particle_q.numpy().copy()
        boundary = {"schema": SCHEMA, "kind": "boundary", "pin_targets": [[0, 0.2, 0]] * 4,
                    "rigid_pin_targets": [], "colliders": []}
        update_boundary(model, boundary)
        np.testing.assert_array_equal(state.particle_q.numpy(), before)
        np.testing.assert_allclose(model.ipc.pin_targets.numpy(), boundary["pin_targets"])
        bad = deepcopy(boundary)
        bad["pin_targets"] = [[0, 0, 0]]
        with self.assertRaises(ValueError):
            update_boundary(model, bad)


class NativeMotionOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get("IPC_NEWTON_SCENE_EXPORTER")
        cls.exporter = Path(value) if value else Path(__file__).resolve().parents[2] / "build/3D_export_newton_scene"
        if not cls.exporter.is_file():
            raise unittest.SkipTest("Build 3D_export_newton_scene and set IPC_NEWTON_SCENE_EXPORTER")

    def test_gpu_authored_targets_match_original_native_updates(self):
        examples = (
            ["--example", "1", "--twist_nx", "3", "--twist_ny", "3"],
            ["--example", "2"],
            ["--example", "3"],
            ["--example", "12", "--osc_nx", "3", "--osc_nz", "3"],
        )
        for arguments in examples:
            with self.subTest(example=arguments[1]):
                with SceneExporter(self.exporter, [*arguments, "--fps", "30", "--substeps", "10", "--num_frames", "1"]) as exporter:
                    model = create_model(exporter.scene, "cpu")
                    motion = BoundaryMotion(model)
                    dt = exporter.scene["params"]["dt"]
                    for substep in (1, 15, 300, 750):
                        motion.reset(substep)
                        motion.apply(dt)
                        expected = exporter.boundary(substep * dt)
                        np.testing.assert_allclose(model.ipc.pin_targets.numpy(), np.asarray(expected["pin_targets"], dtype=np.float32), rtol=3e-6, atol=3e-6)
                        packed = model.ipc.collider_data.numpy()
                        for i, record in enumerate(expected["colliders"]):
                            np.testing.assert_allclose(packed[i, 1:4], record["point"], rtol=3e-6, atol=3e-6)
                            np.testing.assert_allclose(packed[i, 4:7], record["axis"], rtol=3e-6, atol=3e-6)
                            for name, begin, shift in (("previous_pose", 8, 17), ("current_pose", 20, 29)):
                                np.testing.assert_allclose(packed[i, begin:begin + 9], record[name]["rotation"], rtol=3e-6, atol=3e-6)
                                np.testing.assert_allclose(packed[i, shift:shift + 3], record[name]["translation"], rtol=3e-6, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
