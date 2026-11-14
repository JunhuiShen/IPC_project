"""Native scenes represented by the actual Newton Model/State GPU layout.

Only immutable rest data and prescribed boundary inputs are prepared here.
All global vertices keep their native indices. Rigid proxy vertices are
inactive Newton particles, while their poses live in genuine Newton body
state arrays. Newton's own mesh adjacency and coloring are used by SolverIPC.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np


SCHEMA = "ipc-newton-scene-v1"


class SceneExporter:
    def __init__(self, executable, arguments):
        self.process = subprocess.Popen([str(executable), *arguments], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            self.scene = self._read("scene")
        except BaseException:
            self.close()
            raise

    def _read(self, kind):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Native scene exporter stopped before emitting {kind} (exit {self.process.poll()})")
        payload = json.loads(line)
        if payload.get("schema") != SCHEMA or payload.get("kind") != kind:
            raise ValueError(f"Expected a {kind} export message")
        return payload

    def boundary(self, time):
        if not math.isfinite(time) or time < 0:
            raise ValueError("Boundary time must be finite and nonnegative")
        self.process.stdin.write(format(float(time), ".17g") + "\n")
        self.process.stdin.flush()
        return self._read("boundary")

    def close(self):
        process = getattr(self, "process", None)
        if process is not None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
            if process.stdout:
                process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def find_exporter(explicit=None):
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise ValueError(f"Native scene exporter does not exist: {candidate}")
    root = Path(__file__).resolve().parents[1]
    for candidate in (root / "gpu/build/3D_export_newton_scene", root / "build/3D_export_newton_scene",
                      root / "gpu/build/cpu/3D_export_newton_scene"):
        if candidate.is_file():
            return candidate
    raise RuntimeError("Build 3D_export_newton_scene and pass --scene-exporter /path/to/3D_export_newton_scene")


def _array(values, width, dtype, name):
    result = np.asarray(values, dtype=dtype)
    if result.size % width:
        raise ValueError(f"Malformed {name}")
    return result.reshape((-1, width))


def _finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite {name}")


def _collider_data(records):
    """Flat original SDF records; poses describe material motion, not geometry."""
    result = np.zeros((len(records), 32), dtype=np.float32)
    kinds = {"plane": 0, "cylinder": 1, "sphere": 2}
    for i, record in enumerate(records):
        if record["kind"] not in kinds:
            raise ValueError("Unsupported native SDF collider")
        result[i, 0] = kinds[record["kind"]]
        result[i, 1:4] = record["point"]
        result[i, 4:7] = record["axis"]
        result[i, 7] = record["radius"]
        for key, r, t in (("previous_pose", 8, 17), ("current_pose", 20, 29)):
            pose = record[key]
            result[i, r:r + 9] = np.asarray(pose["rotation"]).reshape(9)
            result[i, t:t + 3] = pose["translation"]
    _finite(result, "SDF geometry and poses")
    return result


def create_model(payload, device="cuda:0"):
    """Build an actual Newton Model and attach native immutable ``model.ipc`` data."""
    import newton
    import warp as wp

    if payload.get("schema") != SCHEMA or payload.get("kind") != "scene":
        raise ValueError("Not an exported native IPC scene")
    p = dict(payload["params"])
    for flag in ("use_dat", "use_ogc", "use_cloth_grid", "use_jacobi_level_based_ccd"):
        if p.get(flag, False):
            raise ValueError(f"{flag} requests a different solver; this backend uses our GS equations and CCD")
    x = _array(payload["positions"], 3, np.float64, "positions")
    velocity = _array(payload["velocities"], 3, np.float64, "velocities")
    mass = np.asarray(payload["mass"], dtype=np.float64)
    owner = np.asarray(payload.get("node_to_rb", [-1] * len(x)), dtype=np.int32)
    bodies = payload.get("bodies", [])
    n = len(x)
    if n == 0 or velocity.shape != x.shape or mass.shape != (n,) or owner.shape != (n,):
        raise ValueError("Inconsistent native particle arrays")
    if np.any(owner < -1) or np.any(owner >= len(bodies)) or np.any(mass < 0):
        raise ValueError("Invalid native mass/ownership")
    for values, name in ((x, "positions"), (velocity, "velocities"), (mass, "mass")):
        _finite(values, name)
    # Conversion is explicit: Model/State retain the upstream float32 layout.
    x32, velocity32 = x.astype(np.float32), velocity.astype(np.float32)
    _finite(x32, "float32 positions"); _finite(velocity32, "float32 velocities")
    triangles = _array(payload.get("tris", []), 3, np.int32, "triangles")
    tets = _array(payload.get("tets", []), 4, np.int32, "tetrahedra")
    for cells in (triangles, tets):
        if cells.size and (np.any(cells < 0) or np.any(cells >= n)):
            raise ValueError("Native topology has an out-of-range particle")
    deformable = (owner < 0).astype(np.int32)
    solid = np.zeros(n, dtype=np.int32)
    solid[np.asarray(payload.get("tet_nodes", []), dtype=np.int32)] = 1
    surface = 1 - solid
    surface[np.asarray(payload.get("surface_nodes", []), dtype=np.int32)] = 1
    pinned = np.zeros(n, dtype=np.int32)
    targets = _array(payload.get("pin_targets", np.zeros_like(x)), 3, np.float32, "pin targets")
    if targets.shape != x.shape:
        raise ValueError("Pin target array must retain every global particle")
    for pin in payload.get("pins", []):
        node = int(pin["node"])
        if not 0 <= node < n or owner[node] >= 0:
            raise ValueError("Soft pins must address independently deformable particles")
        pinned[node] = 1
    _finite(targets, "pin targets")

    builder = newton.ModelBuilder(gravity=tuple(p["gravity"]))
    flags = np.where((deformable != 0) & (mass > 0), int(newton.ParticleFlags.ACTIVE), 0).astype(np.int32)
    active_mass = np.where(deformable != 0, mass, 0.0)
    active_mass32 = active_mass.astype(np.float32)
    if not np.isfinite(active_mass32).all() or np.any((active_mass > 0) & (active_mass32 <= 0)):
        raise ValueError("Native particle mass is outside the float32 state range")
    builder.add_particles(x32.tolist(), velocity32.tolist(), active_mass32.tolist(), radius=[0.0] * n, flags=flags.tolist())
    native_inverse = _array(payload.get("triangle_dm_inverse", []), 4, np.float32, "triangle rest matrices")
    native_area = np.asarray(payload.get("triangle_areas", []), dtype=np.float32)
    if len(native_inverse) != len(native_area) or len(native_inverse) > len(triangles):
        raise ValueError("Inconsistent native shell rest records")
    tri_active = np.zeros(len(triangles), dtype=np.int32)
    for i, tri in enumerate(triangles[:len(native_inverse)]):
        tri_active[i] = int(np.all(deformable[tri] != 0) and np.all(solid[tri] == 0))
    if len(triangles):
        builder.add_triangles(triangles[:, 0], triangles[:, 1], triangles[:, 2],
            tri_ke=(tri_active * float(p["mu"])).tolist(), tri_ka=(tri_active * float(p["lambda"])).tolist(),
            tri_kd=[0.0] * len(triangles), tri_drag=[0.0] * len(triangles), tri_lift=[0.0] * len(triangles))
        if len(builder.tri_indices) != len(triangles):
            raise ValueError("Newton rejected a degenerate native surface triangle; global mapping cannot be preserved")
        for i in np.flatnonzero(tri_active):
            builder.tri_poses[i] = native_inverse[i].reshape(2, 2).tolist()
            builder.tri_areas[i] = float(native_area[i])
    rest_tets = payload.get("tet_rest_data", [])
    if len(rest_tets) != len(tets):
        raise ValueError("Native tetrahedron rest records are incomplete")
    builder.tet_indices = tets.tolist()
    builder.tet_poses = [np.asarray(r["dm_inverse"], dtype=np.float32).reshape(3, 3).tolist() for r in rest_tets]
    builder.tet_activations = [0.0] * len(tets)
    builder.tet_materials = [(float(p["solid_mu"]), float(p["solid_lambda"]), 0.0)] * len(tets)
    grad_n = np.asarray([r["grad_N"] for r in rest_tets], dtype=np.float32).reshape((-1, 4, 3))

    # Preserve all collision edges, including boundary and rigid edges. Only
    # native shell hinges carry bending stiffness or add bending-color edges.
    edge_faces = {}
    for tri in triangles:
        for j in range(3):
            a, b, opposite = int(tri[j]), int(tri[(j + 1) % 3]), int(tri[(j + 2) % 3])
            edge_faces.setdefault(tuple(sorted((a, b))), []).append((a, b, opposite))
    native_hinges = {tuple(sorted(h["nodes"][:2])): h for h in payload.get("hinges", [])}
    coefficients, native_orders, edge_active, edge_specs, edge_angles, edge_stiffness = [], [], [], [], [], []
    for key in sorted(edge_faces):
        hinge = native_hinges.get(key)
        active = hinge is not None and all(deformable[v] and not solid[v] for v in hinge["nodes"])
        if active:
            e0, e1, opposite0, opposite1 = map(int, hinge["nodes"])
            coefficient = float(hinge["coefficient"])
            rest_angle = float(hinge["rest_angle"])
        else:
            e0, e1 = key
            incident = edge_faces[key]
            positive = next((v for a, b, v in incident if a == e0), -1)
            negative = next((v for a, b, v in incident if a == e1), -1)
            opposite0, opposite1 = positive, negative
            coefficient, rest_angle = 0.0, 0.0
        edge_specs.append((opposite0, opposite1, e0, e1))
        edge_angles.append(rest_angle)
        # Our hinge evaluator applies the native geometric coefficient once.
        # Newton's material slot therefore stores the unweighted bending kB.
        edge_stiffness.append(float(p["kB"]) if active else 0.0)
        coefficients.append(coefficient); native_orders.append((e0, e1, opposite0, opposite1)); edge_active.append(int(active))
    if edge_specs:
        specs = np.asarray(edge_specs, dtype=np.int32)
        builder.add_edges(specs[:, 0], specs[:, 1], specs[:, 2], specs[:, 3], rest=edge_angles,
                          edge_ke=edge_stiffness, edge_kd=[0.0] * len(specs))

    references = np.zeros((n, 3), dtype=np.float32)
    body_nodes, body_offsets = [], [0]
    for i, body in enumerate(bodies):
        ids = np.asarray(body["nodes"], dtype=np.int32)
        refs = _array(body["reference_positions"], 3, np.float32, "body reference")
        if len(ids) != len(refs) or np.any(owner[ids] != i):
            raise ValueError("Rigid reference records do not match global ownership")
        moment = np.asarray(body["I_hat"], dtype=np.float64).reshape(3, 3)
        inertia = np.trace(moment) * np.eye(3) - moment
        _finite(inertia, "rigid inertia")
        qw, qx, qy, qz = map(float, body["orientation"])
        b = builder.add_link(xform=wp.transform(wp.vec3(*body["com"]), wp.quat(qx, qy, qz, qw)),
            com=wp.vec3(0.0), mass=float(body["mass"]), inertia=wp.mat33(*inertia.reshape(-1)),
            lock_inertia=True, is_kinematic=not (body["translate"] or body["rotate"]), label=f"ipc_body_{i}")
        builder.body_qd[b] = wp.spatial_vector(*body["velocity"], *body["omega"])
        references[ids] = refs
        body_nodes.extend(ids.tolist()); body_offsets.append(len(body_nodes))
    builder.color(include_bending=True)
    # The native exporter already owns the physical rest data. Preserve it;
    # all structural/topological validations remain enabled.
    model = builder.finalize(device=device, skip_validation_inertia=True)
    if model.particle_count != n or model.tri_count != len(triangles) or model.tet_count != len(tets):
        raise RuntimeError("Newton changed native scene indexing")

    def upload(values, dtype, shape=None):
        values = np.asarray(values)
        if shape is not None:
            values = values.reshape(shape)
        return wp.array(values, dtype=dtype, device=model.device)

    pins = payload.get("rigid_pins", [])
    collider_host = _collider_data(payload.get("colliders", []))
    ipc = SimpleNamespace(payload=payload, params=p, precision="float32", node_count=n,
        node_body=upload(owner, wp.int32), deformable=upload(deformable, wp.int32), solid=upload(solid, wp.int32),
        surface=upload(surface, wp.int32), pinned=upload(pinned, wp.int32), pin_targets=upload(targets, wp.vec3),
        original_mass=upload(mass.astype(np.float32), wp.float32), tri_inverse=model.tri_poses, tri_area=model.tri_areas,
        tri_active=upload(tri_active, wp.int32), tet_inverse=model.tet_poses,
        tet_measure=upload([r["measure"] for r in rest_tets], wp.float32),
        tet_grad_N=wp.array(grad_n, dtype=wp.vec3, ndim=2, device=model.device),
        edge_coefficient=upload(coefficients, wp.float32), edge_native_order=upload(native_orders, wp.vec4i, (-1, 4)),
        edge_active=upload(edge_active, wp.int32), body_references=upload(references, wp.vec3),
        body_nodes=upload(body_nodes, wp.int32), body_node_offsets=upload(body_offsets, wp.int32),
        body_translate=upload([int(b["translate"]) for b in bodies], wp.int32),
        body_rotate=upload([int(b["rotate"]) for b in bodies], wp.int32),
        body_second_moment=upload([b["I_hat"] for b in bodies], wp.mat33, (-1, 3, 3)),
        rigid_pin_bodies=upload([pin["body"] for pin in pins], wp.int32),
        rigid_pin_references=upload([pin["local_position"] for pin in pins], wp.vec3, (-1, 3)),
        rigid_pin_targets=upload(payload.get("rigid_pin_targets", [pin["initial_target"] for pin in pins]), wp.vec3, (-1, 3)),
        collider_records=payload.get("colliders", []), collider_data=wp.array(collider_host, dtype=wp.float32, device=model.device),
        collider_host=collider_host)
    model.ipc = ipc
    return model


def update_boundary(model, payload):
    """Upload authored inputs without touching evolving simulation state."""
    if payload.get("schema") != SCHEMA or payload.get("kind") not in ("boundary", "scene"):
        raise ValueError("Invalid native boundary message")
    targets = _array(payload["pin_targets"], 3, np.float32, "pin targets")
    rigid = _array(payload.get("rigid_pin_targets", []), 3, np.float32, "rigid pin targets")
    if targets.shape != (model.particle_count, 3) or rigid.shape != (model.ipc.rigid_pin_targets.size, 3):
        raise ValueError("Boundary input dimensions changed")
    _finite(targets, "pin targets"); _finite(rigid, "rigid pin targets")
    model.ipc.pin_targets.assign(targets)
    model.ipc.rigid_pin_targets.assign(rigid)
    colliders = _collider_data(payload.get("colliders", []))
    if colliders.shape != model.ipc.collider_host.shape:
        raise ValueError("Collider topology changed during a simulation")
    model.ipc.collider_records = payload.get("colliders", [])
    model.ipc.collider_host = colliders
    model.ipc.collider_data.assign(colliders)
    return model.ipc
