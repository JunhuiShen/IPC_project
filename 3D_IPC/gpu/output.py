"""Native-compatible frame and checkpoint IO for actual Newton State arrays."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np


FORMATS = {"obj": ".obj", "ply": ".ply", "geo": ".geo", "usd": ".usda"}
_FIELDS = (("positions", 3), ("velocities", 3), ("com", 3), ("com_velocity", 3), ("orientation", 4), ("omega", 3))
_COUNT = struct.Struct("=Q")
_MAGIC = b"RBSTATE1"


@contextmanager
def atomic_file(path, mode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, mode, **({"encoding": "utf-8", "newline": "\n"} if "b" not in mode else {})) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, payload):
    with atomic_file(path, "w") as stream:
        json.dump(payload, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")


def snapshot(model, state):
    """Requested download only; simulation remains in Newton's device arrays."""
    positions = state.particle_q.numpy().astype(np.float64)
    velocities = state.particle_qd.numpy().astype(np.float64)
    body_q = state.body_q.numpy() if model.body_count else np.empty((0, 7))
    body_qd = state.body_qd.numpy() if model.body_count else np.empty((0, 6))
    result = {"positions": positions, "velocities": velocities, "com": body_q[:, :3].astype(np.float64),
              "com_velocity": body_qd[:, :3].astype(np.float64),
              "orientation": body_q[:, [6, 3, 4, 5]].astype(np.float64), "omega": body_qd[:, 3:].astype(np.float64)}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("Nonfinite Newton IPC state cannot be exported")
    return result


def restore(model, state, values):
    """Validate a complete native checkpoint before uploading float32 state."""
    prepared = {}
    for name, width in _FIELDS:
        count = model.particle_count if name in ("positions", "velocities") else model.body_count
        data = np.asarray(values[name], dtype=np.float64)
        if data.shape != (count, width) or not np.isfinite(data).all():
            raise ValueError(f"Invalid checkpoint {name}")
        data32 = data.astype(np.float32)
        if not np.isfinite(data32).all():
            raise ValueError(f"Checkpoint {name} is outside the float32 state range")
        prepared[name] = data32
    q = np.empty((0, 4), dtype=np.float32)
    if model.body_count:
        q = prepared["orientation"][:, [1, 2, 3, 0]]
        norms = np.linalg.norm(q, axis=1)
        if np.any(np.abs(norms - 1) > 1e-5):
            raise ValueError("Checkpoint has invalid rigid orientation")
    state.particle_q.assign(prepared["positions"])
    state.particle_qd.assign(prepared["velocities"])
    if model.body_count:
        state.body_q.assign(np.column_stack((prepared["com"], q)))
        state.body_qd.assign(np.column_stack((prepared["com_velocity"], prepared["omega"])))


def write_checkpoint(path, values):
    arrays = {}
    for name, width in _FIELDS:
        data = np.asarray(values[name], dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != width or not np.isfinite(data).all():
            raise ValueError(f"Invalid checkpoint {name}")
        arrays[name] = np.asarray(data, dtype="=f8", order="C")
    if arrays["positions"].shape != arrays["velocities"].shape:
        raise ValueError("Particle checkpoint arrays disagree")
    if len({len(arrays[name]) for name in ("com", "com_velocity", "orientation", "omega")}) != 1:
        raise ValueError("Rigid checkpoint arrays disagree")
    with atomic_file(path, "wb") as stream:
        for i, (name, _) in enumerate(_FIELDS):
            if i == 2:
                stream.write(_MAGIC)
            stream.write(_COUNT.pack(len(arrays[name])))
            stream.write(arrays[name].tobytes())


def read_checkpoint(path, *, particle_count, body_count, legacy_defaults=None):
    data = memoryview(Path(path).read_bytes())
    position = 0
    values = {}
    for i, (name, width) in enumerate(_FIELDS):
        if i == 2:
            if position == len(data):
                if legacy_defaults is None:
                    raise ValueError("Legacy checkpoint needs explicit rigid defaults")
                values.update({key: np.asarray(legacy_defaults[key], dtype=np.float64).copy() for key, _ in _FIELDS[2:]})
                break
            if bytes(data[position:position + 8]) != _MAGIC:
                raise ValueError("Invalid generalized checkpoint tag")
            position += 8
        if position + 8 > len(data):
            raise ValueError("Truncated checkpoint count")
        count = _COUNT.unpack_from(data, position)[0]
        position += 8
        expected = particle_count if i < 2 else body_count
        if count != expected or count > (len(data) - position) // (8 * width):
            raise ValueError(f"Invalid checkpoint {name} count")
        end = position + count * width * 8
        values[name] = np.frombuffer(data[position:end], dtype="=f8").reshape((count, width)).copy()
        if not np.isfinite(values[name]).all():
            raise ValueError(f"Nonfinite checkpoint {name}")
        position = end
    if position != len(data):
        raise ValueError("Unexpected trailing checkpoint bytes")
    return values


def _mesh(positions, triangles):
    points = np.asarray(positions, dtype=np.float64).reshape((-1, 3))
    raw = np.asarray(triangles)
    if raw.size and not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("Mesh connectivity must be integer")
    tris = raw.astype(np.int64).reshape((-1, 3))
    if not np.isfinite(points).all() or (tris.size and (np.any(tris < 0) or np.any(tris >= len(points)))):
        raise ValueError("Invalid output mesh")
    return points, tris


def write_mesh(path, positions, triangles, format):
    points, tris = _mesh(positions, triangles)
    if format not in FORMATS:
        raise ValueError(f"Unsupported output format {format}")
    if format == "ply":
        with atomic_file(path, "wb") as out:
            header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\nproperty double x\nproperty double y\nproperty double z\nelement face {len(tris)}\nproperty list uchar int vertex_indices\nend_header\n"
            out.write(header.encode("ascii")); out.write(points.astype("<f8").tobytes())
            for tri in tris:
                out.write(struct.pack("<Biii", 3, *map(int, tri)))
        return
    with atomic_file(path, "w") as out:
        if format == "obj":
            for p in points:
                out.write("v " + " ".join(format_number(x) for x in p) + "\n")
            for t in tris:
                out.write("f " + " ".join(str(int(v) + 1) for v in t) + "\n")
        elif format == "usd":
            out.write('#usda 1.0\n(\n    upAxis = "Y"\n)\ndef Mesh "mesh" {\n')
            out.write("    point3f[] points = [" + ",".join("(" + ",".join(format_number(x) for x in p) + ")" for p in points) + "]\n")
            out.write("    int[] faceVertexCounts = [" + ",".join("3" for _ in tris) + "]\n")
            out.write("    int[] faceVertexIndices = [" + ",".join(str(int(v)) for v in tris.reshape(-1)) + "]\n")
            out.write('    uniform token subdivisionScheme = "none"\n}\n')
        else:
            attribute = [["scope", "public", "type", "numeric", "name", "P", "options", {}],
                         ["size", 3, "storage", "fpreal64", "values", ["size", 3, "storage", "fpreal64", "tuples", points.tolist()]]]
            geometry = ["fileversion", "18.5.408", "hasindex", False, "pointcount", len(points),
                "vertexcount", int(tris.size), "primitivecount", len(tris), "info", {},
                "topology", ["pointref", ["indices", tris.reshape(-1).tolist()]],
                "attributes", ["pointattributes", [attribute]],
                "primitives", [[["type", "Polygon_run"], ["startvertex", 0,
                                "nprimitives", len(tris), "nvertices_rle", [3, len(tris)]]]]]
            json.dump(geometry, out, allow_nan=False, separators=(",", ":"))
            out.write("\n")


def format_number(value):
    return format(float(value), ".17g")
