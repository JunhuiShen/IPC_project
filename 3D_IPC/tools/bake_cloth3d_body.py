#!/usr/bin/env python3
"""Bake a CLOTH3D sequence's SMPL pose stream into per-frame body OBJs.

CLOTH3D ships per-frame SMPL pose parameters (info.mat) but not pre-evaluated
body meshes. This script evaluates the SMPL female/male model per frame and
writes body_NNNN.obj files alongside the dress, in the same coordinate
system (z-up, with the sequence's zrot applied) so frame 0 lines up with
Dress.obj.

Usage:
    python tools/bake_cloth3d_body.py \\
        --sequence ~/cloth3d_one/train/01001 \\
        --smpl-models-dir ~/Downloads/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models \\
        --outdir ~/cloth3d_one/train/01001/body_obj
"""

import argparse
import os
import pickle
import struct
import sys
import types
from pathlib import Path

import numpy as np
from scipy.io import loadmat


# --- PC16 reader (CLOTH3D's per-frame garment positions) --------------------
def read_pc16_frame(path, frame):
    """Read a single frame of a .pc16 file. Returns (n_pts, 3) float32 array."""
    with open(path, 'rb') as f:
        f.seek(16)
        n_pts = struct.unpack('<i', f.read(4))[0]
        f.seek(28)
        n_samples = struct.unpack('<i', f.read(4))[0]
        if frame >= n_samples:
            raise IndexError(f'frame {frame} out of range [0,{n_samples})')
        size = n_pts * 3 * 2  # float16
        f.seek(size * frame, 1)
        return np.frombuffer(f.read(size), dtype=np.float16).astype(np.float32).reshape(n_pts, 3)


def read_pc16_meta(path):
    with open(path, 'rb') as f:
        f.seek(16)
        n_pts = struct.unpack('<i', f.read(4))[0]
        f.seek(28)
        n_samples = struct.unpack('<i', f.read(4))[0]
    return n_pts, n_samples


def read_obj_faces(path):
    """Read face lines from an OBJ. Supports tris and quads (split to tris)."""
    faces = []
    with open(path) as f:
        for line in f:
            if line.startswith('f '):
                idx = [int(t.split('/')[0]) - 1 for t in line[2:].split()]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) == 4:
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])
    return np.array(faces, dtype=np.int32)


def write_obj(path, verts, faces):
    with open(path, 'w') as f:
        for v in verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for tri in faces:
            f.write(f'f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n')


# --- chumpy shim --------------------------------------------------------------
# MPI's SMPL .pkl files were pickled with `chumpy` autograd arrays. We don't
# need their autograd capability — we just want the underlying numpy data.
# Try real chumpy first; fall back to a minimal shim that intercepts
# __setstate__ to extract the underlying numpy array.
def install_chumpy_shim():
    try:
        import chumpy  # noqa: F401
        return
    except Exception:
        pass

    class Ch:
        def __setstate__(self, state):
            if isinstance(state, dict):
                self._x = state.get('x', state.get('_x', None))
            else:
                self._x = state

        @property
        def r(self):
            return np.asarray(self._x)

        def __array__(self, dtype=None):
            return np.asarray(self._x, dtype=dtype) if dtype else np.asarray(self._x)

    chumpy = types.ModuleType('chumpy')
    chumpy_ch = types.ModuleType('chumpy.ch')
    chumpy.Ch = Ch
    chumpy_ch.Ch = Ch
    chumpy.array = np.array
    sys.modules['chumpy'] = chumpy
    sys.modules['chumpy.ch'] = chumpy_ch


def to_numpy(x):
    if hasattr(x, 'r'):
        return np.asarray(x.r)
    return np.asarray(x)


# --- SMPL forward kinematics --------------------------------------------------
class SMPLModel:
    """Pure-numpy SMPL forward kinematics. Adapted from DeePSD/smpl_np.py."""

    def __init__(self, model_path):
        install_chumpy_shim()
        with open(model_path, 'rb') as f:
            params = pickle.load(f, encoding='latin1')
        self.J_regressor = to_numpy(params['J_regressor'].todense()
                                    if hasattr(params['J_regressor'], 'todense')
                                    else params['J_regressor'])
        self.weights     = to_numpy(params['weights'])
        self.posedirs    = to_numpy(params['posedirs'])
        self.v_template  = to_numpy(params['v_template'])
        # SMPL v1.1.0 ships 300 shape PCs; CLOTH3D only stores the first 10
        # (PCs 1-10 are identical to v1.0.0). Truncate for compatibility.
        self.shapedirs   = to_numpy(params['shapedirs'])[..., :10]
        self.faces       = np.asarray(params['f'], np.int32)
        self.kintree     = np.asarray(params['kintree_table'])

        id_to_col = {self.kintree[1, i]: i for i in range(self.kintree.shape[1])}
        self.parent = {i: id_to_col[self.kintree[0, i]]
                       for i in range(1, self.kintree.shape[1])}

    def __call__(self, pose, beta, trans, no_trans=False):
        """pose: (24,3) axis-angle | beta: (10,) | trans: (3,) -> verts (N,3).
        Pelvis is re-centered to origin in y-up canonical before LBS, so that
        with trans=t the body's pelvis ends up at t (CLOTH3D's convention).
        With no_trans=True, returns body with pelvis at origin (no trans added).
        """
        # 1) shape blend
        v_shaped = self.shapedirs.dot(beta) + self.v_template
        # Re-center: subtract J[0] (canonical pelvis location) so the body's
        # pelvis sits at origin in y-up canonical. Without this, smpl_np-style
        # forward kinematics leaves the pelvis at J[0]_canonical and adding
        # trans would place it at trans+J[0] — a ~0.21m vertical offset that
        # mismatches CLOTH3D's .pc16 frame (which assumes pelvis-at-trans).
        J_canonical_pelvis = self.J_regressor.dot(v_shaped)[0]
        v_shaped = v_shaped - J_canonical_pelvis
        J = self.J_regressor.dot(v_shaped)
        # 2) rotation matrices for every joint
        R = self._rodrigues(pose.reshape(-1, 1, 3))
        # 3) global transforms via kintree
        G = np.empty((self.kintree.shape[1], 4, 4))
        G[0] = self._with_h(np.hstack((R[0], J[0].reshape(3, 1))))
        for i in range(1, self.kintree.shape[1]):
            local = self._with_h(np.hstack((R[i],
                                            (J[i] - J[self.parent[i]]).reshape(3, 1))))
            G[i] = G[self.parent[i]].dot(local)
        # subtract rest-pose joint contribution
        zero4 = np.hstack((J, np.zeros((24, 1)))).reshape(24, 4, 1)
        G_rest = np.zeros((24, 4, 4))
        G_rest[..., :, 3:] = np.matmul(G, zero4)
        G = G - G_rest
        # 4) pose blend shapes
        I_cube = np.broadcast_to(np.eye(3), (R.shape[0] - 1, 3, 3))
        lrotmin = (R[1:] - I_cube).ravel()
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        # 5) skinning
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        v_h = np.hstack((v_posed, np.ones((v_posed.shape[0], 1))))
        verts = np.matmul(T, v_h.reshape(-1, 4, 1)).reshape(-1, 4)[:, :3]
        if no_trans:
            return verts
        return verts + trans.reshape(1, 3)

    @staticmethod
    def _rodrigues(r):
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        theta = np.maximum(theta, np.finfo(np.float64).eps)
        r_hat = r / theta
        cos = np.cos(theta)
        z = np.zeros(theta.shape[0])
        m = np.dstack([z, -r_hat[:, 0, 2], r_hat[:, 0, 1],
                       r_hat[:, 0, 2], z, -r_hat[:, 0, 0],
                       -r_hat[:, 0, 1], r_hat[:, 0, 0], z]).reshape(-1, 3, 3)
        I = np.broadcast_to(np.eye(3), (theta.shape[0], 3, 3))
        outer = np.matmul(np.transpose(r_hat, (0, 2, 1)), r_hat)
        return cos * I + (1 - cos) * outer + np.sin(theta) * m

    @staticmethod
    def _with_h(x):
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))


# --- coordinate-frame helper --------------------------------------------------
def zrot_matrix(zrot):
    c, s = np.cos(zrot), np.sin(zrot)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


# --- vertex fusion (body micro-edge cleanup) ----------------------------------
# SMPL has ~24 sub-millimeter edges concentrated at face features (lips, eyelids).
# IPC's d_hat constraint requires d_hat <= 0.5 * min_edge_length, so without
# cleanup the barrier is forced into the sub-millimeter regime — too stiff for
# practical sim. Compute a vertex remap from the rest-pose body once, then
# apply it to every frame so per-frame topology stays stable.
def build_fuse_remap(verts, faces, distance):
    """Returns (new_verts_count, remap, new_faces) where remap[i] is the new
    index for old vertex i. Vertices closer than `distance` are fused; among
    a fused cluster the lowest-index vertex is the canonical representative."""
    n = verts.shape[0]
    remap = np.arange(n)
    # Bucket-based neighbor finding to avoid an N^2 pass.
    cell = (verts / distance).astype(np.int64)
    buckets = {}
    for i in range(n):
        key = tuple(cell[i])
        buckets.setdefault(key, []).append(i)
    # For each vertex, look in its 3x3x3 cell neighborhood for closer vertices.
    d2 = distance * distance
    for i in range(n):
        if remap[i] != i:
            continue
        ci = tuple(cell[i])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (ci[0] + dx, ci[1] + dy, ci[2] + dz)
                    for j in buckets.get(key, ()):
                        if j <= i:
                            continue
                        if remap[j] != j:
                            continue
                        if np.sum((verts[i] - verts[j]) ** 2) <= d2:
                            remap[j] = i
    # Compress to dense indices.
    keep = np.where(remap == np.arange(n))[0]
    compress = -np.ones(n, dtype=np.int64)
    for new_idx, old_idx in enumerate(keep):
        compress[old_idx] = new_idx
    final = compress[remap]
    new_faces = final[faces]
    # Drop degenerate triangles (after fusion two corners may now coincide).
    keep_face = ~((new_faces[:, 0] == new_faces[:, 1])
                  | (new_faces[:, 1] == new_faces[:, 2])
                  | (new_faces[:, 0] == new_faces[:, 2]))
    new_faces = new_faces[keep_face]
    return len(keep), final, new_faces, keep


# --- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sequence', required=True, type=Path,
                    help='Path to the CLOTH3D sequence folder (contains info.mat)')
    ap.add_argument('--smpl-models-dir', required=True, type=Path,
                    help='Directory containing basicmodel_{f,m,neutral}_*.pkl')
    ap.add_argument('--outdir', required=True, type=Path,
                    help='Output directory for body_NNNN.obj files')
    ap.add_argument('--every', type=int, default=1,
                    help='Bake every Nth frame (default: 1 = every frame)')
    ap.add_argument('--garment', default='Dress',
                    help='Garment name to bake from .pc16 (default: Dress). '
                         'Set empty string to skip garment baking.')
    ap.add_argument('--keep-zup', action='store_true',
                    help='Keep CLOTH3D z-up convention. Default rotates to '
                         'y-up to match the IPC sim (gravity = (0,-9.81,0)).')
    ap.add_argument('--fuse-distance', type=float, default=0.001,
                    help='Fuse body vertices closer than this many meters in '
                         'the rest pose, dropping resulting degenerate '
                         'triangles. Default 1mm cleans SMPL face micro-edges '
                         'so IPC d_hat can be set ~5mm. Set 0 to disable.')
    args = ap.parse_args()

    info = loadmat(str(args.sequence / 'info.mat'))
    poses = info['poses']            # (72, F)
    beta  = info['shape'][0]         # (10,)
    trans = info['trans']            # (3, F)
    gender = int(info['gender'][0, 0])
    zrot   = float(info['zrot'][0, 0])
    F = poses.shape[1]

    pkls = list(args.smpl_models_dir.glob('basicmodel_*lbs_10*.pkl'))
    pkls += list(args.smpl_models_dir.glob('basicModel_*lbs_10*.pkl'))
    pick = {0: 'basicmodel_f', 1: 'basicmodel_m'}
    chosen = next((p for p in pkls if p.name.startswith(pick[gender])
                                    or p.name.startswith(pick[gender].replace('basicmodel', 'basicModel'))),
                  None)
    if chosen is None:
        sys.exit(f'No SMPL .pkl matching gender={gender} in {args.smpl_models_dir}')
    print(f'SMPL model:    {chosen.name}  (gender={"female" if gender==0 else "male"})')

    smpl = SMPLModel(str(chosen))
    print(f'template verts: {smpl.v_template.shape[0]}  faces: {smpl.faces.shape[0]}')

    # Build a fusion remap from the rest-pose body so the SMPL face micro-edges
    # are merged consistently across all frames. SMPL's skinning weights keep
    # fused vertices co-located across poses, so a remap from frame-0 is safe
    # for the whole sequence.
    body_faces = smpl.faces
    if args.fuse_distance > 0.0:
        rest_verts = smpl.v_template + smpl.shapedirs.dot(beta)
        fused_n, fuse_map, body_faces, keep_idx = build_fuse_remap(
            rest_verts, smpl.faces, args.fuse_distance)
        dropped_v = smpl.v_template.shape[0] - fused_n
        dropped_f = smpl.faces.shape[0] - body_faces.shape[0]
        print(f'fuse@{args.fuse_distance*1000:.2f}mm: '
              f'{smpl.v_template.shape[0]} -> {fused_n} verts '
              f'(dropped {dropped_v}), '
              f'{smpl.faces.shape[0]} -> {body_faces.shape[0]} faces '
              f'(dropped {dropped_f})')
    else:
        fuse_map = None
        keep_idx = None

    args.outdir.mkdir(parents=True, exist_ok=True)
    R_z = zrot_matrix(zrot)
    # CLOTH3D bakes the SMPL y-up -> z-up rotation INTO each frame's
    # global_orient (the first 3 axis-angle entries of `poses`). So the SMPL
    # eval's output is already in z-up world. We just apply zrot, then
    # optionally swap to y-up for the IPC sim (gravity = (0,-9.81,0)).
    if args.keep_zup:
        zup_to_yup = np.eye(3)
    else:
        zup_to_yup = np.array([[1.0,  0.0, 0.0],
                               [0.0,  0.0, 1.0],
                               [0.0, -1.0, 0.0]], dtype=np.float64)
    final = zup_to_yup @ R_z

    # Optional garment baking from .pc16 (per-frame simulated drape).
    garment_pc16 = None
    garment_faces = None
    if args.garment:
        pc16_path = args.sequence / f'{args.garment}.pc16'
        obj_path  = args.sequence / f'{args.garment}.obj'
        if pc16_path.exists() and obj_path.exists():
            garment_pc16 = pc16_path
            garment_faces = read_obj_faces(obj_path)
            n_pts, n_samples = read_pc16_meta(pc16_path)
            print(f'garment .pc16: {pc16_path.name}  pts={n_pts}  frames={n_samples}')
        else:
            print(f'warn: no garment data for "{args.garment}" in {args.sequence}')

    for fi in range(0, F, args.every):
        # Body: SMPL eval already returns z-up world (global_orient bakes the
        # axis change). Just apply zrot then optional y-up swap.
        pose_t  = poses[:, fi].reshape(24, 3)
        trans_t = trans[:, fi]
        verts = smpl(pose_t, beta, trans_t) @ final.T
        if keep_idx is not None:
            verts = verts[keep_idx]
        write_obj(args.outdir / f'body_{fi:04d}.obj', verts, body_faces)

        # Garment: .pc16 is in pelvis-anchored z-up frame (i.e. before trans).
        # Add trans, then apply same zrot + y-up swap as the body.
        if garment_pc16 is not None:
            g_verts = read_pc16_frame(str(garment_pc16), fi)
            g_verts = (g_verts + trans_t.reshape(1, 3)) @ final.T
            write_obj(args.outdir / f'{args.garment.lower()}_{fi:04d}.obj', g_verts, garment_faces)

        if fi % 50 == 0 or fi == F - 1:
            print(f'  frame {fi:4d}/{F-1}')

    n = (F + args.every - 1) // args.every
    print(f'wrote {n} body OBJs' + (f' + {n} {args.garment.lower()} OBJs' if garment_pc16 is not None else '')
          + f' to {args.outdir}')


if __name__ == '__main__':
    main()
