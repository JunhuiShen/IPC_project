#!/usr/bin/env python3
"""Decimate a Wavefront OBJ via quadric edge-collapse and write the result.

Used to coarsen the xyzrgb dragon (125k verts) down to a tractable size for
example 4 so the original example-1 parameter set converges (the convergence
criterion dt^2 * E < rho * h * l^2 scales with the squared min edge length;
coarsening lets us raise E or drop substeps).

Usage:
    python tools/decimate_obj.py <input.obj> <output.obj> --target-faces 24000
"""

import argparse
import sys

import trimesh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument(
        "--target-faces",
        type=int,
        default=24000,
        help="Target triangle count (vertices come out at roughly half this).",
    )
    args = ap.parse_args()

    mesh = trimesh.load(args.input, process=False)
    print(f"loaded:    {len(mesh.vertices):>7} verts  {len(mesh.faces):>7} faces  ({args.input})")

    out = mesh.simplify_quadric_decimation(face_count=args.target_faces)
    print(f"decimated: {len(out.vertices):>7} verts  {len(out.faces):>7} faces")

    # Write a minimal OBJ (v + f records only). Matches what load_obj_mesh in
    # make_shape.cpp expects -- it ignores everything else anyway.
    with open(args.output, "w") as f:
        for v in out.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in out.faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")
    print(f"wrote:     {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
