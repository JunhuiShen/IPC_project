# Wrecking-ball meshes

`link.obj` and `ball.obj` are copied without geometric modification from the
Rigid IPC reference implementation's `meshes/wrecking-ball` directory:

- https://github.com/ipc-sim/rigid-ipc
- Copyright (c) 2021 Zachary Ferguson and the IPC Simulation organization
- Licensed under the MIT License; see [`LICENSE`](LICENSE) in this directory

They are used by Example 13 for the rigid IPC paper's Figure 8 wrecking-ball
fixture. Example 16 uses a coarser derivative for its triangulated rigid chain-net
adaptation and adds procedurally generated and gear-mesh payloads.

## Intermediate link for Example 17

`link_128.obj` is a closed, outward-oriented derivative with 128 vertices and
256 triangles (16 longitudinal rings × 8 tube samples). It retains the source
attribution and MIT license above. The original `link.obj` is unchanged.
`generate_link_128.py --check` verifies deterministic regeneration;
`--report` prints dimensions, volume, cross-section and ring coordinates.
The generator prints OBJ data to stdout by default and does not write files.

Outer bounds match the original mesh, and the axis-aligned opening spans remain
0.6 × 1.1 before scene scaling. Coarser chords alter the local hole outline;
the new mesh is not simply a subset of the original solid. The enclosed volume
is approximately 12.1% lower, so fixed density gives lower link mass, with
inertia recomputed by the scene builder. Different contact normals and dynamic
results are expected. Example 17 uses this intermediate link. Example 13 always uses the original
220-vertex mesh.

## Compact Example 16 meshes

`link_32.obj` is a 32-vertex, 64-triangle derivative of the same MIT-licensed
link. It preserves the outer dimensions and axis opening spans, using eight
longitudinal stations and a diamond-shaped tube cross section. Fewer cap
segments alter the local opening and enclosed volume. Example 16 selects this mesh directly, with no resolution flag. Other
examples retain their existing link assets.
`generate_compact_meshes.py --check` verifies deterministic regeneration of
this link and `../gear_z18_180.obj`; `--write` regenerates both assets.
The compact gear uses 180 vertices and 360 triangles, keeps all 18 teeth and
the bore, and samples the root and tip outline of the existing procedural
`gear_z18_coarse.obj` asset. Its thickness and outer bounds are preserved.
