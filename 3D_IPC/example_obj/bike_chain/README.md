# Bike-chain source assets

These OBJ meshes and `fixture.json` come from Rigid IPC commit
`23b6ba6fbf8434056444ae106356fd2209136988`, Figure 2's bike-chain scene.
`source_manifest.json` records the upstream file URLs and blob hashes.
The original files are retained at their source resolution.

The fixture contains 34 chain segments: 68 plates (`link.obj`), 34 rollers
(`barring.obj`, the upstream spelling), 34 pins, and two sprockets. Every
component is a separate rigid body. The complete scene has 138 bodies,
47,761 vertices and 95,658 triangles.

`generate_initial_bodies.py --check` verifies the native pose table against
the source JSON; `--write` regenerates it. Transforms preserve the authored
model origins, scale 0.01, and XYZ Euler angles in degrees (`Rz Ry Rx`).
The source's 20-tooth sprocket drive is -25 degrees/second about world z.
In the source simulation, both sprocket centers and their x/y rotations are fixed.
Our Example 18 currently imports frame 0 only; mechanism motion is not enabled.

The upstream license notices are copied as `LICENSE-chain.txt`,
`LICENSE-sprocket.txt`, and `LICENSE-rigid-ipc.txt`. Their terms apply to
the corresponding source assets and code.
