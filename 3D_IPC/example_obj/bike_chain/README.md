# Bike-chain source assets

These OBJ meshes and `fixture.json` come from Rigid IPC commit
`23b6ba6fbf8434056444ae106356fd2209136988`, Figure 2's bike-chain scene.
`source_manifest.json` records the upstream file URLs and blob hashes.
Example 18 loads the original meshes and authored dimensions without geometry
simplification: **138 independent rigid bodies, 47,761 vertices and 95,658
triangles**.

| Mesh | Instances | Vertices each | Triangles each |
|---|---:|---:|---:|
| `link.obj` | 68 | 438 | 880 |
| `barring.obj` (upstream spelling) | 34 | 180 | 360 |
| `pin.obj` | 34 | 302 | 600 |
| `sprocket-20teeth.obj` | 1 | 1,056 | 2,112 |
| `sprocket-8teeth.obj` | 1 | 533 | 1,066 |

`generate_initial_bodies.py --check` verifies the native pose table against
the source JSON; `--write` regenerates it. Transforms preserve the authored
model origins, scale 0.01, and XYZ Euler angles in degrees (`Rz Ry Rx`).
Closed-volume mass and inertia are computed from the original meshes.
The source's 20-tooth sprocket drive is −25 degrees/second about world z.
In the source simulation, both sprocket centers and their x/y rotations are fixed.

Our Example 18 uses four moving soft-pin motor points on the large sprocket
and two stationary shaft points on each sprocket.
`--bike_chain_motor_radius` sets the motor points' radius in metres (default
0.03); changing this lever arm changes the motor torque response at the same
`kpin`.
`--bike_chain_shaft_span` sets the axial distance from each gear center to
either stationary support point (default 0.05 m); changing this distance changes
tilt stiffness without changing the meshes or fixing their rotation about z.
Both centers stay fixed. The soft supports permit small tilt and tracking
errors; they approximate the source's axis constraints. The preset uses zero
gravity. The existing solver updates and CCD are retained. See the main README
for the motion command.

Initial motion follows `initial_bodies.txt`: the chain components and passive
sprocket start at rest, while the large sprocket starts at −25 degrees/s about
world z. Every body has zero initial COM velocity. `--bike_chain_drive_speed`
sets the large sprocket's initial angular velocity and motor target speed;
the chain and passive sprocket acquire motion through contact. Keep motor speed
and FPS unchanged when restarting an existing run so its motor target phase
stays consistent.

The upstream license notices are copied as `LICENSE-chain.txt`,
`LICENSE-sprocket.txt`, and `LICENSE-rigid-ipc.txt`. Their terms apply to
the corresponding source assets and code.
