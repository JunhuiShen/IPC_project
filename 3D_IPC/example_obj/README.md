# Example mesh assets

This directory contains OBJ meshes and supporting geometry files used by the
simulation examples. See the [main README](../README.md#scene-assets) for the
assets required by each scene and how to select their paths.

Example 21 uses `hyperboloid_gears/out/gear1_crank.obj`, `gear2.obj`, and
`stand.obj` as three rigid bodies. One moving spring target inside the handle
grip drives the fused handle/crank/shaft/first gear; tooth contact drives the
second gear. Two stationary shaft pins support each gear, and the stand remains
fixed. Both gears start at rest. The updated `hyperboloid_gears ` folder (with a
trailing space) is also accepted without renaming the source assets.
The loader preserves the assembly and converts its millimeter, Z-up coordinates
to meters, Y-up with `(x, y, z) -> 0.001 * (x, z, -y)`; its fixed hinge geometry
comes from the bundled `out/scene.json`. Pass `--datadir example_obj` or
`--datadir example_obj/hyperboloid_gears`. See the
[gear asset notes](hyperboloid_gears/README.md) for geometry details.

The meshes in `bunny_coarse/` and `armadillo_coarse/` are derived from the
Stanford Bunny and Stanford Armadillo models. We thank the Stanford University
Computer Graphics Laboratory for making these models available through the
[Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/).
This credit also applies to their simplified meshes and supporting tetrahedral
geometry files. See the source repository for its acknowledgment and use terms.

The meshes in `spot/` named `spot_*` and their supporting tetrahedral geometry
files are derived from Spot by Keenan Crane. We thank him for making the model
available through his
[3D Model Repository](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/),
where Spot is released under CC0 1.0. The bundled simplified versions retain
this source credit.

Selected OBJ assets, including those in `wrecking_ball/`, `bike_chain/`,
`anchor/`, and `arch/`, are sourced from or adapted from
[Rigid IPC](https://github.com/ipc-sim/rigid-ipc). We thank the Rigid IPC authors
and the original mesh creators for making these assets available.

See the license files in the asset subdirectories for applicable terms and
additional attribution. The [bike-chain source manifest](bike_chain/source_manifest.json)
records its upstream files and revision.
