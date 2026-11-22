# Hyperboloid gear pair (crank-driven) — simulation meshes

The simulator loads the three prebuilt OBJ meshes in `out/`. To regenerate the geometry, run `python3 gen_hyperboloid_gears.py` in this directory with NumPy, SciPy, trimesh, manifold3d, Shapely, mapbox-earcut, and python-fcl installed. Coordinates are in millimeters.

| file | body | notes |
|---|---|---|
| `out/gear1_crank.obj` | gear 1 + shaft + crank arm + grip (front gear) | one watertight manifold; hinge point/axis in `scene.json` |
| `out/gear2.obj` | gear 2 + shaft (back gear) | one watertight manifold; hinge in `scene.json` |
| `out/stand.obj` | filleted base plate + plinth, two tapered leaning pillars with cylindrical bearing housings | static collider, one watertight manifold |
| `out/scene.json` | hinge points/axes (world frame), `world_from_sim`, base height, parameters, interference result | |

**World frame**: Z up (gravity −Z), X left-right, Y depth (camera looks along +Y). Both shafts lie in the vertical planes
Y=0 (gear 1, front) and Y=50 (gear 2, back). Gear 1 rises +30° to the right and is cantilevered from the tall right post with
the crank at its upper end; gear 2 drops −30° to the shorter left post — the video's layout. Base top at Z = −92.9. The script also checks gears + crank against the stand over a full turn (clear; only the 0.7 mm shaft-bore gap remains).
Note: `world_from_sim` has det = −1, i.e. the exported world scene is the mirror image of the sim-frame pair (opposite-hand
teeth). That is an equally valid gear pair; meshes are exported with consistent outward winding.

## Geometry summary (defaults)
* N = 24 teeth, throat (pitch) radius r_p = 25, skew angle β = 30° → **shaft angle 60°**, ratio 1:1, centre distance 50.
* Full hyperboloids: each gear spans z ∈ [−50, 50] around its throat (100 mm long; end radius 38.6 vs throat 25).
  Pitch surface r(z)² = r_p² + (z tanβ)².
* Section at the throat = hub circle (r = 22.2) + 24 trapezoidal teeth (2.0 mm thick at the pitch surface measured normal
  to the ruling, 14° flanks, 2.2 mm addendum, filleted tips). Each z-slice is that section scaled by r(z)/r_p and rotated by
  atan(z tanβ / r_p), so pitch points run on straight rulings and the teeth twist like in the video.
* Gear 2 = gear 1 rotated 180° about the common ruling (through (r_p,0,0), direction (0, sinβ, cosβ)), phased by half a tooth.
  Gear 1 hinge: origin, +z. Gear 2 hinge: (50,0,0), axis (0, 0.866, 0.5).
* Verified at this length: zero boolean overlap through a full tooth pitch of 1:1 counter-rotation; running clearance ≈ 0.4 mm;
  backlash ±1.0° (gear 2 held). Tighten/loosen with `fin_thickness_normal`.

## Why the shaft angle is 60° and not 90°
Off the common ruling the two pitch hyperboloids separate by ≈ w²·cos2β / (r_p cos²β). For β < 45° (shaft angle < 90°) that is
positive and quadratic — the surfaces behave like crossed helical cylinders and the tooth centre-lines of the two gears never
come closer than half the normal pitch, so full-depth teeth work. At exactly 90° (β = 45°) the surfaces are tangent along *both*
generators through the throat and hug each other (4th-order gap); straight teeth of the two gears then cross and collide
unless the teeth are thin fins and the throat region is cut away. Beyond 90° they interpenetrate. Any `skew_angle` ≤ ~40°
works; the script re-checks interference.

## Simulation hints
* Drive with a torque/motor on gear 1's hinge (or push the handle). Expect θ₂ = −θ₁ up to ±1° backlash.
* Contact is between slightly skewed tooth flanks along the common ruling — a moving line/point contact. Friction matters little.
* ~42k triangles per gear (60 loft slices × 24 teeth). Reduce `hub_slices` for a lighter mesh; the check still runs.
