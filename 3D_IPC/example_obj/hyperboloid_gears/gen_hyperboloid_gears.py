#!/usr/bin/env python3
"""
Hyperboloid (skew-axis, straight-tooth) gear pair generator  -  for rigid-body / IPC simulation.

Geometry
--------
* Pitch surface of each gear is a hyperboloid of one sheet with throat radius r_p at z=0:
  r(z)^2 = r_p^2 + (z tan beta)^2 ,  beta = skew angle of the rulings.  The gear spans
  z in [z_min, z_max] (throat_offset, face_width); z_min < 0 < z_max gives the full "waisted" shape.
* The whole gear section (hub circle + N trapezoidal teeth) is defined in the throat plane z=0 and
  lofted along z with a similarity map: scale k(z) = r(z)/r_p, rotation psi(z) = atan(z tan(beta)/r_p).
  Every point on the pitch circle then travels along a straight ruling of the pitch hyperboloid,
  teeth stay radial everywhere, and tooth thickness/gaps scale with the local radius.
* Gear 2 is the image of gear 1 under a 180 deg rotation about the common ruling
  (through (r_p,0,0) with direction (0, sin b, cos b)).  That rotation swaps the two axes
  (shaft angle Sigma = 2 beta, centre distance 2 r_p) so the pitch surfaces touch along that
  line.  Gear 2 is additionally phased by half a tooth pitch so fins interleave.
* Shaft angle matters: off the common ruling the two pitch surfaces separate by
  ~ w^2 cos(2 beta) / (r_p cos^2 beta)  (w = distance from the ruling).  For beta < 45 deg
  (shaft angle < 90 deg) they separate quadratically like crossed helical gears and the tooth
  centre-lines never come closer than half the normal pitch -> full-depth teeth and full
  hyperboloids are fine.  At exactly 90 deg (beta = 45) the surfaces are also tangent along the
  second generator through the throat and hug each other (4th-order gap), so straight teeth of
  the two gears cross and collide unless they are thin fins and the throat region is cut away;
  beyond 90 deg they interpenetrate.  The script checks interference by boolean intersection over
  one tooth pitch of rotation and reports the worst case (should be 0).

Outputs (mm), in ./out :
  gear1_crank.obj        gear 1 + shaft + crank arm + handle  (one watertight body)
  gear2.obj              gear 2 + shaft (one watertight body, in world position)
  gear2_local_frame.obj  gear 2 in its own frame (axis = +z, throat at z=0); world = T2 * local
  stand_bearings.obj     four static bearing blocks on the shafts (optional)
  scene.glb              everything, for a quick look
  scene.json             hinge axes, transforms, parameters, interference result
  parts/                 the un-fused pieces (hub, fins, shaft, crank, handle)
"""
import json, os
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix
from scipy.spatial import cKDTree

# ----------------------------------------------------------------- parameters
P = dict(
    teeth=24,                 # N
    pitch_radius=25.0,        # r_p  (throat radius of the pitch hyperboloid, at z=0)
    skew_angle=30.0,          # beta [deg]; shaft angle Sigma = 2*beta (keep < 45 deg, see notes), 1:1 ratio
    throat_offset=-50.0,      # z_min: gear spans z in [z_min, L]; negative = full hyperboloid, waist in the middle
    face_width=50.0,          # L = z_max
    addendum=2.2,             # tooth height above the pitch surface
    dedendum=2.8,             # radial offset pitch surface -> hub surface (> addendum + clearance)
    fin_thickness_normal=2.0, # tooth thickness at the pitch surface, measured normal to the ruling
    flank_angle=14.0,         # deg, trapezoidal flanks (thicker at root, thinner at tip)
    tip_fillet=0.5,           # mm
    hub_slices=60,            # loft resolution of the hub along z
    tip_segments=8,           # roundness of the fin tip
    shaft_radius=5.0,
    shaft_extra=58.0,         # shaft length past the supported end of the gear (through the post)
    bearing_inset=20.0,       # bearing (post) centre sits this far inside the shaft end (> boss half-length 0.9*post_t)
    shaft_free=10.0,          # shaft stub past the free end
    crank_len=48.0,           # crank arm length (radial)
    crank_w=9.0,              # crank arm section
    handle_len=45.0,          # grip length (parallel to shaft)
    handle_r=6.0,
    post_w=26.0,              # stand post width (x) and depth (y)
    post_t=16.0,
    base=(250.0, 130.0, 12.0),# base plate size (x, y, z)
    check_phases=9,           # interference check: rotation samples over one tooth pitch
)
for _k, _v in json.loads(os.environ.get("GEAR_OVERRIDES", "{}")).items():
    P[_k] = _v
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_SUFFIX = os.environ.get("GEAR_OUT", "out")
OUT = os.path.join(HERE, OUT_SUFFIX)
os.makedirs(os.path.join(OUT, "parts"), exist_ok=True)

N, rp, L, Z0 = P["teeth"], P["pitch_radius"], P["face_width"], P["throat_offset"]
beta = np.radians(P["skew_angle"]); tanb = np.tan(beta)
ra = rp + P["addendum"]           # fin tip radius at the throat
rh = rp - P["dedendum"]           # hub radius at the throat
w  = P["fin_thickness_normal"] / np.cos(beta)   # fin width in the throat plane (along e_t)
pitch_n = 2 * np.pi * rp / N * np.cos(beta)     # normal pitch between rulings at the throat

# ------------------------------------------------------------------ helpers
def close_loft(V, M, nz):
    """Triangulate a loft of nz rings of M vertices (CCW from +z) into a watertight mesh."""
    F = []
    for i in range(nz - 1):
        a, b = i * M, (i + 1) * M
        for j in range(M):
            j2 = (j + 1) % M
            F += [[a + j, a + j2, b + j2], [a + j, b + j2, b + j]]
    V = [v for v in V]
    for ring, sgn in ((0, -1), (nz - 1, +1)):
        base = ring * M
        ci = len(V); V.append(np.mean(V[base:base + M], axis=0))
        for j in range(M):
            j2 = (j + 1) % M
            F.append([ci, base + j, base + j2] if sgn > 0 else [ci, base + j2, base + j])
    m = trimesh.Trimesh(np.array(V), np.array(F), process=True)
    m.fix_normals()
    assert m.is_watertight
    return m

def hub_hyperboloid(ded, z0, z1, nz, sections=128):
    """Hub = pitch hyperboloid offset inward by a constant radial 'ded' (so mating fin tips,
    whose height above the pitch surface shrinks with z, always clear it)."""
    th = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    zs = np.linspace(z0, z1, nz)
    V = np.vstack([np.column_stack([(np.hypot(rp, z * tanb) - ded) * np.cos(th),
                                    (np.hypot(rp, z * tanb) - ded) * np.sin(th),
                                    np.full(sections, z)]) for z in zs])
    return close_loft(V, sections, nz)

def fin_section():
    """2-D tooth outline in local (radial u, tangential v) coords, CCW.
    Trapezoid: half-width w/2 at the pitch radius, flanks at flank_angle, filleted tip corners."""
    ta = np.tan(np.radians(P["flank_angle"])) / np.cos(beta)   # flank slope in the throat plane
    hw = lambda u: w / 2 - (u - rp) * ta
    u0, f = rh, P["tip_fillet"]
    n = max(2, P["tip_segments"] // 2)
    pts = [(u0, -hw(u0)), (ra - f, -hw(ra - f))]
    cu, cv = ra - f, -(hw(ra) - f)                                # right corner centre
    pts += [(cu + f * np.cos(t), cv + f * np.sin(t)) for t in np.linspace(-np.pi / 2, 0, n + 1)[1:]]
    cu, cv = ra - f, (hw(ra) - f)                                 # left corner centre
    pts += [(cu + f * np.cos(t), cv + f * np.sin(t)) for t in np.linspace(0, np.pi / 2, n + 1)[1:]]
    pts += [(ra - f, hw(ra - f)), (u0, hw(u0))]
    return np.array(pts)

def gear_polygon():
    """Closed CCW outline of the gear section at the throat plane (hub circle + N teeth), as (x,y)."""
    sec = fin_section()                       # (u,v) local, starts/ends on the hub radius
    pitch = 2 * np.pi / N
    pts = []
    for k in range(N):
        thk = k * pitch
        er = np.array([np.cos(thk), np.sin(thk)]); et = np.array([-np.sin(thk), np.cos(thk)])
        pts += [u * er + v * et for (u, v) in sec]
        # root arc to the next tooth
        a0 = thk + np.arctan2(sec[-1][1], sec[-1][0]); a1 = thk + pitch + np.arctan2(sec[0][1], sec[0][0])
        for a in np.linspace(a0, a1, 4)[1:-1]:
            pts.append(np.array([rh * np.cos(a), rh * np.sin(a)]))
    return np.array(pts)

def gear_loft(poly, zs):
    """Hyperboloid loft: slice at z = throat section scaled by k(z)=r(z)/r_p and rotated by
    psi(z)=atan(z tan(beta)/r_p).  Pitch-circle points then move on straight rulings."""
    V = []
    for z in zs:
        s_ = z * tanb
        k = np.hypot(rp, s_) / rp; psi = np.arctan2(s_, rp)
        R = k * np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
        xy = poly @ R.T
        V.append(np.column_stack([xy, np.full(len(poly), z)]))
    return close_loft(np.vstack(V), len(poly), len(zs))

def cylinder(r, z0, z1, sections=48):
    c = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=sections)
    c.apply_translation([0, 0, (z0 + z1) / 2]); return c

def box(size, center):
    b = trimesh.creation.box(extents=size); b.apply_translation(center); return b

def union(parts):
    u = trimesh.boolean.union(parts, engine="manifold")
    u.merge_vertices(); u.fix_normals()
    assert u.is_watertight, "union not watertight"
    return u

def build_gear():
    poly = gear_polygon()
    zs = np.linspace(Z0, L, P["hub_slices"] + 1)
    g = gear_loft(poly, zs)
    hub = hub_hyperboloid(P["dedendum"], Z0, L, P["hub_slices"] + 1)
    return g, hub, None

# ------------------------------------------------------------------ frames
# Simulation ("sim") frame: gear 1 axis = +z through the origin, throat at z=0; gear 2 obtained by a
# 180 deg rotation about the common ruling.  Display/world frame: Z up, X left-right, Y depth; the two
# shafts lie in vertical planes Y=0 (gear 1, front) and Y=2 r_p (gear 2), tilted +/- beta from horizontal.
d_common = np.array([0.0, np.sin(beta), np.cos(beta)])
T2 = rotation_matrix(np.pi, d_common, point=[rp, 0, 0]) @ rotation_matrix(np.pi / N, [0, 0, 1])
b_h = np.array([0.0, np.sin(beta), np.cos(beta)])          # bisector of the two axes -> world X
u_up = np.array([0.0, -np.cos(beta), np.sin(beta)])        # world Z (up): gear 1 rises to the right (crank end high)
W = np.eye(4); W[:3, :3] = np.vstack([b_h, [1.0, 0.0, 0.0], u_up])   # rows: world X, Y, Z in sim coords. det=-1: the
# world scene is the mirror image of the sim-frame pair (opposite-hand teeth), equally valid; meshes are re-wound.

# ------------------------------------------------------------------ gear 1 (+ shaft + crank), in sim frame
gear, hub, _ = build_gear()
zs_end = L + P["shaft_extra"]                       # supported (crank) end of gear 1 at +z
shaft1 = cylinder(P["shaft_radius"], Z0 - P["shaft_free"], zs_end + P["crank_w"])
arm_z = zs_end + P["crank_w"] / 2
arm = box([P["crank_len"] + P["crank_w"], P["crank_w"], P["crank_w"]], [P["crank_len"] / 2, 0, arm_z])
arm = trimesh.boolean.union([arm, cylinder(P["crank_w"] / 2 + 0.5, zs_end, zs_end + P["crank_w"], 32).apply_translation([P["crank_len"], 0, 0])], engine="manifold")
# tapered grip: capsule-ish (revolved profile)
def grip(r0, r1, length, segs=40):
    prof_r = np.array([r0 * 0.6, r0, r0 * 0.95, r1 * 0.9, r1, r1 * 0.7])
    prof_z = np.array([0.0, 0.08, 0.35, 0.7, 0.9, 1.0]) * length
    th = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    V = np.vstack([np.column_stack([r * np.cos(th), r * np.sin(th), np.full(segs, z)]) for r, z in zip(prof_r, prof_z)])
    return close_loft(V, segs, len(prof_r))
handle = grip(P["handle_r"] * 0.8, P["handle_r"], P["handle_len"])
handle.apply_translation([P["crank_len"], 0, arm_z])
gear1 = union([gear, shaft1, arm, handle])

# ------------------------------------------------------------------ gear 2 (+ shaft), supported at its -z end
shaft2 = cylinder(P["shaft_radius"], Z0 - P["shaft_extra"], L + P["shaft_free"])
gear2_local = union([gear.copy(), shaft2])
gear2 = gear2_local.copy(); gear2.apply_transform(T2)
axis2_origin = T2[:3, 3].copy()
axis2_dir = T2[:3, :3] @ np.array([0, 0, 1.0])

print(f"N={N} r_p={rp} tip={ra} hub={rh}  normal pitch={pitch_n:.2f} mm  tooth={P['fin_thickness_normal']} mm  z in [{Z0},{L}]")
print(f"gear1: V={len(gear1.vertices)} F={len(gear1.faces)} watertight={gear1.is_watertight}")
print(f"gear2: V={len(gear2.vertices)} F={len(gear2.faces)} watertight={gear2.is_watertight}")
print("shaft angle", np.degrees(np.arccos(np.clip(axis2_dir[2], -1, 1))).round(2), "deg")

# ------------------------------------------------------------------ interference check (sim frame)
def pose(phi):
    g1 = gear1.copy(); g1.apply_transform(rotation_matrix(phi, [0, 0, 1]))
    g2 = gear2.copy(); g2.apply_transform(rotation_matrix(-phi, axis2_dir, point=axis2_origin))
    return g1, g2

worst = 0.0
for phi in np.linspace(0, 2 * np.pi / N, P["check_phases"]):
    g1, g2 = pose(phi)
    inter = trimesh.boolean.intersection([g1, g2], engine="manifold")
    vol, depth = 0.0, 0.0
    if len(inter.faces):
        vol = inter.volume
        pts = inter.sample(20000)
        t1 = cKDTree(g1.sample(400000)); t2 = cKDTree(g2.sample(400000))
        depth = float(np.maximum(t1.query(pts)[0], t2.query(pts)[0]).max())
    worst = max(worst, depth)
    print(f"  phi={np.degrees(phi):5.1f} deg  overlap={vol:7.3f} mm^3  depth~{depth:.3f} mm")
print("worst-case interference over one tooth pitch:", round(worst, 3), "mm  (0 = clean)")

# ------------------------------------------------------------------ stand (static), built in world frame
def to_world(m):
    m = m.copy(); m.apply_transform(W)
    if np.linalg.det(W[:3, :3]) < 0 and not np.allclose(m.face_normals[:10], (W[:3, :3] @ gear1.face_normals[:0].T).T if False else m.face_normals[:10]):
        pass
    m.fix_normals(); return m
gear1_w, gear2_w = to_world(gear1), to_world(gear2)
hinge1_pt, hinge1_ax = W[:3, :3] @ np.zeros(3), W[:3, :3] @ np.array([0, 0, 1.0])
hinge2_pt, hinge2_ax = W[:3, :3] @ axis2_origin, W[:3, :3] @ axis2_dir

z_low = min(gear1_w.bounds[0][2], gear2_w.bounds[0][2])
base_top = z_low - 32.0
bx, by, bz = P["base"]
base_cx = (gear1_w.bounds[0][0] + gear2_w.bounds[1][0]) / 2
base_cy = rp   # midway between the two shaft planes (Y=0 and Y=2 r_p)

from shapely.geometry import box as shp_box
def rounded_slab(w, h, r, thickness, z0, cx, cy):
    poly = shp_box(-w / 2, -h / 2, w / 2, h / 2).buffer(r, join_style=1, resolution=12).buffer(-r + 1e-6, join_style=1, resolution=12)
    m = trimesh.creation.extrude_polygon(poly, thickness)
    m.apply_translation([cx, cy, z0]); return m

def superellipse_ring(cx, cy, z, ax_, ay_, n=48, p=3.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s_ = np.cos(t), np.sin(t)
    x = ax_ * np.sign(c) * np.abs(c) ** (2 / p); y = ay_ * np.sign(s_) * np.abs(s_) ** (2 / p)
    return np.column_stack([cx + x, cy + y, np.full(n, z)])

def pillar(c, shaft_dir, lean_x):
    """Tapered superellipse pillar on a curved spine from the base up to the bearing centre c,
    blended into a cylindrical bearing housing aligned with the shaft."""
    pw, pt = P["post_w"], P["post_t"]
    foot = np.array([c[0] + lean_x, c[1], base_top - 2.0])
    top = np.array([c[0], c[1], c[2]])
    ctrl = np.array([foot[0], foot[1], 0.55 * foot[2] + 0.45 * top[2]])      # vertical start, lean near the top
    rings = []
    nseg = 26
    for i, t in enumerate(np.linspace(0, 1, nseg)):
        pos = (1 - t) ** 2 * foot + 2 * (1 - t) * t * ctrl + t ** 2 * top     # quadratic Bezier spine
        ease = 1 - (1 - t) ** 2
        ax_ = (1.55 - 0.70 * ease) * pw / 2 * (1.0 + 0.25 * (t < 0.08))        # flared foot, slimmer shaft
        ay_ = (1.45 - 0.55 * ease) * pt / 2 * (1.0 + 0.25 * (t < 0.08))
        rings.append(superellipse_ring(pos[0], pos[1], pos[2], ax_, ay_))
    body = close_loft(np.vstack(rings), 48, nseg)
    R = trimesh.geometry.align_vectors([0, 0, 1], shaft_dir)
    hr = pw * 0.62
    boss = cylinder(hr, -pt * 0.95, pt * 0.95, 64); boss.apply_transform(R); boss.apply_translation(c)
    rim1 = cylinder(hr * 0.85, -pt * 1.10, pt * 1.10, 64); rim1.apply_transform(R); rim1.apply_translation(c)   # proud end faces
    cap = trimesh.creation.capsule(radius=hr * 0.55, height=pt * 1.0, count=[24, 24])
    cap.apply_transform(R); cap.apply_translation(c)
    bore = cylinder(P["shaft_radius"] + 0.7, -pt * 2.5, pt * 2.5, 48); bore.apply_transform(R); bore.apply_translation(c)
    collar = cylinder(P["shaft_radius"] + 2.2, -pt * 1.1, pt * 1.1, 48); collar.apply_transform(R); collar.apply_translation(c)
    p = trimesh.boolean.union([body, boss, rim1, cap, collar], engine="manifold")
    p = trimesh.boolean.difference([p, bore], engine="manifold")
    cut = box([pw * 6, pw * 6, 200], [c[0], c[1], base_top - 100])
    return trimesh.boolean.difference([p, cut], engine="manifold")

base = trimesh.boolean.union([
    rounded_slab(bx, by, 14, bz, base_top - bz, base_cx, base_cy),                       # main plate
    rounded_slab(bx - 22, by - 22, 10, 4.0, base_top, base_cx, base_cy),                 # raised plinth
], engine="manifold")
base_top = base_top + 4.0                                                               # pillars stand on the plinth
post1 = pillar(hinge1_pt + (L + P["shaft_extra"] - P["bearing_inset"]) * hinge1_ax, hinge1_ax, lean_x=+6.0)
post2 = pillar(hinge2_pt + (Z0 - P["shaft_extra"] + P["bearing_inset"]) * hinge2_ax, hinge2_ax, lean_x=-6.0)
stand = union([base, post1, post2])
print(f"stand: watertight={stand.is_watertight}; base top at Z={base_top:.1f}")
# gears vs stand clearance over a full crank turn (world frame)
from trimesh.collision import CollisionManager
cm = CollisionManager(); cm.add_object("stand", stand)
gap_min = 1e9
for phi in np.linspace(0, 2 * np.pi, 12, endpoint=False):
    a = gear1_w.copy(); a.apply_transform(rotation_matrix(phi, hinge1_ax, point=hinge1_pt))
    b = gear2_w.copy(); b.apply_transform(rotation_matrix(-phi, hinge2_ax, point=hinge2_pt))
    for m in (a, b):
        hit = cm.in_collision_single(m)
        gap_min = min(gap_min, -1.0 if hit else cm.min_distance_single(m))
print(f"gears/crank vs stand over a full turn: {'COLLISION' if gap_min < 0 else f'clear, min gap {gap_min:.2f} mm (shaft bore gap is 0.7)'}")

# ------------------------------------------------------------------ export
gear1_w.export(os.path.join(OUT, "gear1_crank.obj"))
gear2_w.export(os.path.join(OUT, "gear2.obj"))
gear.export(os.path.join(OUT, "gear_local_frame.obj"))          # bare gear, axis +z, throat at z=0
stand.export(os.path.join(OUT, "stand.obj"))
for name, m in dict(hub_only=hub, shaft1=shaft1, crank_arm=arm, handle=handle, gear_only=gear).items():
    m.export(os.path.join(OUT, "parts", f"{name}_simframe.obj"))
trimesh.Scene({"gear1_crank": gear1_w, "gear2": gear2_w, "stand": stand}).export(os.path.join(OUT, "scene.glb"))

meta = dict(
    units="mm", parameters=P,
    derived=dict(tip_radius=ra, hub_radius=rh, normal_pitch=pitch_n, shaft_angle_deg=2 * P["skew_angle"],
                 centre_distance=2 * rp, gear_ratio=1.0, worst_case_interference_mm=float(worst),
                 common_ruling=dict(point=[rp, 0, 0], direction=d_common.tolist())),
    frame="world: Z up (gravity -Z), X left-right, Y depth. base top at Z=base_top.",
    base_top=float(base_top),
    world_from_sim=W.tolist(),
    bodies=dict(
        gear1_crank=dict(file="gear1_crank.obj", hinge_point=hinge1_pt.tolist(), hinge_axis=hinge1_ax.tolist(),
                         mass_kg_if_PLA=float(gear1.volume * 1e-9 * 1240),
                         note="drive by applying torque about the hinge, or push the grip"),
        gear2=dict(file="gear2.obj", hinge_point=hinge2_pt.tolist(), hinge_axis=hinge2_ax.tolist(),
                   mass_kg_if_PLA=float(gear2.volume * 1e-9 * 1240)),
        stand=dict(file="stand.obj", static=True),
    ),
    kinematics="theta2 = -theta1 about each body's own hinge axis (1:1). Teeth interleave along the common ruling.",
)
with open(os.path.join(OUT, "scene.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("written to", OUT)
