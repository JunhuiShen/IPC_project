#pragma once

#include "physics.h"
#include "ipc_args.h"
#include <vector>

// Example 1: counter-rotating clamps about +x.
struct TwistSpec {
    std::vector<int>  left_pin_indices;
    std::vector<int>  right_pin_indices;
    std::vector<Vec3> left_initial_targets;
    std::vector<Vec3> right_initial_targets;
    Vec3   axis_point{Vec3::Zero()};
    double omega_left  = 0.0;     // rad/s, signed
    double omega_right = 0.0;
};

// Square cloth, two short edges clamped, counter-rotating at args.twist_rate Hz.
void build_twisting_cloth_example(const IPCArgs3D& args,
                                  RefMesh& ref_mesh,
                                  DeformedState& state,
                                  std::vector<Vec2>& X,
                                  std::vector<Pin>& pins,
                                  TwistSpec& spec);

void update_twist_pins(std::vector<Pin>& pins, const TwistSpec& spec, double t);

// Example 2: closed-loop cloth strips wrap two cylinders that counter-rotate
// about +y; pins, visual mesh, and SDF axes all yaw together so the wrap pin
// (orbiting at pin_r > r) never crosses the SDF surface.
struct CylinderTwistSpec {
    std::vector<int>  top_pin_indices;
    std::vector<int>  bot_pin_indices;
    std::vector<Vec3> top_initial_targets;
    std::vector<Vec3> bot_initial_targets;
    Vec3   top_axis_point{Vec3::Zero()};
    Vec3   bot_axis_point{Vec3::Zero()};
    double omega_top     = 0.0;
    double omega_bot     = 0.0;
    double t_settle      = 0.0;
    double t_ramp        = 0.0;
    double max_abs_theta = 0.0;   // 0 disables the cap
    bool   untwist       = false;
    double t_hold        = 0.0;

    std::vector<Vec3> static_x_rest;
    int top_v_begin = 0;
    int top_v_end   = 0;
    int bot_v_begin = 0;
    int bot_v_end   = 0;
};

// Build the scene described above. Pins counter-rotate at tcyl_twist_rate Hz
// up to tcyl_max_turn turns; tcyl_untwist=true mirrors the trapezoid back to
// 0. Visual cylinder mesh is appended to static_x/static_tris.
void build_two_cylinder_twist_example(const IPCArgs3D& args,
                                      RefMesh& ref_mesh,
                                      DeformedState& state,
                                      std::vector<Vec2>& X,
                                      std::vector<Pin>& pins,
                                      SimParams& params,
                                      std::vector<Vec3>& static_x,
                                      std::vector<int>&  static_tris,
                                      CylinderTwistSpec& spec);

void update_cylinder_twist_pins(std::vector<Pin>& pins,
                                const CylinderTwistSpec& spec,
                                double t);

// Yaws the visual cylinder vertices in `static_x` about +y by effective_theta(t).
void update_cylinder_visuals(std::vector<Vec3>& static_x,
                             const CylinderTwistSpec& spec,
                             double t);

// Yaws the SDF cylinder axes about +y by the same effective_theta(t) the pins
// use, so the collision surface co-rotates with the wrap pin's orbit.
void update_cylinder_sdfs(SimParams& params,
                          const CylinderTwistSpec& spec,
                          double t);

// Example 3: rectangular cloth (tu_width x tu_size) wrapping a single
// horizontal cylinder's underside. Both top edges are statically pinned
// (stretchy clamping bars), and the bottom-wrap rows co-rotate with the
// cylinder. SDF axis yaws about +y in lock-step with the wrap pins,
// twisting the cloth between rotating wrap and fixed bars.
struct TwistUntwistSpec {
    std::vector<int>  end_pin_indices;
    std::vector<Vec3> end_initial_targets;
    std::vector<int>  wrap_pin_indices;
    std::vector<Vec3> wrap_initial_targets;

    Vec3   cyl_axis_point{Vec3::Zero()};
    double omega         = 0.0;            // rad/s
    double t_settle      = 0.0;
    double t_ramp        = 0.0;
    double max_abs_theta = 0.0;
    bool   untwist       = false;
    double t_hold        = 0.0;

    // Visual cylinder slice in static_x; visual_v_rest holds pre-rotation
    // positions so each frame rotates from rest (no drift).
    int               cyl_sdf_index  = -1;
    int               visual_v_begin = 0;
    int               visual_v_end   = 0;
    std::vector<Vec3> visual_v_rest;
};

void build_twist_untwist_example(const IPCArgs3D& args,
                                 RefMesh& ref_mesh,
                                 DeformedState& state,
                                 std::vector<Vec2>& X,
                                 std::vector<Pin>& pins,
                                 SimParams& params,
                                 std::vector<Vec3>& static_x,
                                 std::vector<int>&  static_tris,
                                 TwistUntwistSpec& spec);

void update_twist_untwist_pins(std::vector<Pin>& pins,
                               const TwistUntwistSpec& spec, double t);

// Per-substep: yaws the SDF axis about +y by effective_theta(t) so the
// collider stays co-rotated with the wrap pins.
void update_twist_untwist_sdf(SimParams& params,
                              const TwistUntwistSpec& spec, double t);

// Per-frame: yaws the visual cylinder vertices about +y to match the SDF.
void update_twist_untwist_visual(std::vector<Vec3>& static_x,
                                 const TwistUntwistSpec& spec, double t);

// Example 4: avatar wearing simulated clothing.
void build_avatar_clothing_example(const IPCArgs3D& args,
                                   RefMesh& ref_mesh,
                                   DeformedState& state,
                                   std::vector<Pin>& pins,
                                   SimParams& params,
                                   std::vector<Vec3>& static_x,
                                   std::vector<int>&  static_tris);

// Example 5: a freely rotating rigid tennis racket with no gravity.
void build_rotating_tennis_racket_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params);

// Example 6: a freely rotating "|-" space tool initialized near its
// intermediate principal axis to demonstrate the Dzhanibekov effect.
void build_rotating_space_tool_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params);

// Example 7: a rigid box and an extruded hexagon falling under gravity onto a
// horizontal ground SDF.
void build_rigid_box_drop_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    std::vector<Vec3>& static_x, std::vector<int>& static_tris);

// Example 8: two rigid polygonal prisms at the same height moving toward one
// another with zero gravity.
void build_two_rigid_polygon_collision_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params);

// Example 9: twenty vertically aligned rigid polygonal prisms initialized as a
// stationary stack immediately above a horizontal ground plane.
void build_twenty_rigid_polygon_static_stack_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    std::vector<Vec3>& static_x, std::vector<int>& static_tris);

// Example 10: five equally oriented rigid polygonal prisms aligned in one
// vertical column and falling onto one another above a horizontal ground plane.
void build_five_rigid_polygon_drop_scatter_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    std::vector<Vec3>& static_x, std::vector<int>& static_tris);

// Example 11: one hundred varied regular polygonal prisms initialized in five
// collision-free layers and falling into a wide open-top box.
void build_hundred_rigid_polygon_box_drop_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    std::vector<Vec3>& static_x, std::vector<int>& static_tris);

// Example 12: fifty small rigid polygonal prisms (3 through 12 sides) falling
// onto a large horizontal rectangular cloth whose four corners are pinned.
void build_fifty_rigid_polygons_drop_on_pinned_cloth_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params);

// Example 15: ten small rigid and ten larger tetrahedralized deformable
// polygonal prisms (3 through 12 sides), all initialized flat, falling onto a
// large horizontal cloth whose two opposite sides are pinned.
void build_twenty_rigid_deformable_polygons_drop_on_pinned_cloth_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params);

// Example 13: one flat tetrahedralized deformable octagonal prism falling
// onto a horizontal SDF ground plane. The solid uses a mass density of
// 900 kg/m^3.
void build_single_deformable_solid_ground_drop_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    std::vector<Vec3>& static_x, std::vector<int>& static_tris);

// Example 14: one flat tetrahedralized deformable octagonal prism falling
// onto a large horizontal cloth whose two opposite sides are pinned.
void build_single_deformable_solid_drop_on_pinned_cloth_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params);

// Example 16: ten total polygonal prisms, alternating five rigid bodies and
// five tetrahedralized deformable solids, falling flat onto one another above
// a cloth whose two opposite sides are pinned.
void build_ten_alternating_rigid_solid_flat_stack_on_pinned_cloth_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params);

// Example 17: Bunny-solid / Spot-solid / rigid-cube / rigid-gear cycles,
// repeated twice in one vertical stack above a cloth pinned on opposite sides.
void build_two_bunny_spot_cube_gear_cycles_on_pinned_cloth_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params);

// Example 18: a fully dynamic threaded bolt starts deeply engaged in a nut and
// falls coaxially under gravity, rotating as it follows the thread. The nut's
// translation and orientation are both fixed by its rigid update label.
void build_dynamic_bolt_into_fixed_nut_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params);

// Example 19: one tetrahedral Armadillo starts in the nip between two
// fixed-center, counter-rotating gear crushers. Their angular velocities are
// initial conditions and subsequently evolve through inertia and contact.
void build_armadillo_through_gear_crushers_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params);

// Example 20: four level rows of a tetrahedral Bunny, tetrahedral Spot, rigid
// cube, and rigid gear falling onto a horizontal cloth pinned along two
// opposite sides. All bodies retain their authored, untilted orientations.
void build_four_bunny_spot_cube_gear_rows_on_pinned_cloth_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params);

// Example 21: a flat-rest cloth starts wound into a roll above an inclined
// plane SDF joined to a horizontal ground SDF. Its uphill short edge is pinned
// while gravity unwinds the free length down the ramp. A finite wedge and
// ground quad are exported through static_x/static_tris for visualization only.
void build_cloth_unrolling_down_fixed_ramp_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    std::vector<Vec3>& static_x, std::vector<int>& static_tris);

// Example 22: three parallel cloth sheets form a close stack. One complete
// short edge of every sheet is fixed, while the opposite edge is prescribed
// to oscillate along +y; the two lateral edges are free. The driven targets
// are evaluated from their immutable t=0 positions so checkpoint restarts
// recover the same phase without depending on updater call history.
struct OscillatingClothLayersSpec {
    std::vector<int> driven_pin_indices;
    std::vector<Vec3> driven_initial_targets;
    Vec3 motion_direction{Vec3::UnitY()};
    double amplitude = 0.0;
    double frequency_hz = 0.0;
};

void build_oscillating_cloth_layers_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    OscillatingClothLayersSpec& spec);

void update_oscillating_cloth_layer_pins(
    std::vector<Pin>& pins,
    const OscillatingClothLayersSpec& spec,
    double t);

// Example 23: the rigid IPC paper's wrecking-ball benchmark. Thirteen
// interlinked rigid rings and a ball with an integrated terminal ring swing
// from one fixed top link into a wall of 560 rigid cubes. The reference's
// complete scene is translated +1 m in y; its fixed plane is represented at
// y=0 by an analytic SDF ground plus a visual quad.
void build_wrecking_ball_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    std::vector<Vec3>& static_x, std::vector<int>& static_tris);
