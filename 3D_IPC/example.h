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

// Example 9: five vertically aligned rigid polygonal prisms falling under
// gravity and stacking on a horizontal ground plane.
void build_five_rigid_polygon_drop_stack_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    std::vector<Vec3>& static_x, std::vector<int>& static_tris);

// Example 10: five differently oriented rigid polygonal prisms falling under
// gravity, tumbling apart, and scattering across a horizontal ground plane.
void build_five_rigid_polygon_drop_scatter_example(
    const IPCArgs3D& args, RefMesh& ref_mesh,
    DeformedState& state, std::vector<Vec2>& X,
    std::vector<Pin>& pins, SimParams& params,
    std::vector<Vec3>& static_x, std::vector<int>& static_tris);
