#pragma once

#include "physics.h"
#include "ipc_args.h"
#include <vector>

// Build the "two side-by-side sheets with pinned inner corners" example scene.
// Populates ref_mesh, state, X and pins; overwrites prior contents.
void build_two_sheets_example(const IPCArgs3D& args,
                              RefMesh& ref_mesh,
                              DeformedState& state,
                              std::vector<Vec2>& X,
                              std::vector<Pin>& pins);

// Cloth-stack scene, low-resolution variant: pinned ground cloth with three
// small 0.30x0.30 cloths (nx=ny=4) dropped from rest at increasing heights.
// Cheap to run; useful for quick correctness checks.
void build_cloth_stack_example_low_res(RefMesh& ref_mesh,
                                       DeformedState& state,
                                       std::vector<Vec2>& X,
                                       std::vector<Pin>& pins);

// Cloth-stack scene, high-resolution stress variant: pinned ground cloth with
// five small 0.35x0.35 cloths (nx=ny=8), tightly spaced and given an initial
// downward velocity so layers engage contact near-simultaneously.
// Needs --d_hat > 0 on the command line for contacts to work.
void build_cloth_stack_example_high_res(RefMesh& ref_mesh,
                                        DeformedState& state,
                                        std::vector<Vec2>& X,
                                        std::vector<Pin>& pins);

// Cloth + cylinder drape scene: a static horizontal cylinder and a vertical
// stack of falling cloths. The ground and cylinder are excluded from ref_mesh;
// their geometry is returned in static_x / static_tris for a one-time write.
// SDF colliders handle contact. Needs --d_hat > 0 for contacts to work.
void build_cloth_cylinder_drop_example(const IPCArgs3D& args,
                                       RefMesh& ref_mesh,
                                       DeformedState& state,
                                       std::vector<Vec2>& X,
                                       std::vector<Pin>& pins,
                                       SimParams& params,
                                       std::vector<Vec3>& static_x,
                                       std::vector<int>&  static_tris);

// Cloth + sphere drape scene: a static ground and sphere (excluded from
// ref_mesh; geometry returned in static_x / static_tris for a one-time
// write), and a vertical stack of falling cloths. SDF colliders handle
// contact. Needs --d_hat > 0 for cloth-cloth contacts to work.
void build_cloth_sphere_drop_example(const IPCArgs3D& args,
                                     RefMesh& ref_mesh,
                                     DeformedState& state,
                                     std::vector<Vec2>& X,
                                     std::vector<Pin>& pins,
                                     SimParams& params,
                                     std::vector<Vec3>& static_x,
                                     std::vector<int>&  static_tris);

// Drives the pin-target motion for the twisting-cloth example: two groups of
// pins counter-rotate about a common +x axis (through axis_point) at their
// own signed angular speeds.
struct TwistSpec {
    std::vector<int>  left_pin_indices;       // indices into pins[]
    std::vector<int>  right_pin_indices;
    std::vector<Vec3> left_initial_targets;   // pin target positions at t=0
    std::vector<Vec3> right_initial_targets;
    Vec3   axis_point{Vec3::Zero()};
    double omega_left  = 0.0;                 // rad/s (signed)
    double omega_right = 0.0;                 // rad/s (signed)
};

// Square cloth clamped on two opposite short edges. Clamps counter-rotate
// at a fixed relative rate of args.twist_rate Hz, so total turns scales with
// run duration. Needs --d_hat > 0 for self-contact.
void build_twisting_cloth_example(const IPCArgs3D& args,
                                  RefMesh& ref_mesh,
                                  DeformedState& state,
                                  std::vector<Vec2>& X,
                                  std::vector<Pin>& pins,
                                  TwistSpec& spec);

// Rotates each pin's target about +x (through spec.axis_point) by omega * t
// radians, using omega_left for the left group and omega_right for the right.
// Call once per substep.
void update_twist_pins(std::vector<Pin>& pins, const TwistSpec& spec, double t);

// Drives example 7: N closed-loop cloth strips wrapping two horizontal
// cylinders (long axis along +x). Each cylinder rotates about +y through its
// own center, in opposite directions, swinging the cylinder ends in the x-z
// plane and dragging the pinned wrap segments with them.
//
// Pin-target angle profile (see effective_theta):
//   1. settle  : t in [0, t_settle], theta = 0
//   2. forward : trapezoidal (ramp-up, steady at omega, ramp-down) to ±max_abs_theta
//   3. hold    : dwell t_hold at peak (only if untwist == true)
//   4. reverse : mirror of forward, back to 0   (only if untwist == true)
// max_abs_theta == 0 disables the cap and runs the simple ramp-then-steady profile.
struct CylinderTwistSpec {
    std::vector<int>  top_pin_indices;
    std::vector<int>  bot_pin_indices;
    std::vector<Vec3> top_initial_targets;
    std::vector<Vec3> bot_initial_targets;
    Vec3   top_axis_point{Vec3::Zero()};
    Vec3   bot_axis_point{Vec3::Zero()};
    double omega_top     = 0.0;   // rad/s, signed
    double omega_bot     = 0.0;   // rad/s, signed
    double t_settle      = 0.0;   // seconds with omega clamped to zero
    double t_ramp        = 0.0;   // seconds to ramp omega between 0 and full
    double max_abs_theta = 0.0;   // |theta| cap (radians); 0 means no cap
    bool   untwist       = false; // after hitting peak, reverse back to 0
    double t_hold        = 0.0;   // seconds to dwell at peak before reversing

    // Rest positions of the visual cylinder vertices and the index ranges
    // (in static_x) belonging to the top and bottom cylinders. Used by
    // update_cylinder_visuals to rotate the cylinders per frame so the
    // exported collider geometry matches the live pin rotation.
    std::vector<Vec3> static_x_rest;
    int top_v_begin = 0;
    int top_v_end   = 0;
    int bot_v_begin = 0;
    int bot_v_end   = 0;
};

// Closed-loop cloth strips between two horizontal +x-axis cylinders. Each
// strip's top-half wrap is pinned to the top cylinder; bot-half wrap to the
// bottom cylinder. Pins counter-rotate about +y at args.tcyl_twist_rate Hz
// up to args.tcyl_max_turn turns; with args.tcyl_untwist == true the motion
// then reverses back to the starting orientation (twist + untwist). The
// visual cylinder mesh is appended to static_x / static_tris (and rotated
// per frame by update_cylinder_visuals); no SDF colliders are pushed —
// cloth–cylinder contact is enforced only by the pin spring, plus the
// visual radius being shrunk by args.tcyl_visual_shrink so any solver-side
// drop-row lag never visually pokes through the cylinder.
void build_two_cylinder_twist_example(const IPCArgs3D& args,
                                      RefMesh& ref_mesh,
                                      DeformedState& state,
                                      std::vector<Vec2>& X,
                                      std::vector<Pin>& pins,
                                      SimParams& params,
                                      std::vector<Vec3>& static_x,
                                      std::vector<int>&  static_tris,
                                      CylinderTwistSpec& spec);

// Rotates each pin's target about +y (through the corresponding cylinder's
// axis_point) by effective_theta(omega_{top,bot}, t, ...). Call once per substep.
void update_cylinder_twist_pins(std::vector<Pin>& pins,
                                const CylinderTwistSpec& spec,
                                double t);

// Rotates the visual cylinder vertices in `static_x` so the exported
// collider geometry matches the live pin rotation. Top cylinder verts
// (range [top_v_begin, top_v_end)) rotate about spec.top_axis_point at
// omega_top, bottom verts about bot_axis_point at omega_bot, both about
// the +y axis with the same settle/ramp curve as update_cylinder_twist_pins.
void update_cylinder_visuals(std::vector<Vec3>& static_x,
                             const CylinderTwistSpec& spec,
                             double t);
