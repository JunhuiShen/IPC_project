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

// Example 3: a stack of small cloths is dropped onto a static sphere; each
// cloth briefly drapes on the sphere, slides off, and falls onto a static
// ground plane below. Both obstacles are SDFs (sphere + plane), each with a
// visual mesh appended to static_x / static_tris for export -- the
// simulation only sees them through their analytic SDF.
void build_cloth_pile_example(const IPCArgs3D& args,
                              RefMesh& ref_mesh,
                              DeformedState& state,
                              std::vector<Vec2>& X,
                              std::vector<Pin>& pins,
                              SimParams& params,
                              std::vector<Vec3>& static_x,
                              std::vector<int>&  static_tris);
