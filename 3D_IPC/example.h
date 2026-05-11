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

// Example 3: deformable dragon (loaded from --dragon_path) squeezed between
// one or two translating plane SDFs. Gravity is disabled inside the build
// so the scene is driven only by the moving plates. Per-triangle Dm_inverse
// and per-hinge c_e are rebuilt from the 3D rest pose because the dragon
// is a closed surface with no global 2D parameterization.
//
// build_dragon_squeeze_example contains three variants behind an #if chain:
//   1. floor rises into a top-pinned dragon
//   2. ceiling descends onto a bottom-pinned dragon
//   3. floor + ceiling press toward each other, no pins (active)
// Each plate has its own rise_velocity (sign sets direction); rise_max and
// t_settle are shared so plates stay in lock-step.
struct DragonSqueezePlate {
    int    plane_index = -1;            // params.sdf_planes index
    Vec3   plane_point_rest{Vec3::Zero()};
    Vec3   rise_velocity{Vec3::Zero()}; // m/s; sign sets translation direction

    // Visual mesh slice in static_x. visual_v_rest stores the pre-motion
    // positions so update_dragon_squeeze_visual can re-translate from rest
    // each frame (avoids drift from accumulating per-frame deltas).
    int    visual_v_begin = 0;
    int    visual_v_end   = 0;
    std::vector<Vec3> visual_v_rest;
};

struct DragonSqueezeSpec {
    std::vector<DragonSqueezePlate> plates;
    double rise_max = 0.0;  // cap on |displacement| applied to every plate (m); 0 = no cap
    double t_settle = 0.0;  // seconds at the start with no motion (shared across plates)
};

void build_dragon_squeeze_example(const IPCArgs3D& args,
                                  RefMesh& ref_mesh,
                                  DeformedState& state,
                                  std::vector<Vec2>& X,
                                  std::vector<Pin>& pins,
                                  SimParams& params,
                                  std::vector<Vec3>& static_x,
                                  std::vector<int>&  static_tris,
                                  DragonSqueezeSpec& spec);

// Per-substep: snaps each plate's SDF point to plane_point_rest plus the
// clamped translation. Per-substep (not per-frame) so the dragon never sees
// a stale plane mid-substep -- same lesson as example 2's cylinders.
void update_dragon_squeeze_sdf(SimParams& params,
                               const DragonSqueezeSpec& spec, double t);

// Per-frame: re-translates each plate's visual vertices from visual_v_rest
// for the export.
void update_dragon_squeeze_visual(std::vector<Vec3>& static_x,
                                  const DragonSqueezeSpec& spec, double t);


// Example 4: clothing on an animated person. The body is delivered as a
// per-frame OBJ sequence (e.g. CLOTH3D baked via tools/bake_cloth3d_body.py)
// and treated as fully-pinned cloth — every body vertex has a pin spring whose
// target is interpolated from the source sequence each substep. The dress is
// loaded from a single OBJ as the IPC initial state and simulated forward.
//
// Topology must be stable across all body frames (vertex count and order
// identical). The builder concatenates body + dress into one RefMesh.
struct ClothingSpec {
    // Per-frame body vertex positions, indexed [frame][i_in_body].
    // body_frames.size() == number of source frames loaded.
    std::vector<std::vector<Vec3>> body_frames;

    // Map from local body-vertex index (0..body_count-1) to the index in pins[]
    // for that vertex. Used by update_clothing_pins to write each substep's
    // interpolated target.
    std::vector<int> body_pin_indices;

    // Body vertex range in the combined mesh: [body_v_begin, body_v_end).
    int body_v_begin = 0;
    int body_v_end   = 0;

    double source_fps  = 30.0; // framerate of body_frames
    int    first_frame = 0;    // source-frame index at sim t=0
};

// Concatenates body (frame 0 OBJ in args.cloth_body_dir) and dress
// (args.cloth_dress_obj) into one RefMesh; loads the rest of the body sequence
// into spec.body_frames; pins every body vertex to its frame-0 position.
// Disables gravity-driven body motion (pins enforce kinematic body).
void build_clothing_example(const IPCArgs3D& args,
                            RefMesh& ref_mesh,
                            DeformedState& state,
                            std::vector<Vec2>& X,
                            std::vector<Pin>& pins,
                            SimParams& params,
                            ClothingSpec& spec);

// Per-substep: linearly interpolates between adjacent source frames at sim
// time t and writes the result into each body pin's target_position. After
// the source sequence ends, pins clamp to the last available frame.
void update_clothing_pins(std::vector<Pin>& pins,
                          const ClothingSpec& spec,
                          double t);
