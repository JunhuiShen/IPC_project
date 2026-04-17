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

// Cloth + pinned-cylinder drape scene: a 2.4x2.4 ground cloth (corners pinned),
// one short horizontal cylinder above it (axis along z, every vertex Dirichlet-
// pinned to act as a static collider), and a vertical stack of fifteen
// 0.50x0.50 cloths (nx=ny=16) dropped from rest onto the cylinder. The cloths
// drape over it, slide off, and pile on the ground.
// Needs --d_hat > 0 on the command line for contacts to work.
void build_cloth_cylinder_drop_example(RefMesh& ref_mesh,
                                       DeformedState& state,
                                       std::vector<Vec2>& X,
                                       std::vector<Pin>& pins);

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

// Square cloth clamped on two opposite short edges. The clamps counter-rotate
// about the cloth's midline axis so the relative twist reaches
// args.twist_turns full turns over args.num_frames / args.fps seconds.
// Needs --d_hat > 0 for self-contact.
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
