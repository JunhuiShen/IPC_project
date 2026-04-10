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
