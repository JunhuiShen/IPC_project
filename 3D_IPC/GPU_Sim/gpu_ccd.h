#pragma once
#include "gpu_mesh.h"
#include "physics.h"
#include "broad_phase.h"
#include <vector>

// --------------------------------------------------------------------------
// gpu_ccd_min_toi
// Parallel min reduction over all CCD candidate pairs.
// Both nt and ss pairs are checked; returns the minimum TOI found.
// d_x and d_dx are flat device buffers: [num_verts * 3], interleaved xyz.
// Returns 1.0 if no collision is found.
// --------------------------------------------------------------------------
double gpu_ccd_min_toi(const GPUBroadPhaseCache& bp,
                       const DeviceBuffer<double>& d_x,
                       const DeviceBuffer<double>& d_dx);

// --------------------------------------------------------------------------
// gpu_ccd_initial_guess
// Drop-in replacement for ccd_initial_guess in solver.cpp.
// BroadPhase build and dx computation stay on CPU.
// The two reduction loops run on the GPU via gpu_ccd_min_toi.
// --------------------------------------------------------------------------
std::vector<Vec3> gpu_ccd_initial_guess(const std::vector<Vec3>& x,
                                        const std::vector<Vec3>& xhat,
                                        const RefMesh& ref_mesh);
