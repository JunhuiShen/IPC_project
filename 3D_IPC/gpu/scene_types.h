#pragma once

#include "elastic.h"
#include "broadphase.h"
#include "sdf_types.h"

namespace ipc_gpu::solver {

// Geometry selector for ordinary (non-DAT) contact assembly. Robust keeps the
// qualified exact-fallback evaluator; Original selects the independent
// evaluate_contact_original path used by the CPU exact_computation_fallback
// == false mode. This is a routing option, not a second numerical model.
enum class ContactGeometryPolicy { Robust, Original };

// Linear clipping/commit selector. Robust is the existing conservative
// certificate. OriginalSolver reproduces the CPU original-solver commit rule;
// OriginalColoredGuess reproduces the CPU original colored-predictor rule.
enum class LinearClipMode { Robust, OriginalSolver, OriginalColoredGuess };

// Scheduler-family metadata for the CPU original 1e-28 cooperative cutoff.
// This is NOT a second geometry toggle: it only decides whether the
// solver-computed cooperative node mask participates in the step filter.
enum class OriginalUpdateConvention { StoredNoop, CooperativeSquaredNorm };

// Numerical storage for the GPU solver. Every pointer is on the device.
// A rigid body's proxy vertices share one graph block, after deformable blocks.
struct RigidScene {
    std::size_t bodies;
    const int* body_blocks;
    const int* node_body; // -1 for deformable nodes
    const double (*reference)[3]; // body-local, indexed by global vertex
    const std::size_t* node_offsets;
    const int* nodes;
    const double* mass;
    const double (*second_moment)[9];
    const unsigned char *translate, *rotate;
};

struct RigidState {
    double (*com)[3], (*velocity)[3], (*orientation)[4], (*omega)[3];
    double (*previous_com)[3], (*previous_velocity)[3];
    double (*previous_orientation)[4], (*previous_omega)[3];
};

struct RigidWorkspace {
    Derivatives* systems;
    double (*proposed_com)[3], (*proposed_omega)[3], (*proposed_orientation)[4];
    // Exact represented accepted endpoints are stored separately; never
    // re-form an endpoint from a scalar after its CCD validation.
    double (*accepted_com)[3], (*accepted_orientation)[4];
    double (*accepted_omega)[3];
};

struct ContactParameters {
    double d_hat, k_barrier, dt, friction, eps_v, eps;
};

struct SdfScene {
    const SdfCollider* colliders;
    int count;
    SdfParameters parameters;
};

} // namespace ipc_gpu::solver
