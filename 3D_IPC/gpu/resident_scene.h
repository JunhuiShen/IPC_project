#pragma once

#include "resident_elastic.h"
#include "resident_broadphase.h"
#include "resident_sdf.cuh"

namespace ipc_gpu::resident {

// Numerical storage for the resident solver. Every pointer is on the device.
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

} // namespace ipc_gpu::resident
