#pragma once

#include "physics.h"

#include <vector>
#include <memory>

class BroadPhase;

// One immutable mesh/material/pin snapshot per solve; numerical contact and
// element data are refreshed on every step. Re-prepare after editing topology,
// rest state, pins, material, or the rigid-proxy policy. Not concurrently usable.
class JacobiStepWorkspace {
public:
    JacobiStepWorkspace();
    ~JacobiStepWorkspace();
    void prepare(const RefMesh&, const VertexTriangleMap&, const std::vector<Pin>&,
        const SimParams&, std::size_t count, bool hold_rigid_proxies = false);
    const std::vector<ShapeGrads>& rest_gradients() const;
    // Optional candidates must cover d_hat at these exact positions and retain
    // deformable vertex incidence with the same solid-surface/ownership policy.
    // They remain immutable until the call joins.
    std::vector<Vec3> step(const RefMesh&, const VertexTriangleMap&,
        const std::vector<Pin>&, const SimParams&, const std::vector<Vec3>& positions,
        const std::vector<Vec3>& target, const std::vector<Vec3>& velocity,
        const std::vector<Vec3>* previous = nullptr, const BroadPhase* candidates = nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The mixed solver may hold rigid proxies fixed during particle assembly;
// their generalized-coordinate proposals are assembled separately.
std::vector<Vec3> jacobi_step(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params, const std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat, const std::vector<Vec3>& v, const std::vector<Vec3>* previous_positions = nullptr, bool hold_rigid_proxies = false);
