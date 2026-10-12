#pragma once
#include "physics.h"
#include "resident_solver.h"

namespace ipc_gpu::resident {
// Host-only setup and prescribed boundary conversion; these functions never
// assemble evolving energy/contact derivatives or perform a numerical update.
SceneDescription describe_scene(const RefMesh&, const DeformedState&,
                                const std::vector<Pin>&, const SimParams&);
StepParameters describe_step(const SimParams&);
std::vector<SdfCollider> describe_colliders(const SimParams&);
std::vector<Vector3> describe_pin_targets(std::size_t nodes, const std::vector<Pin>&);
void copy_snapshot(const StateSnapshot&, DeformedState&);
} // namespace ipc_gpu::resident
