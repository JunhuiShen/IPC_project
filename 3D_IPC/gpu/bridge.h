#pragma once
#include "physics.h"
#include "solver.h"

namespace ipc_gpu::solver {
// Host-only setup and prescribed boundary conversion; these functions never
// assemble evolving energy/contact derivatives or perform a numerical update.
SceneDescription describe_scene(const RefMesh&, const DeformedState&,
                                const std::vector<Pin>&, const SimParams&);
// original_reference_threads models the logical CPU reference team used by the
// original cooperative contact dispatcher. It defaults to the documented
// CPU64 Example 1 recipe and is independent of the host OpenMP team size.
StepParameters describe_step(const SimParams&,int original_reference_threads=64);
std::vector<SdfCollider> describe_colliders(const SimParams&);
std::vector<Vector3> describe_pin_targets(std::size_t nodes, const std::vector<Pin>&);
void copy_snapshot(const StateSnapshot&, DeformedState&);
} // namespace ipc_gpu::solver
