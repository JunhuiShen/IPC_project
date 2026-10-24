#pragma once
#include "resident_solver.h"
#include <memory>

namespace ipc_gpu::resident {
struct StepParameters;

// Device-only DAT geometry, contact assembly and truncation. The host schedules
// colors/detections and reads BVH counts; evolving numerical arrays stay on GPU.
// The owner validates cloth-only topology/parameters and begins/finishes the
// deformable substep. Device status is preserved; 60+ContactStatus is a contact
// input/capacity/zero-gap failure and 70 is SDF failure. Nonfinite local Newton systems
// retain the previously accepted displacement, matching CPU DAT.
class ResidentDAT {
public:
    ResidentDAT();
    ~ResidentDAT();
    ResidentDAT(const ResidentDAT&)=delete;
    ResidentDAT& operator=(const ResidentDAT&)=delete;
    void solve(const DeformableScene&,const DeformableState&,const ElasticWorkspace&,const BroadPhaseTopology&,const SdfScene&,const StepParameters&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Standalone cloth/particle owner. Its library has no ordinary CCD solver or
// CPU numerical dependency; state remains resident between requested snapshots.
class ResidentDATSolver {
public:
    explicit ResidentDATSolver(const SceneDescription&);
    ~ResidentDATSolver();
    ResidentDATSolver(const ResidentDATSolver&)=delete;
    ResidentDATSolver& operator=(const ResidentDATSolver&)=delete;
    void advance_substep(const StepParameters&,const std::vector<Vector3>& pins,const std::vector<SdfCollider>& colliders);
    StateSnapshot snapshot()const;
    void begin_profile_frame(int frame);
    void end_profile_frame();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ipc_gpu::resident
