#include "linear_pipeline.h"
#include "rigid_translation_proxy_owner.h"
#include "solver.h"
namespace ipc_gpu::solver {
void GpuSolver::advance_substep(const StepParameters&,const std::vector<Vector3>&,
                              const std::vector<SdfCollider>&) {
    GpuLinearClipper clipper;rigid_proxy_owner::Storage storage;
    rigid_proxy_owner::Scope scope(&clipper,storage);
}
}
