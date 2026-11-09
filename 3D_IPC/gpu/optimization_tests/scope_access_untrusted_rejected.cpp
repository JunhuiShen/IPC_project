#include "linear_pipeline.h"
#include "rigid_translation_proxy_owner.h"
void untrusted(ipc_gpu::solver::GpuLinearClipper& clipper) {
    ipc_gpu::solver::rigid_proxy_owner::Storage storage;
    ipc_gpu::solver::rigid_proxy_owner::Scope scope(&clipper,storage);
}
