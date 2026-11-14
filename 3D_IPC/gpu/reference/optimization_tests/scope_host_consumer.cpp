#include "rigid_translation_proxy_owner.h"
namespace ipc_gpu::solver::rigid_proxy_owner {
// Exactly one definition in this HOST-ONLY two-TU test program. Production
// defines it in linear_pipeline.cu instead; this file is never linked to CUDA.
thread_local Scope* current_scope=nullptr;
}
std::size_t proxy_scope_host_consume(const ipc_gpu::solver::GpuLinearClipper* owner,
    const ipc_gpu::solver::rigid_proxy_owner::Storage& current,bool enabled) {
    return ipc_gpu::solver::rigid_proxy_owner::Scope::consume(owner,current,enabled).proxy_nodes();
}
bool proxy_scope_host_bound(const ipc_gpu::solver::GpuLinearClipper* owner) {
    return ipc_gpu::solver::rigid_proxy_owner::Scope::binding(owner)!=nullptr;
}
