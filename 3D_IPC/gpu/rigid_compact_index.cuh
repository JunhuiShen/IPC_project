#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define IPC_RIGID_COMPACT_HD __host__ __device__
#else
#define IPC_RIGID_COMPACT_HD
#endif
namespace ipc_gpu::solver::rigid_compact {
// end is exclusive; an empty range needs no encoded incident ID. This form
// avoids overflowing size_t on a 32-bit host and retains UINT32_MAX itself.
IPC_RIGID_COMPACT_HD constexpr bool absolute_ids_fit(std::size_t end) {
    return end==0||end-1<=static_cast<std::size_t>(UINT32_MAX);
}
}
#undef IPC_RIGID_COMPACT_HD
