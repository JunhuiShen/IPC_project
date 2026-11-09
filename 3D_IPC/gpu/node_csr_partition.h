#pragma once

#include <cstddef>

// Integer-only helpers for the isolated active-node CSR partition prototype.
// The header owns no CUDA storage and requires no CUDA runtime headers.
#if defined(__CUDACC__)
#define IPC_GPU_NODE_CSR_HD __host__ __device__
#else
#define IPC_GPU_NODE_CSR_HD
#endif

namespace ipc_gpu::solver::node_csr_partition {

constexpr std::size_t maximum_index = static_cast<std::size_t>(-1);

struct Plan {
    std::size_t first_row = 0;
    std::size_t row_stride = 0;
    std::size_t first_rank = 0;
    std::size_t rank_stride = 0;
};

IPC_GPU_NODE_CSR_HD constexpr Plan make_plan(std::size_t active,
    std::size_t grid, std::size_t block, std::size_t threads,
    std::size_t lane, bool enabled) noexcept {
    const Plan invalid{active, 0, 0, 0};
    if (!active || !grid || block >= grid || !threads || lane >= threads)
        return invalid;
    if (!enabled || active >= grid)
        return {block, grid, lane, threads};

    const std::size_t row = block % active;
    const std::size_t part = block / active;
    // row < active < grid, so both subtractions and the final +1 are safe.
    const std::size_t parts = 1 + (grid - 1 - row) / active;
    if (parts > maximum_index / threads) return invalid;
    const std::size_t stride = parts * threads;
    // block < grid implies part < parts. Together with lane < threads,
    // this bounds part*threads+lane strictly below the checked stride.
    return {row, active, part * threads + lane, stride};
}

IPC_GPU_NODE_CSR_HD constexpr std::size_t next_index(std::size_t current,
    std::size_t end, std::size_t stride) noexcept {
    if (current >= end || !stride || stride >= end - current) return end;
    return current + stride;
}

IPC_GPU_NODE_CSR_HD constexpr std::size_t next_row(std::size_t row,
    std::size_t active, Plan plan) noexcept {
    return next_index(row, active, plan.row_stride);
}

IPC_GPU_NODE_CSR_HD constexpr std::size_t next_rank(std::size_t rank,
    std::size_t count, Plan plan) noexcept {
    return next_index(rank, count, plan.rank_stride);
}

} // namespace ipc_gpu::solver::node_csr_partition

#undef IPC_GPU_NODE_CSR_HD
