#pragma once

#include <cstddef>

// Integer-only launch and CSR indexing for rigid proxy execution.
// No CUDA runtime headers, storage, numerical state, or physics belong here.
#if defined(__CUDACC__)
#define IPC_GPU_PROXY_INDEX_HD __host__ __device__
#else
#define IPC_GPU_PROXY_INDEX_HD
#endif

namespace ipc_gpu::solver::proxy_index {

constexpr unsigned width = 128;
constexpr unsigned maximum_blocks = 65535;
constexpr std::size_t invalid_index = static_cast<std::size_t>(-1);

struct Plan {
    unsigned tiles_per_body = 0;
    unsigned body_batches = 0;
    unsigned blocks = 0;
};

IPC_GPU_PROXY_INDEX_HD constexpr Plan make_plan(std::size_t bodies) noexcept {
    const unsigned tiles = bodies <= 64 ? 128u : 1u;
    const unsigned maximum_batches = maximum_blocks / tiles;
    const unsigned batches = bodies < maximum_batches
        ? static_cast<unsigned>(bodies) : maximum_batches;
    return {tiles, batches, tiles * batches};
}

// A flat block owns one tile of each body in its strided body lane.
IPC_GPU_PROXY_INDEX_HD constexpr std::size_t first_body(
    unsigned block, Plan plan) noexcept {
    if (!plan.tiles_per_body || !plan.body_batches || block >= plan.blocks)
        return invalid_index;
    const unsigned body_lane = block / plan.tiles_per_body;
    return body_lane < plan.body_batches ? body_lane : invalid_index;
}

IPC_GPU_PROXY_INDEX_HD constexpr unsigned tile(unsigned block, Plan plan) noexcept {
    if (!plan.tiles_per_body || first_body(block, plan) == invalid_index)
        return plan.tiles_per_body;
    return block % plan.tiles_per_body;
}

// Return bodies as the end sentinel. Subtraction makes even bodies==SIZE_MAX
// safe: no overflowing speculative body+stride is formed.
IPC_GPU_PROXY_INDEX_HD constexpr std::size_t next_body(
    std::size_t body, std::size_t bodies, Plan plan) noexcept {
    const std::size_t stride = plan.body_batches;
    if (body >= bodies || !stride || stride >= bodies - body) return bodies;
    return body + stride;
}

struct RowCursor {
    std::size_t first = 0;
    std::size_t relative = 0;
    std::size_t count = 0;
    std::size_t stride = 0;

    // The extent guard also makes explicitly constructed synthetic cursors
    // safe; row_cursor() already guarantees this property by construction.
    IPC_GPU_PROXY_INDEX_HD constexpr bool valid() const noexcept {
        return stride && count <= invalid_index - first && relative < count;
    }

    IPC_GPU_PROXY_INDEX_HD constexpr std::size_t slot() const noexcept {
        return valid() ? first + relative : invalid_index;
    }

    // End by invalidating the cursor, so for(; c.valid(); c.advance()) cannot
    // repeat its final slot. No count-relative or relative+stride may wrap.
    IPC_GPU_PROXY_INDEX_HD constexpr void advance() noexcept {
        if (!valid() || stride >= count - relative) relative = count;
        else relative += stride;
    }
};

IPC_GPU_PROXY_INDEX_HD constexpr RowCursor row_cursor(
    std::size_t first, std::size_t last, unsigned body_tile,
    unsigned lane, Plan plan) noexcept {
    const std::size_t tiles = plan.tiles_per_body;
    if (!tiles || body_tile >= tiles || lane >= width
        || tiles > invalid_index / width)
        return {};
    const std::size_t stride = tiles * width;
    // body_tile < tiles and lane < width imply relative < stride, so this
    // multiply/add is safe once the stride multiplication has been checked.
    const std::size_t relative = std::size_t(body_tile) * width + lane;
    return {first, relative, last > first ? last - first : 0, stride};
}

} // namespace ipc_gpu::solver::proxy_index

#undef IPC_GPU_PROXY_INDEX_HD
