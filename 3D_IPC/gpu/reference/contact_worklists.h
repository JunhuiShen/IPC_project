#pragma once
#include <cstddef>

namespace ipc_gpu::solver {
// Each color contains its retained candidates once, in ascending original
// pair order. Optional box certificates remove only provably separated pairs
// AFTER graph coloring. Device offsets/indices are rebuilt with the broad phase. The host
// maximum is allocation/launch metadata only; no geometry is read back.
struct ColorContactWorklist {
    const std::size_t* offsets=nullptr;
    const int* pair_indices=nullptr;
    std::size_t max_color_pairs=0;
    // Rebuild-scoped separation proof, valid for every motion inside the blue
    // boxes. Present even when indexed color queues are disabled.
    const unsigned char* box_separated=nullptr;
};
}
