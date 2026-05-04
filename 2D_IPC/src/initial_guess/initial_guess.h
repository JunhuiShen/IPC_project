#pragma once

#include "../chain.h"
#include "../ipc_math.h"
#include <vector>
#include <stdexcept>

// ======================================================
// Unified initial guess wrapper
//
// This matches the original monolithic code:
// - arbitrary number of contiguous blocks
// - arbitrary offsets
// - supports Trivial / Affine / CCD / TrustRegion
// ======================================================

namespace initial_guess {

    enum class Type {
        Trivial,
        Affine,
        CCD,
        TrustRegion
    };

    // Description of one contiguous block in the global indexing
    struct BlockRef {
        Chain* chain;
        Vec* xnew;
        int offset;
    };

    // ------------------------------------------------------
    // Helpers for global assembly
    // ------------------------------------------------------

    inline int total_nodes(const std::vector<BlockRef>& blocks) {
        int total = 0;
        for (const auto& b : blocks)
            total += b.chain->N;
        return total;
    }

    inline void scatter_block_positions(Vec& x_combined, const Vec& x_block, int offset, int N_block) {
        for (int i = 0; i < N_block; ++i)
            set_xi(x_combined, offset + i, get_xi(x_block, i));
    }

    inline void scatter_block_velocities(Vec& v_combined, const Vec& v_block, int offset, int N_block) {
        for (int i = 0; i < N_block; ++i)
            set_xi(v_combined, offset + i, get_xi(v_block, i));
    }

    inline void build_x_combined_from_current_positions(Vec& x_combined, const std::vector<BlockRef>& blocks) {
        for (const auto& b : blocks)
            scatter_block_positions(x_combined, b.chain->x, b.offset, b.chain->N);
    }

    inline void build_x_combined_from_xnew(Vec& x_combined, const std::vector<BlockRef>& blocks) {
        for (const auto& b : blocks)
            scatter_block_positions(x_combined, *b.xnew, b.offset, b.chain->N);
    }

    inline void build_v_combined_from_chain_velocities(Vec& v_combined, const std::vector<BlockRef>& blocks) {
        for (const auto& b : blocks)
            scatter_block_velocities(v_combined, b.chain->v, b.offset, b.chain->N);
    }

    // Unified entry point
    void apply(Type initial_guess_type,
               const std::vector<BlockRef>& blocks,
               Vec& x_combined, Vec& v_combined,
               const std::vector<char>& segment_valid,
               double dt, double dhat, double eta);

} // namespace initial_guess
