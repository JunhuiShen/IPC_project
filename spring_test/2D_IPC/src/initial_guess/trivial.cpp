#include "trivial.h"

namespace initial_guess::trivial {

    void apply(const std::vector<BlockRef>& blocks,
               Vec& x_combined,
               Vec& v_combined) {
        const int total = total_nodes(blocks);
        x_combined.assign(2 * total, 0.0);
        v_combined.assign(2 * total, 0.0);

        for (const auto& b : blocks)
            *b.xnew = b.chain->x;

        build_v_combined_from_chain_velocities(v_combined, blocks);
        build_x_combined_from_current_positions(x_combined, blocks);
    }

} // namespace initial_guess::trivial
