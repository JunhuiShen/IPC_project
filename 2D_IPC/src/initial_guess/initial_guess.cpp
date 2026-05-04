#include "initial_guess.h"
#include "trivial.h"
#include "affine.h"
#include "ccd.h"
#include "trust_region.h"

#include <stdexcept>

namespace initial_guess {

    void apply(Type initial_guess_type,
               const std::vector<BlockRef>& blocks,
               Vec& x_combined,
               Vec& v_combined,
               const std::vector<char>& segment_valid,
               double dt,
               double dhat,
               double eta) {
        (void)dhat; // currently unused by the modular initial-guess implementations

        if (initial_guess_type == Type::Trivial) {
            trivial::apply(blocks, x_combined, v_combined);
            return;
        }

        if (initial_guess_type == Type::Affine) {
            affine::Params ap = affine::compute_affine_params_global(blocks);
            const int total = total_nodes(blocks);

            x_combined.assign(2 * total, 0.0);
            v_combined.assign(2 * total, 0.0);

            affine::build_v_combined_from_affine(v_combined, blocks, ap);

            for (const auto& b : blocks) {
                affine::apply_to_block(ap, b, dt);
            }

            build_x_combined_from_xnew(x_combined, blocks);
            return;
        }

        if (initial_guess_type == Type::CCD) {
            ccd::apply(blocks, x_combined, v_combined, segment_valid, dt, eta);
            return;
        }

        if (initial_guess_type == Type::TrustRegion) {
            trust_region::apply(blocks, x_combined, v_combined, segment_valid, dt, eta);
            return;
        }

        throw std::runtime_error("Unknown initial_guess::Type");
    }

} // namespace initial_guess
