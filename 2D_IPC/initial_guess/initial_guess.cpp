#include "initial_guess.h"
#include "trivial.h"
#include "affine.h"
#include "ccd.h"

#include <stdexcept>

namespace initial_guess {

    void apply(Type initial_guess_type,
               const State2D& state,
               const RefMesh& ref_mesh,
               Vec& xnew,
               double dt,
               double eta) {
        if (initial_guess_type == Type::Trivial) {
            trivial::apply(state, xnew);
            return;
        }

        if (initial_guess_type == Type::Affine) {
            affine::Params ap = affine::compute_affine_params(state);
            affine::apply(ap, state, xnew, dt);
            return;
        }

        if (initial_guess_type == Type::CCD) {
            ccd::apply(state, ref_mesh, xnew, dt, eta);
            return;
        }

        throw std::runtime_error("Unknown initial_guess::Type");
    }

} // namespace initial_guess
