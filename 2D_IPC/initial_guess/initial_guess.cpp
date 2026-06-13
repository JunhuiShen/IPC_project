#include "initial_guess.h"
#include "trivial.h"
#include "affine.h"
#include "ccd.h"

#include <stdexcept>

void apply_initial_guess(InitialGuessType initial_guess_type, const State2D& state,
                         const RefMesh& ref_mesh, Vec& xnew, double dt, double eta) {
    if (initial_guess_type == InitialGuessType::Trivial) {
        apply_trivial_initial_guess(state, xnew);
        return;
    }

    if (initial_guess_type == InitialGuessType::Affine) {
        const AffineInitialGuessParams params = compute_affine_initial_guess_params(state);
        apply_affine_initial_guess(params, state, xnew, dt);
        return;
    }

    if (initial_guess_type == InitialGuessType::CCD) {
        apply_ccd_initial_guess(state, ref_mesh, xnew, dt, eta);
        return;
    }

    throw std::runtime_error("Unknown InitialGuessType");
}
