#include "initial_guess.h"
#include "trivial.h"
#include "affine.h"
#include "ccd.h"

#include <stdexcept>

void apply_initial_guess(InitialGuessType initial_guess_type, const DeformedState& state,
                         const RefMesh& ref_mesh, const std::vector<Pin>& pins,
                         Vec& xnew, double dt, double eta) {
    if (initial_guess_type == InitialGuessType::Trivial) {
        apply_trivial_initial_guess(state, xnew);
        return;
    }

    if (initial_guess_type == InitialGuessType::Affine) {
        const AffineInitialGuessParams params = compute_affine_initial_guess_params(state, ref_mesh, pins);
        apply_affine_initial_guess(params, state, pins, xnew, dt);
        return;
    }

    if (initial_guess_type == InitialGuessType::CCD) {
        apply_ccd_initial_guess(state, ref_mesh, pins, xnew, dt, eta);
        return;
    }

    throw std::runtime_error("Unknown InitialGuessType");
}
