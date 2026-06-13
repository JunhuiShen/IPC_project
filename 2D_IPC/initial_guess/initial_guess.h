#pragma once

#include "../ipc_math.h"
#include "../mesh.h"
#include "../state.h"

// ======================================================
// Unified initial guess wrapper over global state and explicit edges.
// ======================================================

namespace initial_guess {

    enum class Type {
        Trivial,
        Affine,
        CCD
    };

    void apply(Type initial_guess_type,
               const State2D& state, const RefMesh& ref_mesh,
               Vec& xnew,
               double dt, double eta);

} // namespace initial_guess
