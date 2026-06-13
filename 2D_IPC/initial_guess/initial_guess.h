#pragma once

#include "../ipc_math.h"
#include "../mesh.h"
#include "../state.h"

// ======================================================
// Unified initial guess wrapper over global state and explicit edges.
// ======================================================

enum class InitialGuessType {
    Trivial,
    Affine,
    CCD
};

void apply_initial_guess(InitialGuessType initial_guess_type, const State2D& state,
                         const RefMesh& ref_mesh, Vec& xnew, double dt, double eta);
