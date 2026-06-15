#pragma once

#include "../ipc_math.h"
#include "../physics.h"

void apply_initial_guess(InitialGuessType initial_guess_type, const DeformedState& state,
                         const RefMesh& ref_mesh, const std::vector<Pin>& pins,
                         Vec& xnew, double dt, double eta);
