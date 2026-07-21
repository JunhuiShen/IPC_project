#pragma once

#include "physics.h"

#include <string>

void serialize_state(const std::string& dir, int frame, const DeformedState& state);

bool deserialize_state(const std::string& dir, int frame, DeformedState& state);
