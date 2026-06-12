#pragma once

#include "state.h"
#include <string>

bool write_checkpoint(const std::string& filename, const State2D& state);
bool read_checkpoint(const std::string& filename, State2D& state);
