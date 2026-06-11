#pragma once

#include "chain.h"
#include <string>
#include <vector>

bool write_checkpoint(const std::string& filename, const std::vector<Chain>& chains);
bool read_checkpoint(const std::string& filename, std::vector<Chain>& chains);
