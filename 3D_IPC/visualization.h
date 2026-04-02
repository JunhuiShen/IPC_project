#pragma once
#include "physics.h"
#include <string>
#include <vector>

void export_obj(const std::string& filename, const std::vector<Vec3>& x, const std::vector<int>& tris);

void export_frame(const std::string& outdir, int frame, const std::vector<Vec3>& x, const std::vector<int>& tris);
