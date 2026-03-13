#pragma once

#include "ipc_math.h"
#include <string>
#include <vector>
#include <utility>

// ======================================================
// Visualization utilities
//
// Export simulation state as Wavefront .obj files.
// Each frame becomes one file: outdir/frame_XXXX.obj
// ======================================================

void export_obj(const std::string& filename,
                const Vec& x,
                const std::vector<std::pair<int,int>>& edges);

void export_frame(const std::string& outdir,
                  int frame,
                  const Vec& x_combined,
                  const std::vector<std::pair<int,int>>& edges_combined);
