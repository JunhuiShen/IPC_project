#pragma once

#include "ipc_math.h"
#include <string>
#include <vector>
#include <utility>

// ======================================================
// Visualization utilities
//
// Export simulation state as Wavefront .obj or Houdini
// JSON .geo files.
// Each frame becomes one file: outdir/frame_XXXX.obj
//                           or outdir/frame_XXXX.geo
// ======================================================

enum class OutputFormat { OBJ, GEO };

void export_obj(const std::string& filename,
                const Vec& x,
                const std::vector<std::pair<int,int>>& edges);

void export_geo(const std::string& filename,
                const Vec& x,
                const std::vector<std::pair<int,int>>& edges);

void export_frame(const std::string& outdir,
                  int frame,
                  const Vec& x,
                  const std::vector<std::pair<int,int>>& edges,
                  OutputFormat format = OutputFormat::OBJ);

void export_substep_frame(const std::string& outdir,
                          int substep,
                          const Vec& x,
                          const std::vector<std::pair<int,int>>& edges,
                          OutputFormat format = OutputFormat::OBJ);
