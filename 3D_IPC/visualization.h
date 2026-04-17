#pragma once
#include "physics.h"
#include "broad_phase.h"
#include <string>
#include <vector>

enum class ExportFormat { OBJ, GEO, PLY, USD };

void export_obj(const std::string& filename, const std::vector<Vec3>& x, const std::vector<int>& tris);
void export_geo(const std::string& filename, const std::vector<Vec3>& x, const std::vector<int>& tris, const std::vector<std::vector<int>>* color_groups = nullptr);
void export_ply(const std::string& filename, const std::vector<Vec3>& x, const std::vector<int>& tris);
void export_usd(const std::string& filename, const std::vector<Vec3>& x, const std::vector<int>& tris);

void export_frame(const std::string& outdir, int frame, const std::vector<Vec3>& x, const std::vector<int>& tris,
                  ExportFormat fmt, const std::vector<std::vector<int>>* color_groups = nullptr);

// Writes all broad phase AABBs (node, triangle, edge) as a wireframe OBJ.
// Three named groups are written: node_boxes, tri_boxes, edge_boxes.
// Load in Houdini and toggle groups to inspect each type independently.
void export_broad_phase_boxes(const std::string& filename, const BroadPhase& bp);

// Writes a flat list of AABBs as a wireframe OBJ.
void export_aabb_list(const std::string& filename, const std::vector<AABB>& boxes);

// Writes all BVH nodes at a given depth as a wireframe OBJ.
// Leaf nodes shallower than depth are included since they cannot split further.
// Returns the number of boxes written (0 means the depth is beyond the tree).
int export_bvh_level(const std::string& filename, const std::vector<BVHNode>& nodes, int root, int depth);
