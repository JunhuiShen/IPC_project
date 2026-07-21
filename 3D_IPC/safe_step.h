#pragma once

#include "IPC_math.h"

#include <atomic>
#include <functional>
#include <vector>

class BroadPhase;

// Returns gamma_p times the minimum incident NT/SS distance.
double compute_trust_region_bound_for_vertex(int vi, const std::vector<Vec3>& x, const BroadPhase& broad_phase, double gamma_p);

// Clips targets to cached boxes, then limits x += alpha*dx by CCD or the OGC bound.
void per_vertex_safe_step(const BroadPhase& broad_phase, std::vector<Vec3>& x, const std::function<Vec3(int)>& x_new_fn, double safety = 0.9, bool clip_ccd = true, bool use_ticcd = true, bool use_ogc = false, const std::vector<std::vector<int>>* color_groups = nullptr, std::atomic<int>* clip_count = nullptr);
