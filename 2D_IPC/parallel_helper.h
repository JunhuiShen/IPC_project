#pragma once

#include "broad_phase.h"

#include <utility>
#include <vector>

struct RedBoxes {
    std::vector<AABB> segment;
};

struct GreenBoxes {
    std::vector<AABB> segment;
};

AABB arc_node_aabb(const Vec2& x_com, double theta, const Vec2& X, double eps);

double box_safe_step(const AABB& box, const Vec2& x0, const Vec2& displacement);

void build_blue_boxes(const Vec& positions, const std::vector<double>& node_radii, std::vector<AABB>& blue_boxes);

void build_blue_boxes_rb(const Vec& positions, const Vec& x_coms, const std::vector<double>& thetas, const std::vector<double>& theta_radii, 
    const std::vector<double>& com_radii, const std::vector<std::vector<int>>& rb_nodes, const std::vector<Vec>& ref_positions, std::vector<AABB>& blue_boxes);

void build_red_boxes(const std::vector<std::pair<int, int>>& edges, const std::vector<AABB>& blue_boxes, RedBoxes& red_boxes);

void build_green_boxes(const RedBoxes& red_boxes, double d_hat, GreenBoxes& green_boxes);

BroadPhase::Cache register_barrier_pairs_from_blue_and_green(const std::vector<std::pair<int, int>>& edges, const std::vector<AABB>& blue_boxes, const GreenBoxes& green_boxes);

std::vector<std::vector<int>> build_elastic_adj(const std::vector<std::pair<int, int>>& edges, int total_nodes);

std::vector<std::vector<int>> build_contact_adj(const std::vector<NodeSegmentPair>& contact_pairs, int total_nodes);

std::vector<std::vector<int>> union_adjacency(const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b);

std::vector<std::vector<int>> greedy_color_conflict_graph(const std::vector<std::vector<int>>& graph);
