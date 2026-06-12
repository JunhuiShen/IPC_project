#pragma once

#include "broad_phase/broad_phase.h"

#include <utility>
#include <vector>

std::vector<std::vector<int>>
build_conflict_graph(const std::vector<std::pair<int, int>>& edges,
                     const std::vector<contact::NodeSegmentPair>& contact_pairs,
                     int total_nodes);

std::vector<std::vector<int>>
greedy_color_conflict_graph(const std::vector<std::vector<int>>& graph);
