#pragma once

#include "physics.h"
#include "solver.h"
#include <utility>
#include <vector>

std::vector<std::vector<int>>
build_conflict_graph(const std::vector<BlockView>& blocks,
                     const std::vector<physics::NodeSegmentPair>& barrier_pairs,
                     int total_nodes);

std::vector<std::vector<int>>
greedy_color_conflict_graph(const std::vector<std::vector<int>>& graph);

std::vector<std::pair<int, int>>
build_global_to_block_local(const std::vector<BlockView>& blocks, int total_nodes);

