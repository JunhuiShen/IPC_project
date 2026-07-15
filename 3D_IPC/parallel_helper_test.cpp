#include "parallel_helper.h"
#include "physics.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

namespace {

void build_contact_adj_pair_scan_reference(const BroadPhase::Cache& cache, int nv,
                                           std::vector<std::vector<int>>& out) {
    out.assign(nv, {});
    auto add_clique = [&](const int verts[4]) {
        for (int a = 0; a < 4; ++a) {
            if (verts[a] < 0 || verts[a] >= nv) continue;
            for (int b = a + 1; b < 4; ++b) {
                if (verts[b] < 0 || verts[b] >= nv) continue;
                out[verts[a]].push_back(verts[b]);
                out[verts[b]].push_back(verts[a]);
            }
        }
    };
    for (const auto& p : cache.nt_pairs) {
        const int verts[4] = {p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2]};
        add_clique(verts);
    }
    for (const auto& p : cache.ss_pairs) add_clique(p.v);
    for (auto& row : out) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
}

void populate_contact_incidence(BroadPhase::Cache& cache, int nv) {
    cache.vertex_nt.assign(nv, {});
    cache.vertex_ss.assign(nv, {});
    for (std::size_t i = 0; i < cache.nt_pairs.size(); ++i) {
        const auto& p = cache.nt_pairs[i];
        const int verts[4] = {p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2]};
        for (int role = 0; role < 4; ++role)
            if (verts[role] >= 0 && verts[role] < nv)
                cache.vertex_nt[verts[role]].push_back({i, role});
    }
    for (std::size_t i = 0; i < cache.ss_pairs.size(); ++i) {
        const auto& p = cache.ss_pairs[i];
        for (int role = 0; role < 4; ++role)
            if (p.v[role] >= 0 && p.v[role] < nv)
                cache.vertex_ss[p.v[role]].push_back({i, role});
    }
}

} // namespace

TEST(ParallelHelper, ContactAdjacencyMatchesPairScanExactly) {
    constexpr int nv = 6;
    BroadPhase::Cache cache;

    NodeTrianglePair nt{};
    nt.node = 0;
    nt.tri_v[0] = 1;
    nt.tri_v[1] = 2;
    nt.tri_v[2] = 3;
    cache.nt_pairs.push_back(nt);

    SegmentSegmentPair ss{};
    ss.v[0] = 1;
    ss.v[1] = 2;
    ss.v[2] = 4;
    ss.v[3] = 5;
    cache.ss_pairs.push_back(ss);

    populate_contact_incidence(cache, nv);

    std::vector<std::vector<int>> expected;
    build_contact_adj_pair_scan_reference(cache, nv, expected);

    std::vector<std::vector<int>> actual{{99}};
    build_contact_adj(cache, nv, actual);
    EXPECT_EQ(actual, expected);

    BroadPhase::Cache empty;
    empty.vertex_nt.resize(nv);
    empty.vertex_ss.resize(nv);
    build_contact_adj(empty, nv, actual);
    EXPECT_EQ(actual, std::vector<std::vector<int>>(nv));
}

TEST(GreedyColorConflictGraph, DeterministicColoringAndScratchReuse) {
    const std::vector<std::vector<int>> graph = {
        {1, 2},
        {0, 2},
        {0, 1, 3},
        {2},
        {},
    };

    std::vector<std::vector<int>> groups{{99}, {98}, {97}};
    greedy_color_conflict_graph(graph, groups);
    EXPECT_EQ(groups, (std::vector<std::vector<int>>{{0, 3, 4}, {1}, {2}}));

    greedy_color_conflict_graph({}, groups);
    EXPECT_TRUE(groups.empty());
}
