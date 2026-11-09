#include "two_rigid_color_queue_policy.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace policy = ipc_gpu::solver::two_rigid_color_queue;

namespace {

// Only metadata used by the real helper. Full scene validation, proper graph
// coloring, allocation/release and numerical execution are separate gates.
struct MockScene {
    std::vector<int> positions, com;
    int block_count = 0;
    std::vector<int> body_blocks;
    std::vector<unsigned char> deformable;
    std::vector<int> node_body, node_blocks;
};

std::size_t eligibility_checks = 0, decision_checks = 0;
std::size_t pattern_checks = 0, row_checks = 0, lifecycle_checks = 0;

void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

bool same_scene(const MockScene& a, const MockScene& b) {
    return a.positions == b.positions && a.com == b.com
        && a.block_count == b.block_count && a.body_blocks == b.body_blocks
        && a.deformable == b.deformable && a.node_body == b.node_body
        && a.node_blocks == b.node_blocks;
}

MockScene rigid_scene(std::size_t nodes, bool swap_blocks = false) {
    MockScene h;
    h.positions.assign(nodes, 0);
    h.com.assign(2, 0);
    h.block_count = 2;
    h.body_blocks = swap_blocks ? std::vector<int>{1, 0}
                               : std::vector<int>{0, 1};
    h.deformable.assign(nodes, 0);
    h.node_body.resize(nodes);
    h.node_blocks.resize(nodes);
    for (std::size_t node = 0; node < nodes; ++node) {
        // Neither node order nor an even number of nodes is a premise.
        const int body = static_cast<int>((node * 5 + node / 3) % 2);
        h.node_body[node] = body;
        h.node_blocks[node] = h.body_blocks[static_cast<std::size_t>(body)];
    }
    return h;
}

void check_eligibility(const MockScene& scene, bool expected,
                       const std::string& label) {
    const auto before = scene;
    require(policy::eligible_validated_scene(scene) == expected,
        "eligibility: " + label);
    require(same_scene(scene, before), "eligibility mutated input: " + label);
    ++eligibility_checks;
}

void eligibility_cases() {
    for (std::size_t nodes : {std::size_t(0), std::size_t(1), std::size_t(2),
                              std::size_t(7), std::size_t(1025)})
        for (bool swap : {false, true})
            check_eligibility(rigid_scene(nodes, swap), true,
                "two rigid blocks with arbitrary proxy count/order");

    const auto valid = rigid_scene(7);
    for (std::size_t bodies : {std::size_t(0), std::size_t(1), std::size_t(3)}) {
        auto h = valid;
        h.com.resize(bodies);
        check_eligibility(h, false, "body count differs from two");
    }
    for (int blocks : {-1, 0, 1, 3, INT_MAX}) {
        auto h = valid;
        h.block_count = blocks;
        check_eligibility(h, false, "block count differs from two");
    }
    for (std::size_t count : {std::size_t(0), std::size_t(1), std::size_t(3)}) {
        auto h = valid;
        h.body_blocks.resize(count);
        check_eligibility(h, false, "body block extent differs from two");
    }
    for (int body : {0, 1}) {
        for (int block : {-1, 2, INT_MIN, INT_MAX}) {
            auto h = valid;
            h.body_blocks[static_cast<std::size_t>(body)] = block;
            check_eligibility(h, false, "body block is outside palette");
        }
    }
    for (int block : {0, 1}) {
        auto h = valid;
        h.body_blocks.assign(2, block);
        check_eligibility(h, false, "both bodies share one block");
    }

    for (bool grow : {false, true}) {
        const auto extent = grow ? valid.positions.size() + 1
                                 : valid.positions.size() - 1;
        auto h = valid;
        h.deformable.resize(extent);
        check_eligibility(h, false, "classification extent mismatch");
        h = valid;
        h.node_body.resize(extent);
        check_eligibility(h, false, "owner extent mismatch");
        h = valid;
        h.node_blocks.resize(extent);
        check_eligibility(h, false, "node block extent mismatch");
    }
    for (std::size_t node : {std::size_t(0), std::size_t(3), std::size_t(6)}) {
        for (unsigned char flag : {static_cast<unsigned char>(1),
                                   static_cast<unsigned char>(2),
                                   static_cast<unsigned char>(255)}) {
            auto h = valid;
            h.deformable[node] = flag;
            check_eligibility(h, false, "deformable proxy at first/middle/last");
        }
        for (int owner : {-1, 2, INT_MIN, INT_MAX}) {
            auto h = valid;
            h.node_body[node] = owner;
            check_eligibility(h, false, "invalid owner at first/middle/last");
        }
        for (int block : {-1, 2, INT_MIN, INT_MAX}) {
            auto h = valid;
            h.node_blocks[node] = block;
            check_eligibility(h, false, "invalid block at first/middle/last");
        }
        auto h = valid;
        h.node_blocks[node] = 1 - h.node_blocks[node];
        check_eligibility(h, false, "valid block belongs to the other body");
        h = valid;
        h.node_body[node] = 1 - h.node_body[node];
        check_eligibility(h, false, "owner and stored block disagree");
    }
    auto h = valid;
    h.node_body.assign(h.positions.size(), 1);
    h.node_blocks.assign(h.positions.size(), h.body_blocks[1]);
    check_eligibility(h, true, "one body has no proxy nodes");
}

struct CountCase {
    std::size_t pairs, capacity;
    bool above;
};

void decision_truth_table() {
    const auto maximum = std::numeric_limits<std::size_t>::max();
    const auto imax = static_cast<std::size_t>(INT_MAX);
    const std::array<CountCase, 20> counts{{
        {0, 1, false}, {1, 1, false}, {2, 1, true},
        {65535, 65536, false}, {65536, 65536, false}, {65537, 65536, true},
        {262143, 262144, false}, {262144, 262144, false}, {262145, 262144, true},
        {1048575, 1048576, false}, {1048576, 1048576, false},
        {1048577, 1048576, true},
        {imax - 1, imax, false}, {imax, imax, false}, {imax + 1, imax, true},
        {maximum - 1, maximum, false}, {maximum, maximum, false},
        {maximum, maximum - 1, true}, {0, maximum, false}, {maximum, 1, true}
    }};
    struct Palette {int count; bool two;};
    const std::array<Palette, 6> palettes{{
        {-1, false}, {0, false}, {1, false}, {2, true}, {3, false}, {INT_MAX, false}
    }};
    // Bits: trusted owner, feature enabled, explicit COLOR_PAIR_QUEUES=1.
    // Exactly the trusted + enabled + unforced entry permits omission.
    const std::array<bool, 8> permitted{{false, false, false, true,
                                        false, false, false, false}};
    for (unsigned mask = 0; mask < permitted.size(); ++mask) {
        const bool trusted = (mask & 1u) != 0;
        const bool enabled = (mask & 2u) != 0;
        const bool forced = (mask & 4u) != 0;
        for (const auto& count : counts)
            for (const auto& blocks : palettes)
                for (const auto& colors : palettes) {
                    const bool expected = permitted[mask] && count.above
                        && blocks.two && colors.two;
                    require(policy::omit_color_pairs(trusted, enabled, forced,
                        count.pairs, count.capacity, blocks.count, colors.count)
                            == expected,
                        "omission truth table mismatch");
                    ++decision_checks;
                }
    }
}

// Independent host incidence construction: sort the four vertex colors and
// append each unique color once. This is a set-membership oracle, not a copy
// of the production emit_color_pairs kernel's prior-role duplicate checks.
std::array<std::vector<std::size_t>, 2> color_rows(
    const std::vector<unsigned>& patterns, bool swap_blocks, bool swap_colors) {
    const std::array<int, 2> blocks = swap_blocks ? std::array<int, 2>{{1, 0}}
                                                  : std::array<int, 2>{{0, 1}};
    const std::array<int, 2> colors = swap_colors ? std::array<int, 2>{{1, 0}}
                                                  : std::array<int, 2>{{0, 1}};
    std::array<std::vector<std::size_t>, 2> rows;
    for (std::size_t pair = 0; pair < patterns.size(); ++pair) {
        const unsigned pattern = patterns[pair];
        require(pattern > 0 && pattern < 15,
            "same-block candidate was not rejected before worklist construction");
        std::array<int, 4> role_colors{};
        for (unsigned role = 0; role < 4; ++role) {
            const unsigned body = (pattern >> role) & 1u;
            role_colors[role] = colors[static_cast<std::size_t>(blocks[body])];
        }
        std::sort(role_colors.begin(), role_colors.end());
        const auto end = std::unique(role_colors.begin(), role_colors.end());
        require(end - role_colors.begin() == 2,
            "cross-block candidate did not cover both proper colors");
        for (auto color = role_colors.begin(); color != end; ++color)
            rows[static_cast<std::size_t>(*color)].push_back(pair);
    }
    return rows;
}

void cross_block_patterns() {
    for (bool swap_blocks : {false, true})
        for (bool swap_colors : {false, true}) {
            std::vector<unsigned> all_cross;
            for (unsigned pattern = 0; pattern < 16; ++pattern) {
                unsigned first = pattern & 1u;
                bool all_same = true;
                for (unsigned role = 1; role < 4; ++role)
                    all_same = all_same && (((pattern >> role) & 1u) == first);
                const bool broadphase_keeps = !all_same;
                require(broadphase_keeps == (pattern != 0 && pattern != 15),
                    "independent same-block rejection differs");
                if (broadphase_keeps) all_cross.push_back(pattern);
                ++pattern_checks;
            }
            require(all_cross.size() == 14, "four-role cross-pattern count differs");
            // These four are the actual homogeneous-primitive rigid NT/EE
            // patterns: PT 0001/1110, EE 0011/1100. The 14-pattern set above
            // proves the two-color property for the stronger general case.
            const std::vector<unsigned> physical{1, 14, 3, 12};
            for (const auto& palette : {all_cross, physical})
                for (std::size_t count : {std::size_t(1), std::size_t(2),
                        std::size_t(3), std::size_t(4), std::size_t(127),
                        std::size_t(128), std::size_t(129), std::size_t(257)}) {
                    std::vector<unsigned> patterns;
                    for (std::size_t pair = 0; pair < count; ++pair)
                        patterns.push_back(palette[(pair * 5 + pair / 7) % palette.size()]);
                    const auto rows = color_rows(patterns, swap_blocks, swap_colors);
                    for (const auto& row : rows) {
                        require(row.size() == count,
                            "proper color row does not contain all global candidates");
                        for (std::size_t pair = 0; pair < count; ++pair)
                            require(row[pair] == pair,
                                "color row duplicated, omitted or reordered a candidate");
                    }
                    const std::size_t capacity = count > 1 ? count - 1 : 1;
                    require(policy::omit_color_pairs(true, true, false,
                        count, capacity, 2, 2) == (count > 1),
                        "per-color all-N premise disagrees with cap decision");
                    ++row_checks;
                }
        }
}

void lifecycle_decisions() {
    // Decision-level large/small/empty transitions only. Actual retained
    // buffers, publication generations and releases need the native witness.
    const std::array<std::size_t, 9> pairs{{4, 3, 0, 1, 4, 2, 5, 0, 4}};
    const std::array<bool, 9> expected{{true, false, false, false, true,
                                       false, true, false, true}};
    for (int repeat = 0; repeat < 3; ++repeat)
        for (std::size_t epoch = 0; epoch < pairs.size(); ++epoch) {
            require(policy::omit_color_pairs(true, true, false,
                pairs[epoch], 3, 2, 2) == expected[epoch],
                "large-small-large omission decision retained stale state");
            require(!policy::omit_color_pairs(true, false, false,
                pairs[epoch], 3, 2, 2), "disabled feature changed policy");
            require(!policy::omit_color_pairs(false, true, false,
                pairs[epoch], 3, 2, 2), "raw pipeline received trusted behavior");
            require(!policy::omit_color_pairs(true, true, true,
                pairs[epoch], 3, 2, 2), "explicit color queues lost precedence");
            ++lifecycle_checks;
        }
}

} // namespace

int main() {
    try {
        eligibility_cases();
        decision_truth_table();
        cross_block_patterns();
        lifecycle_decisions();
        std::cout << "Two-rigid color queue host policy PASS: eligibility="
                  << eligibility_checks << " decisions=" << decision_checks
                  << " four-role-patterns=" << pattern_checks
                  << " ordered-color-rows=" << row_checks
                  << " lifecycle-epochs=" << lifecycle_checks << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
