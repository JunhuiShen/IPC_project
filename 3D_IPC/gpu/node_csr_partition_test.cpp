#include "node_csr_partition.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace partition = ipc_gpu::solver::node_csr_partition;

namespace {
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
std::uint64_t random_state = 0x4a13de67208bf95cull;
std::uint64_t random_value() {
    random_state = random_state * 6364136223846793005ull + 1442695040888963407ull;
    return random_state;
}

constexpr auto split = partition::make_plan(2, 7, 6, 4, 3, true);
static_assert(split.first_row == 0 && split.row_stride == 2
    && split.first_rank == 15 && split.rank_stride == 16);
constexpr auto shorter = partition::make_plan(2, 7, 5, 4, 3, true);
static_assert(shorter.first_row == 1 && shorter.first_rank == 11 && shorter.rank_stride == 12);
constexpr auto unchanged = partition::make_plan(7, 7, 6, 4, 3, true);
static_assert(unchanged.first_row == 6 && unchanged.row_stride == 7
    && unchanged.first_rank == 3 && unchanged.rank_stride == 4);

// Serial row/rank enumeration is the independent coverage oracle. No rank
// formula, remainder, tile count or scheduler stride is used by the oracle.
void coverage(std::size_t active, std::size_t grid, std::size_t threads,
    bool enabled, std::size_t seed, std::size_t uniform_count = partition::maximum_index) {
    std::vector<std::size_t> counts(active), offsets(active + 1, 0);
    for (std::size_t row = 0; row < active; ++row) {
        counts[row] = uniform_count == partition::maximum_index
            ? (row * 7 + seed) % (2 * threads + 5) : uniform_count;
        offsets[row + 1] = offsets[row] + counts[row];
    }
    std::vector<unsigned> visits(offsets.back(), 0);
    if (!grid || !threads || !active) {
        const auto invalid = partition::make_plan(active, grid, 0, threads, 0, enabled);
        require(invalid.first_row == active && !invalid.row_stride && !invalid.rank_stride,
            "zero active/grid/threads did not produce an empty plan");
        return;
    }
    for (std::size_t block = 0; block < grid; ++block)
        for (std::size_t lane = 0; lane < threads; ++lane) {
            const auto plan = partition::make_plan(active, grid, block, threads, lane, enabled);
            std::size_t row_steps = 0;
            for (auto row = plan.first_row; row < active;
                 row = partition::next_row(row, active, plan)) {
                require(++row_steps <= active, "row traversal did not terminate");
                std::size_t rank_steps = 0;
                for (auto rank = plan.first_rank; rank < counts[row];
                     rank = partition::next_rank(rank, counts[row], plan)) {
                    require(++rank_steps <= counts[row], "rank traversal did not terminate");
                    ++visits[offsets[row] + rank];
                }
            }
        }
    for (std::size_t row = 0; row < active; ++row)
        for (std::size_t rank = 0; rank < counts[row]; ++rank)
            require(visits[offsets[row] + rank] == 1,
                "serial (row,rank) oracle found a missing or duplicated job");
    const auto bad_block = partition::make_plan(active, grid, grid, threads, 0, enabled);
    const auto bad_lane = partition::make_plan(active, grid, 0, threads, threads, enabled);
    require(bad_block.first_row == active && !bad_block.rank_stride,
        "out-of-grid block produced work");
    require(bad_lane.first_row == active && !bad_lane.rank_stride,
        "out-of-block lane produced work");
}

std::size_t checked_add_next(std::size_t current, std::size_t end, std::size_t stride) {
    if (!stride || current >= end) return end;
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (current > maximum - stride) return end;
    const auto sum = current + stride;
    return sum < end ? sum : end;
}

void boundaries() {
    const auto maximum = partition::maximum_index;
    const auto huge = partition::make_plan(1, maximum, maximum - 1, 1, 0, true);
    require(huge.first_row == 0 && huge.row_stride == 1
        && huge.first_rank == maximum - 1 && huge.rank_stride == maximum,
        "representable SIZE_MAX split plan differs");
    require(partition::next_rank(huge.first_rank, maximum, huge) == maximum,
        "SIZE_MAX split rank did not stop safely");
    const auto huge_rows = partition::make_plan(maximum, maximum, maximum - 1, 1, 0, true);
    require(huge_rows.first_row == maximum - 1
        && partition::next_row(huge_rows.first_row, maximum, huge_rows) == maximum,
        "SIZE_MAX unsplit row did not stop safely");
    const auto huge_lanes = partition::make_plan(1, 1, 0, maximum, maximum - 1, true);
    require(huge_lanes.first_rank == maximum - 1 && huge_lanes.rank_stride == maximum,
        "SIZE_MAX thread/lane plan differs");
    for (const auto threads : {std::size_t(2), maximum}) {
        const auto overflow = partition::make_plan(1, maximum, maximum - 1, threads, 0, true);
        require(overflow.first_row == 1 && !overflow.row_stride && !overflow.rank_stride,
            "unrepresentable rank stride did not fail closed");
    }
    for (unsigned sample = 0; sample < 20000; ++sample) {
        const auto end = static_cast<std::size_t>(random_value());
        const auto current = static_cast<std::size_t>(random_value());
        const auto stride = static_cast<std::size_t>(random_value());
        const auto expected = checked_add_next(current, end, stride);
        const partition::Plan plan{0, stride, 0, stride};
        require(partition::next_row(current, end, plan) == expected,
            "large next_row differs from independent checked-add oracle");
        require(partition::next_rank(current, end, plan) == expected,
            "large next_rank differs from independent checked-add oracle");
    }

}
} // namespace

int main() {
    try {
        for (std::size_t active = 0; active <= 12; ++active)
            for (std::size_t grid = 0; grid <= 17; ++grid)
                for (std::size_t threads = 1; threads <= 8; ++threads)
                    for (const bool enabled : {false, true})
                        coverage(active, grid, threads, enabled, active + grid + threads);
        for (const std::size_t grid : {1u, 2u, 3u, 7u, 15u, 31u, 33u, 65u, 127u})
            for (const std::size_t active : {std::size_t(0), std::size_t(1), grid - 1,
                 grid, grid + 1, 3 * grid + 1})
                for (const std::size_t threads : {1u, 3u, 7u, 32u, 128u})
                    for (const bool enabled : {false, true}) {
                        coverage(active, grid, threads, enabled, 17);
                        for (const auto count : {threads - 1, threads, threads + 1,
                             2 * threads - 1, 2 * threads + 1})
                            coverage(active, grid, threads, enabled, 0, count);
                    }
        for (unsigned sample = 0; sample < 1000; ++sample)
            coverage(random_value() % 41, 1 + random_value() % 47,
                1 + random_value() % 33, (random_value() & 1) != 0, random_value() % 97);
        for (const std::size_t grid : {127u, 128u, 129u, 257u})
            for (const std::size_t count : {0u, 1u, 127u, 128u, 129u,
                 16383u, 16384u, 16385u, 32769u, 65537u})
                for (const bool enabled : {false, true})
                    coverage(1, grid, 128, enabled, 0, count);
        for (const std::size_t grid : {1023u,1024u})
            for (const std::size_t count : {131071u,131072u,131073u})
                for (const bool enabled : {false,true}) {
                    coverage(1,grid,128,enabled,0,count);
                    // Seven uneven rows receive146 or147 CTAs at grid1024;
                    // use long rows as well as varying short/empty counts.
                    coverage(7,grid,128,enabled,0,count);
                    coverage(7,grid,128,enabled,31);
                }
        for (const std::size_t active : {15u,16u,17u,1023u,1024u,1025u})
            for (const bool enabled : {false,true})
                coverage(active,1024,128,enabled,53);
        coverage(3, 5, 0, true, 0);
        boundaries();
        std::cout << "Node CSR partition host oracle PASS: exhaustive/random exact coverage, "
            "launch/rank boundaries through grid1024/rank131073, uneven rows and SIZE_MAX guards\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Node CSR partition host oracle failed: " << error.what() << '\n';
        return 1;
    }
}
