#include "rigid_proxy_index.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace proxy = ipc_gpu::solver::proxy_index;

namespace {
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

constexpr bool constexpr_tail() {
    const auto plan = proxy::make_plan(1);
    auto cursor = proxy::row_cursor(7, 7 + 16385, 0, 0, plan);
    if (!cursor.valid() || cursor.slot() != 7) return false;
    cursor.advance();
    if (!cursor.valid() || cursor.slot() != 7 + 16384) return false;
    cursor.advance();
    return !cursor.valid() && cursor.slot() == proxy::invalid_index;
}
static_assert(proxy::width == 128);
static_assert(proxy::make_plan(0).blocks == 0);
static_assert(proxy::make_plan(64).tiles_per_body == 128);
static_assert(proxy::make_plan(65).tiles_per_body == 1);
static_assert(constexpr_tail());

// The oracle walks ordinary serial CSR rows. The candidate traversal uses
// flat blocks, body batches, tiles, lanes and cursors; compare every slot.
void enumerate(const std::vector<std::size_t>& counts, std::size_t base = 0) {
    const std::size_t bodies = counts.size();
    const auto plan = proxy::make_plan(bodies);
    require(plan.blocks <= 65535, "launch exceeds bounded grid");
    require(plan.blocks == plan.tiles_per_body * plan.body_batches,
        "plan block product differs");
    require(plan.body_batches <= bodies, "body batches exceed body count");
    std::vector<std::size_t> offsets(bodies + 1, base);
    for (std::size_t body = 0; body < bodies; ++body) {
        require(counts[body] <= proxy::invalid_index - offsets[body],
            "test fixture offset overflow");
        offsets[body + 1] = offsets[body] + counts[body];
    }
    const auto extent = offsets.back() - base;
    // Reverse order and gaps make global node IDs unrelated to CSR slots.
    std::vector<std::uint64_t> node_ids(extent);
    for (std::size_t i = 0; i < extent; ++i)
        node_ids[i] = std::uint64_t(extent - 1 - i) * 7 + 23;
    std::vector<unsigned> visits(extent, 0), body_tiles(bodies, 0);
    std::vector<std::uint64_t> observed_nodes, expected_nodes;
    for (std::size_t body = 0; body < bodies; ++body)
        for (std::size_t slot = offsets[body]; slot < offsets[body + 1]; ++slot)
            expected_nodes.push_back(node_ids[slot - base]);

    for (unsigned block = 0; block < plan.blocks; ++block) {
        const auto body_tile = proxy::tile(block, plan);
        require(body_tile < plan.tiles_per_body, "valid block has invalid tile");
        for (auto body = proxy::first_body(block, plan); body < bodies;
             body = proxy::next_body(body, bodies, plan)) {
            ++body_tiles[body];
            if (offsets[body] == offsets[body + 1]) {
                for (const unsigned lane : {0u, 127u})
                    require(!proxy::row_cursor(offsets[body], offsets[body + 1],
                        body_tile, lane, plan).valid(), "empty row visited");
                continue;
            }
            for (unsigned lane = 0; lane < proxy::width; ++lane) {
                auto cursor = proxy::row_cursor(offsets[body], offsets[body + 1],
                    body_tile, lane, plan);
                std::size_t iterations = 0;
                for (; cursor.valid(); cursor.advance()) {
                    const auto slot = cursor.slot();
                    require(slot >= offsets[body] && slot < offsets[body + 1],
                        "slot escaped its body's CSR row");
                    require(slot >= base && slot - base < extent,
                        "slot escaped full CSR extent");
                    ++visits[slot - base];
                    observed_nodes.push_back(node_ids[slot - base]);
                    require(++iterations <= counts[body], "cursor did not terminate");
                }
                require(cursor.slot() == proxy::invalid_index,
                    "finished cursor exposed a valid-looking slot");
            }
        }
    }
    require(std::all_of(visits.begin(), visits.end(), [](unsigned n) { return n == 1; }),
        "CSR slots were skipped or visited more than once");
    require(std::all_of(body_tiles.begin(), body_tiles.end(),
        [plan](unsigned n) { return n == plan.tiles_per_body; }),
        "a body did not receive exactly one of every tile");
    std::sort(expected_nodes.begin(), expected_nodes.end());
    std::sort(observed_nodes.begin(), observed_nodes.end());
    require(expected_nodes == observed_nodes, "noncontiguous global node coverage differs");
    require(proxy::first_body(plan.blocks, plan) == proxy::invalid_index,
        "out-of-grid block exposed a body");
}

void enumerate_near_max(std::size_t count) {
    const auto maximum = proxy::invalid_index;
    const auto first = maximum - count;
    const auto plan = proxy::make_plan(1);
    std::vector<unsigned> visited(count, 0);
    for (unsigned tile = 0; tile < plan.tiles_per_body; ++tile)
        for (unsigned lane = 0; lane < proxy::width; ++lane) {
            auto cursor = proxy::row_cursor(first, maximum, tile, lane, plan);
            for (; cursor.valid(); cursor.advance()) {
                require(cursor.slot() >= first && cursor.slot() < maximum,
                    "near-SIZE_MAX slot wrapped or aliased end sentinel");
                ++visited[cursor.slot() - first];
            }
        }
    require(std::all_of(visited.begin(), visited.end(), [](unsigned n) { return n == 1; }),
        "near-SIZE_MAX row coverage differs");
}

// Independent checked-add oracle: implementation instead compares the stride
// with remaining extent before adding. Exercise huge states without iterating
// over huge synthetic rows/body counts.
std::size_t checked_next(std::size_t current, std::size_t end, std::size_t stride) {
    if (current >= end || stride == 0) return end;
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (current > maximum - stride) return end;
    const auto sum = current + stride;
    return sum < end ? sum : end;
}

void boundaries() {
    const auto maximum = proxy::invalid_index;
    const auto plan = proxy::make_plan(maximum);
    require(plan.tiles_per_body == 1 && plan.body_batches == 65535
        && plan.blocks == 65535, "SIZE_MAX body plan differs");
    for (const std::size_t bodies : {std::size_t(0), std::size_t(1),
         std::size_t(64), std::size_t(65), std::size_t(65535), maximum}) {
        const auto candidate = proxy::make_plan(bodies);
        for (const auto body : {std::size_t(0), std::size_t(1),
             maximum - 65536, maximum - 65535, maximum - 1, maximum})
            require(proxy::next_body(body, bodies, candidate)
                == checked_next(body, bodies, candidate.body_batches),
                "body advancement differs from checked-add oracle");
    }
    for (const auto first : {std::size_t(0), std::size_t(9), maximum - 1, maximum})
        for (const auto last : {std::size_t(0), std::size_t(9), maximum - 1, maximum})
            if (last <= first) {
                auto cursor = proxy::row_cursor(first, last, 0, 0, plan);
                require(!cursor.valid(), "reversed/empty legacy row visited");
                cursor.advance();
                require(!cursor.valid(), "invalid cursor advanced into row");
            }
    require(!proxy::row_cursor(0, 1, 0, 128, plan).valid(), "invalid lane accepted");
    require(!proxy::row_cursor(0, 1, 1, 0, plan).valid(), "invalid tile accepted");
    require(!proxy::row_cursor(0, 1, 0, 0, {}).valid(), "zero-stride plan accepted");
    require(proxy::first_body(0, {}) == maximum, "empty plan exposed body");
    require(proxy::first_body(std::numeric_limits<unsigned>::max(), plan) == maximum,
        "huge out-of-grid block exposed body");
    proxy::RowCursor malformed{maximum, 1, 2, 1};
    require(!malformed.valid() && malformed.slot() == maximum,
        "malformed cursor extent overflow accepted");

    std::uint64_t random = 0xd3c61b7e95284a0full;
    const auto next_random = [&]() {
        random = random * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<std::size_t>(random);
    };
    for (unsigned sample = 0; sample < 10000; ++sample) {
        const auto count = next_random();
        const auto relative = count ? next_random() % count : 0;
        const auto stride = next_random();
        const auto expected = checked_next(relative, count, stride);
        proxy::RowCursor cursor{0, relative, count, stride};
        cursor.advance();
        require(cursor.valid() ? cursor.slot() == expected : expected == count,
            "large cursor advancement differs from checked-add oracle");
    }
}
} // namespace

int main() {
    try {
        enumerate({});
        enumerate({0});
        enumerate({0, 3, 0, 1, 0, 257, 0}, 11);
        for (const std::size_t count : {127u, 128u, 129u, 16383u, 16384u, 16385u}) {
            enumerate({count}, 7);
            enumerate_near_max(count);
        }
        for (const std::size_t bodies : {63u, 64u, 65u, 127u, 65535u, 65536u, 70003u}) {
            std::vector<std::size_t> counts(bodies, 0);
            for (std::size_t body = 0; body < bodies; ++body)
                if (body % 17 == 0) counts[body] = 1 + body % 7;
            if (bodies <= 127) counts[bodies / 2] = 16385;
            enumerate(counts, 19);
        }
        boundaries();
        std::cout << "Rigid proxy index host oracle PASS: exact CSR/node coverage, "
            "empty/reversed rows, 64/65 and 65535 boundaries, tails, SIZE_MAX and overflow guards\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Rigid proxy index host oracle failed: " << error.what() << '\n';
        return 1;
    }
}
