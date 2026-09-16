#include "grid_coloring.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <omp.h>
#include <stdexcept>

namespace {

using solver_detail::ClothGridSchedule;
using solver_detail::GridCell;

std::vector<AABB> boxes(const std::vector<Vec3>& positions, double radius = 0.1) {
    std::vector<AABB> result;
    for (const auto& p : positions)
        result.emplace_back(p - Vec3::Constant(radius), p + Vec3::Constant(radius));
    return result;
}

const GridCell& cell_for_vertex(const ClothGridSchedule& schedule, int vertex) {
    for (const auto& cell : schedule.cells)
        if (std::find(cell.vertices.begin(), cell.vertices.end(), vertex) != cell.vertices.end())
            return cell;
    throw std::runtime_error("Test vertex missing from schedule");
}

void expect_one_batch_per_parity(const ClothGridSchedule& schedule,
    const std::vector<AABB>& node_boxes,
    const std::vector<std::vector<int>>& dependencies) {
    std::array<int, 8> batch_for_color;
    batch_for_color.fill(-1);
    ASSERT_LE(schedule.batches.size(), 8u);
    std::vector<int> occurrences(node_boxes.size(), 0);
    for (std::size_t batch = 0; batch < schedule.batches.size(); ++batch) {
        ASSERT_FALSE(schedule.batches[batch].empty());
        const int color = schedule.cells[schedule.batches[batch].front()].color_id;
        ASSERT_GE(color, 0);
        ASSERT_LT(color, 8);
        EXPECT_EQ(batch_for_color[color], -1) << "repeated parity=" << color;
        batch_for_color[color] = static_cast<int>(batch);
        for (int cell_index : schedule.batches[batch]) {
            const auto& cell = schedule.cells[cell_index];
            EXPECT_EQ(cell.color_id, color);
            EXPECT_EQ(cell.batch_id, static_cast<int>(batch));
            EXPECT_TRUE(std::is_sorted(cell.vertices.begin(), cell.vertices.end()));
            for (int vertex : cell.vertices) ++occurrences[vertex];
        }
    }
    for (int count : occurrences) EXPECT_EQ(count, 1);
    for (std::size_t vertex = 0; vertex < dependencies.size(); ++vertex)
        for (int other : dependencies[vertex]) {
            const auto& a = cell_for_vertex(schedule, static_cast<int>(vertex));
            const auto& b = cell_for_vertex(schedule, other);
            if (a.index != b.index) EXPECT_NE(a.color_id, b.color_id);
        }
    for (std::size_t a = 0; a < schedule.cells.size(); ++a)
        for (std::size_t b = a + 1; b < schedule.cells.size(); ++b) {
            if (schedule.cells[a].color_id != schedule.cells[b].color_id) continue;
            for (int first : schedule.cells[a].vertices)
                for (int second : schedule.cells[b].vertices)
                    EXPECT_FALSE(aabb_intersects(node_boxes[first], node_boxes[second]));
        }
}

struct RestoreOpenMP {
    int threads = omp_get_max_threads();
    int dynamic = omp_get_dynamic();
    ~RestoreOpenMP() {
        omp_set_num_threads(threads);
        omp_set_dynamic(dynamic);
    }
};

} // namespace

TEST(ClothGridSchedule, FourByFourXYCheckerboardAndSortedOwnership) {
    std::vector<Vec3> positions;
    // Deliberately reverse the input cell order; ownership must still sort by
    // grid index, while vertices within a cell keep ascending vertex IDs.
    for (int y = 3; y >= 0; --y)
        for (int x = 3; x >= 0; --x) {
            positions.emplace_back(x + 0.25, y + 0.25, 0.25);
            positions.emplace_back(x + 0.75, y + 0.75, 0.25);
        }
    ClothGridSchedule schedule;
    schedule.build(positions, boxes(positions), {}, 1.0);
    ASSERT_EQ(schedule.cells.size(), 16u);
    ASSERT_EQ(schedule.batches.size(), 4u);
    EXPECT_EQ(schedule.min_index, (std::array<std::int64_t, 3>{{0, 0, 0}}));
    EXPECT_EQ(schedule.max_index, (std::array<std::int64_t, 3>{{3, 3, 0}}));
    for (std::size_t i = 0; i < schedule.cells.size(); ++i) {
        const auto& cell = schedule.cells[i];
        if (i) EXPECT_LT(schedule.cells[i - 1].index, cell.index);
        EXPECT_TRUE(std::is_sorted(cell.vertices.begin(), cell.vertices.end()));
        EXPECT_EQ(cell.color_id, (cell.index[0] % 2) + 2 * (cell.index[1] % 2));
        EXPECT_EQ(cell.batch_id, cell.color_id);
        EXPECT_EQ(cell.bounds.min[0], cell.index[0]);
        EXPECT_EQ(cell.bounds.max[0], cell.index[0] + 1);
    }
    for (int color = 0; color < 4; ++color) {
        EXPECT_EQ(schedule.batches[color].size(), 4u);
        EXPECT_EQ(schedule.vertex_color_groups[color].size(), 8u);
    }
}

TEST(ClothGridSchedule, EightColorsInThreeDimensions) {
    std::vector<Vec3> positions;
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x)
                positions.emplace_back(x + 0.5, y + 0.5, z + 0.5);
    ClothGridSchedule schedule;
    schedule.build(positions, boxes(positions), {}, 1.0);
    ASSERT_EQ(schedule.batches.size(), 8u);
    for (int vertex = 0; vertex < 8; ++vertex) {
        EXPECT_EQ(cell_for_vertex(schedule, vertex).color_id, vertex);
        EXPECT_EQ(schedule.vertex_color_groups[vertex], std::vector<int>{vertex});
    }
}

TEST(ClothGridSchedule, NegativeCoordinatesUseFloorAndStableWorldOrigin) {
    const std::vector<Vec3> positions = {
        Vec3(-0.1, -0.1, -0.1), Vec3(-1.0, -2.0, -3.0),
        Vec3(-1.1, 0.5, 0.5), Vec3(0.1, 0.5, 0.5)
    };
    ClothGridSchedule schedule;
    schedule.build(positions, boxes(positions, 0.05), {}, 1.0);
    EXPECT_EQ(cell_for_vertex(schedule, 0).index, (std::array<std::int64_t, 3>{{-1, -1, -1}}));
    EXPECT_EQ(cell_for_vertex(schedule, 0).color_id, 7);
    EXPECT_EQ(cell_for_vertex(schedule, 1).index, (std::array<std::int64_t, 3>{{-1, -2, -3}}));
    EXPECT_EQ(cell_for_vertex(schedule, 1).color_id, 5);
    EXPECT_EQ(cell_for_vertex(schedule, 2).index[0], -2);
    EXPECT_EQ(cell_for_vertex(schedule, 2).color_id, 0);
    EXPECT_EQ(cell_for_vertex(schedule, 3).index[0], 0);
    EXPECT_EQ(cell_for_vertex(schedule, 3).color_id, 0);
}

TEST(ClothGridSchedule, SameParityNodeBoxesCannotTouchNearCellBoundaries) {
    const std::vector<Vec3> positions = {
        Vec3(0.999999999, 0.01, 0.01), Vec3(2.0, 0.01, 0.01),
        Vec3(0.01, 2.0, 0.01), Vec3(0.01, 0.01, 2.0),
        Vec3(-1.000000001, 0.01, 0.01)
    };
    const auto node_boxes = boxes(positions, 0.499999999);
    ClothGridSchedule schedule;
    schedule.build(positions, node_boxes, {}, 1.0);
    ASSERT_EQ(schedule.batches.size(), 1u);
    for (std::size_t a = 0; a < node_boxes.size(); ++a)
        for (std::size_t b = a + 1; b < node_boxes.size(); ++b)
            EXPECT_FALSE(aabb_intersects(node_boxes[a], node_boxes[b])) << a << ", " << b;
    // The domain includes empty cells reached by the node boxes.
    EXPECT_EQ(schedule.min_index, (std::array<std::int64_t, 3>{{-2, -1, -1}}));
    EXPECT_EQ(schedule.max_index, (std::array<std::int64_t, 3>{{2, 2, 2}}));
}

TEST(ClothGridSchedule, RejectsInvalidDxNodeBoxesAndUnrepresentableGridRanges) {
    const std::vector<Vec3> positions{Vec3(0.0, 0.0, 0.0)};
    const auto node_boxes = boxes(positions);
    ClothGridSchedule schedule;
    for (double dx : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN()})
        EXPECT_THROW(schedule.build(positions, node_boxes, {}, dx), std::invalid_argument);
    EXPECT_THROW(schedule.build(positions, {}, {}, 1.0), std::invalid_argument);
    EXPECT_THROW(schedule.build(positions, boxes(positions, 0.5), {}, 1.0), std::invalid_argument);
    EXPECT_THROW(schedule.build(positions, boxes(positions, 0.6), {}, 1.0), std::invalid_argument);
    auto invalid = node_boxes;
    invalid[0].min[0] = 0.01;
    EXPECT_THROW(schedule.build(positions, invalid, {}, 1.0), std::invalid_argument);
    invalid = node_boxes;
    invalid[0].max[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(schedule.build(positions, invalid, {}, 1.0), std::invalid_argument);
    invalid = node_boxes;
    invalid[0].min[2] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(schedule.build(positions, invalid, {}, 1.0), std::invalid_argument);
    EXPECT_THROW(schedule.build({Vec3::Constant(std::numeric_limits<double>::infinity())},
                                node_boxes, {}, 1.0), std::invalid_argument);
    const std::vector<Vec3> huge{Vec3(1e100, 0.0, 0.0)};
    EXPECT_THROW(schedule.build(huge, boxes(huge, 0.0), {}, 1.0), std::invalid_argument);
    const std::vector<Vec3> imprecise{Vec3(1e16, 0.0, 0.0)};
    EXPECT_THROW(schedule.build(imprecise, boxes(imprecise, 0.0), {}, 1.0), std::invalid_argument);
}

TEST(ClothGridSchedule, OneSidedLongDependenciesSplitParityIntoIndependentBatches) {
    const std::vector<Vec3> positions = {
        Vec3(0.1, 0.1, 0.1), Vec3(2.1, 0.1, 0.1),
        Vec3(4.1, 0.1, 0.1), Vec3(6.1, 0.1, 0.1),
        Vec3(1.1, 0.1, 0.1), Vec3(0.2, 0.1, 0.1)
    };
    // Reverse-only edges force a triangle of same-parity cells to use three
    // batches. Same-cell and cross-parity edges need no additional batch.
    std::vector<std::vector<int>> dependencies(positions.size());
    dependencies[1] = {0, 0};
    dependencies[2] = {0, 1};
    dependencies[4] = {0, 1, 2, 3};
    dependencies[5] = {0, 5};
    ClothGridSchedule schedule;
    schedule.build(positions, boxes(positions, 0.05), dependencies, 1.0);
    ASSERT_EQ(schedule.batches.size(), 4u);
    EXPECT_EQ(schedule.vertex_color_groups[0], (std::vector<int>{0, 5, 3}));
    EXPECT_EQ(schedule.vertex_color_groups[1], std::vector<int>{1});
    EXPECT_EQ(schedule.vertex_color_groups[2], std::vector<int>{2});
    EXPECT_EQ(schedule.vertex_color_groups[3], std::vector<int>{4});
    for (std::size_t vertex = 0; vertex < dependencies.size(); ++vertex)
        for (int other : dependencies[vertex]) {
            const auto& a = cell_for_vertex(schedule, static_cast<int>(vertex));
            const auto& b = cell_for_vertex(schedule, other);
            if (a.index != b.index) EXPECT_NE(a.batch_id, b.batch_id);
        }
}

TEST(ClothGridSchedule, InvalidDependenciesPreservePreviousSchedule) {
    const std::vector<Vec3> positions{Vec3(0.1, 0.1, 0.1)};
    ClothGridSchedule schedule;
    schedule.build(positions, boxes(positions), {}, 1.0);
    const auto groups = schedule.vertex_color_groups;
    EXPECT_THROW(schedule.build(positions, boxes(positions), {{1}}, 2.0), std::invalid_argument);
    EXPECT_THROW(schedule.build(positions, boxes(positions), {{-1}}, 2.0), std::invalid_argument);
    EXPECT_THROW(schedule.build(positions, boxes(positions), {{}, {}}, 2.0), std::invalid_argument);
    EXPECT_EQ(schedule.dx, 1.0);
    EXPECT_EQ(schedule.vertex_color_groups, groups);
}

TEST(ClothGridSchedule, AutoDxEliminatesLongOneSidedDependencySubBatches) {
    const std::vector<Vec3> positions = {
        Vec3(0.1, 0.1, 0.1), Vec3(2.1, 0.1, 0.1),
        Vec3(4.1, 0.1, 0.1), Vec3(6.1, 0.1, 0.1), Vec3(0.2, 0.1, 0.1)
    };
    std::vector<std::vector<int>> dependencies(positions.size());
    dependencies[1] = {0, 0};
    dependencies[2] = {0, 1};
    dependencies[4] = {0, 4}; // Same-cell and self edges are harmless.
    const auto node_boxes = boxes(positions, 0.05);
    ClothGridSchedule manual, automatic;
    manual.build(positions, node_boxes, dependencies, 1.0);
    ASSERT_GT(manual.batches.size(), 1u);
    automatic.build_auto_dx(positions, node_boxes, dependencies, 1.0);
    EXPECT_GE(automatic.dx, 1.0);
    EXPECT_GT(static_cast<long double>(automatic.dx),
        static_cast<long double>(positions[2].x()) - positions[0].x());
    expect_one_batch_per_parity(automatic, node_boxes, dependencies);
    // Opting in must not change the manual scheduling contract.
    ClothGridSchedule unchanged;
    unchanged.build(positions, node_boxes, dependencies, 1.0);
    EXPECT_DOUBLE_EQ(unchanged.dx, manual.dx);
    EXPECT_EQ(unchanged.batches, manual.batches);
    EXPECT_EQ(unchanged.vertex_color_groups, manual.vertex_color_groups);
}

TEST(ClothGridSchedule, AutoDxUsesActualAsymmetricNodeBoxReach) {
    const std::vector<Vec3> positions = {
        Vec3(0.25, -0.25, 0.25), Vec3(4.25, -0.25, 0.25)
    };
    auto node_boxes = boxes(positions, 0.1);
    node_boxes[0].min.x() = positions[0].x() - 0.8;
    node_boxes[1].max.z() = positions[1].z() + 0.7;
    ClothGridSchedule automatic;
    automatic.build_auto_dx(positions, node_boxes, {}, 0.01);
    for (std::size_t vertex = 0; vertex < positions.size(); ++vertex)
        for (int axis = 0; axis < 3; ++axis) {
            const long double lower = static_cast<long double>(positions[vertex][axis])
                - node_boxes[vertex].min[axis];
            const long double upper = static_cast<long double>(node_boxes[vertex].max[axis])
                - positions[vertex][axis];
            EXPECT_GT(static_cast<long double>(automatic.dx), 2 * std::max(lower, upper));
        }
    expect_one_batch_per_parity(automatic, node_boxes, {});
}

TEST(ClothGridSchedule, AutoDxPreservesAdequateMinimumAndHandlesEmptyInput) {
    const std::vector<Vec3> positions = {
        Vec3(-0.1, 0.2, 0.3), Vec3(0.4, 0.2, 0.3)
    };
    const auto node_boxes = boxes(positions, 0.02);
    const std::vector<std::vector<int>> dependencies = {{}, {0}};
    ClothGridSchedule automatic;
    automatic.build_auto_dx(positions, node_boxes, dependencies, 5.0);
    EXPECT_DOUBLE_EQ(automatic.dx, 5.0);
    expect_one_batch_per_parity(automatic, node_boxes, dependencies);
    automatic.build_auto_dx({}, {}, {}, 0.125);
    EXPECT_DOUBLE_EQ(automatic.dx, 0.125);
    EXPECT_TRUE(automatic.cells.empty());
    EXPECT_TRUE(automatic.batches.empty());
    EXPECT_TRUE(automatic.vertex_color_groups.empty());
    EXPECT_EQ(automatic.min_index, (std::array<std::int64_t, 3>{{0, 0, 0}}));
    EXPECT_EQ(automatic.max_index, (std::array<std::int64_t, 3>{{-1, -1, -1}}));
}

TEST(ClothGridSchedule, AutoDxHandlesNegativeCoordinatesAndAdjacentBoundaryValues) {
    const double infinity = std::numeric_limits<double>::infinity();
    const std::vector<Vec3> positions = {
        Vec3(-2.0, -2.0, -2.0), Vec3(-1.0, -1.0, -1.0),
        Vec3(std::nextafter(-1.0, -infinity), 0.0, 0.0),
        Vec3(std::nextafter(-1.0, infinity), 1.0, 1.0),
        Vec3(0.0, 2.0, 2.0), Vec3(8.0, 8.0, 8.0)
    };
    const std::vector<std::vector<int>> dependencies = {
        {}, {0}, {1}, {2}, {3}, {}
    };
    const auto node_boxes = boxes(positions, 0.01);
    ClothGridSchedule automatic;
    automatic.build_auto_dx(positions, node_boxes, dependencies, 0.02);
    EXPECT_GT(automatic.dx, 1.0);
    expect_one_batch_per_parity(automatic, node_boxes, dependencies);
    // Stable world origin and correct floor, including negative cells.
    for (std::size_t vertex = 0; vertex < positions.size(); ++vertex) {
        const auto& cell = cell_for_vertex(automatic, static_cast<int>(vertex));
        for (int axis = 0; axis < 3; ++axis)
            EXPECT_EQ(cell.index[axis], static_cast<std::int64_t>(
                std::floor(static_cast<long double>(positions[vertex][axis]) / automatic.dx)));
    }
}

TEST(ClothGridSchedule, AutoDxInvalidInputsAndOverflowPreservePreviousSchedule) {
    const std::vector<Vec3> positions = {Vec3(0.1, 0.1, 0.1), Vec3(2.1, 0.1, 0.1)};
    const auto node_boxes = boxes(positions, 0.05);
    const std::vector<std::vector<int>> dependencies = {{}, {0}};
    ClothGridSchedule automatic;
    automatic.build_auto_dx(positions, node_boxes, dependencies, 0.1);
    const auto original = automatic;
    const auto expect_preserved = [&]() {
        EXPECT_DOUBLE_EQ(automatic.dx, original.dx);
        EXPECT_EQ(automatic.min_index, original.min_index);
        EXPECT_EQ(automatic.max_index, original.max_index);
        EXPECT_EQ(automatic.batches, original.batches);
        EXPECT_EQ(automatic.vertex_color_groups, original.vertex_color_groups);
        EXPECT_EQ(automatic.cells.size(), original.cells.size());
    };
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (double minimum : {0.0, -1.0, infinity, nan}) {
        EXPECT_THROW(automatic.build_auto_dx(positions, node_boxes, dependencies, minimum),
            std::invalid_argument);
        expect_preserved();
    }
    EXPECT_THROW(automatic.build_auto_dx(positions, {}, dependencies, 0.1), std::invalid_argument);
    expect_preserved();
    for (const auto& invalid : std::vector<std::vector<std::vector<int>>>{
             {{}, {-1}}, {{}, {2}}, {{0}}}) {
        EXPECT_THROW(automatic.build_auto_dx(positions, node_boxes, invalid, 0.1),
            std::invalid_argument);
        expect_preserved();
    }
    auto invalid_positions = positions;
    invalid_positions[0].x() = nan;
    EXPECT_THROW(automatic.build_auto_dx(invalid_positions, node_boxes, dependencies, 0.1),
        std::invalid_argument);
    expect_preserved();
    auto invalid_boxes = node_boxes;
    invalid_boxes[0].min.x() = positions[0].x() + 0.01;
    EXPECT_THROW(automatic.build_auto_dx(positions, invalid_boxes, dependencies, 0.1),
        std::invalid_argument);
    expect_preserved();
    invalid_boxes = node_boxes;
    invalid_boxes[1].max.z() = infinity;
    EXPECT_THROW(automatic.build_auto_dx(positions, invalid_boxes, dependencies, 0.1),
        std::invalid_argument);
    expect_preserved();
    // Finite endpoints can have a span larger than any finite double dx.
    const double huge = 0.75 * std::numeric_limits<double>::max();
    const std::vector<Vec3> distant = {Vec3(-huge, 0.0, 0.0), Vec3(huge, 0.0, 0.0)};
    EXPECT_THROW(automatic.build_auto_dx(distant, boxes(distant, 0.0), dependencies, 0.1),
        std::invalid_argument);
    expect_preserved();
}

TEST(ClothGridSchedule, AutoDxRecomputesAfterMotionAndCanReturnToConfiguredMinimum) {
    std::vector<Vec3> positions = {Vec3(-4.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0)};
    const std::vector<std::vector<int>> dependencies = {{}, {0}};
    ClothGridSchedule automatic;
    automatic.build_auto_dx(positions, boxes(positions, 0.01), dependencies, 0.5);
    const double initial_dx = automatic.dx;
    EXPECT_GT(initial_dx, 8.0);
    positions = {Vec3(-0.1, 0.0, 0.0), Vec3(0.1, 0.0, 0.0)};
    const auto smaller_boxes = boxes(positions, 0.01);
    automatic.build_auto_dx(positions, smaller_boxes, dependencies, 0.5);
    EXPECT_DOUBLE_EQ(automatic.dx, 0.5);
    EXPECT_LT(automatic.dx, initial_dx);
    expect_one_batch_per_parity(automatic, smaller_boxes, dependencies);
    positions[1].z() = 12.0;
    const auto stretched_boxes = boxes(positions, 0.01);
    automatic.build_auto_dx(positions, stretched_boxes, dependencies, 0.5);
    EXPECT_GT(automatic.dx, 12.0);
    expect_one_batch_per_parity(automatic, stretched_boxes, dependencies);
}

TEST(ClothGridSchedule, PrioritizesSummedCellCostsWithStableTiesAndPreservesOwnership) {
    const std::vector<Vec3> positions = {
        Vec3(0.2, 0.2, 0.2), Vec3(2.2, 0.2, 0.2),
        Vec3(4.2, 0.2, 0.2), Vec3(6.2, 0.2, 0.2),
        Vec3(0.7, 0.2, 0.2), Vec3(2.7, 0.2, 0.2),
        Vec3(4.7, 0.2, 0.2), Vec3(6.7, 0.2, 0.2)
    };
    const auto node_boxes = boxes(positions);
    ClothGridSchedule schedule;
    schedule.build(positions, node_boxes, {}, 1.0);
    const auto original = schedule;
    // Cell sums are 4, 10, 10, 7; sum-based ordering differs from ordering by
    // the largest individual cost, and cells 1 and 2 exercise the tie break.
    schedule.prioritize_cells({2, 5, 9, 1, 2, 5, 1, 6});
    ASSERT_EQ(schedule.batches.size(), 1u);
    EXPECT_EQ(schedule.batches[0], (std::vector<int>{1, 2, 3, 0}));
    EXPECT_EQ(schedule.vertex_color_groups[0], (std::vector<int>{1, 5, 2, 6, 3, 7, 0, 4}));
    EXPECT_EQ(schedule.min_index, original.min_index);
    EXPECT_EQ(schedule.max_index, original.max_index);
    EXPECT_EQ(schedule.dx, original.dx);
    ASSERT_EQ(schedule.cells.size(), original.cells.size());
    for (std::size_t i = 0; i < schedule.cells.size(); ++i) {
        const auto& cell = schedule.cells[i];
        const auto& before = original.cells[i];
        EXPECT_EQ(cell.index, before.index);
        EXPECT_EQ(cell.vertices, before.vertices);
        EXPECT_EQ(cell.color_id, before.color_id);
        EXPECT_EQ(cell.batch_id, before.batch_id);
        EXPECT_EQ(cell.bounds.min, before.bounds.min);
        EXPECT_EQ(cell.bounds.max, before.bounds.max);
        for (std::size_t j = i + 1; j < schedule.cells.size(); ++j)
            for (int a : cell.vertices)
                for (int b : schedule.cells[j].vertices)
                    EXPECT_FALSE(aabb_intersects(node_boxes[a], node_boxes[b]));
    }
    // Equal costs restore the deterministic cell-index order even after a
    // previous prioritization; zero cost is valid for an inactive vertex.
    schedule.prioritize_cells(std::vector<std::size_t>(positions.size(), 0));
    EXPECT_EQ(schedule.batches, original.batches);
    EXPECT_EQ(schedule.vertex_color_groups, original.vertex_color_groups);
}

TEST(ClothGridSchedule, InvalidSchedulingCostsPreservePreviousOrder) {
    const std::vector<Vec3> positions = {
        Vec3(0.2, 0.2, 0.2), Vec3(0.7, 0.2, 0.2), Vec3(2.2, 0.2, 0.2)
    };
    ClothGridSchedule schedule;
    schedule.build(positions, boxes(positions), {}, 1.0);
    schedule.prioritize_cells({1, 1, 3});
    const auto batches = schedule.batches;
    const auto groups = schedule.vertex_color_groups;
    EXPECT_THROW(schedule.prioritize_cells({1, 1}), std::invalid_argument);
    EXPECT_THROW(schedule.prioritize_cells({1, 1, 1, 1}), std::invalid_argument);
    EXPECT_THROW(schedule.prioritize_cells({std::numeric_limits<std::size_t>::max(), 1, 1}),
                 std::overflow_error);
    EXPECT_EQ(schedule.batches, batches);
    EXPECT_EQ(schedule.vertex_color_groups, groups);

    ClothGridSchedule empty;
    EXPECT_NO_THROW(empty.prioritize_cells({}));
    EXPECT_THROW(empty.prioritize_cells({1}), std::invalid_argument);
}

TEST(ClothGridSchedule, SerialAndParallelPreserveArithmeticAndBarriersAcrossTeamChanges) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    std::vector<Vec3> positions;
    for (int cell = 0; cell < 24; ++cell) {
        positions.emplace_back(2 * cell + 0.25, 0.25, 0.25);
        positions.emplace_back(2 * cell + 0.75, 0.25, 0.25);
    }
    std::vector<std::vector<int>> dependencies(positions.size());
    for (int v = 0; v < static_cast<int>(positions.size()); ++v) {
        if (v % 2) dependencies[v].push_back(v - 1); // Serial within cell.
        if (v >= 2) dependencies[v].push_back(v - 2); // Batch barrier.
        if (v >= 10) dependencies[v].push_back(v - 10); // Longer edge.
    }
    ClothGridSchedule schedule;
    schedule.build(positions, boxes(positions), dependencies, 1.0);
    const auto original = schedule;
    std::vector<std::size_t> costs(positions.size());
    for (std::size_t vertex = 0; vertex < costs.size(); ++vertex)
        costs[vertex] = 1 + (vertex * 7) % 13;
    schedule.prioritize_cells(costs);
    ASSERT_NE(schedule.batches, original.batches);
    ASSERT_EQ(schedule.batches.size(), original.batches.size());
    for (std::size_t batch = 0; batch < schedule.batches.size(); ++batch) {
        auto members = schedule.batches[batch];
        std::sort(members.begin(), members.end());
        EXPECT_EQ(members, original.batches[batch]);
    }
    for (int threads : {1, 2, 4, 8}) {
        std::vector<double> expected(positions.size(), 1.0), actual = expected;
        auto original_order = expected;
        for (int iteration = 0; iteration < 16; ++iteration) {
            auto update = [&](std::vector<double>& state, int vertex) {
                double value = state[vertex];
                for (int neighbor : dependencies[vertex]) value += state[neighbor];
                // Reassociation would change this ordered arithmetic.
                value += 1e16;
                value += vertex + iteration;
                value -= 1e16;
                state[vertex] = std::fmod(value, 1048576.0);
            };
            original.run(false, [&](int v) { update(original_order, v); });
            schedule.run(false, [&](int v) { update(expected, v); });
            omp_set_num_threads(iteration % 3 == 0 ? 1 : threads);
            schedule.run(true, [&](int v) { update(actual, v); });
            EXPECT_EQ(0, std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(double)))
                << "threads=" << threads << ", iteration=" << iteration;
            EXPECT_EQ(0, std::memcmp(expected.data(), original_order.data(), expected.size() * sizeof(double)))
                << "prioritization changed arithmetic: threads=" << threads << ", iteration=" << iteration;
        }
    }
}

TEST(ClothGridSchedule, CallbackFailureJoinsBatchAndSkipsLaterBatches) {
    RestoreOpenMP restore;
    omp_set_num_threads(4);
    const std::vector<Vec3> positions = {
        Vec3(0.1, 0.1, 0.1), Vec3(0.2, 0.1, 0.1),
        Vec3(2.1, 0.1, 0.1), Vec3(1.1, 0.1, 0.1)
    };
    ClothGridSchedule schedule;
    schedule.build(positions, boxes(positions, 0.05), {}, 1.0);
    for (bool parallel : {false, true}) {
        std::atomic<bool> reached_later_vertex{false}, reached_later_batch{false};
        EXPECT_THROW(schedule.run(parallel, [&](int vertex) {
            if (vertex == 0) throw std::runtime_error("grid callback failure");
            if (vertex == 1) reached_later_vertex.store(true);
            if (vertex == 3) reached_later_batch.store(true);
        }), std::runtime_error);
        EXPECT_FALSE(reached_later_vertex.load());
        EXPECT_FALSE(reached_later_batch.load());
    }
}

TEST(ClothGridSchedule, EmptyAndMovedRebuildsDiscardOldOwnership) {
    ClothGridSchedule schedule;
    schedule.build({}, {}, {}, 1.0);
    EXPECT_TRUE(schedule.cells.empty());
    EXPECT_TRUE(schedule.batches.empty());
    EXPECT_EQ(schedule.min_index, (std::array<std::int64_t, 3>{{0, 0, 0}}));
    EXPECT_EQ(schedule.max_index, (std::array<std::int64_t, 3>{{-1, -1, -1}}));
    int calls = 0;
    schedule.run(false, [&](int) { ++calls; });
    schedule.run(true, [&](int) { ++calls; });
    EXPECT_EQ(calls, 0);
    std::vector<Vec3> positions{Vec3(0.2, 0.2, 0.2), Vec3(0.3, 0.3, 0.3)};
    schedule.build(positions, boxes(positions), {}, 1.0);
    ASSERT_EQ(schedule.cells.size(), 1u);
    positions[1] = Vec3(3.3, 2.3, -0.3);
    schedule.build(positions, boxes(positions), {}, 1.0);
    ASSERT_EQ(schedule.cells.size(), 2u);
    EXPECT_EQ(cell_for_vertex(schedule, 1).index, (std::array<std::int64_t, 3>{{3, 2, -1}}));
    EXPECT_EQ(cell_for_vertex(schedule, 1).color_id, 5);
    schedule.build({}, {}, {}, 2.0);
    EXPECT_EQ(schedule.dx, 2.0);
    EXPECT_TRUE(schedule.cells.empty());
    EXPECT_TRUE(schedule.batches.empty());
    EXPECT_TRUE(schedule.vertex_color_groups.empty());
}
