#include "grid_coloring.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace solver_detail {
namespace {

using CellIndex = std::array<std::int64_t, 3>;

std::int64_t grid_index(double position, double dx) {
    const long double value = std::floor(static_cast<long double>(position) / dx);
    // The upper endpoint is exclusive: converting 2^63 to int64 is undefined.
    // Avoid comparing against INT64_MAX rounded to double on ARM platforms.
    constexpr long double limit = 9223372036854775808.0L;
    if (!std::isfinite(value) || value < -limit || value >= limit)
        throw std::invalid_argument("Cloth grid index exceeds the supported integer range");
    const auto index = static_cast<std::int64_t>(value);
    if (index == std::numeric_limits<std::int64_t>::max())
        throw std::invalid_argument("Cloth grid cell upper index would overflow");
    return index;
}

AABB cell_bounds(const CellIndex& index, double dx) {
    AABB result;
    for (int axis = 0; axis < 3; ++axis) {
        result.min[axis] = static_cast<double>(static_cast<long double>(index[axis]) * dx);
        result.max[axis] = static_cast<double>(static_cast<long double>(index[axis] + 1) * dx);
        if (!std::isfinite(result.min[axis]) || !std::isfinite(result.max[axis]) ||
            !(result.min[axis] < result.max[axis]))
            throw std::invalid_argument("Cloth grid cell bounds are not representable at this dx");
    }
    return result;
}

int parity_color(const CellIndex& index) {
    // Unsigned conversion defines negative-index parity portably.
    return static_cast<int>(static_cast<std::uint64_t>(index[0]) & 1u)
         + 2 * static_cast<int>(static_cast<std::uint64_t>(index[1]) & 1u)
         + 4 * static_cast<int>(static_cast<std::uint64_t>(index[2]) & 1u);
}

} // namespace

void ClothGridSchedule::build(const std::vector<Vec3>& positions,
                              const std::vector<AABB>& node_boxes,
                              const std::vector<std::vector<int>>& dependencies,
                              double cell_dx) {
    if (!std::isfinite(cell_dx) || !(cell_dx > 0.0))
        throw std::invalid_argument("Cloth grid dx must be finite and positive");
    if (positions.size() != node_boxes.size())
        throw std::invalid_argument("Cloth grid requires one node box per vertex");
    if (!dependencies.empty() && dependencies.size() != positions.size())
        throw std::invalid_argument("Cloth grid requires one dependency row per vertex");
    if (positions.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Cloth grid has too many vertices");

    ClothGridSchedule next;
    next.dx = cell_dx;
    std::map<CellIndex, std::vector<int>> occupied;
    const long double half_dx = static_cast<long double>(cell_dx) / 2;
    for (std::size_t vertex = 0; vertex < positions.size(); ++vertex) {
        const Vec3& p = positions[vertex];
        const AABB& box = node_boxes[vertex];
        CellIndex index;
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(p[axis]) || !std::isfinite(box.min[axis]) ||
                !std::isfinite(box.max[axis]) || box.min[axis] > p[axis] ||
                box.max[axis] < p[axis])
                throw std::invalid_argument("Cloth grid node boxes must be finite and contain their vertices");
            const long double lower_reach = static_cast<long double>(p[axis]) - box.min[axis];
            const long double upper_reach = static_cast<long double>(box.max[axis]) - p[axis];
            if (!(lower_reach < half_dx) || !(upper_reach < half_dx))
                throw std::invalid_argument("Cloth grid node boxes must extend strictly less than dx/2 from each vertex");
            index[axis] = grid_index(p[axis], cell_dx);
            const auto lo = grid_index(box.min[axis], cell_dx);
            const auto hi = grid_index(box.max[axis], cell_dx);
            if (vertex == 0) {
                next.min_index[axis] = lo;
                next.max_index[axis] = hi;
            } else {
                next.min_index[axis] = std::min(next.min_index[axis], lo);
                next.max_index[axis] = std::max(next.max_index[axis], hi);
            }
        }
        occupied[index].push_back(static_cast<int>(vertex));
    }

    if (!positions.empty()) {
        // Validate the outer empty cells as well, since visualization covers
        // the entire node-box domain rather than just occupied cells.
        cell_bounds(next.min_index, cell_dx);
        cell_bounds(next.max_index, cell_dx);
    }
    std::vector<int> vertex_to_cell(positions.size());
    next.cells.reserve(occupied.size());
    for (auto& entry : occupied) {
        GridCell cell;
        cell.index = entry.first;
        cell.bounds = cell_bounds(cell.index, cell_dx);
        cell.color_id = parity_color(cell.index);
        cell.vertices = std::move(entry.second);
        const int cell_id = static_cast<int>(next.cells.size());
        for (int vertex : cell.vertices) vertex_to_cell[vertex] = cell_id;
        next.cells.push_back(std::move(cell));
    }

    // Dependencies may be one-sided. Both endpoints must see each edge even
    // if the input row happens to belong to the later cell in traversal order.
    std::vector<std::vector<int>> neighbors(next.cells.size());
    for (std::size_t vertex = 0; vertex < dependencies.size(); ++vertex) {
        const int a = vertex_to_cell[vertex];
        for (int other : dependencies[vertex]) {
            if (other < 0 || static_cast<std::size_t>(other) >= positions.size())
                throw std::invalid_argument("Cloth grid dependency contains an invalid vertex index");
            const int b = vertex_to_cell[other];
            if (a != b && next.cells[a].color_id == next.cells[b].color_id) {
                neighbors[a].push_back(b);
                neighbors[b].push_back(a);
            }
        }
    }
    for (auto& row : neighbors) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }

    std::vector<int> forbidden;
    for (int color = 0; color < 8; ++color) {
        const int first_batch = static_cast<int>(next.batches.size());
        forbidden.clear();
        for (int cell_id = 0; cell_id < static_cast<int>(next.cells.size()); ++cell_id) {
            auto& cell = next.cells[cell_id];
            if (cell.color_id != color) continue;
            for (int other : neighbors[cell_id]) {
                const int batch = next.cells[other].batch_id;
                if (batch >= first_batch) forbidden[batch - first_batch] = cell_id;
            }
            int local_batch = 0;
            while (local_batch < static_cast<int>(forbidden.size()) &&
                   forbidden[local_batch] == cell_id)
                ++local_batch;
            if (local_batch == static_cast<int>(forbidden.size())) {
                forbidden.push_back(-1);
                next.batches.emplace_back();
                next.vertex_color_groups.emplace_back();
            }
            cell.batch_id = first_batch + local_batch;
            next.batches[cell.batch_id].push_back(cell_id);
            auto& vertices = next.vertex_color_groups[cell.batch_id];
            vertices.insert(vertices.end(), cell.vertices.begin(), cell.vertices.end());
        }
    }
    *this = std::move(next);
}

void ClothGridSchedule::prioritize_cells(const std::vector<std::size_t>& vertex_costs) {
    std::size_t vertex_count = 0;
    for (const auto& cell : cells) {
        if (cell.vertices.size() > std::numeric_limits<std::size_t>::max() - vertex_count)
            throw std::overflow_error("Cloth grid vertex count overflow");
        vertex_count += cell.vertices.size();
    }
    if (vertex_costs.size() != vertex_count)
        throw std::invalid_argument("Cloth grid requires one scheduling cost per vertex");

    std::vector<std::size_t> cell_costs(cells.size(), 0);
    for (std::size_t cell = 0; cell < cells.size(); ++cell) {
        for (int vertex : cells[cell].vertices) {
            if (vertex < 0 || static_cast<std::size_t>(vertex) >= vertex_costs.size())
                throw std::invalid_argument("Cloth grid cell contains an invalid vertex index");
            const auto cost = vertex_costs[vertex];
            if (cost > std::numeric_limits<std::size_t>::max() - cell_costs[cell])
                throw std::overflow_error("Cloth grid cell scheduling cost overflow");
            cell_costs[cell] += cost;
        }
    }

    // Build both views before replacing either, including allocation failures.
    auto ordered_batches = batches;
    std::vector<std::vector<int>> ordered_vertices(batches.size());
    for (std::size_t b = 0; b < ordered_batches.size(); ++b) {
        auto& batch = ordered_batches[b];
        std::sort(batch.begin(), batch.end(), [&](int a, int other) {
            if (cell_costs[a] != cell_costs[other])
                return cell_costs[a] > cell_costs[other];
            return a < other;
        });
        auto& vertices = ordered_vertices[b];
        for (int cell : batch)
            vertices.insert(vertices.end(), cells[cell].vertices.begin(), cells[cell].vertices.end());
    }
    batches.swap(ordered_batches);
    vertex_color_groups.swap(ordered_vertices);
}

} // namespace solver_detail
