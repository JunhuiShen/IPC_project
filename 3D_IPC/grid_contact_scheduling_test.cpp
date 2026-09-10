#include "grid_contact_scheduling.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {

struct RestoreOpenMP {
    int threads = omp_get_max_threads();
    int dynamic = omp_get_dynamic();
    ~RestoreOpenMP() {
        omp_set_num_threads(threads);
        omp_set_dynamic(dynamic);
    }
};

solver_detail::ClothGridSchedule make_schedule(
    std::vector<std::vector<int>> cell_vertices,
    std::vector<std::vector<int>> batches) {
    solver_detail::ClothGridSchedule schedule;
    schedule.dx = 1.0;
    schedule.cells.resize(cell_vertices.size());
    for (int cell = 0; cell < static_cast<int>(cell_vertices.size()); ++cell) {
        schedule.cells[cell].vertices = std::move(cell_vertices[cell]);
        schedule.cells[cell].index = {{2 * cell, 0, 0}};
        schedule.cells[cell].color_id = cell & 1;
    }
    schedule.batches = std::move(batches);
    schedule.vertex_color_groups.resize(schedule.batches.size());
    for (int batch = 0; batch < static_cast<int>(schedule.batches.size()); ++batch) {
        for (int cell : schedule.batches[batch]) {
            schedule.cells[cell].batch_id = batch;
            const auto& vertices = schedule.cells[cell].vertices;
            schedule.vertex_color_groups[batch].insert(
                schedule.vertex_color_groups[batch].end(), vertices.begin(), vertices.end());
        }
    }
    return schedule;
}

BroadPhase::Cache make_cache(const std::vector<std::pair<int, int>>& counts) {
    BroadPhase::Cache cache;
    cache.vertex_nt.resize(counts.size());
    cache.vertex_ss.resize(counts.size());
    for (int v = 0; v < static_cast<int>(counts.size()); ++v) {
        cache.vertex_nt[v].resize(counts[v].first);
        cache.vertex_ss[v].resize(counts[v].second);
    }
    return cache;
}

int bit_count(std::uint64_t bits) {
    int count = 0;
    while (bits) {
        bits &= bits - 1;
        ++count;
    }
    return count;
}

class SyntheticModel {
public:
    SyntheticModel(const BroadPhase::Cache& cache,
                   std::vector<std::vector<int>> dependencies)
        : cache_(cache), dependencies_(std::move(dependencies)), x(cache.vertex_nt.size()),
          step(x.size()), completed_(new std::atomic<int>[x.size()]),
          workers_(new std::atomic<std::uint64_t>[x.size()]) {
        for (int v = 0; v < static_cast<int>(x.size()); ++v)
            x[v] = 1.0 + 0.25 * v;
        reset_sweep();
    }

    void reset_sweep() {
        for (int v = 0; v < static_cast<int>(x.size()); ++v) {
            completed_[v].store(0, std::memory_order_relaxed);
            workers_[v].store(0, std::memory_order_relaxed);
        }
    }

    int contact_count(int v) const {
        return static_cast<int>(cache_.vertex_nt[v].size() + cache_.vertex_ss[v].size());
    }

    static bool active(int j) { return j == 0 || j % 4 != 0; }
    static bool certified_clear(int j) { return j > 0 && j % 8 == 0; }

    unsigned compute(int v, int j, solver_detail::ContactContribution& value) {
        const int worker = omp_get_thread_num();
        if (worker < 64)
            workers_[v].fetch_or(std::uint64_t(1) << worker, std::memory_order_relaxed);
        const bool is_active = active(j);
        if (is_active) value.gradient[0] = gradient(v, j);
        return (is_active ? 1u : 0u) | (certified_clear(j) ? 2u : 0u);
    }

    void apply(int v, const solver_detail::ContactContribution* values,
               const solver_detail::ContactMaskWord* mask) {
        double result = values[0].gradient[0];
        solver_detail::for_active_contact(mask, contact_count(v),
            [&](int j) { result += values[j].gradient[0]; });
        step[v] = std::fmod(result, modulus);
    }

    bool ccd(int v, int j, solver_detail::ContactContribution& value, bool certified) {
        if (certified != certified_clear(j))
            certificate_mismatches.fetch_add(1, std::memory_order_relaxed);
        if (!collision(v, j, certified)) return false;
        value.toi = toi(v, j);
        return true;
    }

    void commit(int v, const solver_detail::ContactContribution* values,
                const solver_detail::ContactMaskWord* mask) {
        double result = 0.0;
        solver_detail::for_active_contact(mask, contact_count(v),
            [&](int j) { result += values[j].toi; });
        x[v] = std::fmod(step[v] + result, modulus);
        completed_[v].store(1, std::memory_order_release);
    }

    void scalar_vertex(int v) {
        baseline_calls.fetch_add(1, std::memory_order_relaxed);
        const int count = contact_count(v);
        double result = gradient(v, 0);
        for (int j = 1; j <= count; ++j)
            if (active(j)) result += gradient(v, j);
        step[v] = std::fmod(result, modulus);
        result = 0.0;
        for (int j = 1; j <= count; ++j)
            if (collision(v, j, certified_clear(j))) result += toi(v, j);
        x[v] = std::fmod(step[v] + result, modulus);
        completed_[v].store(1, std::memory_order_release);
    }

    int workers_for(int v) const {
        return bit_count(workers_[v].load(std::memory_order_relaxed));
    }

    std::vector<double> x;
    std::vector<double> step;
    std::atomic<int> dependency_violations{0};
    std::atomic<int> certificate_mismatches{0};
    std::atomic<int> baseline_calls{0};

private:
    static constexpr double modulus = 1048576.0;

    double dependency_value(int v) {
        double value = x[v] + 0.125 * (v + 1);
        for (int dependency : dependencies_[v]) {
            if (completed_[dependency].load(std::memory_order_acquire))
                value += x[dependency];
            else {
                dependency_violations.fetch_add(1, std::memory_order_relaxed);
                // Do not read a concurrently written value after detecting a
                // missing serial/cross-batch dependency.
                value -= 4096.0 + dependency;
            }
        }
        return std::fmod(value, modulus);
    }

    double gradient(int v, int j) {
        const double dependency = dependency_value(v);
        if (j == 0) return dependency;
        if (j % 4 == 1) return 1e16;
        if (j % 4 == 3) return -1e16;
        return std::fmod(dependency + 0.5 * j, modulus);
    }

    bool collision(int v, int j, bool certified) const {
        return j > 0 && !certified && (j % 13 == 0 || j == contact_count(v));
    }

    double toi(int v, int j) const {
        return std::fmod(step[v] + 0.25 * (v + 1) + j, 1024.0);
    }

    const BroadPhase::Cache& cache_;
    std::vector<std::vector<int>> dependencies_;
    std::unique_ptr<std::atomic<int>[]> completed_;
    std::unique_ptr<std::atomic<std::uint64_t>[]> workers_;
};

void run_scalar(const solver_detail::ClothGridSchedule& schedule, SyntheticModel& model) {
    for (const auto& batch : schedule.batches)
        for (int cell : batch)
            for (int v : schedule.cells[cell].vertices)
                model.scalar_vertex(v);
}

void run_sweep(solver_detail::ClothGridContactSweep& sweep,
               const solver_detail::ClothGridSchedule& schedule,
               SyntheticModel& model, bool profile = false) {
    sweep.run(schedule,
        [&](int v, int j, solver_detail::ContactContribution& value) {
            return model.compute(v, j, value);
        },
        [&](int v, const solver_detail::ContactContribution* values,
            const solver_detail::ContactMaskWord* mask) { model.apply(v, values, mask); },
        [&](int v) { model.scalar_vertex(v); },
        [&](int v, int j, solver_detail::ContactContribution& value, bool certified) {
            return model.ccd(v, j, value, certified);
        },
        [&](int v, const solver_detail::ContactContribution* values,
            const solver_detail::ContactMaskWord* mask) { model.commit(v, values, mask); }, profile);
}

void expect_bitwise_equal(const std::vector<double>& expected,
                          const std::vector<double>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    EXPECT_EQ(0, std::memcmp(expected.data(), actual.data(),
                            expected.size() * sizeof(double)));
}

} // namespace

TEST(ClothGridContactSweep, SharesOneCellContactsButKeepsItsVerticesSerial) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(8);

    const auto schedule = make_schedule({{0, 1, 2, 3}}, {{0}});
    const auto cache = make_cache({{65, 64}, {72, 73}, {130, 127}, {137, 136}});
    const std::vector<std::vector<int>> dependencies = {{}, {0}, {1}, {2}};

    SyntheticModel expected(cache, dependencies), actual(cache, dependencies);
    run_scalar(schedule, expected);
    solver_detail::ClothGridContactSweep sweep;
    sweep.prepare(schedule, cache);
    run_sweep(sweep, schedule, actual);

    expect_bitwise_equal(expected.x, actual.x);
    EXPECT_EQ(actual.dependency_violations.load(), 0);
    EXPECT_EQ(actual.certificate_mismatches.load(), 0);
    EXPECT_EQ(actual.baseline_calls.load(), 0);
    ASSERT_EQ(sweep.cooperative_cells, (std::vector<int>{0}));
    for (int v = 0; v < 4; ++v)
        EXPECT_GT(actual.workers_for(v), 1) << "vertex=" << v;
}

TEST(ClothGridContactSweep, RunsIndependentCellsAndHonorsCrossBatchBarriers) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);

    const auto schedule = make_schedule({{0, 1}, {2, 3}, {4, 5}, {6}},
                                        {{0, 1}, {2}, {3}});
    const auto cache = make_cache({{68, 65}, {71, 74}, {66, 67}, {73, 72},
                                   {80, 69}, {77, 76}, {83, 78}});
    const std::vector<std::vector<int>> dependencies = {
        {}, {0}, {}, {2}, {1, 3}, {4}, {5}};

    SyntheticModel expected(cache, dependencies), actual(cache, dependencies);
    run_scalar(schedule, expected);
    solver_detail::ClothGridContactSweep sweep;
    sweep.prepare(schedule, cache);
    run_sweep(sweep, schedule, actual);

    expect_bitwise_equal(expected.x, actual.x);
    EXPECT_EQ(actual.dependency_violations.load(), 0);
    EXPECT_EQ(actual.certificate_mismatches.load(), 0);
    EXPECT_EQ(actual.baseline_calls.load(), 0);
    EXPECT_GE(sweep.cooperative_cells.size(), 4u);
}

TEST(ClothGridContactSweep, PreservesActiveAndCertifiedMasksThroughPartialTailWord) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(4);

    constexpr int nt_count = 72;
    constexpr int ss_count = 73;
    constexpr int contacts = nt_count + ss_count; // 145: not a multiple of 16.
    const auto schedule = make_schedule({{0}}, {{0}});
    const auto cache = make_cache({{nt_count, ss_count}});
    std::unique_ptr<std::atomic<int>[]> computed(new std::atomic<int>[contacts + 1]);
    std::unique_ptr<std::atomic<int>[]> ccd_seen(new std::atomic<int>[contacts + 1]);
    for (int j = 0; j <= contacts; ++j) {
        computed[j].store(0, std::memory_order_relaxed);
        ccd_seen[j].store(0, std::memory_order_relaxed);
    }
    std::vector<unsigned char> applied(contacts + 1, 0), committed(contacts + 1, 0);
    std::atomic<int> certificate_mismatches{0}, baseline_calls{0};

    solver_detail::ClothGridContactSweep sweep;
    sweep.prepare(schedule, cache);
    sweep.run(schedule,
        [&](int, int j, solver_detail::ContactContribution& value) {
            computed[j].fetch_add(1, std::memory_order_relaxed);
            value.gradient[0] = j;
            return (SyntheticModel::active(j) ? 1u : 0u)
                | (SyntheticModel::certified_clear(j) ? 2u : 0u);
        },
        [&](int, const solver_detail::ContactContribution*,
            const solver_detail::ContactMaskWord* mask) {
            solver_detail::for_active_contact(mask, contacts,
                [&](int j) { applied[j] = 1; });
        },
        [&](int) { baseline_calls.fetch_add(1, std::memory_order_relaxed); },
        [&](int, int j, solver_detail::ContactContribution& value, bool certified) {
            ccd_seen[j].fetch_add(1, std::memory_order_relaxed);
            if (certified != SyntheticModel::certified_clear(j))
                certificate_mismatches.fetch_add(1, std::memory_order_relaxed);
            const bool hit = j > 0 && !certified && (j % 13 == 0 || j == contacts);
            if (hit) value.toi = j;
            return hit;
        },
        [&](int, const solver_detail::ContactContribution*,
            const solver_detail::ContactMaskWord* mask) {
            solver_detail::for_active_contact(mask, contacts,
                [&](int j) { committed[j] = 1; });
        });

    EXPECT_EQ(baseline_calls.load(), 0);
    EXPECT_EQ(certificate_mismatches.load(), 0);
    for (int j = 0; j <= contacts; ++j) {
        EXPECT_EQ(computed[j].load(), 1) << "compute j=" << j;
        EXPECT_EQ(ccd_seen[j].load(), 1) << "ccd j=" << j;
        EXPECT_EQ(applied[j] != 0, j > 0 && SyntheticModel::active(j))
            << "apply j=" << j;
        const bool expected_hit = j > 0 && !SyntheticModel::certified_clear(j)
            && (j % 13 == 0 || j == contacts);
        EXPECT_EQ(committed[j] != 0, expected_hit) << "commit j=" << j;
    }
    EXPECT_TRUE(applied[contacts]);
    EXPECT_TRUE(committed[contacts]);
}

TEST(ClothGridContactSweep, MatchesScalarOrderAcrossTeamsAndPreparedTeamMismatch) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);

    const auto schedule = make_schedule({{0, 1, 2}, {3, 4}, {5, 6}, {7}},
                                        {{0, 1}, {2}, {3}});
    const auto cache = make_cache({{65, 64}, {79, 66}, {131, 126}, {70, 63},
                                   {91, 58}, {129, 128}, {75, 78}, {140, 133}});
    const std::vector<std::vector<int>> dependencies = {
        {}, {0}, {1}, {}, {3}, {2, 4}, {5}, {6}};

    struct TeamCase { int prepared, actual; };
    for (const TeamCase team : {TeamCase{1, 1}, {2, 2}, {4, 4}, {8, 8}, {12, 12},
                                {8, 2}, {4, 1}, {2, 4}, {12, 4}, {4, 12}}) {
        SyntheticModel expected(cache, dependencies), actual(cache, dependencies);
        omp_set_num_threads(team.prepared);
        solver_detail::ClothGridContactSweep sweep;
        sweep.prepare(schedule, cache);
        EXPECT_EQ(sweep.prepared_team, team.prepared);
        omp_set_num_threads(team.actual);
        for (int iteration = 0; iteration < 7; ++iteration) {
            expected.reset_sweep();
            actual.reset_sweep();
            run_scalar(schedule, expected);
            run_sweep(sweep, schedule, actual);
        }
        expect_bitwise_equal(expected.x, actual.x);
        EXPECT_EQ(actual.dependency_violations.load(), 0)
            << "prepared=" << team.prepared << " actual=" << team.actual;
        EXPECT_EQ(actual.certificate_mismatches.load(), 0)
            << "prepared=" << team.prepared << " actual=" << team.actual;
    }
}

TEST(ClothGridContactSweep, TwelveWorkersAlternateLeaderOnlyAndSharedVerticesWithProfiling) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(12);
    const auto schedule = make_schedule({{0, 1, 2, 3, 4, 5, 6, 7}}, {{0}});
    const auto cache = make_cache({{0, 0}, {64, 63}, {1, 0}, {257, 256},
                                   {10, 5}, {83, 80}, {0, 0}, {200, 199}});
    const std::vector<std::vector<int>> dependencies =
        {{}, {0}, {1}, {2}, {3}, {4}, {5}, {6}};
    SyntheticModel expected(cache, dependencies), actual(cache, dependencies);
    solver_detail::ClothGridContactSweep sweep;
    sweep.prepare(schedule, cache);
    ASSERT_EQ(sweep.cooperative_cells, (std::vector<int>{0}));
    ASSERT_EQ(sweep.assignments[0].lanes, 4);
    for (int iteration = 0; iteration < 12; ++iteration) {
        expected.reset_sweep();
        actual.reset_sweep();
        run_scalar(schedule, expected);
        const bool profile = (iteration % 2) == 0;
        run_sweep(sweep, schedule, actual, profile);
        expect_bitwise_equal(expected.x, actual.x);
        EXPECT_EQ(actual.dependency_violations.load(), 0);
        EXPECT_EQ(actual.certificate_mismatches.load(), 0);
        if (profile) {
            ASSERT_EQ(sweep.batch_stats.size(), 1u);
            const auto& stats = sweep.batch_stats[0];
            EXPECT_EQ(stats.cooperative_cells, 1);
            EXPECT_GT(stats.wall_seconds, 0.0);
            EXPECT_GE(stats.worker_busy_seconds, 0.0);
            EXPECT_GE(stats.group_wait_seconds, 0.0);
            EXPECT_GE(stats.barrier_wait_seconds, 0.0);
            EXPECT_GT(stats.max_cell_seconds, 0.0);
            EXPECT_LE(stats.max_cell_seconds, stats.wall_seconds);
        } else {
            EXPECT_TRUE(sweep.batch_stats.empty());
        }
    }
}

TEST(ClothGridContactSweep, LeaderOnlyFailureInsideCooperativeCellJoinsAllTwelveWorkers) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    omp_set_num_threads(12);
    const auto schedule = make_schedule({{0, 1, 2, 3}, {4}}, {{0}, {1}});
    const auto cache = make_cache({{200, 199}, {1, 0}, {257, 256}, {0, 0}, {0, 0}});
    solver_detail::ClothGridContactSweep sweep;
    sweep.prepare(schedule, cache);
    ASSERT_EQ(sweep.assignments[0].lanes, 4);
    for (int failed_vertex : {1, 2}) {
        std::array<std::atomic<int>, 5> committed{};
        for (auto& count : committed) count.store(0, std::memory_order_relaxed);
        const auto invoke = [&] {
            sweep.run(schedule,
                [&](int v, int j, solver_detail::ContactContribution&) {
                    if (v == failed_vertex && j == 17)
                        throw std::runtime_error("mixed cell compute failure");
                    return 0u;
                },
                [&](int, const solver_detail::ContactContribution*,
                    const solver_detail::ContactMaskWord*) {},
                [&](int v) {
                    if (v == failed_vertex)
                        throw std::runtime_error("mixed cell leader failure");
                    committed[v].fetch_add(1, std::memory_order_relaxed);
                },
                [&](int, int, solver_detail::ContactContribution&, bool) { return false; },
                [&](int v, const solver_detail::ContactContribution*,
                    const solver_detail::ContactMaskWord*) {
                    committed[v].fetch_add(1, std::memory_order_relaxed);
                }, true);
        };
        EXPECT_THROW(invoke(), std::runtime_error);
        for (int v = 0; v < failed_vertex; ++v) EXPECT_EQ(committed[v].load(), 1);
        for (int v = failed_vertex; v < 5; ++v) EXPECT_EQ(committed[v].load(), 0);
    }
}

TEST(ClothGridContactSweep, CallbackFailuresJoinCurrentVertexAndSkipLaterBatches) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);

    enum class Failure { Compute, Apply, Baseline, CCD, Commit };
    const auto schedule = make_schedule({{0, 1, 2}, {3}}, {{0}, {1}});
    const auto cache = make_cache({{80, 65}, {129, 128}, {75, 70}, {72, 73}});

    for (Failure failure : {Failure::Compute, Failure::Apply, Failure::Baseline,
                            Failure::CCD, Failure::Commit}) {
        omp_set_num_threads(4);
        solver_detail::ClothGridContactSweep sweep;
        sweep.prepare(schedule, cache);
        // A mismatched actual team deliberately exercises the whole-cell
        // fallback's exception path; the other phases use cooperative sharing.
        if (failure == Failure::Baseline) omp_set_num_threads(1);

        std::unique_ptr<std::atomic<int>[]> committed(new std::atomic<int>[4]);
        for (int v = 0; v < 4; ++v) committed[v].store(0, std::memory_order_relaxed);
        const auto invoke = [&] {
            sweep.run(schedule,
                [&](int v, int j, solver_detail::ContactContribution& value) {
                    if (failure == Failure::Compute && v == 1 && j == 17)
                        throw std::runtime_error("compute failure");
                    value.gradient[0] = j;
                    return 1u;
                },
                [&](int v, const solver_detail::ContactContribution*,
                    const solver_detail::ContactMaskWord*) {
                    if (failure == Failure::Apply && v == 1)
                        throw std::runtime_error("apply failure");
                },
                [&](int v) {
                    if (failure == Failure::Baseline && v == 1)
                        throw std::runtime_error("baseline failure");
                    committed[v].fetch_add(1, std::memory_order_relaxed);
                },
                [&](int v, int j, solver_detail::ContactContribution&, bool) {
                    if (failure == Failure::CCD && v == 1 && j == 17)
                        throw std::runtime_error("ccd failure");
                    return false;
                },
                [&](int v, const solver_detail::ContactContribution*,
                    const solver_detail::ContactMaskWord*) {
                    if (failure == Failure::Commit && v == 1)
                        throw std::runtime_error("commit failure");
                    committed[v].fetch_add(1, std::memory_order_relaxed);
                });
        };

        EXPECT_THROW(invoke(), std::runtime_error);
        EXPECT_EQ(committed[0].load(), 1);
        EXPECT_EQ(committed[1].load(), 0);
        EXPECT_EQ(committed[2].load(), 0);
        EXPECT_EQ(committed[3].load(), 0) << "later batch ran after failure";
    }
}
