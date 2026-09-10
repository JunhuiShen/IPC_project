#include "contact_scheduling.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

TEST(OrderedContactTasks, LeaderWorkPrecedesOrderedAccumulationForEveryPath) {
    struct Restore { int threads = omp_get_max_threads(); ~Restore() { omp_set_num_threads(threads); } } restore;
    const std::vector<std::vector<int>> groups = {{0}, {1, 2}};
    for (int count : {0, 17, 257}) {
        std::vector<double> reference;
        for (int threads : {1, 2, 8, 64}) {
            omp_set_num_threads(threads);
            std::vector<double> values(3);
            std::vector<int> calls(3);
            solver_detail::for_each_colored_block(groups,
                [](int) { return std::size_t(257); },
                [&](int block, bool cooperative) {
                    double sum = std::numeric_limits<double>::quiet_NaN();
                    const std::function<void()> leader_work = [&] {
                        ++calls[block];
                        sum = block == 0 ? 1.0 : values[0];
                    };
                    solver_detail::ordered_contact_tasks(count, cooperative,
                        [](int i) { return i % 3 == 0 ? 1e16 : (i % 3 == 1 ? 1.0 : -1e16); },
                        [&](double value) { sum += value; }, &leader_work);
                    values[block] = sum;
                });
            EXPECT_EQ(calls, (std::vector<int>{1, 1, 1}));
            if (threads == 1) reference = values;
            else EXPECT_EQ(0, std::memcmp(reference.data(), values.data(), values.size() * sizeof(double)));
        }
    }
}

TEST(OrderedContactTasks, LeaderFailureJoinsHelpersBeforeRethrowing) {
    struct Restore { int threads = omp_get_max_threads(); ~Restore() { omp_set_num_threads(threads); } } restore;
    for (int threads : {1, 8, 64}) {
        omp_set_num_threads(threads);
        std::atomic<int> active{0};
        int accumulated = 0, leader_calls = 0;
        bool visited_next_color = false;
        const auto run = [&] {
            solver_detail::for_each_colored_block(std::vector<std::vector<int>>{{0}, {1}},
                [](int) { return std::size_t(257); },
                [&](int block, bool cooperative) {
                    if (block == 1) visited_next_color = true;
                    const std::function<void()> leader_work = [&] {
                        ++leader_calls;
                        throw std::logic_error("leader failure");
                    };
                    solver_detail::ordered_contact_tasks(257, cooperative,
                        [&](int i) {
                            struct Active {
                                std::atomic<int>& count;
                                Active(std::atomic<int>& c) : count(c) { ++count; }
                                ~Active() { --count; }
                            } guard(active);
                            if (i == 3 || i == 17) throw std::runtime_error("contact failure");
                            return i;
                        }, [&](int) { ++accumulated; }, &leader_work);
                });
        };
        EXPECT_THROW(run(), std::logic_error);
        EXPECT_EQ(active.load(), 0);
        EXPECT_EQ(accumulated, 0);
        EXPECT_EQ(leader_calls, 1);
        EXPECT_FALSE(visited_next_color);
    }
}

TEST(OrderedContactTasks, PreservesContactOrderAndColorDependencies) {
    const int saved = omp_get_max_threads();
    const std::vector<std::vector<int>> groups = {{0}, {1, 2}, {3, 4, 5, 6, 7, 8, 9, 10}};
    std::vector<double> reference;
    for (int threads : {1, 2, 3, 5, 7, 8, 64}) {
        omp_set_num_threads(threads);
        std::vector<double> values(11, 0.0);
        solver_detail::for_each_colored_block(groups,
            [](int block) -> std::size_t { return block == 3 ? 8192 : 257; },
            [&](int block, bool cooperative) {
                // Depend only on preceding colors, with cancellation that
                // detects a reassociated or completion-order reduction.
                double sum = block == 0 ? 1.0 : (block < 3 ? values[0] : values[1] + values[2]);
                const int count = block == 3 ? 8192 : 257;
                solver_detail::ordered_contact_tasks(count, cooperative,
                    [](int i) { return i % 3 == 0 ? 1e16 : (i % 3 == 1 ? 1.0 : -1e16); },
                    [&](double contribution) { sum += contribution; });
                values[block] = sum;
            });
        if (threads == 1) reference = values;
        else EXPECT_EQ(0, std::memcmp(reference.data(), values.data(), values.size() * sizeof(double)));
    }
    omp_set_num_threads(saved);
}

TEST(OrderedContactTasks, RethrowsContactFailureAfterJoiningTasks) {
    const int saved = omp_get_max_threads();
    omp_set_num_threads(4);
    bool visited_next_color = false;
    const auto fail = [&] {
        solver_detail::for_each_colored_block(
            std::vector<std::vector<int>>{{0}, {1}}, [](int) { return 256; },
            [&](int block, bool cooperative) {
                if (block == 1) visited_next_color = true;
            solver_detail::ordered_contact_tasks(256, cooperative,
                [](int i) {
                    if (i == 3 || i == 17) throw std::runtime_error("contact failure");
                    return i;
                }, [](int) {});
        });
    };
    EXPECT_THROW(fail(), std::runtime_error);
    EXPECT_FALSE(visited_next_color);
    omp_set_num_threads(saved);
}

// Reuse the same call site across empty colors, changing teams, several
// contact stages per block, and both small and large colors with outliers.
TEST(OrderedContactTasks, ReusesHelpersAcrossStagesAndSchedulingChanges) {
    struct Restore {
        int threads = omp_get_max_threads(), dynamic = omp_get_dynamic();
        ~Restore() { omp_set_num_threads(threads); omp_set_dynamic(dynamic); }
    } restore;
    omp_set_dynamic(0);
    const std::vector<std::vector<int>> groups = {
        {}, {0}, {1, 2, 3}, {}, {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
    };
    std::vector<double> expected;
    for (int threads : {1, 8, 2, 64, 7, 3, 4, 1, 8}) {
        omp_set_num_threads(threads);
        std::vector<double> values(20);
        for (int repeat = 0; repeat < 5; ++repeat) {
            solver_detail::for_each_colored_block(groups,
                [repeat](int block) -> std::size_t {
                    return (block == 0 || block == 7) ? 8192 + repeat : 257;
                }, [&](int block, bool cooperative) {
                    double sum = block == 0 ? 1.0 : (block < 4 ? values[0] : values[1] + values[2]);
                    for (int stage = 0; stage < 5; ++stage) {
                        const int count = stage == 2 ? 0 :
                            ((block == 0 || block == 7) ? 8192 + repeat : 257);
                        solver_detail::ordered_contact_tasks(count, cooperative,
                            [stage](int i) { return i % 3 == 0 ? 1e16 : (i % 3 == 1 ? double(stage + 1) : -1e16); },
                            [&](double contribution) { sum += contribution; });
                    }
                    values[block] = sum;
                });
        }
        if (expected.empty()) expected = values;
        else EXPECT_EQ(0, std::memcmp(expected.data(), values.data(), values.size() * sizeof(double)))
            << "threads=" << threads;
    }
}

TEST(ColoredContactSweep, PreservesOrderedArithmeticDependenciesAndTeamChanges) {
    struct Restore {
        int threads = omp_get_max_threads(), dynamic = omp_get_dynamic();
        ~Restore() {
            omp_set_num_threads(threads);
            omp_set_dynamic(dynamic);
        }
    } restore;
    omp_set_dynamic(0);
    for (int threads : {1, 2, 8, 16})
        for (bool change_team : {false, true}) {
            omp_set_num_threads(threads);
            std::vector<std::vector<int>> groups(9);
            std::vector<int> color;
            for (int c = 1; c < 9; ++c)
                for (int i = 0; i < c; ++i) {
                    groups[c].push_back(color.size());
                    color.push_back(c);
                }
            std::vector<double> expected(color.size(), 1), actual = expected;
            for (int reference = 0; reference < 2; ++reference) {
                auto &x = reference == 0 ? expected : actual;
                BroadPhase::Cache cache;
                cache.vertex_nt.resize(color.size());
                cache.vertex_ss.resize(color.size());
                std::vector<double> step(x.size());
                solver_detail::ColoredContactSweep sweep;
                auto active = [](int j) {
                    return j == 0 || j % 5 == 2 || j % 5 == 3 || j % 5 == 4;
                };
                auto clear = [](int j) { return j % 5 == 1; };
                auto compute = [&](int v, int j,
                                   solver_detail::ContactContribution &value) -> unsigned {
                    if (!active(j))
                        return clear(j) ? 2u : 0u;
                    double r = x[v] + j;
                    for (int u = 0; u < (int)x.size(); ++u)
                        if (color[u] != color[v])
                            r += x[u];
                    // Reordering the active additions changes rounding.
                    value.gradient[0] = j % 5 == 2   ? 1e16
                                        : j % 5 == 4 ? -1e16
                                                     : std::fmod(r, 1048576.);
                    return 1u;
                };
                auto apply = [&](int v, const solver_detail::ContactContribution *values,
                                 const solver_detail::ContactMaskWord *mask) {
                    double r = values[0].gradient[0];
                    solver_detail::for_active_contact(mask, cache.vertex_nt[v].size(),
                                                      [&](int j) { r += values[j].gradient[0]; });
                    step[v] = std::fmod(r, 1048576.);
                };
                auto ccd = [&](int v, int j, solver_detail::ContactContribution &value,
                               bool certified) {
                    EXPECT_EQ(certified, clear(j));
                    if (j == 0 || certified || j % 3 != 0)
                        return false;
                    value.toi = std::fmod(step[v] + j, 1024.);
                    return true;
                };
                auto commit = [&](int v, const solver_detail::ContactContribution *values,
                                  const solver_detail::ContactMaskWord *mask) {
                    double r = 0;
                    solver_detail::for_active_contact(mask, cache.vertex_nt[v].size(),
                                                      [&](int j) { r += values[j].toi; });
                    x[v] = std::fmod(r + step[v], 1048576.);
                };
                // Independent scalar reference: no masks or scheduler helpers.
                auto whole = [&](int v) {
                    int n = 1 + cache.vertex_nt[v].size();
                    std::vector<solver_detail::ContactContribution> values(n);
                    for (int j = 0; j < n; ++j)
                        compute(v, j, values[j]);
                    double r = values[0].gradient[0];
                    for (int j = 1; j < n; ++j)
                        if (active(j))
                            r += values[j].gradient[0];
                    step[v] = std::fmod(r, 1048576.);
                    r = 0;
                    for (int j = 1; j < n; ++j)
                        if (ccd(v, j, values[j], clear(j)))
                            r += values[j].toi;
                    x[v] = std::fmod(r + step[v], 1048576.);
                };
                for (int iter = 0; iter < 12; ++iter) {
                    if (iter % 4 == 0) {
                        for (int v = 0; v < (int)x.size(); ++v)
                            cache.vertex_nt[v].resize(513 + (v * 317 + iter * 19) % 1009);
                        sweep.prepare(groups, cache);
                    }
                    if (reference == 0)
                        for (const auto &g : groups)
                            for (int v : g)
                                whole(v);
                    else {
                        omp_set_num_threads(change_team && iter % 3 == 0 ? 1 : threads);
                        sweep.run(groups, compute, apply, whole, ccd, commit);
                        omp_set_num_threads(threads);
                    }
                }
            }
            EXPECT_EQ(0,
                      std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(double)))
                << "threads=" << threads << " change_team=" << change_team;
        }
}

TEST(OrderedContactTasks, NestedContactEvaluationPreservesArithmeticOrder) {
    struct Restore { int threads = omp_get_max_threads(); ~Restore() { omp_set_num_threads(threads); } } restore;
    std::vector<double> expected;
    for (int threads : {1, 3, 8, 64}) {
        omp_set_num_threads(threads);
        std::vector<double> values(2);
        solver_detail::for_each_colored_block(std::vector<std::vector<int>>{{0,1}},
            [](int) { return std::size_t(129); },
            [&](int block, bool cooperative) {
                double sum = 0;
                solver_detail::ordered_contact_tasks(129, cooperative,
                    [&](int i) {
                        double inner = 0;
                        solver_detail::ordered_contact_tasks(65, cooperative,
                            [i](int j) { return j % 3 == 0 ? 1e16 : (j % 3 == 1 ? double(i + 1) : -1e16); },
                            [&](double value) { inner += value; });
                        return inner;
                    }, [&](double value) { sum += value; });
                values[block] = sum;
            });
        if (expected.empty()) expected = values;
        else EXPECT_EQ(0, std::memcmp(expected.data(), values.data(), values.size() * sizeof(double)));
    }
}
