#include "contact_scheduling.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <thread>

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
            for (int reference = 0; reference < 3; ++reference) {
                auto &x = reference == 0 ? expected : actual;
                x.assign(color.size(), 1.0);
                std::atomic<int> assigned_calls{0};
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
                        if (reference == 1) sweep.run(groups, compute, apply, whole, ccd, commit);
                        else {
                            const auto assigned = [&](const solver_detail::ColoredContactSweep::Assignment& assignment,
                                solver_detail::ContactContribution* values,
                                solver_detail::ContactMaskWord* masks) {
                                ++assigned_calls;
                                constexpr int grain = solver_detail::contact_grain;
                                for (int start = assignment.lane * grain; start < assignment.count;
                                     start += assignment.lanes * grain) {
                                    unsigned bits = 0, clear_bits = 0;
                                    for (int j = start; j < std::min(start + grain, assignment.count); ++j) {
                                        const unsigned flags = compute(assignment.vertex, j, values[j]);
                                        bits |= (flags & 1u) << (j - start);
                                        clear_bits |= ((flags >> 1) & 1u) << (j - start);
                                    }
                                    masks[start / grain] = {bits, clear_bits};
                                }
                            };
                            sweep.run_assigned(groups, compute, apply, whole, ccd, commit, nullptr, assigned);
                        }
                        omp_set_num_threads(threads);
                    }
                }
                if (reference != 0) EXPECT_EQ(0,
                    std::memcmp(expected.data(), actual.data(), expected.size()*sizeof(double)));
                if (reference == 2 && threads > 1) EXPECT_GT(assigned_calls.load(), 0);
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

namespace {
struct RestoreColoredSweepThreads {
    int threads = omp_get_max_threads(), dynamic = omp_get_dynamic();
    RestoreColoredSweepThreads() { omp_set_dynamic(0); }
    ~RestoreColoredSweepThreads() {
        omp_set_num_threads(threads);
        omp_set_dynamic(dynamic);
    }
};
std::vector<int> colored_sweep_teams() {
    const int native = std::max(1, std::min(64, omp_get_num_procs()));
    // Repeated large/small teams reuse barrier storage; nine workers also
    // exercise a partial final group on the native 64-thread server.
    return {1, std::min(2, native), std::min(8, native), native,
        std::min(9, native), std::min(3, native), 1, native};
}
struct ColoredSweepReference {
    std::vector<std::vector<int>> groups;
    std::vector<int> color;
    explicit ColoredSweepReference(const std::vector<int>& sizes) {
        for (int size : sizes) {
            const int c = static_cast<int>(groups.size());
            groups.emplace_back();
            for (int local = 0; local < size; ++local) {
                groups.back().push_back(static_cast<int>(color.size()));
                color.push_back(c);
            }
        }
    }
    std::vector<double> initial() const {
        std::vector<double> values(color.size());
        for (std::size_t v = 0; v < values.size(); ++v)
            values[v] = 0.0625 * (v % 23) + 0.0125;
        return values;
    }
    void update(std::vector<double>& values, int v) const {
        double value = values[v] * 0.25 + 0.125;
        // Every other color is a dependency, so moving across any color
        // barrier changes the reference arithmetic. Same-color nodes remain
        // independent and may execute in either first-wave or tail order.
        for (int c = 0; c < static_cast<int>(groups.size()); ++c)
            if (c != color[v])
                for (int other : groups[c]) value += values[other] / (1.0 + other % 13);
        values[v] = std::fmod(value, 64.0) + 0.000125 * v;
    }
    void run(std::vector<double>& values, int sweeps) const {
        for (int sweep = 0; sweep < sweeps; ++sweep)
            for (const auto& group : groups)
                for (int v : group) update(values, v);
    }
};
void expect_colored_sweep_equal(const std::vector<double>& actual,
                              const std::vector<double>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t v = 0; v < actual.size(); ++v) {
        ASSERT_TRUE(std::isfinite(actual[v])) << "vertex=" << v;
        EXPECT_EQ(0, std::memcmp(&actual[v], &expected[v], sizeof(double))) << "vertex=" << v;
    }
}
} // namespace

TEST(ColoredVertexSweep, PreservesEveryColorBarrierAndExactArithmeticAcrossTeams) {
    RestoreColoredSweepThreads restore;
    solver_detail::ColoredVertexSweep scheduler;
    const std::vector<std::vector<int>> configurations = {
        {0, 1, 3, 7, 8, 9, 15, 16, 17, 0},
        {0, 63, 64, 65, 129, 0},
        {1}, {0, 0, 0}, {5, 131, 2, 35}
    };
    for (int threads : colored_sweep_teams()) {
        omp_set_num_threads(threads);
        for (std::size_t scene = 0; scene < configurations.size(); ++scene) {
            SCOPED_TRACE(::testing::Message() << "threads=" << threads << " scene=" << scene);
            ColoredSweepReference reference(configurations[scene]);
            auto expected = reference.initial(), actual = expected;
            const int vertices = static_cast<int>(actual.size());
            auto generations = std::make_unique<std::atomic<int>[]>(vertices);
            for (int v = 0; v < vertices; ++v) generations[v].store(0);
            std::atomic<bool> ordered{true};
            std::atomic<int> calls{0};
            int total = 0;
            for (int sweeps : {1, 3, 2}) {
                SCOPED_TRACE(::testing::Message() << "sweeps=" << sweeps);
                reference.run(expected, sweeps);
                scheduler.run(reference.groups, sweeps, [&](int v) {
                    const int generation = generations[v].load() + 1;
                    for (int c = 0; c < static_cast<int>(reference.groups.size()); ++c)
                        if (c != reference.color[v])
                            for (int other : reference.groups[c]) {
                                const int required = c < reference.color[v] ? generation : generation - 1;
                                if (generations[other].load(std::memory_order_acquire) != required)
                                    ordered.store(false);
                            }
                    reference.update(actual, v);
                    generations[v].store(generation, std::memory_order_release);
                    calls.fetch_add(1);
                });
                total += sweeps;
                EXPECT_TRUE(ordered.load());
                EXPECT_EQ(calls.load(), vertices * total);
                for (int v = 0; v < vertices; ++v) EXPECT_EQ(generations[v].load(), total);
                expect_colored_sweep_equal(actual, expected);
            }
        }
    }
}

TEST(ColoredVertexSweep, JoinsThrowingColorSkipsLaterColorsAndReusesTheWorkspace) {
    RestoreColoredSweepThreads restore;
    solver_detail::ColoredVertexSweep scheduler;
    const ColoredSweepReference reference({0, 5, 67, 9, 0});
    for (int threads : colored_sweep_teams()) {
        omp_set_num_threads(threads);
        for (int failing_vertex : {0, 36, 80}) {
            SCOPED_TRACE(::testing::Message() << "threads=" << threads
                << " failing_vertex=" << failing_vertex);
            std::atomic<int> active{0};
            std::atomic<bool> later_color{false};
            EXPECT_THROW(scheduler.run(reference.groups, 3, [&](int v) {
                struct Guard {
                    std::atomic<int>& active;
                    explicit Guard(std::atomic<int>& a) : active(a) { active.fetch_add(1); }
                    ~Guard() { active.fetch_sub(1); }
                } guard(active);
                if (reference.color[v] > reference.color[failing_vertex]) later_color.store(true);
                if (v == failing_vertex) throw std::runtime_error("injected color failure");
            }), std::runtime_error);
            EXPECT_EQ(active.load(), 0);
            EXPECT_FALSE(later_color.load());
            auto expected = reference.initial(), actual = expected;
            reference.run(expected, 3);
            scheduler.run(reference.groups, 3, [&](int v) { reference.update(actual, v); });
            expect_colored_sweep_equal(actual, expected);
        }
    }
}

TEST(ColoredVertexSweep, EmptyScheduleAndZeroSweepsDoNoWork) {
    RestoreColoredSweepThreads restore;
    solver_detail::ColoredVertexSweep scheduler;
    int calls = 0;
    scheduler.run({}, 3, [&](int) { ++calls; });
    scheduler.run({{0, 1}}, 0, [&](int) { ++calls; });
    EXPECT_EQ(calls, 0);
}

namespace {
// Model a color-start geometry snapshot: inactive neighbors must reflect all
// preceding updates, while every node in this color sees the same snapshot.
struct ColorPreparationProbe {
    const ColoredSweepReference& reference;
    std::vector<double> live, gathered;
    std::vector<int> visits, current_color;
    std::atomic<bool> ordered{true};
    std::atomic<int> actual_team{0};

    ColorPreparationProbe(const ColoredSweepReference& ref, int workers)
        : reference(ref), live(ref.initial()),
          gathered(live.size(), std::numeric_limits<double>::quiet_NaN()),
          visits(workers), current_color(workers, -1) {}

    void prepare(std::size_t color) noexcept {
        const int worker = omp_get_thread_num();
        actual_team.store(omp_get_num_threads());
        if (color != static_cast<std::size_t>(visits[worker]) % reference.groups.size())
            ordered.store(false);
        ++visits[worker];
        current_color[worker] = static_cast<int>(color);
#pragma omp for schedule(static)
        for (int v = 0; v < static_cast<int>(live.size()); ++v)
            gathered[v] = live[v];
        // The workshare's implicit barrier must precede every compute call.
    }

    double proposed(int v) {
        if (current_color[omp_get_thread_num()] != reference.color[v]
            || gathered[v] != live[v]) ordered.store(false);
        double value = gathered[v] * 0.25 + 0.125;
        for (int c = 0; c < static_cast<int>(reference.groups.size()); ++c)
            if (c != reference.color[v])
                for (int other : reference.groups[c]) {
                    if (gathered[other] != live[other]) ordered.store(false);
                    value += gathered[other] / (1.0 + other % 13);
                }
        return std::fmod(value, 64.0) + 0.000125 * v;
    }

    void expect_visits(int sweeps) const {
        EXPECT_TRUE(ordered.load());
        ASSERT_GT(actual_team.load(), 0);
        for (int worker = 0; worker < actual_team.load(); ++worker)
            EXPECT_EQ(visits[worker], sweeps * static_cast<int>(reference.groups.size()));
    }
};
} // namespace

TEST(ColoredContactSweep, CollectivePreparationPrecedesCooperationAndFallbacks) {
    RestoreColoredSweepThreads restore;
    const ColoredSweepReference reference({0, 1, 3, 0, 9, 1});
    for (int runtime_threads : {1, 2, 4}) {
        SCOPED_TRACE(runtime_threads);
        omp_set_num_threads(4);
        BroadPhase::Cache cache;
        cache.vertex_nt.resize(reference.color.size());
        cache.vertex_ss.resize(reference.color.size());
        for (int v = 0; v < static_cast<int>(reference.color.size()); ++v)
            if (reference.color[v] != 4) cache.vertex_nt[v].resize(513);
        solver_detail::ColoredContactSweep scheduler;
        scheduler.prepare(reference.groups, cache);
        ASSERT_GT(scheduler.split_count[1], 0); // Several helpers for one node.
        ASSERT_EQ(scheduler.split_count[4], 0); // Independent whole-node work.
        omp_set_num_threads(runtime_threads); // Also exercise prepared-team mismatch.
        ColorPreparationProbe probe(reference, 4);
        std::vector<double> proposed(reference.color.size());
        std::atomic<int> computed{0}, baseline_calls{0};
        auto expected = reference.initial();
        for (int repeat = 0; repeat < 3; ++repeat) {
            reference.run(expected, 1);
            scheduler.run(reference.groups,
                [&](int v, int j, solver_detail::ContactContribution& value) -> unsigned {
                    computed.fetch_add(1);
                    const double next = probe.proposed(v); // Also check helper lanes.
                    if (j != 0) return 0u;
                    value.gradient[0] = next;
                    return 1u;
                },
                [&](int v, const solver_detail::ContactContribution* values,
                    const solver_detail::ContactMaskWord*) { proposed[v] = values[0].gradient[0]; },
                [&](int v) { ++baseline_calls; probe.live[v] = probe.proposed(v); },
                [](int, int, solver_detail::ContactContribution&, bool) { return false; },
                [&](int v, const solver_detail::ContactContribution*,
                    const solver_detail::ContactMaskWord*) { probe.live[v] = proposed[v]; },
                [&](std::size_t color) noexcept { probe.prepare(color); });
            probe.expect_visits(repeat + 1);
            expect_colored_sweep_equal(probe.live, expected);
        }
        EXPECT_GT(baseline_calls.load(), 0);
        if (probe.actual_team.load() == scheduler.team) EXPECT_GT(computed.load(), 0);
        else EXPECT_EQ(computed.load(), 0);
    }
}

TEST(ColoredVertexSweep, CollectivePreparationRefreshesEveryColorAndSweep) {
    RestoreColoredSweepThreads restore;
    const ColoredSweepReference reference({0, 1, 7, 0, 13, 2, 0});
    solver_detail::ColoredVertexSweep scheduler;
    for (int threads : {1, 4}) {
        omp_set_num_threads(threads);
        ColorPreparationProbe probe(reference, threads);
        auto expected = reference.initial();
        int total = 0;
        for (int sweeps : {1, 3, 2}) {
            reference.run(expected, sweeps);
            scheduler.run(reference.groups, sweeps,
                [&](int v) { probe.live[v] = probe.proposed(v); },
                [&](std::size_t color) noexcept { probe.prepare(color); });
            total += sweeps;
            probe.expect_visits(total);
            expect_colored_sweep_equal(probe.live, expected);
        }
    }
}

TEST(ColoredVertexSweep, CollectivePreparationDrainsAfterFailureAndAllowsReuse) {
    RestoreColoredSweepThreads restore;
    const ColoredSweepReference reference({0, 1, 0, 9, 2, 0});
    solver_detail::ColoredVertexSweep scheduler;
    for (int threads : {1, 4}) {
        omp_set_num_threads(threads);
        ColorPreparationProbe probe(reference, threads);
        std::atomic<int> processed{0};
        EXPECT_THROW(scheduler.run(reference.groups, 3,
            [&](int v) {
                ++processed;
                if (v == 0) throw std::runtime_error("after collective preparation");
            }, [&](std::size_t color) noexcept { probe.prepare(color); }), std::runtime_error);
        // Every worker must enter every callback, even while draining later
        // colors/sweeps after the first node fails. Otherwise its omp for hangs.
        EXPECT_EQ(processed.load(), 1);
        probe.expect_visits(3);
        auto expected = reference.initial();
        reference.run(expected, 2);
        scheduler.run(reference.groups, 2,
            [&](int v) { probe.live[v] = probe.proposed(v); },
            [&](std::size_t color) noexcept { probe.prepare(color); });
        probe.expect_visits(5);
        expect_colored_sweep_equal(probe.live, expected);
    }
}

TEST(ColoredVertexSweep, ExceptionsAtColorAndSweepBoundariesAlwaysJoinTheTeam) {
    RestoreColoredSweepThreads restore;
    solver_detail::ColoredVertexSweep scheduler;
    const int native = std::max(1, std::min(64, omp_get_num_procs()));
    std::vector<int> teams{1, std::min(2, native), std::min(9, native), native};
    std::sort(teams.begin(), teams.end());
    teams.erase(std::unique(teams.begin(), teams.end()), teams.end());
    const int wide = 2 * native + 3;
    struct Case {
        std::vector<int> sizes;
        int failing_vertex;
        int failing_sweep;
    };
    const std::vector<Case> cases = {
        {{0, 1, 0, 1, 0}, 0, 0},
        {{wide, 1, 0, 3, 0}, wide, 0},
        {{1, wide, 0, 3, 0}, 0, 1},
        {{0, wide, 0, 1, 0, 3}, wide, 2}
    };
    for (int threads : teams) {
        omp_set_num_threads(threads);
        for (std::size_t scenario = 0; scenario < cases.size(); ++scenario) {
            const auto& item = cases[scenario];
            const ColoredSweepReference reference(item.sizes);
            const int vertices = static_cast<int>(reference.color.size());
            const int failing_color = reference.color[item.failing_vertex];
            ASSERT_EQ(reference.groups[failing_color].size(), 1u);
            // An immediate exception in the next color must not let a worker
            // still leaving the preceding color abandon the team. Empty and
            // uneven colors, sweep transitions, and varied arrival timing
            // repeatedly expose that asynchronous boundary without hooks into
            // the scheduler or synchronization inside the callbacks.
            for (int repeat = 0; repeat < 64; ++repeat) {
                SCOPED_TRACE(::testing::Message() << "threads=" << threads
                    << " scenario=" << scenario << " repeat=" << repeat);
                auto calls = std::make_unique<std::atomic<int>[]>(vertices);
                for (int v = 0; v < vertices; ++v) calls[v].store(0);
                EXPECT_THROW(scheduler.run(reference.groups, 4, [&](int v) {
                    const int visit = calls[v].fetch_add(1);
                    if (v == item.failing_vertex && visit == item.failing_sweep)
                        throw std::runtime_error("boundary failure");
                    if ((v + repeat) % 17 == 0) std::this_thread::yield();
                }), std::runtime_error);
                for (int v = 0; v < vertices; ++v) {
                    const int expected = item.failing_sweep
                        + (reference.color[v] <= failing_color ? 1 : 0);
                    EXPECT_EQ(calls[v].load(), expected) << "vertex=" << v;
                }
            }
            // Reuse after repeated failures, including all of the previously
            // skipped colors and sweeps, without reconstructing the workspace.
            auto actual = reference.initial(), expected = actual;
            reference.run(expected, 3);
            scheduler.run(reference.groups, 3,
                [&](int v) { reference.update(actual, v); });
            expect_colored_sweep_equal(actual, expected);
        }
    }
}

TEST(OrderedContactTasks, BatchedSweepsPreserveHelpersDependenciesAndJoinedProxyTasks) {
    RestoreColoredSweepThreads restore;
    const int native = std::max(1, std::min(64, omp_get_num_procs()));
    const ColoredSweepReference reference({0, 1, 3, 0, 2 * native + 3, 1});
    constexpr int proxies = 131, contacts = 257;
    for (int threads : {1, std::min(2, native), std::min(9, native), native, 1, native}) {
        SCOPED_TRACE(threads);
        omp_set_num_threads(threads);
        auto expected = reference.initial(), actual = expected;
        std::vector<double> positions(actual.size() * proxies);
        for (std::size_t v = 0; v < actual.size(); ++v)
            std::fill_n(positions.begin() + v * proxies, proxies, actual[v]);
        std::atomic<bool> joined{true};
        std::atomic<int> helpers{0}, actual_team{0}, calls{0};
        int total = 0;
        for (int sweeps : {1, 3, 2}) {
            for (int sweep = 0; sweep < sweeps; ++sweep)
                for (const auto& group : reference.groups)
                    for (int v : group) {
                        reference.update(expected, v);
                        for (int i = 0; i < contacts; ++i)
                            expected[v] += i % 3 == 0 ? 1e16 : (i % 3 == 1 ? 1.0 : -1e16);
                    }
            solver_detail::for_each_colored_block(reference.groups,
                [](int v) -> std::size_t { return v < 4 ? 4096 : 0; },
                [&](int v, bool cooperative) {
                    actual_team.store(omp_get_num_threads());
                    if (cooperative) helpers.fetch_add(1);
                    for (int other = 0; other < static_cast<int>(actual.size()); ++other)
                        if (reference.color[other] != reference.color[v])
                            for (int i : {0, proxies - 1})
                                if (positions[other * proxies + i] != actual[other]) joined.store(false);
                    reference.update(actual, v);
                    solver_detail::ordered_contact_tasks(contacts, cooperative,
                        [](int i) { return i % 3 == 0 ? 1e16 : (i % 3 == 1 ? 1.0 : -1e16); },
                        [&](double term) { actual[v] += term; });
                    const double value = actual[v];
                    // Like rigid translation/placement, this implicit taskgroup
                    // must finish every proxy write before the next color.
#pragma omp taskloop grainsize(32) firstprivate(v, value) shared(positions)
                    for (int i = 0; i < proxies; ++i) positions[v * proxies + i] = value;
                    calls.fetch_add(1);
                }, sweeps);
            total += sweeps;
            EXPECT_TRUE(joined.load());
            EXPECT_EQ(calls.load(), total * static_cast<int>(actual.size()));
            expect_colored_sweep_equal(actual, expected);
            for (std::size_t v = 0; v < actual.size(); ++v)
                for (int i = 0; i < proxies; ++i) EXPECT_EQ(positions[v * proxies + i], actual[v]);
        }
        if (threads > 1 && actual_team.load() == threads) EXPECT_GT(helpers.load(), 0);
    }
}

TEST(OrderedContactTasks, BatchedExceptionsDrainHelpersAtSweepBoundariesAndAllowReuse) {
    RestoreColoredSweepThreads restore;
    const int native = std::max(1, std::min(64, omp_get_num_procs()));
    const std::vector<std::vector<int>> groups{{}, {0}, {}, {1, 2}, {3}, {}};
    for (int threads : {1, std::min(2, native), std::min(9, native), native}) {
        omp_set_num_threads(threads);
        for (bool contact_failure : {false, true}) {
            for (int repeat = 0; repeat < 32; ++repeat) {
                SCOPED_TRACE(::testing::Message() << threads << ',' << contact_failure << ',' << repeat);
                std::atomic<int> active{0};
                std::vector<int> visits(4);
                const auto run = [&](bool fail) {
                    solver_detail::for_each_colored_block(groups,
                        [](int) { return std::size_t(4096); },
                        [&](int block, bool cooperative) {
                            const bool inject = fail && block == 0 && visits[block] == 1;
                            ++visits[block];
                            if (inject && !contact_failure) throw std::runtime_error("leader failure");
                            solver_detail::ordered_contact_tasks(257, cooperative,
                                [&](int i) {
                                    struct Guard {
                                        std::atomic<int>& count;
                                        explicit Guard(std::atomic<int>& value) : count(value) { ++count; }
                                        ~Guard() { --count; }
                                    } guard(active);
                                    if (inject && i == 17) throw std::runtime_error("contact failure");
                                    return i;
                                }, [](int) {});
                        }, 4);
                };
                EXPECT_THROW(run(true), std::runtime_error);
                EXPECT_EQ(active.load(), 0);
                EXPECT_EQ(visits, (std::vector<int>{2, 1, 1, 1}));
                std::fill(visits.begin(), visits.end(), 0);
                EXPECT_NO_THROW(run(false));
                EXPECT_EQ(visits, (std::vector<int>{4, 4, 4, 4}));
                EXPECT_EQ(active.load(), 0);
            }
        }
    }
}
