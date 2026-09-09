#include "colored_contact_sweep.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

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
