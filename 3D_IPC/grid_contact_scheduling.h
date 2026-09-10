#pragma once

#include "contact_scheduling.h"
#include "grid_coloring.h"

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <thread>

namespace solver_detail {

// Share the contacts of one vertex among a fixed worker group. The group's
// leader commits that vertex before any worker visits the next vertex in the
// cell. Different groups and whole-cell workers only visit cells in the same
// conflict-free grid batch; no nested OpenMP teams or per-vertex tasks are used.
struct ClothGridContactSweep {
    struct Assignment {
        int cell = -1, lane = 0, lanes = 1;
        std::size_t offset = 0, mask_offset = 0;
    };
    struct alignas(64) State {
        std::atomic<int> arrived{0};
        std::atomic<std::uint64_t> ready{0};
        std::atomic<std::uint64_t> failed_phase{0};
    };
    struct BatchStats {
        double wall_seconds = 0.0;
        double worker_busy_seconds = 0.0;
        double group_wait_seconds = 0.0;
        double barrier_wait_seconds = 0.0;
        double max_cell_seconds = 0.0;
        int cooperative_cells = 0;
    };

    int prepared_team = 0;
    std::vector<int> cooperative_cells;
    std::vector<Assignment> assignments;
    std::vector<int> split_count;
    std::vector<BatchStats> batch_stats;

    void prepare(const ClothGridSchedule& schedule, const BroadPhase::Cache& cache) {
        prepared_team = std::max(1, omp_get_max_threads());
        const std::size_t batch_count = schedule.batches.size();
        assignments.assign(batch_count * prepared_team, Assignment{});
        split_count.assign(batch_count, 0);
        cooperative_cells.clear();
        batch_stats.clear();
        states = std::make_unique<State[]>(schedule.cells.size());
        next_whole = std::make_unique<std::atomic<int>[]>(batch_count);
        vertex_counts.resize(cache.vertex_nt.size());
        for (std::size_t v = 0; v < vertex_counts.size(); ++v)
            vertex_counts[v] = 1 + static_cast<int>(cache.vertex_nt[v].size())
                + static_cast<int>(cache.vertex_ss[v].size());

        std::vector<std::uint64_t> costs(schedule.cells.size(), 0);
        std::vector<int> maxima(schedule.cells.size(), 0);
        for (std::size_t cell = 0; cell < schedule.cells.size(); ++cell) {
            for (int v : schedule.cells[cell].vertices) {
                costs[cell] += vertex_counts[v];
                maxima[cell] = std::max(maxima[cell], vertex_counts[v]);
            }
        }
        std::size_t maximum_values = 0, maximum_masks = 0;
        for (std::size_t b = 0; b < batch_count; ++b) {
            const auto& batch = schedule.batches[b];
            if (batch.empty() || prepared_team == 1) continue;
            std::uint64_t total_cost = 0;
            for (int cell : batch) total_cost += costs[cell];
            const bool undersubscribed = batch.size() < static_cast<std::size_t>(prepared_team);
            int selected = 0;
            int budget = prepared_team;
            if (undersubscribed) {
                selected = static_cast<int>(batch.size());
            } else {
                // Keep at least a quarter of the team available for dynamic
                // whole-cell work. Only unusually expensive cells justify
                // reserving workers when there are ample independent cells.
                budget = std::max(0, 3 * prepared_team / 4);
                const int limit = std::min(budget / 2,
                    static_cast<int>(batch.size()));
                for (; selected < limit; ++selected) {
                    const int cell = batch[selected];
                    if (maxima[cell] < cooperative_threshold || costs[cell] < 512
                        || static_cast<double>(costs[cell])
                            < 0.75 * static_cast<double>(total_cost) / prepared_team)
                        break;
                }
            }
            if (selected == 0) continue;
            std::vector<int> lanes(selected, 1);
            int used = selected;
            while (used < budget) {
                int best = -1;
                for (int i = 0; i < selected; ++i) {
                    const int cell = batch[i];
                    // Every vertex still has an ordered leader reduction and
                    // commit. Small groups limit join costs and avoid requiring
                    // a large gang of workers to be scheduled simultaneously.
                    const int useful_lanes = maxima[cell] < cooperative_threshold ? 1
                        : std::min(4, (maxima[cell] + contact_grain - 1) / contact_grain);
                    if (lanes[i] >= useful_lanes) continue;
                    if (best < 0 || static_cast<double>(costs[cell]) / lanes[i]
                            > static_cast<double>(costs[batch[best]]) / lanes[best])
                        best = i;
                }
                if (best < 0) break;
                ++lanes[best];
                ++used;
            }
            // If contact work is too small to share, dynamic whole-cell
            // scheduling avoids reserving workers for these cells at all.
            if (used == selected) continue;
            split_count[b] = selected;
            std::size_t offset = 0, mask_offset = 0;
            int worker = 0;
            for (int i = 0; i < selected; ++i) {
                const int cell = batch[i];
                for (int lane = 0; lane < lanes[i]; ++lane)
                    assignments[b * prepared_team + worker++] =
                        {cell, lane, lanes[i], offset, mask_offset};
                if (lanes[i] == 1) continue;
                cooperative_cells.push_back(cell);
                offset += static_cast<std::size_t>(maxima[cell]);
                // Eight contributions span an integral number of cache lines.
                offset = (offset + 7) & ~std::size_t(7);
                mask_offset += (maxima[cell] + contact_grain - 1) / contact_grain;
            }
            maximum_values = std::max(maximum_values, offset);
            maximum_masks = std::max(maximum_masks, mask_offset);
        }
        values.resize(maximum_values);
        masks.resize(maximum_masks);
    }

    template <class Compute, class Apply, class Baseline, class CCD, class Commit>
    void run(const ClothGridSchedule& schedule, const Compute& compute,
             const Apply& apply, const Baseline& baseline, const CCD& ccd,
             const Commit& commit, bool profile = false) {
        if (profile) run_impl<true>(schedule, compute, apply, baseline, ccd, commit);
        else run_impl<false>(schedule, compute, apply, baseline, ccd, commit);
    }

private:
    static constexpr int cooperative_threshold = 128;
    std::vector<int> vertex_counts;
    std::vector<ContactContribution, CacheAlignedAllocator<ContactContribution>> values;
    std::vector<ContactMaskWord> masks;
    std::unique_ptr<State[]> states;
    std::unique_ptr<std::atomic<int>[]> next_whole;

    template <bool Profile, class Compute, class Apply, class Baseline, class CCD, class Commit>
    void run_impl(const ClothGridSchedule& schedule, const Compute& compute,
                  const Apply& apply, const Baseline& baseline, const CCD& ccd,
                  const Commit& commit) {
        const std::size_t batch_count = schedule.batches.size();
        batch_stats.clear();
        if (batch_count == 0) return;
        for (std::size_t b = 0; b < batch_count; ++b)
            next_whole[b].store(split_count[b], std::memory_order_relaxed);
        for (int cell : cooperative_cells) {
            states[cell].arrived.store(0, std::memory_order_relaxed);
            states[cell].ready.store(0, std::memory_order_relaxed);
            states[cell].failed_phase.store(0, std::memory_order_relaxed);
        }
        std::vector<std::exception_ptr> errors(batch_count);
        std::vector<BatchStats> worker_stats;
        // Dynamic teams can be smaller or larger than the team used by prepare.
        // The latter takes the whole-cell fallback, but still needs stats slots.
        const int stats_team = std::max(prepared_team, omp_get_max_threads());
        if constexpr (Profile) worker_stats.resize(batch_count * stats_team);
#pragma omp parallel shared(errors, worker_stats)
        {
            const int worker = omp_get_thread_num();
            for (std::size_t b = 0; b < batch_count; ++b) {
                const auto& batch = schedule.batches[b];
                double begin = 0.0, group_wait = 0.0, max_cell = 0.0;
                int cooperative_count = 0;
                if constexpr (Profile) begin = omp_get_wtime();
                const auto record_error = [&](State* state, std::uint64_t phase = 0) {
                    if (state) {
                        std::uint64_t unset = 0;
                        state->failed_phase.compare_exchange_strong(unset, phase,
                            std::memory_order_release, std::memory_order_relaxed);
                    }
#pragma omp critical(ipc_cloth_grid_contact_error)
                    { if (!errors[b]) errors[b] = std::current_exception(); }
                };
                const auto process_whole = [&](int cell) {
                    double start = 0.0;
                    if constexpr (Profile) start = omp_get_wtime();
                    try {
                        for (int v : schedule.cells[cell].vertices) baseline(v);
                    } catch (...) {
                        record_error(nullptr);
                    }
                    if constexpr (Profile)
                        max_cell = std::max(max_cell, omp_get_wtime() - start);
                };
                if (split_count[b] == 0 || omp_get_num_threads() != prepared_team) {
#pragma omp for schedule(dynamic, 1) nowait
                    for (int i = 0; i < static_cast<int>(batch.size()); ++i)
                        process_whole(batch[i]);
                } else {
                    const Assignment a = assignments[b * prepared_team + worker];
                    if (a.cell >= 0 && a.lanes == 1) {
                        process_whole(a.cell);
                    } else if (a.cell >= 0) {
                        double cell_start = 0.0;
                        if constexpr (Profile) {
                            if (a.lane == 0) {
                                cell_start = omp_get_wtime();
                                ++cooperative_count;
                            }
                        }
                        State& state = states[a.cell];
                        auto* result = values.data() + a.offset;
                        auto* mask = masks.data() + a.mask_offset;
                        const auto wait_until = [&](const auto& ready) {
                            double start = 0.0;
                            if constexpr (Profile) start = omp_get_wtime();
                            unsigned spins = 0, yields = 0;
                            while (!ready()) {
                                if (++spins < 65536) {
                                    contact_spin_hint();
                                } else {
                                    spins = 0;
                                    // Busy workers can otherwise prevent a
                                    // descheduled group member from making
                                    // progress, especially on mixed-speed cores.
                                    if (yields < 4) {
                                        ++yields;
                                        std::this_thread::yield();
                                    } else {
                                        std::this_thread::sleep_for(
                                            std::chrono::microseconds(10));
                                    }
                                }
                            }
                            if constexpr (Profile) group_wait += omp_get_wtime() - start;
                        };
                        std::uint64_t phase = 0;
                        for (int v : schedule.cells[a.cell].vertices) {
                            const int count = vertex_counts[v];
                            phase += 2;
                            if (count < cooperative_threshold) {
                                if (a.lane == 0) {
                                    try { baseline(v); }
                                    catch (...) { record_error(&state, phase); }
                                    state.ready.store(phase, std::memory_order_release);
                                } else {
                                    wait_until([&] {
                                        return state.ready.load(std::memory_order_acquire) >= phase;
                                    });
                                }
                            } else {
                                try {
                                    for (int start = a.lane * contact_grain; start < count;
                                         start += a.lanes * contact_grain) {
                                        unsigned bits = 0, clear = 0;
                                        for (int j = start;
                                             j < std::min(start + contact_grain, count); ++j) {
                                            const unsigned flags = compute(v, j, result[j]);
                                            bits |= (flags & 1u) << (j - start);
                                            clear |= ((flags >> 1) & 1u) << (j - start);
                                        }
                                        mask[start / contact_grain].bits = bits;
                                        mask[start / contact_grain].clear = clear;
                                    }
                                } catch (...) { record_error(&state, phase); }
                                state.arrived.fetch_add(1, std::memory_order_acq_rel);
                                if (a.lane == 0) {
                                    wait_until([&] {
                                        return state.arrived.load(std::memory_order_acquire) == a.lanes;
                                    });
                                    if (!state.failed_phase.load(std::memory_order_acquire)) {
                                        try { apply(v, result, mask); }
                                        catch (...) { record_error(&state, phase); }
                                    }
                                    state.ready.store(phase - 1, std::memory_order_release);
                                } else {
                                    wait_until([&] {
                                        return state.ready.load(std::memory_order_acquire) >= phase - 1;
                                    });
                                }
                                if (!state.failed_phase.load(std::memory_order_acquire)) {
                                    try {
                                        for (int start = a.lane * contact_grain; start < count;
                                             start += a.lanes * contact_grain) {
                                            unsigned bits = 0;
                                            const unsigned clear = mask[start / contact_grain].clear;
                                            for (int j = start;
                                                 j < std::min(start + contact_grain, count); ++j)
                                                if (ccd(v, j, result[j],
                                                        (clear >> (j - start)) & 1u))
                                                    bits |= 1u << (j - start);
                                            mask[start / contact_grain].bits = bits;
                                        }
                                    } catch (...) { record_error(&state, phase); }
                                }
                                state.arrived.fetch_add(1, std::memory_order_acq_rel);
                                if (a.lane == 0) {
                                    wait_until([&] {
                                        return state.arrived.load(std::memory_order_acquire)
                                            == 2 * a.lanes;
                                    });
                                    if (!state.failed_phase.load(std::memory_order_acquire)) {
                                        try { commit(v, result, mask); }
                                        catch (...) { record_error(&state, phase); }
                                    }
                                    state.arrived.store(0, std::memory_order_relaxed);
                                    state.ready.store(phase, std::memory_order_release);
                                } else {
                                    // This join keeps the next vertex from reading
                                    // positions or scratch before this commit ends.
                                    wait_until([&] {
                                        return state.ready.load(std::memory_order_acquire) >= phase;
                                    });
                                }
                            }
                            // A fast lane may already have failed in the next
                            // vertex. Every lane must still join that vertex,
                            // so only exit for this (or an earlier) phase.
                            const auto failed = state.failed_phase.load(std::memory_order_acquire);
                            if (failed != 0 && failed <= phase) break;
                        }
                        if constexpr (Profile)
                            if (a.lane == 0)
                                max_cell = std::max(max_cell, omp_get_wtime() - cell_start);
                    }
                    // Finished groups can also drain remaining independent cells.
                    if (split_count[b] < static_cast<int>(batch.size())) {
                        while (true) {
                            const int i = next_whole[b].fetch_add(1, std::memory_order_relaxed);
                            if (i >= static_cast<int>(batch.size())) break;
                            process_whole(batch[i]);
                        }
                    }
                }
                double work_end = 0.0;
                if constexpr (Profile) work_end = omp_get_wtime();
#pragma omp barrier
                if constexpr (Profile) {
                    const double end = omp_get_wtime();
                    auto& stats = worker_stats[b * stats_team + worker];
                    stats.wall_seconds = end - begin;
                    stats.worker_busy_seconds = std::max(0.0, work_end - begin - group_wait);
                    stats.group_wait_seconds = group_wait;
                    stats.barrier_wait_seconds = end - work_end;
                    stats.max_cell_seconds = max_cell;
                    stats.cooperative_cells = cooperative_count;
                }
                if (errors[b]) break;
            }
        }
        if constexpr (Profile) {
            batch_stats.resize(batch_count);
            for (std::size_t b = 0; b < batch_count; ++b) {
                auto& batch = batch_stats[b];
                for (int worker = 0; worker < stats_team; ++worker) {
                    const auto& stats = worker_stats[b * stats_team + worker];
                    batch.wall_seconds = std::max(batch.wall_seconds, stats.wall_seconds);
                    batch.worker_busy_seconds += stats.worker_busy_seconds;
                    batch.group_wait_seconds += stats.group_wait_seconds;
                    batch.barrier_wait_seconds += stats.barrier_wait_seconds;
                    batch.max_cell_seconds = std::max(batch.max_cell_seconds, stats.max_cell_seconds);
                    batch.cooperative_cells += stats.cooperative_cells;
                }
            }
        }
        for (const auto& error : errors)
            if (error) std::rethrow_exception(error);
    }
};

} // namespace solver_detail
