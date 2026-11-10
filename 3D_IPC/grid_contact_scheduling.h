#pragma once

#include "contact_scheduling.h"
#include "grid_coloring.h"

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <thread>

namespace solver_detail {

// Per-vertex scratch for grid cooperation. A helper group finishes the current
// vertex's assembly, safe-step tests and commit before moving to the next one.
struct ClothGridContactStepState {
    std::vector<Vec3> steps;
    std::vector<unsigned char> nonzero_step, short_step;

    void resize(std::size_t vertices) {
        steps.resize(vertices);
        nonzero_step.resize(vertices);
        short_step.resize(vertices);
    }
};

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
        std::atomic<std::uint64_t> commit_ready{0};
    };

    int prepared_team = 0;
    std::vector<int> cooperative_cells;
    std::vector<Assignment> assignments;
    std::vector<int> split_count;

    void prepare(const ClothGridSchedule& schedule, const BroadPhase::Cache& cache) {
        prepared_team = std::max(1, omp_get_max_threads());
        const std::size_t batch_count = schedule.batches.size();
        assignments.assign(batch_count * prepared_team, Assignment{});
        split_count.assign(batch_count, 0);
        cooperative_cells.clear();
        reserve_contact_scratch(states, states_capacity, schedule.cells.size());
        reserve_contact_scratch(next_whole, queue_capacity, batch_count);
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
        resize_contact_scratch(values, maximum_values);
        masks.resize(maximum_masks);
    }

    // With cooperative_commit, fixed helpers also serve ordered contact work
    // requested by the leader's commit callback after the ordinary CCD join.
    template <class Compute, class Apply, class Baseline, class CCD, class Commit>
    void run(const ClothGridSchedule& schedule, const Compute& compute,
             const Apply& apply, const Baseline& baseline, const CCD& ccd,
             const Commit& commit, bool cooperative_commit = false) {
        run_impl(schedule, compute, apply, baseline, ccd, commit, cooperative_commit);
    }

private:
    static constexpr int cooperative_threshold = 128;
    std::vector<int> vertex_counts;
    std::vector<ContactContribution, CacheAlignedAllocator<ContactContribution>> values;
    std::vector<ContactMaskWord> masks;
    std::unique_ptr<State[]> states;
    std::unique_ptr<ContactTaskGroup[]> commit_contexts;
    std::unique_ptr<std::atomic<int>[]> next_whole;
    std::size_t states_capacity = 0, queue_capacity = 0;
    std::size_t commit_context_capacity = 0;

    template <class Compute, class Apply, class Baseline, class CCD, class Commit>
    void run_impl(const ClothGridSchedule& schedule, const Compute& compute,
                  const Apply& apply, const Baseline& baseline, const CCD& ccd,
                  const Commit& commit, bool cooperative_commit) {
        const std::size_t batch_count = schedule.batches.size();
        if (batch_count == 0) return;
        if (cooperative_commit)
            reserve_contact_scratch(commit_contexts, commit_context_capacity, schedule.cells.size());
        for (std::size_t b = 0; b < batch_count; ++b)
            next_whole[b].store(split_count[b], std::memory_order_relaxed);
        for (int cell : cooperative_cells) {
            states[cell].arrived.store(0, std::memory_order_relaxed);
            states[cell].ready.store(0, std::memory_order_relaxed);
            states[cell].failed_phase.store(0, std::memory_order_relaxed);
            if (cooperative_commit)
                states[cell].commit_ready.store(0, std::memory_order_relaxed);
        }
        std::vector<std::exception_ptr> errors(batch_count);
#pragma omp parallel shared(errors)
        {
            const int worker = omp_get_thread_num();
            for (std::size_t b = 0; b < batch_count; ++b) {
                const auto& batch = schedule.batches[b];
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
                    try {
                        for (int v : schedule.cells[cell].vertices) baseline(v);
                    } catch (...) {
                        record_error(nullptr);
                    }
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
                        State& state = states[a.cell];
                        auto* result = values.data() + a.offset;
                        auto* mask = masks.data() + a.mask_offset;
                        const auto wait_until = [&](const auto& ready) {
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
                                    if (cooperative_commit) {
                                        ContactTaskGroup& context = commit_contexts[a.cell];
                                        context.helpers = a.lanes - 1;
                                        context.sequence.store(0, std::memory_order_relaxed);
                                        // All CCD readers joined. Publish the new
                                        // context before helpers inspect its sequence.
                                        state.commit_ready.store(phase, std::memory_order_release);
                                        ContactTaskGroup* previous = active_contact_task_group;
                                        active_contact_task_group = &context;
                                        if (!state.failed_phase.load(std::memory_order_acquire)) {
                                            try { commit(v, result, mask); }
                                            catch (...) { record_error(&state, phase); }
                                        }
                                        active_contact_task_group = previous;
                                        context.sequence.store(-1, std::memory_order_release);
                                        // A helper must leave help(), not merely
                                        // finish its last range, before this context
                                        // can be reused by the next vertex.
                                        wait_until([&] {
                                            return state.arrived.load(std::memory_order_acquire)
                                                == 3 * a.lanes - 1;
                                        });
                                    } else if (!state.failed_phase.load(std::memory_order_acquire)) {
                                        try { commit(v, result, mask); }
                                        catch (...) { record_error(&state, phase); }
                                    }
                                    state.arrived.store(0, std::memory_order_relaxed);
                                    state.ready.store(phase, std::memory_order_release);
                                } else {
                                    if (cooperative_commit) {
                                        wait_until([&] {
                                            return state.commit_ready.load(std::memory_order_acquire) >= phase;
                                        });
                                        commit_contexts[a.cell].help();
                                        state.arrived.fetch_add(1, std::memory_order_acq_rel);
                                    }
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
                #pragma omp barrier
                if (errors[b]) break;
            }
        }
        for (const auto& error : errors)
            if (error) std::rethrow_exception(error);
    }
};

} // namespace solver_detail
