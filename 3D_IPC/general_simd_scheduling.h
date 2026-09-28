#pragma once

#include "contact_scheduling.h"
#include "SIMD.h"
#include <array>

namespace solver_detail {

struct GeneralSimdBatch {
    std::array<int, ipc_simd::tile_width> blocks{};
    std::size_t size = 0, contact_cost = 0;
};

// Only pack independent particle blocks of ONE color. A rigid COM/orientation
// update remains indivisible, and a heavy particle keeps its own helper team.
struct GeneralSimdBatches {
    std::vector<GeneralSimdBatch> batches;
    std::vector<std::vector<int>> colors;

    template<class Cost>
    void prepare(const std::vector<std::vector<int>>& groups, int rigid_begin,
                 const Cost& cost) {
        batches.clear();
        colors.assign(groups.size(), {});
        const auto team = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
        for (std::size_t c = 0; c < groups.size(); ++c) {
            GeneralSimdBatch pending;
            const std::size_t width = std::min(ipc_simd::tile_width,
                std::max(std::size_t(1), groups[c].size() / (2 * team)));
            const auto flush = [&] {
                if (!pending.size) return;
                colors[c].push_back(static_cast<int>(batches.size()));
                batches.push_back(pending);
                pending = GeneralSimdBatch{};
            };
            for (int block : groups[c]) {
                const auto weight = cost(block);
                if (block >= rigid_begin || weight >= 128) {
                    flush();
                    pending.blocks[0] = block;
                    pending.size = 1;
                    pending.contact_cost = weight;
                    flush();
                } else {
                    pending.blocks[pending.size++] = block;
                    pending.contact_cost += weight;
                    if (pending.size == width) flush();
                }
            }
            flush();
        }
    }
};

// Match the general solver's adaptive whole-work policy at batch granularity:
// cheap/uniform colors use cursor-free worker strides; uneven colors without
// helper assignments use heavy-first, adaptive dynamic claims. Remaining work
// alongside fixed helper teams still uses dynamic claims. Each rigid COM and
// orientation update stays one indivisible block under its assigned team.
// A single persistent team keeps every color barrier. On failure join helpers,
// restore the ENTIRE failed color
// (including rigid proxies/generalized coordinates), then rethrow on caller.
template<class Process, class Save, class Restore>
void run_general_simd_batches(const GeneralSimdBatches& batches, int sweeps,
    const Process& process, const Save& save, const Restore& restore) {
    const auto& groups = batches.colors;
    if (sweeps <= 0 || groups.empty()) return;
    if (omp_get_max_threads() == 1 || omp_in_parallel()) {
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            for (const auto& group : groups) {
                for (int item : group) save(item);
                try { for (int item : group) process(item, false); }
                catch (...) {
                    for (int item : group) restore(item);
                    throw;
                }
            }
        }
        return;
    }
    ColoredBlockTeams workspace;
    workspace.prepare(groups, [&](int item) { return batches.batches[item].contact_cost; });
    workspace.barrier.reserve(omp_get_max_threads());
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::vector<std::atomic<bool>> failed_colors(groups.size());
    for (auto& flag : failed_colors) flag.store(false, std::memory_order_relaxed);
    const auto invoke = [&](int item, std::size_t color, bool cooperative) {
        // Save even after another batch failed: rollback covers this color.
        save(item);
        if (failed.load(std::memory_order_relaxed)) return;
        try { process(item, cooperative); }
        catch (...) {
            failed_colors[color].store(true, std::memory_order_relaxed);
            if (!failed.exchange(true, std::memory_order_relaxed))
                error = std::current_exception();
        }
    };
    #pragma omp parallel
    {
        const int worker = omp_get_thread_num(), team = omp_get_num_threads();
        const bool planned = team == workspace.team;
        #pragma omp single
        { workspace.barrier.initialize(team); }
        unsigned phase = 0;
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            for (std::size_t c = 0; c < groups.size(); ++c) {
                const auto& whole = workspace.whole[c];
                if (!planned || whole.size() == groups[c].size()) {
                    const bool dynamic = planned && workspace.dynamic_whole[c];
                    const auto& group = dynamic ? whole : groups[c];
                    const int size = static_cast<int>(group.size());
                    if (planned && !dynamic) {
                        for (int item = worker; item < size; item += team)
                            invoke(group[item], c, false);
                    } else {
                        // Reserve the first wave without cursor contention.
                        // A runtime team-size mismatch disables helper plans
                        // and safely claims all remaining batches one at a time.
                        if (worker < size) invoke(group[worker], c, false);
                        for (;;) {
                            const int remaining = size - team
                                - workspace.next[c].load(std::memory_order_relaxed);
                            if (remaining <= 0) break;
                            const int count = dynamic
                                ? std::min(8, std::max(1, remaining / (4 * team))) : 1;
                            const int first = team + workspace.next[c].fetch_add(count, std::memory_order_relaxed);
                            for (int item = first; item < std::min(first + count, size); ++item)
                                invoke(group[item], c, false);
                        }
                    }
                } else {
                    const auto assignment = workspace.assignments[c * workspace.team + worker];
                    if (assignment.block >= 0) {
                        auto& context = workspace.contexts[assignment.context];
                        if (assignment.lane == 0) {
                            auto* previous = active_contact_task_group;
                            active_contact_task_group = assignment.lanes > 1 ? &context : nullptr;
                            invoke(assignment.block, c, assignment.lanes > 1);
                            active_contact_task_group = previous;
                            context.sequence.store(-1, std::memory_order_release);
                        } else context.help();
                    }
                    for (;;) {
                        const int item = workspace.next[c].fetch_add(1, std::memory_order_relaxed);
                        if (item >= static_cast<int>(whole.size())) break;
                        invoke(whole[item], c, false);
                    }
                }
                workspace.barrier.wait(worker, ++phase, [&] {
                    if (failed_colors[c].load(std::memory_order_relaxed)) {
                        for (int item : groups[c]) restore(item);
                        failed_colors[c].store(false, std::memory_order_relaxed);
                    }
                    workspace.next[c].store(0, std::memory_order_relaxed);
                    for (int context : workspace.helper_contexts[c])
                        workspace.contexts[context].sequence.store(0, std::memory_order_relaxed);
                });
            }
        }
    }
    if (error) std::rethrow_exception(error);
}

} // namespace solver_detail
