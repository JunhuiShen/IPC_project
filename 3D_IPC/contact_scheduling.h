#pragma once

#include "broad_phase.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <new>
#include <numeric>
#include <omp.h>
#include <optional>
#include <type_traits>
#include <vector>

// Contact scheduling for the cloth, rigid-body, and mixed solvers. All paths
// preserve color barriers and each block's original arithmetic order.
namespace solver_detail {

// -----------------------------------------------------------------------------
// Shared ordered contact evaluation and mixed-block scheduling
// -----------------------------------------------------------------------------

template <class T> struct CacheAlignedAllocator {
    using value_type = T;
    static_assert(alignof(T) <= 64, "Contact storage exceeds cache-line alignment");
    CacheAlignedAllocator() = default;
    template <class U> CacheAlignedAllocator(const CacheAlignedAllocator<U> &) {}
    T *allocate(std::size_t n) {
        return static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t(64)));
    }
    void deallocate(T *p, std::size_t) { ::operator delete(p, std::align_val_t(64)); }
};
template <class T, class U>
bool operator==(const CacheAlignedAllocator<T> &, const CacheAlignedAllocator<U> &) {
    return true;
}
template <class T, class U>
bool operator!=(const CacheAlignedAllocator<T> &, const CacheAlignedAllocator<U> &) {
    return false;
}

// Run ranges on the current team and join them before returning. The compiled
// dispatcher avoids duplicating OpenMP task outlining for every contact type.
// Optional leader work must be independent of the range evaluations. It joins
// before accumulation; its failure takes precedence after all helpers finish.
void evaluate_contact_ranges(int count, const std::function<void(int, int)>& evaluate, int alignment = 1,
    const std::function<void()>* leader_work = nullptr);

// Evaluate independently, then accumulate in original contact order. Each
// caller owns its scratch and holds contact positions fixed until the join.
template <class Evaluate, class Accumulate>
void parallel_contact_tasks(int count,
                            const Evaluate& evaluate, const Accumulate& accumulate,
                            const std::function<void()>* leader_work = nullptr) {
    using Value = decltype(evaluate(0));
    // The caller lends its previous buffer to this invocation. A nested call
    // receives a separate buffer, and exceptions release the borrowed storage.
    using Storage = std::vector<std::optional<Value>, CacheAlignedAllocator<std::optional<Value>>>;
    static thread_local Storage reusable;
    Storage values;
    values.swap(reusable);
    values.resize(count);
    constexpr int alignment = 64 / std::gcd(std::size_t(64), sizeof(std::optional<Value>));
    evaluate_contact_ranges(count, [&](int begin, int end) {
        for (int i = begin; i < end; ++i) values[i] = evaluate(i);
    }, alignment, leader_work);
    for (const auto& value : values) accumulate(*value);
    values.swap(reusable);
}

// Keep the scalar traversal outside the OpenMP task region so it can inline
// into existing small-block and single-thread assembly without scratch copies.
template <class Evaluate, class Accumulate>
void ordered_contact_tasks(int count, bool cooperative,
                           const Evaluate& evaluate, const Accumulate& accumulate,
                           const std::function<void()>* leader_work = nullptr) {
    if (!cooperative || count < 32 || omp_get_num_threads() == 1) {
        if (leader_work) (*leader_work)();
        for (int i = 0; i < count; ++i) accumulate(evaluate(i));
    } else {
        parallel_contact_tasks(count, evaluate, accumulate, leader_work);
    }
}

inline void contact_spin_hint();

// Reusable color barrier. The caller reserves storage before entering a team
// and initializes it from an omp single region. Callbacks must join any tasks
// they create before arriving: this barrier only joins the team's workers.
class ColoredSweepBarrier {
    struct alignas(64) WorkerGroup {
        std::atomic<int> arrived{0};
        std::atomic<unsigned> phase{0};
        int size = 0;
    };
    static constexpr int group_size = 8;
    std::unique_ptr<WorkerGroup[]> groups_;
    int capacity_ = 0, group_count_ = 0;
    alignas(64) std::atomic<int> arrived_{0};

public:
    void reserve(int threads) {
        const int needed = (threads + group_size - 1) / group_size;
        if (needed > capacity_) {
            groups_ = std::make_unique<WorkerGroup[]>(needed);
            capacity_ = needed;
        }
    }

    void initialize(int threads) {
        group_count_ = (threads + group_size - 1) / group_size;
        for (int g = 0; g < group_count_; ++g) {
            groups_[g].size = std::min(group_size, threads - g * group_size);
            groups_[g].arrived.store(0, std::memory_order_relaxed);
            groups_[g].phase.store(0, std::memory_order_relaxed);
        }
        arrived_.store(0, std::memory_order_relaxed);
    }

    template <class Finish>
    void wait(int worker, unsigned phase, const Finish& finish) {
        WorkerGroup& group = groups_[worker / group_size];
        // Acquire every worker's writes through the two arrival levels, then
        // publish completion and any scratch resets to all groups together.
        if (group.arrived.fetch_add(1, std::memory_order_acq_rel) == group.size - 1) {
            group.arrived.store(0, std::memory_order_relaxed);
            if (arrived_.fetch_add(1, std::memory_order_acq_rel) == group_count_ - 1) {
                arrived_.store(0, std::memory_order_relaxed);
                finish();
                for (int g = 0; g < group_count_; ++g)
                    groups_[g].phase.store(phase, std::memory_order_release);
            }
        }
        while (group.phase.load(std::memory_order_acquire) != phase)
            contact_spin_hint();
    }
};

// A block leader publishes independent contact ranges to its assigned helpers.
// Contributions still return to the leader for accumulation in contact order.
struct ContactTaskGroup {
    alignas(64) std::atomic<int> sequence{0};
    const std::function<void(int, int)>* evaluator = nullptr;
    int count = 0, grain = 16, helpers = 0, first_error = 0;
    std::exception_ptr error;
    alignas(64) std::atomic<int> cursor{0};
    alignas(64) std::atomic<int> remaining{0};

    void work() {
        for (;;) {
            const int begin = cursor.fetch_add(grain, std::memory_order_relaxed);
            if (begin >= count) break;
            try {
                (*evaluator)(begin, std::min(begin + grain, count));
            } catch (...) {
#pragma omp critical(ipc_contact_task_error)
                {
                    if (begin < first_error) {
                        first_error = begin;
                        error = std::current_exception();
                    }
                }
            }
        }
    }

    void dispatch(int n, const std::function<void(int, int)>& evaluate, int alignment,
                  const std::function<void()>* leader_work = nullptr) {
        evaluator = &evaluate;
        count = n;
        grain = std::max(4, n / (4 * (helpers + 1)));
        grain = (grain + alignment - 1) / alignment * alignment;
        first_error = n;
        error = nullptr;
        cursor.store(0, std::memory_order_relaxed);
        remaining.store(helpers, std::memory_order_relaxed);
        sequence.fetch_add(1, std::memory_order_release);
        // Helpers evaluate contacts while the leader assembles independent
        // terms. Drain the published work even if the leader's work fails.
        std::exception_ptr leader_error;
        try { if (leader_work) (*leader_work)(); }
        catch (...) { leader_error = std::current_exception(); }
        work();
        while (remaining.load(std::memory_order_acquire) != 0)
            contact_spin_hint();
        if (leader_error) std::rethrow_exception(leader_error);
        if (error) std::rethrow_exception(error);
    }

    void help() {
        int seen = 0;
        for (;;) {
            const int command = sequence.load(std::memory_order_acquire);
            if (command < 0) return;
            if (command == seen) {
                contact_spin_hint();
                continue;
            }
            seen = command;
            work();
            remaining.fetch_sub(1, std::memory_order_release);
        }
    }
};

inline thread_local ContactTaskGroup* active_contact_task_group = nullptr;

struct ColoredBlockTeams {
    struct Assignment { int block = -1, context = -1, lane = 0, lanes = 1; };
    struct Choice { int block; std::size_t cost; int lanes = 1, limit = 1; };
    std::vector<Assignment> assignments;
    std::vector<std::vector<int>> whole;
    std::vector<std::vector<int>> helper_contexts;
    std::unique_ptr<ContactTaskGroup[]> contexts;
    std::unique_ptr<std::atomic<int>[]> next;
    ColoredSweepBarrier barrier;
    std::size_t context_capacity = 0, color_capacity = 0;
    int team = 1;
    bool prepared = false;
    std::vector<std::vector<int>> saved_groups;
    std::vector<std::size_t> saved_costs;

    template <class Cost>
    void prepare(const std::vector<std::vector<int>>& groups, const Cost& cost) {
        bool same = prepared && team == omp_get_max_threads() && groups == saved_groups;
        std::size_t index = 0;
        if (same) {
            for (const auto& group : groups)
                for (int block : group)
                    if (saved_costs[index++] != cost(block)) same = false;
        }
        if (same) {
            for (std::size_t c = 0; c < groups.size(); ++c) next[c].store(0, std::memory_order_relaxed);
            for (const auto& assignment : assignments)
                if (assignment.block >= 0 && assignment.lane == 0)
                    contexts[assignment.context].sequence.store(0, std::memory_order_relaxed);
            return;
        }
        prepared = false;
        saved_groups = groups;
        saved_costs.clear();
        for (const auto& group : groups)
            for (int block : group) saved_costs.push_back(cost(block));
        team = omp_get_max_threads();
        const std::size_t count = groups.size() * static_cast<std::size_t>(team);
        assignments.assign(count, Assignment{});
        whole.resize(groups.size());
        helper_contexts.resize(groups.size());
        if (count > context_capacity) {
            contexts = std::make_unique<ContactTaskGroup[]>(count);
            context_capacity = count;
        }
        if (groups.size() > color_capacity) {
            next = std::make_unique<std::atomic<int>[]>(groups.size());
            color_capacity = groups.size();
        }
        for (std::size_t c = 0; c < groups.size(); ++c) {
            const auto& group = groups[c];
            const bool small = group.size() < static_cast<std::size_t>(team);
            std::size_t total = 0;
            for (int block : group) total += cost(block);
            const std::size_t average = total / std::max<std::size_t>(1, group.size());
            std::vector<Choice> chosen;
            for (int block : group) {
                const std::size_t weight = cost(block);
                if (small || (weight >= 512 && weight > 4 * average)) {
                    const int limit = weight < 32 ? 1
                        : static_cast<int>(std::min<std::size_t>(team, (weight + 7) / 8));
                    chosen.push_back({block, weight, 1, limit});
                }
            }
            if (!small) {
                std::stable_sort(chosen.begin(), chosen.end(),
                    [](const Choice& a, const Choice& b) { return a.cost > b.cost; });
                chosen.resize(std::min(chosen.size(), static_cast<std::size_t>(std::max(1, team / 8))));
            }
            const int budget = small ? team : team / 2;
            int used = static_cast<int>(chosen.size());
            while (used < budget) {
                int best = -1;
                double score = -1;
                for (int i = 0; i < static_cast<int>(chosen.size()); ++i) {
                    const auto& item = chosen[i];
                    if (item.lanes > item.limit / 2 || item.lanes > budget - used) continue;
                    const double value = static_cast<double>(item.cost) / item.lanes;
                    if (value > score) { score = value; best = i; }
                }
                if (best < 0) break;
                used += chosen[best].lanes;
                chosen[best].lanes *= 2;
            }
            // Largest power-of-two groups first gives aligned contiguous
            // worker intervals, reducing cross-cache helper synchronization.
            std::stable_sort(chosen.begin(), chosen.end(),
                [](const Choice& a, const Choice& b) { return a.lanes > b.lanes; });
            int worker = 0;
            helper_contexts[c].clear();
            for (const Choice& item : chosen) {
                const int context = static_cast<int>(c) * team + worker;
                if (item.lanes > 1) helper_contexts[c].push_back(context);
                contexts[context].helpers = item.lanes - 1;
                contexts[context].sequence.store(0, std::memory_order_relaxed);
                for (int lane = 0; lane < item.lanes; ++lane)
                    assignments[c * team + worker++] = {item.block, context, lane, item.lanes};
            }
            auto& unsplit = whole[c];
            unsplit.clear();
            for (int block : group) {
                const bool selected = std::any_of(chosen.begin(), chosen.end(),
                    [block](const Choice& item) { return item.block == block; });
                if (!selected) unsplit.push_back(block);
            }
            next[c].store(0, std::memory_order_relaxed);
        }
        prepared = true;
    }
};

// Preserve every color barrier and ordered contact accumulation. Fixed-iteration
// callers may run multiple sweeps on one team, stopping at the next cache rebuild.
// Contact helpers stay assigned to their original block throughout its update.
template <class Cost, class Process>
void for_each_colored_block(const std::vector<std::vector<int>>& groups,
                            const Cost& cost, const Process& process,
                            int sweeps = 1) {
    if (sweeps <= 0 || groups.empty()) return;
    if (omp_get_max_threads() == 1 || omp_in_parallel()) {
        for (int sweep = 0; sweep < sweeps; ++sweep)
            for (const auto& group : groups)
                for (int block : group) process(block, false);
        return;
    }
    static thread_local ColoredBlockTeams storage;
    ColoredBlockTeams& workspace = storage;
    workspace.prepare(groups, cost);
    workspace.barrier.reserve(omp_get_max_threads());
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    const auto invoke = [&](int block, bool cooperative) {
        if (failed.load(std::memory_order_relaxed)) return;
        try { process(block, cooperative); }
        catch (...) {
            if (!failed.exchange(true, std::memory_order_relaxed))
                error = std::current_exception();
        }
    };
#pragma omp parallel shared(error, failed, workspace)
    {
        const int worker = omp_get_thread_num();
        const int team = omp_get_num_threads();
        const bool planned_team = team == workspace.team;
#pragma omp single
        { workspace.barrier.initialize(team); }
        unsigned phase = 0;
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            for (std::size_t c = 0; c < groups.size(); ++c) {
                const auto& whole = workspace.whole[c];
                if (!planned_team || whole.size() == groups[c].size()) {
                    // Reserve the first wave without contending on a cursor.
                    // Runtime team-size changes safely use whole-block work.
                    const auto& group = groups[c];
                    const int size = static_cast<int>(group.size());
                    if (planned_team) {
                        // Preserve the existing round-robin whole-block path
                        // without a runtime workshare or shared cursor.
                        for (int item = worker; item < size; item += team)
                            invoke(group[item], false);
                    } else {
                        if (worker < size) invoke(group[worker], false);
                        while (workspace.next[c].load(std::memory_order_relaxed) < size - team) {
                            const int item = team + workspace.next[c].fetch_add(1, std::memory_order_relaxed);
                            if (item < size) invoke(group[item], false);
                        }
                    }
                } else {
                    const auto assignment = workspace.assignments[c * workspace.team + worker];
                    if (assignment.block >= 0) {
                        ContactTaskGroup& context = workspace.contexts[assignment.context];
                        if (assignment.lane == 0) {
                            ContactTaskGroup* previous = active_contact_task_group;
                            active_contact_task_group = assignment.lanes > 1 ? &context : nullptr;
                            invoke(assignment.block, assignment.lanes > 1);
                            active_contact_task_group = previous;
                            context.sequence.store(-1, std::memory_order_release);
                        } else {
                            context.help();
                        }
                    }
                    const int size = static_cast<int>(whole.size());
                    while (workspace.next[c].load(std::memory_order_relaxed) < size) {
                        const int item = workspace.next[c].fetch_add(1, std::memory_order_relaxed);
                        if (item < size) invoke(whole[item], false);
                    }
                }
                workspace.barrier.wait(worker, ++phase, [&] {
                    // No helper or cursor reader survives this color barrier.
                    // Reset before publishing it, so the next sweep can reuse
                    // each context without seeing the preceding stop command.
                    workspace.next[c].store(0, std::memory_order_relaxed);
                    for (int context : workspace.helper_contexts[c])
                        workspace.contexts[context].sequence.store(0, std::memory_order_relaxed);
                });
                // On failure, drain the identical remaining barriers on every
                // worker. A next-color failure must not split the current team.
            }
        }
    }
    if (error) std::rethrow_exception(error);
}

// -----------------------------------------------------------------------------
// Cloth scheduler: fixed worker groups, compact masks, and reusable scratch
// -----------------------------------------------------------------------------

inline constexpr int contact_grain = 16;
// Each worker owns strided groups of 16 contacts and their compact masks.
// Leaders skip inactive derivative storage during ordered accumulation.
struct alignas(32) ContactMaskWord {
    unsigned bits = 0, clear = 0;
};
template <class F> void for_active_contact(const ContactMaskWord *mask, int count, const F &f) {
    for (int word = 0; word <= (count / contact_grain); ++word) {
        unsigned bits = mask[word].bits;
        while (bits) {
            int j = contact_grain * word + __builtin_ctz(bits);
            bits &= bits - 1;
            if (j > 0 && j <= count)
                f(j);
        }
    }
}
struct ContactContribution {
    Vec3 gradient;
    Mat33 hessian;
    double toi;
};
inline void contact_spin_hint() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

// Split expensive vertices among fixed worker subsets, and let spare workers
// process other vertices of the same color. The global color barrier and each
// vertex's original floating-point accumulation order are preserved.
// Compute returns bit 0 for an active contribution and bit 1 for a certified
// AABB rejection. CCD receives that certificate and returns collision status.
struct ColoredContactSweep {
    struct Assignment {
        int vertex = -1, lane = 0, lanes = 1, count = 0, offset = 0, mask_offset = 0;
    };
    struct alignas(64) State {
        std::atomic<int> arrived{0}, ready{0};
    };
    std::vector<Assignment> assignments;
    std::vector<ContactContribution, CacheAlignedAllocator<ContactContribution>> values;
    std::vector<ContactMaskWord> masks;
    std::unique_ptr<State[]> states;
    int team = 0, threshold = 0;
    std::vector<int> split_count;
    std::unique_ptr<std::atomic<int>[]> next_whole;
    std::vector<int> cooperative_vertices;
    std::vector<Vec3> steps;
    std::vector<unsigned char> nonzero_step, short_step;
    void prepare(const std::vector<std::vector<int>> &groups, const BroadPhase::Cache &cache) {
        team = omp_get_max_threads();
        threshold = std::max(2, std::min(32, team / 2));
        steps.resize(cache.vertex_nt.size());
        nonzero_step.resize(cache.vertex_nt.size());
        short_step.resize(cache.vertex_nt.size());
        cooperative_vertices.clear();
        assignments.assign(groups.size() * team, Assignment{});
        states = std::make_unique<State[]>(cache.vertex_nt.size());
        const int heavy_count = std::min(24, std::max(1, team * 24 / 64));
        const int heavy_budget = 48 * team / 64;
        split_count.assign(groups.size(), 0);
        next_whole = std::make_unique<std::atomic<int>[]>(groups.size());
        std::size_t maximum = 0, maximum_masks = 0;
        for (int c = 0; c < static_cast<int>(groups.size()); ++c) {
            const auto &group = groups[c];
            if (group.empty())
                continue;
            const bool small = group.size() < static_cast<std::size_t>(threshold);
            const int selected =
                small ? group.size() : std::min(heavy_count, static_cast<int>(group.size()));
            const int budget = small ? team : std::min(heavy_budget, team - 1);
            if (selected <= 0 || budget < selected)
                continue;
            if (!small && cache.vertex_nt[group[0]].size() + cache.vertex_ss[group[0]].size() < 512)
                continue;
            split_count[c] = selected;
            cooperative_vertices.insert(cooperative_vertices.end(), group.begin(),
                                        group.begin() + selected);
            std::vector<int> lanes(selected, 1), counts(selected);
            for (int i = 0; i < selected; ++i)
                counts[i] = 1 + cache.vertex_nt[group[i]].size() + cache.vertex_ss[group[i]].size();
            for (int t = selected; t < budget; ++t) {
                int best = 0;
                for (int i = 1; i < selected; ++i)
                    if (static_cast<double>(counts[i]) / lanes[i] >
                        static_cast<double>(counts[best]) / lanes[best])
                        best = i;
                ++lanes[best];
            }
            int t = 0, offset = 0, mask_offset = 0;
            for (int i = 0; i < selected; ++i) {
                for (int lane = 0; lane < lanes[i]; ++lane)
                    assignments[c * team + t++] = {group[i],  lane,   lanes[i],
                                                   counts[i], offset, mask_offset};
                offset += counts[i];
                offset = (offset + 7) & ~7;
                mask_offset += (counts[i] + contact_grain - 1) / contact_grain;
            }
            maximum = std::max(maximum, static_cast<std::size_t>(offset));
            maximum_masks = std::max(maximum_masks, static_cast<std::size_t>(mask_offset));
        }
        values.resize(maximum);
        masks.resize(maximum_masks);
    }
    // Optional before_color is called collectively by every worker before any
    // updates of that color, including fallback paths. It must not throw and
    // must join its writes before returning (e.g. an omp for without nowait).
    // The default adds no work or barriers for existing callers.
    template <class Compute, class Apply, class Baseline, class CCD, class Commit,
              class BeforeColor = std::nullptr_t>
    void run(const std::vector<std::vector<int>> &groups, const Compute &compute,
             const Apply &apply, const Baseline &baseline, const CCD &ccd, const Commit &commit,
             const BeforeColor& before_color = nullptr) {
        constexpr int chunk = contact_grain;
        for (int c = 0; c < static_cast<int>(groups.size()); ++c)
            next_whole[c].store(split_count[c], std::memory_order_relaxed);
        // Reset before entering the team, including after any runtime team-size
        // fallback. Do not assume that every preceding sweep used this path.
        for (int v : cooperative_vertices) {
            states[v].arrived.store(0, std::memory_order_relaxed);
            states[v].ready.store(0, std::memory_order_relaxed);
        }
#pragma omp parallel
        {
            for (int c = 0; c < static_cast<int>(groups.size()); ++c) {
                const auto &group = groups[c];
                if constexpr (!std::is_same_v<BeforeColor, std::nullptr_t>)
                    before_color(static_cast<std::size_t>(c));
                if (split_count[c] == 0 || omp_get_num_threads() != team) {
#pragma omp for schedule(dynamic, 1)
                    for (int i = 0; i < static_cast<int>(group.size()); ++i)
                        baseline(group[i]);
                } else {
                    const Assignment a = assignments[c * team + omp_get_thread_num()];
                    if (a.vertex >= 0) {
                        auto *result = values.data() + a.offset;
                        State &state = states[a.vertex];
                        for (int start = a.lane * chunk; start < a.count;
                             start += a.lanes * chunk) {
                            unsigned bits = 0, clear = 0;
                            for (int j = start; j < std::min(start + chunk, a.count); ++j) {
                                unsigned flags = compute(a.vertex, j, result[j]);
                                bits |= (flags & 1u) << (j - start);
                                clear |= ((flags >> 1) & 1u) << (j - start);
                            }
                            masks[a.mask_offset + start / contact_grain].bits = bits;
                            masks[a.mask_offset + start / contact_grain].clear = clear;
                        }
                        state.arrived.fetch_add(1, std::memory_order_acq_rel);
                        if (a.lane == 0) {
                            while (state.arrived.load(std::memory_order_acquire) != a.lanes)
                                contact_spin_hint();
                            apply(a.vertex, result, masks.data() + a.mask_offset);
                            state.ready.store(1, std::memory_order_release);
                        } else
                            while (state.ready.load(std::memory_order_acquire) < 1)
                                contact_spin_hint();
                        for (int start = a.lane * chunk; start < a.count;
                             start += a.lanes * chunk) {
                            unsigned bits = 0,
                                     clear = masks[a.mask_offset + start / contact_grain].clear;
                            for (int j = start; j < std::min(start + chunk, a.count); ++j) {
                                if (ccd(a.vertex, j, result[j], (clear >> (j - start)) & 1u))
                                    bits |= 1u << (j - start);
                            }
                            masks[a.mask_offset + start / contact_grain].bits = bits;
                        }
                        state.arrived.fetch_add(1, std::memory_order_acq_rel);
                        if (a.lane == 0) {
                            while (state.arrived.load(std::memory_order_acquire) != 2 * a.lanes)
                                contact_spin_hint();
                            commit(a.vertex, result, masks.data() + a.mask_offset);
                        }
                    }
                    if (split_count[c] < static_cast<int>(group.size())) {
                        while (true) {
                            const int size = group.size();
                            const int remaining =
                                size - next_whole[c].load(std::memory_order_relaxed);
                            if (remaining <= 0)
                                break;
                            // Batch only while ample independent work remains; use one
                            // vertex near the tail to limit end-of-color imbalance.
                            const int batch = remaining > 2 * team ? 2 : 1;
                            const int i = next_whole[c].fetch_add(batch, std::memory_order_relaxed);
                            for (int j = i; j < std::min(i + batch, size); ++j)
                                baseline(group[j]);
                        }
                    }
#pragma omp barrier
                }
            }
        }
    }

    // V2 compacts each worker's strided contacts without changing ownership.
    // compute_assigned receives vertex-relative result/mask pointers and must
    // initialize its mask words, finish its writes, and return without throwing.
    // ccd_candidates optionally returns eligible bits for a word after apply;
    // the caller must prove that omitted entries cannot affect the CCD result.
    template <class Compute, class Apply, class Baseline, class CCD, class Commit,
              class BeforeColor = std::nullptr_t, class ComputeAssigned = std::nullptr_t,
              class BaselineBatch = std::nullptr_t, class CCDCandidates = std::nullptr_t>
    void run_assigned(const std::vector<std::vector<int>> &groups, const Compute &compute,
             const Apply &apply, const Baseline &baseline, const CCD &ccd, const Commit &commit,
             const BeforeColor& before_color = nullptr, const ComputeAssigned& compute_assigned = nullptr,
             const BaselineBatch& baseline_batch = nullptr, int sweeps = 1,
             const CCDCandidates& ccd_candidates = nullptr) {
        if (sweeps <= 0 || groups.empty()) return;
        ColoredSweepBarrier barrier;
        barrier.reserve(omp_get_max_threads());
        for (int c = 0; c < static_cast<int>(groups.size()); ++c)
            next_whole[c].store(split_count[c], std::memory_order_relaxed);
        // Reset before entering the team, including after any runtime team-size
        // fallback. Do not assume that every preceding sweep used this path.
        for (int v : cooperative_vertices) {
            states[v].arrived.store(0, std::memory_order_relaxed);
            states[v].ready.store(0, std::memory_order_relaxed);
        }
#pragma omp parallel
        {
            const int worker = omp_get_thread_num();
#pragma omp single
            { barrier.initialize(omp_get_num_threads()); }
            unsigned phase = 0;
            for (int sweep = 0; sweep < sweeps; ++sweep) {
                for (int c = 0; c < static_cast<int>(groups.size()); ++c) {
                    const auto &group = groups[c];
                    if constexpr (!std::is_same_v<BeforeColor, std::nullptr_t>)
                        before_color(static_cast<std::size_t>(c));
                    if (split_count[c] == 0 || omp_get_num_threads() != team) {
                        if constexpr (std::is_same_v<BaselineBatch, std::nullptr_t>) {
#pragma omp for schedule(dynamic, 1) nowait
                            for (int i = 0; i < static_cast<int>(group.size()); ++i)
                                baseline(group[i]);
                        } else {
                            const int workers = omp_get_num_threads();
                            const int batch = std::clamp(
                                (static_cast<int>(group.size()) + 4 * workers - 1) / (4 * workers), 1, 4);
#pragma omp for schedule(dynamic, 1) nowait
                            for (int i = 0; i < static_cast<int>(group.size()); i += batch)
                                baseline_batch(group, i, std::min(i + batch, static_cast<int>(group.size())));
                        }
                    } else {
                        const Assignment a = assignments[c * team + omp_get_thread_num()];
                        if (a.vertex >= 0) {
                            auto *result = values.data() + a.offset;
                            State &state = states[a.vertex];
                            if constexpr (!std::is_same_v<ComputeAssigned, std::nullptr_t>) {
                                compute_assigned(a, result, masks.data() + a.mask_offset);
                            } else {
                                for (int start = a.lane * contact_grain; start < a.count;
                                     start += a.lanes * contact_grain) {
                                    unsigned bits = 0, clear = 0;
                                    for (int j = start; j < std::min(start + contact_grain, a.count); ++j) {
                                        const unsigned flags = compute(a.vertex, j, result[j]);
                                        bits |= (flags & 1u) << (j - start);
                                        clear |= ((flags >> 1) & 1u) << (j - start);
                                    }
                                    masks[a.mask_offset + start / contact_grain].bits = bits;
                                    masks[a.mask_offset + start / contact_grain].clear = clear;
                                }
                            }
                            state.arrived.fetch_add(1, std::memory_order_acq_rel);
                            if (a.lane == 0) {
                                while (state.arrived.load(std::memory_order_acquire) != a.lanes)
                                    contact_spin_hint();
                                apply(a.vertex, result, masks.data() + a.mask_offset);
                                state.ready.store(1, std::memory_order_release);
                            } else
                                while (state.ready.load(std::memory_order_acquire) < 1)
                                    contact_spin_hint();
                            for (int start = a.lane * contact_grain; start < a.count;
                                 start += a.lanes * contact_grain) {
                                unsigned bits = 0;
                                const unsigned clear = masks[a.mask_offset + start / contact_grain].clear;
                                unsigned pending = (1u << std::min(contact_grain, a.count - start)) - 1u;
                                if constexpr (!std::is_same_v<CCDCandidates, std::nullptr_t>)
                                    pending &= ccd_candidates(a.vertex, start, clear);
                                while (pending) {
                                    const int bit = __builtin_ctz(pending);
                                    pending &= pending - 1u;
                                    if (ccd(a.vertex, start + bit, result[start + bit], (clear >> bit) & 1u))
                                        bits |= 1u << bit;
                                }
                                masks[a.mask_offset + start / contact_grain].bits = bits;
                            }
                            state.arrived.fetch_add(1, std::memory_order_acq_rel);
                            if (a.lane == 0) {
                                while (state.arrived.load(std::memory_order_acquire) != 2 * a.lanes)
                                    contact_spin_hint();
                                commit(a.vertex, result, masks.data() + a.mask_offset);
                                // Every helper has finished accessing this state.
                                state.arrived.store(0, std::memory_order_relaxed);
                                state.ready.store(0, std::memory_order_relaxed);
                            }
                        }
                        if (split_count[c] < static_cast<int>(group.size())) {
                            while (true) {
                                const int size = group.size();
                                const int remaining =
                                    size - next_whole[c].load(std::memory_order_relaxed);
                                if (remaining <= 0)
                                    break;
                                // Batch only while ample independent work remains; use one
                                // vertex near the tail to limit end-of-color imbalance.
                                const int batch = std::is_same_v<BaselineBatch, std::nullptr_t>
                                    ? (remaining > 2 * team ? 2 : 1)
                                    : std::clamp((remaining + 4 * team - 1) / (4 * team), 1, 4);
                                const int i = next_whole[c].fetch_add(batch, std::memory_order_relaxed);
                                if constexpr (std::is_same_v<BaselineBatch, std::nullptr_t>) {
                                    for (int j = i; j < std::min(i + batch, size); ++j)
                                        baseline(group[j]);
                                } else if (i < size)
                                    baseline_batch(group, i, std::min(i + batch, size));
                            }
                        }
                    }
                    barrier.wait(worker, ++phase, [&] {
                        // Reset after all cursor readers and vertex updates finish.
                        next_whole[c].store(split_count[c], std::memory_order_relaxed);
                    });
                }
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Whole-vertex colored sweeps
// -----------------------------------------------------------------------------

// Preserve all color/sweep barriers while avoiding a runtime workshare for
// every short color. A workspace is reusable across sequential calls/teams.
class ColoredVertexSweep {
    ColoredSweepBarrier barrier_;
    alignas(64) std::atomic<int> cursor_{0};

public:
    // Same collective, nonthrowing before_color contract as ColoredContactSweep.
    template <class Process, class BeforeColor = std::nullptr_t>
    void run(const std::vector<std::vector<int>>& colors, int sweeps,
             const Process& process, const BeforeColor& before_color = nullptr) {
        if (sweeps <= 0 || colors.empty()) return;
        barrier_.reserve(omp_get_max_threads());
        std::atomic<bool> failed{false};
        std::exception_ptr error;
        const auto invoke = [&](int vertex) {
            if (failed.load(std::memory_order_relaxed)) return;
            try { process(vertex); }
            catch (...) {
                if (!failed.exchange(true, std::memory_order_relaxed))
                    error = std::current_exception();
            }
        };
#pragma omp parallel shared(failed, error)
        {
            const int worker = omp_get_thread_num();
            const int team = omp_get_num_threads();
#pragma omp single
            {
                barrier_.initialize(team);
                cursor_.store(team, std::memory_order_relaxed);
            }
            unsigned phase = 0;
            for (int sweep = 0; sweep < sweeps; ++sweep) {
                for (std::size_t c = 0; c < colors.size(); ++c) {
                    const auto& color = colors[c];
                    // All workers must still enter a collective callback when
                    // draining a failed sweep, just like the end-color barrier.
                    if constexpr (!std::is_same_v<BeforeColor, std::nullptr_t>)
                        before_color(c);
                    const int count = static_cast<int>(color.size());
                    if (worker < count) invoke(color[worker]);
                    while (cursor_.load(std::memory_order_relaxed) < count) {
                        const int item = cursor_.fetch_add(1, std::memory_order_relaxed);
                        if (item < count) invoke(color[item]);
                    }
                    barrier_.wait(worker, ++phase, [&] {
                        cursor_.store(team, std::memory_order_relaxed);
                    });
                    // Drain every barrier even on failure. A faster worker
                    // can throw in the next color before a slower one leaves
                    // this barrier; testing failed here could split the team.
                }
            }
        }
        if (error) std::rethrow_exception(error);
    }
};

} // namespace solver_detail
