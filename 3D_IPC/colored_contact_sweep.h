#pragma once
#include "broad_phase.h"
#include <algorithm>
#include <atomic>
#include <memory>
#include <new>
#include <omp.h>
#include <vector>

namespace solver_detail {
inline constexpr int contact_grain = 16;
// A worker owns 16 consecutive contacts. Compact masks let leaders skip
// inactive records without loading their gradient/Hessian storage.
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

// Eight 104-byte contributions occupy thirteen 64-byte cache lines. Align the
// storage and each vertex's first block so workers never share a writable line.
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
    template <class Compute, class Apply, class Baseline, class CCD, class Commit>
    void run(const std::vector<std::vector<int>> &groups, const Compute &compute,
             const Apply &apply, const Baseline &baseline, const CCD &ccd, const Commit &commit) {
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
};
} // namespace solver_detail
