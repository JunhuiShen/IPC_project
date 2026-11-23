#pragma once
#include "ccd_direct_fp32.cuh"
#include "ccd_clearance.cuh"

#if defined(__CUDACC__)
#define IPC_DIRECT_AXES_HD __host__ __device__
#else
#define IPC_DIRECT_AXES_HD
#endif

namespace ipc_gpu::ccd32direct_multi {
using ccd32::LinearQuery;
using ccd32::Status;

namespace detail {

// Try the three coordinate axes and the geometric direction with the largest
// starting separation from native GS. FP64 only selects the geometric axis.
// Recompute all proof bounds for the actual FP32 direction used below.
template <bool verify_endpoint>
IPC_DIRECT_AXES_HD inline float try_alternatives(const LinearQuery& q) {
    const int mover = ccd32direct::detail::moving_vertex(q);
    float best = 0.0f;
    for (int i = 0; i < 4; ++i) {
        ccd32::detail::Vec axis{};
        if (i < 3) {
            axis.v[i] = 1.0f;
        } else {
            const auto n = ccd_clearance::best_axis(q.start, q.vertex_face);
            double scale = 0.0;
            for (int k = 0; k < 3; ++k)
                scale = ccd::detail::maximum(scale, ::fabs(n.v[k]));
            if (!(scale > 0.0) || !ccd::detail::finite(scale)) break;
            for (int k = 0; k < 3; ++k)
                axis.v[k] = static_cast<float>(n.v[k] / scale);
        }
        float threshold;
        if (!ccd32direct::detail::orient_axis(q, axis, threshold)) continue;
        if constexpr (verify_endpoint) {
            if (ccd32direct::detail::certify_axis(q, axis, threshold)) return 1.0f;
        } else {
            const float fraction = ccd32direct::detail::fraction_on_axis(
                q, mover, axis, threshold, nullptr);
            if (fraction == 1.0f) return 1.0f;
            best = ccd32::detail::maximum(best, fraction);
        }
    }
    return best;
}

} // namespace detail

IPC_DIRECT_AXES_HD inline float propose_fraction(const LinearQuery& q) {
    const float fraction = ccd32direct::propose_fraction(q);
    return fraction == 0.0f ? detail::try_alternatives<false>(q) : fraction;
}

IPC_DIRECT_AXES_HD inline Status validate_linear(const LinearQuery& q) {
    const Status status = ccd32direct::validate_linear(q);
    if (status != Status::Uncertain) return status;
    return detail::try_alternatives<true>(q) > 0.0f ? Status::Safe : Status::Uncertain;
}

} // namespace ipc_gpu::ccd32direct_multi
#undef IPC_DIRECT_AXES_HD
