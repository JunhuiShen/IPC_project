#pragma once

#include "physics.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

class BroadPhase;

// One representable binary64 step, with std::nextafter's exceptional behavior.
// These functions do not change the rounding mode or combine arithmetic: callers
// must still perform each operation before widening its result.
namespace directed_rounding {
namespace detail {

static_assert(sizeof(double) == sizeof(std::uint64_t)
    && std::numeric_limits<double>::is_iec559
    && std::numeric_limits<double>::radix == 2
    && std::numeric_limits<double>::digits == 53
    && std::numeric_limits<double>::min_exponent == -1021
    && std::numeric_limits<double>::max_exponent == 1024,
    "directed_rounding requires IEEE-754 binary64");

// Direction is +1 for +infinity, -1 for -infinity, and 0 for positive zero.
template<int Direction>
inline double step(double x) {
    static_assert(Direction >= -1 && Direction <= 1, "invalid step direction");
    std::uint64_t pattern;
    std::memcpy(&pattern, &x, sizeof(pattern));
    const std::uint64_t magnitude = pattern & UINT64_C(0x7fffffffffffffff);

    // Keep both the input and output normal. Zeros,
    // subnormals, the normal-range endpoints, infinities, and all NaNs retain
    // the platform libm's result, errno, and floating-point exception behavior.
    if (magnitude > UINT64_C(0x0010000000000000)
        && magnitude < UINT64_C(0x7fefffffffffffff)) {
        if constexpr (Direction == 0) {
            --pattern;
        } else {
            const bool negative = (pattern >> 63) != 0;
            if (negative == (Direction > 0)) --pattern;
            else ++pattern;
        }
        double result;
        std::memcpy(&result, &pattern, sizeof(result));
        return result;
    }

    constexpr double target = Direction > 0 ? std::numeric_limits<double>::infinity()
        : Direction < 0 ? -std::numeric_limits<double>::infinity() : 0.0;
    return std::nextafter(x, target);
}

} // namespace detail

inline double next_up(double x) { return detail::step<1>(x); }
inline double next_down(double x) { return detail::step<-1>(x); }

// The target is +0.0, including when x is -0.0. This intentionally matches
// std::nextafter(x, 0.0), which returns positive zero for either zero input.
inline double toward_positive_zero(double x) { return detail::step<0>(x); }

} // namespace directed_rounding

// Reusable scratch: every call refreshes all numerical buffers, so a workspace
// is safe across solves that supply external candidates. Its internal topology
// cache assumes fixed topology within a solve. Null-workspace calls stay stateless.
class DATWorkspace {
public:
    DATWorkspace();
    ~DATWorkspace();
    DATWorkspace(const DATWorkspace&) = delete;
    DATWorkspace& operator=(const DATWorkspace&) = delete;
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Planar Divide and Truncate for simultaneous, straight-line vertex updates.
// Returns displacements; neither input array nor the mesh is modified.
//
// 1. Build the existing BVH from blue boxes centered at current_positions,
//    with half-extent query_radius * node_box_padding (default padding: 1.2).
// 2. Keep nonincident vertex/triangle and edge/edge pairs at distance < r_q.
// 3. Allocate their separating planes from the RAW incremental motion using
//    Newton's proportional rule, a 5% interior floor, and crossing-only
//    relaxation; then apply the minimum plane-intersection factor per vertex.
// 4. Apply the radius cap AFTER the planes, then certify the represented
//    endpoints with outward-rounded projections.
//
// Requires 0 < gamma < 1, finite positive query_radius, padding >= 1, and
// initially separated nonincident primitives. Invalid inputs throw
// std::invalid_argument; unresolved/zero-gap separation throws std::domain_error.
// For solids, mesh.tet_nodes and mesh.surface_nodes must classify interior and
// boundary nodes consistently, as required by BroadPhase::initialize_surface_nodes.
// This is a surface-contact filter: it does not prevent tetrahedron inversion,
// enforce rigid-body motion, handle curved rotational paths, or query SDFs.
std::vector<Vec3> planar_DAT(
    double gamma,
    double query_radius,
    const RefMesh& mesh,
    const std::vector<Vec3>& incremental_update,
    const std::vector<Vec3>& current_positions,
    double node_box_padding = 1.2,
    const BroadPhase* candidates = nullptr, DATWorkspace* workspace = nullptr);

struct GeneralDATResult {
    std::vector<double> body_factors;
    // The mixed path returns the exact represented endpoints/poses certified
    // by the filter, so callers publish them without recomputing arithmetic.
    std::vector<Vec3> positions, centers, omega;
    std::vector<Vec4> orientations;
};

// Mixed planar DAT. Deformables follow straight lines; rigid vertices follow
// c(t)=c+t*dc, q(t)=E(omega+t*domega,dt)*q_n. One factor scales both
// generalized increments of a body. All inputs are immutable sweep snapshots.
GeneralDATResult general_planar_DAT(double gamma, double query_radius,
    const RefMesh& mesh, const std::vector<Vec3>& particle_updates,
    const std::vector<Vec3>& positions, const std::vector<Vec3>& centers,
    const std::vector<Vec3>& omega, const std::vector<Vec4>& previous_orientations,
    const std::vector<Vec3>& delta_centers, const std::vector<Vec3>& delta_omega,
    double dt, double node_box_padding = 1.2,
    const BroadPhase* candidates = nullptr, DATWorkspace* workspace = nullptr);

// Optional candidates must conservatively cover the query-radius neighborhood
// at the supplied positions, with the same ownership and solid-surface policy.
// They may come from an earlier enclosing motion envelope. In particular, boxes
// with half-width >= query_radius cover a colored sweep in which each node/body
// moves once and its whole path is capped at gamma*query_radius/2: a current
// query neighbor was initially within (1+gamma)*query_radius < 2*query_radius.
// Pair geometry and separating planes are recomputed at the supplied positions.

// Local particle assembly and block coloring for the OGC-style DAT sweep.
// prepare() refreshes rest/pin/classification data and retains elasticity-only
// colors when connectivity and ownership are unchanged. Contact pairs never
// enter the coloring graph. Before every color, freeze_contacts() recomputes
// numerical mesh-contact derivatives from the accepted current positions.
// An optional selected color limits contact work to its particle blocks and
// their incident pair union; generalized rigid blocks are assembled by the
// enclosing general solver. Only selected particles may then call proposal().
// proposal() reads that color's immutable derivatives together with LIVE
// elastic/SDF positions, and never mutates the workspace. Physical previous
// positions remain fixed throughout the substep.
// Colors contain particle node ids and, in general mode, rigid blocks encoded
// as count+body. All proxies of one body share that single block. The enclosing
// general solver assembles rigid proposals and owns DAT truncation.
class DATColoredStepWorkspace {
public:
    DATColoredStepWorkspace();
    ~DATColoredStepWorkspace();
    DATColoredStepWorkspace(const DATColoredStepWorkspace&) = delete;
    DATColoredStepWorkspace& operator=(const DATColoredStepWorkspace&) = delete;

    void prepare(const RefMesh&, const VertexTriangleMap&, const std::vector<Pin>&,
        const SimParams&, std::size_t count, bool hold_rigid_proxies = false);
    const std::vector<ShapeGrads>& rest_gradients() const;
    // Cached elasticity-only colors, reused across solves with unchanged
    // structural connectivity/ownership and independent of contact snapshots.
    const std::vector<std::vector<int>>& colors() const;
    void freeze_contacts(const SimParams&, const std::vector<Vec3>& snapshot,
        const std::vector<Vec3>* previous, const BroadPhase& candidates,
        const std::vector<int>* selected_blocks = nullptr);
    Vec3 proposal(int node, const RefMesh&, const VertexTriangleMap&,
        const std::vector<Pin>&, const SimParams&,
        const std::vector<Vec3>& live, const std::vector<Vec3>& target,
        const std::vector<Vec3>* previous) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Checkpointed straight-deformable DAT geometry shared by the canonical DAT
// (planar) and OGC (isotropic) solvers. The supplied candidates must cover
// every nonincident VT/EE pair closer than query_radius at reference_positions.
// Ownership/surface exclusions belong to BroadPhase. Reset again whenever
// detection runs; do not reset the reference while reusing old candidates.
// The filter consumes TOTAL displacement from that immutable reference,
// including all accepted colors since detection.
//
// Planar follows Newton's proportional plane placement, 5% interior floor,
// crossing-only relaxation, and a final isotropic radius cap. The per-pair
// plane half-band is max(separation_epsilon, 4*FLT_EPSILON*max(1,abs(world
// coordinates))). Isotropic uses gamma/2 times the minimum reference VT/FV/EE
// distance incident on each vertex (query_radius when no closer pair exists).
//
// Both modes require strictly separated initial nonincident primitives and
// throw for an uncertifiable pair instead of silently skipping it. They do
// not repair intersections, prevent tet inversion, query SDFs, or certify
// curved rigid trajectories. Rigid meshes are rejected. A checkpoint is not
// safe for simultaneous filter calls because scratch storage is reused.
namespace dat {

enum class Mode { Planar, Isotropic };

struct Options {
    Mode mode = Mode::Planar;
    double gamma = 0.85;
    double query_radius = 0.0; // A finite, strictly positive detection radius.
    double separation_epsilon = 1.0e-6;
    double minimum_plane_fraction = 0.05;
    bool use_parallel = false;
};

struct Result {
    // Publish positions directly: these are the represented endpoints checked
    // by the filter. Displacements are measured from the checkpoint positions.
    std::vector<Vec3> positions;
    std::vector<Vec3> displacements;
    std::vector<double> factors;
};

class Checkpoint {
public:
    Checkpoint();
    ~Checkpoint();
    Checkpoint(Checkpoint&&) noexcept;
    Checkpoint& operator=(Checkpoint&&) noexcept;
    Checkpoint(const Checkpoint&) = delete;
    Checkpoint& operator=(const Checkpoint&) = delete;

    void reset(const RefMesh& mesh, const std::vector<Vec3>& reference_positions,
        const BroadPhase& candidates, const Options& options);
    Result filter(const std::vector<Vec3>& accumulated_displacements) const;
    void filter_into(const std::vector<Vec3>& accumulated_displacements,
        Result& output) const;

    const std::vector<Vec3>& reference_positions() const;
    const std::vector<double>& isotropic_bounds() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dat
