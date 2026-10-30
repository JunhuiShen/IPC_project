#pragma once

#include "scene_types.h"
#include "rigid_pins_types.h"
#include <array>
#include <memory>
#include <vector>

namespace ipc_gpu::solver {

using Vector3 = std::array<double,3>;
using Quaternion = std::array<double,4>;
using Matrix3 = std::array<double,9>;

// Upload-only scene description. Rest data/topology preparation and file I/O
// may run on the host. No evolving element/contact data belongs in this type.
struct SceneDescription {
    std::vector<Vector3> positions, velocities, pin_targets, body_reference;
    std::vector<unsigned char> deformable, solid, pinned, surface;
    std::vector<double> mass;
    std::vector<int> node_blocks, node_body;
    std::vector<TriangleIncident> triangles;
    std::vector<SolidIncident> solids;
    std::vector<HingeIncident> hinges;
    std::vector<std::size_t> triangle_offsets, solid_offsets, hinge_offsets;
    std::vector<BpTriangle> surface_triangles;
    std::vector<BpEdge> edges;
    std::vector<BpDependency> dependencies;
    int block_count = 0;
    std::vector<int> body_blocks, body_nodes;
    std::vector<std::size_t> body_node_offsets;
    std::vector<double> body_mass;
    std::vector<Matrix3> body_second_moment;
    std::vector<unsigned char> body_translate, body_rotate;
    std::vector<Vector3> com, com_velocity, omega;
    std::vector<Quaternion> orientation;
    // Static material-point attachments, stably grouped by body.
    std::vector<RigidPinConstraint> rigid_pins;
    std::vector<std::size_t> rigid_pin_offsets;
};

struct StepParameters {
    Parameters elastic{};
    SolidParameters solid{};
    ContactParameters contact{};
    SdfParameters sdf{};
    double dt = 1.0/300.0;
    double damping = 1.0;
    double box_radius_min = 0.001;
    double box_radius = 0.01;
    double rotation_radius = 0.1;
    int iterations = 10;
    int rebuild_interval = 10;
    int colored_guess_sweeps = 0;
    // Fixed initial-guess target fraction; never changes the solver's xhat.
    double colored_guess_fraction = 1.0;
    // Optional contact-only trial/BFS level schedule for colored CCD guesses.
    bool colored_guess_level_based = false;
    // Optional exact separating-plane proposal; acceptance still uses CCD.
    bool ccd_exact_step_bound = false;
    // Standard CPU cloth predictor: all vertices move simultaneously,
    // then one global 0.9 * earliest-TOI fraction is used for the whole mesh.
    // A configured colored guess takes precedence in describe_step(). The
    // general (solid/mixed/rigid) path ignores this flag, matching the CPU.
    bool global_ccd_guess = false;
    // Particle-update CCD only. Rigid updates and colored guesses always use
    // CCD, as in the production general solver. Does not disable barriers.
    bool collisions = true;
    // Optional frozen-state Jacobi updates followed by planar DAT. This
    // deformable-only mode skips all initial guesses and PBGS/vertex CCD.
    bool jacobi_dat = false;
    double dat_gamma = 0.9;
    double dat_query_radius = 0.0; // 0 selects the CPU automatic-radius rule.
    double dat_node_box_padding = 1.2;
    double rigid_pin_time = 0.0;
    // Authoritative geometry/CCD mode for the ordinary solver. true keeps the
    // robust exact-fallback path; false selects the original non-exact CPU
    // algorithms (original contact geometry, original linear CCD, CPU
    // componentwise clip and raw-displacement commit). Appended last so old
    // aggregate initialization and default construction stay compatible.
    bool exact_computation_fallback = true;
    // Logical CPU reference team size used only to model the CPU original
    // cooperative contact dispatcher (ColoredContactSweep::prepare). It is
    // positive and bounded; 1 selects the noncooperative StoredNoop rule.
    // The host OpenMP team size never changes this value.
    int original_reference_threads = 64;
    // Scheduler-FAMILY metadata derived by describe_step from the actual CPU
    // dispatch. It is not a geometry toggle: StoredNoop keeps CPU scalar/basic
    // semantics, CooperativeSquaredNorm enables the original cooperative
    // selected-node 1e-28 cutoff. Default StoredNoop preserves all callers.
    OriginalUpdateConvention original_update_convention =
        OriginalUpdateConvention::StoredNoop;
    // Narrow predictor-intent metadata, appended last so every existing
    // aggregate initialization and default construction is unchanged. The CPU
    // validates forced positions/targets (and builds nothing else) even when
    // zero colored-guess sweeps are requested, so "enabled with zero sweeps"
    // is NOT the same request as "disabled". Effective activity is this flag
    // OR a positive sweep budget OR the level-based flag; it is never a
    // geometry toggle.
    bool colored_guess_enabled = false;
    // Resolved CPU energy-dispatch intent, independent of the GPU host team.
    // Original v2 cloth uses its CPU blue-box cache even when parallelism is
    // off or friction is enabled. Direct API defaults keep unrequested policy.
    bool cpu_energy_simd = false;
};

struct StateSnapshot {
    std::vector<Vector3> positions, velocities, com, com_velocity, omega;
    std::vector<Quaternion> orientation;
};

// Persistent device state. Upload prescribed boundary data per substep;
// download only for requested visualization/checkpoints. Host orchestration
// of kernel launches, counts, and error reporting is not a CPU numerical path.
class GpuSolver {
public:
    explicit GpuSolver(const SceneDescription&);
    ~GpuSolver();
    GpuSolver(const GpuSolver&) = delete;
    GpuSolver& operator=(const GpuSolver&) = delete;
    void advance_substep(const StepParameters&, const std::vector<Vector3>& pin_targets,
                         const std::vector<SdfCollider>& colliders);
    StateSnapshot snapshot() const;
    // Optional diagnostics, enabled only by IPC_GPU_PROFILE_CSV. They do not
    // change simulation parameters or the numerical update path.
    void begin_profile_frame(int frame);
    void end_profile_frame();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ipc_gpu::solver
