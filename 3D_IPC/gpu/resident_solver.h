#pragma once

#include "resident_scene_types.h"
#include <array>
#include <memory>
#include <vector>

namespace ipc_gpu::resident {

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
};

struct StateSnapshot {
    std::vector<Vector3> positions, velocities, com, com_velocity, omega;
    std::vector<Quaternion> orientation;
};

// Persistent device state. Upload prescribed boundary data per substep;
// download only for requested visualization/checkpoints. Host orchestration
// of kernel launches, counts, and error reporting is not a CPU numerical path.
class ResidentSolver {
public:
    explicit ResidentSolver(const SceneDescription&);
    ~ResidentSolver();
    ResidentSolver(const ResidentSolver&) = delete;
    ResidentSolver& operator=(const ResidentSolver&) = delete;
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

} // namespace ipc_gpu::resident
