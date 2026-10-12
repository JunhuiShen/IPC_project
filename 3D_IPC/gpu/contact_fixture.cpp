#include "barrier_energy.h"
#include "broad_phase.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "physics.h"
#include "solver.h"
#include "time_integration.h"
#ifdef IPC_GPU_FIXTURE
#include "kernels.h"
#endif

#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ContactCounts {
    std::size_t nt_candidates = 0, ss_candidates = 0;
    std::size_t active_nt = 0, active_ss = 0;
    double minimum_distance = std::numeric_limits<double>::infinity();
    double maximum_normal_load = 0;
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

ContactCounts count_contacts(const BroadPhase& broad_phase,
    const std::vector<Vec3>& positions, const SimParams& params) {
    ContactCounts result;
    const auto& cache = broad_phase.cache();
    result.nt_candidates = cache.nt_pairs.size();
    result.ss_candidates = cache.ss_pairs.size();
    for (const auto& pair : cache.nt_pairs) {
        const std::array<Vec3, 4> corners{positions[pair.node],
            positions[pair.tri_v[0]], positions[pair.tri_v[1]], positions[pair.tri_v[2]]};
        const auto evaluation = make_node_triangle_contact_evaluation(corners,
            params.d_hat, params.k_barrier, 1e-12, nullptr, params.exact_computation_fallback);
        result.minimum_distance = std::min(result.minimum_distance, evaluation.dr.distance);
        if (evaluation.active && evaluation.normal_load > 0) ++result.active_nt;
        result.maximum_normal_load = std::max(result.maximum_normal_load, evaluation.normal_load);
    }
    for (const auto& pair : cache.ss_pairs) {
        const std::array<Vec3, 4> corners{positions[pair.v[0]], positions[pair.v[1]],
            positions[pair.v[2]], positions[pair.v[3]]};
        const auto evaluation = make_segment_segment_contact_evaluation(corners,
            params.d_hat, params.k_barrier, 1e-12, nullptr, params.exact_computation_fallback);
        result.minimum_distance = std::min(result.minimum_distance, evaluation.dr.distance);
        if (evaluation.active && evaluation.normal_load > 0) ++result.active_ss;
        result.maximum_normal_load = std::max(result.maximum_normal_load, evaluation.normal_load);
    }
    return result;
}

void print_counts(const ContactCounts& counts) {
    std::cout << "{\"nt_candidates\":" << counts.nt_candidates
              << ",\"ss_candidates\":" << counts.ss_candidates
              << ",\"active_nt\":" << counts.active_nt
              << ",\"active_ss\":" << counts.active_ss
              << ",\"minimum_distance\":";
    if (std::isfinite(counts.minimum_distance)) std::cout << counts.minimum_distance;
    else std::cout << "null";
    std::cout << ",\"maximum_normal_load\":" << counts.maximum_normal_load << '}';
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("Usage: contact_fixture [0|0.1]");
        double friction = 0;
        if (argc == 2) {
            std::size_t consumed = 0;
            friction = std::stod(argv[1], &consumed);
            if (consumed != std::string(argv[1]).size() || (friction != 0 && friction != .1))
                throw std::invalid_argument("Friction must be 0 or 0.1");
        }
        // Select the worker count externally, e.g. OMP_NUM_THREADS=8. Both
        // executables call exactly the same public solver with these flags.
        omp_set_dynamic(0);
        SimParams params = SimParams::zeros();
        params.fps = 30;
        params.substeps = 2;
        params.mu = 20;
        params.lambda = 25;
        params.density = 1;
        params.thickness = .1;
        params.kB = .003;
        params.kpin = 1e6;
        params.gravity = Vec3::Zero();
        params.max_global_iters = 4;
        params.fixed_iters = true;
        params.use_parallel = true;
        params.use_basic_experimental = true;
        params.use_basic_experimental_v2 = true;
        params.use_simd = true;
        params.use_ccd = true;
        params.use_ticcd = false;
        params.exact_computation_fallback = true;
        params.node_box_min = .001;
        params.node_box_max = .01;
        params.node_box_update_count = 2;
        params.d_hat = .012;
        params.k_barrier = 1;
        params.friction_coefficient = friction;
        params.friction_velocity_epsilon = .01;

        RefMesh mesh;
        DeformedState state;
        std::vector<Vec2> material;
        constexpr int subdivisions = 8;
        constexpr int sheet_vertices = (subdivisions + 1) * (subdivisions + 1);
        build_square_mesh(mesh, state, material, subdivisions, subdivisions,
            .45, .45, Vec3(-.31, -.022, -.23));
        build_square_mesh(mesh, state, material, subdivisions, subdivisions,
            .45, .45, Vec3(-.303, -.014, -.227));
        require(!mesh.hinges.empty(), "Contact fixture has no bending hinges");
        state.velocities.resize(state.deformed_positions.size());
        for (std::size_t node = 0; node < state.deformed_positions.size(); ++node) {
            auto& position = state.deformed_positions[node];
            // Keep the flat reference hinges, then bend both nearly parallel
            // sheets smoothly. Their .008 gap is below the .012 barrier range.
            position.x() *= 1.025;
            position.y() += .001 * std::sin(8 * position.x()) * std::sin(7 * position.z());
            const double direction = node < sheet_vertices ? 1 : -1;
            state.velocities[node] = Vec3(.035 * direction, .05 * direction, .025 * direction);
        }
        std::vector<Pin> pins;
        for (int node : {0, subdivisions, sheet_vertices, sheet_vertices + subdivisions})
            append_pin(pins, node, state.deformed_positions);
        pins.back().target_position.x() += .0002;
        mesh.build_lumped_mass(params.density, params.thickness);
        const auto adjacency = build_incident_triangle_map(mesh.tris);
        BroadPhase broad_phase;
        std::vector<AABB> boxes;
        boxes.reserve(state.deformed_positions.size());
        for (const auto& position : state.deformed_positions)
            boxes.emplace_back(position - Vec3::Constant(params.node_box_max),
                position + Vec3::Constant(params.node_box_max));
        broad_phase.initialize(boxes, mesh, params.d_hat, BroadPhase::InitializationMode::DeformableSolver);
        const ContactCounts initial = count_contacts(broad_phase, state.deformed_positions, params);
        require(initial.active_nt + initial.active_ss > 0,
            "Fixture must have actual active barriers before its first solve");
        require(initial.minimum_distance > 0 && std::isfinite(initial.maximum_normal_load)
                && initial.maximum_normal_load > 0,
            "Initial contact geometry or normal barrier load is invalid");

        constexpr int substep_count = 4;
        std::array<ContactCounts, substep_count> after;
        for (int substep = 0; substep < substep_count; ++substep) {
            const auto previous = state.deformed_positions;
            std::vector<Vec3> predicted;
            build_xhat(predicted, previous, state.velocities, params.dt());
            auto next = previous;
            const auto result = global_gauss_seidel_solver_basic_experimental_v2(mesh,
                adjacency, pins, params, next, predicted, state.velocities,
                broad_phase, "", &previous);
            require(result.converged && result.iterations == params.max_global_iters,
                "Contact fixture did not complete its four fixed solver sweeps");
            for (const auto& position : next)
                require(position.allFinite(), "Contact fixture produced nonfinite positions");
            update_velocity(state.velocities, next, previous, params.dt());
            state.deformed_positions = std::move(next);
            after[substep] = count_contacts(broad_phase, state.deformed_positions, params);
            require(after[substep].minimum_distance > 0,
                "Contact fixture reached a zero-gap contact");
        }

        std::cout << std::setprecision(17)
                  << "{\"friction\":" << friction << ",\"threads\":" << omp_get_max_threads()
                  << ",\"substeps\":" << substep_count << ",\"iterations_per_substep\":4"
                  << ",\"vertices\":" << state.deformed_positions.size()
                  << ",\"triangles\":" << mesh.tris.size() / 3
                  << ",\"hinges\":" << mesh.hinges.size() << ",\"initial_contacts\":";
        print_counts(initial);
        std::cout << ",\"contacts_after_substeps\":[";
        for (int substep = 0; substep < substep_count; ++substep) {
            if (substep) std::cout << ',';
            print_counts(after[substep]);
        }
        std::cout << ']';
#ifdef IPC_GPU_FIXTURE
        const auto device_stats=ipc_gpu::statistics();
        require(device_stats.launches>0 && device_stats.triangles>0 && device_stats.hinges>0,
            "GPU contact fixture did not execute CUDA derivatives");
        std::cout << ",\"cuda_launches\":" << device_stats.launches;
#endif
        std::cout << ",\"positions\":[";
        for (std::size_t node = 0; node < state.deformed_positions.size(); ++node) {
            if (node) std::cout << ',';
            const auto& position = state.deformed_positions[node];
            std::cout << '[' << position.x() << ',' << position.y() << ',' << position.z() << ']';
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Contact fixture failed: " << error.what() << '\n';
        return 1;
    }
}
