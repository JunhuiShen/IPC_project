#include "barrier_energy.h"
#include "broad_phase.h"
#include "make_shape.h"
#include "mesh_utils.h"
#include "simulation.h"
#include "solid_ipc.h"
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

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct ContactCounts {
    std::size_t nt_candidates = 0, ss_candidates = 0;
    std::size_t active_nt = 0, active_ss = 0;
    std::size_t cloth_solid = 0, cloth_rigid = 0, solid_rigid = 0;
    double minimum_distance = std::numeric_limits<double>::infinity();
    double maximum_normal_load = 0.0;
};

ContactCounts count_contacts(const BroadPhase& broad_phase,
    const std::vector<Vec3>& positions, const SimParams& params,
    const std::vector<unsigned char>& kinds) {
    ContactCounts result;
    const auto& cache = broad_phase.cache();
    result.nt_candidates = cache.nt_pairs.size();
    result.ss_candidates = cache.ss_pairs.size();
    const auto record = [&](const std::array<int, 4>& nodes, const auto& evaluation,
                            bool segment) {
        result.minimum_distance = std::min(result.minimum_distance, evaluation.dr.distance);
        result.maximum_normal_load = std::max(result.maximum_normal_load, evaluation.normal_load);
        if (!evaluation.active || evaluation.normal_load <= 0.0) return;
        if (segment) ++result.active_ss;
        else ++result.active_nt;
        unsigned kind = 0;
        for (int node : nodes) kind |= kinds[node];
        if ((kind & 3) == 3) ++result.cloth_solid;
        if ((kind & 5) == 5) ++result.cloth_rigid;
        if ((kind & 6) == 6) ++result.solid_rigid;
    };
    for (const auto& pair : cache.nt_pairs) {
        const std::array<int, 4> nodes{pair.node, pair.tri_v[0], pair.tri_v[1], pair.tri_v[2]};
        const std::array<Vec3, 4> corners{positions[nodes[0]], positions[nodes[1]],
            positions[nodes[2]], positions[nodes[3]]};
        record(nodes, make_node_triangle_contact_evaluation(corners,
            params.d_hat, params.k_barrier, 1e-12, nullptr,
            params.exact_computation_fallback), false);
    }
    for (const auto& pair : cache.ss_pairs) {
        const std::array<int, 4> nodes{pair.v[0], pair.v[1], pair.v[2], pair.v[3]};
        const std::array<Vec3, 4> corners{positions[nodes[0]], positions[nodes[1]],
            positions[nodes[2]], positions[nodes[3]]};
        record(nodes, make_segment_segment_contact_evaluation(corners,
            params.d_hat, params.k_barrier, 1e-12, nullptr,
            params.exact_computation_fallback), true);
    }
    return result;
}

void print_counts(const ContactCounts& counts) {
    std::cout << "{\"nt_candidates\":" << counts.nt_candidates
              << ",\"ss_candidates\":" << counts.ss_candidates
              << ",\"active_nt\":" << counts.active_nt
              << ",\"active_ss\":" << counts.active_ss
              << ",\"cloth_solid\":" << counts.cloth_solid
              << ",\"cloth_rigid\":" << counts.cloth_rigid
              << ",\"solid_rigid\":" << counts.solid_rigid
              << ",\"minimum_distance\":";
    if (std::isfinite(counts.minimum_distance)) std::cout << counts.minimum_distance;
    else std::cout << "null";
    std::cout << ",\"maximum_normal_load\":" << counts.maximum_normal_load << '}';
}

template<class Vector>
void print_vectors(const char* name, const std::vector<Vector>& values) {
    std::cout << ",\"" << name << "\":[";
    for (std::size_t entry = 0; entry < values.size(); ++entry) {
        if (entry) std::cout << ',';
        std::cout << '[';
        for (int axis = 0; axis < values[entry].size(); ++axis) {
            if (axis) std::cout << ',';
            std::cout << values[entry][axis];
        }
        std::cout << ']';
    }
    std::cout << ']';
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("Usage: mixed_fixture [0|0.1]");
        double friction = 0.0;
        if (argc == 2) {
            std::size_t consumed = 0;
            friction = std::stod(argv[1], &consumed);
            if (consumed != std::string(argv[1]).size() || (friction != 0.0 && friction != 0.1))
                throw std::invalid_argument("Friction must be 0 or 0.1");
        }
        omp_set_dynamic(0);
#ifdef IPC_GPU_FIXTURE
        ipc_gpu::reset_statistics();
#endif
        SimParams params = SimParams::zeros();
        params.fps = 30.0;
        params.substeps = 1;
        params.mu = 2.0;
        params.lambda = 3.0;
        params.kB = 1e-4;
        params.kpin = 1000.0;
        params.solid_mu = 2.0;
        params.solid_lambda = 3.0;
        params.gravity = Vec3(0.0, -0.1, 0.0);
        params.node_box_min = 0.05;
        params.node_box_max = 0.05;
        params.theta_box_min = 0.02;
        params.theta_box_max = 0.02;
        params.node_box_update_count = 2;
        params.max_global_iters = 4;
        params.fixed_iters = true;
        params.damping = 0.1;
        params.use_parallel = true;
        params.use_basic_experimental = true;
        params.use_basic_experimental_v2 = true;
        params.use_simd = true;
        params.use_ccd = true;
        params.use_ticcd = false;
        params.use_ccd_guess = false;
        params.exact_computation_fallback = true;
        params.d_hat = 0.03;
        params.k_barrier = 1.0;
        params.friction_coefficient = friction;
        params.friction_velocity_epsilon = 0.01;

        RefMesh mesh;
        DeformedState state;
        std::vector<Vec2> material;
        std::vector<Pin> pins;
        build_square_mesh(mesh, state, material, 6, 6, 1.0, 1.0,
            Vec3(-0.5, 0.0, -0.5));
        mesh.build_lumped_mass(900.0, 0.001);
        state.velocities.assign(state.deformed_positions.size(), Vec3(-0.005, -0.003, 0.002));
        for (Vec3& position : state.deformed_positions) {
            position.x() *= 1.005;
            position.y() += 0.0005 * std::sin(4.0 * position.x()) * std::cos(3.0 * position.z());
        }
        append_pin(pins, 0, state.deformed_positions);
        append_pin(pins, 6, state.deformed_positions);
        const int cloth_count = static_cast<int>(state.deformed_positions.size());

        // Four tetrahedra share an interior node; all three object types are
        // separated by positive gaps smaller than the barrier activation gap.
        std::vector<Vec3> solid_positions{
            Vec3(-0.08, 0.006, -0.08), Vec3(0.08, 0.006, -0.08),
            Vec3(-0.08, 0.006, 0.08), Vec3(-0.08, 0.012, -0.08)};
        solid_positions.push_back(0.25 * (solid_positions[0] + solid_positions[1]
            + solid_positions[2] + solid_positions[3]));
        create_solid(solid_positions,
            {4, 2, 1, 3, 0, 4, 1, 3, 0, 2, 4, 3, 0, 2, 1, 4}, 900.0, mesh, state);
        for (int node = cloth_count; node < static_cast<int>(state.velocities.size()); ++node)
            state.velocities[node] = Vec3(0.003, -0.002, -0.004);
        state.deformed_positions[cloth_count + 3] += Vec3(0.0002, 0.0001, -0.0001);
        state.deformed_positions[cloth_count + 4] += Vec3(-0.0001, 0.00005, 0.0001);
        append_pin(pins, cloth_count + 1, state.deformed_positions);
        pins.back().target_position.x() += 0.00002;

        const int rigid = create_rigid_body(
            {Vec3(-0.4, 0.018, -0.4), Vec3(0.4, 0.018, -0.4), Vec3(0.0, 0.018, 0.4)},
            Vec3(0.01, -0.01, 0.02), Vec4(1.0, 0.0, 0.0, 0.0),
            Vec3(0.02, -0.01, 0.01), 1.0, mesh, state,
            RigidBodyUpdateMode::TranslationAndOrientation);
        mesh.tris.insert(mesh.tris.end(), mesh.rb_nodes[rigid].begin(), mesh.rb_nodes[rigid].end());
        mesh.node_to_rb.resize(state.deformed_positions.size(), -1);
        mesh.build_deformable_nodes();
        const auto adjacency = build_incident_triangle_map(mesh.tris);
        std::vector<unsigned char> kinds(state.deformed_positions.size(), 1);
        for (int node : mesh.tet_nodes) kinds[node] = 2;
        for (const auto& body : mesh.rb_nodes)
            for (int node : body) kinds[node] = 4;
        require(!mesh.hinges.empty() && !mesh.tets.empty() && !mesh.rb_nodes.empty(),
            "Mixed fixture must include cloth bending, solids, and a rigid body");
        require(mesh.surface_nodes.size() < mesh.tet_nodes.size(),
            "Mixed fixture must include a tetrahedral interior node");

        BroadPhase broad_phase;
        std::vector<AABB> boxes;
        for (const auto& position : state.deformed_positions)
            boxes.emplace_back(position - Vec3::Constant(params.node_box_max),
                position + Vec3::Constant(params.node_box_max));
        broad_phase.initialize_surface_nodes(boxes, mesh, params.d_hat,
            BroadPhase::InitializationMode::GeneralSolver);
        const ContactCounts initial = count_contacts(broad_phase, state.deformed_positions, params, kinds);
        require(initial.cloth_solid > 0 && initial.cloth_rigid > 0 && initial.solid_rigid > 0,
            "Mixed fixture needs actual cloth/solid, cloth/rigid, and solid/rigid barriers");
        require(initial.minimum_distance > 0.0 && std::isfinite(initial.maximum_normal_load)
                && initial.maximum_normal_load > 0.0,
            "Mixed fixture has invalid initial contact geometry");

        constexpr int substep_count = 4;
        std::array<ContactCounts, substep_count> after;
        for (int substep = 0; substep < substep_count; ++substep) {
            std::vector<Vec3> predicted;
            build_xhat(predicted, state.deformed_positions, state.velocities, params.dt());
            for (std::size_t node = 0; node < predicted.size(); ++node)
                if (mesh.node_to_rb[node] >= 0) predicted[node] = state.deformed_positions[node];
            auto positions = state.deformed_positions;
            auto centers = state.x_coms;
            auto orientations = state.orientations;
            std::vector<Vec3> omega(state.omega.size(), Vec3::Zero());
            const auto result = global_gauss_seidel_solver_general_experimental_v2(mesh,
                state, adjacency, pins, params, positions, predicted, centers,
                orientations, omega, broad_phase, "");
            require(result.converged && result.iterations == params.max_global_iters,
                "Mixed fixture did not finish all fixed sweeps");
            for (std::size_t node = 0; node < positions.size(); ++node) {
                require(positions[node].allFinite(), "Mixed fixture produced nonfinite positions");
                if (mesh.node_to_rb[node] >= 0) continue;
                state.velocities[node] = (positions[node] - state.deformed_positions[node]) / params.dt();
                state.deformed_positions[node] = positions[node];
            }
            for (std::size_t body = 0; body < centers.size(); ++body)
                require(centers[body].allFinite() && orientations[body].allFinite() && omega[body].allFinite(),
                    "Mixed fixture produced nonfinite rigid coordinates");
            update_velocity(state.v_coms, centers, state.x_coms, params.dt());
            state.x_coms = std::move(centers);
            state.orientations = std::move(orientations);
            state.omega = std::move(omega);
            sync_rigid_body_particles(mesh, state);
            after[substep] = count_contacts(broad_phase, state.deformed_positions, params, kinds);
            require(after[substep].minimum_distance > 0.0, "Mixed fixture reached a zero-gap contact");
        }

        std::cout << std::setprecision(17) << "{\"friction\":" << friction
                  << ",\"threads\":" << omp_get_max_threads()
                  << ",\"substeps\":" << substep_count << ",\"iterations_per_substep\":4"
                  << ",\"vertices\":" << state.deformed_positions.size()
                  << ",\"cloth_vertices\":" << cloth_count
                  << ",\"solid_vertices\":" << mesh.tet_nodes.size()
                  << ",\"rigid_bodies\":" << mesh.rb_nodes.size()
                  << ",\"triangles\":" << mesh.Dm_inverse.size()
                  << ",\"hinges\":" << mesh.hinges.size()
                  << ",\"tetrahedra\":" << mesh.tets.size() / 4
                  << ",\"initial_contacts\":";
        print_counts(initial);
        std::cout << ",\"contacts_after_substeps\":[";
        for (int substep = 0; substep < substep_count; ++substep) {
            if (substep) std::cout << ',';
            print_counts(after[substep]);
        }
        std::cout << ']';
#ifdef IPC_GPU_FIXTURE
        const auto stats = ipc_gpu::statistics();
        require(stats.launches > 0 && stats.points > 0 && stats.triangles > 0
                && stats.hinges > 0 && stats.solids > 0 && stats.rigid_contacts > 0
                && stats.rigid_inertia > 0,
            "Mixed GPU fixture did not execute every requested derivative family");
        std::cout << ",\"cuda_launches\":" << stats.launches
                  << ",\"cuda_points\":" << stats.points
                  << ",\"cuda_triangles\":" << stats.triangles
                  << ",\"cuda_hinges\":" << stats.hinges
                  << ",\"cuda_solids\":" << stats.solids
                  << ",\"cuda_rigid_contacts\":" << stats.rigid_contacts
                  << ",\"cuda_rigid_inertia\":" << stats.rigid_inertia;
#endif
        print_vectors("positions", state.deformed_positions);
        print_vectors("velocities", state.velocities);
        print_vectors("x_coms", state.x_coms);
        print_vectors("v_coms", state.v_coms);
        print_vectors("orientations", state.orientations);
        print_vectors("omega", state.omega);
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Mixed fixture failed: " << error.what() << '\n';
        return 1;
    }
}
