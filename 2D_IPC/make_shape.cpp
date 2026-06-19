#include "make_shape.h"
#include "rigid_body_ipc.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

constexpr double kPi = 3.14159265358979323846;

void rebuild_ref_mesh(RefMesh& ref_mesh, const Vec& rest_positions) {
    ref_mesh.initialize(static_cast<int>(rest_positions.size()), ref_mesh.edges, rest_positions);
}

}  // namespace

Chain make_chain(Vec2 start, Vec2 end, int N, double density, double thickness) {
    Chain c;
    c.N = N;
    c.deformed_positions.resize(N);
    c.velocities.assign(N, Vec2{0.0, 0.0});
    c.mass.assign(N, 0.0);

    for (int i = 0; i < N; ++i) {
        double t = (N == 1) ? 0.0 : double(i) / (N - 1);
        Vec2 xi{start.x + t * (end.x - start.x), start.y + t * (end.y - start.y)};
        set_xi(c.deformed_positions, i, xi);
    }

    for (int s = 0; s < N - 1; ++s) {
        Vec2 edge = get_xi(c.deformed_positions, s + 1) - get_xi(c.deformed_positions, s);
        double segment_length = norm(edge);
        double segment_mass = thickness * thickness * segment_length * density;
        c.mass[s] += 0.5 * segment_mass;
        c.mass[s + 1] += 0.5 * segment_mass;
    }

    return c;
}

void assemble_chains(const std::vector<Chain>& chains, DeformedState& state, RefMesh& ref_mesh, std::vector<Pin>& pins) {
    int total_nodes = 0;
    for (const Chain& chain : chains) total_nodes += chain.N;

    state.deformed_positions.assign(total_nodes, Vec2{0.0, 0.0});
    state.velocities.assign(total_nodes, Vec2{0.0, 0.0});
    std::vector<double> mass(total_nodes, 0.0);
    pins.clear();

    ref_mesh.edges.clear();
    ref_mesh.ref_positions.clear();
    ref_mesh.inertia_tensor.clear();
    ref_mesh.total_mass.clear();
    ref_mesh.rb_nodes.clear();

    int offset = 0;
    for (const Chain& chain : chains) {
        for (int i = 0; i < chain.N; ++i) {
            set_xi(state.deformed_positions, offset + i, get_xi(chain.deformed_positions, i));
            set_xi(state.velocities, offset + i, get_xi(chain.velocities, i));
            mass[offset + i] = chain.mass[i];
        }
        for (int i = 0; i + 1 < chain.N; ++i) {
            ref_mesh.edges.emplace_back(offset + i, offset + i + 1);
        }
        for (const Pin& pin : chain.pins) {
            pins.push_back({offset + pin.vertex_index, pin.target_position});
        }
        offset += chain.N;
    }

    rebuild_ref_mesh(ref_mesh, state.deformed_positions);
    ref_mesh.mass = std::move(mass);
}

int append_rigid_pentagon(DeformedState& state, RefMesh& ref_mesh, Vec2 center, double radius, double density, double thickness, Vec2 v_com, double theta, double omega) {

    const int rb = static_cast<int>(state.x_coms.size());
    const int base = static_cast<int>(state.deformed_positions.size());

    Vec x_local(5);
    for (int i = 0; i < 5; ++i) {
        const double angle = theta + 2.0 * kPi * double(i) / 5.0;
        x_local[i] = {center.x + radius * std::cos(angle), center.y + radius * std::sin(angle)
        };
    }

    // A regular pentagon has area A = (5 / 2) R^2 sin(2 pi / 5)
    const double area = 0.5 * 5.0 * radius * radius * std::sin(2.0 * kPi / 5.0);
    const double total_mass = area * thickness * density;
    const double nodal_mass = total_mass / 5.0;

    state.deformed_positions.insert(state.deformed_positions.end(), x_local.begin(), x_local.end());
    state.velocities.insert(state.velocities.end(), 5, v_com);

    if (ref_mesh.mass.size() < static_cast<std::size_t>(base)) {
        ref_mesh.mass.resize(base, 0.0);
    }
    ref_mesh.mass.insert(ref_mesh.mass.end(), 5, nodal_mass);

    for (int i = 0; i < 5; ++i) {
        ref_mesh.edges.emplace_back(base + i, base + ((i + 1) % 5));
    }

    Vec2 x_com;
    Vec2 v_com_out;
    double theta_out;
    double omega_out;
    Mat2 inertia;
    Vec ref_positions;
    create_rigid_body(x_local, v_com, theta, omega, total_mass, x_com, v_com_out, theta_out, omega_out, inertia, ref_positions);

    state.x_coms.push_back(x_com);
    state.v_coms.push_back(v_com_out);
    state.theta.push_back(theta_out);
    state.omega.push_back(omega_out);

    ref_mesh.ref_positions.push_back(ref_positions);
    ref_mesh.inertia_tensor.push_back(inertia);
    ref_mesh.total_mass.push_back(total_mass);
    ref_mesh.rb_nodes.push_back({base, base + 1, base + 2, base + 3, base + 4});

    rebuild_ref_mesh(ref_mesh, state.deformed_positions);
    return rb;
}
