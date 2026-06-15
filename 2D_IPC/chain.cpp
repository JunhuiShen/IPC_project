#include "chain.h"

#include <utility>

Chain make_chain(Vec2 start, Vec2 end, int N, double density, double thickness) {
    Chain c;
    c.N = N;
    c.deformed_positions.resize(N);
    c.velocities.assign(N, Vec2{0.0, 0.0});
    c.mass.assign(N,0.0);

    //create chain geom
    for (int i = 0; i < N; ++i) {
        double t = (N == 1) ? 0.0 : double(i) / (N - 1);
        Vec2 xi{start.x + t * (end.x - start.x), start.y + t * (end.y - start.y)};
        set_xi(c.deformed_positions, i, xi);
    }

    //set nodal masses
    for(int s=0;s<N-1;s++){
        Vec2 edge=get_xi(c.deformed_positions, s+1)-get_xi(c.deformed_positions, s);
        double segment_length=norm(edge);
        double segment_mass=thickness*thickness*segment_length*density;
        c.mass[s] += .5*segment_mass; c.mass[s+1] += .5*segment_mass;
    }

    return c;
}

void assemble_chains(const std::vector<Chain>& chains, DeformedState& state,
                     RefMesh& ref_mesh, std::vector<Pin>& pins) {
    int total_nodes = 0;
    for (const Chain& chain : chains) total_nodes += chain.N;

    state.deformed_positions.assign(total_nodes, Vec2{0.0, 0.0});
    state.velocities.assign(total_nodes, Vec2{0.0, 0.0});
    std::vector<double> mass(total_nodes, 0.0);
    pins.clear();

    std::vector<std::pair<int, int>> edges;
    int offset = 0;
    for (const Chain& chain : chains) {
        for (int i = 0; i < chain.N; ++i) {
            set_xi(state.deformed_positions, offset + i, get_xi(chain.deformed_positions, i));
            set_xi(state.velocities, offset + i, get_xi(chain.velocities, i));
            mass[offset + i] = chain.mass[i];
        }
        for (int i = 0; i + 1 < chain.N; ++i) {
            edges.emplace_back(offset + i, offset + i + 1);
        }
        for (const Pin& pin : chain.pins) {
            pins.push_back({offset + pin.vertex_index, pin.target_position});
        }
        offset += chain.N;
    }

    ref_mesh.initialize(total_nodes, edges, state.deformed_positions);
    ref_mesh.mass = std::move(mass);
}
