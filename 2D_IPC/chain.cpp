#include "chain.h"

Chain make_chain(Vec2 start, Vec2 end, int N, double density, double thickness) {
    Chain c;
    c.N = N;
    c.x.resize(2 * N);
    c.v.assign(2 * N, 0.0);
    c.xpin.assign(2 * N, 0.0);
    c.mass.assign(N,0.0);
    c.is_pinned.assign(N, 0);

    //create chain geom
    for (int i = 0; i < N; ++i) {
        double t = (N == 1) ? 0.0 : double(i) / (N - 1);
        Vec2 xi{start.x + t * (end.x - start.x), start.y + t * (end.y - start.y)};
        set_xi(c.x, i, xi);
        set_xi(c.xpin, i, xi);
    }

    //set nodal masses
    for(int s=0;s<N-1;s++){
        Vec2 edge=get_xi(c.x, s+1)-get_xi(c.x, s);
        double segment_length=norm(edge);
        double segment_mass=thickness*thickness*segment_length*density;
        c.mass[s] += .5*segment_mass; c.mass[s+1] += .5*segment_mass;
    }

    return c;
}

void assemble_chains(const std::vector<Chain>& chains, State2D& state, RefMesh& ref_mesh) {
    int total_nodes = 0;
    for (const Chain& chain : chains) total_nodes += chain.N;

    state.x.assign(2 * total_nodes, 0.0);
    state.v.assign(2 * total_nodes, 0.0);
    state.xhat.assign(2 * total_nodes, 0.0);
    state.xpin.assign(2 * total_nodes, 0.0);
    state.mass.assign(total_nodes, 0.0);
    state.is_pinned.assign(total_nodes, 0);

    std::vector<std::pair<int, int>> edges;
    int offset = 0;
    for (const Chain& chain : chains) {
        for (int i = 0; i < chain.N; ++i) {
            set_xi(state.x, offset + i, get_xi(chain.x, i));
            set_xi(state.v, offset + i, get_xi(chain.v, i));
            set_xi(state.xpin, offset + i, get_xi(chain.xpin, i));
            state.mass[offset + i] = chain.mass[i];
            state.is_pinned[offset + i] = chain.is_pinned[i];
        }
        for (int i = 0; i + 1 < chain.N; ++i) {
            edges.emplace_back(offset + i, offset + i + 1);
        }
        offset += chain.N;
    }

    initialize_ref_mesh(ref_mesh, total_nodes, edges, state.x);
}
