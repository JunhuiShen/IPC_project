#include "chain.h"

void RefMesh::initialize_from_chains(const std::vector<Chain>& chains,
                                     const std::vector<int>& node_offsets) {
    num_positions = 0;
    edges.clear();
    rest_lengths.clear();
    chain_node_offsets = node_offsets;
    chain_rest_offsets.assign(chains.size(), 0);

    for (std::size_t b = 0; b < chains.size(); ++b) {
        const Chain& c = chains[b];
        const int offset = node_offsets[b];
        chain_rest_offsets[b] = static_cast<int>(rest_lengths.size());
        num_positions += c.N;

        for (int i = 0; i < c.N - 1; ++i) {
            edges.emplace_back(offset + i, offset + i + 1);
            rest_lengths.push_back(math::node_distance(c.x, i, i + 1));
        }
    }
}

Chain make_chain(Vec2 start, Vec2 end, int N, double density, double thickness) {
    Chain c;
    c.N = N;
    c.x.resize(2 * N);
    c.v.assign(2 * N, 0.0);
    c.xhat.assign(2 * N, 0.0);
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
        double segment_length=math::norm(edge);
        double segment_mass=thickness*thickness*segment_length*density;
        c.mass[s] += .5*segment_mass; c.mass[s+1] += .5*segment_mass;
    }

    return c;
}

std::vector<int> compute_node_offsets(const std::vector<Chain>& chains) {
    std::vector<int> offsets(chains.size(), 0);
    for (std::size_t b = 1; b < chains.size(); ++b) {
        offsets[b] = offsets[b - 1] + chains[b - 1].N;
    }
    return offsets;
}

RefMesh build_ref_mesh(const std::vector<Chain>& chains,
                       const std::vector<int>& node_offsets) {
    RefMesh ref_mesh;
    ref_mesh.initialize_from_chains(chains, node_offsets);
    return ref_mesh;
}

void build_xhat(Chain& c, double dt) {
    for (int i = 0; i < c.N; ++i) {
        Vec2 xi = get_xi(c.x, i);
        Vec2 vi = get_xi(c.v, i);
        set_xi(c.xhat, i, {xi.x + dt * vi.x, xi.y + dt * vi.y});
    }
}

void update_velocity(Chain& c, const Vec& xnew, double dt) {
    for (int i = 0; i < c.N; ++i) {
        Vec2 xi_new = get_xi(xnew, i);
        Vec2 xi_old = get_xi(c.x, i);
        set_xi(c.v, i, {(xi_new.x - xi_old.x) / dt, (xi_new.y - xi_old.y) / dt});
    }
    c.x = xnew;
}

void scatter_positions(Vec& x_combined, const Vec& x_block, int offset, int N_block) {
    for (int i = 0; i < N_block; ++i)
        set_xi(x_combined, offset + i, get_xi(x_block, i));
}

void scatter_chain_positions(Vec& x_combined, const Chain& c, int offset) {
    scatter_positions(x_combined, c.x, offset, c.N);
}
