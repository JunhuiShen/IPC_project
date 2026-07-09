#include "parallel_helper.h"
#include "IPC_math.h"
#include "ccd.h"
#include "make_shape.h"
#include "ogc_trust_region.h"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline int ph_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static inline int ph_thread_num() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

void compute_local_newton_direction(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins,
                                    const SimParams& params, const std::vector<Vec3>& x, const std::vector<Vec3>& xhat,
                                    const BroadPhase::Cache& bp_cache, Vec3& g_out, Mat33& H_out, Vec3& delta_out,
                                    const PinMap* pin_map,
                                    const std::vector<TriPrecompute>* tri_cache,
                                    const std::vector<HingePrecompute>* hinge_cache){
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x, xhat, pin_map, tri_cache, hinge_cache);

    if (params.d_hat > 0.0) {
        const double dt2k = params.dt2() * params.k_barrier;

        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            auto [bg, bH] = node_triangle_barrier_gradient_and_hessian( x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, entry.dof);
            g += dt2k * bg;
            H += dt2k * bH;
        }

        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(
                    x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, entry.dof);
            g += dt2k * bg;
            H += dt2k * bH;
        }
    }

    g_out = g;
    H_out = H;

    delta_out = matrix3d_inverse(H) * g;
}

std::vector<std::vector<int>> build_elastic_adj(const RefMesh& ref_mesh, const VertexTriangleMap& adj, int nv){
    std::vector<std::vector<int>> out(nv);
    #pragma omp parallel for schedule(static)
    for (int vi = 0; vi < nv; ++vi) {
        auto it = adj.find(vi);
        if (it == adj.end()) continue;
        std::vector<int>& row = out[vi];
        for (const auto& [ti, local_a] : it->second) {
            for (int local_b = 0; local_b < 3; ++local_b) {
                const int vj = tri_vertex(ref_mesh, ti, local_b);
                if (vj == vi || vj < 0 || vj >= nv) continue;
                row.push_back(vj);
            }
        }

        // A hinge's two apex vertices (h.v[2], h.v[3]) are coupled through the
        // bending term but share no triangle, so the 1-ring adjacency above
        // misses that pair. The shared-edge endpoints (h.v[0], h.v[1]) are
        // already adjacent to both apexes via the two triangles.
        auto hinge_it = ref_mesh.hinge_adj.find(vi);
        if (hinge_it != ref_mesh.hinge_adj.end()) {
            for (const auto& [hi, role] : hinge_it->second) {
                if (role < 2) continue;
                const Hinge& h = ref_mesh.hinges[hi];
                const int other_apex = h.v[role == 2 ? 3 : 2];
                if (other_apex != vi && other_apex >= 0 && other_apex < nv) row.push_back(other_apex);
            }
        }

        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return out;
}

static void prepare_int_rows(std::vector<std::vector<int>>& rows, int n) {
    if (static_cast<int>(rows.size()) == n) {
        for (auto& row : rows) row.clear();
    } else {
        rows.assign(n, {});
    }
}

static std::vector<std::vector<std::vector<int>>>& prepare_contact_local_rows(ContactAdjacencyScratch& scratch, int threads, int nv) {
    if (scratch.threads == threads && scratch.vertices == nv) {
        for (auto& thread_rows : scratch.local_nbr) {
            for (auto& row : thread_rows) row.clear();
        }
    } else {
        scratch.local_nbr.assign(threads, std::vector<std::vector<int>>(nv));
        scratch.threads = threads;
        scratch.vertices = nv;
    }
    return scratch.local_nbr;
}

void build_contact_adj(const BroadPhase::Cache& bp_cache, int nv, std::vector<std::vector<int>>& out, ContactAdjacencyScratch* scratch) {
    const int T = ph_max_threads();
    ContactAdjacencyScratch local_scratch;
    auto& local_nbr = scratch? prepare_contact_local_rows(*scratch, T, nv): prepare_contact_local_rows(local_scratch, T, nv);
    const int n_nt = static_cast<int>(bp_cache.nt_pairs.size());
    const int n_ss = static_cast<int>(bp_cache.ss_pairs.size());

    #pragma omp parallel
    {
        auto& lg = local_nbr[ph_thread_num()];
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n_nt; ++i) {
            const auto& p = bp_cache.nt_pairs[i];
            const int verts[4] = { p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2] };
            for (int a = 0; a < 4; ++a) {
                if (verts[a] < 0 || verts[a] >= nv) continue;
                for (int b = a + 1; b < 4; ++b) {
                    if (verts[b] < 0 || verts[b] >= nv) continue;
                    lg[verts[a]].push_back(verts[b]);
                    lg[verts[b]].push_back(verts[a]);
                }
            }
        }
        #pragma omp for schedule(static)
        for (int i = 0; i < n_ss; ++i) {
            const auto& p = bp_cache.ss_pairs[i];
            for (int a = 0; a < 4; ++a) {
                const int va = p.v[a];
                if (va < 0 || va >= nv) continue;
                for (int b = a + 1; b < 4; ++b) {
                    const int vb = p.v[b];
                    if (vb < 0 || vb >= nv) continue;
                    lg[va].push_back(vb);
                    lg[vb].push_back(va);
                }
            }
        }
    }

    prepare_int_rows(out, nv);
    #pragma omp parallel for schedule(dynamic, 64)
    for (int vi = 0; vi < nv; ++vi) {
        std::size_t total = 0;
        for (int t = 0; t < T; ++t) total += local_nbr[t][vi].size();
        out[vi].reserve(total);
        for (int t = 0; t < T; ++t) {
            out[vi].insert(out[vi].end(), local_nbr[t][vi].begin(), local_nbr[t][vi].end());
        }
        std::sort(out[vi].begin(), out[vi].end());
        out[vi].erase(std::unique(out[vi].begin(), out[vi].end()), out[vi].end());
    }
}

void union_adjacency(const std::vector<std::vector<int>>& a,const std::vector<std::vector<int>>& b, std::vector<std::vector<int>>& out) {
    const int nv = static_cast<int>(std::max(a.size(), b.size()));
    prepare_int_rows(out, nv);
    #pragma omp parallel for schedule(dynamic, 64)
    for (int vi = 0; vi < nv; ++vi) {
        const auto* pa = (vi < static_cast<int>(a.size())) ? &a[vi] : nullptr;
        const auto* pb = (vi < static_cast<int>(b.size())) ? &b[vi] : nullptr;
        if (!pa || pa->empty()) {
            if (pb) out[vi].insert(out[vi].end(), pb->begin(), pb->end());
            continue;
        }
        if (!pb || pb->empty()) {
            out[vi].insert(out[vi].end(), pa->begin(), pa->end());
            continue;
        }
        out[vi].reserve(pa->size() + pb->size());
        std::set_union(pa->begin(), pa->end(), pb->begin(), pb->end(), std::back_inserter(out[vi]));
    }
}

void greedy_color_conflict_graph(const std::vector<std::vector<int>>& graph, std::vector<std::vector<int>>& groups) {
    const int nv = static_cast<int>(graph.size());
    std::vector<int> color(nv, -1);
    int max_color = -1;

    for (int vi = 0; vi < nv; ++vi) {
        std::vector<char> used(max_color + 2, 0);
        for (int nb : graph[vi]) {
            if (nb >= 0 && nb < nv && color[nb] >= 0)
                used[color[nb]] = 1;
        }
        int c = 0;
        while (c < static_cast<int>(used.size()) && used[c]) ++c;
        color[vi] = c;
        max_color = std::max(max_color, c);
    }

    const int num_groups = max_color + 1;
    if (static_cast<int>(groups.size()) == num_groups) {
        for (auto& group : groups) group.clear();
    } else {
        groups.assign(num_groups, {});
    }
    for (int vi = 0; vi < nv; ++vi) {
        if (color[vi] >= 0) groups[color[vi]].push_back(vi);
    }
}

double compute_safe_step_for_vertex(int vi, const RefMesh& ref_mesh, const SimParams& params, const std::vector<Vec3>& x, const Vec3& delta,
                                    const BroadPhase::Cache& bp_cache){
    if (params.d_hat <= 0.0) return 1.0;

    const Vec3 dx = -delta;
    const bool tr = params.use_ogc;

    double safe_min = 1.0;

    if (vi >= 0 && vi < static_cast<int>(bp_cache.vertex_nt.size())) {
        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            // dof: 0=node, 1=tri_v[0], 2=tri_v[1], 3=tri_v[2]
            if (entry.dof == 0) {
                // vi is the lone moving node vs a static triangle.
                if (tr) {
                    auto r = trust_region_vertex_triangle_gauss_seidel(
                        x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], dx);
                    safe_min = std::min(safe_min, r.omega);
                } else {
                    CCDResult r = node_triangle_only_one_node_moves(
                        x[p.node],     dx,
                        x[p.tri_v[0]], Vec3::Zero(),
                        x[p.tri_v[1]], Vec3::Zero(),
                        x[p.tri_v[2]], Vec3::Zero(),
                        /*eps=*/1.0e-12, params.use_ticcd);
                    if (r.collision) safe_min = std::min(safe_min, r.t);
                }
            } else {
                // vi is a moving tri corner vs a static node.
                if (tr) {
                    auto r = trust_region_vertex_triangle_gauss_seidel(
                        x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], dx);
                    safe_min = std::min(safe_min, r.omega);
                } else {
                    Vec3 dxv[3] = {Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};
                    dxv[entry.dof - 1] = dx;
                    CCDResult r = node_triangle_only_one_node_moves(
                        x[p.node],     Vec3::Zero(),
                        x[p.tri_v[0]], dxv[0],
                        x[p.tri_v[1]], dxv[1],
                        x[p.tri_v[2]], dxv[2],
                        /*eps=*/1.0e-12, params.use_ticcd);
                    if (r.collision) safe_min = std::min(safe_min, r.t);
                }
            }
        }
    }

    if (vi >= 0 && vi < static_cast<int>(bp_cache.vertex_ss.size())) {
        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            // dof: 0=v[0], 1=v[1], 2=v[2], 3=v[3]. vi is on the first edge iff dof in {0,1}.
            if (tr) {
                auto r = trust_region_edge_edge_gauss_seidel(
                    x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], dx);
                safe_min = std::min(safe_min, r.omega);
            } else {
                CCDResult r;
                const double eps_ccd = 1.0e-12;
                if (entry.dof == 0)
                    r = segment_segment_only_one_node_moves(x[p.v[0]], dx, x[p.v[1]], x[p.v[2]], x[p.v[3]], eps_ccd, params.use_ticcd);
                else if (entry.dof == 1)
                    r = segment_segment_only_one_node_moves(x[p.v[1]], dx, x[p.v[0]], x[p.v[2]], x[p.v[3]], eps_ccd, params.use_ticcd);
                else if (entry.dof == 2)
                    r = segment_segment_only_one_node_moves(x[p.v[2]], dx, x[p.v[3]], x[p.v[0]], x[p.v[1]], eps_ccd, params.use_ticcd);
                else
                    r = segment_segment_only_one_node_moves(x[p.v[3]], dx, x[p.v[2]], x[p.v[0]], x[p.v[1]], eps_ccd, params.use_ticcd);
                if (r.collision) safe_min = std::min(safe_min, r.t);
            }
        }
    }

    return tr ? safe_min : ((safe_min >= 1.0) ? 1.0 : 0.9 * safe_min);
}
