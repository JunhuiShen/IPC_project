#include "initial_guess.h"

#include "broad_phase.h"
#include "ccd.h"

#include <algorithm>
#include <limits>

std::vector<Vec3> ccd_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, BroadPhase* scratch_broad_phase) {
    const int nv = static_cast<int>(x.size());

    std::vector<Vec3> dx(nv);
    for (int i = 0; i < nv; ++i) dx[i] = xhat[i] - x[i];

    BroadPhase local_bp;
    BroadPhase& ccd_bp = scratch_broad_phase ? *scratch_broad_phase : local_bp;
    ccd_bp.build_ccd_candidates(x, dx, ref_mesh, 1.0);
    const auto& cache = ccd_bp.cache();

    double toi_min = 1.0;

    const int n_nt = static_cast<int>(cache.nt_pairs.size());
    #pragma omp parallel for reduction(min:toi_min) schedule(static)
    for (int i = 0; i < n_nt; ++i) {
        const auto& p = cache.nt_pairs[i];
        toi_min = std::min(toi_min, node_triangle_general_ccd(
            x[p.node],     dx[p.node],
            x[p.tri_v[0]], dx[p.tri_v[0]],
            x[p.tri_v[1]], dx[p.tri_v[1]],
            x[p.tri_v[2]], dx[p.tri_v[2]]));
    }

    const int n_ss = static_cast<int>(cache.ss_pairs.size());
    #pragma omp parallel for reduction(min:toi_min) schedule(static)
    for (int i = 0; i < n_ss; ++i) {
        const auto& p = cache.ss_pairs[i];
        toi_min = std::min(toi_min, segment_segment_general_ccd(
            x[p.v[0]], dx[p.v[0]],
            x[p.v[1]], dx[p.v[1]],
            x[p.v[2]], dx[p.v[2]],
            x[p.v[3]], dx[p.v[3]]));
    }

    const double omega = (toi_min >= 1.0) ? 1.0 : 0.9 * toi_min;

    std::vector<Vec3> xnew(nv);
    for (int i = 0; i < nv; ++i) xnew[i] = x[i] + omega * dx[i];

    return xnew;
}

std::vector<Vec3> verlet_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, const SimParams& params, BroadPhase* scratch_broad_phase) {
    const Vec3 dt2g = params.dt2() * params.gravity;
    std::vector<Vec3> xverlet(xhat.size());
    for (int i = 0; i < static_cast<int>(xhat.size()); ++i) xverlet[i] = xhat[i] + dt2g;
    return ccd_initial_guess(x, xverlet, ref_mesh, scratch_broad_phase);
}

namespace {

bool translation_guess_sdf_min_evaluation(const SimParams& params, const Vec3& xi, SDFEvaluation& out) {
    bool any = false;
    out.phi = std::numeric_limits<double>::infinity();
    for (const PlaneSDF& p : params.sdf_planes) {
        const SDFEvaluation s = evaluate_sdf(p, xi);
        if (!any || s.phi < out.phi) { out = s; any = true; }
    }
    for (const CylinderSDF& c : params.sdf_cylinders) {
        const SDFEvaluation s = evaluate_sdf(c, xi);
        if (!any || s.phi < out.phi) { out = s; any = true; }
    }
    for (const SphereSDF& sp : params.sdf_spheres) {
        const SDFEvaluation s = evaluate_sdf(sp, xi);
        if (!any || s.phi < out.phi) { out = s; any = true; }
    }
    return any;
}

}  // namespace

std::vector<Vec3> translation_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh, const std::vector<Pin>& pins, const SimParams& params) {
    std::vector<Vec3> xnew(xhat.size());
    const double dt2 = params.dt2();
    double total_mass = 0.0;
    for (double m: ref_mesh.mass) total_mass += m;

    Vec3 rhs = Vec3::Zero();
    for(int i = 0; i < (int)xhat.size(); ++i){
        rhs += ref_mesh.mass[i] * (xhat[i] - x[i]);
    }
    rhs += dt2 * total_mass * params.gravity;

    double denom = total_mass;
    if (params.kpin > 0.0) {
        for (const Pin& pin : pins) {
            rhs += dt2 * params.kpin * (pin.target_position - x[pin.vertex_index]);
            denom += dt2 * params.kpin;
        }
    }

    Vec3 C = Vec3::Zero();
    if (denom > 0.0) C = rhs / denom;

    if (params.k_sdf > 0.0) {
        Vec3 G = Vec3::Zero();
        Mat33 H = denom * Mat33::Identity();

        for(int i = 0; i < (int)xhat.size(); ++i){
            SDFEvaluation s;
            if (translation_guess_sdf_min_evaluation(params, x[i] + C, s)) {
                G += dt2 * sdf_penalty_gradient(s, params.k_sdf, params.eps_sdf);
                H += dt2 * sdf_penalty_hessian(s, params.k_sdf, params.eps_sdf, false);
            }
        }

        C -= H.ldlt().solve(G);
    }

    for(int i = 0; i < (int)xhat.size(); ++i){
        xnew[i] = x[i] + C;
    }
    return xnew;
}
