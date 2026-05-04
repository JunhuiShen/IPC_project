#include "solver.h"
#include "IPC_math.h"
#include "ccd.h"
#include "make_shape.h"
#include "parallel_helper.h"
#include "ogc_trust_region.h"
#include "node_triangle_distance.h"
#include "segment_segment_distance.h"
#include "barrier_energy.h"

#include <algorithm>
#include <limits>
#include <cstdio>
#include <string>
#include <fstream>
#include <filesystem>
#include "visualization.h"


// CCD initial guess
std::vector<Vec3> ccd_initial_guess(const std::vector<Vec3>& x, const std::vector<Vec3>& xhat, const RefMesh& ref_mesh) {
    const int nv = static_cast<int>(x.size());

    std::vector<Vec3> dx(nv);
    for (int i = 0; i < nv; ++i) dx[i] = xhat[i] - x[i];

    BroadPhase ccd_bp;
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

Vec3 gs_vertex_delta(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                       const std::vector<Vec3>& xhat, std::vector<Vec3>& x, const BroadPhase& broad_phase, const PinMap* pin_map) {
    const auto& bp_cache = broad_phase.cache();
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x, xhat, pin_map);

    if (params.d_hat > 0.0) {
        const double dt2k = params.dt2() * params.k_barrier;

        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, entry.dof);
            g += dt2k * bg;
            H += dt2k * bH;
        }

        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, entry.dof);
            g += dt2k * bg;
            H += dt2k * bH;
        }
    }

    return matrix3d_inverse(H) * g;
}

void update_one_vertex(int vi, const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                       const std::vector<Vec3>& xhat, std::vector<Vec3>& x, const BroadPhase& broad_phase, const PinMap* pin_map) {
    const auto& bp_cache = broad_phase.cache();
    auto [g, H] = compute_local_gradient_and_hessian_no_barrier(vi, ref_mesh, adj, pins, params, x, xhat, pin_map);

    if (params.d_hat > 0.0) {
        const double dt2k = params.dt2() * params.k_barrier;

        for (const auto& entry : bp_cache.vertex_nt[vi]) {
            const auto& p = bp_cache.nt_pairs[entry.pair_index];
            auto [bg, bH] = node_triangle_barrier_gradient_and_hessian(x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], params.d_hat, entry.dof);
            g += dt2k * bg;
            H += dt2k * bH;
        }

        for (const auto& entry : bp_cache.vertex_ss[vi]) {
            const auto& p = bp_cache.ss_pairs[entry.pair_index];
            auto [bg, bH] = segment_segment_barrier_gradient_and_hessian(x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], params.d_hat, entry.dof);
            g += dt2k * bg;
            H += dt2k * bH;
        }
    }

    const Vec3 delta = matrix3d_inverse(H) * g;

    double step = 1.0;
    if (params.d_hat > 0.0) {
        const Vec3 dx = -delta;
        const bool tr = params.use_ogc;

        const auto pairs = broad_phase.query_pairs_for_vertex(x, vi, dx, ref_mesh);

        double safe_min = 1.0;

        // vi as the lone moving node
        for (const auto& p : pairs.nt_node_pairs) {
            if (tr) {
                auto r = trust_region_vertex_triangle_gauss_seidel(
                    x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], dx);
                safe_min = std::min(safe_min, r.omega);
            } else {
                auto r = node_triangle_only_one_node_moves(
                    x[p.node],     dx,
                    x[p.tri_v[0]], Vec3::Zero(),
                    x[p.tri_v[1]], Vec3::Zero(),
                    x[p.tri_v[2]], Vec3::Zero());
                if (r.collision) safe_min = std::min(safe_min, r.t);
            }
        }

        // vi as one moving triangle vertex
        for (const auto& p : pairs.nt_face_pairs) {
            if (tr) {
                auto r = trust_region_vertex_triangle_gauss_seidel(
                    x[p.node], x[p.tri_v[0]], x[p.tri_v[1]], x[p.tri_v[2]], dx);
                safe_min = std::min(safe_min, r.omega);
            } else {
                Vec3 dxv[3] = {Vec3::Zero(), Vec3::Zero(), Vec3::Zero()};
                dxv[p.vi_local] = dx;
                auto r = node_triangle_only_one_node_moves(
                    x[p.node],     Vec3::Zero(),
                    x[p.tri_v[0]], dxv[0],
                    x[p.tri_v[1]], dxv[1],
                    x[p.tri_v[2]], dxv[2]);
                if (r.collision) safe_min = std::min(safe_min, r.t);
            }
        }

        for (const auto& p : pairs.ss_pairs) {
            if (tr) {
                auto r = trust_region_edge_edge_gauss_seidel(
                    x[p.v[0]], x[p.v[1]], x[p.v[2]], x[p.v[3]], dx);
                safe_min = std::min(safe_min, r.omega);
            } else {
                CCDResult r;
                if (p.vi_dof == 0)
                    r = segment_segment_only_one_node_moves(x[p.v[0]], dx, x[p.v[1]], x[p.v[2]], x[p.v[3]]);
                else
                    r = segment_segment_only_one_node_moves(x[p.v[1]], dx, x[p.v[0]], x[p.v[2]], x[p.v[3]]);
                if (r.collision) safe_min = std::min(safe_min, r.t);
            }
        }

        step = tr ? safe_min : ((safe_min >= 1.0) ? 1.0 : 0.9 * safe_min);
    }

    x[vi] -= step * delta;
}

static void export_nt_pairs_geo(const std::string& filename, const std::vector<Vec3>& x,
                                const std::vector<NodeTrianglePair>& nt_pairs) {
    std::ofstream out(filename);
    if (!out) return;

    const int nv    = static_cast<int>(x.size());
    const int nnt   = static_cast<int>(nt_pairs.size());
    const int nverts = nnt * 3;

    std::vector<int> nt_count(nv, 0);
    for (const auto& p : nt_pairs) {
        nt_count[p.node]++;
        nt_count[p.tri_v[0]]++; nt_count[p.tri_v[1]]++; nt_count[p.tri_v[2]]++;
    }

    out << std::setprecision(10);
    out << "[\n";
    out << "    \"fileversion\", \"18.5.408\",\n";
    out << "    \"hasindex\", false,\n";
    out << "    \"pointcount\", " << nv << ",\n";
    out << "    \"vertexcount\", " << nverts << ",\n";
    out << "    \"primitivecount\", " << nnt << ",\n";
    out << "    \"info\", {},\n";

    out << "    \"topology\",\n    [\n";
    out << "        \"pointref\",\n        [\n";
    out << "            \"indices\", [";
    for (int i = 0; i < nnt; ++i) {
        const auto& p = nt_pairs[i];
        if (i > 0) out << ",";
        out << p.tri_v[0] << "," << p.tri_v[1] << "," << p.tri_v[2];
    }
    out << "]\n        ]\n    ],\n";

    out << "    \"attributes\",\n    [\n";
    out << "        \"pointattributes\",\n        [\n";
    out << "            [\n";
    out << "                [\"scope\",\"public\",\"type\",\"numeric\",\"name\",\"P\",\"options\",{}],\n";
    out << "                [\"size\",3,\"storage\",\"fpreal32\",\"values\",\n";
    out << "                    [\"size\",3,\"storage\",\"fpreal32\",\"tuples\",[";
    for (int i = 0; i < nv; ++i) {
        if (i > 0) out << ",";
        out << "[" << x[i].x() << "," << x[i].y() << "," << x[i].z() << "]";
    }
    out << "]]]\n            ],\n";
    out << "            [\n";
    out << "                [\"scope\",\"public\",\"type\",\"numeric\",\"name\",\"nt_pair_count\",\"options\",{}],\n";
    out << "                [\"size\",1,\"storage\",\"int32\",\"values\",\n";
    out << "                    [\"size\",1,\"storage\",\"int32\",\"tuples\",[";
    for (int i = 0; i < nv; ++i) {
        if (i > 0) out << ",";
        out << "[" << nt_count[i] << "]";
    }
    out << "]]]\n            ]\n";
    out << "        ],\n";

    out << "        \"primitiveattributes\",\n        [\n";
    out << "            [\n";
    out << "                [\"scope\",\"public\",\"type\",\"numeric\",\"name\",\"pair_idx\",\"options\",{}],\n";
    out << "                [\"size\",1,\"storage\",\"int32\",\"values\",\n";
    out << "                    [\"size\",1,\"storage\",\"int32\",\"tuples\",[";
    for (int i = 0; i < nnt; ++i) {
        if (i > 0) out << ",";
        out << "[" << i << "]";
    }
    out << "]]]\n            ]\n";
    out << "        ]\n";
    out << "    ],\n";

    out << "    \"primitives\",\n    [\n";
    out << "        [\n";
    out << "            [\"type\",\"Polygon_run\"],\n";
    out << "            [\n";
    out << "                \"startvertex\", 0,\n";
    out << "                \"nprimitives\", " << nnt << ",\n";
    out << "                \"nvertices_rle\", [3," << nnt << "]\n";
    out << "            ]\n        ]\n    ]\n]\n";
}

static void export_ss_pairs_geo(const std::string& filename, const std::vector<Vec3>& x,
                                const std::vector<SegmentSegmentPair>& ss_pairs) {
    std::ofstream out(filename);
    if (!out) return;

    const int nv   = static_cast<int>(x.size());
    const int nss  = static_cast<int>(ss_pairs.size());
    const int nprims = nss * 2;      // 2 edges per pair
    const int nverts = nprims * 2;   // 2 vertices per edge

    // per-point SS pair count
    std::vector<int> ss_count(nv, 0);
    for (const auto& p : ss_pairs) {
        ss_count[p.v[0]]++; ss_count[p.v[1]]++;
        ss_count[p.v[2]]++; ss_count[p.v[3]]++;
    }

    out << std::setprecision(10);
    out << "[\n";
    out << "    \"fileversion\", \"18.5.408\",\n";
    out << "    \"hasindex\", false,\n";
    out << "    \"pointcount\", " << nv << ",\n";
    out << "    \"vertexcount\", " << nverts << ",\n";
    out << "    \"primitivecount\", " << nprims << ",\n";
    out << "    \"info\", {},\n";

    // topology: for each edge primitive, list its 2 point indices
    out << "    \"topology\",\n    [\n";
    out << "        \"pointref\",\n        [\n";
    out << "            \"indices\", [";
    for (int i = 0; i < nss; ++i) {
        const auto& p = ss_pairs[i];
        if (i > 0) out << ",";
        out << p.v[0] << "," << p.v[1] << "," << p.v[2] << "," << p.v[3];
    }
    out << "]\n        ]\n    ],\n";

    // attributes
    out << "    \"attributes\",\n    [\n";

    // point attributes: P and ss_pair_count
    out << "        \"pointattributes\",\n        [\n";
    out << "            [\n";
    out << "                [\"scope\",\"public\",\"type\",\"numeric\",\"name\",\"P\",\"options\",{}],\n";
    out << "                [\"size\",3,\"storage\",\"fpreal32\",\"values\",\n";
    out << "                    [\"size\",3,\"storage\",\"fpreal32\",\"tuples\",[";
    for (int i = 0; i < nv; ++i) {
        if (i > 0) out << ",";
        out << "[" << x[i].x() << "," << x[i].y() << "," << x[i].z() << "]";
    }
    out << "]]]\n            ],\n";
    out << "            [\n";
    out << "                [\"scope\",\"public\",\"type\",\"numeric\",\"name\",\"ss_pair_count\",\"options\",{}],\n";
    out << "                [\"size\",1,\"storage\",\"int32\",\"values\",\n";
    out << "                    [\"size\",1,\"storage\",\"int32\",\"tuples\",[";
    for (int i = 0; i < nv; ++i) {
        if (i > 0) out << ",";
        out << "[" << ss_count[i] << "]";
    }
    out << "]]]\n            ]\n";
    out << "        ],\n";

    // primitive attributes: pair_idx
    out << "        \"primitiveattributes\",\n        [\n";
    out << "            [\n";
    out << "                [\"scope\",\"public\",\"type\",\"numeric\",\"name\",\"pair_idx\",\"options\",{}],\n";
    out << "                [\"size\",1,\"storage\",\"int32\",\"values\",\n";
    out << "                    [\"size\",1,\"storage\",\"int32\",\"tuples\",[";
    for (int i = 0; i < nss; ++i) {
        if (i > 0) out << ",";
        out << "[" << i << "],[" << i << "]";  // two edges per pair share the same pair_idx
    }
    out << "]]]\n            ]\n";
    out << "        ]\n";
    out << "    ],\n";

    // primitives: Polygon_run of 2-vertex open polygons (line segments)
    out << "    \"primitives\",\n    [\n";
    out << "        [\n";
    out << "            [\"type\",\"Polygon_run\"],\n";
    out << "            [\n";
    out << "                \"startvertex\", 0,\n";
    out << "                \"nprimitives\", " << nprims << ",\n";
    out << "                \"nvertices_rle\", [2," << nprims << "]\n";
    out << "            ]\n";
    out << "        ]\n";
    out << "    ]\n";
    out << "]\n";
}

static void write_substep_data(const SimParams& params, const BroadPhase& broad_phase,
                        const std::vector<Vec3>& xnew, const std::string& outdir,
                        const RefMesh* ref_mesh = nullptr,
                        const std::vector<std::vector<int>>* color_groups = nullptr) {
    static int substep_counter = 0;
    const int step = substep_counter++;
    const std::string prefix = (outdir.empty() ? "" : outdir + "/");
    const std::string subdir = prefix + "substep_" + std::to_string(step);
    std::filesystem::create_directories(subdir);
    const auto& bpc = broad_phase.cache();

    // --- barrier distances file ---
    const std::string dist_path = subdir + "/barrier_distances.txt";
    if (FILE* dist_file = fopen(dist_path.c_str(), "w")) {
        fprintf(dist_file, "# substep %d  d_hat=%.6e\n# type node/v0 v1 v2 v3 distance force_norm_sum\n", step, params.d_hat);
        for (const auto& p : bpc.nt_pairs) {
            const auto dr = node_triangle_distance(xnew[p.node], xnew[p.tri_v[0]], xnew[p.tri_v[1]], xnew[p.tri_v[2]]);
            double fsum = 0.0;
            for (int dof = 0; dof < 4; ++dof)
                fsum += node_triangle_barrier_gradient(xnew[p.node], xnew[p.tri_v[0]], xnew[p.tri_v[1]], xnew[p.tri_v[2]], params.d_hat, dof, 1e-12, &dr).norm();
            fprintf(dist_file, "NT %d %d %d %d %.10e %.10e\n", p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2], dr.distance, fsum);
        }
        for (const auto& p : bpc.ss_pairs) {
            const auto dr = segment_segment_distance(xnew[p.v[0]], xnew[p.v[1]], xnew[p.v[2]], xnew[p.v[3]]);
            double fsum = 0.0;
            for (int dof = 0; dof < 4; ++dof)
                fsum += segment_segment_barrier_gradient(xnew[p.v[0]], xnew[p.v[1]], xnew[p.v[2]], xnew[p.v[3]], params.d_hat, dof, 1e-12, &dr).norm();
            fprintf(dist_file, "SS %d %d %d %d %.10e %.10e\n", p.v[0], p.v[1], p.v[2], p.v[3], dr.distance, fsum);
        }
        fclose(dist_file);
    }

    // --- barrier stats file ---
    const int nv = static_cast<int>(xnew.size());
    std::vector<int> vertex_pair_count(nv, 0);
    for (const auto& p : bpc.nt_pairs) {
        vertex_pair_count[p.node]++;
        vertex_pair_count[p.tri_v[0]]++;
        vertex_pair_count[p.tri_v[1]]++;
        vertex_pair_count[p.tri_v[2]]++;
    }
    for (const auto& p : bpc.ss_pairs) {
        vertex_pair_count[p.v[0]]++;
        vertex_pair_count[p.v[1]]++;
        vertex_pair_count[p.v[2]]++;
        vertex_pair_count[p.v[3]]++;
    }
    const int total_pairs = static_cast<int>(bpc.nt_pairs.size() + bpc.ss_pairs.size());

    std::vector<int> sorted_counts = vertex_pair_count;
    std::sort(sorted_counts.begin(), sorted_counts.end());
    const int vmin = sorted_counts.front();
    const int vmax = sorted_counts.back();
    double vmedian;
    if (nv % 2 == 0)
        vmedian = 0.5 * (sorted_counts[nv / 2 - 1] + sorted_counts[nv / 2]);
    else
        vmedian = sorted_counts[nv / 2];

    const std::string stats_path = subdir + "/barrier_stats.txt";
    if (FILE* stats_file = fopen(stats_path.c_str(), "w")) {
        fprintf(stats_file, "substep %d\n", step);
        fprintf(stats_file, "total_pairs %d\n", total_pairs);
        fprintf(stats_file, "nt_pairs %d\n", static_cast<int>(bpc.nt_pairs.size()));
        fprintf(stats_file, "ss_pairs %d\n", static_cast<int>(bpc.ss_pairs.size()));
        fprintf(stats_file, "vertex_pair_count_min %d\n", vmin);
        fprintf(stats_file, "vertex_pair_count_max %d\n", vmax);
        fprintf(stats_file, "vertex_pair_count_median %.1f\n", vmedian);
        fclose(stats_file);
    }

    // --- colored mesh ---
    if (ref_mesh) {
        export_geo(subdir + "/mesh.geo", xnew, ref_mesh->tris, color_groups);
    }

    // --- NT pairs (combined) ---
    export_nt_pairs_geo(subdir + "/nt_pairs.geo", xnew, bpc.nt_pairs);

    // --- NT pairs (one file each) ---
    const std::string nt_dir = subdir + "/nt_pairs";
    std::filesystem::create_directories(nt_dir);
    for (int i = 0; i < static_cast<int>(bpc.nt_pairs.size()); ++i) {
        const auto& p = bpc.nt_pairs[i];
        const std::string path = nt_dir + "/nt_pair_" + std::to_string(i) + ".geo";
        std::ofstream out(path);
        if (!out) continue;
        out << std::setprecision(10);
        out << "[\n";
        out << "    \"fileversion\", \"18.5.408\",\n";
        out << "    \"hasindex\", false,\n";
        out << "    \"pointcount\", 4,\n";
        out << "    \"vertexcount\", 3,\n";
        out << "    \"primitivecount\", 1,\n";
        out << "    \"info\", {},\n";
        out << "    \"topology\",\n    [\n";
        out << "        \"pointref\",\n        [\n";
        out << "            \"indices\", [1,2,3]\n";
        out << "        ]\n    ],\n";
        out << "    \"attributes\",\n    [\n";
        out << "        \"pointattributes\",\n        [\n";
        out << "            [\n";
        out << "                [\"scope\",\"public\",\"type\",\"numeric\",\"name\",\"P\",\"options\",{}],\n";
        out << "                [\"size\",3,\"storage\",\"fpreal32\",\"values\",\n";
        out << "                    [\"size\",3,\"storage\",\"fpreal32\",\"tuples\",[";
        // point 0 = node, points 1-3 = tri verts
        const int pts[4] = {p.node, p.tri_v[0], p.tri_v[1], p.tri_v[2]};
        for (int k = 0; k < 4; ++k) {
            if (k > 0) out << ",";
            const Vec3& pt = xnew[pts[k]];
            out << "[" << pt.x() << "," << pt.y() << "," << pt.z() << "]";
        }
        out << "]]]\n            ]\n        ]\n    ],\n";
        out << "    \"primitives\",\n    [\n";
        out << "        [\n";
        out << "            [\"type\",\"Polygon_run\"],\n";
        out << "            [\n";
        out << "                \"startvertex\", 0,\n";
        out << "                \"nprimitives\", 1,\n";
        out << "                \"nvertices_rle\", [3,1]\n";
        out << "            ]\n        ]\n    ]\n]\n";
    }

    // --- SS pairs (combined) ---
    export_ss_pairs_geo(subdir + "/ss_pairs.geo", xnew, bpc.ss_pairs);

    // --- SS pairs (one file each) ---
    const std::string ss_dir = subdir + "/ss_pairs";
    std::filesystem::create_directories(ss_dir);
    for (int i = 0; i < static_cast<int>(bpc.ss_pairs.size()); ++i) {
        const auto& p = bpc.ss_pairs[i];
        const std::string path = ss_dir + "/ss_pair_" + std::to_string(i) + ".geo";
        std::ofstream out(path);
        if (!out) continue;
        out << std::setprecision(10);
        out << "[\n";
        out << "    \"fileversion\", \"18.5.408\",\n";
        out << "    \"hasindex\", false,\n";
        out << "    \"pointcount\", 4,\n";
        out << "    \"vertexcount\", 4,\n";
        out << "    \"primitivecount\", 2,\n";
        out << "    \"info\", {},\n";
        out << "    \"topology\",\n    [\n";
        out << "        \"pointref\",\n        [\n";
        out << "            \"indices\", [0,1,2,3]\n";
        out << "        ]\n    ],\n";
        out << "    \"attributes\",\n    [\n";
        out << "        \"pointattributes\",\n        [\n";
        out << "            [\n";
        out << "                [\"scope\",\"public\",\"type\",\"numeric\",\"name\",\"P\",\"options\",{}],\n";
        out << "                [\"size\",3,\"storage\",\"fpreal32\",\"values\",\n";
        out << "                    [\"size\",3,\"storage\",\"fpreal32\",\"tuples\",[";
        for (int k = 0; k < 4; ++k) {
            if (k > 0) out << ",";
            const Vec3& pt = xnew[p.v[k]];
            out << "[" << pt.x() << "," << pt.y() << "," << pt.z() << "]";
        }
        out << "]]]\n            ]\n        ]\n    ],\n";
        out << "    \"primitives\",\n    [\n";
        out << "        [\n";
        out << "            [\"type\",\"Polygon_run\"],\n";
        out << "            [\n";
        out << "                \"startvertex\", 0,\n";
        out << "                \"nprimitives\", 2,\n";
        out << "                \"nvertices_rle\", [2,2]\n";
        out << "            ]\n        ]\n    ]\n]\n";
    }
}

SolverResult global_gauss_seidel_solver_basic(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                        std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                        const std::vector<Vec3>& v,
                                        std::vector<double>* residual_history, const std::string& outdir) {

    if (!params.fixed_iters) {
        fprintf(stderr, "global_gauss_seidel_solver_basic: params.fixed_iters must be true\n");
        exit(1);
    }

    //create node (blue) boxes and create broad phase (red boxes) accordingly
    const int nv = static_cast<int>(xnew.size());
    const PinMap pm = build_pin_map(pins, nv);
    static std::vector<double> prev_disp;
    if (static_cast<int>(prev_disp.size()) != nv)
        prev_disp.assign(nv, params.node_box_max);
    constexpr double node_box_padding = 1.2;
    auto node_box_size_fn = [&](int vi) { return std::clamp(prev_disp[vi] * node_box_padding, params.node_box_min, params.node_box_max); };
    std::vector<AABB> blue_boxes(nv);
    for (int i = 0; i < nv; ++i) {
        const double r = node_box_size_fn(i);
        blue_boxes[i] = AABB(xnew[i] - Vec3::Constant(r), xnew[i] + Vec3::Constant(r));
    }
    BroadPhase broad_phase;
    broad_phase.initialize(blue_boxes, ref_mesh, params.d_hat);

    //create colors
    const std::vector<std::vector<int>> color_groups = greedy_color_conflict_graph(
        union_adjacency(build_elastic_adj(ref_mesh, adj, static_cast<int>(xnew.size())),
                        build_contact_adj(broad_phase.cache(), static_cast<int>(xnew.size()))));

    //residual tracking: not going to actually do this and will demand running with fixed iterations
    if (residual_history) residual_history->clear();
    SolverResult result;
    result.initial_residual = 0.0;
    result.final_residual   = result.initial_residual;
    result.iterations       = 0;
    if (residual_history) {
        const int reserve_n = std::max(0, params.max_global_iters);
        residual_history->reserve(static_cast<std::size_t>(reserve_n) + 1);
        residual_history->push_back(result.initial_residual);
    }

    //write substep data
    if (params.write_substeps) {
        write_substep_data(params, broad_phase, xnew, outdir, &ref_mesh, &color_groups);
    }

    const std::vector<Vec3> xnew_substep_start = xnew;

    //gs loop
    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        broad_phase.per_vertex_safe_step(xnew, [&](int vi){ return xnew[vi] - gs_vertex_delta(vi, ref_mesh, adj, pins, params, xhat, xnew, broad_phase, &pm); },
                                         /*safety=*/0.9, /*clip_to_node_box=*/true,
                                         /*clip_ccd=*/params.use_ogc ? false : params.use_ccd,
                                         /*use_ticcd=*/params.use_ticcd,
                                         /*use_ogc=*/params.use_ogc,
                                         params.use_parallel ? &color_groups : nullptr);
        result.final_residual = 0.0;
        result.iterations     = iter;
        if (residual_history) residual_history->push_back(result.final_residual);
    }

    for (int i = 0; i < nv; ++i)
        prev_disp[i] = (xnew[i] - xnew_substep_start[i]).norm();

    if (params.fixed_iters) result.converged = true;
    return result;
}

SolverResult global_gauss_seidel_solver_ogc(const RefMesh& ref_mesh, const VertexTriangleMap& adj, const std::vector<Pin>& pins, const SimParams& params,
                                            std::vector<Vec3>& xnew, const std::vector<Vec3>& xhat,
                                            const std::vector<Vec3>& /*v*/,
                                            std::vector<double>* residual_history, const std::string& outdir) {
    if (!params.fixed_iters) {
        fprintf(stderr, "global_gauss_seidel_solver_ogc: params.fixed_iters must be true\n");
        exit(1);
    }

    const int nv = static_cast<int>(xnew.size());
    const PinMap pm = build_pin_map(pins, nv);

    static std::vector<double> prev_disp;
    if (static_cast<int>(prev_disp.size()) != nv)
        prev_disp.assign(nv, params.node_box_max);
    constexpr double node_box_padding = 1.2;
    auto node_box_size_fn = [&](int vi) { return std::clamp(prev_disp[vi] * node_box_padding, params.node_box_min, params.node_box_max); };

    if (residual_history) residual_history->clear();
    SolverResult result;
    result.initial_residual = 0.0;
    result.final_residual   = 0.0;
    result.iterations       = 0;
    if (residual_history) {
        residual_history->reserve(static_cast<std::size_t>(std::max(0, params.max_global_iters)) + 1);
        residual_history->push_back(0.0);
    }

    BroadPhase broad_phase;
    const std::vector<Vec3> xnew_substep_start = xnew;  // anchor for clip boxes and prev_disp
    const double pad = std::max(params.ogc_box_pad, params.d_hat);

    std::vector<AABB> bvh_node_boxes(nv);
    for (int i = 0; i < nv; ++i) {
        const double r = node_box_size_fn(i) + pad;
        bvh_node_boxes[i] = AABB(xnew[i] - Vec3::Constant(r), xnew[i] + Vec3::Constant(r));
    }
    broad_phase.initialize(bvh_node_boxes, ref_mesh, pad);

    if (params.write_substeps)
        write_substep_data(params, broad_phase, xnew, outdir, &ref_mesh, nullptr);

    auto& bp_cache = broad_phase.mutable_cache();

    for (int iter = 1; iter <= params.max_global_iters; ++iter) {
        for (int vi = 0; vi < nv; ++vi) {
            const Vec3 dx_full = -gs_vertex_delta(vi, ref_mesh, adj, pins, params, xhat, xnew, broad_phase, &pm);
            if (dx_full.squaredNorm() < 1e-28) continue;

            // Clipping
            const double R_vi = node_box_size_fn(vi);
            constexpr double inset = 1e-10;
            const Vec3 clip_min = xnew_substep_start[vi] - Vec3::Constant(R_vi);
            const Vec3 clip_max = xnew_substep_start[vi] + Vec3::Constant(R_vi);
            const Vec3 x_target = (xnew[vi] + dx_full).cwiseMax(clip_min + Vec3::Constant(inset)).cwiseMin(clip_max - Vec3::Constant(inset));
            const Vec3 dx = x_target - xnew[vi];
            if (dx.squaredNorm() < 1e-28) continue;

            // No-pair fallback = half min-extent of the cubic clip box = R_vi.
            double bound = compute_trust_region_bound_for_vertex(vi, xnew, broad_phase.cache(), 0.4);
            if (!std::isfinite(bound)) bound = R_vi;

            const double dx_norm = dx.norm();
            const double toi = (dx_norm > 0.0) ? std::min(1.0, bound / dx_norm) : 1.0;
            xnew[vi] += toi * dx;

            incremental_refresh_vertex(bp_cache, vi, xnew, ref_mesh, pad, R_vi + pad);
        }

        result.iterations = iter;
        if (residual_history) residual_history->push_back(0.0);
    }

    for (int i = 0; i < nv; ++i)
        prev_disp[i] = (xnew[i] - xnew_substep_start[i]).norm();

    result.converged = true;
    return result;
}
