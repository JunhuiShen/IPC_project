#include "jacobi_step.h"
#include "broad_phase.h"
#include "friction_energy.h"
#include "mesh_utils.h"
#include "solid_ipc.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

struct RestoreOpenMP {
    const int threads = omp_get_max_threads();
    const int dynamic = omp_get_dynamic();
    ~RestoreOpenMP() { omp_set_num_threads(threads); omp_set_dynamic(dynamic); }
};

struct Scene {
    RefMesh mesh;
    VertexTriangleMap adj;
    std::vector<Pin> pins;
    SimParams params = SimParams::zeros();
    std::vector<Vec3> x, xhat, velocity;

    Scene(std::vector<Vec3> positions, std::vector<int> triangles = {}) : x(std::move(positions)), xhat(x), velocity(x.size(), Vec3::Zero()) {
        mesh.num_positions = x.size();
        mesh.mass.assign(x.size(), 1.0);
        mesh.node_to_rb.assign(x.size(), -1);
        mesh.tris = std::move(triangles);
        params.fps = 100;
        params.substeps = 1;
        params.damping = 1;
        params.use_parallel = false;
    }
    std::vector<Vec3> solve(const std::vector<Vec3>* previous = nullptr) const {
        return jacobi_step(mesh, adj, pins, params, x, xhat, velocity, previous);
    }
};

Scene cloth() {
    Scene s({Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), Vec3(1,1,0)}, {0,1,2,1,3,2});
    s.mesh.initialize(s.x);
    s.adj = build_incident_triangle_map(s.mesh.tris);
    s.params.fps = 20;
    s.params.mu = 30;
    s.params.lambda = 20;
    s.params.kB = .003;
    s.params.damping = .7;
    s.x[0] += Vec3(-.1,.03,.02);
    s.x[3] += Vec3(.06,-.02,.04);
    return s;
}

Scene point_triangle() {
    Scene s({Vec3(0,0,.1), Vec3(-1,-1,0), Vec3(2,-1,0), Vec3(-1,2,0)}, {1,2,3});
    s.params.d_hat = .2;
    s.params.k_barrier = 100;
    return s;
}

Scene edges() {
    Scene s({Vec3(-1,0,0), Vec3(1,0,0), Vec3(-1,-1,0), Vec3(0,-1,.1), Vec3(0,1,.1), Vec3(1,1,.1)}, {0,1,2,3,4,5});
    s.params.d_hat = .2;
    s.params.k_barrier = 100;
    return s;
}

void expect_same(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(std::memcmp(a[i].data(), b[i].data(), 3*sizeof(double)), 0) << i;
}

// Independent scalar reference: assemble each diagonal system directly from
// the public energy routines and solve it with Eigen's rank-revealing solver.
std::vector<Vec3> reference(const Scene& s, const std::vector<Vec3>* previous = nullptr) {
    BroadPhase phase;
    std::vector<AABB> boxes;
    for (const auto& x : s.x) boxes.emplace_back(x,x);
    phase.initialize(boxes, s.mesh, s.params.d_hat);
    const auto& cache = phase.cache();
    const auto pin_map = build_pin_map(s.pins, static_cast<int>(s.x.size()));
    std::vector<Vec3> result(s.x.size());
    for (std::size_t node = 0; node < s.x.size(); ++node) {
        const auto it = s.adj.find(static_cast<int>(node));
        const IncidentTriangles empty;
        const auto* incident = it == s.adj.end() ? &empty : &it->second;
        auto local = compute_local_gradient_and_hessian_no_barrier(static_cast<int>(node), s.mesh, s.adj, s.pins, s.params, s.x, s.xhat, &pin_map, incident, nullptr, previous);
        Vec3& g = local.first;
        Mat33& H = local.second;
        if (s.params.d_hat > 0 && s.params.k_barrier > 0) {
            const auto add = [&](const auto& points, int role, bool vt) {
                if (vt) {
                    const auto eval = make_node_triangle_contact_evaluation(points, s.params.d_hat, s.params.k_barrier, 1e-12, nullptr, s.params.exact_computation_fallback);
                    const auto [bg,bH] = node_triangle_barrier_self_gradient_and_hessian(points[0], points[1], points[2], points[3], role, eval);
                    g += s.params.dt2()*s.params.k_barrier*bg;
                    H += s.params.dt2()*s.params.k_barrier*bH;
                } else {
                    const auto eval = make_segment_segment_contact_evaluation(points, s.params.d_hat, s.params.k_barrier, 1e-12, nullptr, s.params.exact_computation_fallback);
                    const auto [bg,bH] = segment_segment_barrier_self_gradient_and_hessian(points[0], points[1], points[2], points[3], role, eval);
                    g += s.params.dt2()*s.params.k_barrier*bg;
                    H += s.params.dt2()*s.params.k_barrier*bH;
                }
            };
            for (const auto& entry : cache.vertex_nt[node]) {
                const auto& p = cache.nt_pairs[entry.pair_index];
                const std::array<Vec3,4> points{s.x[p.node],s.x[p.tri_v[0]],s.x[p.tri_v[1]],s.x[p.tri_v[2]]};
                add(points, entry.dof, true);
                if (s.params.friction_coefficient > 0) {
                    const std::array<Vec3,4> old{(*previous)[p.node],(*previous)[p.tri_v[0]],(*previous)[p.tri_v[1]],(*previous)[p.tri_v[2]]};
                    const auto contact = make_node_triangle_frozen_friction_contact(points, old, s.params.d_hat, s.params.k_barrier, s.params.dt(), s.params.friction_velocity_epsilon);
                    const auto [fg,fH] = frozen_friction_role_gradient_and_hessian(contact, entry.dof, s.params.friction_coefficient, s.params.dt2());
                    g += fg;
                    H += fH;
                }
            }
            for (const auto& entry : cache.vertex_ss[node]) {
                const auto& p = cache.ss_pairs[entry.pair_index];
                const std::array<Vec3,4> points{s.x[p.v[0]],s.x[p.v[1]],s.x[p.v[2]],s.x[p.v[3]]};
                add(points, entry.dof, false);
                if (s.params.friction_coefficient > 0) {
                    const std::array<Vec3,4> old{(*previous)[p.v[0]],(*previous)[p.v[1]],(*previous)[p.v[2]],(*previous)[p.v[3]]};
                    const auto contact = make_segment_segment_frozen_friction_contact(points, old, s.params.d_hat, s.params.k_barrier, s.params.dt(), s.params.friction_velocity_epsilon, 1e-12, nullptr, s.params.exact_computation_fallback);
                    const auto [fg,fH] = frozen_friction_role_gradient_and_hessian(contact, entry.dof, s.params.friction_coefficient, s.params.dt2());
                    g += fg;
                    H += fH;
                }
            }
        }
        result[node] = -s.params.damping * H.completeOrthogonalDecomposition().solve(g);
    }
    return result;
}

void expect_close(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_LE((a[i]-b[i]).norm(), 1e-10) << i;
}

} // namespace

TEST(JacobiStep, InertiaGravityAndPinsHaveTheExpectedDisplacement) {
    Scene s({Vec3::Zero(),Vec3(1,2,3)});
    s.mesh.mass = {2,3};
    s.params.fps = 4;
    s.params.gravity = Vec3(0,-4,0);
    s.params.kpin = 8;
    s.params.damping = .5;
    s.xhat[0] = Vec3(.1,-.2,.3);
    s.xhat[1] += Vec3(-.1,.2,-.3);
    s.pins = {Pin{0,Vec3(1,0,0)}};
    const auto original = s.x;
    const auto dx = s.solve();
    const double dt2 = 1.0/16.0;
    const Vec3 g0 = 2*(s.x[0]-s.xhat[0]) - 2*dt2*s.params.gravity + 8*dt2*(s.x[0]-s.pins[0].target_position);
    const Vec3 g1 = 3*(s.x[1]-s.xhat[1]) - 3*dt2*s.params.gravity;
    EXPECT_LE((dx[0] + .5*g0/(2+8*dt2)).norm(), 1e-13);
    EXPECT_LE((dx[1] + .5*g1/3).norm(), 1e-13);
    expect_same(s.x, original);
    s.params.damping = 0;
    for (const auto& value : s.solve()) EXPECT_TRUE(value.isZero());
}

TEST(JacobiStep, CoupledClothReadsOneFrozenState) {
    auto s = cloth();
    const auto original = s.x;
    const auto dx = s.solve();
    expect_close(dx, reference(s));
    expect_same(s.x, original);
    auto live = s;
    std::vector<Vec3> sequential(dx.size());
    for (std::size_t node = 0; node < dx.size(); ++node) {
        const auto [g,H] = compute_local_gradient_and_hessian_no_barrier(static_cast<int>(node), live.mesh, live.adj, live.pins, live.params, live.x, live.xhat);
        sequential[node] = -live.params.damping * H.completeOrthogonalDecomposition().solve(g);
        live.x[node] += sequential[node];
    }
    EXPECT_GT((sequential[1]-dx[1]).norm(), 1e-8);
}

TEST(JacobiStep, RepeatedSweepsRelaxClothWithoutChangingTheTarget) {
    auto s = cloth();
    s.params.damping = .5;
    const auto target = s.xhat;
    const double initial = compute_incremental_potential_no_barrier(s.mesh, s.pins, s.params, s.x, s.xhat);
    for (int iteration = 0; iteration < 30; ++iteration) {
        const auto dx = s.solve();
        for (std::size_t node = 0; node < s.x.size(); ++node) s.x[node] += dx[node];
    }
    const double final = compute_incremental_potential_no_barrier(s.mesh, s.pins, s.params, s.x, s.xhat);
    EXPECT_LT(final, .01*initial);
    expect_same(s.xhat, target);
}

TEST(JacobiStep, OwnedBVHMatchesVTAndEEReferenceWithFriction) {
    for (bool vt : {true,false}) for (double friction : {0.0,.3}) {
        SCOPED_TRACE(::testing::Message() << "VT=" << vt << " friction=" << friction);
        auto s = vt ? point_triangle() : edges();
        s.params.friction_coefficient = friction;
        auto previous = s.x;
        previous[0] -= Vec3(.02,.005,0);
        s.velocity[0] = Vec3(2,1,-.2);
        for (std::size_t i = 0; i < s.x.size(); ++i) s.xhat[i] = previous[i] + .01*s.velocity[i];
        expect_close(s.solve(&previous), reference(s, &previous));
        expect_close(s.solve(), s.solve(&previous));
    }
}

TEST(JacobiStep, OwnedBVHRefreshesWhenPositionsChange) {
    auto s = point_triangle();
    EXPECT_GT(s.solve()[0].z(), 0);
    s.x[0] = Vec3(100,100,100);
    s.xhat = s.x;
    for (const auto& value : s.solve()) EXPECT_TRUE(value.isZero());
    s.x[0] = Vec3(0,0,.1);
    s.xhat = s.x;
    EXPECT_GT(s.solve()[0].z(), 0);
}

TEST(JacobiStep, SerialAndParallelSweepsAreBitwiseEqual) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    for (bool vt : {true,false}) {
        auto s = vt ? point_triangle() : edges();
        s.params.friction_coefficient = .3;
        auto previous = s.x;
        previous[0] -= Vec3(.02,.005,0);
        omp_set_num_threads(1);
        const auto serial = s.solve(&previous);
        s.params.use_parallel = true;
        omp_set_num_threads(std::min(4,omp_get_thread_limit()));
        expect_same(s.solve(&previous), serial);
    }
    auto s = cloth();
    const auto serial = s.solve();
    s.params.use_parallel = true;
    expect_same(s.solve(), serial);
}

TEST(JacobiStep, SolidParticlesUseTheirVolumetricBlocks) {
    Scene s({});
    DeformedState state;
    create_solid({Vec3::Zero(),Vec3(1,0,0),Vec3(0,1,0),Vec3(0,0,1)}, {0,1,2,3}, 1.0, s.mesh, state);
    s.x = state.deformed_positions;
    s.xhat = s.x;
    s.velocity.assign(s.x.size(), Vec3::Zero());
    s.x[1].x() += .1;
    s.params.solid_mu = 4;
    s.params.solid_lambda = 6;
    s.params.use_parallel = true;
    const auto dx = s.solve();
    BroadPhase phase;
    for (std::size_t node = 0; node < dx.size(); ++node) {
        const auto [g,H] = compute_solid_local_gradient_and_block(static_cast<int>(node), s.mesh, s.pins, s.params, s.x, s.xhat, phase);
        const Vec3 expected = -H.completeOrthogonalDecomposition().solve(g);
        EXPECT_LE((dx[node]-expected).norm(), 1e-11);
    }
}

TEST(JacobiStep, WorkerFailuresPropagateAndWorkspaceCanBeReused) {
    auto s = point_triangle();
    s.params.use_parallel = true;
    s.x[0].z() = 0;
    EXPECT_THROW(s.solve(), std::exception);
    s.x[0].z() = .1;
    for (const auto& value : s.solve()) EXPECT_TRUE(value.allFinite());
    Scene overflow({Vec3::Zero()});
    overflow.params.use_parallel = true;
    overflow.mesh.mass[0] = std::numeric_limits<double>::max();
    overflow.xhat[0].x() = std::numeric_limits<double>::max();
    EXPECT_THROW(overflow.solve(), std::runtime_error);
    EXPECT_TRUE(overflow.x[0].isZero());
    overflow.mesh.mass[0] = 1;
    overflow.xhat[0].x() = .1;
    EXPECT_NEAR(overflow.solve()[0].x(), .1, 1e-14);
}

TEST(JacobiStep, RejectsInvalidStateAndUnsupportedRigidBlocks) {
    Scene empty({});
    EXPECT_TRUE(empty.solve().empty());
    Scene s({Vec3::Zero()});
    s.xhat.clear();
    EXPECT_THROW(s.solve(), std::invalid_argument);
    s.xhat = s.x;
    s.pins = {Pin{1,Vec3::Zero()}};
    EXPECT_THROW(s.solve(), std::out_of_range);
    s.pins.clear();
    s.params.damping = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(s.solve(), std::invalid_argument);
    s.params.damping = 1;
    s.mesh.rb_nodes = {{0}};
    EXPECT_THROW(s.solve(), std::invalid_argument);
}

TEST(JacobiStep, FrictionValidatesPreviousPositions) {
    auto s = point_triangle();
    s.params.friction_coefficient = .3;
    std::vector<Vec3> previous(1,Vec3::Zero());
    EXPECT_THROW(s.solve(&previous), std::invalid_argument);
    previous = s.x;
    previous[0].x() = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(s.solve(&previous), std::invalid_argument);
}
