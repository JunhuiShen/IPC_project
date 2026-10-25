#include "ipc_args.h"
#include "mesh_utils.h"
#include "dat_contact.h"
#include "parallel_helper.h"
#include "simulation.h"
#include "solver.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct RestoreOpenMP {
    const int threads = omp_get_max_threads();
    const int dynamic = omp_get_dynamic();
    ~RestoreOpenMP() {
        omp_set_num_threads(threads);
        omp_set_dynamic(dynamic);
    }
};

struct Scene {
    RefMesh mesh;
    VertexTriangleMap adjacency;
    std::vector<Pin> pins;
    std::vector<Vec3> initial, target, velocity;
    SimParams params = SimParams::zeros();

    Scene(std::vector<Vec3> positions, std::vector<int> triangles = {}) : initial(std::move(positions)), target(initial), velocity(initial.size(), Vec3::Zero()) {
        mesh.num_positions = initial.size();
        mesh.mass.assign(initial.size(), 1.0);
        mesh.node_to_rb.assign(initial.size(), -1);
        mesh.tris = std::move(triangles);
        params.fixed_iters = true;
        params.max_global_iters = 1;
        params.damping = 1.0;
        params.d_hat = 1.0;
        params.k_barrier = 0.0;
        params.dat_relaxation = 0.8;
        params.dat_detection_interval = 100;
        // Inertial fixtures deliberately supply empty elastic adjacency.
        // The triangles are contact geometry; they contribute no elasticity.
        params.use_parallel = false;
        params.node_box_min = params.node_box_max = 1.0;
    }

    std::vector<Vec3> solve(const std::vector<Vec3>* previous = nullptr) {
        auto positions = initial;
        const auto result = global_gauss_seidel_solver_dat(mesh, adjacency, pins, params, positions, target, velocity, "", previous);
        EXPECT_TRUE(result.converged);
        EXPECT_EQ(result.iterations, params.max_global_iters);
        for(const auto& position : positions) EXPECT_TRUE(position.allFinite());
        return positions;
    }
};

Scene point_triangle(double gap = 0.1) {
    return Scene({Vec3(0,0,gap), Vec3(-1,-1,0), Vec3(2,-1,0), Vec3(-1,2,0)}, {1,2,3});
}

void expect_close(const Vec3& actual, const Vec3& expected, double tolerance = 2e-9) {
    EXPECT_LE((actual-expected).cwiseAbs().maxCoeff(), tolerance) << "actual=" << actual.transpose() << " expected=" << expected.transpose();
}

void expect_same(const std::vector<Vec3>& actual, const std::vector<Vec3>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for(std::size_t i=0;i<actual.size();++i) {
        EXPECT_EQ(std::memcmp(actual[i].data(), expected[i].data(), 3*sizeof(double)), 0) << "vertex=" << i;
    }
}

bool parse(IPCArgs3D& args, std::vector<std::string> words) {
    std::vector<char*> argv;
    for(auto& word : words) argv.push_back(word.data());
    return args.parse(static_cast<int>(argv.size()), argv.data());
}

} // namespace

TEST(PlanarDATSolver, TangentialAndSeparatingMotionUseFullNewtonStep) {
    for(const Vec3& displacement : {Vec3(.2,0,0), Vec3(0,0,.2)}) {
        auto scene = point_triangle(.01);
        scene.target[0] += displacement;
        const auto positions = scene.solve();
        expect_close(positions[0], scene.target[0]);
        // This is much larger than the OGC spherical limit 0.4*gap.
        EXPECT_GT((positions[0]-scene.initial[0]).norm(), .1);
        for(int i=1;i<4;++i) expect_close(positions[i], scene.initial[i]);
    }
}

TEST(PlanarDATSolver, ApproachingPointStopsOnOriginalSide) {
    auto scene = point_triangle();
    scene.target[0].z() = -.4;
    const auto positions = scene.solve();
    // A stationary triangle gives the point almost the full gap; gamma
    // leaves 20% of the gap, plus the interior-plane numerical margin.
    EXPECT_NEAR(positions[0].z(), .02 + 8e-10, 1e-13);
    EXPECT_GT(positions[0].z(), 0.0);
}

TEST(PlanarDATSolver, MovingTriangleConstrainsEveryTriangleVertex) {
    auto scene = point_triangle();
    for(int i=1;i<4;++i) scene.target[i].z() = .5;
    const auto positions = scene.solve();
    expect_close(positions[0], scene.initial[0]);
    for(int i=1;i<4;++i) {
        EXPECT_NEAR(positions[i].z(), .08 - 8e-10, 1e-13) << "vertex=" << i;
        EXPECT_LT(positions[i].z(), positions[0].z());
    }
}

TEST(PlanarDATSolver, OpposingProposalsShareTheAvailableGap) {
    auto scene = point_triangle();
    scene.target[0].z() -= .5;
    for(int i=1;i<4;++i) scene.target[i].z() += .5;
    const auto positions = scene.solve();
    EXPECT_NEAR(positions[0].z(), .06, 1e-11);
    for(int i=1;i<4;++i) {
        EXPECT_NEAR(positions[i].z(), .04, 1e-11);
        EXPECT_NEAR(positions[0].z()-positions[i].z(), .02, 1e-11);
    }
}

TEST(PlanarDATSolver, UnequalApproachesGetProportionalRoom) {
    auto scene = point_triangle();
    scene.target[0].z() -= .8;
    for (int i = 1; i < 4; ++i) scene.target[i].z() += .2;
    const auto positions = scene.solve();
    EXPECT_NEAR(positions[0].z(), .036, 1e-12);
    for (int i = 1; i < 4; ++i) {
        EXPECT_NEAR(positions[i].z(), .016, 1e-12);
        EXPECT_NEAR(scene.initial[0].z() - positions[0].z(), 4 * positions[i].z(), 1e-12);
        EXPECT_NEAR(positions[0].z() - positions[i].z(), .02, 1e-12);
    }
}

TEST(PlanarDATSolver, EdgeEdgePlanesConstrainAllFourEndpoints) {
    // The two long edges cross in projection. Every nonincident vertex is
    // farther than rq=.3 from the opposite triangle, isolating EE bounds.
    const std::vector<Vec3> initial={Vec3(-1,0,0), Vec3(1,0,0), Vec3(-1,-1,0), Vec3(0,-1,.1), Vec3(0,1,.1), Vec3(1,1,.1)};
    for(int moving : {0,1,3,4}) {
        SCOPED_TRACE(::testing::Message() << "moving endpoint=" << moving);
        Scene scene(initial, {0,1,2,3,4,5});
        scene.params.d_hat = .2;
        scene.target[moving].z() += moving < 3 ? .5 : -.5;
        const auto positions = scene.solve();
        EXPECT_NEAR(std::abs(positions[moving].z()-initial[moving].z()), .08 - 8e-10, 1e-13);
        for(int i=0;i<6;++i)
            if(i!=moving) expect_close(positions[i], initial[i]);
    }
}

TEST(PlanarDATSolver, QueryIncludesPairsBeyondBarrierActivationDistance) {
    auto scene = point_triangle(.12);
    scene.params.d_hat = .1; // gap is inactive, but lies inside rq=.15.
    scene.target[0].z() -= .5;
    for(int i=1;i<4;++i) scene.target[i].z() += .5;
    const auto positions = scene.solve();
    // Without this pair, both sides would exhaust the .06 query motion cap
    // and meet. Its adaptive plane instead leaves 20% of the original gap.
    EXPECT_NEAR(positions[0].z(), .072, 1e-11);
    for(int i=1;i<4;++i) {
        EXPECT_NEAR(positions[i].z(), .048, 1e-11);
        EXPECT_NEAR(positions[0].z()-positions[i].z(), .024, 1e-11);
    }
}

TEST(PlanarDATSolver, OmittedFarPairStillHasTheQueryMotionCap) {
    auto scene = point_triangle(.4);
    scene.params.d_hat = .1;
    scene.target[0].z() = -.4;
    const auto positions = scene.solve();
    EXPECT_NEAR(positions[0].z(), .34, 1e-11);
    EXPECT_NEAR((positions[0]-scene.initial[0]).norm(), .06, 1e-11);
    EXPECT_GT(positions[0].z(), 0.0);
}

TEST(PlanarDATSolver, DetectionRefreshAllowsFurtherProgress) {
    Scene stale({Vec3::Zero()});
    stale.params.d_hat = .1;
    stale.params.max_global_iters = 3;
    stale.target[0].x() = 1.0;
    const auto without_refresh = stale.solve();
    EXPECT_NEAR(without_refresh[0].x(), .06, 1e-11);

    Scene refreshed({Vec3::Zero()});
    refreshed.params = stale.params;
    refreshed.params.dat_detection_interval = 1;
    refreshed.target = stale.target;
    const auto with_refresh = refreshed.solve();
    EXPECT_NEAR(with_refresh[0].x(), .18, 1e-11);
    EXPECT_GT(with_refresh[0].x(), without_refresh[0].x());
}

TEST(PlanarDATSolver, ZeroNewtonIncrementPreservesAcceptedAccumulatedMotion) {
    auto scene = point_triangle();
    scene.params.max_global_iters = 3;
    scene.target[0].z() = .05;
    const auto positions = scene.solve();
    expect_close(positions[0], scene.target[0]);
}

TEST(PlanarDATSolver, ObtuseTriangleUsesCorrectedDistanceAndSeparatingPlane) {
    Scene scene({Vec3(.5,-.01,.1), Vec3::Zero(), Vec3(1,0,0), Vec3(-10,.1,0)}, {1,2,3});
    const std::array<Vec3, 4> points{scene.initial[0], scene.initial[1], scene.initial[2], scene.initial[3]};
    const auto distance = ogc_detail::node_triangle_distance(points[0], points[1], points[2], points[3]);
    ASSERT_EQ(distance.region, NodeTriangleRegion::Edge12);
    ASSERT_TRUE(distance.closest_point.isApprox(Vec3(.5,0,0)));
    ASSERT_TRUE(dat_detail::make_vertex_triangle_geometry(points, distance).valid_plane);
    scene.target[0] += Vec3(0,.02,-.2);
    const auto positions = scene.solve();
    // This ray would touch the triangle edge at its midpoint. The stationary
    // triangle leaves the point room to approach, with a positive gap retained.
    expect_close(positions[0], scene.initial[0] + .4 * (1.0 - 1e-8) * Vec3(0,.02,-.2), 1e-12);
    EXPECT_GT(positions[0].z(), 0.0);
}

TEST(OGCSolver, ObtuseTriangleUsesCorrectedDistanceForMotionBound) {
    Scene scene({Vec3(.5,-.01,.1), Vec3::Zero(), Vec3(1,0,0), Vec3(-10,.1,0)}, {1,2,3});
    const Vec3 proposal(0,.02,-.2);
    scene.target[0] += proposal;
    auto positions = scene.initial;
    const auto result = global_gauss_seidel_solver_ogc(scene.mesh, scene.adjacency, scene.pins, scene.params, positions, scene.target, scene.velocity, "", &scene.initial);
    ASSERT_TRUE(result.converged);
    // The correct gap is sqrt(.01^2+.1^2), rather than the legacy .51.
    expect_close(positions[0], scene.initial[0] + .2 * proposal, 1e-13);
    EXPECT_NEAR((positions[0] - scene.initial[0]).norm(), .4 * std::sqrt(.0101), 1e-13);
    for (int i = 1; i < 4; ++i) expect_close(positions[i], scene.initial[i]);
}

TEST(OGCDATContact, CorrectedDistanceActivatesBarrierAndFrictionAssembly) {
    for (bool dat : {false, true}) for (double friction : {0.0, .3}) {
        SCOPED_TRACE(::testing::Message() << "dat=" << dat << " friction=" << friction);
        Scene scene({Vec3(.5,-.01,.1), Vec3::Zero(), Vec3(1,0,0), Vec3(-10,.1,0)}, {1,2,3});
        scene.params.d_hat = .2;
        scene.params.k_barrier = 100;
        scene.params.friction_coefficient = friction;
        auto positions = scene.initial;
        if (dat) global_gauss_seidel_solver_dat(scene.mesh, scene.adjacency, scene.pins, scene.params, positions, scene.target, scene.velocity, "", &scene.initial);
        else global_gauss_seidel_solver_ogc(scene.mesh, scene.adjacency, scene.pins, scene.params, positions, scene.target, scene.velocity, "", &scene.initial);
        const Vec3 separation(0,-.01,.1);
        EXPECT_GT(separation.dot(positions[0] - scene.initial[0]), 0.0);
    }
}

TEST(PlanarDATSolver, RoundedWorldCoordinatesStayStrictlyAboveThePlane) {
    constexpr double plane = 1e12;
    const double next = std::nextafter(plane, std::numeric_limits<double>::infinity());
    Scene scene({Vec3(0,0,next), Vec3(-.5,-.5,plane), Vec3(.5,-.5,plane), Vec3(0,.5,plane)}, {1,2,3});
    scene.params.d_hat = .01;
    scene.target[0].z() = plane-.1;
    const auto positions = scene.solve();
    // The analytic bound allows most of one ULP, which would round onto the
    // plane. The stored-endpoint guard must retain a representable safe gap.
    EXPECT_GT(positions[0].z(), plane);
    EXPECT_DOUBLE_EQ(positions[0].z(), next);
    expect_same(positions,scene.initial);
}

TEST(PlanarDATSolver, NonfiniteNewtonProposalKeepsTheAcceptedState) {
    Scene scene({Vec3::Zero()});
    scene.params.fps = 1.0;
    const double largest = std::numeric_limits<double>::max();
    scene.target[0].x() = largest;
    scene.params.gravity.x() = largest;
    // Both inputs are finite, but the combined inertial/gravity gradient
    // overflows. A nonfinite Newton proposal must not enter the DAT filter
    // or turn zero scaling into 0*infinity/NaN coordinates.
    ASSERT_TRUE(scene.target[0].allFinite());
    ASSERT_TRUE(scene.params.gravity.allFinite());
    const auto positions = scene.solve();
    expect_same(positions,scene.initial);
}

TEST(PlanarDATSolver, SerialAndParallelColorStagesAgreeWithFriction) {
    RestoreOpenMP restore;
    omp_set_dynamic(0);
    std::vector<Vec3> frictionless;
    for(double friction : {0.0,.3}) {
        auto scene = point_triangle(.125);
        scene.mesh.compute_dm_inverse(scene.initial);
        scene.adjacency = build_incident_triangle_map(scene.mesh.tris);
        scene.params.d_hat = .2;
        scene.params.k_barrier = 2.0;
        scene.params.friction_coefficient = friction;
        scene.params.max_global_iters = 4;
        scene.params.dat_detection_interval = 2;
        scene.target[0] += Vec3(.03125,0,-.015625);
        auto previous = scene.initial;
        previous[0].x() -= .015625;
        const auto previous_copy = previous;
        const auto reference = scene.solve(&previous);
        if(friction == 0.0) frictionless = reference;
        else EXPECT_LT(reference[0].x(), frictionless[0].x());
        for(int threads : {1,4,8}) {
            omp_set_num_threads(threads);
            scene.params.use_parallel = true;
            const auto parallel = scene.solve(&previous);
            expect_same(parallel, reference);
        }
        expect_same(previous, previous_copy);
        // Repeating the same public solve must reset its detection anchor.
        scene.params.use_parallel = false;
        expect_same(scene.solve(&previous), reference);
    }
}

TEST(PlanarDATSolver, PreviousPositionsAreOptionalOnlyWhenFrictionAllowsIt) {
    auto scene = point_triangle(.125);
    scene.params.d_hat = .2;
    scene.params.k_barrier = 2.0;
    scene.target[0].x() += .03125;
    const std::vector<Vec3> wrong_size(1,Vec3::Zero());
    const auto zero_friction = scene.solve();
    expect_same(scene.solve(&wrong_size), zero_friction);

    scene.params.friction_coefficient = .3;
    EXPECT_THROW(scene.solve(&wrong_size), std::invalid_argument);
    // Construct exactly the same history used by the public reconstruction.
    scene.velocity[0] = Vec3(.75,0,0);
    auto reconstructed = scene.target;
    for(std::size_t i=0;i<reconstructed.size();++i)
        reconstructed[i] -= scene.params.dt()*scene.velocity[i];
    expect_same(scene.solve(), scene.solve(&reconstructed));
}

TEST(PlanarDATSolver, RejectsInvalidDirectSolverParameters) {
    auto scene = point_triangle();
    const auto good = scene.params;
    scene.params.fixed_iters = false;
    EXPECT_THROW(scene.solve(), std::invalid_argument);
    for(double d_hat : {0.0,-1.0,std::numeric_limits<double>::max(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        scene.params = good;scene.params.d_hat = d_hat;
        EXPECT_THROW(scene.solve(), std::invalid_argument);
    }
    for(double relaxation : {0.0,1.0,-.1, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        scene.params = good;scene.params.dat_relaxation = relaxation;
        EXPECT_THROW(scene.solve(), std::invalid_argument);
    }
    scene.params = good;scene.params.dat_detection_interval = 0;
    EXPECT_THROW(scene.solve(), std::invalid_argument);
    scene.params = good;scene.params.max_global_iters = 0;
    EXPECT_THROW(scene.solve(), std::invalid_argument);
    scene.params = good;scene.mesh.tets = {0,1,2,3};
    EXPECT_THROW(scene.solve(), std::invalid_argument);
    scene.mesh.tets.clear();scene.mesh.node_to_rb[0] = 0;
    EXPECT_THROW(scene.solve(), std::invalid_argument);
}

TEST(PlanarDATSolver, DefaultsAndInactiveKnobsPreserveOtherSolverSelection) {
    const auto zero = SimParams::zeros();
    EXPECT_FALSE(zero.use_dat_solver);
    EXPECT_DOUBLE_EQ(zero.dat_relaxation,.8);
    EXPECT_EQ(zero.dat_detection_interval,5);
    IPCArgs3D args;
    EXPECT_FALSE(args.to_sim_params().use_dat_solver);
    args.dat_relaxation = std::numeric_limits<double>::quiet_NaN();
    args.dat_detection_interval = 0;
    EXPECT_NO_THROW(args.to_sim_params());
}

TEST(PlanarDATSolver, ArgumentsSelectDATAndValidateItsSettings) {
    IPCArgs3D args;
    ASSERT_TRUE(parse(args,{"3D_sim","--use_dat_solver","true", "--fixed_iters","true","--d_hat","0.02", "--dat_relaxation","0.75","--dat_detection_interval","3"}));
    const auto params = args.to_sim_params();
    EXPECT_TRUE(params.use_dat_solver);
    EXPECT_FALSE(params.use_ogc_solver);
    EXPECT_DOUBLE_EQ(params.dat_relaxation,.75);
    EXPECT_EQ(params.dat_detection_interval,3);
    args.fixed_iters = false;
    EXPECT_THROW(args.to_sim_params(), std::invalid_argument);
    args.fixed_iters = true;args.use_ogc_solver = true;
    EXPECT_THROW(args.to_sim_params(), std::invalid_argument);
    args.use_ogc_solver = false;args.use_ogc = true;
    EXPECT_THROW(args.to_sim_params(), std::invalid_argument);
    args.use_ogc = false;args.ccd_exact_step_bound = true;
    args.use_ticcd = false;
    EXPECT_THROW(args.to_sim_params(), std::invalid_argument);
    args.ccd_exact_step_bound = false;
    args.use_ogc_solver = false;args.use_cloth_grid = true;
    EXPECT_THROW(args.to_sim_params(), std::invalid_argument);
}

TEST(PlanarDATSolver, FrameDriverSelectsDATAndStartsFromPreviousPositions) {
    auto scene = point_triangle(.01);
    scene.params.d_hat = .1;
    scene.params.use_dat_solver = true;
    scene.params.use_verlet_guess = true;
    scene.params.use_translation_guess = true;
    scene.params.use_colored_ccd_guess = true;
    scene.params.use_basic_experimental = true;
    scene.params.use_basic_experimental_v2 = true;
    scene.params.use_simd = true;
    scene.velocity[0].z() = -3.3;
    scene.target = scene.initial;
    scene.target[0] += scene.params.dt()*scene.velocity[0];
    const auto expected = scene.solve(&scene.initial);

    DeformedState state;
    state.deformed_positions = scene.initial;
    state.velocities = scene.velocity;
    BroadPhase broad_phase;
    const auto result = advance_one_frame(state,scene.mesh,scene.adjacency, scene.pins,scene.params,broad_phase);
    ASSERT_TRUE(result.converged);
    expect_same(state.deformed_positions, expected);
    // Allow the conservative rounding margin in the supporting projections.
    EXPECT_NEAR(state.deformed_positions[0].z(), .002 + 8e-11, 1e-13);
}
