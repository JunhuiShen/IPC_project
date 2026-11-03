#include "DAT.h"
#include "broad_phase.h"
#include "mesh_utils.h"
#include "physics.h"
#include "solver.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr double gamma = 0.9;
constexpr double query_radius = 1.0;
constexpr double tolerance = 1e-12;

struct Scene {
    RefMesh mesh;
    std::vector<Vec3> positions;
    std::vector<Vec3> update;
};

Scene point_triangle(double gap = 0.4) {
    Scene scene;
    scene.positions = {Vec3(0, 0, gap), Vec3(-3, -3, 0),
        Vec3(3, -3, 0), Vec3(0, 3, 0)};
    scene.update.assign(scene.positions.size(), Vec3::Zero());
    scene.mesh.num_positions = scene.positions.size();
    scene.mesh.tris = {1, 2, 3};
    return scene;
}

Scene crossed_edges(double gap = 0.4) {
    Scene scene;
    // Only edges (0,1) and (3,4) are within the query radius. The
    // off-plane apices keep every cross-triangle VT and other EE pair far.
    scene.positions = {Vec3(-10, 0, 0), Vec3(10, 0, 0),
        Vec3(0, -10, -10), Vec3(0, -10, gap),
        Vec3(0, 10, gap), Vec3(10, 0, 10 + gap)};
    scene.update = {Vec3(0, 0, 0.4), Vec3(0, 0, 0.4),
        Vec3(0, 0, 0.4), Vec3(0, 0, -0.1),
        Vec3(0, 0, -0.1), Vec3(0, 0, -0.1)};
    scene.mesh.num_positions = scene.positions.size();
    scene.mesh.tris = {0, 1, 2, 3, 4, 5};
    return scene;
}

Scene point_two_triangles() {
    Scene scene = point_triangle();
    for (int vertex = 1; vertex < 4; ++vertex) {
        scene.positions[vertex].z() = 0.2;
    }
    scene.positions.insert(scene.positions.end(), {
        Vec3(-3, -3, 0.1), Vec3(3, -3, 0.1), Vec3(0, 3, 0.1)});
    scene.mesh.tris.insert(scene.mesh.tris.end(), {4, 5, 6});
    scene.mesh.num_positions = scene.positions.size();
    scene.update.assign(scene.positions.size(), Vec3::Zero());
    scene.update[0] = Vec3(0, 0, -0.4);
    return scene;
}

std::vector<Vec3> filter(const Scene& scene, double padding = 1.2) {
    return planar_DAT(gamma, query_radius, scene.mesh, scene.update,
        scene.positions, padding);
}

void expect_vector_near(const Vec3& actual, const Vec3& expected) {
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_NEAR(actual[axis], expected[axis], tolerance);
    }
}

void expect_vectors_identical(const std::vector<Vec3>& actual,
    const std::vector<Vec3>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t vertex = 0; vertex < actual.size(); ++vertex) {
        SCOPED_TRACE(vertex);
        EXPECT_EQ(std::memcmp(actual[vertex].data(), expected[vertex].data(),
            3 * sizeof(double)), 0);
    }
}

class PlanarDAT : public testing::Test {
protected:
    int previous_threads = omp_get_max_threads();
    int previous_dynamic = omp_get_dynamic();

    void SetUp() override {
        omp_set_dynamic(0);
        omp_set_num_threads(1);
    }

    void TearDown() override {
        omp_set_num_threads(previous_threads);
        omp_set_dynamic(previous_dynamic);
    }
};

TEST_F(PlanarDAT, CapsEachUnconstrainedUpdateWithoutGlobalDamping) {
    Scene scene;
    scene.positions = {Vec3::Zero(), Vec3(1, 0, 0), Vec3(2, 0, 0)};
    scene.update = {Vec3(3, 4, 0), Vec3(0.1, -0.2, 0.1), Vec3::Zero()};
    scene.mesh.num_positions = scene.positions.size();

    const auto result = filter(scene);
    ASSERT_EQ(result.size(), scene.update.size());
    expect_vector_near(result[0], Vec3(0.27, 0.36, 0));
    expect_vector_near(result[1], scene.update[1]);
    expect_vector_near(result[2], Vec3::Zero());
    EXPECT_NEAR(result[0].norm(), 0.5 * gamma * query_radius, tolerance);
}

TEST_F(PlanarDAT, IgnoresIncidentPrimitivesOfASingleTriangle) {
    Scene scene;
    scene.positions = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0)};
    scene.update.assign(3, Vec3(0.1, 0.2, 0.1));
    scene.mesh.num_positions = 3;
    scene.mesh.tris = {0, 1, 2};
    const auto result = filter(scene);
    ASSERT_EQ(result.size(), 3u);
    for (int vertex = 0; vertex < 3; ++vertex) {
        expect_vector_near(result[vertex], scene.update[vertex]);
    }
}

TEST_F(PlanarDAT, ProportionalVertexTrianglePlaneRelaxesExactlyOnce) {
    Scene scene = point_triangle();
    scene.update[0] = Vec3(0, 0, -0.4);
    for (int vertex = 1; vertex < 4; ++vertex) {
        scene.update[vertex] = Vec3(0, 0, 0.1);
    }

    const auto positions_before = scene.positions;
    const auto update_before = scene.update;
    const auto result = filter(scene);
    ASSERT_EQ(result.size(), 4u);
    // The gap is 0.4 and the closing speed 0.5: alpha=0.9*0.4/0.5.
    for (int vertex = 0; vertex < 4; ++vertex) {
        expect_vector_near(result[vertex], 0.72 * scene.update[vertex]);
    }
    const double final_gap = scene.positions[0].z() + result[0].z()
        - scene.positions[1].z() - result[1].z();
    EXPECT_NEAR(final_gap, 0.04, tolerance);
    expect_vectors_identical(scene.positions, positions_before);
    expect_vectors_identical(scene.update, update_before);
}

TEST_F(PlanarDAT, VertexTriangleTruncationIsInvariantUnderUniformScaling) {
    Scene unscaled = point_triangle();
    unscaled.update[0] = Vec3(0, 0, -0.4);
    for (int vertex = 1; vertex < 4; ++vertex) {
        unscaled.update[vertex] = Vec3(0, 0, 0.1);
    }
    for (double scale : {1e-100, 1e80}) {
        SCOPED_TRACE(scale);
        Scene scene = unscaled;
        for (Vec3& position : scene.positions) position *= scale;
        for (Vec3& update : scene.update) update *= scale;
        const auto result = planar_DAT(gamma, query_radius * scale, scene.mesh,
            scene.update, scene.positions);
        ASSERT_EQ(result.size(), 4u);
        for (int vertex = 0; vertex < 4; ++vertex) {
            // Compare in the original units so a tiny-scale zero result
            // cannot pass an absolute world-space tolerance accidentally.
            expect_vector_near(result[vertex] / scale,
                0.72 * unscaled.update[vertex]);
        }
    }
}

TEST_F(PlanarDAT, OneSidedApproachReceivesTheWholeAvailableGap) {
    for (bool triangle_moves : {false, true}) {
        SCOPED_TRACE(triangle_moves);
        Scene scene = point_triangle();
        if (triangle_moves) {
            for (int vertex = 1; vertex < 4; ++vertex) {
                scene.update[vertex] = Vec3(0, 0, 0.4);
            }
        } else {
            scene.update[0] = Vec3(0, 0, -0.4);
        }
        const auto result = filter(scene);
        ASSERT_EQ(result.size(), 4u);
        for (int vertex = 0; vertex < 4; ++vertex) {
            expect_vector_near(result[vertex], gamma * scene.update[vertex]);
        }
    }
}

TEST_F(PlanarDAT, TangentialAndSeparatingMotionRemainUnchanged) {
    for (bool separating : {false, true}) {
        SCOPED_TRACE(separating);
        Scene scene = point_triangle();
        scene.update[0] = Vec3(0.1, 0.2, separating ? 0.1 : 0.0);
        for (int vertex = 1; vertex < 4; ++vertex) {
            scene.update[vertex] = Vec3(-0.2, 0.1, separating ? -0.1 : 0.0);
        }
        const auto result = filter(scene);
        ASSERT_EQ(result.size(), scene.update.size());
        for (std::size_t vertex = 0; vertex < result.size(); ++vertex) {
            expect_vector_near(result[vertex], scene.update[vertex]);
        }
    }
}

TEST_F(PlanarDAT, TriangleUsesMaximumCornerApproach) {
    Scene scene = point_triangle();
    scene.update[0] = Vec3(0, 0, -0.2);
    scene.update[1] = Vec3(0, 0, 0.4);
    const auto result = filter(scene);
    ASSERT_EQ(result.size(), 4u);
    // A nearest-point barycentric velocity would underestimate this corner.
    expect_vector_near(result[0], Vec3(0, 0, -0.12));
    expect_vector_near(result[1], Vec3(0, 0, 0.24));
    expect_vector_near(result[2], Vec3::Zero());
    expect_vector_near(result[3], Vec3::Zero());
}

TEST_F(PlanarDAT, ConstructsAdaptivePlanesFromAlreadyCappedUpdates) {
    Scene scene = point_triangle();
    scene.update[0] = Vec3(0, 0, -4);
    for (int vertex = 1; vertex < 4; ++vertex) {
        scene.update[vertex] = Vec3(0, 0, 0.1);
    }
    const auto result = filter(scene);
    ASSERT_EQ(result.size(), 4u);
    const double alpha = gamma * 0.4 / (0.45 + 0.1);
    expect_vector_near(result[0], Vec3(0, 0, -0.45 * alpha));
    for (int vertex = 1; vertex < 4; ++vertex) {
        expect_vector_near(result[vertex], Vec3(0, 0, 0.1 * alpha));
    }
    EXPECT_NEAR(0.4 + result[0].z() - result[1].z(), 0.04, tolerance);
}

TEST_F(PlanarDAT, QueryBoundaryAndFarPairsNeedOnlyTheDisplacementCap) {
    for (double gap : {1.0, 1.01, 3.0}) {
        SCOPED_TRACE(gap);
        Scene scene = point_triangle(gap);
        scene.update[0] = Vec3(0, 0, -10);
        for (int vertex = 1; vertex < 4; ++vertex) {
            scene.update[vertex] = Vec3(0, 0, 10);
        }
        const auto result = filter(scene);
        ASSERT_EQ(result.size(), 4u);
        expect_vector_near(result[0], Vec3(0, 0, -0.45));
        for (int vertex = 1; vertex < 4; ++vertex) {
            expect_vector_near(result[vertex], Vec3(0, 0, 0.45));
        }
        EXPECT_GE(gap + result[0].z() - result[1].z(), 0.1 - tolerance);
    }
}

TEST_F(PlanarDAT, SliverTriangleUsesFiniteTriangleDistanceForRadiusFiltering) {
    Scene scene;
    scene.positions = {Vec3(0.5, 1e-3, 1e-4), Vec3(0, 0, 0),
        Vec3(1, 0, 0), Vec3(1, 1e-9, 0)};
    scene.update.assign(4, Vec3::Zero());
    scene.update[0] = Vec3(0, 0, -1e-4);
    scene.mesh.num_positions = 4;
    scene.mesh.tris = {1, 2, 3};

    // The point is near the supporting plane but outside the sliver. Its
    // true finite-triangle distance exceeds r_q, even though the AABBs overlap.
    const auto result = planar_DAT(gamma, 5e-4, scene.mesh,
        scene.update, scene.positions);
    ASSERT_EQ(result.size(), 4u);
    for (int vertex = 0; vertex < 4; ++vertex) {
        expect_vector_near(result[vertex], scene.update[vertex]);
    }
}

TEST_F(PlanarDAT, ObtuseTriangleTruncatesAgainstTheActualNearestEdge) {
    for (double scale : {1.0, 1e-4}) {
        const std::array<Vec3, 3> triangle{{Vec3::Zero(), Vec3(1, 0, 0), Vec3(-10, .1, 0)}};
        std::array<int, 3> order{{0, 1, 2}};
        do {
            Scene scene;
            scene.positions = {scale * Vec3(.5, -.01, .1), scale * triangle[order[0]], scale * triangle[order[1]], scale * triangle[order[2]]};
            scene.update.assign(4, Vec3::Zero());
            scene.update[0] = scale * Vec3(0, .02, -.2);
            scene.mesh.num_positions = 4;
            scene.mesh.tris = {1, 2, 3};
            // The actual gap sqrt(.0101) is inside r_q=.3. The old
            // barycentric classification returned .51 and missed this pair.
            const auto result = planar_DAT(gamma, .3 * scale, scene.mesh, scene.update, scene.positions);
            expect_vector_near(result[0] / scale, gamma * Vec3(0, .01, -.1));
            for (int vertex = 1; vertex < 4; ++vertex) expect_vector_near(result[vertex], Vec3::Zero());
        } while (std::next_permutation(order.begin(), order.end()));
    }
}

TEST_F(PlanarDAT, EdgeEdgeContactTruncatesItsFourEndpointsOnly) {
    const Scene scene = crossed_edges();
    const auto result = filter(scene);
    ASSERT_EQ(result.size(), 6u);
    for (int vertex : {0, 1, 3, 4}) {
        expect_vector_near(result[vertex], 0.72 * scene.update[vertex]);
    }
    for (int vertex : {2, 5}) {
        expect_vector_near(result[vertex], scene.update[vertex]);
    }
    EXPECT_NEAR(0.4 + result[3].z() - result[0].z(), 0.04, tolerance);
}

TEST_F(PlanarDAT, SharedVertexTakesMinimumIndependentOfTriangleOrder) {
    Scene scene = point_two_triangles();
    const auto result = filter(scene);
    ASSERT_EQ(result.size(), scene.positions.size());
    // The two independent limits are 0.45 and 0.675.
    expect_vector_near(result[0], Vec3(0, 0, -0.18));
    for (std::size_t vertex = 1; vertex < result.size(); ++vertex) {
        expect_vector_near(result[vertex], Vec3::Zero());
    }
    scene.mesh.tris = {4, 5, 6, 1, 2, 3};
    expect_vectors_identical(filter(scene), result);
}

TEST_F(PlanarDAT, PaddingDoesNotChangeTheFilteredResult) {
    Scene scene = point_triangle();
    scene.update[0] = Vec3(0, 0, -0.4);
    const auto result = filter(scene, 1.0);
    expect_vector_near(result[0], Vec3(0, 0, -0.36));
    expect_vectors_identical(filter(scene, 1.2), result);
    expect_vectors_identical(filter(scene, 3.0), result);
}

TEST_F(PlanarDAT, SerialAndOpenMPResultsAreBitwiseIdentical) {
    Scene scene;
    for (int copy = 0; copy < 48; ++copy) {
        const Scene component = copy % 2 ? crossed_edges() : point_two_triangles();
        const int base = static_cast<int>(scene.positions.size());
        for (const Vec3& position : component.positions) {
            scene.positions.push_back(position + Vec3(30 * copy, 0, 0));
        }
        scene.update.insert(scene.update.end(),
            component.update.begin(), component.update.end());
        for (int vertex : component.mesh.tris) {
            scene.mesh.tris.push_back(base + vertex);
        }
    }
    scene.mesh.num_positions = scene.positions.size();
    omp_set_num_threads(1);
    const auto serial = filter(scene);
    omp_set_num_threads(4);
    for (int repeat = 0; repeat < 3; ++repeat) {
        SCOPED_TRACE(repeat);
        expect_vectors_identical(filter(scene), serial);
    }
}

TEST_F(PlanarDAT, SolidBoundaryContactsExcludeInteriorTetVertices) {
    Scene scene;
    // Four tetrahedra share interior node 4; node 5 approaches the outer
    // bottom face. The interior update stays inside the solid throughout.
    scene.positions = {Vec3(-3, -3, 0), Vec3(3, -3, 0), Vec3(0, 3, 0),
        Vec3(0, 0, 3), Vec3(0, 0, 0.1), Vec3(0, 0, -0.2)};
    scene.update.assign(scene.positions.size(), Vec3::Zero());
    scene.update[4] = Vec3(0, 0, -0.095);
    scene.update[5] = Vec3(0, 0, 0.4);
    scene.mesh.num_positions = scene.positions.size();
    scene.mesh.tris = {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3};
    scene.mesh.tets = {4, 0, 2, 1, 4, 0, 1, 3, 4, 1, 2, 3, 4, 2, 0, 3};
    scene.mesh.tet_nodes = {0, 1, 2, 3, 4};
    scene.mesh.surface_nodes = {0, 1, 2, 3};

    const auto result = filter(scene);
    ASSERT_EQ(result.size(), 6u);
    for (int vertex = 0; vertex < 4; ++vertex) {
        expect_vector_near(result[vertex], Vec3::Zero());
    }
    expect_vector_near(result[4], scene.update[4]);
    expect_vector_near(result[5], Vec3(0, 0, 0.18));
}

TEST_F(PlanarDAT, RejectsInitiallyTouchingVertexTriangleAndEdgeEdgePairs) {
    EXPECT_THROW(filter(point_triangle(0.0)), std::domain_error);
    EXPECT_THROW(filter(crossed_edges(0.0)), std::domain_error);
}

TEST_F(PlanarDAT, RejectsInvalidScalarParameters) {
    const Scene scene = point_triangle();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double invalid : {-0.1, 0.0, 1.0, 1.1, nan, inf}) {
        SCOPED_TRACE(invalid);
        EXPECT_THROW(planar_DAT(invalid, query_radius, scene.mesh,
            scene.update, scene.positions), std::invalid_argument);
    }
    for (double invalid : {-1.0, 0.0, nan, inf}) {
        SCOPED_TRACE(invalid);
        EXPECT_THROW(planar_DAT(gamma, invalid, scene.mesh,
            scene.update, scene.positions), std::invalid_argument);
    }
    for (double invalid : {-1.0, 0.99, nan, inf}) {
        SCOPED_TRACE(invalid);
        EXPECT_THROW(filter(scene, invalid), std::invalid_argument);
    }
}

TEST_F(PlanarDAT, RejectsMismatchedSizesAndMalformedTriangleIndices) {
    const Scene valid = point_triangle();
    Scene scene = valid;
    scene.mesh.num_positions += 1;
    EXPECT_THROW(filter(scene), std::invalid_argument);
    scene = valid;
    scene.positions.pop_back();
    EXPECT_THROW(filter(scene), std::invalid_argument);
    scene = valid;
    scene.update.pop_back();
    EXPECT_THROW(filter(scene), std::invalid_argument);
    scene = valid;
    scene.mesh.tris.pop_back();
    EXPECT_THROW(filter(scene), std::invalid_argument);
    for (int invalid : {-1, 4}) {
        scene = valid;
        scene.mesh.tris[0] = invalid;
        EXPECT_THROW(filter(scene), std::invalid_argument);
    }
}

TEST_F(PlanarDAT, RejectsNonfinitePositionsAndUpdates) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double invalid : {nan, inf, -inf}) {
        SCOPED_TRACE(invalid);
        Scene scene = point_triangle();
        scene.positions[0].x() = invalid;
        EXPECT_THROW(filter(scene), std::invalid_argument);
        scene = point_triangle();
        scene.update[2].z() = invalid;
        EXPECT_THROW(filter(scene), std::invalid_argument);
    }
}

TEST_F(PlanarDAT, EmptyMeshProducesAnEmptyUpdate) {
    EXPECT_TRUE(filter(Scene{}).empty());
}

TEST_F(PlanarDAT, PairOnlyBroadPhaseAndReusableScratchMatchStandaloneAfterMotion) {
    auto scene=point_triangle(.4);
    scene.update[0]=Vec3(0,0,-.8);
    DATWorkspace workspace;
    BroadPhase phase;
    for(int sweep=0;sweep<4;++sweep) {
        std::vector<AABB> boxes;
        for(const auto& x:scene.positions)boxes.emplace_back(x-Vec3::Constant(2),x+Vec3::Constant(2));
        phase.initialize(boxes,scene.mesh,0,BroadPhase::InitializationMode::DeformableSolver,false);
        ASSERT_TRUE(phase.cache().vertex_nt.empty());
        ASSERT_TRUE(phase.cache().vertex_ss.empty());
        const auto expected=planar_DAT(gamma,query_radius,scene.mesh,scene.update,scene.positions);
        const auto actual=planar_DAT(gamma,query_radius,scene.mesh,scene.update,
            scene.positions,1.2,&phase,&workspace);
        for(std::size_t i=0;i<actual.size();++i)
            EXPECT_EQ(std::memcmp(actual[i].data(),expected[i].data(),3*sizeof(double)),0);
        scene.positions[0]+=actual[0]*.2;
    }
}

struct OpenMPRestore {
    int threads=omp_get_max_threads(),dynamic=omp_get_dynamic();
    ~OpenMPRestore(){omp_set_num_threads(threads);omp_set_dynamic(dynamic);}
};
struct ContactCloud {
    RefMesh mesh;
    DeformedState state;
    SimParams params=SimParams::zeros();
    VertexTriangleMap adjacency;
    std::vector<Pin> pins;
    explicit ContactCloud(RigidBodyUpdateMode mode,double friction) {
        create_rigid_body({Vec3(-4,0,-4),Vec3(4,0,-4),Vec3(0,0,4)},
            Vec3::Zero(),Vec4(1,0,0,0),Vec3::Zero(),3,mesh,state,RigidBodyUpdateMode::None);
        mesh.tris={0,2,1};
        std::vector<Vec3> points;
        for(int j=0;j<24;++j)for(int i=0;i<24;++i)
            points.emplace_back(-.4+.8*i/23.,.02+.003*std::sin(i+j),-.4+.8*j/23.);
        create_rigid_body(points,Vec3(.2,-.05,.1),Vec4(1,0,0,0),Vec3(.05,.07,.02),
            double(points.size()),mesh,state,mode);
        mesh.initialize(state.deformed_positions);mesh.build_deformable_nodes();
        adjacency=build_incident_triangle_map(mesh.tris);
        params.fps=100;params.substeps=1;params.fixed_iters=true;params.max_global_iters=3;
        params.damping=.5;params.d_hat=.05;params.k_barrier=30;
        params.dat_query_radius=.1;params.use_jacobi_dat=true;params.friction_coefficient=friction;
    }
};
struct Result {
    std::vector<Vec3> positions,centers,omega;
    std::vector<Vec4> orientations;
    SolverResult statistics;
};
Result solve(ContactCloud& scene,int threads,bool parallel) {
    omp_set_num_threads(threads);
    SimParams params=scene.params;params.use_parallel=parallel;
    Result out{scene.state.deformed_positions,scene.state.x_coms,
        std::vector<Vec3>(scene.state.omega.size(),Vec3::Zero()),scene.state.orientations};
    BroadPhase broad;
    const auto result=global_jacobi_solver_dat_general(scene.mesh,scene.state,scene.adjacency,
        scene.pins,params,out.positions,scene.state.deformed_positions,out.centers,
        out.orientations,out.omega,broad);
    if(params.fixed_iters)EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations,params.max_global_iters);
    out.statistics=result;
    return out;
}
template<class Vector>
void expect_same(const std::vector<Vector>& actual,const std::vector<Vector>& expected) {
    ASSERT_EQ(actual.size(),expected.size());
    for(std::size_t i=0;i<actual.size();++i) {
        ASSERT_TRUE(actual[i].allFinite())<<i;
        EXPECT_EQ(std::memcmp(actual[i].data(),expected[i].data(),Vector::SizeAtCompileTime*sizeof(double)),0)<<i;
    }
}
void expect_same(const Result& actual,const Result& expected) {
    expect_same(actual.positions,expected.positions);expect_same(actual.centers,expected.centers);
    expect_same(actual.omega,expected.omega);expect_same(actual.orientations,expected.orientations);
}
} // namespace

TEST(DATParallel, HeavySingleBodyContactsPreserveSerialArithmeticAcrossThreadsAndModes) {
    OpenMPRestore restore;omp_set_dynamic(0);
    for(auto mode:{RigidBodyUpdateMode::TranslationAndOrientation,
                   RigidBodyUpdateMode::TranslationOnly,RigidBodyUpdateMode::OrientationOnly})
        for(double friction:{0.,.3}) {
            SCOPED_TRACE(::testing::Message()<<"mode="<<int(mode)<<" friction="<<friction);
            ContactCloud scene(mode,friction);
            const auto original=scene.state.deformed_positions;
            const auto serial=solve(scene,1,false);
            for(int threads:{2,4,8}) {
                expect_same(solve(scene,threads,true),serial);
                expect_same(scene.state.deformed_positions,original);
            }
        }
}

TEST(DATParallel, CooperativeFailureJoinsAndAllowsTheNextSolve) {
    OpenMPRestore restore;omp_set_dynamic(0);
    ContactCloud scene(RigidBodyUpdateMode::TranslationAndOrientation,.3);
    const auto original=scene.state.deformed_positions;
    scene.params.k_barrier=std::numeric_limits<double>::max();
    EXPECT_THROW((void)solve(scene,8,true),std::exception);
    expect_same(scene.state.deformed_positions,original);
    scene.params.k_barrier=30;
    const auto serial=solve(scene,1,false);
    expect_same(solve(scene,8,true),serial);
}

TEST(DATParallel, HeavyRigidInitialResidualKeepsOrderedArithmeticAcrossThreads) {
    OpenMPRestore restore;omp_set_dynamic(0);
    for(double friction:{0.,.3}) {
        ContactCloud scene(RigidBodyUpdateMode::TranslationAndOrientation,friction);
        scene.params.fixed_iters=false;scene.params.max_global_iters=0;
        const auto serial=solve(scene,1,false);
        ASSERT_TRUE(serial.statistics.has_residual);
        ASSERT_GT(serial.statistics.final_rigid_residual,0);
        for(int threads:{2,4,8}) {
            const auto parallel=solve(scene,threads,true);
            expect_same(parallel,serial);
            EXPECT_EQ(std::memcmp(&parallel.statistics.initial_residual,
                &serial.statistics.initial_residual,sizeof(double)),0);
            EXPECT_EQ(std::memcmp(&parallel.statistics.final_rigid_residual,
                &serial.statistics.final_rigid_residual,sizeof(double)),0);
        }
    }
}

TEST(DATParallel, EndpointBacktrackingKeepsRepresentedParticleStepsInsideTheCap) {
    OpenMPRestore restore;omp_set_dynamic(0);
    RefMesh mesh;DeformedState state;
    create_rigid_body({Vec3(-.125,-.125,-.125),Vec3(.125,.125,-.125),
        Vec3(.125,-.125,.125),Vec3(-.125,.125,.125)},Vec3::Zero(),
        Vec4(1,0,0,0),Vec3::Zero(),1,mesh,state,RigidBodyUpdateMode::None);
    auto positions=state.deformed_positions;
    for(int i=0;i<300;++i)positions.emplace_back(1e12+4*i,0,0);
    mesh.num_positions=positions.size();mesh.node_to_rb.resize(positions.size(),-1);
    std::vector<Vec3> updates(positions.size(),Vec3(.3,0,0));
    for(int i=0;i<4;++i)updates[i].setZero();
    const std::vector<Vec3> zeros(1,Vec3::Zero());
    const auto run=[&](int threads) {
        omp_set_num_threads(threads);
        return general_planar_DAT(.9,2./3,mesh,updates,positions,state.x_coms,
            zeros,state.orientations,zeros,zeros,.01);
    };
    ASSERT_GT((positions[4].x()+.3)-positions[4].x(),.3);
    const auto serial=run(1),parallel=run(8);
    expect_same(parallel.positions,serial.positions);
    expect_same(parallel.centers,serial.centers);
    expect_same(parallel.omega,serial.omega);
    for(std::size_t i=4;i<positions.size();++i) {
        const double displacement=parallel.positions[i].x()-positions[i].x();
        EXPECT_GT(displacement,0);
        EXPECT_LT(displacement,.25); // A real shortened retry, not a zero update.
        EXPECT_LE(displacement,.3);
    }
    expect_same(state.deformed_positions,std::vector<Vec3>(positions.begin(),positions.begin()+4));
}
