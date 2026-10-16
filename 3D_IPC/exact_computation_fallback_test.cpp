#include "ipc_args.h"
#include "ccd.h"
#include "SIMD.h"
#include "broad_phase.h"
#include "safe_step.h"
#include <gtest/gtest.h>

static bool parse_flags(IPCArgs3D& args, std::vector<std::string> values) {
    std::vector<char*> argv;
    for (auto& value : values) argv.push_back(value.data());
    return args.parse(static_cast<int>(argv.size()), argv.data());
}

TEST(LinearCCDVersions, FallbackDefaultsToModifiedAndOffSelectsOriginal) {
    IPCArgs3D args;
    ASSERT_TRUE(parse_flags(args, {"3D_sim"}));
    EXPECT_TRUE(args.exact_computation_fallback);
    EXPECT_FALSE(args.to_sim_params().use_original_linear_ccd());
    EXPECT_FALSE(args.to_sim_params().use_ticcd);
    IPCArgs3D v1;
    ASSERT_TRUE(parse_flags(v1, {"3D_sim", "--exact_computation_fallback", "false"}));
    EXPECT_TRUE(v1.to_sim_params().use_original_linear_ccd());
}

TEST(LinearCCDVersions, TightInclusionRemainsIndependentOfFallbackFlag) {
    IPCArgs3D args;
    ASSERT_TRUE(parse_flags(args, {"3D_sim", "--use_ticcd", "true",
                                  "--exact_computation_fallback", "true"}));
    EXPECT_TRUE(args.to_sim_params().use_ticcd);
    EXPECT_FALSE(args.to_sim_params().use_original_linear_ccd());
    IPCArgs3D conflict;
    EXPECT_TRUE(parse_flags(conflict, {"3D_sim", "--use_ticcd", "true",
                                       "--exact_computation_fallback", "false"}));
    EXPECT_TRUE(conflict.to_sim_params().use_ticcd);
    EXPECT_FALSE(conflict.to_sim_params().use_original_linear_ccd());
}

TEST(LinearCCDVersions, OriginalVersionRetainsItsNearContactBehavior) {
    const Vec3 a(0,0,0), b(1,0,0), c(0,1,0), zero=Vec3::Zero();
    const Vec3 x(.2,.2,1e-12), away(0,0,1);
    const auto v1=node_triangle_linear_ccd_v1(x,away,a,zero,b,zero,c,zero);
    const auto v2=node_triangle_only_one_node_moves(x,away,a,zero,b,zero,c,zero,1e-12,false);
    EXPECT_TRUE(v1.collision);
    EXPECT_DOUBLE_EQ(v1.t,0.0);
    EXPECT_FALSE(v2.collision);
}

TEST(LinearCCDVersions, BothVersionsResolveAnOrdinaryCrossing) {
    const Vec3 a(0,0,0), b(1,0,0), c(0,1,0), zero=Vec3::Zero();
    const Vec3 x(.2,.2,1), down(0,0,-2);
    const auto v1=node_triangle_linear_ccd_v1(x,down,a,zero,b,zero,c,zero);
    const auto v2=node_triangle_only_one_node_moves(x,down,a,zero,b,zero,c,zero,1e-12,false);
    ASSERT_TRUE(v1.collision);
    ASSERT_TRUE(v2.collision);
    EXPECT_NEAR(v1.t,.5,1e-12);
    EXPECT_DOUBLE_EQ(v1.t,v2.t);
}

TEST(LinearCCDVersions, TightInclusionQueryIgnoresLinearVersion) {
    const Vec3 a(0,0,0), b(1,0,0), c(0,1,0), zero=Vec3::Zero();
    const Vec3 x(.2,.2,1), down(0,0,-2);
    const auto first=node_triangle_only_one_node_moves(x,down,a,zero,b,zero,c,zero,1e-12,true,true);
    const auto second=node_triangle_only_one_node_moves(x,down,a,zero,b,zero,c,zero,1e-12,true,false);
    ASSERT_TRUE(first.collision);
    EXPECT_EQ(first.collision,second.collision);
    EXPECT_DOUBLE_EQ(first.t,second.t);
}

TEST(ExactComputationFallback, DefaultsOnAndPreservesColoredGuessSettings) {
    IPCArgs3D defaults;
    ASSERT_TRUE(parse_flags(defaults, {"3D_sim"}));
    EXPECT_TRUE(defaults.to_sim_params().exact_computation_fallback);
    IPCArgs3D original;
    ASSERT_TRUE(parse_flags(original, {"3D_sim", "--exact_computation_fallback", "false", "--use_colored_ccd_guess", "true",
        "--colored_ccd_guess_iters", "7"}));
    const auto params = original.to_sim_params();
    EXPECT_FALSE(params.exact_computation_fallback);
    EXPECT_TRUE(params.use_original_linear_ccd());
    EXPECT_TRUE(params.use_colored_ccd_guess);
    EXPECT_EQ(params.colored_ccd_guess_iters, 7);
    IPCArgs3D modified;
    ASSERT_TRUE(parse_flags(modified, {"3D_sim", "--exact_computation_fallback", "true"}));
    EXPECT_FALSE(modified.to_sim_params().use_original_linear_ccd());
}

TEST(ExactComputationFallback, OffRestoresOriginalNearParallelFeatureSelection) {
    const Vec3 a(0,0,0), b(1,0,0), c(0,-5e-9,.125), d(1,5e-9,.125);
    const auto original = segment_segment_distance(a,b,c,d,1e-12,false);
    const auto exact = segment_segment_distance(a,b,c,d);
    EXPECT_FALSE(original.robust);
    EXPECT_EQ(original.region, SegmentSegmentRegion::ParallelSegments);
    EXPECT_DOUBLE_EQ(original.s, 0.0);
    EXPECT_TRUE((original.separation.array()
        == (original.closest_point_1-original.closest_point_2).array()).all());
    EXPECT_TRUE(exact.robust);
    EXPECT_EQ(exact.region, SegmentSegmentRegion::Interior);
    EXPECT_DOUBLE_EQ(exact.s, .5);
    EXPECT_DOUBLE_EQ(exact.t, .5);
}

TEST(ExactComputationFallback, BothModesKeepTinyRepresentableVertexMoves) {
    RefMesh mesh;
    mesh.num_positions=1;
    mesh.mass={1.0};
    AABB box;
    box.min=Vec3(-1,-1,-1);
    box.max=Vec3(1,1,1);
    BroadPhase broad_phase;
    broad_phase.initialize({box},mesh,0.0);
    for (bool fallback:{false,true}) {
        std::vector<Vec3> x{Vec3::Zero()};
        const Vec3 target(1e-15,0,0);
        const double step=per_vertex_safe_step(broad_phase,x,0,target,.9,true,false,
                                              false,false,nullptr,fallback);
        EXPECT_DOUBLE_EQ(step,1.0);
        EXPECT_DOUBLE_EQ(x[0].x(),target.x());
    }
}

TEST(ExactComputationFallback, OGCTrustRadiusUsesTheSelectedDistancePolicy) {
    // The legacy absolute determinant threshold selects a boundary feature
    // for these short crossing edges; exact fallback resolves the interior gap.
    constexpr double length=1e-7, gap=1e-9;
    const std::vector<Vec3> initial{
        Vec3(-length/2,0,0),Vec3(length/2,0,0),
        Vec3(0,-length/2,gap),Vec3(0,length/2,gap)};
    BroadPhase phase;
    auto& cache=phase.mutable_cache();
    cache.node_boxes.assign(4,AABB(Vec3::Constant(-1),Vec3::Constant(1)));
    cache.vertex_nt.resize(4);
    cache.vertex_ss.resize(4);
    SegmentSegmentPair pair{};
    for(int i=0;i<4;++i)pair.v[i]=i;
    cache.ss_pairs.push_back(pair);
    cache.vertex_ss[0].push_back({0,0});
    for(bool fallback:{false,true}) {
        const double expected=.4*(fallback ? gap : std::hypot(length/2,gap));
        EXPECT_NEAR(compute_trust_region_bound_for_vertex(0,initial,phase,.4,fallback),expected,1e-20);
        for(bool ticcd:{false,true}) {
            auto x=initial;
            const Vec3 target=x[0]+Vec3(0,0,length);
            const double step=per_vertex_safe_step(phase,x,0,target,.9,false,ticcd,
                                                  true,false,nullptr,fallback);
            EXPECT_NEAR(step,expected/length,1e-13);
            EXPECT_NEAR(x[0].z(),expected,1e-20);
        }
    }
}

TEST(ExactComputationFallback, ScalarAndSIMDUseMatchingOriginalContactDerivatives) {
    const std::array<Vec3,4> x{Vec3(0,0,0),Vec3(1,0,0),
        Vec3(0,-5e-9,.125),Vec3(1,5e-9,.125)};
    const auto original = make_segment_segment_contact_evaluation(x,.5,100,1e-12,nullptr,false);
    const auto exact = make_segment_segment_contact_evaluation(x,.5,100);
    ASSERT_FALSE(original.dr.robust);
    ASSERT_TRUE(exact.dr.robust);
    for (int role=0;role<4;++role) {
        ipc_simd::MeshContactInput input;
        input.positions=x;
        input.previous_positions=x;
        input.role=role;
        input.segment_segment=true;
        input.exact_computation_fallback=false;
        ipc_simd::MeshContactOutput result;
        ipc_simd::mesh_contact_derivatives_tile(&input,1,.5,100,0,.01,.01,&result);
        const auto scalar=segment_segment_barrier_self_gradient_and_hessian(
            x[0],x[1],x[2],x[3],role,original);
        const auto direct=segment_segment_barrier_self_gradient_and_hessian(
            x[0],x[1],x[2],x[3],.5,role,1e-12,nullptr,nullptr,nullptr,false);
        EXPECT_LE((result.gradient-scalar.first).norm(),1e-12*(1+scalar.first.norm()));
        EXPECT_LE((result.hessian-scalar.second).norm(),1e-12*(1+scalar.second.norm()));
        EXPECT_TRUE((direct.first.array()==scalar.first.array()).all());
        EXPECT_TRUE((direct.second.array()==scalar.second.array()).all());
    }
    const auto old_gradient=segment_segment_barrier_gradient(x[0],x[1],x[2],x[3],1,original);
    const auto new_gradient=segment_segment_barrier_gradient(x[0],x[1],x[2],x[3],1,exact);
    EXPECT_DOUBLE_EQ(old_gradient.norm(),0.0);
    EXPECT_GT(new_gradient.norm(),0.0);
}
