#include "node_triangle_distance.h"

#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <string>

namespace {

    constexpr double kTol = 1e-10;

    bool approx(double a, double b, double tol = kTol){
        return std::abs(a - b) <= tol;
    }

    bool approx_vec(const Vec3& a, const Vec3& b, double tol = kTol){
        return (a - b).norm() <= tol;
    }

    void print_result(const std::string& name, const NodeTriangleDistanceResult& r){
        std::cout << name << "\n";
        std::cout << "  region   = " << to_string(r.region) << "\n";
        std::cout << "  phi      = " << r.phi << "\n";
        std::cout << "  distance = " << r.distance << "\n";
        std::cout << "  tilde_x  = " << r.tilde_x.transpose() << "\n";
        std::cout << "  closest  = " << r.closest_point.transpose() << "\n";
        std::cout << "  bary     = ["
                  << r.barycentric_tilde_x[0] << ", "
                  << r.barycentric_tilde_x[1] << ", "
                  << r.barycentric_tilde_x[2] << "]\n";
    }

} // namespace

TEST(NodeTriangleDistance, FaceCase){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);
    const Vec3 x(0.25, 0.25, 2.0);

    const auto r = node_triangle_distance(x, x1, x2, x3);
    print_result("Face case", r);

    EXPECT_EQ(r.region, NodeTriangleRegion::FaceInterior) << "Face case: wrong region";
    EXPECT_TRUE(approx(r.phi, 2.0)) << "Face case: wrong phi";
    EXPECT_TRUE(approx(r.distance, 2.0)) << "Face case: wrong distance";
    EXPECT_TRUE(approx_vec(r.tilde_x, Vec3(0.25, 0.25, 0.0))) << "Face case: wrong tilde_x";
    EXPECT_TRUE(approx_vec(r.closest_point, Vec3(0.25, 0.25, 0.0))) << "Face case: wrong closest_point";
}

TEST(NodeTriangleDistance, Edge12Case){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);
    const Vec3 x(0.5, -0.2, 1.0);

    const auto r = node_triangle_distance(x, x1, x2, x3);
    print_result("Edge12 case", r);

    EXPECT_EQ(r.region, NodeTriangleRegion::Edge12) << "Edge12 case: wrong region";
    EXPECT_TRUE(approx_vec(r.closest_point, Vec3(0.5, 0.0, 0.0))) << "Edge12 case: wrong closest_point";
}

TEST(NodeTriangleDistance, Edge23Case){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);
    const Vec3 x(0.8, 0.8, 1.5);

    const auto r = node_triangle_distance(x, x1, x2, x3);
    print_result("Edge23 case", r);

    EXPECT_EQ(r.region, NodeTriangleRegion::Edge23) << "Edge23 case: wrong region";
    EXPECT_TRUE(approx_vec(r.closest_point, Vec3(0.5, 0.5, 0.0))) << "Edge23 case: wrong closest_point";
}

TEST(NodeTriangleDistance, Edge31Case){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);
    const Vec3 x(-0.2, 0.5, 0.7);

    const auto r = node_triangle_distance(x, x1, x2, x3);
    print_result("Edge31 case", r);

    EXPECT_EQ(r.region, NodeTriangleRegion::Edge31) << "Edge31 case: wrong region";
    EXPECT_TRUE(approx_vec(r.closest_point, Vec3(0.0, 0.5, 0.0))) << "Edge31 case: wrong closest_point";
}

TEST(NodeTriangleDistance, Vertex1Case){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);
    const Vec3 x(-0.3, -0.4, 0.2);

    const auto r = node_triangle_distance(x, x1, x2, x3);
    print_result("Vertex1 case", r);

    EXPECT_EQ(r.region, NodeTriangleRegion::Vertex1) << "Vertex1 case: wrong region";
    EXPECT_TRUE(approx_vec(r.closest_point, x1)) << "Vertex1 case: wrong closest_point";
}

TEST(NodeTriangleDistance, Vertex2Case){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);
    const Vec3 x(1.4, -0.2, 0.6);

    const auto r = node_triangle_distance(x, x1, x2, x3);
    print_result("Vertex2 case", r);

    EXPECT_EQ(r.region, NodeTriangleRegion::Vertex2) << "Vertex2 case: wrong region";
    EXPECT_TRUE(approx_vec(r.closest_point, x2)) << "Vertex2 case: wrong closest_point";
}

TEST(NodeTriangleDistance, Vertex3Case){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);
    const Vec3 x(-0.2, 1.3, 0.4);

    const auto r = node_triangle_distance(x, x1, x2, x3);
    print_result("Vertex3 case", r);

    EXPECT_EQ(r.region, NodeTriangleRegion::Vertex3) << "Vertex3 case: wrong region";
    EXPECT_TRUE(approx_vec(r.closest_point, x3)) << "Vertex3 case: wrong closest_point";
}

TEST(NodeTriangleDistance, SignedDistance){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(0.0, 1.0, 0.0);

    const auto r_plus  = node_triangle_distance(Vec3(0.2, 0.2,  3.0), x1, x2, x3);
    const auto r_minus = node_triangle_distance(Vec3(0.2, 0.2, -3.0), x1, x2, x3);

    print_result("Signed distance positive side", r_plus);
    print_result("Signed distance negative side", r_minus);

    EXPECT_TRUE(approx(r_plus.phi,  3.0)) << "Signed distance: wrong positive phi";
    EXPECT_TRUE(approx(r_minus.phi, -3.0)) << "Signed distance: wrong negative phi";
    EXPECT_TRUE(approx(r_plus.distance, 3.0)) << "Signed distance: wrong positive distance";
    EXPECT_TRUE(approx(r_minus.distance, 3.0)) << "Signed distance: wrong negative distance";
}

TEST(NodeTriangleDistance, DegenerateTriangle){
    const Vec3 x1(0.0, 0.0, 0.0);
    const Vec3 x2(1.0, 0.0, 0.0);
    const Vec3 x3(2.0, 0.0, 0.0);
    const Vec3 x(0.4, 1.0, 0.0);

    const auto r = node_triangle_distance(x, x1, x2, x3);
    print_result("Degenerate triangle case", r);

    EXPECT_EQ(r.region, NodeTriangleRegion::DegenerateTriangle) << "Degenerate case: wrong region";
    EXPECT_TRUE(approx_vec(r.closest_point, Vec3(0.4, 0.0, 0.0))) << "Degenerate case: wrong closest_point";
    EXPECT_TRUE(approx(r.distance, 1.0)) << "Degenerate case: wrong distance";
}

#include <random>
#include <algorithm>
#include <Eigen/Geometry>

TEST(NodeTriangleDistance, ObtuseTriangleUsesActualNearestEdge) {
    const auto r = node_triangle_distance(Vec3(.5,-.01,.1), Vec3::Zero(),
        Vec3(1,0,0), Vec3(-10,.1,0));
    EXPECT_EQ(r.region, NodeTriangleRegion::Edge12);
    EXPECT_LE((r.closest_point-Vec3(.5,0,0)).norm(), 1e-14);
    EXPECT_NEAR(r.distance, std::sqrt(.0101), 1e-14);
}

TEST(NodeTriangleDistance, ClampedBoundaryPointUsesVertexFeature) {
    const auto r = node_triangle_distance(Vec3(1.4,.1,.6), Vec3::Zero(),
        Vec3(1,0,0), Vec3(0,1,0));
    EXPECT_EQ(r.region, NodeTriangleRegion::Vertex2);
    EXPECT_LE((r.closest_point-Vec3(1,0,0)).norm(), 1e-14);
}

TEST(NodeTriangleDistance, SmallTriangleOutsideIsNotAnInfinitePlane) {
    const auto r = node_triangle_distance(Vec3(2e-4,2e-5,3e-5), Vec3::Zero(),
        Vec3(1e-4,0,0), Vec3(0,1e-4,0));
    EXPECT_EQ(r.region, NodeTriangleRegion::Vertex2);
    EXPECT_LE((r.closest_point-Vec3(1e-4,0,0)).norm(), 1e-16);
    EXPECT_NEAR(r.distance, std::sqrt(1e-8+4e-10+9e-10), 1e-16);
}

TEST(NodeTriangleDistance, SmallTriangleInteriorHasNonzeroBarycentrics) {
    const auto r = node_triangle_distance(Vec3(2e-5,3e-5,4e-5), Vec3::Zero(),
        Vec3(1e-4,0,0), Vec3(0,1e-4,0));
    EXPECT_EQ(r.region, NodeTriangleRegion::FaceInterior);
    EXPECT_NEAR(r.barycentric_tilde_x[0], .5, 1e-14);
    EXPECT_NEAR(r.barycentric_tilde_x[1], .2, 1e-14);
    EXPECT_NEAR(r.barycentric_tilde_x[2], .3, 1e-14);
    EXPECT_NEAR(r.distance, 4e-5, 1e-16);
}

TEST(NodeTriangleDistance, ObtuseFeatureSurvivesPermutationsRotationAndScale) {
    const Mat33 rotation = Eigen::AngleAxisd(.73, Vec3(1,2,3).normalized()).toRotationMatrix();
    for (double scale : {1.0, .001, .0001}) {
        const auto tr = [&](const Vec3& x) -> Vec3 { return scale*(rotation*x+Vec3(2,-3,4)); };
        const std::array<Vec3,3> tri{{tr(Vec3::Zero()),tr(Vec3(1,0,0)),tr(Vec3(-10,.1,0))}};
        std::array<int,3> order{{0,1,2}};
        do {
            const auto r=node_triangle_distance(tr(Vec3(.5,-.01,.1)),tri[order[0]],tri[order[1]],tri[order[2]]);
            EXPECT_LE((r.closest_point-tr(Vec3(.5,0,0))).norm(), 2e-12*scale);
            EXPECT_NEAR(r.distance, scale*std::sqrt(.0101), 2e-12*scale);
        } while(std::next_permutation(order.begin(),order.end()));
    }
}

namespace {
// Independent dot-product Voronoi reference, evaluated in long double.
using LVec = Eigen::Matrix<long double,3,1>;
LVec voronoi_reference(const LVec& p,const LVec& a,const LVec& b,const LVec& c) {
    const LVec ab=b-a,ac=c-a,ap=p-a;
    const long double d1=ab.dot(ap),d2=ac.dot(ap);
    if(d1<=0&&d2<=0)return a;
    const LVec bp=p-b;
    const long double d3=ab.dot(bp),d4=ac.dot(bp);
    if(d3>=0&&d4<=d3)return b;
    const long double vc=d1*d4-d3*d2;
    if(vc<=0&&d1>=0&&d3<=0)return a+(d1/(d1-d3))*ab;
    const LVec cp=p-c;
    const long double d5=ab.dot(cp),d6=ac.dot(cp);
    if(d6>=0&&d5<=d6)return c;
    const long double vb=d5*d2-d1*d6;
    if(vb<=0&&d2>=0&&d6<=0)return a+(d2/(d2-d6))*ac;
    const long double va=d3*d6-d5*d4;
    if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0)return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b);
    const long double inv=1/(va+vb+vc);
    return a+(vb*inv)*ab+(vc*inv)*ac;
}
}

TEST(NodeTriangleDistance, RandomTrianglesAgreeWithIndependentVoronoiReference) {
    std::mt19937_64 rng(20261027);
    std::uniform_real_distribution<double> uniform(-2,2);
    for(int i=0;i<2000;++i){
        const auto sample=[&](){return Vec3(uniform(rng),uniform(rng),uniform(rng));};
        const Vec3 p=sample(),a=sample(),b=sample(),c=sample();
        if((b-a).cross(c-a).norm()<.001){--i;continue;}
        const LVec q=voronoi_reference(p.cast<long double>(),a.cast<long double>(),b.cast<long double>(),c.cast<long double>());
        const auto actual=node_triangle_distance(p,a,b,c);
        EXPECT_LE((actual.closest_point-q.cast<double>()).norm(),3e-12)<<i;
        EXPECT_NEAR(actual.distance,(p.cast<long double>()-q).norm(),3e-12)<<i;
    }
}
