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
