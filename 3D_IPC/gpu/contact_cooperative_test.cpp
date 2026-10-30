// Standalone, no Eigen/CPU solver dependency. Build host with:
// c++ -std=c++17 -O2 -ffp-contract=off -I gpu \
//   gpu/contact_cooperative_test.cpp -o /tmp/contact_cooperative_test
// CUDA companion includes this file and also checks real cooperative launches.
#include "contact_cooperative.cuh"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ipc_gpu::solver;
namespace cooperative_test {
namespace coop = contact_cooperative_detail;
std::size_t comparisons = 0;
void require(bool okay, const std::string& label) {
    if (!okay) throw std::runtime_error(label);
}
void same(double a, double b, const std::string& label) {
    std::uint64_t x, y; std::memcpy(&x, &a, 8); std::memcpy(&y, &b, 8);
    require(x == y, label + " bits " + std::to_string(x) + " != " + std::to_string(y));
    ++comparisons;
}
void compare(const ContactOutput& a, const ContactOutput& b, const std::string& label) {
    require(a.status == b.status && a.active == b.active, label + " status/activity");
    same(a.distance, b.distance, label + " distance");
    for (int i = 0; i < 4; ++i) {
        same(a.weights[i], b.weights[i], label + " weight");
        for (int k = 0; k < 3; ++k) {
            same(a.barrier_gradient[i][k], b.barrier_gradient[i][k], label + " gradient");
            for (int j = 0; j < 4; ++j) for (int l = 0; l < 3; ++l)
                same(a.barrier_hessian[i][j][3*k+l], b.barrier_hessian[i][j][3*k+l], label + " Hessian");
        }
    }
    for (int k = 0; k < 3; ++k) {
        same(a.normal[k], b.normal[k], label + " normal");
        same(a.friction_relative_gradient[k], b.friction_relative_gradient[k], label + " friction gradient");
    }
    for (int k = 0; k < 9; ++k)
        same(a.friction_relative_hessian[k], b.friction_relative_hessian[k], label + " friction Hessian");
}
using Point = std::array<double, 3>;
ContactInput make(ContactKind kind, const std::array<Point, 4>& points, double friction = .1) {
    ContactInput in; in.kind = kind; in.d_hat = .8; in.dt = .031;
    in.k_barrier = 370; in.eps_v = .1; in.friction = friction;
    for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) {
        in.positions[i][k] = points[i][k];
        in.previous[i][k] = points[i][k] - .001*(i+1)*(k+1);
    }
    return in;
}
std::vector<ContactInput> fixtures() {
    std::vector<ContactInput> geometry;
    const std::vector<Point> points{{.2,.3,.17}, {.2,.3,-.17}, {.3,-.2,.17},
        {.7,.7,.17}, {-.2,.3,.17}, {-.2,-.3,.17}, {1.3,-.1,.17},
        {-.1,1.3,.17}, {.2,.3,1.1}, {0,0,.125}, {1,0,.125}};
    for (const auto& point : points)
        geometry.push_back(make(ContactKind::PointTriangle, {point, Point{0,0,0}, Point{1,0,0}, Point{0,1,0}}));
    geometry.push_back(make(ContactKind::PointTriangle, {Point{.5,.125,0}, Point{0,0,0}, Point{1,0,0}, Point{2,0,0}}));
    // CPU eps-degenerate contact semantics must also survive cooperative
    // exact preparation and its seven independent conversion lanes.
    for(double height:{1e-13,1e-12})
        geometry.push_back(make(ContactKind::PointTriangle,
            {Point{.25,2e-14,.1},Point{0,0,0},Point{1,0,0},Point{0,height,0}}));
    for(double px:{5e-15,2e-14})
        geometry.push_back(make(ContactKind::PointTriangle,
            {Point{px,.2,.1},Point{0,0,0},Point{1,0,0},Point{2,0,0}}));
    geometry.push_back(make(ContactKind::PointTriangle,
        {Point{0,0,.1},Point{0,0,0},Point{0,0,0},Point{0,0,0}}));
    for (double x : {-.6,.2,1.6}) for (double y : {-.6,.2,1.6})
        geometry.push_back(make(ContactKind::EdgeEdge, {Point{0,0,0}, Point{1,0,0}, Point{x,y,.17}, Point{x,y+1,.17}}));
    geometry.push_back(make(ContactKind::EdgeEdge, {Point{0,0,0}, Point{1,0,0}, Point{.25,.125,0}, Point{.75,.125,0}}));
    geometry.push_back(make(ContactKind::EdgeEdge, {Point{0,0,0}, Point{0,0,0}, Point{.25,.125,0}, Point{.75,.125,0}}));
    geometry.push_back(make(ContactKind::EdgeEdge, {Point{0,0,0}, Point{0,0,0}, Point{.25,.125,0}, Point{.25,.125,0}}));
    for (int exponent : {-30,-50,-500,-1074}) {
        const double gap = std::ldexp(1., exponent);
        auto in = make(ContactKind::PointTriangle, {Point{.25,.25,gap}, Point{0,0,0}, Point{1,0,0}, Point{0,1,0}}, 0);
        geometry.push_back(in);
    }
    auto signed_zero = geometry.front();
    signed_zero.positions[1][0] = -0.0; signed_zero.positions[1][1] = -0.0;
    signed_zero.previous[1][0] = -0.0; signed_zero.previous[1][1] = 0.0;
    geometry.push_back(signed_zero);
    std::mt19937_64 random(0x6c414e45534ULL);
    for (int sample = 0; sample < 128; ++sample) {
        std::array<Point, 4> p;
        for (auto& point : p) for (double& coordinate : point)
            coordinate = (int(random()%2049)-1024)/512.;
        geometry.push_back(make(sample%2 ? ContactKind::PointTriangle : ContactKind::EdgeEdge,
            p, sample%3 ? .1 : 0));
    }
    // Include invalid, zero-gap, overflow/nonfinite derivative, zero barrier,
    // and extreme-coordinate capacity paths. The original evaluator remains
    // authoritative for all unsuccessful cooperative evaluations.
    auto invalid = geometry[0]; invalid.positions[0][0] = std::numeric_limits<double>::infinity(); geometry.push_back(invalid);
    invalid = geometry[0]; invalid.previous[0][0] = std::numeric_limits<double>::quiet_NaN(); geometry.push_back(invalid);
    invalid = geometry[0]; invalid.dt = 0; geometry.push_back(invalid);
    invalid = geometry[0]; invalid.k_barrier = 0; geometry.push_back(invalid);
    invalid = geometry[0]; invalid.positions[0][2] = 0; geometry.push_back(invalid);
    invalid = geometry[0]; invalid.positions[0][0] = std::ldexp(1., 1000); invalid.positions[0][2] = std::ldexp(1., -1000); geometry.push_back(invalid);
    std::vector<ContactInput> records;
    for (const auto& in : geometry) for (unsigned mask : {1u,1u<<5,1u<<10,1u<<15}) {
        records.push_back(in); records.back().hessian_blocks = mask;
    }
    for (std::size_t i = 0; i < 12; ++i) for (unsigned mask : {0u,0xffffu,0x23u,0x8ce0u,0x10000u}) {
        records.push_back(geometry[i]); records.back().hessian_blocks = mask;
    }
    return records;
}
template<int C> ContactOutput evaluate_portable(const ContactInput& in, unsigned* prepared_count = nullptr,
    bool parallel_conversions = false) {
    const int role = coop::self_block(in.hessian_blocks);
    if (role < 0) return evaluate_contact_exact<C>(in);
    coop::GeometryOwner<C> owner;
    coop::SharedGeometry shared{}; ContactOutput out;
    bool prepared;
    if (parallel_conversions) {
        prepared = coop::prepare_feature(in, owner, out);
        if (prepared) {
            coop::SharedConversions conversions;
            coop::publish_conversions(owner, conversions);
            bool okay = true;
            // Reverse visitation proves that independent conversions have no
            // ordering dependency, including signed-zero and failure cases.
            for (int component = 6; component >= 0; --component) {
                const auto result = coop::convert_geometry_component<C>(conversions, component);
                conversions.values[component] = result.value;
                okay = okay && result.ok;
            }
            if (!okay) return evaluate_contact_exact<C>(in);
            for (int k = 0; k < 3; ++k) owner.residual[k] = conversions.values[k];
            for (int i = 0; i < 4; ++i) out.weights[i] = conversions.values[3+i];
            prepared = coop::prepare_derivatives(in, role, owner, out, shared);
        }
    } else prepared = coop::prepare_geometry(in, role, owner, out, shared);
    if (out.status != ContactStatus::Ok) return evaluate_contact_exact<C>(in);
    if (!prepared) return out;
    if (prepared_count) ++*prepared_count;
    for (int entry = 0; entry < 9; ++entry) {
        const auto result = coop::compute_entry<C>(shared, entry);
        if (!result.ok) return evaluate_contact_exact<C>(in);
        out.barrier_hessian[role][role][entry] = result.value;
    }
    coop::finish_output(in, owner, out);
    return out.status == ContactStatus::Ok ? out : evaluate_contact_exact<C>(in);
}
template<int C> void host_cases(const std::vector<ContactInput>& records) {
    for (bool parallel : {false, true}) {
        unsigned prepared = 0;
        for (std::size_t i = 0; i < records.size(); ++i)
            compare(evaluate_portable<C>(records[i], &prepared, parallel), evaluate_contact_exact<C>(records[i]),
                "host C="+std::to_string(C)+" parallel="+std::to_string(parallel)+" case="+std::to_string(i));
        if (C >= 64) require(prepared > 0, "cooperative branch must be exercised");
        std::cout << "host C=" << C << " parallel=" << parallel << " cases=" << records.size()
            << " cooperative=" << prepared << '\n';
    }
}
} // namespace cooperative_test

#if defined(IPC_CONTACT_COOPERATIVE_CUDA_TEST)
void cooperative_cuda_cases(const std::vector<ContactInput>&);
#endif

int main() {
    try {
        const auto records = cooperative_test::fixtures();
        cooperative_test::host_cases<2>(records);
        cooperative_test::host_cases<8>(records);
        cooperative_test::host_cases<64>(records);
        cooperative_test::host_cases<256>(records);
#if defined(IPC_CONTACT_COOPERATIVE_CUDA_TEST)
        cooperative_cuda_cases(records);
#endif
        std::cout << "PASS: " << cooperative_test::comparisons << " exact output components\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
