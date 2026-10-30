#include "broadphase.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ipc_gpu::solver;

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
static bool no_solver_data(const BroadPhaseView& view) {
    return !view.adjacency_offsets && !view.adjacency && !view.directed_edge_count
        && !view.block_colors && !view.color_blocks && !view.color_offsets && !view.color_count;
}
template<class T> struct Device {
    T* data = nullptr;
    std::size_t count;
    explicit Device(std::size_t n) : count(n) {
        if (n) check(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(T)));
    }
    explicit Device(const std::vector<T>& values) : Device(values.size()) {
        if (count) check(cudaMemcpy(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Device() { if (data) cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
template<class T> static std::vector<T> download(const T* pointer, std::size_t count) {
    std::vector<T> result(count);
    if (count) check(cudaMemcpy(result.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}
static BpAabb point_box(double x, double y, double z, double radius = 0.01) {
    return {{x - radius, y - radius, z - radius}, {x + radius, y + radius, z + radius}};
}
static BpAabb join(BpAabb a, const BpAabb& b) {
    for (int k = 0; k < 3; ++k) {
        a.lo[k] = std::min(a.lo[k], b.lo[k]); a.hi[k] = std::max(a.hi[k], b.hi[k]);
    }
    return a;
}
static bool intersects(BpAabb a, const BpAabb& b, double d_hat) {
    for (int k = 0; k < 3; ++k) {
        a.lo[k] = std::nextafter(a.lo[k] - d_hat, -std::numeric_limits<double>::infinity());
        a.hi[k] = std::nextafter(a.hi[k] + d_hat, std::numeric_limits<double>::infinity());
        if (a.lo[k] > b.hi[k] || a.hi[k] < b.lo[k]) return false;
    }
    return true;
}
using ContactKey = std::array<int, 5>;
static ContactKey key(const BpContact& c) {
    return {{c.kind, c.vertices[0], c.vertices[1], c.vertices[2], c.vertices[3]}};
}
static bool same_block(const BpContact& c, const std::vector<int>& owners) {
    for (int k = 1; k < 4; ++k) if (owners[c.vertices[k]] != owners[c.vertices[0]]) return false;
    return true;
}
static std::set<ContactKey> reference_contacts(const std::vector<BpAabb>& boxes,
    const std::vector<BpTriangle>& triangles, const std::vector<BpEdge>& edges,
    const std::vector<int>& owners, const std::vector<unsigned char>& surface, double d_hat) {
    std::set<ContactKey> result;
    for (int i = 0; i < static_cast<int>(boxes.size()); ++i) {
        if (!surface.empty() && !surface[i]) continue;
        for (const BpTriangle& t : triangles) {
            bool skip = false;
            for (int j = 0; j < 3; ++j) skip |= t.vertices[j] == i;
            if (skip) continue;
            const BpAabb box = join(join(boxes[t.vertices[0]], boxes[t.vertices[1]]), boxes[t.vertices[2]]);
            const BpContact c{{i, t.vertices[0], t.vertices[1], t.vertices[2]}, 0};
            if (intersects(boxes[i], box, d_hat) && !same_block(c, owners)) result.insert(key(c));
        }
    }
    for (std::size_t i = 0; i < edges.size(); ++i) for (std::size_t j = i + 1; j < edges.size(); ++j) {
        const BpEdge a = edges[i], b = edges[j];
        bool skip = false;
        for (int ia = 0; ia < 2; ++ia) for (int ib = 0; ib < 2; ++ib) skip |= a.vertices[ia] == b.vertices[ib];
        if (skip) continue;
        const BpContact c{{a.vertices[0], a.vertices[1], b.vertices[0], b.vertices[1]}, 1};
        if (intersects(join(boxes[a.vertices[0]], boxes[a.vertices[1]]),
            join(boxes[b.vertices[0]], boxes[b.vertices[1]]), d_hat) && !same_block(c, owners)) result.insert(key(c));
    }
    return result;
}
static std::set<std::pair<int,int>> reference_graph(const std::vector<BpTriangle>& triangles,
    const std::vector<BpDependency>& dependencies, const std::set<ContactKey>& contacts,
    const std::vector<int>& owners) {
    std::set<std::pair<int,int>> result;
    auto clique = [&](const int* vertices, int count) {
        for (int i = 0; i < count; ++i) for (int j = i + 1; j < count; ++j) {
            const int a = owners[vertices[i]], b = owners[vertices[j]];
            if (a != b) { result.emplace(a, b); result.emplace(b, a); }
        }
    };
    for (const auto& t : triangles) clique(t.vertices, 3);
    for (const auto& d : dependencies) clique(d.vertices, d.count);
    for (const auto& c : contacts) clique(c.data() + 1, 4);
    return result;
}

static void run_case(const std::vector<BpAabb>& boxes, const std::vector<BpTriangle>& triangles,
    const std::vector<BpEdge>& edges, const std::vector<BpDependency>& dependencies,
    const std::vector<int>& owners, const std::vector<unsigned char>& surface,
    int block_count, double d_hat, std::size_t minimum_contacts = 0,
    bool include_triangle_dependencies = true, bool protect_linear_gap = false) {
    Device<BpAabb> db(boxes);
    Device<BpTriangle> dt(triangles);
    Device<BpEdge> de(edges);
    Device<BpDependency> dd(dependencies);
    Device<int> owner(owners);
    Device<unsigned char> ds(surface);
    BroadPhaseTopology topology;
    topology.vertex_count = static_cast<int>(boxes.size()); topology.block_count = block_count;
    topology.triangle_count = static_cast<int>(triangles.size()); topology.edge_count = static_cast<int>(edges.size());
    topology.dependency_count = static_cast<int>(dependencies.size());
    topology.triangles = dt.data; topology.edges = de.data; topology.dependencies = dd.data;
    topology.vertex_block = owner.data; topology.surface_query = ds.data;
    topology.include_triangle_dependencies = include_triangle_dependencies;
    topology.protect_linear_gap = protect_linear_gap;
    // Reuse the allocation owner across changing geometry/topology/candidate
    // counts, including rebuilding an empty scene after a populated scene.
    static GpuBroadPhase broadphase;
    const BroadPhaseView view = broadphase.rebuild(topology, db.data, d_hat);
    const auto contacts = download(view.contacts, view.contact_count);
    std::set<ContactKey> actual;
    for (const auto& c : contacts) actual.insert(key(c));
    double query_padding=d_hat;
    if (protect_linear_gap) {
        double scene_scale=0;
        for(const auto& box:boxes)for(int k=0;k<3;++k)
            scene_scale=std::max(scene_scale,std::max(std::abs(box.lo[k]),std::abs(box.hi[k])));
        query_padding=std::nextafter(std::max({d_hat,1e-8,128*std::numeric_limits<double>::epsilon()*scene_scale}),
            std::numeric_limits<double>::infinity());
    }
    const auto reference = reference_contacts(boxes, triangles, edges, owners, surface, query_padding);
    require(actual == reference, "GPU BVH contacts differ from uncapped brute-force reference");
    require(actual.size() == contacts.size(), "duplicate GPU contacts");
    require(view.contact_count >= minimum_contacts, "stress fixture has too few contacts");

    const auto offsets = download(view.adjacency_offsets, block_count + 1);
    const auto adjacency = download(view.adjacency, view.directed_edge_count);
    require(offsets.front() == 0 && offsets.back() == adjacency.size(), "invalid GPU CSR bounds");
    std::set<std::pair<int,int>> graph;
    for (int i = 0; i < block_count; ++i) for (std::size_t j = offsets[i]; j < offsets[i + 1]; ++j)
        graph.emplace(i, adjacency[j]);
    require(graph == reference_graph(include_triangle_dependencies ? triangles : std::vector<BpTriangle>{}, dependencies, actual, owners), "GPU dependency graph mismatch");
    require(graph.size() == adjacency.size(), "duplicate graph edges");

    const auto colors = download(view.block_colors, block_count);
    std::vector<unsigned int> stages(block_count),expected_stages(view.color_count,0);
    for(int block=0;block<block_count;++block) {
        // Include fully fixed, single-mode, and combined rigid blocks.
        stages[block]=static_cast<unsigned int>(block%8);
        expected_stages[colors[block]]|=stages[block];
    }
    Device<unsigned int> device_stages(stages),device_color_stages(view.color_count);
    collect_color_stage_masks(view,block_count,device_stages.data,device_color_stages.data);
    require(download(device_color_stages.data,view.color_count)==expected_stages,
        "GPU stage masks mismatch actual mixed dependency coloring");
    const auto color_blocks = download(view.color_blocks, block_count);
    const auto color_offsets = download(view.color_offsets, view.color_count + 1);
    require(color_offsets.front() == 0 && color_offsets.back() == block_count, "invalid color bounds");
    for (const auto& e : graph) require(colors[e.first] != colors[e.second], "dependent update blocks share a color");
    std::vector<int> visits(block_count, 0);
    for (int c = 0; c < view.color_count; ++c) {
        int previous = -1;
        for (int i = color_offsets[c]; i < color_offsets[c + 1]; ++i) {
            const int b = color_blocks[i];
            require(b > previous && b < block_count, "color blocks not deterministically sorted");
            require(colors[b] == c, "color block indexing mismatch");
            ++visits[b]; previous = b;
        }
    }
    for (int count : visits) require(count == 1, "missing or repeated color block");
    const auto candidates_only = broadphase.rebuild(topology, db.data, d_hat, BroadPhaseMode::CandidatesOnly);
    const auto candidate_contacts = download(candidates_only.contacts, candidates_only.contact_count);
    require(no_solver_data(candidates_only) && no_solver_data(broadphase.view()),
        "candidate-only rebuild exposed stale graph or colors");
    require(candidate_contacts.size() == contacts.size()
        && (contacts.empty() || std::memcmp(candidate_contacts.data(), contacts.data(), contacts.size()*sizeof(BpContact)) == 0),
        "candidate-only rebuild changed contact bytes/order");
    const auto repeat = broadphase.rebuild(topology, db.data, d_hat);
    if(!std::getenv("IPC_GPU_COLORING")||std::string(std::getenv("IPC_GPU_COLORING"))!="cusparse")
        require(colors == download(repeat.block_colors, block_count), "coloring is not deterministic");
    if(!std::getenv("IPC_GPU_COLORING")||std::string(std::getenv("IPC_GPU_COLORING"))!="cusparse")
        require(color_blocks == download(repeat.color_blocks, block_count), "color ordering changed on rebuild");
    const auto unchanged=download(db.data,boxes.size());
    for(std::size_t i=0;i<boxes.size();++i)for(int k=0;k<3;++k)
        require(unchanged[i].lo[k]==boxes[i].lo[k]&&unchanged[i].hi[k]==boxes[i].hi[k],
            "query gap padding changed blue movement limits");
}

static void color_stage_test() {
    const std::vector<int> colors{0,1,1,0,2,2,3,3};
    Device<int> device_colors(colors);
    const unsigned int both=RigidTranslationColorStage|RigidOrientationColorStage;
    Device<unsigned int> stages(std::vector<unsigned int>{DeformableColorStage,RigidTranslationColorStage,
        RigidOrientationColorStage,DeformableColorStage,0,0,both,0});
    Device<unsigned int> result(4);
    BroadPhaseView view;view.block_colors=device_colors.data;view.color_count=4;
    collect_color_stage_masks(view,8,stages.data,result.data);
    require(download(result.data,4)==std::vector<unsigned int>({DeformableColorStage,both,0,both}),
        "mixed color stages or fixed-only color mismatch");
    check(cudaMemset(stages.data,0,8*sizeof(unsigned int)));
    collect_color_stage_masks(view,8,stages.data,result.data);
    require(download(result.data,4)==std::vector<unsigned int>(4,0),
        "all-fixed stage masks retained stale activity");
    require(download(device_colors.data,8)==colors,"stage scheduling changed graph colors");
    collect_color_stage_masks({},0,nullptr,nullptr);
    bool rejected=false;
    try {collect_color_stage_masks(view,0,nullptr,nullptr);}
    catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"inconsistent color-stage counts accepted");
}

static void motion_test() {
    const std::vector<double> start{0.0, 1.0, 2.0, -3.0, 0.0, 4.0};
    const std::vector<double> end{2.0, -1.0, 2.0, -1.0, 0.0, -2.0};
    Device<double> a(start), b(end);
    Device<BpAabb> boxes(2);
    make_motion_boxes(a.data, b.data, 2, 0.2, 0.001, boxes.data);
    const auto result = download(boxes.data, 2);
    for (int i = 0; i < 2; ++i) for (int k = 0; k < 3; ++k) {
        const double x = start[3 * i + k], y = end[3 * i + k];
        const double pad = 0.1 * std::abs(y - x) + 0.001;
        require(result[i].lo[k] <= std::min(x, y) - pad && result[i].hi[k] >= std::max(x, y) + pad,
                "motion box padding is not conservative");
    }
    bool rejected = false;
    try { make_motion_boxes(a.data, b.data, 2, -0.2, 0.0, boxes.data); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid padding was accepted");
}

static void gap_padding_test() {
    // At this world scale, the protected gap is about .0284, although the
    // local y coordinate is small. Motion does not intersect the triangle,
    // but y=.04 -> .02 must remain a candidate for endpoint gap validation.
    constexpr double world=1e12;
    const double gap=128*std::numeric_limits<double>::epsilon()*world;
    require(.02<gap&&gap<.04,"large-world gap fixture is invalid");
    for(bool edge_edge:{false,true}) {
        const std::vector<double> start=edge_edge
            ? std::vector<double>{world,.04,-1, world,.04,1, world-1,0,0, world+1,0,0}
            : std::vector<double>{world,.04,0, world-1,0,-1, world+1,0,-1, world,0,1};
        auto end=start;end[1]=.02;if(edge_edge)end[4]=.02;
        Device<double> a(start),b(end);Device<BpAabb> output(4);
        make_motion_boxes(a.data,b.data,4,.2,1e-8,output.data);
        const auto boxes=download(output.data,4);
        for(int node=0;node<4;++node)for(int axis=0;axis<3;++axis)
            require(boxes[node].lo[axis]<=std::min(start[3*node+axis],end[3*node+axis])-gap&&
                    boxes[node].hi[axis]>=std::max(start[3*node+axis],end[3*node+axis])+gap,
                "motion padding omitted the global protected gap");
        run_case(boxes,edge_edge?std::vector<BpTriangle>{}:std::vector<BpTriangle>{{{1,2,3}}},
            edge_edge?std::vector<BpEdge>{{{0,1}},{{2,3}}}:std::vector<BpEdge>{},
            {},{0,1,2,3},edge_edge?std::vector<unsigned char>{}:std::vector<unsigned char>{1,0,0,0},
            4,0.0,1,false);
    }
    // Target scale counts even when every starting coordinate is small.
    std::vector<double> start{0,.04,0,-1,0,-1,1,0,-1,0,0,1,0,10,0},end=start;
    end[1]=.02;end[12]=2*world;
    Device<double> a(start),b(end);Device<BpAabb> output(5);
    make_motion_boxes(a.data,b.data,5,.2,0.0,output.data);
    const auto boxes=download(output.data,5);
    require(boxes[1].lo[1]<=-2*gap&&boxes[1].hi[1]>=2*gap,"target-only scene scale ignored");
    run_case(boxes,{{{1,2,3}}},{},{},{0,1,2,3,4},{1,0,0,0,0},5,0.0,1,false);

    // Ordinary solver regions are intentionally disjoint. Enlarging ONLY the
    // query pad must recover NT/EE near-gap candidates without allowing nodes
    // to move outside these original blue regions.
    const std::vector<BpAabb> nt_boxes{point_box(world,.04,0,.012),
        point_box(world-1,0,-1,.001),point_box(world+1,0,-1,.001),point_box(world,0,1,.001)};
    require(reference_contacts(nt_boxes,{{{1,2,3}}},{},{0,1,2,3},{1,0,0,0},0).empty(),
        "ordinary near-gap fixture unexpectedly intersects without protection");
    run_case(nt_boxes,{{{1,2,3}}},{},{},{0,1,2,3},{1,0,0,0},4,0,0,false,false);
    run_case(nt_boxes,{{{1,2,3}}},{},{},{0,1,2,3},{1,0,0,0},4,0,1,false,true);
    const std::vector<BpAabb> ee_boxes{point_box(world,.04,-1,.01),point_box(world,.04,1,.01),
        point_box(world-1,0,0,.005),point_box(world+1,0,0,.005)};
    run_case(ee_boxes,{},{{{0,1}},{{2,3}}},{},{0,1,2,3},{},4,0,0,false,false);
    run_case(ee_boxes,{},{{{0,1}},{{2,3}}},{},{0,1,2,3},{},4,0,1,false,true);
}


// Focused public-API check. Snapshots own every downloaded payload before a
// later rebuild can reuse the GpuBroadPhase allocation owner.
class ScopedTestEnvironment {
    const char* name_;
    bool present_;
    std::string previous_;
public:
    explicit ScopedTestEnvironment(const char* name) : name_(name),
        present_(std::getenv(name) != nullptr),
        previous_(present_ ? std::getenv(name) : "") {}
    ~ScopedTestEnvironment() {
        if (present_) (void)setenv(name_, previous_.c_str(), 1);
        else (void)unsetenv(name_);
    }
    void assign(const char* value) {
        require((value ? setenv(name_, value, 1) : unsetenv(name_)) == 0,
            "cannot configure focused broadphase test environment");
    }
    ScopedTestEnvironment(const ScopedTestEnvironment&) = delete;
    ScopedTestEnvironment& operator=(const ScopedTestEnvironment&) = delete;
};

struct OwnedBroadPhaseSnapshot {
    std::vector<BpContact> contacts;
    std::vector<std::size_t> adjacency_offsets;
    std::vector<int> adjacency, colors, color_blocks, color_offsets;
    int color_count = 0;
};
static OwnedBroadPhaseSnapshot own_broadphase(const BroadPhaseView& view, int blocks) {
    OwnedBroadPhaseSnapshot out;
    out.contacts = download(view.contacts, view.contact_count);
    out.adjacency_offsets = download(view.adjacency_offsets, std::size_t(blocks) + 1);
    out.adjacency = download(view.adjacency, view.directed_edge_count);
    out.colors = download(view.block_colors, blocks);
    out.color_blocks = download(view.color_blocks, blocks);
    out.color_offsets = download(view.color_offsets, std::size_t(view.color_count) + 1);
    out.color_count = view.color_count;
    return out;
}
static bool same_contact_payload(const std::vector<BpContact>& a,
    const std::vector<BpContact>& b) {
    // All five fields are ints; assert the absence of padding before comparing
    // the complete raw orientation/order payload, including the contact kind.
    static_assert(sizeof(BpContact) == 5 * sizeof(int), "BpContact padding changed");
    return a.size() == b.size() && (a.empty()
        || std::memcmp(a.data(), b.data(), a.size() * sizeof(BpContact)) == 0);
}
static void require_same_broadphase(const OwnedBroadPhaseSnapshot& expected,
    const OwnedBroadPhaseSnapshot& actual) {
    require(same_contact_payload(expected.contacts, actual.contacts),
        "warp dispatch changed complete contact payload/order");
    require(expected.adjacency_offsets == actual.adjacency_offsets
        && expected.adjacency == actual.adjacency,
        "warp dispatch changed complete dependency CSR");
    require(expected.colors == actual.colors && expected.color_count == actual.color_count,
        "warp dispatch changed exact labels/palette");
    require(expected.color_blocks == actual.color_blocks
        && expected.color_offsets == actual.color_offsets,
        "warp dispatch changed complete sequential color order/offsets");
}
static void warp_dispatch_test(const std::string& mode) {
    require(mode == "degree" || mode == "spatial", "focused dispatch needs degree or spatial");
    ScopedTestEnvironment coloring("IPC_GPU_COLORING"), flag("IPC_GPU_WARP_ORDERED_COLOR"),
        spatial_dx("IPC_GPU_SPATIAL_COLOR_DX");
    coloring.assign(mode.c_str()); spatial_dx.assign("0.05");
    constexpr int count = 33; // Also exercises a partial last 32-color palette batch.
    std::vector<BpAabb> boxes;
    std::vector<int> owners;
    for (int i = 0; i < count; ++i) {
        boxes.push_back(point_box(.025 * (i % 6), .027 * (i / 6), .011 * (i % 3), .1));
        owners.push_back((10 * i) % count); // A bijection, deliberately nonidentity.
    }
    std::vector<BpTriangle> triangles;
    std::vector<BpEdge> edges;
    for (int i = 0; i < count; i += 3) {
        triangles.push_back({{i, i + 1, i + 2}});
        edges.push_back({{i, i + 1}}); edges.push_back({{i + 1, i + 2}});
        edges.push_back({{i + 2, i}});
    }
    const std::vector<unsigned char> surface(count, 1);
    Device<BpAabb> db(boxes); Device<BpTriangle> dt(triangles); Device<BpEdge> de(edges);
    Device<int> blocks(owners); Device<unsigned char> ds(surface);
    BroadPhaseTopology topology;
    topology.vertex_count = topology.block_count = count;
    topology.triangle_count = int(triangles.size()); topology.edge_count = int(edges.size());
    topology.triangles = dt.data; topology.edges = de.data;
    topology.vertex_block = blocks.data; topology.surface_query = ds.data;
    GpuBroadPhase broadphase;
    flag.assign("0");
    const auto serial = own_broadphase(broadphase.rebuild(topology, db.data, .01), count);
    require(!serial.contacts.empty() && serial.adjacency.size() == count * (count - 1)
        && serial.color_count == count, "focused fixture did not exercise the full 33-color graph");
    for (const char* value : {"1", static_cast<const char*>(nullptr)}) {
        flag.assign(value);
        require_same_broadphase(serial,
            own_broadphase(broadphase.rebuild(topology, db.data, .01), count));
    }
    // A malformed selector is parsed only when ordered coloring is reached.
    flag.assign("invalid");
    const auto candidates = broadphase.rebuild(topology, db.data, .01, BroadPhaseMode::CandidatesOnly);
    require(no_solver_data(candidates) && same_contact_payload(serial.contacts,
        download(candidates.contacts, candidates.contact_count)),
        "candidate-only rebuild parsed warp flag or changed contact payload");
    bool rejected = false;
    try { (void)broadphase.rebuild(topology, db.data, .01); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && no_solver_data(broadphase.view()) && !broadphase.view().contacts
        && !broadphase.view().contact_count, "invalid ordered warp flag did not invalidate view");
    if (mode == "degree") {
        coloring.assign("greedy");
        const auto generic = own_broadphase(broadphase.rebuild(topology, db.data, .01), count);
        require(same_contact_payload(serial.contacts, generic.contacts)
            && serial.adjacency_offsets == generic.adjacency_offsets
            && serial.adjacency == generic.adjacency && generic.color_count == count,
            "generic coloring parsed warp flag or changed contact/graph payload");
        coloring.assign(mode.c_str());
    }
    flag.assign("0");
    require_same_broadphase(serial,
        own_broadphase(broadphase.rebuild(topology, db.data, .01), count));
}


// Compare complete public payloads under independent defaults and rollbacks.
// Every Tree owner is constructed after its bounds selector is set.
static void broadphase_options_test(const std::string& mode) {
    require(mode == "degree" || mode == "spatial", "broadphase options need degree or spatial");
    ScopedTestEnvironment coloring("IPC_GPU_COLORING"), warp("IPC_GPU_WARP_ORDERED_COLOR"),
        shared("IPC_GPU_WARP_SHARED_MARKS"), bounds("IPC_GPU_PARALLEL_SCENE_BOUNDS"),
        spatial_dx("IPC_GPU_SPATIAL_COLOR_DX");
    coloring.assign(mode.c_str()); spatial_dx.assign("0.05");
    constexpr int count = 33;
    std::vector<BpAabb> boxes;
    std::vector<int> owners;
    for (int i = 0; i < count; ++i) {
        // Nonzero normal endpoints exercise the parallel reduction itself.
        boxes.push_back(point_box(1.125 + .025 * (i % 6),
            2.25 + .027 * (i / 6), 3.375 + .011 * (i % 3), .1));
        owners.push_back((10 * i) % count);
    }
    std::vector<BpTriangle> triangles;
    std::vector<BpEdge> edges;
    for (int i = 0; i < count; i += 3) {
        triangles.push_back({{i, i + 1, i + 2}});
        edges.push_back({{i, i + 1}}); edges.push_back({{i + 1, i + 2}});
        edges.push_back({{i + 2, i}});
    }
    Device<BpAabb> db(boxes); Device<BpTriangle> dt(triangles); Device<BpEdge> de(edges);
    Device<int> blocks(owners); Device<unsigned char> surface(std::vector<unsigned char>(count, 1));
    BroadPhaseTopology topology;
    topology.vertex_count = topology.block_count = count;
    topology.triangle_count = int(triangles.size()); topology.edge_count = int(edges.size());
    topology.triangles = dt.data; topology.edges = de.data;
    topology.vertex_block = blocks.data; topology.surface_query = surface.data;
    const auto snapshot = [&](const char* s, const char* b, const char* w) {
        shared.assign(s); bounds.assign(b); warp.assign(w);
        GpuBroadPhase owner;
        return own_broadphase(owner.rebuild(topology, db.data, .01), count);
    };
    const auto serial = snapshot("0", "0", "0");
    require(!serial.contacts.empty() && serial.color_count == count
        && serial.adjacency.size() == count * (count - 1), "options fixture lost its full graph");
    for (const char* s : {static_cast<const char*>(nullptr), "0", "1"})
        for (const char* b : {static_cast<const char*>(nullptr), "0", "1"})
            require_same_broadphase(serial, snapshot(s, b, "1"));
    // Ordered-color rollback must bypass even an unused malformed shared flag.
    require_same_broadphase(serial, snapshot("invalid", "0", "0"));
    shared.assign("invalid"); bounds.assign("0"); warp.assign("1");
    GpuBroadPhase invalid_shared;
    const auto candidates = invalid_shared.rebuild(topology, db.data, .01, BroadPhaseMode::CandidatesOnly);
    require(no_solver_data(candidates) && same_contact_payload(serial.contacts,
        download(candidates.contacts, candidates.contact_count)), "candidate-only path parsed shared selector");
    bool rejected = false;
    try { (void)invalid_shared.rebuild(topology, db.data, .01); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && no_solver_data(invalid_shared.view()) && !invalid_shared.view().contacts
        && !invalid_shared.view().contact_count,
        "invalid shared selector did not fail closed");
    shared.assign("0"); bounds.assign("invalid"); rejected = false;
    try { GpuBroadPhase invalid_bounds; }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid bounds selector was silently accepted");
    // Above the 48 KiB shared palette, default/on must retain the global Warp.
    constexpr int large = 12289;
    boxes.clear(); owners.clear();
    for (int i = 0; i < large; ++i) {
        boxes.push_back(point_box(1.125 + .25 * i, 2.25, 3.375, .001));
        owners.push_back(i);
    }
    Device<BpAabb> large_boxes(boxes); Device<int> large_blocks(owners);
    BroadPhaseTopology sparse; sparse.vertex_count = sparse.block_count = large;
    sparse.vertex_block = large_blocks.data;
    bounds.assign("1"); warp.assign("1"); shared.assign("0");
    GpuBroadPhase global;
    const auto reference = own_broadphase(global.rebuild(sparse, large_boxes.data, .01), large);
    require(reference.contacts.empty() && reference.color_count == 1
        && std::all_of(reference.colors.begin(), reference.colors.end(), [](int c) { return c == 0; }),
        "large sparse fallback fixture changed");
    for (const char* s : {static_cast<const char*>(nullptr), "1"}) {
        shared.assign(s); GpuBroadPhase fallback;
        require_same_broadphase(reference,
            own_broadphase(fallback.rebuild(sparse, large_boxes.data, .01), large));
    }
}

int main(int argc, char** argv) {
    try {
        if (argc != 1) {
            require(argc == 3, "unknown broadphase test selector");
            if (std::string(argv[1]) == "--broadphase-options") {
                broadphase_options_test(argv[2]);
                std::cout << "GPU broadphase options: " << argv[2]
                          << " defaults/rollbacks/shared-budget fallback payloads PASS\n";
            } else {
                require(std::string(argv[1]) == "--warp-dispatch", "unknown broadphase test selector");
                warp_dispatch_test(argv[2]);
                std::cout << "GPU focused warp dispatch: " << argv[2] << " serial/warp/default payloads PASS\n";
            }
            return 0;
        }
        color_stage_test();
        motion_test();
        gap_padding_test();
        // Cloth hinge, solid tetrahedron and two rigid update blocks. All are
        // close enough to exercise mixed contacts, shared-vertex exclusions,
        // same-body exclusions and the solid-interior NT query mask.
        std::vector<BpAabb> boxes;
        for (int i = 0; i < 14; ++i) boxes.push_back(point_box(0.08 * (i % 4), 0.025 * (i / 4), 0.01 * (i % 3)));
        std::vector<BpTriangle> triangles{{{0,1,2}}, {{0,2,3}}, {{4,5,6}}, {{8,9,10}}, {{11,12,13}}};
        std::vector<BpEdge> edges{{{0,1}}, {{1,2}}, {{0,2}}, {{0,3}}, {{2,3}},
            {{4,5}}, {{5,6}}, {{4,6}}, {{8,9}}, {{9,10}}, {{8,10}}, {{11,12}}, {{12,13}}, {{11,13}}};
        std::vector<BpDependency> dependencies{{{0,1,2,3},4}, {{4,5,6,7},4}};
        std::vector<int> owners{0,1,2,3,4,5,6,7,8,8,8,9,9,9};
        std::vector<unsigned char> surface(14, 1); surface[7] = 0;
        run_case(boxes, triangles, edges, dependencies, owners, surface, 10, 0.04, 1);
        run_case(boxes, triangles, edges, dependencies, owners, surface, 10, 0.0);
        // Move each physical object far away on the next rebuild. Stale
        // cross-object pairs and their dependency edges must disappear.
        for (int i = 0; i < 14; ++i) {
            const int object = i < 4 ? 0 : (i < 8 ? 1 : (i < 11 ? 2 : 3));
            boxes[i].lo[0] += 10.0 * object; boxes[i].hi[0] += 10.0 * object;
        }
        run_case(boxes, triangles, edges, dependencies, owners, surface, 10, 0.04);
        // Initial guess has contact-only coloring, with no elastic cliques.
        run_case(boxes, triangles, edges, {}, owners, surface, 10, 0.04, 0, false);
        run_case({}, {}, {}, {}, {}, {}, 0, 0.0);
        run_case({point_box(0,0,0), point_box(5,0,0)}, {}, {}, {}, {0,1}, {}, 2, 0.01);

        // More contacts for one primitive than small fixed-size neighbor lists
        // would hold. No candidate count or adjacency cap is permitted.
        boxes.assign(203, point_box(0,0,0)); owners.resize(203);
        for (int i = 0; i < 203; ++i) owners[i] = i;
        run_case(boxes, {{{200,201,202}}}, {}, {}, owners, {}, 203, 0.005, 200);

        // Invalid device topology must fail closed, never read invalid memory.
        Device<BpAabb> db(std::vector<BpAabb>{point_box(0,0,0)});
        Device<int> invalid_owner(std::vector<int>{2});
        BroadPhaseTopology invalid; invalid.vertex_count = 1; invalid.block_count = 1;
        invalid.vertex_block = invalid_owner.data;
        GpuBroadPhase broadphase;
        Device<int> valid_owner(std::vector<int>{0});
        for(const auto mode:{BroadPhaseMode::Solver,BroadPhaseMode::CandidatesOnly}) {
            auto valid=invalid;valid.vertex_block=valid_owner.data;
            broadphase.rebuild(valid,db.data,0.01);
            bool rejected = false;
            try { broadphase.rebuild(invalid, db.data, 0.01, mode); }
            catch (const std::runtime_error&) { rejected = true; }
            const auto failed=broadphase.view();
            require(rejected && !failed.contacts && !failed.contact_count && no_solver_data(failed),
                "invalid topology did not invalidate every view field");
            const auto recovered=broadphase.rebuild(valid,db.data,0.01,mode);
            require(mode==BroadPhaseMode::Solver?recovered.color_count==1:no_solver_data(recovered),
                "failed rebuild could not recover with selected build mode");
        }
        std::cout << "GPU broadphase: motion boxes, mixed contacts, uncapped compaction, CSR, coloring PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GPU broadphase test failed: " << error.what() << '\n';
        return 1;
    }
}
