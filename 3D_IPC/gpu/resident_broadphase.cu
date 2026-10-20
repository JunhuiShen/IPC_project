#include "resident_broadphase.h"
#include "resident_profile.h"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdlib>
#include <climits>
#include <fstream>
#include <thrust/fill.h>
#include <math_constants.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ipc_gpu {
namespace resident {
namespace {

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
void launched(const char* operation) { check(cudaGetLastError(), operation); }

template<class T> struct Buffer {
    T* data = nullptr;
    std::size_t capacity = 0;
    ~Buffer() { if (data) cudaFree(data); }
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void reserve(std::size_t n) {
        if (n <= capacity) return;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::overflow_error("resident broadphase allocation overflow");
        T* next = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&next), n * sizeof(T)), "broadphase cudaMalloc");
        if (data) check(cudaFree(data), "broadphase cudaFree");
        data = next;
        capacity = n;
    }
};

constexpr int threads = 128;
int blocks(std::size_t count) {
    return static_cast<int>(std::min<std::size_t>((count + threads - 1) / threads, 65535));
}

__global__ void color_stage_masks(const int* colors, int count,
                                  const unsigned int* block_stages,
                                  unsigned int* color_stages) {
    for (int b=blockIdx.x*blockDim.x+threadIdx.x;b<count;b+=blockDim.x*gridDim.x) {
        const unsigned int stages=block_stages[b];
        if (stages) atomicOr(color_stages+colors[b],stages);
    }
}

__device__ BpAabb empty_box() {
    BpAabb box;
    for (int k = 0; k < 3; ++k) { box.lo[k] = CUDART_INF; box.hi[k] = -CUDART_INF; }
    return box;
}
__device__ BpAabb join(BpAabb a, const BpAabb& b) {
    for (int k = 0; k < 3; ++k) {
        a.lo[k] = fmin(a.lo[k], b.lo[k]); a.hi[k] = fmax(a.hi[k], b.hi[k]);
    }
    return a;
}
__device__ BpAabb expand(BpAabb box, double pad) {
    for (int k = 0; k < 3; ++k) {
        box.lo[k] = nextafter(box.lo[k] - pad, -CUDART_INF);
        box.hi[k] = nextafter(box.hi[k] + pad, CUDART_INF);
    }
    return box;
}
__device__ bool overlap(const BpAabb& a, const BpAabb& b) {
    for (int k = 0; k < 3; ++k)
        if (a.lo[k] > b.hi[k] || a.hi[k] < b.lo[k]) return false;
    return true;
}

__global__ void motion_scale(const double* start, const double* end, int count,
                             unsigned long long* largest_bits, int* invalid) {
    double largest=0.0;
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) {
        for (int k=0;k<3;++k) {
            const double a=start[3*static_cast<std::size_t>(i)+k];
            const double b=end[3*static_cast<std::size_t>(i)+k];
            if (!isfinite(a)||!isfinite(b)) atomicExch(invalid,1);
            else largest=fmax(largest,fmax(fabs(a),fabs(b)));
        }
    }
    // Positive finite binary64 values have monotonically ordered bit patterns.
    // Atomic max is an exact, deterministic device reduction; no host geometry
    // or floating-point scale reduction is involved.
    atomicMax(largest_bits,static_cast<unsigned long long>(__double_as_longlong(largest)));
}

__global__ void box_scale(const BpAabb* boxes,int count,unsigned long long* largest_bits) {
    double largest=0.0;
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x)
        for (int k=0;k<3;++k)
            largest=fmax(largest,fmax(fabs(boxes[i].lo[k]),fabs(boxes[i].hi[k])));
    atomicMax(largest_bits,static_cast<unsigned long long>(__double_as_longlong(largest)));
}
__global__ void set_query_padding(double d_hat,bool protect_gap,
                                  const unsigned long long* scale_bits,double* result) {
    double padding=d_hat;
    if (protect_gap) {
        const double scale=__longlong_as_double(static_cast<long long>(*scale_bits));
        padding=nextafter(fmax(padding,fmax(1e-8,128.0*DBL_EPSILON*scale)),CUDART_INF);
    }
    *result=padding;
}

__global__ void motion_boxes(const double* start, const double* end, int count,
                             double relative_pad, double absolute_pad,
                             const unsigned long long* scene_scale_bits,
                             BpAabb* boxes, int* invalid) {
    const double scale=__longlong_as_double(static_cast<long long>(*scene_scale_bits));
    const double gap_pad=nextafter(fmax(fmax(absolute_pad,1e-8),128.0*DBL_EPSILON*scale),CUDART_INF);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        BpAabb box;
        for (int k = 0; k < 3; ++k) {
            const double a = start[3 * static_cast<std::size_t>(i) + k];
            const double b = end[3 * static_cast<std::size_t>(i) + k];
            const double pad = (0.5 * relative_pad) * fabs(b - a) + gap_pad;
            box.lo[k] = nextafter(fmin(a, b) - pad, -CUDART_INF);
            box.hi[k] = nextafter(fmax(a, b) + pad, CUDART_INF);
            if (!isfinite(a) || !isfinite(b) || !isfinite(box.lo[k]) || !isfinite(box.hi[k]))
                atomicExch(invalid, 1);
        }
        boxes[i] = box;
    }
}

__global__ void validate(BroadPhaseTopology topology, const BpAabb* boxes, int count,
                         int* invalid) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        if (i < topology.vertex_count) {
            const int owner = topology.vertex_block[i];
            if (owner < 0 || owner >= topology.block_count) atomicExch(invalid, 2);
            for (int k = 0; k < 3; ++k)
                if (!isfinite(boxes[i].lo[k]) || !isfinite(boxes[i].hi[k]) || boxes[i].lo[k] > boxes[i].hi[k])
                    atomicExch(invalid, 3);
        }
        if (i < topology.triangle_count)
            for (int k = 0; k < 3; ++k)
                if (topology.triangles[i].vertices[k] < 0 || topology.triangles[i].vertices[k] >= topology.vertex_count)
                    atomicExch(invalid, 4);
        if (i < topology.edge_count)
            for (int k = 0; k < 2; ++k)
                if (topology.edges[i].vertices[k] < 0 || topology.edges[i].vertices[k] >= topology.vertex_count)
                    atomicExch(invalid, 5);
        if (i < topology.dependency_count) {
            const BpDependency d = topology.dependencies[i];
            if (d.count < 2 || d.count > 4) atomicExch(invalid, 6);
            else for (int k = 0; k < d.count; ++k)
                if (d.vertices[k] < 0 || d.vertices[k] >= topology.vertex_count) atomicExch(invalid, 7);
        }
    }
}

__global__ void primitive_boxes(const BpAabb* nodes, BroadPhaseTopology topology,
                                BpAabb* triangles, BpAabb* edges, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        if (i < topology.triangle_count) {
            BpAabb box = empty_box();
            for (int k = 0; k < 3; ++k) box = join(box, nodes[topology.triangles[i].vertices[k]]);
            triangles[i] = box;
        }
        if (i < topology.edge_count) {
            const BpEdge edge = topology.edges[i];
            edges[i] = join(nodes[edge.vertices[0]], nodes[edge.vertices[1]]);
        }
    }
}

// One device thread is deliberate here: a conservative baseline bounds
// reduction. No geometry is downloaded to construct the hierarchy.
__global__ void scene_bounds(const BpAabb* boxes, int count, BpAabb* result) {
    BpAabb total = empty_box();
    for (int i = 0; i < count; ++i) total = join(total, boxes[i]);
    *result = total;
}
__device__ unsigned spread10(unsigned x) {
    x &= 0x000003ffu;
    x = (x | (x << 16)) & 0x030000ffu;
    x = (x | (x << 8)) & 0x0300f00fu;
    x = (x | (x << 4)) & 0x030c30c3u;
    return (x | (x << 2)) & 0x09249249u;
}
__global__ void morton_keys(const BpAabb* boxes, const BpAabb* bounds, int count,
                            std::uint64_t* keys, int* indices) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        unsigned q[3];
        for (int k = 0; k < 3; ++k) {
            const double center = 0.5 * boxes[i].lo[k] + 0.5 * boxes[i].hi[k];
            const double span = bounds->hi[k] - bounds->lo[k];
            double fraction = (span > 0.0 && isfinite(span)) ? (center - bounds->lo[k]) / span : 0.5;
            fraction = fmin(1.0, fmax(0.0, fraction));
            q[k] = static_cast<unsigned>(1023.0 * fraction);
        }
        const unsigned code = spread10(q[0]) | (spread10(q[1]) << 1) | (spread10(q[2]) << 2);
        keys[i] = (static_cast<std::uint64_t>(code) << 32) | static_cast<unsigned>(i);
        indices[i] = i;
    }
}
__global__ void fill_leaves(const BpAabb* source, const int* indices, int count,
                            int leaf_base, BpAabb* nodes) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < leaf_base;
         i += blockDim.x * gridDim.x)
        nodes[leaf_base + i] = i < count ? source[indices[i]] : empty_box();
}
__global__ void join_level(BpAabb* nodes, int first, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        const int node = first + i;
        nodes[node] = join(nodes[2 * node], nodes[2 * node + 1]);
    }
}


#include "parallel_scene_bounds.cuh"

struct TreeView { const BpAabb* nodes; const int* indices; int leaf_base, count; };
struct Tree {
    const bool parallel_bounds=parallel_scene_bounds_option();
    Buffer<SceneBoundsPartial> bounds_partials;
    Buffer<BpAabb> nodes, bounds;
    Buffer<int> indices;
    Buffer<std::uint64_t> keys;
    int leaf_base = 0, count = 0;
    TreeView view() const { return {nodes.data, indices.data, leaf_base, count}; }
    void build(const BpAabb* source, int size) {
        count = size;
        if (!size) { leaf_base = 0; return; }
        if (size > (1 << 29)) throw std::overflow_error("resident BVH exceeds supported index range");
        leaf_base = 1;
        while (leaf_base < size) leaf_base *= 2;
        nodes.reserve(2 * static_cast<std::size_t>(leaf_base));
        bounds.reserve(1); indices.reserve(size); keys.reserve(size);
        if(parallel_bounds) {
            const int groups=std::min(blocks(size),256);
            bounds_partials.reserve(groups);
            scene_bounds_partials<<<groups,128>>>(source,size,bounds_partials.data);
            launched("parallel BVH bounds partials");
            scene_bounds_finish<<<1,128>>>(source,size,bounds_partials.data,groups,bounds.data);
            launched("parallel BVH bounds finish");
        } else scene_bounds<<<1, 1>>>(source, size, bounds.data);
        morton_keys<<<blocks(size), threads>>>(source, bounds.data, size, keys.data, indices.data);
        launched("GPU BVH Morton keys");
        thrust::sort_by_key(thrust::device, thrust::device_pointer_cast(keys.data),
            thrust::device_pointer_cast(keys.data + size), thrust::device_pointer_cast(indices.data));
        fill_leaves<<<blocks(leaf_base), threads>>>(source, indices.data, size, leaf_base, nodes.data);
        for (int first = leaf_base / 2; first; first /= 2)
            join_level<<<blocks(first), threads>>>(nodes.data, first, first);
        launched("GPU BVH bottom-up build");
    }
};

__device__ bool same_block(const BpContact& c, const int* owner) {
    const int b = owner[c.vertices[0]];
    for (int k = 1; k < 4; ++k) if (owner[c.vertices[k]] != b) return false;
    return true;
}

template<bool Fill>
__device__ std::size_t query_contacts(int query, BroadPhaseTopology topology,
                                      const BpAabb* nodes, const BpAabb* edge_boxes,
                                      TreeView triangles, TreeView edges, double d_hat,
                                      BpContact* out) {
    const bool nt = query < topology.vertex_count;
    const int source = nt ? query : query - topology.vertex_count;
    if (nt && topology.surface_query && !topology.surface_query[source]) return 0;
    const TreeView tree = nt ? triangles : edges;
    if (!tree.count) return 0;
    const BpAabb query_box = expand(nt ? nodes[source] : edge_boxes[source], d_hat);
    // Balanced binary trees have <= 29 levels by the checked build limit.
    // DFS therefore needs <= 30 entries; this is a traversal stack, NOT a
    // candidate cap. Every overlapping leaf is counted/emitted.
    int stack[64], top = 0;
    stack[top++] = 1;
    std::size_t count = 0;
    while (top) {
        const int node = stack[--top];
        if (!overlap(query_box, tree.nodes[node])) continue;
        if (node < tree.leaf_base) {
            stack[top++] = 2 * node + 1;
            stack[top++] = 2 * node;
            continue;
        }
        const int leaf = node - tree.leaf_base;
        if (leaf >= tree.count) continue;
        const int primitive = tree.indices[leaf];
        BpContact c;
        c.kind = nt ? 0 : 1;
        bool skip = false;
        if (nt) {
            c.vertices[0] = source;
            for (int k = 0; k < 3; ++k) {
                c.vertices[k + 1] = topology.triangles[primitive].vertices[k];
                skip |= source == c.vertices[k + 1];
            }
        } else {
            if (primitive <= source) continue;
            const BpEdge a = topology.edges[source], b = topology.edges[primitive];
            c.vertices[0] = a.vertices[0]; c.vertices[1] = a.vertices[1];
            c.vertices[2] = b.vertices[0]; c.vertices[3] = b.vertices[1];
            for (int i = 0; i < 2; ++i) for (int j = 2; j < 4; ++j) skip |= c.vertices[i] == c.vertices[j];
        }
        if (skip || same_block(c, topology.vertex_block)) continue;
        if (Fill) out[count] = c;
        ++count;
    }
    return count;
}

template<bool Fill>
__global__ void contact_pass(BroadPhaseTopology topology, const BpAabb* nodes,
                              const BpAabb* edge_boxes, TreeView triangles,
                              TreeView edges, const double* query_padding, std::size_t query_count,
                              std::size_t* offsets, BpContact* contacts) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < query_count;
         i += blockDim.x * gridDim.x) {
        const std::size_t n = query_contacts<Fill>(static_cast<int>(i), topology, nodes,
            edge_boxes, triangles, edges, *query_padding, Fill ? contacts + offsets[i] : nullptr);
        if (!Fill) offsets[i] = n;
    }
}

__device__ BpDependency dependency_at(std::size_t index, BroadPhaseTopology topology,
                                      const BpContact* contacts) {
    BpDependency result;
    const int triangle_dependencies = topology.include_triangle_dependencies ? topology.triangle_count : 0;
    if (index < static_cast<std::size_t>(triangle_dependencies)) {
        result.count = 3;
        for (int k = 0; k < 3; ++k) result.vertices[k] = topology.triangles[index].vertices[k];
    } else if (index - triangle_dependencies < static_cast<std::size_t>(topology.dependency_count)) {
        result = topology.dependencies[index - triangle_dependencies];
    } else {
        const BpContact c = contacts[index - triangle_dependencies - topology.dependency_count];
        result.count = 4;
        for (int k = 0; k < 4; ++k) result.vertices[k] = c.vertices[k];
    }
    return result;
}
template<bool Fill>
__global__ void dependency_pass(BroadPhaseTopology topology, const BpContact* contacts,
                                 std::size_t count, std::size_t* offsets,
                                 std::uint64_t* keys) {
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += blockDim.x * gridDim.x) {
        const BpDependency d = dependency_at(index, topology, contacts);
        std::size_t n = 0;
        for (int i = 0; i < d.count; ++i) for (int j = i + 1; j < d.count; ++j) {
            const unsigned a = topology.vertex_block[d.vertices[i]], b = topology.vertex_block[d.vertices[j]];
            if (a == b) continue;
            if (Fill) {
                keys[offsets[index] + n] = (static_cast<std::uint64_t>(a) << 32) | b;
                keys[offsets[index] + n + 1] = (static_cast<std::uint64_t>(b) << 32) | a;
            }
            n += 2;
        }
        if (!Fill) offsets[index] = n;
    }
}

__global__ void csr_rows(const std::uint64_t* keys, std::size_t count,
                         std::size_t* offsets, int* columns) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        const unsigned row = static_cast<unsigned>(keys[i] >> 32);
        atomicAdd(reinterpret_cast<unsigned long long*>(offsets + row), 1ull);
        columns[i] = static_cast<int>(static_cast<unsigned>(keys[i]));
    }
}
#include "pbgs_coloring.cuh"
#include "pbgs_warp_ordered_exact.cuh"
#include "pbgs_warp_shared_forbidden_exact.cuh"
#include "pbgs_label_compaction.cuh"

__global__ void greedy_color(const std::size_t* offsets, const int* adjacency,
                             int count, int* colors, int* forbidden,
                             int* color_count) {
    int largest = -1;
    for (int vertex = 0; vertex < count; ++vertex) {
        for (std::size_t i = offsets[vertex]; i < offsets[vertex + 1]; ++i) {
            const int neighbor = adjacency[i];
            if (neighbor < vertex) forbidden[colors[neighbor]] = vertex;
        }
        int color = 0;
        while (color < count && forbidden[color] == vertex) ++color;
        colors[vertex] = color;
        if (color > largest) largest = color;
    }
    *color_count = largest + 1;
}
__global__ void count_colors(const int* colors, int count, int* offsets) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) atomicAdd(offsets + colors[i], 1);
}

std::size_t scan_counts(Buffer<std::size_t>& offsets, std::size_t count) {
    check(cudaMemset(offsets.data + count, 0, sizeof(std::size_t)), "GPU count sentinel");
    auto begin = thrust::device_pointer_cast(offsets.data);
    thrust::exclusive_scan(thrust::device, begin, begin + count + 1, begin);
    std::size_t total;
    check(cudaMemcpy(&total, offsets.data + count, sizeof(total), cudaMemcpyDeviceToHost), "GPU count readback");
    return total;
}
void require_status(Buffer<int>& error, const char* operation) {
    int value = 0;
    check(cudaMemcpy(&value, error.data, sizeof(value), cudaMemcpyDeviceToHost), operation);
    if (value) throw std::runtime_error(std::string(operation) + ": invalid input (code " + std::to_string(value) + ")");
}

} // namespace

void collect_color_stage_masks(const BroadPhaseView& view, int block_count,
                              const unsigned int* block_stages,
                              unsigned int* color_stages) {
    if (block_count<0 || view.color_count<0 || view.color_count>block_count ||
        (block_count && (!view.color_count || !view.block_colors || !block_stages || !color_stages)))
        throw std::invalid_argument("invalid resident color-stage arrays/counts");
    if (!block_count) return;
    check(cudaMemsetAsync(color_stages,0,static_cast<std::size_t>(view.color_count)*sizeof(unsigned int)),
          "clear GPU color-stage masks");
    color_stage_masks<<<blocks(block_count),threads>>>(view.block_colors,block_count,block_stages,color_stages);
    launched("GPU color-stage masks");
}

struct ResidentBroadPhase::Impl {
    Buffer<int> error;
    Buffer<unsigned long long> scene_scale;
    Buffer<double> query_padding;
    Buffer<BpAabb> triangles, edges;
    Tree triangle_tree, edge_tree;
    Buffer<std::size_t> query_offsets, dependency_offsets, adjacency_offsets;
    Buffer<BpContact> contacts;
    Buffer<std::uint64_t> graph_keys;
    Buffer<int> adjacency, colors, forbidden, color_count, color_blocks, sorted_colors, color_offsets;
    PbgsColoringStorage coloring;
    BroadPhaseView result;
};

void make_motion_boxes(const double* start, const double* end, int count,
                       double padding_fraction, double absolute_pad, BpAabb* outboxes) {
    if (count < 0 || !std::isfinite(padding_fraction) || padding_fraction < 0.0 ||
        !std::isfinite(absolute_pad) || absolute_pad < 0.0)
        throw std::invalid_argument("invalid GPU motion box parameters");
    if (count > (1 << 29)) throw std::overflow_error("GPU motion boxes exceed supported index range");
    if (!count) return;
    if (!start || !end || !outboxes) throw std::invalid_argument("null GPU motion box storage");
    Buffer<int> invalid;
    Buffer<unsigned long long> scene_scale;
    invalid.reserve(1);
    scene_scale.reserve(1);
    check(cudaMemset(invalid.data, 0, sizeof(int)), "clear GPU motion box status");
    check(cudaMemset(scene_scale.data,0,sizeof(unsigned long long)),"clear GPU motion scale");
    motion_scale<<<blocks(count),threads>>>(start,end,count,scene_scale.data,invalid.data);
    launched("GPU motion scene scale");
    motion_boxes<<<blocks(count), threads>>>(start, end, count, padding_fraction, absolute_pad, scene_scale.data, outboxes, invalid.data);
    launched("GPU motion boxes");
    require_status(invalid, "GPU motion box validation");
}

ResidentBroadPhase::ResidentBroadPhase() : impl_(new Impl) {}
ResidentBroadPhase::~ResidentBroadPhase() = default;
BroadPhaseView ResidentBroadPhase::view() const { return impl_->result; }

BroadPhaseView ResidentBroadPhase::rebuild(const BroadPhaseTopology& t,
                                           const BpAabb* blue_boxes, double d_hat,
                                           BroadPhaseMode mode) {
    ResidentProfileScope profile("broadphase");
    static_assert(sizeof(std::size_t) == sizeof(unsigned long long), "GPU CSR requires 64-bit size_t");
    Impl& p = *impl_;
    // A failed rebuild invalidates the old view: stale candidates must not be
    // mistaken for a successful rebuild of the new permitted motion region.
    p.result = {};
    if (mode != BroadPhaseMode::Solver && mode != BroadPhaseMode::CandidatesOnly)
        throw std::invalid_argument("invalid resident broadphase build mode");
    if (t.vertex_count < 0 || t.block_count < 0 || t.triangle_count < 0 || t.edge_count < 0 ||
        t.dependency_count < 0 || !std::isfinite(d_hat) || d_hat < 0.0)
        throw std::invalid_argument("invalid resident broadphase counts/d_hat");
    if (std::max({t.vertex_count, t.block_count, t.triangle_count, t.edge_count, t.dependency_count}) > (1 << 29))
        throw std::overflow_error("resident topology exceeds supported 29-bit item index range");
    if ((t.vertex_count && (!blue_boxes || !t.vertex_block || !t.block_count)) ||
        (t.triangle_count && !t.triangles) || (t.edge_count && !t.edges) ||
        (t.dependency_count && !t.dependencies))
        throw std::invalid_argument("missing resident broadphase device arrays");
    if (static_cast<std::size_t>(t.vertex_count) + t.edge_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("too many resident broadphase queries");
    {
    ResidentProfileScope validate_profile("validate");
    const int count = std::max({t.vertex_count, t.triangle_count, t.edge_count, t.dependency_count});
    p.error.reserve(1);
    check(cudaMemset(p.error.data, 0, sizeof(int)), "clear GPU topology status");
    if (count) {
        validate<<<blocks(count), threads>>>(t, blue_boxes, count, p.error.data);
        launched("GPU broadphase topology validation");
        require_status(p.error, "GPU broadphase topology validation");
    }
    }
    {
    ResidentProfileScope bounds_profile("bounds_bvh");
    p.scene_scale.reserve(1);p.query_padding.reserve(1);
    check(cudaMemset(p.scene_scale.data,0,sizeof(unsigned long long)),"clear GPU broadphase scale");
    if (t.protect_linear_gap && t.vertex_count)
        box_scale<<<blocks(t.vertex_count),threads>>>(blue_boxes,t.vertex_count,p.scene_scale.data);
    set_query_padding<<<1,1>>>(d_hat,t.protect_linear_gap,p.scene_scale.data,p.query_padding.data);
    launched("GPU protected query padding");
    p.triangles.reserve(t.triangle_count); p.edges.reserve(t.edge_count);
    const int primitive_count = std::max(t.triangle_count, t.edge_count);
    if (primitive_count) {
        primitive_boxes<<<blocks(primitive_count), threads>>>(blue_boxes, t, p.triangles.data, p.edges.data, primitive_count);
        launched("GPU primitive boxes");
    }
    p.triangle_tree.build(p.triangles.data, t.triangle_count);
    p.edge_tree.build(p.edges.data, t.edge_count);
    }
    std::size_t contacts=0;
    {
    ResidentProfileScope candidate_profile("candidates");
    const std::size_t queries = static_cast<std::size_t>(t.vertex_count) + t.edge_count;
    p.query_offsets.reserve(queries + 1);
    if (queries) contact_pass<false><<<blocks(queries), threads>>>(t, blue_boxes, p.edges.data,
        p.triangle_tree.view(), p.edge_tree.view(), p.query_padding.data, queries, p.query_offsets.data, nullptr);
    launched("GPU contact count pass");
    contacts = scan_counts(p.query_offsets, queries);
    p.contacts.reserve(contacts);
    if (queries && contacts) contact_pass<true><<<blocks(queries), threads>>>(t, blue_boxes, p.edges.data,
        p.triangle_tree.view(), p.edge_tree.view(), p.query_padding.data, queries, p.query_offsets.data, p.contacts.data);
        launched("GPU contact fill pass");
    }

    if (mode == BroadPhaseMode::CandidatesOnly) {
        // Preserve the synchronous rebuild contract and publish only after
        // completion. Retained graph allocations are deliberately not exposed.
        check(cudaDeviceSynchronize(), "resident candidate broadphase completion");
        p.result = {};
        p.result.contacts = p.contacts.data;
        p.result.contact_count = contacts;
        resident_profile_broadphase(contacts, 0, 0);
        return p.result;
    }

    std::size_t graph_edges = 0;
    {
    ResidentProfileScope graph_profile("dependency_graph");
    const std::size_t static_dependencies = static_cast<std::size_t>(t.include_triangle_dependencies ? t.triangle_count : 0) + t.dependency_count;
    if (contacts > std::numeric_limits<std::size_t>::max() / 12 - static_dependencies)
        throw std::overflow_error("resident graph incidence count overflow");
    const std::size_t dependencies = static_dependencies + contacts;
    p.dependency_offsets.reserve(dependencies + 1);
    if (dependencies) dependency_pass<false><<<blocks(dependencies), threads>>>(t, p.contacts.data,
        dependencies, p.dependency_offsets.data, nullptr);
    launched("GPU dependency count pass");
    const std::size_t raw_edges = scan_counts(p.dependency_offsets, dependencies);
    p.graph_keys.reserve(raw_edges);
    if (raw_edges) {
        dependency_pass<true><<<blocks(dependencies), threads>>>(t, p.contacts.data,
            dependencies, p.dependency_offsets.data, p.graph_keys.data);
        launched("GPU dependency fill pass");
        auto first = thrust::device_pointer_cast(p.graph_keys.data);
        thrust::sort(thrust::device, first, first + raw_edges);
        graph_edges = thrust::unique(thrust::device, first, first + raw_edges) - first;
    }
    p.adjacency.reserve(graph_edges); p.adjacency_offsets.reserve(static_cast<std::size_t>(t.block_count) + 1);
    check(cudaMemset(p.adjacency_offsets.data, 0, (static_cast<std::size_t>(t.block_count) + 1) * sizeof(std::size_t)), "clear GPU graph rows");
    if (graph_edges) csr_rows<<<blocks(graph_edges), threads>>>(p.graph_keys.data, graph_edges,
        p.adjacency_offsets.data, p.adjacency.data);
    launched("GPU CSR rows");
    scan_counts(p.adjacency_offsets, t.block_count);
    }

    int colors = 0;
    int vendor_palette_extent = -1; // Diagnostic only; never an allocation/index bound.
    const int color_mode=pbgs_coloring_mode();
    {
    ResidentProfileScope coloring_profile("coloring");
    p.colors.reserve(t.block_count); p.forbidden.reserve(t.block_count); p.color_count.reserve(1);
    if (t.block_count) check(cudaMemset(p.forbidden.data, 0xff, t.block_count * sizeof(int)), "clear GPU greedy color marks");
    if(color_mode==1&&t.block_count&&graph_edges) {
        if(graph_edges>INT_MAX)throw std::overflow_error("cuSPARSE coloring requires 32-bit edge indices");
        p.coloring.vendor_init();p.coloring.offsets.reserve(t.block_count+1);p.coloring.values.reserve(graph_edges);
        pbgs_convert_offsets<<<blocks(t.block_count+1),threads>>>(p.adjacency_offsets.data,p.coloring.offsets.data,t.block_count);
        thrust::fill(thrust::device,thrust::device_pointer_cast(p.coloring.values.data),thrust::device_pointer_cast(p.coloring.values.data+graph_edges),1.f);
        check(cudaMemset(p.colors.data,0xff,std::size_t(t.block_count)*sizeof(int)),"initialize vendor output labels");
        const float fraction=1.f;
        sparse_check(cusparseScsrcolor(p.coloring.handle,t.block_count,int(graph_edges),p.coloring.descriptor,
            p.coloring.values.data,p.coloring.offsets.data,p.adjacency.data,&fraction,&vendor_palette_extent,p.colors.data,nullptr,p.coloring.info));
        // Vendor labels can have holes. Preserve each occupied independent set
        // and its ascending execution order while deriving a dense GPU palette.
        colors=pbgs_compact_occupied_labels(p.coloring.occupied,p.colors.data,t.block_count,p.error.data);
        require_status(p.error,"cuSPARSE nonnegative output label validation");
    } else if(color_mode>=2&&t.block_count) {
        p.coloring.order.reserve(t.block_count);p.coloring.keys.reserve(t.block_count);
        double dx=.05;
        if(color_mode==3) {
        p.coloring.anchors.reserve(t.block_count);
        // Use INT_MAX exactly, including blocks with no owned vertex.
        thrust::fill(thrust::device,thrust::device_pointer_cast(p.coloring.anchors.data),thrust::device_pointer_cast(p.coloring.anchors.data+t.block_count),INT_MAX);
        if(t.vertex_count)pbgs_anchors<<<blocks(t.vertex_count),threads>>>(t.vertex_block,t.vertex_count,p.coloring.anchors.data);
        if(const char* value=std::getenv("IPC_GPU_SPATIAL_COLOR_DX")){char* end=nullptr;dx=std::strtod(value,&end);if(!end||*end)throw std::invalid_argument("invalid spatial ordering dx");}
        if(!std::isfinite(dx)||dx<=0)throw std::invalid_argument("spatial ordering dx must be finite and positive");
        }
        // dx only orders vertices. Safety never relies on a spatial radius:
        // every original dependency remains in CSR and is checked below.
        pbgs_priority<<<blocks(t.block_count),threads>>>(p.adjacency_offsets.data,t.block_count,p.coloring.keys.data,
            p.coloring.anchors.data,blue_boxes,color_mode,dx,p.error.data);
        require_status(p.error,"spatial ordering coordinates");
        thrust::sequence(thrust::device,thrust::device_pointer_cast(p.coloring.order.data),thrust::device_pointer_cast(p.coloring.order.data+t.block_count));
        thrust::stable_sort_by_key(thrust::device,thrust::device_pointer_cast(p.coloring.keys.data),thrust::device_pointer_cast(p.coloring.keys.data+t.block_count),thrust::device_pointer_cast(p.coloring.order.data));
        check(cudaMemset(p.colors.data,0xff,t.block_count*sizeof(int)),"initialize ordered colors");
        if(pbgs_warp_ordered_enabled()) {
            if(pbgs_warp_shared_marks_enabled())
                pbgs_launch_warp_shared_marks_exact(p.adjacency_offsets.data,p.adjacency.data,t.block_count,p.coloring.order.data,p.colors.data,p.forbidden.data,p.color_count.data);
            else
                pbgs_ordered_greedy_warp_exact<<<1,32>>>(p.adjacency_offsets.data,p.adjacency.data,t.block_count,p.coloring.order.data,p.colors.data,p.forbidden.data,p.color_count.data);
        } else
            pbgs_ordered_greedy<<<1,1>>>(p.adjacency_offsets.data,p.adjacency.data,t.block_count,p.coloring.order.data,p.colors.data,p.forbidden.data,p.color_count.data);
        check(cudaMemcpy(&colors,p.color_count.data,sizeof(int),cudaMemcpyDeviceToHost),"ordered color count");
    } else {
        greedy_color<<<1, 1>>>(p.adjacency_offsets.data, p.adjacency.data, t.block_count,
            p.colors.data, p.forbidden.data, p.color_count.data);
        check(cudaMemcpy(&colors, p.color_count.data, sizeof(int), cudaMemcpyDeviceToHost), "GPU color count readback");
    }
    launched("GPU graph coloring");
    if(colors<0||colors>t.block_count||(t.block_count&&colors==0))throw std::runtime_error("invalid PBGS color count");
    if(t.block_count)pbgs_validate<<<blocks(t.block_count),threads>>>(p.adjacency_offsets.data,p.adjacency.data,t.block_count,p.colors.data,colors,p.error.data);
    require_status(p.error,"PBGS full dependency conflict validation");
    p.color_blocks.reserve(t.block_count); p.sorted_colors.reserve(t.block_count); p.color_offsets.reserve(colors + 1);
    check(cudaMemset(p.color_offsets.data, 0, (colors + 1) * sizeof(int)), "clear GPU color counts");
    if (t.block_count) {
        check(cudaMemcpy(p.sorted_colors.data, p.colors.data, t.block_count * sizeof(int), cudaMemcpyDeviceToDevice), "GPU color key copy");
        thrust::sequence(thrust::device, thrust::device_pointer_cast(p.color_blocks.data),
            thrust::device_pointer_cast(p.color_blocks.data + t.block_count));
        thrust::stable_sort_by_key(thrust::device, thrust::device_pointer_cast(p.sorted_colors.data),
            thrust::device_pointer_cast(p.sorted_colors.data + t.block_count), thrust::device_pointer_cast(p.color_blocks.data));
        count_colors<<<blocks(t.block_count), threads>>>(p.colors.data, t.block_count, p.color_offsets.data);
        launched("GPU color histogram");
    }
    auto color_offsets = thrust::device_pointer_cast(p.color_offsets.data);
    thrust::exclusive_scan(thrust::device, color_offsets, color_offsets + colors + 1, color_offsets);
    check(cudaDeviceSynchronize(), "resident broadphase completion");
    }
    if(const char* path=std::getenv("IPC_GPU_COLORING_CSV")) {
        if(!p.coloring.diagnostics.is_open())p.coloring.diagnostics.open(path,std::ios::app);
        if(!p.coloring.diagnostics)throw std::runtime_error("cannot open coloring diagnostics");
        std::vector<int> offsets(colors+1);check(cudaMemcpy(offsets.data(),p.color_offsets.data,offsets.size()*sizeof(int),cudaMemcpyDeviceToHost),"color distribution diagnostic");
        p.coloring.diagnostics<<(t.include_triangle_dependencies?"solver":"guess")<<','<<color_mode<<','<<t.block_count<<','<<graph_edges<<','<<colors<<',';
        for(int c=0;c<colors;++c){if(c)p.coloring.diagnostics<<';';p.coloring.diagnostics<<offsets[c+1]-offsets[c];}
        p.coloring.diagnostics<<','<<vendor_palette_extent<<'\n';p.coloring.diagnostics.flush();
        if(!p.coloring.diagnostics)throw std::runtime_error("cannot write coloring diagnostics");
    }
    p.result = {p.contacts.data, contacts, p.adjacency_offsets.data, p.adjacency.data, graph_edges,
                p.colors.data, p.color_blocks.data, p.color_offsets.data, colors};
    resident_profile_broadphase(contacts,colors,graph_edges);
    return p.result;
}

} // namespace resident
} // namespace ipc_gpu
