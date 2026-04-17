#pragma once
#include "physics.h"
#include "broad_phase.h"
#include <cstddef>
#include <vector>

// --------------------------------------------------------------------------
// DeviceBuffer<T>
// RAII buffer that lives in "device" memory.
//   - On a real CUDA build (gpu_mesh.cu):  cudaMalloc / cudaMemcpy
//   - On the CPU stub   (gpu_mesh_stub.cpp): plain new[] / memcpy
// Non-copyable, movable. Auto-frees on destruction.
// --------------------------------------------------------------------------
template<typename T>
struct DeviceBuffer {
    T*  ptr   = nullptr;
    int count = 0;

    DeviceBuffer()  = default;
    ~DeviceBuffer() { free(); }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), count(o.count) {
        o.ptr = nullptr; o.count = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        free(); ptr = o.ptr; count = o.count; o.ptr = nullptr; o.count = 0; return *this;
    }

    // Allocate n elements (frees any existing allocation first).
    void alloc(int n);
    // Release the allocation.
    void free();
    // Allocate and copy n elements from host.
    void upload(const T* host_data, int n);
    // Copy count elements to host (host_data must point to count*sizeof(T) bytes).
    void download(T* host_data) const;
};

extern template struct DeviceBuffer<int>;
extern template struct DeviceBuffer<double>;
extern template struct DeviceBuffer<float>;

// --------------------------------------------------------------------------
// GPURefMesh
// Device-side mirror of RefMesh.
//
//   tris    : [num_tris * 3]  triangle vertex indices
//   Dm_inv  : [num_tris * 4]  per-triangle Dm^-1, Eigen column-major
//                             layout [a00, a10, a01, a11]
//   area    : [num_tris]      undeformed triangle areas
//   mass    : [num_verts]     lumped vertex masses
//
// Hinge data (discrete-shell bending, only populated when kB > 0):
//   hinge_v         : [num_hinges * 4]  vertex indices per hinge [v0,v1,v2,v3]
//   hinge_bar_theta : [num_hinges]      rest dihedral complement
//   hinge_ce        : [num_hinges]      |e|^2 / (A_A + A_B)
//
// Hinge adjacency (CSR): vertex vi's hinges at
//   [hinge_adj_offsets[vi] .. hinge_adj_offsets[vi+1])
//   hinge_adj_hi   : hinge index
//   hinge_adj_role : local role in hinge (0-3)
// --------------------------------------------------------------------------
struct GPURefMesh {
    int num_verts   = 0;
    int num_tris    = 0;
    int num_hinges  = 0;

    DeviceBuffer<int>    tris;    // num_tris * 3
    DeviceBuffer<double> Dm_inv;  // num_tris * 4
    DeviceBuffer<double> area;    // num_tris
    DeviceBuffer<double> mass;    // num_verts

    DeviceBuffer<int>    hinge_v;           // num_hinges * 4
    DeviceBuffer<double> hinge_bar_theta;   // num_hinges
    DeviceBuffer<double> hinge_ce;          // num_hinges

    DeviceBuffer<int>    hinge_adj_offsets; // num_verts + 1
    DeviceBuffer<int>    hinge_adj_hi;      // total hinge entries
    DeviceBuffer<int>    hinge_adj_role;    // total hinge entries

    void upload(const RefMesh& cpu);
};

// --------------------------------------------------------------------------
// GPUDeformedState
// Device-side mirror of DeformedState.
//
//   positions  : [num_verts * 3]  interleaved x, y, z
//   velocities : [num_verts * 3]  interleaved x, y, z
// --------------------------------------------------------------------------
struct GPUDeformedState {
    int num_verts = 0;

    DeviceBuffer<double> positions;   // num_verts * 3
    DeviceBuffer<double> velocities;  // num_verts * 3

    void upload(const DeformedState& cpu);
    void download(DeformedState& cpu) const;
};

// --------------------------------------------------------------------------
// GPUPins
// Device-side mirror of std::vector<Pin>.
//
//   indices : [count]      vertex indices of pinned vertices
//   targets : [count * 3]  pin target positions, interleaved x, y, z
// --------------------------------------------------------------------------
struct GPUPins {
    int count = 0;

    DeviceBuffer<int>    indices;  // count
    DeviceBuffer<double> targets;  // count * 3

    void upload(const std::vector<Pin>& cpu);
};

// --------------------------------------------------------------------------
// GPUAdjacency
// Device-side mirror of VertexTriangleMap (CSR format).
//
// For vertex vi, its incident (triangle, local_corner) pairs are stored at:
//   tri_idx  [offsets[vi] .. offsets[vi+1])
//   tri_local[offsets[vi] .. offsets[vi+1])
//
//   offsets   : [num_verts + 1]
//   tri_idx   : [total_entries]  triangle index for each entry
//   tri_local : [total_entries]  local corner (0,1,2) for each entry
// --------------------------------------------------------------------------
struct GPUAdjacency {
    int num_verts = 0;

    DeviceBuffer<int> offsets;    // num_verts + 1
    DeviceBuffer<int> tri_idx;    // total_entries
    DeviceBuffer<int> tri_local;  // total_entries

    void upload(const VertexTriangleMap& adj, int nv);
};

// --------------------------------------------------------------------------
// GPURefPositions
// Device-side mirror of std::vector<Vec2> X (2-D rest positions).
//
//   data : [count * 2]  interleaved x, y
// --------------------------------------------------------------------------
struct GPURefPositions {
    int count = 0;

    DeviceBuffer<double> data;  // count * 2

    void upload(const std::vector<Vec2>& cpu);
};

// --------------------------------------------------------------------------
// GPUPinMap
// Device-side mirror of PinMap (vector<int>).
// pin_map[vi] = index into pins array, or -1 if vertex is not pinned.
//
//   data : [num_verts]
// --------------------------------------------------------------------------
struct GPUPinMap {
    int num_verts = 0;

    DeviceBuffer<int> data;  // num_verts

    void upload(const PinMap& pm);
};

// --------------------------------------------------------------------------
// GPUSimParams
// Plain POD mirror of SimParams — safe to pass by value into a kernel.
// dt and dt2 are precomputed on upload so no division happens in device code.
// --------------------------------------------------------------------------
struct GPUSimParams {
    double dt_val;
    double dt2_val;
    double mu, lambda, density, thickness, kpin, kB;
    double d_hat;
    double gx, gy, gz;
    double tol_abs, tol_rel, step_weight;
    int    max_global_iters;
    bool   use_parallel;
    bool   ccd_check;
    bool   use_trust_region;
    bool   use_incremental_refresh;
    bool   mass_normalize_residual;

    double dt()  const { return dt_val;  }
    double dt2() const { return dt2_val; }

    static GPUSimParams from(const SimParams& p);
};

// --------------------------------------------------------------------------
// GPUBroadPhaseCache
// Device-side mirror of BroadPhase::Cache.
//
// Pair lists (flat):
//   nt_data : [num_nt * 4]   node-triangle pairs, layout per pair:
//             [node, tri_v0, tri_v1, tri_v2]
//   ss_data : [num_ss * 4]   segment-segment pairs, layout per pair:
//             [v0, v1, v2, v3]
//
// Per-vertex pair lookups (CSR, same pattern as GPUAdjacency):
//   vnt_offsets  : [num_verts + 1]
//   vnt_pair_idx : [total_nt_entries]  index into nt_data (pair index)
//   vnt_dof      : [total_nt_entries]  role of vi in this pair (0-3)
//
//   vss_offsets  : [num_verts + 1]
//   vss_pair_idx : [total_ss_entries]  index into ss_data (pair index)
//   vss_dof      : [total_ss_entries]  role of vi in this pair (0-3)
// --------------------------------------------------------------------------
struct GPUBroadPhaseCache {
    int num_verts = 0;
    int num_nt    = 0;
    int num_ss    = 0;

    DeviceBuffer<int> nt_data;       // num_nt * 4
    DeviceBuffer<int> ss_data;       // num_ss * 4

    DeviceBuffer<int> vnt_offsets;   // num_verts + 1
    DeviceBuffer<int> vnt_pair_idx;  // total NT entries
    DeviceBuffer<int> vnt_dof;       // total NT entries

    DeviceBuffer<int> vss_offsets;   // num_verts + 1
    DeviceBuffer<int> vss_pair_idx;  // total SS entries
    DeviceBuffer<int> vss_dof;       // total SS entries

    void upload(const BroadPhase::Cache& cache, int nv);
};
