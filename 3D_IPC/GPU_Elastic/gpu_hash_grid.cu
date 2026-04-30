// ============================================================================
// GPU Hash Grid — uniform spatial hash broad phase for NT (node-triangle)
// pair extraction. Mirrors CPU `BroadPhase::initialize(blue_boxes, mesh, d_hat)`
// for the gauss_seidel_basic solver path.
//
// Pipeline (one substep cost; not run per iter):
//
//   1. Compute mesh AABB → grid origin + dimensions
//   2. cell_size = 2 * node_box_size + d_hat   (worst-case overlap diameter)
//   3. Per-triangle: count cells the red box overlaps  (count-tri kernel)
//   4. Exclusive prefix sum of counts → scatter offsets   (cub)
//   5. Per-triangle: scatter (cell_id, tri_id) tuples     (scatter kernel)
//   6. Sort tuples by cell_id                              (cub::DeviceRadixSort)
//   7. Build per-cell [start, end) ranges                  (range kernel)
//   8. Per-vertex: walk cells in blue box, narrow-phase AABB overlap, emit
//      pair candidates with global atomic counter          (query kernel)
//   9. Sort + unique the candidate list to dedupe         (cub)
//   10. Copy back to host as NodeTrianglePair[]
// ============================================================================

#include "gpu_hash_grid.h"
#include "../physics.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
struct GridParams {
    double ox, oy, oz;     // origin
    double cell_size;
    int    nx, ny, nz;     // dims (cells)
    int    n_cells;        // nx * ny * nz
};

__device__ __host__ inline int floor_div(double v, double s) {
    // floor toward -inf, returns int
    return static_cast<int>(floor(v / s));
}

__device__ __host__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __host__ inline int cell_idx(int cx, int cy, int cz, GridParams g) {
    return cx + cy * g.nx + cz * g.nx * g.ny;
}

// AABB-AABB overlap test (closed intervals), all double.
__device__ inline bool aabb_overlap(
    double amn_x, double amn_y, double amn_z,
    double amx_x, double amx_y, double amx_z,
    double bmn_x, double bmn_y, double bmn_z,
    double bmx_x, double bmx_y, double bmx_z)
{
    return amx_x >= bmn_x && amn_x <= bmx_x
        && amx_y >= bmn_y && amn_y <= bmx_y
        && amx_z >= bmn_z && amn_z <= bmx_z;
}

// Red triangle box = union of 3 per-vertex blue boxes, then padded by d_hat.
// Each vertex's blue box uses its own radius from R[]. Done in CPU's order
// (blue box first, then d_hat pad) for FP bit-identity with the CPU side.
__device__ inline void red_tri_box(
    const double* __restrict__ X, const double* __restrict__ R,
    int v0, int v1, int v2, double d_hat,
    double& mn_x, double& mn_y, double& mn_z,
    double& mx_x, double& mx_y, double& mx_z)
{
    const double p0x = X[3*v0+0], p0y = X[3*v0+1], p0z = X[3*v0+2];
    const double p1x = X[3*v1+0], p1y = X[3*v1+1], p1z = X[3*v1+2];
    const double p2x = X[3*v2+0], p2y = X[3*v2+1], p2z = X[3*v2+2];
    const double r0 = R[v0], r1 = R[v1], r2 = R[v2];
    mn_x = fmin(fmin(p0x - r0, p1x - r1), p2x - r2) - d_hat;
    mn_y = fmin(fmin(p0y - r0, p1y - r1), p2y - r2) - d_hat;
    mn_z = fmin(fmin(p0z - r0, p1z - r1), p2z - r2) - d_hat;
    mx_x = fmax(fmax(p0x + r0, p1x + r1), p2x + r2) + d_hat;
    mx_y = fmax(fmax(p0y + r0, p1y + r1), p2y + r2) + d_hat;
    mx_z = fmax(fmax(p0z + r0, p1z + r1), p2z + r2) + d_hat;
}

// Red edge box = union of 2 per-vertex blue boxes, then padded by d_hat.
__device__ inline void red_edge_box(
    const double* __restrict__ X, const double* __restrict__ R,
    int v0, int v1, double d_hat,
    double& mn_x, double& mn_y, double& mn_z,
    double& mx_x, double& mx_y, double& mx_z)
{
    const double p0x = X[3*v0+0], p0y = X[3*v0+1], p0z = X[3*v0+2];
    const double p1x = X[3*v1+0], p1y = X[3*v1+1], p1z = X[3*v1+2];
    const double r0 = R[v0], r1 = R[v1];
    mn_x = fmin(p0x - r0, p1x - r1) - d_hat;
    mn_y = fmin(p0y - r0, p1y - r1) - d_hat;
    mn_z = fmin(p0z - r0, p1z - r1) - d_hat;
    mx_x = fmax(p0x + r0, p1x + r1) + d_hat;
    mx_y = fmax(p0y + r0, p1y + r1) + d_hat;
    mx_z = fmax(p0z + r0, p1z + r1) + d_hat;
}

// ---------------------------------------------------------------------------
// Kernel 1: count cells per triangle
// ---------------------------------------------------------------------------
__global__ void k_count_tri_cells(
    int n_tri, const int* __restrict__ tris,
    const double* __restrict__ X, const double* __restrict__ R, double d_hat, GridParams g,
    int* __restrict__ counts)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tri) return;

    const int v0 = tris[3*t+0];
    const int v1 = tris[3*t+1];
    const int v2 = tris[3*t+2];

    double mn_x, mn_y, mn_z, mx_x, mx_y, mx_z;
    red_tri_box(X, R, v0, v1, v2, d_hat, mn_x, mn_y, mn_z, mx_x, mx_y, mx_z);

    const int lo_x = clampi(floor_div(mn_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int lo_y = clampi(floor_div(mn_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int lo_z = clampi(floor_div(mn_z - g.oz, g.cell_size), 0, g.nz - 1);
    const int hi_x = clampi(floor_div(mx_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int hi_y = clampi(floor_div(mx_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int hi_z = clampi(floor_div(mx_z - g.oz, g.cell_size), 0, g.nz - 1);

    counts[t] = (hi_x - lo_x + 1) * (hi_y - lo_y + 1) * (hi_z - lo_z + 1);
}

// ---------------------------------------------------------------------------
// Kernel 2: scatter (cell_id, tri_id) per triangle into pre-allocated slots
// ---------------------------------------------------------------------------
__global__ void k_scatter_tri_cells(
    int n_tri, const int* __restrict__ tris,
    const double* __restrict__ X, const double* __restrict__ R, double d_hat, GridParams g,
    const int* __restrict__ offsets,
    int* __restrict__ out_cell_ids, int* __restrict__ out_tri_ids)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tri) return;

    const int v0 = tris[3*t+0];
    const int v1 = tris[3*t+1];
    const int v2 = tris[3*t+2];

    double mn_x, mn_y, mn_z, mx_x, mx_y, mx_z;
    red_tri_box(X, R, v0, v1, v2, d_hat, mn_x, mn_y, mn_z, mx_x, mx_y, mx_z);

    const int lo_x = clampi(floor_div(mn_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int lo_y = clampi(floor_div(mn_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int lo_z = clampi(floor_div(mn_z - g.oz, g.cell_size), 0, g.nz - 1);
    const int hi_x = clampi(floor_div(mx_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int hi_y = clampi(floor_div(mx_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int hi_z = clampi(floor_div(mx_z - g.oz, g.cell_size), 0, g.nz - 1);

    int p = offsets[t];
    for (int cz = lo_z; cz <= hi_z; ++cz)
    for (int cy = lo_y; cy <= hi_y; ++cy)
    for (int cx = lo_x; cx <= hi_x; ++cx) {
        out_cell_ids[p] = cell_idx(cx, cy, cz, g);
        out_tri_ids[p]  = t;
        ++p;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: build [start, end) ranges per cell from sorted cell_ids.
// Initialized with cell_start = -1 / cell_end = 0; written only at boundaries.
// ---------------------------------------------------------------------------
__global__ void k_build_cell_ranges(
    int n_entries, const int* __restrict__ sorted_cell_ids,
    int* __restrict__ cell_start, int* __restrict__ cell_end)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_entries) return;
    const int c = sorted_cell_ids[i];
    if (i == 0 || sorted_cell_ids[i-1] != c)        cell_start[c] = i;
    if (i == n_entries-1 || sorted_cell_ids[i+1] != c) cell_end[c] = i + 1;
}

// ---------------------------------------------------------------------------
// Kernel 4a: per-vertex query (count only) — walk cells in blue box,
// AABB-overlap each triangle, self-incidence skip, write count.
// Counts may include duplicates (same (vi, t) found via multiple cells);
// dedupe is on host after emit.
// ---------------------------------------------------------------------------
// Warp-per-vertex variant: 32 threads cooperate on one vertex's query work.
// All threads in the warp visit the same cells in lockstep. Within each cell,
// candidates are striped across the 32 lanes. At the end the local counts are
// warp-reduced via __shfl_xor and lane 0 writes the total. Intent: trade
// thread-level parallelism (1 thread/vertex) for warp-cooperative work
// distribution that better tolerates per-cell candidate variance and amortizes
// the cell_start/cell_end loads.
__global__ void k_query_count(
    int n_vert, const double* __restrict__ X, const double* __restrict__ R, double d_hat,
    GridParams g, const int* __restrict__ cell_start,
    const int* __restrict__ cell_end, const int* __restrict__ sorted_tri_ids,
    const int* __restrict__ tris,
    int* __restrict__ out_counts)
{
    const int vi   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // /32
    const int lane = threadIdx.x & 31;
    if (vi >= n_vert) return;

    const double vx = X[3*vi+0], vy = X[3*vi+1], vz = X[3*vi+2];
    const double r = R[vi];
    const double bmn_x = vx - r, bmn_y = vy - r, bmn_z = vz - r;
    const double bmx_x = vx + r, bmx_y = vy + r, bmx_z = vz + r;

    const int lo_x = clampi(floor_div(bmn_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int lo_y = clampi(floor_div(bmn_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int lo_z = clampi(floor_div(bmn_z - g.oz, g.cell_size), 0, g.nz - 1);
    const int hi_x = clampi(floor_div(bmx_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int hi_y = clampi(floor_div(bmx_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int hi_z = clampi(floor_div(bmx_z - g.oz, g.cell_size), 0, g.nz - 1);

    int n = 0;
    for (int cz = lo_z; cz <= hi_z; ++cz)
    for (int cy = lo_y; cy <= hi_y; ++cy)
    for (int cx = lo_x; cx <= hi_x; ++cx) {
        const int cell = cell_idx(cx, cy, cz, g);
        const int s = cell_start[cell];
        if (s < 0) continue;
        const int e = cell_end[cell];
        // Stripe candidates across the 32 lanes.
        for (int k = s + lane; k < e; k += 32) {
            const int t = sorted_tri_ids[k];
            const int tv0 = tris[3*t+0];
            const int tv1 = tris[3*t+1];
            const int tv2 = tris[3*t+2];
            if (vi == tv0 || vi == tv1 || vi == tv2) continue;
            double tmn_x, tmn_y, tmn_z, tmx_x, tmx_y, tmx_z;
            red_tri_box(X, R, tv0, tv1, tv2, d_hat,
                        tmn_x, tmn_y, tmn_z, tmx_x, tmx_y, tmx_z);
            if (!aabb_overlap(bmn_x, bmn_y, bmn_z, bmx_x, bmx_y, bmx_z,
                              tmn_x, tmn_y, tmn_z, tmx_x, tmx_y, tmx_z))
                continue;
            ++n;
        }
    }
    // Warp-reduce
    for (int o = 16; o > 0; o >>= 1) n += __shfl_xor_sync(0xffffffff, n, o);
    if (lane == 0) out_counts[vi] = n;
}

// ---------------------------------------------------------------------------
// Kernel 4b: per-vertex query (emit) — same walk, write into pre-allocated
// per-vertex slot at offsets[vi] .. offsets[vi+1]-1.
// ---------------------------------------------------------------------------
// Warp-per-vertex emit. 32 lanes cooperate. Per-iteration warp ballot + popc
// gives each emitting lane its slot offset within the warp's allocation. The
// warp's running write pointer is held in lane 0 (broadcast each iter) so we
// don't need shared memory.
__global__ void k_query_emit(
    int n_vert, const double* __restrict__ X, const double* __restrict__ R, double d_hat,
    GridParams g, const int* __restrict__ cell_start,
    const int* __restrict__ cell_end, const int* __restrict__ sorted_tri_ids,
    const int* __restrict__ tris, const int* __restrict__ offsets,
    int* __restrict__ out_node, int* __restrict__ out_tri)
{
    const int vi   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (vi >= n_vert) return;

    const double vx = X[3*vi+0], vy = X[3*vi+1], vz = X[3*vi+2];
    const double r = R[vi];
    const double bmn_x = vx - r, bmn_y = vy - r, bmn_z = vz - r;
    const double bmx_x = vx + r, bmx_y = vy + r, bmx_z = vz + r;

    const int lo_x = clampi(floor_div(bmn_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int lo_y = clampi(floor_div(bmn_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int lo_z = clampi(floor_div(bmn_z - g.oz, g.cell_size), 0, g.nz - 1);
    const int hi_x = clampi(floor_div(bmx_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int hi_y = clampi(floor_div(bmx_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int hi_z = clampi(floor_div(bmx_z - g.oz, g.cell_size), 0, g.nz - 1);

    // Warp's running write pointer — held in lane 0, broadcast to all lanes.
    int warp_p = offsets[vi];

    for (int cz = lo_z; cz <= hi_z; ++cz)
    for (int cy = lo_y; cy <= hi_y; ++cy)
    for (int cx = lo_x; cx <= hi_x; ++cx) {
        const int cell = cell_idx(cx, cy, cz, g);
        const int s = cell_start[cell];
        if (s < 0) continue;
        const int e = cell_end[cell];
        // Stripe candidates across lanes; each iteration handles up to 32.
        for (int base = s; base < e; base += 32) {
            const int k = base + lane;
            int t = -1;
            bool emit = false;
            if (k < e) {
                t = sorted_tri_ids[k];
                const int tv0 = tris[3*t+0];
                const int tv1 = tris[3*t+1];
                const int tv2 = tris[3*t+2];
                if (vi != tv0 && vi != tv1 && vi != tv2) {
                    double tmn_x, tmn_y, tmn_z, tmx_x, tmx_y, tmx_z;
                    red_tri_box(X, R, tv0, tv1, tv2, d_hat,
                                tmn_x, tmn_y, tmn_z, tmx_x, tmx_y, tmx_z);
                    emit = aabb_overlap(bmn_x, bmn_y, bmn_z, bmx_x, bmx_y, bmx_z,
                                        tmn_x, tmn_y, tmn_z, tmx_x, tmx_y, tmx_z);
                }
            }
            const unsigned ballot = __ballot_sync(0xffffffff, emit);
            if (emit) {
                const int prefix = __popc(ballot & ((1u << lane) - 1u));
                const int slot = warp_p + prefix;
                out_node[slot] = vi;
                out_tri[slot]  = t;
            }
            warp_p += __popc(ballot);
        }
    }
}

// ===========================================================================
// SS pair pipeline — same shape as NT, but on edges instead of triangles.
// Edge primitive AABB = red edge box (union of 2 blue boxes + d_hat pad).
// Query: per edge, walk cells of its red box, find other edges, AABB overlap
// (red×red), shared-vertex skip, emit candidate. Pairs deduped on host.
// ===========================================================================

// Kernel: count cells per edge red box.
__global__ void k_count_edge_cells(
    int n_edges, const int* __restrict__ edges,
    const double* __restrict__ X, const double* __restrict__ R, double d_hat, GridParams g,
    int* __restrict__ counts)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    const int v0 = edges[2*e+0];
    const int v1 = edges[2*e+1];
    double mn_x, mn_y, mn_z, mx_x, mx_y, mx_z;
    red_edge_box(X, R, v0, v1, d_hat, mn_x, mn_y, mn_z, mx_x, mx_y, mx_z);
    const int lo_x = clampi(floor_div(mn_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int lo_y = clampi(floor_div(mn_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int lo_z = clampi(floor_div(mn_z - g.oz, g.cell_size), 0, g.nz - 1);
    const int hi_x = clampi(floor_div(mx_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int hi_y = clampi(floor_div(mx_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int hi_z = clampi(floor_div(mx_z - g.oz, g.cell_size), 0, g.nz - 1);
    counts[e] = (hi_x - lo_x + 1) * (hi_y - lo_y + 1) * (hi_z - lo_z + 1);
}

// Kernel: scatter (cell_id, edge_id) pairs.
__global__ void k_scatter_edge_cells(
    int n_edges, const int* __restrict__ edges,
    const double* __restrict__ X, const double* __restrict__ R, double d_hat, GridParams g,
    const int* __restrict__ offsets,
    int* __restrict__ out_cell_ids, int* __restrict__ out_edge_ids)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    const int v0 = edges[2*e+0];
    const int v1 = edges[2*e+1];
    double mn_x, mn_y, mn_z, mx_x, mx_y, mx_z;
    red_edge_box(X, R, v0, v1, d_hat, mn_x, mn_y, mn_z, mx_x, mx_y, mx_z);
    const int lo_x = clampi(floor_div(mn_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int lo_y = clampi(floor_div(mn_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int lo_z = clampi(floor_div(mn_z - g.oz, g.cell_size), 0, g.nz - 1);
    const int hi_x = clampi(floor_div(mx_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int hi_y = clampi(floor_div(mx_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int hi_z = clampi(floor_div(mx_z - g.oz, g.cell_size), 0, g.nz - 1);
    int p = offsets[e];
    for (int cz = lo_z; cz <= hi_z; ++cz)
    for (int cy = lo_y; cy <= hi_y; ++cy)
    for (int cx = lo_x; cx <= hi_x; ++cx) {
        out_cell_ids[p] = cell_idx(cx, cy, cz, g);
        out_edge_ids[p] = e;
        ++p;
    }
}

// Kernel: per-edge query (count) — walk cells in this edge's red box, check
// every other edge, AABB overlap (red×red), shared-vertex skip. Edge pair
// (a, b) is registered only when a < b to avoid double-counting; both
// orientations would otherwise visit the same pair.
__global__ void k_query_count_ss(
    int n_edges, const int* __restrict__ edges,
    const double* __restrict__ X, const double* __restrict__ R, double d_hat, GridParams g,
    const int* __restrict__ cell_start, const int* __restrict__ cell_end,
    const int* __restrict__ sorted_edge_ids,
    int* __restrict__ out_counts)
{
    const int ea   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (ea >= n_edges) return;
    const int a0 = edges[2*ea+0], a1 = edges[2*ea+1];
    double amn_x, amn_y, amn_z, amx_x, amx_y, amx_z;
    red_edge_box(X, R, a0, a1, d_hat, amn_x, amn_y, amn_z, amx_x, amx_y, amx_z);
    const int lo_x = clampi(floor_div(amn_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int lo_y = clampi(floor_div(amn_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int lo_z = clampi(floor_div(amn_z - g.oz, g.cell_size), 0, g.nz - 1);
    const int hi_x = clampi(floor_div(amx_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int hi_y = clampi(floor_div(amx_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int hi_z = clampi(floor_div(amx_z - g.oz, g.cell_size), 0, g.nz - 1);

    int n = 0;
    for (int cz = lo_z; cz <= hi_z; ++cz)
    for (int cy = lo_y; cy <= hi_y; ++cy)
    for (int cx = lo_x; cx <= hi_x; ++cx) {
        const int cell = cell_idx(cx, cy, cz, g);
        const int s = cell_start[cell];
        if (s < 0) continue;
        const int e = cell_end[cell];
        for (int k = s + lane; k < e; k += 32) {
            const int eb = sorted_edge_ids[k];
            if (eb <= ea) continue;
            const int b0 = edges[2*eb+0], b1 = edges[2*eb+1];
            if (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1) continue;
            double bmn_x, bmn_y, bmn_z, bmx_x, bmx_y, bmx_z;
            red_edge_box(X, R, b0, b1, d_hat, bmn_x, bmn_y, bmn_z, bmx_x, bmx_y, bmx_z);
            if (!aabb_overlap(amn_x, amn_y, amn_z, amx_x, amx_y, amx_z,
                              bmn_x, bmn_y, bmn_z, bmx_x, bmx_y, bmx_z))
                continue;
            ++n;
        }
    }
    for (int o = 16; o > 0; o >>= 1) n += __shfl_xor_sync(0xffffffff, n, o);
    if (lane == 0) out_counts[ea] = n;
}

// Warp-per-edge emit. Mirrors NT emit: ballot + popc gives each emitting
// lane its slot offset within the warp's pre-allocated range.
__global__ void k_query_emit_ss(
    int n_edges, const int* __restrict__ edges,
    const double* __restrict__ X, const double* __restrict__ R, double d_hat, GridParams g,
    const int* __restrict__ cell_start, const int* __restrict__ cell_end,
    const int* __restrict__ sorted_edge_ids,
    const int* __restrict__ offsets,
    int* __restrict__ out_a, int* __restrict__ out_b)
{
    const int ea   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (ea >= n_edges) return;
    const int a0 = edges[2*ea+0], a1 = edges[2*ea+1];
    double amn_x, amn_y, amn_z, amx_x, amx_y, amx_z;
    red_edge_box(X, R, a0, a1, d_hat, amn_x, amn_y, amn_z, amx_x, amx_y, amx_z);
    const int lo_x = clampi(floor_div(amn_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int lo_y = clampi(floor_div(amn_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int lo_z = clampi(floor_div(amn_z - g.oz, g.cell_size), 0, g.nz - 1);
    const int hi_x = clampi(floor_div(amx_x - g.ox, g.cell_size), 0, g.nx - 1);
    const int hi_y = clampi(floor_div(amx_y - g.oy, g.cell_size), 0, g.ny - 1);
    const int hi_z = clampi(floor_div(amx_z - g.oz, g.cell_size), 0, g.nz - 1);

    int warp_p = offsets[ea];
    for (int cz = lo_z; cz <= hi_z; ++cz)
    for (int cy = lo_y; cy <= hi_y; ++cy)
    for (int cx = lo_x; cx <= hi_x; ++cx) {
        const int cell = cell_idx(cx, cy, cz, g);
        const int s = cell_start[cell];
        if (s < 0) continue;
        const int e = cell_end[cell];
        for (int base = s; base < e; base += 32) {
            const int k = base + lane;
            int eb = -1;
            bool emit = false;
            if (k < e) {
                eb = sorted_edge_ids[k];
                if (eb > ea) {
                    const int b0 = edges[2*eb+0], b1 = edges[2*eb+1];
                    if (a0 != b0 && a0 != b1 && a1 != b0 && a1 != b1) {
                        double bmn_x, bmn_y, bmn_z, bmx_x, bmx_y, bmx_z;
                        red_edge_box(X, R, b0, b1, d_hat, bmn_x, bmn_y, bmn_z, bmx_x, bmx_y, bmx_z);
                        emit = aabb_overlap(amn_x, amn_y, amn_z, amx_x, amx_y, amx_z,
                                            bmn_x, bmn_y, bmn_z, bmx_x, bmx_y, bmx_z);
                    }
                }
            }
            const unsigned ballot = __ballot_sync(0xffffffff, emit);
            if (emit) {
                const int prefix = __popc(ballot & ((1u << lane) - 1u));
                const int slot = warp_p + prefix;
                out_a[slot] = ea;
                out_b[slot] = eb;
            }
            warp_p += __popc(ballot);
        }
    }
}

// ---------------------------------------------------------------------------
// Pack two int32 IDs into a uint64 key for radix sort + unique dedup.
// High 32 bits = first id, low 32 = second.
// ---------------------------------------------------------------------------
__global__ void k_pack_keys(int n, const int* __restrict__ a,
                            const int* __restrict__ b,
                            unsigned long long* __restrict__ keys)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    keys[i] = (static_cast<unsigned long long>(static_cast<unsigned int>(a[i])) << 32)
            |  static_cast<unsigned long long>(static_cast<unsigned int>(b[i]));
}

// ---------------------------------------------------------------------------
// Cuda error helper
// ---------------------------------------------------------------------------
inline void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[gpu_hash_grid] %s: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}

// Stream-ordered allocation. CUDA 11.2+ has a memory pool: subsequent
// d_alloc calls reuse memory freed by d_free at near-zero cost. Avoids
// the per-call cudaMalloc/cudaFree storm. Stream 0 = legacy default; the
// pool still works since allocations are pool-tracked, not stream-bound.
template<typename T>
inline cudaError_t d_alloc(T** p, std::size_t bytes) {
    return cudaMallocAsync(reinterpret_cast<void**>(p), bytes, 0);
}
inline cudaError_t d_free(void* p) {
    return cudaFreeAsync(p, 0);
}

}  // namespace

// ===========================================================================
// Host-side orchestrator
// ===========================================================================
GpuBroadPhaseResult gpu_hash_grid_build_pairs(
    const std::vector<Vec3>&    positions,
    const RefMesh&              ref_mesh,
    const std::vector<double>&  per_vertex_radii,
    double                      d_hat)
{
    GpuBroadPhaseResult result;
    const int nv = static_cast<int>(positions.size());
    const int nt = static_cast<int>(ref_mesh.tris.size()) / 3;
    if (nv == 0 || nt == 0) return result;
    if (static_cast<int>(per_vertex_radii.size()) != nv) {
        std::fprintf(stderr,
            "[gpu_hash_grid] per_vertex_radii size %zu != positions size %d\n",
            per_vertex_radii.size(), nv);
        std::abort();
    }

    // Instrumentation: GPU_HG_PROFILE=1 prints per-call phase timings.
    const bool prof = (std::getenv("GPU_HG_PROFILE") != nullptr);
    using Clock = std::chrono::high_resolution_clock;
    auto t_start = Clock::now();
    double t_setup = 0, t_nt = 0, t_edges = 0, t_ss = 0, t_dedup = 0;
    int n_nt_emitted_total = 0, n_ss_emitted_total = 0;
    auto mark = [&prof](const char* what, Clock::time_point& t0, double& acc) {
        if (!prof) { t0 = Clock::now(); return; }
        cudaDeviceSynchronize();
        auto t1 = Clock::now();
        acc += std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
    };
    auto t_phase = Clock::now();

    // -----------------------------------------------------------------------
    // Compute mesh AABB → grid origin + dims. Per-vertex pad uses each
    // vertex's own radius; for grid sizing we conservatively use max radius.
    // -----------------------------------------------------------------------
    double max_r = 0.0;
    for (double r : per_vertex_radii) max_r = std::max(max_r, r);

    Vec3 mn = positions[0];
    Vec3 mx = positions[0];
    for (int i = 1; i < nv; ++i) {
        mn = mn.cwiseMin(positions[i]);
        mx = mx.cwiseMax(positions[i]);
    }
    // Pad mesh AABB by worst-case red box extent so every primitive's red
    // box lies inside the grid. Red box = blue box (±radius) + d_hat.
    const double pad = max_r + d_hat;
    mn.array() -= pad;
    mx.array() += pad;

    GridParams g{};
    // Cell size scales with worst-case overlap diameter to keep cell counts
    // bounded for the hot loops that walk per-primitive cell ranges.
    g.cell_size = 2.0 * max_r + d_hat;
    if (g.cell_size <= 0.0) g.cell_size = 1.0;  // degenerate: all radii zero
    g.ox = mn(0); g.oy = mn(1); g.oz = mn(2);
    g.nx = std::max(1, static_cast<int>(std::ceil((mx(0) - mn(0)) / g.cell_size)) + 1);
    g.ny = std::max(1, static_cast<int>(std::ceil((mx(1) - mn(1)) / g.cell_size)) + 1);
    g.nz = std::max(1, static_cast<int>(std::ceil((mx(2) - mn(2)) / g.cell_size)) + 1);
    g.n_cells = g.nx * g.ny * g.nz;

    // -----------------------------------------------------------------------
    // Upload positions (as flat double[]) and triangle indices.
    // -----------------------------------------------------------------------
    std::vector<double> h_X(3 * nv);
    for (int i = 0; i < nv; ++i) {
        h_X[3*i+0] = positions[i](0);
        h_X[3*i+1] = positions[i](1);
        h_X[3*i+2] = positions[i](2);
    }

    double* d_X = nullptr;
    double* d_R = nullptr;
    int*    d_tris = nullptr;
    cudaCheck(d_alloc(&d_X, sizeof(double) * 3 * nv), "malloc d_X");
    cudaCheck(d_alloc(&d_R, sizeof(double) * nv),     "malloc d_R");
    cudaCheck(d_alloc(&d_tris, sizeof(int) * 3 * nt), "malloc d_tris");
    cudaCheck(cudaMemcpy(d_X, h_X.data(), sizeof(double) * 3 * nv,
                         cudaMemcpyHostToDevice), "memcpy d_X");
    cudaCheck(cudaMemcpy(d_R, per_vertex_radii.data(), sizeof(double) * nv,
                         cudaMemcpyHostToDevice), "memcpy d_R");
    cudaCheck(cudaMemcpy(d_tris, ref_mesh.tris.data(), sizeof(int) * 3 * nt,
                         cudaMemcpyHostToDevice), "memcpy d_tris");

    // -----------------------------------------------------------------------
    // Pass 1: per-triangle, count cells overlapped.
    // -----------------------------------------------------------------------
    int* d_counts = nullptr;
    cudaCheck(d_alloc(&d_counts, sizeof(int) * (nt + 1)), "malloc d_counts");
    {
        const int block = 256;
        const int grid_b = (nt + block - 1) / block;
        k_count_tri_cells<<<grid_b, block>>>(nt, d_tris, d_X, d_R, d_hat, g, d_counts);
        cudaCheck(cudaGetLastError(), "k_count_tri_cells");
    }

    // -----------------------------------------------------------------------
    // Exclusive prefix sum → offsets[]; total = offsets[nt].
    // cub::DeviceScan needs a temporary buffer; query size, then run.
    // -----------------------------------------------------------------------
    int* d_offsets = nullptr;
    cudaCheck(d_alloc(&d_offsets, sizeof(int) * (nt + 1)), "malloc d_offsets");
    void* d_tmp = nullptr;
    std::size_t tmp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, d_counts, d_offsets, nt + 1);
    cudaCheck(d_alloc(&d_tmp, tmp_bytes), "malloc cub tmp1");
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_counts, d_offsets, nt + 1);
    cudaCheck(d_free(d_tmp), "free cub tmp1");

    int total_entries = 0;
    cudaCheck(cudaMemcpy(&total_entries, d_offsets + nt, sizeof(int),
                         cudaMemcpyDeviceToHost), "read total_entries");

    // -----------------------------------------------------------------------
    // Pass 2: scatter (cell_id, tri_id) tuples.
    // -----------------------------------------------------------------------
    int* d_cell_ids = nullptr;
    int* d_tri_ids  = nullptr;
    int* d_cell_ids_sorted = nullptr;
    int* d_tri_ids_sorted  = nullptr;
    cudaCheck(d_alloc(&d_cell_ids, sizeof(int) * total_entries), "malloc cell_ids");
    cudaCheck(d_alloc(&d_tri_ids,  sizeof(int) * total_entries), "malloc tri_ids");
    cudaCheck(d_alloc(&d_cell_ids_sorted, sizeof(int) * total_entries), "malloc cell_ids_s");
    cudaCheck(d_alloc(&d_tri_ids_sorted,  sizeof(int) * total_entries), "malloc tri_ids_s");
    {
        const int block = 256;
        const int grid_b = (nt + block - 1) / block;
        k_scatter_tri_cells<<<grid_b, block>>>(nt, d_tris, d_X, d_R, d_hat, g,
                                               d_offsets, d_cell_ids, d_tri_ids);
        cudaCheck(cudaGetLastError(), "k_scatter_tri_cells");
    }

    // -----------------------------------------------------------------------
    // Sort (cell_id, tri_id) by cell_id. cub::DeviceRadixSort sorts pairs.
    // -----------------------------------------------------------------------
    {
        d_tmp = nullptr; tmp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, tmp_bytes,
            d_cell_ids, d_cell_ids_sorted,
            d_tri_ids,  d_tri_ids_sorted,
            total_entries);
        cudaCheck(d_alloc(&d_tmp, tmp_bytes), "malloc cub tmp2");
        cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes,
            d_cell_ids, d_cell_ids_sorted,
            d_tri_ids,  d_tri_ids_sorted,
            total_entries);
        cudaCheck(d_free(d_tmp), "free cub tmp2");
    }

    // -----------------------------------------------------------------------
    // Build cell→[start, end) range tables.
    // -----------------------------------------------------------------------
    int* d_cell_start = nullptr;
    int* d_cell_end   = nullptr;
    cudaCheck(d_alloc(&d_cell_start, sizeof(int) * g.n_cells), "malloc cell_start");
    cudaCheck(d_alloc(&d_cell_end,   sizeof(int) * g.n_cells), "malloc cell_end");
    cudaCheck(cudaMemset(d_cell_start, 0xFF, sizeof(int) * g.n_cells), "memset start"); // -1
    cudaCheck(cudaMemset(d_cell_end,   0,    sizeof(int) * g.n_cells), "memset end");
    if (total_entries > 0) {
        const int block = 256;
        const int grid_b = (total_entries + block - 1) / block;
        k_build_cell_ranges<<<grid_b, block>>>(total_entries, d_cell_ids_sorted,
                                                d_cell_start, d_cell_end);
        cudaCheck(cudaGetLastError(), "k_build_cell_ranges");
    }

    // -----------------------------------------------------------------------
    // Pass 4a: count candidates per vertex (with duplicates from multi-cell
    // entries), prefix-sum to per-vertex offsets, allocate exactly.
    // Pass 4b: emit at each vertex's allocated slot.
    // -----------------------------------------------------------------------
    int* d_v_counts  = nullptr;
    int* d_v_offsets = nullptr;
    cudaCheck(d_alloc(&d_v_counts,  sizeof(int) * (nv + 1)), "malloc v_counts");
    cudaCheck(d_alloc(&d_v_offsets, sizeof(int) * (nv + 1)), "malloc v_offsets");
    {
        // Warp-per-vertex: 32 threads per vertex; 8 vertices per block of 256.
        const int block = 256;
        const int verts_per_block = block / 32;
        const int grid_b = (nv + verts_per_block - 1) / verts_per_block;
        k_query_count<<<grid_b, block>>>(nv, d_X, d_R, d_hat, g,
                                          d_cell_start, d_cell_end,
                                          d_tri_ids_sorted, d_tris,
                                          d_v_counts);
        cudaCheck(cudaGetLastError(), "k_query_count");
    }
    {
        d_tmp = nullptr; tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes,
                                      d_v_counts, d_v_offsets, nv + 1);
        cudaCheck(d_alloc(&d_tmp, tmp_bytes), "malloc cub tmp3");
        cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes,
                                      d_v_counts, d_v_offsets, nv + 1);
        cudaCheck(d_free(d_tmp), "free cub tmp3");
    }
    int n_emitted = 0;
    cudaCheck(cudaMemcpy(&n_emitted, d_v_offsets + nv, sizeof(int),
                         cudaMemcpyDeviceToHost), "read n_emitted");
    n_nt_emitted_total = n_emitted;
    mark("nt", t_phase, t_nt);

    int* d_out_node = nullptr;
    int* d_out_tri  = nullptr;
    if (n_emitted > 0) {
        cudaCheck(d_alloc(&d_out_node, sizeof(int) * n_emitted), "malloc out_node");
        cudaCheck(d_alloc(&d_out_tri,  sizeof(int) * n_emitted), "malloc out_tri");
        const int block = 256;
        const int verts_per_block = block / 32;
        const int grid_b = (nv + verts_per_block - 1) / verts_per_block;
        k_query_emit<<<grid_b, block>>>(nv, d_X, d_R, d_hat, g,
                                        d_cell_start, d_cell_end,
                                        d_tri_ids_sorted, d_tris,
                                        d_v_offsets, d_out_node, d_out_tri);
        cudaCheck(cudaGetLastError(), "k_query_emit");
    }

    // -----------------------------------------------------------------------
    // GPU dedup: pack (node, tri) into uint64 keys, radix-sort, select unique.
    // The result keeps the unique key array on device (transferred to result).
    // -----------------------------------------------------------------------
    if (n_emitted > 0) {
        unsigned long long* d_keys = nullptr;
        unsigned long long* d_keys_sorted = nullptr;
        unsigned long long* d_keys_unique = nullptr;
        int* d_n_unique = nullptr;
        cudaCheck(d_alloc(&d_keys,        sizeof(unsigned long long) * n_emitted), "malloc keys");
        cudaCheck(d_alloc(&d_keys_sorted, sizeof(unsigned long long) * n_emitted), "malloc keys_sorted");
        cudaCheck(d_alloc(&d_keys_unique, sizeof(unsigned long long) * n_emitted), "malloc keys_unique");
        cudaCheck(d_alloc(&d_n_unique,    sizeof(int)),                              "malloc n_unique");
        {
            const int block = 256;
            const int grid_b = (n_emitted + block - 1) / block;
            k_pack_keys<<<grid_b, block>>>(n_emitted, d_out_node, d_out_tri, d_keys);
            cudaCheck(cudaGetLastError(), "k_pack_keys nt");
        }
        {
            d_tmp = nullptr; tmp_bytes = 0;
            cub::DeviceRadixSort::SortKeys(nullptr, tmp_bytes,
                d_keys, d_keys_sorted, n_emitted);
            cudaCheck(d_alloc(&d_tmp, tmp_bytes), "malloc cub tmp_sortkeys");
            cub::DeviceRadixSort::SortKeys(d_tmp, tmp_bytes,
                d_keys, d_keys_sorted, n_emitted);
            cudaCheck(d_free(d_tmp), "free cub tmp_sortkeys");
        }
        {
            d_tmp = nullptr; tmp_bytes = 0;
            cub::DeviceSelect::Unique(nullptr, tmp_bytes,
                d_keys_sorted, d_keys_unique, d_n_unique, n_emitted);
            cudaCheck(d_alloc(&d_tmp, tmp_bytes), "malloc cub tmp_unique");
            cub::DeviceSelect::Unique(d_tmp, tmp_bytes,
                d_keys_sorted, d_keys_unique, d_n_unique, n_emitted);
            cudaCheck(d_free(d_tmp), "free cub tmp_unique");
        }
        int n_unique = 0;
        cudaCheck(cudaMemcpy(&n_unique, d_n_unique, sizeof(int),
                             cudaMemcpyDeviceToHost), "read n_unique");
        // Transfer ownership of d_keys_unique to the result.
        result.d_nt_keys  = d_keys_unique;
        result.n_nt_pairs = n_unique;
        d_free(d_keys);
        d_free(d_keys_sorted);
        d_free(d_n_unique);
    }

    // =======================================================================
    // SS pipeline
    // =======================================================================

    // Edge enumeration AND device upload are static across the simulation
    // — cached by triangle array identity. Saves ~1.5ms per call (host
    // enumeration ~1ms + d_edges H2D ~0.35ms).
    struct EdgeCache {
        const int* tris_ptr = nullptr;
        std::size_t tris_size = 0;
        std::vector<std::array<int, 2>> h_edges;
        int* d_edges = nullptr;
    };
    static EdgeCache edge_cache;
    if (edge_cache.tris_ptr != ref_mesh.tris.data() ||
        edge_cache.tris_size != ref_mesh.tris.size()) {
        edge_cache.h_edges.clear();
        struct PairHash {
            std::size_t operator()(const std::pair<int,int>& p) const noexcept {
                return std::hash<long long>{}(
                    (static_cast<long long>(p.first) << 32) | (unsigned)p.second);
            }
        };
        std::unordered_set<std::pair<int,int>, PairHash> seen;
        seen.reserve(static_cast<std::size_t>(nt) * 2);
        edge_cache.h_edges.reserve(static_cast<std::size_t>(nt) * 3);
        for (int t = 0; t < nt; ++t) {
            int corners[3] = {
                ref_mesh.tris[3*t + 0],
                ref_mesh.tris[3*t + 1],
                ref_mesh.tris[3*t + 2]
            };
            for (int i = 0; i < 3; ++i) {
                int a = corners[i];
                int b = corners[(i + 1) % 3];
                if (a > b) std::swap(a, b);
                if (seen.insert({a, b}).second)
                    edge_cache.h_edges.push_back({a, b});
            }
        }
        // Upload to device once, free any prior allocation first.
        if (edge_cache.d_edges) d_free(edge_cache.d_edges);
        const int ne_new = static_cast<int>(edge_cache.h_edges.size());
        cudaCheck(d_alloc(&edge_cache.d_edges, sizeof(int) * 2 * ne_new),
                  "malloc d_edges (cache)");
        cudaCheck(cudaMemcpy(edge_cache.d_edges, edge_cache.h_edges.data(),
                             sizeof(int) * 2 * ne_new, cudaMemcpyHostToDevice),
                  "memcpy d_edges (cache)");
        edge_cache.tris_ptr = ref_mesh.tris.data();
        edge_cache.tris_size = ref_mesh.tris.size();
    }
    const std::vector<std::array<int, 2>>& h_edges = edge_cache.h_edges;
    const int ne = static_cast<int>(h_edges.size());
    // Expose the cached edge table on the result (non-owning pointer).
    result.d_edges = edge_cache.d_edges;
    result.n_edges = ne;
    mark("edges", t_phase, t_edges);

    if (ne > 0) {
        int* d_edges = edge_cache.d_edges;  // cached, persistent across calls

        // Pass 1: count cells per edge.
        int* d_e_counts = nullptr;
        int* d_e_offsets = nullptr;
        cudaCheck(d_alloc(&d_e_counts,  sizeof(int) * (ne + 1)), "malloc e_counts");
        cudaCheck(d_alloc(&d_e_offsets, sizeof(int) * (ne + 1)), "malloc e_offsets");
        {
            const int block = 256;
            const int grid_b = (ne + block - 1) / block;
            k_count_edge_cells<<<grid_b, block>>>(ne, d_edges, d_X, d_R, d_hat, g, d_e_counts);
            cudaCheck(cudaGetLastError(), "k_count_edge_cells");
        }
        {
            d_tmp = nullptr; tmp_bytes = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes,
                                          d_e_counts, d_e_offsets, ne + 1);
            cudaCheck(d_alloc(&d_tmp, tmp_bytes), "malloc cub tmp4");
            cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes,
                                          d_e_counts, d_e_offsets, ne + 1);
            cudaCheck(d_free(d_tmp), "free cub tmp4");
        }
        int e_total = 0;
        cudaCheck(cudaMemcpy(&e_total, d_e_offsets + ne, sizeof(int),
                             cudaMemcpyDeviceToHost), "read e_total");

        // Pass 2: scatter (cell, edge) pairs.
        int* d_e_cell_ids = nullptr;
        int* d_e_edge_ids = nullptr;
        int* d_e_cell_ids_sorted = nullptr;
        int* d_e_edge_ids_sorted = nullptr;
        cudaCheck(d_alloc(&d_e_cell_ids,        sizeof(int) * e_total), "malloc e_cell_ids");
        cudaCheck(d_alloc(&d_e_edge_ids,        sizeof(int) * e_total), "malloc e_edge_ids");
        cudaCheck(d_alloc(&d_e_cell_ids_sorted, sizeof(int) * e_total), "malloc e_cell_ids_s");
        cudaCheck(d_alloc(&d_e_edge_ids_sorted, sizeof(int) * e_total), "malloc e_edge_ids_s");
        if (e_total > 0) {
            const int block = 256;
            const int grid_b = (ne + block - 1) / block;
            k_scatter_edge_cells<<<grid_b, block>>>(ne, d_edges, d_X, d_R, d_hat, g,
                                                     d_e_offsets,
                                                     d_e_cell_ids, d_e_edge_ids);
            cudaCheck(cudaGetLastError(), "k_scatter_edge_cells");
        }

        // Sort by cell_id.
        if (e_total > 0) {
            d_tmp = nullptr; tmp_bytes = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, tmp_bytes,
                d_e_cell_ids, d_e_cell_ids_sorted,
                d_e_edge_ids, d_e_edge_ids_sorted,
                e_total);
            cudaCheck(d_alloc(&d_tmp, tmp_bytes), "malloc cub tmp5");
            cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes,
                d_e_cell_ids, d_e_cell_ids_sorted,
                d_e_edge_ids, d_e_edge_ids_sorted,
                e_total);
            cudaCheck(d_free(d_tmp), "free cub tmp5");
        }

        // Build cell→range map (separate from NT's; SS hashes edges, not tris).
        int* d_e_cell_start = nullptr;
        int* d_e_cell_end   = nullptr;
        cudaCheck(d_alloc(&d_e_cell_start, sizeof(int) * g.n_cells), "malloc e_cell_start");
        cudaCheck(d_alloc(&d_e_cell_end,   sizeof(int) * g.n_cells), "malloc e_cell_end");
        cudaCheck(cudaMemset(d_e_cell_start, 0xFF, sizeof(int) * g.n_cells), "memset e_start");
        cudaCheck(cudaMemset(d_e_cell_end,   0,    sizeof(int) * g.n_cells), "memset e_end");
        if (e_total > 0) {
            const int block = 256;
            const int grid_b = (e_total + block - 1) / block;
            k_build_cell_ranges<<<grid_b, block>>>(e_total, d_e_cell_ids_sorted,
                                                    d_e_cell_start, d_e_cell_end);
            cudaCheck(cudaGetLastError(), "k_build_cell_ranges (ss)");
        }

        // Per-edge query: count then emit.
        int* d_q_counts  = nullptr;
        int* d_q_offsets = nullptr;
        cudaCheck(d_alloc(&d_q_counts,  sizeof(int) * (ne + 1)), "malloc q_counts");
        cudaCheck(d_alloc(&d_q_offsets, sizeof(int) * (ne + 1)), "malloc q_offsets");
        {
            const int block = 256;
            const int edges_per_block = block / 32;
            const int grid_b = (ne + edges_per_block - 1) / edges_per_block;
            k_query_count_ss<<<grid_b, block>>>(ne, d_edges, d_X, d_R, d_hat, g,
                                                 d_e_cell_start, d_e_cell_end,
                                                 d_e_edge_ids_sorted,
                                                 d_q_counts);
            cudaCheck(cudaGetLastError(), "k_query_count_ss");
        }
        {
            d_tmp = nullptr; tmp_bytes = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes,
                                          d_q_counts, d_q_offsets, ne + 1);
            cudaCheck(d_alloc(&d_tmp, tmp_bytes), "malloc cub tmp6");
            cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes,
                                          d_q_counts, d_q_offsets, ne + 1);
            cudaCheck(d_free(d_tmp), "free cub tmp6");
        }
        int n_ss_emitted = 0;
        cudaCheck(cudaMemcpy(&n_ss_emitted, d_q_offsets + ne, sizeof(int),
                             cudaMemcpyDeviceToHost), "read n_ss_emitted");
        n_ss_emitted_total = n_ss_emitted;
        mark("ss", t_phase, t_ss);

        int* d_out_a = nullptr;
        int* d_out_b = nullptr;
        if (n_ss_emitted > 0) {
            cudaCheck(d_alloc(&d_out_a, sizeof(int) * n_ss_emitted), "malloc out_a");
            cudaCheck(d_alloc(&d_out_b, sizeof(int) * n_ss_emitted), "malloc out_b");
            const int block = 256;
            const int edges_per_block = block / 32;
            const int grid_b = (ne + edges_per_block - 1) / edges_per_block;
            k_query_emit_ss<<<grid_b, block>>>(ne, d_edges, d_X, d_R, d_hat, g,
                                                d_e_cell_start, d_e_cell_end,
                                                d_e_edge_ids_sorted,
                                                d_q_offsets,
                                                d_out_a, d_out_b);
            cudaCheck(cudaGetLastError(), "k_query_emit_ss");
        }

        // GPU dedup: pack → radix sort → unique on device. The unique key
        // array stays on device and ownership transfers to the result.
        if (n_ss_emitted > 0) {
            unsigned long long* d_keys = nullptr;
            unsigned long long* d_keys_sorted = nullptr;
            unsigned long long* d_keys_unique = nullptr;
            int* d_n_unique = nullptr;
            cudaCheck(d_alloc(&d_keys,        sizeof(unsigned long long) * n_ss_emitted), "malloc ss_keys");
            cudaCheck(d_alloc(&d_keys_sorted, sizeof(unsigned long long) * n_ss_emitted), "malloc ss_keys_sorted");
            cudaCheck(d_alloc(&d_keys_unique, sizeof(unsigned long long) * n_ss_emitted), "malloc ss_keys_unique");
            cudaCheck(d_alloc(&d_n_unique,    sizeof(int)),                                  "malloc ss_n_unique");
            {
                const int block = 256;
                const int grid_b = (n_ss_emitted + block - 1) / block;
                k_pack_keys<<<grid_b, block>>>(n_ss_emitted, d_out_a, d_out_b, d_keys);
                cudaCheck(cudaGetLastError(), "k_pack_keys ss");
            }
            {
                d_tmp = nullptr; tmp_bytes = 0;
                cub::DeviceRadixSort::SortKeys(nullptr, tmp_bytes,
                    d_keys, d_keys_sorted, n_ss_emitted);
                cudaCheck(d_alloc(&d_tmp, tmp_bytes), "malloc cub tmp_ss_sortkeys");
                cub::DeviceRadixSort::SortKeys(d_tmp, tmp_bytes,
                    d_keys, d_keys_sorted, n_ss_emitted);
                cudaCheck(d_free(d_tmp), "free cub tmp_ss_sortkeys");
            }
            {
                d_tmp = nullptr; tmp_bytes = 0;
                cub::DeviceSelect::Unique(nullptr, tmp_bytes,
                    d_keys_sorted, d_keys_unique, d_n_unique, n_ss_emitted);
                cudaCheck(d_alloc(&d_tmp, tmp_bytes), "malloc cub tmp_ss_unique");
                cub::DeviceSelect::Unique(d_tmp, tmp_bytes,
                    d_keys_sorted, d_keys_unique, d_n_unique, n_ss_emitted);
                cudaCheck(d_free(d_tmp), "free cub tmp_ss_unique");
            }
            int n_unique = 0;
            cudaCheck(cudaMemcpy(&n_unique, d_n_unique, sizeof(int),
                                 cudaMemcpyDeviceToHost), "read ss n_unique");
            // Transfer ownership of d_keys_unique to the result.
            result.d_ss_keys  = d_keys_unique;
            result.n_ss_pairs = n_unique;
            d_free(d_keys);
            d_free(d_keys_sorted);
            d_free(d_n_unique);
        }

        // d_edges is owned by edge_cache — do not free.
        d_free(d_e_counts);
        d_free(d_e_offsets);
        d_free(d_e_cell_ids);
        d_free(d_e_edge_ids);
        d_free(d_e_cell_ids_sorted);
        d_free(d_e_edge_ids_sorted);
        d_free(d_e_cell_start);
        d_free(d_e_cell_end);
        d_free(d_q_counts);
        d_free(d_q_offsets);
        if (d_out_a) d_free(d_out_a);
        if (d_out_b) d_free(d_out_b);
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    d_free(d_X);
    d_free(d_R);
    d_free(d_tris);
    d_free(d_counts);
    d_free(d_offsets);
    d_free(d_cell_ids);
    d_free(d_tri_ids);
    d_free(d_cell_ids_sorted);
    d_free(d_tri_ids_sorted);
    d_free(d_cell_start);
    d_free(d_cell_end);
    d_free(d_v_counts);
    d_free(d_v_offsets);
    if (d_out_node) d_free(d_out_node);
    if (d_out_tri)  d_free(d_out_tri);

    if (prof) {
        mark("dedup", t_phase, t_dedup);
        auto t_end = Clock::now();
        double t_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::fprintf(stderr,
            "[gpu_hg] total=%.2fms  nt=%.2fms  edges=%.2fms  ss=%.2fms  "
            "dedup=%.2fms  nv=%d nt=%d ne=%d  nt_emit=%d ss_emit=%d  "
            "nt_pairs=%d ss_pairs=%d\n",
            t_total, t_nt, t_edges, t_ss, t_dedup,
            nv, nt, ne, n_nt_emitted_total, n_ss_emitted_total,
            result.n_nt_pairs, result.n_ss_pairs);
    }
    return result;
}

// ============================================================================
// Result lifetime + host-copy helpers
// ============================================================================
GpuBroadPhaseResult::~GpuBroadPhaseResult() {
    if (d_nt_keys) cudaFreeAsync(d_nt_keys, 0);
    if (d_ss_keys) cudaFreeAsync(d_ss_keys, 0);
    // d_edges is non-owning (cached).
}

std::vector<unsigned long long> gpu_broad_phase_nt_keys_to_host(const GpuBroadPhaseResult& r) {
    std::vector<unsigned long long> out;
    if (r.n_nt_pairs == 0 || r.d_nt_keys == nullptr) return out;
    out.resize(r.n_nt_pairs);
    cudaMemcpy(out.data(), r.d_nt_keys,
               sizeof(unsigned long long) * r.n_nt_pairs,
               cudaMemcpyDeviceToHost);
    return out;
}

std::vector<unsigned long long> gpu_broad_phase_ss_keys_to_host(const GpuBroadPhaseResult& r) {
    std::vector<unsigned long long> out;
    if (r.n_ss_pairs == 0 || r.d_ss_keys == nullptr) return out;
    out.resize(r.n_ss_pairs);
    cudaMemcpy(out.data(), r.d_ss_keys,
               sizeof(unsigned long long) * r.n_ss_pairs,
               cudaMemcpyDeviceToHost);
    return out;
}

std::vector<int> gpu_broad_phase_edges_to_host(const GpuBroadPhaseResult& r) {
    std::vector<int> out;
    if (r.n_edges == 0 || r.d_edges == nullptr) return out;
    out.resize(2 * r.n_edges);
    cudaMemcpy(out.data(), r.d_edges,
               sizeof(int) * 2 * r.n_edges,
               cudaMemcpyDeviceToHost);
    return out;
}
