// gpu_mesh_stub.cpp
// CPU-only implementation of the GPU mirror structs.
// DeviceBuffer<T> uses plain heap allocation so the rest of the codebase
// can be written and tested on any machine without a CUDA toolchain.
// On a real GPU build, compile gpu_mesh.cu instead and omit this file.

#include "gpu_mesh.h"
#include <algorithm>
#include <cstring>
#include <vector>

// --------------------------------------------------------------------------
// DeviceBuffer  (stub: heap allocation, memcpy "transfer")
// --------------------------------------------------------------------------

template<typename T>
void DeviceBuffer<T>::alloc(int n) {
    free();
    count = n;
    ptr   = new T[n];
}

template<typename T>
void DeviceBuffer<T>::free() {
    delete[] ptr;
    ptr   = nullptr;
    count = 0;
}

template<typename T>
void DeviceBuffer<T>::upload(const T* host_data, int n) {
    alloc(n);
    std::memcpy(ptr, host_data, static_cast<size_t>(n) * sizeof(T));
}

template<typename T>
void DeviceBuffer<T>::download(T* host_data) const {
    std::memcpy(host_data, ptr, static_cast<size_t>(count) * sizeof(T));
}

template struct DeviceBuffer<int>;
template struct DeviceBuffer<double>;
template struct DeviceBuffer<float>;

// --------------------------------------------------------------------------
// GPURefMesh
// --------------------------------------------------------------------------

void GPURefMesh::upload(const RefMesh& cpu) {
    num_verts = static_cast<int>(cpu.num_positions);
    num_tris  = static_cast<int>(cpu.tris.size()) / 3;

    tris.upload(cpu.tris.data(), static_cast<int>(cpu.tris.size()));
    area.upload(cpu.area.data(), num_tris);
    mass.upload(cpu.mass.data(), num_verts);

    // Mat22 is Eigen column-major: .data() gives [a00, a10, a01, a11].
    std::vector<double> flat(num_tris * 4);
    for (int t = 0; t < num_tris; ++t)
        std::copy(cpu.Dm_inverse[t].data(), cpu.Dm_inverse[t].data() + 4, flat.data() + t * 4);
    Dm_inv.upload(flat.data(), num_tris * 4);

    // --- hinges ---
    num_hinges = static_cast<int>(cpu.hinges.size());
    if (num_hinges > 0) {
        std::vector<int>    hv(num_hinges * 4);
        std::vector<double> hbt(num_hinges), hce(num_hinges);
        for (int h = 0; h < num_hinges; ++h) {
            for (int k = 0; k < 4; ++k) hv[h*4+k] = cpu.hinges[h].v[k];
            hbt[h] = cpu.hinges[h].bar_theta;
            hce[h] = cpu.hinges[h].c_e;
        }
        hinge_v.upload(hv.data(), num_hinges * 4);
        hinge_bar_theta.upload(hbt.data(), num_hinges);
        hinge_ce.upload(hce.data(), num_hinges);
    }

    // --- hinge adjacency (CSR) ---
    std::vector<int> ha_offsets(num_verts + 1, 0);
    for (int vi = 0; vi < num_verts; ++vi) {
        auto it = cpu.hinge_adj.find(vi);
        if (it != cpu.hinge_adj.end())
            ha_offsets[vi + 1] = static_cast<int>(it->second.size());
    }
    for (int vi = 0; vi < num_verts; ++vi)
        ha_offsets[vi + 1] += ha_offsets[vi];

    const int ha_total = ha_offsets[num_verts];
    std::vector<int> ha_hi(ha_total), ha_role(ha_total);
    for (int vi = 0; vi < num_verts; ++vi) {
        auto it = cpu.hinge_adj.find(vi);
        if (it == cpu.hinge_adj.end()) continue;
        int pos = ha_offsets[vi];
        for (const auto& [hi, role] : it->second) {
            ha_hi[pos]   = hi;
            ha_role[pos] = role;
            ++pos;
        }
    }
    hinge_adj_offsets.upload(ha_offsets.data(), num_verts + 1);
    if (ha_total > 0) {
        hinge_adj_hi.upload(ha_hi.data(), ha_total);
        hinge_adj_role.upload(ha_role.data(), ha_total);
    }
}

// --------------------------------------------------------------------------
// GPUDeformedState
// --------------------------------------------------------------------------

void GPUDeformedState::upload(const DeformedState& cpu) {
    num_verts = static_cast<int>(cpu.deformed_positions.size());

    std::vector<double> pos_flat(num_verts * 3);
    std::vector<double> vel_flat(num_verts * 3);
    for (int i = 0; i < num_verts; ++i) {
        pos_flat[i*3+0] = cpu.deformed_positions[i](0);
        pos_flat[i*3+1] = cpu.deformed_positions[i](1);
        pos_flat[i*3+2] = cpu.deformed_positions[i](2);
        vel_flat[i*3+0] = cpu.velocities[i](0);
        vel_flat[i*3+1] = cpu.velocities[i](1);
        vel_flat[i*3+2] = cpu.velocities[i](2);
    }
    positions.upload(pos_flat.data(), num_verts * 3);
    velocities.upload(vel_flat.data(), num_verts * 3);
}

void GPUDeformedState::download(DeformedState& cpu) const {
    std::vector<double> pos_flat(num_verts * 3);
    std::vector<double> vel_flat(num_verts * 3);
    positions.download(pos_flat.data());
    velocities.download(vel_flat.data());

    cpu.deformed_positions.resize(num_verts);
    cpu.velocities.resize(num_verts);
    for (int i = 0; i < num_verts; ++i) {
        cpu.deformed_positions[i] = Vec3(pos_flat[i*3+0], pos_flat[i*3+1], pos_flat[i*3+2]);
        cpu.velocities[i]         = Vec3(vel_flat[i*3+0], vel_flat[i*3+1], vel_flat[i*3+2]);
    }
}

// --------------------------------------------------------------------------
// GPUPins
// --------------------------------------------------------------------------

void GPUPins::upload(const std::vector<Pin>& cpu) {
    count = static_cast<int>(cpu.size());
    if (count == 0) return;

    std::vector<int>    idx(count);
    std::vector<double> tgt(count * 3);
    for (int i = 0; i < count; ++i) {
        idx[i]     = cpu[i].vertex_index;
        tgt[i*3+0] = cpu[i].target_position(0);
        tgt[i*3+1] = cpu[i].target_position(1);
        tgt[i*3+2] = cpu[i].target_position(2);
    }
    indices.upload(idx.data(), count);
    targets.upload(tgt.data(), count * 3);
}

// --------------------------------------------------------------------------
// GPUAdjacency
// --------------------------------------------------------------------------

void GPUAdjacency::upload(const VertexTriangleMap& adj, int nv) {
    num_verts = nv;

    // Build per-vertex entry counts then prefix-sum into offsets.
    std::vector<int> offsets_cpu(nv + 1, 0);
    for (int vi = 0; vi < nv; ++vi) {
        auto it = adj.find(vi);
        if (it != adj.end())
            offsets_cpu[vi + 1] = static_cast<int>(it->second.size());
    }
    for (int vi = 0; vi < nv; ++vi)
        offsets_cpu[vi + 1] += offsets_cpu[vi];

    const int total = offsets_cpu[nv];
    std::vector<int> tri_idx_cpu(total);
    std::vector<int> tri_local_cpu(total);

    for (int vi = 0; vi < nv; ++vi) {
        auto it = adj.find(vi);
        if (it == adj.end()) continue;
        int pos = offsets_cpu[vi];
        for (const auto& [tri, local] : it->second) {
            tri_idx_cpu[pos]   = tri;
            tri_local_cpu[pos] = local;
            ++pos;
        }
    }

    offsets.upload(offsets_cpu.data(), nv + 1);
    tri_idx.upload(tri_idx_cpu.data(), total);
    tri_local.upload(tri_local_cpu.data(), total);
}

// --------------------------------------------------------------------------
// GPURefPositions
// --------------------------------------------------------------------------

void GPURefPositions::upload(const std::vector<Vec2>& cpu) {
    count = static_cast<int>(cpu.size());
    if (count == 0) return;

    std::vector<double> flat(count * 2);
    for (int i = 0; i < count; ++i) {
        flat[i*2+0] = cpu[i](0);
        flat[i*2+1] = cpu[i](1);
    }
    data.upload(flat.data(), count * 2);
}

// --------------------------------------------------------------------------
// GPUPinMap
// --------------------------------------------------------------------------

void GPUPinMap::upload(const PinMap& pm) {
    num_verts = static_cast<int>(pm.size());
    if (num_verts == 0) return;
    data.upload(pm.data(), num_verts);
}

// --------------------------------------------------------------------------
// GPUSimParams
// --------------------------------------------------------------------------

GPUSimParams GPUSimParams::from(const SimParams& p) {
    GPUSimParams g;
    g.dt_val                 = p.dt();
    g.dt2_val                = p.dt2();
    g.mu                     = p.mu;
    g.lambda                 = p.lambda;
    g.density                = p.density;
    g.thickness              = p.thickness;
    g.kpin                   = p.kpin;
    g.kB                     = p.kB;
    g.d_hat                  = p.d_hat;
    g.gx                     = p.gravity(0);
    g.gy                     = p.gravity(1);
    g.gz                     = p.gravity(2);
    g.tol_abs                = p.tol_abs;
    g.tol_rel                = p.tol_rel;
    g.step_weight            = p.step_weight;
    g.max_global_iters       = p.max_global_iters;
    g.use_parallel           = p.use_parallel;
    g.ccd_check              = p.ccd_check;
    g.use_trust_region       = p.use_trust_region;
    g.use_incremental_refresh   = p.use_incremental_refresh;
    g.mass_normalize_residual   = p.mass_normalize_residual;
    return g;
}

// --------------------------------------------------------------------------
// GPUBroadPhaseCache
// --------------------------------------------------------------------------

void GPUBroadPhaseCache::upload(const BroadPhase::Cache& cache, int nv) {
    num_verts = nv;
    num_nt    = static_cast<int>(cache.nt_pairs.size());
    num_ss    = static_cast<int>(cache.ss_pairs.size());

    // --- flat pair lists ---
    {
        std::vector<int> flat(num_nt * 4);
        for (int i = 0; i < num_nt; ++i) {
            const auto& p  = cache.nt_pairs[i];
            flat[i*4+0] = p.node;
            flat[i*4+1] = p.tri_v[0];
            flat[i*4+2] = p.tri_v[1];
            flat[i*4+3] = p.tri_v[2];
        }
        if (num_nt > 0) nt_data.upload(flat.data(), num_nt * 4);
    }
    {
        std::vector<int> flat(num_ss * 4);
        for (int i = 0; i < num_ss; ++i) {
            const auto& p  = cache.ss_pairs[i];
            flat[i*4+0] = p.v[0];
            flat[i*4+1] = p.v[1];
            flat[i*4+2] = p.v[2];
            flat[i*4+3] = p.v[3];
        }
        if (num_ss > 0) ss_data.upload(flat.data(), num_ss * 4);
    }

    // --- per-vertex NT lookup (CSR) ---
    {
        std::vector<int> offsets_cpu(nv + 1, 0);
        for (int vi = 0; vi < nv; ++vi)
            if (vi < static_cast<int>(cache.vertex_nt.size()))
                offsets_cpu[vi + 1] = static_cast<int>(cache.vertex_nt[vi].size());
        for (int vi = 0; vi < nv; ++vi)
            offsets_cpu[vi + 1] += offsets_cpu[vi];

        const int total = offsets_cpu[nv];
        std::vector<int> pidx(total), dof(total);
        for (int vi = 0; vi < nv; ++vi) {
            if (vi >= static_cast<int>(cache.vertex_nt.size())) continue;
            int pos = offsets_cpu[vi];
            for (const auto& e : cache.vertex_nt[vi]) {
                pidx[pos] = static_cast<int>(e.pair_index);
                dof[pos]  = e.dof;
                ++pos;
            }
        }
        vnt_offsets.upload(offsets_cpu.data(), nv + 1);
        if (total > 0) { vnt_pair_idx.upload(pidx.data(), total); vnt_dof.upload(dof.data(), total); }
    }

    // --- per-vertex SS lookup (CSR) ---
    {
        std::vector<int> offsets_cpu(nv + 1, 0);
        for (int vi = 0; vi < nv; ++vi)
            if (vi < static_cast<int>(cache.vertex_ss.size()))
                offsets_cpu[vi + 1] = static_cast<int>(cache.vertex_ss[vi].size());
        for (int vi = 0; vi < nv; ++vi)
            offsets_cpu[vi + 1] += offsets_cpu[vi];

        const int total = offsets_cpu[nv];
        std::vector<int> pidx(total), dof(total);
        for (int vi = 0; vi < nv; ++vi) {
            if (vi >= static_cast<int>(cache.vertex_ss.size())) continue;
            int pos = offsets_cpu[vi];
            for (const auto& e : cache.vertex_ss[vi]) {
                pidx[pos] = static_cast<int>(e.pair_index);
                dof[pos]  = e.dof;
                ++pos;
            }
        }
        vss_offsets.upload(offsets_cpu.data(), nv + 1);
        if (total > 0) { vss_pair_idx.upload(pidx.data(), total); vss_dof.upload(dof.data(), total); }
    }
}
