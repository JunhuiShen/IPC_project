#pragma once
#include "kernels.h"
#include "physics.h"
#include <vector>

namespace ipc_gpu {
// Host organizer. Gather, device compute, and result-layout conversion are
// distinct passes. Existing physics code performs the ordered accumulation.
class ColorBridge {
    std::vector<Point> points;
    std::vector<Triangle> triangles;
    std::vector<Hinge> hinges;
    std::vector<Derivatives> po, to, ho;
    std::vector<Vec3> pg, tg, hg;
    std::vector<Mat33> ph, th, hh;
    std::vector<std::size_t> triangle_offsets, hinge_offsets;
    static void unpack(const std::vector<Derivatives>& in,
                       std::vector<Vec3>& g, std::vector<Mat33>& h) {
        g.resize(in.size()); h.resize(in.size());
        #pragma omp parallel for schedule(static) if(in.size() >= 128)
        for (std::size_t e=0; e<in.size(); ++e) {
            for (int i=0; i<3; ++i) {
                g[e][i]=in[e].gradient[i];
                for (int j=0; j<3; ++j) h[e](i,j)=in[e].hessian[3*i+j];
            }
        }
    }
public:
    template<class Materials>
    void prepare(const std::vector<int>& group, const RefMesh& mesh,
                 const Materials& materials, const std::vector<Vec3>& positions,
                 const std::vector<Vec3>& predicted, const std::vector<Pin>& pins,
                 const PinMap& pin_map, const SimParams& params,
                 std::vector<physics_detail::SimdDerivativeView>& point_views,
                 std::vector<physics_detail::SimdDerivativeView>& triangle_views,
                 std::vector<physics_detail::SimdDerivativeView>& hinge_views) {
        const auto count=group.size();
        triangle_offsets.resize(count+1); hinge_offsets.resize(count+1);
        triangle_offsets[0]=hinge_offsets[0]=0;
        for (std::size_t i=0; i<count; ++i) {
            const int vi=group[i];
            triangle_offsets[i+1]=triangle_offsets[i]+materials.triangles.node_offsets[vi+1]-materials.triangles.node_offsets[vi];
            hinge_offsets[i+1]=hinge_offsets[i]+(params.kB>0 ? materials.hinges.node_offsets[vi+1]-materials.hinges.node_offsets[vi] : 0);
        }
        points.resize(count); triangles.resize(triangle_offsets.back()); hinges.resize(hinge_offsets.back());
        po.resize(points.size()); to.resize(triangles.size()); ho.resize(hinges.size());
        // Gather AoS records from an unchanged, conflict-free color snapshot.
        #pragma omp parallel for schedule(static) if(count >= 128)
        for (std::size_t i=0; i<count; ++i) {
            const int vi=group[i];
            auto& p=points[i];
            p.mass=mesh.mass[vi]; p.pinned=pin_map[vi]>=0;
            for (int a=0; a<3; ++a) {
                p.position[a]=positions[vi][a]; p.predicted[a]=predicted[vi][a];
                p.target[a]=p.pinned ? pins[pin_map[vi]].target_position[a] : positions[vi][a];
            }
            auto out=triangle_offsets[i];
            for (auto e=materials.triangles.node_offsets[vi]; e<materials.triangles.node_offsets[vi+1]; ++e,++out) {
                auto& t=triangles[out];
                const auto base=3*materials.triangles.triangle_indices[e];
                for (int v=0; v<3; ++v) for (int a=0; a<3; ++a)
                    t.positions[3*v+a]=positions[mesh.tris[base+v]][a];
                for (int r=0; r<2; ++r) {
                    t.shape[r]=materials.triangles.shape_gradients[e][r];
                    for (int c=0; c<2; ++c) t.dm_inverse[2*r+c]=materials.triangles.dm_inverse[e](r,c);
                }
                t.area=materials.triangles.areas[e];
            }
            if (params.kB>0) {
                out=hinge_offsets[i];
                for (auto e=materials.hinges.node_offsets[vi]; e<materials.hinges.node_offsets[vi+1]; ++e,++out) {
                    auto& h=hinges[out];
                    const auto& definition=mesh.hinges[materials.hinges.hinge_indices[e]];
                    for (int v=0; v<4; ++v) for (int a=0; a<3; ++a) h.positions[3*v+a]=positions[definition.v[v]][a];
                    h.active_node=materials.hinges.active_nodes[e];
                    h.coefficient=materials.hinges.coefficients[e]; h.rest_angle=materials.hinges.rest_angles[e];
                }
            }
        }
        Parameters p{};
        for (int a=0; a<3; ++a) p.gravity[a]=params.gravity[a];
        p.kpin=params.kpin; p.dt2=params.dt2(); p.mu=params.mu; p.lambda=params.lambda; p.bending_stiffness=params.kB;
        compute(points.data(),points.size(),triangles.data(),triangles.size(),hinges.data(),hinges.size(),p,po.data(),to.data(),ho.data());
        unpack(po,pg,ph); unpack(to,tg,th); unpack(ho,hg,hh);
        for (std::size_t i=0; i<count; ++i) {
            const int vi=group[i];
            point_views[vi]={pg.data()+i,ph.data()+i,1};
            const auto t=triangle_offsets[i], nt=triangle_offsets[i+1]-t;
            triangle_views[vi]={nt ? tg.data()+t : nullptr,nt ? th.data()+t : nullptr,nt};
            if (params.kB>0) {
                const auto h=hinge_offsets[i], nh=hinge_offsets[i+1]-h;
                hinge_views[vi]={nh ? hg.data()+h : nullptr,nh ? hh.data()+h : nullptr,nh};
            }
        }
    }
};
} // namespace ipc_gpu
