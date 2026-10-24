#include "resident_dat.h"
#include "resident_memory.h"
#include "resident_profile.h"
#include <cmath>
#include <climits>
#include <mutex>

namespace ipc_gpu::resident {

namespace {
void validate_scene(const SceneDescription& h) {
    const auto n=h.positions.size();
    if(n>std::size_t(INT_MAX)||h.surface_triangles.size()>std::size_t(INT_MAX)||h.edges.size()>std::size_t(INT_MAX)||h.dependencies.size()>std::size_t(INT_MAX))throw std::invalid_argument("GPU DAT topology too large");
    const auto size=[n](std::size_t count){if(count!=n)throw std::invalid_argument("GPU DAT scene array size");};
    size(h.velocities.size());size(h.pin_targets.size());size(h.deformable.size());size(h.solid.size());size(h.pinned.size());size(h.surface.size());size(h.mass.size());size(h.node_blocks.size());size(h.node_body.size());
    if(!h.com.empty()||!h.solids.empty()||h.block_count<0)throw std::invalid_argument("GPU DAT supports cloth and particles only");
    std::vector<unsigned char> seen(static_cast<std::size_t>(h.block_count),0);
    for(std::size_t i=0;i<n;++i) {
        if(!h.deformable[i]||h.solid[i]||h.node_body[i]!=-1)throw std::invalid_argument("GPU DAT supports cloth and particles only");
        const int block=h.node_blocks[i];if(block<0||block>=h.block_count||seen[block]++)throw std::invalid_argument("GPU DAT block mapping");
        if(!(h.mass[i]>=0)||!std::isfinite(h.mass[i]))throw std::invalid_argument("GPU DAT invalid mass");
        for(int k=0;k<3;++k)if(!std::isfinite(h.positions[i][k])||!std::isfinite(h.velocities[i][k]))throw std::invalid_argument("GPU DAT nonfinite initial state");
    }
    const auto node=[n](int v){if(v<0||std::size_t(v)>=n)throw std::invalid_argument("GPU DAT topology index");};
    for(const auto& t:h.surface_triangles)for(int v:t.vertices)node(v);
    for(const auto& e:h.edges)for(int v:e.vertices)node(v);
    for(const auto& d:h.dependencies){if(d.count<1||d.count>4)throw std::invalid_argument("GPU DAT dependency size");for(int k=0;k<d.count;++k)node(d.vertices[k]);}
    const auto offsets=[n,&node](const auto& off,const auto& records) {
        if(off.size()!=n+1||off.front()!=0||off.back()!=records.size())throw std::invalid_argument("GPU DAT element offsets");
        for(std::size_t i=0;i<n;++i){if(off[i]>off[i+1]||off[i+1]>records.size())throw std::invalid_argument("GPU DAT element offsets");for(auto j=off[i];j<off[i+1];++j){if(records[j].owner!=int(i))throw std::invalid_argument("GPU DAT incident owner");for(int v:records[j].nodes)node(v);}}
    };
    offsets(h.triangle_offsets,h.triangles);offsets(h.solid_offsets,h.solids);offsets(h.hinge_offsets,h.hinges);
}
void validate_step(const StepParameters& p) {
    if(!p.planar_dat||!(p.dt>0)||!std::isfinite(p.dt)||p.elastic.dt2!=p.dt*p.dt||!(p.elastic.dt2>0)||!std::isfinite(p.elastic.dt2)
        ||p.contact.dt!=p.dt||p.sdf.dt!=p.dt||!std::isfinite(p.damping)||p.iterations<=0||p.dat_detection_interval<=0
        ||!(p.contact.d_hat>0)||!std::isfinite(1.5*p.contact.d_hat)||!(p.dat_relaxation>0&&p.dat_relaxation<1)
        ||!std::isfinite(p.contact.k_barrier)||p.contact.k_barrier<0||!std::isfinite(p.contact.friction)||p.contact.friction<0
        ||!(p.contact.eps>0)||!std::isfinite(p.contact.eps)||!(p.contact.eps_v>0)||!std::isfinite(p.contact.eps_v))
        throw std::invalid_argument("GPU DAT invalid step parameters");
}
std::mutex dat_solver_mutex;
}
struct ResidentDATSolver::Impl {
    DeviceArray<Vector3> positions,velocities,previous,predicted,pins,proposed;
    DeviceArray<unsigned char> deformable,solid,pinned,surface;
    DeviceArray<int> node_blocks,error;
    DeviceArray<double> mass;
    DeviceArray<TriangleIncident> triangles;
    DeviceArray<SolidIncident> solids;
    DeviceArray<HingeIncident> hinges;
    DeviceArray<std::size_t> triangle_offsets,solid_offsets,hinge_offsets;
    DeviceArray<BpTriangle> surface_triangles;
    DeviceArray<BpEdge> edges;
    DeviceArray<BpDependency> dependencies;
    DeviceArray<ipc_gpu::Point> points;
    DeviceArray<ipc_gpu::Triangle> triangle_records;
    DeviceArray<ipc_gpu::Solid> solid_records;
    DeviceArray<ipc_gpu::Hinge> hinge_records;
    DeviceArray<ipc_gpu::Derivatives> point_output,triangle_output,solid_output,hinge_output,systems;
    DeviceArray<SdfCollider> colliders;
    DeformableScene scene{};DeformableState state{};ElasticWorkspace elastic{};BroadPhaseTopology topology{};
    ResidentDAT dat;
    bool invalid=false;
    static double (*v3(DeviceArray<Vector3>& a))[3]{return reinterpret_cast<double(*)[3]>(a.data());}
public:
    explicit Impl(const SceneDescription& h) {
        validate_scene(h);const auto n=h.positions.size();positions.upload(h.positions);velocities.upload(h.velocities);previous.resize(n);predicted.resize(n);pins.upload(h.pin_targets);proposed.resize(n);
        deformable.upload(h.deformable);solid.upload(h.solid);pinned.upload(h.pinned);surface.upload(h.surface);node_blocks.upload(h.node_blocks);mass.upload(h.mass);
        triangles.upload(h.triangles);solids.upload(h.solids);hinges.upload(h.hinges);triangle_offsets.upload(h.triangle_offsets);solid_offsets.upload(h.solid_offsets);hinge_offsets.upload(h.hinge_offsets);
        surface_triangles.upload(h.surface_triangles);edges.upload(h.edges);dependencies.upload(h.dependencies);error.resize(1);error.clear();
        points.resize(n);triangle_records.resize(h.triangles.size());solid_records.resize(h.solids.size());hinge_records.resize(h.hinges.size());
        point_output.resize(n);triangle_output.resize(h.triangles.size());solid_output.resize(h.solids.size());hinge_output.resize(h.hinges.size());systems.resize(n);
        scene={n,h.triangles.size(),h.solids.size(),h.hinges.size(),node_blocks.data(),deformable.data(),solid.data(),pinned.data(),mass.data(),v3(pins),triangles.data(),solids.data(),hinges.data(),triangle_offsets.data(),solid_offsets.data(),hinge_offsets.data()};
        state={v3(positions),v3(velocities),v3(previous),v3(predicted)};
        elastic={points.data(),triangle_records.data(),solid_records.data(),hinge_records.data(),point_output.data(),triangle_output.data(),solid_output.data(),hinge_output.data(),systems.data(),v3(proposed),error.data()};
        topology.vertex_count=n;topology.block_count=h.block_count;topology.triangle_count=h.surface_triangles.size();topology.edge_count=h.edges.size();topology.dependency_count=h.dependencies.size();
        topology.triangles=surface_triangles.data();topology.edges=edges.data();topology.dependencies=dependencies.data();topology.vertex_block=node_blocks.data();topology.surface_query=surface.data();
    }
    void advance_substep(const StepParameters& p,const std::vector<Vector3>& targets,const std::vector<SdfCollider>& obstacles) {
        pins.upload(targets);colliders.upload(obstacles);error.clear();begin_deformable_substep(scene,state,p.dt);
        dat.solve(scene,state,elastic,topology,{colliders.data(),static_cast<int>(obstacles.size()),p.sdf},p);
        finish_deformable_substep(scene,state,p.dt);cuda_checked(cudaDeviceSynchronize());const int status=error.download()[0];if(status)throw std::runtime_error("GPU DAT device status "+std::to_string(status));
    }
    StateSnapshot snapshot(){StateSnapshot out;out.positions=positions.download();out.velocities=velocities.download();return out;}
};

ResidentDATSolver::ResidentDATSolver(const SceneDescription& scene):impl_(std::make_unique<Impl>(scene)){}
ResidentDATSolver::~ResidentDATSolver()=default;
void ResidentDATSolver::advance_substep(const StepParameters& p,const std::vector<Vector3>& pins,const std::vector<SdfCollider>& colliders) {
    const std::lock_guard<std::mutex> lock(dat_solver_mutex);auto& s=*impl_;
    if(s.invalid)throw std::runtime_error("GPU DAT state invalidated by earlier failure");
    validate_step(p);
    if(pins.size()!=s.scene.nodes||colliders.size()>std::size_t(INT_MAX))throw std::invalid_argument("GPU DAT prescribed input size");
    try{s.advance_substep(p,pins,colliders);}catch(...){s.invalid=true;if(active_resident_profiler){active_resident_profiler->abandon_frame();active_resident_profiler=nullptr;}throw;}
}
StateSnapshot ResidentDATSolver::snapshot()const {
    const std::lock_guard<std::mutex> lock(dat_solver_mutex);
    if(impl_->invalid)throw std::runtime_error("GPU DAT cannot export invalid state");
    return impl_->snapshot();
}
void ResidentDATSolver::begin_profile_frame(int frame) {
    const std::lock_guard<std::mutex> lock(dat_solver_mutex);auto& profile=resident_profiler();
    if(profile.enabled()){profile.begin_frame(frame);active_resident_profiler=&profile;}
}
void ResidentDATSolver::end_profile_frame() {
    const std::lock_guard<std::mutex> lock(dat_solver_mutex);
    if(active_resident_profiler){active_resident_profiler->end_frame();active_resident_profiler=nullptr;}
}
} // namespace ipc_gpu::resident
