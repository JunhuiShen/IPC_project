#include "solver.h"
#include "device_memory.h"
#include "contact_pipeline.h"
#include "linear_pipeline.h"
#include "rigid_translation_proxy_owner.h"
#include "rigid_pipeline.h"
#include "adaptive_boxes.h"
#include "global_guess.h"
#include "level_schedule.h"
#include "level_schedule.cuh"
#include "contact_execution.h"
#include "contact_schedule.h"
#include "two_rigid_color_queue_policy.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>

namespace ipc_gpu::solver {
namespace {
static_assert(sizeof(Vector3)==3*sizeof(double));
static_assert(sizeof(Quaternion)==4*sizeof(double));
static_assert(sizeof(Matrix3)==9*sizeof(double));
double (*v3(DeviceArray<Vector3>& a))[3] {return reinterpret_cast<double(*)[3]>(a.data());}
double (*q4(DeviceArray<Quaternion>& a))[4] {return reinterpret_cast<double(*)[4]>(a.data());}
double (*m9(DeviceArray<Matrix3>& a))[9] {return reinterpret_cast<double(*)[9]>(a.data());}
unsigned blocks(std::size_t n) {return static_cast<unsigned>((n+127)/128);}
__global__ void colored_guess_targets(DeformableScene s,DeformableState x,double fraction,
                                     double (*targets)[3],double (*raw_out)[3],int* error) {
    const auto n=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=s.nodes || *error)return;
    for(int a=0;a<3;++a) {
        // Same subtract, scale, add order as the CPU predictor. Rigid proxies
        // remain obstacles; the full inertial prediction is never overwritten.
        double target=x.previous[n][a];
        double raw=0.0;
        if(s.deformable[n]) {
            const double displacement=(x.predicted[n][a]-x.previous[n][a])*fraction;
            // CPU stores this raw intended displacement and only then forms
            // target = previous + raw. The original predictor's movement boxes
            // and padding must use THIS value, never target-start.
            raw=displacement;
            target+=displacement;
        }
        if(!isfinite(target)||!isfinite(raw)) {atomicCAS(error,0,2);return;}
        if(raw_out)raw_out[n][a]=raw;
        if(targets)targets[n][a]=target;
    }
}
__global__ void predicted_targets(DeformableScene s,const double (*targets)[3],DeformableState x,
                                   ElasticWorkspace w,const int* colors,int color,
                                   const int* predictor_done) {
    const auto n=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=s.nodes || !s.deformable[n] || colors[s.node_blocks[n]]!=color || *w.error)return;
    // Once a whole sweep has converged (no stored movement or every target
    // squared error < 1e-28) later sweeps write the current position so the
    // proposal and clipper become a no-op, matching the CPU stop rule without
    // any host readback of positions.
    if(predictor_done&&*predictor_done) {
        for(int a=0;a<3;++a)w.proposed[n][a]=x.positions[n][a];
        return;
    }
    for(int a=0;a<3;++a)w.proposed[n][a]=targets[n][a];
}
__global__ void predictor_sweep_status(DeformableScene s,const double (*positions)[3],
    const double (*sweep_start)[3],const double (*targets)[3],unsigned* bits) {
    const auto n=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=s.nodes||!s.deformable[n])return;
    bool moved=false;
    double remaining=0.0;
    for(int a=0;a<3;++a) {
        moved=moved||(positions[n][a]!=sweep_start[n][a]);
        const double d=targets[n][a]-positions[n][a];
        remaining+=d*d;
    }
    // Shared CPU rule (see level_schedule.cuh): after the whole sweep joins,
    // stop on no stored movement OR every target squared error below 1e-28.
    const unsigned value=level_predictor::predictor_sweep_bits(moved,remaining);
    if(value&1u)atomicOr(bits,1u);
    if(value&2u)atomicOr(bits,2u);
}
__global__ void predictor_join_status(const unsigned* bits,int* done) {
    if(blockIdx.x||threadIdx.x)return;
    // No per-color stop and no probe-derived convergence: only this joined
    // whole-sweep decision may end the predictor.
    *done=level_predictor::predictor_sweep_finished(*bits)?1:0;
}
// CPU ColoredContactSweep::prepare model. Marks exactly the cooperative
// selected deformable nodes for a logical reference team. Stable heavy-first
// order uses incidence counts with canonical (ascending block) ties. A
// reference team of one is the noncooperative StoredNoop rule and marks none.
__global__ void original_cooperative_mask(DeformableScene scene,BroadPhaseView broad,
    const std::size_t* incidence_offsets,const std::size_t* raw_counts,
    const int* block_nodes,int block_count,int team,unsigned char* mask) {
    const auto node=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(node>=scene.nodes)return;
    mask[node]=0;
    if(team<2||!broad.block_colors||!broad.color_offsets||
        broad.color_count<=0)return;
    const int block=scene.node_blocks[node];
    if(block<0||block>=block_count)return;
    const int color=broad.block_colors[block];
    if(color<0||color>=broad.color_count)return;
    const int first=broad.color_offsets[color];
    const int last=broad.color_offsets[color+1];
    if(first<0||last<first||last>block_count)return;
    if(cpu_contact_schedule::selected(block,block_count,broad.color_blocks,first,last,
        incidence_offsets,raw_counts,block_nodes,team))mask[node]=1;
}
void length(std::size_t actual,std::size_t expected,const char* what) {
    if(actual!=expected)throw std::invalid_argument(std::string("GPU scene size: ")+what);
}
template<class T> void offsets(const std::vector<std::size_t>& off,std::size_t nodes,
                              const std::vector<T>& records) {
    length(off.size(),nodes+1,"element offsets");
    if(off.front()!=0 || off.back()!=records.size())throw std::invalid_argument("GPU element offset extent");
    for(std::size_t n=0;n<nodes;++n)
        if(off[n]>off[n+1] || off[n+1]>records.size())
            throw std::invalid_argument("GPU invalid element offsets");
    for(std::size_t n=0;n<nodes;++n) {
        for(auto e=off[n];e<off[n+1];++e) {
            if(records[e].owner!=static_cast<int>(n))throw std::invalid_argument("GPU incident owner mismatch");
            for(int node:records[e].nodes)if(node<0||static_cast<std::size_t>(node)>=nodes)
                throw std::invalid_argument("GPU incident node index");
        }
    }
}
void validate(const SceneDescription& h) {
    const auto n=h.positions.size(),b=h.com.size();
    if(n>std::numeric_limits<int>::max() || h.block_count<0)throw std::invalid_argument("GPU scene too large");
    if(h.surface_triangles.size()>std::numeric_limits<int>::max() || h.edges.size()>std::numeric_limits<int>::max()
        ||h.dependencies.size()>std::numeric_limits<int>::max())throw std::invalid_argument("GPU topology too large");
    length(h.velocities.size(),n,"velocities");length(h.pin_targets.size(),n,"pins");
    length(h.body_reference.size(),n,"reference");length(h.deformable.size(),n,"deformable");
    length(h.solid.size(),n,"solid");length(h.pinned.size(),n,"pinned");length(h.surface.size(),n,"surface");
    length(h.mass.size(),n,"mass");length(h.node_blocks.size(),n,"blocks");length(h.node_body.size(),n,"owners");
    offsets(h.triangle_offsets,n,h.triangles);offsets(h.solid_offsets,n,h.solids);offsets(h.hinge_offsets,n,h.hinges);
    length(h.body_blocks.size(),b,"body blocks");length(h.body_mass.size(),b,"body mass");
    length(h.body_second_moment.size(),b,"body moment");length(h.body_translate.size(),b,"body translate");
    length(h.body_rotate.size(),b,"body rotate");length(h.com_velocity.size(),b,"body velocity");
    length(h.orientation.size(),b,"orientation");length(h.omega.size(),b,"omega");
    if(!h.rigid_pins.empty()||!h.rigid_pin_offsets.empty()) {
        length(h.rigid_pin_offsets.size(),b+1,"rigid pin offsets");
        if(h.rigid_pin_offsets.front()!=0||h.rigid_pin_offsets.back()!=h.rigid_pins.size())
            throw std::invalid_argument("GPU rigid pin offset extent");
        for(std::size_t body=0;body<b;++body) {
            const auto first=h.rigid_pin_offsets[body],last=h.rigid_pin_offsets[body+1];
            if(first>last||last>h.rigid_pins.size())throw std::invalid_argument("GPU rigid pin offsets");
            for(auto index=first;index<last;++index) {
                const auto& pin=h.rigid_pins[index];
                if(pin.body!=static_cast<int>(body)||!std::isfinite(pin.angular_velocity_z))
                    throw std::invalid_argument("GPU rigid pin owner/motion");
                for(int a=0;a<3;++a)
                    if(!std::isfinite(pin.local_position[a])||!std::isfinite(pin.initial_target[a])
                        ||!std::isfinite(pin.rotation_center[a]))
                        throw std::invalid_argument("GPU nonfinite rigid pin");
            }
        }
    }
    length(h.body_node_offsets.size(),b+1,"body offsets");
    if(h.body_node_offsets.front()!=0 || h.body_node_offsets.back()!=h.body_nodes.size())
        throw std::invalid_argument("GPU body offset extent");
    for(std::size_t body=0;body<b;++body)
        if(h.body_node_offsets[body]>h.body_node_offsets[body+1] || h.body_node_offsets[body+1]>h.body_nodes.size())
            throw std::invalid_argument("GPU invalid body offsets");
    std::vector<unsigned char> seen_blocks(static_cast<std::size_t>(h.block_count),0);
    for(std::size_t i=0;i<n;++i) {
        if(h.node_blocks[i]<0||h.node_blocks[i]>=h.block_count)throw std::invalid_argument("GPU block index");
        if(h.node_body[i]<-1||h.node_body[i]>=static_cast<int>(b))throw std::invalid_argument("GPU body index");
        if(bool(h.deformable[i])!=(h.node_body[i]<0))throw std::invalid_argument("GPU node classification");
        if(h.deformable[i] && seen_blocks[h.node_blocks[i]]++)
            throw std::invalid_argument("GPU deformable nodes must have distinct graph blocks");
        if(!(h.mass[i]>=0)||!std::isfinite(h.mass[i]))throw std::invalid_argument("GPU invalid mass");
        for(int a=0;a<3;++a)if(!std::isfinite(h.positions[i][a])||!std::isfinite(h.velocities[i][a]))
            throw std::invalid_argument("GPU nonfinite initial state");
    }
    std::vector<unsigned char> seen_body_nodes(n,0);
    for(std::size_t body=0;body<b;++body) {
        if(h.body_blocks[body]<0||h.body_blocks[body]>=h.block_count||seen_blocks[h.body_blocks[body]]++)
            throw std::invalid_argument("GPU rigid body graph block must be distinct");
        for(auto i=h.body_node_offsets[body];i<h.body_node_offsets[body+1];++i) {
            const auto node=h.body_nodes[i];
            if(node<0||static_cast<std::size_t>(node)>=n||h.node_body[node]!=static_cast<int>(body)
                ||h.node_blocks[node]!=h.body_blocks[body])throw std::invalid_argument("GPU body node ownership");
            if(seen_body_nodes[node]++)throw std::invalid_argument("GPU duplicate rigid node");
        }
    }
    for(std::size_t node=0;node<n;++node)
        if((h.node_body[node]>=0)!=bool(seen_body_nodes[node]))throw std::invalid_argument("GPU missing rigid node");
    const auto node_valid=[n](int node){if(node<0||static_cast<std::size_t>(node)>=n)throw std::invalid_argument("GPU topology node index");};
    for(const auto& triangle:h.surface_triangles) {
        for(int node:triangle.vertices)node_valid(node);
        const int owner=h.node_body[triangle.vertices[0]];
        if(owner!=h.node_body[triangle.vertices[1]]||owner!=h.node_body[triangle.vertices[2]])
            throw std::invalid_argument("GPU triangle has mixed rigid ownership");
    }
    for(const auto& edge:h.edges) {
        for(int node:edge.vertices)node_valid(node);
        if(h.node_body[edge.vertices[0]]!=h.node_body[edge.vertices[1]])
            throw std::invalid_argument("GPU edge has mixed rigid ownership");
    }
    for(const auto& dep:h.dependencies) {
        if(dep.count<1||dep.count>4)throw std::invalid_argument("GPU dependency size");
        for(int k=0;k<dep.count;++k)node_valid(dep.vertices[k]);
    }
}
// Serialize host orchestration of the default-stream GPU solver. Exact
// workers have independent scratch; one solver's internal jobs still run in
// parallel on the device.
std::mutex gpu_mutex;
}

struct GpuSolver::Impl {
    DeviceArray<Vector3> positions,velocities,previous,predicted,pin_targets,reference,proposed,accepted;
    DeviceArray<unsigned char> deformable,solid,pinned,surface,body_translate,body_rotate;
    DeviceArray<unsigned char> original_selected;
    DeviceArray<double> mass,body_mass;
    DeviceArray<int> node_blocks,node_body,body_blocks,body_nodes,error,canonical_block_nodes;
    DeviceArray<unsigned int> block_stages,color_stages;
    std::vector<unsigned int> active_color_stages;
    unsigned int enabled_stages=0;
    DeviceArray<std::size_t> triangle_offsets,solid_offsets,hinge_offsets,body_node_offsets;
    DeviceArray<TriangleIncident> triangles;
    DeviceArray<SolidIncident> solids;
    DeviceArray<HingeIncident> hinges;
    DeviceArray<BpTriangle> surface_triangles;
    DeviceArray<BpEdge> edges;
    DeviceArray<BpDependency> dependencies;
    DeviceArray<BpAabb> boxes,body_boxes;
    DeviceArray<Point> gathered_points;
    DeviceArray<Triangle> gathered_triangles;
    DeviceArray<Solid> gathered_solids;
    DeviceArray<Hinge> gathered_hinges;
    DeviceArray<Derivatives> point_outputs,triangle_outputs,solid_outputs,hinge_outputs,systems,body_systems;
    DeviceArray<Vector3> com,com_velocity,omega,previous_com,previous_com_velocity,previous_omega;
    DeviceArray<Quaternion> orientation,previous_orientation,box_orientation;
    DeviceArray<Matrix3> second_moment;
    DeviceArray<Vector3> proposed_com,proposed_omega,accepted_com,accepted_omega;
    DeviceArray<Quaternion> proposed_orientation,accepted_orientation;
    DeviceArray<SdfCollider> colliders;
    DeviceArray<RigidPinConstraint> rigid_pins;
    DeviceArray<std::size_t> rigid_pin_offsets;
    RigidPinScene pin_scene{};
    DeformableScene scene{};DeformableState state{};ElasticWorkspace elastic{};
    RigidScene rigid{};RigidState rigid_state{};RigidWorkspace rigid_work{};
    BroadPhaseTopology topology{};GpuBroadPhase broadphase;BroadPhaseView broad{};
    // Keep packet storage bounded while amortizing the large rigid scenes'
    // gather/reduction launches. Direct pipeline capacities are unchanged.
    GpuContactPipeline contacts{static_cast<std::size_t>(ExactContactSchedule::environment_integer(
        "IPC_GPU_CONTACT_PACKET_CAPACITY",1<<18,1<<20))};
    GpuLinearClipper clipper;
    // validate(h) rejects duplicate/out-of-range CSR proxy nodes before these
    // owners can launch; the DeviceArrays own disjoint storage. Reference
    // values still retain device preflight checks.
    GpuRigidRotationClipper rotation_clipper{true};
    GpuRigidProxyPipeline proxy_pipeline{true};
    const bool rotation_batch_enabled=contact_execution_option("IPC_GPU_RIGID_ROTATION_BATCH",true);
    GpuElasticWorklists elastic_worklists;
    GpuAdaptiveBoxes adaptive_boxes;
    GpuGlobalGuess global_guess;
    std::unique_ptr<level_predictor::GpuLevelSchedule> level_schedule;
    GpuBroadPhase level_broadphase;
    DeviceArray<BpAabb> level_boxes;
    DeviceArray<Vector3> fraction_targets;
    // CPU raw intended displacement ((xhat-x)*fraction) of the current original
    // predictor epoch. Fixed obstacles store exactly zero.
    DeviceArray<Vector3> original_guess_raw;
    DeviceArray<Vector3> predictor_sweep_start;
    DeviceArray<unsigned> predictor_bits;
    DeviceArray<int> predictor_done;
    bool invalid=false,cloth_only=false;

    explicit Impl(const SceneDescription& h) {
        validate(h);const auto n=h.positions.size(),b=h.com.size();
        if(two_rigid_color_queue::eligible_validated_scene(h))
            contacts.bind_validated_two_rigid_blocks(n);
        rigid_pins.upload(h.rigid_pins);
        rigid_pin_offsets.upload(h.rigid_pin_offsets.empty()?std::vector<std::size_t>(b+1,0):h.rigid_pin_offsets);
        pin_scene={b,h.rigid_pins.size(),rigid_pins.data(),rigid_pin_offsets.data()};
        cloth_only=b==0&&std::none_of(h.solid.begin(),h.solid.end(),
            [](unsigned char solid){return solid!=0;});
        positions.upload(h.positions);velocities.upload(h.velocities);previous.resize(n);predicted.resize(n);
        pin_targets.upload(h.pin_targets);reference.upload(h.body_reference);proposed.resize(n);accepted.resize(n);
        deformable.upload(h.deformable);solid.upload(h.solid);pinned.upload(h.pinned);surface.upload(h.surface);
        original_selected.resize(n);
        if(cloth_only) {
            std::vector<int> block_nodes(static_cast<std::size_t>(h.block_count),-1);
            for(std::size_t node=0;node<n;++node)
                if(h.deformable[node])block_nodes[h.node_blocks[node]]=static_cast<int>(node);
            canonical_block_nodes.upload(block_nodes);
        }
        body_translate.upload(h.body_translate);body_rotate.upload(h.body_rotate);
        mass.upload(h.mass);body_mass.upload(h.body_mass);node_blocks.upload(h.node_blocks);node_body.upload(h.node_body);
        body_blocks.upload(h.body_blocks);body_nodes.upload(h.body_nodes);error.resize(1);error.clear();
        // These are immutable object/update-mode metadata, not numerical state.
        // Soft-pinned cloth/solid nodes must remain active.
        std::vector<unsigned int> stages(static_cast<std::size_t>(h.block_count),0);
        for(std::size_t node=0;node<n;++node)
            if(h.deformable[node])stages[h.node_blocks[node]]|=DeformableColorStage;
        for(std::size_t body=0;body<b;++body) {
            if(h.body_translate[body])stages[h.body_blocks[body]]|=RigidTranslationColorStage;
            if(h.body_rotate[body])stages[h.body_blocks[body]]|=RigidOrientationColorStage;
        }
        for(const auto stages_for_block:stages)enabled_stages|=stages_for_block;
        block_stages.upload(stages);color_stages.resize(stages.size());
        triangle_offsets.upload(h.triangle_offsets);solid_offsets.upload(h.solid_offsets);hinge_offsets.upload(h.hinge_offsets);
        body_node_offsets.upload(h.body_node_offsets);triangles.upload(h.triangles);solids.upload(h.solids);hinges.upload(h.hinges);
        surface_triangles.upload(h.surface_triangles);edges.upload(h.edges);dependencies.upload(h.dependencies);
        boxes.resize(n);body_boxes.resize(b);gathered_points.resize(n);gathered_triangles.resize(h.triangles.size());
        gathered_solids.resize(h.solids.size());gathered_hinges.resize(h.hinges.size());point_outputs.resize(n);
        triangle_outputs.resize(h.triangles.size());solid_outputs.resize(h.solids.size());hinge_outputs.resize(h.hinges.size());
        systems.resize(n);body_systems.resize(b);
        com.upload(h.com);com_velocity.upload(h.com_velocity);orientation.upload(h.orientation);omega.upload(h.omega);
        previous_com.resize(b);previous_com_velocity.resize(b);previous_orientation.resize(b);previous_omega.resize(b);box_orientation.resize(b);
        second_moment.upload(h.body_second_moment);proposed_com.resize(b);proposed_omega.resize(b);proposed_orientation.resize(b);
        accepted_com.resize(b);accepted_omega.resize(b);accepted_orientation.resize(b);

        scene={n,h.triangles.size(),h.solids.size(),h.hinges.size(),node_blocks.data(),deformable.data(),solid.data(),pinned.data(),
            mass.data(),v3(pin_targets),triangles.data(),solids.data(),hinges.data(),triangle_offsets.data(),solid_offsets.data(),hinge_offsets.data()};
        elastic={gathered_points.data(),gathered_triangles.data(),gathered_solids.data(),gathered_hinges.data(),point_outputs.data(),
            triangle_outputs.data(),solid_outputs.data(),hinge_outputs.data(),systems.data(),v3(proposed),error.data()};
        rigid={b,body_blocks.data(),node_body.data(),v3(reference),body_node_offsets.data(),body_nodes.data(),body_mass.data(),
            m9(second_moment),body_translate.data(),body_rotate.data()};
        refresh_state_views();
        topology.vertex_count=static_cast<int>(n);topology.block_count=h.block_count;
        topology.triangle_count=static_cast<int>(h.surface_triangles.size());topology.edge_count=static_cast<int>(h.edges.size());
        topology.dependency_count=static_cast<int>(h.dependencies.size());topology.triangles=surface_triangles.data();
        topology.edges=edges.data();topology.dependencies=dependencies.data();topology.vertex_block=node_blocks.data();topology.surface_query=surface.data();
        topology.protect_linear_gap=true;
    }
    // Bind non-owning views to the current device storage owners.
    void refresh_state_views() {
        state={v3(positions),v3(velocities),v3(previous),v3(predicted)};
        rigid_state={v3(com),v3(com_velocity),q4(orientation),v3(omega),v3(previous_com),v3(previous_com_velocity),q4(previous_orientation),v3(previous_omega)};
        rigid_work={body_systems.data(),v3(proposed_com),v3(proposed_omega),q4(proposed_orientation),v3(accepted_com),q4(accepted_orientation),v3(accepted_omega)};
    }
    void check_status() {
        const auto code=error.download()[0];
        if(code) {invalid=true;throw std::runtime_error("GPU substep failed, device status "+std::to_string(code)+"; no CPU fallback and no frame output");}
    }
    auto prepare_colored_targets(double fraction) -> const double (*)[3] {
        // Preserve the old default's exact arithmetic and launch/allocation path.
        if(fraction==1.)return state.predicted;
        fraction_targets.resize(scene.nodes);
        if(scene.nodes)colored_guess_targets<<<blocks(scene.nodes),128>>>(scene,state,fraction,v3(fraction_targets),nullptr,error.data());
        cuda_checked(cudaGetLastError());check_status();
        return v3(fraction_targets);
    }
    // ORIGINAL predictor targets: the CPU always forms target = previous + raw
    // with raw = (xhat - previous) * fraction, even for fraction 1 (where that
    // sum need not reproduce the rounded xhat). The same kernel publishes the
    // raw array that the CPU false movement boxes are padded with.
    auto prepare_original_guess_targets(double fraction) -> const double (*)[3] {
        fraction_targets.resize(scene.nodes);
        original_guess_raw.resize(scene.nodes);
        if(scene.nodes)colored_guess_targets<<<blocks(scene.nodes),128>>>(scene,state,fraction,
            v3(fraction_targets),v3(original_guess_raw),error.data());
        cuda_checked(cudaGetLastError());check_status();
        return v3(fraction_targets);
    }
    auto original_guess_raw_view() -> const double (*)[3] {return v3(original_guess_raw);}
    // The CPU 1e-28 cutoff machinery belongs only to the cloth
    // ColoredContactSweep family. Solid/mixed/rigid scenes use a different CPU
    // dispatcher, so their vertices stay StoredNoop. This is scheduler-FAMILY
    // metadata: it does NOT depend on the exact_computation_fallback geometry
    // flag, because the CPU builds the same selected set for both flags.
    bool cooperative_family(const StepParameters& p) const {
        return cloth_only
            && p.original_update_convention
                ==OriginalUpdateConvention::CooperativeSquaredNorm
            && p.original_reference_threads>1;
    }
    // CPU validate_cooperative_step = exact && use_ccd && !ticcd && !ogc.
    // It selects the exact stored-change test for selected vertices instead of
    // the 1e-28 squared cutoff. Tight-Inclusion CCD is unported here, so the
    // GPU equivalent is `collisions`.
    bool cooperative_exact_change(const StepParameters& p) const {
        return p.exact_computation_fallback && p.collisions;
    }
    // Selected vertices use the squared cutoff exactly when the CPU's
    // validate_cooperative_step rule is false: original mode with CCD on or
    // off, and robust exact mode with CCD disabled.
    bool selected_squared_cutoff(const StepParameters& p) const {
        return cooperative_family(p) && !cooperative_exact_change(p);
    }
    // Effective predictor activity: the narrow appended intent flag, a positive
    // sweep budget, or the level-based request. All three validate forced
    // positions/targets even for zero sweeps, as the CPU helper does.
    bool colored_guess_active(const StepParameters& p) const {
        return p.colored_guess_enabled || p.colored_guess_sweeps>0
            || p.colored_guess_level_based;
    }
    void rebuild(const StepParameters& p,bool guess,const double (*guess_targets)[3]=nullptr) {
        if(guess&&scene.nodes&&!guess_targets)throw std::logic_error("GPU colored guess targets missing");
        {
        if(guess)make_motion_boxes(reinterpret_cast<const double*>(positions.data()),reinterpret_cast<const double*>(guess_targets),
            topology.vertex_count,.2,p.exact_computation_fallback?1e-8:
                (p.contact.d_hat>0.0?p.contact.d_hat:1e-8),boxes.data(),
            !p.exact_computation_fallback);
        else adaptive_boxes.build(scene,state,boxes.data(),error.data());
        build_rigid_blue_boxes(rigid,rigid_state,scene.nodes,p.box_radius,body_boxes.data(),boxes.data(),error.data(),
            p.rotation_radius,q4(box_orientation),state.positions);
        cuda_checked(cudaGetLastError());check_status();
        }
        auto graph=topology;
        graph.protect_linear_gap=p.exact_computation_fallback;
        if(guess){graph.dependency_count=0;graph.include_triangle_dependencies=false;}
        broad=broadphase.rebuild(graph,boxes.data(),p.contact.d_hat);
        elastic_worklists.rebuild(scene,broad.block_colors,topology.block_count,broad.color_count);
        {
        // Certificates are invalidated/recomputed with every blue-box rebuild;
        // the original graph/coloring above remains untouched. Without CCD the
        // solver is not bound to these motion boxes, so disable this cache.
        ContactBoxPruning pruning;
        if((guess||p.collisions)&&p.exact_computation_fallback)
            pruning={boxes.data(),scene.deformable,state.positions,state.previous,p.contact};
        else if(!guess&&!p.exact_computation_fallback&&p.cpu_energy_simd
            &&cloth_only&&p.collisions&&p.contact.d_hat>1e-8
            &&std::isfinite(p.contact.d_hat)&&p.contact.k_barrier>=0.0
            &&std::isfinite(p.contact.k_barrier)&&p.contact.dt>0.0
            &&std::isfinite(p.contact.dt)&&std::isfinite(p.contact.dt*p.contact.dt)
            &&std::isfinite((p.contact.dt*p.contact.dt)*p.contact.k_barrier))
            pruning={boxes.data(),scene.deformable,state.positions,state.previous,p.contact,true};
        contacts.rebuild(broad,topology.block_count,node_blocks.data(),surface.data(),pruning);
        // Cooperative family: rebuild the selected-node mask from THIS
        // rebuild's colors and incidence counts. Mode-dependent caches are
        // invalidated every rebuild. The mask is only computed when the CPU
        // selected-vertex squared cutoff actually applies (original mode with
        // either CCD setting, or robust exact mode with CCD disabled), so the
        // robust exact CCD path performs no extra work. Only the cloth
        // experimental solver owns CPU ColoredContactSweep, so a
        // general/solid/rigid scene stays StoredNoop.
        if(selected_squared_cutoff(p)&&scene.nodes)
            original_cooperative_mask<<<blocks(scene.nodes),128>>>(scene,broad,
                contacts.incidence().block_offsets,contacts.incidence().raw_block_counts,
                canonical_block_nodes.data(),topology.block_count,
                p.original_reference_threads,original_selected.data());
        }
        collect_color_stage_masks(broad,topology.block_count,block_stages.data(),color_stages.data());
        active_color_stages.resize(static_cast<std::size_t>(broad.color_count));
        if(broad.color_count)cuda_checked(cudaMemcpy(active_color_stages.data(),color_stages.data(),
            active_color_stages.size()*sizeof(unsigned int),cudaMemcpyDeviceToHost));
    }
    // `raw_guess` is the CPU raw intended displacement ((xhat-x)*fraction).
    // Original mode requires it for the movement boxes; robust ignores it.
    void run_level_guess(const StepParameters& p,const double (*targets)[3],
                         const double (*raw_guess)[3]) {
        if(!scene.nodes||p.colored_guess_sweeps<=0)return;
        if(!level_schedule)level_schedule=std::make_unique<level_predictor::GpuLevelSchedule>();
        // The trial does not touch adaptive-box history, previous positions,
        // velocities or rigid state. Rigid predicted positions equal starts.
        // Keep its boxes separate from the normal solver's blue regions.
        level_boxes.resize(scene.nodes);
        if(p.exact_computation_fallback)
            make_motion_boxes(reinterpret_cast<const double*>(positions.data()),
                reinterpret_cast<const double*>(targets),topology.vertex_count,.2,1e-8,
                level_boxes.data(),false);
        else {
            if(!raw_guess)throw std::logic_error("GPU original predictor boxes require raw displacement");
            make_original_guess_boxes(reinterpret_cast<const double*>(positions.data()),
                reinterpret_cast<const double*>(targets),
                reinterpret_cast<const double*>(raw_guess),topology.vertex_count,
                level_boxes.data());
        }
        auto graph=topology;graph.dependency_count=0;graph.include_triangle_dependencies=false;
        graph.protect_linear_gap=p.exact_computation_fallback;
        // protect_linear_gap pads queries from the complete allowed boxes,
        // including large-coordinate endpoint floors, entirely on the GPU.
        const auto candidates=level_broadphase.rebuild(graph,level_boxes.data(),1e-8,BroadPhaseMode::CandidatesOnly);
        // CPU false predictor: fixed 1e-8 gap, no heavy-first reorder, and the
        // false projected supporting-plane certificate. Robust keeps the
        // existing outward certificate and heavy-first order.
        level_schedule->prepare(topology.vertex_count,candidates.contacts,candidates.contact_count,
            state.positions,targets,level_boxes.data(),scene.deformable,1e-8,
            p.exact_computation_fallback,error.data(),!p.exact_computation_fallback);
        // Positive-budget inputs and movement boxes are ready for the trial.
        check_status();
        predictor_sweep_start.resize(scene.nodes);
        predictor_bits.resize(1);predictor_done.resize(1);predictor_done.clear();
        const auto bind=[&](level_predictor::DeviceSchedule schedule) {
            BroadPhaseView phase{};
            phase.contacts=level_schedule->retained_contacts();
            phase.contact_count=level_schedule->retained_contact_count();
            phase.block_colors=level_schedule->project_blocks(scene.node_blocks,topology.block_count,error.data());
            phase.color_count=schedule.groups;
            // The retained array is EXACTLY the graph's pruning result.
            // Disable secondary pruning and rebuild all compacted pair IDs.
            contacts.rebuild(phase,topology.block_count,scene.node_blocks,surface.data());
            return phase;
        };
        const auto sweep=[&](const BroadPhaseView& phase,const int* predictor_join) {
            for(int color=0;color<phase.color_count;++color) {
                if(scene.nodes)predicted_targets<<<blocks(scene.nodes),128>>>(scene,targets,state,elastic,
                    phase.block_colors,color,predictor_join);
                const auto incidence=contacts.incidence(color);
                clipper.clip_deformable_color(scene,state,elastic,phase,
                    incidence.block_offsets,incidence.pair_indices,level_boxes.data(),phase.block_colors,color,
                    v3(accepted),error.data(),true,nullptr,incidence.color_pairs,p.ccd_exact_step_bound,
                    p.exact_computation_fallback?LinearClipMode::Robust:LinearClipMode::OriginalColoredGuess,
                    OriginalUpdateConvention::StoredNoop,nullptr);
                // Vertex trial colors can outnumber update blocks because
                // rigid proxy vertices remain separate graph obstacles. The
                // existing full-node commit avoids elastic worklist limits.
                commit_deformable_endpoints(scene,state,elastic,phase.block_colors,color,v3(accepted));
            }
            check_status();
        };
        const auto trial_phase=bind(level_schedule->trial());
        // Exactly one discarded ORIGINAL-color probe; it never consumes the
        // requested actual budget and never establishes convergence.
        sweep(trial_phase,nullptr);
        const auto scheduled=level_schedule->finish_trial(state.positions,error.data());
        if(scene.nodes)cuda_checked(cudaMemcpy(positions.data(),level_schedule->starts(),
            scene.nodes*sizeof(Vector3),cudaMemcpyDeviceToDevice));
        const auto actual_phase=bind(scheduled);
        // The discarded sweep never consumes the requested actual budget or
        // establishes convergence. Default-stream launch order joins every
        // color group before the next one can inspect its positions.
        predictor_done.clear();
        for(int iteration=0;iteration<p.colored_guess_sweeps;++iteration) {
            cuda_checked(cudaMemcpyAsync(predictor_sweep_start.data(),positions.data(),
                scene.nodes*sizeof(Vector3),cudaMemcpyDeviceToDevice));
            predictor_bits.clear();
            sweep(actual_phase,predictor_done.data());
            predictor_sweep_status<<<blocks(scene.nodes),128>>>(scene,v3(positions),
                v3(predictor_sweep_start),targets,predictor_bits.data());
            predictor_join_status<<<1,1>>>(predictor_bits.data(),predictor_done.data());
            cuda_checked(cudaGetLastError());
        }
        check_status();
    }
    // CPU LEGACY original colored guess (`use_colored_ccd_guess` without the
    // level-based flag). It shares the ORIGINAL-color preparation with the
    // level carrier — the same CPU false movement boxes, fixed 1e-8 candidate
    // gap, false complete-four-box projected prune, contact-only ascending
    // greedy cliques and fixed targets — but gains NO discarded probe and no
    // BFS scheduling: its first actual sweep consumes the requested budget.
    // Whole-sweep stopping is the CPU rule: after all colors join, stop once
    // no vertex moved or every target squared error is below 1e-28. The flag
    // stays on the device (no host readback); later sweeps become no-ops.
    void run_legacy_original_guess(const StepParameters& p,const double (*targets)[3],
                                   const double (*raw_guess)[3]) {
        if(!scene.nodes||p.colored_guess_sweeps<=0)return;
        if(!level_schedule)level_schedule=std::make_unique<level_predictor::GpuLevelSchedule>();
        if(!raw_guess)throw std::logic_error("GPU original predictor boxes require raw displacement");
        level_boxes.resize(scene.nodes);
        make_original_guess_boxes(reinterpret_cast<const double*>(positions.data()),
            reinterpret_cast<const double*>(targets),
            reinterpret_cast<const double*>(raw_guess),topology.vertex_count,
            level_boxes.data());
        auto graph=topology;graph.dependency_count=0;graph.include_triangle_dependencies=false;
        graph.protect_linear_gap=false;
        const auto candidates=level_broadphase.rebuild(graph,level_boxes.data(),1e-8,
            BroadPhaseMode::CandidatesOnly);
        // False projected certificate, ascending contact-only coloring (no
        // heavy-first reordering), fixed 1e-8 gap.
        level_schedule->prepare(topology.vertex_count,candidates.contacts,candidates.contact_count,
            state.positions,targets,level_boxes.data(),scene.deformable,1e-8,
            false,error.data(),true);
        // Positive-budget inputs and movement boxes are ready for the sweep.
        check_status();
        predictor_sweep_start.resize(scene.nodes);
        predictor_bits.resize(1);predictor_done.resize(1);predictor_done.clear();
        const auto colors=level_schedule->trial();
        BroadPhaseView phase{};
        phase.contacts=level_schedule->retained_contacts();
        phase.contact_count=level_schedule->retained_contact_count();
        phase.block_colors=level_schedule->project_blocks(scene.node_blocks,topology.block_count,
            error.data());
        phase.color_count=colors.groups;
        // The retained array is exactly this graph's pruning result; disable
        // secondary pruning and renumber the compacted pairs.
        contacts.rebuild(phase,topology.block_count,scene.node_blocks,surface.data());
        for(int iteration=0;iteration<p.colored_guess_sweeps;++iteration) {
            cuda_checked(cudaMemcpyAsync(predictor_sweep_start.data(),positions.data(),
                scene.nodes*sizeof(Vector3),cudaMemcpyDeviceToDevice));
            predictor_bits.clear();
            for(int color=0;color<phase.color_count;++color) {
                if(scene.nodes)predicted_targets<<<blocks(scene.nodes),128>>>(scene,targets,state,
                    elastic,phase.block_colors,color,predictor_done.data());
                const auto incidence=contacts.incidence(color);
                clipper.clip_deformable_color(scene,state,elastic,phase,
                    incidence.block_offsets,incidence.pair_indices,level_boxes.data(),
                    phase.block_colors,color,v3(accepted),error.data(),true,nullptr,
                    incidence.color_pairs,p.ccd_exact_step_bound,
                    LinearClipMode::OriginalColoredGuess,
                    OriginalUpdateConvention::StoredNoop,nullptr);
                commit_deformable_endpoints(scene,state,elastic,phase.block_colors,color,
                    v3(accepted));
            }
            check_status();
            predictor_sweep_status<<<blocks(scene.nodes),128>>>(scene,v3(positions),
                v3(predictor_sweep_start),targets,predictor_bits.data());
            predictor_join_status<<<1,1>>>(predictor_bits.data(),predictor_done.data());
            cuda_checked(cudaGetLastError());
        }
        check_status();
    }
    void deformable_step(int color,bool use_ccd,bool exact_step_bound,
        LinearClipMode mode,OriginalUpdateConvention convention,
        const unsigned char* original_cooperative_selected) {
        const auto incidence=contacts.incidence(color);
        {
        clipper.clip_deformable_color(scene,state,elastic,broad,incidence.block_offsets,incidence.pair_indices,
            boxes.data(),broad.block_colors,color,v3(accepted),error.data(),use_ccd,nullptr,incidence.color_pairs,exact_step_bound,
            mode,convention,original_cooperative_selected);
        }
        commit_deformable_endpoints(scene,state,elastic,broad.block_colors,color,v3(accepted),nullptr,elastic_worklists.color(color));
    }
};

GpuSolver::GpuSolver(const SceneDescription& h):impl_(std::make_unique<Impl>(h)){}
GpuSolver::~GpuSolver()=default;


void GpuSolver::advance_substep(const StepParameters& p,const std::vector<Vector3>& pins,
                                     const std::vector<SdfCollider>& colliders) {
    const std::lock_guard<std::mutex> lock(gpu_mutex);
    auto& s=*impl_;
    if(s.invalid)throw std::runtime_error("GPU state invalidated by earlier failure");
    // Direct GpuSolver API must reject the unqualified original+direct-bound
    // combination before any mutation, exactly like the CPU/bridge validation.
    if(!p.exact_computation_fallback&&p.ccd_exact_step_bound)
        throw std::invalid_argument(
            "GPU original mode does not support ccd_exact_step_bound");
    if(p.original_reference_threads<=0||p.original_reference_threads>4096)
        throw std::invalid_argument(
            "GPU original_reference_threads must be a positive bounded logical count");
    // The cooperative selected-node cutoff models the CPU's canonical greedy
    // color order. Unqualified alternative GPU colorings are rejected rather
    // than assumed to match it. The family is independent of the geometry
    // flag, so this guard also covers robust exact mode with CCD disabled.
    if(s.selected_squared_cutoff(p)) {
        const char* coloring=std::getenv("IPC_GPU_COLORING");
        if(coloring&&*coloring&&std::string(coloring)!="greedy")
            throw std::invalid_argument(
                "GPU cooperative selected-step mode requires the default greedy coloring");
    }
    if(!(p.dt>0)||!std::isfinite(p.dt)||!std::isfinite(p.damping)||p.iterations<0||p.colored_guess_sweeps<0
        ||!std::isfinite(p.colored_guess_fraction)||p.colored_guess_fraction<0||p.colored_guess_fraction>1)
        throw std::invalid_argument("GPU invalid step parameters");
    if(!(p.box_radius_min>1e-10)||!std::isfinite(p.box_radius_min)
        ||!(p.box_radius>=p.box_radius_min)||!std::isfinite(p.box_radius)
        ||!std::isfinite(p.rotation_radius)||p.rotation_radius<0||p.rotation_radius>3.141592653589793
        ||p.rebuild_interval<=0||(p.global_ccd_guess&&(p.colored_guess_sweeps||p.colored_guess_level_based))
        ||((p.colored_guess_level_based||p.colored_guess_sweeps)
            &&(!std::isfinite(p.contact.d_hat)||p.contact.d_hat<0)))
        throw std::invalid_argument("GPU invalid colored solver parameters");
    if(s.pin_scene.count&&p.elastic.kpin>0.0&&!std::isfinite(p.rigid_pin_time))
        throw std::invalid_argument("GPU nonfinite rigid pin time");
    if(p.contact.dt!=p.dt || p.sdf.dt!=p.dt || p.elastic.dt2!=p.dt*p.dt
        ||!(p.elastic.dt2>0)||!std::isfinite(p.elastic.dt2))
        throw std::invalid_argument("GPU inconsistent or unrepresentable timestep");
    if(colliders.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("GPU too many SDF colliders");
    length(pins.size(),s.scene.nodes,"prescribed pins");
    try {
    {
    {
    s.pin_targets.upload(pins);s.colliders.upload(colliders);s.error.clear();
    begin_deformable_substep(s.scene,s.state,p.dt);
    begin_rigid_substep(s.rigid,s.rigid_state,s.error.data());
    }
    const SdfScene sdf{s.colliders.data(),static_cast<int>(colliders.size()),p.sdf};
    const RigidPinAssembly rigid_pins{s.pin_scene,p.elastic.kpin,p.rigid_pin_time};
    // The CPU general driver disables the legacy global particle guess for
    // solid/mixed scenes. Only its cloth driver honors this default flag.
    if(p.global_ccd_guess && s.cloth_only && (s.enabled_stages&DeformableColorStage)) {
        s.global_guess.apply(s.scene,s.state,s.topology,s.error.data());
        s.check_status();
    }
    const bool guess_active=s.colored_guess_active(p)
        &&(s.enabled_stages&DeformableColorStage);
    // The CPU validates positions, intended displacements and targets even
    // for a zero budget, then returns BEFORE padded boxes, contacts or a trial.
    // Run the existing target kernel in validation-only mode; unlike the
    // robust fraction-one shortcut, this checks subtraction/scaling overflow.
    if(guess_active&&p.colored_guess_sweeps==0) {
        if(s.scene.nodes)colored_guess_targets<<<blocks(s.scene.nodes),128>>>(
            s.scene,s.state,p.colored_guess_fraction,nullptr,nullptr,s.error.data());
        cuda_checked(cudaGetLastError());s.check_status();
    }
    const bool guess_updates=guess_active&&p.colored_guess_sweeps>0;
    // Original updating predictors retain the raw displacement used by their
    // CPU movement boxes. Robust positive-budget target arithmetic is intact.
    const double (*raw_guess)[3]=nullptr;
    const auto targets=!guess_updates?s.state.predicted
        :(p.exact_computation_fallback?s.prepare_colored_targets(p.colored_guess_fraction)
          :s.prepare_original_guess_targets(p.colored_guess_fraction));
    if(!guess_updates) {
        // An input-only predictor request was validated above; no boxes or
        // graph/probe/update work is allowed for its zero budget.
    } else if(!p.exact_computation_fallback) {
        raw_guess=s.original_guess_raw_view();
        if(p.colored_guess_level_based) {
            // Positive budget: build boxes and execute the scheduled predictor.
            s.run_level_guess(p,targets,raw_guess);
        } else {
            s.run_legacy_original_guess(p,targets,raw_guess);
        }
    } else if(p.colored_guess_level_based) {
        // Positive budget: build boxes and execute the scheduled predictor.
        s.run_level_guess(p,targets,nullptr);
    } else if(p.colored_guess_sweeps) {
        s.rebuild(p,true,targets);
        for(int iteration=0;iteration<p.colored_guess_sweeps;++iteration)
            for(int color=0;color<s.broad.color_count;++color) {
                if(!(s.active_color_stages[color]&DeformableColorStage))continue;
                if(s.scene.nodes)predicted_targets<<<blocks(s.scene.nodes),128>>>(s.scene,targets,s.state,s.elastic,s.broad.block_colors,color,nullptr);
                // A CCD initial guess must remain collision-safe even when
                // subsequent particle solver updates explicitly disable CCD.
                s.deformable_step(color,true,p.ccd_exact_step_bound,
                    LinearClipMode::Robust,
                    OriginalUpdateConvention::StoredNoop,nullptr);
            }
        s.check_status();
    }
    // Match the CPU solver's displacement history: its entry anchor is after
    // the initial guess, while inertial displacement uses substep velocities.
    {
        s.adaptive_boxes.begin_solver(s.scene,s.state,p.dt,p.box_radius_min,p.box_radius,s.error.data());
    }
    for(int iteration=0;iteration<p.iterations && s.enabled_stages;++iteration) {
        if(iteration%p.rebuild_interval==0)s.rebuild(p,false);
        for(int color=0;color<s.broad.color_count;++color) {
            const unsigned int stages=s.active_color_stages[color];
            if(stages&DeformableColorStage) {
                const auto worklist=s.elastic_worklists.color(color);
                assemble_elastic_color(s.scene,s.state,s.elastic,s.broad.block_colors,color,p.elastic,p.solid,nullptr,worklist);
                s.contacts.assemble_color(s.scene,s.state,s.elastic,s.rigid,s.rigid_state,s.rigid_work,sdf,p.contact,color,ContactAssemblyPhase::Deformable,
                    ContactAssemblyMode::ColorSweep,{},
                    p.exact_computation_fallback?ContactGeometryPolicy::Robust:ContactGeometryPolicy::Original);
                {
                propose_deformable_color(s.scene,s.state,s.elastic,s.broad.block_colors,color,p.damping,nullptr,worklist);
                }
                s.deformable_step(color,p.collisions,p.ccd_exact_step_bound,
                    p.exact_computation_fallback?LinearClipMode::Robust:LinearClipMode::OriginalSolver,
                    s.cooperative_family(p)
                        ?OriginalUpdateConvention::CooperativeSquaredNorm
                        :OriginalUpdateConvention::StoredNoop,
                    s.selected_squared_cutoff(p)?s.original_selected.data():nullptr);
            }
            if(stages&RigidTranslationColorStage) {
                {
                assemble_rigid_inertia_color(s.rigid,s.rigid_state,s.rigid_work,s.broad.block_colors,color,false,p.dt,p.elastic.gravity[1],s.error.data());
                }
                s.contacts.assemble_color(s.scene,s.state,s.elastic,s.rigid,s.rigid_state,s.rigid_work,sdf,p.contact,color,
                    ContactAssemblyPhase::RigidTranslation,ContactAssemblyMode::ColorSweep,rigid_pins,
                    p.exact_computation_fallback?ContactGeometryPolicy::Robust:ContactGeometryPolicy::Original);
                {
                propose_rigid_color(s.rigid,s.rigid_state,s.rigid_work,s.broad.block_colors,color,false,s.body_boxes.data(),p.dt,p.damping,s.error.data(),p.rotation_radius,q4(s.box_orientation));
                }
                const auto incidence=s.contacts.incidence();
                {
                // validate(h) established unique/disjoint rigid CSR ownership;
                // these current DeviceArray views are bound for this one call.
                rigid_proxy_owner::Storage proxy_owner;
                proxy_owner.vertices=s.scene.nodes;proxy_owner.proxy_nodes=s.body_nodes.size();
                proxy_owner.bodies=s.rigid.bodies;proxy_owner.blocks=static_cast<std::size_t>(s.topology.block_count);
                proxy_owner.positions=s.state.positions;proxy_owner.com=s.rigid_state.com;
                proxy_owner.proposed_com=s.rigid_work.proposed_com;proxy_owner.accepted_com=s.rigid_work.accepted_com;
                proxy_owner.accepted=v3(s.accepted);proxy_owner.boxes=s.boxes.data();
                proxy_owner.colors=s.broad.block_colors;proxy_owner.error=s.error.data();
                proxy_owner.node_offsets=s.rigid.node_offsets;proxy_owner.nodes=s.rigid.nodes;
                proxy_owner.node_body=s.rigid.node_body;proxy_owner.body_blocks=s.rigid.body_blocks;
                proxy_owner.translate=s.rigid.translate;proxy_owner.rotate=s.rigid.rotate;
                rigid_proxy_owner::Scope proxy_scope(&s.clipper,proxy_owner);
                s.clipper.clip_rigid_translation_color(s.rigid,s.rigid_state,s.rigid_work,s.state,s.broad,
                    incidence.block_offsets,incidence.pair_indices,s.boxes.data(),s.broad.block_colors,color,v3(s.accepted),s.error.data(),true,nullptr,
                    p.exact_computation_fallback?LinearClipMode::Robust:LinearClipMode::OriginalSolver);
                }
                s.proxy_pipeline.commit_translation_color(s.rigid,s.rigid_state,s.rigid_work,s.state,s.broad.block_colors,color,v3(s.accepted),s.error.data());
            }
            if(stages&RigidOrientationColorStage) {
                {
                assemble_rigid_inertia_color(s.rigid,s.rigid_state,s.rigid_work,s.broad.block_colors,color,true,p.dt,p.elastic.gravity[1],s.error.data());
                }
                s.contacts.assemble_color(s.scene,s.state,s.elastic,s.rigid,s.rigid_state,s.rigid_work,sdf,p.contact,color,
                    ContactAssemblyPhase::RigidOrientation,ContactAssemblyMode::ColorSweep,rigid_pins,
                    p.exact_computation_fallback?ContactGeometryPolicy::Robust:ContactGeometryPolicy::Original);
                {
                propose_rigid_color(s.rigid,s.rigid_state,s.rigid_work,s.broad.block_colors,color,true,s.body_boxes.data(),p.dt,p.damping,s.error.data(),p.rotation_radius,q4(s.box_orientation));
                }
                // Match general-v2: rigid CCD is mandatory, independently of the
                // particle use_ccd switch.
                {
                if(s.rotation_batch_enabled)
                    s.rotation_clipper.clip_color(s.rigid,s.rigid_state,s.rigid_work,s.state,s.broad,s.broad.block_colors,color,p.dt,.9,true,v3(s.accepted),s.error.data());
                else
                clip_rigid_rotation_color(s.rigid,s.rigid_state,s.rigid_work,s.state,s.broad,s.broad.block_colors,color,p.dt,.9,true,v3(s.accepted),s.error.data());
                }
                s.proxy_pipeline.commit_rotation_color(s.rigid,s.rigid_state,s.rigid_work,s.state,s.broad.block_colors,color,v3(s.accepted),s.error.data());
            }
        }
        s.check_status();
    }
    // The colored Gauss-Seidel path retains its sequential color barriers.
    {
    s.adaptive_boxes.finish_solver(s.scene,s.state,s.error.data());
    finish_deformable_substep(s.scene,s.state,p.dt);
    s.proxy_pipeline.finish_substep(s.rigid,s.rigid_state,s.state,p.dt,s.error.data());
    cuda_checked(cudaDeviceSynchronize());s.check_status();
    }
    }
    } catch(...) {
        // A failure can follow partial color commits. Never expose that state
        // as a completed substep or permit an accidental retry from it.
        s.invalid=true;
        throw;
    }
}

StateSnapshot GpuSolver::snapshot()const {
    const std::lock_guard<std::mutex> lock(gpu_mutex);
    const auto& s=*impl_;
    if(s.invalid)throw std::runtime_error("GPU cannot export invalid state");
    StateSnapshot out;
    out.positions=s.positions.download();out.velocities=s.velocities.download();out.com=s.com.download();
    out.com_velocity=s.com_velocity.download();out.orientation=s.orientation.download();out.omega=s.omega.download();
    return out;
}
} // namespace ipc_gpu::solver
