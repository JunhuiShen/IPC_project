#include "resident_solver.h"
#include "resident_memory.h"
#include "resident_contact_pipeline.h"
#include "resident_linear_pipeline.h"
#include "resident_rigid_pipeline.h"
#include "resident_adaptive_boxes.h"
#include "resident_global_guess.h"
#include "resident_level_schedule.h"
#include "resident_profile.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

namespace ipc_gpu::resident {
namespace {
static_assert(sizeof(Vector3)==3*sizeof(double));
static_assert(sizeof(Quaternion)==4*sizeof(double));
static_assert(sizeof(Matrix3)==9*sizeof(double));
double (*v3(DeviceArray<Vector3>& a))[3] {return reinterpret_cast<double(*)[3]>(a.data());}
double (*q4(DeviceArray<Quaternion>& a))[4] {return reinterpret_cast<double(*)[4]>(a.data());}
double (*m9(DeviceArray<Matrix3>& a))[9] {return reinterpret_cast<double(*)[9]>(a.data());}
unsigned blocks(std::size_t n) {return static_cast<unsigned>((n+127)/128);}
__global__ void predicted_targets(DeformableScene s,DeformableState x,ElasticWorkspace w,
                                   const int* colors,int color) {
    const auto n=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=s.nodes || !s.deformable[n] || colors[s.node_blocks[n]]!=color || *w.error)return;
    for(int a=0;a<3;++a)w.proposed[n][a]=x.predicted[n][a];
}
void length(std::size_t actual,std::size_t expected,const char* what) {
    if(actual!=expected)throw std::invalid_argument(std::string("resident scene size: ")+what);
}
template<class T> void offsets(const std::vector<std::size_t>& off,std::size_t nodes,
                              const std::vector<T>& records) {
    length(off.size(),nodes+1,"element offsets");
    if(off.front()!=0 || off.back()!=records.size())throw std::invalid_argument("resident element offset extent");
    for(std::size_t n=0;n<nodes;++n)
        if(off[n]>off[n+1] || off[n+1]>records.size())
            throw std::invalid_argument("resident invalid element offsets");
    for(std::size_t n=0;n<nodes;++n) {
        for(auto e=off[n];e<off[n+1];++e) {
            if(records[e].owner!=static_cast<int>(n))throw std::invalid_argument("resident incident owner mismatch");
            for(int node:records[e].nodes)if(node<0||static_cast<std::size_t>(node)>=nodes)
                throw std::invalid_argument("resident incident node index");
        }
    }
}
void validate(const SceneDescription& h) {
    const auto n=h.positions.size(),b=h.com.size();
    if(n>std::numeric_limits<int>::max() || h.block_count<0)throw std::invalid_argument("resident scene too large");
    if(h.surface_triangles.size()>std::numeric_limits<int>::max() || h.edges.size()>std::numeric_limits<int>::max()
        ||h.dependencies.size()>std::numeric_limits<int>::max())throw std::invalid_argument("resident topology too large");
    length(h.velocities.size(),n,"velocities");length(h.pin_targets.size(),n,"pins");
    length(h.body_reference.size(),n,"reference");length(h.deformable.size(),n,"deformable");
    length(h.solid.size(),n,"solid");length(h.pinned.size(),n,"pinned");length(h.surface.size(),n,"surface");
    length(h.mass.size(),n,"mass");length(h.node_blocks.size(),n,"blocks");length(h.node_body.size(),n,"owners");
    offsets(h.triangle_offsets,n,h.triangles);offsets(h.solid_offsets,n,h.solids);offsets(h.hinge_offsets,n,h.hinges);
    length(h.body_blocks.size(),b,"body blocks");length(h.body_mass.size(),b,"body mass");
    length(h.body_second_moment.size(),b,"body moment");length(h.body_translate.size(),b,"body translate");
    length(h.body_rotate.size(),b,"body rotate");length(h.com_velocity.size(),b,"body velocity");
    length(h.orientation.size(),b,"orientation");length(h.omega.size(),b,"omega");
    length(h.body_node_offsets.size(),b+1,"body offsets");
    if(h.body_node_offsets.front()!=0 || h.body_node_offsets.back()!=h.body_nodes.size())
        throw std::invalid_argument("resident body offset extent");
    for(std::size_t body=0;body<b;++body)
        if(h.body_node_offsets[body]>h.body_node_offsets[body+1] || h.body_node_offsets[body+1]>h.body_nodes.size())
            throw std::invalid_argument("resident invalid body offsets");
    std::vector<unsigned char> seen_blocks(static_cast<std::size_t>(h.block_count),0);
    for(std::size_t i=0;i<n;++i) {
        if(h.node_blocks[i]<0||h.node_blocks[i]>=h.block_count)throw std::invalid_argument("resident block index");
        if(h.node_body[i]<-1||h.node_body[i]>=static_cast<int>(b))throw std::invalid_argument("resident body index");
        if(bool(h.deformable[i])!=(h.node_body[i]<0))throw std::invalid_argument("resident node classification");
        if(h.deformable[i] && seen_blocks[h.node_blocks[i]]++)
            throw std::invalid_argument("resident deformable nodes must have distinct graph blocks");
        if(!(h.mass[i]>=0)||!std::isfinite(h.mass[i]))throw std::invalid_argument("resident invalid mass");
        for(int a=0;a<3;++a)if(!std::isfinite(h.positions[i][a])||!std::isfinite(h.velocities[i][a]))
            throw std::invalid_argument("resident nonfinite initial state");
    }
    std::vector<unsigned char> seen_body_nodes(n,0);
    for(std::size_t body=0;body<b;++body) {
        if(h.body_blocks[body]<0||h.body_blocks[body]>=h.block_count||seen_blocks[h.body_blocks[body]]++)
            throw std::invalid_argument("resident rigid body graph block must be distinct");
        for(auto i=h.body_node_offsets[body];i<h.body_node_offsets[body+1];++i) {
            const auto node=h.body_nodes[i];
            if(node<0||static_cast<std::size_t>(node)>=n||h.node_body[node]!=static_cast<int>(body)
                ||h.node_blocks[node]!=h.body_blocks[body])throw std::invalid_argument("resident body node ownership");
            if(seen_body_nodes[node]++)throw std::invalid_argument("resident duplicate rigid node");
        }
    }
    for(std::size_t node=0;node<n;++node)
        if((h.node_body[node]>=0)!=bool(seen_body_nodes[node]))throw std::invalid_argument("resident missing rigid node");
    const auto node_valid=[n](int node){if(node<0||static_cast<std::size_t>(node)>=n)throw std::invalid_argument("resident topology node index");};
    for(const auto& triangle:h.surface_triangles)for(int node:triangle.vertices)node_valid(node);
    for(const auto& edge:h.edges)for(int node:edge.vertices)node_valid(node);
    for(const auto& dep:h.dependencies) {
        if(dep.count<1||dep.count>4)throw std::invalid_argument("resident dependency size");
        for(int k=0;k<dep.count;++k)node_valid(dep.vertices[k]);
    }
}
// Serialize host orchestration of the default-stream resident solver. Exact
// workers have independent scratch; one solver's internal jobs still run in
// parallel on the device.
std::mutex resident_mutex;
}

struct ResidentSolver::Impl {
    DeviceArray<Vector3> positions,velocities,previous,predicted,pin_targets,reference,proposed,accepted;
    DeviceArray<unsigned char> deformable,solid,pinned,surface,body_translate,body_rotate;
    DeviceArray<double> mass,body_mass;
    DeviceArray<int> node_blocks,node_body,body_blocks,body_nodes,error;
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
    DeformableScene scene{};DeformableState state{};ElasticWorkspace elastic{};
    RigidScene rigid{};RigidState rigid_state{};RigidWorkspace rigid_work{};
    BroadPhaseTopology topology{};ResidentBroadPhase broadphase;BroadPhaseView broad{};
    ResidentContactPipeline contacts;ResidentLinearClipper clipper;
    ResidentElasticWorklists elastic_worklists;
    ResidentAdaptiveBoxes adaptive_boxes;
    ResidentGlobalGuess global_guess;
    std::unique_ptr<level_predictor::ResidentLevelSchedule> level_schedule;
    ResidentBroadPhase level_broadphase;
    DeviceArray<BpAabb> level_boxes;
    bool invalid=false,cloth_only=false;

    explicit Impl(const SceneDescription& h) {
        validate(h);const auto n=h.positions.size(),b=h.com.size();
        cloth_only=b==0&&std::none_of(h.solid.begin(),h.solid.end(),
            [](unsigned char solid){return solid!=0;});
        positions.upload(h.positions);velocities.upload(h.velocities);previous.resize(n);predicted.resize(n);
        pin_targets.upload(h.pin_targets);reference.upload(h.body_reference);proposed.resize(n);accepted.resize(n);
        deformable.upload(h.deformable);solid.upload(h.solid);pinned.upload(h.pinned);surface.upload(h.surface);
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
        state={v3(positions),v3(velocities),v3(previous),v3(predicted)};
        elastic={gathered_points.data(),gathered_triangles.data(),gathered_solids.data(),gathered_hinges.data(),point_outputs.data(),
            triangle_outputs.data(),solid_outputs.data(),hinge_outputs.data(),systems.data(),v3(proposed),error.data()};
        rigid={b,body_blocks.data(),node_body.data(),v3(reference),body_node_offsets.data(),body_nodes.data(),body_mass.data(),
            m9(second_moment),body_translate.data(),body_rotate.data()};
        rigid_state={v3(com),v3(com_velocity),q4(orientation),v3(omega),v3(previous_com),v3(previous_com_velocity),q4(previous_orientation),v3(previous_omega)};
        rigid_work={body_systems.data(),v3(proposed_com),v3(proposed_omega),q4(proposed_orientation),v3(accepted_com),q4(accepted_orientation),v3(accepted_omega)};
        topology.vertex_count=static_cast<int>(n);topology.block_count=h.block_count;
        topology.triangle_count=static_cast<int>(h.surface_triangles.size());topology.edge_count=static_cast<int>(h.edges.size());
        topology.dependency_count=static_cast<int>(h.dependencies.size());topology.triangles=surface_triangles.data();
        topology.edges=edges.data();topology.dependencies=dependencies.data();topology.vertex_block=node_blocks.data();topology.surface_query=surface.data();
        topology.protect_linear_gap=true;
    }
    void check_status() {
        ResidentProfileScope profile("status_wait");
        const auto code=error.download()[0];
        if(code) {invalid=true;throw std::runtime_error("resident GPU substep failed, device status "+std::to_string(code)+"; no CPU fallback and no frame output");}
    }
    void rebuild(const StepParameters& p,bool guess) {
        ResidentProfileScope profile("rebuild");
        {
        ResidentProfileScope boxes_profile("adaptive_boxes");
        if(guess)make_motion_boxes(reinterpret_cast<const double*>(positions.data()),reinterpret_cast<const double*>(predicted.data()),
            topology.vertex_count,.2,1e-8,boxes.data());
        else adaptive_boxes.build(scene,state,boxes.data(),error.data());
        build_rigid_blue_boxes(rigid,rigid_state,scene.nodes,p.box_radius,body_boxes.data(),boxes.data(),error.data(),
            p.rotation_radius,q4(box_orientation),state.positions);
        cuda_checked(cudaGetLastError());check_status();
        }
        auto graph=topology;
        if(guess){graph.dependency_count=0;graph.include_triangle_dependencies=false;}
        broad=broadphase.rebuild(graph,boxes.data(),p.contact.d_hat);
        elastic_worklists.rebuild(scene,broad.block_colors,topology.block_count,broad.color_count);
        {
        ResidentProfileScope incidence_profile("contact_incidence");
        // Certificates are invalidated/recomputed with every blue-box rebuild;
        // the original graph/coloring above remains untouched. Without CCD the
        // solver is not bound to these motion boxes, so disable this cache.
        ContactBoxPruning pruning;
        if(guess||p.collisions)pruning={boxes.data(),scene.deformable,state.positions,state.previous,p.contact};
        contacts.rebuild(broad,topology.block_count,node_blocks.data(),surface.data(),pruning);
        }
        ResidentProfileScope masks_profile("color_masks");
        collect_color_stage_masks(broad,topology.block_count,block_stages.data(),color_stages.data());
        active_color_stages.resize(static_cast<std::size_t>(broad.color_count));
        if(broad.color_count)cuda_checked(cudaMemcpy(active_color_stages.data(),color_stages.data(),
            active_color_stages.size()*sizeof(unsigned int),cudaMemcpyDeviceToHost));
    }
    void run_level_guess(const StepParameters& p) {
        ResidentProfileScope profile("level_colored_guess");
        if(!level_schedule)level_schedule=std::make_unique<level_predictor::ResidentLevelSchedule>();
        // The trial does not touch adaptive-box history, previous positions,
        // velocities or rigid state. Rigid predicted positions equal starts.
        // Keep its boxes separate from the normal solver's blue regions.
        level_boxes.resize(scene.nodes);
        make_motion_boxes(reinterpret_cast<const double*>(positions.data()),
            reinterpret_cast<const double*>(predicted.data()),topology.vertex_count,.2,1e-8,level_boxes.data());
        auto graph=topology;graph.dependency_count=0;graph.include_triangle_dependencies=false;
        // protect_linear_gap pads queries from the complete allowed boxes,
        // including large-coordinate endpoint floors, entirely on the GPU.
        const auto candidates=level_broadphase.rebuild(graph,level_boxes.data(),1e-8,BroadPhaseMode::CandidatesOnly);
        level_schedule->prepare(topology.vertex_count,candidates.contacts,candidates.contact_count,
            state.positions,state.predicted,level_boxes.data(),scene.deformable,1e-8,true,error.data());
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
        const auto sweep=[&](const BroadPhaseView& phase) {
            for(int color=0;color<phase.color_count;++color) {
                if(scene.nodes)predicted_targets<<<blocks(scene.nodes),128>>>(scene,state,elastic,phase.block_colors,color);
                const auto incidence=contacts.incidence(color);
                clipper.clip_deformable_color(scene,state,elastic,phase,
                    incidence.block_offsets,incidence.pair_indices,level_boxes.data(),phase.block_colors,color,
                    v3(accepted),error.data(),true,nullptr,incidence.color_pairs,p.ccd_exact_step_bound);
                // Vertex trial colors can outnumber update blocks because
                // rigid proxy vertices remain separate graph obstacles. The
                // existing full-node commit avoids elastic worklist limits.
                commit_deformable_endpoints(scene,state,elastic,phase.block_colors,color,v3(accepted));
            }
            check_status();
        };
        const auto trial_phase=bind(level_schedule->trial());
        sweep(trial_phase); // Exactly one discarded original-color sweep.
        const auto scheduled=level_schedule->finish_trial(state.positions,error.data());
        if(scene.nodes)cuda_checked(cudaMemcpy(positions.data(),level_schedule->starts(),
            scene.nodes*sizeof(Vector3),cudaMemcpyDeviceToDevice));
        const auto actual_phase=bind(scheduled);
        // The discarded sweep never consumes the requested actual budget or
        // establishes convergence. Default-stream launch order joins every
        // color group before the next one can inspect its positions.
        for(int iteration=0;iteration<p.colored_guess_sweeps;++iteration)sweep(actual_phase);
    }
    void deformable_step(int color,bool use_ccd,bool exact_step_bound) {
        const auto incidence=contacts.incidence(color);
        {
        ResidentProfileScope ccd_profile("ccd");
        clipper.clip_deformable_color(scene,state,elastic,broad,incidence.block_offsets,incidence.pair_indices,
            boxes.data(),broad.block_colors,color,v3(accepted),error.data(),use_ccd,nullptr,incidence.color_pairs,exact_step_bound);
        }
        ResidentProfileScope commit_profile("commit");
        commit_deformable_endpoints(scene,state,elastic,broad.block_colors,color,v3(accepted),nullptr,elastic_worklists.color(color));
    }
};

ResidentSolver::ResidentSolver(const SceneDescription& h):impl_(std::make_unique<Impl>(h)){}
ResidentSolver::~ResidentSolver()=default;

void ResidentSolver::begin_profile_frame(int frame) {
    const std::lock_guard<std::mutex> lock(resident_mutex);
    auto& profiler=resident_profiler();
    if(!profiler.enabled())return;
    profiler.begin_frame(frame);active_resident_profiler=&profiler;
}
void ResidentSolver::end_profile_frame() {
    const std::lock_guard<std::mutex> lock(resident_mutex);
    if(!active_resident_profiler)return;
    auto* profiler=active_resident_profiler;
    try {profiler->end_frame();active_resident_profiler=nullptr;}
    catch(...) {profiler->abandon_frame();active_resident_profiler=nullptr;throw;}
}

void ResidentSolver::advance_substep(const StepParameters& p,const std::vector<Vector3>& pins,
                                     const std::vector<SdfCollider>& colliders) {
    const std::lock_guard<std::mutex> lock(resident_mutex);
    auto& s=*impl_;
    if(s.invalid)throw std::runtime_error("resident GPU state invalidated by earlier failure");
    if(!(p.dt>0)||!std::isfinite(p.dt)||!(p.box_radius_min>1e-10)||!std::isfinite(p.box_radius_min)
        ||!(p.box_radius>=p.box_radius_min)||!std::isfinite(p.box_radius)
        ||!std::isfinite(p.damping)||!std::isfinite(p.rotation_radius)||p.rotation_radius<0
        ||p.rotation_radius>3.141592653589793||p.iterations<0||p.rebuild_interval<=0||p.colored_guess_sweeps<0
        ||(p.global_ccd_guess&&(p.colored_guess_sweeps||p.colored_guess_level_based))
        ||((p.colored_guess_level_based||p.colored_guess_sweeps)
            &&(!std::isfinite(p.contact.d_hat)||p.contact.d_hat<0)))
        throw std::invalid_argument("resident GPU invalid step parameters");
    if(p.contact.dt!=p.dt || p.sdf.dt!=p.dt || p.elastic.dt2!=p.dt*p.dt
        ||!(p.elastic.dt2>0)||!std::isfinite(p.elastic.dt2))
        throw std::invalid_argument("resident GPU inconsistent or unrepresentable timestep");
    if(colliders.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("resident GPU too many SDF colliders");
    length(pins.size(),s.scene.nodes,"prescribed pins");
    try {
    {
    ResidentProfileScope substep_profile("substep");
    resident_profile_counter(ResidentProfileCounter::Substeps);
    {
    ResidentProfileScope begin_profile("upload_begin");
    s.pin_targets.upload(pins);s.colliders.upload(colliders);s.error.clear();
    begin_deformable_substep(s.scene,s.state,p.dt);
    begin_rigid_substep(s.rigid,s.rigid_state,s.error.data());
    }
    // The CPU general driver disables the legacy global particle guess for
    // solid/mixed scenes. Only its cloth driver honors this default flag.
    if(p.global_ccd_guess && s.cloth_only && (s.enabled_stages&DeformableColorStage)) {
        ResidentProfileScope guess_profile("global_guess");
        s.global_guess.apply(s.scene,s.state,s.topology,s.error.data());
        s.check_status();
    }
    if(p.colored_guess_level_based&&p.colored_guess_sweeps&&(s.enabled_stages&DeformableColorStage)) {
        s.run_level_guess(p);
    } else if(p.colored_guess_sweeps && (s.enabled_stages&DeformableColorStage)) {
        ResidentProfileScope guess_profile("colored_guess");
        s.rebuild(p,true);
        for(int iteration=0;iteration<p.colored_guess_sweeps;++iteration)
            for(int color=0;color<s.broad.color_count;++color) {
                if(!(s.active_color_stages[color]&DeformableColorStage))continue;
                if(s.scene.nodes)predicted_targets<<<blocks(s.scene.nodes),128>>>(s.scene,s.state,s.elastic,s.broad.block_colors,color);
                // A CCD initial guess must remain collision-safe even when
                // subsequent particle solver updates explicitly disable CCD.
                s.deformable_step(color,true,p.ccd_exact_step_bound);
            }
        s.check_status();
    }
    // Match the CPU solver's displacement history: its entry anchor is after
    // the initial guess, while inertial displacement uses substep velocities.
    {
        ResidentProfileScope box_profile("adaptive_setup");
        s.adaptive_boxes.begin_solver(s.scene,s.state,p.dt,p.box_radius_min,p.box_radius,s.error.data());
    }
    const SdfScene sdf{s.colliders.data(),static_cast<int>(colliders.size()),p.sdf};
    for(int iteration=0;iteration<p.iterations && s.enabled_stages;++iteration) {
        resident_profile_counter(ResidentProfileCounter::Iterations);
        if(iteration%p.rebuild_interval==0)s.rebuild(p,false);
        for(int color=0;color<s.broad.color_count;++color) {
            resident_profile_counter(ResidentProfileCounter::ColorVisits);
            const unsigned int stages=s.active_color_stages[color];
            if(stages&DeformableColorStage) {
                ResidentProfileScope deformable_profile("deformable");
                const auto worklist=s.elastic_worklists.color(color);
                assemble_elastic_color(s.scene,s.state,s.elastic,s.broad.block_colors,color,p.elastic,p.solid,nullptr,worklist);
                s.contacts.assemble_color(s.scene,s.state,s.elastic,s.rigid,s.rigid_state,s.rigid_work,sdf,p.contact,color,ContactAssemblyPhase::Deformable);
                {
                ResidentProfileScope proposal_profile("local_solve");
                propose_deformable_color(s.scene,s.state,s.elastic,s.broad.block_colors,color,p.damping,nullptr,worklist);
                }
                s.deformable_step(color,p.collisions,p.ccd_exact_step_bound);
            }
            if(stages&RigidTranslationColorStage) {
                ResidentProfileScope rigid_profile("rigid_translation");
                {
                ResidentProfileScope inertia_profile("inertia");
                assemble_rigid_inertia_color(s.rigid,s.rigid_state,s.rigid_work,s.broad.block_colors,color,false,p.dt,p.elastic.gravity[1],s.error.data());
                }
                s.contacts.assemble_color(s.scene,s.state,s.elastic,s.rigid,s.rigid_state,s.rigid_work,sdf,p.contact,color,ContactAssemblyPhase::RigidTranslation);
                {
                ResidentProfileScope proposal_profile("local_solve");
                propose_rigid_color(s.rigid,s.rigid_state,s.rigid_work,s.broad.block_colors,color,false,s.body_boxes.data(),p.dt,p.damping,s.error.data(),p.rotation_radius,q4(s.box_orientation));
                }
                const auto incidence=s.contacts.incidence();
                {
                ResidentProfileScope ccd_profile("ccd");
                s.clipper.clip_rigid_translation_color(s.rigid,s.rigid_state,s.rigid_work,s.state,s.broad,
                    incidence.block_offsets,incidence.pair_indices,s.boxes.data(),s.broad.block_colors,color,v3(s.accepted),s.error.data(),true);
                }
                ResidentProfileScope commit_profile("commit");
                commit_rigid_translation_color(s.rigid,s.rigid_state,s.rigid_work,s.state,s.broad.block_colors,color,v3(s.accepted),s.error.data());
            }
            if(stages&RigidOrientationColorStage) {
                ResidentProfileScope rigid_profile("rigid_orientation");
                {
                ResidentProfileScope inertia_profile("inertia");
                assemble_rigid_inertia_color(s.rigid,s.rigid_state,s.rigid_work,s.broad.block_colors,color,true,p.dt,p.elastic.gravity[1],s.error.data());
                }
                s.contacts.assemble_color(s.scene,s.state,s.elastic,s.rigid,s.rigid_state,s.rigid_work,sdf,p.contact,color,ContactAssemblyPhase::RigidOrientation);
                {
                ResidentProfileScope proposal_profile("local_solve");
                propose_rigid_color(s.rigid,s.rigid_state,s.rigid_work,s.broad.block_colors,color,true,s.body_boxes.data(),p.dt,p.damping,s.error.data(),p.rotation_radius,q4(s.box_orientation));
                }
                // Match general-v2: rigid CCD is mandatory, independently of the
                // particle use_ccd switch.
                {
                ResidentProfileScope ccd_profile("ccd");
                clip_rigid_rotation_color(s.rigid,s.rigid_state,s.rigid_work,s.state,s.broad,s.broad.block_colors,color,p.dt,.9,true,v3(s.accepted),s.error.data());
                }
                ResidentProfileScope commit_profile("commit");
                commit_rigid_rotation_color(s.rigid,s.rigid_state,s.rigid_work,s.state,s.broad.block_colors,color,v3(s.accepted),s.error.data());
            }
        }
        s.check_status();
    }
    {
    ResidentProfileScope finish_profile("finish");
    s.adaptive_boxes.finish_solver(s.scene,s.state,s.error.data());
    finish_deformable_substep(s.scene,s.state,p.dt);
    finish_rigid_substep(s.rigid,s.rigid_state,s.state,p.dt,s.error.data());
    cuda_checked(cudaDeviceSynchronize());s.check_status();
    }
    }
    if(active_resident_profiler)active_resident_profiler->flush();
    } catch(...) {
        // A failure can follow partial color commits. Never expose that state
        // as a completed substep or permit an accidental retry from it.
        s.invalid=true;
        if(active_resident_profiler) {
            active_resident_profiler->abandon_frame();active_resident_profiler=nullptr;
        }
        throw;
    }
}

StateSnapshot ResidentSolver::snapshot()const {
    const std::lock_guard<std::mutex> lock(resident_mutex);
    const auto& s=*impl_;
    if(s.invalid)throw std::runtime_error("resident GPU cannot export invalid state");
    ResidentProfileScope profile("snapshot");
    StateSnapshot out;
    out.positions=s.positions.download();out.velocities=s.velocities.download();out.com=s.com.download();
    out.com_velocity=s.com_velocity.download();out.orientation=s.orientation.download();out.omega=s.omega.download();
    return out;
}
} // namespace ipc_gpu::resident
