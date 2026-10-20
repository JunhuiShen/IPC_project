#include "resident_bridge.h"
#include "general_simd_materials.h"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace ipc_gpu::resident {
namespace {
Vector3 vector3(const Vec3& v) { return {v[0],v[1],v[2]}; }
Quaternion quaternion(const Vec4& q) { return {q[0],q[1],q[2],q[3]}; }
Matrix3 matrix3(const Mat33& m) {
    Matrix3 out{};
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) out[3*r+c]=m(r,c);
    return out;
}
SdfPose pose(const SDFMaterialPose& p) {
    SdfPose out;
    for(int r=0;r<3;++r) {
        out.translation[r]=p.translation[r];
        for(int c=0;c<3;++c) out.rotation[3*r+c]=p.rotation(r,c);
    }
    return out;
}
void copy_vectors(const std::vector<Vector3>& source,std::vector<Vec3>& target) {
    target.resize(source.size());
    for(std::size_t i=0;i<source.size();++i)
        for(int a=0;a<3;++a) target[i][a]=source[i][a];
}
}

std::vector<Vector3> describe_pin_targets(std::size_t nodes,const std::vector<Pin>& pins) {
    std::vector<Vector3> out(nodes);
    for(const auto& pin:pins) out.at(pin.vertex_index)=vector3(pin.target_position);
    return out;
}

SceneDescription describe_scene(const RefMesh& mesh,const DeformedState& state,
    const std::vector<Pin>& pins,const SimParams& params) {
    SceneDescription out;
    const auto n=state.deformed_positions.size();
    if(state.velocities.size()!=n || mesh.mass.size()!=n)
        throw std::invalid_argument("resident scene: inconsistent node arrays");
    out.positions.reserve(n);out.velocities.reserve(n);
    for(const auto& p:state.deformed_positions)out.positions.push_back(vector3(p));
    for(const auto& v:state.velocities)out.velocities.push_back(vector3(v));
    out.mass=mesh.mass;out.pin_targets=describe_pin_targets(n,pins);
    out.deformable.assign(n,1);out.solid.assign(n,0);out.pinned.assign(n,0);
    out.surface.assign(n,0);out.node_body.assign(n,-1);out.node_blocks.assign(n,-1);
    out.body_reference.resize(n);
    for(const auto& pin:pins)out.pinned.at(pin.vertex_index)=1;
    for(int node:mesh.tet_nodes)out.solid.at(node)=1;
    for(std::size_t b=0;b<mesh.rb_nodes.size();++b) {
        if(mesh.ref_positions.at(b).size()!=mesh.rb_nodes[b].size())
            throw std::invalid_argument("resident scene: rigid reference size");
        for(std::size_t j=0;j<mesh.rb_nodes[b].size();++j) {
            const int node=mesh.rb_nodes[b][j];
            if(out.node_body.at(node)>=0)throw std::invalid_argument("resident scene: multiple rigid owners");
            out.node_body[node]=static_cast<int>(b);out.deformable[node]=0;
            out.body_reference[node]=vector3(mesh.ref_positions[b][j]);
        }
    }
    for(std::size_t node=0;node<n;++node)
        if(out.deformable[node])out.node_blocks[node]=out.block_count++;
    out.body_node_offsets.push_back(0);
    for(std::size_t b=0;b<mesh.rb_nodes.size();++b) {
        out.body_blocks.push_back(out.block_count++);
        for(int node:mesh.rb_nodes[b]) {
            out.node_blocks[node]=out.body_blocks.back();out.body_nodes.push_back(node);
        }
        out.body_node_offsets.push_back(out.body_nodes.size());
        out.body_mass.push_back(mesh.total_mass.at(b));
        out.body_second_moment.push_back(matrix3(mesh.I_hat.at(b)));
        const auto mode=mesh.rb_update_modes.at(b);
        out.body_translate.push_back(updates_rigid_translation(mode));
        out.body_rotate.push_back(updates_rigid_orientation(mode));
        out.com.push_back(vector3(state.x_coms.at(b)));
        out.com_velocity.push_back(vector3(state.v_coms.at(b)));
        out.orientation.push_back(quaternion(state.orientations.at(b)));
        out.omega.push_back(vector3(state.omega.at(b)));
    }
    std::vector<IncidentTriangles> incident(n);
    std::set<std::pair<int,int>> edges;
    if(mesh.tris.size()%3 || mesh.tets.size()%4)
        throw std::invalid_argument("resident scene: malformed connectivity");
    for(std::size_t t=0;t<mesh.tris.size()/3;++t) {
        BpTriangle tri{};
        for(int j=0;j<3;++j) {
            const int node=mesh.tris[3*t+j];
            tri.vertices[j]=node;out.surface.at(node)=1;
            incident.at(node).push_back({static_cast<int>(t),j});
            int a=node,b=mesh.tris[3*t+(j+1)%3];
            if(a>b)std::swap(a,b);edges.insert({a,b});
        }
        out.surface_triangles.push_back(tri);
    }
    for(auto [a,b]:edges)out.edges.push_back({{a,b}});
    for(const auto& h:mesh.hinges)out.dependencies.push_back({{h.v[0],h.v[1],h.v[2],h.v[3]},4});
    for(std::size_t t=0;t<mesh.tets.size()/4;++t)
        out.dependencies.push_back({{mesh.tets[4*t],mesh.tets[4*t+1],mesh.tets[4*t+2],mesh.tets[4*t+3]},4});

    // This cache only contains immutable rest/material data and topology.
    // Dynamic positions are gathered later, entirely on the device.
    solver_detail::GeneralSimdMaterials materials;
    materials.prepare(mesh,incident,n,params.kB>0);
    out.triangle_offsets=materials.triangles.node_offsets;
    out.solid_offsets=materials.tets.node_offsets;
    out.hinge_offsets=materials.hinges.node_offsets;
    for(std::size_t node=0;node<n;++node) {
        const auto& t=materials.triangles;
        for(auto i=t.node_offsets[node];i<t.node_offsets[node+1];++i) {
            TriangleIncident record{};record.owner=static_cast<int>(node);
            std::copy(t.nodes[i].begin(),t.nodes[i].end(),record.nodes);
            for(int r=0;r<2;++r) {
                record.material.shape[r]=t.shape_gradients[i][r];
                for(int c=0;c<2;++c)record.material.dm_inverse[2*r+c]=t.dm_inverse[i](r,c);
            }
            record.material.area=t.measures[i];out.triangles.push_back(record);
        }
        const auto& s=materials.tets;
        for(auto i=s.node_offsets[node];i<s.node_offsets[node+1];++i) {
            SolidIncident record{};record.owner=static_cast<int>(node);
            std::copy(s.nodes[i].begin(),s.nodes[i].end(),record.nodes);
            for(int r=0;r<3;++r) {
                record.material.shape[r]=s.shape_gradients[i][r];
                for(int c=0;c<3;++c)record.material.dm_inverse[3*r+c]=s.dm_inverse[i](r,c);
            }
            record.material.measure=s.measures[i];out.solids.push_back(record);
        }
        const auto& h=materials.hinges;
        for(auto i=h.node_offsets[node];i<h.node_offsets[node+1];++i) {
            HingeIncident record{};record.owner=static_cast<int>(node);
            std::copy(h.nodes[i].begin(),h.nodes[i].end(),record.nodes);
            record.material.active_node=h.active_nodes[i];
            record.material.coefficient=h.coefficients[i];
            record.material.rest_angle=h.rest_angles[i];out.hinges.push_back(record);
        }
    }
    return out;
}

StepParameters describe_step(const SimParams& p) {
    p.validate_ccd_exact_step_bound_parameters();
    p.validate_colored_ccd_guess_parameters();
    // Never silently reroute an unported solver/initial-guess mode to CPU.
    if(!p.fixed_iters || p.use_ogc || p.use_ogc_solver || p.use_cloth_grid)
        throw std::invalid_argument("resident GPU currently requires fixed iterations and conflict-graph coloring, with OGC/grid disabled");
    if(!(p.use_colored_ccd_guess || p.use_colored_ccd_guess_level_based) && (p.use_verlet_guess || p.use_translation_guess))
        throw std::invalid_argument("resident GPU: Verlet and translation initial guesses are not ported");
    if(!p.exact_computation_fallback)
        throw std::invalid_argument("resident GPU requires the robust exact fallback enabled");
    if(p.use_ticcd)
        throw std::invalid_argument("resident GPU uses device interval/exact linear CCD; the Tight-Inclusion backend is not ported (use --use_ticcd false)");
    StepParameters out;
    out.dt=p.dt();out.elastic.dt2=p.dt2();out.elastic.kpin=p.kpin;
    for(int a=0;a<3;++a)out.elastic.gravity[a]=p.gravity[a];
    out.elastic.mu=p.mu;out.elastic.lambda=p.lambda;out.elastic.bending_stiffness=p.kB;
    out.solid={p.solid_mu,p.solid_lambda};
    out.contact={p.d_hat,p.k_barrier,p.dt(),p.friction_coefficient,p.friction_velocity_epsilon,1e-12};
    out.sdf.stiffness=p.k_sdf;out.sdf.epsilon=p.eps_sdf;out.sdf.dt=p.dt();
    out.sdf.friction=p.friction_coefficient;out.sdf.eps_v=p.friction_velocity_epsilon;
    out.sdf.include_curvature=0;
    out.damping=p.damping;out.box_radius_min=p.node_box_min;out.box_radius=p.node_box_max;
    out.rotation_radius=p.theta_box_max;
    out.iterations=p.max_global_iters;out.rebuild_interval=p.node_box_update_count;
    const bool colored=p.use_colored_ccd_guess||p.use_colored_ccd_guess_level_based;
    out.colored_guess_sweeps=colored?p.colored_ccd_guess_iters:0;
    out.colored_guess_fraction=p.colored_ccd_guess_fraction;
    out.colored_guess_level_based=p.use_colored_ccd_guess_level_based;
    out.ccd_exact_step_bound=p.ccd_exact_step_bound;
    out.global_ccd_guess=p.use_ccd_guess&&!colored;
    out.collisions=p.use_ccd;
    return out;
}

std::vector<SdfCollider> describe_colliders(const SimParams& p) {
    std::vector<SdfCollider> out;
    for(const auto& s:p.sdf_planes) {
        SdfCollider x{};x.kind=SdfKind::Plane;
        for(int a=0;a<3;++a){x.point[a]=s.point[a];x.axis[a]=s.normal[a];}
        x.previous_pose=pose(s.material_motion.previous);x.current_pose=pose(s.material_motion.current);
        out.push_back(x);
    }
    for(const auto& s:p.sdf_cylinders) {
        SdfCollider x{};x.kind=SdfKind::Cylinder;x.radius=s.radius;
        for(int a=0;a<3;++a){x.point[a]=s.point[a];x.axis[a]=s.axis[a];}
        x.previous_pose=pose(s.material_motion.previous);x.current_pose=pose(s.material_motion.current);
        out.push_back(x);
    }
    for(const auto& s:p.sdf_spheres) {
        SdfCollider x{};x.kind=SdfKind::Sphere;x.radius=s.radius;
        for(int a=0;a<3;++a)x.point[a]=s.center[a];
        x.previous_pose=pose(s.material_motion.previous);x.current_pose=pose(s.material_motion.current);
        out.push_back(x);
    }
    return out;
}

void copy_snapshot(const StateSnapshot& s,DeformedState& state) {
    copy_vectors(s.positions,state.deformed_positions);copy_vectors(s.velocities,state.velocities);
    copy_vectors(s.com,state.x_coms);copy_vectors(s.com_velocity,state.v_coms);
    copy_vectors(s.omega,state.omega);state.orientations.resize(s.orientation.size());
    for(std::size_t b=0;b<s.orientation.size();++b)
        for(int a=0;a<4;++a)state.orientations[b][a]=s.orientation[b][a];
}
} // namespace ipc_gpu::resident
