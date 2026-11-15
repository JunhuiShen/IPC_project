"""Rigid equations, contact graph and COM/rotation GS stages.

Public quaternions use Newton/Warp's xyzw convention. Native derivatives use
normalized scalar-first quaternions, with float32 state boundaries and the
maintained binary64 signed-curvature/LDLT routines. Ordinary graph kernels
remain in separate Warp modules from the cold block-size-one exact phase.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo

from . import contact, sdf
from .contact import FullPacket, decode_candidate
from .ccd import round_inside
from .warp_build import configure_module as _configure


HEADERS=("rigid.cuh",)
BODY_HEADERS=("rigid.cuh", "original_rigid.cuh")

_HELPERS=r"""
#ifndef IPC_WARP_RIGID_F32_BOUNDARY
#define IPC_WARP_RIGID_F32_BOUNDARY
namespace ipc_warp_rigid {
CUDA_CALLABLE inline bool unit(float x,float y,float z,float w,double* q) {
    q[0]=w;q[1]=x;q[2]=y;q[3]=z;
    const double norm=sqrt(((q[0]*q[0]+q[1]*q[1])+q[2]*q[2])+q[3]*q[3]);
    if(!(norm>0)||!isfinite(norm))return false;
    for(int k=0;k<4;++k)q[k]/=norm;
    return true;
}
CUDA_CALLABLE inline void vec(const wp::vec_t<3,float>& input,double* output) {
    for(int k=0;k<3;++k)output[k]=input[k];
}
}
#endif
"""


def configure_module(module=None):
    return _configure(__name__ if module is None else module,HEADERS,extra_preamble=_HELPERS)


_UNIT=r"""
    double q[4];
    const bool unit=ipc_warp_rigid::unit(previous_q[0],previous_q[1],previous_q[2],previous_q[3],q);
"""
_OUTPUT=r"""
    for(int a=0;a<3;++a){gradient[a]=static_cast<float>(value.gradient[a]);
        for(int b=0;b<3;++b)hessian.data[a][b]=static_cast<float>(value.hessian[3*a+b]);}
"""


@wp.func_native(_UNIT+r"""
    wp::vec_t<4,float> out;
    if(!unit){for(int k=0;k<4;++k)out[k]=static_cast<float>(NAN);return out;}
    double angular[3],target[4];ipc_warp_rigid::vec(omega,angular);
    ipc_gpu::solver::rigid_orientation(q,angular,dt,target);
    out[0]=static_cast<float>(target[1]);out[1]=static_cast<float>(target[2]);out[2]=static_cast<float>(target[3]);out[3]=static_cast<float>(target[0]);
    return out;
""")
def _orientation(previous_q: wp.quat,omega: wp.vec3,dt: float)->wp.vec4: ...


@wp.func
def orientation(previous_q: wp.quat,omega: wp.vec3,dt: float):
    value=_orientation(previous_q,omega,dt)
    return wp.normalize(wp.quat(value[0],value[1],value[2],value[3]))


@wp.func_native(r"""
    double q[4],x[3],center[3],result[3];
    wp::vec_t<3,float> output;
    if(!ipc_warp_rigid::unit(rotation[0],rotation[1],rotation[2],rotation[3],q)) {
        for(int k=0;k<3;++k)output[k]=static_cast<float>(NAN);return output;
    }
    ipc_warp_rigid::vec(reference,x);ipc_warp_rigid::vec(com,center);
    ipc_gpu::solver::rigid_world_position(x,center,q,result);
    for(int k=0;k<3;++k)output[k]=static_cast<float>(result[k]);return output;
""")
def world_point(reference: wp.vec3,com: wp.vec3,rotation: wp.quat)->wp.vec3: ...


@wp.func
def world_position(reference: wp.vec3,pose: wp.transform):
    return world_point(reference,wp.transform_get_translation(pose),wp.transform_get_rotation(pose))


@wp.func_native(_UNIT+r"""
    double angular[3],x[3],j[9],c[3][9];
    ipc_warp_rigid::vec(omega,angular);ipc_warp_rigid::vec(reference,x);
    const auto kin=ipc_gpu::solver::rigid_kinematics(q,angular,dt);
    ipc_gpu::solver::rigid_reference_derivatives(x,kin,j,c);
    for(int r=0;r<3;++r)for(int s=0;s<3;++s){
        jacobian.data[r][s]=unit?static_cast<float>(j[3*r+s]):static_cast<float>(NAN);
        cx.data[r][s]=unit?static_cast<float>(c[0][3*r+s]):static_cast<float>(NAN);
        cy.data[r][s]=unit?static_cast<float>(c[1][3*r+s]):static_cast<float>(NAN);
        cz.data[r][s]=unit?static_cast<float>(c[2][3*r+s]):static_cast<float>(NAN);
    }
""")
def _reference(reference: wp.vec3,previous_q: wp.quat,omega: wp.vec3,dt: float,
               jacobian: wp.ref[wp.mat33],cx: wp.ref[wp.mat33],cy: wp.ref[wp.mat33],cz: wp.ref[wp.mat33]): ...


@wp.func
def reference_derivatives(reference: wp.vec3,previous_q: wp.quat,omega: wp.vec3,dt: float):
    j=wp.mat33();cx=wp.mat33();cy=wp.mat33();cz=wp.mat33()
    _reference(reference,previous_q,omega,dt,j,cx,cy,cz)
    return j,cx,cy,cz


@wp.func
def translation_inertia(com: wp.vec3,previous_com: wp.vec3,previous_velocity: wp.vec3,
                        mass: float,gravity_y: float,dt: float):
    gradient=mass*(com-previous_com-dt*previous_velocity)
    gradient[1]-=dt*dt*mass*gravity_y
    return gradient,mass*wp.identity(n=3,dtype=float)


@wp.func_native(_UNIT+r"""
    double old_angular[3],angular[3],moment[9];
    ipc_warp_rigid::vec(previous_omega,old_angular);ipc_warp_rigid::vec(omega,angular);
    for(int r=0;r<3;++r)for(int s=0;s<3;++s)moment[3*r+s]=second_moment.data[r][s];
    const auto prepared=ipc_gpu::solver::build_rigid_inertia(q,old_angular,angular,dt,moment);
    auto value=ipc_gpu::evaluate_rigid_inertia(prepared);
    if(!unit){for(int k=0;k<3;++k)value.gradient[k]=NAN;}
"""+_OUTPUT)
def _rotation(previous_q: wp.quat,previous_omega: wp.vec3,omega: wp.vec3,dt: float,
              second_moment: wp.mat33,gradient: wp.ref[wp.vec3],hessian: wp.ref[wp.mat33]): ...


@wp.func
def rotation_inertia(previous_q: wp.quat,previous_omega: wp.vec3,omega: wp.vec3,dt: float,second_moment: wp.mat33):
    g=wp.vec3();h=wp.mat33()
    _rotation(previous_q,previous_omega,omega,dt,second_moment,g,h)
    return g,h


@wp.func_native(_UNIT+r"""
    using namespace ipc_gpu;
    if(packet[176]!=0){for(int k=0;k<3;++k)gradient[k]=static_cast<float>(NAN);return;}
    if(packet[177]==0){for(int k=0;k<3;++k)gradient[k]=0;for(int r=0;r<3;++r)for(int s=0;s<3;++s)hessian.data[r][s]=0;return;}
    bool selected[4]{};RigidPreparedContact p{};
    p.segment_segment=kind!=0;
    p.flags=rotational?RigidOrientationGradient|RigidOrientationHessian|RigidUpperCrossBlocks
        :RigidTranslationGradient|RigidTranslationHessian;
    p.flags|=RigidFriction;p.update_mask=rotational?2u:1u;
    double angular[3];ipc_warp_rigid::vec(omega,angular);
    const auto kin=ipc_gpu::solver::rigid_kinematics(q,angular,dt);
    bool valid=unit,any=false;
    for(int i=0;i<4;++i){
        const int vertex=nodes[i];
        if(vertex<0||vertex>=owners.shape[0]||vertex>=references.shape[0]){valid=false;continue;}
        selected[i]=owners.data[vertex]==body;
        if(!selected[i])continue;
        any=true;
        p.friction_weights[i]=packet[1+i];
        for(int k=0;k<3;++k)p.gradients[i][k]=packet[8+3*i+k];
        if(rotational){double x[3];ipc_warp_rigid::vec(references.data[vertex],x);
            ipc_gpu::solver::rigid_reference_derivatives(x,kin,p.jacobians[i],p.curvature[i]);}
    }
    for(int k=0;k<3;++k)p.friction_gradient[k]=packet[164+k];
    for(int k=0;k<9;++k)p.friction_hessian[k]=packet[167+k];
    if(!rotational&&kind==0&&any){for(int k=0;k<9;++k)p.translation_hessian[k]=packet[20+k];}
    else for(int i=0;i<4;++i)if(selected[i])for(int j=i;j<4;++j)if(selected[j])
        for(int r=0;r<3;++r)for(int s=0;s<3;++s){
            p.hessians[i][j][3*r+s]=packet[20+36*i+9*j+3*r+s];
            if(!rotational&&i!=j)p.hessians[j][i][3*s+r]=p.hessians[i][j][3*r+s];
        }
    const auto result=evaluate_rigid(p);
    const double* g=rotational?result.barrier.orientation_gradient:result.barrier.translation_gradient;
    const double* h=rotational?result.barrier.orientation_orientation_hessian:result.barrier.translation_translation_hessian;
    const double* fg=rotational?result.friction.orientation_gradient:result.friction.translation_gradient;
    const double* fh=rotational?result.friction.orientation_orientation_hessian:result.friction.translation_translation_hessian;
    for(int r=0;r<3;++r){gradient[r]=valid?static_cast<float>(normal_scale*g[r]+fg[r]):static_cast<float>(NAN);
        for(int s=0;s<3;++s)hessian.data[r][s]=static_cast<float>(normal_scale*h[3*r+s]+fh[3*r+s]);}
""")
def _contact(packet: FullPacket,nodes: wp.vec4i,owners: wp.array[int],references: wp.array[wp.vec3],
              body: int,previous_q: wp.quat,omega: wp.vec3,dt: float,kind: int,rotational: int,normal_scale: float,
              gradient: wp.ref[wp.vec3],hessian: wp.ref[wp.mat33]): ...


@wp.func
def contact_terms(packet: FullPacket,nodes: wp.vec4i,owners: wp.array[int],references: wp.array[wp.vec3],
                  body: int,previous_q: wp.quat,omega: wp.vec3,dt: float,kind: int,rotational: int,normal_scale: float):
    g=wp.vec3();h=wp.mat33()
    _contact(packet,nodes,owners,references,body,previous_q,omega,dt,kind,rotational,normal_scale,g,h)
    return g,h


@wp.func_native(_UNIT+r"""
    ipc_gpu::solver::RigidPinConstraint pin{};double center[3],angular[3];
    ipc_warp_rigid::vec(com,center);ipc_warp_rigid::vec(omega,angular);
    for(int k=0;k<3;++k){pin.local_position[k]=reference[k];pin.initial_target[k]=target[k];}
    const auto kin=ipc_gpu::solver::rigid_kinematics(q,angular,dt);
    ipc_gpu::RigidDerivatives result{};
    valid=unit&&ipc_gpu::solver::evaluate_rigid_pin(pin,center,kin,dt,kpin,0.0,rotational!=0,result);
    const double* g=rotational?result.orientation_gradient:result.translation_gradient;
    const double* h=rotational?result.orientation_orientation_hessian:result.translation_translation_hessian;
    for(int r=0;r<3;++r){gradient[r]=static_cast<float>(g[r]);for(int s=0;s<3;++s)hessian.data[r][s]=static_cast<float>(h[3*r+s]);}
""")
def _pin(reference: wp.vec3,target: wp.vec3,com: wp.vec3,previous_q: wp.quat,omega: wp.vec3,
          dt: float,kpin: float,rotational: int,gradient: wp.ref[wp.vec3],hessian: wp.ref[wp.mat33],valid: wp.ref[int]): ...


@wp.func
def point_pin_derivatives(reference: wp.vec3,target: wp.vec3,com: wp.vec3,previous_q: wp.quat,
                          omega: wp.vec3,dt: float,kpin: float,rotational: int):
    g=wp.vec3();h=wp.mat33();valid=int(0)
    _pin(reference,target,com,previous_q,omega,dt,kpin,rotational,g,h,valid)
    return g,h,valid


@wp.func_native(r"""
    wp::vec_t<3,float> target;
    bool okay=isfinite(rate)&&isfinite(time)&&isfinite(ramp)&&ramp>=0&&isfinite(start)&&isfinite(duration);
    double target64[3],offset[3],axis64[3];
    for(int k=0;k<3;++k){target64[k]=initial[k];offset[k]=static_cast<double>(initial[k])-center[k];axis64[k]=axis[k];
        okay=okay&&isfinite(initial[k])&&isfinite(center[k])&&isfinite(translation[k]);}
    if(rate!=0&&!(ramp>0&&time<=0)){
        double largest=fmax(fabs(axis64[0]),fmax(fabs(axis64[1]),fabs(axis64[2])));
        okay=okay&&largest>0&&isfinite(largest);
        if(okay){
            for(int k=0;k<3;++k)axis64[k]/=largest;
            const double norm=sqrt((axis64[0]*axis64[0]+axis64[1]*axis64[1])+axis64[2]*axis64[2]);
            for(int k=0;k<3;++k)axis64[k]/=norm;
            double rotation_time=time;
            if(ramp>0){if(time>=ramp)rotation_time=static_cast<double>(time)-0.5*static_cast<double>(ramp);
                else {const double u=static_cast<double>(time)/ramp;rotation_time=static_cast<double>(ramp)*u*u*u*u*(2.5+u*(-3.0+u));}}
            const double theta=static_cast<double>(rate)*rotation_time,c=cos(theta),s=sin(theta);
            okay=okay&&isfinite(theta);
            if(axis[0]==0&&axis[1]==0&&axis[2]>0){
                target64[0]=center[0]+(c*offset[0]-s*offset[1]);
                target64[1]=center[1]+(s*offset[0]+c*offset[1]);target64[2]=center[2]+offset[2];
            }else{
                const double dot=(axis64[0]*offset[0]+axis64[1]*offset[1])+axis64[2]*offset[2];
                const double cross[3]{axis64[1]*offset[2]-axis64[2]*offset[1],axis64[2]*offset[0]-axis64[0]*offset[2],axis64[0]*offset[1]-axis64[1]*offset[0]};
                for(int k=0;k<3;++k)target64[k]=center[k]+c*offset[k]+s*cross[k]+((1-c)*dot)*axis64[k];
            }
        }
    }
    const bool moves=translation[0]!=0||translation[1]!=0||translation[2]!=0;
    okay=okay&&(!moves||duration>0);
    if(moves&&time>start&&okay){
        const double elapsed=static_cast<double>(time)-start,u=elapsed>=duration?1.0:elapsed/duration;
        const double weight=u*u*u*(10.0+u*(-15.0+6.0*u));
        for(int k=0;k<3;++k)target64[k]+=weight*translation[k];
    }
    valid=okay?1:0;
    for(int k=0;k<3;++k){target[k]=okay?static_cast<float>(target64[k]):static_cast<float>(NAN);if(!isfinite(target[k]))valid=0;}
    return target;
""")
def _target(initial: wp.vec3,center: wp.vec3,axis: wp.vec3,rate: float,time: float,ramp: float,
            translation: wp.vec3,start: float,duration: float,valid: wp.ref[int])->wp.vec3: ...


@wp.func
def pin_target(initial: wp.vec3,center: wp.vec3,axis: wp.vec3,rate: float,time: float,
               ramp: float,translation: wp.vec3,start: float,duration: float):
    valid=int(0)
    target=_target(initial,center,axis,rate,time,ramp,translation,start,duration,valid)
    return target,valid


@wp.func_native(r"""
    double H[9],g[3],x[3];
    for(int r=0;r<3;++r){g[r]=force[r];for(int s=0;s<3;++s)H[3*r+s]=hessian.data[r][s];}
    valid=ipc_gpu::solver::rigid_ldlt3(H,g,x)?1:0;
    for(int k=0;k<3;++k)delta[k]=static_cast<float>(x[k]);
""")
def _solve(hessian: wp.mat33,force: wp.vec3,delta: wp.ref[wp.vec3],valid: wp.ref[int]): ...


@wp.func
def solve3(hessian: wp.mat33,gradient: wp.vec3):
    delta=wp.vec3();valid=int(0)
    _solve(hessian,gradient,delta,valid)
    return delta,valid


@wp.kernel(module="unique", enable_backward=False)
def begin(poses: wp.array[wp.transform], velocity: wp.array[wp.spatial_vector],
           gravity: wp.array[wp.vec3], body_world: wp.array[int], dt: float,
           previous_pose: wp.array[wp.transform], previous_velocity: wp.array[wp.spatial_vector],
           omega: wp.array[wp.vec3]):
    b = wp.tid()
    previous_pose[b] = poses[b]
    old = velocity[b]
    # Linear components encode the inertial velocity used by the existing
    # translation system: v_hat = v_old + dt*g. Angular components remain the
    # frozen physical step-start omega used by rotational inertia.
    linear = wp.vec3(old[0], old[1], old[2]) + dt * gravity[body_world[b]]
    previous_velocity[b] = wp.spatial_vector(linear[0], linear[1], linear[2], old[3], old[4], old[5])
    # The CPU rotational unknown is the step's angular increment / dt, and
    # starts at zero. Previous physical angular velocity enters inertia only.
    omega[b] = wp.vec3(0.0)


@wp.func
def primitive_owners(nodes: wp.vec4i, kind: int, owners: wp.array[int]):
    first = owners[nodes[0]]
    second = owners[nodes[1]]
    valid = True
    if kind == 0:
        valid = second == owners[nodes[2]] and second == owners[nodes[3]]
    else:
        second = owners[nodes[2]]
        valid = first == owners[nodes[1]] and second == owners[nodes[3]]
    return first, second, valid


@wp.kernel(module="unique", enable_backward=False)
def count_incidence(ids: wp.array[int], count: wp.array[int], triangles: wp.array2d[int],
                      edges: wp.array2d[int], info: TriMeshCollisionInfo, owners: wp.array[int],
                      counts: wp.array[int], conflicts: wp.array2d[wp.uint32], error: wp.array[int]):
    index = wp.tid()
    if index < count[0] and error[0] == 0:
        nodes, kind = decode_candidate(ids[index], triangles, edges, info)
        first, second, valid = primitive_owners(nodes, kind, owners)
        if not valid:
            wp.atomic_max(error, 0, 602)
        else:
            if first >= 0:
                wp.atomic_add(counts, first, 1)
            if second >= 0 and second != first:
                wp.atomic_add(counts, second, 1)
            if first >= 0 and second >= 0 and first != second:
                wp.atomic_or(conflicts, first, second // 32, wp.uint32(1) << wp.uint32(second % 32))
                wp.atomic_or(conflicts, second, first // 32, wp.uint32(1) << wp.uint32(first % 32))


@wp.kernel(module="unique", enable_backward=False)
def prefix_incidence(counts: wp.array[int], offsets: wp.array[int]):
    total = int(0)
    for body in range(counts.shape[0]):
        offsets[body] = total
        total += counts[body]


@wp.kernel(module="unique", enable_backward=False)
def fill_incidence(ids: wp.array[int], count: wp.array[int], triangles: wp.array2d[int],
                     edges: wp.array2d[int], info: TriMeshCollisionInfo, owners: wp.array[int],
                     offsets: wp.array[int], cursors: wp.array[int], incidence: wp.array[int], error: wp.array[int]):
    index = wp.tid()
    if index < count[0] and error[0] == 0:
        nodes, kind = decode_candidate(ids[index], triangles, edges, info)
        first, second, valid = primitive_owners(nodes, kind, owners)
        if valid:
            if first >= 0:
                slot = offsets[first] + wp.atomic_add(cursors, first, 1)
                incidence[slot] = index
            if second >= 0 and second != first:
                slot = offsets[second] + wp.atomic_add(cursors, second, 1)
                incidence[slot] = index


@wp.kernel(module="unique", enable_backward=False)
def color_bodies(conflicts: wp.array2d[wp.uint32], colors: wp.array[int], marks: wp.array[int],
                    count: wp.array[int], error: wp.array[int]):
    if error[0] == 0:
        maximum = int(0)
        for b in range(colors.shape[0]):
            for a in range(b):
                adjacent = (conflicts[b, a // 32] & (wp.uint32(1) << wp.uint32(a % 32))) != wp.uint32(0)
                if adjacent:
                    marks[colors[a]] = b + 1
            color = int(0)
            while marks[color] == b + 1:
                color += 1
            colors[b] = color
            maximum = wp.max(maximum, color + 1)
        count[0] = maximum


@wp.kernel(module="unique", enable_backward=False)
def start_colors(color: wp.array[int], count: wp.array[int], condition: wp.array[int], error: wp.array[int]):
    color[0] = 0
    condition[0] = int(count[0] > 0 and error[0] == 0)


@wp.kernel(module="unique", enable_backward=False)
def next_color(color: wp.array[int], count: wp.array[int], condition: wp.array[int], error: wp.array[int]):
    color[0] += 1
    condition[0] = int(color[0] < count[0] and error[0] == 0)


@wp.kernel(module="unique", enable_backward=False)
def finish(poses: wp.array[wp.transform], previous: wp.array[wp.transform], omega: wp.array[wp.vec3],
            translate: wp.array[int], rotate: wp.array[int], target_pose: wp.array[wp.transform],
            target_velocity: wp.array[wp.spatial_vector], dt: float, error: wp.array[int]):
    b = wp.tid()
    if error[0] == 0:
        velocity = wp.vec3(0.0)
        angular = wp.vec3(0.0)
        if translate[b] != 0:
            velocity = (wp.transform_get_translation(poses[b]) - wp.transform_get_translation(previous[b])) / dt
        if rotate[b] != 0:
            angular = omega[b]
        target_pose[b] = poses[b]
        target_velocity[b] = wp.spatial_vector(velocity[0], velocity[1], velocity[2], angular[0], angular[1], angular[2])


@wp.struct
class BodyData:
    q: wp.array[wp.vec3]
    previous: wp.array[wp.vec3]
    poses: wp.array[wp.transform]
    previous_pose: wp.array[wp.transform]
    previous_velocity: wp.array[wp.spatial_vector]
    omega: wp.array[wp.vec3]
    anchors: wp.array[wp.transform]
    mass: wp.array[float]
    second_moment: wp.array[wp.mat33]
    translate: wp.array[int]
    rotate: wp.array[int]
    owners: wp.array[int]
    reference: wp.array[wp.vec3]
    nodes: wp.array[int]
    node_offsets: wp.array[int]
    surface: wp.array[int]
    pin_bodies: wp.array[int]
    pin_references: wp.array[wp.vec3]
    pin_targets: wp.array[wp.vec3]
    triangles: wp.array2d[int]
    edges: wp.array2d[int]
    info: TriMeshCollisionInfo
    candidates: wp.array[int]
    incidence: wp.array[int]
    offsets: wp.array[int]
    colors: wp.array[int]
    color: wp.array[int]
    contact_parameters: contact.Parameters
    colliders: wp.array[sdf.Collider]
    sdf_parameters: sdf.Parameters
    words: wp.array[wp.uint32]
    slots: wp.array[int]
    error: wp.array[int]
    dt: float
    damping: float
    gravity_y: float
    kpin: float
    radius: wp.float64
    angle: float
    use_ccd: int
    use_sdf: int
    exact_geometry: int


@wp.func_native(r"""
    using namespace ipc_gpu::solver;
    wp::vec_t<4,float> result;
    if(angle==0||damping==0||(delta[0]==0&&delta[1]==0&&delta[2]==0)){
        for(int k=0;k<4;++k)result[k]=current_q[k];return result;
    }
    double previous[4],now[4],anchor[4],angular[3],change[3],target[4],bounded[4];
    bool okay=ipc_warp_rigid::unit(previous_q[0],previous_q[1],previous_q[2],previous_q[3],previous)
        &&ipc_warp_rigid::unit(current_q[0],current_q[1],current_q[2],current_q[3],now)
        &&ipc_warp_rigid::unit(anchor_q[0],anchor_q[1],anchor_q[2],anchor_q[3],anchor);
    const double largest=fmax(fabs(delta[0]),fmax(fabs(delta[1]),fabs(delta[2])));
    double scale=damping;
    if(largest>0){double squared=0;for(int k=0;k<3;++k){const double a=delta[k]/largest;squared+=a*a;}
        const double allowed=(double(angle)/dt)/largest/sqrt(squared);if(fabs(scale)>allowed)scale=copysign(allowed,scale);}
    for(int k=0;k<3;++k){change[k]=delta[k];angular[k]=double(omega[k])-scale*change[k];okay=okay&&isfinite(angular[k]);}
    if(okay){rigid_orientation(previous,angular,dt,target);
        okay=rotation_detail::normalize(target,target)
            &&bound_rigid_orientation(now,now,target,angle,bounded)==RigidRotationStatus::Success;
        if(okay){double relative[4];rotation_detail::relative(bounded,now,relative);
            okay=rotation_detail::normalize(relative,relative);
            const double actual=2*atan2(rotation_detail::vec(relative+1).norm(),relative[0]);
            if(okay&&actual>angle)okay=interpolate_rigid_orientation(now,bounded,double(angle)/actual,bounded)==RigidRotationStatus::Success;}
        if(okay)okay=bound_rigid_orientation(anchor,now,bounded,angle,bounded)==RigidRotationStatus::Success;
    }
    // Certify the stored float32 orientation against the same checkpoint cap.
    if(okay){double rounded[4],unit[4],relative[4];for(int k=0;k<4;++k)rounded[k]=double(static_cast<float>(bounded[k]));
        okay=rotation_detail::normalize(rounded,unit);
        rotation_detail::relative(unit,anchor,relative);
        const double actual=2*atan2(rotation_detail::vec(relative+1).norm(),relative[0]);
        if(actual>double(angle)){
            const double inset=fmax(0.0,double(angle)-64*FLT_EPSILON);
            okay=interpolate_rigid_orientation(anchor,unit,actual>0?inset/actual:0,bounded)==RigidRotationStatus::Success;
        }
    }
    for(int k=0;k<4;++k)result[k]=okay?static_cast<float>(bounded[(k+1)%4]):static_cast<float>(NAN);
    return result;
""")
def _rotation_target(previous_q: wp.quat, current_q: wp.quat, anchor_q: wp.quat,
                     omega: wp.vec3, delta: wp.vec3, dt: float, damping: float, angle: float) -> wp.vec4: ...


@wp.func_native(r"""
    wp::vec_t<4,float> out;double a[4],b[4],q[4];
    const bool okay=ipc_warp_rigid::unit(current[0],current[1],current[2],current[3],a)
        &&ipc_warp_rigid::unit(target[0],target[1],target[2],target[3],b)
        &&ipc_gpu::solver::interpolate_rigid_orientation(a,b,fraction,q)==ipc_gpu::solver::RigidRotationStatus::Success;
    for(int k=0;k<4;++k)out[k]=okay?static_cast<float>(q[(k+1)%4]):static_cast<float>(NAN);return out;
""")
def _interpolate(current: wp.quat, target: wp.quat, fraction: wp.float64) -> wp.vec4: ...


@wp.func_native(r"""
    wp::vec_t<3,float> out;double old[4],q[4],w[3];
    const bool okay=ipc_warp_rigid::unit(previous[0],previous[1],previous[2],previous[3],old)
        &&ipc_warp_rigid::unit(current[0],current[1],current[2],current[3],q)
        &&ipc_gpu::solver::rigid_angular_velocity_from_orientation(q,old,dt,w)==ipc_gpu::solver::RigidRotationStatus::Success;
    for(int k=0;k<3;++k)out[k]=okay?static_cast<float>(w[k]):static_cast<float>(NAN);return out;
""")
def _angular_velocity(previous: wp.quat, current: wp.quat, dt: float) -> wp.vec3: ...


# Each call owns a complete cold-worker pool. The caller supplies one unique
# worker index and launches with block_dim=1, as the contact exact queue does.
@wp.func_native(r"""
    using namespace ipc_gpu;
    using namespace ipc_gpu::solver;
    wp::vec_t<2,double> output;output[0]=1;output[1]=0;
    double start[4][3],end[4][3],ideal[4][3],center[3],target_center[3],q[4],target_q[4];
    bool okay=ipc_warp_rigid::unit(current_rotation[0],current_rotation[1],current_rotation[2],current_rotation[3],q)
        &&ipc_warp_rigid::unit(target_rotation[0],target_rotation[1],target_rotation[2],target_rotation[3],target_q);
    for(int k=0;k<3;++k){center[k]=current_com[k];target_center[k]=target_com[k];}
    const bool face=kind==0,first=owners.data[nodes[0]]==body;
    bool second=owners.data[nodes[face?1:2]]==body,moved=false;
    for(int i=0;i<4;++i){const bool selected=owners.data[nodes[i]]==body;
        okay=okay&&(selected==((i<(face?1:2))?first:second));
        double ref[3],point[3];for(int k=0;k<3;++k)ref[k]=reference.data[nodes[i]][k];
        if(selected)rigid_world_position(ref,target_center,target_q,point);
        for(int k=0;k<3;++k){start[i][k]=positions.data[nodes[i]][k];
            end[i][k]=selected?double(static_cast<float>(point[k])):start[i][k];
            moved=moved||end[i][k]!=start[i][k];okay=okay&&isfinite(start[i][k])&&isfinite(end[i][k]);}}
    if(!okay||first==second){output[1]=2;return output;}
    if(!moved)return output;
#if defined(__CUDA_ARCH__)
    exact::DevicePool pool;exact::initialize_pool(pool,words.data+size_t(worker)*2048*512,
        slots.data+size_t(worker)*512,2048,512);
#endif
    auto clean=[&](){
#if defined(__CUDA_ARCH__)
        return !pool.failed&&pool.available==pool.slot_count;
#else
        return true;
#endif
    };
    auto represented=[&](const double a[4][3],const double b[4][3]){
        auto result=original_rigid::represented_path<256>(a,b,face);
        if(clean()&&result==original_rigid::RepresentationStatus::CapacityExceeded)
            result=original_rigid::represented_path<2048>(a,b,face);
        return result;
    };
    using R=original_rigid::RepresentationStatus;
    auto publish=[&](R result){if(result==R::Uncertain)output[0]=fmin(output[0],0.5);
        else if(result==R::InitialContact)output[1]=1;
        else if(result==R::Invalid)output[1]=2;
        else if(result==R::CapacityExceeded)output[1]=3;};
    // A true initial intersection is an error even when the ORIGINAL TOI
    // returns a zero fraction; do not turn it into a silent successful freeze.
    publish(represented(start,start));
    if(!rotational){
        const double displacement[3]={target_center[0]-center[0],target_center[1]-center[1],target_center[2]-center[2]};
        // Preserve our original relative-frame TOI, then independently prove
        // the rounded multi-vertex endpoint path actually stored in Newton.
        if(output[1]==0&&!exact_geometry&&original_rigid::translation_boxes_overlap(start,face,first,displacement)){
            const auto query=original_rigid::translation_query(start,face,first,displacement);
            auto hit=original_ccd::evaluate_fast(query);
            if(hit.status==original_ccd::Status::NeedsExact){hit=original_ccd::evaluate_exact<256>(query);
                if(clean()&&hit.status==original_ccd::Status::CapacityExceeded)hit=original_ccd::evaluate_exact<2048>(query);}
            if(hit.status==original_ccd::Status::CapacityExceeded)output[1]=3;
            else if(hit.status!=original_ccd::Status::Ok)output[1]=2;
            else if(hit.collision)output[0]=fmin(output[0],0.9*hit.toi);
        }
        if(output[1]==0&&output[0]==1)publish(represented(start,end));
    }else{
        double relative[4],reverse[4];rotation_detail::relative(target_q,q,relative);
        rotation_detail::relative(q,target_q,reverse);
        okay=rotation_detail::normalize(relative,relative)&&rotation_detail::normalize(reverse,reverse);
        if(!okay)output[1]=2;
        if(output[1]==0){RigidRotationQuery query{};query.vertex_face=face;
            for(int k=0;k<3;++k)query.com[k]=center[k];
            for(int i=0;i<4;++i){query.current[i]=first?q[i]:(i==0?1:0);
                query.target[i]=first?target_q[i]:reverse[i];
                for(int k=0;k<3;++k)query.positions[i][k]=start[i][k];}
            const auto result=rigid_rotation_ccd(query);
            if(result.status!=RigidRotationStatus::Success)output[1]=2;
            else if(result.collision)output[0]=fmin(output[0],0.9*result.time);
            if(output[1]==0&&output[0]==1){
                // The analytic arc starts at the represented float32 proxy.
                // Fixed-reference projection may round its endpoint differently.
                // Prove the short joining segment rather than assuming equality.
                const double zero[3]={0,0,0};
                for(int i=0;i<4;++i){if(owners.data[nodes[i]]==body){double offset[3],point[3];
                        for(int k=0;k<3;++k)offset[k]=start[i][k]-center[k];
                        rigid_world_position(offset,zero,relative,point);
                        for(int k=0;k<3;++k)ideal[i][k]=point[k]+center[k];}
                    else for(int k=0;k<3;++k)ideal[i][k]=start[i][k];}
                publish(represented(ideal,end));
            }
        }
    }
    if(exact_geometry&&output[1]==0&&output[0]==1){
        double scale=0;for(int i=0;i<4;++i)for(int k=0;k<3;++k)scale=fmax(scale,fmax(fabs(start[i][k]),fabs(end[i][k])));
        const double gap=fmax(1e-8,128*DBL_EPSILON*scale);
        auto verdict=exact::validate_endpoint_gap<256>(start,end,face,gap);
        if(clean()&&verdict==exact::Status::CapacityExceeded)verdict=exact::validate_endpoint_gap<2048>(start,end,face,gap);
        if(verdict==exact::Status::Unsafe)output[0]=0.5;
        else if(verdict==exact::Status::InitialContact)output[1]=1;
        else if(verdict==exact::Status::Invalid)output[1]=2;
        else if(verdict==exact::Status::CapacityExceeded)output[1]=3;
    }
    if(!clean())output[1]=3;
#if defined(__CUDA_ARCH__)
    exact::release_pool();
#endif
    return output;
""")
def certify_pair(positions: wp.array[wp.vec3], reference: wp.array[wp.vec3], owners: wp.array[int],
                 nodes: wp.vec4i, kind: int, body: int, current_com: wp.vec3, current_rotation: wp.quat,
                 target_com: wp.vec3, target_rotation: wp.quat, rotational: int, exact_geometry: int,
                 words: wp.array[wp.uint32], slots: wp.array[int], worker: int) -> wp.vec2d: ...


@wp.func
def _system(data: BodyData, body: int, rotational: int, worker: int):
    pose = data.poses[body]
    center = wp.transform_get_translation(pose)
    old = data.previous_pose[body]
    old_q = wp.transform_get_rotation(old)
    velocity = data.previous_velocity[body]
    gradient = wp.vec3(0.0)
    hessian = wp.mat33(0.0)
    if rotational != 0:
        gradient, hessian = rotation_inertia(old_q, wp.vec3(velocity[3], velocity[4], velocity[5]),
            data.omega[body], data.dt, data.second_moment[body])
    else:
        gradient, hessian = translation_inertia(center, wp.transform_get_translation(old),
            wp.vec3(velocity[0], velocity[1], velocity[2]), data.mass[body], data.gravity_y, data.dt)
    for slot in range(data.offsets[body], data.offsets[body + 1]):
        candidate = data.candidates[data.incidence[slot]]
        nodes, kind = contact.decode_candidate(candidate, data.triangles, data.edges, data.info)
        packet = contact.evaluate_full(data.q, data.previous, nodes, kind, data.contact_parameters)
        status = int(packet[176])
        if status == 2 and data.exact_geometry != 0:
            packet = contact.evaluate_full_exact(data.q, data.previous, nodes, kind, data.contact_parameters,
                data.words, data.slots, worker)
            status = int(packet[176])
        if status != 0:
            wp.atomic_max(data.error, 0, 100 + status)
        else:
            g, h = contact_terms(packet, nodes, data.owners, data.reference, body,
                old_q, data.omega[body], data.dt, kind, rotational,
                wp.float32(data.contact_parameters.k_barrier) * data.dt * data.dt)
            gradient += g
            hessian += h
    for pin in range(data.pin_bodies.shape[0]):
        if data.pin_bodies[pin] == body:
            g, h, valid = point_pin_derivatives(data.pin_references[pin], data.pin_targets[pin],
                center, old_q, data.omega[body], data.dt, data.kpin, rotational)
            if valid == 0:
                wp.atomic_max(data.error, 0, 603)
            gradient += g
            hessian += h
    if data.use_sdf != 0:
        for slot in range(data.node_offsets[body], data.node_offsets[body + 1]):
            node = data.nodes[slot]
            if data.surface[node] != 0:
                sdf_packet = sdf.evaluate(data.q[node], data.previous[node], data.colliders, data.sdf_parameters)
                if sdf_packet[41] != wp.float64(0.0):
                    wp.atomic_max(data.error, 0, 49)
                else:
                    jacobian = wp.identity(n=3, dtype=float)
                    if rotational != 0:
                        jacobian, cx, cy, cz = reference_derivatives(data.reference[node], old_q,
                            data.omega[body], data.dt)
                    g, h = sdf.rigid_terms(sdf_packet, jacobian, data.dt, rotational)
                    gradient += g
                    hessian += h
    return gradient, hessian


@wp.func
def _accept(data: BodyData, body: int, target: wp.transform, rotational: int, worker: int):
    current = data.poses[body]
    center = wp.transform_get_translation(current)
    rotation = wp.transform_get_rotation(current)
    target_center = wp.transform_get_translation(target)
    target_rotation = wp.transform_get_rotation(target)
    fraction = wp.float64(1.0)
    done = bool(False)
    for attempt in range(65):
        if not done and data.error[0] == 0:
            trial_center = center
            trial_rotation = rotation
            if rotational != 0:
                value = _interpolate(rotation, target_rotation, fraction)
                trial_rotation = wp.quat(value[0], value[1], value[2], value[3])
            else:
                for k in range(3):
                    trial_center[k] = wp.float32(wp.float64(center[k]) + fraction *
                        (wp.float64(target_center[k]) - wp.float64(center[k])))
            same = bool(True)
            for k in range(3):
                same = same and trial_center[k] == center[k]
            for k in range(4):
                same = same and trial_rotation[k] == rotation[k]
            # A zero proposal is a true no-op. Reprojection would introduce a
            # second, unrequested rounding move and change finite-pin states.
            if same or fraction == wp.float64(0.0):
                done = True
            else:
                safe = wp.float64(1.0)
                if data.use_ccd != 0:
                    for slot in range(data.offsets[body], data.offsets[body + 1]):
                        nodes, kind = contact.decode_candidate(data.candidates[data.incidence[slot]],
                            data.triangles, data.edges, data.info)
                        result = certify_pair(data.q, data.reference, data.owners, nodes, kind, body,
                            center, rotation, trial_center, trial_rotation, rotational, data.exact_geometry,
                            data.words, data.slots, worker)
                        safe = wp.min(safe, result[0])
                        if result[1] != wp.float64(0.0):
                            wp.atomic_max(data.error, 0, 610 + int(result[1]))
                if safe == wp.float64(1.0) and data.error[0] == 0:
                    data.poses[body] = wp.transform(trial_center, trial_rotation)
                    if rotational != 0:
                        data.omega[body] = _angular_velocity(wp.transform_get_rotation(data.previous_pose[body]),
                            trial_rotation, data.dt)
                    for slot in range(data.node_offsets[body], data.node_offsets[body + 1]):
                        node = data.nodes[slot]
                        data.q[node] = world_point(data.reference[node], trial_center, trial_rotation)
                    done = True
                else:
                    fraction *= wp.max(wp.float64(0.0), safe)
    if not done and data.error[0] == 0:
        wp.atomic_max(data.error, 0, 614)


@wp.kernel(module="unique", enable_backward=False)
def _phase(data: BodyData, rotational: int, workers: int):
    worker = wp.tid()
    for body in range(worker, data.poses.shape[0], workers):
        enabled = data.rotate[body] if rotational != 0 else data.translate[body]
        if enabled != 0 and data.colors[body] == data.color[0] and data.error[0] == 0:
            gradient, hessian = _system(data, body, rotational, worker)
            delta, valid = solve3(hessian, gradient)
            if valid == 0:
                wp.atomic_max(data.error, 0, 604)
            elif data.error[0] == 0:
                pose = data.poses[body]
                center = wp.transform_get_translation(pose)
                rotation = wp.transform_get_rotation(pose)
                if rotational == 0:
                    anchor = wp.transform_get_translation(data.anchors[body])
                    proposed = center - data.damping * delta
                    center = round_inside(center, proposed, anchor, data.radius)
                else:
                    value = _rotation_target(wp.transform_get_rotation(data.previous_pose[body]), rotation,
                        wp.transform_get_rotation(data.anchors[body]), data.omega[body], delta,
                        data.dt, data.damping, data.angle)
                    rotation = wp.quat(value[0], value[1], value[2], value[3])
                finite = bool(True)
                for k in range(3):
                    finite = finite and wp.isfinite(center[k])
                for k in range(4):
                    finite = finite and wp.isfinite(rotation[k])
                if not finite:
                    wp.atomic_max(data.error, 0, 605)
                else:
                    _accept(data, body, wp.transform(center, rotation), rotational, worker)


class BodySolver:
    """Rigid block physics called by the inherited SolverVBD iteration hooks."""

    def __init__(self, solver):
        self.solver, self.model, self.device = solver, solver.model, solver.model.device
        m, ipc, c, p = self.model, self.model.ipc, solver.contact_backend, solver.params
        self.workers = min(m.body_count, c.workers)
        if c.capacity >= 2**30:
            raise ValueError("Rigid incidence exceeds the device int32 index capacity")
        self.radius = float(p["node_box_max"])
        self.angle = float(p["theta_box_max"])
        if not math.isfinite(self.angle) or self.angle < 0 or self.angle > math.pi:
            raise ValueError("The analytic rigid CCD cap requires 0 <= theta_box_max <= pi")
        self.previous_pose = wp.empty_like(m.body_q)
        # Private bank: linear entries hold v_old+dt*gravity; angular entries
        # retain old omega. The heavy translation/rotation systems share this
        # layout without changing their compiled kernel or native arithmetic.
        self.previous_velocity = wp.empty_like(m.body_qd)
        self.omega = wp.zeros(m.body_count, wp.vec3, device=self.device)
        self.anchors = wp.clone(m.body_q)
        self.counts = wp.zeros(m.body_count + 1, int, device=self.device)
        self.offsets = wp.zeros(m.body_count + 1, int, device=self.device)
        self.cursors = wp.zeros(m.body_count, int, device=self.device)
        self.incidence = wp.empty(2 * c.capacity, int, device=self.device)
        self.conflicts = wp.zeros((m.body_count, (m.body_count + 31) // 32), wp.uint32, device=self.device)
        self.colors = wp.zeros(m.body_count, int, device=self.device)
        self.color_marks = wp.zeros(m.body_count, int, device=self.device)
        self.color_count = wp.zeros(1, int, device=self.device)
        self.color = wp.zeros(1, int, device=self.device)
        self.condition = wp.zeros(1, int, device=self.device)
        references = ipc.body_references.numpy()
        maximum_radius = float(np.max(np.linalg.norm(references.astype(np.float64), axis=1), initial=0.0))
        arc = 2.0 * maximum_radius * math.sin(self.angle / 2.0)
        # Actual detector is AABB-only: each side can move by this amount per
        # coordinate. Include float32 projection rounding at both endpoints.
        coordinate = float(np.max(np.abs(m.particle_q.numpy()), initial=0.0))
        rounding = 64.0 * np.finfo(np.float32).eps * max(1.0, coordinate, maximum_radius)
        self.query_radius = max(float(p["d_hat"]), 1e-8) + 2.0 * (self.radius + arc + rounding)

    def refresh(self, state):
        """Called after the upstream BVH candidate rebuild, entirely on-device."""
        m, c = self.model, self.solver.contact_backend
        wp.copy(self.anchors, state.body_q)
        self.counts.zero_()
        self.cursors.zero_()
        self.conflicts.zero_()
        self.color_count.zero_()
        self.color_marks.zero_()
        if c.capacity:
            args = [c.candidate_ids, c.candidate_count, m.tri_indices, m.edge_indices, c.info, m.ipc.node_body]
            wp.launch(count_incidence, c.capacity,
                [*args, self.counts, self.conflicts, self.solver.error], device=self.device)
            # Body counts are small; this persistent single-thread scan avoids
            # CUB's temporary allocation in every captured detection pass.
            wp.launch(prefix_incidence, 1, [self.counts, self.offsets], device=self.device)
            wp.launch(fill_incidence, c.capacity,
                [*args, self.offsets, self.cursors, self.incidence, self.solver.error], device=self.device)
        else:
            self.offsets.zero_()
        wp.launch(color_bodies, 1,
            [self.conflicts, self.colors, self.color_marks, self.color_count, self.solver.error], device=self.device)

    def begin(self, state, dt):
        wp.launch(begin, self.model.body_count,
            [state.body_q, state.body_qd, self.model.gravity, self.model.body_world, float(dt),
             self.previous_pose, self.previous_velocity, self.omega], device=self.device)

    def _data(self, state, dt):
        m, ipc, c, p, s = self.model, self.model.ipc, self.solver.contact_backend, self.solver.params, self.solver.sdf_backend
        data = BodyData()
        data.q, data.previous = state.particle_q, self.solver.particle_q_prev
        data.poses, data.previous_pose, data.previous_velocity = state.body_q, self.previous_pose, self.previous_velocity
        data.omega, data.anchors = self.omega, self.anchors
        data.mass, data.second_moment = m.body_mass, ipc.body_second_moment
        data.translate, data.rotate = ipc.body_translate, ipc.body_rotate
        data.owners, data.reference = ipc.node_body, ipc.body_references
        data.nodes, data.node_offsets, data.surface = ipc.body_nodes, ipc.body_node_offsets, ipc.surface
        data.pin_bodies, data.pin_references, data.pin_targets = ipc.rigid_pin_bodies, ipc.rigid_pin_references, ipc.rigid_pin_targets
        data.triangles, data.edges, data.info = m.tri_indices, m.edge_indices, c.info
        data.candidates, data.incidence, data.offsets = c.candidate_ids, self.incidence, self.offsets
        data.colors, data.color = self.colors, self.color
        data.contact_parameters = contact.parameters(dt=dt, **c.options)
        data.colliders, data.sdf_parameters = s.colliders, s.params
        data.words, data.slots, data.error = c.words, c.slots, self.solver.error
        data.dt, data.damping, data.gravity_y = dt, p["damping"], 0.0
        data.kpin, data.radius, data.angle = p["kpin"], self.radius, self.angle
        data.use_ccd, data.use_sdf, data.exact_geometry = int(p["use_ccd"]), int(s.enabled), int(p["exact_computation_fallback"])
        return data

    def iterate(self, state, dt):
        data = self._data(state, dt)
        wp.launch(start_colors, 1, [self.color, self.color_count, self.condition, self.solver.error], device=self.device)

        def round_body():
            # COM commits before the fresh angular system is evaluated. Bodies
            # of one color share no candidate, so these writes are disjoint.
            wp.launch(_phase, self.workers, [data, 0, self.workers], block_dim=1, device=self.device)
            wp.launch(_phase, self.workers, [data, 1, self.workers], block_dim=1, device=self.device)
            wp.launch(next_color, 1, [self.color, self.color_count, self.condition, self.solver.error], device=self.device)

        wp.capture_while(self.condition, round_body)

    def finish(self, state_in, state_out, dt):
        wp.launch(finish, self.model.body_count,
            [state_in.body_q, self.previous_pose, self.omega, self.model.ipc.body_translate, self.model.ipc.body_rotate,
             state_out.body_q, state_out.body_qd, dt, self.solver.error], device=self.device)


def configure_body_module(module=None):
    """Attach the complete native preamble needed by the cold rigid phase."""
    target = __name__ if module is None else module
    contact.configure_module(target, include_filter=False)
    configure_module(target)
    sdf.configure_module(target)
    return _configure(target, BODY_HEADERS)


configure_body_module()
configure_body_module(_phase)
for _graph_kernel in (begin, count_incidence, prefix_incidence, fill_incidence,
                      color_bodies, start_colors, next_color, finish):
    wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": True},
                          module=_graph_kernel.module)
del _graph_kernel
