"""Our SDF union/penalty/friction law on Newton float32 particle/pose data."""

import math

import numpy as np
import warp as wp

from .warp_build import configure_module as _configure


HEADERS=("sdf.cuh",)
Collider=wp.types.vector(length=32,dtype=wp.float64)
Packet=wp.types.vector(length=42,dtype=wp.float64)


def configure_module(module=None):
    return _configure(__name__ if module is None else module,HEADERS)


@wp.struct
class Parameters:
    stiffness: wp.float64
    epsilon: wp.float64
    dt: wp.float64
    friction: wp.float64
    eps_v: wp.float64
    include_curvature: int


def parameters(*,stiffness=0.0,epsilon=0.0,dt=1/300,friction=0.0,eps_v=.001,include_curvature=False):
    for name,value in (("stiffness",stiffness),("epsilon",epsilon),("friction",friction)):
        if isinstance(value,bool) or not math.isfinite(value) or value<0:raise ValueError(f"{name} must be finite and nonnegative")
    for name,value in (("dt",dt),("eps_v",eps_v)):
        if isinstance(value,bool) or not math.isfinite(value) or value<=0:raise ValueError(f"{name} must be finite and positive")
    result=Parameters()
    result.stiffness=stiffness;result.epsilon=epsilon;result.dt=dt
    result.friction=friction;result.eps_v=eps_v;result.include_curvature=int(include_curvature)
    return result


def pack_colliders(records):
    """Immutable host metadata only; dynamic collider motion is updated on GPU."""
    result=np.zeros((len(records),32),dtype=np.float32)
    kinds={"plane":0,"cylinder":1,"sphere":2}
    for i,record in enumerate(records):
        result[i,0]=kinds[record["kind"]]
        result[i,1:4]=record["point"];result[i,4:7]=record.get("axis",[0,1,0]);result[i,7]=record.get("radius",0)
        for name,offset in (("previous_pose",8),("current_pose",20)):
            pose=record.get(name,{})
            result[i,offset:offset+9]=np.asarray(pose.get("rotation",np.eye(3)),dtype=np.float32).reshape(9)
            result[i,offset+9:offset+12]=pose.get("translation",[0,0,0])
    return result


@wp.func_native(r"""
    double r[9];for(int k=0;k<9;++k)r[k]=data[offset+k];
    double error=0;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        const double value=(r[i]*r[j]+r[3+i]*r[3+j])+r[6+i]*r[6+j];
        error=fmax(error,fabs(value-(i==j?1.0:0.0)));
    }
    const double det=r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]);
    valid=error<=2e-6&&fabs(det-1.0)<=4e-6?1:0;
    for(int k=0;k<9;++k)if(!isfinite(r[k]))valid=0;
    if(valid){
        double a[3]{r[0],r[3],r[6]},b[3]{r[1],r[4],r[7]};
        const double norm=sqrt((a[0]*a[0]+a[1]*a[1])+a[2]*a[2]);
        for(int k=0;k<3;++k)a[k]/=norm;
        const double dot=(a[0]*b[0]+a[1]*b[1])+a[2]*b[2];
        for(int k=0;k<3;++k)b[k]-=dot*a[k];
        const double bn=sqrt((b[0]*b[0]+b[1]*b[1])+b[2]*b[2]);
        for(int k=0;k<3;++k)b[k]/=bn;
        const double c[3]{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
        for(int k=0;k<3;++k){out[offset+3*k]=a[k];out[offset+3*k+1]=b[k];out[offset+3*k+2]=c[k];}
    }
""")
def _pose(data: Collider,offset: int,out: wp.ref[Collider],valid: wp.ref[int]): ...


@wp.kernel
def prepare_colliders(source: wp.array2d[float],prepared: wp.array[Collider],check_material: int,error: wp.array[int]):
    i=wp.tid()
    data=Collider()
    for k in range(32):data[k]=wp.float64(source[i,k])
    same=True
    for k in range(12):
        if data[8+k]!=data[20+k]:same=False
    if check_material!=0 and not same:
        valid0=int(0);valid1=int(0)
        _pose(data,8,data,valid0)
        _pose(data,20,data,valid1)
        if valid0==0 or valid1==0:wp.atomic_max(error,0,48)
        for k in range(3):
            if not wp.isfinite(data[17+k]) or not wp.isfinite(data[29+k]):wp.atomic_max(error,0,48)
    prepared[i]=data


_PACK=r"""
    wp::vec_t<42,double> packed;
    packed[0]=out.phi;
    for(int k=0;k<3;++k){packed[1+k]=out.grad_phi[k];packed[13+k]=out.surface_point[k];packed[16+k]=out.gradient[k];packed[28+k]=out.friction_gradient[k];}
    for(int k=0;k<9;++k){packed[4+k]=out.hess_phi[k];packed[19+k]=out.hessian[k];packed[31+k]=out.friction_hessian[k];}
    packed[40]=out.collider;packed[41]=static_cast<int>(out.status);return packed;
"""


@wp.func_native(r"""
    using namespace ipc_gpu::solver;
    SdfParameters p;p.stiffness=params.stiffness;p.epsilon=params.epsilon;p.dt=params.dt;
    p.friction=params.friction;p.eps_v=params.eps_v;p.include_curvature=params.include_curvature;
    double x[3],old[3];for(int k=0;k<3;++k){x[k]=position[k];old[k]=previous[k];}
    SdfOutput out=evaluate_sdf_contact(nullptr,0,x,old,p);
    SdfCollider selected;int selected_index=-1;
    if(out.status==SdfStatus::Ok)for(int i=0;i<colliders.shape[0];++i){
        const auto& data=colliders.data[i];SdfCollider item;
        if(!(data[0]==0||data[0]==1||data[0]==2)){out.status=SdfStatus::InvalidInput;break;}
        item.kind=static_cast<SdfKind>(static_cast<int>(data[0]));item.radius=data[7];
        for(int k=0;k<3;++k){item.point[k]=data[1+k];item.axis[k]=data[4+k];
            item.previous_pose.translation[k]=data[17+k];item.current_pose.translation[k]=data[29+k];}
        for(int k=0;k<9;++k){item.previous_pose.rotation[k]=data[8+k];item.current_pose.rotation[k]=data[20+k];}
        SdfOutput candidate;
        if(!sdf_detail::geometry(item,x,candidate)){out.status=SdfStatus::InvalidInput;break;}
        if(selected_index<0||candidate.phi<out.phi){selected=item;selected_index=i;out=candidate;}
    }
    if(out.status==SdfStatus::Ok&&selected_index>=0){out=evaluate_sdf_contact(&selected,1,x,old,p);out.collider=selected_index;}
"""+_PACK)
def evaluate(position: wp.vec3,previous: wp.vec3,colliders: wp.array[Collider],params: Parameters)->Packet: ...


@wp.func
def vertex_terms(packet: Packet,dt: float):
    """Physical force/Hessian for the inherited VBD float32 accumulators."""
    f=wp.vec3();h=wp.mat33()
    inv_dt2=wp.float64(1.0)/(wp.float64(dt)*wp.float64(dt))
    for r in range(3):
        f[r]=wp.float32(-packet[16+r]-inv_dt2*packet[28+r])
        for c in range(3):h[r,c]=wp.float32(packet[19+3*r+c]+inv_dt2*packet[31+3*r+c])
    return f,h


@wp.func_native(r"""
    ipc_gpu::solver::SdfOutput s;
    for(int k=0;k<3;++k){s.gradient[k]=packet[16+k];s.friction_gradient[k]=packet[28+k];}
    for(int k=0;k<9;++k){s.hessian[k]=packet[19+k];s.friction_hessian[k]=packet[31+k];}
    double j[9];for(int r=0;r<3;++r)for(int c=0;c<3;++c)j[3*r+c]=jacobian.data[r][c];
    const auto result=ipc_gpu::solver::sdf_rigid_pullback(s,j,static_cast<double>(dt)*dt);
    const double* g=rotational?result.orientation_gradient:result.translation_gradient;
    const double* h=rotational?result.orientation_orientation_hessian:result.translation_translation_hessian;
    for(int r=0;r<3;++r){gradient[r]=static_cast<float>(g[r]);for(int c=0;c<3;++c)hessian.data[r][c]=static_cast<float>(h[3*r+c]);}
""")
def _rigid(packet: Packet,jacobian: wp.mat33,dt: float,rotational: int,
            gradient: wp.ref[wp.vec3],hessian: wp.ref[wp.mat33]): ...


@wp.func
def rigid_terms(packet: Packet,jacobian: wp.mat33,dt: float,rotational: int):
    g=wp.vec3();h=wp.mat33()
    _rigid(packet,jacobian,dt,rotational,g,h)
    return g,h


@wp.kernel
def accumulate(positions: wp.array[wp.vec3],previous: wp.array[wp.vec3],flags: wp.array[int],
               colors: wp.array[int],color: int,solid: wp.array[int],surface: wp.array[int],
               colliders: wp.array[Collider],params: Parameters,forces: wp.array[wp.vec3],
               hessians: wp.array[wp.mat33],error: wp.array[int]):
    i=wp.tid()
    if (flags[i]&1)!=0 and colors[i]==color and (solid[i]==0 or surface[i]!=0):
        packet=evaluate(positions[i],previous[i],colliders,params)
        if packet[41]!=wp.float64(0.0):wp.atomic_max(error,0,49)
        else:
            f,h=vertex_terms(packet,wp.float32(params.dt))
            valid=True
            for a in range(3):
                valid=valid and wp.isfinite(f[a])
                for b in range(3):valid=valid and wp.isfinite(h[a,b])
            if valid:
                forces[i]+=f
                hessians[i]+=h
            else:wp.atomic_max(error,0,49)


class SdfBackend:
    """Persistent prepared collider bank and graph-safe color accumulation."""
    def __init__(self,model,error,*,include_curvature=None,precision=None):
        self.model,self.error=model,error
        self.device=model.device
        self.source=model.ipc.collider_data
        self.colliders=wp.empty(self.source.shape[0],dtype=Collider,device=self.device)
        self.prepared=self.colliders
        p=model.ipc.params
        if include_curvature is None:include_curvature=p.get("sdf_include_curvature",False)
        if not isinstance(include_curvature,bool):raise TypeError("SDF curvature mode must be a boolean")
        if precision is None:precision=p.get("sdf_precision","float32")
        if precision not in ("float32","float64"):raise ValueError("SDF precision must be float32 or float64")
        self.precision=precision
        self.params=parameters(stiffness=p["k_sdf"],epsilon=p["eps_sdf"],dt=p["dt"],
            friction=p["friction_coefficient"],eps_v=p["friction_velocity_epsilon"],include_curvature=include_curvature)
        self.enabled=self.source.shape[0]>0 and p["k_sdf"]>0
        self.fast_context=None
        self._fast_accumulate=self._replay=None
        self._checked_accumulate=self._prepared_accumulate=None
        # Small scenes otherwise launch too few 256-thread blocks to use this
        # GPU's SMs. Recorded-input timing favors one-warp blocks in that range.
        self.block_dim=(32 if self.device.is_cuda and model.particle_count<=32*self.device.sm_count else 256)
        # This describes the current prepared bank, not the mutable source.
        # A zero-friction prepare deliberately does not validate material poses.
        self._poses_validated=False
        if precision=="float32":
            from . import sdf_fast
            ctx=sdf_fast.Context()
            ctx.pending=wp.empty(model.particle_count,int,device=self.device)
            ctx.count=wp.zeros(1,int,device=self.device)
            ctx.error=error
            ctx.workers=self.device.sm_count*128 if self.device.is_cuda else 1
            self.fast_context=ctx
            self._fast_accumulate=sdf_fast.accumulate
            self._checked_accumulate=sdf_fast.accumulate
            self._prepared_accumulate=sdf_fast.accumulate_prepared
            self._replay=sdf_fast.make_replay()

    def prepare(self,dt):
        self.params.dt=dt
        self._poses_validated=False
        if self.enabled:
            check_material=int(self.params.friction>0)
            wp.launch(prepare_colliders,dim=self.source.shape[0],
                inputs=[self.source,self.colliders,check_material,self.error],device=self.device)
            # The CUDA stream orders this validation before accumulation. Any
            # invalid pose latches error 48, which both fast kernels check.
            self._poses_validated=bool(check_material)

    def accumulate(self,q,previous,color,forces,hessians,dt):
        if self.enabled:
            self.params.dt=dt
            common=[q,previous,self.model.particle_flags,self.model.particle_colors,color,
                    self.model.ipc.solid,self.model.ipc.surface,self.colliders,self.params,forces,hessians]
            if self.fast_context is None:
                wp.launch(accumulate,dim=self.model.particle_count,inputs=[*common,self.error],device=self.device)
            else:
                self._fast_accumulate=(self._prepared_accumulate if self._poses_validated
                                       else self._checked_accumulate)
                self.fast_context.count.zero_()
                wp.launch(self._fast_accumulate,dim=self.model.particle_count,
                    inputs=[*common,self.fast_context],device=self.device,block_dim=self.block_dim)
                wp.launch(self._replay,dim=self.fast_context.workers,
                    inputs=[q,previous,self.colliders,self.params,forces,hessians,self.fast_context],
                    device=self.device,block_dim=128 if self.device.is_cuda else 1)


configure_module()
