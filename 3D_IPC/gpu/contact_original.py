"""GS-owned original/zero-friction contact assembly.

A light one-sided geometry certificate discards proven inactive pairs before
launching the unchanged native original law with compact all-four diagonal
storage. The uncertain queue is rebuilt at every color. Other contact policies
inherit ContactBackend unchanged. SolverDAT uses that shared default backend.
"""
from functools import cache
import warp as wp
from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo
from .contact import ContactBackend, Parameters, Result, _INPUT, decode_candidate, scatter_self, parameters
from .warp_build import configure_module

HEADERS=("contact_original.cuh","separation.cuh")

# Match evaluate_self's exact preflight and existing certificate. Invalid or
# uncertified inputs always reach the full evaluator and retain its error.
# Represented geometry is promoted to binary64 exactly as in the shared wrapper.
@wp.func_native(r"""
    for(int i=0;i<4;++i)
        if(nodes[i]<0||nodes[i]>=positions.shape[0]||nodes[i]>=previous.shape[0])return 1;
    if(!(params.d_hat>0&&params.k_barrier>0))return 0;
    bool filterable=ipc_gpu::separation::finite(params.k_barrier)
        &&ipc_gpu::separation::finite(params.dt)&&params.dt>0
        &&ipc_gpu::separation::finite(params.friction)&&params.friction>=0
        &&ipc_gpu::separation::finite(params.eps_v)&&params.eps_v>0
        &&ipc_gpu::separation::finite(params.eps)&&params.eps>0;
    double points[4][3];
    for(int i=0;i<4;++i)for(int k=0;k<3;++k){
        points[i][k]=double(positions.data[nodes[i]][k]);
        filterable=filterable&&ipc_gpu::separation::finite(double(previous.data[nodes[i]][k]));
    }
    return !filterable||!ipc_gpu::separation::safe(points,points,kind==0,params.d_hat);
""")
def needs_original_evaluation(positions:wp.array[wp.vec3],previous:wp.array[wp.vec3],nodes:wp.vec4i,
                              kind:int,params:Parameters)->int: ...


@wp.func_native(_INPUT+r"""
    if(params.exact_geometry!=0||params.friction!=0.0){
        wp::vec_t<50,float> invalid;for(int k=0;k<50;++k)invalid[k]=0;invalid[48]=1;return invalid;
    }
    original_contact_storage::OriginalSelfZeroOutput out;
    if(params.d_hat>0&&params.k_barrier>0){
        bool filterable=ipc_gpu::separation::finite(in.k_barrier)
            &&ipc_gpu::separation::finite(in.dt)&&in.dt>0
            &&ipc_gpu::separation::finite(in.friction)&&in.friction>=0
            &&ipc_gpu::separation::finite(in.eps_v)&&in.eps_v>0
            &&ipc_gpu::separation::finite(in.eps)&&in.eps>0;
        for(int i=0;i<4;++i)for(int k=0;k<3;++k)
            filterable=filterable&&ipc_gpu::separation::finite(in.previous[i][k]);
        if(!filterable||!ipc_gpu::separation::safe(in.positions,in.positions,kind==0,params.d_hat))
            out=evaluate_contact_original_self_zero(in);
    }
    wp::vec_t<50,float> packed;for(int k=0;k<50;++k)packed[k]=0;
    packed[48]=static_cast<float>(static_cast<int>(out.status));packed[49]=static_cast<float>(out.active);
    if(out.status==ContactStatus::Ok&&out.active){
        const double inverse_dt2=1.0/(params.dt*params.dt);
        for(int role=0;role<4;++role){
            const double weight=out.weights[role];
            for(int k=0;k<3;++k){
                // Retain these original zero-friction operations, including
                // signed zero and inverse-dt overflow behavior.
                const double g=params.k_barrier*out.barrier_gradient[role][k]+(weight*0.0)*inverse_dt2;
                packed[12*role+k]=static_cast<float>(-g);
                for(int l=0;l<3;++l){
                    const double h=params.k_barrier*out.self_hessian[role][3*k+l]
                        +((weight*0.0)*weight)*inverse_dt2;
                    packed[12*role+3+3*k+l]=static_cast<float>(h);
                }
            }
        }
        for(int k=0;k<48;++k)if(!isfinite(packed[k])){packed[48]=4;packed[49]=0;break;}
    }
    return packed;
""")
def evaluate_original(positions:wp.array[wp.vec3],previous:wp.array[wp.vec3],nodes:wp.vec4i,
                      kind:int,params:Parameters)->Result: ...


@cache
def make_filter_kernel():
    @wp.kernel(module="unique",enable_backward=False)
    def filter_kernel(positions:wp.array[wp.vec3],previous:wp.array[wp.vec3],
        triangles:wp.array2d[int],edges:wp.array2d[int],info:TriMeshCollisionInfo,
        ids:wp.array[int],count:wp.array[int],colors:wp.array[int],flags:wp.array[int],mass:wp.array[float],
        color:int,params:Parameters,workers:int,pending:wp.array[int],pending_count:wp.array[int],error:wp.array[int]):
        if error[0]==0:
            for at in range(wp.tid(),count[0],workers):
                nodes,kind=decode_candidate(ids[at],triangles,edges,info)
                touches=False
                for role in range(4):
                    node=nodes[role]
                    if colors[node]==color and flags[node]&1!=0 and mass[node]>0.0:touches=True
                if touches and needs_original_evaluation(positions,previous,nodes,kind,params)!=0:
                    slot=wp.atomic_add(pending_count,0,1)
                    if slot<pending.shape[0]:pending[slot]=ids[at]
                    else:wp.atomic_max(error,0,12)
    configure_module(filter_kernel,("separation.cuh",))
    return filter_kernel


@cache
def make_original_kernel():
    @wp.kernel(module="unique",enable_backward=False)
    def original_kernel(positions:wp.array[wp.vec3],previous:wp.array[wp.vec3],
        triangles:wp.array2d[int],edges:wp.array2d[int],info:TriMeshCollisionInfo,
        ids:wp.array[int],count:wp.array[int],colors:wp.array[int],flags:wp.array[int],mass:wp.array[float],
        color:int,params:Parameters,workers:int,forces:wp.array[wp.vec3],hessians:wp.array[wp.mat33],error:wp.array[int]):
        if error[0]==0:
            for at in range(wp.tid(),count[0],workers):
                nodes,kind=decode_candidate(ids[at],triangles,edges,info)
                touches=False
                for role in range(4):
                    node=nodes[role]
                    if colors[node]==color and flags[node]&1!=0 and mass[node]>0.0:touches=True
                if touches:
                    result=evaluate_original(positions,previous,nodes,kind,params)
                    if int(result[48])!=0:wp.atomic_max(error,0,100+int(result[48]))
                    elif result[49]!=0.0:scatter_self(result,nodes,colors,flags,mass,color,forces,hessians)
    configure_module(original_kernel,HEADERS)
    return original_kernel


class ContactBackendIPC(ContactBackend):
    """Our GS backend; DAT and shared ContactBackend retain their defaults."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._original_zero_friction=(not self.options['exact_geometry'] and self.options['friction']==0.0)
        if self._original_zero_friction:
            # These private launch settings were measured on frozen real GS
            # inputs. Shared fast_workers and cold exact arenas stay intact.
            self._original_block_dim=64
            self._original_workers=max(128,self.device.sm_count*256) if self.device.is_cuda else self.fast_workers
            self._filter_original=make_filter_kernel()
            self._evaluate_original=make_original_kernel()

    def accumulate(self,positions,previous,color,forces,hessians,dt):
        if not self._original_zero_friction or self.options['exact_geometry'] or self.options['friction']!=0.0:
            return super().accumulate(positions,previous,color,forces,hessians,dt)
        p=parameters(dt=float(dt),**self.options)
        self.pending_count.zero_()
        if self.capacity:
            wp.launch(self._filter_original,self._original_workers,
                [positions,previous,self.model.tri_indices,self.model.edge_indices,self.info,
                 self.candidate_ids,self.candidate_count,self.model.particle_colors,self.model.particle_flags,
                 self.model.particle_mass,int(color),p,self._original_workers,self.pending,self.pending_count,self.error],device=self.device,block_dim=self._original_block_dim)
            # In original mode NeedsExact fails closed; this queue holds encoded
            # uncertain pair IDs, never exact requests. Every color rebuilds it.
            wp.launch(self._evaluate_original,self._original_workers,
                [positions,previous,self.model.tri_indices,self.model.edge_indices,self.info,
                 self.pending,self.pending_count,self.model.particle_colors,self.model.particle_flags,
                 self.model.particle_mass,int(color),p,self._original_workers,forces,hessians,self.error],device=self.device,block_dim=self._original_block_dim)


configure_module(__name__,HEADERS)
