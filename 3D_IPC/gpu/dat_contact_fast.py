"""Compact float32-first DAT contact energy with ordinary/exact replay requests.

The returned packet has the established ``contact.Result`` ABI. Status 6 is
private to this specialization and requests high precision in the separate
workspace replay kernel; it must never be consumed as an inactive contact.
"""
import warp as wp
from . import contact
from .warp_build import configure_module as _configure


def configure_module(module=None):
    return _configure(__name__ if module is None else module, ("dat_contact_fast.cuh",))


@wp.func_native(r"""
    wp::vec_t<50,float> packed;
    for(int k=0;k<50;++k)packed[k]=0.0f;
    using namespace ipc_gpu::dat_contact_fast;
    Input in;
    for(int i=0;i<4;++i) {
        if(nodes[i]<0||nodes[i]>=positions.shape[0]||nodes[i]>=previous.shape[0]) {
            packed[48]=1.0f;return packed;
        }
        for(int k=0;k<3;++k) {
            in.x[i][k]=positions.data[nodes[i]][k];
            in.previous[i][k]=previous.data[nodes[i]][k];
        }
    }
    // Match the established zero-barrier/no-radius fast no-force packet.
    if(params.d_hat<=0.0||params.k_barrier<=0.0)return packed;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        if(!finite(in.x[i][k])||!finite(in.previous[i][k])) {
            packed[48]=1.0f;return packed;
        }
    }
    in.kind=kind;
    in.d_hat=static_cast<float>(params.d_hat);in.stiffness=static_cast<float>(params.k_barrier);
    in.dt=static_cast<float>(params.dt);in.friction=static_cast<float>(params.friction);in.eps_v=static_cast<float>(params.eps_v);
    // A cast or validation ambiguity belongs to the original evaluator.
    if((kind!=0&&kind!=1)||!finite(in.d_hat)||!finite(in.stiffness)||!finite(in.dt)
       ||!finite(in.friction)||!finite(in.eps_v)||!(in.d_hat>0.0f)||!(in.stiffness>0.0f)
       ||!(in.dt>0.0f)||in.friction<0.0f||(in.friction==0.0f&&params.friction!=0.0)
       ||!(in.eps_v>0.0f)||params.eps!=1.0e-12
       ||in.d_hat<FLT_MIN||in.stiffness<FLT_MIN||in.dt<FLT_MIN||in.eps_v<FLT_MIN
       ||(in.friction>0.0f&&in.friction<FLT_MIN)) {
        packed[48]=6.0f;return packed;
    }
    const int status=evaluate(in,&packed[0]);
    if(status) {for(int k=0;k<48;++k)packed[k]=0.0f;packed[48]=static_cast<float>(status);packed[49]=0.0f;}
    return packed;
""")
def evaluate_self(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3], nodes: wp.vec4i,
                  kind: int, params: contact.Parameters) -> contact.Result: ...


configure_module()
