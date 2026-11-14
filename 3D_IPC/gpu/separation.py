"""One-sided F32 endpoint-hull separation proof; uncertainty is not acceptance."""

import warp as wp

from .warp_build import configure_module as _configure


HEADERS = ("separation.cuh",)


def configure_module(module=None):
    return _configure(__name__ if module is None else module, HEADERS)


@wp.func_native(r"""
    if(kind!=0&&kind!=1)return false;
    float a[4][3],b[4][3];
    const bool same=start.data==end.data;
    for(int i=0;i<4;++i){
        if(nodes[i]<0||nodes[i]>=start.shape[0]||nodes[i]>=end.shape[0])return false;
        for(int k=0;k<3;++k){a[i][k]=start.data[nodes[i]][k];
            if(!same)b[i][k]=end.data[nodes[i]][k];}
    }
    return ipc_gpu::separation::safe(a,same?a:b,kind==0,gap);
""")
def safe(start: wp.array[wp.vec3],end: wp.array[wp.vec3],nodes: wp.vec4i,kind: int,gap: wp.float64)->bool: ...


configure_module()
