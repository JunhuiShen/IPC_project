"""Canonical node-ID greedy coloring, evaluated through its dependency DAG.

A vertex publishes a color only after every lower-ID neighbor has a color.
The resulting minimum free color is exactly the serial ordered-greedy result;
rounds change execution only. All arithmetic in this module is integer.
"""
import warp as wp
from .parallel_levels import _validate_rows, _start_loop, _next_loop


@wp.func_native(r"""
    unsigned long long forbidden[2]={0,0};
    for(uint64_t at=rows.data[vertex];at<rows.data[vertex+1];++at) {
        const int neighbor=adjacency.data[at];
        if(neighbor>=vertex)break;
        const int color=colors.data[neighbor];
        if(color<0)return -1;
        if(color<128)forbidden[color/64]|=1ull<<(color%64);
    }
    for(int word=0;word<2;++word) {
        const unsigned long long available=~forbidden[word];
        if(available) {
#if defined(__CUDA_ARCH__)
            return 64*word+__ffsll(static_cast<long long>(available))-1;
#else
            for(int bit=0;bit<64;++bit)if(available&(1ull<<bit))return 64*word+bit;
#endif
        }
    }
    // No color-count cap: graphs requiring more than 128 colors retain the
    // same exact minimum-free-color search as the original implementation.
    for(int color=128;color<colors.shape[0];++color) {
        bool available=true;
        for(uint64_t at=rows.data[vertex];at<rows.data[vertex+1];++at) {
            const int neighbor=adjacency.data[at];
            if(neighbor>=vertex)break;
            if(colors.data[neighbor]==color){available=false;break;}
        }
        if(available)return color;
    }
    return -1;
""")
def _ready_color(rows:wp.array[wp.uint64],adjacency:wp.array[int],colors:wp.array[int],vertex:int)->int: ...


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    const unsigned mask=__activemask();
    const unsigned votes=__ballot_sync(mask,pending!=0);
    return votes!=0 && static_cast<int>(threadIdx.x%32)==__ffs(mask)-1;
#else
    return pending!=0;
#endif
""")
def _pending_leader(pending:int)->bool: ...


@wp.kernel
def _color_pass(rows:wp.array[wp.uint64],adjacency:wp.array[int],source:wp.array[int],
                target:wp.array[int],pending:wp.array[int],error:wp.array[int]):
    vertex=wp.tid()
    remaining=int(0)
    if error[0]==0:
        color=source[vertex]
        if color<0:color=_ready_color(rows,adjacency,source,vertex)
        target[vertex]=color
        remaining=int(color<0)
    if _pending_leader(remaining):wp.atomic_or(pending,0,1)


@wp.kernel
def _publish_colors(colors:wp.array[int],trial_order:wp.array[int],metadata:wp.array[int],error:wp.array[int]):
    vertex=wp.tid()
    if error[0]==0:
        color=colors[vertex]
        if color<0 or color>=colors.shape[0]:wp.atomic_max(error,0,371)
        else:
            wp.atomic_max(metadata,0,color+1)
            wp.atomic_max(metadata,1,color+1)
            trial_order[vertex]=vertex
        if vertex==0:metadata[2]=colors.shape[0]


class ParallelColors:
    def __init__(self,graph):
        self.graph,self.n,self.device=graph,graph.n,graph.device
        self.temporary=wp.empty(self.n,int,device=self.device)
        self.pending=wp.zeros(1,int,device=self.device)
        self.condition=wp.zeros(1,int,device=self.device)
        self.iterations=wp.zeros(1,int,device=self.device)

    def apply(self,contacts):
        g,n,d=self.graph,self.n,self.device
        if n==0:raise ValueError('Empty model is not a supported predictor epoch')
        wp.launch(_validate_rows,n,[g.rows,g.adjacency,contacts.error],device=d)
        g.colors.fill_(-1);g.metadata.zero_()
        wp.launch(_start_loop,1,[self.condition,self.iterations,contacts.error],device=d)
        def body():
            self.pending.zero_()
            wp.launch(_color_pass,n,[g.rows,g.adjacency,g.colors,self.temporary,self.pending,contacts.error],device=d)
            wp.launch(_color_pass,n,[g.rows,g.adjacency,self.temporary,g.colors,self.pending,contacts.error],device=d)
            wp.launch(_next_loop,1,[self.pending,self.condition,self.iterations,n,contacts.error],device=d)
        wp.capture_while(self.condition,body)
        wp.launch(_publish_colors,n,[g.colors,g.trial_order,g.metadata,contacts.error],device=d)


wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})
