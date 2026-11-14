"""Replay actual unchanged resolve kernels, including their worker-private pool."""
import numpy as np
import warp as wp
from newton._src.geometry.tri_mesh_collision import TriMeshCollisionInfo
from ..ccd import resolve_original as original_resolve
from ..ccd import resolve_cached as cached_resolve


class SingleQueryResolver:
    def __init__(self,device):
        self.device=wp.get_device(device)
        self.q=wp.empty(4,wp.vec3,device=self.device);self.end=wp.empty_like(self.q)
        self.ids=wp.empty(1,int,device=self.device)
        self.triangles=wp.array([[1,2,3]],int,device=self.device)
        self.edges=wp.array([[-1,-1,0,1],[-1,-1,2,3]],int,device=self.device)
        self.info=TriMeshCollisionInfo()
        self.info.vertex_colliding_triangles=wp.array([0,0],int,device=self.device)
        self.info.edge_colliding_edges=wp.array([0,1],int,device=self.device)
        self.words=wp.empty(2048*512 if self.device.is_cuda else 0,wp.uint32,device=self.device)
        self.slots=wp.empty(512 if self.device.is_cuda else 0,int,device=self.device)
        self.pending=wp.zeros(1,int,device=self.device);self.count=wp.ones(1,int,device=self.device)
        self.fraction=wp.ones(1,wp.float64,device=self.device);self.error=wp.zeros(1,int,device=self.device)

    def upload(self,case):
        self.q.assign(case['start']);self.end.assign(case['end']);self.ids.fill_(0 if case['kind']==0 else -1)
        self.exact=int(case['exact_geometry']);self.gap=int(case['preserve_gap'])

    def launch(self,*,cached):
        self.fraction.fill_(1.0);self.error.zero_()
        kernel=cached_resolve if cached else original_resolve
        wp.launch(kernel,1,[self.q,self.end,self.ids,self.triangles,self.edges,self.info,
            self.exact,self.gap,self.words,self.slots,1,
            self.pending,self.count,self.fraction,self.error],device=self.device,block_dim=1)

    def packet(self):
        fraction=float(self.fraction.numpy()[0]);error=int(self.error.numpy()[0]);status=error-200 if error else 0
        # For one queued query the reducer is lossless: on success it writes
        # precisely PairResult[0]; every failure retains1 and maps status+200.
        # The original clip source sets full iff status==0 and fraction==1.
        return np.array([fraction,float(status),float(status==0 and fraction==1.0)],dtype=np.float64)

    def evaluate(self,case,*,cached):
        self.upload(case);self.launch(cached=cached)
        return self.packet()


class FixedBatchResolver:
    """Reindex frozen queries without changing any decoded geometry/input bits."""
    def __init__(self,cases,device,*,workers=16):
        self.device=wp.get_device(device);self.cases=cases;self.workers=workers
        if workers<1 or not cases:raise ValueError('Nonempty queries and positive worker count required')
        n=len(cases);ar=lambda values,dtype:wp.array(values,dtype,device=self.device)
        self.q=ar(np.concatenate([x['start'] for x in cases]),wp.vec3)
        self.end=ar(np.concatenate([x['end'] for x in cases]),wp.vec3)
        self.ids=ar([i if x['kind']==0 else -i-1 for i,x in enumerate(cases)],int)
        self.triangles=ar([[4*i+1,4*i+2,4*i+3] for i in range(n)],int)
        self.edges=ar([edge for i in range(n) for edge in [[-1,-1,4*i,4*i+1],[-1,-1,4*i+2,4*i+3]]],int)
        self.info=TriMeshCollisionInfo()
        self.info.vertex_colliding_triangles=ar([v for i in range(n) for v in [4*i,i]],int)
        self.info.edge_colliding_edges=ar([v for i in range(n) for v in [2*i,2*i+1]],int)
        self.words=wp.empty(workers*2048*512 if self.device.is_cuda else 0,wp.uint32,device=self.device)
        self.slots=wp.empty(workers*512 if self.device.is_cuda else 0,int,device=self.device)
        self.pending=wp.empty(n,int,device=self.device);self.count=wp.zeros(1,int,device=self.device)
        self.fraction=wp.ones(1,wp.float64,device=self.device);self.error=wp.zeros(1,int,device=self.device)

    def select(self,first,last):
        batch=self.cases[first:last]
        if not batch:raise ValueError('Empty batch')
        modes={(x['exact_geometry'],x['preserve_gap']) for x in batch}
        if len(modes)!=1:raise ValueError('One original resolve invocation has one exact/gap mode')
        self.exact,self.gap=next(iter(modes))
        self.pending.zero_()
        # Upload outside all timing/capture. Trailing slots are never consumed.
        values=np.zeros(len(self.cases),np.int32);values[:last-first]=np.arange(first,last,dtype=np.int32)
        self.pending.assign(values);self.count.fill_(last-first)

    def launch(self,*,cached):
        self.fraction.fill_(1.0);self.error.zero_()
        kernel=cached_resolve if cached else original_resolve
        wp.launch(kernel,self.workers,[self.q,self.end,self.ids,self.triangles,self.edges,self.info,
            self.exact,self.gap,self.words,self.slots,self.workers,self.pending,self.count,
            self.fraction,self.error],device=self.device,block_dim=1)

    def result(self):
        return float(self.fraction.numpy()[0]),int(self.error.numpy()[0])
