#pragma once
#include "linear_pipeline_launch.h"
#include "linear_pipeline_constants.h"
#include "ccd.cuh"
#include <cfloat>

namespace ipc_gpu::solver {
namespace {
namespace cert=ipc_gpu::ccd;
using namespace linear_detail;

__device__ void fail(int*error,LinearClipError code) { atomicCAS(error,0,int(code)); }

__device__ bool finite(double v) { return v-v==0.0; }

__device__ bool inside(const double*p,const BpAabb&box) {
    for(int k=0;k<3;++k)
        if(!finite(p[k])||!finite(box.lo[k])||!finite(box.hi[k])||p[k]<box.lo[k]||p[k]>box.hi[k])return false;
    return true;
}

__device__ double pair_floor(const double p[4][3]) {
    double scale=0.;for(int i=0;i<4;++i)for(int k=0;k<3;++k)scale=fmax(scale,fabs(p[i][k]));
    return fmax(1e-8,128.0*DBL_EPSILON*scale);
}

__device__ bool box_fraction(const double*position,const double*delta,
    const BpAabb&box,double&fraction,int*error) {
    if(!inside(position,box)) { fail(error,LinearClipError::OutsideBlueBox);return false; }
    for(int k=0;k<3;++k) {
        if(!finite(delta[k])) { fail(error,LinearClipError::InvalidInput);return false; }
        if(delta[k]>0.)fraction=fmin(fraction,(box.hi[k]-position[k])/delta[k]);
        if(delta[k]<0.)fraction=fmin(fraction,(box.lo[k]-position[k])/delta[k]);
    }
    fraction=fmax(0.,fmin(1.,fraction));return true;
}

__device__ cert::LinearQuery node_query(const BpContact&contact,const double(*positions)[3],
    int node,const double*endpoint) {
    cert::LinearQuery q;q.vertex_face=contact.kind==0;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        q.start[i][k]=positions[contact.vertices[i]][k];
        q.end[i][k]=contact.vertices[i]==node?endpoint[k]:q.start[i][k];
    }
    q.gap_floor=pair_floor(q.start);return q;
}

__device__ cert::LinearQuery body_query(const BpContact&contact,RigidScene scene,
    DeformableState vertices,int body,const double(*accepted)[3]) {
    cert::LinearQuery q;q.vertex_face=contact.kind==0;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        const int node=contact.vertices[i];q.start[i][k]=vertices.positions[node][k];
        q.end[i][k]=scene.node_body[node]==body?accepted[node][k]:q.start[i][k];
    }
    q.gap_floor=pair_floor(q.start);return q;
}
} // namespace
} // namespace ipc_gpu::solver
