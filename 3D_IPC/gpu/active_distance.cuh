#pragma once
#include "strict_vertex_selector.cuh"
#if defined(__CUDACC__)
#define IPC_POINT_ACTIVE_HD __host__ __device__
#else
#define IPC_POINT_ACTIVE_HD
#endif
namespace ipc_gpu::resident::strict_active_distance_study {
namespace interval=ipc_gpu::resident_ccd::detail;
struct Certificate {interval::Interval squared,threshold;bool proven=false;};
// Requires a separately certified strict vertex before interpreting proven as
// primitive contact activity. Alone this proves only point-to-vertex distance.
IPC_POINT_ACTIVE_HD inline Certificate certify(const double positions[4][3],int vertex,double d_hat) {
 Certificate out{interval::whole(),interval::whole(),false};
 if(vertex<1||vertex>3||!(d_hat>0)||!interval::finite(d_hat))return out;
 for(int i=0;i<4;++i)for(int k=0;k<3;++k)if(!interval::finite(positions[i][k]))return out;
 auto squared=interval::point(0.0);
 for(int axis=0;axis<3;++axis) {
  const auto residual=interval::subtract(interval::point(positions[0][axis]),interval::point(positions[vertex][axis]));
  squared=interval::add(squared,interval::multiply(residual,residual));
 }
 out.squared=squared;out.threshold=interval::multiply(interval::point(d_hat),interval::point(d_hat));
 const bool bounded=interval::finite(out.squared.lo)&&interval::finite(out.squared.hi)
  &&interval::finite(out.threshold.lo)&&interval::finite(out.threshold.hi)
  &&out.squared.lo<=out.squared.hi&&out.threshold.lo<=out.threshold.hi;
 out.proven=bounded&&out.squared.hi>=0.0&&out.squared.hi<out.threshold.lo;
 return out;
}
} // namespace ipc_gpu::resident::strict_active_distance_study
#undef IPC_POINT_ACTIVE_HD
