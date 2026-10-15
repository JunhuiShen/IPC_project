#pragma once
#include "vertex_voronoi_certificate.cuh"
#if defined(__CUDACC__)
#define IPC_STRICT_SELECTOR_HD __host__ __device__
#else
#define IPC_STRICT_SELECTOR_HD
#endif
namespace ipc_gpu::resident::strict_vertex_selector_study {
namespace interval=ipc_gpu::resident_ccd::detail;
// Only the strict vertex is requested. This does not replace callers needing
// every closed-mask/dot enclosure in the original Certificate descriptor.
// Any accepted vertex has the same two original interval dot computations.
IPC_STRICT_SELECTOR_HD inline int select(const double positions[4][3],unsigned* diagnostic_dots=nullptr) {
 if(diagnostic_dots)*diagnostic_dots=0;
 for(int i=0;i<4;++i)for(int axis=0;axis<3;++axis)
  if(!interval::finite(positions[i][axis]))return -1;
 for(int vertex=1;vertex<4;++vertex) {
  bool strict=true;
  for(int other=1;other<4;++other)if(other!=vertex) {
   auto dot=interval::point(0.0);
   for(int axis=0;axis<3;++axis) {
    const auto residual=interval::subtract(interval::point(positions[0][axis]),interval::point(positions[vertex][axis]));
    const auto direction=interval::subtract(interval::point(positions[other][axis]),interval::point(positions[vertex][axis]));
    dot=interval::add(dot,interval::multiply(residual,direction));
   }
   if(diagnostic_dots)++*diagnostic_dots;
   const bool bounded=interval::finite(dot.lo)&&interval::finite(dot.hi)&&dot.lo<=dot.hi;
   if(!(bounded&&dot.hi<0.0)){strict=false;break;}
  }
  if(strict)return vertex;
 }
 return -1;
}
} // namespace ipc_gpu::resident::strict_vertex_selector_study
#undef IPC_STRICT_SELECTOR_HD
