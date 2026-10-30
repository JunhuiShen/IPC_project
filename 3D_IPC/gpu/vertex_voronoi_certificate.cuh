#pragma once
// Read-only proof/coverage helper. No production ordinary or exact path calls it.
#include "ccd.cuh"
#if defined(__CUDACC__)
#define IPC_VORONOI_HD __host__ __device__
#else
#define IPC_VORONOI_HD
#endif
namespace ipc_gpu::solver::voronoi_study {
namespace interval=ipc_gpu::ccd::detail;
struct Certificate {
    interval::Interval products[3][2];
    unsigned closed_mask=0,strict_mask=0;
    bool finite_coordinates=true;
};
IPC_VORONOI_HD inline Certificate certify(const double positions[4][3]) {
    Certificate out;
    for(int j=0;j<3;++j)for(int edge=0;edge<2;++edge)out.products[j][edge]=interval::whole();
    for(int i=0;i<4;++i)for(int axis=0;axis<3;++axis)
        if(!interval::finite(positions[i][axis]))out.finite_coordinates=false;
    if(!out.finite_coordinates)return out;
    for(int vertex=1;vertex<4;++vertex) {
        bool closed=true,strict=true;int edge=0;
        for(int other=1;other<4;++other)if(other!=vertex) {
            auto dot=interval::point(0.0);
            for(int axis=0;axis<3;++axis) {
                const auto residual=interval::subtract(interval::point(positions[0][axis]),interval::point(positions[vertex][axis]));
                const auto direction=interval::subtract(interval::point(positions[other][axis]),interval::point(positions[vertex][axis]));
                dot=interval::add(dot,interval::multiply(residual,direction));
            }
            out.products[vertex-1][edge++]=dot;
            const bool bounded=interval::finite(dot.lo)&&interval::finite(dot.hi)&&dot.lo<=dot.hi;
            closed=closed&&bounded&&dot.hi<=0.0;
            strict=strict&&bounded&&dot.hi<0.0;
        }
        if(closed)out.closed_mask|=1u<<vertex;
        if(strict)out.strict_mask|=1u<<vertex;
    }
    return out;
}
} // namespace ipc_gpu::solver::voronoi_study
#undef IPC_VORONOI_HD
