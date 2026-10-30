#pragma once
#include <cmath>
#include <cstdint>
#include <limits>

// Adapted from Tight-Inclusion's ccd.cpp, interval_root_finder.cpp, avx.cpp,
// and interval.cpp. Original license:
// MIT License
// Copyright (c) 2020 Bolun Wang, Zachary Ferguson, Teseo Schneider,
// Xin Jiang, Marco Attene, and Daniele Panozzo
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifdef __CUDACC__
#define IPC_GUESS_HD __host__ __device__
#else
#define IPC_GUESS_HD
#endif

namespace ipc_gpu::solver::global_guess_detail {
constexpr double minimum_separation = 1e-10;
constexpr double distance_tolerance = 1e-6;
constexpr int max_iterations = 1000000;

struct Input { double start[4][3], end[4][3]; bool vertex_face; };
// Every bisection interval is exactly [k/2^d,(k+1)/2^d]. Storing k,d instead
// of both reduced dyadic endpoints is lossless and saves queue memory.
struct Cell { std::uint64_t index[3]; unsigned char depth[3]; int level; };
struct Result { double toi=1; int status=0; bool capacity=false; };
IPC_GUESS_HD inline double maximum(double a,double b){return a>b?a:b;}
IPC_GUESS_HD inline double minimum(double a,double b){return a<b?a:b;}
IPC_GUESS_HD inline double absolute(double a){return a<0?-a:a;}
IPC_GUESS_HD inline bool finite(double a){return a==a&&absolute(a)<=1.7976931348623157e308;}

// Match NumCCD::value and function_vf/ee's numerator-then-denominator order.
IPC_GUESS_HD inline void endpoint(const Cell& c,int axis,int upper,
                                 double& numerator,double& denominator) {
    auto n=c.index[axis]+upper; unsigned d=c.depth[axis];
    if(n==0)d=0;
    else while(d && !(n&1)){n>>=1;--d;}
    numerator=static_cast<double>(n);
    denominator=static_cast<double>(std::uint64_t(1)<<d);
}
IPC_GUESS_HD inline double endpoint_value(const Cell& c,int axis,int upper) {
    double n,d;endpoint(c,axis,upper,n,d);return n/d;
}
IPC_GUESS_HD inline bool time_before(const Cell& a,const Cell& b) {
    const unsigned d=a.depth[0]>b.depth[0]?a.depth[0]:b.depth[0];
    return (a.index[0]<<(d-a.depth[0])) < (b.index[0]<<(d-b.depth[0]));
}
IPC_GUESS_HD inline bool worse(const Cell& a,const Cell& b) {
    return a.level!=b.level ? a.level>b.level : time_before(b,a);
}
IPC_GUESS_HD inline void push(Cell* heap,int& size,const Cell& value) {
    int hole=size++;
    while(hole>0) {
        const int parent=(hole-1)/2;
        if(!worse(heap[parent],value))break;
        heap[hole]=heap[parent];hole=parent;
    }
    heap[hole]=value;
}
// libstdc++ priority_queue adjustment including its right-child tie rule.
IPC_GUESS_HD inline Cell pop(Cell* heap,int& size) {
    Cell first=heap[0],value=heap[--size];
    if(!size)return first;
    int hole=0,child=2;
    while(child<size) {
        if(worse(heap[child],heap[child-1]))--child;
        heap[hole]=heap[child];hole=child;child=2*(hole+1);
    }
    if(child==size){heap[hole]=heap[child-1];hole=child-1;}
    while(hole>0) {
        const int parent=(hole-1)/2;
        if(!worse(heap[parent],value))break;
        heap[hole]=heap[parent];hole=parent;
    }
    heap[hole]=value;return first;
}

IPC_GUESS_HD inline void tolerances(const Input& q,double* tol,double* error) {
    // p[t][u][v], the eight corners of the trilinear inclusion function.
    double p[2][2][2][3];
    for(int t=0;t<2;++t)for(int a=0;a<3;++a) {
        const double (*x)[3]=t?q.end:q.start;
        p[t][0][0][a]=x[0][a]-x[q.vertex_face?1:2][a];
        p[t][0][1][a]=x[0][a]-x[3][a];
        p[t][1][0][a]=q.vertex_face?x[0][a]-x[2][a]:x[1][a]-x[2][a];
        p[t][1][1][a]=q.vertex_face?x[0][a]-(x[2][a]+x[3][a]-x[1][a]):x[1][a]-x[3][a];
    }
    double lengths[3]={0,0,0};
    for(int a=0;a<3;++a)for(int i=0;i<2;++i)for(int j=0;j<2;++j) {
        lengths[0]=maximum(lengths[0],absolute(p[1][i][j][a]-p[0][i][j][a]));
        lengths[1]=maximum(lengths[1],absolute(p[i][1][j][a]-p[i][0][j][a]));
        lengths[2]=maximum(lengths[2],absolute(p[i][j][1][a]-p[i][j][0][a]));
    }
    for(int a=0;a<3;++a) {
        lengths[a]*=3;
        tol[a]=lengths[a]==0?INFINITY:distance_tolerance/lengths[a];
        double m=0;
        for(int v=0;v<4;++v) m=maximum(m,maximum(absolute(q.start[v][a]),absolute(q.end[v][a])));
        const double d=minimum(m,1.0);
        error[a]=(q.vertex_face?7.549516567451064e-15:7.105427357601002e-15)*((d*d)*d);
    }
}

// Tight inclusion evaluates the eight corners in the original operation order
// and expands the image by its analytically derived numerical error filter.
// Build this CUDA translation unit with --fmad=false, as the parent library.
IPC_GUESS_HD inline bool overlaps(const Input& q,const Cell& c,const double* error,
                                 bool& contained,double* widths,int& status) {
    contained=true;
    for(int axis=0;axis<3;++axis) {
        double low=INFINITY,high=-INFINITY;
        for(int corner=0;corner<8;++corner) {
            double tn,td,un,ud,vn,vd;
            endpoint(c,0,(corner>>2)&1,tn,td);
            endpoint(c,1,(corner>>1)&1,un,ud);
            endpoint(c,2,corner&1,vn,vd);
            double x[4];
            for(int v=0;v<4;++v)x[v]=(q.end[v][axis]-q.start[v][axis])*tn/td+q.start[v][axis];
            const double f=q.vertex_face
                ? x[0]-((x[2]-x[1])*un/ud+(x[3]-x[1])*vn/vd+x[1])
                : ((x[1]-x[0])*un/ud+x[0])-((x[3]-x[2])*vn/vd+x[2]);
            if(!finite(f)){status=51;return false;}
            low=minimum(low,f);high=maximum(high,f);
        }
        widths[axis]=high-low;
        if(!finite(widths[axis])){status=51;return false;}
        const double e=error[axis]+minimum_separation;
        if(low>e || high<-e)return false;
        if(low<-e || high>e)contained=false;
    }
    return true;
}

IPC_GUESS_HD inline bool simplex(const Cell& c) {
    const unsigned d=c.depth[1]>c.depth[2]?c.depth[1]:c.depth[2];
    return (c.index[1]<<(d-c.depth[1]))+(c.index[2]<<(d-c.depth[2])) <= (std::uint64_t(1)<<d);
}

IPC_GUESS_HD inline Result solve(const Input& q,Cell* heap,int capacity) {
    Result result;
    for(int v=0;v<4;++v)for(int a=0;a<3;++a)
        if(!finite(q.start[v][a])||!finite(q.end[v][a])){result.status=50;return result;}
    double tol[3],error[3];tolerances(q,tol,error);
    for(int a=0;a<3;++a)if(!(tol[a]>0)||!finite(error[a])){result.status=51;return result;}
    if(capacity<1){result.capacity=true;return result;}
    int size=0,refine=0,current_level=-2,box_in_level=-2;
    bool this_level_less_tol=true,find_level_root=false,use_skip=false;
    Cell skip{};
    double temp_toi=1;
    Cell first{};first.level=-1;push(heap,size,first);
    while(size) {
        const Cell current=pop(heap,size);
        if(use_skip&&!time_before(current,skip))continue;
        if(box_in_level!=current.level){box_in_level=current.level;this_level_less_tol=true;}
        ++refine;
        bool contained;double true_tol[3];
        if(!overlaps(q,current,error,contained,true_tol,result.status)) {
            if(result.status)return result;
            continue;
        }
        double widths[3];bool tol_condition=true,condition1=true;
        for(int a=0;a<3;++a) {
            widths[a]=endpoint_value(current,a,1)-endpoint_value(current,a,0);
            tol_condition=tol_condition&&true_tol[a]<=distance_tolerance;
            condition1=condition1&&widths[a]<=tol[a];
        }
        const bool condition2=contained&&this_level_less_tol;
        if(!tol_condition)this_level_less_tol=false;
        if(condition1||condition2||this_level_less_tol) {
            result.toi=endpoint_value(current,0,0);return result;
        }
        if(current_level!=current.level){current_level=current.level;find_level_root=false;}
        if(!find_level_root){temp_toi=endpoint_value(current,0,0);find_level_root=true;}
        if(refine>max_iterations){result.toi=temp_toi;return result;}
        if(tol_condition||contained) {
            if(!use_skip||time_before(current,skip))skip=current;
            use_skip=true;continue;
        }
        int split=0;double largest=-INFINITY;
        for(int a=0;a<3;++a) {
            const double value=widths[a]>tol[a]?widths[a]/tol[a]:-INFINITY;
            if(value>largest){largest=value;split=a;}
        }
        // Upstream NumCCD uses 64-bit dyadics. Explicit failure replaces its
        // overflow assertion/undefined behavior, never a spurious safe result.
        if(current.depth[split]>=62){result.status=53;return result;}
        Cell halves[2]={current,current};
        for(int side=0;side<2;++side) {
            ++halves[side].depth[split];halves[side].index[split]=2*current.index[split]+side;
            ++halves[side].level;
        }
        // Same second-half then first-half insertion order as Tight-Inclusion.
        for(int side=1;side>=0;--side) {
            if(q.vertex_face&&split!=0&&!simplex(halves[side]))continue;
            if(size==capacity){result.capacity=true;return result;}
            push(heap,size,halves[side]);
        }
    }
    if(use_skip)result.toi=endpoint_value(skip,0,0);
    return result;
}
} // namespace ipc_gpu::solver::global_guess_detail
#undef IPC_GUESS_HD
