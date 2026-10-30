#pragma once
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)
#define IPC_LEVEL_HD __host__ __device__
#else
#define IPC_LEVEL_HD
#endif

namespace ipc_gpu::solver::level_predictor {

// CUDA exposes the device overload globally; standard host C++ guarantees the
// classification function in std. Keep identical finite/nonfinite decisions
// without depending on a host library's optional global C-name exports.
IPC_LEVEL_HD inline bool finite_value(double value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

// This component schedules an already validated, immutable predictor epoch.
// It neither accepts a CCD step nor writes any physical solver state.
enum class Status : int { Ok=0, InvalidInput=71, Capacity=72, Nonfinite=73 };
struct Shape { int colors=0, groups=0, active=0; };
IPC_LEVEL_HD inline int group_of(int index,int groups,const int* offsets) {
    int lo=0,hi=groups;
    while(lo+1<hi){int mid=lo+(hi-lo)/2;if(offsets[mid]<=index)lo=mid;else hi=mid;}
    return lo;
}
struct Scratch {
    int* colors; int* seen; int* levels; int* component; int* queue;
    int* trial_order; int* order; int* offsets; int* level_offsets;
    std::size_t vertices;
};
IPC_LEVEL_HD inline double add_rn(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a,b);
#else
    return a+b;
#endif
}
IPC_LEVEL_HD inline double mul_rn(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a,b);
#else
    return a*b;
#endif
}
IPC_LEVEL_HD inline double div_rn(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __ddiv_rn(a,b);
#else
    return a/b;
#endif
}
IPC_LEVEL_HD inline double sub_rn(double a,double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a,-b);
#else
    return a-b;
#endif
}
IPC_LEVEL_HD inline double sqrt_rn(double a) {
#if defined(__CUDA_ARCH__)
    return __dsqrt_rn(a);
#else
    return ::sqrt(a);
#endif
}
IPC_LEVEL_HD inline double fma_rn(double a,double b,double c) {
#if defined(__CUDA_ARCH__)
    return __fma_rn(a,b,c);
#else
    return ::fma(a,b,c);
#endif
}
// Eigen's stableNorm algorithm for one three-component block: scale first,
// saturate inverse scale for subnormals, then scaled squared norm. The observed
// native CPU scalar path rounds y*y first and uses two FMAs. Preserve that
// order explicitly: exact movement ties select BFS seeds. This primitive must
// pass the native Eigen snapshot oracle; global CUDA contraction stays off.
IPC_LEVEL_HD inline double stable_norm3(const double d[3]) {
    double scale=0;
    for(int a=0;a<3;++a) {
        if(!finite_value(d[a]))return INFINITY;
        if(::fabs(d[a])>scale)scale=::fabs(d[a]);
    }
    if(scale==0)return 0;
    double inv=div_rn(1,scale);
    if(inv>DBL_MAX) { inv=DBL_MAX;scale=div_rn(1,inv); }
    const double x=mul_rn(d[0],inv),y=mul_rn(d[1],inv),z=mul_rn(d[2],inv);
    const double sum=fma_rn(z,z,fma_rn(x,x,mul_rn(y,y)));
    return mul_rn(scale,sqrt_rn(sum));
}
// CPU colored-predictor whole-sweep stopping tests (initial_guess.cpp). Every
// vertex contributes bit 0 when its STORED position moved during the sweep and
// bit 1 when its remaining target squared error is not below 1e-28. The sweep
// only decides after all colors have joined, and it stops when either bit is
// clear: no stored movement at all, or every vertex within the target
// tolerance even when a vertex is not exactly at its target. The same helper
// is used by the device status kernels and by host controls.
IPC_LEVEL_HD inline unsigned predictor_sweep_bits(bool moved,double remaining_squared) {
    unsigned bits=0;
    if(moved)bits|=1u;
    if(!(remaining_squared<1e-28))bits|=2u;
    return bits;
}
IPC_LEVEL_HD inline bool predictor_sweep_finished(unsigned bits) {
    return !(bits&1u)||!(bits&2u);
}
IPC_LEVEL_HD inline bool enough(int n,const Scratch& s) {
    return n>=0&&s.vertices>=static_cast<std::size_t>(n)
        &&s.offsets&&s.level_offsets&&(!n||(s.colors&&s.seen&&s.levels
        &&s.component&&s.queue&&s.trial_order&&s.order));
}
// Input CSR is symmetric, sorted, unique, and contains vertex IDs (NOT rigid
// body/update-block IDs), built only from the retained contact cliques. It
// includes fixed rigid proxies. Caller validates symmetry before this call.
IPC_LEVEL_HD inline Status color(int n,const std::size_t* row,const int* adj,
    Scratch s,Shape& out) {
    out={};
    if(!enough(n,s)||!row||row[0]!=0||(!adj&&row[n]))return Status::InvalidInput;
    for(int v=0;v<n;++v)s.seen[v]=-1;
    for(int v=0;v<n;++v) {
        if(row[v]>row[v+1])return Status::InvalidInput;
        int prior=-1;
        for(auto j=row[v];j<row[v+1];++j) {
            const int nb=adj[j];
            if(nb<0||nb>=n||nb==v||nb<=prior)return Status::InvalidInput;
            prior=nb;
            if(nb<v)s.seen[s.colors[nb]]=v;
        }
        int c=0;while(c<n&&s.seen[c]==v)++c;
        if(c==n)return Status::InvalidInput;
        s.colors[v]=c;
        if(c+1>out.colors)out.colors=c+1;
    }
    for(int v=0;v<n;++v)s.trial_order[v]=v;
    return Status::Ok;
}
// The GPU wrapper sorts trial_order with this total order. It is identical to
// ascending CPU color groups, optionally heavy-first within each color. The
// counts are incidence counts BEFORE adjacency deduplication.
struct TrialLess {
    const int* colors;const std::uint64_t* incidence_count;
    const unsigned char* deformable;bool heavy_first;
    IPC_LEVEL_HD bool operator()(int a,int b)const {
        if(colors[a]!=colors[b])return colors[a]<colors[b];
        if(heavy_first) {
            const auto ca=deformable[a]?incidence_count[a]:0;
            const auto cb=deformable[b]?incidence_count[b]:0;
            if(ca!=cb)return ca>cb;
        }
        return a<b;
    }
};
// Same trial snapshot, same pruned contacts, same node classification are
// required by the CPU scheduler oracle. Every exact maximum tie is seeded,
// separately per component, including zero-movement components. Rigid-only
// components get level zero and produce no scheduled updates.
IPC_LEVEL_HD inline Status schedule(int n,const std::size_t* row,const int* adj,
    const unsigned char* deformable,const double* movement,Scratch s,Shape& out) {
    if(!enough(n,s)||!row||(!deformable&&n)||(!movement&&n))return Status::InvalidInput;
    for(int v=0;v<n;++v) {
        if(deformable[v]>1)return Status::InvalidInput;
        if(!finite_value(movement[v])||movement[v]<0)return Status::Nonfinite;
        s.levels[v]=-2;s.seen[v]=0;
    }
    // Validate the externally sorted trial order, avoiding uninitialized color
    // reads or duplicate vertices even if a caller violates its epoch contract.
    for(int i=0;i<n;++i) {
        int v=s.trial_order[i];
        if(v<0||v>=n||s.seen[v]++||s.colors[v]<0||s.colors[v]>=n)return Status::InvalidInput;
    }
    for(int start=0;start<n;++start) {
        if(s.levels[start]!=-2)continue;
        int count=1;s.component[0]=start;s.levels[start]=-1;
        for(int head=0;head<count;++head) {
            int v=s.component[head];
            for(auto j=row[v];j<row[v+1];++j) {
                int nb=adj[j];
                if(s.levels[nb]!=-2)continue;
                s.levels[nb]=-1;s.component[count++]=nb;
            }
        }
        double largest=-1;
        for(int i=0;i<count;++i) {
            int v=s.component[i];
            if(deformable[v]&&movement[v]>largest)largest=movement[v];
        }
        int tail=0;
        for(int i=0;i<count;++i) {
            int v=s.component[i];
            if(deformable[v]&&movement[v]==largest) {s.levels[v]=0;s.queue[tail++]=v;}
        }
        if(!tail) {for(int i=0;i<count;++i)s.levels[s.component[i]]=0;continue;}
        for(int head=0;head<tail;++head) {
            int v=s.queue[head];
            for(auto j=row[v];j<row[v+1];++j) {
                int nb=adj[j];
                if(s.levels[nb]!=-1)continue;
                s.levels[nb]=s.levels[v]+1;s.queue[tail++]=nb;
            }
        }
    }
    for(int i=0;i<=n;++i)s.level_offsets[i]=0;
    out.active=0;out.groups=0;
    for(int i=0;i<n;++i) {
        int v=s.trial_order[i];
        if(deformable[v]) {++s.level_offsets[s.levels[v]+1];++out.active;}
    }
    for(int i=1;i<=n;++i)s.level_offsets[i]+=s.level_offsets[i-1];
    for(int i=0;i<n;++i) {
        int v=s.trial_order[i];
        if(deformable[v])s.order[s.level_offsets[s.levels[v]]++]=v;
    }
    s.offsets[0]=0;
    for(int i=1;i<out.active;++i) {
        int a=s.order[i-1],b=s.order[i];
        if(s.levels[a]!=s.levels[b]||s.colors[a]!=s.colors[b])s.offsets[++out.groups]=i;
    }
    if(out.active)s.offsets[++out.groups]=out.active;
    return Status::Ok;
}
} // namespace ipc_gpu::solver::level_predictor

namespace ipc_gpu::solver::cpu_contact_schedule {
// Shared host/device implementation of the CPU scheduling policy. Geometry and
// physical state are absent: order uses RAW costs, the 512 gate uses RETAINED
// cost of the raw-first vertex, and ties use canonical vertex IDs.
IPC_LEVEL_HD inline bool selected(int block,int block_count,
    const int* group,int first,int last,const std::size_t* incidence_offsets,
    const std::size_t* raw_counts,const int* block_nodes,int team) {
    if(team<2||block<0||block>=block_count||first<0||last<first
        ||last>block_count||!group||!incidence_offsets)return false;
    const int size=last-first;
    const int half=team/2;
    const int threshold=half<2?2:(half>32?32:half);
    const int scaled=team*24/64;
    const int heavy_count=scaled<1?1:(scaled>24?24:scaled);
    const int heavy_budget=48*team/64;
    const bool small=size<threshold;
    const int selected=small?size:(heavy_count<size?heavy_count:size);
    const int budget=small?team:(heavy_budget<team-1?heavy_budget:team-1);
    if(selected<=0||budget<selected)return false;
    const auto raw_cost=[&](int id) {
        return static_cast<int>(raw_counts?raw_counts[id]
            :incidence_offsets[id+1]-incidence_offsets[id]);
    };
    const auto canonical=[&](int id) {
        const int vertex=block_nodes?block_nodes[id]:id;
        return vertex>=0?vertex:id;
    };
    int first_raw=-1,maximum=-1;
    for(int i=first;i<last;++i) {
        const int other=group[i];
        if(other<0||other>=block_count)return false;
        const int count=raw_cost(other);
        if(first_raw<0||count>maximum
            ||(count==maximum&&canonical(other)<canonical(first_raw))) {
            first_raw=other;maximum=count;
        }
    }
    // CPU sorts before pruning; prepare then tests only retained group[0].
    if(!small&&(first_raw<0
        ||incidence_offsets[first_raw+1]-incidence_offsets[first_raw]<512))return false;
    const int mine=raw_cost(block);
    int rank=0;
    for(int i=first;i<last;++i) {
        const int other=group[i];
        if(other==block)continue;
        const int count=raw_cost(other);
        if(count>mine||(count==mine&&canonical(other)<canonical(block)))++rank;
    }
    return rank<selected;
}
} // namespace ipc_gpu::solver::cpu_contact_schedule

#undef IPC_LEVEL_HD
