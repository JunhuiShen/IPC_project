#pragma once
// Isolated ClassA mathematical prototype. No production kernel includes this.
#include "resident_contact_cooperative.cuh"
#include "guard103_prefix.cuh"
#include "vertex_voronoi_certificate.cuh"
#if defined(IPC_GPU_TU_ACTIVE_POINT)
#include "strict_vertex_selector.cuh"
#include "active_distance.cuh"
#endif
static_assert(__cplusplus>=201703L,"Compact-point pool proof requires C++17");
#if defined(__CUDACC__)
#define IPC_COMPACT_POINT_HD __host__ __device__
#else
#define IPC_COMPACT_POINT_HD
#endif
namespace ipc_gpu::resident::point_vertex_compact_prefix {
namespace ex=resident_exact;
namespace cp=contact_cooperative_detail;
constexpr int compact_capacity=8;
enum class Reason:unsigned { Eligible,UnsupportedMaskOrKind,NotOriginalGuard103,
    NotStrictVertex,InvalidDyadic,OriginalCapacityUnproved,CompactCapacityUnproved };
struct Gate {
    Reason reason=Reason::UnsupportedMaskOrKind;
    int role=-1,vertex=-1,unit=0,coordinate_bits=0;
    int original_working_bits=0,compact_working_bits=0;
    const ContactInput* input=nullptr;
    IPC_COMPACT_POINT_HD bool eligible() const{return reason==Reason::Eligible;}
};
IPC_COMPACT_POINT_HD inline int magnitude_bits(std::uint64_t m){
    if(!m)return 0; // Never invoke host CLZ on zero.
#if defined(__CUDA_ARCH__)
    return 64-__clzll(static_cast<unsigned long long>(m));
#elif defined(__GNUC__) || defined(__clang__)
    return 64-__builtin_clzll(static_cast<unsigned long long>(m));
#else
    int bits=0;while(m){m>>=1;++bits;}return bits;
#endif
}
// No Integer construction or current_pool lookup occurs during this gate.
// This separately source-restored prefix preserves the original final103
// predicate; no ordinary Hessian, output or friction evaluation is required.
IPC_COMPACT_POINT_HD inline Gate classify(const ContactInput& in,int role){
    Gate gate;gate.input=&in;gate.role=role;
    if(in.kind!=ContactKind::PointTriangle||role<0||role>=4||cp::self_block(in.hessian_blocks)!=role)return gate;
    gate.reason=Reason::NotOriginalGuard103;
#if defined(IPC_GPU_TU_ACTIVE_POINT)
    if(!point_vertex_guard103::eligible_before_inactive(in))return gate;
    const int selected=strict_vertex_selector_study::select(in.positions);
    const auto active=strict_active_distance_study::certify(in.positions,selected,in.d_hat);
    // Preserve original inactivity priority; the selected feature remains local.
    if(!active.proven&&certified_contact_inactive(in))return gate;
    gate.reason=Reason::NotStrictVertex;
    if(selected<1)return gate;
    gate.vertex=selected;
#else
    if(!point_vertex_guard103::eligible(in))return gate;
    const auto certificate=voronoi_study::certify(in.positions);const unsigned mask=certificate.strict_mask;
    gate.reason=Reason::NotStrictVertex;
    if(!mask||(mask&(mask-1)))return gate;
    for(int vertex=1;vertex<4;++vertex)if(mask==(1u<<vertex))gate.vertex=vertex;
    if(gate.vertex<1)return gate;
#endif
    bool have_unit=false;int maximum_top=0;
    gate.reason=Reason::InvalidDyadic;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k){
        const auto value=ex::decompose(in.positions[i][k]);
        if(!value.valid)return gate;
        if(value.sign){
            const int top=magnitude_bits(value.mantissa)+value.exponent;
            if(!have_unit||value.exponent<gate.unit)gate.unit=value.exponent;
            if(!have_unit||top>maximum_top)maximum_top=top;
            have_unit=true;
        }
    }
    if(!have_unit)return gate;
    gate.coordinate_bits=maximum_top-gate.unit;
    gate.original_working_bits=10*gate.coordinate_bits+22+54;
    gate.compact_working_bits=2*gate.coordinate_bits+4+54;
    gate.reason=Reason::OriginalCapacityUnproved;
    if(gate.original_working_bits>32*cp::capacity)return gate;
    gate.reason=Reason::CompactCapacityUnproved;
    if(gate.compact_working_bits>32*compact_capacity)return gate;
    gate.reason=Reason::Eligible;return gate;
}

// Every active lane must agree BEFORE generic original preparation is skipped.
// No pool lookup on inactive/uninitialized lanes; use this on explicit valid
// original pool pointers. Host metadata checks do not model CUDA allocation.
IPC_COMPACT_POINT_HD inline bool pool_admits(const ex::DevicePool* pool,bool owner){
    const int slots=owner?cp::owner_slots:cp::helper_slots;
    const int required=owner?256:128;
    return pool&&!pool->failed&&pool->words&&pool->free_slots
        &&pool->words_per_slot==cp::capacity&&pool->slot_count==slots
        &&pool->available>=required&&pool->available<=slots;
}

// REQUIRE a fresh GeometryOwner<8>, immutable same input/role as the gate, and
// agreed healthy original pool headroom before its placement construction.
// All unsuccessful preparation/entries/finish must run original256/1600 logic
// after complete owner/adapter destruction. Never publish a partial output.
template<bool ParallelConversions,bool InplaceRatio>
IPC_COMPACT_POINT_HD inline bool prepare(const ContactInput& in,int role,const Gate& gate,
    cp::GeometryOwner<compact_capacity>& owner,ContactOutput& out,cp::SharedGeometry& shared){
    using I=ex::Integer<compact_capacity>;
    if(!gate.eligible()||gate.input!=&in||gate.role!=role||gate.vertex<1||gate.vertex>3
        ||cp::self_block(in.hessian_blocks)!=role||owner.feature.have||owner.feature.free_parameters!=0)return false;
    ex::Arithmetic<compact_capacity> arithmetic;out=ContactOutput{};owner.unit=gate.unit;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k){
        owner.points.p[i].v[k]=ex::from_double<compact_capacity>(in.positions[i][k],gate.unit);
        arithmetic.ok=arithmetic.ok&&owner.points.p[i].v[k].valid;
    }
    if(!arithmetic.ok){out.status=ContactStatus::ExactCapacityExceeded;return false;}
    auto& feature=owner.feature;
    feature.weight[0]=I(1);feature.weight[gate.vertex]=I(-1);
    contact_exact_detail::finish_feature(arithmetic,owner.points,feature);
    if(!arithmetic.ok||!feature.have){out.status=ContactStatus::ExactCapacityExceeded;return false;}
    if(!arithmetic.sign(feature.squared_residual)){out.status=ContactStatus::ZeroDistance;return false;}
    bool okay=true;
    if constexpr(ParallelConversions){
        // Portable shadow of the original seven independent conversion lanes;
        // a production kernel must publish the same owned views before helpers.
        cp::SharedConversions conversions{};cp::publish_conversions(owner,conversions);
        for(int component=0;component<7;++component){
            const auto result=cp::convert_geometry_component<compact_capacity,InplaceRatio>(conversions,component);
            conversions.values[component]=result.value;okay=okay&&result.ok;
        }
        for(int k=0;k<3;++k)owner.residual[k]=conversions.values[k];
        for(int i=0;i<4;++i)out.weights[i]=conversions.values[3+i];
    }else{
        for(int k=0;k<3;++k)owner.residual[k]=cp::contact_ratio<compact_capacity,InplaceRatio>(feature.residual[k],feature.denominator,owner.unit,&okay);
        for(int i=0;i<4;++i)out.weights[i]=cp::contact_ratio<compact_capacity,InplaceRatio>(feature.weight[i],feature.denominator,0,&okay);
    }
    if(!okay){out.status=ContactStatus::ExactCapacityExceeded;return false;}
    return cp::prepare_derivatives(in,role,owner,out,shared);
}
} // namespace ipc_gpu::resident::point_vertex_compact_prefix
#undef IPC_COMPACT_POINT_HD
