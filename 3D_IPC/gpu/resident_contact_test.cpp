#include "resident_contact.cuh"
#include "resident_contact_exact.cuh"
#include "barrier_energy.h"
#include "friction_energy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef IPC_RESIDENT_CONTACT_CUDA_TEST
void resident_contact_compute_gpu(const ipc_gpu::resident::ContactInput*,
    ipc_gpu::resident::ContactOutput*,std::size_t);
void resident_contact_compute_exact_gpu(const ipc_gpu::resident::ContactInput*,
    ipc_gpu::resident::ContactOutput*,std::size_t);
#endif

using ipc_gpu::resident::ContactInput;
using ipc_gpu::resident::ContactOutput;
using ipc_gpu::resident::ContactKind;
using ipc_gpu::resident::ContactStatus;

namespace {
std::size_t components=0;
double worst_scaled_error=0;
std::string worst_label;
void require(bool ok,const std::string& message) {if(!ok)throw std::runtime_error(message);}
void compare(double actual,double expected,const std::string& label) {
    require(std::isfinite(actual)&&std::isfinite(expected),label+": nonfinite");
    const double error=std::abs(actual-expected)/(1+std::abs(expected));
    if(error>worst_scaled_error){worst_scaled_error=error;worst_label=label;}
    require(error<=2e-9,label+": mismatch "+std::to_string(actual)+" vs "+std::to_string(expected));
    ++components;
}
ContactInput make_input(ContactKind kind,const std::array<Vec3,4>& positions,int index) {
    ContactInput in{};
    in.kind=kind;in.d_hat=.8;in.k_barrier=370;in.dt=.031;in.friction=.2;in.eps_v=.1;
    const double slip=index%3==0?0:(index%3==1?1e-6:1);
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        in.positions[i][k]=positions[i][k];
        in.previous[i][k]=positions[i][k]-slip*(i+1)*.001*(k+1);
    }
    return in;
}
std::vector<ContactInput> fixtures() {
    std::vector<ContactInput> records;
    // All PT active feature classes, both orientations, all sixteen Cartesian
    // Hessian blocks (the rigid solver requires off-diagonal blocks too).
    const std::vector<Vec3> points={{.2,.3,.17},{.2,.3,-.17},{.3,-.2,.17},
        {.7,.7,.17},{-.2,.3,.17},{-.2,-.3,.17},{1.3,-.1,.17},{-.1,1.3,.17},
        {.2,.3,1.1}};
    for(const auto& point:points)records.push_back(make_input(ContactKind::PointTriangle,
        {point,Vec3::Zero(),Vec3(1,0,0),Vec3(0,1,0)},records.size()));
    // Segment interiors, four edges, four corners, inactive contact.
    for(const double px:{-.6,.2,1.6})for(const double py:{-.6,.2,1.6}) {
        records.push_back(make_input(ContactKind::EdgeEdge,
            {Vec3(0,0,0),Vec3(1,0,0),Vec3(px,py,.17),Vec3(px,py+1,.17)},records.size()));
    }
    // Rotated/skewed non-axis-aligned features guard against a formula which
    // happens to work only for horizontal faces and orthogonal edges.
    const auto base=records;
    const Mat33 R=Eigen::AngleAxisd(.57,Vec3(.3,-.7,.4).normalized()).toRotationMatrix();
    for(int copy=0;copy<5;++copy)for(auto in:base) {
        const double scale=.7+.23*copy;
        for(int i=0;i<4;++i) {
            Vec3 p(in.positions[i][0],in.positions[i][1],in.positions[i][2]);
            Vec3 prev(in.previous[i][0],in.previous[i][1],in.previous[i][2]);
            p=scale*R*p+Vec3(.5,-.7,.2);prev=scale*R*prev+Vec3(.5,-.7,.2);
            for(int k=0;k<3;++k){in.positions[i][k]=p[k];in.previous[i][k]=prev[k];}
        }
        records.push_back(in);
    }
    return records;
}
void reference_check(const ContactInput& in,const ContactOutput& out,std::size_t index,bool exact=false) {
    const std::string tag="record "+std::to_string(index);
    require(out.status==ContactStatus::Ok,tag+": unexpected failure status "+std::to_string(unsigned(out.status)));
    std::array<Vec3,4> x,previous;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k){x[i][k]=in.positions[i][k];previous[i][k]=in.previous[i][k];}
    std::array<Vec3,4> g;
    std::array<std::array<Mat33,4>,4> h;
    FrozenFrictionContact f;
    std::array<double,4> weights;
    double distance;
    if(in.kind==ContactKind::PointTriangle) {
        const auto eval=make_node_triangle_contact_evaluation(x,in.d_hat,in.k_barrier);
        distance=eval.dr.distance;
        weights=node_triangle_contact_weights(x[0],x[1],x[2],x[3],in.eps,&eval.dr);
        f=make_node_triangle_frozen_friction_contact(x,previous,eval,in.dt,in.eps_v);
        for(int i=0;i<4;++i) {
            g[i]=node_triangle_barrier_gradient(x[0],x[1],x[2],x[3],i,eval);
            for(int j=0;j<4;++j)h[i][j]=node_triangle_barrier_cross_hessian(x[0],x[1],x[2],x[3],i,j,eval);
        }
    } else {
        const auto eval=make_segment_segment_contact_evaluation(x,in.d_hat,in.k_barrier);
        distance=eval.dr.distance;
        require(exact||!eval.dr.robust,tag+": fixture unexpectedly requires exact reference");
        weights=segment_segment_contact_weights(x[0],x[1],x[2],x[3],in.eps,&eval.dr);
        f=make_segment_segment_frozen_friction_contact(x,previous,eval,in.dt,in.eps_v);
        for(int i=0;i<4;++i) {
            g[i]=segment_segment_barrier_gradient(x[0],x[1],x[2],x[3],i,eval);
            for(int j=0;j<4;++j)h[i][j]=segment_segment_barrier_cross_hessian(x[0],x[1],x[2],x[3],i,j,eval);
        }
    }
    compare(out.distance,distance,tag+" distance");
    const auto friction=frozen_friction_relative_gradient_and_hessian(f,in.friction,in.dt*in.dt);
    for(int i=0;i<4;++i) {
        compare(out.weights[i],weights[i],tag+" weight");
        for(int k=0;k<3;++k) {
            compare(out.barrier_gradient[i][k],g[i][k],tag+" gradient");
            for(int j=0;j<4;++j)for(int l=0;l<3;++l)
                compare(out.barrier_hessian[i][j][3*k+l],h[i][j](k,l),tag+" Hessian "+std::to_string(i)+std::to_string(j)+std::to_string(k)+std::to_string(l));
        }
    }
    for(int k=0;k<3;++k) {
        compare(out.friction_relative_gradient[k],friction.first[k],tag+" friction gradient");
        for(int l=0;l<3;++l)compare(out.friction_relative_hessian[3*k+l],friction.second(k,l),tag+" friction Hessian");
    }
}

#ifdef IPC_RESIDENT_CONTACT_CUDA_TEST
std::vector<ContactOutput> evaluate(const std::vector<ContactInput>& in) {
    std::vector<ContactOutput> result(in.size());
    resident_contact_compute_gpu(in.data(),result.data(),in.size());
    return result;
}
#else
std::vector<ContactOutput> evaluate(const std::vector<ContactInput>& in) {
    std::vector<ContactOutput> result;
    for(const auto& record:in)result.push_back(ipc_gpu::resident::evaluate_contact(record));
    return result;
}
#endif

std::vector<ContactOutput> evaluate_exact(const std::vector<ContactInput>& in) {
    std::vector<ContactOutput> result(in.size());
#ifdef IPC_RESIDENT_CONTACT_CUDA_TEST
    resident_contact_compute_exact_gpu(in.data(),result.data(),in.size());
#else
    for(std::size_t i=0;i<in.size();++i) {
        result[i]=ipc_gpu::resident::evaluate_contact_exact<256>(in[i]);
        if(result[i].status==ContactStatus::ExactCapacityExceeded)
            result[i]=ipc_gpu::resident::evaluate_contact_exact<1600>(in[i]);
    }
#endif
    return result;
}
void masked_check(const std::vector<ContactInput>& inputs,bool exact) {
    // Include gradient-only, each self block, PT COM, rigid EE upper blocks,
    // and a triangle-side orientation mask. The device and host wrapper run
    // exactly the same cases; masks must not alter any requested arithmetic.
    constexpr unsigned masks[]={0u,1u,1u<<5,1u<<10,1u<<15,
        (1u<<0)|(1u<<1)|(1u<<5),
        (1u<<5)|(1u<<6)|(1u<<7)|(1u<<10)|(1u<<11)|(1u<<15),0xffffu};
    const auto baseline=exact?evaluate_exact(inputs):evaluate(inputs);
    std::vector<ContactInput> requests;
    for(const auto& in:inputs)for(unsigned mask:masks) {
        requests.push_back(in);requests.back().hessian_blocks=mask;
    }
    const auto outputs=exact?evaluate_exact(requests):evaluate(requests);
    for(std::size_t record=0;record<requests.size();++record) {
        const auto& actual=outputs[record];const auto& expected=baseline[record/8];
        require(actual.status==ContactStatus::Ok&&actual.status==expected.status,"masked contact status");
        require(actual.active==expected.active,"masked contact activity");
        const auto same=[&](double a,double b) {
            require(std::memcmp(&a,&b,sizeof(double))==0,"masked contact changed requested arithmetic");++components;
        };
        same(actual.distance,expected.distance);
        for(int i=0;i<4;++i) {
            same(actual.weights[i],expected.weights[i]);
            for(int k=0;k<3;++k) {
                same(actual.barrier_gradient[i][k],expected.barrier_gradient[i][k]);
                for(int j=0;j<4;++j)for(int l=0;l<3;++l) {
                    if(requests[record].hessian_blocks&(1u<<(4*i+j)))
                        same(actual.barrier_hessian[i][j][3*k+l],expected.barrier_hessian[i][j][3*k+l]);
                    else require(actual.barrier_hessian[i][j][3*k+l]==0,"unused Hessian block must be zero");
                }
            }
        }
        for(int k=0;k<3;++k) {
            same(actual.normal[k],expected.normal[k]);same(actual.friction_relative_gradient[k],expected.friction_relative_gradient[k]);
            for(int l=0;l<3;++l)same(actual.friction_relative_hessian[3*k+l],expected.friction_relative_hessian[3*k+l]);
        }
    }
}
}

int main() {
    try {
        const auto inputs=fixtures();
        const auto outputs=evaluate(inputs);
        for(std::size_t i=0;i<inputs.size();++i)reference_check(inputs[i],outputs[i],i);
        std::vector<ContactInput> failures;
        failures.push_back(make_input(ContactKind::EdgeEdge,{Vec3(0,0,0),Vec3(1,0,0),Vec3(0,.1,0),Vec3(1,.1,0)},0));
        failures.push_back(make_input(ContactKind::PointTriangle,{Vec3(.2,.2,1e-12),Vec3(0,0,0),Vec3(1,0,0),Vec3(0,1,0)},0));
        failures.push_back(make_input(ContactKind::PointTriangle,{Vec3(.2,.2,0),Vec3(0,0,0),Vec3(1,0,0),Vec3(0,1,0)},0));
        failures.push_back(inputs[0]);failures.back().dt=0;
        failures.push_back(make_input(ContactKind::PointTriangle,{Vec3(.2,.2,.1),Vec3(0,0,0),Vec3(1,0,0),Vec3(2,0,0)},0));
        failures.push_back(inputs[0]);failures.back().hessian_blocks=1u<<16;
        const auto failed=evaluate(failures);
        require(failed[0].status==ContactStatus::NeedsExactGeometry,"parallel contact must request exact geometry");
        require(failed[1].status==ContactStatus::NeedsExactGeometry,"tiny gap must request exact geometry");
        require(failed[2].status==ContactStatus::ZeroDistance,"zero gap must fail closed");
        require(failed[3].status==ContactStatus::InvalidInput,"invalid input must fail closed");
        require(failed[4].status==ContactStatus::NeedsExactGeometry,"degenerate triangle must request exact geometry");
        require(failed[5].status==ContactStatus::InvalidInput,"out-of-range Hessian mask must fail closed");
        require(evaluate_exact({failures[5]})[0].status==ContactStatus::InvalidInput,"exact mask validation");
        for(const auto& out:failed)require(!out.active,"failed contact must not advertise valid derivatives");
        std::vector<ContactInput> exact_inputs{failures[0],failures[1],failures[4]};
        exact_inputs.push_back(make_input(ContactKind::EdgeEdge,
            {Vec3(0,0,0),Vec3(1,0,0),Vec3(.2,-.5,1e-16),Vec3(.2,.5,1e-16)},0));
        exact_inputs.push_back(make_input(ContactKind::EdgeEdge,
            {Vec3(0,0,0),Vec3(0,0,0),Vec3(.1,-.5,.17),Vec3(.1,.5,.17)},1));
        exact_inputs.push_back(make_input(ContactKind::EdgeEdge,
            {Vec3(0,0,0),Vec3(1,0,0),Vec3(.1,.1,.17),Vec3(.7,.1+1e-14,.17)},2));
        const auto full_width=make_input(ContactKind::PointTriangle,
            {Vec3(.25,.25,std::ldexp(1.0,-500)),Vec3(0,0,0),Vec3(1,0,0),Vec3(0,1,0)},0);
        require(ipc_gpu::resident::evaluate_contact_exact<256>(full_width).status==ContactStatus::ExactCapacityExceeded,
            "full-width fixture must exercise capacity escalation");
        exact_inputs.push_back(full_width);
        // Repeated robust geometry with different current coefficients and
        // endpoint bits exercises the upstream exact-geometry caching contract.
        exact_inputs.push_back(exact_inputs[0]);
        exact_inputs.back().d_hat=.5;exact_inputs.back().k_barrier=19;
        exact_inputs.back().friction=0;exact_inputs.back().dt=.017;
        exact_inputs.push_back(exact_inputs[0]);
        exact_inputs.back().positions[2][1]=std::nextafter(exact_inputs.back().positions[2][1],1.0);
        const auto exact_outputs=evaluate_exact(exact_inputs);
        for(std::size_t i=0;i<exact_inputs.size();++i)reference_check(exact_inputs[i],exact_outputs[i],1000+i,true);
        const auto ordinary_exact=evaluate_exact(inputs);
        for(std::size_t i=0;i<inputs.size();++i)reference_check(inputs[i],ordinary_exact[i],2000+i,true);
        const auto overflow=ipc_gpu::resident::evaluate_contact_exact<2>(inputs[0]);
        require(overflow.status==ContactStatus::ExactCapacityExceeded,"insufficient exact capacity must be explicit");
        masked_check(std::vector<ContactInput>(inputs.begin(),inputs.begin()+18),false);
        masked_check(exact_inputs,true);
        std::cout<<"Resident contacts: "<<inputs.size()<<" contacts, "<<components
            <<" reference components, maximum scaled error "<<worst_scaled_error
            <<" at "<<worst_label
#ifdef IPC_RESIDENT_CONTACT_CUDA_TEST
            <<" (CUDA device execution)\n";
#else
            <<" (host arithmetic only; not CUDA validation)\n";
#endif
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
