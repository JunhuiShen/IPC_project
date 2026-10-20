#pragma once
// Selected ordinary evaluator; callers prove the exact single-self/friction-zero precondition.
#include "resident_contact_compact.cuh"
#ifdef __CUDACC__
#define IPC_SELECTED_HD __host__ __device__
#else
#define IPC_SELECTED_HD
#endif
namespace ipc_gpu::resident::selected_ordinary {
// Preserve all geometry fields and all12 gradients for exact status checks.
// Only the one requested9-entry Hessian is stored; friction fields are +0.
struct Output {
    double distance=0.0,weights[4]{},normal[3]{},barrier_gradient[4][3]{},selected_hessian[9]{};
    ContactStatus status=ContactStatus::Ok;
    unsigned active=0;
};
static_assert(sizeof(Output)==240);
// Precondition enforced by try_store below: role0..3, sole matching self block,
// either signed friction zero. Original input validation is still executed.
IPC_SELECTED_HD inline Output evaluate_eligible(const ContactInput& in,int role) {
    using namespace contact_detail;
    Output out{};
    double coordinate_scale=0;
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        if(!finite(in.positions[i][k])||!finite(in.previous[i][k])) {
            out.status=ContactStatus::InvalidInput;return out;
        }
        if(absval(in.positions[i][k])>coordinate_scale)coordinate_scale=absval(in.positions[i][k]);
    }
    if(!finite(in.d_hat)||!(in.d_hat>0)||!finite(in.k_barrier)||in.k_barrier<0
        ||!finite(in.dt)||!(in.dt>0)||!finite(in.friction)||in.friction<0
        ||!finite(in.eps_v)||!(in.eps_v>0)||!finite(in.eps)||!(in.eps>0)
        ||(in.kind!=ContactKind::PointTriangle&&in.kind!=ContactKind::EdgeEdge)
        ||(in.hessian_blocks&~0xffffu)) {
        out.status=ContactStatus::InvalidInput;return out;
    }
    if(in.k_barrier==0)return out;
    Feature feature{};
    out.status=in.kind==ContactKind::PointTriangle?point_triangle(in,feature,out.normal):edge_edge(in,feature,out.normal);
    if(out.status!=ContactStatus::Ok) {
        if(out.status==ContactStatus::NeedsExactGeometry&&certified_contact_inactive(in)) {
            Output inactive{};inactive.distance=in.d_hat;return inactive;
        }
        return out;
    }
    out.distance=feature.distance;
    for(int i=0;i<4;++i)out.weights[i]=feature.w[i];
    if(!finite(out.distance)){out.status=ContactStatus::NeedsExactGeometry;return out;}
    if(out.distance>=in.d_hat)return out;
    if(out.distance==0){out.status=ContactStatus::ZeroDistance;return out;}
    // The Schur curvature subtraction below is cancellation-sensitive as the
    // gap tends to zero. Reject it instead of dividing roundoff by a tiny gap.
    // This stricter-than-production guard is intentional and is not a gap floor.
    if(out.distance<=sqrt(DBL_EPSILON)*coordinate_scale) {
        out.status=ContactStatus::NeedsExactGeometry;return out;
    }
    const double d=out.distance,s=d-in.d_hat,logratio=log(d/in.d_hat),ratio=in.d_hat/d;
    const double bp=-2*s*logratio-s*s/d,bpp=ratio*ratio+2*ratio-3-2*logratio;
    const int n=feature.free_parameters;
    double inverse[2][2]{};
    if(n==1) inverse[0][0]=1/dot(feature.u[0],feature.u[0]);
    if(n==2) {
        const double A=dot(feature.u[0],feature.u[0]),B=dot(feature.u[0],feature.u[1]),C=dot(feature.u[1],feature.u[1]);
        const double determinant=A*C-B*B;
        if(!(determinant>64*DBL_EPSILON*A*C)) {out.status=ContactStatus::NeedsExactGeometry;return out;}
        inverse[0][0]=C/determinant;inverse[0][1]=inverse[1][0]=-B/determinant;inverse[1][1]=A/determinant;
    }
    // Differentiate q=0.5*||sum_i w_i*x_i||^2 after minimizing over the
    // active feature's free parameters. The Schur complement is the full
    // Hessian of minimized q, NOT frozen-weight/Gauss-Newton differentiation.
    double q_xt[4][3][2]{};
    for(int i=0;i<4;++i)for(int k=0;k<3;++k)for(int a=0;a<n;++a)
        q_xt[i][k][a]=feature.w[i]*feature.u[a][k]+feature.sigma[a][i]*feature.residual[k];
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        const double gi=feature.w[i]*out.normal[k];
        out.barrier_gradient[i][k]=bp*gi;
        for(int j=0;j<4;++j)if(in.hessian_blocks&(1u<<(4*i+j)))for(int l=0;l<3;++l) {
            const double gj=feature.w[j]*out.normal[l];
            double hq=(k==l?feature.w[i]*feature.w[j]:0);
            for(int a=0;a<n;++a)for(int b=0;b<n;++b)hq-=q_xt[i][k][a]*inverse[a][b]*q_xt[j][l][b];
            out.selected_hessian[3*k+l]=bpp*gi*gj+bp*(hq-gi*gj)/d;
        }
    }
    // Friction branch omitted only under the exact signed-zero eligibility gate.
    for(int i=0;i<4;++i)for(int k=0;k<3;++k) {
        if(!finite(out.barrier_gradient[i][k])||!finite(0.0 /* original initialized friction gradient */))out.status=ContactStatus::NonFiniteDerivative;
        for(int j=0;j<4;++j)for(int l=0;l<3;++l)
            if(!finite(((i==role&&j==role)?out.selected_hessian[3*k+l]:0.0))||!finite(0.0 /* original initialized friction Hessian */))out.status=ContactStatus::NonFiniteDerivative;
    }
    out.active=out.status==ContactStatus::Ok;
    return out;
}

IPC_SELECTED_HD inline bool try_store(const ContactInput& in,bool deformable_phase,
    CompactDeformableContact* compact,unsigned char* output_activity,ContactStatus& status) {
    if(!deformable_phase||!compact||in.friction!=0.0)return false;
    const int role=compact_contact_role(in.hessian_blocks);
    if(role<0)return false;
    const auto out=evaluate_eligible(in,role);status=out.status;
    // Same publication order/early returns as original store_contact_output.
    if(output_activity) {
        *output_activity=static_cast<unsigned char>(out.status!=ContactStatus::Ok||out.active!=0);
        if(!*output_activity)return true;
    }
    compact->status=out.status;compact->active=out.active;compact->role=role;
    if(out.status!=ContactStatus::Ok||!out.active)return true;
    compact->weight=out.weights[role];
    for(int k=0;k<3;++k) {
        compact->gradient[k]=out.barrier_gradient[role][k];
        compact->friction_gradient[k]=0.0;
    }
    for(int k=0;k<9;++k) {
        compact->hessian[k]=out.selected_hessian[k];
        compact->friction_hessian[k]=0.0;
    }
    return true;
}
} // namespace ipc_gpu::resident::selected_ordinary
#undef IPC_SELECTED_HD
