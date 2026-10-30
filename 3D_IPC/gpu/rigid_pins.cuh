#pragma once

#include "rigid_pins_types.h"
#include "rigid.cuh"
#include <cmath>

#ifdef __CUDACC__
#define IPC_RIGID_PIN_HD __host__ __device__
#else
#define IPC_RIGID_PIN_HD
#endif

namespace ipc_gpu::solver {
namespace rigid_pin_detail {

IPC_RIGID_PIN_HD inline bool finite(double x) {
#ifdef __CUDA_ARCH__
    return isfinite(x);
#else
    return std::isfinite(x);
#endif
}
IPC_RIGID_PIN_HD inline bool finite3(const double* x) {
    return finite(x[0]) && finite(x[1]) && finite(x[2]);
}
IPC_RIGID_PIN_HD inline bool finite_derivatives(const RigidDerivatives& d) {
    for (int k=0;k<3;++k)
        if (!finite(d.translation_gradient[k]) || !finite(d.orientation_gradient[k])) return false;
    for (int k=0;k<9;++k)
        if (!finite(d.translation_translation_hessian[k]) ||
            !finite(d.translation_orientation_hessian[k]) ||
            !finite(d.orientation_orientation_hessian[k])) return false;
    return true;
}
IPC_RIGID_PIN_HD inline bool valid_pin(const RigidPinConstraint& p) {
    return finite3(p.local_position) && finite3(p.initial_target) &&
        finite3(p.rotation_center) && finite(p.angular_velocity_z);
}

} // namespace rigid_pin_detail

IPC_RIGID_PIN_HD inline bool rigid_pin_target(const RigidPinConstraint& pin,
    double time, double* target) {
    if (!rigid_pin_detail::valid_pin(pin) || !rigid_pin_detail::finite(time)) return false;
    const double theta = pin.angular_velocity_z*time;
    if (!rigid_pin_detail::finite(theta)) return false;
    const double c = cos(theta), s = sin(theta);
    double offset[3];
    for (int k=0;k<3;++k) offset[k] = pin.initial_target[k]-pin.rotation_center[k];
    target[0] = pin.rotation_center[0]+(c*offset[0]-s*offset[1]);
    target[1] = pin.rotation_center[1]+(s*offset[0]+c*offset[1]);
    target[2] = pin.rotation_center[2]+offset[2];
    return rigid_pin_detail::finite3(target);
}

// Exact pullback of dt^2*kpin/2*||com+R(q(omega))*X-target(time)||^2.
// with_orientation_hessian includes the signed quaternion-curvature term;
// it is deliberately not a positive-semidefinite Gauss-Newton replacement.
IPC_RIGID_PIN_HD inline bool evaluate_rigid_pin(const RigidPinConstraint& pin,
    const double* com, const RigidKinematics& kin, double dt, double kpin,
    double time, bool with_orientation_hessian, RigidDerivatives& out) {
    out = {};
    if (!rigid_pin_detail::finite(kpin)) return false;
    if (kpin<=0) return true; // Same inactive-spring rule as CPU.
    if (!rigid_pin_detail::finite(dt) || dt<0 || !rigid_pin_detail::finite3(com)) return false;
    const double scale = dt*dt*kpin;
    if (!rigid_pin_detail::finite(scale)) return false;
    for (int a=0;a<4;++a) {
        if (!rigid_pin_detail::finite(kin.orientation[a])) return false;
        for (int k=0;k<3;++k)
            if (!rigid_pin_detail::finite(kin.orientation_jacobian[a][k])) return false;
        if (with_orientation_hessian) for (int k=0;k<9;++k)
            if (!rigid_pin_detail::finite(kin.orientation_hessians[a][k])) return false;
    }
    double target[3], position[3];
    if (!rigid_pin_target(pin,time,target)) return false;
    rigid_world_position(pin.local_position,com,kin.orientation,position);
    double error[3], J[9], curvature[3][9];
    for (int k=0;k<3;++k) error[k] = position[k]-target[k];
    if (!rigid_pin_detail::finite3(error)) return false;
    rigid_reference_derivatives(pin.local_position,kin,J,curvature);
    for (int r=0;r<3;++r) {
        out.translation_gradient[r] = scale*error[r];
        out.orientation_gradient[r] = ((scale*J[r])*error[0]+(scale*J[3+r])*error[1])
            +(scale*J[6+r])*error[2];
        out.translation_translation_hessian[3*r+r] = scale;
    }
    for (int k=0;k<9;++k) out.translation_orientation_hessian[k] = scale*J[k];
    if (with_orientation_hessian) {
        double H[9];
        for (int r=0;r<3;++r) for (int c=0;c<3;++c) {
            double value = (J[r]*J[c]+J[3+r]*J[3+c])+J[6+r]*J[6+c];
            for (int axis=0;axis<3;++axis) value += error[axis]*curvature[axis][3*r+c];
            H[3*r+c] = value;
        }
        for (int r=0;r<3;++r) for (int c=0;c<3;++c)
            out.orientation_orientation_hessian[3*r+c] = scale*(0.5*(H[3*r+c]+H[3*c+r]));
    }
    return rigid_pin_detail::finite_derivatives(out);
}

// One body owns its aggregate; there are no floating-point atomics or races.
// Invalid metadata/derivatives return an error without publishing a partial sum.
IPC_RIGID_PIN_HD inline int evaluate_rigid_body_pins(const RigidPinScene& pins,
    std::size_t body, const double* com, const RigidKinematics& kin, double dt,
    double kpin, double time, bool with_orientation_hessian, RigidDerivatives& out) {
    out = {};
    if (!rigid_pin_detail::finite(kpin)) return RigidPinInvalidInput;
    if (kpin<=0 || pins.count==0) return 0;
    if (body>=pins.bodies || !pins.pins || !pins.offsets ||
        pins.offsets[0]!=0 || pins.offsets[pins.bodies]!=pins.count) return RigidPinInvalidInput;
    const auto begin=pins.offsets[body], end=pins.offsets[body+1];
    if (begin>end || end>pins.count) return RigidPinInvalidInput;
    RigidDerivatives sum{};
    for (std::size_t i=begin;i<end;++i) {
        if (pins.pins[i].body!=static_cast<int>(body) || !rigid_pin_detail::valid_pin(pins.pins[i]))
            return RigidPinInvalidInput;
        RigidDerivatives value{};
        if (!evaluate_rigid_pin(pins.pins[i],com,kin,dt,kpin,time,with_orientation_hessian,value))
            return RigidPinInvalidDerivative;
        for (int k=0;k<3;++k) {
            sum.translation_gradient[k] += value.translation_gradient[k];
            sum.orientation_gradient[k] += value.orientation_gradient[k];
        }
        for (int k=0;k<9;++k) {
            sum.translation_translation_hessian[k] += value.translation_translation_hessian[k];
            sum.translation_orientation_hessian[k] += value.translation_orientation_hessian[k];
            sum.orientation_orientation_hessian[k] += value.orientation_orientation_hessian[k];
        }
    }
    if (!rigid_pin_detail::finite_derivatives(sum)) return RigidPinInvalidDerivative;
    out = sum;
    return 0;
}

} // namespace ipc_gpu::solver

#undef IPC_RIGID_PIN_HD
