#pragma once

#include "rigid_pins_types.h"

namespace ipc_gpu::solver {

struct RigidPinOracleInput {
    RigidPinConstraint pin;
    double com[3], previous_orientation[4], omega[3];
    double dt, kpin, time;
    unsigned with_orientation_hessian;
};

struct RigidPinAssemblyFixture {
    std::size_t bodies=4, count=6;
    RigidPinConstraint pins[6];
    std::size_t offsets[5]{0,2,3,5,6};
    int body_blocks[4]{3,7,1,0};
    int colors[8]{1,0,-1,0,-1,-1,-1,1};
    unsigned char translate[4]{1,0,1,0}, rotate[4]{1,1,0,0};
    double com[4][3], committed_com[4][3], previous_orientation[4][4], omega[4][3];
    double dt, kpin, time;
    Derivatives initial[4], translation_expected[4], orientation_expected[4];
};

void rigid_pin_oracle_cuda(const RigidPinOracleInput*,std::size_t,
    RigidDerivatives*,int*,double* targets);
void rigid_pin_assembly_cuda(const RigidPinAssemblyFixture&);

} // namespace ipc_gpu::solver
