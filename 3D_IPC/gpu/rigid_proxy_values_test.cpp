#include "rigid_proxy_values.cuh"
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

using V=Eigen::Vector3d;
using namespace ipc_gpu::solver;
static void require(bool value,const char* why) {if(!value)throw std::runtime_error(why);}
static bool finite(const double* a) {return std::isfinite(a[0])&&std::isfinite(a[1])&&std::isfinite(a[2]);}

int main() {
    try {
        std::mt19937_64 random(0x52ab831);
        std::uniform_real_distribution<double> scalar(-1,1);
        for(int sample=0;sample<4096;++sample) {
            V com,previous,spin,reference;
            for(int k=0;k<3;++k) {
                com[k]=scalar(random);previous[k]=com[k]+.01*scalar(random);
                spin[k]=scalar(random);reference[k]=scalar(random);
            }
            Eigen::Quaterniond q(scalar(random),scalar(random),scalar(random),scalar(random));q.normalize();
            const double orientation[4]={q.w(),q.x(),q.y(),q.z()};
            const double dt=.001+.1*std::abs(scalar(random));
            for(bool translation:{false,true})for(bool rotation:{false,true}) {
                double velocity[3],omega[3],actual[3];
                proxy_values::finish_body(com.data(),previous.data(),spin.data(),translation,
                    rotation,dt,velocity,omega);
                V expected_velocity=V::Zero(),expected_omega=V::Zero();
                if(translation)expected_velocity=(com-previous)/dt;
                if(rotation)expected_omega=spin;
                for(int k=0;k<3;++k) {
                    require(velocity[k]==expected_velocity[k],"body velocity differs from independent Eigen oracle");
                    require(omega[k]==expected_omega[k],"body angular phase differs from mode oracle");
                }
                proxy_values::finish_proxy(reference.data(),orientation,velocity,omega,actual);
                const V expected=expected_velocity+expected_omega.cross(q.toRotationMatrix()*reference);
                require(finite(actual),"healthy proxy velocity is nonfinite");
                for(int k=0;k<3;++k)
                    require(std::abs(actual[k]-expected[k])<=3e-12,"proxy world/cross differs from Eigen matrix oracle");
            }
        }
        const double nan=std::numeric_limits<double>::quiet_NaN();
        const double infinity=std::numeric_limits<double>::infinity();
        const double maximum=std::numeric_limits<double>::max();
        const double identity[4]={1,0,0,0},zero[3]={0,0,0};
        const double disabled_bad[3]={nan,infinity,-infinity};
        double velocity[3],omega[3],actual[3];
        proxy_values::finish_body(disabled_bad,disabled_bad,disabled_bad,false,false,.01,velocity,omega);
        require(finite(velocity)&&finite(omega),"disabled generalized values were consumed");
        for(int k=0;k<3;++k)require(velocity[k]==0&&omega[k]==0&&!std::signbit(velocity[k])&&!std::signbit(omega[k]),
            "disabled generalized values changed original positive zero");
        // Fixed proxies must still run the original world/cross arithmetic.
        // Replacing it by constant zero would mask this baseline error path.
        const double invalid_reference[3]={infinity,0,0};
        proxy_values::finish_proxy(invalid_reference,identity,velocity,omega,actual);
        require(!finite(actual),"fixed 0*Inf proxy path was incorrectly skipped");
        const double huge[3]={maximum,0,0},opposite[3]={-maximum,0,0};
        proxy_values::finish_body(huge,opposite,zero,true,false,.01,velocity,omega);
        require(!finite(velocity),"finite generalized subtraction overflow was hidden");
        const double large_spin[3]={maximum,maximum,0},large_reference[3]={2,2,2};
        proxy_values::finish_proxy(large_reference,identity,zero,large_spin,actual);
        require(!finite(actual),"finite proxy cross overflow was hidden");
        std::cout<<"Host proxy value oracle PASS: 4096 poses x4 modes, Eigen world/cross/dt, disabled gates, fixed invalid references, finite overflow\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
